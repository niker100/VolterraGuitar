"""Regularized truncated Volterra-series emulator.

A Volterra series models a nonlinear, fading-memory system as a sum of
multidimensional convolutions ("kernels")::

    y[n] = h0
         + sum_i      h1[i]            x[n-i]
         + sum_{i<=j} h2[i,j]          x[n-i] x[n-j]
         + sum_{i<=j<=k} h3[i,j,k]     x[n-i] x[n-j] x[n-k]

This is the *correct* way to obtain Volterra kernels for our circuits: identify
them directly by least squares from input/output data. (Contrast the legacy
``kernel_est`` approach, which tried to read kernels out of trained neural-net
weights -- that is not a valid identification and is not reproduced here.)

WHY REGULARIZATION. The number of free kernel coefficients explodes with order
and memory length (the "curse of dimensionality"): an order-3 kernel of memory
``M`` has ~``M^3`` entries. Plain least squares over so many correlated lagged
products is wildly ill-conditioned and overfits. Following Birpoutsoukis,
Marconato, Hellings & Schoukens, "Regularized nonparametric Volterra kernel
estimation" (Automatica, 2017), we treat each kernel as a draw from a Gaussian
process whose prior encodes the two things we know physically: the kernel is
**smooth** (neighbouring taps are correlated) and **decays** with lag (fading
memory). That prior becomes a quadratic penalty added to the LS cost, i.e. a
generalized Tikhonov term ``reg * h^T P h``. The penalty tames the conditioning
so we can keep useful memory with far fewer *effective* degrees of freedom than
raw coefficient counts suggest.

SYMMETRY. ``x[n-i] x[n-j]`` is symmetric in ``i,j``, so only the upper-triangular
index set is identifiable; we parametrize exactly those (and the triangular
"simplex" ``i<=j<=k`` for order 3), cutting the free-parameter count by ~2x for
order 2 and ~6x for order 3.

LIMITATION. A truncated Volterra series is a polynomial in the input and
therefore *diverges outside the amplitude range it was trained on* (large inputs
blow up the high-order terms). This is mitigated -- not eliminated -- by
exciting with amplitude-rich signals spanning the expected drive range (see
:class:`vguitar.config.DataConfig.drive_levels`); for hard clipping it remains a
fundamental weakness of the method.

Memory is kept modest (defaults ``mem1=128, mem2=32, mem3=12``) so identification
and real-time evaluation stay tractable on CPU.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.signal import lfilter

from vguitar.models.base import FitReport, Model, register_model

if TYPE_CHECKING:
    from vguitar.config import TrainConfig
    from vguitar.data import Dataset


@register_model
class VolterraReg(Model):
    """Regularized truncated Volterra series identified by penalized least squares.

    Args:
        mem1: memory (taps) of the linear kernel ``h1``.
        mem2: memory of the quadratic kernel ``h2`` (symmetric, triangular).
        mem3: memory of the cubic kernel ``h3`` (symmetric, simplex).
        max_order: highest kernel order to fit (1, 2 or 3).
        reg: regularization strength ``lambda`` for the smoothness/decay prior.
    """

    name: ClassVar[str] = "volterra"
    description: ClassVar[str] = "Regularized truncated Volterra series (orders 1-3)."
    latency_samples: int = 0  # causal: y[n] depends only on x[n], x[n-1], ...

    def __init__(
        self,
        mem1: int = 128,
        mem2: int = 32,
        mem3: int = 12,
        max_order: int = 3,
        reg: float = 1e-2,
    ) -> None:
        if not 1 <= max_order <= 3:
            raise ValueError("max_order must be 1, 2 or 3")
        self.mem1 = int(mem1)
        self.mem2 = int(mem2)
        self.mem3 = int(mem3)
        self.max_order = int(max_order)
        self.reg = float(reg)

        # Learned parameters (None until fit/load). Bias + one flat coeff vector
        # per active order; kernels are stored in their reduced (symmetric) form.
        self.h0: float = 0.0
        self.h1: np.ndarray | None = None  # (mem1,)
        self.h2: np.ndarray | None = None  # (n_pairs,)  upper-triangular i<=j
        self.h3: np.ndarray | None = None  # (n_triples,) simplex i<=j<=k

        # Precomputed symmetric index sets (multiplicities fold into the coeffs).
        self._idx2 = _triangular_indices(self.mem2, 2)  # (n_pairs, 2)
        self._idx3 = _triangular_indices(self.mem3, 3)  # (n_triples, 3)

        # Streaming state: input ring buffer of length = longest memory.
        self._maxmem = max(self.mem1, self.mem2, self.mem3)
        self._buf = np.zeros(self._maxmem - 1, dtype=np.float64)

    # --- learning ---------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Identify kernels by regularized least squares (penalized normal equations).

        Builds the regression matrix ``Phi`` of lagged products (one column per
        free, symmetry-reduced coefficient), then solves
        ``(Phi^T Phi + lambda * P) theta = Phi^T y`` where ``P`` is the
        block-diagonal smoothness/decay prior precision. Solving the normal
        equations (rather than a full least-squares of the tall ``Phi``) keeps the
        problem ``n_params x n_params`` and lets us add ``P`` cleanly.
        """
        x = np.asarray(train.x, dtype=np.float64)
        y = np.asarray(train.y, dtype=np.float64)

        cols, sizes = self._design_columns(x)
        phi = np.column_stack([np.ones_like(x), *cols])  # leading bias column

        # Block-diagonal prior precision: bias unpenalized, each kernel gets its
        # own smoothness+decay block (Birpoutsoukis et al. 2017, GP view).
        blocks = [np.zeros((1, 1))]
        blocks.append(_prior_precision(self.mem1))  # h1: full square memory
        if self.max_order >= 2:
            blocks.append(_simplex_prior_precision(self._idx2, self.mem2))
        if self.max_order >= 3:
            blocks.append(_simplex_prior_precision(self._idx3, self.mem3))
        prior = _block_diag(blocks)

        # Penalized normal equations. Scale lambda by the data energy so ``reg``
        # is roughly dimensionless / drive-independent.
        gram = phi.T @ phi
        rhs = phi.T @ y
        scale = float(np.trace(gram)) / max(gram.shape[0], 1)
        reg_gram = gram + self.reg * scale * prior
        theta = np.linalg.solve(reg_gram, rhs)

        # Unpack the flat solution back into kernels.
        self.h0 = float(theta[0])
        off = 1
        self.h1 = theta[off : off + self.mem1].copy()
        off += self.mem1
        if self.max_order >= 2:
            self.h2 = theta[off : off + sizes[1]].copy()
            off += sizes[1]
        if self.max_order >= 3:
            self.h3 = theta[off : off + sizes[2]].copy()
            off += sizes[2]

        resid = phi @ theta - y
        train_mse = float(np.mean(resid**2))
        history: dict[str, list[float]] = {"train_loss": [train_mse]}
        if val is not None:
            vy = np.asarray(val.y, dtype=np.float64)
            verr = self.process(np.asarray(val.x, dtype=np.float32)).astype(np.float64) - vy
            history["val_loss"] = [float(np.mean(verr**2))]
        return FitReport(
            history=history,
            info={"num_params": self.num_params(), "cond": float(np.linalg.cond(reg_gram))},
        )

    # --- offline inference ------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a whole signal; same length as ``x``, float32.

        Order 1 is an FIR convolution via :func:`scipy.signal.lfilter`. Orders 2/3
        sum symmetric lagged products over their (small) supports, vectorized over
        time using one precomputed lagged-input matrix.
        """
        if self.h1 is None:
            raise RuntimeError("model is not fitted")
        xd = np.asarray(x, dtype=np.float64).reshape(-1)
        n = xd.shape[0]
        y = np.full(n, self.h0, dtype=np.float64)

        # Linear term: y += conv(h1, x), causal.
        y += lfilter(self.h1, [1.0], xd)

        if self.max_order >= 2 and self.h2 is not None:
            y += self._poly_term(xd, self._idx2, self.h2, self.mem2)
        if self.max_order >= 3 and self.h3 is not None:
            y += self._poly_term(xd, self._idx3, self.h3, self.mem3)
        return y.astype(np.float32)

    # --- streaming inference (realtime) -----------------------------------
    def reset(self) -> None:
        self._buf = np.zeros(self._maxmem - 1, dtype=np.float64)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process one block causally; same length as ``x``, float32.

        Prepends the input ring buffer (the last ``maxmem-1`` past samples) so the
        first samples of the block see correct history, runs the same offline math
        on the concatenation, then keeps the new tail as history. Output equals
        :meth:`process` over the full stream up to float rounding (``latency=0``).
        """
        if self.h1 is None:
            raise RuntimeError("model is not fitted")
        xb = np.asarray(x, dtype=np.float64).reshape(-1)
        ext = np.concatenate([self._buf, xb])
        y_ext = self._process_array(ext)
        out = y_ext[self._buf.shape[0] :]
        self._buf = ext[-(self._maxmem - 1) :] if self._maxmem > 1 else self._buf
        return out.astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save kernels + hyperparameters to a ``.npz``."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:  # file handle => no '.npz' appended; save/load symmetric
            np.savez(
                f,
                mem1=self.mem1,
                mem2=self.mem2,
                mem3=self.mem3,
                max_order=self.max_order,
                reg=self.reg,
                h0=self.h0,
                h1=self.h1 if self.h1 is not None else np.array([]),
                h2=self.h2 if self.h2 is not None else np.array([]),
                h3=self.h3 if self.h3 is not None else np.array([]),
            )

    @classmethod
    def load(cls, path: str | Path) -> VolterraReg:
        with np.load(path, allow_pickle=False) as f:
            m = cls(
                mem1=int(f["mem1"]),
                mem2=int(f["mem2"]),
                mem3=int(f["mem3"]),
                max_order=int(f["max_order"]),
                reg=float(f["reg"]),
            )
            m.h0 = float(f["h0"])
            m.h1 = f["h1"].astype(np.float64) if f["h1"].size else None
            m.h2 = f["h2"].astype(np.float64) if f["h2"].size else None
            m.h3 = f["h3"].astype(np.float64) if f["h3"].size else None
        return m

    # --- introspection ----------------------------------------------------
    def num_params(self) -> int:
        """Count of free kernel coefficients after symmetry reduction (+ bias)."""
        n = 1 + self.mem1  # bias + linear kernel
        if self.max_order >= 2:
            n += self._idx2.shape[0]
        if self.max_order >= 3:
            n += self._idx3.shape[0]
        return n

    # --- internals --------------------------------------------------------
    def _process_array(self, xd: np.ndarray) -> np.ndarray:
        """Float64 forward pass over a raw array (shared by process/process_block)."""
        y = np.full(xd.shape[0], self.h0, dtype=np.float64)
        assert self.h1 is not None
        y += lfilter(self.h1, [1.0], xd)
        if self.max_order >= 2 and self.h2 is not None:
            y += self._poly_term(xd, self._idx2, self.h2, self.mem2)
        if self.max_order >= 3 and self.h3 is not None:
            y += self._poly_term(xd, self._idx3, self.h3, self.mem3)
        return y

    def _design_columns(self, x: np.ndarray) -> tuple[list[np.ndarray], dict[int, int]]:
        """Regression columns (one per free coeff) and per-order column counts."""
        cols: list[np.ndarray] = []
        sizes: dict[int, int] = {}
        for i in range(self.mem1):  # linear: lagged copies of x
            cols.append(_shift(x, i))
        if self.max_order >= 2:
            c2 = [_lagged_product(x, idx) for idx in self._idx2]
            cols.extend(c2)
            sizes[1] = len(c2)
        if self.max_order >= 3:
            c3 = [_lagged_product(x, idx) for idx in self._idx3]
            cols.extend(c3)
            sizes[2] = len(c3)
        return cols, sizes

    @staticmethod
    def _poly_term(xd: np.ndarray, idx: np.ndarray, coef: np.ndarray, _mem: int) -> np.ndarray:
        """Sum a symmetric polynomial kernel over its support, vectorized in time."""
        out = np.zeros_like(xd)
        for row, c in zip(idx, coef, strict=True):
            prod = np.full(xd.shape[0], c, dtype=np.float64)
            for lag in row:
                prod *= _shift(xd, int(lag))
            out += prod
        return out


# --- module-level helpers (pure, testable) --------------------------------
def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Causal shift: ``x[n-lag]`` with zeros for ``n < lag`` (same length)."""
    if lag == 0:
        return x
    out = np.zeros_like(x)
    out[lag:] = x[:-lag]
    return out


def _lagged_product(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Pointwise product ``prod_k x[n - lags[k]]`` (one regression column)."""
    out = np.ones_like(x)
    for lag in lags:
        out = out * _shift(x, int(lag))
    return out


def _triangular_indices(mem: int, order: int) -> np.ndarray:
    """Upper-triangular (sorted, with repeats) lag tuples for a symmetric kernel.

    Returns shape ``(n_terms, order)``; ``n_terms = C(mem + order - 1, order)``.
    Choosing only sorted tuples removes the redundancy of the symmetric kernel.
    """
    if order == 1:
        return np.arange(mem, dtype=np.int64).reshape(-1, 1)
    return np.array(list(combinations_with_replacement(range(mem), order)), dtype=np.int64)


def _prior_precision(mem: int) -> np.ndarray:
    """Smoothness + exponential-decay prior *precision* for a length-``mem`` kernel.

    Implements the Birpoutsoukis et al. (2017) idea as a quadratic penalty: a
    second-difference (roughness) operator ``D`` penalizes non-smooth kernels, and
    a diagonal ``exp(+lag)`` weighting penalizes energy at long lags (encoding the
    fading-memory / exponential-decay prior). The returned matrix is
    ``D^T W D + epsilon I`` (the ``epsilon I`` keeps it full-rank / invertible).
    """
    if mem == 1:
        return np.ones((1, 1))
    # Second-difference operator (roughness): (mem-2) x mem.
    d = np.zeros((mem - 2, mem)) if mem >= 3 else np.zeros((0, mem))
    for r in range(mem - 2):
        d[r, r : r + 3] = (1.0, -2.0, 1.0)
    decay = np.exp(np.linspace(0.0, 4.0, mem))  # grows with lag -> penalize tails
    w = np.diag(decay[: max(mem - 2, 0)]) if mem >= 3 else np.zeros((0, 0))
    rough = d.T @ w @ d if mem >= 3 else np.zeros((mem, mem))
    # Diagonal decay term + small ridge for invertibility.
    return rough + np.diag(decay) + 1e-6 * np.eye(mem)


def _simplex_prior_precision(idx: np.ndarray, mem: int) -> np.ndarray:
    """Decay prior precision for a higher-order kernel on its simplex coefficients.

    Full multidimensional smoothness over a triangular index set is awkward to
    assemble; we keep it simple and use a diagonal exponential-decay prior keyed
    on the total lag of each term (sum of its lags), which still enforces fading
    memory and conditions the (otherwise rank-deficient) high-order block.
    """
    decay_per_lag = np.exp(np.linspace(0.0, 4.0, mem))
    diag = np.array([float(np.prod(decay_per_lag[row])) for row in idx])
    return np.diag(diag) + 1e-6 * np.eye(idx.shape[0])


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    """Assemble a block-diagonal matrix (small, so a plain loop is fine)."""
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total))
    o = 0
    for b in blocks:
        k = b.shape[0]
        out[o : o + k, o : o + k] = b
        o += k
    return out
