"""Parallel-cascade (LN) Volterra identification.

Korenberg's parallel-cascade method (M. J. Korenberg, "Parallel cascade
identification and kernel estimation for nonlinear systems", *Annals of
Biomedical Engineering* 19 (1991) 429-455). Any finite-memory, finite-order
Volterra system can be represented exactly as a *finite* sum of parallel
LN cascades, where each cascade is a linear FIR filter ``h_k`` (the "L" stage)
followed by a static polynomial nonlinearity ``g_k`` of degree ``<= poly_deg``
(the "N" stage)::

    y(n) ~= sum_k  g_k( (h_k * x)(n) )

This is *Volterra-equivalent* but far cheaper to run than an explicit Volterra
series: each path costs one 1-D convolution plus a pointwise polynomial, so the
total cost is linear in the number of paths and memory length rather than
exponential in the Volterra order.

Identification is greedy/iterative. Starting from residual ``e = y``, each
iteration fits one new cascade to the *current residual*:

1.  Estimate a candidate filter ``h_k`` from a slice of the input-residual
    cross-correlation (Korenberg's construction: a Volterra kernel slice is a
    valid cascade filter). Random restarts diversify the directions explored.
2.  Convolve ``v = h_k * x`` and fit the polynomial ``g_k`` mapping ``v -> e``
    by linear least squares on the (normalised) powers of ``v``.
3.  Subtract ``g_k(v)`` from the residual.

The loop stops when the residual energy stops decreasing meaningfully (relative
improvement below ``tol``) or ``n_paths`` cascades have been added. Because each
stage only ever *reduces* the residual energy, convergence is monotone.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.signal import lfilter

from vguitar.models.base import FitReport, Model, register_model

if TYPE_CHECKING:
    from vguitar.config import TrainConfig
    from vguitar.data import Dataset


@register_model
class VolterraPC(Model):
    """Sum of LN cascades identified by parallel-cascade regression.

    Args:
        n_paths: maximum number of cascades (parallel LN paths).
        mem: FIR filter length (memory) of each cascade's linear stage.
        poly_deg: degree of each cascade's static polynomial nonlinearity.
        tol: relative residual-energy improvement below which fitting stops.
    """

    name: ClassVar[str] = "volterra_pc"
    description: ClassVar[str] = "Korenberg parallel-cascade (LN) Volterra model"
    latency_samples: int = 0

    def __init__(
        self, n_paths: int = 8, mem: int = 64, poly_deg: int = 3, tol: float = 1e-4
    ) -> None:
        self.n_paths = int(n_paths)
        self.mem = int(mem)
        self.poly_deg = int(poly_deg)
        self.tol = float(tol)
        # Learned parameters, populated by ``fit`` / ``load``.
        # filters: (k, mem) FIR taps; coeffs: (k, poly_deg+1) polynomial weights
        # (ascending power order, evaluated on the *normalised* filter output).
        # scales: (k,) per-path normalisation so the polynomial sees unit-ish
        # inputs (improves conditioning; folded into the coeffs at eval time).
        self.filters: np.ndarray = np.zeros((0, self.mem), dtype=np.float64)
        self.coeffs: np.ndarray = np.zeros((0, self.poly_deg + 1), dtype=np.float64)
        self.scales: np.ndarray = np.zeros((0,), dtype=np.float64)
        # Streaming state: one lfilter delay-line per path (set by ``reset``).
        self._zi: list[np.ndarray] = []

    # --- core math --------------------------------------------------------
    def _poly(self, v: np.ndarray, k: int) -> np.ndarray:
        """Evaluate path ``k``'s polynomial on filter output ``v`` (vectorised)."""
        # Horner on normalised input; np.polyval wants descending coeffs.
        c = self.coeffs[k][::-1]
        return np.polyval(c, v / self.scales[k])

    def _filtered(self, x: np.ndarray, k: int) -> np.ndarray:
        """Offline FIR output ``h_k * x`` for path ``k`` (zero initial state)."""
        return lfilter(self.filters[k], (1.0,), x)

    # --- learning ---------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Greedily add LN cascades, each fit to the running residual.

        See the module docstring for the algorithm. ``val``/``cfg`` are accepted
        for interface uniformity; identification uses only the training signals.
        """
        seed = cfg.seed if cfg is not None else 0
        rng = np.random.default_rng(seed)
        x = train.x.astype(np.float64)
        y = train.y.astype(np.float64)

        filters: list[np.ndarray] = []
        coeffs: list[np.ndarray] = []
        scales: list[float] = []

        residual = y.copy()
        e0 = float(residual @ residual) + 1e-30
        history: list[float] = [e0]

        for _ in range(self.n_paths):
            h = self._propose_filter(x, residual, rng)
            v = lfilter(h, (1.0,), x)
            scale = float(np.std(v)) or 1.0
            c = self._fit_poly(v / scale, residual)
            contrib = np.polyval(c[::-1], v / scale)

            new_res = residual - contrib
            e_new = float(new_res @ new_res)
            # Reject a path that does not reduce energy (numerical safety).
            if e_new >= history[-1]:
                break
            filters.append(h)
            coeffs.append(c)
            scales.append(scale)
            residual = new_res
            history.append(e_new)
            if (history[-2] - e_new) / history[-2] < self.tol:
                break

        self.filters = np.asarray(filters, dtype=np.float64).reshape(-1, self.mem)
        self.coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1, self.poly_deg + 1)
        self.scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        self.reset()

        esr = history[-1] / e0  # error-to-signal energy ratio on train
        return FitReport(
            history={"train_residual_energy": history},
            info={"n_paths": int(self.filters.shape[0]), "train_esr": esr},
        )

    def _propose_filter(
        self, x: np.ndarray, residual: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Estimate a cascade filter from the input-residual cross-correlation.

        A slice of the first-order Volterra kernel of the residual is a valid
        cascade filter (Korenberg 1991). We add a small random perturbation so
        successive paths explore different directions (random-restart flavour).
        """
        m = self.mem
        # Cross-correlation r_xe[tau] = sum_n x[n-tau] residual[n], tau=0..m-1.
        # Implemented as correlation of residual against a windowed x.
        xc = np.correlate(residual, x, mode="full")
        mid = len(x) - 1  # zero-lag index in 'full' correlation
        h = xc[mid : mid + m]
        if h.shape[0] < m:  # short signals: pad
            h = np.pad(h, (0, m - h.shape[0]))
        h = h + rng.standard_normal(m) * (np.std(h) * 0.1 + 1e-9)
        nrm = float(np.linalg.norm(h))
        return h / nrm if nrm > 0 else h

    def _fit_poly(self, v: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Least-squares polynomial coeffs (ascending powers) mapping v->target."""
        # Vandermonde of powers [v^0, v^1, ..., v^deg]; solve V c ~= target.
        powers = np.vander(v, self.poly_deg + 1, increasing=True)
        c, *_ = np.linalg.lstsq(powers, target, rcond=None)
        return c

    # --- offline inference ------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Sum every cascade ``g_k(h_k * x)`` over the whole signal."""
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.zeros_like(xf)
        for k in range(self.filters.shape[0]):
            y += self._poly(self._filtered(xf, k), k)
        return y.astype(np.float32)

    # --- streaming inference ----------------------------------------------
    def reset(self) -> None:
        """Reset every path's FIR delay line to zero (silent initial state)."""
        self._zi = [
            np.zeros(max(self.mem - 1, 0), dtype=np.float64) for _ in range(self.filters.shape[0])
        ]

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process one block, carrying each path's FIR state across calls.

        ``lfilter`` with explicit ``zi`` makes the FIR convolution stateful, so
        block-by-block output matches :meth:`process` exactly (the polynomial is
        memoryless). Latency is therefore 0.
        """
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        if not self._zi:
            self.reset()
        y = np.zeros_like(xf)
        for k in range(self.filters.shape[0]):
            v, self._zi[k] = lfilter(self.filters[k], (1.0,), xf, zi=self._zi[k])
            y += self._poly(v, k)
        return y.astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist hyperparameters and learned cascades to a ``.npz`` archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:  # file handle => no '.npz' appended; save/load symmetric
            np.savez(
                f,
                n_paths=np.int64(self.n_paths),
                mem=np.int64(self.mem),
                poly_deg=np.int64(self.poly_deg),
                tol=np.float64(self.tol),
                filters=self.filters,
                coeffs=self.coeffs,
                scales=self.scales,
            )

    @classmethod
    def load(cls, path: str | Path) -> VolterraPC:
        """Reconstruct an untrained instance and restore its parameters."""
        with np.load(path, allow_pickle=False) as f:
            obj = cls(
                n_paths=int(f["n_paths"]),
                mem=int(f["mem"]),
                poly_deg=int(f["poly_deg"]),
                tol=float(f["tol"]),
            )
            obj.filters = f["filters"].astype(np.float64).reshape(-1, obj.mem)
            obj.coeffs = f["coeffs"].astype(np.float64).reshape(-1, obj.poly_deg + 1)
            obj.scales = f["scales"].astype(np.float64).reshape(-1)
        obj.reset()
        return obj

    # --- introspection ----------------------------------------------------
    def num_params(self) -> int:
        """Total learned parameters: FIR taps + polynomial coeffs across paths."""
        return int(self.filters.size + self.coeffs.size)
