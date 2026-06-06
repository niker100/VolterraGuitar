"""Linear FIR baseline — the LTI sanity floor.

A length-``M`` finite impulse response identified by ridge-regularized least
squares, mapping ``x`` to ``y``. Being strictly linear and time-invariant it can
only capture the circuit's *linear frequency response* (its small-signal
transfer function); it is blind to the harmonic distortion that defines a guitar
overdrive/fuzz. We keep it as the benchmark's lower bound: any nonlinear model
worth its complexity must beat the best LTI fit on ESR/THD.

Identification is ordinary linear regression on the lagged (Toeplitz) design
matrix. With taps ``b`` and lagged inputs ``X`` (row ``n`` holds
``[x[n], x[n-1], ..., x[n-M+1]]``) the model is ``y ≈ X b``; ridge solves
``(XᵀX + λI) b = Xᵀy`` (Tikhonov regularization), which stabilizes the normal
equations when the excitation is band-limited and ``XᵀX`` is near-singular. We
accumulate the small ``M*M`` Gram matrix ``XᵀX`` and ``Xᵀy`` in chunks so peak
memory is ``O(chunk·M)`` rather than ``O(N·M)``.

Self-consistency: ``process`` is ``scipy.signal.lfilter(b, [1], x)``; streaming
carries ``lfilter``'s state ``zi`` across blocks, so the concatenated
``process_block`` output equals ``process`` exactly (causal, zero latency) —
``check_streaming`` passes to floating-point round-off.
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
class LinearFIR(Model):
    """Ridge-fit length-``M`` FIR (LTI) baseline; captures linear response only."""

    name: ClassVar[str] = "fir"
    description: ClassVar[str] = "Linear FIR baseline (ridge LS); LTI sanity floor."
    latency_samples: int = 0  # causal FIR, no lookahead

    def __init__(self, memory: int = 256, ridge: float = 1e-6) -> None:
        """Construct an (untrained) FIR.

        Args:
            memory: number of taps ``M`` (filter length / FIR memory).
            ridge: Tikhonov regularization strength ``λ`` for the normal equations.
        """
        if memory < 1:
            raise ValueError("memory must be >= 1")
        self.memory = int(memory)
        self.ridge = float(ridge)
        # Identity impulse response => passthrough before fitting.
        self.b: np.ndarray = np.zeros(self.memory, dtype=np.float64)
        self.b[0] = 1.0
        self._zi: np.ndarray | None = None  # streaming filter state

    # --- learning ---------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Solve ridge least squares for the taps via chunked normal equations.

        Accumulates the Gram matrix ``XᵀX`` (``M*M``) and ``Xᵀy`` over chunks of
        the lagged design matrix, then solves ``(XᵀX + λI) b = Xᵀy``. Chunking
        bounds peak memory to ``O(chunk·M)`` regardless of dataset length.
        """
        x = train.x.astype(np.float64)
        y = train.y.astype(np.float64)
        n, m = x.shape[0], self.memory

        gram = np.zeros((m, m), dtype=np.float64)
        rhs = np.zeros(m, dtype=np.float64)
        chunk = max(m, 1 << 16)  # rows per chunk; cap the temporary X block
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            xlag = _lag_matrix(x, m, start, stop)  # (rows, M)
            yc = y[start:stop]
            gram += xlag.T @ xlag
            rhs += xlag.T @ yc

        gram.flat[:: m + 1] += self.ridge  # add λ to the diagonal
        self.b = np.linalg.solve(gram, rhs)
        self.reset()

        # Training-set residual error (relative) for a quick fit diagnostic.
        resid = y - lfilter(self.b, [1.0], x)
        denom = float(np.sum(y * y)) or 1.0
        train_esr = float(np.sum(resid * resid) / denom)
        return FitReport(
            history={"train_esr": [train_esr]},
            info={"memory": m, "ridge": self.ridge},
        )

    # --- offline inference ------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Whole-signal FIR convolution via direct-form filtering."""
        y = lfilter(self.b, [1.0], np.asarray(x, dtype=np.float64))
        return y.astype(np.float32)

    # --- streaming inference ---------------------------------------------
    def reset(self) -> None:
        """Zero the FIR delay line (``lfilter`` state)."""
        self._zi = np.zeros(self.memory - 1, dtype=np.float64)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Filter one block, carrying ``lfilter`` state across calls."""
        if self._zi is None:
            self.reset()
        y, self._zi = lfilter(self.b, [1.0], np.asarray(x, dtype=np.float64), zi=self._zi)
        return y.astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist taps + hyperparameters to the exact ``path`` (npz format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:  # file handle => no '.npz' appended; save/load symmetric
            np.savez(f, b=self.b, memory=np.int64(self.memory), ridge=np.float64(self.ridge))

    @classmethod
    def load(cls, path: str | Path) -> LinearFIR:
        """Reconstruct an instance and restore its learned taps."""
        with np.load(path, allow_pickle=False) as f:
            model = cls(memory=int(f["memory"]), ridge=float(f["ridge"]))
            model.b = np.asarray(f["b"], dtype=np.float64)
        model.reset()
        return model

    # --- introspection ----------------------------------------------------
    def num_params(self) -> int:
        """Learned parameters = number of FIR taps."""
        return self.memory


def _lag_matrix(x: np.ndarray, m: int, start: int, stop: int) -> np.ndarray:
    """Lagged design rows for outputs ``start:stop``.

    Row ``i`` (output index ``start+i``) is ``[x[k], x[k-1], ..., x[k-M+1]]`` with
    ``k = start+i``; out-of-range (negative) lags are zero, matching the implicit
    zero initial state of a causal FIR. Built by left-padding then striding so no
    Python-level loop is needed.
    """
    rows = stop - start
    pad = np.concatenate([np.zeros(m - 1, dtype=x.dtype), x[:stop]])
    # Window over the padded signal: column j is lag j (reversed for tap order).
    base = pad[start : start + rows + m - 1]
    win = np.lib.stride_tricks.sliding_window_view(base, m)  # (rows, M), lag 0..M-1
    return win[:, ::-1]
