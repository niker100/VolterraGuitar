"""The :class:`Model` contract.

Every approach — Volterra, block-oriented, neural — implements this so the
benchmark and the live engine can treat them identically.

Two processing paths, deliberately:

* :meth:`Model.process` — offline, whole-signal, vectorized. Used for training
  diagnostics and evaluation (ESR/THD on the test set).
* :meth:`Model.reset` + :meth:`Model.process_block` — streaming, stateful,
  one block at a time. This is what the realtime engine calls and what we time
  to get the real-time factor (RTF). ``process(x)`` and a sequence of
  ``process_block`` calls over the same ``x`` must produce the same output
  (up to ``latency_samples``); :func:`vguitar.models.base.check_streaming`
  verifies this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np

if TYPE_CHECKING:
    from vguitar.config import TrainConfig
    from vguitar.data import Dataset


@dataclass
class FitReport:
    """What :meth:`Model.fit` returns: training curves + free-form info."""

    history: dict[str, list[float]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def final_val_loss(self) -> float | None:
        v = self.history.get("val_loss")
        return v[-1] if v else None


class Model(ABC):
    """A learnable emulator of a circuit's input->output behavior."""

    name: ClassVar[str]
    description: ClassVar[str] = ""
    #: Algorithmic output latency in samples (lookahead). 0 for causal models.
    latency_samples: int = 0

    # --- learning ---------------------------------------------------------
    @abstractmethod
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Identify/train the model from a dataset."""
        ...

    # --- offline inference ------------------------------------------------
    @abstractmethod
    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a whole signal at once; returns output the same length as ``x``."""
        ...

    # --- streaming inference (realtime) -----------------------------------
    @abstractmethod
    def reset(self) -> None:
        """Clear streaming state (ring buffers, hidden state) before live use."""
        ...

    @abstractmethod
    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process one block, advancing internal state; returns same-length output."""
        ...

    # --- persistence ------------------------------------------------------
    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Model: ...

    # --- introspection (for the benchmark) --------------------------------
    @abstractmethod
    def num_params(self) -> int:
        """Number of free parameters (kernel taps, weights, ...)."""
        ...


def check_streaming(
    model: Model, n: int = 8192, block: int = 128, seed: int = 0, atol: float = 1e-4
) -> float:
    """Assert ``process`` and blockwise ``process_block`` agree; return max abs error.

    Accounts for ``model.latency_samples`` by comparing the overlapping region.
    Useful as a unit test for every model implementation.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32) * 0.3
    y_off = np.asarray(model.process(x), dtype=np.float32)
    model.reset()
    chunks = [model.process_block(x[i : i + block]) for i in range(0, n, block)]
    y_stream = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])[:n]
    lat = model.latency_samples
    a = y_off[lat:] if lat else y_off
    b = y_stream[lat:] if lat else y_stream
    m = min(len(a), len(b))
    err = float(np.max(np.abs(a[:m] - b[:m]))) if m else float("inf")
    if not np.isfinite(err) or err > atol:
        raise AssertionError(
            f"{model.name}: streaming mismatch (max abs err {err:.2e} > {atol:.0e})"
        )
    return err


# --- registry -------------------------------------------------------------
_REGISTRY: dict[str, type[Model]] = {}

_ModelT = TypeVar("_ModelT", bound="Model")


def register_model(cls: type[_ModelT]) -> type[_ModelT]:
    """Class decorator: register a Model under its ``name`` (identity-preserving)."""
    key = cls.name.lower()
    if key in _REGISTRY:
        raise ValueError(f"duplicate model name: {cls.name!r}")
    _REGISTRY[key] = cls
    return cls


def get_model(name: str) -> type[Model]:
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; have {sorted(_REGISTRY)}") from None


def all_models() -> dict[str, type[Model]]:
    return dict(_REGISTRY)
