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
from collections.abc import Callable
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
    #: True for models that require an exogenous-control dataset (see CIRCE);
    #: such models are skipped by the unconditioned benchmark / generic tests.
    conditioned: ClassVar[bool] = False

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


def check_streaming_moving(
    model: Any,
    control_traj: np.ndarray | Callable[[int], np.ndarray] | None = None,
    n: int = 8192,
    block: int = 128,
    seed: int = 0,
    atol: float = 2e-3,
) -> float:
    """Assert conditioned streaming is correct under a MOVING control (GATE-4).

    ``check_streaming`` only exercises a constant/default control. Here the
    control changes per block (a knob being turned), and we verify the output is
    **invariant to block size**: streaming at ``block`` must equal streaming
    sample-by-sample (``block=1``) given the *same* per-block control schedule.
    Because the conv state-carry is exact and FiLM is piecewise-constant per
    block, the two agree to floating-point tolerance — proving the live engine
    can re-block freely and move the knob mid-stream without artifacts.

    Args:
        model: a conditioned model whose ``process_block(x, c)`` accepts a control.
        control_traj: ``(n_blocks, K)`` per-block control, or a
            ``callable(block_index) -> c``; ``None`` uses a smooth 0->1 ramp
            across the model's ``n_control`` controls.
        n, block, seed, atol: signal length, block size, RNG seed, tolerance.

    Returns:
        Max abs difference between the block and per-sample streamed outputs.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32) * 0.3
    n_blocks = (n + block - 1) // block
    k = int(getattr(model, "n_control", 1))
    if control_traj is None:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_blocks))  # smooth 0->1
        traj = np.repeat(ramp[:, None], k, axis=1).astype(np.float32)
    elif isinstance(control_traj, np.ndarray):
        traj = np.asarray(control_traj, dtype=np.float32)
        if traj.ndim == 1:
            traj = traj[:, None]
    else:  # callable(block_index) -> control vector
        traj = np.asarray([np.atleast_1d(control_traj(i)) for i in range(n_blocks)], dtype=np.float32)

    model.reset()
    y_block = np.concatenate(
        [
            np.asarray(model.process_block(x[i * block : (i + 1) * block], traj[i]), dtype=np.float32)
            for i in range(n_blocks)
        ]
    )[:n]
    model.reset()  # per-sample, but reusing each block's control (so the schedule matches)
    y_sample = np.concatenate(
        [np.asarray(model.process_block(x[j : j + 1], traj[j // block]), dtype=np.float32) for j in range(n)]
    )[:n]
    err = float(np.max(np.abs(y_block - y_sample))) if n else float("inf")
    if not np.isfinite(err) or err > atol:
        raise AssertionError(
            f"{model.name}: moving-control streaming mismatch (max abs err {err:.2e} > {atol:.0e})"
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


# --- device selection -----------------------------------------------------
def pick_device(prefer: str = "auto") -> str:
    """Choose a TRAINING device: ``"cuda"`` if available, else ``"cpu"``.

    Training is the only place a GPU helps here — the real-time deployment target
    (streaming ``process_block`` / RTF) is always CPU, so :func:`to_inference_cpu`
    moves a model back to the CPU after training. ``prefer="cpu"`` forces CPU
    (e.g. for honest RTF or when reproducing CPU numbers).
    """
    if prefer == "cpu":
        return "cpu"
    try:
        import torch

        if prefer in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def to_inference_cpu(model: Any) -> Any:
    """Move a (possibly GPU-trained) model to the CPU for honest inference/RTF.

    The benchmark's real-time factor and streaming equivalence must reflect the
    CPU deployment target, and some models stream via torch on their device. After
    training on the GPU, call this so ``process`` / ``process_block`` / RTF all run
    on the CPU. No-op for models already on CPU or without a torch device.
    """
    dev = getattr(model, "device", None)
    if dev is not None and str(dev) != "cpu":
        import torch

        model.device = torch.device("cpu")
        net = getattr(model, "net", None)
        if net is not None:
            net.to("cpu")
        if hasattr(model, "reset"):
            model.reset()
    return model
