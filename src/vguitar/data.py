"""The :class:`Dataset` contract: aligned input/target audio at one sample rate.

A dataset is two mono float32 arrays of equal length plus the sample rate.
Every model trains and is evaluated on this. Generated datasets are stored as
``.npz`` so they are reproducible and cheap to load.

Conditioned / parametric models (see ``docs/CIRCE-design.md``) also need
*exogenous controls* — continuous pots, discrete switches, slow-drift
temperature/supply-sag — that vary the circuit's behaviour. A dataset may
therefore carry an optional ``controls`` array of shape ``(N, C)`` aligned
sample-for-sample with ``x``/``y``, named by ``control_names`` and (optionally)
typed by ``control_kinds``.

The in-memory representation is intentionally *dense*: every sample has its own
control row, so :meth:`Dataset.slice` / :meth:`Dataset.split` index it exactly
like ``x``/``y`` and models receive one uniform array regardless of control
kind. Piecewise-constant controls (the common SPICE-sweep case, where each
audio segment is rendered at one fixed setting) cost almost nothing on disk:
``np.savez_compressed`` collapses the long constant runs to roughly their
run-length-encoded size. :meth:`Dataset.from_segments` builds the dense array
from a compact per-segment spec for exactly that case.

Backward compatibility: control-free ``.npz`` files written before this field
existed load fine — ``controls`` simply defaults to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Allowed values for :attr:`Dataset.control_kinds`. ``continuous`` = a pot/knob
#: swept over a range; ``discrete`` = a switch/selector (integer-valued, stored
#: as float); ``drift`` = a slow sub-Hz exogenous variable (temperature, supply
#: sag) the model treats with a bounded-pole path.
CONTROL_KINDS: frozenset[str] = frozenset({"continuous", "discrete", "drift"})


@dataclass
class Dataset:
    """Aligned ``(x, y)`` signals, optionally conditioned on exogenous controls.

    Attributes:
        x: input signal, shape ``(N,)``, float32. The guitar/DI signal fed into
            the circuit (already scaled to volts at the circuit input).
        y: target output signal, shape ``(N,)``, float32. The circuit's response
            at node ``out``, anti-alias-decimated to ``sr``.
        sr: sample rate in Hz (canonical: ``vguitar.AUDIO_SR``).
        name: human-readable label (e.g. ``"bjt"``).
        meta: free-form provenance (circuit netlist hash, excitation spec, ...).
        controls: optional exogenous control matrix, shape ``(N, C)``, float32,
            aligned sample-for-sample with ``x``/``y``. ``None`` for an
            unconditioned dataset. A 1-D ``(N,)`` array is accepted and treated
            as a single control column.
        control_names: names of the ``C`` control columns. Defaults to
            ``["c0", "c1", ...]`` when ``controls`` is given without names.
            ``None`` iff ``controls`` is ``None``.
        control_kinds: optional per-column kind, each in :data:`CONTROL_KINDS`.
            ``None`` leaves the kind unspecified (models may then assume
            ``continuous``).
    """

    x: np.ndarray
    y: np.ndarray
    sr: int
    name: str = "dataset"
    meta: dict[str, Any] = field(default_factory=dict)
    controls: np.ndarray | None = None
    control_names: list[str] | None = None
    control_kinds: list[str] | None = None

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float32).reshape(-1)
        self.y = np.ascontiguousarray(self.y, dtype=np.float32).reshape(-1)
        if self.x.shape != self.y.shape:
            raise ValueError(f"x/y length mismatch: {self.x.shape} vs {self.y.shape}")
        self._validate_controls()

    def _validate_controls(self) -> None:
        if self.controls is None:
            # No controls => no per-column metadata may dangle.
            if self.control_names is not None or self.control_kinds is not None:
                raise ValueError("control_names/control_kinds given but controls is None")
            return

        controls = np.ascontiguousarray(self.controls, dtype=np.float32)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
        elif controls.ndim != 2:
            raise ValueError(f"controls must be 1-D or 2-D, got {controls.ndim}-D")
        if controls.shape[1] == 0:
            raise ValueError("controls has zero columns; use controls=None instead")
        if controls.shape[0] != len(self):
            raise ValueError(
                f"controls length mismatch: {controls.shape[0]} rows vs {len(self)} samples"
            )
        self.controls = controls
        n_cols = controls.shape[1]

        if self.control_names is None:
            self.control_names = [f"c{i}" for i in range(n_cols)]
        else:
            self.control_names = [str(s) for s in self.control_names]
            if len(self.control_names) != n_cols:
                raise ValueError(
                    f"control_names has {len(self.control_names)} entries "
                    f"but controls has {n_cols} columns"
                )

        if self.control_kinds is not None:
            self.control_kinds = [str(s) for s in self.control_kinds]
            if len(self.control_kinds) != n_cols:
                raise ValueError(
                    f"control_kinds has {len(self.control_kinds)} entries "
                    f"but controls has {n_cols} columns"
                )
            bad = sorted(set(self.control_kinds) - CONTROL_KINDS)
            if bad:
                raise ValueError(
                    f"unknown control_kinds {bad}; allowed: {sorted(CONTROL_KINDS)}"
                )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def duration_s(self) -> float:
        return len(self) / self.sr

    @property
    def n_controls(self) -> int:
        """Number of control columns ``C`` (0 if unconditioned)."""
        return 0 if self.controls is None else int(self.controls.shape[1])

    @classmethod
    def from_segments(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        sr: int,
        boundaries: list[int] | np.ndarray,
        values: np.ndarray,
        *,
        name: str = "dataset",
        meta: dict[str, Any] | None = None,
        control_names: list[str] | None = None,
        control_kinds: list[str] | None = None,
    ) -> Dataset:
        """Build a conditioned dataset from piecewise-constant control segments.

        The natural form for a SPICE parametric sweep: the signal is the
        concatenation of ``S`` segments, each rendered at one fixed control
        setting. ``boundaries`` holds the ``S + 1`` sample offsets delimiting the
        segments (``boundaries[0] == 0``, ``boundaries[-1] == N``), and
        ``values`` is the ``(S, C)`` matrix of per-segment control rows. The
        controls are expanded to the dense ``(N, C)`` array each segment holds
        constant.

        Args:
            x: input signal, shape ``(N,)``.
            y: target signal, shape ``(N,)``.
            sr: sample rate in Hz.
            boundaries: ``S + 1`` ascending sample offsets; first must be ``0``
                and last must equal ``N``.
            values: per-segment control rows, shape ``(S, C)`` (a ``(S,)`` vector
                is treated as one column).
            name, meta, control_names, control_kinds: passed through to the
                constructor.
        """
        bounds = np.ascontiguousarray(boundaries, dtype=np.int64).reshape(-1)
        seg_values = np.ascontiguousarray(values, dtype=np.float32)
        if seg_values.ndim == 1:
            seg_values = seg_values.reshape(-1, 1)
        n = int(np.asarray(x).reshape(-1).shape[0])
        n_seg = seg_values.shape[0]
        if bounds.shape[0] != n_seg + 1:
            raise ValueError(
                f"boundaries must have S+1={n_seg + 1} entries for {n_seg} segments, "
                f"got {bounds.shape[0]}"
            )
        if bounds[0] != 0 or bounds[-1] != n:
            raise ValueError(
                f"boundaries must run 0..N; got {bounds[0]}..{bounds[-1]} for N={n}"
            )
        if np.any(np.diff(bounds) < 0):
            raise ValueError("boundaries must be non-decreasing")
        controls = np.empty((n, seg_values.shape[1]), dtype=np.float32)
        for s in range(n_seg):
            controls[bounds[s] : bounds[s + 1]] = seg_values[s]
        return cls(
            x=x,
            y=y,
            sr=sr,
            name=name,
            meta=meta or {},
            controls=controls,
            control_names=control_names,
            control_kinds=control_kinds,
        )

    def slice(self, start: int, stop: int) -> Dataset:
        controls = None if self.controls is None else self.controls[start:stop]
        return Dataset(
            self.x[start:stop],
            self.y[start:stop],
            self.sr,
            self.name,
            dict(self.meta),
            controls=controls,
            control_names=None if self.control_names is None else list(self.control_names),
            control_kinds=None if self.control_kinds is None else list(self.control_kinds),
        )

    def split(
        self, val_fraction: float = 0.1, test_fraction: float = 0.1
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Contiguous train/val/test split (no shuffling — preserves time order).

        Sequence models need contiguous audio; splitting by time also keeps the
        test set honest (the model never sees adjacent samples during training).
        Controls (if present) are sliced alongside ``x``/``y``.
        """
        n = len(self)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        n_train = n - n_val - n_test
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError("split fractions leave an empty partition")
        train = self.slice(0, n_train)
        val = self.slice(n_train, n_train + n_val)
        test = self.slice(n_train + n_val, n)
        for d, tag in ((train, "train"), (val, "val"), (test, "test")):
            d.name = f"{self.name}:{tag}"
        return train, val, test

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "x": self.x,
            "y": self.y,
            "sr": np.int64(self.sr),
            "name": self.name,
            "meta": repr(self.meta),
        }
        # Only emit control keys when present, so files stay byte-identical to
        # the unconditioned format and old readers keep working.
        if self.controls is not None:
            arrays["controls"] = self.controls
            arrays["control_names"] = np.asarray(self.control_names, dtype=str)
            if self.control_kinds is not None:
                arrays["control_kinds"] = np.asarray(self.control_kinds, dtype=str)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        with np.load(path, allow_pickle=False) as f:
            meta_raw = str(f["meta"]) if "meta" in f else "{}"
            try:
                meta = eval(meta_raw, {"__builtins__": {}}, {})
            except Exception:
                meta = {}
            # Missing control keys => pre-conditioning (or unconditioned) file.
            keys = f.files
            controls = f["controls"] if "controls" in keys else None
            control_names = (
                [str(s) for s in f["control_names"]] if "control_names" in keys else None
            )
            control_kinds = (
                [str(s) for s in f["control_kinds"]] if "control_kinds" in keys else None
            )
            return cls(
                x=f["x"],
                y=f["y"],
                sr=int(f["sr"]),
                name=str(f["name"]),
                meta=meta if isinstance(meta, dict) else {},
                controls=controls,
                control_names=control_names,
                control_kinds=control_kinds,
            )
