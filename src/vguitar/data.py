"""The :class:`Dataset` contract: aligned input/target audio at one sample rate.

A dataset is just two mono float32 arrays of equal length plus the sample rate.
Every model trains and is evaluated on this. Generated datasets are stored as
``.npz`` so they are reproducible and cheap to load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """Aligned ``(x, y)`` signals.

    Attributes:
        x: input signal, shape ``(N,)``, float32. The guitar/DI signal fed into
            the circuit (already scaled to volts at the circuit input).
        y: target output signal, shape ``(N,)``, float32. The circuit's response
            at node ``out``, anti-alias-decimated to ``sr``.
        sr: sample rate in Hz (canonical: ``vguitar.AUDIO_SR``).
        name: human-readable label (e.g. ``"bjt"``).
        meta: free-form provenance (circuit netlist hash, excitation spec, ...).
    """

    x: np.ndarray
    y: np.ndarray
    sr: int
    name: str = "dataset"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float32).reshape(-1)
        self.y = np.ascontiguousarray(self.y, dtype=np.float32).reshape(-1)
        if self.x.shape != self.y.shape:
            raise ValueError(f"x/y length mismatch: {self.x.shape} vs {self.y.shape}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def duration_s(self) -> float:
        return len(self) / self.sr

    def slice(self, start: int, stop: int) -> Dataset:
        return Dataset(self.x[start:stop], self.y[start:stop], self.sr, self.name, dict(self.meta))

    def split(
        self, val_fraction: float = 0.1, test_fraction: float = 0.1
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Contiguous train/val/test split (no shuffling — preserves time order).

        Sequence models need contiguous audio; splitting by time also keeps the
        test set honest (the model never sees adjacent samples during training).
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
        np.savez_compressed(
            path, x=self.x, y=self.y, sr=np.int64(self.sr), name=self.name, meta=repr(self.meta)
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        with np.load(path, allow_pickle=False) as f:
            meta_raw = str(f["meta"]) if "meta" in f else "{}"
            try:
                meta = eval(meta_raw, {"__builtins__": {}}, {})
            except Exception:
                meta = {}
            return cls(
                x=f["x"],
                y=f["y"],
                sr=int(f["sr"]),
                name=str(f["name"]),
                meta=meta if isinstance(meta, dict) else {},
            )
