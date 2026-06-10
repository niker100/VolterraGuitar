"""Shared helpers for the experiment scripts — the consolidated boilerplate.

Every probe duplicated the same three things: a flushed timestamped file logger
(so background runs are tailable + stall-detectable, per the logging convention),
a latency-aligned held-ESR eval on a circuit's held-out test segment, and dataset
loading. They live here once.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vguitar import metrics as M
from vguitar.data import Dataset


def make_log(name: str) -> Callable[[str], None]:
    """Return ``log(msg)`` writing flushed, ``t0``-relative timestamped lines to BOTH
    stdout and ``outputs/logs/<name>.log`` (the dedicated file survives the
    background-task stdout buffering, so a run is always tailable)."""
    path = Path(f"outputs/logs/{name}.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    return log


def held_esr(model: object, test: Dataset, *, warmup: int = 2048) -> float:
    """Held-out ESR on ``test``'s first constant-control segment, latency-aligned.

    Feeds the segment's input at its control value through ``model.process``, shifts
    by ``model.latency_samples`` (oversampling group delay), and scores ESR past the
    warm-up. The single eval every probe uses to compare configs."""
    from vguitar.models.archive.circe3 import _segments

    s, e = _segments(test.controls)[0]
    x = np.ascontiguousarray(test.x[s:e], np.float32)
    y = np.ascontiguousarray(test.y[s:e], np.float32)
    g = float(test.controls[s, 0])
    pred = np.asarray(model.process(x, np.array([g], np.float32)), np.float32)  # type: ignore[attr-defined]
    lat = int(getattr(model, "latency_samples", 0))
    n = min(len(y), len(pred))
    yc, pc = y[:n], pred[:n]
    if lat > 0:
        pc = pc[lat:]
        yc = yc[: len(pc)]
    return float(M.esr(yc[warmup:], pc[warmup:]))


def load_pair(sweep: str, test: str) -> tuple[Dataset, Dataset]:
    """Load a ``(sweep, test)`` dataset pair from ``data/<name>.npz``."""
    return Dataset.load(f"data/{sweep}.npz"), Dataset.load(f"data/{test}.npz")
