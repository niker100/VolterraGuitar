"""Contract + conditioning tests for CIRCE (the conditioned model)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.metrics import esr
from vguitar.models.base import check_streaming
from vguitar.models.circe import CIRCE

SR = 8000


def _seg(g: float, n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A drive-conditioned segment: target = tanh(g * 2 * lowpass(x))."""
    x = (0.6 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)
    lp = lfilter([0.3], [1.0, -0.7], x)
    return x, np.tanh(g * 2.0 * lp).astype(np.float32)


@pytest.fixture(scope="module")
def drive_ds() -> Dataset:
    grid = [0.5, 1.0, 2.0, 4.0]
    xs, ys, vals, bounds = [], [], [], [0]
    for j, g in enumerate(grid):
        x, y = _seg(g, seed=j)
        xs.append(x)
        ys.append(y)
        vals.append([g])
        bounds.append(bounds[-1] + len(x))
    return Dataset.from_segments(
        np.concatenate(xs), np.concatenate(ys), SR, bounds, np.asarray(vals, np.float32),
        control_names=["drive"], control_kinds=["continuous"],
    )


@pytest.fixture(scope="module")
def trained(drive_ds: Dataset) -> CIRCE:
    torch.manual_seed(0)
    tr, va, _ = drive_ds.split(0.15, 0.15)
    m = CIRCE(n_control=1, channels=8, n_blocks=2, n_layers=6)
    m.fit(tr, va, TrainConfig(epochs=20, seq_len=1024, batch_size=16, lr=3e-3, warmup=128))
    return m


def test_streaming_equivalence_default_control(trained: CIRCE) -> None:
    # The generic contract (process == streamed process_block) at the default control.
    assert check_streaming(trained, n=4096, block=128, atol=2e-3) < 2e-3


def test_streaming_equivalence_at_a_control(trained: CIRCE) -> None:
    x, _ = _seg(1.3, seed=99)
    c = np.array([1.3], np.float32)
    trained.reset()
    y_off = trained.process(x, c)
    trained.reset()
    y_st = np.concatenate([trained.process_block(x[i : i + 128], c) for i in range(0, len(x), 128)])
    assert np.max(np.abs(y_off - y_st[: len(x)])) < 2e-3


def test_save_load_roundtrip(trained: CIRCE, tmp_path) -> None:
    path = tmp_path / "circe.model"
    trained.save(path)
    reloaded = CIRCE.load(path)
    x, _ = _seg(1.7, seed=7)
    c = np.array([1.7], np.float32)
    assert np.max(np.abs(trained.process(x, c) - reloaded.process(x, c))) < 1e-5


def test_interpolates_to_unseen_controls(trained: CIRCE) -> None:
    # g = 0.75 and 3.0 are NOT in the training grid [0.5, 1, 2, 4].
    for g in (0.75, 3.0):
        x, y = _seg(g, seed=500 + int(g * 10))
        e = esr(y, trained.process(x, np.array([g], np.float32)))
        assert np.isfinite(e) and e < 0.6  # well below the ESR=1 trivial floor


def test_requires_controls() -> None:
    rng = np.random.default_rng(0)
    plain = Dataset(rng.standard_normal(2000).astype(np.float32),
                    rng.standard_normal(2000).astype(np.float32), SR)
    with pytest.raises(ValueError, match="control"):
        CIRCE(n_control=1).fit(plain)
