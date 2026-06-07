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
    assert reloaded.dcblock_fc == trained.dcblock_fc  # hparam persisted
    x, _ = _seg(1.7, seed=7)
    c = np.array([1.7], np.float32)
    assert np.max(np.abs(trained.process(x, c) - reloaded.process(x, c))) < 1e-5


def test_dcblock_silences_zero_input(trained: CIRCE) -> None:
    """With the DC-blocker on, zero input is silent even at the hottest control
    (the drive-dependent DC offset is removed)."""
    assert trained.dcblock_fc > 0.0
    c = np.array([4.0], np.float32)
    y = trained.process(np.zeros(SR, np.float32), c)
    # After the ~1/fc settling transient the output is essentially zero.
    tail = y[SR // 2 :]
    assert float(np.sqrt(np.mean(tail**2))) < 1e-3


def test_dcblock_streaming_still_exact(trained: CIRCE) -> None:
    # The DC-blocker is applied identically offline (zero IC) and streamed (zi),
    # so process == streamed process_block still holds with it enabled.
    assert check_streaming(trained, n=4096, block=128, atol=2e-3) < 2e-3
    x, _ = _seg(2.5, seed=11)
    c = np.array([2.5], np.float32)
    trained.reset()
    y_off = trained.process(x, c)
    trained.reset()
    y_st = np.concatenate([trained.process_block(x[i : i + 96], c) for i in range(0, len(x), 96)])
    assert np.max(np.abs(y_off - y_st[: len(x)])) < 2e-3


def test_dcblock_disabled_passthrough() -> None:
    # fc<=0 disables the DC-blocker (ablation / back-compat path).
    m = CIRCE(n_control=1, channels=6, n_blocks=1, n_layers=4, dcblock_fc=0.0)
    x = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    c = np.array([1.0], np.float32)
    assert np.all(np.isfinite(m.process(x, c)))
    assert check_streaming(m, n=2048, block=64, atol=2e-3) < 2e-3


def test_invalid_saturator_rejected() -> None:
    with pytest.raises(ValueError, match="saturator"):
        CIRCE(n_control=1, saturator="bogus")


@pytest.mark.parametrize("saturator", ["clamp", "adaa1", "adaa2"])
def test_saturator_streaming_exact_and_roundtrip(drive_ds: Dataset, tmp_path, saturator: str) -> None:
    """Every output saturator keeps process == streamed process_block, and the
    choice round-trips through save/load."""
    torch.manual_seed(0)
    tr, va, _ = drive_ds.split(0.15, 0.15)
    m = CIRCE(n_control=1, channels=8, n_blocks=2, n_layers=5, saturator=saturator)
    m.fit(tr, va, TrainConfig(epochs=4, seq_len=1024, batch_size=16, lr=3e-3, warmup=128))

    assert check_streaming(m, n=4096, block=128, atol=2e-3) < 2e-3
    from vguitar.models.base import check_streaming_moving

    assert check_streaming_moving(m, n=4096, block=128, atol=2e-3) < 2e-3

    path = tmp_path / f"circe_{saturator}.model"
    m.save(path)
    reloaded = CIRCE.load(path)
    assert reloaded.saturator == saturator
    x, _ = _seg(1.7, seed=7)
    c = np.array([1.7], np.float32)
    assert np.max(np.abs(m.process(x, c) - reloaded.process(x, c))) < 1e-5


def test_interpolates_to_unseen_controls(trained: CIRCE) -> None:
    # g = 0.75 and 3.0 are NOT in the training grid [0.5, 1, 2, 4].
    for g in (0.75, 3.0):
        x, y = _seg(g, seed=500 + int(g * 10))
        e = esr(y, trained.process(x, np.array([g], np.float32)))
        assert np.isfinite(e) and e < 0.6  # well below the ESR=1 trivial floor


def test_packaged_bjt_checkpoint_loads() -> None:
    """The shipped bjt checkpoint loads and runs (skipped if not packaged)."""
    from pathlib import Path

    p = Path("assets/checkpoints/bjt.circe.model")
    if not p.exists():
        pytest.skip("packaged bjt checkpoint not present")
    m = CIRCE.load(p)
    assert m.conditioned and m.n_control == 1
    y = m.process(np.random.default_rng(0).standard_normal(2048).astype(np.float32),
                  np.array([0.08], np.float32))
    assert y.shape == (2048,) and np.all(np.isfinite(y))


def test_requires_controls() -> None:
    rng = np.random.default_rng(0)
    plain = Dataset(rng.standard_normal(2000).astype(np.float32),
                    rng.standard_normal(2000).astype(np.float32), SR)
    with pytest.raises(ValueError, match="control"):
        CIRCE(n_control=1).fit(plain)
