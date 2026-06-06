"""Tests for the CIRCE validation suite (vguitar.benchmark.circe_eval).

The no-ngspice tests cover the validation *logic* (moving-control equivalence,
stability, interpolation stats, moving-knob RTF) on a tiny trained CIRCE. The
ngspice-gated tests exercise the end-to-end pipeline pieces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from vguitar.benchmark.circe_eval import _dbfs, _interp_stats, _rtf_moving, _stability_checks
from vguitar.circuits import get_circuit
from vguitar.config import Config, Paths, TrainConfig
from vguitar.data import Dataset
from vguitar.models.base import check_streaming_moving
from vguitar.models.circe import CIRCE

SR = 8000


def _ngspice_ok() -> bool:
    try:
        from vguitar.spice.runner import simulate

        t = np.arange(64) / 44_100
        simulate(get_circuit("diode"), (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), 44_100)
        return True
    except Exception:
        return False


HAVE_NGSPICE = _ngspice_ok()


def _seg(g: float, n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = (0.6 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)
    lp = lfilter([0.3], [1.0, -0.7], x)
    return x, np.tanh(g * 2.0 * lp).astype(np.float32)


@pytest.fixture(scope="module")
def trained() -> CIRCE:
    torch.manual_seed(0)
    grid = [0.5, 1.0, 2.0, 4.0]
    xs, ys, vals, bounds = [], [], [], [0]
    for j, g in enumerate(grid):
        x, y = _seg(g, seed=j)
        xs.append(x)
        ys.append(y)
        vals.append([g])
        bounds.append(bounds[-1] + len(x))
    ds = Dataset.from_segments(
        np.concatenate(xs), np.concatenate(ys), SR, bounds, np.asarray(vals, np.float32),
        control_names=["drive"], control_kinds=["continuous"],
    )
    tr, va, _ = ds.split(0.15, 0.15)
    m = CIRCE(n_control=1, channels=8, n_blocks=2, n_layers=6)
    m.fit(tr, va, TrainConfig(epochs=12, seq_len=1024, batch_size=16, lr=3e-3, warmup=128))
    return m


# --- no-ngspice: validation logic -----------------------------------------
def test_moving_control_streaming_equivalence(trained: CIRCE) -> None:
    assert check_streaming_moving(trained, n=4096, block=128, atol=2e-3) < 2e-3


def test_moving_control_per_sample(trained: CIRCE) -> None:
    # smaller block vs per-sample re-blocking under a moving knob
    assert check_streaming_moving(trained, n=2048, block=64, atol=2e-3) < 2e-3


def test_zero_input_is_quiet(trained: CIRCE) -> None:
    c = np.array([2.0], np.float32)
    driven = _dbfs(trained.process(_seg(2.0, seed=7)[0], c))
    quiet = _dbfs(trained.process(np.zeros(SR, np.float32), c))
    assert np.isfinite(quiet) and quiet < driven - 20.0  # >=20 dB below a driven signal


def test_hot_input_saturates(trained: CIRCE) -> None:
    stab = _stability_checks(trained, [0.5, 2.0, 4.0], sr=SR)
    assert stab["hot_finite"]
    bound = stab["out_bound"] * (1 + 1e-4)
    assert stab["hot_offline_peak"] <= bound
    assert stab["hot_stream_peak"] <= bound


def test_interp_stats_worstcase() -> None:
    rows = [
        {"drive": 0.5, "held": False, "esr": 0.10},
        {"drive": 1.0, "held": False, "esr": 0.10},
        {"drive": 0.75, "held": True, "esr": 0.30},
        {"drive": 0.90, "held": True, "esr": 0.20},
    ]
    s = _interp_stats(rows, [0.5, 1.0])
    assert s["held_worst"] == pytest.approx(0.30)
    assert s["held_mean"] == pytest.approx(0.25)
    # 0.75 is 0.25 from the nearest grid point; span 0.5 -> normalized distance 0.5
    assert any(p["dist"] == pytest.approx(0.5) for p in s["per_held"])


def test_rtf_moving_runs(trained: CIRCE) -> None:
    r = _rtf_moving(trained, [0.5, 1.0, 2.0, 4.0], sr=SR, block=128, dur_s=0.5)
    assert np.isfinite(r["rtf"]) and r["rtf"] > 0


# --- ngspice-gated: end-to-end pieces --------------------------------------
pytestmark_spice = pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice unavailable")


@pytestmark_spice
def test_di_eval_finite(trained: CIRCE) -> None:
    from vguitar.benchmark.circe_eval import _eval_on_di

    rows = _eval_on_di(trained, get_circuit("bjt"), [0.02], sr=44_100, dur_s=0.2)
    assert rows and np.isfinite(rows[0]["esr"]) and np.isfinite(rows[0]["stft"])


@pytestmark_spice
def test_ab_render_writes_wavs(trained: CIRCE, tmp_path) -> None:
    from vguitar.benchmark.circe_eval import _render_ab

    wavs = _render_ab(trained, get_circuit("bjt"), [0.05], sr=44_100, outdir=tmp_path, dur_s=0.2)
    assert len(wavs) == 3  # di + circuit + circe for the one setting
    assert all(str(p).endswith(".wav") and Path(p).exists() for p in wavs)


@pytestmark_spice
def test_run_circe_eval_smoke(tmp_path) -> None:
    from vguitar.benchmark.circe_eval import run_circe_eval

    cfg = Config(paths=Paths(data=tmp_path / "data", outputs=tmp_path / "out", runs=tmp_path / "runs"))
    rows = run_circe_eval(
        "bjt", drives=[0.02, 0.08], held=[0.04], cfg=cfg, retrain=True, regen=True,
        seg_dur_s=0.3, epochs=3, channels=6, eval_di=False, render=False, heatmap=False,
    )
    assert len(rows) == 3
    assert (tmp_path / "out" / "figs" / "bjt_circe_esr.png").exists()
