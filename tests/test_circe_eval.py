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

from vguitar.benchmark.circe_eval import (
    _as_control_rows,
    _dbfs,
    _interp_stats,
    _rtf_moving,
    _split_control,
    _stability_checks,
)
from vguitar.circuits import get_circuit
from vguitar.circuits.base import ControlSpec
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
    # Steady-state tail (drop the DC-blocker's sub-audio settle transient).
    quiet = _dbfs(trained.process(np.zeros(SR, np.float32), c)[SR // 2 :])
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


# --- no-ngspice: multi-control (vector) helper paths -----------------------
def test_as_control_rows_floats_and_vectors() -> None:
    rows = _as_control_rows([0.5, 1.0, 2.0])
    assert len(rows) == 3 and all(r.shape == (1,) for r in rows)
    rows2 = _as_control_rows([[0.5, 0.1], [1.0, 0.9]])
    assert len(rows2) == 2 and all(r.shape == (2,) for r in rows2)


def test_split_control_pregain_and_netlist() -> None:
    specs = [
        ControlSpec("drive", "continuous", 0.0, 1.0, 0.2, "pregain"),
        ControlSpec("tone", "continuous", 0.0, 1.0, 0.5, "netlist"),
    ]
    g, params = _split_control(np.array([0.4, 0.7]), specs)
    assert g == pytest.approx(0.4)
    assert params == {"tone": pytest.approx(0.7)}
    # specs=None -> every column is a pre-gain (product), no netlist params.
    g2, params2 = _split_control(np.array([0.4, 0.5]), None)
    assert g2 == pytest.approx(0.2) and params2 is None


def test_interp_stats_vector_path() -> None:
    specs = [
        ControlSpec("drive", "continuous", 0.0, 1.0, 0.2, "pregain"),
        ControlSpec("tone", "continuous", 0.0, 1.0, 0.5, "netlist"),
    ]
    trained = np.array([[0.0, 0.0], [1.0, 1.0]], np.float32)
    rows = [
        {"control": np.array([0.0, 0.0], np.float32), "held": False, "esr": 0.10},
        {"control": np.array([1.0, 1.0], np.float32), "held": False, "esr": 0.10},
        {"control": np.array([0.5, 0.5], np.float32), "held": True, "esr": 0.30},
    ]
    s = _interp_stats(rows, trained, key=None, specs=specs)
    assert s["held_worst"] == pytest.approx(0.30)
    assert s["per_held"][0]["dist"] == pytest.approx(np.sqrt(0.5**2 + 0.5**2))
    assert "control" in s["per_held"][0]


def test_stability_checks_vector_rows(trained: CIRCE) -> None:
    # The 1-control fixture still works when fed (S, 1) control rows.
    stab = _stability_checks(trained, np.array([[0.5], [2.0], [4.0]], np.float32), sr=SR)
    assert stab["hot_finite"]
    assert len(stab["zero"]) == 3 and "control" in stab["zero"][0]


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
