"""Tests for the fixed-point cross-method shootout."""

from __future__ import annotations

import numpy as np
import pytest

from vguitar.benchmark.shootout import _predict, _seg_metrics, _uncond
from vguitar.data import Dataset


def _ngspice_ok() -> bool:
    try:
        from vguitar.circuits import get_circuit
        from vguitar.spice.runner import simulate

        t = np.arange(64) / 44_100
        simulate(get_circuit("diode"), (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), 44_100)
        return True
    except Exception:
        return False


HAVE_NGSPICE = _ngspice_ok()


# --- pure helpers ----------------------------------------------------------
def test_uncond_collapses_control_to_input() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    y = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    ds = Dataset.from_segments(x, y, 8000, [0, 2, 4], np.array([[0.5], [2.0]], np.float32),
                               control_names=["drive"])
    u = _uncond(ds)
    assert u.controls is None
    np.testing.assert_allclose(u.x, [0.5, 1.0, 6.0, 8.0])  # x * per-segment control


def test_predict_conditioned_vs_unconditioned_match_on_identity() -> None:
    # A trivial "model" that returns its input: conditioned path divides by g then
    # the (fake) model would see dry; here we just check the routing/shape.
    class _Id:
        def process(self, x, c=None):  # echoes its input; c is accepted but unused
            return np.asarray(x, np.float32)

    ci = np.array([0.4, 0.8], np.float32)
    uncond = _predict(_Id(), ci, 2.0, conditioned=False)
    cond = _predict(_Id(), ci, 2.0, conditioned=True)
    np.testing.assert_allclose(uncond, ci)          # raw input voltage
    np.testing.assert_allclose(cond, ci / 2.0)      # dry = circuit_input / g


def test_seg_metrics_perfect_model_is_zero() -> None:
    # Build a tiny conditioned test set; a model that reproduces the target exactly
    # should score ESR ~ 0 (per-segment, warmup-trimmed).
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.standard_normal(n).astype(np.float32)
    y = np.tanh(x).astype(np.float32)
    ds = Dataset.from_segments(x, y, 8000, [0, n], np.array([[1.0]], np.float32),
                               control_names=["drive"])

    class _Oracle:
        def process(self, xx, c=None):
            return np.tanh(np.asarray(xx, np.float32)).astype(np.float32)

    esr, stft = _seg_metrics(_Oracle(), ds, conditioned=False, warmup=128)
    assert esr < 1e-6 and stft < 1e-3


# --- ngspice-gated end-to-end smoke ---------------------------------------
@pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice unavailable")
def test_run_shootout_smoke(tmp_path) -> None:
    from vguitar.benchmark.shootout import run_shootout
    from vguitar.config import Config, Paths

    cfg = Config(paths=Paths(data=tmp_path / "data", outputs=tmp_path / "out", runs=tmp_path / "runs"))
    out = run_shootout(
        ["diode"], models=("fir", "wh"), cfg=cfg, epochs=2, seg_dur_s=0.4, di_dur_s=0.3, regen=True,
    )
    names = {r["model"] for r in out["rows"]}
    assert {"fir", "wh", "circe"} <= names           # baselines + circe all raced
    assert out["matrix"].shape == (1, 3)             # 1 circuit x (2 baselines + circe)
    assert (tmp_path / "out" / "figs" / "shootout_diode_leaderboard.png").exists()
    assert (tmp_path / "out" / "figs" / "shootout_circuit_model_esr.png").exists()
    assert (tmp_path / "out" / "audio" / "shootout" / "diode" / "circuit.wav").exists()
    assert (tmp_path / "out" / "audio" / "shootout" / "diode" / "circe.wav").exists()
