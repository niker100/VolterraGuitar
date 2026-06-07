"""Tests for cross-circuit validation + the new comparison plot builders.

Pure figure-builder tests run anywhere; the run_validation smoke is ngspice-gated.
"""

from __future__ import annotations

import numpy as np
import pytest

from vguitar import plotting as plot


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


def _no_top_right(fig) -> bool:
    ax = fig.axes[0]
    return not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()


# --- pure plot builders ----------------------------------------------------
def test_fig_circuit_model_esr_returns_figure() -> None:
    m = np.array([[0.08, 0.24], [0.001, 1.02]])
    fig = plot.fig_circuit_model_esr(m, ["bjt", "diode"], ["circe", "tcn"])
    assert fig.axes and _no_top_right(fig)


def test_fig_circuit_model_esr_tolerates_nan() -> None:
    m = np.array([[0.08, np.nan], [0.001, 1.02]])
    fig = plot.fig_circuit_model_esr(m, ["bjt", "diode"], ["circe", "tcn"])
    assert fig.axes


def test_fig_cross_circuit_summary_returns_figure() -> None:
    rows = [
        {"circuit": "bjt", "trained_esr": 0.076, "held_mean": 0.088, "rtf_m": 7.7},
        {"circuit": "tube_screamer", "trained_esr": 0.10, "held_mean": 0.12, "rtf_m": 8.0},
    ]
    fig = plot.fig_cross_circuit_summary(rows)
    assert _no_top_right(fig)


def test_fig_multimodel_control_response_returns_figure() -> None:
    cv = np.array([0.1, 0.2, 0.3, 0.4])
    circ = np.array([0.1, 0.2, 0.35, 0.5])
    models = {"circe": np.array([0.1, 0.21, 0.34, 0.49]), "tcn": np.array([0.12, 0.25, 0.4, 0.55])}
    fig = plot.fig_multimodel_control_response(cv, circ, models, held_mask=np.array([False, True, False, False]))
    assert _no_top_right(fig)


def test_fig_control_esr_heatmap_returns_figure() -> None:
    vi = np.array([0.0, 0.5, 1.0])
    vj = np.array([0.0, 0.5, 1.0])
    grid = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.2], [0.3, 0.2, 0.1]])
    fig = plot.fig_control_esr_heatmap(vi, vj, grid, names=("drive", "tone"))
    assert fig.axes and not fig.axes[0].spines["top"].get_visible()


# --- ngspice-gated: end-to-end cross-circuit smoke -------------------------
@pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice unavailable")
def test_run_validation_smoke(tmp_path) -> None:
    from vguitar.benchmark.validate import run_validation
    from vguitar.config import Config, Paths

    cfg = Config(paths=Paths(data=tmp_path / "data", outputs=tmp_path / "out", runs=tmp_path / "runs"))
    out = run_validation(
        ["bjt", "diode"], models=("circe",), cfg=cfg, retrain=True, regen=True,
        seg_dur_s=0.25, epochs=2, channels=6,
    )
    assert out["circuits"] == ["bjt", "diode"]
    assert out["matrix"].shape == (2, 1)
    assert len(out["summary"]) == 2
    assert (tmp_path / "out" / "figs" / "validate_cross_circuit.png").exists()
    assert (tmp_path / "out" / "figs" / "validate_circuit_model_esr.png").exists()
