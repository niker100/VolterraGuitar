"""Smoke + structure tests for the fidelity figures (no ngspice/GPU needed)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from vguitar import plotting as plot

_SR = 44_100


def _tone(f: float = 1000.0, dur: float = 0.1) -> np.ndarray:
    t = np.arange(int(dur * _SR)) / _SR
    return np.sin(2 * np.pi * f * t).astype(np.float32)


def test_fig_transfer_has_residual_panel() -> None:
    x = _tone(90.0, 0.1)
    yref = np.tanh(3 * x)
    fig = plot.fig_transfer(x, yref, {"circe3": np.tanh(3 * x) * 0.98, "tcn": np.tanh(2.5 * x)})
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # main curve + residual panel
    plt.close(fig)


def test_fig_harmonics_has_error_panel() -> None:
    yref = np.tanh(4 * _tone())
    fig = plot.fig_harmonics(yref, {"circe3": np.tanh(4 * _tone()) * 0.97}, _SR)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # stack + per-harmonic error
    plt.close(fig)


def test_fig_spectrum_log_freq_and_error() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(_SR).astype(np.float32) * 0.3
    yref = np.tanh(3 * x)
    fig = plot.fig_spectrum(yref, {"circe3": np.tanh(3 * x) * 0.99}, _SR)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xscale() == "log"
    plt.close(fig)


def test_fig_spectrogram_compare_three_panels() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(_SR // 2).astype(np.float32) * 0.3
    yref = np.tanh(3 * x)
    fig = plot.fig_spectrogram_compare(yref, np.tanh(3 * x) * 0.98, _SR)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 3  # circuit, model, difference (+ colorbar)
    plt.close(fig)


def test_fig_aliasing_overlay() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(_SR).astype(np.float32) * 0.3
    yref = np.tanh(3 * x)
    fig = plot.fig_aliasing(yref, {"1x": np.tanh(3 * x), "2x": np.tanh(3 * x) * 0.99}, _SR)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_ylabel().startswith("magnitude")
    plt.close(fig)


def test_fig_transfer_family_one_panel_per_drive() -> None:
    x = _tone(90.0, 0.1)
    curves = [
        (f"{d:g}V", (d * x).astype(np.float32), np.tanh(8 * d * x),
         np.tanh(8 * d * x) * 0.98, d == 0.15)
        for d in (0.05, 0.1, 0.15, 0.2)
    ]
    fig = plot.fig_transfer_family(curves, name="bjt")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    plt.close(fig)
