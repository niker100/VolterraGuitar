"""Light tests for the benchmark/compare module (pure helpers; no ngspice/GPU)."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from vguitar.benchmark.compare import _fig_generalization, _uncond
from vguitar.data import Dataset


def test_uncond_folds_drive_into_input() -> None:
    """The unconditioned view feeds the circuit's true input g*x; targets unchanged."""
    n = 400
    x = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    y = np.tanh(x).astype(np.float32)
    g = np.full((n, 1), 0.5, np.float32)
    ds = Dataset(x, y, 44_100, controls=g, control_names=["drive"], control_kinds=["continuous"])
    u = _uncond(ds)
    assert u.controls is None
    np.testing.assert_allclose(u.x, 0.5 * x, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(u.y, y)


def test_fig_generalization_returns_figure() -> None:
    drives = np.array([0.05, 0.1, 0.2, 0.4])
    c3 = np.array([0.002, 0.0018, 0.0021, 0.0025])
    tcn = np.array([0.05, 0.002, 0.06, 0.2])
    held = np.array([False, True, False, True])
    fig = _fig_generalization(drives, c3, tcn, held, g_nom=0.1, name="bjt")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    for sp in ("top", "right"):
        assert not ax.spines[sp].get_visible()
    import matplotlib.pyplot as plt

    plt.close(fig)
