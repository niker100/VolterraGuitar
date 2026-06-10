"""Sanity tests for the non-audio Duffing testbed."""

from __future__ import annotations

import numpy as np

from vguitar.models.archive.systems.duffing import make_duffing_dataset, simulate_duffing


def test_simulate_shape_and_finite() -> None:
    rng = np.random.default_rng(0)
    u = (rng.standard_normal(2000) * 0.3).astype(np.float32)
    y = simulate_duffing(u, 44_100, f0=200.0, zeta=0.3, beta=1.0)
    assert y.shape == u.shape
    assert np.all(np.isfinite(y))
    assert y.dtype == np.float32


def test_beta_zero_is_linear_beta_positive_is_not() -> None:
    """beta=0 => linear (superposition: 2*forcing -> ~2*response). beta>0 => not."""
    rng = np.random.default_rng(1)
    u = (rng.standard_normal(4000) * 0.3).astype(np.float32)
    w = 2000  # skip startup transient
    # Linear case: response scales with input.
    y1 = simulate_duffing(u, 44_100, beta=0.0)[w:]
    y2 = simulate_duffing(2.0 * u, 44_100, beta=0.0)[w:]
    lin_err = np.max(np.abs(y2 - 2.0 * y1)) / (np.max(np.abs(2.0 * y1)) + 1e-9)
    assert lin_err < 1e-3, f"beta=0 should be linear, got rel err {lin_err:.2e}"
    # Nonlinear case: the cubic breaks superposition appreciably.
    z1 = simulate_duffing(u, 44_100, beta=3.0)[w:]
    z2 = simulate_duffing(2.0 * u, 44_100, beta=3.0)[w:]
    nl_err = np.max(np.abs(z2 - 2.0 * z1)) / (np.max(np.abs(2.0 * z1)) + 1e-9)
    assert nl_err > 0.05, f"beta>0 should be nonlinear, got rel err {nl_err:.2e}"


def test_make_duffing_dataset_shapes() -> None:
    from vguitar.circuits.base import ControlSpec

    specs = [
        ControlSpec("amp", "continuous", 0.5, 3.0, 1.0, "pregain"),
        ControlSpec("beta", "continuous", 0.0, 3.0, 1.0, "netlist"),
    ]
    grid = np.array([[1.0, 0.0], [2.0, 1.5], [3.0, 3.0]], dtype=np.float32)
    ds = make_duffing_dataset(grid, specs, seg_dur_s=0.2, seed=0)
    assert ds.controls is not None
    assert ds.n_controls == 2
    assert ds.control_names == ["amp", "beta"]
    assert len(ds) == len(ds.x) == len(ds.y)
    # three constant-control segments
    from vguitar.models.archive.circe3 import _segments

    assert len(_segments(ds.controls)) == 3
