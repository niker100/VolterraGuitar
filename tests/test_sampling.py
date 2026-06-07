"""Pure (no-ngspice) tests for control-grid sampling + parameterized netlists."""

from __future__ import annotations

import numpy as np
import pytest

from vguitar.circuits.base import Circuit, ControlSpec
from vguitar.spice.sampling import control_grid, holdout_grid, nn_distance

DRIVE = ControlSpec("drive", "continuous", 0.5, 4.0, 1.0, "pregain")
TONE = ControlSpec("tone", "continuous", 1e-9, 3e-8, 1e-8, "netlist")
MODE = ControlSpec("mode", "discrete", 0, 2, 0, "netlist")


def test_factorial_shape_and_corners() -> None:
    grid = control_grid([DRIVE, TONE], n_axis=4, mode="factorial")
    assert grid.shape == (16, 2)
    # Axis ranges are spanned exactly at the corners.
    assert grid[:, 0].min() == pytest.approx(0.5)
    assert grid[:, 0].max() == pytest.approx(4.0)
    assert grid[:, 1].min() == pytest.approx(1e-9)
    assert grid[:, 1].max() == pytest.approx(3e-8)


def test_auto_switches_to_sobol_for_three_axes() -> None:
    # 5**3 = 125 > default budget 36 -> Sobol with the budget count.
    grid = control_grid([DRIVE, TONE, MODE], n_axis=5, budget=24, mode="auto")
    assert grid.shape == (24, 3)


def test_factorial_budget_cap_raises() -> None:
    with pytest.raises(ValueError, match="budget"):
        control_grid([DRIVE, TONE], n_axis=8, budget=10, mode="factorial")


def test_discrete_axis_is_integer_valued() -> None:
    grid = control_grid([MODE], n_axis=3, mode="factorial")
    assert np.all(grid == np.round(grid))
    assert grid.min() >= 0 and grid.max() <= 2


def test_sobol_is_deterministic_in_seed() -> None:
    a = control_grid([DRIVE, TONE], budget=16, mode="sobol", seed=7)
    b = control_grid([DRIVE, TONE], budget=16, mode="sobol", seed=7)
    c = control_grid([DRIVE, TONE], budget=16, mode="sobol", seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_nn_distance_and_holdout_stay_away_from_trained() -> None:
    trained = control_grid([DRIVE, TONE], n_axis=3, mode="factorial")
    held = holdout_grid([DRIVE, TONE], trained, n=3, seed=2, min_dist=0.05)
    assert held.shape[0] >= 1 and held.shape[1] == 2
    # Every held-out point is at least min_dist from all trained points.
    assert np.all(nn_distance(held, trained, [DRIVE, TONE]) >= 0.05)


def test_nn_distance_infinite_when_no_trained() -> None:
    pts = control_grid([DRIVE], n_axis=4, mode="factorial")
    d = nn_distance(pts, np.empty((0, 1), np.float32), [DRIVE])
    assert np.all(np.isinf(d))


# --- parameterized netlists ------------------------------------------------
class _ParamDiode(Circuit):
    """A diode clipper with a netlist-mode tone cap + a pregain drive (test only)."""

    name = "paramdiode_test"
    controls = (
        ControlSpec("drive", "continuous", 0.5, 2.0, 1.0, "pregain"),
        ControlSpec("cap", "continuous", 1e-9, 3e-8, 1e-8, "netlist"),
    )
    _TEMPLATE = (
        "* test param diode\n"
        "Vin in 0 dc 0\n"
        "R1 in out 1k\n"
        "D1 out 0 D1N4148\n"
        "D2 0 out D1N4148\n"
        "C1 out 0 {cap:g}\n"
        ".model D1N4148 D(IS=2.52n N=1.752 RS=0.568)\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        return self._TEMPLATE.format(**self._resolve_netlist_params(params))


def test_netlist_for_default_uses_declared_default() -> None:
    nl = _ParamDiode().netlist()
    assert "C1 out 0 1e-08" in nl  # the declared cap default


def test_netlist_for_substitutes_value() -> None:
    nl = _ParamDiode().netlist_for({"cap": 2.2e-8})
    assert "C1 out 0 2.2e-08" in nl


def test_netlist_for_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown netlist control"):
        _ParamDiode().netlist_for({"bogus": 1.0})


def test_base_netlist_for_default_ignores_params() -> None:
    # The two shipped circuits have no netlist-mode controls -> params ignored.
    from vguitar.circuits import get_circuit

    diode = get_circuit("diode")
    assert diode.netlist_for(None) == diode.netlist()
    assert diode.netlist_for({}) == diode.netlist()


# --- DI window helper (pure) -----------------------------------------------
def test_di_window_unit_peak_and_length() -> None:
    from vguitar.spice.runner import _di_window

    di = (0.4 * np.sin(2 * np.pi * np.arange(2000) / 53.0)).astype(np.float32)
    w = _di_window(di, 256, offset_seed=3)
    assert w.shape == (256,)
    assert abs(float(np.max(np.abs(w))) - 1.0) < 1e-5  # re-peak-normalized


def test_di_window_deterministic_and_offset_varies() -> None:
    from vguitar.spice.runner import _di_window

    di = (0.4 * np.sin(2 * np.pi * np.arange(2000) / 53.0)).astype(np.float32)
    assert np.array_equal(_di_window(di, 256, 3), _di_window(di, 256, 3))
    assert not np.array_equal(_di_window(di, 256, 3), _di_window(di, 256, 9))


def test_di_window_tiles_when_short() -> None:
    from vguitar.spice.runner import _di_window

    di = np.linspace(-1, 1, 100, dtype=np.float32)
    w = _di_window(di, 350, offset_seed=0)  # n > len(di) -> tile
    assert w.shape == (350,) and np.all(np.isfinite(w))
