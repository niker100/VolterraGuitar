"""End-to-end ngspice tests (skipped if the shared library is unavailable)."""

from __future__ import annotations

import numpy as np
import pytest

from vguitar.circuits import get_circuit
from vguitar.circuits.base import Circuit, ControlSpec
from vguitar.config import SimConfig

SR = 44_100


class _ParamDiode(Circuit):
    """Diode clipper with a netlist-mode tone cap + a pregain drive (test only)."""

    name = "paramdiode_spice_test"
    controls = (
        ControlSpec("drive", "continuous", 0.8, 2.0, 1.0, "pregain"),
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


def _ngspice_ok() -> bool:
    try:
        from vguitar.spice.runner import simulate

        t = np.arange(64) / SR
        simulate(get_circuit("diode"), (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), SR)
        return True
    except Exception:
        return False


HAVE_NGSPICE = _ngspice_ok()
pytestmark = pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice shared library not available")


def test_diode_clips_and_distorts() -> None:
    from vguitar.metrics import thd
    from vguitar.spice.runner import simulate

    t = np.arange(int(0.02 * SR)) / SR
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = simulate(get_circuit("diode"), x, SR, SimConfig())

    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
    # Anti-parallel diodes clamp the output well below the 1 V input peak.
    assert np.max(np.abs(y)) < np.max(np.abs(x))

    distortion = thd(
        lambda xx: simulate(get_circuit("diode"), xx, SR, SimConfig()),
        sr=SR,
        amp=1.0,
        dur_s=0.05,
    )
    assert distortion > 0.01


def test_make_drive_dataset_schema() -> None:
    from vguitar.spice.runner import make_drive_dataset

    ds = make_drive_dataset(get_circuit("diode"), [0.5, 1.0], seg_dur_s=0.15, seed=0)
    assert ds.n_controls == 1
    assert ds.control_names == ["drive"]
    assert ds.controls is not None and ds.controls.shape == (len(ds), 1)
    assert ds.name == "diode_drive"
    assert ds.meta["control"] == "drive"
    assert ds.meta["drive_values"] == [0.5, 1.0]
    # Each segment is held at its drive value.
    assert set(np.unique(ds.controls[:, 0]).tolist()) == {0.5, 1.0}


def test_make_control_dataset_two_axis_roundtrip() -> None:
    from vguitar.spice.runner import make_control_dataset
    from vguitar.spice.sampling import control_grid

    circ = _ParamDiode()
    specs = list(circ.controls)
    grid = control_grid(specs, n_axis=2, mode="factorial")  # (4, 2)
    ds = make_control_dataset(circ, grid, specs, seg_dur_s=0.12, seed=0)

    assert ds.n_controls == 2
    assert ds.control_names == ["drive", "cap"]
    assert ds.control_kinds == ["continuous", "continuous"]
    assert ds.controls is not None and ds.controls.shape == (len(ds), 2)
    assert np.all(np.isfinite(ds.y))
    # All four (drive, cap) combinations are present as held segments.
    combos = {tuple(np.round(r, 12)) for r in np.unique(ds.controls, axis=0)}
    assert len(combos) == 4


def test_simulate_params_change_output() -> None:
    from vguitar.spice.runner import simulate

    circ = _ParamDiode()
    t = np.arange(int(0.02 * SR)) / SR
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y_small = simulate(circ, x, SR, params={"cap": 1e-9})
    y_large = simulate(circ, x, SR, params={"cap": 3e-8})
    assert len(y_small) == len(x) and len(y_large) == len(x)
    # A 30x larger cap rolls off highs differently -> outputs must differ.
    assert not np.allclose(y_small, y_large, atol=1e-4)
