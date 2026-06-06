"""End-to-end ngspice tests (skipped if the shared library is unavailable)."""

from __future__ import annotations

import numpy as np
import pytest

from vguitar.circuits import get_circuit
from vguitar.config import SimConfig

SR = 44_100


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
