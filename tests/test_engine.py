"""Conditioned offline-render tests for the realtime engine (no audio device)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from vguitar.models.circe3 import CIRCE3
from vguitar.realtime.engine import _control_for_block, render_file


def _write_wav(path, n: int = 4000, sr: int = 8000) -> str:
    x = (0.4 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
    sf.write(path, x, sr)
    return str(path)


def _read(path) -> np.ndarray:
    y, _ = sf.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(y[:, 0], dtype=np.float32)


# --- pure control resolution ----------------------------------------------
def test_control_for_block_modes() -> None:
    assert _control_for_block(None, 0, 3) is None
    # constant vector
    np.testing.assert_array_equal(_control_for_block(np.array([0.5, 1.0]), 7, 3), [0.5, 1.0])
    # per-block schedule (clamps past the end)
    sched = np.array([[0.0], [1.0], [2.0]], np.float32)
    np.testing.assert_array_equal(_control_for_block(sched, 1, 5), [1.0])
    np.testing.assert_array_equal(_control_for_block(sched, 9, 5), [2.0])
    # callable
    np.testing.assert_array_equal(_control_for_block(lambda i: np.array([i]), 4, 5), [4.0])


# --- render_file with control ----------------------------------------------
@pytest.fixture(scope="module")
def model() -> CIRCE3:
    # Untrained is fine: render equivalence is about deterministic streaming, not accuracy.
    return CIRCE3(n_control=1, channels=6, n_blocks=1, n_layers=4)


def test_render_constant_vector_equals_callable(model: CIRCE3, tmp_path) -> None:
    inp = _write_wav(tmp_path / "in.wav")
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    render_file(model, inp, str(a), sr=8000, control=np.array([1.5], np.float32))
    render_file(model, inp, str(b), sr=8000, control=lambda _bi: np.array([1.5], np.float32))
    assert np.allclose(_read(a), _read(b), atol=1e-6)


def test_render_length_and_none_control(model: CIRCE3, tmp_path) -> None:
    inp = _write_wav(tmp_path / "in.wav", n=3333)
    out = tmp_path / "o.wav"
    render_file(model, inp, str(out), sr=8000, control=None)  # unconditioned path
    assert _read(out).shape[0] == 3333


def test_render_schedule_matches_manual_loop(model: CIRCE3, tmp_path) -> None:
    """render_file with a knob automation == a manual block loop feeding the same
    per-block control to process_block (the wiring is exact)."""
    inp = _write_wav(tmp_path / "in.wav")
    x = _read(inp)  # 8 kHz file, no resample

    def sweep(bi: int) -> np.ndarray:
        return np.array([0.3 + 0.2 * bi], np.float32)  # rising knob across blocks

    out = tmp_path / "o.wav"
    render_file(model, inp, str(out), sr=8000, control=sweep)
    got = _read(out)

    model.reset()
    block = 1024
    man = np.empty_like(x)
    for bi, i in enumerate(range(0, len(x), block)):
        ch = x[i : i + block]
        man[i : i + len(ch)] = np.asarray(model.process_block(ch, sweep(bi)), np.float32)
    np.clip(man, -1.0, 1.0, out=man)
    # Round-trip `man` through the same wav codec so any (identical) quantization
    # cancels -> exact equality proves the per-block control wiring.
    man_path = tmp_path / "man.wav"
    sf.write(str(man_path), man, 8000)
    assert np.allclose(got, _read(man_path), atol=1e-6)


def test_live_engine_constructs_with_control_fn(model: CIRCE3) -> None:
    from vguitar.config import RealtimeConfig
    from vguitar.realtime.engine import LiveEngine

    eng = LiveEngine(model, RealtimeConfig(), control_fn=lambda bi: np.array([float(bi)], np.float32))
    assert eng._cbuf.shape == (1,)  # preallocated control buffer sized to n_control
