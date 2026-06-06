"""Tests for evaluation metrics (vguitar.metrics)."""

from __future__ import annotations

import numpy as np

from vguitar.metrics import esr, mse, multi_stft, thd

SR = 44_100


def _tone(f: float = 1000.0, n: int = SR) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_esr_zero_for_identical() -> None:
    y = _tone()
    assert esr(y, y) == 0.0


def test_esr_one_for_zero_prediction() -> None:
    y = _tone()
    # error == signal => ESR == 1 by construction.
    assert abs(esr(y, np.zeros_like(y)) - 1.0) < 1e-6


def test_mse_nonnegative_and_zero() -> None:
    y = _tone()
    assert mse(y, y) == 0.0
    assert mse(y, np.zeros_like(y)) > 0.0


def test_multi_stft_zero_for_identical() -> None:
    y = _tone()
    assert multi_stft(y, y) < 1e-5


def test_thd_identity_is_small() -> None:
    val = thd(lambda x: x, sr=SR)
    assert val < 1e-3


def test_thd_cubic_clip_is_positive() -> None:
    # A cubic soft-clip injects odd harmonics -> measurable THD.
    val = thd(lambda x: np.tanh(4.0 * x).astype(np.float32), sr=SR)
    assert val > 0.05
