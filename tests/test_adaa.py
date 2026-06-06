"""Tests for antiderivative anti-aliasing (vguitar.nonlinear.adaa)."""

from __future__ import annotations

import numpy as np

from vguitar.nonlinear.adaa import HARDCLIP, TANH, ADAAProcessor, adaa1

SR = 44_100


def test_adaa1_constant_input_no_nan() -> None:
    # Equal consecutive samples hit the (x == x_prev) fallback branch.
    x = np.full(16, 0.4, dtype=np.float32)
    y, prev = adaa1(TANH, x, x_prev=0.4)
    assert np.all(np.isfinite(y))
    assert np.allclose(y, np.tanh(0.4), atol=1e-5)
    assert np.isfinite(prev)


def test_adaa_processor_streaming_finite_and_continuous() -> None:
    rng = np.random.default_rng(0)
    x = (0.8 * rng.standard_normal(4096)).astype(np.float32)
    proc = ADAAProcessor(TANH, order=1)
    proc.reset()
    blocks = [proc.process_block(x[i : i + 128]) for i in range(0, len(x), 128)]
    y = np.concatenate(blocks)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def _inharmonic_energy(y: np.ndarray, f0: float) -> float:
    """Energy at bins that are NOT harmonics of f0 — i.e. aliased fold-back.

    Hard-clipping a tone produces odd harmonics; the high ones exceed Nyquist
    and fold back onto inharmonic bins. A band-limited result has ~none there.
    """
    mag = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    harmonic = np.zeros(len(f), dtype=bool)
    for k in range(1, int(SR / 2 / f0) + 1):
        harmonic |= np.abs(f - k * f0) < 60.0
    return float(np.sum(mag[~harmonic] ** 2))


def test_adaa_reduces_aliasing_vs_naive() -> None:
    # Drive a 6 kHz tone *into* hard clipping (amp 2.0): the 5th/7th/... harmonics
    # land above Nyquist and fold back to inharmonic bins. First-order ADAA should
    # leave less inharmonic (aliased) energy than naive pointwise clipping.
    f0 = 6000.0  # bin-aligned at n = SR//2
    x = (2.0 * np.sin(2 * np.pi * f0 * np.arange(SR // 2) / SR)).astype(np.float32)
    naive = HARDCLIP.f(x)
    proc = ADAAProcessor(HARDCLIP, order=1)
    proc.reset()
    aa = proc.process_block(x)
    assert _inharmonic_energy(aa, f0) < _inharmonic_energy(naive, f0)
