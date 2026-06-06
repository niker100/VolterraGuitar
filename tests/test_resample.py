"""Resampling: length ~ ratio, identity, amplitude/peak-frequency preservation, to_uniform."""

from __future__ import annotations

import numpy as np

from vguitar.resample import resample, to_uniform


def test_length_tracks_ratio() -> None:
    n = 8000
    y = np.zeros(n, dtype=np.float32)
    out = resample(y, 8000, 4000)
    # Decimation by 2 should roughly halve the length (soxr edge taps -> approx).
    assert abs(len(out) - n // 2) <= 8


def test_identity_when_rates_equal() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(256).astype(np.float32)
    out = resample(y, 44_100, 44_100)
    np.testing.assert_array_equal(out, y)


def test_sine_amplitude_and_peak_frequency_preserved() -> None:
    sr_in, sr_out, f0, dur = 352_800, 44_100, 1000.0, 0.5
    n = int(sr_in * dur)
    t = np.arange(n) / sr_in
    y = np.sin(2.0 * np.pi * f0 * t).astype(np.float32)

    out = resample(y, sr_in, sr_out)
    assert np.all(np.isfinite(out))

    # Amplitude: a 1 kHz tone is far below the 22.05 kHz Nyquist, so the steep
    # anti-alias filter passes it untouched -> peak stays ~1.
    interior = out[len(out) // 4 : -len(out) // 4]  # drop filter edge transients
    assert abs(float(np.max(np.abs(interior))) - 1.0) < 0.02

    # Peak-frequency: the dominant rfft bin must land on f0.
    spec = np.abs(np.fft.rfft(interior))
    freqs = np.fft.rfftfreq(len(interior), d=1.0 / sr_out)
    assert abs(freqs[int(np.argmax(spec))] - f0) < 5.0


def test_to_uniform_recovers_linear_ramp() -> None:
    # A line is reproduced exactly by the (shape-preserving) PCHIP interpolant.
    sr = 1000
    t = np.array([0.0, 0.001, 0.0035, 0.006, 0.01], dtype=np.float64)  # irregular grid
    y = 2.0 * t + 0.5  # known straight line
    out = to_uniform(t, y, sr)
    grid = t[0] + np.arange(len(out)) / sr
    np.testing.assert_allclose(out, 2.0 * grid + 0.5, atol=1e-5)
