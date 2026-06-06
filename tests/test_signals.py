"""Tests for excitation design (vguitar.signals)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vguitar.config import DataConfig
from vguitar.signals import (
    amplitude_staircase,
    build_training_excitation,
    exp_sweep,
    load_di,
    multisine,
    noise,
    sine,
)

SR = 8000


def test_noise_shape_dtype_peak() -> None:
    x = noise(0.1, SR, color="white", seed=0, peak=0.5)
    assert x.dtype == np.float32
    assert abs(len(x) - int(0.1 * SR)) <= 2
    assert np.max(np.abs(x)) <= 0.5 + 1e-6


def test_noise_determinism() -> None:
    a = noise(0.1, SR, seed=1)
    b = noise(0.1, SR, seed=1)
    c = noise(0.1, SR, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_pink_has_more_low_end_than_white() -> None:
    rng_len = 1.0
    w = noise(rng_len, SR, color="white", seed=0)
    p = noise(rng_len, SR, color="pink", seed=0)
    # Pink (~1/f) should put proportionally more energy in the low band.
    def low_frac(x: np.ndarray) -> float:
        mag = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return float(np.sum(mag[f < 200] ** 2) / (np.sum(mag**2) + 1e-12))

    assert low_frac(p) > low_frac(w)


def test_sine_frequency_and_peak() -> None:
    x = sine(1000.0, 0.5, SR, peak=0.8)
    assert np.max(np.abs(x)) <= 0.8 + 1e-4
    mag = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    assert abs(f[np.argmax(mag)] - 1000.0) < 20.0


def test_exp_sweep_deconvolves_to_impulse() -> None:
    sweep, inv = exp_sweep(50.0, 3000.0, 1.0, SR)
    assert sweep.dtype == np.float32
    # For an LTI identity system, sweep * inverse is a sharp impulse.
    y = np.convolve(sweep, inv)
    peak = np.max(np.abs(y))
    assert peak > 20.0 * np.median(np.abs(y))


def test_multisine_shape_peak() -> None:
    x = multisine(0.5, SR, n_tones=32, seed=0, peak=1.0)
    assert x.dtype == np.float32
    assert np.max(np.abs(x)) <= 1.0 + 1e-4


def test_amplitude_staircase_tiles_and_scales() -> None:
    base = sine(200.0, 0.05, SR, peak=1.0)
    levels = (0.1, 0.5, 1.0)
    out = amplitude_staircase(base, levels)
    assert len(out) == len(base) * len(levels)
    # Each segment's peak should track its level.
    seg = len(base)
    for i, lvl in enumerate(levels):
        assert abs(np.max(np.abs(out[i * seg : (i + 1) * seg])) - lvl) < 0.05


def test_load_di_mono_and_normalized() -> None:
    wav = Path("assets/guitar_di_loop.wav")
    if not wav.exists():  # asset optional in some checkouts
        return
    x = load_di(str(wav), SR, peak=0.9)
    assert x.dtype == np.float32
    assert x.ndim == 1
    assert np.max(np.abs(x)) <= 0.9 + 1e-4


def test_build_training_excitation_deterministic_nonempty() -> None:
    cfg = DataConfig(duration_s=1.0, seed=0)
    a = build_training_excitation(cfg)
    b = build_training_excitation(cfg)
    assert a.dtype == np.float32
    assert len(a) > SR  # at least ~1 s of content
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a))
