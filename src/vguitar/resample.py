"""Two-step bridge from a raw SPICE transient to a band-limited audio target.

ngspice picks its own (non-uniform) time steps, and at the high simulation rate
the nonlinear circuit generates harmonics well above the audio Nyquist. Naively
sampling those points at the audio rate would fold that energy back down as
aliasing. We avoid this in two stages:

1. ``to_uniform``: place the irregular ``(t, y)`` samples on a uniform grid at
   the *simulation* rate using a monotonic cubic (PCHIP) interpolant. PCHIP is
   shape-preserving (Fritsch & Carlson, 1980), so unlike a natural cubic spline
   it adds no ringing/overshoot near the sharp clipping edges typical of guitar
   circuits.
2. ``resample``: rate-convert down to the audio rate with a high-quality
   polyphase filter (soxr 'VHQ'). Its steep anti-aliasing lowpass removes the
   above-Nyquist harmonics *before* decimation, so they are discarded rather
   than folded back in.

``to_audio`` chains both: this is THE canonical raw-SPICE to audio-target path.
"""

from __future__ import annotations

import numpy as np
import soxr
from scipy.interpolate import PchipInterpolator


def to_uniform(t: np.ndarray, y: np.ndarray, sr: int) -> np.ndarray:
    """Resample irregular samples ``(t, y)`` onto a uniform grid at ``sr`` Hz.

    Uses a monotonic cubic (PCHIP) interpolant over ``[t[0], t[-1]]`` for a
    smooth, overshoot-free reconstruction.
    """
    interp = PchipInterpolator(t, y)
    n = round((t[-1] - t[0]) * sr) + 1
    grid = t[0] + np.arange(n) / sr
    return interp(grid).astype(np.float32)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality rate conversion via soxr 'VHQ' (anti-aliased)."""
    if sr_in == sr_out:
        return y.astype(np.float32)
    return soxr.resample(y, sr_in, sr_out, quality="VHQ").astype(np.float32)


def to_audio(t: np.ndarray, y: np.ndarray, sim_sr: int, sr: int) -> np.ndarray:
    """Raw SPICE transient ``(t, y)`` to band-limited audio at ``sr`` Hz."""
    return resample(to_uniform(t, y, sim_sr), sim_sr, sr)
