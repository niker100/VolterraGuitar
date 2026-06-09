"""Excitation design for nonlinear system identification.

Identifying a nonlinear circuit well requires probing it across both *frequency*
and *amplitude*: a model only learns the behaviour it actually saw. This module
provides the building blocks (noise, sweeps, multisines, a guitar DI) and a
single :func:`build_training_excitation` that stitches them into one
deterministic training input.

Design rationale, by component:

* **Pink + white noise** — broadband, statistically rich excitation. Pink noise
  matches the roughly ``1/f`` spectrum of real instrument signals (more energy
  where the ear and the circuit's response live); white noise tops up the high
  end to exercise fast nonlinear dynamics.
* **Schroeder-phase multisine** — deterministic, periodic, *low crest factor*
  (Schroeder 1970) so we can push high RMS energy into the circuit without
  clipping the input source, giving a clean per-tone frequency picture.
* **Exponential sine sweep (ESS)** — the bridge to diagonal Volterra /
  Hammerstein kernels: deconvolving the response with the matched inverse filter
  yields harmonic impulse responses, one per nonlinear order (Farina 2000;
  Novak et al., JAES 2015).
* **Amplitude staircase** — every component is replayed at each drive level so
  the model sees the full input-voltage range. Narrow amplitude coverage was a
  v1 failure mode: models diverged outside their trained range.
* **Guitar DI (optional)** — a real direct-input loop so the identified model is
  validated on the kind of signal it will actually process live.

All public functions return ``float32`` and use ``np.random.default_rng`` for
reproducibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vguitar.config import DataConfig


def _peak_normalize(x: np.ndarray, peak: float) -> np.ndarray:
    """Scale ``x`` so its maximum absolute value equals ``peak`` (float32)."""
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > 0.0:
        x = x * (peak / m)
    return x.astype(np.float32)


def noise(
    duration_s: float,
    sr: int,
    *,
    color: str = "white",
    seed: int = 0,
    peak: float = 1.0,
) -> np.ndarray:
    """White or pink noise, peak-normalized to ``+-peak``.

    Pink (``1/f``) noise is synthesized with the Voss-McCartney algorithm: a sum
    of independent random sources each refreshed at a successively halved rate,
    which approximates a ``-3 dB/octave`` spectrum cheaply and without filter
    ringing. White noise is i.i.d. uniform.

    Args:
        duration_s: length in seconds.
        sr: sample rate in Hz.
        color: ``"white"`` or ``"pink"``.
        seed: RNG seed.
        peak: target peak amplitude.

    Returns:
        float32 array of shape ``(round(duration_s * sr),)``.
    """
    rng = np.random.default_rng(seed)
    n = round(duration_s * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if color == "white":
        out = rng.uniform(-1.0, 1.0, size=n)
    elif color == "pink":
        out = _voss_pink(n, rng)
    else:
        raise ValueError(f"unknown color {color!r}; use 'white' or 'pink'")
    return _peak_normalize(out, peak)


def _voss_pink(n: int, rng: np.random.Generator, n_rows: int = 16) -> np.ndarray:
    """Voss-McCartney pink noise: sum of ``n_rows`` octave-spaced random sources.

    Row ``k`` is refreshed every ``2**k`` samples; summing the rows yields an
    approximate ``1/f`` spectrum. Implemented vectorized via cumulative-update
    indexing rather than a per-sample loop.
    """
    # Each row holds one held value at a time; precompute when each row updates.
    rows = np.empty((n_rows, n), dtype=np.float64)
    for k in range(n_rows):
        step = 1 << k
        n_updates = n // step + 1
        held = rng.uniform(-1.0, 1.0, size=n_updates)
        rows[k] = np.repeat(held, step)[:n]
    return rows.sum(axis=0)


def sine(
    freq: float,
    duration_s: float,
    sr: int,
    peak: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """A single sinusoid of amplitude ``peak`` (float32)."""
    n = round(duration_s * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    out = peak * np.sin(2.0 * np.pi * freq * t + phase)
    return out.astype(np.float32)


def tone_bank(
    duration_s: float,
    sr: int,
    *,
    f_lo: float = 80.0,
    f_hi: float = 6000.0,
    n_tones: int = 16,
    peak: float = 1.0,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Concatenated ISOLATED single sines at log-spaced frequencies (float32).

    Unlike :func:`multisine` (all tones sounding at once, so the response is full
    of intermodulation products), this plays **one tone at a time**, so the model
    sees each tone's *clean per-tone harmonic structure*. That is the probe that
    uniquely constrains the nonlinearity's harmonic generation: broadband / multi-
    tone excitation under-determines it (many nonlinearities match the broadband
    spectrum but differ on a single tone), which lets a model invent spurious
    overtones a band-limited circuit never produces. Each segment is edge-faded to
    suppress the click (broadband splatter) at concatenation boundaries.
    """
    n_total = round(duration_s * sr)
    if n_total <= 0 or n_tones <= 0:
        return np.zeros(0, dtype=np.float32)
    freqs = np.logspace(np.log10(f_lo), np.log10(min(f_hi, 0.45 * sr)), n_tones)
    seg = max(n_total // n_tones, 1)
    fade = min(int(fade_ms * 1e-3 * sr), seg // 2)
    win = np.ones(seg, dtype=np.float64)
    if fade > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
        win[:fade] = ramp
        win[-fade:] = ramp[::-1]
    t = np.arange(seg, dtype=np.float64) / sr
    segs = [np.sin(2.0 * np.pi * f * t) * win for f in freqs]
    x = np.concatenate(segs)[:n_total]
    return _peak_normalize(x.astype(np.float32), peak)


def exp_sweep(
    f1: float,
    f2: float,
    duration_s: float,
    sr: int,
    peak: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synchronized exponential sine sweep (ESS) and its inverse filter.

    The sweep instantaneous frequency rises exponentially from ``f1`` to ``f2``.
    Convolving a nonlinear system's response to this sweep with the returned
    ``inverse_filter`` (a time-reversed, amplitude-equalized copy of the sweep)
    deconvolves it into a train of *harmonic impulse responses*: the ``m``-th
    harmonic's IR appears advanced in time by ``dt_m = L * ln(m)`` ahead of the
    linear (``m = 1``) IR, where ``L = T / ln(f2/f1)`` (Farina 2000; Novak et
    al., JAES 2015). Because each harmonic order is thus separated in time, the
    ESS is the bridge to *diagonal* Volterra / Hammerstein kernels: windowing
    each separated IR gives the kernel of one nonlinear order directly.

    This implementation uses Novak's *synchronized* sweep, where the total
    duration is rounded so the phase ``f1`` and ``f2`` lock to an integer number
    of cycles; this removes the start/stop transients that otherwise smear the
    harmonic IRs.

    Args:
        f1: start frequency (Hz, > 0).
        f2: end frequency (Hz, > f1).
        duration_s: nominal sweep length in seconds.
        sr: sample rate in Hz.
        peak: peak amplitude of the sweep.

    Returns:
        ``(sweep, inverse_filter)``, both float32. The sweep has ``peak``
        amplitude; the inverse filter is scaled so that
        ``convolve(sweep, inverse_filter)`` approximates a unit Dirac.
    """
    if not (f1 > 0.0 and f2 > f1):
        raise ValueError("require 0 < f1 < f2")
    ratio = f2 / f1
    # Synchronized rate L: round T so f1 starts and f2 ends at zero phase
    # (Novak 2015, eq. for L = round(f1*T/ln(ratio)) / f1).
    t_nom = duration_s
    big_l = round(f1 * t_nom / np.log(ratio)) / f1
    t = big_l * np.log(ratio)  # synchronized duration
    n = round(t * sr)
    tt = np.arange(n, dtype=np.float64) / sr
    phase = 2.0 * np.pi * f1 * big_l * (np.exp(tt / big_l) - 1.0)
    sweep = peak * np.sin(phase)

    # Inverse filter: time-reversed sweep with +6 dB/oct amplitude envelope to
    # flatten the sweep's pink-ish spectrum (Farina 2000). Then scale for unit
    # deconvolution gain at the fundamental.
    inv = sweep[::-1] * np.exp(-tt / big_l)
    # Normalize so the linear convolution peak is ~1.
    norm = np.sum(sweep * inv[::-1])
    if norm != 0.0:
        inv = inv / norm
    return sweep.astype(np.float32), inv.astype(np.float32)


def multisine(
    duration_s: float,
    sr: int,
    f_lo: float = 20.0,
    f_hi: float = 20000.0,
    n_tones: int = 64,
    seed: int = 0,
    peak: float = 1.0,
) -> np.ndarray:
    """Schroeder-phase multisine, peak-normalized to ``+-peak`` (float32).

    A sum of ``n_tones`` equal-amplitude sinusoids logarithmically spaced from
    ``f_lo`` to ``f_hi``, with Schroeder phases ``phi_k = -pi * k*(k-1)/n``
    (Schroeder 1970). Schroeder phases give a near-minimal crest factor, so the
    signal packs high RMS energy into the circuit for a given input-voltage limit
    — maximizing excitation per volt without input-source clipping. ``seed`` is
    accepted for API uniformity and only perturbs phases negligibly; the spectrum
    is deterministic.

    Args:
        duration_s: length in seconds.
        sr: sample rate in Hz.
        f_lo: lowest tone frequency (Hz).
        f_hi: highest tone frequency (Hz, clamped below Nyquist).
        n_tones: number of tones.
        seed: RNG seed (used only for a tiny phase dither for reproducibility).
        peak: target peak amplitude.

    Returns:
        float32 array of shape ``(round(duration_s * sr),)``.
    """
    n = round(duration_s * sr)
    if n <= 0 or n_tones <= 0:
        return np.zeros(max(n, 0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    nyq = 0.5 * sr
    f_hi = min(f_hi, 0.99 * nyq)
    freqs = np.geomspace(f_lo, f_hi, n_tones)
    k = np.arange(1, n_tones + 1, dtype=np.float64)
    # Schroeder phases for low crest factor (Schroeder, IEEE TIT 1970).
    phases = -np.pi * k * (k - 1.0) / n_tones
    phases = phases + 1e-6 * rng.standard_normal(n_tones)  # tiny seeded dither
    t = np.arange(n, dtype=np.float64) / sr
    out = np.sin(2.0 * np.pi * np.outer(t, freqs) + phases).sum(axis=1)
    return _peak_normalize(out, peak)


def amplitude_staircase(base: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    """Concatenate copies of ``base`` scaled to each level in ``levels``.

    Drives the same waveform through every amplitude in ``levels`` so the model
    sees the full input-voltage range. ``base`` is assumed peak-normalized to
    ``+-1``; each block is scaled to peak ``level`` (volts at the circuit input).
    Broad amplitude coverage prevents the out-of-range divergence that was a v1
    failure mode.

    Args:
        base: peak-normalized 1-D waveform.
        levels: target peak amplitudes (volts).

    Returns:
        float32 array of length ``len(base) * len(levels)``.
    """
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    if not levels:
        return np.zeros(0, dtype=np.float32)
    blocks = [(base * np.float32(level)) for level in levels]
    return np.concatenate(blocks).astype(np.float32)


def load_di(path: str, sr: int, *, peak: float = 1.0) -> np.ndarray:
    """Load a WAV direct-input loop, downmix to mono, resample, peak-normalize.

    Args:
        path: path to a WAV file.
        sr: target sample rate (Hz).
        peak: target peak amplitude.

    Returns:
        float32 mono array at ``sr``, peak-normalized to ``+-peak``.
    """
    import soundfile as sf

    from vguitar.resample import resample

    data, src_sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix channels
    if src_sr != sr:
        mono = resample(mono, src_sr, sr)
    return _peak_normalize(mono, peak)


def build_training_excitation(cfg: DataConfig) -> np.ndarray:
    """Build the full training excitation from ``cfg`` (a ``DataConfig``).

    Concatenates, in order: pink noise, white noise, a Schroeder-phase
    multisine, an exponential sweep, and (if the asset exists) a guitar DI loop.
    Each component is replayed across ``cfg.drive_levels`` via
    :func:`amplitude_staircase`, so the model sees every waveform at every input
    voltage. The component budget is split to total roughly ``cfg.duration_s``
    seconds after the amplitude staircase replication, and everything is
    deterministic from ``cfg.seed``.

    Component justification (see module docstring): noise gives broadband
    statistical coverage; the multisine gives high-RMS, low-crest frequency
    coverage; the ESS gives the harmonic-IR structure that anchors diagonal
    Volterra/Hammerstein kernels; the DI grounds the model in real playing
    signals. Amplitude coverage across ``drive_levels`` prevents the
    out-of-range divergence that was a v1 failure mode.

    Args:
        cfg: a :class:`vguitar.config.DataConfig`.

    Returns:
        float32 array of input-voltage samples at ``cfg.sr``.
    """
    from pathlib import Path

    sr = cfg.sr
    levels = cfg.drive_levels
    n_levels = max(len(levels), 1)

    # Total duration is consumed by len(levels) replicas of each component, so
    # each component's pre-staircase length is the share divided by n_levels.
    # Weights sum to 1; the DI (if present) takes its own share.
    di_path = Path("C:/Users/nicks/Documents/Projekte/VolterraGuitar/assets/guitar_di_loop.wav")
    has_di = di_path.exists()
    # Relative time budget per component (pre-staircase, before /n_levels). The
    # isolated-tone bank uniquely constrains the per-tone harmonic structure
    # (prevents the model inventing overtones a band-limited circuit lacks), which
    # noise / multisine / sweep under-determine.
    weights = {"pink": 0.25, "white": 0.15, "multisine": 0.18, "sweep": 0.14, "tones": 0.16}
    if has_di:
        weights["di"] = 0.12
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    def secs(name: str) -> float:
        return cfg.duration_s * weights[name] / n_levels

    # Distinct seeds per stochastic component, all derived from cfg.seed.
    pink = noise(secs("pink"), sr, color="pink", seed=cfg.seed, peak=1.0)
    white = noise(secs("white"), sr, color="white", seed=cfg.seed + 1, peak=1.0)
    ms = multisine(secs("multisine"), sr, seed=cfg.seed + 2, peak=1.0)
    sweep, _inv = exp_sweep(20.0, 0.45 * sr, secs("sweep"), sr, peak=1.0)
    tb = tone_bank(secs("tones"), sr)

    components = [pink, white, ms, sweep, tb]
    if has_di:
        di = load_di(str(di_path), sr, peak=1.0)
        # Trim/replicate DI to its budget so it does not dominate the total.
        target = round(secs("di") * sr)
        if di.size and target > 0:
            reps = int(np.ceil(target / di.size))
            di = np.tile(di, reps)[:target]
            components.append(di)

    staircased = [amplitude_staircase(c, levels) for c in components if c.size]
    if not staircased:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(staircased).astype(np.float32)
