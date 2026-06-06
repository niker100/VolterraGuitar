"""Evaluation metrics and comparison plots for circuit-emulation models.

Pure ``numpy`` / ``scipy`` / ``matplotlib`` — this module is imported by
non-neural models too, so it deliberately never imports ``torch``.

The primary metric is the error-to-signal ratio (ESR), the standard objective
for black-box guitar-amp/distortion modelling (Wright et al., "Real-Time Guitar
Amplifier Emulation with Deep Learning", Appl. Sci. 2020; Damskaegg et al.,
"Deep Learning for Tube Amplifier Emulation", ICASSP 2019). We additionally
report a multi-resolution STFT distance (perceptually-motivated, after Yamamoto
et al. Parallel WaveGAN, ICASSP 2020) and total harmonic distortion (THD), a
classic measure of how well a model reproduces a nonlinearity's harmonics.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a screen
import matplotlib.pyplot as plt
import numpy as np

_EPS = 1e-12  # guards against division by zero on silent signals


def _prep(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to 1-D float64 and check lengths (float64 keeps sums accurate)."""
    a = np.asarray(y_true, dtype=np.float64).reshape(-1)
    b = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    return a, b


def esr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error-to-signal ratio: ``sum((y-yhat)**2) / sum(y**2)``.

    The primary benchmark metric (Wright/Damskaegg). It is the squared-error
    energy normalised by the target's energy, so it is scale-invariant and
    comparable across circuits and drive levels. Lower is better; 0 is perfect.
    """
    a, b = _prep(y_true, y_pred)
    return float(np.sum((a - b) ** 2) / (np.sum(a**2) + _EPS))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error (unnormalised). Lower is better."""
    a, b = _prep(y_true, y_pred)
    return float(np.mean((a - b) ** 2))


def _stft_mag(y: np.ndarray, n_fft: int) -> np.ndarray:
    """Magnitude STFT via framed ``rfft`` (Hann window, 75% overlap).

    Returns an array of shape ``(frames, n_fft // 2 + 1)``. We frame by hand
    with numpy rather than pulling in extra deps; 75% overlap (hop = n_fft/4) is
    the common choice for the multi-resolution STFT loss.
    """
    hop = n_fft // 4
    win = np.hanning(n_fft).astype(np.float64)
    if y.shape[0] < n_fft:  # pad short signals so at least one frame exists
        y = np.pad(y, (0, n_fft - y.shape[0]))
    n_frames = 1 + (y.shape[0] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * win  # (frames, n_fft)
    return np.abs(np.fft.rfft(frames, axis=-1))


def multi_stft(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ffts: tuple[int, ...] = (512, 1024, 2048),
) -> float:
    """Multi-resolution STFT distance (lower is better).

    For each FFT size we sum two terms (Yamamoto et al., Parallel WaveGAN):

    * spectral convergence ``||S - S_hat||_F / ||S||_F`` — emphasises large
      spectral peaks;
    * log-magnitude L1 ``mean |log S - log S_hat|`` — emphasises detail/quiet
      partials.

    The result is averaged over the scales, giving a single resolution-robust
    number that complements time-domain ESR.
    """
    a, b = _prep(y_true, y_pred)
    scores: list[float] = []
    for n_fft in ffts:
        s, sh = _stft_mag(a, n_fft), _stft_mag(b, n_fft)
        sc = np.linalg.norm(s - sh) / (np.linalg.norm(s) + _EPS)
        log_l1 = np.mean(np.abs(np.log(s + _EPS) - np.log(sh + _EPS)))
        scores.append(float(sc + log_l1))
    return float(np.mean(scores))


def thd(
    process: Callable[[np.ndarray], np.ndarray],
    f0: float = 1000.0,
    sr: int = 44_100,
    amp: float = 0.3,
    n_harmonics: int = 8,
    dur_s: float = 1.0,
) -> float:
    """Total harmonic distortion of a processor at a single tone.

    Drives a pure sine of amplitude ``amp`` at ``f0`` through ``process`` and
    measures ``sqrt(sum P_k for k>=2) / sqrt(P_1)`` from the magnitude spectrum,
    i.e. the RMS of harmonics 2..``n_harmonics`` relative to the fundamental.
    THD quantifies how much harmonic energy the nonlinearity adds; it should
    match the reference circuit's THD if the model captures the nonlinearity.

    ``f0`` is snapped to an exact FFT bin so each harmonic lands on one bin,
    avoiding spectral leakage (no window needed).
    """
    n = round(dur_s * sr)
    k0 = max(1, round(f0 * n / sr))  # bin index of the fundamental
    f0 = k0 * sr / n  # exact bin-aligned frequency
    t = np.arange(n, dtype=np.float64) / sr
    y = process(amp * np.sin(2.0 * np.pi * f0 * t).astype(np.float32))
    mag = np.abs(np.fft.rfft(np.asarray(y, dtype=np.float64)))
    fund = mag[k0]
    n_bins = mag.shape[0]
    harm = np.array([mag[k * k0] for k in range(2, n_harmonics + 1) if k * k0 < n_bins])
    return float(np.sqrt(np.sum(harm**2)) / (fund + _EPS))


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_spectrogram(y: np.ndarray, sr: int, path: str | Path) -> None:
    """Save a log-magnitude spectrogram PNG of ``y`` to ``path``."""
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_spectrogram(ax, np.asarray(y, dtype=np.float64).reshape(-1), sr, "spectrogram")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _draw_spectrogram(ax: plt.Axes, y: np.ndarray, sr: int, title: str) -> None:
    """Draw a dB spectrogram onto ``ax`` (shared by single/compare plots)."""
    n_fft, _hop = 1024, 256
    mag = _stft_mag(y, n_fft)  # (frames, bins)
    db = 20.0 * np.log10(mag.T + _EPS)  # (bins, frames), low freq at bottom
    extent = (0.0, y.shape[0] / sr, 0.0, sr / 2)
    ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmax=db.max(),
        vmin=db.max() - 80.0,
    )
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")


def plot_compare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sr: int,
    path: str | Path,
) -> None:
    """Save a 2x2 comparison figure: waveforms, error trace, two spectrograms.

    Top-left: target vs. prediction overlaid on a zoomed window (~5 ms) so the
    shape of the waveform is legible. Top-right: the sample-wise error. Bottom:
    side-by-side spectrograms. The figure title reports ESR for quick triage.
    """
    a, b = _prep(y_true, y_pred)
    path = _ensure_parent(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_wav, ax_err), (ax_st, ax_sp) = axes

    # Zoom to ~5 ms taken from the middle (steady region), capped at signal len.
    win = min(a.shape[0], max(1, int(0.005 * sr)))
    start = max(0, a.shape[0] // 2 - win // 2)
    sl = slice(start, start + win)
    t = np.arange(start, start + win) / sr
    ax_wav.plot(t, a[sl], label="target", lw=1.2)
    ax_wav.plot(t, b[sl], label="prediction", lw=1.0, alpha=0.85)
    ax_wav.set_title("waveform (zoom)")
    ax_wav.set_xlabel("time (s)")
    ax_wav.set_ylabel("amplitude")
    ax_wav.legend(loc="upper right")

    t_full = np.arange(a.shape[0]) / sr
    ax_err.plot(t_full, a - b, color="crimson", lw=0.6)
    ax_err.set_title("error (target - prediction)")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("amplitude")

    _draw_spectrogram(ax_st, a, sr, "target")
    _draw_spectrogram(ax_sp, b, sr, "prediction")

    fig.suptitle(f"ESR = {esr(a, b):.4e}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
