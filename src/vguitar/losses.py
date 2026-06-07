"""Torch training losses for the neural emulators (tcn/rnn/ssm).

The primary objective is the **error-to-signal ratio** (ESR), the standard metric
for black-box amplifier modelling: it is the squared error normalized by the
target's energy, so it is invariant to overall gain and comparable across signals
(Wright et al., "Real-Time Black-Box Modelling with Recurrent Neural Networks",
DAFx-19). We add a DC term to suppress the slow offsets RNNs tend to accumulate
and an optional multi-resolution STFT term (Yamamoto et al., "Parallel
WaveGAN", ICASSP-20) to sharpen high-frequency / harmonic content.

All losses accept ``(B, T)`` batches or a single ``(T,)`` signal.
"""

from __future__ import annotations

import torch


def _flatten(y: torch.Tensor) -> torch.Tensor:
    """Promote a ``(T,)`` signal to ``(1, T)`` so reductions are uniform."""
    return y.unsqueeze(0) if y.ndim == 1 else y


def pre_emphasis(y: torch.Tensor, coef: float) -> torch.Tensor:
    """First-order high-pass pre-emphasis ``y[n] - coef*y[n-1]`` (per row).

    A +6 dB/octave tilt that amplifies high-frequency content. Applied to BOTH
    prediction and target before ESR, it makes the loss weight the high harmonics
    (which carry little energy and are otherwise ignored), so the model learns the
    sharp clipping edges / high-order distortion that actually shape the sound
    (Wright & Valimaki, "Perceptual Loss Function for Neural Modeling of Audio
    Systems", ICASSP 2020). ``coef`` ~0.85-0.95; higher = stronger HF emphasis.
    """
    yf = _flatten(y)
    out = yf.clone()
    out[..., 1:] = yf[..., 1:] - coef * yf[..., :-1]
    return out


def esr_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8, pre_emph: float = 0.0
) -> torch.Tensor:
    """Error-to-signal ratio: ``sum((y - yhat)**2) / (sum(y**2) + eps)`` (DAFx-19).

    With ``pre_emph > 0`` the error and the target are high-pass pre-emphasised
    first (see :func:`pre_emphasis`), turning this into the perceptual,
    high-frequency-weighted ESR that forces the model to match high harmonics and
    clipping transients rather than just the low-order, high-energy partials.
    """
    if pre_emph > 0.0:
        y_pred = pre_emphasis(y_pred, pre_emph)
        y_true = pre_emphasis(y_true, pre_emph)
    err = (y_true - y_pred).pow(2).sum()
    return err / (y_true.pow(2).sum() + eps)


def dc_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Squared error of the per-signal mean offset, normalized by signal energy."""
    yp, yt = _flatten(y_pred), _flatten(y_true)
    offset = (yt.mean(dim=-1) - yp.mean(dim=-1)).pow(2).sum()
    return offset / (yt.pow(2).mean(dim=-1).sum() + eps)


def multi_stft_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, ffts: tuple[int, ...] = (256, 512, 1024, 2048)
) -> torch.Tensor:
    """Multi-resolution STFT loss: spectral convergence + log-magnitude L1 (ICASSP-20).

    The log-magnitude term is the high-frequency workhorse (log compresses dynamic
    range so a quiet high harmonic counts like the fundamental); the short 256
    window adds the time resolution that resolves sharp clipping edges/transients.
    """
    yp, yt = _flatten(y_pred), _flatten(y_true)
    total = yp.new_zeros(())
    # Skip FFT sizes larger than the signal: torch.stft center-pads by n_fft//2
    # and errors if that exceeds the input length (happens on short windows).
    sizes = [n for n in ffts if n <= yp.shape[-1]]
    if not sizes:
        return total
    for n_fft in sizes:
        win = torch.hann_window(n_fft, device=yp.device, dtype=yp.dtype)
        hop = n_fft // 4
        mp = torch.stft(
            yp, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True
        ).abs()
        mt = torch.stft(
            yt, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True
        ).abs()
        sc = torch.linalg.norm(mt - mp) / (torch.linalg.norm(mt) + 1e-8)
        mag = (torch.log(mt + 1e-8) - torch.log(mp + 1e-8)).abs().mean()
        total = total + sc + mag
    return total / len(sizes)


def combined_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    stft_weight: float = 0.0,
    dc_weight: float = 0.0,
    pre_emph: float = 0.0,
) -> torch.Tensor:
    """ESR (optionally pre-emphasised) plus weighted DC and multi-STFT terms."""
    loss = esr_loss(y_pred, y_true, pre_emph=pre_emph)
    if dc_weight:
        loss = loss + dc_weight * dc_loss(y_pred, y_true)
    if stft_weight:
        loss = loss + stft_weight * multi_stft_loss(y_pred, y_true)
    return loss
