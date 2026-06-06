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


def esr_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Error-to-signal ratio: ``sum((y - yhat)**2) / (sum(y**2) + eps)`` (DAFx-19)."""
    err = (y_true - y_pred).pow(2).sum()
    return err / (y_true.pow(2).sum() + eps)


def dc_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Squared error of the per-signal mean offset, normalized by signal energy."""
    yp, yt = _flatten(y_pred), _flatten(y_true)
    offset = (yt.mean(dim=-1) - yp.mean(dim=-1)).pow(2).sum()
    return offset / (yt.pow(2).mean(dim=-1).sum() + eps)


def multi_stft_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, ffts: tuple[int, ...] = (512, 1024, 2048)
) -> torch.Tensor:
    """Multi-resolution STFT loss: spectral convergence + log-magnitude L1 (ICASSP-20)."""
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
) -> torch.Tensor:
    """ESR plus optionally weighted DC and multi-resolution STFT terms."""
    loss = esr_loss(y_pred, y_true)
    if dc_weight:
        loss = loss + dc_weight * dc_loss(y_pred, y_true)
    if stft_weight:
        loss = loss + stft_weight * multi_stft_loss(y_pred, y_true)
    return loss
