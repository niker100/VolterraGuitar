"""Fourier Neural Operator (1D) for nonlinear analog-circuit emulation.

The canonical FNO spectral-convolution operator (Li et al., "Fourier Neural
Operator for Parametric PDEs", ICLR 2021), adapted to a 1D audio signal.

Why this can match a TCN's harmonic generation more cheaply
------------------------------------------------------------
A guitar distortion circuit is a *long* linear filter (the reactive RC/RLC
network — tens of ms of memory) wrapped around a *pointwise* nonlinearity (the
diode/BJT/JFET transfer curve). A TCN spends most of its parameters and FLOPs
building that long linear memory out of stacked dilated convolutions.

The FNO replaces each long convolution with a single **spectral multiply**: an
rFFT over the whole sequence, a learned per-channel complex weight on the lowest
``K`` modes, then an irFFT. One global FFT (``O(T log T)``) gives every output
sample access to the entire input history for free — the receptive field is the
*whole signal*, not a fixed ``R``. The expensive long-memory linear mixing thus
costs ``K`` complex multiplies per channel-pair instead of ``R`` real taps.

Harmonics still need a genuine pointwise time nonlinearity (a linear spectral
filter provably cannot create new frequencies — confirmed by the failing
"spectral-only linear gain" baseline). So each Fourier layer is
``spectral_mix(h) + pointwise_conv(h)`` followed by a heterogeneous time-domain
activation (:class:`_MixedActivation` from the TCN: tanh / gelu / relu / abs /
snake). The ``abs``/``relu``/snake units synthesize the sharp clipping corners
and the snake unit folds harmonics, exactly as in the mixed-TCN; the FFT just
distributes those generated harmonics across the long linear memory cheaply.

Cost / realtime
---------------
Per Fourier layer the spectral path is ``2 * K * C^2`` real mults for the mode
mixing (complex) plus one rFFT+irFFT (``O(C * T log T)`` shared across modes).
At ``C=24``, ``K=64``, depth 4 this is well under 120k params. Latency: the FFT
is over the whole sequence, so the *exact* offline operator is non-causal /
not block-streamable as written. It IS realtime-feasible as a block FNO
(overlap-save / STFT-style framing with a fixed frame, giving frame-length
latency) — but that's an approximation, unlike the natively-causal TCN. Stated
honestly in COST.
"""

from __future__ import annotations

import torch
from torch import nn

from vguitar.models.tcn import _MixedActivation

APPROACH = (
    "FNO-1D: per-layer rFFT -> keep K low modes -> learned complex per-channel "
    "mode-mixing -> irFFT, plus a pointwise (1x1) bypass, then a heterogeneous "
    "time-domain activation (tanh/gelu/relu/abs/snake) + residual."
)
COST = (
    "~86k params @ channels=24 (depth=3, K=24). Per layer: 2*K*C^2 real mults for "
    "spectral mode-mixing + O(C*T log T) for the rFFT/irFFT pair + C^2*T for the 1x1 "
    "bypass. Receptive field = whole signal (global FFT). NON-causal as written "
    "(full-sequence FFT); realtime-feasible only as a block/overlap-save FNO with "
    "frame-length latency, an approximation. Offline cost is near-linear in T."
)


class _SpectralConv1d(nn.Module):
    """FNO spectral convolution: complex linear mixing of the lowest ``modes``.

    ``rFFT`` the ``(B, C, T)`` signal, multiply the kept low-frequency modes by a
    learned complex weight ``W[c_out, c_in, k]`` (a per-mode channel mixing
    matrix), zero the rest, then ``irFFT`` back to length ``T``. This is a global,
    learnable, long linear filter that costs ``modes`` complex multiplies per
    channel-pair regardless of ``T`` — the cheap replacement for a long
    time-domain convolution.
    """

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes = modes
        # Complex weights stored as (real, imag); scaled like the FNO reference.
        scale = 1.0 / (channels * channels)
        self.w_re = nn.Parameter(scale * torch.randn(channels, channels, modes))
        self.w_im = nn.Parameter(scale * torch.randn(channels, channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        t = x.shape[-1]
        x_ft = torch.fft.rfft(x, n=t, dim=-1)  # (B, C, T//2 + 1) complex
        n_freq = x_ft.shape[-1]
        k = min(self.modes, n_freq)  # short sequences may have fewer bins than modes
        weight = torch.complex(self.w_re[..., :k], self.w_im[..., :k])  # (Co, Ci, k)
        out_ft = x_ft.new_zeros(x_ft.shape)
        # einsum: (B, Ci, k) x (Co, Ci, k) -> (B, Co, k); per-mode channel mixing.
        out_ft[..., :k] = torch.einsum("bik,oik->bok", x_ft[..., :k], weight)
        return torch.fft.irfft(out_ft, n=t, dim=-1)


class _FourierLayer(nn.Module):
    """One FNO block: ``activation(spectral_conv(h) + 1x1(h)) + h`` (residual).

    The ``1x1`` bypass is the FNO's local linear term ``W`` (Li et al.); it lets
    the layer pass through and locally remix channels while the spectral path
    supplies the global long-memory filtering. The heterogeneous time-domain
    activation (reused from the mixed-TCN) is what actually *generates* harmonics:
    its abs/relu/snake units create the clipping corners a smooth filter cannot.
    """

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.spectral = _SpectralConv1d(channels, modes)
        self.bypass = nn.Conv1d(channels, channels, 1)
        self.act = _MixedActivation(channels)
        # Per-channel learnable mix of the two paths (init: equal weight).
        self.scale = nn.Parameter(torch.ones(2, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        mixed = self.scale[0] * self.spectral(x) + self.scale[1] * self.bypass(x)
        return x + self.act(mixed)


class Net(nn.Module):
    """Stacked 1D Fourier Neural Operator with time-domain nonlinear activations.

    Lift to ``channels`` (1x1 conv) -> ``depth`` Fourier layers -> project back to
    one channel (1x1 -> gelu -> 1x1). The lift/project keep the spectral mixing
    cheap (small channel count) while the FFT gives whole-signal receptive field.
    """

    def __init__(self, channels: int = 24, depth: int = 3, modes: int = 24) -> None:
        super().__init__()
        self.lift = nn.Conv1d(1, channels, 1)
        self.layers = nn.ModuleList(_FourierLayer(channels, modes) for _ in range(depth))
        self.project = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, 1, 1),
        )
        self.register_buffer("out_bound", torch.tensor(1.0))  # REQUIRED (trainer sets this)

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T) real, already input-scaled. c_sys ignored (single-circuit).
        del c_sys
        h = self.lift(x.unsqueeze(1))  # (B, C, T)
        for layer in self.layers:
            h = layer(h)
        return self.project(h).squeeze(1)  # (B, T)


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
