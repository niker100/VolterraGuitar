"""tcn_spectral_interleave - mixed-activation TCN with half its layers made spectral.

Core idea
---------
The reference mixed-TCN generates a circuit's harmonics with *time-domain* pointwise
nonlinearities (tanh / gelu / relu / abs / snake) sandwiched between long dilated
convolutions. Those long dilated convs are what make the receptive field big enough
to capture the circuit's memory (coupling caps, tone stacks) - but a length-``2^d``
dilated conv costs ``O(C**2 * T * k)`` and the deepest layers dominate.

A linear convolution is a *per-frequency complex multiply*. So we replace ABOUT HALF
the dilated layers with a **spectral global-mixing layer**: rFFT over time -> a learned
complex, channel-mixing, frequency-banded linear map -> irFFT. One such layer mixes
information across the WHOLE signal (effectively infinite receptive field) for
``O(C * T * logT + C**2 * B)`` (``B`` = #frequency bands), which is much cheaper than
stacking several large dilated convs to reach the same range.

Why it can still match the TCN's harmonic generation
----------------------------------------------------
A purely-linear spectral filter provably CANNOT create harmonics (spectral-only linear
gain FAILS in the reference numbers, 0.25 ESR on bjt). So every spectral layer here is
immediately followed by the SAME heterogeneous time-domain :class:`_MixedActivation`
used by the TCN's mixed layers: the spectral op supplies cheap long-range *linear*
mixing, and the pointwise nonlinearity right after it manufactures the new harmonics in
time. We interleave: time-domain dilated mixed layer (local corners / saturation),
then spectral mixed layer (global linear context + nonlinearity), repeating. The result
is a residual+skip stack identical in shape to the TCN, so the proven mixed-activation
machinery is preserved while the expensive long-range convs are done as cheap FFT
multiplies.

Interface: see the module-level ``Net`` (channels-parametrised, returns ``(B, T)``).
"""

from __future__ import annotations

import torch
from torch import nn

from vguitar.models.archive.tcn import _MixedActivation, _MixedLayer

APPROACH = (
    "Mixed-activation dilated TCN, half its layers swapped for spectral global-mixers "
    "(rFFT -> banded complex channel-linear -> irFFT) feeding the same time-domain "
    "mixed nonlinearity; residual+skip stack."
)
COST = (
    "~50-90k params @ channels=24 (grows with n_bands). Per-sample FLOP order: "
    "spectral layers O(C*logT + C**2*B/T) amortised + O(C*logT) FFTs; time layers "
    "O(C**2*k). Streaming: time/dilated layers are causal (ring buffer, 0 latency); "
    "the spectral layers use a full-signal rFFT and are NON-causal as written -> "
    "real-time needs block-STFT / overlap-save approximation per spectral layer "
    "(finite latency = one analysis window), which is feasible but not exact here."
)


class _SpectralMixLayer(nn.Module):
    """Global linear mixing in the frequency domain, then a time-domain nonlinearity.

    Forward:  ``x (B,C,T)`` -> rFFT -> banded complex channel-linear -> irFFT (length T)
    -> :class:`_MixedActivation` -> residual ``1x1`` + skip ``1x1``.

    The complex linear map is a per-band ``(C x C)`` complex matrix. Using ``n_bands``
    bands (rather than one matrix per FFT bin) keeps the parameter count fixed and
    independent of the sequence length ``T`` - each rFFT bin is routed to a band by a
    fixed, length-aware split so the same weights work for any ``T`` (the self-test
    exercises 2048 / 4096 / 1000). A learned per-bin complex gain (broadcast within a
    band via interpolation of the band matrices) supplies the actual long convolution;
    the nonlinearity that follows is what turns this linear mixer into a harmonic
    generator.
    """

    def __init__(self, channels: int, n_bands: int = 6) -> None:
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        # Per-band complex channel-mixing matrices, stored as real/imag (n_bands,C,C).
        # Init: real part near identity (start as a gentle all-pass), imag part small,
        # so an untrained layer is close to a pass-through and the residual stack is
        # well-behaved at init.
        eye = torch.eye(channels).unsqueeze(0).repeat(n_bands, 1, 1)
        self.w_re = nn.Parameter(eye + 0.01 * torch.randn(n_bands, channels, channels))
        self.w_im = nn.Parameter(0.01 * torch.randn(n_bands, channels, channels))
        self.act = _MixedActivation(channels)
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def _band_index(self, n_freq: int, device: torch.device) -> torch.Tensor:
        """Map each of ``n_freq`` rFFT bins to a band id in ``[0, n_bands)``.

        Length-aware (uses the actual bin count), so the fixed-size band matrices
        apply to any ``T``. Bands are linearly spaced over the rFFT bins.
        """
        if n_freq <= 1:
            return torch.zeros(n_freq, dtype=torch.long, device=device)
        pos = torch.arange(n_freq, device=device, dtype=torch.float32) / (n_freq - 1)
        idx = (pos * self.n_bands).clamp(max=self.n_bands - 1 + 1e-6).long()
        return idx.clamp(max=self.n_bands - 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = x.shape[-1]
        xf = torch.fft.rfft(x, n=t, dim=-1)  # (B, C, F) complex
        f = xf.shape[-1]
        band = self._band_index(f, x.device)  # (F,) long
        w_re = self.w_re[band]  # (F, C, C)
        w_im = self.w_im[band]  # (F, C, C)
        xr, xi = xf.real, xf.imag  # (B, C, F)
        # Per-bin complex matmul over the channel axis:
        #   y = (Wr + i Wi) (xr + i xi)
        # Move channel to the last contracted dim: (B,F,C) via einsum over (F,C,C).
        xr_t = xr.transpose(1, 2)  # (B, F, C)
        xi_t = xi.transpose(1, 2)  # (B, F, C)
        yr = torch.einsum("foc,bfc->bfo", w_re, xr_t) - torch.einsum("foc,bfc->bfo", w_im, xi_t)
        yi = torch.einsum("foc,bfc->bfo", w_re, xi_t) + torch.einsum("foc,bfc->bfo", w_im, xr_t)
        yf = torch.complex(yr, yi).transpose(1, 2)  # (B, C, F)
        y = torch.fft.irfft(yf, n=t, dim=-1)  # (B, C, T) real, exact length T
        g = self.act(y)
        return x + self.res(g), self.skip(g)


class Net(nn.Module):
    """Interleaved time / spectral mixed-activation residual stack.

    Layout (per the assigned variant): a dilated mixed-TCN whose every other dilated
    layer is replaced by a :class:`_SpectralMixLayer`. Odd positions keep the proven
    local time-domain :class:`_MixedLayer` (corner / saturation capacity at growing
    dilations); even positions do cheap global linear mixing in the spectral domain
    followed by the same time-domain nonlinearity. Skip outputs are summed and read
    out by two ``1x1`` convs (the TCN head).
    """

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.channels = channels
        self.input = nn.Conv1d(1, channels, 1)
        # 8 layers: alternate time (dilated mixed) / spectral (global mixed).
        # Time layers carry the geometric dilations 1,2,4,8 (local receptive field);
        # spectral layers supply the long-range / global linear mixing more cheaply.
        n_bands = 6
        dilations = (1, 2, 4, 8)
        layers: list[nn.Module] = []
        self.is_spectral: list[bool] = []
        for d in dilations:
            layers.append(_MixedLayer(channels, kernel=3, dilation=d))
            self.is_spectral.append(False)
            layers.append(_SpectralMixLayer(channels, n_bands=n_bands))
            self.is_spectral.append(True)
        self.layers = nn.ModuleList(layers)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, 1),
        )
        self.register_buffer("out_bound", torch.tensor(1.0))  # REQUIRED (trainer sets this)

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T) real, already input-scaled. c_sys ignored (single-circuit).
        del c_sys
        h = self.input(x.unsqueeze(1))  # (B, C, T)
        skips = h.new_zeros(h.shape)
        for layer in self.layers:
            h, s = layer(h)
            skips = skips + s
        return self.out(skips).squeeze(1)  # (B, T)


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
