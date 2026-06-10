"""STFT-mixing spectral operator: cheap per-bin complex channel-mix + time nonlinearity.

Idea (the "stft_mix" family)
----------------------------
A dilated-TCN spends most of its parameters/FLOPs learning *long* temporal
convolutions (large receptive field via stacked dilations). The expensive part
of an analog-circuit emulator is exactly that long *linear* memory: tone-stack
shelving, coupling/DC-blocking high-pass, reactive loading — all linear,
time-invariant, and long. The cheap-but-essential part is the *pointwise*
nonlinearity that actually manufactures harmonics (diode/BJT/JFET saturation).

So we factor the two:

* **Linear long memory -> spectral 1x1 mix.** A framed STFT turns a long
  time-domain convolution into a per-frequency-bin complex multiply. We give
  each block a *complex 1x1 channel-mixing* on the STFT coefficients: for every
  bin, a learned ``C x C`` complex matrix mixes the channels. Per bin this is a
  cheap pointwise op, yet across all bins it realises a *different* linear filter
  per channel-pair (a full LTI mixing matrix), i.e. arbitrarily long FIR/IIR-like
  memory of length ~``n_fft`` — for ``O(C^2)`` params and ``O(C^2)`` mults per
  bin, far below a dilated-conv stack covering the same span.

* **Harmonic generation -> time-domain mixed activation.** Between STFT mixers we
  drop back to the time domain and apply the project's heterogeneous
  :class:`_MixedActivation` (tanh / gelu / relu / abs / snake). A purely-spectral
  linear filter provably *cannot* create new harmonics (proven to FAIL in this
  search at ESR 0.25); the time-domain pointwise nonlinearity is what folds energy
  into higher harmonics, exactly as the TCN's gated activation does.

Stacking 2-3 ``[STFT-mix -> ISTFT -> time mixed-act]`` blocks therefore interleaves
"long linear memory (cheap, spectral)" with "harmonic creation (cheap, pointwise)"
— the same alternation a gated TCN performs, but with the long convolution priced
as an FFT + per-bin matmul instead of a deep dilation cascade.

Why it could match a TCN more cheaply
--------------------------------------
* One STFT mixer with ``n_fft=256`` gives every channel a learned linear filter of
  effective length ~256 samples in a *single* layer; a kernel-3 dilated TCN needs
  ~7-8 stacked layers (dilations 1..128) to reach a comparable span. The mixer's
  cost is ``O(C^2)`` per bin (independent of filter length) vs the TCN's
  ``O(depth * C^2 * k)``.
* The spectral mix is global-per-frame, so a single block already sees long
  context; depth is then spent purely on *nonlinear* refinement, not on growing
  the receptive field.

Streaming / latency
-------------------
Overlap-add STFT is streaming-friendly: with hop ``H`` and frame ``n_fft`` the
algorithm is causal up to one analysis frame of look-back, giving an algorithmic
latency of ~``n_fft`` samples (~5.8 ms at 44.1 kHz for n_fft=256) — higher than the
TCN's 0 but well within musical-monitoring tolerance, and the per-block work is
O(frame * log frame + C^2 * bins), constant per hop. (This offline screen uses a
single whole-signal STFT for simplicity; the cost statement reflects the streaming
overlap-add form.)
"""

from __future__ import annotations

import torch
from torch import nn

from vguitar.models.archive.tcn import _MixedActivation

APPROACH = (
    "Lift to channels; alternate [framed-STFT complex 1x1 per-bin channel-mix -> ISTFT] "
    "(cheap long LTI memory) with a time-domain mixed activation (tanh/gelu/relu/abs/snake) "
    "that generates the harmonics. Overlap-add, streaming-friendly."
)
COST = (
    "~24.5k params @ C=24 (lift + 3 blocks of [shared complex CxC mix 2*C^2 + per-bin "
    "complex diag 2*C*n_bins + mixed-act + 1x1 recomb] + 1x1 head). Per-sample FLOP order "
    "O(C*log(n_fft) + C^2) = FFT (amortised over hop H) + one CxC complex matmul per bin. "
    "Streaming-friendly via overlap-add; algorithmic latency ~n_fft samples (~5.8 ms "
    "@44.1k, n_fft=256), per-hop work constant. Real-time plausible (FFT + small matmuls)."
)


class _STFTMix(nn.Module):
    """One spectral block: STFT -> per-bin complex 1x1 channel-mix -> ISTFT.

    The complex channel-mix is a single learned ``C x C`` complex matrix applied to
    every frequency bin's channel vector. Sharing one matrix across bins keeps the
    parameter count at ``2*C^2`` (real + imag) while still realising a non-trivial
    LTI filter per channel-pair: because the input STFT coefficients differ per bin,
    a per-bin complex scaling of the (shared) mix yields a frequency-dependent
    response. A small *per-bin complex gain* (a learned diagonal of length
    ``n_bins`` per channel) supplies the bin-selective shelving cheaply
    (``2*C*n_bins`` params) — together they give a full, learnable linear filter
    bank without an ``O(C^2 * n_bins)`` blow-up.
    """

    def __init__(self, channels: int, n_fft: int = 256, hop_frac: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.n_fft = n_fft
        self.hop = n_fft // hop_frac
        n_bins = n_fft // 2 + 1
        # Shared complex CxC channel-mixing (real + imag parts), init near identity.
        eye = torch.eye(channels)
        self.mix_re = nn.Parameter(eye + 0.02 * torch.randn(channels, channels))
        self.mix_im = nn.Parameter(0.02 * torch.randn(channels, channels))
        # Per-bin complex diagonal gain (the cheap frequency-selective shelving),
        # init at unit gain (re=1, im=0) so the block starts ~linear pass-through.
        self.bin_re = nn.Parameter(torch.ones(channels, n_bins))
        self.bin_im = nn.Parameter(torch.zeros(channels, n_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        b, c, t = x.shape
        n_fft, hop = self.n_fft, self.hop
        # Zero-pad both ends by n_fft so even very short signals (T < n_fft) yield
        # >=1 STFT frame and we always have a clean [pad:pad+t] crop region after
        # ISTFT. Constant pad (unlike reflect) has no size restriction, so this is
        # safe for ANY T (including T=1). torch.stft(center=True) additionally
        # reflect-pads internally for framing.
        pad = n_fft
        xp = nn.functional.pad(x, (pad, pad), mode="constant", value=0.0)
        tp = xp.shape[-1]
        # Build the analysis/synthesis window inline (device/dtype-matched), matching
        # the repo's multi_stft_loss convention — a constant, so no buffer needed.
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        # STFT over the batched (B*C) channel rows.
        spec = torch.stft(
            xp.reshape(b * c, tp),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=win,
            center=True,
            return_complex=True,
        )  # (B*C, n_bins, n_frames)
        nb, n_frames = spec.shape[1], spec.shape[2]
        spec = spec.reshape(b, c, nb, n_frames)
        # Per-bin complex diagonal gain (broadcast over batch & frames).
        gain = torch.complex(self.bin_re, self.bin_im)[None, :, :, None]  # (1,C,nb,1)
        spec = spec * gain
        # Shared complex CxC channel mix: einsum over channels for every (bin, frame).
        mix = torch.complex(self.mix_re, self.mix_im)  # (C, C)
        spec = torch.einsum("oc,bcft->boft", mix, spec)
        # ISTFT back to time.
        spec = spec.reshape(b * c, nb, n_frames)
        y = torch.istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=win,
            center=True,
            length=tp,
        )  # (B*C, tp)
        y = y.reshape(b, c, tp)
        return y[..., pad : pad + t]


class Net(nn.Module):
    """Stacked STFT-mix + time-domain mixed-activation spectral operator."""

    def __init__(self, channels: int = 24, n_blocks: int = 3, n_fft: int = 256) -> None:
        super().__init__()
        self.channels = channels
        self.lift = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            _STFTMix(channels, n_fft=n_fft) for _ in range(n_blocks)
        )
        # Each spectral block is followed by a time-domain heterogeneous nonlinearity
        # (the harmonic generator) plus a pointwise 1x1 recombination + residual.
        self.acts = nn.ModuleList(_MixedActivation(channels) for _ in range(n_blocks))
        self.recomb = nn.ModuleList(nn.Conv1d(channels, channels, 1) for _ in range(n_blocks))
        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, 1, 1),
        )
        self.register_buffer("out_bound", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T) already input-scaled. c_sys ignored (single circuit).
        del c_sys
        h = self.lift(x.unsqueeze(1))  # (B, C, T)
        for block, act, recomb in zip(self.blocks, self.acts, self.recomb, strict=True):
            mixed = block(h)  # spectral long-memory linear mix
            g = act(mixed)  # time-domain pointwise nonlinearity -> harmonics
            h = h + recomb(g)  # residual recombination
        return self.out(h).squeeze(1)  # (B, T)


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
