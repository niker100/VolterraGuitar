"""Recurrent spectral operator: STFT frames -> per-frame time-domain waveshaper
-> spectral encode -> GRU across frames (long memory) -> spectral decode -> ISTFT.

Why this can match a TCN's harmonic generation while being cheaper
------------------------------------------------------------------
A guitar distortion circuit is a long-memory (reactive: caps/inductors) system
wrapped around a pointwise nonlinearity (diode/BJT/JFET). A TCN spends almost all
of its parameters/FLOPs building a long receptive field out of many short dilated
convolutions so it can see far enough back to resolve that memory.

Here the long-memory job is split into two cheap pieces:

* **Across-frame memory is a GRU.** One STFT hop (here 256 @ 44.1 kHz ~ 6 ms) of
  latent state is carried frame-to-frame by a small GRU. Its receptive field is
  effectively unbounded (IIR-like) at a per-hop cost of one GRU step — far cheaper
  than stacking dilations to reach 1000+ taps. This is the "long convolution as a
  recurrence" trade.
* **Within-frame mixing is spectral.** Each frame's long linear filtering is a
  complex multiply / small MLP in the STFT domain (O(F) per frame), replacing the
  long time-domain convolution a TCN approximates with many taps.

The harmonics — which a purely-linear spectral filter provably cannot create —
come from a genuine **time-domain pointwise waveshaper** applied to the *windowed
time-domain frames* before analysis, plus a learnable mixed activation inside the
per-frame encoder. Generating harmonics in time and doing the long linear work in
the spectral/recurrent domain is the whole point.

Streaming / latency
--------------------
The STFT is causal-friendly: process one hop at a time, keep the GRU hidden state
and a 1-frame overlap-add tail. Algorithmic latency = one window (here 512 samples
~ 12 ms) from the OLA reconstruction; reducible by shrinking the window. So this is
plausibly real-time, though heavier per-block than the fully-causal TCN.
"""

from __future__ import annotations

import torch
from torch import nn

from vguitar.models.tcn import _MixedActivation

APPROACH = (
    "STFT frames -> per-frame time-domain waveshaper (harmonics) -> spectral encode "
    "-> GRU across frames (long memory) -> spectral decode -> overlap-add ISTFT + "
    "time-domain nonlinear residual."
)
COST = (
    "38.5k params @ channels=24 (well under the 120k budget). Per-hop cost "
    "O(C*F) for the complex spectral filter/mix + O(H^2) for one GRU step "
    "(H=2C) plus two size-win FFTs (O(win*log win)); amortized per sample this "
    "is small and roughly constant in T. Streaming-feasible: carry GRU state + "
    "1-frame OLA tail, process one hop at a time. Algorithmic latency ~ one "
    "window (512 samples ~ 12 ms @ 44.1 kHz), shrinkable by reducing win; "
    "not zero-latency like the fully-causal TCN."
)


class _Waveshaper(nn.Module):
    """Per-channel learnable time-domain pointwise nonlinearity (harmonic source).

    Acts on the windowed time-domain frames *before* spectral analysis, so the new
    spectral content it injects is genuine harmonic distortion (a linear spectral
    op downstream can shape but never create it). A learnable mix of tanh (smooth
    saturation), a biased corner (asymmetric clip, like a single diode), and a
    cubic term (odd expansion) covers the common BJT/JFET/diode transfer shapes.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.w_tanh = nn.Parameter(torch.ones(1, channels, 1))
        self.w_corner = nn.Parameter(torch.zeros(1, channels, 1))
        self.w_cubic = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L) -> (B, C, L)
        z = self.pre * x + self.bias
        cubic = z - z.pow(3) / 3.0  # soft odd expansion, bounded slope near 0
        return self.w_tanh * torch.tanh(z) + self.w_corner * torch.relu(z) + self.w_cubic * cubic


class Net(nn.Module):
    """Recurrent spectral operator (see module docstring).

    Pipeline (offline, any length T):
      1. Lift the 1-channel signal to ``channels`` time-domain channels (1x1 conv).
      2. Frame with a Hann window (win=512, hop=256), apply :class:`_Waveshaper`
         per channel -> genuine time-domain harmonics inside each frame.
      3a. rFFT each frame; multiply by a learnable per-(channel, bin) COMPLEX gain
          (the long convolution as a cheap frequency-domain filter).
      3b. Mean-pool the filtered magnitude into ``n_bands`` bands per channel -> a
          compact per-frame spectral envelope; a small :class:`_MixedActivation`
          encoder maps it to the GRU latent.
      4. A GRU runs across frames carrying long memory (IIR-like receptive field).
      5. Decode the GRU state to a per-channel COMPLEX modulation gain that
         re-weights the filtered spectrum, then a learnable complex channel-mix
         collapses channels to one spectrum per frame -> irFFT -> windowed frames.
      6. Overlap-add back to a (B, T) signal; add a light time-domain nonlinear
         residual path so DC/edge content and very-local nonlinearity survive the
         STFT round-trip.
    """

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.channels = channels
        self.win = 512
        self.hop = 256
        n_freq = self.win // 2 + 1  # rFFT bins
        self.register_buffer("window", torch.hann_window(self.win, periodic=True))

        self.n_freq = n_freq

        # 1. time-domain lift + per-frame waveshaper (harmonic source)
        self.lift = nn.Conv1d(1, channels, 1)
        self.shaper = _Waveshaper(channels)

        # 3a. cheap LINEAR spectral mixing: a learnable per-(channel, bin) COMPLEX
        # gain. This IS the long time-domain convolution, done as a frequency
        # multiply (O(F) per channel-frame, not O(taps)). Stored as real/imag.
        self.filt_re = nn.Parameter(torch.ones(channels, n_freq) * 0.1)
        self.filt_im = nn.Parameter(torch.zeros(channels, n_freq))

        # 3b. per-frame summary fed to the GRU: pool the post-filter magnitude over a
        # few frequency bands per channel -> compact vector (keeps the GRU small).
        self.n_bands = 8
        summ_in = channels * self.n_bands
        hidden = 2 * channels
        self.enc = nn.Sequential(
            nn.Linear(summ_in, hidden),
            _LinActWrap(_MixedActivation(hidden)),
        )

        # 4. recurrence across frames -> long IIR-like memory
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)

        # 5. decode GRU state -> a per-(out-channel) COMPLEX modulation gain that
        # multiplies the filtered spectrum, plus a channel->1 spectral mix. The
        # GRU thus supplies frame-to-frame memory by re-weighting the spectrum.
        self.mod = nn.Linear(hidden, 2 * channels)  # complex gain per channel
        self.chan_mix_re = nn.Parameter(torch.ones(channels) / channels)
        self.chan_mix_im = nn.Parameter(torch.zeros(channels))

        # 6. light time-domain nonlinear residual (preserves very-local detail)
        self.res_in = nn.Conv1d(1, channels, 1)
        self.res_shaper = _Waveshaper(channels)
        self.res_out = nn.Conv1d(channels, 1, 1)
        self.mix = nn.Parameter(torch.tensor(0.1))

        self.register_buffer("out_bound", torch.tensor(1.0))  # REQUIRED (trainer sets this)

    def _stft_frames(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """``x``: (B, C, T) -> windowed frames (B, C, n_frames, win); returns pad."""
        t = x.shape[2]
        # center-pad so frame i covers [i*hop, i*hop+win); guarantees >=1 frame and
        # full coverage of the signal for any T.
        n_frames = max(1, (t + self.hop - 1) // self.hop)
        needed = (n_frames - 1) * self.hop + self.win
        pad = needed - t
        xp = nn.functional.pad(x, (0, pad))
        # unfold into overlapping frames
        frames = xp.unfold(dimension=2, size=self.win, step=self.hop)  # (B,C,n_frames,win)
        frames = frames * self.window.view(1, 1, 1, -1)
        return frames, pad

    def _overlap_add(self, frames: torch.Tensor, t: int, pad: int) -> torch.Tensor:
        """Inverse of :meth:`_stft_frames` for a single channel: (B, n_frames, win)
        -> (B, T) via windowed overlap-add with window-power normalization."""
        b, n_frames, win = frames.shape
        frames = frames * self.window.view(1, 1, -1)
        length = t + pad
        # fold expects (B, C*kernel, L) with C=1
        folded = nn.functional.fold(
            frames.transpose(1, 2),  # (B, win, n_frames)
            output_size=(1, length),
            kernel_size=(1, win),
            stride=(1, self.hop),
        ).view(b, length)
        # normalize by summed squared window (OLA energy) to invert the analysis win
        wsq = (self.window**2).view(1, 1, -1).expand(1, n_frames, -1)
        norm = nn.functional.fold(
            wsq.transpose(1, 2),
            output_size=(1, length),
            kernel_size=(1, win),
            stride=(1, self.hop),
        ).view(1, length)
        out = folded / norm.clamp(min=1e-8)
        return out[:, :t]

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        del c_sys  # single-circuit screen
        b, t = x.shape
        xin = x.unsqueeze(1)  # (B, 1, T)

        c, nf = self.channels, self.n_freq

        # 1-2. lift + per-frame time-domain waveshaping (HARMONIC generation in time)
        h = self.lift(xin)  # (B, C, T)
        frames, pad = self._stft_frames(h)  # (B, C, n_frames, win)
        n_frames = frames.shape[2]
        frames = self.shaper(frames.reshape(b, c, n_frames * self.win)).reshape(
            b, c, n_frames, self.win
        )

        # 3a. analyze + LINEAR spectral mixing (per-(channel,bin) complex multiply =
        # the long convolution as a cheap frequency-domain gain).
        spec = torch.fft.rfft(frames, dim=-1)  # (B, C, n_frames, nf) complex
        filt = torch.complex(self.filt_re, self.filt_im).view(1, c, 1, nf)
        filtered = spec * filt  # (B, C, n_frames, nf)

        # 3b. compact per-frame summary for the GRU: mean band-magnitudes per channel.
        mag = filtered.abs()  # (B, C, n_frames, nf)
        bands = self._band_pool(mag)  # (B, C, n_frames, n_bands)
        summ = bands.permute(0, 2, 1, 3).reshape(b, n_frames, c * self.n_bands)
        enc = self.enc(summ)  # (B, n_frames, hidden)

        # 4. recurrence across frames -> long memory
        rec, _ = self.gru(enc)  # (B, n_frames, hidden)

        # 5. decode to a per-(channel) complex modulation gain, apply, then mix
        # channels down to one complex spectrum per frame.
        mod = self.mod(rec)  # (B, n_frames, 2C)
        mod_c = torch.complex(mod[..., :c], mod[..., c:])  # (B, n_frames, C)
        modulated = filtered * mod_c.permute(0, 2, 1).unsqueeze(-1)  # (B, C, n_frames, nf)
        cmix = torch.complex(self.chan_mix_re, self.chan_mix_im).view(1, c, 1, 1)
        out_spec = (modulated * cmix).sum(dim=1)  # (B, n_frames, nf) complex
        out_frames = torch.fft.irfft(out_spec, n=self.win, dim=-1)  # (B, n_frames, win)

        # 6. overlap-add back to time domain
        y_spec = self._overlap_add(out_frames, t, pad)  # (B, T)

        # local time-domain nonlinear residual
        r = self.res_out(self.res_shaper(self.res_in(xin))).squeeze(1)  # (B, T)
        return y_spec + self.mix * r

    def _band_pool(self, mag: torch.Tensor) -> torch.Tensor:
        """Mean-pool the (B, C, n_frames, nf) magnitude into ``n_bands`` contiguous
        frequency bands -> (B, C, n_frames, n_bands). Cheap fixed pooling (no params)
        gives the GRU a low-dimensional, length-stable spectral envelope per frame."""
        nf = mag.shape[-1]
        edges = torch.linspace(0, nf, self.n_bands + 1).long()
        out = [
            mag[..., edges[i] : edges[i + 1]].mean(dim=-1, keepdim=True)
            for i in range(self.n_bands)
        ]
        return torch.cat(out, dim=-1)


class _LinActWrap(nn.Module):
    """Adapt :class:`_MixedActivation` (expects (B, C, T)) to a (B, ..., C) Linear
    stack by treating the feature axis as channels with a dummy time axis."""

    def __init__(self, act: nn.Module) -> None:
        super().__init__()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., C) -> (..., C)
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).unsqueeze(-1)  # (N, C, 1)
        out = self.act(flat).squeeze(-1)  # (N, C)
        return out.reshape(shape)


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
