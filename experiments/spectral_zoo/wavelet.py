"""Wavelet-operator distortion net (spectral-zoo variant "wavelet").

Idea
----
A multi-level orthogonal DWT (learnable, init = Daubechies-4) splits the signal
into critically-sampled subbands.  Each subband is processed by a *small*
time-domain nonlinearity + a 1x1 channel mix; an inverse DWT reconstructs the
signal, and a final residual time nonlinearity closes the loop.

Why this can match a TCN's harmonic generation, cheaper
-------------------------------------------------------
These circuits are ``LTI-filter ∘ instantaneous nonlinearity ∘ LTI-filter``.  A
TCN spends most of its FLOPs on the two *long linear* convolutions (the formant /
tone-stack memory); the harmonics come from a *local* waveshaper.  A wavelet
transform is exactly a cheap, perfectly-invertible bank of those long
convolutions: one DWT level is an O(N) depthwise stride-2 conv, and L levels give
a logarithmic, multi-resolution decomposition for free (lowpass branch keeps
halving the rate, so total work is < 2N taps regardless of L).  We then place the
NONLINEARITY *inside each subband*, at that band's natural (decimated) rate:

* harmonics of a sharp corner are wideband, but a circuit's *audible* harmonic
  structure is band-limited and shaped per octave — a waveshaper run on the
  detail coefficients of one octave injects harmonics whose spectral support is
  controlled by that subband's synthesis filter, so the operator naturally
  produces *band-shaped* harmonics (what the formant filtering does physically);
* the mixed activation (tanh/relu/abs/snake) supplies genuine corners so a
  slope-discontinuity (crossover / hard-clip) can be synthesized natively;
* a residual full-rate waveshaper guarantees full-band harmonic content even for
  the highest octave, so the net is never spectrally starved.

Because the per-band nonlinearity acts on *decimated* coefficients, the dominant
1x1 mixing cost is shared across far fewer samples than a full-rate TCN of the
same channel width — multi-resolution buys depth (receptive field) at log cost.
"""

from __future__ import annotations

import torch
from torch import nn

from vguitar.models.tcn import _MixedActivation

APPROACH = "Learnable multi-level DWT -> per-subband mixed nonlinearity + 1x1 mix -> IDWT + residual waveshaper"
COST = (
    "~6.7k params @ ch=24 (1x1 mixes are channels^2 each, shared over few blocks). "
    "Per-sample FLOP order O(channels^2): the 1x1 mixes dominate, but each subband "
    "block runs at a DECIMATED rate (sum over octaves of 2^-l < 2 full-rate passes), "
    "so total multiply-adds are well below a same-width full-rate TCN. DWT/IDWT add "
    "only O(channels*taps) per sample. Streaming/causal: feasible offline-equivalent "
    "via per-level FIR ring buffers (orthogonal DWT is FIR -> finite latency); this "
    "offline build uses reflect padding (non-causal), latency = synthesis group delay "
    "~= taps*2^levels samples (low single-digit ms at 44.1 kHz)."
)

# Daubechies-4 (db2) orthonormal scaling (lowpass) coefficients — a compact,
# smooth wavelet that decorrelates audio well while staying short (4 taps).
_DB4_LO = (
    0.48296291314469025,
    0.836516303737469,
    0.22414386804185735,
    -0.12940952255092145,
)


def _qmf_hi(lo: torch.Tensor) -> torch.Tensor:
    """Quadrature-mirror highpass from a lowpass: g[k] = (-1)^k h[L-1-k]."""
    rev = torch.flip(lo, dims=[0])
    signs = torch.tensor([(-1.0) ** k for k in range(lo.numel())], dtype=lo.dtype)
    return signs * rev


class _DWT1d(nn.Module):
    """One level of an (optionally learnable) orthogonal DWT and its inverse.

    Analysis: depthwise conv with the lowpass/highpass pair, stride 2.  Synthesis:
    the matching stride-2 transposed convs.  Filters are stored as a single
    learnable lowpass; the highpass is derived by the QMF relation each call, so a
    learned filter stays a valid (perfect-reconstruction-shaped) wavelet pair.
    Reflect padding keeps the transform length-exact for arbitrary even inputs.
    """

    def __init__(self, learnable: bool = True) -> None:
        super().__init__()
        lo = torch.tensor(_DB4_LO, dtype=torch.float32)
        self.taps = lo.numel()
        if learnable:
            self.lo = nn.Parameter(lo)
        else:
            self.register_buffer("lo", lo)

    def _filters(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo = self.lo
        # renormalize so analysis energy stays ~unit even as the filter is learned
        lo = lo / (lo.norm() + 1e-8)
        hi = _qmf_hi(lo)
        return lo, hi

    def analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` (B, C, T even) -> (approx, detail) each (B, C, T/2)."""
        b, c, t = x.shape
        lo, hi = self._filters()
        w = torch.stack([lo, hi]).unsqueeze(1)  # (2, 1, taps)
        w = w.repeat(c, 1, 1)  # depthwise: (2C, 1, taps)
        # reflect needs pad < length; for tiny bands fall back to zero pad.
        pad = self.taps - 1
        if t > pad:
            xp = nn.functional.pad(x, (pad, pad), mode="reflect")
        else:
            xp = nn.functional.pad(x, (pad, pad), mode="constant")
        y = nn.functional.conv1d(xp, w, stride=2, groups=c)  # (B, 2C, ?)
        y = y.view(b, c, 2, -1)
        half = t // 2
        a = y[:, :, 0, :half]
        d = y[:, :, 1, :half]
        return a, d

    def synthesis(self, a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`analysis`; (B, C, T/2) pair -> (B, C, T)."""
        b, c, half = a.shape
        lo, hi = self._filters()
        w = torch.stack([lo, hi]).unsqueeze(1).repeat(c, 1, 1)  # (2C,1,taps)
        coef = torch.stack([a, d], dim=2).view(b, 2 * c, half)  # interleave bands
        up = nn.functional.conv_transpose1d(coef, w, stride=2, groups=c)  # (B, C, 2*half+taps-2)
        # transposed conv adds (taps-2) trailing samples; center-crop to 2*half
        start = (up.shape[-1] - 2 * half) // 2
        return up[:, :, start : start + 2 * half]


class _SubbandBlock(nn.Module):
    """Per-subband processor: mixed time-nonlinearity + 1x1 channel mix (residual)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act = _MixedActivation(channels)
        self.mix1 = nn.Conv1d(channels, channels, 1)
        self.mix2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix2(self.act(self.mix1(x)))


class Net(nn.Module):
    """Multi-resolution wavelet waveshaper.

    Lift to ``channels`` -> ``levels`` cascaded DWTs (process each detail band and
    the final approximation with a :class:`_SubbandBlock`) -> IDWT back -> a final
    residual full-rate mixed waveshaper -> project to one channel.
    """

    out_bound: torch.Tensor

    def __init__(self, channels: int = 24, levels: int = 4, learnable_dwt: bool = True) -> None:
        super().__init__()
        self.levels = levels
        self.lift = nn.Conv1d(1, channels, 1)
        # One DWT module per level (its own learnable filter) + one block per band.
        self.dwts = nn.ModuleList(_DWT1d(learnable=learnable_dwt) for _ in range(levels))
        self.detail_blocks = nn.ModuleList(_SubbandBlock(channels) for _ in range(levels))
        self.approx_block = _SubbandBlock(channels)
        # Final full-rate residual waveshaper guarantees full-band harmonics.
        self.res_act = _MixedActivation(channels)
        self.res_mix = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, 1, 1)
        self.register_buffer("out_bound", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        del c_sys  # single-circuit screen
        t_in = x.shape[-1]
        # Pad to a multiple of 2**levels so every decimation stays length-exact.
        m = 1 << self.levels
        pad = (-t_in) % m
        xp = nn.functional.pad(x.unsqueeze(1), (0, pad)) if pad else x.unsqueeze(1)
        h = self.lift(xp)  # (B, C, T)

        # --- analysis: cascade DWTs on the lowpass branch, keep details ---
        details: list[torch.Tensor] = []
        a = h
        for lvl in range(self.levels):
            a, d = self.dwts[lvl].analysis(a)
            details.append(self.detail_blocks[lvl](d))
        a = self.approx_block(a)

        # --- synthesis: reconstruct from the (processed) coarse-to-fine bands ---
        for lvl in reversed(range(self.levels)):
            a = self.dwts[lvl].synthesis(a, details[lvl])

        # --- residual full-rate waveshaper + readout ---
        a = a + self.res_mix(self.res_act(a))
        y = self.proj(a).squeeze(1)  # (B, T_padded)
        return y[:, :t_in]


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
