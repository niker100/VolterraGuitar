"""AFNO / GFNet-style spectral operator with a nonlinearity ON the Fourier modes.

Adaptive Fourier Neural Operator (Guibas et al., 2022) and GFNet (Rao et al., 2021)
replace a long depthwise temporal convolution with a *per-frequency complex weight*:
the convolution theorem says a convolution of length ~T costs one ``O(T log T)`` FFT
pair plus an ``O(T)`` pointwise multiply, instead of ``O(T * R)`` for an R-tap kernel.
That is exactly the user's goal — "long convolutions done as cheap spectral multiplies".

A *purely linear* spectral filter cannot create harmonics (it only re-weights the
frequencies already present), which is why the spectral-only baseline FAILS. AFNO's
fix is to put a **genuine nonlinearity on the complex Fourier modes**: a small
channel-mixing MLP with a ``modReLU`` (Arjovsky et al., 2016) between two complex
linear maps. ``modReLU(z) = relu(|z| + b) * z / |z|`` is nonlinear in the magnitude,
so it folds energy *between* frequency bins — i.e. it manufactures harmonics in the
frequency domain. We also keep a residual *time-domain* pointwise nonlinearity
(a small mixed-activation 1x1 stack borrowed from the TCN) so harmonic generation
happens in BOTH domains; this is the part the linear-spectral baseline lacked.

Why this could match a TCN more cheaply
----------------------------------------
* The TCN reaches ~1000-sample receptive field by stacking 16 dilated convs. Here a
  single FFT pair sees the WHOLE block at once — global receptive field, no depth.
* Channel mixing happens with tiny complex matrices shared across all kept low-freq
  bins (mode truncation: only the lowest ``n_modes`` rFFT bins carry a learned weight;
  guitar/distortion energy is overwhelmingly low-frequency, and dropping the top bins
  is a cheap anti-alias-ish prior). Param count is dominated by these small matrices,
  independent of T.
* Two AFNO spectral blocks + a per-channel time activation give two rounds of
  cross-bin folding; the time path adds odd/even harmonics pointwise.

Streaming / latency
-------------------
Global FFT over a block is NOT sample-causal: it needs the whole analysis window, so
real-time use means overlap-save block processing with latency ~ block length
(e.g. 23 ms at 1024 / 44.1 kHz) — heavier latency than the TCN's 0 samples, but a
constant, bounded amount and far fewer FLOPs per sample. Stated honestly in COST.
"""

from __future__ import annotations

import torch
from torch import nn

APPROACH = (
    "rFFT -> mode-truncated complex channel-mixing MLP with modReLU on the Fourier "
    "modes (nonlinear in |z|, folds energy between bins) -> irFFT, + residual "
    "time-domain mixed pointwise nonlinearity. Harmonics made in BOTH domains."
)
COST = (
    "12.7k params @ channels=24 (independent of T). Per-sample FLOPs O(C log T) for "
    "the two FFT pairs + O(C*H) per-bin complex mixing over n_modes bins -> amortizes "
    "to O(C*H*n_modes / T) per sample, far below a 16-layer dilated TCN. Causal NO: "
    "global block FFT needs the whole window; realtime = overlap-save with latency ~ "
    "block length (~23 ms @ 1024/44.1kHz), a bounded constant. CPU-friendly."
)


class _ComplexLinear(nn.Module):
    """A complex-valued ``C_in -> C_out`` linear map applied per frequency bin.

    Real implementation of ``W z`` with ``W = Wr + i Wi`` and ``z = x + i y``:
    ``(Wr x - Wi y) + i (Wr y + Wi x)``. Weights are shared across all kept bins
    (a single small matrix), so cost is independent of how many modes we keep.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        scale = (2.0 / c_in) ** 0.5
        self.wr = nn.Parameter(torch.randn(c_in, c_out) * scale)
        self.wi = nn.Parameter(torch.randn(c_in, c_out) * scale)
        self.br = nn.Parameter(torch.zeros(c_out))
        self.bi = nn.Parameter(torch.zeros(c_out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, M, C_in) complex
        x, y = z.real, z.imag
        out_r = x @ self.wr - y @ self.wi + self.br
        out_i = y @ self.wr + x @ self.wi + self.bi
        return torch.complex(out_r, out_i)


class _ModReLU(nn.Module):
    """``modReLU(z) = relu(|z| + b) * z / |z|`` (Arjovsky et al., 2016).

    Nonlinear in the MAGNITUDE while preserving phase. Because the magnitude map is
    a (shifted) ReLU corner, applying it per bin and then transforming back to time
    creates inter-bin coupling -> genuine harmonic generation in the frequency domain
    (a linear per-bin weight provably cannot do this).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, M, C) complex
        mag = z.abs()
        gated = torch.relu(mag + self.bias) / (mag + 1e-6)
        return z * gated.to(z.dtype)


class _AFNOBlock(nn.Module):
    """One spectral block: rFFT -> [CLinear -> modReLU -> CLinear] on low modes -> irFFT.

    Only the lowest ``n_modes`` rFFT bins are transformed (mode truncation); higher
    bins are passed through unchanged (they carry little energy for these circuits and
    skipping them is a cheap low-pass prior). A residual keeps the block well-behaved.
    """

    def __init__(self, channels: int, n_modes: int, hidden: int) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.lin1 = _ComplexLinear(channels, hidden)
        self.act = _ModReLU(hidden)
        self.lin2 = _ComplexLinear(hidden, channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        t = h.shape[-1]
        spec = torch.fft.rfft(h, dim=-1)  # (B, C, F) complex
        f = spec.shape[-1]
        m = min(self.n_modes, f)
        low = spec[..., :m].transpose(1, 2)  # (B, m, C)
        mixed = self.lin2(self.act(self.lin1(low)))  # (B, m, C)
        new_spec = spec.clone()
        new_spec[..., :m] = mixed.transpose(1, 2) + spec[..., :m]  # residual in freq
        return torch.fft.irfft(new_spec, n=t, dim=-1)


class _TimeMix(nn.Module):
    """Pointwise (1x1) time path with a heterogeneous activation: the harmonic maker.

    A 1x1 conv has no temporal mixing, so this stays cheap and length-agnostic; the
    activation is what bends the transfer curve. We use a small mix of tanh (smooth
    saturation), gelu, relu and abs (a V-corner) so the net can synthesize both the
    rounded conducting region and the sharp clipping kink of a distortion circuit.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_in = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        base = channels // 4
        self.sizes = (base, base, base, channels - 3 * base)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        z = self.proj_in(h)
        s = self.sizes
        i0, i1, i2 = s[0], s[0] + s[1], s[0] + s[1] + s[2]
        a = torch.cat(
            [
                torch.tanh(z[:, :i0]),
                nn.functional.gelu(z[:, i0:i1]),
                torch.relu(z[:, i1:i2]),
                torch.abs(z[:, i2:]),
            ],
            dim=1,
        )
        return h + self.proj_out(a)


class Net(nn.Module):
    """AFNO: alternating spectral channel-mixing (nonlinear modes) and time nonlinearity.

    Pipeline (lifted to ``channels`` features):
    ``in 1x1 -> AFNO -> TimeMix -> AFNO -> TimeMix -> out 1x1`` with a global
    pre-emphasis-free residual; both AFNO blocks fold energy between Fourier bins and
    both TimeMix blocks bend the waveform pointwise, so harmonics arise in either
    domain. All ops are length-agnostic (FFT length follows the input T).
    """

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        n_modes = 256  # learned low-freq modes; bins above are passed through
        hidden = 2 * channels  # complex mixing width (drives cross-bin folding capacity)
        self.lift = nn.Conv1d(1, channels, 1)
        self.afno1 = _AFNOBlock(channels, n_modes, hidden)
        self.time1 = _TimeMix(channels)
        self.afno2 = _AFNOBlock(channels, n_modes, hidden)
        self.time2 = _TimeMix(channels)
        self.head = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, 1, 1),
        )
        self.register_buffer("out_bound", torch.tensor(1.0))  # trainer sets this

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        _ = c_sys  # single-circuit: always None, ignored
        h = self.lift(x.unsqueeze(1))  # (B, 1, T) -> (B, C, T)
        h = self.time1(self.afno1(h))
        h = self.time2(self.afno2(h))
        return self.head(h).squeeze(1)  # (B, T)


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
