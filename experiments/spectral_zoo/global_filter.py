"""global_filter: conv-as-spectral-multiply, harmonics from time-domain mixed acts.

The literal "replace conv with a spectral multiply" idea. A long linear filter
``y = h * x`` is a *diagonal* multiply in the Fourier basis: ``Y = H .* X``. So
instead of paying ``O(T*K)`` for a length-``K`` time-domain convolution we pay one
rFFT, a per-channel complex multiply (the learned diagonal ``H``), and one irFFT —
``O(T log T)`` total, independent of how long the linear memory is. That makes a
"global" (whole-signal-length) linear receptive field essentially free, which is
the user's goal: long convolutions done as cheap spectral multiplies.

A purely-linear spectral filter cannot create harmonics (it only re-weights the
frequencies already present), so each block is::

    x -> [global spectral filter] -> [+ bias] -> [MIXED pointwise nonlinearity]
      -> [1x1 channel mix] -> residual add

The MIXED nonlinearity (reused from :mod:`vguitar.models.tcn`: tanh / gelu / relu
/ abs / snake, one group per channel) is what synthesizes the distortion harmonics
in TIME; the spectral diagonal does the cheap linear mixing / long-memory shaping
*between* nonlinear stages — exactly the role the dilated convs play in the TCN,
but at ``O(T log T)`` instead of ``O(T*R)``.

Why this can match a TCN cheaply: a dilated-conv stack spends most of its weights
and FLOPs building a long linear receptive field out of many short kernels. Here a
single learned per-channel filter spans the whole window for ~``K`` params and one
FFT pair, and the harmonic-generating capacity is concentrated in the (cheap)
pointwise mixed activations between filters. Stacking a few such blocks composes
linear-shape -> distort -> linear-shape, the same compositional structure that lets
a Wiener/Hammerstein cascade approximate a real analog distortion stage.

Run: uv run python experiments/spectral_zoo/global_filter.py
"""

from __future__ import annotations

import torch
from torch import nn

from vguitar.models.tcn import _MixedActivation

APPROACH = (
    "Alternate time-domain mixed-activation nonlinearities with a GLOBAL learned "
    "per-channel diagonal complex multiply (rFFT -> H.*X -> irFFT) = long linear "
    "conv done as a spectral multiply; residual blocks."
)
COST = (
    "~30-70k params at channels=24 (filters dominate: n_blocks*channels*kernel_len). "
    "Per-sample work is O(log T) amortized (a few length-T rFFT/irFFT pairs across "
    "the whole window) plus O(channels) pointwise; far below a dilated TCN's "
    "O(receptive_field) per sample. Streaming/causal: the spectral multiply is a "
    "circular (non-causal) global conv, so exact realtime needs causal "
    "block-overlap-add of a one-sided (causal) kernel -> a fixed lookahead/latency "
    "of ~kernel_len samples per block; feasible but NOT zero-latency like the TCN."
)


class _GlobalSpectralFilter(nn.Module):
    """Per-channel long linear filter applied as a frequency-domain diagonal multiply.

    The filter is stored as a short *time-domain* kernel ``h`` of length ``kernel``
    (one per channel). On each forward we rFFT the (zero-padded) signal and the
    kernel to a common length, multiply ``X .* H`` per channel, and irFFT back —
    i.e. a genuine linear convolution realised as a spectral multiply. Storing ``h``
    in time (not a fixed-length spectrum) keeps the parameter count tiny and, more
    importantly, makes the op length-agnostic: the same learned kernel works for any
    input ``T`` (the FFT length adapts), so the self-test's 2048/4096/1000 all run.

    Linear convolution (not circular) is obtained by padding to ``T + kernel - 1``
    and cropping the first ``T`` outputs, which also makes the kernel one-sided
    (causal): output sample ``t`` depends only on inputs ``<= t``. That is what
    makes a streaming overlap-add build possible later (see COST).
    """

    def __init__(self, channels: int, kernel: int = 256) -> None:
        super().__init__()
        self.kernel = kernel
        # Small random kernel; first tap biased to ~1 so the block starts near a
        # pass-through (identity-ish) filter and the residual path stays well-behaved.
        h = 0.02 * torch.randn(channels, kernel)
        h[:, 0] += 1.0
        self.h = nn.Parameter(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        t = x.shape[-1]
        n = t + self.kernel - 1  # linear (non-circular) conv length
        x_f = torch.fft.rfft(x, n=n, dim=-1)  # (B, C, F)
        h_f = torch.fft.rfft(self.h, n=n, dim=-1)  # (C, F)
        y = torch.fft.irfft(x_f * h_f.unsqueeze(0), n=n, dim=-1)
        return y[..., :t]  # causal crop back to length T


class _SpectralBlock(nn.Module):
    """One block: global spectral filter -> mixed pointwise nonlinearity -> 1x1 mix.

    The spectral filter does cheap long-range LINEAR mixing; the heterogeneous
    pointwise activation (tanh/gelu/relu/abs/snake) generates the HARMONICS; the
    ``1x1`` conv recombines the activation groups; everything is wrapped in a
    residual so blocks compose without vanishing gradients.
    """

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.filt = _GlobalSpectralFilter(channels, kernel)
        self.bias = nn.Parameter(torch.zeros(channels, 1))
        self.act = _MixedActivation(channels)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        g = self.act(self.filt(x) + self.bias)
        return x + self.mix(g)


class Net(nn.Module):
    """Stacked global-spectral-filter + mixed-nonlinearity residual blocks.

    ``channels`` lifts the 1-D signal into a feature space; ``n_blocks`` blocks each
    apply a cheap global linear filter (spectral diagonal) then a harmonic-generating
    pointwise nonlinearity; a ``1x1`` head collapses back to one channel.
    """

    def __init__(
        self, channels: int = 24, n_blocks: int = 4, kernel: int = 256
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            _SpectralBlock(channels, kernel) for _ in range(n_blocks)
        )
        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, 1, 1),
        )
        # REQUIRED: the trainer overwrites this with the dataset's output bound.
        self.register_buffer("out_bound", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T) already input-scaled. c_sys is always None (single-circuit).
        del c_sys
        h = self.input(x.unsqueeze(1))  # (B, C, T)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h).squeeze(1)  # (B, T)


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
