"""Learned analysis/synthesis filterbank — a trainable, possibly non-orthogonal
transform that replaces the fixed FFT for cheap long-range linear mixing.

Idea
----
The "spectral-only linear gain" baseline fails because a linear filter (any
basis, FFT included) cannot create harmonics. So we keep the structure that
makes spectral methods cheap — *transform -> pointwise op -> inverse transform* —
but (a) make the transform itself **learnable** (a strided 1D conv analysis
filterbank, its transposed-conv mirror for synthesis) and (b) put genuine
**time-domain pointwise nonlinearities** inside the transform domain so the net
actually generates the circuit's distortion harmonics.

Why a learned transform could beat the FFT here
-----------------------------------------------
* A guitar-distortion circuit's memory (coupling caps, tone stack) lives in a
  handful of decaying modes, not the full DFT basis. A learned filterbank can
  put its ``2*channels`` analysis atoms exactly on those modes, so a few
  coefficients summarise a long history — the long convolution becomes a cheap
  strided multiply. The FFT spends its whole basis uniformly regardless.
* The transform need not be orthogonal: analysis and synthesis are independent
  learnable conv kernels, so the net can pick an over-complete, well-conditioned
  pair tuned for *reconstruction after a nonlinearity*, which an orthonormal FFT
  cannot.
* Harmonic generation happens **in the (downsampled) coefficient stream**: each
  coefficient frame already pools a wide receptive field (one stride hop), so a
  pointwise nonlinearity on it mixes that whole context — exactly what a deep
  dilated TCN achieves with many layers, but in a single analysis hop.

Pipeline (per :class:`Net`)
---------------------------
1. **Analysis**: ``Conv1d(1 -> A, kernel=K, stride=H)`` with reflect/zero padding
   -> coefficient tensor ``(B, A, L)`` at hop rate ``H`` (the learned transform).
2. **Coefficient nonlinearity + mixing**: a small residual stack operating on the
   ``(B, A, L)`` coefficients — ``1x1`` linear mixing across atoms (cheap, this is
   the "spectral mixing") interleaved with :class:`_MixedActivation`
   (tanh/gelu/relu/abs/snake) so harmonics are genuinely created. A learnable
   per-atom complex-style gate (two 1x1 convs, multiplicative) adds the
   amplitude-dependent shaping a distortion circuit needs.
3. **Synthesis**: ``ConvTranspose1d(A -> 1, kernel=K, stride=H)`` (the learned
   inverse transform) -> reconstruct to exactly the (padded) input length, then
   crop to ``T``.
4. A tiny **time-domain residual nonlinearity** (depthwise causal conv + tanh) is
   added at the original rate so the model can place sharp, sample-accurate
   corners the downsampled path would smear.

Cost: see :data:`COST`. The expensive long convolution is replaced by strided
analysis at hop ``H`` (``L = T/H`` frames), so the dominant work is ``O(T*A*K/H)``
for the two filterbanks plus ``O(L*A^2)`` for coefficient mixing — far below the
TCN's ``O(T * depth * channels^2)`` once ``H`` is moderate.
"""

from __future__ import annotations

import torch
from torch import nn

from vguitar.models.archive.tcn import _MixedActivation

APPROACH = (
    "Learned strided-conv analysis filterbank -> per-coefficient mixed "
    "nonlinearity + 1x1 atom mixing + multiplicative gate -> transposed-conv "
    "synthesis; a trainable non-orthogonal transform replacing the FFT, with a "
    "small time-domain corrective nonlinearity."
)
COST = (
    "~35k params at channels=24 (A=2*channels=48 atoms, K=64 analysis/synthesis "
    "kernels, hop H=16, 3 coeff blocks). Per-sample FLOP order O(A*K/H + A^2/H) "
    "~ a few hundred MACs/sample — dominated by the two filterbanks evaluated at "
    "hop rate, well below a deep dilated TCN (O(depth*channels^2) per sample). "
    "Streaming/causal: feasible as overlap-save block processing (analysis and "
    "synthesis are FIR; carry K-1 history per block) with latency ~= one hop H + "
    "(K-H) overlap (~K samples ~1.5 ms @44.1k). The current module uses "
    "reflect/zero padding for the offline screen, so as written it is non-causal; "
    "a causal variant uses left-only padding and adds ~K-sample latency."
)


class _CoeffBlock(nn.Module):
    """One residual block on the transform coefficients ``(B, A, L)``.

    Cheap ``1x1`` cross-atom mixing (the "spectral mixing" that replaces a long
    time-domain convolution) wrapped around a heterogeneous :class:`_MixedActivation`
    so the block genuinely synthesises harmonics, plus a multiplicative gate
    ``v * sigmoid(g)`` giving the amplitude-dependent shaping a distortion stage
    needs. All ops are pointwise in ``L`` (no extra receptive field), so the only
    long-range pooling is the analysis stride itself.
    """

    def __init__(self, atoms: int) -> None:
        super().__init__()
        self.mix_in = nn.Conv1d(atoms, atoms, 1)
        self.act = _MixedActivation(atoms)
        self.gate_v = nn.Conv1d(atoms, atoms, 1)
        self.gate_g = nn.Conv1d(atoms, atoms, 1)
        self.mix_out = nn.Conv1d(atoms, atoms, 1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        h = self.act(self.mix_in(c))
        h = self.gate_v(h) * torch.sigmoid(self.gate_g(h))  # multiplicative shaping
        return c + self.mix_out(h)


class Net(nn.Module):
    """Learned analysis/synthesis filterbank with in-transform nonlinearity.

    ``channels`` controls capacity: the transform uses ``A = 2*channels`` learned
    atoms. Kernel ``K`` and hop ``H`` are fixed (long enough to capture circuit
    memory, strided for cheap downsampled processing).
    """

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        atoms = 2 * channels
        self.kernel = 64
        self.hop = 16
        self.atoms = atoms

        # Learned (non-orthogonal) forward transform: strided analysis filterbank.
        self.analysis = nn.Conv1d(1, atoms, self.kernel, stride=self.hop, bias=True)
        # Coefficient-domain harmonic generation + cheap linear atom mixing.
        self.coeff = nn.ModuleList(_CoeffBlock(atoms) for _ in range(3))
        # Learned inverse transform: strided synthesis filterbank.
        self.synthesis = nn.ConvTranspose1d(atoms, 1, self.kernel, stride=self.hop, bias=True)

        # Tiny time-domain corrective path: a short depthwise conv lifted to a few
        # channels with a genuine pointwise nonlinearity, so sharp corners that the
        # downsampled filterbank smears can be placed at full sample rate.
        tc = max(4, channels // 3)
        self.t_in = nn.Conv1d(1, tc, 9, padding=4)
        self.t_act = nn.GELU()
        self.t_out = nn.Conv1d(tc, 1, 1)
        self.t_scale = nn.Parameter(torch.zeros(1))  # start at 0: pure filterbank first

        # Final blend of filterbank reconstruction + time residual.
        self.out_gain = nn.Parameter(torch.ones(1))

        self.register_buffer("out_bound", torch.tensor(1.0))  # REQUIRED (trainer sets this)

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        del c_sys  # single-circuit: always None
        t = x.shape[-1]
        xc = x.unsqueeze(1)  # (B, 1, T)

        # Pad so the strided analysis covers the whole signal and the transposed
        # synthesis reconstructs at least T samples; crop back to T afterwards.
        # ConvTranspose of a length-L analysis output yields (L-1)*H + K samples.
        # Choose left/right reflect padding that makes the round-trip length >= T.
        pad_total = self.kernel  # one full kernel of context on each side region
        left = pad_total // 2
        right = pad_total - left
        # Also pad on the right so (T + pad) - K is divisible by H (clean stride).
        eff = t + left + right
        rem = (eff - self.kernel) % self.hop
        if rem != 0:
            right += self.hop - rem
        # reflect needs pad < dim and t > 1; fall back to zero-pad otherwise.
        if t > 1 and left < t and right < t:
            xp = nn.functional.pad(xc, (left, right), mode="reflect")
        else:
            xp = nn.functional.pad(xc, (left, right))

        # --- learned forward transform ---
        c = self.analysis(xp)  # (B, A, L)
        # --- coefficient-domain nonlinear mixing (harmonic generation) ---
        for blk in self.coeff:
            c = blk(c)
        # --- learned inverse transform ---
        rec = self.synthesis(c)  # (B, 1, ~eff)
        # crop the synthesis output back to the padded-input region, then to T
        rec = rec[:, :, left : left + t]
        if rec.shape[-1] < t:  # safety: pad if synthesis came up short
            rec = nn.functional.pad(rec, (0, t - rec.shape[-1]))

        # --- time-domain corrective nonlinearity at full rate ---
        tr = self.t_out(self.t_act(self.t_in(xc)))  # (B, 1, T)

        y = self.out_gain * rec + self.t_scale * tr
        return y.squeeze(1)  # (B, T)


if __name__ == "__main__":
    for T in (2048, 4096, 1000):  # must handle arbitrary lengths
        net = Net(channels=16)
        y = net(torch.randn(2, T))
        assert y.shape == (2, T) and torch.isfinite(y).all(), (T, tuple(y.shape))
    print("OK", sum(p.numel() for p in Net(channels=24).parameters()), "params")
