"""A small diagonal state-space sequence model (S4D / S5-lite).

State-space models (SSMs) emulate a circuit by learning a bank of linear
recurrences interleaved with pointwise nonlinearities. We use the *diagonal*
S4D parametrization (Gu, Gupta, Goel & Re, "On the Parameterization and
Initialization of Diagonal State Space Models", NeurIPS 2022), which keeps each
mode's recurrence to a single complex multiply per step yet matches the full
S4 kernel. Comunita, Steinmetz & Reiss ("Differentiable Black-Box and Grey-Box
Modeling of Nonlinear Audio Effects", 2025) found such SSMs the best overall
black-box analog-effect emulator -- but warn that *large* S4 stacks miss
real-time (RTF > 1), so this implementation is deliberately tiny.

Architecture (per layer)::

    u -> [diagonal SSM along time] + D*u -> GELU -> Linear(d_model->d_model)
                                                        |
                                  residual + LayerNorm <+

with a ``1 -> d_model`` input projection and a ``d_model -> 1`` output
projection. The GELU between the linear recurrences is what makes the whole
stack nonlinear (a pure SSM is linear and could only model a filter).

**Stability.** Each diagonal mode has a continuous pole ``-softplus(a_re) +
1j*a_im``; ``softplus`` keeps the real part strictly negative, so the
discrete pole ``A = exp(dt * pole)`` has magnitude ``< 1`` for any learnable
``log_dt``. The recurrence therefore cannot blow up. The real input is driven
through complex ``B`` and read out as ``2*Re(C @ h)`` (the standard real-SSM
convention: conjugate modes are folded into the factor of two).

**Streaming.** The recurrence is a first-order scan, so the exact same code
runs offline (full sequence, zero initial state) and block-by-block (carrying
the complex per-layer state across :meth:`process_block` calls). With zero
initial state the two agree to floating-point tolerance, so
``latency_samples == 0`` and :func:`check_streaming` passes (expect ~1e-3 max
abs error: the scan accumulates float32 rounding differently when chunked).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import nn

from vguitar.config import TrainConfig
from vguitar.losses import combined_loss
from vguitar.models.base import FitReport, Model, register_model

if TYPE_CHECKING:
    from vguitar.data import Dataset


class _S4DLayer(nn.Module):
    """One diagonal-SSM block: recurrence + GELU + linear mix + residual/norm.

    The diagonal state has shape ``(d_model, d_state)``: every model channel
    owns ``d_state`` independent complex modes. Parameters follow S4D-Lin
    (Gu et al. 2022): ``log_dt`` (timestep), ``a_re``/``a_im`` (pole), complex
    ``B``/``C`` (per-channel input/output gains), real ``D`` (skip).
    """

    _MAX_CHUNK: int = 2048  # FFT-scan chunk length (bounds memory; state carried)

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # log_dt spread over [1e-3, 1e-1] s (S4D init); broad timescale coverage.
        log_dt = torch.rand(d_model) * (np.log(1e-1) - np.log(1e-3)) + np.log(1e-3)
        self.log_dt = nn.Parameter(log_dt)
        # Pole real part via softplus(a_re); init a_re so softplus ~= 0.5 (HiPPO-ish).
        self.a_re = nn.Parameter(torch.full((d_model, d_state), float(np.log(np.e**0.5 - 1))))
        # Imaginary parts pi*n: distinct oscillation frequencies per mode (S4D-Lin).
        a_im = torch.pi * torch.arange(d_state).float().expand(d_model, d_state).clone()
        self.a_im = nn.Parameter(a_im)
        # Complex B, C stored as real/imag pairs (torch optimizers want real params).
        self.b_re = nn.Parameter(torch.ones(d_model, d_state))
        self.b_im = nn.Parameter(torch.zeros(d_model, d_state))
        c = torch.randn(d_model, d_state, 2) / np.sqrt(d_state)
        self.c_re = nn.Parameter(c[..., 0].contiguous())
        self.c_im = nn.Parameter(c[..., 1].contiguous())
        self.d = nn.Parameter(torch.ones(d_model))  # real skip connection
        self.mix = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _discrete(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return discrete pole ``A`` and complex ``B``/``C`` (all ``(d_model, d_state)``).

        Zero-order-hold-free S4D uses ``A_bar = exp(dt * pole)`` and folds the
        usual ``(A_bar - 1)/A * B`` discretization gain into the learned B, so
        a plain ``exp`` is both correct and cheaper.
        """
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (d_model, 1)
        pole = -nn.functional.softplus(self.a_re) + 1j * self.a_im
        a_bar = torch.exp(dt * pole)  # |a_bar| < 1 by construction -> stable
        b = torch.complex(self.b_re, self.b_im)
        c = torch.complex(self.c_re, self.c_im)
        return a_bar, b, c

    def _ssm(self, u: torch.Tensor, h0: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the diagonal recurrence over time via a chunked FFT convolution.

        The recurrence ``h_t = A h_{t-1} + B u_t``, ``y_t = 2 Re(C h_t) + D u_t``
        is computed in parallel over time using the closed form
        ``h_t = A^{t+1} h_0 + sum_{s<=t} A^{t-s} B u_s`` -- a causal convolution of
        the per-mode drive with the geometric kernel ``A^k`` (done by FFT). Long
        sequences are split into ``_MAX_CHUNK`` chunks with state carried across
        them, so this is far faster than a Python per-sample scan yet still
        *exactly* the recurrence: offline ``process`` and block-streaming agree.

        Args:
            u: input, shape ``(B, T, d_model)``.
            h0: optional initial complex state ``(B, d_model, d_state)``; ``None``
                means zeros (offline / :meth:`process`).

        Returns:
            ``(y, h_last)``: real ``(B, T, d_model)`` output and the complex state
            after the last step.
        """
        bsz, t_len, _ = u.shape
        a_bar, b, c = self._discrete()  # (d_model, d_state), complex
        h = (
            torch.zeros(bsz, self.d_model, self.d_state, dtype=a_bar.dtype, device=u.device)
            if h0 is None
            else h0
        )
        if t_len == 0:
            return u.new_zeros(bsz, 0, self.d_model), h
        outs: list[torch.Tensor] = []
        for start in range(0, t_len, self._MAX_CHUNK):
            yc, h = self._ssm_chunk(u[:, start : start + self._MAX_CHUNK], h, a_bar, b, c)
            outs.append(yc)
        return torch.cat(outs, dim=1) + u * self.d, h  # real skip D*u

    def _ssm_chunk(
        self,
        u: torch.Tensor,
        h0: torch.Tensor,
        a_bar: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One chunk of the diagonal recurrence by FFT; returns ``(y, h_last)``."""
        lc = u.shape[1]
        k = torch.arange(lc, device=u.device, dtype=torch.float32)
        a_pows = a_bar.unsqueeze(-1) ** k  # (d_model, d_state, lc) = A^k
        v = (u.unsqueeze(-1).to(b.dtype) * b).permute(0, 2, 3, 1)  # (B, d_model, d_state, lc)
        nfft = 1 << ((2 * lc - 1).bit_length())  # power of two >= 2*lc (linear conv)
        conv = torch.fft.ifft(
            torch.fft.fft(v, n=nfft, dim=-1) * torch.fft.fft(a_pows, n=nfft, dim=-1), dim=-1
        )[..., :lc]  # sum_{s<=t} A^{t-s} B u_s
        state = h0.unsqueeze(-1) * (a_pows * a_bar.unsqueeze(-1)).unsqueeze(0)  # A^{t+1} h0
        h_all = conv + state  # (B, d_model, d_state, lc)
        y = 2.0 * (c.unsqueeze(0).unsqueeze(-1) * h_all).real.sum(2)  # (B, d_model, lc)
        return y.permute(0, 2, 1), h_all[..., -1]

    def forward(
        self, x: torch.Tensor, h0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the full block; returns ``(y, h_last)`` with ``y`` shape ``(B, T, d_model)``."""
        y, h = self._ssm(x, h0)
        y = nn.functional.gelu(y)
        y = self.mix(y)
        return self.norm(x + y), h


class _SSMNet(nn.Module):
    """Input projection -> stacked :class:`_S4DLayer` -> output projection."""

    def __init__(self, d_model: int, d_state: int, n_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(_S4DLayer(d_model, d_state) for _ in range(n_layers))
        self.out_proj = nn.Linear(d_model, 1)
        # Zero-init the readout so the untrained net outputs ~0 (ESR starts ~1,
        # not amplifying). Standard for residual stacks; training grows it.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, states: list[torch.Tensor | None] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Map ``(B, T)`` audio to ``(B, T)`` audio, threading per-layer states.

        ``states[i]`` is the complex state entering layer ``i`` (``None`` for
        zeros). Returns the output and the list of updated states for streaming.
        """
        h = self.in_proj(x.unsqueeze(-1))  # (B, T, d_model)
        new_states: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            h, s = layer(h, None if states is None else states[i])
            new_states.append(s)
        return self.out_proj(h).squeeze(-1), new_states


@register_model
class SSM(Model):
    """Small diagonal state-space sequence model (S4D/S5-lite) for circuit emulation."""

    name: ClassVar[str] = "ssm"
    description: ClassVar[str] = "Diagonal state-space model (S4D-lite) with GELU nonlinearities"
    latency_samples: int = 0  # causal, zero-init -> offline and streaming align

    def __init__(
        self, d_model: int = 16, d_state: int = 16, n_layers: int = 3, device: str = "cpu"
    ) -> None:
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.device = torch.device(device)
        self.net = _SSMNet(d_model, d_state, n_layers).to(self.device)
        self._states: list[torch.Tensor | None] = [None] * n_layers  # streaming state

    # --- learning ---------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Train with truncated-BPTT windows, ESR(+STFT) loss, Adam, keep best-val.

        Matches the other neural emulators: random ``seq_len`` windows per batch,
        a ``warmup`` prefix excluded from the loss so the recurrence settles, and
        ESR plus a light multi-resolution STFT term (see :mod:`vguitar.losses`).
        """
        cfg = cfg or TrainConfig()
        gen = torch.Generator().manual_seed(cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        xt = torch.from_numpy(np.asarray(train.x, dtype=np.float32)).to(self.device)
        yt = torch.from_numpy(np.asarray(train.y, dtype=np.float32)).to(self.device)
        val_xy = None
        if val is not None and len(val) > cfg.warmup:
            val_xy = (
                torch.from_numpy(np.asarray(val.x, dtype=np.float32)).to(self.device),
                torch.from_numpy(np.asarray(val.y, dtype=np.float32)).to(self.device),
            )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        seq_len = min(cfg.seq_len, len(train))
        warmup = min(cfg.warmup, seq_len - 1)
        # Several gradient steps per epoch (random windows). Capped so training
        # cost stays bounded on large datasets: without the cap, steps scale with
        # dataset length and a 10 s clip would need thousands of (batched) steps
        # per epoch -- hours. Each step already sees `batch_size` windows.
        steps_per_epoch = max(1, min((len(train) - seq_len) // seq_len, 40))

        for _ in range(cfg.epochs):
            self.net.train()
            ep_loss = 0.0
            for _ in range(steps_per_epoch):
                xb, yb = self._sample_batch(xt, yt, seq_len, cfg.batch_size, gen)
                opt.zero_grad()
                pred, _ = self.net(xb)
                loss = combined_loss(pred[:, warmup:], yb[:, warmup:], stft_weight=0.1, dc_weight=0.1)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)  # tame scan gradients
                opt.step()
                ep_loss += float(loss.detach())
            history["train_loss"].append(ep_loss / steps_per_epoch)

            vloss = self._val_loss(val_xy, warmup)
            history["val_loss"].append(vloss)
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

        self.net.load_state_dict(best_state)  # restore best-val weights
        self.reset()
        return FitReport(
            history=history,
            info={"best_val_loss": best_val, "num_params": self.num_params()},
        )

    def _sample_batch(
        self,
        xt: torch.Tensor,
        yt: torch.Tensor,
        seq_len: int,
        batch_size: int,
        gen: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``batch_size`` random contiguous ``seq_len`` windows as ``(B, T)`` tensors."""
        max_start = len(xt) - seq_len
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=gen)
        idx = starts.unsqueeze(1) + torch.arange(seq_len).unsqueeze(0)  # (B, seq_len)
        return xt[idx], yt[idx]

    @torch.no_grad()
    def _val_loss(self, val_xy: tuple[torch.Tensor, torch.Tensor] | None, warmup: int) -> float:
        """ESR(+STFT) on the full validation signal; ``inf`` if no val set given."""
        if val_xy is None:
            return float("inf")
        self.net.eval()
        pred, _ = self.net(val_xy[0].unsqueeze(0))
        loss = combined_loss(
            pred[:, warmup:], val_xy[1].unsqueeze(0)[:, warmup:], stft_weight=0.1, dc_weight=0.1
        )
        return float(loss)

    # --- offline inference ------------------------------------------------
    @torch.no_grad()
    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a whole signal with zero initial state (matches streaming)."""
        self.net.eval()
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(-1)).to(self.device)
        y, _ = self.net(xt.unsqueeze(0))  # zero init state -> block-vs-offline agree
        return y.squeeze(0).cpu().numpy().astype(np.float32)

    # --- streaming inference (realtime) -----------------------------------
    def reset(self) -> None:
        """Zero every layer's complex state so the next block starts fresh."""
        self._states = [None] * self.n_layers

    @torch.no_grad()
    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process one block, carrying the per-layer complex state forward."""
        self.net.eval()
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(-1)).to(self.device)
        if xt.numel() == 0:
            return np.empty(0, dtype=np.float32)
        y, self._states = self.net(xt.unsqueeze(0), self._states)
        return y.squeeze(0).cpu().numpy().astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save hyperparameters + weights so :meth:`load` reconstructs exactly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hparams": {
                    "d_model": self.d_model,
                    "d_state": self.d_state,
                    "n_layers": self.n_layers,
                },
                "state_dict": self.net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SSM:
        """Reconstruct an :class:`SSM` from a file written by :meth:`save`."""
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = cls(**ckpt["hparams"], device="cpu")
        model.net.load_state_dict(ckpt["state_dict"])
        model.reset()
        return model

    # --- introspection ----------------------------------------------------
    def num_params(self) -> int:
        """Number of learnable scalar parameters across the whole network."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
