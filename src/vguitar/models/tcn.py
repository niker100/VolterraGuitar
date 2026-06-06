"""Feedforward WaveNet-style gated dilated TCN — the primary realtime candidate.

A causal temporal convolutional network closely following the black-box amplifier
emulators of Damskaegg, Juvela & Valimaeki ("Deep Learning for Tube Amplifier
Emulation", ICASSP 2019) and the GuitarML *PedalNetRT* implementation.

Architecture (per :class:`_TCNNet`):

* ``n_blocks`` repeats of ``n_layers`` dilated layers; dilation doubles within a
  block (``1, 2, 4, ..., 2**(n_layers-1)``) and resets at each block start. This
  geometric dilation grows the receptive field exponentially with depth.
* Each layer: a *causal* dilated conv producing ``2*channels`` features split into
  a gated activation ``tanh(a) * sigmoid(b)`` (the WaveNet gate), then two ``1x1``
  convs — one feeding a residual sum back into the layer, one feeding a skip path.
* All skip outputs are summed -> ReLU -> ``1x1`` -> ``1x1`` -> single output channel.

Receptive field::

    R = n_blocks * (2**n_layers - 1) * (kernel - 1) + 1

(each block contributes ``(2**n_layers - 1)*(kernel-1)`` from its summed dilations).
With the defaults (n_blocks=2, n_layers=8, kernel=3) that is ``2*255*2 + 1 = 1021``
samples (~23 ms at 44.1 kHz).

The whole network is purely causal, so blockwise streaming reproduces the offline
pass exactly and ``latency_samples == 0``. Streaming keeps a ring buffer holding
the last ``R-1`` input samples so each block's convolution sees its full history;
this is why the TCN — fixed cost per sample, no recurrence — is our main realtime
candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import nn

from vguitar.config import TrainConfig
from vguitar.losses import esr_loss, multi_stft_loss
from vguitar.models.base import FitReport, Model, register_model

if TYPE_CHECKING:
    from vguitar.data import Dataset


class _GatedLayer(nn.Module):
    """One causal dilated WaveNet layer: gated conv -> residual + skip.

    The dilated conv emits ``2*channels`` channels; the first half drives a
    ``tanh`` (filter) and the second a ``sigmoid`` (gate), per WaveNet (van den
    Oord et al., 2016). Causality is enforced by left-padding the input by
    ``dilation*(kernel-1)`` and never padding on the right.
    """

    def __init__(self, channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(channels, 2 * channels, kernel, dilation=dilation)
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(nn.functional.pad(x, (self.pad, 0)))
        a, b = h.chunk(2, dim=1)
        g = torch.tanh(a) * torch.sigmoid(b)
        return x + self.res(g), self.skip(g)


class _TCNNet(nn.Module):
    """The dilated TCN stack wrapped by :class:`TCN` (see module docstring)."""

    def __init__(self, channels: int, n_blocks: int, n_layers: int, kernel: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        dilations = [2**i for i in range(n_layers)]
        self.layers = nn.ModuleList(
            _GatedLayer(channels, kernel, d) for _ in range(n_blocks) for d in dilations
        )
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 1, T)`` -> ``(B, 1, T)``."""
        h = self.input(x)
        skips = h.new_zeros(h.shape)
        for layer in self.layers:
            h, s = layer(h)
            skips = skips + s
        return self.out(skips)


@register_model
class TCN(Model):
    """Gated dilated temporal convolutional network (ICASSP-19); realtime primary."""

    name: ClassVar[str] = "tcn"
    description: ClassVar[str] = "Feedforward WaveNet-style gated dilated causal TCN."
    latency_samples: int = 0  # fully causal

    def __init__(
        self,
        channels: int = 16,
        n_blocks: int = 2,
        n_layers: int = 8,
        kernel: int = 3,
        device: str = "cpu",
    ) -> None:
        self.channels = channels
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.kernel = kernel
        self.device = torch.device(device)
        # Receptive field: each block sums dilations 1..2^(n_layers-1).
        self.receptive_field = n_blocks * (2**n_layers - 1) * (kernel - 1) + 1
        self.net = _TCNNet(channels, n_blocks, n_layers, kernel).to(self.device)
        self._buf: np.ndarray | None = None  # streaming ring of the last R-1 inputs

    # --- helpers ----------------------------------------------------------
    def _hparams(self) -> dict[str, int]:
        return {
            "channels": self.channels,
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "kernel": self.kernel,
        }

    def _make_windows(
        self, ds: Dataset, seq_len: int, hop: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tile ``ds`` into overlapping ``(N, seq_len)`` (x, y) window batches."""
        x = torch.from_numpy(np.ascontiguousarray(ds.x, dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(ds.y, dtype=np.float32))
        n = len(ds) - seq_len + 1
        if n <= 0:  # signal shorter than a window -> single zero-padded window
            pad = seq_len - len(ds)
            x = nn.functional.pad(x, (0, pad))
            y = nn.functional.pad(y, (0, pad))
            return x.unsqueeze(0), y.unsqueeze(0)
        starts = torch.arange(0, n, hop)
        idx = starts[:, None] + torch.arange(seq_len)[None, :]
        return x[idx], y[idx]

    # --- learning ---------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Train on truncated windows; the first ``cfg.warmup`` samples are not scored.

        Loss is ESR (DAFx-19) plus a small multi-resolution STFT term to sharpen
        harmonic content; optimized with Adam. The best validation-ESR weights are
        retained and restored at the end.
        """
        cfg = cfg or TrainConfig()
        torch.manual_seed(cfg.seed)
        dev = self.device
        warmup = min(cfg.warmup, cfg.seq_len - 1)

        xb, yb = self._make_windows(train, cfg.seq_len, hop=cfg.seq_len - warmup)
        xb, yb = xb.to(dev), yb.to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(cfg.seed)

        history: dict[str, list[float]] = {"train_loss": [], "esr": [], "val_esr": []}
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

        for _ in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(xb.shape[0], generator=gen)
            ep_loss = ep_esr = 0.0
            n_batches = 0
            for i in range(0, len(perm), cfg.batch_size):
                sel = perm[i : i + cfg.batch_size]
                xin = xb[sel].unsqueeze(1)  # (B, 1, T)
                pred = self.net(xin).squeeze(1)[:, warmup:]
                tgt = yb[sel][:, warmup:]
                loss = esr_loss(pred, tgt) + 0.1 * multi_stft_loss(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                ep_esr += esr_loss(pred.detach(), tgt).item()
                n_batches += 1
            history["train_loss"].append(ep_loss / max(n_batches, 1))
            history["esr"].append(ep_esr / max(n_batches, 1))

            ve = self._eval_esr(val, cfg) if val is not None else history["esr"][-1]
            history["val_esr"].append(ve)
            if ve < best_val:
                best_val = ve
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

        self.net.load_state_dict(best_state)
        return FitReport(
            history=history,
            info={"best_val_esr": best_val, "receptive_field": self.receptive_field},
        )

    def _eval_esr(self, ds: Dataset, cfg: TrainConfig) -> float:
        """Whole-signal ESR on ``ds`` (warm-up samples excluded)."""
        pred = torch.from_numpy(self.process(ds.x))
        tgt = torch.from_numpy(np.ascontiguousarray(ds.y, dtype=np.float32))
        w = min(cfg.warmup, len(ds) - 1)
        return float(esr_loss(pred[w:], tgt[w:]))

    # --- offline inference ------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Run the whole signal causally, with input-domain zero history.

        We left-pad the input with ``receptive_field - 1`` zeros and drop the
        corresponding warm-up outputs. This matters: the conv layers have biases,
        so per-layer zero *padding* is NOT the same as a zero *input* history
        (``conv(0) = bias != 0``). Streaming uses a zero-initialised input ring
        buffer, so feeding the same input-domain zero history here makes
        ``process`` and blockwise ``process_block`` agree bit-for-bit.
        """
        self.net.eval()
        hist = self.receptive_field - 1
        xv = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        xin = np.concatenate([np.zeros(hist, dtype=np.float32), xv]) if hist else xv
        with torch.no_grad():
            t = torch.from_numpy(xin).to(self.device).view(1, 1, -1)
            y = self.net(t).view(-1)[hist:]
        return y.cpu().numpy().astype(np.float32)

    # --- streaming inference ----------------------------------------------
    def reset(self) -> None:
        """Clear the causal history ring buffer."""
        self._buf = None

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process one block; prepend ``R-1`` past samples so output == offline."""
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        hist = self.receptive_field - 1
        if self._buf is None:
            self._buf = np.zeros(hist, dtype=np.float32)
        ctx = np.concatenate([self._buf, x])  # (hist + block,)
        self.net.eval()
        with torch.no_grad():
            xin = torch.from_numpy(ctx).to(self.device).view(1, 1, -1)
            y = self.net(xin).view(-1)[hist:]  # drop the warm-up context
        self._buf = ctx[-hist:].copy() if hist else self._buf
        return y.cpu().numpy().astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Pickle ``state_dict`` + hyperparameters via ``torch.save``."""
        torch.save({"hparams": self._hparams(), "state_dict": self.net.state_dict()}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> TCN:
        """Reconstruct an untrained TCN from hyperparameters, then restore weights."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**blob["hparams"])
        model.net.load_state_dict(blob["state_dict"])
        return model

    # --- introspection ----------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
