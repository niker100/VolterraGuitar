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
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch import nn

from vguitar.config import TrainConfig
from vguitar.models.archive.losses import esr_loss, multi_stft_loss
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
        self.channels = channels
        self.pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(channels, 2 * channels, kernel, dilation=dilation)
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(nn.functional.pad(x, (self.pad, 0)))
        a, b = h.chunk(2, dim=1)
        g = torch.tanh(a) * torch.sigmoid(b)
        return x + self.res(g), self.skip(g)


class _MixedActivation(nn.Module):
    """Heterogeneous per-channel activation: 5 groups, each a DIFFERENT function.

    The channels are split into five (near-)equal groups carrying ``tanh`` (smooth
    saturation), ``gelu`` (smooth gate), ``relu`` (one-sided corner), ``abs``
    (V-shaped corner — a natural fit for a symmetric dead-zone), and Snake
    ``x + sin(alpha*x)^2 / alpha`` (periodic / harmonic folding; per-channel
    learnable ``alpha`` clamped >= 0.1 so the ``1/alpha`` scale never explodes).

    A smooth (Lipschitz) ``tanh``/``sigmoid`` gate cannot represent a *corner*
    (a slope discontinuity) without enormous capacity; giving every layer
    ``abs``/``relu``/Snake units lets the net synthesize the kink natively while
    the smooth units round the conducting region. The layer's downstream ``1x1``
    convs recombine the heterogeneous groups, so the net can also *suppress* the
    corner units where they are not needed (smooth circuits) — which is why this
    does not regress on smooth transfer curves the way an input-level rectified
    feature basis does.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        base = channels // 5
        # tanh, gelu, relu, abs, snake (snake gets the remainder)
        self.sizes = (base, base, base, base, channels - 4 * base)
        self.alpha = nn.Parameter(torch.ones(self.sizes[4]))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        s = self.sizes
        i0, i1, i2, i3 = s[0], s[0] + s[1], s[0] + s[1] + s[2], s[0] + s[1] + s[2] + s[3]
        sn = h[:, i3:]
        al = self.alpha.clamp(min=0.1).view(1, -1, 1)
        return torch.cat(
            [
                torch.tanh(h[:, :i0]),
                nn.functional.gelu(h[:, i0:i1]),
                torch.relu(h[:, i1:i2]),
                torch.abs(h[:, i2:i3]),
                sn + torch.sin(al * sn) ** 2 / al,
            ],
            dim=1,
        )


class _MixedLayer(nn.Module):
    """A WaveNet residual+skip layer whose gate is :class:`_MixedActivation`.

    Drop-in for :class:`_GatedLayer` (same constructor signature, same ``forward``
    contract returning ``(residual, skip)``) but the dilated conv emits ``channels``
    (not ``2*channels``) features, which the heterogeneous activation transforms
    before the ``1x1`` residual/skip projections. Used by CIRCE3 when
    ``block_act='mixed'`` to add corner-capacity for sharp-discontinuity circuits.
    """

    def __init__(self, channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.channels = channels
        self.pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.act = _MixedActivation(channels)
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.act(self.conv(nn.functional.pad(x, (self.pad, 0))))
        return x + self.res(g), self.skip(g)


def _mixed_sizes(channels: int) -> tuple[int, int, int, int, int]:
    """Group sizes for :class:`_MixedActivation` (tanh, gelu, relu, abs, snake)."""
    base = channels // 5
    return (base, base, base, base, channels - 4 * base)


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
        # Cached streaming state (weights baked to numpy + per-layer ring buffers),
        # built lazily on the first process_block after a reset. See process_block.
        self._stream: dict[str, Any] | None = None

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
        self.reset()  # invalidate any cached streaming weights (params changed)
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
        """Offline forward over the whole signal (per-layer causal zero-padding).

        Matches blockwise ``process_block`` to ~1e-6: that path caches each
        dilated layer's activation history initialised to zero, which is exactly
        the per-layer zero-padding ``self.net`` applies here.
        """
        self.net.eval()
        xv = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            y = self.net(torch.from_numpy(xv).to(self.device).view(1, 1, -1)).view(-1)
        return y.cpu().numpy().astype(np.float32)

    # --- streaming inference (cached incremental dilated conv) -------------
    def reset(self) -> None:
        """Drop streaming state; rebuilt (weights baked to numpy) on the next block."""
        self._stream = None

    def _build_stream(self) -> dict[str, Any]:
        """Bake net weights to numpy arrays + allocate per-layer ring buffers.

        Weights are read from ``state_dict()`` by dotted key (a plain
        ``dict[str, Tensor]``) rather than by submodule attribute access — same
        values, but clean to type-check.
        """
        self.net.eval()
        c, n_lyr = self.channels, self.n_blocks * self.n_layers
        sd = {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        dil = [2 ** (i % self.n_layers) for i in range(n_lyr)]
        layers = [
            (
                sd[f"layers.{i}.conv.weight"],  # (2c, c, k)
                sd[f"layers.{i}.conv.bias"],  # (2c,)
                sd[f"layers.{i}.res.weight"][:, :, 0],  # (c, c)
                sd[f"layers.{i}.res.bias"],
                sd[f"layers.{i}.skip.weight"][:, :, 0],
                sd[f"layers.{i}.skip.bias"],
            )
            for i in range(n_lyr)
        ]
        return {
            "c": c,
            "k": self.kernel,
            "dil": dil,
            "w_in": sd["input.weight"][:, 0, 0],
            "b_in": sd["input.bias"],
            "layers": layers,
            "o1w": sd["out.1.weight"][:, :, 0],
            "o1b": sd["out.1.bias"],
            "o3w": sd["out.3.weight"][:, :, 0],
            "o3b": sd["out.3.bias"],
            "buf": [np.zeros((c, (self.kernel - 1) * d), dtype=np.float32) for d in dil],
        }

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Stream one block via cached incremental dilated convs ("Fast-WaveNet").

        Each dilated layer keeps a ring buffer of its last ``(kernel-1)*dilation``
        inputs, so a block costs O(block), not O(receptive_field) — the key to
        real-time (~3x at block 128 in pure numpy). The 1x1 input/residual/skip/
        output convs are pointwise (no memory). Equals :meth:`process` to ~1e-6;
        ``latency_samples == 0``.
        """
        if self._stream is None:
            self._stream = self._build_stream()
        s = self._stream
        xb = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        nb = xb.shape[0]
        if nb == 0:
            return np.empty(0, dtype=np.float32)
        c, k, dil = s["c"], s["k"], s["dil"]
        h = s["w_in"][:, None] * xb[None, :] + s["b_in"][:, None]  # (c, nb), 1x1 input conv
        skip = np.zeros((c, nb), dtype=np.float32)
        for i, (cw, cb, rw, rb, sw, sb) in enumerate(s["layers"]):
            d = dil[i]
            ctx = np.concatenate([s["buf"][i], h], axis=1)  # (c, (k-1)d + nb)
            conv = cb[:, None] + sum(cw[:, :, t] @ ctx[:, t * d : t * d + nb] for t in range(k))
            if k > 1:
                s["buf"][i] = ctx[:, -(k - 1) * d :].copy()
            # gated activation; clip the sigmoid arg to avoid exp overflow on
            # out-of-range inputs (no effect in-range — sigmoid is saturated there).
            g = np.tanh(conv[:c]) * (1.0 / (1.0 + np.exp(-np.clip(conv[c:], -30.0, 30.0))))
            h = h + (rw @ g + rb[:, None])  # residual
            skip += sw @ g + sb[:, None]
        o = np.maximum(skip, 0.0)
        o = np.maximum(s["o1w"] @ o + s["o1b"][:, None], 0.0)
        o = s["o3w"] @ o + s["o3b"][:, None]
        return o[0].astype(np.float32)

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
