"""CIRCE — a conditioned, real-time circuit emulator (the project's SOTA model).

The architecture vetted in ``docs/CIRCE-design.md``, with the GATE-0 backbone
pivot baked in: a **small dilated TCN** (the gated-conv blocks of
:mod:`vguitar.models.tcn`) conditioned by **per-block FiLM** from a tiny
conditioner network, with a **fixed output saturator of last resort** so the
output stays bounded across the whole control + amplitude range.

* **Exogenous controls** ``c`` (continuous pots, discrete switches, slow drift)
  enter only through FiLM ``h <- gamma(c)*h + beta(c)`` applied before each gated
  layer. The conditioner never touches the conv kernels, so the cached-streaming
  kernel (built once) and carried state stay valid as the knobs move.
* **Real-time**: the streaming path is the Fast-WaveNet cached incremental
  convolution from :class:`vguitar.models.tcn.TCN`, plus a per-block FiLM eval
  (a couple of tiny matmuls) -- all pure numpy. ``latency_samples == 0``.
* **Interpolation**: gamma is centred at 1 and the conditioner is zero-init
  (untrained == identity), controls are normalized; FiLM's affine form
  interpolates smoothly between trained settings.
* **Stability**: gamma is bounded (``1 + tanh``) and the output passes through
  ``A*tanh(y/A)`` (``A`` ~ 1.2x the trained output range), so it saturates
  rather than diverging outside the trained range.

``process``/``process_block`` take an optional control vector ``c``; with
``c=None`` they use the (normalized) zero control, so :func:`check_streaming`
works unchanged and the offline and streaming paths agree to ~1e-6.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch import nn

from vguitar.config import TrainConfig
from vguitar.losses import esr_loss, multi_stft_loss
from vguitar.models.base import FitReport, Model, register_model
from vguitar.models.tcn import _GatedLayer

if TYPE_CHECKING:
    from vguitar.data import Dataset


class _CIRCENet(nn.Module):
    """Dilated gated-TCN backbone + FiLM conditioner + output saturator."""

    # Typed buffers (annotations let the checker see them as Tensors, not the
    # loose Tensor|Module that nn.Module.__getattr__ returns).
    c_mean: torch.Tensor
    c_std: torch.Tensor
    out_bound: torch.Tensor

    def __init__(
        self,
        channels: int,
        n_blocks: int,
        n_layers: int,
        kernel: int,
        n_control: int,
        cond_hidden: int,
    ) -> None:
        super().__init__()
        self.n_total = n_blocks * n_layers
        self.channels = channels
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
        # Conditioner: control vector -> per-layer (gamma, beta). Tanh hidden so
        # the numpy streaming path matches exactly. Zero-init the readout => at
        # init gamma=1, beta=0 (identity), so the untrained net is the plain TCN.
        cond_out = nn.Linear(cond_hidden, 2 * channels * self.n_total)
        nn.init.zeros_(cond_out.weight)
        nn.init.zeros_(cond_out.bias)
        self.cond = nn.Sequential(nn.Linear(n_control, cond_hidden), nn.Tanh(), cond_out)
        # Control normalization + output bound, fit from data (registered so they
        # save/load and move with the module).
        self.register_buffer("c_mean", torch.zeros(n_control))
        self.register_buffer("c_std", torch.ones(n_control))
        self.register_buffer("out_bound", torch.tensor(1.0))

    def film(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Control ``c`` (B, K) -> per-layer ``gamma, beta`` each (B, n_total, C)."""
        cn = (c - self.c_mean) / self.c_std
        gb = self.cond(cn).view(-1, self.n_total, 2, self.channels)
        gamma = 1.0 + torch.tanh(gb[:, :, 0, :])  # centred at 1, bounded (0, 2)
        beta = gb[:, :, 1, :]
        return gamma, beta

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``x`` (B, T), ``c`` (B, K) -> (B, T)."""
        h = self.input(x.unsqueeze(1))  # (B, C, T)
        gamma, beta = self.film(c)
        skip = h.new_zeros(h.shape)
        for i, layer in enumerate(self.layers):
            h = gamma[:, i, :, None] * h + beta[:, i, :, None]  # FiLM before the layer
            h, s = layer(h)
            skip = skip + s
        y = self.out(skip).squeeze(1)  # (B, T)
        a = self.out_bound
        return a * torch.tanh(y / a)  # saturator of last resort


@register_model
class CIRCE(Model):
    """Conditioned real-time circuit emulator: FiLM-conditioned cached-streaming TCN."""

    name: ClassVar[str] = "circe"
    description: ClassVar[str] = "Conditioned (FiLM) real-time TCN — interactive circuit emulator."
    latency_samples: int = 0
    conditioned: ClassVar[bool] = True

    def __init__(
        self,
        n_control: int = 1,
        channels: int = 8,
        n_blocks: int = 2,
        n_layers: int = 7,
        kernel: int = 3,
        cond_hidden: int = 16,
        device: str = "cpu",
    ) -> None:
        self.n_control = n_control
        self.channels = channels
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.kernel = kernel
        self.cond_hidden = cond_hidden
        self.device = torch.device(device)
        self.net = _CIRCENet(channels, n_blocks, n_layers, kernel, n_control, cond_hidden).to(
            self.device
        )
        self._stream: dict[str, Any] | None = None

    def _hparams(self) -> dict[str, int]:
        return {
            "n_control": self.n_control,
            "channels": self.channels,
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "kernel": self.kernel,
            "cond_hidden": self.cond_hidden,
        }

    # --- learning ---------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Train on conditioned windows: each window carries the control row at
        its start (controls are piecewise-constant per SPICE-sweep segment)."""
        cfg = cfg or TrainConfig()
        if train.controls is None or train.n_controls != self.n_control:
            raise ValueError(
                f"CIRCE needs a dataset with {self.n_control} control column(s); "
                f"got {train.n_controls}"
            )
        torch.manual_seed(cfg.seed)
        dev = self.device
        warmup = min(cfg.warmup, cfg.seq_len - 1)

        # Control normalization + output bound from the training data.
        c_all = np.asarray(train.controls, dtype=np.float32)
        self.net.c_mean.copy_(torch.from_numpy(c_all.mean(0)))
        self.net.c_std.copy_(torch.from_numpy(c_all.std(0) + 1e-6))
        self.net.out_bound.copy_(torch.tensor(1.2 * float(np.max(np.abs(train.y)) + 1e-6)))

        xb, yb, cb = self._windows(train, cfg.seq_len, hop=cfg.seq_len - warmup)
        xb, yb, cb = xb.to(dev), yb.to(dev), cb.to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(cfg.seed)

        history: dict[str, list[float]] = {"train_loss": [], "val_esr": []}
        best_val, best_state = float("inf"), self._clone_state()
        for _ in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(xb.shape[0], generator=gen)
            ep = 0.0
            n_batch = 0
            for i in range(0, len(perm), cfg.batch_size):
                sel = perm[i : i + cfg.batch_size]
                pred = self.net(xb[sel], cb[sel])[:, warmup:]
                tgt = yb[sel][:, warmup:]
                loss = esr_loss(pred, tgt) + 0.1 * multi_stft_loss(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep += loss.item()
                n_batch += 1
            history["train_loss"].append(ep / max(n_batch, 1))
            ve = self._eval_esr(val, cfg) if val is not None else history["train_loss"][-1]
            history["val_esr"].append(ve)
            if ve < best_val:
                best_val, best_state = ve, self._clone_state()

        self._load_state(best_state)
        self.reset()
        return FitReport(history=history, info={"best_val_esr": best_val})

    def _clone_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def _load_state(self, state: dict[str, torch.Tensor]) -> None:
        self.net.load_state_dict(state)

    def _windows(
        self, ds: Dataset, seq_len: int, hop: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tile into (x, y, c) windows; each window's control is the row at its start."""
        assert ds.controls is not None
        x = np.ascontiguousarray(ds.x, dtype=np.float32)
        y = np.ascontiguousarray(ds.y, dtype=np.float32)
        c = np.ascontiguousarray(ds.controls, dtype=np.float32)
        n = len(ds) - seq_len + 1
        starts = np.arange(0, max(n, 1), hop)
        idx = starts[:, None] + np.arange(seq_len)[None, :]
        xb = torch.from_numpy(x[idx])
        yb = torch.from_numpy(y[idx])
        cb = torch.from_numpy(c[starts])  # (n_win, K)
        return xb, yb, cb

    @torch.no_grad()
    def _eval_esr(self, ds: Dataset, cfg: TrainConfig) -> float:
        """Mean per-segment test ESR (each segment evaluated at its own control)."""
        from vguitar.metrics import esr

        if ds.controls is None:
            return float("inf")
        self.net.eval()
        errs, segs = [], _segments(ds.controls)
        for s, e in segs:
            pred = self.process(ds.x[s:e], ds.controls[s])
            w = min(cfg.warmup, (e - s) - 1)
            errs.append(esr(ds.y[s:e][w:], pred[w:]))
        return float(np.mean(errs)) if errs else float("inf")

    # --- offline inference ------------------------------------------------
    def process(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        """Offline forward at a CONSTANT control ``c`` (``None`` => zero control)."""
        self.net.eval()
        xv = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        cv = self._control_vec(c)
        with torch.no_grad():
            xt = torch.from_numpy(xv).to(self.device).view(1, -1)
            ct = torch.from_numpy(cv).to(self.device).view(1, -1)
            y = self.net(xt, ct).view(-1)
        return y.cpu().numpy().astype(np.float32)

    def _control_vec(self, c: np.ndarray | None) -> np.ndarray:
        if c is None:
            return self.net.c_mean.detach().cpu().numpy().astype(np.float32)
        cv = np.asarray(c, dtype=np.float32).reshape(-1)
        if cv.shape[0] != self.n_control:
            raise ValueError(f"control has {cv.shape[0]} entries, expected {self.n_control}")
        return cv

    # --- streaming inference ----------------------------------------------
    def reset(self) -> None:
        self._stream = None

    def _build_stream(self) -> dict[str, Any]:
        c = self.channels
        sd = {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        n_lyr = self.n_blocks * self.n_layers
        dil = [2 ** (i % self.n_layers) for i in range(n_lyr)]
        layers = [
            (
                sd[f"layers.{i}.conv.weight"],
                sd[f"layers.{i}.conv.bias"],
                sd[f"layers.{i}.res.weight"][:, :, 0],
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
            "n_lyr": n_lyr,
            "w_in": sd["input.weight"][:, 0, 0],
            "b_in": sd["input.bias"],
            "layers": layers,
            "o1w": sd["out.1.weight"][:, :, 0],
            "o1b": sd["out.1.bias"],
            "o3w": sd["out.3.weight"][:, :, 0],
            "o3b": sd["out.3.bias"],
            "cw0": sd["cond.0.weight"],
            "cb0": sd["cond.0.bias"],
            "cw1": sd["cond.2.weight"],
            "cb1": sd["cond.2.bias"],
            "c_mean": sd["c_mean"],
            "c_std": sd["c_std"],
            "out_bound": float(sd["out_bound"]),
            "buf": [np.zeros((c, (self.kernel - 1) * d), dtype=np.float32) for d in dil],
        }

    def _film_np(self, s: dict[str, Any], c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Numpy FiLM: control -> per-layer (gamma, beta), each (n_lyr, C)."""
        cn = (c - s["c_mean"]) / s["c_std"]
        gb = s["cw1"] @ np.tanh(s["cw0"] @ cn + s["cb0"]) + s["cb1"]
        gb = gb.reshape(s["n_lyr"], 2, s["c"])
        gamma = 1.0 + np.tanh(gb[:, 0, :])
        beta = gb[:, 1, :]
        return gamma.astype(np.float32), beta.astype(np.float32)

    def process_block(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        """Stream one block at control ``c`` (cached conv + per-block FiLM)."""
        if self._stream is None:
            self._stream = self._build_stream()
        s = self._stream
        xb = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        nb = xb.shape[0]
        if nb == 0:
            return np.empty(0, dtype=np.float32)
        gamma, beta = self._film_np(s, self._control_vec(c))
        ch, k, dil = s["c"], s["k"], s["dil"]
        h = s["w_in"][:, None] * xb[None, :] + s["b_in"][:, None]
        skip = np.zeros((ch, nb), dtype=np.float32)
        for i, (cw, cb, rw, rb, sw, sb) in enumerate(s["layers"]):
            h = gamma[i][:, None] * h + beta[i][:, None]  # FiLM before the layer
            d = dil[i]
            ctx = np.concatenate([s["buf"][i], h], axis=1)
            conv = cb[:, None] + sum(cw[:, :, t] @ ctx[:, t * d : t * d + nb] for t in range(k))
            if k > 1:
                s["buf"][i] = ctx[:, -(k - 1) * d :].copy()
            g = np.tanh(conv[:ch]) * (1.0 / (1.0 + np.exp(-conv[ch:])))
            h = h + (rw @ g + rb[:, None])
            skip += sw @ g + sb[:, None]
        o = np.maximum(skip, 0.0)
        o = np.maximum(s["o1w"] @ o + s["o1b"][:, None], 0.0)
        o = s["o3w"] @ o + s["o3b"][:, None]
        a = s["out_bound"]
        return (a * np.tanh(o[0] / a)).astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save({"hparams": self._hparams(), "state_dict": self.net.state_dict()}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> CIRCE:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**blob["hparams"])
        model.net.load_state_dict(blob["state_dict"])
        model.reset()
        return model

    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def _segments(controls: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous spans over which the control row is constant."""
    controls = np.asarray(controls)
    if len(controls) == 0:
        return []
    change = np.any(np.diff(controls, axis=0) != 0, axis=1)
    bounds = [0, *(np.flatnonzero(change) + 1).tolist(), len(controls)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
