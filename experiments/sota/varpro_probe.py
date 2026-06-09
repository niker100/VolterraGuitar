"""Alternating Variable-Projection (VarPro) training (user idea): a trainable TCN trunk
+ a WIDE feature layer + a LINEAR readout solved by lstsq, ALTERNATING closed-form
readout solves with backprop on the hidden weights.

Pure ELM (random features + lstsq) plateaus ~0.06-0.23 (structural — random features). The
fix: TRAIN the hidden weights so the features become good, while keeping the readout
always-optimal via lstsq. Each outer step: (A) freeze the trunk, solve the linear readout
W in closed form (chunked, no_grad, memory-safe); (B) freeze W, backprop the trunk for a
few epochs. Repeat. Combines trained features with an optimal readout — should leap past
ELM toward the trained frontier, and the always-optimal head can improve conditioning vs
plain joint Adam.

A/B (same trunk + wide proj + linear readout, OS1, equal total epochs): VarPro (lstsq
readout, alternating) vs standard joint training (readout trained by SGD). Memory-safe:
solve via chunked HtH; backprop in minibatch windows; wide features only live per-minibatch.

Run (background): uv run python -m experiments.sota.varpro_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.common import make_log
from experiments.sota.harness import CIRCUITS as HC
from vguitar import metrics as M
from vguitar.losses import esr_loss, preemph_esr_loss
from vguitar.models.circe3 import _segments
from vguitar.models.tcn import _MixedLayer

DEVICE = "cuda"
CHANNELS, N_LAYERS, FEAT_DIM, KERNEL = 24, 9, 512, 3
N_OUTER, EPOCHS_PER = 15, 10  # 150 total backprop epochs (matches the standard baseline)
SEQ_LEN, WARMUP, BATCH, LR, RIDGE = 4096, 2048, 24, 3e-3, 1e-2
CHUNK = 65536
CIRCUITS = ["bjt", "jfet", "tube_screamer", "crossover", "hard_clipper"]
REF = {"bjt": 0.0046, "jfet": 0.0020, "tube_screamer": 0.0145, "crossover": 0.0225,
       "hard_clipper": 0.0558}


class VarProNet(nn.Module):
    """Trainable dilated mixed-activation trunk -> wide ReLU feature layer -> linear
    readout. Readout is a lstsq-solved buffer ``W`` (VarPro) or a trained ``nn.Linear``
    (joint baseline)."""

    out_bound: torch.Tensor

    def __init__(self, trainable_readout: bool, channels: int = CHANNELS,
                 n_layers: int = N_LAYERS, feat_dim: int = FEAT_DIM, kernel: int = KERNEL) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.layers = nn.ModuleList(_MixedLayer(channels, kernel, 2**i) for i in range(n_layers))
        self.proj = nn.Conv1d(channels, feat_dim, 1)
        self.feat_dim = feat_dim
        self.rf = 1 + (kernel - 1) * sum(2**i for i in range(n_layers))
        self.trainable_readout = trainable_readout
        if trainable_readout:
            self.readout = nn.Linear(feat_dim, 1)
        else:
            self.register_buffer("W", torch.zeros(feat_dim + 1))  # [weights, bias], lstsq
        self.register_buffer("out_bound", torch.tensor(1.0))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x.unsqueeze(1) if x.ndim == 2 else x.view(1, 1, -1))
        skip = torch.zeros_like(h)
        for layer in self.layers:
            h, sk = layer(h)
            skip = skip + sk
        return F.relu(self.proj(skip))  # (B, feat_dim, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        if self.trainable_readout:
            o = self.readout(f.transpose(1, 2)).squeeze(-1)
        else:
            o = (self.W[:-1].view(1, -1, 1) * f).sum(1) + self.W[-1]
        a = float(self.out_bound)
        return torch.clamp(o, -a, a)


def _segs(ds) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for s, e in _segments(ds.controls):
        g = float(ds.controls[s, 0])
        out.append(((ds.x[s:e] * g).astype(np.float32), ds.y[s:e].astype(np.float32)))
    return out


@torch.no_grad()
def solve_W(net: VarProNet, segs, device: str, ridge: float = RIDGE, chunk: int = CHUNK) -> torch.Tensor:
    """Closed-form ridge readout over the training segments — chunked, fp64, no graph."""
    nf = net.feat_dim + 1
    a = torch.zeros(nf, nf, device=device, dtype=torch.float64)
    bv = torch.zeros(nf, device=device, dtype=torch.float64)
    for xs, ys in segs:
        x = torch.from_numpy(xs).to(device)
        y = torch.from_numpy(ys).to(device)
        for start in range(0, x.shape[0], chunk):
            n = min(chunk, x.shape[0] - start)
            lo = max(0, start - (net.rf - 1))
            f = net.features(x[lo:start + n].view(1, 1, -1))[0][:, start - lo:]
            fb = torch.cat([f, torch.ones(1, n, device=device)], 0).double()
            a += fb @ fb.T
            bv += fb @ y[start:start + n].double()
            del f, fb
        del x, y
        torch.cuda.empty_cache()
    a += ridge * torch.eye(nf, device=device, dtype=torch.float64)
    return torch.linalg.solve(a, bv).float()


@torch.no_grad()
def predict(net: VarProNet, x: np.ndarray, device: str, chunk: int = CHUNK) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    out = torch.empty(xt.shape[0], device=device)
    for start in range(0, xt.shape[0], chunk):
        n = min(chunk, xt.shape[0] - start)
        lo = max(0, start - (net.rf - 1))
        out[start:start + n] = net(xt[lo:start + n].view(1, -1))[0][start - lo:]
    return out.cpu().numpy()


def _windows(segs, device: str):
    xs, ys = [], []
    for x, y in segs:
        nstart = len(x) - SEQ_LEN + 1
        if nstart <= 0:
            continue
        starts = np.arange(0, nstart, SEQ_LEN - WARMUP)
        idx = starts[:, None] + np.arange(SEQ_LEN)[None, :]
        xs.append(x[idx])
        ys.append(y[idx])
    return (torch.from_numpy(np.concatenate(xs)).to(device),
            torch.from_numpy(np.concatenate(ys)).to(device))


def train(net: VarProNet, segs, device: str, varpro: bool, log) -> None:
    net.to(device)
    net.out_bound.copy_(torch.tensor(1.2 * max(float(np.max(np.abs(y))) for _, y in segs) + 1e-6))
    xb, yb = _windows(segs, device)
    params = [p for n, p in net.named_parameters() if "readout" not in n or not varpro]
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_OUTER * EPOCHS_PER,
                                                       eta_min=LR * 0.01)
    gen = torch.Generator().manual_seed(0)
    for _ in range(N_OUTER):
        if varpro:
            net.eval()
            net.W.copy_(solve_W(net, segs, device))  # closed-form readout (trunk frozen)
        net.train()
        for _ep in range(EPOCHS_PER):
            perm = torch.randperm(xb.shape[0], generator=gen)
            for i in range(0, len(perm), BATCH):
                sel = perm[i:i + BATCH]
                pred = net(xb[sel])[:, WARMUP:]
                tgt = yb[sel][:, WARMUP:]
                loss = esr_loss(pred, tgt) + preemph_esr_loss(pred, tgt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            sched.step()
    if varpro:
        net.eval()
        net.W.copy_(solve_W(net, segs, device))  # final readout


def main() -> None:
    log = make_log("sota_varpro_probe")
    out = Path("outputs/sota/varpro_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"VarPro A/B: trunk ch{CHANNELS}/L{N_LAYERS} + proj feat{FEAT_DIM}, {N_OUTER}x{EPOCHS_PER}ep OS1")
    from vguitar.data import Dataset
    results: dict = {}
    for key in CIRCUITS:
        sweep, test, kind = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        segs = _segs(tr)
        s, e = _segments(ts.controls)[0]
        gt = float(ts.controls[s, 0])
        xt = (ts.x[s:e] * gt).astype(np.float32)
        yt = ts.y[s:e].astype(np.float32)
        row: dict = {"kind": kind, "ref_trained": REF.get(key)}
        for tag, vp in (("joint", False), ("varpro", True)):
            t = time.time()
            torch.manual_seed(0)
            net = VarProNet(trainable_readout=not vp)
            train(net, segs, DEVICE, varpro=vp, log=log)
            pred = predict(net, xt, DEVICE)
            n = min(len(yt), len(pred))
            esr = float(M.esr(yt[:n][2048:], pred[:n][2048:]))
            row[tag] = {"held": esr, "secs": time.time() - t}
            log(f"{key:14s} {tag:7s} held={esr:.4f} ({time.time()-t:.0f}s)  [ref {REF.get(key)}]")
            del net
            torch.cuda.empty_cache()
        d = row["joint"]["held"]
        log(f"  -> {key}: joint {d:.4f} -> varpro {row['varpro']['held']:.4f} "
            f"({100*(row['varpro']['held']-d)/d:+.0f}%)")
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("done — wrote outputs/sota/varpro_probe.json")


if __name__ == "__main__":
    main()
