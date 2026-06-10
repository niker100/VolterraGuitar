"""Convergence-speed test for VarPro: does the closed-form (always-optimal) readout let
the trunk reach low ESR in FEWER epochs than joint training? This is the user's
'closed-form is computationally better than GD' claim, measured as epochs-to-convergence
(the only axis where the closed-form readout could genuinely win, since accuracy = joint).

Caveat: the per-batch fp64 solve is ~6x slower per epoch, so a WALL-CLOCK win needs a big
epoch win AND a cheaper solve cadence. jfet + bjt, held-ESR logged at epoch checkpoints.

Run (background): uv run python -m experiments.sota.varpro_conv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.common import make_log
from experiments.sota.harness import CIRCUITS as HC
from experiments.sota.varpro_probe import (
    BATCH,
    LR,
    RIDGE,
    WARMUP,
    VarProNet,
    _segs,
    _solve_diff,
    _windows,
    predict,
    solve_W,
)
from vguitar import metrics as M
from vguitar.data import Dataset
from vguitar.models.archive.circe3 import _segments
from vguitar.models.archive.losses import esr_loss, preemph_esr_loss

DEVICE = "cuda"
CHECKPOINTS = [5, 15, 30, 60, 100, 150]
CIRCUITS = ["jfet", "bjt"]


def _held(net: VarProNet, segs, xt, yt, device, varpro):
    if varpro:
        net.eval()
        net.W.copy_(solve_W(net, segs, device))
    net.eval()
    pred = predict(net, xt, device)
    n = min(len(yt), len(pred))
    return float(M.esr(yt[:n][2048:], pred[:n][2048:]))


def train_logged(net, segs, xt, yt, device, varpro, log, key, tag):
    net.to(device)
    net.out_bound.copy_(torch.tensor(1.2 * max(float(np.max(np.abs(y))) for _, y in segs) + 1e-6))
    xb, yb = _windows(segs, device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(CHECKPOINTS), eta_min=LR * 0.01)
    gen = torch.Generator().manual_seed(0)
    curve = {}
    for ep in range(1, max(CHECKPOINTS) + 1):
        net.train()
        perm = torch.randperm(xb.shape[0], generator=gen)
        for i in range(0, len(perm), BATCH):
            sel = perm[i:i + BATCH]
            tgt = yb[sel][:, WARMUP:]
            pred = (_solve_diff(net.features(xb[sel]), yb[sel], RIDGE, WARMUP) if varpro
                    else net(xb[sel])[:, WARMUP:])
            loss = esr_loss(pred, tgt) + preemph_esr_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep in CHECKPOINTS:
            curve[ep] = _held(net, segs, xt, yt, device, varpro)
            log(f"{key:6s} {tag:7s} ep{ep:4d} held={curve[ep]:.4f}")
    return curve


def main() -> None:
    log = make_log("sota_varpro_conv")
    out = Path("outputs/sota/varpro_conv.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    log(f"VarPro convergence-speed: checkpoints={CHECKPOINTS}, jfet+bjt joint vs varpro")
    for key in CIRCUITS:
        sweep, test, _ = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        segs = _segs(tr)
        s, e = _segments(ts.controls)[0]
        gt = float(ts.controls[s, 0])
        xt = (ts.x[s:e] * gt).astype(np.float32)
        yt = ts.y[s:e].astype(np.float32)
        row = {}
        for tag, vp in (("joint", False), ("varpro", True)):
            t = time.time()
            torch.manual_seed(0)
            net = VarProNet(trainable_readout=not vp)
            row[tag] = train_logged(net, segs, xt, yt, DEVICE, vp, log, key, tag)
            row[tag]["secs"] = time.time() - t
            del net
            torch.cuda.empty_cache()
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("done — wrote outputs/sota/varpro_conv.json")


if __name__ == "__main__":
    main()
