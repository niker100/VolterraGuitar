"""Generalization A/B: does the mixed-activation TCN block regress on SMOOTH
circuits, where the smooth gated-TCN (CIRCE3) already hits ESR < 0.005?

The radical search proved mixed activations beat the gated-TCN 5-9x on the
class-B crossover dead-zone. The open risk (synthesis): the win is by
construction specific to *corners*; the claim "never hurts on smooth circuits"
was asserted, not measured. This script measures it.

Controlled A/B — identical harness (same windows, loss, epochs, optimizer,
grad-clip); the ONLY difference is the TCN block's activation:
  * MIXED   : conv -> [tanh|gelu|relu|abs|snake groups] -> 1x1 mix -> residual
  * GATED   : conv -> tanh(a)*sigmoid(b) (WaveNet/CIRCE3 gate) -> 1x1 -> residual
Both ~98-99k params, fully causal (left-pad only), 8 dilated blocks 1..128.

Circuits:
  smooth (make-or-break) : bjt, jfet, tube_screamer   (*_bench_sweep/_bench_fp_test)
  hard   (confirm win)   : crossover, wavefolder, asym_clipper (*_edge_sweep/_edge_test)

Writes outputs/radical_generalize_ab.json and logs to
outputs/logs/radical_generalize_ab.log (flushed) so progress is visible live.

Run: uv run python radical_generalize_ab.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vguitar import metrics as M
from vguitar.data import Dataset
from vguitar.losses import esr_loss, preemph_esr_loss
from vguitar.models.base import pick_device
from vguitar.models.circe3 import _segments

EPOCHS = 60
SEQ, HOP, BATCH = 4096, 2048, 12
DILS = (1, 2, 4, 8, 16, 32, 64, 128)
SEEDS = (0, 7)

CIRCUITS = [
    # (key, sweep, test, kind)
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth"),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth"),
    ("tube_screamer", "tube_screamer_bench_sweep", "tube_screamer_bench_fp_test", "smooth"),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard"),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard"),
    ("asym_clipper", "asym_clipper_edge_sweep", "asym_clipper_edge_test", "hard"),
]

LOG = Path("outputs/logs/radical_generalize_ab.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
_t0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - _t0:7.1f}s] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


# --- blocks: the only difference between the two models is the activation -----
class _MixedAct(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        base = ch // 5
        self.sizes = [base, base, base, base, ch - 4 * base]
        self.alpha = nn.Parameter(torch.ones(self.sizes[4]))

    def forward(self, h):
        s = self.sizes
        i0 = s[0]
        i1 = i0 + s[1]
        i2 = i1 + s[2]
        i3 = i2 + s[3]
        sn = h[:, i3:]
        al = self.alpha.clamp(min=0.1).view(1, -1, 1)
        return torch.cat(
            [
                torch.tanh(h[:, :i0]),
                torch.nn.functional.gelu(h[:, i0:i1]),
                torch.relu(h[:, i1:i2]),
                torch.abs(h[:, i2:i3]),
                sn + torch.sin(al * sn) ** 2 / al,
            ],
            dim=1,
        )


class _MixedBlock(nn.Module):
    def __init__(self, ch, dil, k=3):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(ch, ch, k, dilation=dil)
        self.act = _MixedAct(ch)
        self.mix = nn.Conv1d(ch, ch, 1)

    def forward(self, h):
        z = torch.nn.functional.pad(h, (self.pad, 0))
        return h + self.mix(self.act(self.conv(z)))


class _GatedBlock(nn.Module):
    def __init__(self, ch, dil, k=3):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(ch, 2 * ch, k, dilation=dil)
        self.mix = nn.Conv1d(ch, ch, 1)

    def forward(self, h):
        z = torch.nn.functional.pad(h, (self.pad, 0))
        a, b = self.conv(z).chunk(2, dim=1)
        return h + self.mix(torch.tanh(a) * torch.sigmoid(b))


class TCN(nn.Module):
    def __init__(self, block, ch):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList([block(ch, d) for d in DILS])
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.inp(x.unsqueeze(1))
        for b in self.blocks:
            h = b(h)
        return self.out(h).squeeze(1)


def make_windows(sweep: Dataset, dev):
    x = np.ascontiguousarray(sweep.x, np.float32)
    y = np.ascontiguousarray(sweep.y, np.float32)
    g = np.ascontiguousarray(sweep.controls[:, 0], np.float32)
    xin = (x * g).astype(np.float32)
    starts = np.arange(0, len(xin) - SEQ, HOP)
    idx = starts[:, None] + np.arange(SEQ)[None, :]
    return (torch.from_numpy(xin[idx]).to(dev), torch.from_numpy(y[idx]).to(dev))


def train_eval(kind_name, X, Y, test, dev, seed):
    torch.manual_seed(seed)
    ch = 55 if kind_name == "mixed" else 42  # match params ~98-99k
    block = _MixedBlock if kind_name == "mixed" else _GatedBlock
    m = TCN(block, ch).to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=3e-5)
    for _ep in range(EPOCHS):
        perm = torch.randperm(X.shape[0])
        for i in range(0, len(perm), BATCH):
            sel = perm[i:i + BATCH]
            pred = m(X[sel])[:, 2048:]
            tgt = Y[sel][:, 2048:]
            loss = esr_loss(pred, tgt) + preemph_esr_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
    s, e = _segments(test.controls)[0]
    # feed input-scaled signal g*x to match training (bench tests use g!=1)
    xin_t = (test.x[s:e] * test.controls[s:e, 0]).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(xin_t).to(dev).view(1, -1)
        pe = m(xt).view(-1).cpu().numpy()
    held = float(M.esr(test.y[s:e][2048:], pe[2048:]))
    return held, nparam


def main() -> None:
    dev = pick_device()
    log(f"device={dev} epochs={EPOCHS} seeds={SEEDS}")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind in CIRCUITS:
        sweep = Dataset.load(f"data/{sweep_nm}.npz")
        test = Dataset.load(f"data/{test_nm}.npz")
        X, Y = make_windows(sweep, dev)
        results[key] = {"kind": kind, "mixed": [], "gated": [], "params": {}}
        for model in ("gated", "mixed"):
            for seed in SEEDS:
                t = time.time()
                held, npar = train_eval(model, X, Y, test, dev, seed)
                results[key][model].append(held)
                results[key]["params"][model] = npar
                log(f"{key:14s} [{kind:6s}] {model:5s} seed{seed} "
                    f"held-ESR {held:.4f} ({npar/1000:.0f}k, {time.time()-t:.0f}s)")
        gm = float(np.mean(results[key]["gated"]))
        mm = float(np.mean(results[key]["mixed"]))
        delta = (mm - gm) / gm * 100.0
        verdict = "WIN" if mm < gm * 0.97 else ("REGRESS" if mm > gm * 1.10 else "TIE")
        results[key]["summary"] = {"gated_mean": gm, "mixed_mean": mm,
                                   "rel_change_pct": delta, "verdict": verdict}
        log(f"  -> {key}: gated {gm:.4f}  mixed {mm:.4f}  "
            f"({delta:+.0f}%)  {verdict}")
        del X, Y
        if dev == "cuda":
            torch.cuda.empty_cache()

    out = Path("outputs/radical_generalize_ab.json")
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    # concise verdict block
    log("=== GENERALIZATION VERDICT ===")
    for key, _, _, kind in CIRCUITS:
        s = results[key]["summary"]
        log(f"  {key:14s} [{kind:6s}] gated {s['gated_mean']:.4f} -> "
            f"mixed {s['mixed_mean']:.4f}  {s['verdict']}")


if __name__ == "__main__":
    main()
