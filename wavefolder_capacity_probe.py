"""Branch B swing #2 — is the wavefolder 0.18 floor a MODEL limit or a TARGET floor?

Two swings have failed to move it: more oversampling (OS4 strictly worse -> not
aliasing) and a learned periodic output shaper (within seed noise -> an output
fold-primitive is not enough). The decisive remaining question is whether 0.18 is a
capacity/real-time limit at all, or a fundamental property of the (band-limited,
decimated-SPICE) TARGET that NO model can beat.

This probe drops the real-time constraint and throws capacity + bandwidth at the
wavefolder — an unconstrained "teacher". Read:
  * If a big model gets WELL under 0.18 -> the floor is capacity/real-time, and the
    next move is distilling a real-time student toward the teacher (Branch B #3).
  * If even a big model stalls at ~0.18 -> the wall is the band-limited target
    itself (the multi-fold's harmonics exceed what a causal, band-limited operator
    can recover from this excitation) -> a fundamental, honest conclusion; stop
    chasing it with architecture.

Reports held-ESR + params + CPU RTF. Logs to outputs/logs/wavefolder_capacity_probe.log;
writes outputs/wavefolder_capacity_probe.json.

Run: uv run python wavefolder_capacity_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from vguitar import metrics as M
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3, _segments
from vguitar.realtime import measure_rtf

SWEEP = "wavefolder_edge_sweep"
TEST = "wavefolder_edge_test"
SEEDS = (0, 7)
# (label, channels, n_layers, oversample, epochs, lr, grad_clip, n_seeds)
# CORRECTED v2: the first attempt used the ch24-tuned lr=3e-3 + grad_clip=1.0 for ALL
# sizes -> ch48 scored 0.93 (diverged/underfit: clip=1.0 throttles a 4x-bigger grad
# norm). A capacity test must give big models a fair shot, so big configs get a gentler
# lr=1e-3, grad-clip OFF, and more epochs. OS4 is dropped (already shown harmful here).
CONFIGS = [
    ("ref_ch24_L9_OS2", 24, 9, 2, 150, 3e-3, 1.0, 2),  # known-good reference (~0.17)
    ("ch48_lr1e3_noclip", 48, 9, 2, 200, 1e-3, 0.0, 1),  # 4x params, tuned for size
    ("ch48_L11_lr1e3_noclip", 48, 11, 2, 200, 1e-3, 0.0, 1),  # + receptive field
    ("ch64_L11_lr1e3_noclip", 64, 11, 2, 250, 1e-3, 0.0, 1),  # well-tuned big "teacher"
]

LOG = Path("outputs/logs/wavefolder_capacity_probe.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
_t0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - _t0:7.1f}s] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def held_esr(model: CIRCE3, test: Dataset) -> float:
    s, e = _segments(test.controls)[0]
    x = np.ascontiguousarray(test.x[s:e], np.float32)
    y = np.ascontiguousarray(test.y[s:e], np.float32)
    g = float(test.controls[s, 0])
    pred = model.process(x, np.array([g], np.float32))
    lat = model.latency_samples
    n = min(len(y), len(pred))
    yc, pc = y[:n], pred[:n]
    if lat > 0:
        pc = pc[lat:]
        yc = yc[: len(pc)]
    return float(M.esr(yc[2048:], pc[2048:]))


def main() -> None:
    log("wavefolder capacity probe: is 0.18 a model limit or a target floor?")
    tr = Dataset.load(f"data/{SWEEP}.npz")
    ts = Dataset.load(f"data/{TEST}.npz")
    results: dict[str, dict] = {}
    for label, ch, nl, osf, ep, lr, gc, nseeds in CONFIGS:
        vals = []
        rtf = params = None
        for seed in SEEDS[:nseeds]:
            t = time.time()
            torch.manual_seed(seed)
            m = CIRCE3(n_control=1, signal_idx=(0,), channels=ch, n_blocks=2, n_layers=nl,
                       oversample=osf, grad_clip=gc, device="cuda")  # mixed default
            m.fit(tr, ts, TrainConfig(epochs=ep, lr=lr, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=seed))
            h = held_esr(m, ts)
            vals.append(h)
            if rtf is None:
                rtf = float(measure_rtf(m, sr=ts.sr, block=128)["rtf"])
                params = int(m.num_params())
            log(f"{label:24s} seed{seed} held {h:.4f} lr={lr:g} clip={gc:g} "
                f"params={m.num_params()} ({time.time()-t:.0f}s)")
        mean = float(np.mean(vals))
        results[label] = {"channels": ch, "n_layers": nl, "oversample": osf, "epochs": ep,
                          "lr": lr, "grad_clip": gc,
                          "held": mean, "seeds": vals, "params": params, "rtf": rtf}
        rt = "real-time" if (rtf or 0) > 1.0 else "NOT-RT"
        log(f"  -> {label}: held {mean:.4f}  params {params}  RTF {rtf:.2f}x ({rt})")
    Path("outputs/wavefolder_capacity_probe.json").write_text(json.dumps(results, indent=2))
    ref = results["ref_ch24_L9_OS2"]["held"]
    log("=== WAVEFOLDER CAPACITY PROBE (held-ESR vs the ~0.18 reference) ===")
    for label, *_ in CONFIGS:
        r = results[label]
        delta = 100.0 * (r["held"] - ref) / ref
        log(f"  {label:22s} held {r['held']:.4f} ({delta:+.0f}%)  params {r['params']:>7}  "
            f"RTF {r['rtf']:.2f}x")
    log("wrote outputs/wavefolder_capacity_probe.json")


if __name__ == "__main__":
    main()
