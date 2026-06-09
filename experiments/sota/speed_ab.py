"""Training-speed A/B: batch size + LR + bf16 AMP, measuring wall-clock AND held-ESR.

The 4090 is badly underutilized at batch=12 (65% util, 18% mem, 45% power). Bigger batch
fills it, but with fixed epochs it means FEWER optimizer steps, so LR is co-scaled
(~sqrt rule) to hold convergence — hence this measures ESR too, not just speed. AMP
(bf16, weights stay fp32, numpy twin untouched) adds a compute speedup. Goal: the fastest
arm whose held-ESR matches the batch=12 control within noise -> adopt as the harness
default for all future campaigns (compounding: faster multi-seed/ablation + RT).

Config = the unified model (nb2/L10 + n_state=4 + dcblock_off). Battery: jfet
(near-boundary 0.002, convergence-sensitive), hard_clipper (heavy), bjt (guard). 1 seed.

Run (background): uv run python -m experiments.sota.speed_ab
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from experiments.common import held_esr, make_log
from experiments.sota.harness import CIRCUITS as HC
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3

MODEL = {"channels": 24, "n_blocks": 2, "n_layers": 10, "oversample": 2,
         "dcblock_fc": 0.0, "n_state": 4}
# (label, batch, lr, amp)
ARMS = [
    ("b12", 12, 3e-3, False),       # control (current default)
    ("b48", 48, 3e-3, False),       # 4x batch, same LR (isolates the fewer-steps cost)
    ("b48_lr6", 48, 6e-3, False),   # 4x batch, ~sqrt-scaled LR
    ("b96_lr9", 96, 9e-3, False),   # 8x batch, ~sqrt-scaled LR
    ("b48_amp", 48, 6e-3, True),    # 4x batch + bf16 AMP
]
CIRC = ["jfet", "hard_clipper", "bjt"]
EPOCHS = 150


def main() -> None:
    log = make_log("sota_speed_ab")
    out = Path("outputs/sota/speed_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    log(f"speed A/B arms={[a[0] for a in ARMS]} circuits={CIRC} {EPOCHS}ep, unified config")
    for key in CIRC:
        sweep, test, kind = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        row: dict = {"kind": kind}
        for label, bs, lr, amp in ARMS:
            t = time.time()
            torch.manual_seed(0)
            m = CIRCE3(n_control=1, signal_idx=(0,), device="cuda", **MODEL)
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=lr, seq_len=4096, batch_size=bs,
                                      warmup=2048, seed=0, amp=amp))
            esr = held_esr(m, ts)
            secs = time.time() - t
            row[label] = {"held": esr, "secs": secs, "batch": bs, "lr": lr, "amp": amp}
            log(f"{key:14s} {label:9s} held={esr:.4f} ({secs:.0f}s)")
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("=== speed / accuracy summary (secs | held-ESR) ===")
    for key in CIRC:
        for label, _, _, _ in ARMS:
            r = results[key][label]
            log(f"  {key:14s} {label:9s} {r['secs']:5.0f}s  ESR {r['held']:.4f}")
    log("done")


if __name__ == "__main__":
    main()
