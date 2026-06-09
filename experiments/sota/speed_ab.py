"""Training-speed A/B: batch size + LR + bf16 AMP, measuring wall-clock AND held-ESR.

The 4090 is badly underutilized at batch=12 (65% util, 18% mem, 45% power). Bigger batch
fills it, but with fixed epochs it means FEWER optimizer steps, so LR is co-scaled to
hold convergence — hence this measures ESR too, not just speed. AMP (bf16, weights stay
fp32, numpy twin untouched) adds a compute speedup. Goal: the fastest arm whose held-ESR
matches the batch=12 control within noise -> adopt as the harness default for all future
campaigns (compounding: faster multi-seed/ablation + RT).

Round-1 finding (jfet): sqrt-LR scaling holds to b96 (lr9: 0.0038, BETTER than control
0.0050) but DIVERGES at b192/lr12 (ESR 0.91). So the big-batch arms now reuse the proven
lr 9e-3 — fewer steps at a safe LR instead of more LR. Resumes from speed_ab.json so
finished cells are never redone (the round-1 run was killed by a session restart).

Config = the unified model (nb2/L10 + n_state=4 + dcblock_off). Battery: jfet
(near-boundary 0.005, convergence-sensitive), hard_clipper (heavy), bjt (guard). 1 seed.

Run (background): uv run python -m experiments.sota.speed_ab
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from experiments.common import held_esr, make_log
from experiments.sota.harness import CIRCUITS as HC
from experiments.sota.harness import UNIFIED as MODEL
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3

# (label, batch, lr, amp). lr12+ diverged at b192 (round 1) — big batches stay at the
# proven 9e-3. b96_amp vs b96_lr9 isolates the pure bf16 effect. OOM arms are skipped.
ARMS = [
    ("b12", 12, 3e-3, False),         # control (current default)
    ("b96_lr9", 96, 9e-3, False),     # 8x batch, fp32
    ("b96_amp", 96, 9e-3, True),      # 8x batch, bf16  (vs b96_lr9 = pure AMP effect)
    ("b192_amp9", 192, 9e-3, True),   # 16x batch, bf16, safe LR
    # b384: too big for the card (user-confirmed) — WDDM swaps instead of OOM-ing,
    # so the try/except never fires and the box chokes. Do not re-add.
]
CIRC = ["jfet", "hard_clipper", "bjt"]
EPOCHS = 150


def main() -> None:
    log = make_log("sota_speed_ab")
    out = Path("outputs/sota/speed_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = json.loads(out.read_text()) if out.exists() else {}
    log(f"speed A/B arms={[a[0] for a in ARMS]} circuits={CIRC} {EPOCHS}ep, unified "
        f"config (resume: {sum(len(v) - 1 for v in results.values())} cells present)")
    for key in CIRC:
        sweep, test, kind = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        row: dict = results.setdefault(key, {"kind": kind})
        for label, bs, lr, amp in ARMS:
            if row.get(label, {}).get("held") is not None:  # resume
                continue
            t = time.time()
            try:
                torch.manual_seed(0)
                m = CIRCE3(n_control=1, signal_idx=(0,), device="cuda", **MODEL)
                m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=lr, seq_len=4096, batch_size=bs,
                                          warmup=2048, seed=0, amp=amp))
                esr = held_esr(m, ts)
                row[label] = {"held": esr, "secs": time.time() - t, "batch": bs,
                              "lr": lr, "amp": amp}
                log(f"{key:14s} {label:10s} held={esr:.4f} ({time.time()-t:.0f}s)")
                del m
            except Exception as exc:  # OOM at big batch -> skip, free, continue
                torch.cuda.empty_cache()
                row[label] = {"held": None, "secs": time.time() - t, "batch": bs,
                              "error": str(exc)[:120]}
                log(f"{key:14s} {label:10s} FAILED: {str(exc)[:100]}")
            out.write_text(json.dumps(results, indent=2))
    log("=== speed / accuracy summary (secs | held-ESR) ===")
    for key in CIRC:
        for label, _, _, _ in ARMS:
            r = results.get(key, {}).get(label)
            if r is None or r.get("held") is None:
                continue
            log(f"  {key:14s} {label:9s} {r['secs']:5.0f}s  ESR {r['held']:.4f}")
    log("done")


if __name__ == "__main__":
    main()
