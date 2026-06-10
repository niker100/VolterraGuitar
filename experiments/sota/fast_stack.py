"""The quick-learner stack: VarPro x big-batch x LR-warmup on the smooth circuits.

Each lever is individually validated: VarPro reaches the plateau in ~3x fewer epochs
(closed-form readout = always-optimal head from step 1) and sharpens smooth circuits
(jfet 0.0010 vs 0.0045); b96 cuts wall-clock ~1.4-2.2x; the 5-epoch LR warmup removes
the big-batch early-overshoot collapse. This measures the COMBINED stack — varpro-60ep
at b96/lr9/w5 — against the standard-150ep/b12 reference on the circuits where VarPro
is uniform-safe (smooth only; it breaks discontinuity circuits). Target: reference ESR
at ~4-7x less wall-clock = the iteration config for all future smooth-circuit work.

Run (background): uv run python -m experiments.sota.fast_stack
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from experiments.common import held_esr, make_log
from experiments.sota.harness import CIRCUITS as HC
from experiments.sota.harness import UNIFIED
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3

# (label, epochs, batch, lr, varpro, lr_warmup)
ARMS = [
    ("ref_std150_b12", 150, 12, 3e-3, False, 0),   # the leaderboard reference
    ("vp60_b96_w5", 60, 96, 9e-3, True, 5),        # the full quick-learner stack
    ("vp60_b12", 60, 12, 3e-3, True, 0),           # validated varpro speedup (control)
    ("vp150_b96_w5", 150, 96, 9e-3, True, 5),      # stack at full epochs (accuracy ceiling)
]
CIRC = ["jfet", "bjt", "tube_screamer"]


def main() -> None:
    log = make_log("sota_fast_stack")
    out = Path("outputs/sota/fast_stack.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = json.loads(out.read_text()) if out.exists() else {}
    log(f"fast-stack arms={[a[0] for a in ARMS]} circuits={CIRC}")
    for key in CIRC:
        sweep, test, kind = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        row: dict = results.setdefault(key, {"kind": kind})
        for label, ep, bs, lr, vp, warm in ARMS:
            if row.get(label, {}).get("held") is not None:  # resume
                continue
            t = time.time()
            try:
                torch.manual_seed(0)
                m = CIRCE3(n_control=1, signal_idx=(0,), device="cuda", **UNIFIED)
                m.fit(tr, ts, TrainConfig(epochs=ep, lr=lr, seq_len=4096, batch_size=bs,
                                          warmup=2048, seed=0, varpro=vp, lr_warmup=warm))
                esr = held_esr(m, ts)
                row[label] = {"held": esr, "secs": time.time() - t, "epochs": ep,
                              "batch": bs, "varpro": vp, "lr_warmup": warm}
                log(f"{key:14s} {label:16s} held={esr:.4f} ({time.time()-t:.0f}s)")
                del m
            except Exception as exc:
                torch.cuda.empty_cache()
                row[label] = {"held": None, "secs": time.time() - t,
                              "error": str(exc)[:120]}
                log(f"{key:14s} {label:16s} FAILED: {str(exc)[:100]}")
            out.write_text(json.dumps(results, indent=2))
    log("=== fast-stack summary (secs | held-ESR) ===")
    for key in CIRC:
        for label, *_ in ARMS:
            r = results.get(key, {}).get(label)
            if r is None or r.get("held") is None:
                continue
            log(f"  {key:14s} {label:16s} {r['secs']:5.0f}s  ESR {r['held']:.4f}")
    log("done")


if __name__ == "__main__":
    main()
