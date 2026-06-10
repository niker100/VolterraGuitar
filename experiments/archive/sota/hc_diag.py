"""hard_clipper b12 collapse diagnostic.

speed_ab's in-run control (b12/lr3e-3, seed 0) scored held-ESR 1.0000 on
hard_clipper — but the IDENTICAL config/seed scored 0.0578 in unified_varpro.
fit() restores best-val weights, so 1.0 means val never improved: collapse from
epoch ~1. Two hypotheses: (a) GPU nondeterminism on a knife-edge init (same seed
can differ across runs via cuDNN/cuBLAS reductions), or (b) something systematic
changed. Test: run the exact cell twice at seed 0 (+ once at seed 7), capturing
the per-epoch val-ESR history. Repeats disagreeing => (a); both collapsed =>
bisect for (b). Also runs b96/lr9 once — the arm that survived — for the
stability contrast.

Run (background): uv run python -m experiments.sota.hc_diag
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
from vguitar.models.archive.circe3 import CIRCE3

RUNS = [  # (tag, seed, batch, lr)
    ("b12_s0_a", 0, 12, 3e-3),
    ("b12_s0_b", 0, 12, 3e-3),
    ("b12_s7", 7, 12, 3e-3),
    ("b96_s0", 0, 96, 9e-3),
]


def main() -> None:
    log = make_log("sota_hc_diag")
    out = Path("outputs/sota/hc_diag.json")
    sweep, test, _ = HC["hard_clipper"]
    tr = Dataset.load(f"data/{sweep}.npz")
    ts = Dataset.load(f"data/{test}.npz")
    results: dict = json.loads(out.read_text()) if out.exists() else {}
    log(f"hard_clipper collapse diagnostic: {[r[0] for r in RUNS]}")
    for tag, seed, bs, lr in RUNS:
        if tag in results:
            continue
        t = time.time()
        torch.manual_seed(seed)
        m = CIRCE3(n_control=1, signal_idx=(0,), device="cuda", **UNIFIED)
        rep = m.fit(tr, ts, TrainConfig(epochs=150, lr=lr, seq_len=4096, batch_size=bs,
                                        warmup=2048, seed=seed))
        esr = held_esr(m, ts)
        ve = rep.history["val_esr"]
        results[tag] = {"held": esr, "secs": time.time() - t, "val_esr": ve,
                        "first_under_0.1": next((i for i, v in enumerate(ve) if v < 0.1), None)}
        log(f"{tag:10s} held={esr:.4f} val[0]={ve[0]:.3f} val[5]={ve[5]:.3f} "
            f"best={min(ve):.4f} ({time.time()-t:.0f}s)")
        del m
        torch.cuda.empty_cache()
        out.write_text(json.dumps(results, indent=2))
    log("done")


if __name__ == "__main__":
    main()
