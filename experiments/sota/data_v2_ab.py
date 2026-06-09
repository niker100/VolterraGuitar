"""Campaign 6 — data-quality A/B: does ~3x more training data lower held-ESR?

The audit found the hard sweeps carried 41% less data than the smooth ones; regen_data
rebuilt 6 sweeps at seg_dur_s=8 (~3x) to *_v2.npz. This trains the production config
(dcblock_off, OS2, nb2/nl9) on the original vs the _v2 sweep and scores on the UNCHANGED
test — isolating the data-volume lever (test content identical). The user flagged data
quality as historically high-impact; this is the clean measurement of it.

Run (background): uv run python -m experiments.sota.data_v2_ab
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from experiments.common import held_esr, make_log
from experiments.sota.harness import UNIFIED
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3

EPOCHS = 150
# circuit -> (original sweep, v2 sweep, unchanged test)
CIRCUITS = {
    "jfet": ("jfet_bench_sweep", "jfet_bench_sweep_v2", "jfet_bench_fp_test"),
    "tube_screamer": ("tube_screamer_bench_sweep", "tube_screamer_bench_sweep_v2",
                      "tube_screamer_bench_fp_test"),
    "fullwave_rectifier": ("fullwave_rectifier_edge_sweep", "fullwave_rectifier_edge_sweep_v2",
                           "fullwave_rectifier_edge_test"),
    "crossover": ("crossover_classb_edge_sweep", "crossover_classb_edge_sweep_v2",
                  "crossover_classb_edge_test"),
    "hard_clipper": ("hard_clipper_edge_sweep", "hard_clipper_edge_sweep_v2",
                     "hard_clipper_edge_test"),
    "hysteretic_fuzz": ("hysteretic_fuzz_edge_sweep", "hysteretic_fuzz_edge_sweep_v2",
                        "hysteretic_fuzz_edge_test"),
}


def _train_eval(sweep: str, test: str, seed: int = 0) -> float:
    tr = Dataset.load(f"data/{sweep}.npz")
    ts = Dataset.load(f"data/{test}.npz")
    torch.manual_seed(seed)
    # the unified production config: depth L10 + IIR memory + dcblock_off + OS2
    m = CIRCE3(n_control=1, signal_idx=(0,), device="cuda", **UNIFIED)
    m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096, batch_size=12,
                              warmup=2048, seed=seed))
    return held_esr(m, ts)


def main() -> None:
    log = make_log("sota_data_v2_ab")
    out = Path("outputs/sota/data_v2_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    log(f"data-quality A/B: original vs v2 (seg_dur 8s, ~3x) sweeps, dcblock_off prod "
        f"config, {EPOCHS}ep")
    for key, (orig, v2, test) in CIRCUITS.items():
        row: dict = {"kind": "smooth" if key in ("jfet", "tube_screamer") else "hard"}
        for tag, sweep in (("orig", orig), ("v2", v2)):
            t = time.time()
            esr = _train_eval(sweep, test)
            row[tag] = {"held": esr, "secs": time.time() - t}
            log(f"{key:18s} {tag:4s} held={esr:.4f} ({time.time()-t:.0f}s)")
        d = row["orig"]["held"]
        imp = 100.0 * (row["v2"]["held"] - d) / d if d else 0.0
        log(f"  -> {key}: {d:.4f} -> {row['v2']['held']:.4f} ({imp:+.0f}%)")
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("done")


if __name__ == "__main__":
    main()
