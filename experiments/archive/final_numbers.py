"""Authoritative held-ESR table for the SHIPPING CIRCE3 config (ch24/nb2/nl9, OS2,
pre-emph, 1Hz DC-block, grad_clip=1.0 default) at the production 150 epochs,
across all 9 circuits. Refreshes the (now stale) model-card numbers after the
gradient-clipping integration. grad-clip is training-only -> RTF/latency unchanged
(~2.2x CPU, latency 63), so this only needs held-ESR.

Hard/noisy circuits get 2 seeds. Logs to outputs/logs/final_numbers.log; writes
outputs/final_numbers.json.

Run: uv run python final_numbers.py
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
from vguitar.models.archive.circe3 import CIRCE3, _segments

EPOCHS = 150
CIRCUITS = [
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth", 1),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth", 1),
    ("tube_screamer", "tube_screamer_bench_sweep", "tube_screamer_bench_fp_test", "smooth", 1),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard", 2),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard", 2),
    ("asym_clipper", "asym_clipper_edge_sweep", "asym_clipper_edge_test", "hard", 1),
    ("hard_clipper", "hard_clipper_edge_sweep", "hard_clipper_edge_test", "hard", 1),
    ("fullwave_rectifier", "fullwave_rectifier_edge_sweep", "fullwave_rectifier_edge_test", "hard", 1),
    ("hysteretic_fuzz", "hysteretic_fuzz_edge_sweep", "hysteretic_fuzz_edge_test", "hard", 1),
]
SEEDS = (0, 7)

LOG = Path("outputs/logs/final_numbers.log")
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
    log(f"epochs={EPOCHS} shipping config (ch24/L9/OS2/grad_clip=1.0)")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind, nseeds in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        vals = []
        for seed in SEEDS[:nseeds]:
            t = time.time()
            torch.manual_seed(seed)
            # shipping config: n_blocks=2, n_layers=9 (the constructor default is L8!)
            m = CIRCE3(n_control=1, signal_idx=(0,), n_blocks=2, n_layers=9,
                       oversample=2, device="cuda")  # grad_clip=1.0 default
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=seed))
            h = held_esr(m, ts)
            vals.append(h)
            log(f"{key:18s} [{kind:6s}] seed{seed} held-ESR {h:.4f} ({time.time()-t:.0f}s)")
        results[key] = {"kind": kind, "held": float(np.mean(vals)), "seeds": vals}
        log(f"  -> {key}: held-ESR {np.mean(vals):.4f}")
    results["_params"] = sum(p.numel() for p in
                             CIRCE3(n_control=1, n_blocks=2, n_layers=9, oversample=2).net.parameters())
    Path("outputs/final_numbers.json").write_text(json.dumps(results, indent=2))
    log("=== SHIPPING CIRCE3 held-ESR (150ep, grad_clip=1.0, OS2) ===")
    for key, _, _, kind, _ in CIRCUITS:
        log(f"  {key:18s} [{kind:6s}] {results[key]['held']:.4f}")
    log(f"params={results['_params']/1000:.0f}k")
    log("wrote outputs/final_numbers.json")


if __name__ == "__main__":
    main()
