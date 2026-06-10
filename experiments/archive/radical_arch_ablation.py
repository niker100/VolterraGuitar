"""Uniform-architecture search: the radical fair-A/B harness (8-layer single
stack, ch42, residual-only, direct output, OS1) reached crossover 0.07, while
production CIRCE3 (ch24, 2x9=18 layers, skip-sum head, OS2) sits at ~0.31 on the
SAME eval. The OS ablation showed oversampling is NOT the cause (OS1 vs OS2 ~10%
at fixed config). So the lever is the ARCHITECTURE itself.

This isolates depth + width INSIDE CIRCE3 (so any winner is a single uniform
config usable for ALL circuits — no per-circuit tailoring, OS2 kept uniform for
the smooth-circuit anti-aliasing the user requires). Configs:
  A prod        : ch24, nb2, nl9  (18 layers, RF 2045) -- baseline
  B wide-shallow: ch40, nb1, nl8  ( 8 layers, RF  511) -- radical-like
  C wide-deep   : ch40, nb2, nl9  (18 layers)          -- isolates width vs A
(B vs C isolates depth; A vs C isolates width; A vs B is the full change.)

Evaluated on a representative MIX: bjt (smooth-easy), jfet (smooth-memory),
crossover (hard-static), wavefolder (hard-fold). The best UNIFORM config minimises
the worst-case / mean held-ESR across the mix. Logs to
outputs/logs/radical_arch_ablation.log; writes outputs/radical_arch_ablation.json.

Run: uv run python radical_arch_ablation.py
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

EPOCHS = 70
CIRCUITS = [
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth"),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth"),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard"),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard"),
]
CONFIGS = [
    ("prod_ch24_18L", dict(channels=24, n_blocks=2, n_layers=9)),
    ("wide_shallow_ch40_8L", dict(channels=40, n_blocks=1, n_layers=8)),
    ("wide_deep_ch40_18L", dict(channels=40, n_blocks=2, n_layers=9)),
]

LOG = Path("outputs/logs/radical_arch_ablation.log")
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
    w = 2048
    return float(M.esr(yc[w:], pc[w:]))


def main() -> None:
    log(f"epochs={EPOCHS} configs={[c[0] for c in CONFIGS]} (all OS2, uniform)")
    tcfg = TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096, batch_size=12, warmup=2048)
    results: dict[str, dict] = {}
    for cfg_name, kw in CONFIGS:
        results[cfg_name] = {"params": None, "by_circuit": {}}
        for key, sweep_nm, test_nm, kind in CIRCUITS:
            tr = Dataset.load(f"data/{sweep_nm}.npz")
            ts = Dataset.load(f"data/{test_nm}.npz")
            t = time.time()
            torch.manual_seed(0)
            m = CIRCE3(n_control=1, signal_idx=(0,), oversample=2, device="cuda", **kw)
            results[cfg_name]["params"] = m.num_params()
            m.fit(tr, ts, tcfg)
            held = held_esr(m, ts)
            results[cfg_name]["by_circuit"][key] = {"held": held, "kind": kind}
            log(f"{cfg_name:22s} {key:11s} [{kind:6s}] held-ESR {held:.4f} "
                f"({m.num_params()/1000:.0f}k, {time.time()-t:.0f}s)")
        vals = [v["held"] for v in results[cfg_name]["by_circuit"].values()]
        hard = [v["held"] for v in results[cfg_name]["by_circuit"].values() if v["kind"] == "hard"]
        results[cfg_name]["mean"] = float(np.mean(vals))
        results[cfg_name]["worst"] = float(np.max(vals))
        results[cfg_name]["hard_mean"] = float(np.mean(hard))
        log(f"  -> {cfg_name}: mean {results[cfg_name]['mean']:.4f}  "
            f"worst {results[cfg_name]['worst']:.4f}  hard-mean {results[cfg_name]['hard_mean']:.4f}")
    Path("outputs/radical_arch_ablation.json").write_text(json.dumps(results, indent=2))
    log("=== UNIFORM-ARCHITECTURE VERDICT (held-ESR, OS2) ===")
    cols = [c[0] for c in CIRCUITS]
    log(f"  {'config':22s} " + " ".join(f"{c:>11s}" for c in cols) + "   mean   hard")
    for cfg_name, _ in CONFIGS:
        bc = results[cfg_name]["by_circuit"]
        row = " ".join(f"{bc[c]['held']:>11.4f}" for c in cols)
        log(f"  {cfg_name:22s} {row}   {results[cfg_name]['mean']:.4f}  "
            f"{results[cfg_name]['hard_mean']:.4f}")
    log("wrote outputs/radical_arch_ablation.json")


if __name__ == "__main__":
    main()
