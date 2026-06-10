"""Decisive OS ablation: does internal 2x oversampling HURT hard-discontinuity
circuits? Isolates the ONLY variable (oversample in {1, 2}) at the identical
production CIRCE3 config (gated, ch24, n_blocks=2, n_layers=9, 80 epochs, real
.fit + latency-compensated held eval).

Hypothesis (from the mixed-act prod run): OS2 band-limits the sharp dead-zone /
fold corner -> Gibbs ringing on the discontinuity, which dominates the aliasing
benefit OS gives SMOOTH circuits. If true, the right default is circuit-class
dependent: OS2 for smooth saturating circuits, OS1 for sharp-corner circuits.

Circuits: crossover (dead-zone) + wavefolder (multi-fold) [hard] + bjt [smooth
control, where OS is known to help ~6x]. 2 seeds each.

Logs to outputs/logs/radical_os_ablation.log; writes outputs/radical_os_ablation.json.
Run: uv run python radical_os_ablation.py
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

EPOCHS = 80
SEEDS = (0, 7)
CIRCUITS = [
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard"),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard"),
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth"),
]

LOG = Path("outputs/logs/radical_os_ablation.log")
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
    log(f"epochs={EPOCHS} seeds={SEEDS}")
    tcfg = TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096, batch_size=12, warmup=2048)
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        results[key] = {"kind": kind, "os1": [], "os2": []}
        for osf in (1, 2):
            for seed in SEEDS:
                t = time.time()
                torch.manual_seed(seed)
                m = CIRCE3(n_control=1, signal_idx=(0,), channels=24, n_blocks=2,
                           n_layers=9, oversample=osf, block_act="gated", device="cuda")
                m.fit(tr, ts, TrainConfig(**{**tcfg.__dict__, "seed": seed}))
                held = held_esr(m, ts)
                results[key][f"os{osf}"].append(held)
                log(f"{key:11s} [{kind:6s}] OS{osf} seed{seed} held-ESR {held:.4f} "
                    f"({time.time()-t:.0f}s)")
        o1 = float(np.mean(results[key]["os1"]))
        o2 = float(np.mean(results[key]["os2"]))
        delta = (o2 - o1) / o1 * 100.0
        results[key]["summary"] = {"os1_mean": o1, "os2_mean": o2, "os2_vs_os1_pct": delta}
        log(f"  -> {key} [{kind}]: OS1 {o1:.4f}  OS2 {o2:.4f}  (OS2 {delta:+.0f}% vs OS1)")
    Path("outputs/radical_os_ablation.json").write_text(json.dumps(results, indent=2))
    log("=== OS ABLATION VERDICT (held-ESR, gated ch24/L9) ===")
    for key, _, _, kind in CIRCUITS:
        s = results[key]["summary"]
        better = "OS1 better" if s["os1_mean"] < s["os2_mean"] else "OS2 better"
        log(f"  {key:11s} [{kind:6s}] OS1 {s['os1_mean']:.4f} | OS2 {s['os2_mean']:.4f}  -> {better}")
    log("wrote outputs/radical_os_ablation.json")


if __name__ == "__main__":
    main()
