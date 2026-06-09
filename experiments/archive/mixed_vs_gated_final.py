"""Decisive default-decision run: gated vs mixed block, BOTH with grad-clip, at the
shipping config (ch24/L9/OS2, 150 ep), across all 9 circuits. The RTF check already
showed mixed is smaller (54k vs 85k) and FASTER (2.45x vs 2.23x CPU), and grad-clip
fixed mixed's old wavefolder instability — so if mixed is uniformly >= gated here it
should become the default block_act. A win = no smooth-circuit regression and a net
gain on the hard frontier.

Logs to outputs/logs/mixed_vs_gated_final.log; writes outputs/mixed_vs_gated_final.json.
Run: uv run python mixed_vs_gated_final.py
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

LOG = Path("outputs/logs/mixed_vs_gated_final.log")
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
    log(f"epochs={EPOCHS} gated vs mixed (both grad_clip=1.0, ch24/L9/OS2)")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind, nseeds in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        results[key] = {"kind": kind}
        for act in ("gated", "mixed"):
            vals = []
            for seed in SEEDS[:nseeds]:
                t = time.time()
                torch.manual_seed(seed)
                m = CIRCE3(n_control=1, signal_idx=(0,), n_blocks=2, n_layers=9,
                           oversample=2, block_act=act, device="cuda")
                m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                          batch_size=12, warmup=2048, seed=seed))
                vals.append(held_esr(m, ts))
                log(f"{key:18s} [{kind:6s}] {act:5s} seed{seed} held {vals[-1]:.4f} "
                    f"({time.time()-t:.0f}s)")
            results[key][act] = float(np.mean(vals))
        g, mx = results[key]["gated"], results[key]["mixed"]
        results[key]["rel_pct"] = (mx - g) / g * 100.0
        log(f"  -> {key} [{kind}]: gated {g:.4f}  mixed {mx:.4f}  ({(mx-g)/g*100:+.0f}%)")
    Path("outputs/mixed_vs_gated_final.json").write_text(json.dumps(results, indent=2))
    log("=== MIXED vs GATED VERDICT (both grad-clip, 150ep) ===")
    gw = sum(1 for k in results if results[k]["mixed"] < results[k]["gated"] * 0.97)
    gr = sum(1 for k in results if results[k]["mixed"] > results[k]["gated"] * 1.03)
    for key, _, _, kind, _ in CIRCUITS:
        r = results[key]
        v = "mixed" if r["mixed"] < r["gated"] * 0.97 else (
            "gated" if r["mixed"] > r["gated"] * 1.03 else "tie")
        log(f"  {key:18s} [{kind:6s}] gated {r['gated']:.4f} -> mixed {r['mixed']:.4f}  {v}")
    log(f"mixed wins {gw}, gated wins {gr}, of {len(results)} circuits "
        f"(mixed = 54k params, RTF 2.45x vs gated 85k, 2.23x)")
    log("wrote outputs/mixed_vs_gated_final.json")


if __name__ == "__main__":
    main()
