"""Confirm gradient clipping is a UNIFORM win (no regression) on the circuits not
covered by radical_gradclip_test.py. clip0 (off) vs clip1 (the new default) at the
production CIRCE3 config (ch24, nb2, nl9, OS2). Merges into the existing
outputs/radical_gradclip_test.json for one consolidated plot.

Run: uv run python radical_gradclip_rest.py
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
    ("tube_screamer", "tube_screamer_bench_sweep", "tube_screamer_bench_fp_test", "smooth", 1),
    ("hard_clipper", "hard_clipper_edge_sweep", "hard_clipper_edge_test", "hard", 2),
    ("asym_clipper", "asym_clipper_edge_sweep", "asym_clipper_edge_test", "hard", 2),
    ("fullwave_rectifier", "fullwave_rectifier_edge_sweep", "fullwave_rectifier_edge_test", "hard", 2),
    ("hysteretic_fuzz", "hysteretic_fuzz_edge_sweep", "hysteretic_fuzz_edge_test", "hard", 2),
]
CLIPS = [0.0, 1.0]
SEEDS = (0, 7)

LOG = Path("outputs/logs/radical_gradclip_rest.log")
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
    log(f"epochs={EPOCHS} clips={CLIPS} (prod ch24/nb2/nl9, OS2) -- remaining circuits")
    out_p = Path("outputs/radical_gradclip_test.json")
    results = json.loads(out_p.read_text()) if out_p.exists() else {}
    for key, sweep_nm, test_nm, kind, nseeds in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        results[key] = {"kind": kind, "clip0.0": [], "clip1.0": []}
        for clip in CLIPS:
            for seed in SEEDS[:nseeds]:
                t = time.time()
                torch.manual_seed(seed)
                m = CIRCE3(n_control=1, signal_idx=(0,), channels=24, n_blocks=2,
                           n_layers=9, oversample=2, grad_clip=clip, device="cuda")
                m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                          batch_size=12, warmup=2048, seed=seed))
                held = held_esr(m, ts)
                results[key][f"clip{clip}"].append(held)
                log(f"{key:18s} [{kind:6s}] clip{clip} seed{seed} held-ESR {held:.4f} "
                    f"({time.time()-t:.0f}s)")
        c0 = float(np.mean(results[key]["clip0.0"]))
        c1 = float(np.mean(results[key]["clip1.0"]))
        results[key]["summary"] = {"clip0_mean": c0, "clip1_mean": c1,
                                   "rel_pct": (c1 - c0) / c0 * 100.0}
        log(f"  -> {key} [{kind}]: clip0 {c0:.4f}  clip1 {c1:.4f}  "
            f"({(c1-c0)/c0*100:+.0f}%)")
    out_p.write_text(json.dumps(results, indent=2))
    log("=== GRAD-CLIP (remaining) VERDICT ===")
    for key, _, _, kind, _ in CIRCUITS:
        s = results[key]["summary"]
        v = "helps" if s["clip1_mean"] < s["clip0_mean"] * 0.97 else (
            "HURTS" if s["clip1_mean"] > s["clip0_mean"] * 1.03 else "neutral")
        log(f"  {key:18s} [{kind:6s}] clip0 {s['clip0_mean']:.4f} -> "
            f"clip1 {s['clip1_mean']:.4f}  {v}")
    log("wrote (merged) outputs/radical_gradclip_test.json")


if __name__ == "__main__":
    main()
