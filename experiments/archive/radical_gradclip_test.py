"""Does gradient clipping (a UNIFORM, standard, circuit-agnostic, zero-inference-
cost training change) close the hard-circuit gap? CIRCE3.fit did NOT clip; the
radical fair-A/B harness used clip_grad_norm(1.0) and every radical agent reported
crossover training was unstable without it. Unstable optimization on a sharp
discontinuity can land in a much worse minimum (CIRCE3 crossover ~0.38 vs the
clipped radical harness ~0.07 on the same eval).

Compares grad_clip in {0.0 (current), 1.0} at the production CIRCE3 config
(ch24, nb2, nl9, OS2), same .fit pipeline. Hard/noisy circuits get extra seeds.
If clip helps hard circuits AND does not hurt smooth -> integrate as the default.

Logs to outputs/logs/radical_gradclip_test.log; writes outputs/radical_gradclip_test.json.
Run: uv run python radical_gradclip_test.py
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
# (key, sweep, test, kind, n_seeds)  -- more seeds where held-ESR is noisy
CIRCUITS = [
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth", 1),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth", 1),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard", 3),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard", 2),
]
CLIPS = [0.0, 1.0]
SEEDS = (0, 7, 13)

LOG = Path("outputs/logs/radical_gradclip_test.log")
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
    log(f"epochs={EPOCHS} clips={CLIPS} (prod ch24/nb2/nl9, OS2)")
    results: dict[str, dict] = {}
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
                tcfg = TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096, batch_size=12,
                                   warmup=2048, seed=seed)
                m.fit(tr, ts, tcfg)
                held = held_esr(m, ts)
                results[key][f"clip{clip}"].append(held)
                log(f"{key:11s} [{kind:6s}] clip{clip} seed{seed} held-ESR {held:.4f} "
                    f"({time.time()-t:.0f}s)")
        c0 = float(np.mean(results[key]["clip0.0"]))
        c1 = float(np.mean(results[key]["clip1.0"]))
        delta = (c1 - c0) / c0 * 100.0
        results[key]["summary"] = {"clip0_mean": c0, "clip1_mean": c1, "rel_pct": delta}
        log(f"  -> {key} [{kind}]: clip0 {c0:.4f}  clip1 {c1:.4f}  ({delta:+.0f}%)")
    Path("outputs/radical_gradclip_test.json").write_text(json.dumps(results, indent=2))
    log("=== GRAD-CLIP VERDICT (held-ESR, OS2) ===")
    for key, _, _, kind, _ in CIRCUITS:
        s = results[key]["summary"]
        better = "clip helps" if s["clip1_mean"] < s["clip0_mean"] * 0.97 else (
            "clip hurts" if s["clip1_mean"] > s["clip0_mean"] * 1.03 else "neutral")
        log(f"  {key:11s} [{kind:6s}] clip0 {s['clip0_mean']:.4f} -> "
            f"clip1 {s['clip1_mean']:.4f}  {better}")
    log("wrote outputs/radical_gradclip_test.json")


if __name__ == "__main__":
    main()
