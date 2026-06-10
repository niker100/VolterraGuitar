"""Uniform-improvement sweep (circuit-agnostic) with gradient clipping as the new
baseline. Every config is a SINGLE setting applied to ALL circuits — the model is
trained per-circuit but the config never depends on circuit identity.

Questions:
  base      : gated, clip1.0, 70ep, ch24/L9, OS2          -- the new SOTA reference
  mixed     : block_act=mixed, clip1.0                     -- does the corner-capable
                                                              activation help now that
                                                              grad-clip stabilises it?
  clip0.5   : gentler clip                                 -- avoids the mild hard_clipper
                                                              regression at 1.0?
  clip2.0   : stronger clip
  ep120     : 120 epochs                                   -- does longer training help now
                                                              that grad-clip prevents the
                                                              hard-circuit overfit?

Evaluated on a representative mix (smooth: bjt, jfet; hard/diverse: crossover,
wavefolder, fullwave_rectifier). Hard/noisy circuits get 2 seeds. A config "wins"
only if it lowers the hard-circuit mean WITHOUT regressing the smooth circuits.

Logs to outputs/logs/sweep_uniform.log; writes outputs/sweep_uniform.json.
Run: uv run python sweep_uniform.py
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

CIRCUITS = [
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth", 1),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth", 1),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard", 2),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard", 2),
    ("fullwave_rectifier", "fullwave_rectifier_edge_sweep", "fullwave_rectifier_edge_test", "hard", 1),
]
# name -> (model kwargs over defaults, epochs)
CONFIGS = {
    "base":    (dict(block_act="gated", grad_clip=1.0), 70),
    "mixed":   (dict(block_act="mixed", grad_clip=1.0), 70),
    "clip0.5": (dict(block_act="gated", grad_clip=0.5), 70),
    "clip2.0": (dict(block_act="gated", grad_clip=2.0), 70),
    "ep120":   (dict(block_act="gated", grad_clip=1.0), 120),
}
SEEDS = (0, 7)

LOG = Path("outputs/logs/sweep_uniform.log")
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
    log(f"configs={list(CONFIGS)}  circuits={[c[0] for c in CIRCUITS]}  (all ch24/L9 OS2)")
    data = {key: (Dataset.load(f"data/{sw}.npz"), Dataset.load(f"data/{ts}.npz"), kind, ns)
            for key, sw, ts, kind, ns in CIRCUITS}
    results: dict[str, dict] = {}
    for cfg_name, (kw, epochs) in CONFIGS.items():
        results[cfg_name] = {"by_circuit": {}}
        for key, (tr, ts, kind, nseeds) in data.items():
            vals = []
            for seed in SEEDS[:nseeds]:
                t = time.time()
                torch.manual_seed(seed)
                m = CIRCE3(n_control=1, signal_idx=(0,), channels=24, n_blocks=2,
                           n_layers=9, oversample=2, device="cuda", **kw)
                m.fit(tr, ts, TrainConfig(epochs=epochs, lr=3e-3, seq_len=4096,
                                          batch_size=12, warmup=2048, seed=seed))
                h = held_esr(m, ts)
                vals.append(h)
                log(f"{cfg_name:8s} {key:18s} [{kind:6s}] seed{seed} held {h:.4f} "
                    f"({time.time()-t:.0f}s)")
            results[cfg_name]["by_circuit"][key] = {"held": float(np.mean(vals)),
                                                    "kind": kind, "seeds": vals}
        bc = results[cfg_name]["by_circuit"]
        hard = [v["held"] for v in bc.values() if v["kind"] == "hard"]
        smooth = [v["held"] for v in bc.values() if v["kind"] == "smooth"]
        results[cfg_name]["hard_mean"] = float(np.mean(hard))
        results[cfg_name]["smooth_mean"] = float(np.mean(smooth))
        log(f"  -> {cfg_name}: smooth-mean {np.mean(smooth):.4f}  hard-mean {np.mean(hard):.4f}")
    Path("outputs/sweep_uniform.json").write_text(json.dumps(results, indent=2))
    log("=== UNIFORM SWEEP VERDICT (vs base; lower=better) ===")
    base = results["base"]
    cols = [c[0] for c in CIRCUITS]
    log(f"  {'config':8s} " + " ".join(f"{c[:9]:>10s}" for c in cols) + "  smooth   hard")
    for cfg_name in CONFIGS:
        bc = results[cfg_name]["by_circuit"]
        row = " ".join(f"{bc[c]['held']:>10.4f}" for c in cols)
        log(f"  {cfg_name:8s} {row}  {results[cfg_name]['smooth_mean']:.4f}  "
            f"{results[cfg_name]['hard_mean']:.4f}")
    log(f"(base smooth {base['smooth_mean']:.4f} hard {base['hard_mean']:.4f})")
    log("wrote outputs/sweep_uniform.json")


if __name__ == "__main__":
    main()
