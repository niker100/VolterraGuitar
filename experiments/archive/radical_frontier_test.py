"""Definitive frontier probe: can ANY uniform lever crack the two hardest circuits
(wavefolder ~0.17, hard_clipper ~0.09) now that gradient clipping stabilises
training? Capacity was tested WITHOUT grad-clip (arch ablation) and was neutral;
this tests capacity (width, depth) and the mixed activation WITH grad-clip + 120ep.

All configs: OS2, grad_clip=1.0, circuit-agnostic. Reports held-ESR + params + a
crude CPU RTF proxy (params) so plugin viability of wider/deeper configs is visible.
If nothing beats the ~0.17 wavefolder floor, it is confirmed a real-time bandwidth
limit (the folds alias even at 2x OS), not an optimization gap.

Logs to outputs/logs/radical_frontier_test.log.
Run: uv run python radical_frontier_test.py
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

EPOCHS = 120
CIRCUITS = [
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test"),
    ("hard_clipper", "hard_clipper_edge_sweep", "hard_clipper_edge_test"),
]
CONFIGS = {
    "base_ch24_L9": dict(channels=24, n_blocks=2, n_layers=9),
    "wide_ch40_L9": dict(channels=40, n_blocks=2, n_layers=9),
    "deep_ch24_L11": dict(channels=24, n_blocks=2, n_layers=11),
    "mixed_ch24_L9": dict(channels=24, n_blocks=2, n_layers=9, block_act="mixed"),
}
SEEDS = (0, 7)

LOG = Path("outputs/logs/radical_frontier_test.log")
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
    log(f"epochs={EPOCHS} configs={list(CONFIGS)} (all OS2, grad_clip=1.0)")
    results: dict[str, dict] = {}
    for cfg_name, kw in CONFIGS.items():
        results[cfg_name] = {"params": None, "by_circuit": {}}
        for key, sweep_nm, test_nm in CIRCUITS:
            tr = Dataset.load(f"data/{sweep_nm}.npz")
            ts = Dataset.load(f"data/{test_nm}.npz")
            vals = []
            for seed in SEEDS:
                t = time.time()
                torch.manual_seed(seed)
                m = CIRCE3(n_control=1, signal_idx=(0,), oversample=2, device="cuda", **kw)
                results[cfg_name]["params"] = m.num_params()
                m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                          batch_size=12, warmup=2048, seed=seed))
                h = held_esr(m, ts)
                vals.append(h)
                log(f"{cfg_name:14s} {key:13s} seed{seed} held {h:.4f} "
                    f"({m.num_params()/1000:.0f}k, {time.time()-t:.0f}s)")
            results[cfg_name]["by_circuit"][key] = float(np.mean(vals))
        log(f"  -> {cfg_name} ({results[cfg_name]['params']/1000:.0f}k): " +
            "  ".join(f"{k} {v:.4f}" for k, v in results[cfg_name]["by_circuit"].items()))
    Path("outputs/radical_frontier_test.json").write_text(json.dumps(results, indent=2))
    log("=== FRONTIER VERDICT (held-ESR, OS2+grad-clip) ===")
    log(f"  {'config':14s} {'params':>7s}  " + "  ".join(f"{c[0]:>13s}" for c in CIRCUITS))
    for cfg_name in CONFIGS:
        bc = results[cfg_name]["by_circuit"]
        row = "  ".join(f"{bc[c[0]]:>13.4f}" for c in CIRCUITS)
        log(f"  {cfg_name:14s} {results[cfg_name]['params']/1000:6.0f}k  {row}")
    log("wrote outputs/radical_frontier_test.json")


if __name__ == "__main__":
    main()
