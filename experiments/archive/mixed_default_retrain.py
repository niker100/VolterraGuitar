"""Retrain the production CIRCE3 with the NEW default block (block_act='mixed').

The decisive 9-circuit gated-vs-mixed head-to-head (outputs/logs/mixed_vs_gated_final.log)
concluded mixed is the circuit-agnostic default: it wins 4 / ties 3 / loses 2 (the two
losses mild — jfet +14%, tube_screamer +3%), and is SMALLER (54k vs 85k) and FASTER
(RTF 2.45x vs 2.23x). Per the project's uniform-config constraint, the shipped
checkpoints + the model-card numbers must use the default config, so this:

  * retrains all 9 circuits at the shipping config (ch24/nb2/nl9, OS2, pre-emph,
    1 Hz DC-block, grad_clip=1.0) with the new mixed default,
  * RE-SAVES the three shipped checkpoints (bjt/jfet/tube_screamer) as mixed,
  * records authoritative held-ESR for the model card.

grad-clip + OS2 are training/inference unchanged; only the block activation flipped.
Hard/noisy circuits get 2 seeds. Logs to outputs/logs/mixed_default_retrain.log;
writes outputs/mixed_default_retrain.json.

Run: uv run python mixed_default_retrain.py
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
# (key, sweep dataset, test dataset, kind, n_seeds, checkpoint_name|None)
CIRCUITS = [
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth", 1, "bjt"),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth", 1, "jfet"),
    ("tube_screamer", "tube_screamer_bench_sweep", "tube_screamer_bench_fp_test",
     "smooth", 1, "tube_screamer"),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard", 2, None),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard", 2, None),
    ("asym_clipper", "asym_clipper_edge_sweep", "asym_clipper_edge_test", "hard", 1, None),
    ("hard_clipper", "hard_clipper_edge_sweep", "hard_clipper_edge_test", "hard", 1, None),
    ("fullwave_rectifier", "fullwave_rectifier_edge_sweep", "fullwave_rectifier_edge_test",
     "hard", 1, None),
    ("hysteretic_fuzz", "hysteretic_fuzz_edge_sweep", "hysteretic_fuzz_edge_test", "hard", 1, None),
]
SEEDS = (0, 7)
CKPT_DIR = Path("assets/checkpoints")

LOG = Path("outputs/logs/mixed_default_retrain.log")
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
    log(f"epochs={EPOCHS} shipping config (ch24/L9/OS2/grad_clip=1.0) block_act=mixed (DEFAULT)")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind, nseeds, ckpt in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        vals = []
        best_model: CIRCE3 | None = None
        best_h = float("inf")
        for seed in SEEDS[:nseeds]:
            t = time.time()
            torch.manual_seed(seed)
            # shipping config; block_act defaults to 'mixed' now. n_layers=9 (ctor default is 8).
            m = CIRCE3(n_control=1, signal_idx=(0,), n_blocks=2, n_layers=9,
                       oversample=2, device="cuda")
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=seed))
            h = held_esr(m, ts)
            vals.append(h)
            if h < best_h:
                best_h, best_model = h, m
            log(f"{key:18s} [{kind:6s}] seed{seed} held-ESR {h:.4f} "
                f"block_act={m.block_act} params={m.num_params()} ({time.time()-t:.0f}s)")
        results[key] = {"kind": kind, "held": float(np.mean(vals)), "seeds": vals}
        if ckpt is not None and best_model is not None:
            path = CKPT_DIR / f"{ckpt}.circe3.model"
            best_model.save(path)
            log(f"  saved checkpoint {path} (held {best_h:.4f}, block_act={best_model.block_act})")
        log(f"  -> {key}: held-ESR {np.mean(vals):.4f}")
    params = CIRCE3(n_control=1, n_blocks=2, n_layers=9, oversample=2).num_params()
    results["_params"] = params
    results["_block_act"] = "mixed"
    Path("outputs/mixed_default_retrain.json").write_text(json.dumps(results, indent=2))
    log("=== SHIPPING CIRCE3 held-ESR (150ep, grad_clip=1.0, OS2, block_act=mixed) ===")
    for key, _, _, kind, _, _ in CIRCUITS:
        log(f"  {key:18s} [{kind:6s}] {results[key]['held']:.4f}")
    log(f"params={params/1000:.0f}k")
    log("wrote outputs/mixed_default_retrain.json")


if __name__ == "__main__":
    main()
