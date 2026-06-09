"""VarPro on the PRODUCTION CIRCE3 (the integration verification): does closed-form
readout training reach the same held-ESR FASTER on real circuits — including the
memory-heavy hysteretic_fuzz (n_state path)?

Config = the unified production model: nb2/L10 + n_state=4 (IIR memory) + dcblock_off +
OS2. Arms: standard 150 ep vs varpro 60 ep vs varpro 150 ep. If varpro@60ep matches
standard@150ep in much less wall-clock, the closed-form readout is a real training
speedup on the production model. Battery: bjt, jfet, hysteretic_fuzz (memory).

Run (background): uv run python -m experiments.sota.varpro_circe3
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from experiments.common import held_esr, make_log
from experiments.sota.harness import CIRCUITS as HC
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3

MODEL = {"n_control": 1, "signal_idx": (0,), "channels": 24, "n_blocks": 2, "n_layers": 10,
         "oversample": 2, "dcblock_fc": 0.0, "n_state": 4}
CIRCUITS = ["bjt", "jfet", "hysteretic_fuzz"]
ARMS = [("standard_150", False, 150), ("varpro_60", True, 60), ("varpro_150", True, 150)]


def main() -> None:
    log = make_log("sota_varpro_circe3")
    out = Path("outputs/sota/varpro_circe3.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    log(f"VarPro on production CIRCE3 (nb2/L10/n_state4/OS2/dcblock_off): {[a[0] for a in ARMS]}")
    for key in CIRCUITS:
        sweep, test, kind = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        row: dict = {"kind": kind}
        for tag, vp, ep in ARMS:
            t = time.time()
            torch.manual_seed(0)
            m = CIRCE3(**MODEL, device="cuda")
            m.fit(tr, ts, TrainConfig(epochs=ep, lr=3e-3, seq_len=4096, batch_size=12,
                                      warmup=2048, seed=0, varpro=vp))
            row[tag] = {"held": held_esr(m, ts), "secs": time.time() - t, "epochs": ep}
            log(f"{key:16s} {tag:13s} held={row[tag]['held']:.4f} "
                f"({row[tag]['secs']:.0f}s, {ep}ep)")
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("=== VarPro vs standard on production CIRCE3 (held-ESR | wall-clock) ===")
    for key in CIRCUITS:
        for tag, _, _ in ARMS:
            r = results[key][tag]
            log(f"  {key:16s} {tag:13s} {r['held']:.4f}  {r['secs']:.0f}s")
    log("done")


if __name__ == "__main__":
    main()
