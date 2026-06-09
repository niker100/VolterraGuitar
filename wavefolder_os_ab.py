"""Wavefolder aliasing frontier — a clean GPU-bound A/B on the oversampling factor.

The multi-fold wavefolder is the one circuit no uniform lever has cracked
(held-ESR ~0.18 at the shipping OS2 config; OS/depth/width/clip/mixed/epochs/EMA
all neutral). The standing hypothesis is a *real-time-bandwidth floor*: the folds
generate harmonics so high they alias back in-band even at 2x internal
oversampling. That hypothesis has NEVER been tested directly on the wavefolder —
the OS-factor rejection (4x "fails") was measured on the BJT, where 4x gave a
nonsensical held-ESR 0.32 (worse than 2x — physically backwards for an
anti-aliasing knob, i.e. almost certainly a config/training artifact, not a
fundamental result).

This decides it. For the wavefolder, train the production CIRCE3 (mixed block,
grad-clip, 150 ep, ch24/L9) at oversample in {2,3,4} and measure BOTH held-ESR and
CPU RTF (the binding real-time constraint). Two readings:

  * If OS4 lowers held-ESR meaningfully  -> the floor IS aliasing, and the problem
    is RTF engineering (claw back real-time with a smaller net at OS4 -> the
    OS4_ch16 arm tests exactly that).
  * If OS4 does NOT help                  -> the floor is representation/capacity,
    not aliasing, and the "bandwidth floor" framing is wrong.

Either way it is an informative, GPU-bound result. mixed block + grad_clip=1.0 are
the current CIRCE3 defaults. Logs to outputs/logs/wavefolder_os_ab.log; writes
outputs/wavefolder_os_ab.json.

Run: uv run python wavefolder_os_ab.py
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
from vguitar.realtime import measure_rtf

EPOCHS = 150
SWEEP = "wavefolder_edge_sweep"
TEST = "wavefolder_edge_test"
SEEDS = (0, 7)
# (label, oversample, channels, n_seeds)
CONFIGS = [
    ("OS2_ch24", 2, 24, 2),  # current shipping default — the baseline (~0.18)
    ("OS3_ch24", 3, 24, 1),  # intermediate diagnostic
    ("OS4_ch24", 4, 24, 2),  # does more bandwidth help? (aliasing test)
    ("OS4_ch16", 4, 16, 2),  # OS4 with a smaller net — claw back real-time RTF
]

LOG = Path("outputs/logs/wavefolder_os_ab.log")
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
    log(f"wavefolder OS A/B: {EPOCHS}ep mixed/grad-clip ch?/L9, oversample sweep")
    tr = Dataset.load(f"data/{SWEEP}.npz")
    ts = Dataset.load(f"data/{TEST}.npz")
    results: dict[str, dict] = {}
    for label, osf, ch, ns in CONFIGS:
        vals = []
        rtf = None
        params = None
        lat = None
        for seed in SEEDS[:ns]:
            t = time.time()
            torch.manual_seed(seed)
            m = CIRCE3(n_control=1, signal_idx=(0,), channels=ch, n_blocks=2, n_layers=9,
                       oversample=osf, device="cuda")  # mixed block + grad_clip=1.0 defaults
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=seed))
            h = held_esr(m, ts)
            vals.append(h)
            if rtf is None:
                # RTF on CPU (deployment target); m is already moved to CPU by fit().
                rtf = float(measure_rtf(m, sr=ts.sr, block=128)["rtf"])
                params = int(m.num_params())
                lat = int(m.latency_samples)
            log(f"{label:10s} seed{seed} held-ESR {h:.4f} block_act={m.block_act} "
                f"params={m.num_params()} ({time.time()-t:.0f}s)")
        mean = float(np.mean(vals))
        results[label] = {"oversample": osf, "channels": ch, "held": mean, "seeds": vals,
                          "params": params, "rtf": rtf, "latency": lat}
        rt = "REAL-TIME" if (rtf or 0) > 1.0 else "NOT-real-time"
        log(f"  -> {label}: held {mean:.4f}  params {params}  RTF {rtf:.2f}x ({rt})  lat {lat}")
    Path("outputs/wavefolder_os_ab.json").write_text(json.dumps(results, indent=2))
    log("=== WAVEFOLDER OVERSAMPLING A/B (held-ESR / RTF) ===")
    base = results["OS2_ch24"]["held"]
    for label, _, _, _ in CONFIGS:
        r = results[label]
        delta = 100.0 * (r["held"] - base) / base
        log(f"  {label:10s} held {r['held']:.4f} ({delta:+.0f}% vs OS2)  "
            f"RTF {r['rtf']:.2f}x  params {r['params']}")
    log("wrote outputs/wavefolder_os_ab.json")


if __name__ == "__main__":
    main()
