"""Branch B swing #1 — does a learned periodic (Fourier) waveshaper head crack the
wavefolder wall (~0.18)?

The OS diagnostic (wavefolder_os_ab.py) showed more internal bandwidth does NOT
help the wavefolder (OS2 0.190 -> OS3 0.186 -> OS4 0.216, strictly worse) -> the
wall is representation/inductive-bias, not self-aliasing. A smooth TCN cannot
synthesize a multi-fold; this gives it the fold as an explicit primitive:

    y = o + sum_{k=1..K} c_k sin(k . w . o)         (CIRCE3 out_shaper="fourier")

a learned residual Fourier waveshaper on the network's scalar pre-output o
(zero-init c_k => identity, so it's a strict superset of the baseline; pointwise =>
streaming-exact + input-scaling-preserving).

Honest A/B at the shipping config (mixed/grad-clip/OS2/ch24/L9/150ep):
  * wavefolder  (the target — 2 seeds): does the head break 0.18?
  * bjt, jfet   (smooth regression guards — 1 seed): a wavefolder win that wrecks
    the smooth showcase is not a win.
Reports held-ESR + CPU RTF per (circuit, variant). Logs to
outputs/logs/wavefolder_shaper_ab.log; writes outputs/wavefolder_shaper_ab.json.

Run: uv run python wavefolder_shaper_ab.py
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
# (key, sweep, test, n_seeds, role)
CIRCUITS = [
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", 2, "target"),
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", 1, "guard"),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", 1, "guard"),
]
SEEDS = (0, 7)
# (label, out_shaper, shaper_k)
VARIANTS = [
    ("baseline", "none", 0),
    ("fourier8", "fourier", 8),
]

LOG = Path("outputs/logs/wavefolder_shaper_ab.log")
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
    log(f"Fourier-shaper A/B: {EPOCHS}ep mixed/grad-clip/OS2 ch24/L9")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, nseeds, role in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        results[key] = {"role": role, "variants": {}}
        for label, shaper, sk in VARIANTS:
            vals = []
            rtf = None
            params = None
            for seed in SEEDS[:nseeds]:
                t = time.time()
                torch.manual_seed(seed)
                m = CIRCE3(
                    n_control=1,
                    signal_idx=(0,),
                    channels=24,
                    n_blocks=2,
                    n_layers=9,
                    oversample=2,
                    out_shaper=shaper,
                    shaper_k=sk or 8,
                    device="cuda",
                )
                m.fit(
                    tr,
                    ts,
                    TrainConfig(
                        epochs=EPOCHS, lr=3e-3, seq_len=4096, batch_size=12, warmup=2048, seed=seed
                    ),
                )
                h = held_esr(m, ts)
                vals.append(h)
                if rtf is None:
                    rtf = float(measure_rtf(m, sr=ts.sr, block=128)["rtf"])
                    params = int(m.num_params())
                log(
                    f"{key:11s} {label:9s} seed{seed} held {h:.4f} "
                    f"params={m.num_params()} ({time.time() - t:.0f}s)"
                )
            mean = float(np.mean(vals))
            results[key]["variants"][label] = {
                "held": mean,
                "seeds": vals,
                "params": params,
                "rtf": rtf,
            }
            log(f"  -> {key} [{label}]: held {mean:.4f}  RTF {rtf:.2f}x  params {params}")
    Path("outputs/wavefolder_shaper_ab.json").write_text(json.dumps(results, indent=2))
    log("=== FOURIER-SHAPER A/B (held-ESR; baseline -> fourier8) ===")
    for key, _, _, _, role in CIRCUITS:
        b = results[key]["variants"]["baseline"]["held"]
        f = results[key]["variants"]["fourier8"]["held"]
        delta = 100.0 * (f - b) / b
        verdict = "SHAPER WINS" if f < b else "baseline"
        log(
            f"  {key:11s} [{role:6s}] {b:.4f} -> {f:.4f} ({delta:+.0f}%)  {verdict}  "
            f"RTF {results[key]['variants']['fourier8']['rtf']:.2f}x"
        )
    log("wrote outputs/wavefolder_shaper_ab.json")


if __name__ == "__main__":
    main()
