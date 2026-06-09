"""Definitive shipping test: does block_act='mixed' help in the PRODUCTION CIRCE3
pipeline (oversample=2, ch24, L9, real .fit + .process eval), where the controlled
no-OS A/B said the activation is only a marginal, circuit-dependent lever?

Also probes the cheaper rival lever surfaced by the A/B: a plain WIDTH bump
(channels 24 -> 36) on the standard gated block. The radical-search 'baseline
~0.45-0.55' on crossover turned out to be a thin/under-fit config; a fairly sized
gated TCN already reaches ~0.07, so width may matter more than the activation.

Compares, per hard circuit (crossover, wavefolder) + one smooth (bjt) control:
  gated@ch24  |  mixed@ch24  |  gated@ch36
all oversample=2, L9, short epochs, real held eval (latency-compensated, g-scaled
via CIRCE3's own input-scaling). Logs to outputs/logs/radical_mixedact_prod.log.

Run: uv run python radical_mixedact_prod.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from vguitar import metrics as M
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3, _segments

EPOCHS = 80
CIRCUITS = [
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test"),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test"),
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test"),
]
CONFIGS = [
    ("gated_ch24", dict(channels=24, block_act="gated")),
    ("mixed_ch24", dict(channels=24, block_act="mixed")),
    ("gated_ch36", dict(channels=36, block_act="gated")),
]

LOG = Path("outputs/logs/radical_mixedact_prod.log")
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
    if lat > 0:  # polyphase group delay: shift prediction left, drop the tail
        pc = pc[lat:]
        yc = yc[: len(pc)]
    w = 2048
    return float(M.esr(yc[w:], pc[w:]))


def main() -> None:
    log(f"epochs={EPOCHS} configs={[c[0] for c in CONFIGS]}")
    tcfg = TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096, batch_size=12, warmup=2048)
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        results[key] = {}
        for cfg_name, kw in CONFIGS:
            t = time.time()
            m = CIRCE3(n_control=1, signal_idx=(0,), n_blocks=2, n_layers=9,
                       oversample=2, device="cuda", **kw)
            npar = m.num_params()
            m.fit(tr, ts, tcfg)
            held = held_esr(m, ts)
            results[key][cfg_name] = {"held": held, "params": npar}
            log(f"{key:11s} {cfg_name:11s} held-ESR {held:.4f}  ({npar/1000:.0f}k, {time.time()-t:.0f}s)")
        r = results[key]
        log(f"  -> {key}: gated24 {r['gated_ch24']['held']:.4f}  "
            f"mixed24 {r['mixed_ch24']['held']:.4f}  gated36 {r['gated_ch36']['held']:.4f}")
    Path("outputs/radical_mixedact_prod.json").write_text(json.dumps(results, indent=2))
    log("=== PRODUCTION-PIPELINE VERDICT (held-ESR, OS2) ===")
    for key, _, _ in CIRCUITS:
        r = results[key]
        log(f"  {key:11s} gated24 {r['gated_ch24']['held']:.4f} | "
            f"mixed24 {r['mixed_ch24']['held']:.4f} | gated36 {r['gated_ch36']['held']:.4f}")
    log("wrote outputs/radical_mixedact_prod.json")


if __name__ == "__main__":
    main()
