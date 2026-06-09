"""Deep-dive on the spectral-zoo leads: are the circuit-split wins robust + general?

The zoo found two cheap variants that each beat the hybrid on ONE circuit:
stft_mix (best bjt, 24.5k params) and fft_longfir (best jfet, ~85x rtf), plus a
tiny wavelet that crashed (now fixed). This re-runs them MULTI-SEED on bjt + jfet +
tube_screamer (a third formant circuit, tests generalization) against the hybrid
baseline, to see if (a) stft_mix's bjt win + jfet weakness are real or seed noise,
(b) either lead generalizes to tube_screamer, (c) the fixed wavelet is competitive.

Run: uv run python -m experiments.spectral_leads
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import load_pair, make_log
from experiments.spectral_probe import _eval
from experiments.spectral_zoo_run import _build
from vguitar.config import TrainConfig

EPOCHS = 150
SEEDS = (0, 7)
MODELS = ["hybrid", "stft_mix", "fft_longfir", "wavelet"]
CIRCUITS = [("bjt", "bjt_bench_sweep", "bjt_bench_fp_test"),
            ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test"),
            ("tube_screamer", "tube_screamer_bench_sweep", "tube_screamer_bench_fp_test")]
log = make_log("spectral_leads")


def main() -> None:
    log(f"spectral leads deep-dive: {MODELS} x {[c[0] for c in CIRCUITS]} x {len(SEEDS)} seeds, "
        f"{EPOCHS}ep OS1")
    results: dict[str, dict] = {}
    for tag in MODELS:
        results[tag] = {}
        for key, sweep_nm, test_nm in CIRCUITS:
            tr, ts = load_pair(sweep_nm, test_nm)
            esrs, bands = [], []
            for seed in SEEDS:
                t = time.time()
                torch.manual_seed(seed)
                try:
                    m = _build(tag, "cuda")
                    m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                              batch_size=12, warmup=2048, seed=seed))
                    esr, band = _eval(m, ts)
                    esrs.append(esr)
                    bands.append(band)
                    log(f"{tag:12s} {key:13s} seed{seed} ESR {esr:.4f} band>4k {band:.4f} "
                        f"({time.time()-t:.0f}s)")
                except Exception as exc:
                    log(f"{tag:12s} {key:13s} seed{seed} FAILED: {exc}")
            results[tag][key] = {"esr": esrs, "band": bands,
                                 "esr_mean": float(np.mean(esrs)) if esrs else float("nan"),
                                 "band_mean": float(np.mean(bands)) if bands else float("nan")}
    Path("outputs/spectral_leads.json").write_text(json.dumps(results, indent=2))
    log("=== LEADS DEEP-DIVE (mean held-ESR over seeds) ===")
    for key, _, _ in CIRCUITS:
        line = "  ".join(f"{t} {results[t][key]['esr_mean']:.4f}" for t in MODELS)
        log(f"  {key:13s}  {line}")
    log("wrote outputs/spectral_leads.json")


if __name__ == "__main__":
    main()
