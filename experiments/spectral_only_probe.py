"""What if we used ONLY the FFT branch (no time-domain head)?

The spectral branch applies a magnitude-conditioned complex gain G·X to the
EXISTING spectrum — a time-varying linear filter. It can reshape frequency content
that's present but cannot synthesize harmonics at bins where the input has no
energy, which is exactly what a distortion nonlinearity must do. Prediction:
spectral-only fails on overall ESR (worst on strongly-nonlinear bjt), confirming
the hybrid's division of labour (time head = harmonic generation, spectral branch
= linear formant shaping).

3-way A/B on bjt (strong distortion) + jfet (mild): time-only vs spectral-only vs
hybrid, held-ESR + band>4k, OS1, same harness. Run:
    uv run python -m experiments.spectral_only_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from experiments.common import load_pair, make_log
from experiments.spectral_probe import SpectralHybrid, _eval
from vguitar.config import TrainConfig

EPOCHS = 150
CIRCUITS = [("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "strong"),
            ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "mild")]
# (label, use_time, use_spectral, spec_hidden)
MODES = [("time_only", True, False, 16),
         ("spectral_only", False, True, 16),
         ("hybrid", True, True, 16)]
log = make_log("spectral_only_probe")


def main() -> None:
    log(f"FFT-only question: time-only vs spectral-only vs hybrid, {EPOCHS}ep OS1, hidden=16")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind in CIRCUITS:
        tr, ts = load_pair(sweep_nm, test_nm)
        results[key] = {"kind": kind}
        for label, ut, us, hid in MODES:
            t = time.time()
            torch.manual_seed(0)
            m = SpectralHybrid(channels=24, n_layers=9, use_time=ut, use_spectral=us,
                               spec_hidden=hid, device="cuda")
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=0))
            esr, band = _eval(m, ts)
            results[key][label] = {"esr": esr, "band": band, "params": m.num_params()}
            log(f"{key:5s} {label:13s} ESR {esr:.4f} band>4k {band:.4f} "
                f"params={m.num_params()} ({time.time()-t:.0f}s)")
    Path("outputs/spectral_only_probe.json").write_text(json.dumps(results, indent=2))
    log("=== FFT-ONLY vs TIME-ONLY vs HYBRID (held-ESR / band>4k) ===")
    for key, _, _, kind in CIRCUITS:
        r = results[key]
        log(f"  {key:5s} [{kind:6s}] "
            f"time {r['time_only']['esr']:.4f} | spectral {r['spectral_only']['esr']:.4f} | "
            f"hybrid {r['hybrid']['esr']:.4f}   (band>4k "
            f"{r['time_only']['band']:.3f}/{r['spectral_only']['band']:.3f}/"
            f"{r['hybrid']['band']:.3f})")
    log("wrote outputs/spectral_only_probe.json")


if __name__ == "__main__":
    main()
