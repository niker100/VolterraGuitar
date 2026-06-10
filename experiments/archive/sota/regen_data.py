"""Data-quality lever — regenerate TRAINING sweeps with ~3x more data, test unchanged.

The data-quality audit found the *_edge_sweep training sets carry 41% less data per
drive (~2.9 s) than the smooth *_bench_sweep (4.0 s) — a generation-script
inconsistency, not a design choice — and the user flagged data quality as historically
high-impact. This regenerates each sweep at seg_dur_s=8.0 (same 4 drives, same rich
excitation in signals.py) to ``data/<sweep>_v2.npz``. A follow-up campaign trains on the
_v2 sweeps and scores on the UNCHANGED ``<c>_edge_test`` / ``<c>_bench_fp_test``, so any
held-ESR change isolates the data-volume lever (test content is identical).

CPU/ngspice-bound -> runs in parallel with GPU training (the 4090 stays free). Robust
per-circuit (one failure doesn't abort the run); logs timestamped progress for stall
detection. Run (background): uv run python -m experiments.sota.regen_data
"""

from __future__ import annotations

import time
from pathlib import Path

from experiments.common import make_log
from vguitar.circuits import get_circuit
from vguitar.config import Config
from vguitar.spice.runner import make_drive_dataset

SEG_DUR = 8.0
DRIVE_FACTORS = (0.25, 0.5, 1.0, 2.0)
# (circuit registry name, sweep dataset basename) — the unsolved / near-miss circuits.
# asym_clipper + bjt are already < 0.005; wavefolder is a spectral-bias wall data won't fix.
TARGETS = [
    ("jfet", "jfet_bench_sweep"),
    ("tube_screamer", "tube_screamer_bench_sweep"),
    ("fullwave_rectifier", "fullwave_rectifier_edge_sweep"),
    ("crossover_classb", "crossover_classb_edge_sweep"),
    ("hard_clipper", "hard_clipper_edge_sweep"),
    ("hysteretic_fuzz", "hysteretic_fuzz_edge_sweep"),
]


def main() -> None:
    log = make_log("sota_regen_data")
    cfg = Config()
    log(f"regen sweeps seg_dur_s={SEG_DUR} drives={DRIVE_FACTORS}x nominal -> *_v2.npz "
        f"(test sets UNCHANGED)")
    for name, sweep in TARGETS:
        t = time.time()
        try:
            circ = get_circuit(name)
            g = float(circ.nominal_drive_v)
            drives = [round(g * f, 5) for f in DRIVE_FACTORS]
            ds = make_drive_dataset(circ, drives, cfg, seg_dur_s=SEG_DUR, seed=0)
            out = Path(f"data/{sweep}_v2.npz")
            ds.save(out)
            log(f"{name:18s} drives={drives} samples={len(ds)} ({len(ds)/cfg.data.sr:.1f}s) "
                f"-> {out.name} ({time.time()-t:.0f}s)")
        except Exception as exc:
            log(f"{name:18s} FAILED ({time.time()-t:.0f}s): {exc}")
    log("done")


if __name__ == "__main__":
    main()
