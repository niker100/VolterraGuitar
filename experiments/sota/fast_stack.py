"""The quick-learner stack: VarPro x big-batch x LR-warmup on the smooth circuits.

Each lever is individually validated: VarPro reaches the plateau in ~3x fewer epochs
(closed-form readout = always-optimal head from step 1) and sharpens smooth circuits
(jfet 0.0010 vs 0.0045); b96 cuts wall-clock ~1.4-2.7x; the 5-epoch LR warmup removes
the big-batch early-overshoot collapse (except on hard_clipper, which is excluded —
VarPro breaks discontinuity circuits anyway). This measures the COMBINED stack against
the unified_varpro standard-150ep/b12 reference (REF below — not retrained; GPU time
goes to new information only). Target: reference ESR at ~4-7x less wall-clock = the
iteration config for all future smooth-circuit work.

Run (background): uv run python -m experiments.sota.fast_stack
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.common import make_log
from experiments.sota.harness import CIRCUITS as HC
from experiments.sota.harness import UNIFIED, fit_score
from vguitar.data import Dataset

#: the standard-150ep/b12 leaderboard numbers (outputs/sota/unified_varpro.json,
#: config "standard", seed 0) — (held-ESR, secs); secs carry cross-process
#: contention noise, so speed ratios are indicative, not exact.
REF = {"jfet": (0.0045, 326.0), "bjt": (0.0044, 358.0), "tube_screamer": (0.0026, 309.0)}

# (label, epochs, training overrides folded into the config dict)
ARMS = [
    ("vp60_b96_w5", 60, {"varpro": True, "batch_size": 96, "lr": 9e-3}),   # the stack
    ("vp60_b12", 60, {"varpro": True}),                                    # varpro alone
    ("vp150_b96_w5", 150, {"varpro": True, "batch_size": 96, "lr": 9e-3}),  # ceiling
]
CIRC = list(REF)


def main() -> None:
    log = make_log("sota_fast_stack")
    out = Path("outputs/sota/fast_stack.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = json.loads(out.read_text()) if out.exists() else {}
    log(f"fast-stack arms={[a[0] for a in ARMS]} circuits={CIRC}")
    for key in CIRC:
        sweep, test, kind = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        row: dict = results.setdefault(key, {"kind": kind})
        for label, ep, over in ARMS:
            if row.get(label, {}).get("held") is not None:  # resume
                continue
            t = time.time()
            try:
                _, r = fit_score(tr, ts, UNIFIED | over, seed=0, epochs=ep)
                row[label] = r | {"epochs": ep}
                ref_esr, ref_s = REF[key]
                log(f"{key:14s} {label:14s} held={r['held']:.4f} ({r['secs']:.0f}s) "
                    f"[ref {ref_esr:.4f}/{ref_s:.0f}s -> {ref_s / r['secs']:.1f}x]")
            except Exception as exc:
                row[label] = {"held": None, "secs": time.time() - t,
                              "error": str(exc)[:120]}
                log(f"{key:14s} {label:14s} FAILED: {str(exc)[:100]}")
            out.write_text(json.dumps(results, indent=2))
    log("done")


if __name__ == "__main__":
    main()
