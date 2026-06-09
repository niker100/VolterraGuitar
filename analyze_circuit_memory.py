"""How much MEMORY does each circuit actually need? A static (memoryless) map
y=f(x) vs a short-FIR-then-static map quantifies the intrinsic temporal-memory
requirement per circuit — the axis that decides whether a deep large-RF TCN
(good for memory) or a shallow net (good for near-static corners) is the right
UNIFORM inductive bias.

For each circuit's held test segment (input g*x):
  * static     : best instantaneous map (fine-binned median lookup x->y), ESR
  * fir<k>     : least-squares FIR of length k on x, then binned static map, ESR
A low static-ESR => near-memoryless (dead-zone/clipper/folder); a big drop from
static->fir => the circuit needs memory (reactive filtering, e.g. tube_screamer,
hysteresis). This is circuit ANALYSIS (not a model change) — it explains the
smooth-vs-static tension a single uniform model must straddle.

Run: uv run python analyze_circuit_memory.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vguitar import metrics as M
from vguitar.data import Dataset
from vguitar.models.circe3 import _segments

CIRCUITS = [
    ("bjt", "bjt_bench_fp_test", "smooth"),
    ("jfet", "jfet_bench_fp_test", "smooth"),
    ("tube_screamer", "tube_screamer_bench_fp_test", "smooth"),
    ("crossover", "crossover_classb_edge_test", "hard"),
    ("wavefolder", "wavefolder_edge_test", "hard"),
    ("asym_clipper", "asym_clipper_edge_test", "hard"),
    ("hard_clipper", "hard_clipper_edge_test", "hard"),
    ("fullwave_rectifier", "fullwave_rectifier_edge_test", "hard"),
    ("hysteretic_fuzz", "hysteretic_fuzz_edge_test", "hard"),
]


def static_esr(x: np.ndarray, y: np.ndarray, nbins: int = 400) -> float:
    """ESR of the best memoryless map: bin x, predict per-bin mean of y."""
    lo, hi = np.percentile(x, 0.1), np.percentile(x, 99.9)
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    sums = np.bincount(idx, weights=y, minlength=nbins)
    cnts = np.bincount(idx, minlength=nbins).astype(np.float64)
    means = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
    pred = means[idx].astype(np.float32)
    return float(M.esr(y, pred))


def fir_static_esr(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Least-squares causal FIR(k) on x to predict y, then a binned static map on
    the FIR output (captures linear memory + a static nonlinearity)."""
    n = len(x)
    X = np.zeros((n, k), np.float32)
    for i in range(k):
        X[i:, i] = x[: n - i]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    z = (X @ coef).astype(np.float32)
    return static_esr(z, y)


def main() -> None:
    rows = {}
    for key, test_nm, kind in CIRCUITS:
        ts = Dataset.load(f"data/{test_nm}.npz")
        s, e = _segments(ts.controls)[0]
        g = float(ts.controls[s, 0])
        x = (ts.x[s:e] * g).astype(np.float32)
        y = np.ascontiguousarray(ts.y[s:e], np.float32)
        st = static_esr(x, y)
        f8 = fir_static_esr(x, y, 8)
        f64 = fir_static_esr(x, y, 64)
        rows[key] = {"kind": kind, "static": st, "fir8": f8, "fir64": f64,
                     "mem_gain": st / max(f64, 1e-9)}
        print(f"{key:18s} [{kind:6s}] static {st:.4f}  fir8 {f8:.4f}  fir64 {f64:.4f}  "
              f"static/fir64 {st/max(f64,1e-9):.1f}x")
    Path("outputs/circuit_memory.json").write_text(json.dumps(rows, indent=2))
    print("\nINTERPRETATION:")
    print("  low static-ESR  -> near-memoryless (a shallow net suffices; deep RF wasted)")
    print("  big static->fir64 drop -> needs linear memory (deep/large-RF helps)")
    print("wrote outputs/circuit_memory.json")


if __name__ == "__main__":
    main()
