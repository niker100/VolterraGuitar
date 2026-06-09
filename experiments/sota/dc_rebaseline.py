"""Campaign 2 — the DC re-baseline (rank-1 fix: kill the train/eval DC mismatch).

VERIFIED finding: CIRCE3 trains its loss on the raw network output but applies a
1 Hz DC-blocker only at inference (process_block). For circuits whose target carries
real, level-dependent output DC (asymmetric clip / rectify / fold), a *perfect* model
is floored — measured perfect-model ESR through the blocker: asym_clipper 0.0518
(~99% of its 0.0524 score!), wavefolder 0.1153 (~66% of 0.175), vs ~0 for the
symmetric circuits. The DC is genuine signal content (y_mean +0.05 on asym), so the
faithful fix is to NOT discard it: train AND eval on the raw output (dcblock_fc=0).
This is uniform (same flag for all circuits) and removes an inference stage.

This re-baselines all 9 circuits with current code, blocker OFF vs ON, single seed,
to (a) re-establish the true current-code baseline (final_numbers.json is stale at
85k params; current mixed block is 54k) and (b) measure the DC fix per circuit. The
winner (expected: dcblock_fc=0) becomes the new uniform default; circuits still over
0.005 (hard_clipper, crossover, hysteretic, wavefolder) go to the targeted structural
campaigns (IIR memory, learnable corners, depth).

Run (background): uv run python -m experiments.sota.dc_rebaseline
"""

from __future__ import annotations

from experiments.sota.harness import ALL, run_campaign

BASE = {"channels": 24, "n_blocks": 2, "n_layers": 9, "oversample": 2}
CONFIGS = [
    {"label": "dcblock_on", **BASE, "dcblock_fc": 1.0},   # current default (control)
    {"label": "dcblock_off", **BASE, "dcblock_fc": 0.0},  # rank-1 fix
]

if __name__ == "__main__":
    run_campaign("dc_rebaseline", CONFIGS, ALL, seeds=(0,), epochs=150)
