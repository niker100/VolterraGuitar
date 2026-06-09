"""Unified config standard vs VarPro across the FULL suite — does the closed-form readout
(which sharpened jfet 0.0048->0.0010 and trains ~2.5x faster) pull more circuits under
0.005? Config = the unified production model: dcblock_off + nb2/L10 + n_state=4 + OS2.
Also re-establishes the clean unified baseline (unified.json was cleared). Seed 0.

Run (background): uv run python -m experiments.sota.unified_varpro
"""

from __future__ import annotations

from experiments.sota.harness import ALL, run_campaign

BASE = {"channels": 24, "n_blocks": 2, "n_layers": 10, "oversample": 2,
        "dcblock_fc": 0.0, "n_state": 4}
CONFIGS = [
    {"label": "standard", **BASE},
    {"label": "varpro", **BASE, "varpro": True},
]

if __name__ == "__main__":
    run_campaign("unified_varpro", CONFIGS, ALL, seeds=(0,), epochs=150)
