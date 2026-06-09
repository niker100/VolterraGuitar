"""Campaign 5 — depth / receptive-field sweep (rank-4). Corners were inert (Campaign 4),
so the only lever that has ever moved hard_clipper is DEPTH (an un-confirmed lead:
deep ch24/L11 dropped hard_clipper 0.094->0.041, 2.3x). Width/OS/loss/corners are all
dead on it. This confirms it with a clean, default-window comparison.

Both deeper configs are chosen so the receptive field still fits the training window
(warmup 4096 internal at OS2), avoiding RF starvation and keeping seq_len fixed so the
comparison isolates depth (no window confound):
  nb2/L9  RF 2045  (production control)
  nb1/L11 RF 4095  (2x RF, FEWER layers -> also cheaper)
  nb2/L10 RF 4093  (2x RF, more layers)
dcblock_off (new default), OS2. Battery: the sharp-knee + dead-zone + near-miss set
(hard_clipper, crossover, fullwave, jfet) + bjt guard. Excludes hysteretic (IIR's job),
wavefolder (depth made the fold worse), asym (already < 0.005). Reports RTF (depth
trades it). Single seed; winners confirmed multi-seed in the unified campaign.

Run (background): uv run python -m experiments.sota.depth_sweep
"""

from __future__ import annotations

from experiments.sota.harness import run_campaign

BASE = {"channels": 24, "oversample": 2, "dcblock_fc": 0.0}
CONFIGS = [
    {"label": "nb2_L9", **BASE, "n_blocks": 2, "n_layers": 9},
    {"label": "nb1_L11", **BASE, "n_blocks": 1, "n_layers": 11},
    {"label": "nb2_L10", **BASE, "n_blocks": 2, "n_layers": 10},
]
BATTERY = ["hard_clipper", "crossover", "fullwave_rectifier", "jfet", "bjt"]

if __name__ == "__main__":
    run_campaign("depth_sweep", CONFIGS, BATTERY, seeds=(0,), epochs=150)
