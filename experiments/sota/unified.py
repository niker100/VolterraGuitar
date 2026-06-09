"""Campaign 7 — the UNIFIED config: stack the validated levers, measure multi-seed.

config = dcblock_off + depth (nb2/L10, ~2x receptive field) + IIR (n_state=4), OS2, ch24.
The convergence run: how many of the 9 circuits cross 0.005 once the validated levers are
stacked? Evidence so far (single seed): DC fix cracked asym (0.0007); depth cracked jfet
(0.0094->0.0020) and ~halved hard_clipper; IIR helped the memory/dead-zone circuits
(hysteretic -29%, crossover -18%). nb2/L10 keeps bjt under 0.005 (nb1/L11 didn't).

Full 9-circuit suite, seeds (0, 7) for robustness, held-ESR + RTF. The smooth circuits
double as regression guards. Whatever still resists (expected: hard_clipper, wavefolder)
goes to the grey-box waveshaper / complementary-metric stage.

Run (background): uv run python -m experiments.sota.unified
"""

from __future__ import annotations

from experiments.sota.harness import ALL, run_campaign

CONFIG = [
    {"label": "unified", "channels": 24, "n_blocks": 2, "n_layers": 10,
     "oversample": 2, "dcblock_fc": 0.0, "n_state": 4},
]

if __name__ == "__main__":
    run_campaign("unified", CONFIG, ALL, seeds=(0, 7), epochs=150)
