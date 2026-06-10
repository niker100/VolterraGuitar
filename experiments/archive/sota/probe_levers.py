"""Campaign 1 — lever probe: which UNIFORM structural priors crack the hard wall?

The shipping CIRCE3 reaches held-ESR < 0.005 only on bjt (0.0047). The wall is
the discontinuity circuits — wavefolder 0.175, hard_clipper 0.092, asym_clipper
0.052 — whose sharp transfer corners + high harmonics a smooth TCN cannot
synthesize and partly aliases. This probes two cheap, RTF-light structural priors
that target exactly that, both already in CIRCE3 and both off by default:

* ``rect_thr`` — a fixed bank of rectified input features (relu(x-t), relu(-x-t),
  abs(x)) giving the net EXACT slope discontinuities at fixed circuit volts (diode
  knee / dead-zone / clip edge) to combine linearly, instead of approximating a
  kink with a smooth Lipschitz map.
* ``out_shaper='fourier'`` — a residual sine waveshaper head (y = o + sum_k c_k
  sin(k w o)), the explicit multi-fold primitive the wavefolder literally is.

Both are zero-/identity-init (no regression risk at init) and the SAME bank is
used for every circuit (no per-circuit tailoring). A third config adds capacity
(channels 24->40) to see if the wall is representational. Battery = the 3 hardest
+ bjt as a smooth regression guard. Single seed (fast triage); the winner gets a
full multi-seed sweep in Campaign 2.

Run (background): uv run python -m experiments.sota.probe_levers
"""

from __future__ import annotations

from experiments.sota.harness import run_campaign

#: fixed-volt corner bank: small-signal curvature -> diode knee -> hard-clip edge.
RECT = (0.1, 0.3, 0.6, 1.0)

CONFIGS = [
    {"label": "base", "channels": 24, "n_blocks": 2, "n_layers": 9, "oversample": 2},
    {"label": "rect_shaper", "channels": 24, "n_blocks": 2, "n_layers": 9, "oversample": 2,
     "rect_thr": RECT, "out_shaper": "fourier"},
    {"label": "rect_shaper_big", "channels": 40, "n_blocks": 2, "n_layers": 9, "oversample": 2,
     "rect_thr": RECT, "out_shaper": "fourier"},
]
BATTERY = ["wavefolder", "hard_clipper", "asym_clipper", "bjt"]

if __name__ == "__main__":
    run_campaign("probe_levers", CONFIGS, BATTERY, seeds=(0,), epochs=150)
