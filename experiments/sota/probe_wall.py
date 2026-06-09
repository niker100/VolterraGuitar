"""Campaign 1b — wall diagnostic: what actually moves a hard circuit's held-ESR?

Campaign 1 ruled out the cheap structural priors (rect corners, Fourier head) AND
capacity — all inert on the wall (ch40 even destabilized hard_clipper to 0.96). So
the bottleneck is NOT expressivity. The two remaining first-principles suspects:

* **Aliasing** — a pointwise nonlinearity + decimation aliases the (huge) harmonics
  of a sharp transfer curve back into band; the band-limited target then can't be
  matched. Lever: more internal oversampling (``oversample=4``).
* **Loss weighting** — plain ESR is energy-dominated and barely sees the low-energy
  upper harmonics where the hard-circuit residual lives. Levers: stronger phase-aware
  pre-emphasis (``preemph_order=2``) and a multi-resolution STFT magnitude term
  (``stft_weight``), which directly weights the harmonic band.

Battery = the two clearest above-target hard circuits (asym_clipper 0.052,
hard_clipper 0.089) — excludes wavefolder (separately studied, OS already known
weak there) and the divergence-prone ch40. Single seed, uniform configs. Whatever
moves these gets validated on the full hard suite + wavefolder next.

Run (background): uv run python -m experiments.sota.probe_wall
"""

from __future__ import annotations

from experiments.sota.harness import run_campaign

BASE = {"channels": 24, "n_blocks": 2, "n_layers": 9}
CONFIGS = [
    {"label": "base", **BASE, "oversample": 2},
    {"label": "os4", **BASE, "oversample": 4},
    {"label": "preemph2", **BASE, "oversample": 2, "preemph_order": 2},
    {"label": "stft01", **BASE, "oversample": 2, "stft_weight": 0.1},
    {"label": "stft03_pe2", **BASE, "oversample": 2, "stft_weight": 0.3, "preemph_order": 2},
]
BATTERY = ["asym_clipper", "hard_clipper"]

if __name__ == "__main__":
    run_campaign("probe_wall", CONFIGS, BATTERY, seeds=(0,), epochs=150)
