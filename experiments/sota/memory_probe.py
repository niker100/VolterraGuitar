"""Memory probe — push the IIR-state lever where the residual IS memory.

hysteretic_fuzz (0.0139) has bias-recovery tau = 22-440 ms; the unified config gives
it n_state=4 one-poles initialized 5-500 ms and a 93 ms training window (seq 4096).
Three orthogonal extensions, plus the combination, each against the unified control:

  state8       more poles (4 -> 8): denser tau coverage, +~200 params
  tau2s        tau init 5 ms - 2 s: poles where the slowest dynamics live
  state8_tau2s both
  seq8192      186 ms training window (the loss finally SEES slow dynamics across a
               window; batch pinned 48 + lr 6e-3 — the validated half-batch combo —
               so GPU memory stays bounded regardless of harness defaults)

Battery: hysteretic_fuzz (target), crossover (dead-zone, IIR helped at prototype),
bjt (smooth regression guard — zero-init keeps extra poles inert where unneeded).

Run (background): uv run python -m experiments.sota.memory_probe
"""

from __future__ import annotations

from experiments.sota.harness import SCREEN, UNIFIED, run_campaign

CONFIGS = [
    {"label": "base", **UNIFIED},
    {"label": "state8", **UNIFIED, "n_state": 8},
    {"label": "tau2s", **UNIFIED, "state_tau_s": (5e-3, 2.0)},
    {"label": "state8_tau2s", **UNIFIED, "n_state": 8, "state_tau_s": (5e-3, 2.0)},
    {"label": "seq8192", **UNIFIED, "seq_len": 8192, "batch_size": 48, "lr": 6e-3},
]
CIRC = ["hysteretic_fuzz", "crossover", "bjt"]

if __name__ == "__main__":
    run_campaign("memory_probe", CONFIGS, CIRC, seeds=(0,), epochs=150, **SCREEN)
