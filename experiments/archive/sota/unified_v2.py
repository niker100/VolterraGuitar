"""The multi-seed leaderboard run on the adopted v2 data: unified config, finals
training (b12/lr3/warm5 via fit_score defaults, collapse retry/fallback), seeds
(0, 7), all 9 circuits. This is the number that counts against the <0.005-on-every-
circuit goal after the data adoption (ts -47%, crossover -32%, jfet -16% in the A/B).

Run (background): uv run python -m experiments.sota.unified_v2
"""

from __future__ import annotations

from experiments.sota.harness import ALL, UNIFIED, run_campaign

if __name__ == "__main__":
    run_campaign("unified_v2", [{"label": "unified", **UNIFIED}], ALL,
                 seeds=(0, 7), epochs=150)
