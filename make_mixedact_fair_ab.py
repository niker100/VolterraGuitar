"""Fair-baseline A/B: mixed activations vs the standard gated TCN, at EQUAL
capacity (~100k params) in an identical harness (same windows / loss / epochs /
grad-clip), the ONLY difference being the block activation.

The radical-architecture search reported mixed activations beating CIRCE3 "5-9x"
on the crossover dead-zone. That margin was against a *weak* baseline: a fairly
sized gated TCN already reaches ~0.07 on crossover (not 0.45-0.55). Under a fair
baseline the real picture is honest and modest — mixed-act wins a little on some
circuits, ties on others, and REGRESSES on the wavefolder. Reads
outputs/radical_generalize_ab.json.

Run: uv run python make_mixedact_fair_ab.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vguitar import plotting as plot

ORDER = ["bjt", "jfet", "tube_screamer", "crossover", "wavefolder", "asym_clipper"]
PRETTY = {"bjt": "BJT", "jfet": "JFET", "tube_screamer": "Tube\nScreamer",
          "crossover": "crossover\n(dead-zone)", "wavefolder": "wavefolder\n(multi-fold)",
          "asym_clipper": "asym\nclipper"}


def main() -> None:
    res = json.loads(Path("outputs/radical_generalize_ab.json").read_text())
    plot.apply_style()
    keys = [k for k in ORDER if k in res]
    gated = [float(np.mean(res[k]["gated"])) for k in keys]
    mixed = [float(np.mean(res[k]["mixed"])) for k in keys]
    kinds = [res[k]["kind"] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    xs = np.arange(len(keys))
    w = 0.38
    cg, cm = plot.OKABE_ITO["gray"], plot.OKABE_ITO["purple"]
    ax.bar(xs - w / 2, gated, w, label="gated TCN (standard CIRCE3 block)", color=cg,
           edgecolor="white", lw=0.6)
    ax.bar(xs + w / 2, mixed, w, label="mixed activations (tanh·gelu·relu·abs·snake)",
           color=cm, edgecolor="white", lw=0.6)

    for i, (g, m) in enumerate(zip(gated, mixed, strict=True)):
        d = (m - g) / g * 100.0
        verdict = "WIN" if m < g * 0.97 else ("REGRESS" if m > g * 1.10 else "tie")
        col = (plot.OKABE_ITO["green"] if verdict == "WIN"
               else plot.OKABE_ITO["vermillion"] if verdict == "REGRESS" else "k")
        top = max(g, m)
        ax.text(i, top * 1.06, f"{d:+.0f}%\n{verdict}", ha="center", va="bottom",
                fontsize=8.2, color=col, fontweight="bold")

    # smooth | hard divider
    n_smooth = sum(1 for k in kinds if k == "smooth")
    ax.axvline(n_smooth - 0.5, color="k", ls=":", lw=1, alpha=0.4)
    ax.text((n_smooth - 1) / 2, ax.get_ylim()[1] * 0.9, "smooth\n(CIRCE3 already strong)",
            ha="center", fontsize=8.5, alpha=0.6)
    ax.text((n_smooth + len(keys) - 1) / 2, ax.get_ylim()[1] * 0.9,
            "hard / discontinuity", ha="center", fontsize=8.5, alpha=0.6)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([PRETTY[k] for k in keys], fontsize=9)
    ax.set_ylabel("held-out ESR (log, lower = better)")
    ax.set_title("Fair-baseline A/B (equal ~100k params, identical harness): the mixed-activation "
                 "win is marginal and circuit-dependent", fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.set_ylim(top=max(max(gated), max(mixed)) * 2.0)
    fig.tight_layout()
    out = Path("outputs/figs/mixedact_fair_ab.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
