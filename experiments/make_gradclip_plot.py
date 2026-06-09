"""The gradient-clipping win: one uniform, standard, zero-inference-cost training
change (clip global grad-norm at 1.0) vs no clipping, across all circuits, at the
production CIRCE3 config (ch24/nb2/nl9, OS2). Reads
outputs/radical_gradclip_test.json (held-ESR lists per seed for clip 0.0 / 1.0).

Bars = mean held-ESR; whiskers = seed min-max (clipping also COLLAPSES the seed
variance on hard circuits, not just the mean). Annotated with % change. The story:
big win on the hardest circuit (crossover ~8x), helps the smooth circuits too,
regresses none -- a genuinely general improvement, no per-circuit tailoring.

Run: uv run python make_gradclip_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vguitar import plotting as plot

ORDER = ["bjt", "jfet", "tube_screamer", "crossover", "wavefolder", "asym_clipper",
         "hard_clipper", "fullwave_rectifier", "hysteretic_fuzz"]
PRETTY = {"bjt": "BJT", "jfet": "JFET", "tube_screamer": "Tube\nScreamer",
          "crossover": "crossover", "wavefolder": "wavefolder", "asym_clipper": "asym\nclip",
          "hard_clipper": "hard\nclip", "fullwave_rectifier": "fullwave\nrect",
          "hysteretic_fuzz": "hyst.\nfuzz"}


def stat(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.min()), float(a.max())


def main() -> None:
    res = json.loads(Path("outputs/radical_gradclip_test.json").read_text())
    plot.apply_style()
    keys = [k for k in ORDER if k in res]
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    xs = np.arange(len(keys))
    w = 0.38
    cg, cc = plot.OKABE_ITO["gray"], plot.OKABE_ITO["green"]
    for off, tag, col, lab in [(-w / 2, "clip0.0", cg, "no grad-clip (was)"),
                               (w / 2, "clip1.0", cc, "grad-clip 1.0 (new default)")]:
        means, los, his = [], [], []
        for k in keys:
            mn, lo, hi = stat(res[k][tag])
            means.append(mn)
            los.append(mn - lo)
            his.append(hi - mn)
        ax.bar(xs + off, means, w, color=col, edgecolor="white", lw=0.5, label=lab,
               yerr=[los, his], capsize=2.5, error_kw={"lw": 0.8, "alpha": 0.7})

    for i, k in enumerate(keys):
        c0, _, _ = stat(res[k]["clip0.0"])
        c1, _, _ = stat(res[k]["clip1.0"])
        d = (c1 - c0) / c0 * 100.0
        col = (plot.OKABE_ITO["green"] if c1 < c0 * 0.97
               else plot.OKABE_ITO["vermillion"] if c1 > c0 * 1.03 else "k")
        ax.text(i, max(c0, c1) * 1.25, f"{d:+.0f}%", ha="center", va="bottom",
                fontsize=8.2, color=col, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([PRETTY[k] for k in keys], fontsize=8.5)
    ax.set_ylabel("held-out ESR (log, lower = better)")
    ax.set_title("Gradient clipping: one uniform, zero-inference-cost change cuts the hardest "
                 "circuit ~8x and helps the rest", fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.set_ylim(top=max(stat(res[k]["clip0.0"])[2] for k in keys) * 3)
    fig.tight_layout()
    out = Path("outputs/figs/gradclip_win.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
