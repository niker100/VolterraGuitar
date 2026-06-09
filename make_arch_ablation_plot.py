"""Uniform-architecture ablation result: which SINGLE config generalizes best
across the circuit mix (no per-circuit tailoring)? Reads
outputs/radical_arch_ablation.json (written by radical_arch_ablation.py).

Grouped bars: per circuit, one bar per uniform config; log-y held-ESR. The right
panel ranks configs by hard-circuit mean and overall mean — the decision metric
for a single generalized model.

Run: uv run python make_arch_ablation_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vguitar import plotting as plot

CIRC_ORDER = ["bjt", "jfet", "crossover", "wavefolder"]
PRETTY = {"bjt": "BJT", "jfet": "JFET", "crossover": "crossover", "wavefolder": "wavefolder"}


def main() -> None:
    res = json.loads(Path("outputs/radical_arch_ablation.json").read_text())
    plot.apply_style()
    configs = list(res.keys())
    circs = [c for c in CIRC_ORDER if c in next(iter(res.values()))["by_circuit"]]
    palette = [plot.OKABE_ITO["gray"], plot.OKABE_ITO["blue"], plot.OKABE_ITO["vermillion"],
               plot.OKABE_ITO["green"]]

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [2.1, 1]})
    xs = np.arange(len(circs))
    w = 0.8 / len(configs)
    for ci, cfg in enumerate(configs):
        bc = res[cfg]["by_circuit"]
        vals = [bc[c]["held"] for c in circs]
        npar = res[cfg]["params"]
        ax.bar(xs + (ci - (len(configs) - 1) / 2) * w, vals, w,
               label=f"{cfg}  ({npar/1000:.0f}k)", color=palette[ci % len(palette)],
               edgecolor="white", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([PRETTY[c] for c in circs])
    ax.set_ylabel("held-out ESR (log, lower = better)")
    ax.set_title("Uniform-architecture ablation (all OS2): per-circuit held-ESR", loc="left",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    # right: rank configs by hard-mean + overall mean
    hard_means = [res[c]["hard_mean"] for c in configs]
    means = [res[c]["mean"] for c in configs]
    yy = np.arange(len(configs))
    axr.barh(yy + 0.18, hard_means, 0.36, color=plot.OKABE_ITO["vermillion"],
             label="hard-circuit mean")
    axr.barh(yy - 0.18, means, 0.36, color=plot.OKABE_ITO["sky"], label="overall mean")
    for y, (hm, mm) in enumerate(zip(hard_means, means, strict=True)):
        axr.text(hm, y + 0.18, f" {hm:.3f}", va="center", fontsize=7.5)
        axr.text(mm, y - 0.18, f" {mm:.3f}", va="center", fontsize=7.5)
    axr.set_yticks(yy)
    axr.set_yticklabels(configs, fontsize=8)
    axr.invert_yaxis()
    axr.set_xlabel("mean held-ESR")
    axr.set_title("Decision metric: best single config", loc="left", fontsize=11)
    axr.legend(fontsize=8, loc="lower right")
    axr.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Searching one uniform config that lowers hard-circuit loss without hurting "
                 "smooth circuits", fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path("outputs/figs/arch_ablation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
