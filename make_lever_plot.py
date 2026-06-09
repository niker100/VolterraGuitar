"""Beautiful insight plot for a general-lever sweep (e.g. depth) across the
highly-nonlinear circuit set: held-ESR vs the lever per circuit (does it lower
loss *generally*?) + the real-time boundary. Reads a results JSON of the form
{circuit: {lever_value: {"held": float, "rtf": float}}}.

Run: uv run python make_lever_plot.py [results.json] [lever_name] [out.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from vguitar import plotting as plot


def main() -> None:
    res_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/depth_results.json")
    lever = sys.argv[2] if len(sys.argv) > 2 else "n_blocks"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs/figs/circe3_depth_sweep.png")
    res = json.loads(res_path.read_text())
    plot.apply_style()

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.0]})
    palette = list(plot.OKABE_ITO.values())
    for i, (circ, by_lever) in enumerate(sorted(res.items())):
        xs = sorted(int(k) for k in by_lever)
        held = [by_lever[str(x)]["held"] for x in xs]
        rtf = [by_lever[str(x)]["rtf"] for x in xs]
        col = palette[i % len(palette)]
        ax.plot(xs, held, "-o", color=col, lw=1.6, ms=6, label=circ)
        axr.plot(xs, rtf, "-o", color=col, lw=1.4, ms=5, alpha=0.9)
    ax.set(ylabel="held-out ESR (log, lower=better)", yscale="log",
           title=f"CIRCE3 generalization vs {lever} — highly-nonlinear circuits")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    axr.axhline(1.0, color="k", ls=":", lw=1, alpha=0.7)
    axr.text(axr.get_xlim()[0], 1.0, " real-time (RTF=1)", va="bottom", fontsize=7, alpha=0.7)
    axr.set(xlabel=lever, ylabel="RTF (CPU)")
    axr.set_xticks(sorted({int(k) for v in res.values() for k in v}))
    axr.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
