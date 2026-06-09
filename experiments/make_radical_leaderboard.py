"""Radical-architecture search leaderboard — a concise insight figure.

10 radical architectures were each short-trained on the hardest circuit
(class-B crossover dead-zone, held-ESR on crossover_classb_edge_test), to look
for a breakthrough past the smooth gated-TCN CIRCE3 baseline (~0.42-0.56 here).

Left panel:  ranked held-ESR bars vs the CIRCE3 baseline band + breakthrough bar.
Right panel: accuracy-vs-efficiency scatter (held-ESR vs params, log), marker =
             real-time-streamable today (o) or not (x). Bottom-left = ideal.

The single root cause across every winner: the dead-zone is a *static corner*,
and a corner-capable primitive (abs/relu, Snake folding, spline, PWL, phase)
beats the smooth tanh/sigmoid gate. Mixed activations win — best accuracy, a
pure drop-in activation swap, same compute class as a normal TCN.

Run: uv run python make_radical_leaderboard.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from vguitar import plotting as plot

# (short label, held-ESR, params, verdict, real-time-streamable-today)
# verdicts: pursue (general drop-in) / specialist / caution / dead (for plugin)
RESULTS = [
    ("Mixed activations\n(tanh·gelu·relu·abs·snake)", 0.0599, 97934, "pursue", True),
    ("Snake-only TCN", 0.0664, 52601, "pursue", True),
    ("KAN-TCN (B-spline edges)", 0.0710, 45921, "caution", False),
    ("Complex-valued TCN\n(Cardioid)", 0.0735, 48049, "caution", False),
    ("Local self-attention", 0.0887, 41361, "dead", False),
    ("Parallel Wiener-Hammerstein\n(PWL grey-box)", 0.1146, 1417, "specialist", True),
    ("Gradient-boosted TCNs", 0.1768, 20116, "dead", True),
    ("FNO spectral + pointwise NL", 0.2615, 170305, "dead", False),
    ("Neural ODE (RK4)", 0.3989, 3922, "dead", False),
    ("Spiking NN (LIF)", 0.4578, 17827, "dead", False),
]

BASE_LO, BASE_HI = 0.42, 0.56  # CIRCE3 gated-TCN baseline on this circuit
THRESH = 0.25                  # breakthrough bar

VCOLOR = {
    "pursue": plot.OKABE_ITO["green"],
    "specialist": plot.OKABE_ITO["blue"],
    "caution": plot.OKABE_ITO["orange"],
    "dead": plot.OKABE_ITO["gray"],
}
VLABEL = {
    "pursue": "Pursue — general drop-in, real-time",
    "specialist": "Specialist — tiny, static-NL circuits only",
    "caution": "Caution — beats bar but has a tax",
    "dead": "Dead end for a general plugin",
}


def main() -> None:
    plot.apply_style()
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- left: ranked held-ESR bars (best at top) --------------------------
    rows = sorted(RESULTS, key=lambda r: r[1], reverse=True)
    ys = range(len(rows))
    labels = [r[0] for r in rows]
    held = [r[1] for r in rows]
    cols = [VCOLOR[r[3]] for r in rows]
    axb.axvspan(BASE_LO, BASE_HI, color=plot.OKABE_ITO["vermillion"], alpha=0.12, zorder=0)
    axb.axvline((BASE_LO + BASE_HI) / 2, color=plot.OKABE_ITO["vermillion"], ls="--",
                lw=1.2, alpha=0.8, zorder=1)
    axb.text((BASE_LO + BASE_HI) / 2, len(rows) - 0.35, " CIRCE3 baseline\n (smooth gated-TCN)",
             color=plot.OKABE_ITO["vermillion"], fontsize=8, va="top", ha="center")
    axb.axvline(THRESH, color="k", ls=":", lw=1.2, alpha=0.7, zorder=1)
    axb.text(THRESH, -0.7, "breakthrough\nbar (0.25)", fontsize=7.5, ha="center", va="top",
             alpha=0.75)
    axb.barh(list(ys), held, color=cols, height=0.66, zorder=3, edgecolor="white", lw=0.6)
    for yi, r in zip(ys, rows, strict=True):
        mk = "RT" if r[4] else "no-RT"
        axb.text(r[1] + 0.006, yi, f"{r[1]:.3f}  ·  {r[2] / 1000:.0f}k  ·  {mk}",
                 va="center", fontsize=8)
    axb.set_yticks(list(ys))
    axb.set_yticklabels(labels, fontsize=8.5)
    axb.set_xlim(0, 0.66)
    axb.set_xlabel("held-out ESR on crossover dead-zone (lower = better)")
    axb.set_title("Radical-architecture search — 10 ideas vs CIRCE3", fontsize=11, loc="left")
    axb.grid(True, axis="x", alpha=0.25)

    # ---- right: accuracy vs efficiency scatter -----------------------------
    axs.axhspan(BASE_LO, BASE_HI, color=plot.OKABE_ITO["vermillion"], alpha=0.10, zorder=0)
    axs.axhline(THRESH, color="k", ls=":", lw=1.2, alpha=0.7)
    axs.text(1.05e3, THRESH, "breakthrough bar", fontsize=7.5, va="bottom", alpha=0.75)
    for _label, h, p, v, rt in RESULTS:
        m = "o" if rt else "X"
        axs.scatter(p, h, s=130, c=VCOLOR[v], marker=m, edgecolor="k", lw=0.7, zorder=3)
    # annotate the two winners + the tiny specialist
    for label, h, p, v, _rt in RESULTS:
        if v in ("pursue", "specialist"):
            short = label.split("\n")[0]
            axs.annotate(short, (p, h), textcoords="offset points", xytext=(7, 6),
                         fontsize=7.8, color=VCOLOR[v])
    axs.set_xscale("log")
    axs.set_xlabel("parameters (log) — efficiency →")
    axs.set_ylabel("held-out ESR — accuracy ↓")
    axs.set_title("Accuracy vs efficiency  (bottom-left = ideal)", fontsize=11, loc="left")
    axs.set_ylim(0, 0.62)
    axs.grid(True, which="both", alpha=0.25)

    handles = [mpatches.Patch(color=VCOLOR[k], label=VLABEL[k]) for k in VCOLOR]
    handles += [
        plt.Line2D([], [], marker="o", color="k", ls="", mfc="none", label="real-time today"),
        plt.Line2D([], [], marker="X", color="k", ls="", mfc="none", label="not real-time"),
    ]
    axs.legend(handles=handles, loc="upper right", fontsize=7.6, frameon=False)

    fig.suptitle(
        "Corner-capable activations beat the smooth gated-TCN 5-9x on a static discontinuity",
        fontsize=12.5, y=1.005)
    fig.tight_layout()
    out = Path("outputs/figs/radical_leaderboard.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
