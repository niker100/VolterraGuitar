"""House-style figures for the research experiments — one home, regenerable.

Reads the probe result JSONs in ``outputs/`` and renders concise, insightful
figures in the project house style (reuses :func:`vguitar.plotting.apply_style`,
the Okabe-Ito palette, dense + un-annotated). Run ``uv run python -m
experiments.figures`` to regenerate everything under ``outputs/figs/frontier/``.

Consolidates what used to be a scatter of ``make_*.py`` one-off plot scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vguitar.plotting import OKABE_ITO, apply_style

OUT = Path("outputs/figs/frontier")
JSON = Path("outputs")
apply_style()


def _load(name: str) -> dict[str, Any] | None:
    p = JSON / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {OUT / name}.png", flush=True)


def fig_wavefolder_frontier() -> bool:
    """The consolidated wavefolder story: every lever tried, none beats ~0.18, and
    the big ones aren't even real-time. Left: held-ESR bars (green=real-time,
    vermillion=not), ref line at OS2. Right: held-ESR vs RTF (the real-time cliff)."""
    os_ab, shaper, cap = _load("wavefolder_os_ab"), _load("wavefolder_shaper_ab"), _load(
        "wavefolder_capacity_probe"
    )
    if not os_ab:
        return False
    rows: list[tuple[str, float, float]] = []  # (label, held, rtf)
    for label, r in os_ab.items():
        rows.append((label.replace("_ch24", "").replace("_", " "), r["held"], r["rtf"]))
    if shaper and "wavefolder" in shaper:
        v = shaper["wavefolder"]["variants"]
        rows.append(("fourier-head", v["fourier8"]["held"], v["fourier8"]["rtf"]))
    if cap:
        for label, r in cap.items():
            if label.startswith("ref"):
                continue
            rows.append((label.replace("_lr1e3_noclip", "↑cap").replace("_", " "),
                         r["held"], r["rtf"]))
    ref = os_ab.get("OS2_ch24", {}).get("held", 0.18)
    rows.sort(key=lambda t: t[1])
    labels = [r[0] for r in rows]
    held = np.array([r[1] for r in rows])
    rtf = np.array([r[2] for r in rows])
    colors = [OKABE_ITO["green"] if r > 1.0 else OKABE_ITO["vermillion"] for r in rtf]

    fig, (axb, axs) = plt.subplots(1, 2, figsize=(10, 3.6))
    y = np.arange(len(labels))
    axb.barh(y, held, color=colors, edgecolor="black", linewidth=0.4)
    axb.axvline(ref, ls="--", color=OKABE_ITO["black"], lw=1, alpha=0.7)
    axb.set_yticks(y)
    axb.set_yticklabels(labels, fontsize=8)
    axb.set_xlabel("held-out ESR (lower = better)")
    axb.set_title(f"Wavefolder: no lever beats the ~{ref:.2f} floor", fontsize=9)
    for yi, h in zip(y, held, strict=True):
        axb.text(h + 0.003, yi, f"{h:.3f}", va="center", fontsize=7)

    axs.scatter(rtf, held, c=colors, s=45, edgecolor="black", linewidth=0.4, zorder=3)
    for lab, r, h in zip(labels, rtf, held, strict=True):
        axs.annotate(lab, (r, h), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    axs.axvline(1.0, ls=":", color=OKABE_ITO["black"], lw=1)
    axs.axhline(ref, ls="--", color=OKABE_ITO["black"], lw=1, alpha=0.5)
    axs.set_xscale("log")
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    axs.xaxis.set_major_locator(LogLocator(base=10))
    axs.xaxis.set_major_formatter(ScalarFormatter())
    axs.xaxis.set_minor_formatter(NullFormatter())  # decade labels only — no clutter
    axs.set_xlabel("real-time factor (CPU, log) — right of dotted = real-time")
    axs.set_ylabel("held-out ESR")
    axs.set_title("capacity/OS trade RT for no accuracy gain", fontsize=9)
    fig.suptitle("Wavefolder frontier — a hard target floor (representation, not bandwidth/capacity)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "wavefolder_frontier")
    return True


def fig_ab(vals: dict[str, tuple[float, float]], labels: tuple[str, str], title: str,
           fname: str, kinds: dict[str, str] | None = None) -> None:
    """Grouped-bar A/B across circuits: ``vals`` = {circuit: (baseline, variant)}."""
    circuits = list(vals)
    base = np.array([vals[c][0] for c in circuits])
    var = np.array([vals[c][1] for c in circuits])
    x = np.arange(len(circuits))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(circuits)), 3.4))
    ax.bar(x - w / 2, base, w, label=labels[0], color=OKABE_ITO["gray"], edgecolor="black",
           linewidth=0.4)
    ax.bar(x + w / 2, var, w, label=labels[1], color=OKABE_ITO["purple"], edgecolor="black",
           linewidth=0.4)
    for xi, b, v in zip(x, base, var, strict=True):
        d = 100.0 * (v - b) / b if b else 0.0
        ax.text(xi + w / 2, v, f"{d:+.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n[{kinds[c]}]" if kinds and c in kinds else c for c in circuits], fontsize=8
    )
    ax.set_ylabel("held-out ESR")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, fname)


def fig_circex_gate(circex: dict[str, Any]) -> None:
    """Stacked gate-weight composition per circuit — which expert each leans on."""
    circuits = [c for c in circex if isinstance(circex[c], dict) and circex[c].get("gate")]
    if not circuits:
        return
    experts = ["long-smooth", "corner+periodic", "shallow-short"]
    cols = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"]]
    gates = np.array([circex[c]["gate"] for c in circuits])  # (n_circ, 3)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(circuits)), 3.4))
    bottom = np.zeros(len(circuits))
    for j, (e, col) in enumerate(zip(experts, cols, strict=True)):
        ax.bar(circuits, gates[:, j], bottom=bottom, label=e, color=col, edgecolor="black",
               linewidth=0.4)
        bottom += gates[:, j]
    ax.set_ylabel("mean gate weight")
    ax.set_ylim(0, 1)
    ax.set_title("CIRCE-X: which expert each circuit's gate favours", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    _save(fig, "circex_gate")


def fig_spectral_ab(sp: dict[str, Any]) -> None:
    """Two-panel A/B for the spectral hybrid: overall ESR (left) + high-band >4k ESR
    (right, the formant-fidelity metric it exists to move), time-only vs hybrid."""
    circuits = list(sp)
    x = np.arange(len(circuits))
    w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, title in (
        (a1, "esr", "overall held-ESR"),
        (a2, "band", "high-band (>4 kHz) ESR — the formant target"),
    ):
        base = np.array([sp[c]["time_only"][key] for c in circuits])
        hyb = np.array([sp[c]["hybrid"][key] for c in circuits])
        ax.bar(x - w / 2, base, w, label="time-only TCN", color=OKABE_ITO["gray"],
               edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, hyb, w, label="+ spectral branch", color=OKABE_ITO["blue"],
               edgecolor="black", linewidth=0.4)
        for xi, b, h in zip(x, base, hyb, strict=True):
            d = 100.0 * (h - b) / b if b else 0.0
            ax.text(xi + w / 2, h, f"{d:+.0f}%", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n[{sp[c].get('kind', '')}]" for c in circuits], fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=9)
    a1.legend(fontsize=8)
    fig.suptitle("Spectral-domain hybrid: complex STFT branch nails the formant band "
                 "(smooth circuits), neutral-to-worse on the dead-zone", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "spectral_ab")


def main() -> None:
    apply_style()
    made = []
    if fig_wavefolder_frontier():
        made.append("wavefolder_frontier")
    mvg = _load("mixed_vs_gated_final")
    if mvg:
        vals = {c: (r["gated"], r["mixed"]) for c, r in mvg.items() if isinstance(r, dict)}
        kinds = {c: r.get("kind", "") for c, r in mvg.items() if isinstance(r, dict)}
        fig_ab(vals, ("gated", "mixed"), "Block activation: mixed is the default (wins hard, ties smooth)",
               "mixed_vs_gated", kinds)
        made.append("mixed_vs_gated")
    sh = _load("wavefolder_shaper_ab")
    if sh:
        vals = {c: (r["variants"]["baseline"]["held"], r["variants"]["fourier8"]["held"])
                for c, r in sh.items()}
        kinds = {c: r.get("role", "") for c, r in sh.items()}
        fig_ab(vals, ("baseline", "fourier-head"), "Fourier waveshaper head: null + jfet regression (rejected)",
               "shaper_ab", kinds)
        made.append("shaper_ab")
    cx = _load("circex_probe")
    if cx:
        vals = {c: (r["baseline"], r["circex"]) for c, r in cx.items() if isinstance(r, dict)}
        kinds = {c: r.get("kind", "") for c, r in cx.items() if isinstance(r, dict)}
        fig_ab(vals, ("single TCN", "CIRCE-X MoE"), "CIRCE-X multi-modal MoE vs single-branch TCN",
               "circex_suite", kinds)
        fig_circex_gate(cx)
        made.extend(["circex_suite", "circex_gate"])
    sp = _load("spectral_probe")
    if sp:
        fig_spectral_ab(sp)
        made.append("spectral_ab")
    print(f"generated {len(made)} figures: {', '.join(made) or '(none — no JSONs yet)'}", flush=True)


if __name__ == "__main__":
    main()
