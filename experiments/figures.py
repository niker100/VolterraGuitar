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


def fig_spectral_slim(slim: dict[str, Any]) -> None:
    """band>4k vs spectral hidden width — shows the formant win is flat down to a tiny
    net (so it's real-time-affordable; the big MLP was overkill)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cols = {"bjt": OKABE_ITO["vermillion"], "jfet": OKABE_ITO["blue"]}
    for circ, runs in slim.items():
        hs = sorted(v["hidden"] for v in runs.values())
        band = [next(v["band"] for v in runs.values() if v["hidden"] == h) for h in hs]
        base = next(v["band"] for v in runs.values() if v["hidden"] == 0)
        ax.plot(hs, band, "o-", color=cols.get(circ, OKABE_ITO["green"]), label=circ)
        ax.axhline(base, ls=":", color=cols.get(circ, OKABE_ITO["green"]), lw=1, alpha=0.5)
    ax.set_xlabel("spectral hidden width (0 = time-only baseline, dotted)")
    ax.set_ylabel("high-band (>4 kHz) ESR")
    ax.set_title("Spectral formant win is flat down to hidden=16 → real-time-affordable",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "spectral_slim")


def fig_spectral_only(so: dict[str, Any]) -> None:
    """3-way (time-only / spectral-only / hybrid): spectral alone can't distort
    (huge overall ESR, esp. bjt) yet still shapes the formant envelope (band>4k).
    Left: overall ESR (log). Right: band>4k. The division-of-labor proof."""
    circuits = list(so)
    modes = ["time_only", "spectral_only", "hybrid"]
    cols = [OKABE_ITO["gray"], OKABE_ITO["vermillion"], OKABE_ITO["blue"]]
    x = np.arange(len(circuits))
    w = 0.26
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, key, title, logy in (
        (a1, "esr", "overall held-ESR (log) — spectral-only can't generate harmonics", True),
        (a2, "band", "high-band (>4 kHz) ESR — spectral-only still shapes formants", False),
    ):
        for j, (mode, col) in enumerate(zip(modes, cols, strict=True)):
            vals = np.array([so[c][mode][key] for c in circuits])
            ax.bar(x + (j - 1) * w, vals, w, label=mode.replace("_", "-"), color=col,
                   edgecolor="black", linewidth=0.4)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n[{so[c].get('kind', '')}]" for c in circuits], fontsize=8)
        ax.set_title(title, fontsize=8.5)
    a1.legend(fontsize=8)
    fig.suptitle("Only the FFT branch? It filters, it doesn't distort — "
                 "harmonics need the time head", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "spectral_only")


def fig_spectral_zoo(zoo: dict[str, Any]) -> None:
    """The spectral-operator zoo: bjt-ESR vs jfet-ESR (log-log) — baselines starred,
    the 'good corner' is bottom-left. Shows the wins are circuit-split and no variant
    beats the time+spectral hybrid on both. Point size ~ params."""
    res = zoo["results"]
    params = zoo.get("params", {})
    tags = [t for t in res if "bjt" in res[t] and "jfet" in res[t]]
    bjt = np.array([res[t]["bjt"].get("esr", np.nan) for t in tags])
    jf = np.array([res[t]["jfet"].get("esr", np.nan) for t in tags])
    # clip failures (ESR ~1) to the plot edge so they don't blow the axes
    bjt = np.clip(np.nan_to_num(bjt, nan=1.2), 1e-3, 1.2)
    jf = np.clip(np.nan_to_num(jf, nan=1.2), 1e-3, 1.2)
    sz = np.array([max(params.get(t, 1e4), 5e3) / 600 for t in tags])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for t, x, y, s in zip(tags, bjt, jf, sz, strict=True):
        base = t in ("hybrid", "time_only")
        ax.scatter(x, y, s=s if not base else 180, marker="*" if base else "o",
                   color=OKABE_ITO["purple"] if base else OKABE_ITO["sky"],
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(t, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bjt held-ESR (strong distortion) — lower better")
    ax.set_ylabel("jfet held-ESR (mild) — lower better")
    ax.set_title("Spectral-operator zoo: wins are circuit-split; none beats the\n"
                 "time+spectral hybrid on both (stars=baselines, circles=variants, size~params)",
                 fontsize=9)
    ax.axhline(res["hybrid"]["jfet"]["esr"], ls=":", color=OKABE_ITO["purple"], lw=1, alpha=0.5)
    ax.axvline(res["hybrid"]["bjt"]["esr"], ls=":", color=OKABE_ITO["purple"], lw=1, alpha=0.5)
    fig.tight_layout()
    _save(fig, "spectral_zoo")


def fig_spectral_leads(leads: dict[str, Any]) -> None:
    """Multi-seed deep-dive verdict: grouped held-ESR bars over 3 formant circuits
    (bjt/jfet/tube_screamer) for the hybrid vs the three cheap zoo leads, with
    seed min-max whiskers and a log y-axis (values span 0.009 -> 1.0). Tells the
    whole story at a glance: the hybrid is the only flat, low, seed-stable line;
    every cheap operator either spikes, swings across seeds, or fails outright."""
    models = ["hybrid", "stft_mix", "fft_longfir", "wavelet"]
    cols = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["vermillion"]]
    circuits = [c for c in ("bjt", "jfet", "tube_screamer") if c in leads.get("hybrid", {})]
    x = np.arange(len(circuits))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.0))
    for j, (m, col) in enumerate(zip(models, cols, strict=True)):
        means = np.array([leads[m][c]["esr_mean"] for c in circuits])
        lo = np.array([min(leads[m][c]["esr"]) for c in circuits])
        hi = np.array([max(leads[m][c]["esr"]) for c in circuits])
        off = (j - 1.5) * w
        ax.bar(x + off, means, w, label=m, color=col, edgecolor="black", linewidth=0.4,
               yerr=[means - lo, hi - means], capsize=2, error_kw={"lw": 0.8})
        for xi, mu in zip(x, means, strict=True):
            if mu >= 0.99:  # flag the catastrophic failures
                ax.text(xi + off, 1.02, "fail", ha="center", va="bottom", fontsize=6,
                        color=OKABE_ITO["vermillion"], rotation=90)
    ax.set_yscale("log")
    ax.axhline(leads["hybrid"]["jfet"]["esr_mean"], ls=":", color=OKABE_ITO["blue"], lw=1,
               alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(circuits, fontsize=9)
    ax.set_ylabel("held-out ESR (log) — lower better")
    ax.set_title("Spectral leads, multi-seed: only the time+spectral hybrid generalizes\n"
                 "(whiskers = seed min-max; cheap operators spike, swing, or fail)",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    _save(fig, "spectral_leads")


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
    sl = _load("spectral_slim")
    if sl:
        fig_spectral_slim(sl)
        made.append("spectral_slim")
    so = _load("spectral_only_probe")
    if so:
        fig_spectral_only(so)
        made.append("spectral_only")
    zoo = _load("spectral_zoo")
    if zoo:
        fig_spectral_zoo(zoo)
        made.append("spectral_zoo")
    leads = _load("spectral_leads")
    if leads:
        fig_spectral_leads(leads)
        made.append("spectral_leads")
    print(f"generated {len(made)} figures: {', '.join(made) or '(none — no JSONs yet)'}", flush=True)


if __name__ == "__main__":
    main()
