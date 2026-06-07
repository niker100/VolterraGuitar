r"""Figures — the single home for every diagnostic plot.

House style (shared with the author's other projects): serif, font-size 10, no
top/right spines, faint grid, frameless legends, the Okabe-Ito colourblind-safe
palette, and fixed colormap roles (signed -> RdBu_r, magnitude -> cividis,
spectrogram -> magma). Each ``fig_*`` builder takes primitive inputs (arrays,
dicts) and returns a Matplotlib ``Figure``; the CLI (``vguitar plots``) saves
them as PNG. The backend is forced to ``Agg`` for headless runs.

The builders are deliberately dense and un-annotated: one consistent colour per
model across every figure, key numbers in the title, and no callouts or legends
beyond what's needed to read the panel. Together they answer "what is the
circuit doing, and which model captured it":

* :func:`fig_dataset`     — what the model learns from (excitation + response).
* :func:`fig_transfer`    — the static nonlinearity (output-vs-input curve).
* :func:`fig_harmonics`   — harmonic generation under a pure tone (the distortion).
* :func:`fig_waveform`    — circuit vs models on a hard-clipping segment.
* :func:`fig_volterra_kernels` — the identified Volterra kernels (interpretable).
* :func:`fig_leaderboard` — accuracy (ESR) vs speed (RTF) across models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a window
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Okabe & Ito (2008) colourblind-safe qualitative palette.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "gray": "#999999",
}

# One fixed colour per model, used in every figure so the eye tracks a model.
MODEL_COLORS = {
    "fir": OKABE_ITO["gray"],
    "volterra": OKABE_ITO["blue"],
    "volterra_pc": OKABE_ITO["sky"],
    "wh": OKABE_ITO["green"],
    "tcn": OKABE_ITO["vermillion"],
    "rnn": OKABE_ITO["orange"],
    "ssm": OKABE_ITO["purple"],
    "circe": OKABE_ITO["purple"],  # the conditioned hero model
}

# Fixed colormap roles (one meaning each).
CMAP_SIGNED = "RdBu_r"  # signed kernels / fields (diverging, 0 = white)
CMAP_MAG = "cividis"    # non-negative magnitudes
CMAP_SPEC = "magma"     # spectrograms (log power)


def apply_style() -> None:
    """Apply the house Matplotlib style (idempotent)."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "lines.linewidth": 1.3,
    })


apply_style()


def color_for(name: str) -> str:
    """Stable colour for a model name (falls back to a palette cycle)."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    cycle = list(OKABE_ITO.values())
    return cycle[hash(name) % len(cycle)]


# --- helpers -----------------------------------------------------------------
def _spectrogram(ax, y: np.ndarray, sr: int, title: str) -> None:
    """Log-power spectrogram panel, ticks in kHz, grid off."""
    ax.specgram(np.asarray(y, dtype=np.float64), NFFT=1024, Fs=sr, noverlap=768,
                cmap=CMAP_SPEC, mode="magnitude", scale="dB", vmin=-120, vmax=0)
    ax.set(title=title, xlabel="time (s)", ylabel="kHz")
    ax.set_yticks([0, 5000, 10000, 15000, 20000], ["0", "5", "10", "15", "20"])
    ax.grid(False)


# --- the dataset (what the model learns from) --------------------------------
def fig_dataset(x: np.ndarray, y: np.ndarray, sr: int, name: str = "") -> Figure:
    """(a) input & output (peak-normalized) on a short window; (b) output
    spectrogram; (c) input-drive amplitude coverage."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), int(0.006 * sr))  # ~6 ms window
    s = len(x) // 2
    t = np.arange(n) / sr * 1e3  # ms
    xn = x[s : s + n] / (np.max(np.abs(x)) + 1e-30)
    yn = y[s : s + n] / (np.max(np.abs(y)) + 1e-30)

    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.0), constrained_layout=True)
    ax[0].plot(t, xn, color=OKABE_ITO["gray"], label="input")
    ax[0].plot(t, yn, color=OKABE_ITO["blue"], label="output")
    ax[0].set(title=f"(a) {name} signals (norm.)", xlabel="time (ms)", ylabel="amplitude")
    ax[0].legend(loc="upper right", fontsize=8)
    _spectrogram(ax[1], y, sr, "(b) output spectrogram")
    ax[2].hist(np.abs(x), bins=60, color=OKABE_ITO["blue"], alpha=0.85)
    ax[2].set(title="(c) input drive coverage", xlabel="|input| (V)", ylabel="count", yscale="log")
    return fig


# --- the static nonlinearity -------------------------------------------------
def fig_transfer(x_probe: np.ndarray, y_ref: np.ndarray, preds: dict[str, np.ndarray],
                 name: str = "") -> Figure:
    """Output-vs-input curve for a slow sweep: the circuit's nonlinear shape
    (black) with each model overlaid. Loop width = memory; mismatch = error."""
    x_probe = np.asarray(x_probe, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(x_probe, y_ref, color="k", lw=2.2, label="circuit", zorder=3)
    for mname, yp in preds.items():
        ax.plot(x_probe, np.asarray(yp, dtype=np.float64), color=color_for(mname),
                lw=1.1, alpha=0.9, label=mname, zorder=2)
    # Crop to the circuit's range: polynomial models (Volterra/WH) can diverge
    # outside their trained amplitude; clipping keeps the in-range shape legible
    # (a model leaving the frame is itself the diagnostic).
    lim = 1.4 * float(np.max(np.abs(y_ref))) or 1.0
    ax.set_ylim(-lim, lim)
    ax.set(title=f"{name} static transfer  (output vs input)", xlabel="input (V)",
           ylabel="output (V)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# --- harmonic generation -----------------------------------------------------
def _harmonic_levels(y: np.ndarray, sr: int, f0: float, n_harm: int) -> np.ndarray:
    """Level of harmonics 1..n_harm in dB relative to the fundamental."""
    y = np.asarray(y, dtype=np.float64)
    mag = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    n = len(y)
    levels = np.full(n_harm, -120.0)
    fund = 0.0
    for k in range(1, n_harm + 1):
        b = round(k * f0 * n / sr)
        if b + 3 >= len(mag):
            break
        peak = float(mag[max(b - 3, 0) : b + 4].max())  # small window absorbs leakage
        if k == 1:
            fund = peak
        levels[k - 1] = 20.0 * np.log10(peak / (fund + 1e-30) + 1e-12)
    return levels


def fig_harmonics(y_ref: np.ndarray, preds: dict[str, np.ndarray], sr: int,
                  f0: float = 1000.0, n_harm: int = 12, name: str = "") -> Figure:
    """Harmonic stack of a pure ``f0`` tone: level (dB, relative to the
    fundamental) at each harmonic for the circuit (black) and each model. A model
    that tracks the circuit's stack reproduces the distortion; one that drops to
    the floor (e.g. a linear fit) adds no harmonics."""
    ks = np.arange(1, n_harm + 1)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(ks, _harmonic_levels(y_ref, sr, f0, n_harm), "o-", color="k", lw=1.8,
            ms=5, label="circuit", zorder=3)
    for mname, yp in preds.items():
        ax.plot(ks, _harmonic_levels(yp, sr, f0, n_harm), "o-", color=color_for(mname),
                lw=1.0, ms=4, alpha=0.85, label=mname, zorder=2)
    ax.set(title=f"{name} harmonic stack of a {f0:.0f} Hz tone", xlabel="harmonic",
           ylabel="level rel. fundamental (dB)", ylim=(-90, 5), xticks=ks)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# --- waveform comparison -----------------------------------------------------
def fig_waveform(y_ref: np.ndarray, preds: dict[str, np.ndarray], sr: int,
                 start: int | None = None, n: int = 600, name: str = "") -> Figure:
    """Circuit (black) vs each model on a short, hard-clipping segment, with the
    residual (model - circuit) below at the same scale."""
    y_ref = np.asarray(y_ref, dtype=np.float64)
    if start is None:
        start = int(np.argmax(np.abs(y_ref)) - n // 2)  # centre on the loudest part
    start = max(0, min(start, len(y_ref) - n))
    sl = slice(start, start + n)
    t = np.arange(n) / sr * 1e3

    fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True,
                           gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax[0].plot(t, y_ref[sl], color="k", lw=2.0, label="circuit", zorder=3)
    for mname, yp in preds.items():
        yp = np.asarray(yp, dtype=np.float64)
        ax[0].plot(t, yp[sl], color=color_for(mname), lw=1.0, alpha=0.9, label=mname)
        ax[1].plot(t, yp[sl] - y_ref[sl], color=color_for(mname), lw=0.9, alpha=0.9)
    ax[0].set(title=f"{name} waveform (clipping segment)", ylabel="output (V)")
    ax[0].legend(loc="upper right", fontsize=8, ncol=2)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="time (ms)", ylabel="residual")
    fig.tight_layout()
    return fig


# --- Volterra kernels (interpretable) ----------------------------------------
def fig_volterra_kernels(h1: np.ndarray, h2: np.ndarray | None, h3: np.ndarray | None,
                         sr: int) -> Figure:
    """The identified kernels: h1 (impulse response), h2 (quadratic memory map),
    and a central slice of h3 — the circuit's nonlinearity made legible."""
    h1 = np.asarray(h1, dtype=np.float64)
    panels = 1 + (h2 is not None) + (h3 is not None)
    fig, ax = plt.subplots(1, panels, figsize=(3.6 * panels, 3.3), constrained_layout=True)
    ax = np.atleast_1d(ax)
    tau = np.arange(len(h1)) / sr * 1e3
    ax[0].plot(tau, h1, color=OKABE_ITO["blue"])
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(title="$h_1$ (linear kernel)", xlabel="lag (ms)", ylabel="gain")
    i = 1
    if h2 is not None:
        h2 = np.asarray(h2, dtype=np.float64)
        v = float(np.max(np.abs(h2))) or 1.0
        im = ax[i].imshow(h2, cmap=CMAP_SIGNED, vmin=-v, vmax=v, origin="lower")
        ax[i].set(title="$h_2$ (quadratic kernel)", xlabel="lag $j$", ylabel="lag $i$")
        ax[i].grid(False)
        fig.colorbar(im, ax=ax[i], shrink=0.82)
        i += 1
    if h3 is not None:
        h3 = np.asarray(h3, dtype=np.float64)
        sl = h3[:, :, h3.shape[2] // 2]  # central slice k = mem3/2
        v = float(np.max(np.abs(sl))) or 1.0
        im = ax[i].imshow(sl, cmap=CMAP_SIGNED, vmin=-v, vmax=v, origin="lower")
        ax[i].set(title="$h_3$ (cubic, central slice)", xlabel="lag $j$", ylabel="lag $i$")
        ax[i].grid(False)
        fig.colorbar(im, ax=ax[i], shrink=0.82)
    return fig


# --- accuracy vs speed -------------------------------------------------------
def fig_leaderboard(rows: list[dict], name: str = "") -> Figure:
    """The headline tradeoff: ESR (accuracy, log) vs RTF (speed, log), one point
    per model; the dashed line is the real-time threshold (RTF = 1)."""
    ok = [r for r in rows if "error" not in r and r.get("esr", 0) > 0]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

    for r in ok:
        c = color_for(r["model"])
        ax[0].scatter(r["rtf"], r["esr"], s=70, color=c, edgecolor="k", linewidth=0.5, zorder=3)
        ax[0].annotate(r["model"], (r["rtf"], r["esr"]), textcoords="offset points",
                       xytext=(6, 4), fontsize=8)
    ax[0].axvline(1.0, color="k", ls="--", lw=1.0)
    ax[0].set(xscale="log", yscale="log", xlabel="real-time factor (>1 = live)",
              ylabel="ESR (lower = more accurate)", title=f"(a) {name} accuracy vs speed")

    order = sorted(ok, key=lambda r: r["esr"])
    ax[1].barh([r["model"] for r in order], [r["esr"] for r in order],
               color=[color_for(r["model"]) for r in order])
    ax[1].set(xscale="log", xlabel="ESR", title="(b) accuracy ranking")
    ax[1].invert_yaxis()
    fig.tight_layout()
    return fig


# --- conditioned model: response across a control sweep ----------------------
def fig_control_response(control_vals: np.ndarray, circuit_y: np.ndarray, model_y: np.ndarray,
                         *, held_mask: np.ndarray | None = None, ylabel: str = "THD",
                         control_name: str = "drive", name: str = "") -> Figure:
    """A metric (e.g. THD, output level) vs a control knob: circuit (black) vs a
    conditioned model (colour). Trained settings are filled markers; **held-out
    (interpolated) settings are open markers** — if they land on the circuit
    curve, the model interpolates the knob correctly."""
    cv = np.asarray(control_vals, dtype=np.float64)
    order = np.argsort(cv)
    cv = cv[order]
    cy = np.asarray(circuit_y, dtype=np.float64)[order]
    my = np.asarray(model_y, dtype=np.float64)[order]
    held = (np.zeros(len(cv), bool) if held_mask is None else np.asarray(held_mask, bool)[order])
    col = color_for("circe")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(cv, cy, "-o", color="k", lw=1.8, ms=5, label="circuit", zorder=3)
    ax.plot(cv, my, "-", color=col, lw=1.3, label="CIRCE", zorder=2)
    ax.scatter(cv[~held], my[~held], s=45, color=col, zorder=4, label="trained")
    if held.any():
        ax.scatter(cv[held], my[held], s=80, facecolor="white", edgecolor=col, linewidth=1.6,
                   zorder=5, label="held-out (interpolated)")
    ax.set(title=f"{name} {ylabel} vs {control_name}", xlabel=control_name, ylabel=ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_esr_by_control(control_vals: np.ndarray, esr_vals: np.ndarray, *,
                       held_mask: np.ndarray | None = None, control_name: str = "drive",
                       name: str = "") -> Figure:
    """Per-setting test ESR; trained settings filled, held-out (interpolated) hatched."""
    cv = np.asarray(control_vals, dtype=np.float64)
    order = np.argsort(cv)
    cv, ev = cv[order], np.asarray(esr_vals, dtype=np.float64)[order]
    held = np.zeros(len(cv), bool) if held_mask is None else np.asarray(held_mask, bool)[order]
    col = color_for("circe")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(np.arange(len(cv)), ev, color=col)
    for i, h in enumerate(held):
        if h:
            bars[i].set_facecolor("white")
            bars[i].set_edgecolor(col)
            bars[i].set_linewidth(1.6)
            bars[i].set_hatch("//")
    ax.set_xticks(np.arange(len(cv)), [f"{v:g}" for v in cv])
    ax.set(xlabel=control_name, ylabel="test ESR", title=f"{name} ESR by {control_name} (hatched = held-out)")
    fig.tight_layout()
    return fig


def fig_knob_harmonics(drives: list[float], circ: list[np.ndarray], model: list[np.ndarray],
                       held: list[bool], sr: int, f0: float = 1000.0, name: str = "") -> Figure:
    """Harmonic stack at several drive settings: circuit (black) vs CIRCE (colour)."""
    n = len(drives)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.1), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    ks = np.arange(1, 9)
    for a, g, yc, ym, h in zip(axes, drives, circ, model, held, strict=True):
        a.plot(ks, _harmonic_levels(yc, sr, f0, 8), "o-", color="k", ms=4, lw=1.6, label="circuit")
        a.plot(ks, _harmonic_levels(ym, sr, f0, 8), "o-", color=color_for("circe"), ms=3, lw=1.1,
               label="CIRCE")
        a.set(title=f"drive={g * 1000:.0f} mV{' (held-out)' if h else ''}", xlabel="harmonic",
              ylim=(-80, 5), xticks=ks)
    axes[0].set_ylabel("level rel. fund. (dB)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{name} harmonic stack across the drive knob", fontsize=10)
    return fig


def fig_knob_waveforms(drives: list[float], circ: list[np.ndarray], model: list[np.ndarray],
                       held: list[bool], sr: int, n: int = 400, name: str = "") -> Figure:
    """Output waveform at several drive settings: circuit (black) vs CIRCE (colour)."""
    npan = len(drives)
    fig, axes = plt.subplots(1, npan, figsize=(3.4 * npan, 2.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for a, g, yc, ym, h in zip(axes, drives, circ, model, held, strict=True):
        yc = np.asarray(yc, dtype=np.float64)
        ym = np.asarray(ym, dtype=np.float64)
        s = max(0, int(np.argmax(np.abs(yc)) - n // 2))
        sl = slice(s, s + n)
        t = np.arange(len(yc[sl])) / sr * 1e3
        a.plot(t, yc[sl], color="k", lw=1.6, label="circuit")
        a.plot(t, ym[sl], color=color_for("circe"), lw=1.0, label="CIRCE")
        a.set(title=f"drive={g * 1000:.0f} mV{' (held-out)' if h else ''}", xlabel="time (ms)")
    axes[0].set_ylabel("output (V)")
    axes[0].legend(fontsize=8)
    return fig


def fig_training_curve(history: dict[str, list[float]], name: str = "") -> Figure:
    """Training/validation curves vs epoch (log-y)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, col, lab in (("val_esr", color_for("circe"), "val ESR"),
                          ("train_loss", OKABE_ITO["gray"], "train loss")):
        v = history.get(key)
        if v:
            ax.plot(range(1, len(v) + 1), v, color=col, label=lab)
    ax.set(xlabel="epoch", ylabel="loss / ESR", yscale="log", title=f"{name} CIRCE training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_interp_vs_distance(dist: np.ndarray, esr_vals: np.ndarray, held_mask: np.ndarray,
                           *, control_name: str = "drive", name: str = "") -> Figure:
    """ESR vs distance-to-nearest-trained control setting (the interpolation test).

    Trained settings sit at distance 0 (filled); held-out (interpolated) settings
    are open markers — if their ESR is near the trained level, interpolation holds.
    """
    d = np.asarray(dist, dtype=np.float64)
    ev = np.asarray(esr_vals, dtype=np.float64)
    held = np.asarray(held_mask, bool)
    col = color_for("circe")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(d[~held], ev[~held], s=45, color=col, label="trained", zorder=3)
    if held.any():
        ax.scatter(d[held], ev[held], s=80, facecolor="white", edgecolor=col, linewidth=1.6,
                   label="held-out (interpolated)", zorder=4)
    ax.set(xlabel=f"distance to nearest trained {control_name} (normalized)", ylabel="ESR",
           yscale="log", title=f"{name} ESR vs control distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_control_esr_heatmap(vals_i: np.ndarray, vals_j: np.ndarray, esr_grid: np.ndarray,
                            *, names: tuple[str, str] = ("c0", "c1"), name: str = "") -> Figure:
    """ESR over a 2-D control plane (two knobs, others at default): the multi-
    control analogue of the drive x frequency heatmap. ``esr_grid`` is
    ``(len(vals_i), len(vals_j))`` ESR (lower = better), drawn on a log colour
    scale (cividis = magnitude)."""
    from matplotlib.colors import LogNorm

    vi = np.asarray(vals_i, dtype=np.float64)
    vj = np.asarray(vals_j, dtype=np.float64)
    g = np.asarray(esr_grid, dtype=np.float64)
    lo = max(float(np.nanmin(g[g > 0])) if np.any(g > 0) else 1e-3, 1e-4)
    hi = max(float(np.nanmax(g)), lo * 1.001)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    im = ax.imshow(g, aspect="auto", origin="lower", cmap=CMAP_MAG,
                   norm=LogNorm(vmin=lo, vmax=hi),
                   extent=(0.0, float(len(vj)), 0.0, float(len(vi))))
    ax.set_xticks(np.arange(len(vj)) + 0.5, [f"{v:g}" for v in vj])
    ax.set_yticks(np.arange(len(vi)) + 0.5, [f"{v:g}" for v in vi])
    ax.set(xlabel=names[1], ylabel=names[0], title=f"{name} ESR over {names[0]} x {names[1]}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="ESR")
    fig.tight_layout()
    return fig


def fig_drive_freq_error(drives: np.ndarray, freqs: np.ndarray, err_db: np.ndarray,
                         *, control_name: str = "drive", name: str = "") -> Figure:
    """Heatmap of CIRCE-vs-circuit spectral error (dB) across control x frequency."""
    dv = np.asarray(drives, dtype=np.float64)
    fv = np.asarray(freqs, dtype=np.float64)
    err = np.asarray(err_db, dtype=np.float64)  # (n_drives, n_freqs)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(err, aspect="auto", origin="lower", cmap=CMAP_SPEC,
                   extent=(float(fv[0]), float(fv[-1]), 0.0, float(len(dv))))
    ax.set_yticks(np.arange(len(dv)) + 0.5, [f"{d * 1000:.0f}" for d in dv])
    ax.set(xlabel="frequency (Hz)", ylabel=f"{control_name} (mV)",
           title=f"{name} |CIRCE - circuit| magnitude error (dB)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="error (dB)")
    fig.tight_layout()
    return fig
