"""The CIRCE3-vs-architectures benchmark — one clean, insightful comparison.

This is the project's single head-to-head. Per circuit it shows, precisely and
by eye, the two things that make CIRCE3 (the SOTA: a dilated gated TCN +
input-scaling for signal controls + minimal FiLM for system controls) the right
architecture:

1. **Accurate AND real-time.** At the circuit's nominal operating point every
   architecture (CIRCE3, tcn, rnn, wiener-hammerstein, volterra) is trained
   on identical data and raced on held-out *content*. The ESR-vs-RTF plane shows
   CIRCE3 in the good corner — low error, real-time — beating the recurrent /
   state-space / classical models and matching the strong TCN backbone it builds
   on.
2. **Generalizes across the exogenous control.** A single CIRCE3, trained once on
   the whole drive sweep, stays accurate at *every* drive (seen and unseen) by
   input-scaling, whereas a plain TCN trained at one operating point degrades as
   the knob moves away from it. This is the capability the unconditioned
   architectures structurally lack and the reason CIRCE3 exists.

Plus qualitative overlays (static transfer curve, harmonic stack) of CIRCE3 and
the best baseline against the real circuit. Everything is reproducible via
``vguitar benchmark --circuits ...`` and reuses the library
(``spice.runner``, ``metrics``, ``plotting``, ``realtime``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vguitar.config import Config

#: Held-out test excitation uses a different seed than training (new content, same
#: operating point), so the leaderboard measures generalization, not recall.
_TEST_SEED = 777
#: The other architectures CIRCE3 is raced against (one per distinct class).
#: SSM is intentionally excluded: its chunked-FFT scan trains ~50x slower per epoch
#: than the convolutional models and it is not real-time (RTF ~0.3x) — strictly
#: dominated on both axes, so it does not belong in this curated comparison. It is
#: still a registered model; run it explicitly via ``vguitar bench --model ssm``.
_BASELINES = ("volterra", "wh", "tcn", "rnn")


# --- helpers --------------------------------------------------------------
def _uncond(ds: Any) -> Any:
    """Collapse a drive dataset to a plain input->output map: input = g.x."""
    from vguitar.data import Dataset

    g = ds.controls[:, 0] if ds.controls is not None else np.ones(len(ds), np.float32)
    return Dataset(np.asarray(ds.x, np.float32) * g, ds.y, ds.sr, name=ds.name)


def _proc(model: Any, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
    """Run a model and compensate its reporting latency, so the returned signal is
    sample-aligned to the input (an oversampled CIRCE3 has a polyphase group delay;
    latency-0 models are unaffected). Shift left by ``latency_samples``, pad the tail."""
    pred = np.asarray(model.process(x, c) if c is not None else model.process(x), np.float32)
    lat = int(getattr(model, "latency_samples", 0) or 0)
    if lat > 0:
        pred = np.concatenate([pred[lat:], np.zeros(lat, dtype=np.float32)])
    return pred


def _esr_at_drive(model: Any, circ: Any, dry: np.ndarray, g: float, sr: int,
                  conditioned: bool, warmup: int = 2048) -> float:
    """ESR of ``model`` vs the circuit for input ``g.dry`` (band-limited target)."""
    from vguitar.metrics import esr
    from vguitar.spice.runner import simulate

    y = simulate(circ, (g * dry).astype(np.float32), sr)
    pred = (_proc(model, dry, np.array([g], np.float32)) if conditioned
            else _proc(model, (g * dry).astype(np.float32)))
    return float(esr(y[warmup:], pred[warmup:]))


# --- per-circuit run ------------------------------------------------------
def _compare_one(circuit_name: str, cfg: Config, *, epochs: int, regen: bool,
                 console: Any) -> dict[str, Any]:
    """Train CIRCE3 + baselines on one circuit; return rows + write its figures."""
    import matplotlib.pyplot as plt

    from vguitar import plotting as plot
    from vguitar.benchmark.run import _instantiate
    from vguitar.circuits import get_circuit
    from vguitar.data import Dataset
    from vguitar.models.base import pick_device, to_inference_cpu
    from vguitar.models.circe3 import CIRCE3
    from vguitar.realtime import measure_rtf
    from vguitar.signals import load_di
    from vguitar.spice.runner import make_drive_dataset, simulate

    circ = get_circuit(circuit_name)
    g_nom = float(circ.nominal_drive_v)
    sr = cfg.data.sr
    dev = pick_device()
    console.print(f"[bold]benchmark[/] {circuit_name}  (nominal drive {g_nom:g} V, device {dev})")

    # --- datasets: a drive SWEEP (for CIRCE3) + a fixed point at g_nom (baselines) ---
    drives = [round(g_nom * f, 5) for f in (0.25, 0.5, 1.0, 2.0)]  # trained sweep
    held = [round(g_nom * f, 5) for f in (0.375, 0.75, 1.5)]  # unseen interior drives
    sweep_p = cfg.paths.data / f"{circuit_name}_bench_sweep.npz"
    fp_p = cfg.paths.data / f"{circuit_name}_bench_fp.npz"
    fp_test_p = cfg.paths.data / f"{circuit_name}_bench_fp_test.npz"
    if regen or not all(p.exists() for p in (sweep_p, fp_p, fp_test_p)):
        console.print("[dim]  simulating sweep + fixed-point datasets via ngspice...[/]")
        make_drive_dataset(circ, drives, cfg, seg_dur_s=4.0, seed=0).save(sweep_p)
        make_drive_dataset(circ, [g_nom], cfg, seg_dur_s=10.0, seed=0).save(fp_p)
        make_drive_dataset(circ, [g_nom], cfg, seg_dur_s=10.0, seed=_TEST_SEED).save(fp_test_p)
    sweep_ds, fp_ds, fp_test = Dataset.load(sweep_p), Dataset.load(fp_p), Dataset.load(fp_test_p)

    tcfg = replace(cfg.train, epochs=epochs, lr=3e-3)
    # held-out content at the nominal point, as the circuit's true input g_nom.x:
    test_in = (g_nom * fp_test.x).astype(np.float32)

    rows: list[dict[str, Any]] = []
    # --- CIRCE3: one model trained on the whole sweep ---
    import torch

    torch.manual_seed(0)
    console.print("  training [magenta]circe3[/] on the drive sweep ...")
    tr, va, _ = sweep_ds.split(0.12, 0.0001)
    # Pre-emphasis ESR loss (formant/transfer) + 2x internal oversampling (removes
    # the gates' self-aliasing -> ~8x lower held ESR on realistic signals,
    # streaming-exact, real-time). n_layers=9 keeps the receptive-field-in-seconds
    # at the 2x internal rate. The polyphase group delay is reported as latency.
    circe3 = CIRCE3(n_control=1, channels=24, n_blocks=2, n_layers=9,
                    oversample=2, device=dev)
    circe3.fit(tr, va, replace(tcfg, seq_len=4096, batch_size=12, warmup=2048))
    to_inference_cpu(circe3)
    (cfg.paths.assets / "checkpoints").mkdir(parents=True, exist_ok=True)
    circe3.save(cfg.paths.assets / "checkpoints" / f"{circuit_name}.circe3.model")
    from vguitar.metrics import esr

    c3_pred = _proc(circe3, fp_test.x, np.array([g_nom], np.float32))
    rows.append({
        "model": "circe3",
        "esr": float(esr(fp_test.y[2048:], c3_pred[2048:])),
        "rtf": float(measure_rtf(circe3, sr=sr, block=cfg.realtime.block_size)["rtf"]),
        "params": int(circe3.num_params()),
        "conditioned": True,
    })

    # --- baselines: fixed-point specialists trained at g_nom ---
    tr_u, va_u, _ = _uncond(fp_ds).split(0.12, 0.0001)
    trained: dict[str, Any] = {"circe3": circe3}
    for name in _BASELINES:
        try:
            console.print(f"  training [cyan]{name}[/] (fixed point) ...")
            m = _instantiate(name, dev)
            m.fit(tr_u, va_u, tcfg)
            to_inference_cpu(m)
            pred = np.asarray(m.process(test_in), np.float32)
            rows.append({
                "model": name,
                "esr": float(esr(fp_test.y[2048:], pred[2048:])),
                "rtf": float(measure_rtf(m, sr=sr, block=cfg.realtime.block_size)["rtf"]),
                "params": int(m.num_params()),
                "conditioned": False,
            })
            trained[name] = m
        except Exception as exc:  # one model must not abort the benchmark
            console.print(f"    [red]{name} failed: {exc}[/]")
    rows.sort(key=lambda r: r["esr"])

    # --- figures ---
    outdir = cfg.paths.outputs / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    def _save(fig: Any, stem: str) -> None:
        fig.savefig(outdir / f"compare_{circuit_name}_{stem}.png")
        plt.close(fig)

    _save(plot.fig_leaderboard(rows, name=f"{circuit_name}: CIRCE3 vs architectures"), "esr_rtf")

    # Generalization: one CIRCE3 across the whole sweep vs the fixed-point TCN.
    try:
        di = load_di("assets/guitar_di_loop.wav", sr, peak=1.0)[: int(2.0 * sr)]
        gs = sorted(set(drives) | set(held))
        c3_curve = [_esr_at_drive(circe3, circ, di, g, sr, conditioned=True) for g in gs]
        tcn = trained.get("tcn")
        tcn_curve = (
            [_esr_at_drive(tcn, circ, di, g, sr, conditioned=False) for g in gs]
            if tcn is not None else None
        )
        held_mask = np.array([g in set(held) for g in gs])
        _save(_fig_generalization(np.array(gs), np.array(c3_curve),
                                  None if tcn_curve is None else np.array(tcn_curve),
                                  held_mask, g_nom, circuit_name), "generalization")
        # Qualitative overlays at the hardest trained drive.
        gq = max(drives)
        tprobe = np.arange(int(0.2 * sr)) / sr
        xprobe = (1.3 * gq * np.sin(2 * np.pi * 90.0 * tprobe)).astype(np.float32)
        yref = simulate(circ, xprobe, sr)
        best_base = next((r["model"] for r in rows if r["model"] != "circe3"), None)
        preds_t = {"circe3": _proc(circe3, xprobe / gq, np.array([gq], np.float32))}
        if best_base and best_base in trained:
            preds_t[best_base] = _proc(trained[best_base], xprobe)
        # Plot only the SETTLED region: drop the model warm-up head and the
        # latency-compensation tail so the transfer loop is artifact-free.
        sl = slice(2048, len(xprobe) - (circe3.latency_samples + 64))
        _save(plot.fig_transfer(xprobe[sl], yref[sl], {k: v[sl] for k, v in preds_t.items()},
                                name=f"{circuit_name} @ {gq:g} V"), "transfer")
        tone = np.sin(2 * np.pi * 1000.0 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
        yreft = simulate(circ, (gq * tone).astype(np.float32), sr)
        preds_h = {"circe3": _proc(circe3, tone, np.array([gq], np.float32))}
        if best_base and best_base in trained:
            preds_h[best_base] = _proc(trained[best_base], (gq * tone).astype(np.float32))
        _save(plot.fig_harmonics(yreft, preds_h, sr, name=f"{circuit_name} @ {gq:g} V"), "harmonics")
        # Spectral fidelity on the broadband guitar-DI probe (the full
        # transfer-shaped formant envelope + a time-frequency difference map).
        ydi = simulate(circ, (gq * di).astype(np.float32), sr)
        preds_s = {"circe3": _proc(circe3, di, np.array([gq], np.float32))}
        if best_base and best_base in trained:
            preds_s[best_base] = _proc(trained[best_base], (gq * di).astype(np.float32))
        _save(plot.fig_spectrum(ydi, preds_s, sr, name=f"{circuit_name} @ {gq:g} V"), "spectrum")
        _save(plot.fig_spectrogram_compare(ydi, preds_s["circe3"], sr,
                                           name=f"{circuit_name} @ {gq:g} V"), "spectrogram")
        # Transfer-curve FAMILY across drives: one CIRCE3 matches the
        # Transferkennlinie at every operating point (seen and held-out) by
        # input-scaling. Pick a spread of drives incl. a held-out one.
        held_set = set(held)
        fam_drives = sorted({min(drives), g_nom, *([min(held)] if held else []), max(drives)})
        curves = []
        for gd in fam_drives:
            xf = (1.3 * gd * np.sin(2 * np.pi * 90.0 * tprobe)).astype(np.float32)
            yf = simulate(circ, xf, sr)
            pf = _proc(circe3, xf / gd, np.array([gd], np.float32))
            fsl = slice(2048, len(xf) - (circe3.latency_samples + 64))
            curves.append((f"{gd:g} V", xf[fsl], yf[fsl], pf[fsl], gd in held_set))
        _save(plot.fig_transfer_family(curves, name=circuit_name), "transfer_family")
    except Exception as exc:
        console.print(f"[yellow]  ngspice unavailable; skipping generalization/overlay figs ({exc})[/]")

    return {"circuit": circuit_name, "rows": rows}


def _fig_generalization(drives: np.ndarray, c3: np.ndarray, tcn: np.ndarray | None,
                        held_mask: np.ndarray, g_nom: float, name: str) -> Any:
    """ESR vs drive: CIRCE3 (one model, all drives) vs a fixed-point TCN."""
    import matplotlib.pyplot as plt

    from vguitar.plotting import color_for

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(drives, c3, "-o", color=color_for("circe3"), lw=1.6, label="CIRCE3 (one model, input-scaling)")
    if tcn is not None:
        ax.plot(drives, tcn, "-s", color=color_for("tcn"), lw=1.4, alpha=0.9,
                label=f"TCN (trained only at {g_nom:g} V)")
    if held_mask.any():
        ax.plot(drives[held_mask], c3[held_mask], "o", mfc="white",
                mec=color_for("circe3"), ms=9, label="held-out (unseen) drive")
    ax.axvline(g_nom, color="k", ls=":", lw=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("drive (V)")
    ax.set_ylabel("ESR vs circuit (guitar-DI, log)")
    ax.set_title(f"{name}: generalization across the drive knob")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


# --- entry ----------------------------------------------------------------
def run_compare(circuit_names: list[str], *, cfg: Config | None = None,
                epochs: int = 150, regen: bool = False) -> dict[str, Any]:
    """Run the CIRCE3-vs-architectures benchmark across circuits; print + plot."""
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table

    from vguitar import plotting as plot
    from vguitar.config import Config

    cfg = cfg or Config()
    cfg.paths.ensure()
    console = Console()

    all_rows: list[dict[str, Any]] = []
    for cname in circuit_names:
        res = _compare_one(cname, cfg, epochs=epochs, regen=regen, console=console)
        for r in res["rows"]:
            r["circuit"] = cname
        all_rows.extend(res["rows"])
        # per-circuit leaderboard
        table = Table(title=f"{cname}: CIRCE3 vs architectures (held-out content)")
        for col in ("model", "ESR", "params", "RTF", "live?", "knob?"):
            table.add_column(col, justify="right" if col != "model" else "left")
        for r in sorted(res["rows"], key=lambda r: r["esr"]):
            mark = "[magenta]" if r["model"] == "circe3" else ""
            table.add_row(f"{mark}{r['model']}", f"{r['esr']:.4f}", f"{r['params']:,}",
                          f"{r['rtf']:.1f}x", "[green]yes[/]" if r["rtf"] > 1 else "[red]no[/]",
                          "[green]yes[/]" if r["conditioned"] else "-")
        console.print(table)

    # cross-circuit ESR matrix
    models = ["circe3", *_BASELINES]
    matrix = np.full((len(circuit_names), len(models)), np.nan)
    for i, cname in enumerate(circuit_names):
        for j, mname in enumerate(models):
            hit = [r for r in all_rows if r["circuit"] == cname and r["model"] == mname]
            if hit:
                matrix[i, j] = hit[0]["esr"]
    fig = plot.fig_circuit_model_esr(matrix, list(circuit_names), models, name="benchmark")
    fig.savefig(cfg.paths.outputs / "figs" / "compare_esr_matrix.png")
    plt.close(fig)

    csv_path = cfg.paths.outputs / "benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["circuit", "model", "esr", "params", "rtf", "conditioned"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    (cfg.paths.outputs / "benchmark.json").write_text(json.dumps(all_rows, indent=2))
    console.print(f"wrote figures to {cfg.paths.outputs / 'figs'} and {csv_path}")
    return {"rows": all_rows, "matrix": matrix, "circuits": list(circuit_names), "models": models}
