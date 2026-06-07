"""Fixed-operating-point shootout: CIRCE vs the unconditioned baselines.

A clean, apples-to-apples head-to-head. Per circuit, at its **nominal operating
point**, every method is trained and tested on the *same* data and the *same*
held-out signal: the circuit input is identical for all models, and CIRCE
additionally receives the operating point as a control (a constant at one point).
Targets are byte-identical, so ESR/THD/RTF are directly comparable.

It produces the artifacts that make the comparison verifiable by eye and ear:

* a per-circuit **leaderboard** (ESR / multi-STFT / THD / params / RTF) + the
  ESR-vs-RTF Pareto scatter;
* **overlay figures** — static transfer curve, harmonic stack, and a clipping
  waveform with residuals — every model drawn against the circuit (black);
* **A/B audio** — dry DI, the circuit, and each model rendered to wav; and
* a **cross-circuit ESR matrix**.

CIRCE's unique value (turning the knob / interpolating to unseen settings) is a
separate report: `vguitar circe`. Reproducible via `vguitar shootout`.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from vguitar.benchmark.circe_eval import _DI_PATH, _thd_signal, sf_write

if TYPE_CHECKING:
    from vguitar.config import Config

#: Held-out test excitation uses a different seed than training (new content,
#: same operating point) so the leaderboard measures generalization, not recall.
_BENCH_TEST_SEED = 777
#: The unconditioned baselines we race CIRCE against (SSM excluded — slow + weak).
_DEFAULT_BASELINES = ("fir", "volterra", "volterra_pc", "wh", "tcn", "rnn")


# --- pure helpers (unit-testable) -----------------------------------------
def _uncond(ds: Any) -> Any:
    """The unconditioned view of a drive dataset: input = the actual circuit input.

    The drive control is a pre-gain, so the voltage the circuit (and an
    unconditioned model) actually sees is ``control * dry``. This collapses the
    conditioned dataset to a plain input->output map with the SAME targets.
    """
    from vguitar.data import Dataset

    g = ds.controls[:, 0] if ds.controls is not None else np.ones(len(ds), np.float32)
    return Dataset(np.asarray(ds.x, np.float32) * g, ds.y, ds.sr, name=ds.name)


def _predict(model: Any, circuit_input: np.ndarray, g: float, conditioned: bool) -> np.ndarray:
    """Predict the output for a given **circuit input**, per the model's parameterization.

    Unconditioned models map the input voltage directly. CIRCE maps ``(dry, g)``
    where ``g*dry`` is the circuit input, so its dry input is ``circuit_input / g``
    at control ``g`` — yielding the same target the baselines are asked for.
    """
    ci = np.ascontiguousarray(circuit_input, dtype=np.float32)
    if conditioned:
        return np.asarray(model.process(ci / g, np.array([g], np.float32)), dtype=np.float32)
    return np.asarray(model.process(ci), dtype=np.float32)


def _seg_metrics(model: Any, test_ds: Any, conditioned: bool, warmup: int = 512) -> tuple[float, float]:
    """Mean per-segment (ESR, multi-STFT) on the held-out test set."""
    from vguitar.metrics import esr, multi_stft
    from vguitar.models.circe import _segments

    ctrl = test_ds.controls
    segs = _segments(ctrl) if ctrl is not None else [(0, len(test_ds))]
    errs, stfts = [], []
    for s, e in segs:
        g = float(ctrl[s, 0]) if ctrl is not None else 1.0
        pred = _predict(model, np.asarray(test_ds.x[s:e], np.float32) * g, g, conditioned)
        w = min(warmup, (e - s) - 1)
        errs.append(float(esr(test_ds.y[s:e][w:], pred[w:])))
        stfts.append(float(multi_stft(test_ds.y[s:e][w:], pred[w:])))
    return float(np.mean(errs)), float(np.mean(stfts))


# --- per-circuit shootout -------------------------------------------------
def _shootout_one(
    circuit_name: str,
    baselines: tuple[str, ...],
    cfg: Config,
    *,
    epochs: int,
    seg_dur_s: float,
    retrain: bool,
    regen: bool,
    di_dur_s: float,
    console: Any,
) -> list[dict[str, Any]]:
    """Train + evaluate every method on one circuit; write figures + audio; return rows."""
    import matplotlib.pyplot as plt

    from vguitar import plotting as plot
    from vguitar.benchmark.run import _instantiate
    from vguitar.circuits import get_circuit
    from vguitar.data import Dataset
    from vguitar.models import get_model
    from vguitar.models.base import pick_device, to_inference_cpu
    from vguitar.models.circe import CIRCE
    from vguitar.realtime import measure_rtf
    from vguitar.spice.runner import make_drive_dataset, simulate

    circ = get_circuit(circuit_name)
    g_nom = float(circ.nominal_drive_v)
    sr = cfg.data.sr
    # Same gentle lr for every neural model (the 5e-3 default diverges the big
    # feedforward/recurrent nets on the hard circuits); fair shared recipe.
    train_cfg = replace(cfg.train, epochs=epochs, lr=3e-3)
    console.print(f"[bold]shootout[/] {circuit_name} @ drive={g_nom:g} V")

    # --- datasets: one operating point, train + held-out (new content) ---
    tr_path = cfg.paths.data / f"{circuit_name}_shootout_train.npz"
    te_path = cfg.paths.data / f"{circuit_name}_shootout_test.npz"
    if regen or not tr_path.exists() or not te_path.exists():
        console.print("[dim]  simulating train/test via ngspice...[/]")
        make_drive_dataset(circ, [g_nom], cfg, seg_dur_s=seg_dur_s, seed=0).save(tr_path)
        make_drive_dataset(circ, [g_nom], cfg, seg_dur_s=seg_dur_s, seed=_BENCH_TEST_SEED).save(te_path)
    train_ds, test_ds = Dataset.load(tr_path), Dataset.load(te_path)
    tr, va, _ = train_ds.split(0.12, 0.0001)

    # --- train every method on identical data (GPU if available; infer on CPU) ---
    dev = pick_device()
    console.print(f"  training device: [bold]{dev}[/] (inference/RTF measured on CPU)")
    trained: dict[str, tuple[Any, bool]] = {}
    for name in baselines:
        cls = get_model(name)
        if cls.conditioned:
            continue
        try:
            console.print(f"  training [cyan]{name}[/] ...")
            m = _instantiate(name, dev)  # passes device to ctors that accept it
            m.fit(_uncond(tr), _uncond(va), train_cfg)
            trained[name] = (to_inference_cpu(m), False)  # CPU for honest RTF/streaming
        except Exception as exc:  # a model must not abort the shootout
            console.print(f"    [red]{name} failed: {exc}[/]")
    try:
        import torch

        torch.manual_seed(0)
        console.print("  training [magenta]circe[/] ...")
        mc = CIRCE(n_control=1, channels=24, n_blocks=2, n_layers=8, stft_weight=0.2, device=dev)
        mc.fit(tr, va, replace(train_cfg, seq_len=2048, batch_size=16, lr=3e-3, warmup=256))
        trained["circe"] = (to_inference_cpu(mc), True)
    except Exception as exc:
        console.print(f"    [red]circe failed: {exc}[/]")

    # --- metrics (ESR/STFT on held-out test; THD at nominal; RTF) ---
    tone = np.sin(2 * np.pi * 1000.0 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
    rows: list[dict[str, Any]] = []
    preds_tone: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}
    for name, (m, cond) in trained.items():
        e, st = _seg_metrics(m, test_ds, cond)
        rtf = measure_rtf(m, sr=sr, block=cfg.realtime.block_size)
        preds_tone[name] = _predict(m, g_nom * tone, g_nom, cond)
        preds_test[name] = _predict(m, np.asarray(test_ds.x, np.float32) * g_nom, g_nom, cond)
        rows.append({
            "circuit": circuit_name, "model": name, "esr": e, "stft": st,
            "thd": _thd_signal(preds_tone[name], sr), "rtf": float(rtf["rtf"]),
            "realtime": bool(rtf["realtime"]), "params": int(m.num_params()),
        })
    rows.sort(key=lambda r: r["esr"])

    # --- circuit reference + overlay figures + audio (ngspice) ---
    outdir = cfg.paths.outputs / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    def _save(fig: Any, stem: str) -> None:
        fig.savefig(outdir / f"shootout_{circuit_name}_{stem}.png")
        plt.close(fig)

    _save(plot.fig_leaderboard(rows, name=f"{circuit_name} @ drive={g_nom:g}V"), "leaderboard")
    try:
        tone_circuit = simulate(circ, (g_nom * tone).astype(np.float32), sr)
        ref_thd = _thd_signal(tone_circuit, sr)
        for r in rows:
            r["thd_circuit"] = ref_thd
        _save(plot.fig_harmonics(tone_circuit, preds_tone, sr, name=circuit_name), "harmonics")
        _save(plot.fig_waveform(test_ds.y, preds_test, sr, name=circuit_name), "waveform")
        # static transfer: a sine in the guitar band that sweeps the clipping
        # range. Probed at 90 Hz (a real low note), ABOVE any model's output
        # DC-blocker corner, so the curve reflects the nonlinearity rather than a
        # sub-corner high-pass phase shift (a 12 Hz probe opens a spurious loop).
        tprobe = np.arange(int(0.2 * sr)) / sr
        x_probe = (1.3 * g_nom * np.sin(2 * np.pi * 90.0 * tprobe)).astype(np.float32)
        y_ref = simulate(circ, x_probe, sr)
        preds_tr = {n: _predict(m, x_probe, g_nom, c) for n, (m, c) in trained.items()}
        _save(plot.fig_transfer(x_probe, y_ref, preds_tr, name=circuit_name), "transfer")
        _render_audio(circ, trained, g_nom, sr, cfg, circuit_name, di_dur_s, console)
    except Exception as exc:
        console.print(f"[yellow]  ngspice unavailable; skipping overlay figures + audio ({exc})[/]")

    return rows


def _render_audio(circ: Any, trained: dict[str, tuple[Any, bool]], g_nom: float, sr: int,
                  cfg: Config, circuit_name: str, di_dur_s: float, console: Any) -> None:
    """Write dry / circuit / per-model wavs at the nominal operating point."""
    import soundfile as sf

    from vguitar.signals import load_di
    from vguitar.spice.runner import simulate

    di = load_di(_DI_PATH, sr, peak=0.9)[: int(di_dur_s * sr)]
    adir = cfg.paths.outputs / "audio" / "shootout" / circuit_name
    adir.mkdir(parents=True, exist_ok=True)

    def norm(y: np.ndarray) -> np.ndarray:
        p = float(np.max(np.abs(y)))
        return (0.9 * y / p).astype(np.float32) if p > 0 else np.asarray(y, np.float32)

    sf_write(sf, adir / "di.wav", di, sr)
    sf_write(sf, adir / "circuit.wav", norm(simulate(circ, (g_nom * di).astype(np.float32), sr)), sr)
    for name, (m, cond) in trained.items():
        sf_write(sf, adir / f"{name}.wav", norm(_predict(m, g_nom * di, g_nom, cond)), sr)
    console.print(f"  wrote A/B audio ({len(trained) + 2} wavs) to {adir}")


# --- entry ----------------------------------------------------------------
def run_shootout(
    circuit_names: list[str],
    *,
    models: tuple[str, ...] | None = None,
    cfg: Config | None = None,
    epochs: int = 50,
    seg_dur_s: float = 10.0,
    di_dur_s: float = 2.5,
    retrain: bool = False,
    regen: bool = False,
) -> dict[str, Any]:
    """Run the fixed-point shootout across circuits; aggregate + plot the comparison."""
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table

    from vguitar import plotting as plot
    from vguitar.config import Config

    cfg = cfg or Config()
    cfg.paths.ensure()
    console = Console()
    baselines = tuple(m for m in (models or _DEFAULT_BASELINES) if m not in ("ssm", "circe"))

    all_rows: list[dict[str, Any]] = []
    for cname in circuit_names:
        all_rows.extend(_shootout_one(
            cname, baselines, cfg, epochs=epochs, seg_dur_s=seg_dur_s,
            retrain=retrain, regen=regen, di_dur_s=di_dur_s, console=console,
        ))

    # --- per-circuit leaderboard tables ---
    model_order = [*baselines, "circe"]
    for cname in circuit_names:
        crows = sorted((r for r in all_rows if r["circuit"] == cname), key=lambda r: r["esr"])
        if not crows:
            continue
        ref = crows[0].get("thd_circuit")
        title = f"shootout {cname}" + (f"  (circuit THD@1k={ref:.3f})" if ref is not None else "")
        table = Table(title=title)
        for col in ("model", "ESR", "STFT", "THD", "params", "RTF", "live?"):
            table.add_column(col, justify="right" if col != "model" else "left")
        for r in crows:
            mark = "[magenta]" if r["model"] == "circe" else ""
            table.add_row(
                f"{mark}{r['model']}", f"{r['esr']:.4f}", f"{r['stft']:.3f}", f"{r['thd']:.3f}",
                f"{r['params']:,}", f"{r['rtf']:.1f}x",
                "[green]yes[/]" if r["realtime"] else "[red]no[/]",
            )
        console.print(table)

    # --- cross-circuit ESR matrix ---
    matrix = np.full((len(circuit_names), len(model_order)), np.nan, dtype=np.float64)
    for i, cname in enumerate(circuit_names):
        for j, mname in enumerate(model_order):
            hit = [r for r in all_rows if r["circuit"] == cname and r["model"] == mname]
            if hit:
                matrix[i, j] = hit[0]["esr"]
    outdir = cfg.paths.outputs / "figs"
    fig = plot.fig_circuit_model_esr(matrix, list(circuit_names), model_order, name="shootout")
    fig.savefig(outdir / "shootout_circuit_model_esr.png")
    plt.close(fig)

    # --- combined CSV ---
    csv_path = cfg.paths.outputs / "shootout.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["circuit", "model", "esr", "stft", "thd", "thd_circuit",
                           "params", "rtf", "realtime"], extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    console.print(f"wrote leaderboards + figures to {outdir} and {csv_path}")

    return {"rows": all_rows, "matrix": matrix, "circuits": list(circuit_names), "models": model_order}
