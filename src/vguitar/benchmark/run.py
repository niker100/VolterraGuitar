"""Benchmark harness: train every model on a circuit and compare them fairly.

This is the project's whole point — put Volterra, block-oriented, and neural
emulators side by side on the *same* circuit data with the *same* metrics, so
the accuracy/latency tradeoff is visible at a glance.

For each model we report:

* **ESR** — error-to-signal ratio on the held-out test set (primary; lower is
  better). Plus **MSE** and a multi-resolution **STFT** distance.
* **THD** — total harmonic distortion at 1 kHz, compared against the circuit's
  own THD (does the model reproduce the nonlinearity?).
* **params** — model size; **latency** — algorithmic lookahead (samples).
* **RTF** — real-time factor (audio seconds produced per compute second);
  ``RTF > 1`` means it can run live at the configured block size.

A model that raises is recorded and skipped, never aborting the whole run.
"""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from vguitar import metrics
from vguitar.circuits import get_circuit
from vguitar.config import Config
from vguitar.data import Dataset
from vguitar.models import all_models, get_model
from vguitar.realtime import measure_rtf


def _seed_everything(seed: int) -> None:
    """Make a run reproducible. Models seed their own ``default_rng``; here we
    also pin torch (neural models) and legacy global numpy state as a backstop."""
    np.random.seed(seed)  # noqa: NPY002 - global backstop for any legacy np.random use
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _load_or_make_dataset(circuit_name: str, cfg: Config) -> Dataset:
    """Return a cached dataset for ``circuit_name`` or simulate and cache one."""
    path = cfg.paths.data / f"{circuit_name}.npz"
    if path.exists():
        return Dataset.load(path)
    from vguitar.spice import make_dataset  # lazy: needs ngspice

    ds = make_dataset(get_circuit(circuit_name), cfg)
    ds.save(path)
    return ds


def _instantiate(name: str, device: str) -> Any:
    """Construct a registered model, passing ``device`` only if it accepts it."""
    cls: Any = get_model(name)  # heterogeneous ctors across model classes
    if "device" in inspect.signature(cls.__init__).parameters:
        return cls(device=device)
    return cls()


def _evaluate(
    name: str, model: Any, train: Dataset, val: Dataset, test: Dataset, cfg: Config
) -> dict[str, Any]:
    """Fit one model and measure accuracy, size, latency, and speed."""
    model.fit(train, val, cfg.train)
    y_pred = np.asarray(model.process(test.x), dtype=np.float32)
    rtf = measure_rtf(model, sr=cfg.realtime.sr, block=cfg.realtime.block_size)
    return {
        "model": name,
        "esr": metrics.esr(test.y, y_pred),
        "mse": metrics.mse(test.y, y_pred),
        "stft": metrics.multi_stft(test.y, y_pred),
        "thd": metrics.thd(model.process, sr=test.sr),
        "params": int(model.num_params()),
        "latency": int(model.latency_samples),
        "rtf": float(rtf["rtf"]),
        "realtime": bool(rtf["realtime"]),
    }


def run_benchmark(
    circuit_name: str = "bjt",
    model_names: list[str] | None = None,
    cfg: Config | None = None,
    dataset: Dataset | None = None,
) -> list[dict[str, Any]]:
    """Train and compare models on one circuit; print + save a leaderboard.

    Args:
        circuit_name: registered circuit (default ``"bjt"``).
        model_names: subset of registered models, or ``None`` for all.
        cfg: pipeline config (defaults to :class:`Config`).
        dataset: pre-built dataset to use instead of loading/simulating.

    Returns:
        One result dict per model (sorted best-ESR-first; errored models last
        with an ``"error"`` key).
    """
    cfg = cfg or Config()
    cfg.paths.ensure()
    _seed_everything(cfg.train.seed)
    console = Console()

    ds = dataset or _load_or_make_dataset(circuit_name, cfg)
    train, val, test = ds.split(cfg.train.val_fraction, cfg.train.test_fraction)
    console.print(
        f"[bold]circuit[/] {circuit_name}  "
        f"[dim]({ds.duration_s:.1f}s @ {ds.sr} Hz; "
        f"train/val/test = {len(train)}/{len(val)}/{len(test)})[/]"
    )
    # Reference: THD of the actual circuit output on the test tone region.
    ref_thd = metrics.thd(lambda _x: test.y[: len(_x)], sr=test.sr) if len(test) else float("nan")

    names = model_names or sorted(all_models())
    rows: list[dict[str, Any]] = []
    for name in names:
        console.print(f"  training [cyan]{name}[/] ...")
        try:
            model = _instantiate(name, cfg.train.device)
            row = _evaluate(name, model, train, val, test, cfg)
            with _suppress():
                model.save(cfg.paths.runs / f"{circuit_name}_{name}")
            with _suppress():
                metrics.plot_compare(
                    test.y,
                    np.asarray(model.process(test.x), dtype=np.float32),
                    test.sr,
                    cfg.paths.outputs / f"{circuit_name}_{name}.png",
                )
        except Exception as exc:  # one model must not abort the whole suite
            console.print(f"    [red]{name} failed: {exc}[/]")
            row = {"model": name, "error": str(exc)}
        rows.append(row)

    ok = sorted((r for r in rows if "error" not in r), key=lambda r: r["esr"])
    failed = [r for r in rows if "error" in r]
    _print_table(console, circuit_name, ref_thd, ok, failed)
    _write_csv(cfg.paths.outputs / f"{circuit_name}_benchmark.csv", ok + failed)
    with _suppress():
        _scatter(ok, cfg.paths.outputs / f"{circuit_name}_esr_vs_rtf.png", circuit_name)
    return ok + failed


# --- reporting helpers ----------------------------------------------------
class _suppress:
    """Best-effort context manager (plots/saves shouldn't fail the benchmark)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True  # swallow everything


def _print_table(
    console: Console,
    circuit: str,
    ref_thd: float,
    ok: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    table = Table(title=f"vguitar benchmark — {circuit}  (circuit THD@1k = {ref_thd:.3f})")
    for col in ("model", "ESR", "MSE", "STFT", "THD", "params", "latency", "RTF", "live?"):
        table.add_column(col, justify="right" if col != "model" else "left")
    for r in ok:
        live = "[green]yes[/]" if r["realtime"] else "[red]no[/]"
        table.add_row(
            r["model"],
            f"{r['esr']:.4f}",
            f"{r['mse']:.2e}",
            f"{r['stft']:.3f}",
            f"{r['thd']:.3f}",
            f"{r['params']:,}",
            str(r["latency"]),
            f"{r['rtf']:.1f}x",
            live,
        )
    for r in failed:
        table.add_row(r["model"], "[red]ERROR[/]", "", "", "", "", "", "", "")
    console.print(table)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = ["model", "esr", "mse", "stft", "thd", "params", "latency", "rtf", "realtime", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _scatter(ok: list[dict[str, Any]], path: Path, circuit: str) -> None:
    if not ok:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in ok:
        ax.scatter(r["rtf"], r["esr"], s=60)
        ax.annotate(r["model"], (r["rtf"], r["esr"]), textcoords="offset points", xytext=(6, 4))
    ax.axvline(1.0, color="red", ls="--", lw=1, label="real-time threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("real-time factor (>1 = live-capable)")
    ax.set_ylabel("ESR (lower = more accurate)")
    ax.set_title(f"accuracy vs. speed — {circuit}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
