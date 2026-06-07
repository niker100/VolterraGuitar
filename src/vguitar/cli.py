"""Command-line front end: ``vguitar <subcommand>``.

A thin argparse dispatcher over the package's contracts. Each subcommand is a
small handler that wires existing pieces together (circuit/model registries, the
SPICE runner, the benchmark, the live engine) and returns a process exit code.

Design choices:

* ``main`` only parses and dispatches; every subcommand is its own function so
  the control flow reads top-down and stays testable.
* Heavy or optional dependencies (torch via the model registry, matplotlib, the
  benchmark module, the realtime engine, ngspice) are imported lazily *inside*
  the handler that needs them. So ``vguitar list`` and ``--help`` stay instant,
  and a missing optional piece produces a friendly message instead of an import
  traceback at startup.
* Defaults come straight from :class:`vguitar.config.Config`; flags only
  override the few fields a user typically tweaks (duration, seed, epochs).

Subcommands: ``list``, ``gen``, ``train``, ``bench``, ``live``, ``selftest``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vguitar.config import Config

if TYPE_CHECKING:
    from vguitar.data import Dataset
    from vguitar.models.base import Model


# --- shared helpers -------------------------------------------------------
def _dataset_path(cfg: Config, circuit: str) -> Path:
    """Canonical cache path for a circuit's generated dataset."""
    return cfg.paths.data / f"{circuit}.npz"


def _run_path(cfg: Config, circuit: str, model: str) -> Path:
    """Canonical save path for a trained ``model`` on ``circuit``."""
    return cfg.paths.runs / f"{circuit}.{model}.model"


def _fail(msg: str) -> int:
    """Print a friendly error to stderr and return a non-zero exit code."""
    print(f"error: {msg}", file=sys.stderr)
    return 1


def _load_or_make_dataset(cfg: Config, circuit_name: str, *, make: bool) -> Dataset:
    """Load the cached dataset for ``circuit_name`` or generate (and cache) it.

    Generation runs ngspice via :func:`vguitar.spice.make_dataset`; if ``make``
    is False and no cache exists we raise so the caller can print guidance.
    """
    from vguitar.circuits.base import get_circuit
    from vguitar.data import Dataset
    from vguitar.spice import make_dataset

    path = _dataset_path(cfg, circuit_name)
    if path.exists():
        return Dataset.load(path)
    if not make:
        raise FileNotFoundError(
            f"no dataset at {path}; run 'vguitar gen --circuit {circuit_name}' first"
        )
    circuit = get_circuit(circuit_name)
    ds = make_dataset(circuit, cfg)
    ds.save(path)
    return ds


# --- list -----------------------------------------------------------------
def cmd_list(args: argparse.Namespace, cfg: Config) -> int:
    """Print every registered circuit and model with its description."""
    from rich.console import Console
    from rich.table import Table

    from vguitar.circuits.base import all_circuits
    from vguitar.models.base import all_models

    console = Console()
    for title, registry in (("Circuits", all_circuits()), ("Models", all_models())):
        table = Table(title=title)
        table.add_column("name", style="bold cyan")
        table.add_column("description")
        for name, cls in sorted(registry.items()):
            table.add_row(name, getattr(cls, "description", "") or "")
        console.print(table)
    return 0


# --- gen ------------------------------------------------------------------
def cmd_gen(args: argparse.Namespace, cfg: Config) -> int:
    """Generate (and cache) the dataset for a circuit; save a preview plot."""
    from dataclasses import replace

    from vguitar.circuits.base import get_circuit
    from vguitar.metrics import esr, plot_compare
    from vguitar.spice import make_dataset

    cfg.paths.ensure()
    data_cfg = replace(cfg.data, duration_s=args.duration, seed=args.seed)
    cfg = replace(cfg, data=data_cfg)

    try:
        circuit = get_circuit(args.circuit)
    except KeyError as exc:
        return _fail(str(exc))

    try:
        ds = make_dataset(circuit, cfg)
    except RuntimeError as exc:  # ngspice missing / sim failure
        return _fail(str(exc))

    path = ds.save(_dataset_path(cfg, args.circuit))
    # ESR of a silent (all-zero) prediction is the trivial baseline any model
    # must beat; it equals 1.0 by construction (error energy == signal energy).
    baseline = esr(ds.y, np.zeros_like(ds.y))
    print(f"saved {path}")
    print(f"  samples : {len(ds)}  ({ds.duration_s:.2f} s at {ds.sr} Hz)")
    print(f"  zero-prediction ESR baseline : {baseline:.4f}")

    preview = cfg.paths.outputs / f"{args.circuit}_preview.png"
    plot_compare(ds.y, ds.x, ds.sr, preview)  # target vs. raw input, for a quick look
    print(f"  preview : {preview}")
    return 0


# --- train ----------------------------------------------------------------
def cmd_train(args: argparse.Namespace, cfg: Config) -> int:
    """Load-or-generate the dataset, fit one model, save it, print test ESR."""
    from dataclasses import replace

    from vguitar.metrics import esr, plot_compare
    from vguitar.models.base import get_model

    cfg.paths.ensure()
    if args.epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=args.epochs))

    try:
        model_cls = get_model(args.model)
    except KeyError as exc:
        return _fail(str(exc))

    try:
        ds = _load_or_make_dataset(cfg, args.circuit, make=True)
    except (KeyError, RuntimeError) as exc:
        return _fail(str(exc))

    train, val, test = ds.split(cfg.train.val_fraction, cfg.train.test_fraction)
    model = model_cls()
    print(f"training {model.name} on {args.circuit} ({model.num_params()} params)...")
    report = model.fit(train, val, cfg.train)

    pred = model.process(test.x)
    test_esr = esr(test.y, pred)
    run = _run_path(cfg, args.circuit, args.model)
    model.save(run)
    print(f"saved {run}")
    if report.final_val_loss is not None:
        print(f"  final val loss : {report.final_val_loss:.4e}")
    print(f"  test ESR : {test_esr:.4e}")

    fig = cfg.paths.outputs / f"{args.circuit}_{args.model}_test.png"
    plot_compare(test.y, pred, test.sr, fig)
    print(f"  comparison plot : {fig}")
    return 0


# --- bench ----------------------------------------------------------------
def cmd_bench(args: argparse.Namespace, cfg: Config) -> int:
    """Run the benchmark over the requested models and print the leaderboard."""
    cfg.paths.ensure()
    try:
        from vguitar.benchmark import run_benchmark
    except ImportError as exc:
        return _fail(f"benchmark module unavailable: {exc}")

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    try:
        run_benchmark(args.circuit, model_names=models, cfg=cfg)
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        return _fail(str(exc))
    return 0


# --- live -----------------------------------------------------------------
def _control_specs_for(circ: Any, model: Any) -> list:
    """Control specs for a conditioned model: the circuit's, or generic c0..cK-1."""
    from vguitar.circuits.base import ControlSpec

    n = int(getattr(model, "n_control", 0))
    if n <= 0:
        return []
    if circ.controls and len(circ.controls) == n:
        return list(circ.controls)
    names = ["drive"] if n == 1 else [f"c{i}" for i in range(n)]
    return [ControlSpec(nm) for nm in names]


def _control_value_fn(control_str: str | None, automation_path: str | None, specs: list) -> Any:
    """Build ``value_at(t_seconds) -> (K,) vector`` from --control / --automation.

    ``--control "drive=0.08,tone=0.6"`` is a constant; ``--automation file.json``
    is a list of breakpoints ``[{"t": sec, <name>: val, ...}, ...]`` linearly
    interpolated over time. Returns ``None`` when neither is given.
    """
    if not specs:
        return None
    names = [s.name for s in specs]
    base = np.array([s.default for s in specs], dtype=np.float32)
    idx = {nm: i for i, nm in enumerate(names)}

    def vec_from(d: dict) -> np.ndarray:
        v = base.copy()
        for k, val in d.items():
            if k in idx:
                v[idx[k]] = float(val)
        return v

    if automation_path:
        import json

        bps = sorted(json.loads(Path(automation_path).read_text()), key=lambda b: float(b.get("t", 0.0)))
        ts = np.array([float(b.get("t", 0.0)) for b in bps], dtype=np.float64)
        vecs = np.array([vec_from(b) for b in bps], dtype=np.float32)

        def value_at(t: float) -> np.ndarray:
            if t <= ts[0]:
                return vecs[0]
            if t >= ts[-1]:
                return vecs[-1]
            j = int(np.searchsorted(ts, t))
            w = (t - ts[j - 1]) / (ts[j] - ts[j - 1]) if ts[j] > ts[j - 1] else 0.0
            return ((1.0 - w) * vecs[j - 1] + w * vecs[j]).astype(np.float32)

        return value_at

    if control_str:
        d = {k.split("=")[0].strip(): k.split("=", 1)[1].strip()
             for k in control_str.split(",") if "=" in k}
        const = vec_from(d)

        def value_at(_t: float) -> np.ndarray:
            return const

        return value_at
    return None


def cmd_live(args: argparse.Namespace, cfg: Config) -> int:
    """Load a trained model and either render a WAV pair or run the live engine."""
    from dataclasses import replace

    from vguitar.models.base import get_model

    if args.device is not None:
        rt = replace(cfg.realtime, input_device=args.device, output_device=args.device)
        cfg = replace(cfg, realtime=rt)

    run = _run_path(cfg, args.circuit, args.model)
    if not run.exists():
        packaged = cfg.paths.assets / "checkpoints" / f"{args.circuit}.{args.model}.model"
        if packaged.exists():
            run = packaged
        else:
            return _fail(
                f"no trained model at {run}; "
                f"run 'vguitar train --circuit {args.circuit} --model {args.model}' first"
            )

    try:
        model_cls = get_model(args.model)
    except KeyError as exc:
        return _fail(str(exc))
    model = model_cls.load(run)

    # Optional control source for a conditioned model (e.g. CIRCE).
    from vguitar.circuits import get_circuit

    specs = _control_specs_for(get_circuit(args.circuit), model)
    value_at = _control_value_fn(getattr(args, "control", None), getattr(args, "automation", None), specs)
    if value_at is not None and not specs:
        print("  (model is not conditioned; ignoring --control/--automation)")
        value_at = None

    if args.in_wav or args.out_wav:
        if not (args.in_wav and args.out_wav):
            return _fail("both --in and --out are required for file rendering")
        try:
            from vguitar.realtime import render_file
        except ImportError as exc:
            return _fail(f"realtime module unavailable: {exc}")
        control = None
        if value_at is not None:
            sr = cfg.realtime.sr

            def control(bi: int) -> np.ndarray:
                return value_at(bi * 1024 / sr)

        render_file(model, args.in_wav, args.out_wav, cfg.realtime.sr, control=control)
        print(f"rendered {args.in_wav} -> {args.out_wav}")
        return 0

    try:
        import sounddevice as sd

        from vguitar.realtime import LiveEngine
    except ImportError as exc:
        return _fail(f"realtime module unavailable: {exc}")
    control_fn = None
    if value_at is not None:
        bs, sr = cfg.realtime.block_size, cfg.realtime.sr

        def control_fn(bi: int) -> np.ndarray:
            return value_at(bi * bs / sr)

    engine = LiveEngine(model, cfg.realtime, control_fn=control_fn)
    print("starting live engine (Ctrl-C to stop)...")
    try:
        engine.start()
        while True:  # block the main thread; audio runs on PortAudio's thread
            sd.sleep(200)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        engine.stop()
    return 0


# --- selftest -------------------------------------------------------------
def _toy_dataset(n: int = 6144, sr: int = 8000, seed: int = 0) -> Dataset:
    """A short, fast-to-learn synthetic dataset for the model smoke test.

    The target is a soft-clipped, one-pole low-passed copy of a noise input:
    a memoryless ``tanh`` nonlinearity followed by a single real pole. This is
    learnable by every model class in a couple of epochs, so it exercises the
    full ``fit -> process -> streaming -> save/load`` path without needing
    ngspice or a long run.
    """
    from scipy.signal import lfilter

    from vguitar.data import Dataset

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) * 0.5).astype(np.float32)
    shaped = np.tanh(3.0 * x)  # soft clipping (the nonlinearity)
    y = lfilter([0.2], [1.0, -0.8], shaped).astype(np.float32)  # 1-pole low-pass
    return Dataset(x, y, sr, name="selftest")


def _toy_conditioned_dataset(n_seg: int = 2048, sr: int = 8000, seed: int = 0) -> Dataset:
    """A conditioned variant of :func:`_toy_dataset` for conditioned models.

    Two segments at different "drive" settings, each a ``tanh(g*x)`` soft-clip +
    1-pole low-pass, so a conditioned model has a single control column to learn.
    """
    from scipy.signal import lfilter

    from vguitar.data import Dataset

    rng = np.random.default_rng(seed)
    drives = [1.0, 3.0]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    bounds: list[int] = [0]
    vals: list[list[float]] = []
    for g in drives:
        x = (rng.standard_normal(n_seg) * 0.5).astype(np.float32)
        y = lfilter([0.2], [1.0, -0.8], np.tanh(g * x)).astype(np.float32)
        xs.append(x)
        ys.append(y)
        vals.append([g])
        bounds.append(bounds[-1] + n_seg)
    return Dataset.from_segments(
        np.concatenate(xs),
        np.concatenate(ys),
        sr,
        bounds,
        np.asarray(vals, dtype=np.float32),
        name="selftest_cond",
        control_names=["drive"],
        control_kinds=["continuous"],
    )


def _selftest_model(name: str, model_cls: type[Model], ds: Dataset) -> tuple[bool, str]:
    """Fit, streaming-check, and save/load a single model; return (ok, detail)."""
    import tempfile

    from vguitar.config import TrainConfig
    from vguitar.models.base import check_streaming

    # Tiny training budget: a handful of short windows, ~2 epochs.
    cfg = TrainConfig(seq_len=1024, warmup=128, batch_size=8, epochs=2, sr=ds.sr)
    train, val, _ = ds.split(0.2, 0.2)
    model = model_cls()
    model.fit(train, val, cfg)

    check_streaming(model, n=4096, block=128)  # raises AssertionError on mismatch

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.model"
        model.save(path)
        reloaded = model_cls.load(path)
    x = ds.x[:2048]
    a = np.asarray(model.process(x), dtype=np.float32)
    b = np.asarray(reloaded.process(x), dtype=np.float32)
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.isfinite(err) or err > 1e-4:
        return False, f"save/load mismatch (max abs err {err:.2e})"
    return True, f"{model.num_params()} params"


def _selftest_circuit(circuit_name: str) -> tuple[str, bool, str]:
    """0.05 s convergence smoke test for one circuit at its nominal drive.

    Reports PASS, SKIP (ngspice DLL absent), or FAIL. Drives the circuit at
    ``nominal_drive_v`` so a stage that actually clips is exercised; netlist-mode
    controls use their declared defaults.
    """
    from vguitar.circuits.base import get_circuit
    from vguitar.signals import sine
    from vguitar.spice import simulate

    sr = 44_100
    circuit = get_circuit(circuit_name)
    x = sine(220.0, 0.05, sr, peak=float(circuit.nominal_drive_v))
    try:
        y = simulate(circuit, x, sr)
    except RuntimeError as exc:  # ngspice DLL absent -> skip, not a failure
        return "SKIP", True, f"ngspice unavailable: {str(exc).splitlines()[0]}"
    except Exception as exc:
        return "FAIL", False, str(exc)
    if y.shape == x.shape and np.all(np.isfinite(y)):
        return "PASS", True, f"out range +-{float(np.max(np.abs(y))):.3f} V"
    return "FAIL", False, "bad output shape or non-finite values"


def cmd_selftest(args: argparse.Namespace, cfg: Config) -> int:
    """Smoke-test every registered model and the ngspice path; print a table."""
    from rich.console import Console
    from rich.table import Table

    from vguitar.models.base import all_models

    console = Console()
    table = Table(title="vguitar selftest")
    table.add_column("component", style="bold cyan")
    table.add_column("status")
    table.add_column("detail")

    all_ok = True
    ds = _toy_dataset()
    ds_cond = _toy_conditioned_dataset()
    for name, model_cls in sorted(all_models().items()):
        try:
            use_ds = ds_cond if model_cls.conditioned else ds
            ok, detail = _selftest_model(name, model_cls, use_ds)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        table.add_row(f"model:{name}", status, detail)
        all_ok = all_ok and ok

    from vguitar.circuits.base import all_circuits

    for circ_name in sorted(all_circuits()):
        status_str, ng_ok, ng_detail = _selftest_circuit(circ_name)
        color = {"PASS": "green", "SKIP": "yellow", "FAIL": "red"}[status_str]
        table.add_row(f"ngspice:{circ_name}", f"[{color}]{status_str}[/]", ng_detail)
        all_ok = all_ok and ng_ok

    console.print(table)
    return 0 if all_ok else 1


# --- plots ----------------------------------------------------------------
def _find_model_file(cfg: Config, circuit: str, model: str) -> Path | None:
    """Locate a trained model file, tolerating both naming conventions."""
    for cand in (cfg.paths.runs / f"{circuit}.{model}.model", cfg.paths.runs / f"{circuit}_{model}"):
        if cand.exists():
            return cand
    return None


def _read_bench_csv(cfg: Config, circuit: str) -> list[dict]:
    """Parse a saved benchmark CSV into leaderboard rows (for the plot)."""
    import csv

    path = cfg.paths.outputs / f"{circuit}_benchmark.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                rows.append({"model": r["model"], "error": r["error"]})
                continue
            with contextlib.suppress(KeyError, ValueError):
                rows.append({
                    "model": r["model"], "esr": float(r["esr"]), "rtf": float(r["rtf"]),
                    "realtime": r["realtime"] == "True", "params": int(r["params"]),
                })
    return rows


def cmd_plots(args: argparse.Namespace, cfg: Config) -> int:
    """Render the clean diagnostic figure set for a circuit to outputs/figs/."""
    from vguitar import plotting as plot
    from vguitar.circuits import get_circuit
    from vguitar.models import all_models, get_model

    try:
        ds = _load_or_make_dataset(cfg, args.circuit, make=False)
    except FileNotFoundError as exc:
        return _fail(str(exc))

    names = [m.strip() for m in args.models.split(",")] if args.models else sorted(all_models())
    models = {}
    for name in names:
        path = _find_model_file(cfg, args.circuit, name)
        if path is None:
            print(f"  (skip {name}: no trained model in {cfg.paths.runs})")
            continue
        try:
            models[name] = get_model(name).load(path)
        except Exception as exc:  # a stale/incompatible checkpoint shouldn't abort
            print(f"  (skip {name}: load failed: {exc})")

    outdir = cfg.paths.outputs / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    sr = ds.sr
    saved: list[str] = []

    def _save(fig, stem: str) -> None:
        import matplotlib.pyplot as plt

        p = outdir / f"{args.circuit}_{stem}.png"
        fig.savefig(p)
        plt.close(fig)
        saved.append(str(p))

    _save(plot.fig_dataset(ds.x, ds.y, sr, name=args.circuit), "dataset")

    _, _, test = ds.split(cfg.train.val_fraction, cfg.train.test_fraction)
    if models:
        preds = {n: m.process(test.x) for n, m in models.items()}
        _save(plot.fig_waveform(test.y, preds, sr, name=args.circuit), "waveform")

    # Probes through the real circuit (needs ngspice) for transfer + harmonics.
    try:
        from vguitar.spice.runner import simulate

        circ = get_circuit(args.circuit)
        amp = float(np.max(np.abs(ds.x)))  # probe within the trained amplitude range
        x_slow = (amp * np.sin(2 * np.pi * 40 * np.arange(int(0.05 * sr)) / sr)).astype(np.float32)
        y_slow = simulate(circ, x_slow, sr)
        _save(
            plot.fig_transfer(x_slow, y_slow, {n: m.process(x_slow) for n, m in models.items()},
                              name=args.circuit), "transfer")
        x_tone = (circ.nominal_drive_v * np.sin(2 * np.pi * 1000 * np.arange(int(0.2 * sr)) / sr)
                  ).astype(np.float32)
        y_tone = simulate(circ, x_tone, sr)
        _save(
            plot.fig_harmonics(y_tone, {n: m.process(x_tone) for n, m in models.items()}, sr,
                               f0=1000.0, name=args.circuit), "harmonics")
    except Exception as exc:  # ngspice missing / sim error: skip these two
        print(f"  (skip transfer/harmonics: {exc})")

    from vguitar.models.volterra_reg import VolterraReg

    vm = models.get("volterra")
    if isinstance(vm, VolterraReg) and vm.h1 is not None:
        _save(plot.fig_volterra_kernels(vm.h1, vm._H2, vm._H3, sr), "kernels")

    rows = _read_bench_csv(cfg, args.circuit)
    if rows:
        _save(plot.fig_leaderboard(rows, name=args.circuit), "leaderboard")

    print(f"wrote {len(saved)} figures to {outdir}:")
    for s in saved:
        print(f"  {s}")
    return 0


def cmd_circe(args: argparse.Namespace, cfg: Config) -> int:
    """Train/validate the conditioned CIRCE model on a drive knob; emit figures."""
    try:
        from vguitar.benchmark.circe_eval import run_circe_eval
    except ImportError as exc:
        return _fail(f"circe eval unavailable: {exc}")
    try:
        run_circe_eval(args.circuit, retrain=args.retrain, regen=args.regen,
                       eval_di=args.di, render=args.render, heatmap=args.heatmap)
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        return _fail(str(exc))
    return 0


def cmd_shootout(args: argparse.Namespace, cfg: Config) -> int:
    """Fixed-operating-point head-to-head: CIRCE vs baselines, per circuit + audio."""
    try:
        from vguitar.benchmark.shootout import run_shootout
    except ImportError as exc:
        return _fail(f"shootout unavailable: {exc}")
    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    models = tuple(m.strip() for m in args.models.split(",") if m.strip()) if args.models else None
    try:
        run_shootout(circuits, models=models, cfg=cfg, epochs=args.epochs,
                     retrain=args.retrain, regen=args.regen)
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        return _fail(str(exc))
    return 0


def cmd_validate(args: argparse.Namespace, cfg: Config) -> int:
    """Validate CIRCE across a set of circuits; emit cross-circuit comparison figures."""
    try:
        from vguitar.benchmark.validate import run_validation
    except ImportError as exc:
        return _fail(f"validate unavailable: {exc}")
    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    models = tuple(m.strip() for m in args.models.split(",") if m.strip()) if args.models else ("circe",)
    try:
        run_validation(circuits, models=models, cfg=cfg, retrain=args.retrain, regen=args.regen)
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        return _fail(str(exc))
    return 0


# --- argument parser ------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse tree (one subparser per subcommand)."""
    p = argparse.ArgumentParser(prog="vguitar", description=(__doc__ or "vguitar").splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="list registered circuits and models")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("gen", help="generate + cache a circuit dataset via ngspice")
    sp.add_argument("--circuit", required=True, help="circuit name (see 'vguitar list')")
    sp.add_argument(
        "--duration",
        type=float,
        default=Config().data.duration_s,
        help="excitation duration in seconds",
    )
    sp.add_argument("--seed", type=int, default=Config().data.seed, help="excitation RNG seed")
    sp.set_defaults(func=cmd_gen)

    sp = sub.add_parser("train", help="train one model on a circuit dataset")
    sp.add_argument("--circuit", required=True, help="circuit name")
    sp.add_argument("--model", required=True, help="model name (see 'vguitar list')")
    sp.add_argument("--epochs", type=int, default=None, help="override training epochs")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("bench", help="benchmark models on a circuit and print a leaderboard")
    sp.add_argument("--circuit", required=True, help="circuit name")
    sp.add_argument("--models", default=None, help="comma-separated model names (default: all)")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("live", help="run a trained model on a WAV or live audio")
    sp.add_argument("--circuit", required=True, help="circuit name")
    sp.add_argument("--model", required=True, help="model name")
    sp.add_argument("--in", dest="in_wav", default=None, help="input WAV (offline render)")
    sp.add_argument("--out", dest="out_wav", default=None, help="output WAV (offline render)")
    sp.add_argument("--device", type=int, default=None, help="PortAudio device index")
    sp.add_argument("--control", default=None,
                    help="conditioned model: constant knobs, e.g. 'drive=0.08,tone=0.6'")
    sp.add_argument("--automation", default=None,
                    help="conditioned model: JSON breakpoints [{\"t\":sec,<name>:val,...}] (knob automation)")
    sp.set_defaults(func=cmd_live)

    sp = sub.add_parser("selftest", help="smoke-test every model and the ngspice path")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("plots", help="render the clean diagnostic figure set to outputs/figs/")
    sp.add_argument("--circuit", required=True, help="circuit name")
    sp.add_argument("--models", default=None, help="comma-separated model names (default: all trained)")
    sp.set_defaults(func=cmd_plots)

    sp = sub.add_parser("circe", help="train + validate the conditioned CIRCE model; emit figures")
    sp.add_argument("--circuit", default="bjt", help="circuit name")
    sp.add_argument("--retrain", action="store_true", help="retrain even if a saved model exists")
    sp.add_argument("--regen", action="store_true", help="re-simulate the drive-sweep datasets")
    sp.add_argument("--no-di", dest="di", action="store_false", help="skip held-out guitar-DI eval")
    sp.add_argument("--no-render", dest="render", action="store_false", help="skip A/B wav renders")
    sp.add_argument("--no-heatmap", dest="heatmap", action="store_false", help="skip drive-freq heatmap")
    sp.set_defaults(func=cmd_circe)

    sp = sub.add_parser("shootout", help="fixed-point head-to-head: CIRCE vs baselines per circuit (+ audio)")
    sp.add_argument("--circuits", required=True, help="comma-separated circuit names")
    sp.add_argument("--models", default=None,
                    help="comma-separated baselines (default: fir,volterra,volterra_pc,wh,tcn,rnn; circe always)")
    sp.add_argument("--epochs", type=int, default=100, help="training epochs for every neural method")
    sp.add_argument("--retrain", action="store_true", help="ignored placeholder (always trains fresh)")
    sp.add_argument("--regen", action="store_true", help="re-simulate the shootout datasets")
    sp.set_defaults(func=cmd_shootout)

    sp = sub.add_parser("validate", help="validate CIRCE across several circuits; emit comparison figures")
    sp.add_argument("--circuits", required=True, help="comma-separated circuit names")
    sp.add_argument("--models", default=None,
                    help="comma-separated matrix columns (default: circe; add baselines e.g. tcn,volterra)")
    sp.add_argument("--retrain", action="store_true", help="retrain even if saved models exist")
    sp.add_argument("--regen", action="store_true", help="re-simulate the control-sweep datasets")
    sp.set_defaults(func=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the chosen subcommand's handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args, Config()))


if __name__ == "__main__":
    raise SystemExit(main())
