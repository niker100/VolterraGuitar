"""Validation + visualization for CIRCE (the conditioned interactive model).

Runs the conditioned-control experiment end to end for **any circuit and any
control axes** (one knob like the BJT drive, or several like the Tube Screamer's
drive+tone and the Big Muff's sustain+tone+level) and reports, separating
**trained** from **held-out** (interpolated) control settings:

* interpolation accuracy (per-setting + worst-case/p95, vs nearest-trained distance),
* moving-control streaming equivalence (the knob turning mid-stream — GATE-4),
* generalization to a real held-out guitar-DI input,
* stability (zero-input quietness, hot-input saturation),
* real-time factor with a moving knob, and audio A/B renders to listen to.

Everything is reproducible via ``vguitar circe`` and reuses the library
(``spice.runner``, ``spice.sampling``, ``metrics``, ``realtime``, ``plotting``).
The quantitative core lives in :func:`_validate_one` (reused by the cross-circuit
``vguitar validate``); :func:`run_circe_eval` adds the report + figures. The pure
helpers (``_interp_stats``, ``_stability_checks``, ``_rtf_moving``) take plain
arrays/models so they are unit-testable without the full pipeline, and keep their
original scalar-control call signatures for backward compatibility.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vguitar.circuits.base import Circuit, ControlSpec
    from vguitar.config import Config
    from vguitar.models.circe import CIRCE

_DI_PATH = "assets/guitar_di_loop.wav"
_DEFAULT_DRIVES = [0.005, 0.02, 0.04, 0.08, 0.16]
_DEFAULT_HELD = [0.01, 0.06, 0.12]


# --- pure helpers (unit-testable) -----------------------------------------
def _thd_signal(y: np.ndarray, sr: int, f0: float = 1000.0, n_harm: int = 8) -> float:
    """THD of a single-tone output: RMS of harmonics 2..n / fundamental."""
    yv = np.asarray(y, dtype=np.float64)
    mag = np.abs(np.fft.rfft(yv * np.hanning(len(yv))))
    n = len(yv)

    def peak(k: int) -> float:
        b = round(k * f0 * n / sr)
        return float(mag[max(b - 3, 0) : b + 4].max()) if b + 3 < len(mag) else 0.0

    fund = peak(1)
    harm = np.sqrt(sum(peak(k) ** 2 for k in range(2, n_harm + 1)))
    return float(harm / (fund + 1e-12))


def _dbfs(y: np.ndarray) -> float:
    """RMS level in dBFS (full scale = 1.0)."""
    rms = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)) + 1e-20)
    return 20.0 * np.log10(rms)


def _as_control_rows(controls: Any) -> list[np.ndarray]:
    """Normalize floats / vectors / ``(S, C)`` array to a list of 1-D control rows."""
    arr = np.asarray(controls, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]


def _split_control(row: np.ndarray, specs: list[ControlSpec] | None) -> tuple[float, dict[str, float] | None]:
    """Split a control row into (input pre-gain factor, netlist params).

    With ``specs=None`` every column is treated as a pre-gain (the legacy drive
    behaviour): the factor is their product and there are no netlist params.
    """
    r = np.asarray(row, dtype=np.float64).reshape(-1)
    if specs is None:
        return (float(np.prod(r)) if r.size else 1.0), None
    gain = 1.0
    params: dict[str, float] = {}
    for i, s in enumerate(specs):
        if s.mode == "pregain":
            gain *= float(r[i])
        else:
            params[s.name] = float(r[i])
    return gain, (params or None)


def _interp_stats(
    rows: list[dict[str, Any]],
    trained_pts: Any,
    *,
    key: str | None = "drive",
    specs: list[ControlSpec] | None = None,
) -> dict[str, Any]:
    """Held-out interpolation summary + per-held distance-to-nearest-trained.

    Two call styles:

    * scalar (legacy): ``key="drive"``, ``trained_pts`` a list of scalars; each
      row has ``r[key]``. Distance is ``min|r[key]-t| / span``.
    * vector: ``key=None``, ``trained_pts`` a ``(T, C)`` array, ``specs`` given;
      distance is the normalized nearest-neighbour L2 in control space.
    """
    held_rows = [r for r in rows if r["held"]]
    if key is not None:
        tv = [float(t) for t in trained_pts]
        span = (max(tv) - min(tv)) or 1.0
        per = [
            {key: r[key], "dist": min(abs(r[key] - t) for t in tv) / span, "esr": r["esr"]}
            for r in held_rows
        ]
    else:
        from vguitar.spice.sampling import nn_distance

        assert specs is not None
        pts = np.array([r["control"] for r in held_rows], dtype=np.float32).reshape(len(held_rows), -1)
        dist = (
            nn_distance(pts, np.asarray(trained_pts, dtype=np.float32), specs)
            if held_rows
            else np.zeros(0)
        )
        per = [
            {"control": held_rows[i]["control"], "dist": float(dist[i]), "esr": held_rows[i]["esr"]}
            for i in range(len(held_rows))
        ]
    h = [r["esr"] for r in held_rows]
    t = [r["esr"] for r in rows if not r["held"]]
    return {
        "trained_mean": float(np.mean(t)) if t else float("nan"),
        "held_mean": float(np.mean(h)) if h else float("nan"),
        "held_worst": float(np.max(h)) if h else float("nan"),
        "held_p95": float(np.percentile(h, 95)) if h else float("nan"),
        "per_held": per,
    }


def _stability_checks(model: CIRCE, control_rows: Any, sr: int = 44_100) -> dict[str, Any]:
    """Zero-input quietness (probe settings) + hot-input saturation (offline+streaming)."""
    rows = _as_control_rows(control_rows)
    out_bound = float(model.net.out_bound)
    probe = [rows[0], rows[len(rows) // 2], rows[-1]]
    zero = [
        {"control": c, "dbfs": _dbfs(model.process(np.zeros(sr, np.float32), c))} for c in probe
    ]
    # hot input: 4x a unit-peak base, at the last (hottest) control row.
    hot = (4.0 * np.random.default_rng(0).standard_normal(sr)).astype(np.float32)
    cmax = rows[-1]
    y_off = model.process(hot, cmax)
    model.reset()
    y_st = np.concatenate(
        [model.process_block(hot[i : i + 128], cmax) for i in range(0, len(hot), 128)]
    )
    return {
        "out_bound": out_bound,
        "zero": zero,
        "hot_offline_peak": float(np.max(np.abs(y_off))),
        "hot_stream_peak": float(np.max(np.abs(y_st))),
        "hot_finite": bool(np.isfinite(y_off).all() and np.isfinite(y_st).all()),
    }


def _rtf_moving(model: CIRCE, control_rows: Any, sr: int = 44_100, block: int = 128,
                dur_s: float = 3.0) -> dict[str, float | bool]:
    """Real-time factor with the control changed every block (knob turning)."""
    import time

    rows = _as_control_rows(control_rows)
    n_blocks = max(2, int(dur_s * sr / block))
    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal(block).astype(np.float32) for _ in range(n_blocks)]
    ctrls = [rows[i % len(rows)] for i in range(n_blocks)]
    model.reset()
    model.process_block(blocks[0], ctrls[0])  # warm-up (exclude from timing)
    t0 = time.perf_counter()
    for b, c in zip(blocks[1:], ctrls[1:], strict=True):
        model.process_block(b, c)
    dt = time.perf_counter() - t0
    audio = (n_blocks - 1) * block / sr
    rtf = audio / dt if dt > 0 else float("inf")
    return {"rtf": float(rtf), "realtime": bool(rtf > 1.0),
            "ms_per_block": float(dt / max(n_blocks - 1, 1) * 1e3)}


# --- ngspice helpers (control-agnostic) -----------------------------------
def _eval_on_di(model: CIRCE, circ: Any, control_rows: Any, sr: int,
                dur_s: float = 1.5, control_specs: list[ControlSpec] | None = None) -> list[dict[str, Any]]:
    """Generalization to the real guitar-DI input across control settings (ngspice)."""
    from vguitar.metrics import esr, multi_stft
    from vguitar.signals import load_di
    from vguitar.spice.runner import simulate

    di = load_di(_DI_PATH, sr, peak=1.0)[: int(dur_s * sr)]
    out = []
    for c in _as_control_rows(control_rows):
        g, params = _split_control(c, control_specs)
        yc = simulate(circ, (g * di).astype(np.float32), sr, params=params)
        ym = model.process(di, c)
        w = 2048
        out.append({
            "control": c, "esr": float(esr(yc[w:], ym[w:])), "stft": float(multi_stft(yc[w:], ym[w:])),
            "yc": yc, "ym": ym,
        })
    return out


def _control_freq_grid(model: CIRCE, circ: Any, control_rows: Any, sr: int,
                       tones: list[float] | None = None,
                       control_specs: list[ControlSpec] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Relative spectral-magnitude error (dB) of CIRCE vs circuit over setting x tone."""
    from vguitar.spice.runner import simulate

    rows = _as_control_rows(control_rows)
    tones = tones or [200.0, 500.0, 1000.0, 2000.0, 4000.0]
    t = np.arange(int(0.08 * sr)) / sr
    win = np.hanning(len(t))
    err = np.zeros((len(rows), len(tones)), dtype=np.float64)
    for i, c in enumerate(rows):
        g, params = _split_control(c, control_specs)
        for j, f in enumerate(tones):
            x = np.sin(2 * np.pi * f * t).astype(np.float32)
            yc = simulate(circ, (g * x).astype(np.float32), sr, params=params)
            ym = model.process(x, c)
            yc_m = np.abs(np.fft.rfft(yc * win))
            ym_m = np.abs(np.fft.rfft(ym * win))
            rel = float(np.linalg.norm(ym_m - yc_m) / (np.linalg.norm(yc_m) + 1e-12))
            err[i, j] = 20.0 * np.log10(rel + 1e-6)
    return np.asarray(tones), err


def _axis_sweep(model: CIRCE, circ: Any, specs: list[ControlSpec], axis: int, base_row: np.ndarray,
                sr: int, n: int = 7, f0: float = 1000.0) -> dict[str, np.ndarray]:
    """Sweep one control axis (others held at ``base_row``); THD + peak, circ vs model."""
    from vguitar.spice.runner import simulate

    s = specs[axis]
    vals = np.linspace(s.lo, s.hi, n)
    tone = np.sin(2 * np.pi * f0 * np.arange(int(0.12 * sr)) / sr).astype(np.float32)
    thd_c, thd_m, peak_c, peak_m = [], [], [], []
    for v in vals:
        row = np.array(base_row, dtype=np.float32, copy=True)
        row[axis] = v
        g, params = _split_control(row, specs)
        yc = simulate(circ, (g * tone).astype(np.float32), sr, params=params)
        ym = model.process(tone, row)
        thd_c.append(_thd_signal(yc, sr))
        thd_m.append(_thd_signal(ym, sr))
        peak_c.append(float(np.max(np.abs(yc))))
        peak_m.append(float(np.max(np.abs(ym))))
    return {"vals": vals, "thd_c": np.array(thd_c), "thd_m": np.array(thd_m),
            "peak_c": np.array(peak_c), "peak_m": np.array(peak_m)}


def _control_esr_grid(model: CIRCE, circ: Any, specs: list[ControlSpec], base_row: np.ndarray,
                      sr: int, axes: tuple[int, int] = (0, 1), n: int = 5,
                      dur_s: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ESR over a 2-D control plane (axes ``i,j``; others at ``base_row``) — ngspice."""
    from vguitar.metrics import esr as esr_metric
    from vguitar.spice.runner import simulate

    i, j = axes
    vi = np.linspace(specs[i].lo, specs[i].hi, n)
    vj = np.linspace(specs[j].lo, specs[j].hi, n)
    base = np.random.default_rng(0).standard_normal(int(dur_s * sr)).astype(np.float32)
    base = (base / (np.max(np.abs(base)) + 1e-9)).astype(np.float32)
    grid = np.zeros((n, n), dtype=np.float64)
    w = 1024
    for a, va in enumerate(vi):
        for b, vb in enumerate(vj):
            row = np.array(base_row, dtype=np.float32, copy=True)
            row[i], row[j] = va, vb
            g, params = _split_control(row, specs)
            yc = simulate(circ, (g * base).astype(np.float32), sr, params=params)
            ym = model.process(base, row)
            grid[a, b] = float(esr_metric(yc[w:], ym[w:]))
    return vi, vj, grid


def _render_ab(model: CIRCE, circ: Any, control_rows: Any, sr: int, outdir: Any,
               dur_s: float = 2.5, control_specs: list[ControlSpec] | None = None) -> list[str]:
    """Write dry / circuit / CIRCE wavs at a few settings (for A/B listening)."""
    import soundfile as sf

    from vguitar.signals import load_di
    from vguitar.spice.runner import simulate

    outdir.mkdir(parents=True, exist_ok=True)
    di = load_di(_DI_PATH, sr, peak=0.9)[: int(dur_s * sr)]

    def norm(y: np.ndarray) -> np.ndarray:
        p = float(np.max(np.abs(y)))
        return (0.9 * y / p).astype(np.float32) if p > 0 else y.astype(np.float32)

    written = [str(sf_write(sf, outdir / "di.wav", di, sr))]
    for c in _as_control_rows(control_rows):
        g, params = _split_control(c, control_specs)
        tag = "_".join(f"{v:g}" for v in np.asarray(c).reshape(-1))
        yc = simulate(circ, (g * di).astype(np.float32), sr, params=params)
        ym = model.process(di, c)
        written.append(str(sf_write(sf, outdir / f"set_{tag}_circuit.wav", norm(yc), sr)))
        written.append(str(sf_write(sf, outdir / f"set_{tag}_circe.wav", norm(ym), sr)))
    return written


def sf_write(sf: Any, path: Any, y: np.ndarray, sr: int) -> Any:
    sf.write(path, np.asarray(y, dtype=np.float32), sr)
    return path


# --- plan resolution ------------------------------------------------------
def _resolve_plan(
    circuit_name: str,
    control_specs: list[ControlSpec] | None,
    grid: Any,
    drives: list[float] | None,
    held: list[float] | None,
) -> tuple[Circuit, list[ControlSpec], np.ndarray, np.ndarray, bool]:
    """Resolve (circuit, specs, trained grid, held grid, single-drive flag)."""
    from vguitar.circuits import get_circuit
    from vguitar.circuits.base import ControlSpec
    from vguitar.spice.sampling import control_grid, holdout_grid

    circ = get_circuit(circuit_name)
    if control_specs is not None:
        specs: list[ControlSpec] | None = list(control_specs)
    elif circ.controls:
        specs = list(circ.controls)
    else:
        specs = None

    # Implicit single drive knob (the original behaviour for bjt/diode).
    if specs is None:
        dv = list(drives) if drives else _DEFAULT_DRIVES
        hv = list(held) if held is not None else _DEFAULT_HELD
        allv = dv + hv
        specs = [ControlSpec("drive", "continuous", min(allv), max(allv), dv[0], "pregain")]
        g = np.asarray(grid, np.float32).reshape(-1, 1) if grid is not None else np.array([[d] for d in dv], np.float32)
        hg = np.array([[h] for h in hv], np.float32)
        return circ, specs, g, hg, True

    c = len(specs)
    single_drive = c == 1 and specs[0].mode == "pregain"

    if grid is not None:
        g = np.asarray(grid, np.float32)
        if g.ndim == 1:
            g = g.reshape(-1, 1)
    elif single_drive and drives:
        g = np.array([[d] for d in drives], np.float32)
    elif single_drive:
        g = np.array([[d] for d in np.linspace(specs[0].lo, specs[0].hi, 5)], np.float32)
    else:
        n_axis = 4 if c == 2 else 3
        budget = None if c <= 2 else 24
        g = control_grid(specs, n_axis=n_axis, budget=budget, mode="auto", seed=0)

    if held is not None:
        hg = np.asarray(held, np.float32)
        hg = hg.reshape(-1, 1) if hg.ndim == 1 else hg
    else:
        n_held = 3 if c <= 2 else 6
        hg = holdout_grid(specs, g, n=n_held, seed=1, min_dist=0.08)
    return circ, specs, g, hg, single_drive


# --- quantitative validation core (reused by `vguitar validate`) ----------
def _validate_one(
    circuit_name: str = "bjt",
    *,
    control_specs: list[ControlSpec] | None = None,
    grid: Any = None,
    drives: list[float] | None = None,
    held: list[float] | None = None,
    cfg: Config | None = None,
    retrain: bool = False,
    regen: bool = False,
    seg_dur_s: float = 2.0,
    epochs: int = 80,
    channels: int = 10,
    n_blocks: int = 2,
    n_layers: int = 7,
    probe_thd: bool = True,
    console: Any = None,
) -> dict[str, Any]:
    """Train/load CIRCE for one circuit and compute the quantitative validation.

    Returns a dict with the per-setting ``rows`` (each carrying its control
    vector), the interpolation/stability/real-time/streaming aggregates, the
    fitted ``model`` and datasets, and the resolved plan — everything the report,
    the figures, and the cross-circuit ``vguitar validate`` need.
    """
    from rich.console import Console

    from vguitar import metrics
    from vguitar.config import Config, TrainConfig
    from vguitar.data import Dataset
    from vguitar.models.base import check_streaming, check_streaming_moving
    from vguitar.models.circe import CIRCE, _segments
    from vguitar.realtime import measure_rtf
    from vguitar.spice.runner import make_control_dataset, make_drive_dataset, simulate

    cfg = cfg or Config()
    cfg.paths.ensure()
    console = console or Console()
    circ, specs, train_grid, held_grid, single_drive = _resolve_plan(
        circuit_name, control_specs, grid, drives, held
    )
    n_control = len(specs)
    sr = cfg.data.sr

    # --- datasets (cached) ---
    stem = "drive" if single_drive else "ctl"
    train_path = cfg.paths.data / f"{circuit_name}_{stem}.npz"
    test_path = cfg.paths.data / f"{circuit_name}_{stem}_test.npz"
    if regen or not train_path.exists() or not test_path.exists():
        console.print(f"[dim]simulating control sweep via ngspice ({len(train_grid)}+{len(held_grid)} settings)...[/]")
        if single_drive:
            make_drive_dataset(circ, train_grid[:, 0].tolist(), cfg, seg_dur_s=seg_dur_s, seed=0).save(train_path)
            make_drive_dataset(circ, held_grid[:, 0].tolist(), cfg, seg_dur_s=seg_dur_s, seed=100).save(test_path)
        else:
            make_control_dataset(circ, train_grid, specs, cfg, seg_dur_s=seg_dur_s, seed=0).save(train_path)
            make_control_dataset(circ, held_grid, specs, cfg, seg_dur_s=seg_dur_s, seed=100).save(test_path)
    train_ds, test_ds = Dataset.load(train_path), Dataset.load(test_path)

    # --- model (cached); persist training history sidecar when we train ---
    model_path = cfg.paths.runs / f"{circuit_name}.circe.model"
    hist_path = cfg.paths.runs / f"{circuit_name}.circe.history.json"
    if retrain or not model_path.exists():
        import torch

        torch.manual_seed(0)
        tr, va, _ = train_ds.split(0.12, 0.0001)
        model = CIRCE(n_control=n_control, channels=channels, n_blocks=n_blocks, n_layers=n_layers)
        console.print("[dim]training CIRCE...[/]")
        report = model.fit(tr, va, TrainConfig(epochs=epochs, seq_len=2048, batch_size=16, lr=3e-3, warmup=256))
        model.save(model_path)
        hist_path.write_text(json.dumps(report.history))
    else:
        model = CIRCE.load(model_path)

    # --- per-setting metrics from the cached datasets ---
    rows: list[dict[str, Any]] = []
    for ds, is_held in ((train_ds, False), (test_ds, True)):
        assert ds.controls is not None
        for s, e in _segments(ds.controls):
            c_vec = np.ascontiguousarray(ds.controls[s], dtype=np.float32)
            pred = model.process(ds.x[s:e], c_vec)
            w = min(256, (e - s) - 1)
            yv, pv = ds.y[s:e][w:], pred[w:]
            row: dict[str, Any] = {
                "control": c_vec, "held": is_held, "esr": float(metrics.esr(yv, pv)),
                "peak_c": float(np.max(np.abs(yv))), "peak_m": float(np.max(np.abs(pv))),
            }
            if single_drive:
                row["drive"] = float(c_vec[0])
            rows.append(row)
    sort_key = (lambda r: r["drive"]) if single_drive else (lambda r: tuple(r["control"].tolist()))
    rows.sort(key=sort_key)

    # --- tone-based THD per setting (ngspice) ---
    have_spice = True
    if probe_thd:
        tone = np.sin(2 * np.pi * 1000.0 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
        try:
            for r in rows:
                g, params = _split_control(r["control"], specs)
                r["thd_c"] = _thd_signal(simulate(circ, (g * tone).astype(np.float32), sr, params=params), sr)
                r["thd_m"] = _thd_signal(model.process(tone, r["control"]), sr)
        except Exception as exc:  # ngspice optional
            have_spice = False
            console.print(f"[yellow]ngspice unavailable; skipping tone/DI/heatmap probes ({exc})[/]")

    # --- aggregates ---
    all_rows_ctrl = np.array([r["control"] for r in rows], dtype=np.float32)
    if single_drive:
        trained_pts: Any = sorted(r["drive"] for r in rows if not r["held"])
        interp = _interp_stats(rows, trained_pts, key="drive")
    else:
        trained_pts = np.array([r["control"] for r in rows if not r["held"]], dtype=np.float32)
        interp = _interp_stats(rows, trained_pts, key=None, specs=specs)
    stab = _stability_checks(model, all_rows_ctrl, sr)
    rtf_c = measure_rtf(model, sr=sr, block=cfg.realtime.block_size)
    rtf_m = _rtf_moving(model, all_rows_ctrl, sr=sr, block=cfg.realtime.block_size)
    streaming_err = check_streaming(model, n=4096, block=128, atol=2e-3)
    ramp = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None] * (
        np.asarray([s.hi for s in specs], np.float32) - np.asarray([s.lo for s in specs], np.float32)
    ) + np.asarray([s.lo for s in specs], np.float32)
    moving_err = check_streaming_moving(model, control_traj=ramp.astype(np.float32), n=4096, block=128, atol=2e-3)

    return {
        "circuit": circuit_name, "specs": specs, "single_drive": single_drive,
        "train_grid": train_grid, "held_grid": held_grid,
        "model": model, "train_ds": train_ds, "test_ds": test_ds, "hist_path": hist_path,
        "rows": rows, "interp": interp, "stab": stab, "rtf_c": rtf_c, "rtf_m": rtf_m,
        "streaming_err": float(streaming_err), "moving_err": float(moving_err),
        "params": int(model.num_params()), "have_spice": have_spice, "sr": sr,
    }


# --- main entry -----------------------------------------------------------
def run_circe_eval(
    circuit_name: str = "bjt",
    *,
    drives: list[float] | None = None,
    held: list[float] | None = None,
    grid: Any = None,
    control_specs: list[ControlSpec] | None = None,
    cfg: Config | None = None,
    retrain: bool = False,
    regen: bool = False,
    eval_di: bool = True,
    render: bool = True,
    heatmap: bool = True,
    seg_dur_s: float = 2.0,
    epochs: int = 80,
    channels: int = 10,
    n_blocks: int = 2,
    n_layers: int = 7,
) -> list[dict[str, Any]]:
    """Validate CIRCE on a circuit's control axes; print a report and write figures."""
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table

    from vguitar import metrics
    from vguitar import plotting as plot

    console = Console()
    res = _validate_one(
        circuit_name, control_specs=control_specs, grid=grid, drives=drives, held=held,
        cfg=cfg, retrain=retrain, regen=regen, seg_dur_s=seg_dur_s, epochs=epochs,
        channels=channels, n_blocks=n_blocks, n_layers=n_layers, probe_thd=True, console=console,
    )
    from vguitar.circuits import get_circuit
    from vguitar.config import Config

    cfg = cfg or Config()
    circ = get_circuit(circuit_name)
    specs = res["specs"]
    model, rows = res["model"], res["rows"]
    single_drive, have_spice, sr = res["single_drive"], res["have_spice"], res["sr"]
    interp, stab = res["interp"], res["stab"]
    hist_path = res["hist_path"]

    # --- report table ---
    axis_names = [s.name for s in specs]
    table = Table(title=f"CIRCE validation — {circuit_name} ({', '.join(axis_names)})")
    for cname in (*axis_names, "set", "ESR", "THD circ", "THD CIRCE", "peak c", "peak m"):
        table.add_column(cname, justify="center" if cname == "set" else "right")
    for r in rows:
        vals = [f"{v:g}" for v in np.asarray(r["control"]).reshape(-1)]
        table.add_row(
            *vals, "[yellow]held[/]" if r["held"] else "train", f"{r['esr']:.4f}",
            f"{r.get('thd_c', float('nan')):.3f}" if have_spice else "-",
            f"{r.get('thd_m', float('nan')):.3f}" if have_spice else "-",
            f"{r['peak_c']:.2f}", f"{r['peak_m']:.2f}",
        )
    console.print(table)
    zero_str = ", ".join(
        f"[{','.join(f'{v:g}' for v in z['control'])}]:{z['dbfs']:.0f}dB" for z in stab["zero"]
    )
    console.print(
        f"[bold]interpolation[/] trained ESR={interp['trained_mean']:.4f} | "
        f"held-out mean={interp['held_mean']:.4f} worst={interp['held_worst']:.4f} "
        f"p95={interp['held_p95']:.4f}\n"
        f"[bold]streaming[/] constant={res['streaming_err']:.1e}  moving-control={res['moving_err']:.1e}\n"
        f"[bold]real-time[/] constant RTF={res['rtf_c']['rtf']:.1f}x  "
        f"moving-knob RTF={res['rtf_m']['rtf']:.1f}x (realtime={res['rtf_m']['realtime']})\n"
        f"[bold]stability[/] zero-input [{zero_str}]  "
        f"hot-input peak={stab['hot_offline_peak']:.2f}/{stab['hot_stream_peak']:.2f} "
        f"<= bound {stab['out_bound']:.2f} (finite={stab['hot_finite']})\n"
        f"params={res['params']:,}"
    )

    # --- figures ---
    outdir = cfg.paths.outputs / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def _save(fig: Any, name: str) -> None:
        p = outdir / f"{circuit_name}_circe_{name}.png"
        fig.savefig(p)
        plt.close(fig)
        saved.append(str(p))

    from vguitar.spice.sampling import nn_distance

    hm = np.array([r["held"] for r in rows])
    esr_v = np.array([r["esr"] for r in rows])
    trained_ctrl = np.array([r["control"] for r in rows if not r["held"]], dtype=np.float32)
    dist_all = nn_distance(np.array([r["control"] for r in rows], np.float32), trained_ctrl, specs)
    _save(plot.fig_interp_vs_distance(dist_all, esr_v, hm, control_name=", ".join(axis_names),
                                      name=circuit_name), "interp_dist")
    if hist_path.exists():
        _save(plot.fig_training_curve(json.loads(hist_path.read_text()), name=circuit_name), "training")

    if single_drive:
        _run_single_drive_figs(plot, metrics, _save, outdir, circ, model, rows, specs, interp,
                               have_spice, sr, eval_di, render, heatmap, drives, held, cfg, saved)
    else:
        _run_multi_control_figs(plot, metrics, _save, outdir, circ, model, rows, specs, res,
                                have_spice, sr, eval_di, render, heatmap, cfg, saved, console)

    console.print(f"wrote {len(saved)} figures to {outdir}")
    return rows


def _run_single_drive_figs(plot, metrics, _save, outdir, circ, model, rows, specs, interp,
                           have_spice, sr, eval_di, render, heatmap, drives, held, cfg, saved):
    """The original one-knob figure set (drive sweep), kept byte-for-byte in spirit."""
    dr = np.array([r["drive"] for r in rows])
    hm = np.array([r["held"] for r in rows])
    drives = drives or _DEFAULT_DRIVES
    held = held if held is not None else _DEFAULT_HELD
    _save(plot.fig_esr_by_control(dr, np.array([r["esr"] for r in rows]), held_mask=hm,
                                  control_name="drive (V)", name=circ.name), "esr")
    _save(plot.fig_control_response(dr, np.array([r["peak_c"] for r in rows]),
                                    np.array([r["peak_m"] for r in rows]), held_mask=hm,
                                    ylabel="output peak (V)", control_name="drive (V)",
                                    name=circ.name), "gain")
    if have_spice:
        _save(plot.fig_control_response(dr, np.array([r["thd_c"] for r in rows]),
                                        np.array([r["thd_m"] for r in rows]), held_mask=hm,
                                        ylabel="THD", control_name="drive (V)", name=circ.name), "thd")
        tone = np.sin(2 * np.pi * 1000.0 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
        from vguitar.spice.runner import simulate

        held_vals = [r["drive"] for r in rows if r["held"]]
        picks = [rows[0]["drive"], held_vals[len(held_vals) // 2] if held_vals else rows[len(rows) // 2]["drive"],
                 rows[-1]["drive"]]
        pick_held = [g in set(held_vals) for g in picks]
        tone_c = {g: simulate(circ, (g * tone).astype(np.float32), sr) for g in picks}
        tone_m = {g: model.process(tone, np.array([g], np.float32)) for g in picks}
        _save(plot.fig_knob_harmonics(picks, [tone_c[g] for g in picks], [tone_m[g] for g in picks],
                                      pick_held, sr, name=circ.name), "harmonics")
        _save(plot.fig_knob_waveforms(picks, [tone_c[g] for g in picks], [tone_m[g] for g in picks],
                                      pick_held, sr, name=circ.name), "waveforms")
        if eval_di:
            di_rows = _eval_on_di(model, circ, sorted(set(held) | {drives[0], drives[-1]}), sr,
                                  control_specs=specs)
            _print_di(di_rows)
            for r in di_rows:
                g = float(r["control"][0])
                if g in set(held):
                    p = outdir / f"{circ.name}_circe_di_g{g * 1000:.0f}mV.png"
                    metrics.plot_compare(r["yc"], r["ym"], sr, p)
                    saved.append(str(p))
        if heatmap:
            tones, err_db = _control_freq_grid(model, circ, dr, sr, control_specs=specs)
            _save(plot.fig_drive_freq_error(dr, tones, err_db, control_name="drive", name=circ.name),
                  "drive_freq")
        if render:
            trained_d = sorted(r["drive"] for r in rows if not r["held"])
            held_mid = sorted(held)[len(held) // 2]
            wavs = _render_ab(model, circ, [trained_d[len(trained_d) // 2], held_mid, drives[-1]], sr,
                              cfg.paths.outputs / "audio", control_specs=specs)
            print(f"wrote {len(wavs)} A/B wavs to {cfg.paths.outputs / 'audio'}")


def _run_multi_control_figs(plot, metrics, _save, outdir, circ, model, rows, specs, res,
                            have_spice, sr, eval_di, render, heatmap, cfg, saved, console):
    """Multi-control figure set: per-axis response slices + a 2-D ESR heatmap."""
    base_row = np.array([s.default for s in specs], dtype=np.float32)
    if have_spice:
        for axis, s in enumerate(specs):
            sw = _axis_sweep(model, circ, specs, axis, base_row, sr)
            _save(plot.fig_control_response(sw["vals"], sw["thd_c"], sw["thd_m"], held_mask=None,
                                            ylabel="THD", control_name=s.name, name=circ.name),
                  f"axis_{s.name}_thd")
        # 2-D ESR heatmap over the first two axes (others at default).
        if heatmap and len(specs) >= 2:
            vi, vj, eg = _control_esr_grid(model, circ, specs, base_row, sr, axes=(0, 1), n=5)
            _save(plot.fig_control_esr_heatmap(vi, vj, eg, names=(specs[0].name, specs[1].name),
                                               name=circ.name), "esr_grid")
        if eval_di:
            di_rows = _eval_on_di(model, circ, res["held_grid"], sr, control_specs=specs)
            _print_di(di_rows)
            for k, r in enumerate(di_rows[:3]):
                p = outdir / f"{circ.name}_circe_di_{k}.png"
                metrics.plot_compare(r["yc"], r["ym"], sr, p)
                saved.append(str(p))
        if render:
            held_grid = res["held_grid"]
            picks = [res["train_grid"][len(res["train_grid"]) // 2]]
            if len(held_grid):
                picks.append(held_grid[len(held_grid) // 2])
            wavs = _render_ab(model, circ, np.array(picks, np.float32), sr,
                              cfg.paths.outputs / "audio", control_specs=specs)
            console.print(f"wrote {len(wavs)} A/B wavs to {cfg.paths.outputs / 'audio'}")


def _print_di(di_rows: list[dict[str, Any]]) -> None:
    from rich.console import Console

    parts = [
        "[" + ",".join(f"{v:g}" for v in np.asarray(r["control"]).reshape(-1)) + f"] ESR={r['esr']:.3f}"
        for r in di_rows
    ]
    Console().print("held-out guitar-DI: " + "  ".join(parts))
