"""Validation + visualization for CIRCE (the conditioned interactive model).

Runs the conditioned drive-knob experiment end to end and reports, separating
**trained** from **held-out** (interpolated) control settings:

* interpolation accuracy (per-setting + worst-case/p95, vs distance-to-trained),
* moving-control streaming equivalence (the knob turning mid-stream — GATE-4),
* generalization to a real held-out guitar-DI input,
* stability (zero-input quietness, hot-input saturation),
* real-time factor with a moving knob, and audio A/B renders to listen to.

Everything is reproducible via ``vguitar circe`` and reuses the library
(``spice.runner``, ``metrics``, ``realtime``, ``plotting``). The pure helpers
(``_interp_stats``, ``_stability_checks``, ``_rtf_moving``) take plain
arrays/models so they are unit-testable without the full pipeline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vguitar.config import Config
    from vguitar.models.circe import CIRCE

_DI_PATH = "assets/guitar_di_loop.wav"


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


def _interp_stats(rows: list[dict[str, Any]], trained_drives: list[float]) -> dict[str, Any]:
    """Held-out interpolation summary + per-held distance-to-nearest-trained."""
    span = (max(trained_drives) - min(trained_drives)) or 1.0
    per = [
        {"drive": r["drive"], "dist": min(abs(r["drive"] - t) for t in trained_drives) / span,
         "esr": r["esr"]}
        for r in rows if r["held"]
    ]
    h = [r["esr"] for r in rows if r["held"]]
    t = [r["esr"] for r in rows if not r["held"]]
    return {
        "trained_mean": float(np.mean(t)) if t else float("nan"),
        "held_mean": float(np.mean(h)) if h else float("nan"),
        "held_worst": float(np.max(h)) if h else float("nan"),
        "held_p95": float(np.percentile(h, 95)) if h else float("nan"),
        "per_held": per,
    }


def _stability_checks(model: CIRCE, drives: list[float], sr: int = 44_100) -> dict[str, Any]:
    """Zero-input quietness (per drive) + hot-input saturation (offline+streaming)."""
    out_bound = float(model.net.out_bound)
    probe = [drives[0], drives[len(drives) // 2], drives[-1]]
    zero = {}
    for g in probe:
        zero[g] = _dbfs(model.process(np.zeros(sr, np.float32), np.array([g], np.float32)))
    # hot input: 4x the (unit-peak) trained drive base, at the hottest drive
    hot = (4.0 * np.random.default_rng(0).standard_normal(sr)).astype(np.float32)
    cmax = np.array([drives[-1]], np.float32)
    y_off = model.process(hot, cmax)
    model.reset()
    y_st = np.concatenate(
        [model.process_block(hot[i : i + 128], cmax) for i in range(0, len(hot), 128)]
    )
    return {
        "out_bound": out_bound,
        "zero_dbfs": zero,
        "hot_offline_peak": float(np.max(np.abs(y_off))),
        "hot_stream_peak": float(np.max(np.abs(y_st))),
        "hot_finite": bool(np.isfinite(y_off).all() and np.isfinite(y_st).all()),
    }


def _rtf_moving(model: CIRCE, drives: list[float], sr: int = 44_100, block: int = 128,
                dur_s: float = 3.0) -> dict[str, float | bool]:
    """Real-time factor with the control changed every block (knob turning)."""
    import time

    n_blocks = max(2, int(dur_s * sr / block))
    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal(block).astype(np.float32) for _ in range(n_blocks)]
    ctrls = [np.array([drives[i % len(drives)]], np.float32) for i in range(n_blocks)]
    model.reset()
    model.process_block(blocks[0], ctrls[0])  # warm-up (exclude from timing)
    t0 = time.perf_counter()
    for b, c in zip(blocks[1:], ctrls[1:], strict=True):
        model.process_block(b, c)
    dt = time.perf_counter() - t0
    audio = (n_blocks - 1) * block / sr
    rtf = audio / dt if dt > 0 else float("inf")
    return {"rtf": float(rtf), "realtime": bool(rtf > 1.0), "ms_per_block": float(dt / max(n_blocks - 1, 1) * 1e3)}


def _eval_on_di(model: CIRCE, circ: Any, drives: list[float], sr: int,
                dur_s: float = 1.5) -> list[dict[str, Any]]:
    """Generalization to the real guitar-DI input across drive settings (ngspice)."""
    from vguitar.metrics import esr, multi_stft
    from vguitar.signals import load_di
    from vguitar.spice.runner import simulate

    di = load_di(_DI_PATH, sr, peak=1.0)[: int(dur_s * sr)]
    out = []
    for g in drives:
        yc = simulate(circ, (g * di).astype(np.float32), sr)
        ym = model.process(di, np.array([g], np.float32))
        w = 2048
        out.append({
            "drive": g, "esr": float(esr(yc[w:], ym[w:])), "stft": float(multi_stft(yc[w:], ym[w:])),
            "yc": yc, "ym": ym,
        })
    return out


def _drive_freq_grid(model: CIRCE, circ: Any, drives: list[float], sr: int,
                     tones: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Relative spectral-magnitude error (dB) of CIRCE vs circuit over drive x tone."""
    from vguitar.spice.runner import simulate

    tones = tones or [200.0, 500.0, 1000.0, 2000.0, 4000.0]
    t = np.arange(int(0.08 * sr)) / sr
    win = np.hanning(len(t))
    err = np.zeros((len(drives), len(tones)), dtype=np.float64)
    for i, g in enumerate(drives):
        for j, f in enumerate(tones):
            x = np.sin(2 * np.pi * f * t).astype(np.float32)
            yc = simulate(circ, (g * x).astype(np.float32), sr)
            ym = model.process(x, np.array([g], np.float32))
            yc_m = np.abs(np.fft.rfft(yc * win))
            ym_m = np.abs(np.fft.rfft(ym * win))
            rel = float(np.linalg.norm(ym_m - yc_m) / (np.linalg.norm(yc_m) + 1e-12))
            err[i, j] = 20.0 * np.log10(rel + 1e-6)
    return np.asarray(tones), err


def _render_ab(model: CIRCE, circ: Any, drives: list[float], sr: int, outdir: Any,
               dur_s: float = 2.5) -> list[str]:
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
    for g in drives:
        tag = f"g{g * 1000:.0f}mV"
        yc = simulate(circ, (g * di).astype(np.float32), sr)
        ym = model.process(di, np.array([g], np.float32))
        written.append(str(sf_write(sf, outdir / f"{tag}_circuit.wav", norm(yc), sr)))
        written.append(str(sf_write(sf, outdir / f"{tag}_circe.wav", norm(ym), sr)))
    return written


def sf_write(sf: Any, path: Any, y: np.ndarray, sr: int) -> Any:
    sf.write(path, np.asarray(y, dtype=np.float32), sr)
    return path


# --- main entry -----------------------------------------------------------
def run_circe_eval(
    circuit_name: str = "bjt",
    *,
    drives: list[float] | None = None,
    held: list[float] | None = None,
    cfg: Config | None = None,
    retrain: bool = False,
    regen: bool = False,
    eval_di: bool = True,
    render: bool = True,
    heatmap: bool = True,
    seg_dur_s: float = 2.0,
    epochs: int = 80,
    channels: int = 10,
) -> list[dict[str, Any]]:
    """Validate CIRCE on a one-knob (drive) sweep; print a report and write figures."""
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table

    from vguitar import metrics
    from vguitar import plotting as plot
    from vguitar.circuits import get_circuit
    from vguitar.config import Config, TrainConfig
    from vguitar.data import Dataset
    from vguitar.models.base import check_streaming, check_streaming_moving
    from vguitar.models.circe import CIRCE, _segments
    from vguitar.realtime import measure_rtf
    from vguitar.spice.runner import make_drive_dataset, simulate

    cfg = cfg or Config()
    cfg.paths.ensure()
    console = Console()
    drives = drives or [0.005, 0.02, 0.04, 0.08, 0.16]
    held = held or [0.01, 0.06, 0.12]
    circ = get_circuit(circuit_name)
    sr = cfg.data.sr

    # --- datasets (cached) ---
    train_path = cfg.paths.data / f"{circuit_name}_drive.npz"
    test_path = cfg.paths.data / f"{circuit_name}_drive_test.npz"
    if regen or not train_path.exists() or not test_path.exists():
        console.print("[dim]simulating drive sweep via ngspice...[/]")
        make_drive_dataset(circ, drives, cfg, seg_dur_s=seg_dur_s, seed=0).save(train_path)
        make_drive_dataset(circ, held, cfg, seg_dur_s=seg_dur_s, seed=100).save(test_path)
    train_ds, test_ds = Dataset.load(train_path), Dataset.load(test_path)

    # --- model (cached); persist training history sidecar when we train ---
    model_path = cfg.paths.runs / f"{circuit_name}.circe.model"
    hist_path = cfg.paths.runs / f"{circuit_name}.circe.history.json"
    if retrain or not model_path.exists():
        import torch

        torch.manual_seed(0)
        tr, va, _ = train_ds.split(0.12, 0.0001)
        model = CIRCE(n_control=1, channels=channels, n_blocks=2, n_layers=7)
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
            g = float(ds.controls[s, 0])
            pred = model.process(ds.x[s:e], ds.controls[s])
            w = min(256, (e - s) - 1)
            yv, pv = ds.y[s:e][w:], pred[w:]
            rows.append({
                "drive": g, "held": is_held, "esr": float(metrics.esr(yv, pv)),
                "peak_c": float(np.max(np.abs(yv))), "peak_m": float(np.max(np.abs(pv))),
            })
    rows.sort(key=lambda r: r["drive"])
    all_drives = [r["drive"] for r in rows]
    trained_drives = sorted(r["drive"] for r in rows if not r["held"])

    # --- tone-based THD + waveforms (ngspice) ---
    tone = np.sin(2 * np.pi * 1000.0 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
    have_spice, tone_c, tone_m = True, {}, {}
    try:
        for r in rows:
            g = r["drive"]
            tone_c[g] = simulate(circ, (g * tone).astype(np.float32), sr)
            tone_m[g] = model.process(tone, np.array([g], np.float32))
            r["thd_c"] = _thd_signal(tone_c[g], sr)
            r["thd_m"] = _thd_signal(tone_m[g], sr)
    except Exception as exc:  # ngspice optional
        have_spice = False
        console.print(f"[yellow]ngspice unavailable; skipping tone/DI/heatmap probes ({exc})[/]")

    # --- validation aggregates ---
    interp = _interp_stats(rows, trained_drives)
    stab = _stability_checks(model, all_drives, sr)
    rtf_c = measure_rtf(model, sr=sr, block=cfg.realtime.block_size)
    rtf_m = _rtf_moving(model, all_drives, sr=sr, block=cfg.realtime.block_size)
    streaming_err = check_streaming(model, n=4096, block=128, atol=2e-3)
    ramp = np.linspace(min(all_drives), max(all_drives), 32, dtype=np.float32).reshape(-1, 1)
    moving_err = check_streaming_moving(model, control_traj=ramp, n=4096, block=128, atol=2e-3)

    # --- report table ---
    table = Table(title=f"CIRCE validation — {circuit_name} drive knob")
    for c in ("drive (mV)", "set", "ESR", "THD circ", "THD CIRCE", "peak circ (V)", "peak CIRCE (V)"):
        table.add_column(c, justify="center" if c == "set" else "right")
    for r in rows:
        table.add_row(
            f"{r['drive'] * 1000:.1f}", "[yellow]held[/]" if r["held"] else "train",
            f"{r['esr']:.4f}",
            f"{r.get('thd_c', float('nan')):.3f}" if have_spice else "-",
            f"{r.get('thd_m', float('nan')):.3f}" if have_spice else "-",
            f"{r['peak_c']:.2f}", f"{r['peak_m']:.2f}",
        )
    console.print(table)
    zero_str = ", ".join(f"{g * 1000:.0f}mV:{db:.0f}dB" for g, db in stab["zero_dbfs"].items())
    console.print(
        f"[bold]interpolation[/] trained ESR={interp['trained_mean']:.4f} | "
        f"held-out mean={interp['held_mean']:.4f} worst={interp['held_worst']:.4f} "
        f"p95={interp['held_p95']:.4f}\n"
        f"[bold]streaming[/] constant={streaming_err:.1e}  moving-control={moving_err:.1e}\n"
        f"[bold]real-time[/] constant RTF={rtf_c['rtf']:.1f}x  moving-knob RTF={rtf_m['rtf']:.1f}x "
        f"(realtime={rtf_m['realtime']})\n"
        f"[bold]stability[/] zero-input [{zero_str}]  "
        f"hot-input peak={stab['hot_offline_peak']:.2f}/{stab['hot_stream_peak']:.2f} "
        f"<= bound {stab['out_bound']:.2f} (finite={stab['hot_finite']})\n"
        f"params={model.num_params():,}"
    )

    # --- figures ---
    outdir = cfg.paths.outputs / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def _save(fig, stem: str) -> None:
        p = outdir / f"{circuit_name}_circe_{stem}.png"
        fig.savefig(p)
        plt.close(fig)
        saved.append(str(p))

    dr = np.array(all_drives)
    hm = np.array([r["held"] for r in rows])
    _save(plot.fig_esr_by_control(dr, np.array([r["esr"] for r in rows]), held_mask=hm,
                                  control_name="drive (V)", name=circuit_name), "esr")
    _save(plot.fig_control_response(dr, np.array([r["peak_c"] for r in rows]),
                                    np.array([r["peak_m"] for r in rows]), held_mask=hm,
                                    ylabel="output peak (V)", control_name="drive (V)",
                                    name=circuit_name), "gain")
    _save(plot.fig_interp_vs_distance(
        np.array([0.0 if not r["held"] else next(p["dist"] for p in interp["per_held"] if p["drive"] == r["drive"]) for r in rows]),
        np.array([r["esr"] for r in rows]), hm, control_name="drive", name=circuit_name), "interp_dist")
    if hist_path.exists():
        _save(plot.fig_training_curve(json.loads(hist_path.read_text()), name=circuit_name), "training")

    if have_spice:
        _save(plot.fig_control_response(dr, np.array([r["thd_c"] for r in rows]),
                                        np.array([r["thd_m"] for r in rows]), held_mask=hm,
                                        ylabel="THD", control_name="drive (V)", name=circuit_name), "thd")
        held_vals = [r["drive"] for r in rows if r["held"]]
        picks = [rows[0]["drive"], held_vals[len(held_vals) // 2], rows[-1]["drive"]]
        pick_held = [g in set(held_vals) for g in picks]
        _save(plot.fig_knob_harmonics(picks, [tone_c[g] for g in picks], [tone_m[g] for g in picks],
                                      pick_held, sr, name=circuit_name), "harmonics")
        _save(plot.fig_knob_waveforms(picks, [tone_c[g] for g in picks], [tone_m[g] for g in picks],
                                      pick_held, sr, name=circuit_name), "waveforms")
        if eval_di:
            di_rows = _eval_on_di(model, circ, sorted(set(held) | {drives[0], drives[-1]}), sr)
            console.print("held-out guitar-DI: " + "  ".join(
                f"{r['drive'] * 1000:.0f}mV ESR={r['esr']:.3f}" for r in di_rows))
            for r in di_rows:
                if r["drive"] in set(held):
                    metrics.plot_compare(r["yc"], r["ym"], sr,
                                         outdir / f"{circuit_name}_circe_di_g{r['drive'] * 1000:.0f}mV.png")
                    saved.append(str(outdir / f"{circuit_name}_circe_di_g{r['drive'] * 1000:.0f}mV.png"))
        if heatmap:
            tones, err_db = _drive_freq_grid(model, circ, all_drives, sr)
            _save(plot.fig_drive_freq_error(dr, tones, err_db, control_name="drive", name=circuit_name),
                  "drive_freq")
        if render:
            held_mid = sorted(held)[len(held) // 2]
            wavs = _render_ab(model, circ, [trained_drives[len(trained_drives) // 2], held_mid, drives[-1]],
                              sr, cfg.paths.outputs / "audio")
            console.print(f"wrote {len(wavs)} A/B wavs to {cfg.paths.outputs / 'audio'}")

    console.print(f"wrote {len(saved)} figures to {outdir}")
    return rows
