"""Reusable train+score harness for the SOTA push.

One *uniform* CIRCE3 config per circuit, scored by held-ESR (latency-aligned —
the same eval as ``outputs/final_numbers.json``) plus CPU RTF, logged with
timestamps and persisted incrementally so a background run is tailable +
stall-detectable and partial results survive a crash or a kill.

Protocol (identical to ``final_numbers`` / ``compare``): every circuit is a
single drive control (signal-acting, folded into the input), trained on its
drive *sweep* and tested at nominal drive on held-out *content* — so held-ESR is
directly comparable to the shipping baseline. Configs are plain dicts of CIRCE3
ctor kwargs plus a ``"label"``; the same dict is used for all circuits, which is
how "no per-circuit tailoring" is enforced structurally.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common import held_esr, make_log
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3

#: circuit key -> (drive-sweep dataset, held-out test dataset, kind).
#: PROTOCOL CHANGE 2026-06-10: the 6 regenerated *_v2 sweeps (seg_dur 8 s, ~3x data)
#: are adopted after the data_v2_ab A/B (ts -47%, crossover -32%, jfet -16%,
#: hard_clipper -9%, fullwave/hysteretic flat, nothing worse). TEST SETS UNCHANGED,
#: so held-ESR stays comparable; pre-v2 sweep numbers are historical.
CIRCUITS: dict[str, tuple[str, str, str]] = {
    "bjt": ("bjt_bench_sweep", "bjt_bench_fp_test", "smooth"),
    "jfet": ("jfet_bench_sweep_v2", "jfet_bench_fp_test", "smooth"),
    "tube_screamer": ("tube_screamer_bench_sweep_v2", "tube_screamer_bench_fp_test", "smooth"),
    "crossover": ("crossover_classb_edge_sweep_v2", "crossover_classb_edge_test", "hard"),
    "wavefolder": ("wavefolder_edge_sweep", "wavefolder_edge_test", "hard"),
    "asym_clipper": ("asym_clipper_edge_sweep", "asym_clipper_edge_test", "hard"),
    "hard_clipper": ("hard_clipper_edge_sweep_v2", "hard_clipper_edge_test", "hard"),
    "fullwave_rectifier": ("fullwave_rectifier_edge_sweep_v2",
                           "fullwave_rectifier_edge_test", "hard"),
    "hysteretic_fuzz": ("hysteretic_fuzz_edge_sweep_v2", "hysteretic_fuzz_edge_test", "hard"),
}
SMOOTH = [k for k, v in CIRCUITS.items() if v[2] == "smooth"]
HARD = [k for k, v in CIRCUITS.items() if v[2] == "hard"]
ALL = list(CIRCUITS)
TARGET = 0.005  # the held-ESR bar that must hold on every circuit

#: the unified production config (best uniform yet, 4/9 under 0.005) — the single
#: source of truth every campaign driver builds on.
UNIFIED: dict[str, Any] = {"channels": 24, "n_blocks": 2, "n_layers": 10, "oversample": 2,
                           "dcblock_fc": 0.0, "n_state": 4}

#: screening-speed training (speed_ab verdict): fp32 batch 96 / lr 9e-3 is ~1.4-2.2x
#: faster wall-clock with held-ESR at-or-near the b12 control (jfet even improves);
#: bf16 AMP is deterministically UNSAFE (collapses hard_clipper/bjt at some batches —
#: exact-value reproducible, not stochastic). Use for relative A/B probes (the
#: comparison is within-config); final/multi-seed leaderboard runs stay at the
#: accuracy-proven defaults (batch 12 / lr 3e-3).
SCREEN: dict[str, Any] = {"batch_size": 96, "lr": 9e-3}

_DATA = Path("data")


def load_circuit(circuit: str) -> tuple[Dataset, Dataset, str]:
    sweep_nm, test_nm, kind = CIRCUITS[circuit]
    return Dataset.load(_DATA / f"{sweep_nm}.npz"), Dataset.load(_DATA / f"{test_nm}.npz"), kind


# config keys that are NOT CIRCE3 ctor args: campaign label + per-config training
# overrides (a config dict can pin its own window/batch/lr/precision when the lever
# under test needs it — e.g. a longer seq_len arm halves batch to hold GPU memory).
_TRAIN_KEYS = ("label", "varpro", "seq_len", "batch_size", "lr", "amp", "lr_warmup")

#: held-ESR above this = the degenerate predict-mean basin (an output that ignores
#: the input scores ~1.0; the worst GENUINE result in the suite is wavefolder 0.245).
COLLAPSE = 0.5


def build(cfg: dict[str, Any], device: str) -> CIRCE3:
    """CIRCE3 from a config dict (single drive control, folded into the input)."""
    kw = {k: v for k, v in cfg.items() if k not in _TRAIN_KEYS}
    if "rect_thr" in kw:
        kw["rect_thr"] = tuple(kw["rect_thr"])
    return CIRCE3(n_control=1, signal_idx=(0,), device=device, **kw)


def _rtf(model: CIRCE3, block: int = 512) -> float:
    """CPU RTF of the (pure-numpy, bit-exact) streaming twin at a realistic host
    block. block=128 is an unrepresentative worst case for a per-block numpy path
    (per-call overhead dominates); 512 (~12 ms, within the <=1024-sample latency
    budget) is the relative number we track across configs."""
    from vguitar.realtime import measure_rtf

    try:
        return float(measure_rtf(model, sr=44_100, block=block, dur_s=2.0)["rtf"])
    except Exception:
        return float("nan")


def fit_score(
    tr: Dataset,
    ts: Dataset,
    cfg: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    device: str = "cuda",
    batch_size: int = 12,
    lr: float = 3e-3,
    amp: bool = False,
    retries: int = 1,
    fallback: bool = True,
) -> tuple[CIRCE3, dict[str, Any]]:
    """Train one config on one (train, test) pair with collapse handling.

    Training can stochastically land in a degenerate predict-mean basin (held-ESR
    ~1.0; GPU nondeterminism decides — same seed, different outcome). A collapsed
    run (held > COLLAPSE) is retried with a shifted seed, and as a last resort
    falls back to the accuracy-proven safe point (batch 12 / lr 3e-3 / fp32) so a
    campaign cell is never reported as a collapse artifact. Every fit gets a
    5-epoch LR warmup by default (measured free at b12; removes the big-batch
    early-overshoot collapse on all but the knife-edge circuits)."""
    t = time.time()
    plan = [(cfg.get("batch_size", batch_size), cfg.get("lr", lr), cfg.get("amp", amp),
             seed + 1000 * k) for k in range(retries + 1)]
    if fallback and (plan[0][0] != 12 or plan[0][1] != 3e-3 or plan[0][2]):
        plan.append((12, 3e-3, False, seed))
    attempts: list[float] = []
    for bs, lr_k, amp_k, seed_k in plan:
        torch.manual_seed(seed_k)
        model = build(cfg, device)
        model.fit(
            tr,
            ts,
            TrainConfig(epochs=epochs, lr=lr_k, seq_len=cfg.get("seq_len", 4096),
                        batch_size=bs, warmup=2048, seed=seed_k, amp=amp_k,
                        varpro=cfg.get("varpro", False),
                        lr_warmup=cfg.get("lr_warmup", 5)),
        )
        esr = held_esr(model, ts)
        attempts.append(esr)
        if esr < COLLAPSE:
            break
    return model, {
        "held": attempts[-1],
        "params": int(model.num_params()),
        "secs": time.time() - t,
        "attempts": attempts,
    }


def train_eval(
    circuit: str,
    cfg: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    device: str,
    batch_size: int = 12,
    lr: float = 3e-3,
    amp: bool = False,
) -> dict[str, Any]:
    """Train one uniform config on one circuit; return held-ESR + RTF + params."""
    tr, ts, kind = load_circuit(circuit)
    model, r = fit_score(tr, ts, cfg, seed=seed, epochs=epochs, device=device,
                         batch_size=batch_size, lr=lr, amp=amp)
    return r | {"rtf": _rtf(model), "kind": kind}


def run_campaign(
    name: str,
    configs: Sequence[dict[str, Any]],
    circuits: Sequence[str],
    seeds: Sequence[int] = (0,),
    epochs: int = 150,
    device: str = "cuda",
    batch_size: int = 12,
    lr: float = 3e-3,
    amp: bool = False,
) -> dict[str, Any]:
    """Train every (config x circuit x seed) cell serially, logging + persisting
    incrementally. Resumes from any existing ``outputs/sota/<name>.json`` so a
    killed run can be relaunched without redoing finished cells."""
    log = make_log(f"sota_{name}")
    out = Path(f"outputs/sota/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
    labels = [c["label"] for c in configs]
    log(f"campaign={name} configs={labels} circuits={list(circuits)} "
        f"seeds={list(seeds)} epochs={epochs} device={device}")
    for cfg in configs:
        label = cfg["label"]
        results.setdefault(label, {})
        for circuit in circuits:
            cell = results[label].setdefault(
                circuit, {"held": [], "rtf": None, "params": None, "secs": [],
                          "kind": CIRCUITS[circuit][2]}
            )
            for si, seed in enumerate(seeds):
                if len(cell["held"]) > si:  # already computed (resume)
                    continue
                log(f"START {label:16s} {circuit:18s} seed{seed}")
                try:
                    r = train_eval(circuit, cfg, seed=seed, epochs=epochs,
                                   device=device, batch_size=batch_size, lr=lr, amp=amp)
                    cell["held"].append(r["held"])
                    cell["secs"].append(r["secs"])
                    cell["rtf"] = r["rtf"]
                    cell["params"] = r["params"]
                    flag = "<<<" if r["held"] < TARGET else ""
                    note = (f" [retried: {[f'{a:.3f}' for a in r['attempts'][:-1]]}]"
                            if len(r["attempts"]) > 1 else "")
                    log(f"DONE  {label:16s} {circuit:18s} seed{seed} held={r['held']:.4f} "
                        f"rtf={r['rtf']:.2f} params={r['params']} ({r['secs']:.0f}s) {flag}{note}")
                except Exception as exc:
                    log(f"FAIL  {label:16s} {circuit:18s} seed{seed}: {exc}")
                    log(traceback.format_exc())
                out.write_text(json.dumps(results, indent=2))
    _summarize(results, configs, circuits, log)
    return results


def _summarize(
    results: dict[str, Any],
    configs: Sequence[dict[str, Any]],
    circuits: Sequence[str],
    log: Callable[[str], None],
) -> None:
    log("=== SUMMARY: mean held-ESR (<<< = under 0.005) ===")
    for cfg in configs:
        label = cfg["label"]
        worst = 0.0
        for circuit in circuits:
            held = results.get(label, {}).get(circuit, {}).get("held") or []
            mean = float(np.mean(held)) if held else float("nan")
            worst = max(worst, mean if np.isfinite(mean) else worst)
            flag = "<<<" if np.isfinite(mean) and mean < TARGET else ""
            rtf = results.get(label, {}).get(circuit, {}).get("rtf")
            log(f"  {label:16s} {circuit:18s} {mean:.4f} rtf={rtf} {flag}")
        log(f"  {label:16s} {'WORST':18s} {worst:.4f}")
