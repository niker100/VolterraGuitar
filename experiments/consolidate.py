"""Consolidate the neural benchmarking era into the machine-readable knowledge store.

One-shot (re-runnable) builder: a hand-curated CURATION table (question + verdict
per experiment, distilled from docs/archive/sota-campaign.md and
experiments/archive/README.md) plus key numbers, emitted to
``outputs/knowledge/findings.json`` (tracked in git -- the durable record) with
three human summary plots beside it. Raw result JSONs/figures stay local under
``outputs/archive/``; every entry links to its sources, and the builder fails if
a linked file is missing.

Run: uv run python -m experiments.consolidate
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vguitar.plotting import OKABE_ITO, apply_style

OUT = Path("outputs/knowledge")
ARC = "outputs/archive"

# --- the curated record ------------------------------------------------------
# id -> {phase, question, verdict, key_numbers, links}. Verdicts are the distilled
# conclusions; key_numbers are the load-bearing measurements (held-out ESR unless
# stated). Links point at the raw JSON / script / figure for every claim.
CURATION: dict[str, dict[str, Any]] = {
    "families.benchmark": {
        "phase": "model families",
        "question": "Which model family emulates SPICE-simulated circuits best "
                    "(FIR / Volterra / parallel-cascade / Wiener-Hammerstein / TCN / "
                    "RNN / SSM / CIRCE3)?",
        "verdict": "CIRCE3 (input-scaling TCN) wins accuracy on every circuit; classical "
                   "closed-form fits (FIR/Volterra/WH) are seconds-fast but plateau 10-100x "
                   "worse - they lack trained nonlinear features. RNN competitive but "
                   "slower; SSM mid-field.",
        "key_numbers": {"circe3_esr_range": [0.001, 0.005], "volterra_esr_range": [0.03, 0.3]},
        "links": {"results": [f"{ARC}/benchmark.json"],
                  "figures": [f"{ARC}/figs/compare_esr_matrix.png"]},
    },
    "circe.architecture": {
        "phase": "architecture",
        "question": "What architecture choices matter for a TCN circuit emulator?",
        "verdict": "Input-scaling (fold signal controls into input gain) beats conditioning "
                   "nets; minimal zero-init FiLM for system controls; mixed activations "
                   "(tanh/gelu/relu/abs/snake) beat gated on hard circuits, tie smooth; "
                   "2x internal oversampling removes aliasing (OS>2 measured WORSE); "
                   "grad-clip 1.0 optimal; raw channel width inert and destabilizing.",
        "key_numbers": {"adopted": "ch24/nb2/L10, mixed act, OS2, ~60k params"},
        "links": {"results": [f"{ARC}/mixed_vs_gated_final.json", f"{ARC}/radical_arch_ablation.json",
                              f"{ARC}/radical_gradclip_test.json", f"{ARC}/wavefolder_os_ab.json"],
                  "docs": ["docs/archive/optimal-architecture.md", "docs/archive/CIRCE3-modelcard.md"]},
    },
    "spectral.branch": {
        "phase": "architecture",
        "question": "Do spectral/frequency-domain hybrids (FNO, AFNO, STFT-mix, wavelet, "
                    "global filter, 12 variants) beat the time-domain TCN?",
        "verdict": "No uniform win anywhere; dropped entirely.",
        "key_numbers": {},
        "links": {"results": [f"{ARC}/spectral_zoo.json", f"{ARC}/spectral_leads.json"],
                  "figures": [f"{ARC}/figs/frontier/spectral_zoo.png"]},
    },
    "sota.dc_blocker_artifact": {
        "phase": "sota campaign",
        "question": "Why are the asymmetric circuits floored at ESR 0.05-0.17?",
        "verdict": "MEASUREMENT ARTIFACT: training scored raw output but inference applied a "
                   "1 Hz DC blocker; circuits with real level-dependent output DC floored a "
                   "perfect model (asym floor 0.0518). Fix: dcblock off (DC is genuine "
                   "circuit output). asym_clipper 0.0523 -> 0.0007 (75x). Lesson: validate "
                   "the eval path with a perfect-model floor test.",
        "key_numbers": {"asym_clipper": [0.0523, 0.0007], "wavefolder": [0.200, 0.126]},
        "links": {"results": [f"{ARC}/sota/dc_rebaseline.json"],
                  "figures": [f"{ARC}/figs/frontier/sota_dc_rebaseline.png"]},
    },
    "sota.depth": {
        "phase": "sota campaign",
        "question": "Does ~2x receptive field (nb2/L10) move the near-miss circuits?",
        "verdict": "Strong but mixed: cracks jfet (0.0094->0.0020), halves hard_clipper, "
                   "helps fullwave, neutral bjt - but HURTS crossover (dead-zone prefers "
                   "shallow). Adopted nb2/L10.",
        "key_numbers": {"jfet": [0.0094, 0.0020], "hard_clipper": [0.0933, 0.0558],
                        "crossover": [0.0225, 0.0299]},
        "links": {"results": [f"{ARC}/sota/depth_sweep.json"],
                  "figures": [f"{ARC}/figs/frontier/sota_depth_sweep.png"]},
    },
    "sota.iir_memory": {
        "phase": "sota campaign",
        "question": "Can learnable one-pole IIR state channels give the TCN the long memory "
                    "hysteretic_fuzz needs (tau 22-440ms >> 21ms receptive field)?",
        "verdict": "Directionally yes (-29% hysteretic, -18% crossover at prototype), "
                   "adopted as n_state=4 with ZERO-INIT input weights (random init perturbs "
                   "memoryless circuits). But extensions (8 poles, tau to 2s, 2x window) "
                   "were all flat-to-worse: linear memory saturates; true hysteresis needs "
                   "input-DEPENDENT (gated) state - never built (pivot).",
        "key_numbers": {"hysteretic_prototype_delta_pct": -29, "best_hysteretic": 0.0139},
        "links": {"results": [f"{ARC}/sota/iir_probe.json", f"{ARC}/sota/memory_probe.json"],
                  "figures": [f"{ARC}/figs/frontier/lever_iir_probe.png"]},
    },
    "sota.corners_inert": {
        "phase": "sota campaign",
        "question": "Do rectified input features / learnable corner thresholds / Fourier "
                    "output shapers crack the discontinuity circuits?",
        "verdict": "All INERT (thresholds never move from init; SGD finds no gradient use). "
                   "Input-side corner primitives do not help; the untried placement was a "
                   "parallel PWL waveshaper path from the input (Wiener-style).",
        "key_numbers": {"hard_clipper_delta_pct": -1},
        "links": {"results": [f"{ARC}/sota/corner_probe.json", f"{ARC}/sota/probe_levers.json"],
                  "figures": [f"{ARC}/figs/frontier/lever_corner_probe.png"]},
    },
    "sota.data_volume": {
        "phase": "sota campaign",
        "question": "Does 3x training data (seg_dur 8s regenerated sweeps) lower held-ESR?",
        "verdict": "Big where data-limited: tube_screamer -47% (0.0027), crossover -32%, "
                   "jfet -16..-35%, hard_clipper -9%. FLAT on fullwave and hysteretic "
                   "(not data-limited). v2 sweeps were adopted into the protocol.",
        "key_numbers": {"tube_screamer": [0.0051, 0.0027], "crossover": [0.0413, 0.0282],
                        "jfet_b12": [0.0053, 0.0034]},
        "links": {"results": [f"{ARC}/sota/data_v2_ab.json", f"{ARC}/sota/data_v2_ab_b12.json"],
                  "figures": [f"{ARC}/figs/frontier/lever_data_v2_ab.png"]},
    },
    "closedform.elm_hammerstein": {
        "phase": "closed form",
        "question": "Can pure closed-form fits (random features + ridge lstsq; structured "
                    "Hammerstein basis) reach the 0.005 bar without backprop?",
        "verdict": "No: ~5s fits but plateau at 0.03-0.3 ESR regardless of width - a "
                   "feature-QUALITY limit, not feature count. Hammerstein worse than random "
                   "features (too shallow). Closed-form readout on trained features (VarPro) "
                   "is the keeper instead.",
        "key_numbers": {"elm_jfet": 0.035, "elm_hard_clipper": 0.21, "fit_seconds": 5},
        "links": {"results": [f"{ARC}/sota/elm_probe.json", f"{ARC}/sota/hammerstein_probe.json"],
                  "figures": [f"{ARC}/figs/frontier/sota_probe_wall.png"]},
    },
    "closedform.varpro": {
        "phase": "closed form",
        "question": "Does Variable Projection (closed-form ridge solve of the linear readout "
                    "inside the training loop, Golub-Pereyra) beat joint SGD?",
        "verdict": "Major speed/accuracy tool on SMOOTH circuits: ~3x fewer epochs, and on "
                   "the strong architecture also sharper minima (jfet 0.0010 vs 0.0045). "
                   "fp64 solve is a trap on consumer GPUs (1/64 throughput) - fp32 + "
                   "scale-relative ridge matches joint per-epoch cost. NOT uniform-safe: "
                   "breaks discontinuity circuits (wavefolder 0.245 -> 0.84). VarPro x "
                   "big-batch ANTI-compounds (trunk needs small-batch gradient noise).",
        "key_numbers": {"jfet": [0.0045, 0.0010], "epochs_to_plateau_ratio": 3,
                        "vp60_b12_speedup": 3.0},
        "links": {"results": [f"{ARC}/sota/unified_varpro.json", f"{ARC}/sota/varpro_conv.json",
                              f"{ARC}/sota/fast_stack.json"],
                  "figures": [f"{ARC}/figs/frontier/varpro_conv.png", f"{ARC}/figs/frontier/fast_stack.png"]},
    },
    "training.stability": {
        "phase": "training",
        "question": "Why do training runs stochastically collapse to held-ESR ~1.0?",
        "verdict": "A degenerate predict-mean basin entered in the first epochs. bf16 AMP "
                   "lands there DETERMINISTICALLY on some circuit x batch combos (8-bit "
                   "mantissa rounds away early gradient signal - 16-bit training disqualified); "
                   "fp32 big-batch lands there stochastically (GPU nondeterminism; same seed, "
                   "different outcome). Fixes: 5-epoch LR warmup (free), non-finite step "
                   "guard, collapse retry-with-shifted-seed. Wall-clock measurements across "
                   "processes are contaminated by desktop GPU contention - only in-process "
                   "ratios are meaningful.",
        "key_numbers": {"collapse_rate_no_warmup_b96": "2/3 on bjt",
                        "b12_warmup_cost": "none (0.0044 == 0.0044)"},
        "links": {"results": [f"{ARC}/sota/speed_ab.json", f"{ARC}/sota/hc_diag.json"],
                  "figures": [f"{ARC}/figs/frontier/speed_ab.png", f"{ARC}/figs/frontier/hc_diag.png"]},
    },
    "sota.final_state": {
        "phase": "sota campaign",
        "question": "Where did the uniform <0.005 push end?",
        "verdict": "4/9 circuits under held-ESR 0.005 with one uniform config (dcblock off + "
                   "nb2/L10 + n_state=4 + OS2): asym 0.0008, ts 0.0026, bjt 0.0044, jfet "
                   "0.0045; fullwave 0.0057 near. The WORST failures (wavefolder 0.245, "
                   "hard_clipper 0.058, hysteretic 0.014, fullwave) are all SYNTHETIC "
                   "behavioral B-source circuits (sin-fold map, tanh(60x) knee, behavioral "
                   "hysteresis/abs); the one realistic circuit above target, crossover "
                   "(0.0154), was still improving with data (-32% at 3x). This motivated "
                   "the pivot: realistic circuits + physical DK solver instead of "
                   "ever-larger trained black boxes.",
        "key_numbers": {"under_target": 4, "of": 9, "wavefolder_floor": 0.245},
        "links": {"results": [f"{ARC}/sota/unified_varpro.json"],
                  "figures": [f"{ARC}/figs/frontier/sota_unified_varpro.png"],
                  "docs": ["docs/archive/sota-campaign.md"]},
    },
}

#: the final uniform-config leaderboard (unified_varpro.json, config "standard", seed 0)
FINAL_LEADERBOARD = {
    "asym_clipper": 0.0008, "tube_screamer": 0.0026, "bjt": 0.0044, "jfet": 0.0045,
    "fullwave_rectifier": 0.0057, "hysteretic_fuzz": 0.0139, "crossover": 0.0154,
    "hard_clipper": 0.0578, "wavefolder": 0.2453,
}

HEADLINE = [
    "CIRCE3 (input-scaling dilated TCN, mixed activations, zero-init FiLM, 4 IIR state "
    "channels, 2x oversampling, ~60k params) is the frozen neural SOTA: 4/9 benchmark "
    "circuits under held-ESR 0.005 with one uniform config.",
    "The worst resisting circuits (wavefolder 0.245, hard_clipper 0.058, hysteretic "
    "0.014, fullwave) are all SYNTHETIC behavioral stress tests with non-physical "
    "discontinuities; realistic-component circuits were under target, near it, or "
    "still improving with data (crossover). This is the empirical case for the "
    "physical-emulator pivot.",
    "Closed-form readouts (VarPro) are a 3x training speedup and sharper on smooth "
    "circuits, but pure closed-form fits plateau 10-100x off - trained features were "
    "the bottleneck, which a physical solver sidesteps entirely.",
    "The biggest 'model failure' of the campaign was a measurement artifact (DC-blocker "
    "train/eval mismatch): always validate the eval path with a perfect-model floor test.",
    "Training landmines catalogued: bf16 deterministically unsafe for this loss; "
    "big-batch needs LR warmup; GPU training is non-reproducible run-to-run; guard "
    "non-finite steps and retry collapses.",
]


def _check_links(entry_id: str, entry: dict[str, Any]) -> list[str]:
    missing = []
    for paths in entry.get("links", {}).values():
        for p in paths:
            if not Path(p).exists():
                missing.append(f"{entry_id}: {p}")
    return missing


def _plot_leaderboard() -> None:
    circuits = list(FINAL_LEADERBOARD)
    vals = np.array([FINAL_LEADERBOARD[c] for c in circuits])
    colors = [OKABE_ITO["green"] if v < 0.005 else OKABE_ITO["vermillion"] for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(circuits)), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.005, ls="--", color=OKABE_ITO["black"], lw=1.4)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(circuits)))
    ax.set_xticklabels(circuits, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("held-out ESR (log)")
    ax.set_title("Final neural-era leaderboard: CIRCE3 uniform config (frozen 2026-06-10)\n"
                 "green = under the 0.005 target; the worst failures are synthetic "
                 "behavioral stress tests (crossover is the realistic exception)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "leaderboard_final.png")
    plt.close(fig)


def _plot_accuracy_vs_rtf() -> None:
    rows = json.loads(Path(f"{ARC}/benchmark.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    by_model: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r.get("esr") is not None and r.get("rtf"):
            by_model.setdefault(r["model"], []).append((float(r["rtf"]), float(r["esr"])))
    for j, (m, pts) in enumerate(sorted(by_model.items())):
        arr = np.array(pts)
        ax.scatter(arr[:, 0], arr[:, 1], s=42, label=m,
                   color=list(OKABE_ITO.values())[j % len(OKABE_ITO)], edgecolor="black",
                   linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(0.005, ls="--", color=OKABE_ITO["black"], lw=1.2)
    ax.set_xlabel("real-time factor (CPU, higher = faster)")
    ax.set_ylabel("test ESR (log)")
    ax.set_title("Model families: accuracy vs real-time speed (frozen era)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_rtf.png")
    plt.close(fig)


def _plot_timeline() -> None:
    stages = ["baseline", "+dcblock off", "+depth L10", "+IIR state\n(unified)", "+v2 data\n(partial)"]
    under = [1, 2, 3, 4, 4]
    worst = [0.1746, 0.126, 0.126, 0.245, 0.245]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.step(np.arange(len(stages)), under, where="mid", color=OKABE_ITO["blue"], lw=2)
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel("circuits under 0.005 (of 9)", color=OKABE_ITO["blue"])
    ax.set_ylim(0, 9)
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(stages)), worst, "o--", color=OKABE_ITO["vermillion"], lw=1.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("worst-circuit ESR (log)", color=OKABE_ITO["vermillion"])
    ax.set_title("SOTA campaign progress: each lever moved the count; the synthetic\n"
                 "wavefolder never moved - the wall that motivated the pivot", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "campaign_timeline.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [m for eid, e in CURATION.items() for m in _check_links(eid, e)]
    if missing:
        raise SystemExit("findings.json link targets missing:\n  " + "\n  ".join(missing))
    store = {
        "schema_version": 1,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "frozen",
        "headline_findings": HEADLINE,
        "final_leaderboard": {
            c: {"model": "circe3", "held_esr": v, "pass_0p005": v < 0.005}
            for c, v in FINAL_LEADERBOARD.items()
        },
        "experiments": CURATION,
    }
    path = OUT / "findings.json"
    path.write_text(json.dumps(store, indent=2))
    print(f"wrote {path} ({len(CURATION)} experiments)")
    _plot_leaderboard()
    _plot_accuracy_vs_rtf()
    _plot_timeline()
    print(f"wrote {OUT}/leaderboard_final.png, accuracy_vs_rtf.png, campaign_timeline.png")


if __name__ == "__main__":
    main()
