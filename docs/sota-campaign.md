# SOTA campaign — one uniform model, held-ESR < 0.005 on every circuit

**Goal (set 2026-06-09).** Craft one clean, generalized, real-time-affordable
architecture for nonlinear audio with exogenous controls that reaches **held-out
ESR < 0.005 on every benchmarked circuit, with no per-circuit tailoring**. Use the
4090 fully; loosen the real-time bar only slightly (RTF may sit just below 1 if
the accuracy gap is otherwise closed); keep every result; visualize everything;
never sit idle while a benchmark runs. One clean solution — no compatibility shims,
no legacy code.

This is run as a series of GPU-bound **campaigns** through the reusable harness in
`experiments/sota/harness.py` (train one uniform config per circuit → held-ESR
[latency-aligned] + CPU RTF, logged + persisted incrementally so runs are tailable,
stall-detectable, and resumable). Figures auto-render via
`experiments/figures.py::fig_sota_campaign` to `outputs/figs/frontier/sota_*.png`.

## Where we start (the wall)

Shipping CIRCE3 = input-scaling TCN (signal controls folded into input) + minimal
FiLM (system controls) + mixed activation + 2× oversampling + 1 Hz DC-block +
pre-emphasis-ESR loss + grad-clip 1.0. Protocol: 1 drive control, trained on the
drive *sweep*, tested at nominal drive on held-out *content*.

Baseline held-ESR (from `outputs/final_numbers.json`; **note:** that run logged
85k params, but the current `tcn.py` mixed block is 54k — the block was changed
since, so these numbers are a *reference*, not an exact current-code reproduction.
Campaign 2 re-establishes the current-code baseline):

| circuit | kind | held-ESR | under 0.005? |
|---|---|---|---|
| bjt | smooth | 0.0047 | ✅ |
| jfet | smooth | 0.0086 | ❌ |
| fullwave_rectifier | hard | 0.0090 | ❌ |
| tube_screamer | smooth | 0.0132 | ❌ |
| crossover | hard | 0.0261 | ❌ |
| hysteretic_fuzz | hard | 0.0279 | ❌ |
| asym_clipper | hard | 0.0524 | ❌ |
| hard_clipper | hard | 0.0919 | ❌ |
| **wavefolder** | hard | **0.1746** | ❌ (the wall) |

Only **1 of 9** clears the bar. The wall is the discontinuity circuits: their
sharp transfer **corners** (dead-zone edge, diode knee, clip knee, rectifier cusp,
sine folds) need infinite bandwidth a smooth Lipschitz TCN can only approximate,
and the resulting high harmonics partly alias even at 2× OS.

## Levers (all must stay uniform across circuits)

1. **Rectified input features** (`rect_thr`): a fixed bank of `relu(x−t)`,
   `relu(−x−t)`, `abs(x)` features giving exact slope discontinuities at fixed
   circuit volts (where knees physically live). Already in CIRCE3, streaming-exact,
   off by default. *Principled fix for the corner circuits.*
2. **Fourier output shaper** (`out_shaper='fourier'`): residual `y = o + Σ cₖ
   sin(k·w·o)`, the explicit multi-fold primitive the wavefolder literally is.
   Zero-init = identity. *Principled fix for wavefolder.*
3. **Capacity** (`channels`, `n_layers`): is the wall representational?
4. **Oversampling** (`oversample`): alias suppression for the sharpest circuits;
   trades RTF.
5. **Data quality**: denser drive grid, longer, cleaner targets. The user flagged
   this as historically high-impact. Deferred until architecture plateaus.
6. **Loss**: ESR + pre-emph today. A phase-aware / harmonic-weighted term may help
   the hard circuits — explored only if ESR plateaus above target.

Strategy: exhaust the cheap, RTF-light **structural priors** (1, 2) first, then
add capacity/OS, then data, then loss — changing one family at a time, measuring on
a uniform config, with smooth-circuit regression guards throughout.

## Decision log

- **2026-06-09 — Campaign 1 (`probe_levers`)** launched. Battery: 3 hardest
  (wavefolder, hard_clipper, asym_clipper) + bjt (smooth regression guard).
  Configs (uniform): `base` (current-code control) vs `rect_shaper`
  (+rect bank `(0.1,0.3,0.6,1.0)` + Fourier head) vs `rect_shaper_big` (+ch40).
  Single seed for fast triage; winner gets a multi-seed full-suite run.
  *Question: do the structural priors crack the wall, or is it capacity/OS/data?*
- **2026-06-09 — Campaign 1 RESULT: cheap levers + capacity are INERT on the wall.**
  base reproduces the baseline (wavefolder 0.185, hard_clipper 0.089, asym 0.052,
  bjt 0.0046 ✓). `rect_shaper` (rect bank `(0.1,0.3,0.6,1.0)` + Fourier head) was
  flat-to-worse everywhere (wavefolder 0.185→0.192); the Fourier head matches a
  prior *rejected* result, and the rect bank likely sits at the wrong input-volt
  scale (hard_clipper's nominal input ≈ ±0.3 V, so thresholds 0.6/1.0 never fire).
  `+ch40` was inert (wavefolder/asym flat, bjt 0.0046→0.0043) and **destabilizing**
  (hard_clipper diverged to 0.96). ⇒ The wall is **not** expressivity/capacity.
  Decision: drop rect/Fourier/capacity; pivot to the two first-principles suspects,
  **aliasing** and **loss-weighting**.
- **2026-06-09 — Campaign 1b (`probe_wall`)** launched. asym_clipper + hard_clipper
  × {base, os4, preemph2, stft01, stft03_pe2}, single seed. *Question: does more
  oversampling (alias) or a harmonic-weighted loss (preemph2 / STFT) move them?*
  (Note: numpy-twin RTF is 0.04–0.6× — a verification artifact, not deployable
  inference speed; real-time characterization deferred until an ESR winner exists.)
