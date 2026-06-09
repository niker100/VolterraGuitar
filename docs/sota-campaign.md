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
- **2026-06-09 — Campaign 1b RESULT: oversampling + loss-weighting are DEAD on the
  wall.** os4 was *worse* (asym 0.052→0.17, hard_clipper 0.089→0.30); preemph_order=2
  and multi-STFT were inert (asym flat ~0.053, hard_clipper flat ~0.094). Confirms
  the diagnosis below.
- **2026-06-09 — Multi-agent wall diagnosis (workflow `sota-wall-diagnosis`)** +
  independent verification. See the Diagnosis section below. Headline: the biggest
  chunk of the "wall" is a **DC-blocker train/eval artifact**, not a model limit.
- **2026-06-09 — Campaign 2 (`dc_rebaseline`)** launched: all 9 circuits, current
  config, `dcblock_fc` OFF (0.0, the fix) vs ON (1.0, control), single seed.
  *Re-establishes the true current-code baseline AND tests the rank-1 DC fix.*

## Diagnosis — three separable failure modes (not one wall)

A 5-investigator workflow + my own verification established that the "wall" is three
distinct things, and the dominant one is a **measurement artifact**:

**(1) DC-blocker train/eval mismatch [artifact — fix it].** `circe3.fit` computes its
loss on the raw network output (no blocker, `circe3.py:508-516`), but `process_block`
applies a 1 Hz DC-blocker at inference (`:724-729`, default `dcblock_fc=1.0`). For
circuits whose target carries real, level-dependent output DC, a *perfect* model is
floored. **Verified perfect-model ESR through the blocker** (my own measurement,
`M.esr(y, lfilter(y))`, warmup 2048):

| circuit | DC-block floor | baseline ESR | artifact share |
|---|---|---|---|
| asym_clipper | 0.0518 | 0.0524 | ~99% |
| wavefolder | 0.1153 | 0.1746 | ~66% |
| tube_screamer | 0.0022 | 0.0132 | ~17% |
| jfet / fullwave | 0.0017 | 0.0086 / 0.0090 | ~20% |
| hard_clipper | 0.0055 | 0.0919 | ~6% |
| crossover | 0.0026 | 0.0261 | ~10% |
| hysteretic_fuzz | 0.0003 | 0.0279 | ~1% |
| bjt | 0.0000 | 0.0047 | ~0% |

The DC is genuine signal (asym y_mean +0.050, tracks drive — the SPICE netlist has no
output coupling cap), so the *faithful* fix is to reproduce it: train + eval raw
(`dcblock_fc=0`). Uniform, removes an inference stage. Cross-harness proof: a plain
TCN scored raw hits asym **0.0064** on the identical data (`radical_generalize_ab.json`).

**(2) Structural memory deficit [hysteretic_fuzz — needs IIR state].** TCN receptive
field is 21 ms (n_layers=9, OS2) but hysteretic_fuzz's bias-recovery τ = 22–440 ms.
The FIR stack physically can't see the state defining the hysteresis loop. No
capacity/loss/OS lever adds a missing pole — only added IIR/leaky-integrator state
(rank-2 lever) can. (OS2 *halves* RF-in-ms vs OS1, so OS actively hurts memory.)

**(3) Genuine spectral-bias floor [wavefolder, partly hard_clipper — hard limit].**
After (1)+(2) are removed, the wavefolder residual (~0.06–0.15) is real and is NOT
aliasing: targets are soxr-VHQ anti-alias-decimated (zero target aliasing) and the
pointwise-NL-then-decimate alias floor is ~1e-6 at OS2 — 5 orders below the residual.
The wavefolder map `0.45 sin(3.4x)+0.18 sin(9x)` is non-monotonic/multi-fold; a
finite-Lipschitz smooth-activation TCN rounds the fold turnarounds. **Honest verdict:
uniform <0.005 on wavefolder is likely NOT achievable** without a grey-box learnable
static waveshaper (rank-5, speculative, in tension with no-tailoring). hard_clipper's
near-vertical knee (tanh(60x)) may yield to learnable-threshold corners (rank-3) +
depth (rank-4) — one un-confirmed lead (deep ch24/L11) already dropped it 0.09→0.04.

**Dead ends (catalogued — do not retry):** OS>2 on hard circuits (measured worse);
fixed-threshold rect bank + Fourier/sine output head (inert); raw width capacity
(inert + destabilizing); more high-band/STFT loss weight (preemph already 89–97%
HF-weighted; STFT wrecks the transfer curve); MoE/CIRCE-X (regresses smooth, gate
collapses); grad-clip retuning (already optimal); attributing the wall to aliasing.

**Plan after the DC re-baseline:** (rank-2) add a gated leaky-integrator IIR state
channel for hysteretic + crossover; (rank-3) learnable-threshold rectified corners for
hard_clipper/crossover; (rank-4) confirm the depth lead with an RTF gate; (metric) add
a band-decomposed ESR column to every hard eval, and for wavefolder report a
complementary log-spectral/harmonic-match metric rather than gating on 0.005.

- **2026-06-09 — rank-2 IIR prototype built + validated (`iir_probe.py`), queued
  behind the re-baseline.** Learnable one-pole state channels `s_k[n]=a_k s_k[n-1]+
  (1-a_k)x[n]` (τ init 5–500 ms) computed via a stable log-depth parallel scan
  (no torchaudio; self-test matches a sequential reference to 2.4e-7). Costs +100
  params, input-scaling-equivariant, streamable. A/B (state on/off, OS1) on
  hysteretic_fuzz + crossover + bjt guard — launches when the GPU frees up.
