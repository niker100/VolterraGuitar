> **FROZEN 2026-06-10 - historical record of the neural benchmarking era.**
> The project pivoted to the physical DK-method emulator; consolidated verdicts
> live in `outputs/knowledge/findings.json`.

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

## Current state (living section — updated 2026-06-10)

**Model (the unified config, best uniform yet — 4/9 under 0.005):** CIRCE3 =
dilated causal TCN, ch24 / nb2 / L10 (RF ≈ 4093 samples), mixed activations
(tanh/gelu/relu/abs/snake), input-scaling (signal controls fold into the input as
gain) + zero-init FiLM for system controls, `n_state=4` learnable one-pole IIR
memory channels (zero-init input weights; log-depth parallel scan in torch,
carried `lfilter` state in the numpy twin), internal 2× oversampling,
`dcblock_fc=0` (DC is real circuit output — the 1 Hz blocker was a train/eval
artifact), clamp saturator, ~60k params. Loss: ESR + pre-emphasis ESR, grad-clip
1.0. Streaming twin is bit-exact (`process == process_block`, tested).

**Training:** AdamW, seq 4096, warmup 2048, 150 ep. `TrainConfig.varpro` solves
the readout in closed form per step (fp32 ridge, Golub-Pereyra) — 2.5× faster /
5× sharper on smooth circuits, but NOT uniform-safe (breaks discontinuity
circuits). `TrainConfig.amp` = bf16 autocast (fp32 weights, IIR scan fp32).

**Repo organization:** active experiment code = `experiments/sota/` (harness +
campaign drivers) + `experiments/common.py` + `experiments/figures.py` +
`experiments/final_numbers.py` + `experiments/check_runs.py` (stall detector).
All completed pre-SOTA phases live in `experiments/archive/` (see its README
for the script → verdict map). Decisions live here; numbers in `outputs/sota/`;
figures in `outputs/figs/frontier/`.

**Leaderboard (standard training, unified config, seed 0):** asym 0.0008 ✅,
ts 0.0026 ✅, bjt 0.0044 ✅, jfet 0.0045 ✅ — then fullwave 0.0057, hysteretic
0.0139, crossover 0.0154, hard_clipper 0.0578, wavefolder 0.245.

**Prioritized tackle points:**
1. **DONE — speed adoption (`speed_ab`, 4 rounds + `hc_diag`):** bf16 AMP is
   deterministically unsafe (collapses hard_clipper/bjt); fp32 big-batch
   overshoots into a degenerate basin without warmup. Adopted: **5-epoch LR
   warmup everywhere** (free at b12), **SCREEN = b96/lr9/warm5 fp32** for A/B
   probes with **retry-then-fallback** (`harness.fit_score`: held>0.5 → reseeded
   retry → safe-b12 fallback), finals at b12/lr3/warm5. b384 exceeds the card
   (WDDM swaps instead of OOM-ing — never again). Next: `fast_stack` (VarPro ×
   b96 × warmup on smooth circuits, running).
2. **DONE — data volume (`data_v2_ab`):** 3× data wins big where the circuit is
   data-limited (ts −47% → 0.0027, crossover −32%, jfet −16%, hard_clipper −9%)
   and is flat on fullwave/hysteretic. **v2 sweeps adopted into the protocol**
   (tests unchanged); `unified_v2` multi-seed finals queued.
3. **fullwave 0.0057 (near):** NOT data-limited (v2 flat) — multi-seed first,
   then depth/window.
4. **hysteretic 0.0139 (memory):** IIR lever validated (−29% at prototype);
   probe n_state=8, longer τ-init range, seq_len 8192 (τ up to 440 ms vs 93 ms
   training window).
5. **hard_clipper 0.0578 (knee):** depth halved it once; next depth step needs
   seq_len 8192 (RF would exceed the 4096 window) — combine with data-v2.
6. **crossover 0.0154:** depth *hurts* it, IIR helps; data-v2 + IIR-width probe.
7. **wavefolder 0.245 (spectral-bias wall):** grey-box learnable PWL/ADAA
   waveshaper (reuse archived `PWLBank`) is the one untried structural lever;
   if it fails uniform constraints, report the complementary metric honestly
   (band-ESR / log-spectral) per the goal's "unless ESR is not the proper
   metric" clause.
8. **RTF re-measure, serially:** `unified_varpro` logged rtf ≈ 2.1 for several
   varpro cells vs 0.08–0.17 for identical-architecture standard cells —
   measurement contention (RTF sampled while the GPU/CPU was busy), as the user
   predicted. Final RTF table must be measured with nothing else running.

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
- **2026-06-09 — Campaign 2 RESULT: the DC fix works; `dcblock_fc=0` is the new
  default.** dcblock_off vs on (single seed, current code). **asym_clipper 0.0523 →
  0.0007** (75× — cracked, the diagnosis confirmed). **wavefolder 0.200 → 0.126**
  (DC artifact removed; 0.126 genuine fold residual remains). fullwave 0.0089 →
  0.0074. Symmetric circuits within seed noise (jfet 0.0097 flat, ts 0.0137→0.0145,
  crossover 0.0219→0.0233, hard_clipper 0.0911→0.0940, hysteretic 0.0273→0.0280,
  bjt 0.0045↔0.0046). Adopt dcblock_off (faithful DC reproduction; cracks asym, helps
  the DC-bearing circuits, neutral elsewhere). Fig: `sota_dc_rebaseline.png`.

### Post-DC-fix leaderboard (current-code, dcblock_off, 1 seed) — the remaining work

| circuit | held-ESR | under 0.005? | remaining lever |
|---|---|---|---|
| asym_clipper | 0.0007 | ✅ | done |
| bjt | 0.0046 | ✅ | done |
| fullwave_rectifier | 0.0074 | ❌ (near) | depth / data / seed |
| jfet | 0.0097 | ❌ (near) | depth / data / seed |
| tube_screamer | 0.0145 | ❌ | depth / data |
| crossover | 0.0233 | ❌ | learnable corners (rank-3) / depth |
| hysteretic_fuzz | 0.0280 | ❌ | IIR memory (rank-2, running) |
| hard_clipper | 0.0940 | ❌ | learnable corners + depth (rank-3/4) |
| wavefolder | 0.1259 | ❌ | spectral-bias wall — grey-box (rank-5) or report complementary metric |

2/9 under 0.005. Next: rank-2 IIR (running) for hysteretic; rank-3 learnable corners
for hard_clipper/crossover; a depth knob for the near-misses; wavefolder last.

- **2026-06-09 — Campaign 3 RESULT: IIR memory lever WORKS (directionally).** OS1
  prototype A/B (state on/off): hysteretic_fuzz 0.0495→0.0353 (**−29%**), crossover
  0.0716→0.0587 (**−18%**), bjt 0.0296→0.0291 (−2%, smooth guard neutral). The
  shallow-OS1 prototype's absolute ESR is inflated vs production, so these are
  directional "integrate it" signals — true magnitudes come post-integration at
  OS2/nb2. −29% alone won't crack hysteretic; levers must stack. Fig: `lever_iir_probe`.
- **2026-06-09 — Campaign 4 (`corner_probe`)** launched (GPU): learnable-threshold
  corners on/off on hard_clipper + crossover + bjt. Gates the circe3.py integration.
- **2026-06-09 — Data-quality track (`regen_data`)** launched (CPU/ngspice, parallel —
  no GPU contention): regenerate the 6 unsolved/near-miss training sweeps at
  seg_dur_s=8 (~3× data, the audited 41% volume deficit) to `*_v2.npz`, **test sets
  unchanged** so held-ESR stays comparable. A follow-up campaign trains on _v2.
- **Integration plan (next):** once corners are confirmed, add to `circe3.py` as
  default-off uniform flags — `n_state` (K learnable one-pole channels, numpy
  streaming twin = carried one-pole recurrence, bit-exact by construction) and
  learnable `rect_thr` (replacing the dead fixed bank) — with a streaming-exactness
  test + save/load round-trip. Then a production-config (OS2/nb2/dcblock_off)
  combined campaign: baseline vs +IIR vs +corners vs +both vs +depth, multi-seed,
  with the smooth circuits as regression guards.
- **2026-06-09 — Campaign 4 RESULT: learnable corners are INERT (drop them).**
  hard_clipper 0.1092→0.1079 (−1%), crossover 0.0703→0.0676 (−4%), bjt −1%. The
  thresholds barely moved from init (0.3/0.7/1.5 stuck), i.e. SGD found no gradient
  use for them — same dead end as the fixed bank. ⇒ do NOT integrate corners; the
  hard_clipper knee residual is not fixable by input corner primitives. Fig:
  `lever_corner_probe`. So integration = **IIR only**.
- **2026-06-09 — Campaign 5 (`depth_sweep`)** launched (GPU): the only un-confirmed
  hard_clipper lead is depth. nb2/L9 (RF 2045, control) vs nb1/L11 (RF 4095) vs
  nb2/L10 (RF 4093) — both deeper configs keep RF inside the training window (no
  starvation, seq_len fixed → isolates depth). dcblock_off, OS2, on hard_clipper +
  crossover + fullwave + jfet + bjt. *Does ~2× receptive field move the knee/near-miss
  circuits?*
- **2026-06-09 — Campaign 5 RESULT: depth is a strong but MIXED lever.** nb2/L10
  (~2× RF) **cracks jfet** 0.0094→0.0020 ✅, ~halves hard_clipper 0.0933→0.0558,
  helps fullwave 0.0070→0.0057, bjt stays 0.0044 ✅ — but **hurts crossover**
  0.0225→0.0299 (the dead-zone prefers shallow; its lever is IIR, not depth). nb1/L11
  is cheaper/faster (33k, RTF 4.2) but pushes bjt to 0.0053 (over). ⇒ **nb2/L10 is the
  depth choice.** Fig: `sota_depth_sweep`. Best-per-circuit now: asym 0.0007, jfet
  0.0020, bjt 0.0044 (**3/9 clear**), fullwave 0.0057 (near).
- **2026-06-09 — Campaign 7 (`unified`)** launched (GPU): the convergence config
  **dcblock_off + nb2/L10 (depth) + n_state=4 (IIR)**, full 9 circuits, seeds (0,7).
  *How many cross 0.005 with the validated levers stacked?* Expected resisters:
  hard_clipper, wavefolder (→ grey-box / complementary metric). crossover is the
  open question (depth hurts, IIR helps — net TBD).
- **2026-06-09 — IIR regression found + fixed (zero-init), unified re-run.** The
  unified run showed jfet 0.0020 (depth-only) → **0.0055** (both seeds) once IIR was
  added: the IIR state-channel input weights were *random*-init, perturbing the
  memoryless circuits. Fix: **zero-init the state-channel input weights** so the net
  starts identical to no-IIR and learns memory only where it helps (mirrors the
  zero-init FiLM/shaper; 33/33 tests still pass). Killed the tainted unified run; will
  re-run clean. *Lesson: new input channels must be zero-init to stay neutral on
  circuits that don't need them.*
- **2026-06-09 — Training-speed track (user idea).** GPU underutilized at batch=12
  (65% util, 18% mem, 45% power). Added `TrainConfig.amp` (bf16 autocast, fp32 weights
  + numpy twin untouched, IIR scan forced fp32) and `speed_ab.py` (batch × LR × AMP,
  measuring wall-clock AND held-ESR; bigger batch = fewer steps so LR co-scaled).

### Speed + closed-form track (user ideas, 2026-06-09)

- **zero-init fix CONFIRMED:** jfet `b12` unified = 0.0046 (recovered from the tainted
  0.0055; depth-only was 0.0020 — IIR still costs a little but jfet clears 0.005).
- **bigger batch wins:** jfet `b48_lr6` = 0.0041 in 108 s vs 0.0046/150 s at b12 (~1.4×
  faster, ESR holds/improves); GPU util jumped when batch grew. Sweep extended to
  192/384 (OOM-safe) to find the max that fills the 24 GB card → harness default.
- **ELM (random features + lstsq readout) — VERDICT: fast tool, not SOTA.** ~0.05–0.23
  ESR in ~5 s (no backprop), but improves only slowly with width and **plateaus** far
  from 0.005 (jfet 4608→9216 feats: 0.0367→0.0350, 17× off trained; tube_screamer 0.058,
  4×; hard_clipper 0.21). Structural feature-quality limit, not feature-count. Leak fixed
  (chunked HtH accumulation, bounded 3.5 GB). `elm_probe.py`.
- **Alternating VarPro (`varpro_probe.py`):** trainable trunk + wide feature layer +
  lstsq readout. First (alternating, freeze-W-10-epochs) version DIVERGED (bjt 0.03→0.42:
  trunk overfits a stale W, feature collapse). Fixed → **differentiable per-batch solve**
  (Golub-Pereyra, fp64 + scale-relative ridge).

### VarPro / closed-form verdict (user-directed deep dive, 2026-06-09)

Tested the full closed-form family. **Headline: closed-form is NOT an accuracy
gamechanger here — the trained nonlinear features are the lever, and the readout is a
trivial linear layer SGD already optimizes.** Evidence (held-ESR; trained-CIRCE3 refs in
parens):

| method | bjt | jfet | ts | crossover | hard_clipper | notes |
|---|---|---|---|---|---|---|
| ELM (random feats + lstsq) | 0.10 | 0.035 | 0.058 | 0.13 | 0.21 | plateaus with width; ~5 s |
| Hammerstein (structured + lstsq) | 0.18 | 0.083 | 0.090 | 0.30 | 0.30 | *worse* than ELM (shallow) |
| VarPro (trained feats + lstsq) | 0.030 | 0.011 | 0.016 | 0.061 | 0.127 | ≈ joint, 6× slower |
| joint (trained feats + trained readout) | 0.030 | 0.010 | 0.017 | 0.063 | 0.126 | the baseline (weak prototype arch) |
| *(trained CIRCE3 ref)* | *0.0046* | *0.002* | *0.0145* | *0.0225* | *0.0558* | strong arch |

Reads: (1) **VarPro ≈ joint** — the closed-form readout gives no accuracy gain (the
readout is trivial to train). (2) **VarPro ≫ ELM** (bjt 0.03 vs 0.10, ~4×) — *trained*
features beat random ones; the user's "alternate instead of dropping backprop improves
massively" holds **vs ELM**, not vs joint. (3) Pure closed-form (ELM/Hammerstein) plateaus
~0.04–0.3, far from 0.005 — no trained features. (4) The differentiable per-batch fp64
solve is ~6× slower/epoch, so VarPro is slower as implemented.
**Remaining open angle (`varpro_conv.py`, running):** does VarPro converge in FEWER
epochs (the only speed mechanism)? **The genuine computational win found is BIGGER BATCH**
(validated ~1.4×, fills the 24 GB card) — that's the "computationally better" lever to
adopt for the trained path, which remains the route to 0.005.

### VarPro convergence WIN + production integration (2026-06-09)

- **Convergence (`varpro_conv`, fp32):** VarPro reaches the same plateau in **~3× fewer
  epochs** (bjt: plateau 0.0293 by ~ep60 vs joint ep150; jfet similar). The closed-form
  readout trains the trunk against an always-optimal head from step 1. Fig: `varpro_conv`.
- **The fp64 trap:** the 4090 cripples fp64 to ~1/64 of fp32, so the original fp64 solve
  was ~6×/epoch and cancelled the win. **fp32 solve → 18.5 ms/batch vs joint's 16 ms** —
  per-epoch cost ≈ joint, so the 3× epoch win becomes a **~2–3× wall-clock training
  speedup at equal accuracy** (a SPEED win — VarPro=joint on final ESR, not better).
- **INTEGRATED into the production model** (`circe3.py`, `TrainConfig.varpro`): solves the
  final linear readout `out[3]` in closed form each step (fp32 differentiable ridge
  lstsq), trunk backprops; `_varpro_set_readout` writes the global solution into the conv
  at the end. Works WITH the IIR memory channels (`n_state`) + OS2 — the memory-heavy
  path. **The numpy streaming twin is unchanged (readout stays a plain linear conv) —
  bit-exact verified 7.7e-7; +1 test.** `varpro_circe3.py` validates the speedup on the
  production model (bjt/jfet/hysteretic_fuzz; standard-150 vs varpro-60 vs varpro-150).
- **Other closed-form paths (verdict):** ELM/Hammerstein are no-training but plateau
  ~0.04–0.3 (no trained features) — fast screening/init tools, not SOTA. The closed-form
  *readout* (VarPro) is the keeper: it accelerates training of the real model.
- **Production validation (`varpro_circe3`) — VarPro WINS on smooth, neutral on memory:**
  on the production model (nb2/L10 + n_state=4 + OS2 + dcblock_off): bjt varpro-60 0.0050
  ≈ standard-150 0.0044 at **2.5× less wall-clock**; **jfet varpro-150 0.0010 vs standard
  0.0048 — 5× MORE accurate** (revises the "VarPro = joint" call: on the strong arch the
  always-optimal readout also reaches a better minimum). hysteretic_fuzz (memory)
  neutral-to-slower (0.0146 vs 0.0134) — VarPro doesn't fix memory (that's the IIR
  channels). So VarPro = faster + sharper on smooth/near-static circuits. Threaded into
  the harness (`varpro` config key). `unified_varpro` (standard vs varpro, full suite)
  running to see if it pulls more circuits under 0.005.

### Best uniform config so far + VarPro-uniform verdict (`unified_varpro`, 2026-06-09)

**The standard unified config — dcblock_off + nb2/L10 (depth) + n_state=4 (IIR) + OS2 — is
the best uniform config yet: 4/9 cleanly under 0.005.** Fig: `sota_unified_varpro`.

| circuit | standard | varpro |
|---|---|---|
| asym_clipper | **0.0008** OK | 0.0043 OK |
| tube_screamer | **0.0026** OK | 0.0013 OK (depth L10 cracked it; was 0.0145) |
| bjt | **0.0044** OK | 0.0042 OK |
| jfet | **0.0045** OK | 0.0010 OK |
| fullwave | 0.0057 (near) | 0.0171 |
| hysteretic | 0.0139 | 0.0172 |
| crossover | 0.0154 | 0.0976 |
| hard_clipper | 0.0578 | 0.0496 |
| wavefolder | 0.245 | 0.84 (broke) |

**VarPro is NOT uniform-safe:** it sharpens smooth circuits (jfet 0.0010, ts 0.0013) but
BREAKS the discontinuity circuits — the closed-form L2 readout finds a worse minimum where
the transfer curve is sharp. So uniform training = **standard**; VarPro is a tool for the
smooth circuits + fast iteration, not the uniform method.

**Remaining above 0.005 (standard):** fullwave 0.0057 (near), hysteretic 0.0139, crossover
0.0154, hard_clipper 0.0578, wavefolder 0.245. Next levers: data-v2 (3x data, running on
the unified config), multi-seed, capacity/depth for hard_clipper, grey-box / complementary
metric for wavefolder.

### Stochastic NaN-collapse found + fixed (2026-06-10)

The speed A/B (batch x LR x bf16, jfet/hard_clipper/bjt) collapsed **5 of 12 cells to
held-ESR ~1.0, scattered across ALL arm types** — including the b12 fp32 control on
hard_clipper and b96 fp32 on the rock-solid bjt — while `unified_varpro` had 0/18 on
byte-identical training code. Repeat diagnostic (`hc_diag.py`, fig `hc_diag`): the
identical hard_clipper b12/seed-0 cell scored 0.0524 / 0.0497 on two reruns (collapsed
run scored 1.0000), and the two healthy same-seed runs took *different trajectories*
(one still at ESR 1.0 at epoch 5, one at 0.33). **Training is not bit-reproducible
across runs (GPU-nondeterministic reductions), and a spiked batch -> inf/NaN grads ->
`opt.step` poisons the weights permanently**; with NaN val forever, fit's best-val
restore returns the *untrained init* — that is what held-ESR exactly 1.0 means.

**Fix (trainer hardening, uniform):** `fit()` now computes the grad norm every step and
**skips the optimizer step when loss or grad-norm is non-finite**
(`FitReport.info.skipped_steps` counts them). One bad batch costs one step, not the run.

**Measurement hygiene note:** cell wall-clocks vary up to 2x between processes (the same
hard_clipper b12 cell: 184 s in one run, 98 s in another) — desktop/GPU contention. Speed
ratios are only meaningful *within* one process; absolute secs across runs are not
comparable. (The user predicted contention; RTF numbers in `unified_varpro` show the same
artifact — varpro cells logged rtf ~2.1 vs 0.08 for identical architectures.)

### Speed adoption (speed_ab rounds 3-4, 2026-06-10) — warmup + SCREEN + retry/fallback

The guard alone did NOT fix the collapses: the guarded rerun reproduced the bf16
failures at the EXACT same ESR (hard_clipper b192_amp9 1.3108 twice, bjt b96_amp 0.9932
twice — deterministic, no NaN involved), and bjt b96_lr9 fp32 collapsed again (0.9998,
~2/3 of samples). Revised mechanism: **early epochs at full scaled LR overshoot into a
degenerate predict-mean basin** (held-ESR ~1.0 = best-val never left silence). bf16
lands there deterministically on some circuit x batch combos (8-bit mantissa rounds
away the early gradient signal); fp32 lands there stochastically (GPU nondeterminism
decides). Round 4 = the standard medicine, **linear LR warmup** (`TrainConfig.lr_warmup`,
ramp x the exact cosine schedule as before):

| arm | jfet | hard_clipper | bjt | verdict |
|---|---|---|---|---|
| b12 (control) | 0.0050 | 0.0524 | 0.0044 | reference |
| b12_w5 | 0.0047 | 0.0512 | 0.0044 | warmup is FREE at b12 → default everywhere |
| b96_lr9 | 0.0038 | 0.0661 | collapse 2/3 | fast, unstable |
| b96_w5 | 0.0046 | collapse | 0.0054 | warmup fixes bjt, NOT hard_clipper |
| b96_amp / b192_amp9 | ok | collapse | mixed | **bf16 training disqualified** |

**Adopted:** (1) `lr_warmup=5` in every harness fit (free insurance, also covers the
b12 stochastic collapse observed once on hard_clipper); (2) **SCREEN = batch 96 / lr
9e-3 / fp32** (~1.4-2.7x wall-clock) for relative A/B probes; (3) `harness.fit_score`
with **collapse retry-then-fallback**: held>0.5 → retrain with shifted seed → last
resort the safe b12 point, so no campaign cell can report a collapse artifact. Finals/
leaderboard runs stay at b12/lr3/warm5. bf16's honest verdict for the user's 16-bit
idea: training-unsafe for this loss/arch; the RT inference path is already fp32 numpy.
Figs: `speed_ab`, `hc_diag`.

### Quick-learner verdict (`fast_stack`, 2026-06-10) — vp60_b12 is the iteration config

Combined-stack benchmark on the smooth circuits (where VarPro is safe), vs the
standard-150ep/b12 leaderboard reference. Fig: `fast_stack`.

| arm | jfet | bjt | tube_screamer | wall-clock |
|---|---|---|---|---|
| ref std150_b12 | 0.0045 | 0.0044 | 0.0026 | 1x |
| **vp60_b12** | **0.0021** | 0.0049 | **0.0020** | **~2.6-3.0x less** |
| vp60_b96_w5 | 0.0066 | 0.0094 | 0.0082 | ~6-7x less |
| vp150_b96_w5 | 0.0022 | 0.0062 | 0.0033 | ~3x less |

**VarPro-60ep at the safe b12 point = equal-or-SHARPER ESR at ~3x less wall-clock**
(jfet 2.1x sharper than the reference). **VarPro x big-batch anti-compounds** at every
epoch count — the trunk apparently needs small-batch gradient noise when the readout
is solved optimally each step (the b96 stack is uniformly worse than vp60_b12 despite
2x more wall-clock at 150 ep). So the two speed tools serve different jobs:
**vp60_b12** for smooth-circuit iteration (accuracy-critical), **SCREEN b96/w5 +
retry** for standard-training A/B probes (speed-critical, relative comparisons).

### Data-volume verdict + v2 adoption (`data_v2_ab`, 2026-06-10)

Original vs regenerated 3x sweeps (seg_dur 8 s), unchanged tests, SCREEN training,
within-config relative A/B. Fig: `lever_data_v2_ab` (the killed b12-config run's
jfet pair is archived as `lever_data_v2_ab_b12`: 0.0053 → 0.0034, −35% — replicated
here directionally at −16%).

| circuit | orig | v2 | delta | read |
|---|---|---|---|---|
| tube_screamer | 0.0051 | **0.0027** | **−47%** | data-limited, cracked at SCREEN |
| crossover | 0.0413 | 0.0282 | −32% | data helps; (also: crossover degrades badly at b96 — SCREEN numbers ≠ b12 leaderboard) |
| jfet | 0.0046 | 0.0038 | −16% | replicates the b12 finding |
| hard_clipper | 0.0590 | 0.0535 | −9% | small win |
| fullwave_rectifier | 0.0056 | 0.0057 | +1% | **not data-limited** — its 0.0056 plateau needs another lever |
| hysteretic_fuzz | 0.0186 | 0.0194 | +4% | **not data-limited** — memory-limited as diagnosed (`memory_probe` running) |

**Adopted:** the 6 v2 sweeps replace the originals in `harness.CIRCUITS` (protocol
change; tests unchanged so held-ESR stays comparable; pre-v2 numbers historical).
Note: the in-flight `memory_probe` imported the old map — its relative A/B (orig
sweeps) stays valid; winners get confirmed on v2 in finals. `unified_v2` (multi-seed
finals on v2) queued behind it.
