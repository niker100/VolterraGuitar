# CIRCE — Conditioned Interpolatable Real-time Circuit Emulator
## Architecture specification (vetted; design only, no implementation)

> Produced by a research → candidate-design → adversarial-critique → converge →
> **red-team (with empirical RTF measurements)** → revise workflow (29 agents).
> Every claim below is pinned to a measured number or an explicit fallback. The
> headline red-team correction: the look-ahead conditioner thread was **deleted**
> (measured net-negative: RTF 0.78 threaded vs 3.75 inline).

---

## 0. Code-confirmed ground truth (the five facts the spec is built on)

1. **No per-sample recurrence exists.** `ssm.py` streaming `process_block` runs the
   *same* chunked-FFT scan as offline. The per-sample audio kernel is unwritten.
   `_MAX_CHUNK=2048` vs deployment block 128 — a real mismatch for moving-control equivalence.
2. **No conditioning data path.** `Dataset` is `(x, y, sr, name, meta)` — no `c`.
   `process_block(x)` is single-arg. `make_dataset` is single-shot.
3. **Engine forbids per-call allocation** and runs the whole model at `oversample×sr`
   (`config.oversample=2` default doubles model cost — not free).
4. **`|A|<1` bounds STATE, not OUTPUT.** The `out_proj`/`mix`/GELU readout is unbounded;
   the BJT output was measured at `y ∈ [-5.82, +3.62] V`, wildly outside `[-1,1]` and asymmetric.
5. **Streaming-equivalence tolerance is already loose** — the model test runs `atol=2e-3`
   (float32 scan agreement ~1e-3); a "1e-4 under moving controls" target was never real.

---

## 1. Core model and execution paths

**Backbone (unchanged from `ssm.py`):** `in_proj(1→d_model)` → N×[diagonal SSM, stable
pole `-softplus(a_re)+1j·a_im`, `A=exp(dt·pole)`, `+D·u`] → GELU → `mix Linear` →
residual + Norm → `out_proj(d_model→1)`.

**Size (pinned to measured RTF):** start **d_model=16, d_state=16, N=3** (measured bare
RTF 14.7); ceiling **32/16/N=5** (RTF 4.03) only if the *full-stack* GATE-1 clears ~1.5×.

**Two first-class paths:**
- **`process` (offline/training):** keep the chunked-FFT scan for parallel gradients.
- **`process_block` (streaming):** a hand-written **compiled per-sample diagonal recurrence**
  (Numba `@njit(nogil=True, fastmath=False)`, preallocated complex64), one fused loop
  `h = A·h + B·u; y = 2·Re(C·h) + D·u` fused with in_proj/GELU/mix/Norm/out_proj **and the
  inline conditioner + FiLM**. `fastmath=False` for bit-reproducibility vs the FFT path.
  The win is killing PyTorch eager dispatch, not FLOPs.

---

## 2. How the control vector enters (post-recurrence FiLM; conditioner INLINE)

Control vector **c ∈ ℝ^K**. An inline conditioner **G(c) → {(γ_i, β_i)}** per layer, applied
**after the SSM recurrence + mix, on bounded features**: `h ← γ⊙h + β` (Kallinen & Juvela 2025).
Run **inline on the audio thread, once per block, allocation-free** — at `load()` the conditioner
weights are baked to numpy/Numba arrays and evaluated in the compiled extension into preallocated
`(γ,β)` buffers. No torch in the callback.

**Hard rule:** the conditioner **never touches recurrence-internal params** (`a_re,a_im,log_dt,B,C`)
in the real-time config. This preserves: kernel `(A,B,C)` built once at load, carried-state validity
across blocks, stability over the whole control hypercube, and negligible per-sample FiLM cost.

**Bounded exception — slow drift only:** a drift sub-branch may emit a bounded offset *inside the
softplus*: `pole_re = -softplus(a_re + clamp(g_drift(c)))` + clamped `log_dt` offset; `|A|<1` stays
structural; kernel rebuilt ≤ every few seconds with an equal-power crossfade.

**Per exogenous type:**
- **(a) Continuous pots — per block, FiLM.** Taper-warped to [0,1]; **low-frequency B-spline/RBF
  basis** (not Fourier features — they hurt interpolation); one-pole ~10 Hz smoothing; (γ,β) **linearly
  ramped across the block** (no zipper noise). **Drive/gain gets a dedicated pre/post-gain pair around
  the GELU** so the knob *reshapes saturation*, not just scales features.
- **(b) Discrete switches — per block, embedding + equal-power crossfade** (few ms, both sets run in
  parallel during the window). Topology switches = selection among a small bank of conditioning sets.
  **Switch×pot interactions via a low-rank bilinear term** so unseen combinations compose.
- **(c) Slow drift — per few-seconds, bounded-pole path.** Sub-Hz smoothing → bounded pole/log_dt +
  GELU asymmetry offset. **Supply sag reclassified as circuit STATE** (modeled in the netlist:
  source-R + reservoir cap). **Temperature: 2–3 coarse swept values, interpolated, never a live knob.**

---

## 3. Real-time budget (single-threaded, inline, allocation-free) — threading inverted

**The look-ahead conditioner thread is DELETED.** Measured: kernel-alone RTF 4.68 → **0.78 (fails)**
with threaded handoff, but **3.75** with the same conditioner inline. Per-block Event/wakeup/cache
sync costs more than the ~100 µs it hides; oversubscription can starve the audio core (RTF 0.20).
Inline also gives **zero control latency**. `latency_samples = 0` for both audio and control.

| Path (block=128, 1 thread; period 2902 µs) | Cost |
|---|---|
| Bare fused recurrence 16/N=3 / 24/N=4 / 32/N=5 | RTF 14.7 / 6.95 / 4.03 |
| Inline conditioner G(c), allocation-free, per block | ~100 µs (~3.4%) |
| Per-sample FiLM + ramp (fused) | negligible |
| Switch crossfade | doubles only the cheap FiLM/readout apply (kernel reused) |

**Allocation discipline:** a `tracemalloc` delta == 0 assertion across the timed loop is a **hard CI
gate** in `measure_rtf`. **GATE-1 redefined to the FULL stack** (compiled kernel + inline conditioner
+ ADAA-on-railing-stage, measured together): must clear **RTF > 1** (≥1.5× to adopt N=5/32). The
bare-kernel 4–14× is *not* the gate.

**Anti-aliasing without blowing RTF:** ADAA on the **one explicit railing saturator only** (cheap
HARDCLIP F1/F2, no per-sample `spence`); deep-stack residual handled by an **offline teacher-student
alias-free fine-tune** (no runtime oversampling). With ADAA carrying anti-aliasing, default
**`config.oversample` → 1** (oversampling opt-in for the nonlinear stage only).

---

## 4. Scaling to complex multi-stage systems

- **Stacked stability-by-construction blocks** = long memory (slow poles) + compound nonlinearity
  without divergence as depth grows.
- **Each physical stage → one block with its own per-stage (γ,β)** (a shared affine generator can't
  represent controls entering at different depths through intervening nonlinearities).
- **Complexity budget computed first:** measure the stage's effective Volterra order + IR length from
  SPICE; map order→#GELU-layers, memory→slowest pole/#blocks; depth-vs-ESR ablation with gradient-norm
  tracking on a deliberately multi-stage SPICE target.
- **Explicit ceiling:** a feedforward LN/WH cascade **cannot represent nonlinear feedback**
  (bias-shifting, NFB loops, sag-modulated feedback). **R2 is scoped to cascaded feedforward stages**;
  a feedback circuit is added to the benchmark to *measure the gap*; fix (a bounded contractive
  feedback block) is deferred. The real bottleneck is **data, not architecture** (§7).

---

## 5. Interpolation to unseen controls (a trained objective + hard gate, never assumed)

- **Build in:** FiLM convexity + low-frequency smooth control encoding; **Lipschitz-constrained
  conditioner** (the convexity theorem bounds the FiLM-coefficient space, NOT the `c→(γ,β)` MLP — so
  grid-memorization is otherwise an admissible minimum); drive reshapes the GELU so the clip threshold
  moves monotonically.
- **Train in:** **control-space mixup / interpolation-consistency loss** (sample `c` between grid
  nodes, simulate those exact settings in ngspice as the active-learning oracle, penalize ESR there);
  curvature penalty on `||d(γ,β)/dc||`.
- **Certify:** a **frozen** held-out-control test set; lead metric **MRSTFT/THD** (ESR alone hid a
  known failure mode); report **worst-case (95th-pct, max), stratified by region** (clipping-onset,
  switch-boundary, pot-extreme) **and vs distance-to-nearest-trained-setting, per switch combination**.
  **Hard merge gate:** off-grid median ESR ≤ k×on-grid AND worst-case off-grid ESR < 1.

---

## 6. Stability across control + amplitude range

- **Linear sub-dynamics:** `|A| = ρ_max·sigmoid(·)`, ρ_max ∈ [0.999, 0.9995] (avoids `|A|→1⁻`
  ringing), applied to base params AND the drift offset.
- **Output boundedness (NOT given by `|A|<1` — confirmed real):** γ ∈ [0.5, 2]; spectral-normalize
  hyper-generated readout; **a fixed non-conditioned output saturator of last resort after `out_proj`**
  (also fixes amplitude extrapolation — hot inputs rail like the real BJT); replace the railing-stage
  GELU with an **asymmetric saturating** nonlinearity; FiLM **after Norm** so γ isn't renormalized away.
- **Stability under MOVING controls (the real R3 risk):** **dwell-time / slew-limit** bounding per-block
  (γ,β) change-rate vs the slowest pole's settling time; verification over (control vector, control
  velocity) pairs on a hot input, asserting bounded energy and zero-input residual < −130 dBFS.
- **Amplitude:** widen `drive_levels` (currently peaks 1.0 V for a ~6 V-swinging stage) past the rail;
  hot-amplitude test: above max trained level must **saturate, not grow**.

---

## 7. SPICE parametric-sweep data plan (built first; singleton + budget resolved)

- **Pipeline extensions:** templated netlists (pot R, `Vcc`, `.options temp=`, diode-select branch,
  source-R/reservoir-cap for emergent sag); **`Dataset.c` field** `(N,K)` threaded through save/load/
  benchmark; conditioning Model-contract extension (§8).
- **ngspice singleton resolved:** the sweep can't parallelize in-process (one instance/process).
  → **multiprocessing pool of separate workers** (each its own ngspice process); **prototype + measure
  spawn/DLL overhead first**; **1000+ sequential-run leak stress test** on the reset path; **fallback =
  serial + cap supported control dimensionality.**
- **Measured budget:** one 30 s BJT sim = **212 s wall** (7.1× slower than realtime).
  - ≤3 continuous axes: **5-per-axis grid** → 5³=125 sims ≈ 7 serial CPU-h (tractable).
  - 4 pots + 2 switches + 3 temp ≈ 7500 sims ≈ 442 serial CPU-h (explosive).
  - **>3 axes:** LHS/Sobol → **ensemble-disagreement active-learning** (ngspice on-demand oracle);
    **AL must beat random on the single-pot case before any >3-D claim.**
  - **Switches enumerated** (`2^#switches`), **sag = netlist state**, temperature coarse-swept.

---

## 8. Fit with the `Model` contract (additive, backward-compatible)

- **`process_block(x, c=None)`** — optional control; `None` reuses last vector (old callers unchanged).
- **`process(x, c=None)`** — `c` may be `(K,)` constant or `(N,K)` trajectory; for exact equivalence,
  `process` **re-chunks at the deployment `block_size`** and replays the identical per-block control
  schedule + (γ,β) ramp (resolves the `_MAX_CHUNK=2048` vs block=128 mismatch).
- **`fit`** — `train.c` drives the conditioner; loss adds interpolation-consistency + curvature terms.
- **`save`/`load`** — store conditioner weights + control schema (names, tapers, switch tables, basis
  knots); `load` bakes the conditioner to allocation-free arrays.
- **`check_streaming` extended** with a **moving-control variant** (hard CI gate). Engine drives
  conditioning off the model's input-block boundary, not the audio callback, so the oversampled FIFO
  can't desync controls.
- **Equivalence tolerance up front:** float64 accumulation for the `A^{t+1}h₀` state-carry term;
  **empirically calibrate the moving-control atol** so a correct impl passes AND a desynced one fails
  (~1e-3 target); static `check_streaming` keeps `atol=2e-3` parity.

---

## 9. Staged validation — ordered go/no-go gates (cheapest disqualifier first; STOP on failure)

- **GATE-0 — Core recovery.** Unconditioned SSM on the BJT must match the best unconditioned baseline
  (WH / LSTM) before any conditioning. (The SSM is currently the *weakest* contestant.) **No-go ⇒ pivot
  backbone to LSTM/GRU**, keeping the entire conditioning/stability/data layer (the transferable value).
- **GATE-1 — Full-stack real-time kernel.** Compiled recurrence + inline allocation-free conditioner +
  ADAA, measured *together*, block=128, 1 thread, oversample=1. **RTF > 1** (≥1.5× for N=5/32);
  allocation assertion == 0; static `check_streaming` (compiled vs FFT, calibrated atol).
- **GATE-2 — Data substrate + PLAIN-baseline interpolation (true first go/no-go).** Build templated
  netlists + `Dataset.c` + `process_block(x,c)` + the sweep workaround + leak test. Parameterize **one**
  real continuous axis; train the **existing concat-conditioned LSTM** on a 5-point grid, test off-grid.
  **If a plain baseline already interpolates, the FiLM/Lipschitz/mixup apparatus may be unnecessary; if
  it does NOT, it's a data/identifiability problem CIRCE can't fix either** — diagnose before investing.
- **GATE-3 — Conditioned core, single axis.** Add post-recurrence FiLM + drive GELU-gain; beat GATE-2
  off-grid (worst-case, stratified). Full-stack RTF still > 1.
- **GATE-4 — Stability under motion + amplitude.** Moving-control `check_streaming`; swept-parameter
  robustness (hot input, corners, zero-input residual < −130 dBFS); hot-amplitude saturation; set the
  trained-range→limiter handoff; verify dwell-time/slew.
- **GATE-5 — Anti-aliasing.** ADAA on railing stage + teacher-student fine-tune; THD/alias at top gain
  on bright inputs across the control range; full-stack RTF > 1 with anti-aliasing in the loop.
- **GATE-6 — Multi-stage + multi-control.** 2–3-stage cascade + a feedback circuit; depth-vs-ESR
  ablation; **AL-beats-random validated on the single-pot case before scaling**; held-out
  switch-combination gate; report the RTF-vs-ESR Pareto, not a single point.

---

## Executive summary

CIRCE is a conditioned, stability-by-construction diagonal-SSM emulator whose streaming path is a
**single-threaded, inline, allocation-free compiled per-sample recurrence** (the FFT scan is kept for
offline/training only), conditioned by **post-recurrence time-varying FiLM** computed inline once per
block from a Lipschitz-constrained, low-frequency-encoded conditioner that never touches the recurrence
poles (a bounded inside-softplus offset handles slow drift only), with a **fixed output saturator of
last resort**, ADAA-on-the-railing-stage plus offline alias-free fine-tuning, and a **per-circuit SPICE
parametric-sweep data substrate that does not yet exist and is built first** behind a multiprocessing
workaround for the ngspice singleton. The central red-team correction was **deleting the look-ahead
conditioner thread** (measured net-negative) and **redefining the real-time gate to a full-stack
measurement**. The program is governed by **seven ordered go/no-go gates** that run the cheapest
disqualifier first, so it is killed early if its premises fail.

## Residual risks accepted

1. **Interpolation & moving-control stability are plausible but unproven** until GATE-2/3/4 on a data
   substrate that doesn't exist yet (architecture makes them likely; proves none).
2. **High-dimensional control budget is explosive** and rests on active-learning beating random
   (tractable ≤3 axes ~7 CPU-h; 4-pot/2-switch/3-temp ~442 CPU-h). AL unvalidated until GATE-6.
3. **Nonlinear feedback is out of scope** (feedforward cascade provably can't represent it); benchmark
   measures the gap, fix deferred.
4. **Moving-control equivalence tolerance may end up loose (~1e-3)** — discriminating power must be
   empirically defended.
5. **ngspice multiprocessing overhead/leaks unmeasured** — prototype-then-decide, serial fallback caps
   ambition.
6. **Trained-range→hard-limiter handoff trades ESR for stability** — set at GATE-4, not closed-form.
7. **Backbone-pivot contingency (GATE-0):** if the SSM can't recover the unconditioned baseline, the
   audio core swaps to LSTM/GRU (conditioning/data layer transfers; the pole-placement proofs do not).

## Recommended first milestone

**BJT-CE overdrive with ONE continuous drive knob, validated through GATE-2** — in order:
1. **GATE-0 (no conditioning):** unconditioned SSM (16/16/N=3) matches the WH/LSTM baseline ESR on the
   existing BJT dataset. If not, pivot the backbone *now*.
2. **Minimum data substrate:** add `Dataset.c`; template the BJT netlist for one axis (`nominal_drive_v`
   — free in data, no SPICE parallelism needed); 5-point grid + a few off-grid held-out settings (~10
   sims, <1 h serial).
3. **GATE-1 in parallel:** compiled per-sample recurrence with a trivial inline conditioner + the
   `tracemalloc==0` assertion; confirm full-stack RTF > 1 at block=128, 1 thread.
4. **GATE-2 decision:** train the existing concat-conditioned LSTM on the grid, test off-grid — the
   pivotal cheap experiment that decides whether CIRCE's machinery is needed at all.

This touches every load-bearing risk (backbone recovery, full-stack RTF + allocation discipline, the
conditioning data path, plain-baseline interpolation) on the cheapest circuit/axis and needs **no
ngspice parallelism**.

---

## Appendix — red-team must-fix items (all discharged in the spec above)

1. Invert the threading decision (look-ahead thread deleted; inline conditioner) — §3.
2. Make the inline conditioner allocation-free + tracemalloc assertion — §2/§3.
3. Re-run GATE-1 with the full stack, not the bare kernel — §3/§9.
4. Resolve the ngspice singleton for the sweep (multiprocessing or serial cap) + leak test — §7.
5. Build & pass GATE-2 (plain-baseline interpolation) before any CIRCE-specific machinery — §9.
6. Set a realistic moving-control `check_streaming` tolerance up front — §8.

(Generated 2026-06-06 by the `conditioned-sota-design` workflow: 6 research angles, 4 candidates,
16 critiques, converge → red-team → revise.)
