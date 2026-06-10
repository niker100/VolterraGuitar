> **FROZEN 2026-06-10 - historical record of the neural benchmarking era.**
> The project pivoted to the physical DK-method emulator; consolidated verdicts
> live in `outputs/knowledge/findings.json`.

# CIRCE3 — model card

**CIRCE3** is the project's **optimal, minimal, domain-general** emulator of
nonlinear systems with exogenous inputs. It is the empirical conclusion of the
CIRCE → CIRCE2 → CIRCE3 arc (see `docs/optimal-architecture.md` for the full
research record, the theory derivation, and the verdicts that overturned the
heavier designs).

## Headline result — ESR ≈ 0.001–0.005 on every tested control type, real-time on CPU

On **realistic (band-limited) signals** — the deployment-meaningful metric — the
final architecture (**a heterogeneous mixed-activation TCN + input-scaling for
signal controls + minimal FiLM for system controls + a dense control grid**,
trained with **pre-emphasis ESR** under **gradient clipping**, and run with **2×
internal oversampling**) clears the 0.005 target on all three *kinds* of exogenous
control, held-out (interpolated) settings, real-time:

| control kind | testbed | conditioner | held-out ESR | real-time |
|---|---|---|---|---|
| **signal** (drive) | BJT (audio) | input-scaling | **0.0012 (guitar-DI), 0.005 (fp)** | ✅ RTF 2.4× (OS2) |
| **static-map** (β) | Duffing (non-audio) | FiLM + dense grid | **0.0023** (worst 0.0041) | ✅ |
| **dynamics** (tone RC) | JFET (audio) | FiLM + dense grid | **0.0029** (worst 0.0033) | ✅ |

The two decisions that nail the **spectral properties** — a phase-aware
pre-emphasis loss for the harmonic formants + transfer curve, and 2× internal
oversampling to remove the gates' self-aliasing — are detailed in
*Spectral fidelity* and *Internal oversampling* below; they were each a larger
lever than network size. Figures (from `vguitar benchmark`):
`outputs/figs/compare_bjt_{spectrum,transfer,harmonics,spectrogram}.png` +
`circe3_oversample_aliasing.png`.

**Two findings that decided the design** (full record in `docs/optimal-architecture.md`):
1. **The control grid is the first-order lever, not the conditioner.** Densifying
   the system-control grid dropped held-out ESR ~2.7× (Duffing β coarse 0.0063 →
   dense 0.0023), exactly as the interpolation bound `ε ~ (grid spacing)^k`
   predicts. This is what crossed the 0.005 line.
2. **FiLM ≥ concat (the conditioner A/B), on BOTH static-map and dynamics
   controls.** A theory pass predicted concat would beat FiLM by "reaching the
   dynamics"; the mandated A/B overturned that — per-layer FiLM across a deep TCN
   reaches the dynamics in practice, and concat (control only at layer 0) has less
   conditioning surface. **FiLM is the default; concat is a documented option**
   (`system_mode="concat"`, cheaper streaming) but did not earn the default.

The **white-noise broadband ESR (~0.025) is capacity-invariant** (identical at
76k…1.2M params) — a rate–distortion floor of the excitation, not a model limit —
so it is reported only as a diagnostic; realistic signals are the metric.

The whole design is one idea, applied honestly: **split exogenous controls by how
they physically act, and handle each correctly.**

- **Signal-acting** controls — drive, gain, sustain, fuzz, level, a forcing
  amplitude: these *scale the signal*, so they are folded **directly into the
  model input** as a gain (`x ← g·x`). The model is left **unconditioned** on
  them, so it reproduces **any** setting exactly *by construction* —
  interpolation *and* extrapolation — with **zero conditioning parameters** and
  **no interpolation error**.
- **System-acting** controls — tone-stack component values, mode switches, bias,
  a nonlinearity coefficient: these *change the system*, so they drive a
  **minimal per-block FiLM** (a tiny plain conditioner). No Fourier features, no
  Chebyshev head, no per-sample modulator — those were all empirically
  net-negative once the signal controls are handled right.

Backbone = the proven Fast-WaveNet cached dilated TCN spine, by default with a
**heterogeneous mixed-activation block** (`models/tcn._MixedLayer`, see *Mixed
activations* below; the classic WaveNet `tanh*sigmoid` gate `_GatedLayer` remains
available as `block_act="gated"`) + a clamp saturator of last resort + a fixed
1-pole 1 Hz DC-blocker (a low corner — removes only the asymmetric stages' true-DC
silence offset, *not* the audible low band, so it doesn't shift the transfer curve
or widen its hysteresis the way a higher corner does). `conditioned = True`,
pure-numpy streaming
that is **bit-exact and block-size invariant** (`check_streaming` ≤ 1e-4,
moving-knob + ragged-block ≤ 2e-3).

## Spectral fidelity — matching the harmonic formants and the static transfer curve

ESR alone produces an excellent overall fit but leaves a fidelity gap a player
hears: the circuit's **harmonic formants** (the resonant peaks/notches in the
harmonic stack — its poles/zeros shaping the distortion) and the **static transfer
curve** ("Transferkennlinie") were too smooth. The cause is information-theoretic:
ESR is *energy-dominated*, and the formant harmonics sit 16–41 dB below the
fundamental, so they barely register in the loss — training under pure ESR
*smooths them away* (training longer even made the formant error worse). Raising a
magnitude-STFT term recovers the formants but **wrecks the transfer curve**,
because STFT magnitude is **phase-blind** and the transfer curve is a phase-exact
instantaneous relation.

The fix is the default loss: **ESR + a phase-aware pre-emphasis-ESR term**
(`losses.preemph_esr_loss`, a first-order high-pass `H(z) = 1 − 0.95 z⁻¹` on the
error before ESR; Wright & Valimaki, ICASSP-20). It lifts the low-energy formant
band into the gradient *without discarding phase*, so — unlike STFT — it sharpens
the formants **and** the transfer curve at once. On the BJT drive sweep (3-seed
means, ch24, RTF ≈ 4×) it improves every axis simultaneously vs the old
`0.1·STFT`:

| metric | old (ESR + 0.1·STFT) | **CIRCE3 (ESR + 1.0·pre-emph)** |
|---|---|---|
| harmonic-formant-peak error (h5,6,9,10) | 6.8 dB | **4.1 dB** |
| static transfer curve-RMSE | 1.30 | **0.72** |
| clipping-knee-weighted RMSE | 2.54 | **1.27** |
| high-band (>4 kHz) ESR | 0.61 | **0.30** |
| held-out realistic ESR | 0.0375 | **0.0318** |

The win is largest on the strongly-nonlinear BJT (where the formants are richest),
holds on the JFET, and **improves the conditioned FiLM path too** (JFET-tone
held-ESR 0.037 → 0.029). STFT is kept as a constructor option (`stft_weight`) but
is **off by default**. (Note: pre-emphasis slightly *raises* the broadband
white-noise ESR — it chases the unfittable rate-distortion floor there — but that
is the diagnostic-only metric; on realistic signals, which is what matters, it is a
clean win. See `docs/optimal-architecture.md` → "the loss is a first-order lever".)

### Internal oversampling — the dominant spectral lever (removes self-aliasing)

After the loss fix, the *remaining* realistic held-out error turned out to be the
model's **own aliasing**: the TCN's per-layer nonlinearities (tanh/gelu/relu/abs/
Snake) generate harmonics above Nyquist that **fold back into the audio band**, corrupting exactly
the formant region. Running the network at an **internal 2× rate**
(`oversample=2`: upsample → TCN → downsample) pushes the alias-fold to 2× Nyquist
and leaves the in-band spectrum clean. This is the single biggest fidelity lever in
the whole project:

| metric (BJT, ch24) | base rate (1×) | **2× oversampled (shipped)** |
|---|---|---|
| held-out ESR — guitar-DI (the deployment metric) | ~0.03 | **0.0012** |
| held-out ESR — fp test | ~0.032 | **~0.004** |
| high-band (>4 kHz) ESR | 0.33 | **0.11** |
| harmonic-formant-peak error | 5.5 dB | **2–4 dB** |
| real-time factor (CPU, end-to-end) | 4.0× | **2.3×** |

That is an **~8–25× reduction in realistic held-out error** at a still-real-time
RTF. It is **streaming-exact**: a stateful linear-phase polyphase resampler
(`models.circe3._OverSampler`, `process == process_block` to ≤1e-7, block-size
invariant) with a small group delay reported as `latency_samples` (≈1.4 ms at the
default 127 taps). The base-rate capacity / receptive-field / edge-loss ablations
were all *within seed noise* by comparison — confirming the residual was aliasing,
not approximation. The constructor default stays `oversample=1` (latency-0
contract); the **shipped/benchmarked checkpoint uses `oversample=2`** — the
recommended config for nailing the spectral properties. The factor was swept:
**2× is the sweet spot** — 4× fails (held-ESR 0.32, and RTF 0.85× is not real-time)
and a sharper 191-tap FIR is no better than the default 127.

The mechanism is made visible in `outputs/figs/circe3_oversample_aliasing.png`
(reproduce with `uv run python experiments/make_oversample_figure.py`): a 2.5 kHz tone driven
hard produces, in the base-rate (1×) model, a dense forest of **inharmonic alias
spurs** between the true harmonics (its >Nyquist harmonics folded back); the 2×
model removes them, restoring the circuit's clean between-harmonic floor.

### Gradient clipping — the decisive lever for sharp nonlinearities

Training under a fixed **global gradient-norm clip** (`grad_clip=1.0`, the default)
is the single biggest improvement for hard, discontinuous circuits — and it is the
*most* circuit-agnostic lever in the project: one setting, applied to every circuit,
training-only, **zero inference cost**. The dead-zone / fold corner produces large,
spiky gradients that, unclipped, knock Adam into a far-worse minimum and make the
result swing wildly with the random seed. Clipping collapses both the error **and**
its variance:

| metric (class-B crossover dead-zone) | no clip | **grad-clip 1.0** |
|---|---|---|
| held-out ESR (mean of 3 seeds) | 0.27 | **0.033 (~8×)** |
| seed spread | 0.16 – 0.39 | **0.031 – 0.035** |

It is not a hard-circuit-only trick — across the full 9-circuit suite it helps
6/9 (crossover −88%, fullwave −35%, BJT −25%, tube-screamer −15%, hysteretic −14%,
JFET −10%), is neutral on 2 (asym-clipper, wavefolder), and mildly regresses 1
(hard-clipper +11%, within seed noise). It is uniform and training-only, so
streaming, latency and RTF are unchanged. **This — not the rectified-feature
basis — is what tamed the discontinuity circuits** (`experiments/make_gradclip_plot.py` →
`outputs/figs/gradclip_win.png`). A clip-value sweep confirmed 1.0 is the sweet
spot (0.5 ≈ 1.0; 2.0 is worse).

### Mixed activations — the default block (smaller, faster, ≥ the WaveNet gate)

Each TCN layer's nonlinearity defaults to a **heterogeneous mixed activation**
(`block_act="mixed"`, `models/tcn._MixedLayer`) rather than the classic WaveNet
`tanh*sigmoid` gate. The channels are split into five (near-)equal groups carrying
**tanh** (smooth saturation), **gelu** (smooth gate), **relu** (one-sided corner),
**abs** (V-shaped corner — a natural fit for a symmetric dead-zone) and **Snake**
`x + sin(αx)²/α` (periodic / harmonic folding, per-channel learnable `α`). A purely
smooth `tanh`/`sigmoid` gate cannot represent a *corner* (a slope discontinuity)
without enormous capacity; giving every layer abs/relu/Snake units lets the net
synthesize the kink natively, while the layer's `1×1` recombine can *suppress*
those units where they are not needed — which is why, unlike an input-level
rectified-feature basis, mixed activations do **not** regress smooth transfer
curves.

The earlier verdict ("modest gain, inference-cost → opt-in") was overturned once
**gradient clipping** removed the mixed block's training instability. The decisive
9-circuit head-to-head (both grad-clip, 150 ep, ch24/L9/OS2; held-out ESR):

| circuit | gated | **mixed (default)** | winner |
|---|---|---|---|
| bjt [smooth] | 0.0047 | 0.0046 | tie |
| jfet [smooth] | **0.0083** | 0.0095 | gated (+14%) |
| tube_screamer [smooth] | **0.0133** | 0.0137 | gated (+3%) |
| crossover [hard] | 0.0264 | **0.0230** | mixed (−13%) |
| wavefolder [hard] | 0.1982 | **0.1829** | mixed (−8%) |
| asym_clipper [hard] | 0.0525 | 0.0523 | tie |
| hard_clipper [hard] | 0.0935 | **0.0902** | mixed (−4%) |
| fullwave_rectifier [hard] | 0.0091 | **0.0085** | mixed (−7%) |
| hysteretic_fuzz [hard] | 0.0274 | 0.0270 | tie |

Mixed wins 4 (every win on a *hard* circuit), ties 3, and loses 2 — both losses
mild and on the *smoothest* circuits (jfet, tube-screamer). Decisively, mixed is
also **smaller and faster**: its dilated conv emits `channels` features (not
`2·channels`), so the shipping model is **54k params at RTF 2.45×** vs the gate's
**85k at 2.23×** (the `ch→ch` conv outweighs the Snake `sin()` cost). Uniformly
≥ on accuracy where it matters, fewer parameters, faster, real-time, and
streaming-exact (the numpy twin `_mixed_act_np` matches the torch forward to
~7e-8 incl. OS2 + FiLM) — so it is the **circuit-agnostic default**, consistent
with the project's one-config-for-all-circuits constraint. `block_act="gated"`
keeps the original gate for the two smooth circuits where it is marginally ahead.
Reproduce: `experiments/mixed_vs_gated_final.py` → `outputs/mixed_vs_gated_final.json`.

## Why this is the optimal design (not more compute)

The earlier models *learned* what a pre-gain does (a FiLM map `g → (γ,β)`) and
then **mis-interpolated it** — that learned-conditioning interpolation error is
the dominant source of held-out loss. CIRCE3 removes it by construction. Every
building block in CIRCE2 (FiLM-on-drive, rational gate, Chebyshev head, Fourier
features, per-sample modulator) was compensating for the wrong control
parameterization; fixing the parameterization makes them unnecessary *and*
counterproductive. The result is **lower error, fewer parameters, lower variance,
and real-time** — from mathematics, not layers.

## Benchmark vs the other architectures

`vguitar benchmark --circuits bjt,jfet,tube_screamer` races CIRCE3 against every
other **deployable** architecture (tcn, rnn, wiener-hammerstein, volterra) on
identical data. Held-out content at the nominal operating point (ESR, lower =
better):

| circuit | CIRCE3 (OS2, grad-clip, 1 Hz DC-block) | tcn | volterra | wh | rnn |
|---|---|---|---|---|---|
| **bjt** (BJT overdrive) | **0.0046** (RTF 2.4×) | 0.290 | 0.217 | 0.643 | 1.03 |
| **jfet** (square-law) | **0.0093** (RTF 2.4×) | 0.034 | 0.091 | 0.097 | 0.248 |
| **tube_screamer** (op-amp + diode clipper) | **0.0131** (RTF 2.4×) | 0.062 | 0.161 | 0.171 | 0.209 |

(Lowering the output DC-blocker corner from 5 Hz to 1 Hz — it only ever needed to
remove the asymmetric stages' true-DC silence offset — recovered the audible low
band the 5 Hz HPF was stripping: jfet 0.025→0.011, tube_screamer 0.026→0.011,
and it removed the transfer-curve vertical offset + hysteresis widening.)

The oversampled CIRCE3 is the **best real-time architecture on all three
circuits** — on the strongly-nonlinear bjt it is **tens of times more accurate than
the strong TCN backbone** (0.0046 vs 0.29) and orders of magnitude ahead of the
recurrent / classical models. On the **gently nonlinear jfet** it still wins by
**~3.7×** (0.0093 vs 0.034) — while remaining a *single* model trained across the
whole drive sweep, not a fixed-point specialist (the fixed-point baselines are
pinned to one operating point, so their held-out error across the sweep is both
higher and more seed-variable).
That is the capability CIRCE3 is built for and the **only** model here that has it:
it takes the control knob — one CIRCE3, by input-scaling, stays accurate at *every*
drive (seen and unseen), while a TCN pinned to one operating point is ~10× worse
across the sweep (see the generalization figure). (The state-space S4/SSM model is
excluded from the benchmark: it is not real-time — RTF ~0.3× — and its chunked-FFT
scan trains ~50× slower than the convolutional models, so it is strictly dominated;
it remains a registered model runnable via `vguitar bench --model ssm`.) Figures:
`outputs/figs/compare_{bjt,jfet,tube_screamer}_*` —
`esr_rtf`, `generalization`, `transfer` (output-vs-input curve + residual panel),
`harmonics` (stack + per-harmonic error panel), `spectrum` (the Welch magnitude
**formant envelope** vs circuit + error, log-freq), `spectrogram` (circuit / CIRCE3
/ dB-difference time-frequency map) — plus `compare_esr_matrix.png`;
`outputs/benchmark.{csv,json}`.

## Results — Duffing oscillator (NON-AUDIO, general exogenous inputs)

To show the approach is **not audio-specific**, CIRCE3 emulates a driven Duffing
oscillator `ẍ + δẋ + ω₀²x + ω₀²β x³ = ω₀² A u(t)` (`systems/duffing.py`), with two
exogenous controls of *different kinds*: forcing amplitude `A` (signal-acting →
folded into input) and cubic-nonlinearity strength `β` (system-acting → FiLM).
Trained on a 4×4 `(A, β)` grid; tested on **held-out interior settings unseen on
amplitude AND on β**:

| conditioning scheme | held-out mean ESR |
|---|---|
| **signal/system split** (A→input, FiLM on β) | **0.224** |
| full-FiLM (condition on A *and* β, the "naive" way) | 0.996 (fails) |

The split **works and generalizes** on a non-audio nonlinear dynamical system;
conditioning on the amplitude (a signal control) instead **fails catastrophically
(ESR ≈ 1)** there too — exactly as it does for audio. This is the core
generalization result: the control-decomposition principle is **domain-general**.
(Reproducible via the `systems/duffing.py` testbed; on realistic band-limited
forcing with a dense β grid the held-out error drops under 0.005 — see the
headline table above.)

## Honest limits

- **Duffing β-interpolation is the residual** (held-mean 0.224 vs BJT's 0.045).
  The amplitude axis is exact (input-scaling); the error is the *system* control
  (`β`) interpolating through FiLM on a stiffer 2nd-order resonant system. Denser
  system-control sampling / more capacity would lower it — but the demonstration
  (generalizes, and the split is essential) is the point.
- **The single sharpest clipping knee still rounds slightly** — a finite-Lipschitz
  smooth net cannot synthesize a true discontinuity (a fundamental limit, not a
  tuning gap). The pre-emphasis loss roughly halves the knee error (2.54 → 1.27)
  and the formant mismatch, but that one corner stays the residual; CIRCE3 matches
  the loop, the formants, and the harmonics elsewhere.
- **Multi-fold wavefolding is now the single hardest circuit** — the edge-case
  battery (`circuits/{fullwave_rectifier,hard_clipper,wavefolder,asym_clipper,
  crossover_classb,hysteretic_fuzz}.py`) is otherwise largely solved. Held-out ESR
  at the shipping config (150 ep, OS2, grad-clip, mixed block): crossover **0.022**,
  fullwave-rectifier **0.009**, hysteretic-fuzz **0.027**, asym-clipper **0.052**,
  hard-clipper **0.092**. The **class-B crossover dead-zone** that used to break the
  model (≈0.37 with the old config) is now well within range — gradient clipping,
  not the rectified-feature basis, was what fixed it (see *Gradient clipping*
  above). The **multi-fold wavefolder remains the outlier (≈0.18)**: its many folds
  generate harmonics that alias even at 2× oversampling, so the residual is a
  real-time-bandwidth limit (a uniform model cannot raise oversampling per-circuit)
  rather than an optimization one. The opt-in rectified-feature basis
  (`rect_thr=(...)` → concat `ReLU(±x−θ)`, `abs(x)`, streaming-exact,
  input-scaling-preserving) remains available for discontinuity-heavy stages but is
  no longer the primary fix.
- **Mild, near-square-law circuits get faint spurious HF.** On a clean tone the
  JFET produces only h2 + a trace of h3, but CIRCE3 adds a low harmonic stack
  (h3–h12 at ≈ −34…−77 dB) the circuit lacks — the model's ~1% residual error
  surfacing as harmonics where nothing else masks it (the broadband held-ESR is
  blind to it). It tracks the error floor (the lower-error BJT shows none), is
  masked in real playing, and an isolated-tone training component
  (`signals.tone_bank`) trims it modestly.
- **Validated circuits:** 11-circuit suite (diode, BJT, JFET, tube-screamer,
  big-muff + the 6 edge cases); control kinds: one signal axis (BJT drive) and one
  signal+one-system axis (Duffing/JFET tone). Multi-system-axis follows the same
  FiLM path and is the next validation.

## Use

```python
from vguitar.models.circe3 import CIRCE3
# drive is signal-acting -> column 0 folded into the input; no other controls.
m = CIRCE3(n_control=1, signal_idx=(0,))
y = m.process(x, c=[0.08])          # any drive, incl. unseen (exact by input-scaling)
# a system control (e.g. Duffing beta) -> not in signal_idx, conditioned via FiLM:
m = CIRCE3(n_control=2, signal_idx=(0,))   # col 0 = amplitude (signal), col 1 = beta (system)
```

Training auto-uses the GPU (`pick_device`); inference/RTF are measured on the CPU
(`to_inference_cpu`) so the numbers reflect the deployment target. Shipped
checkpoints (mixed block, `assets/checkpoints/`):
`{bjt,jfet,tube_screamer}.circe3.model`.
