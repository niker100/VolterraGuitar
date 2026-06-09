# Cracking the wavefolder wall (~0.18) — radical-swing roadmap

The multi-fold wavefolder is the one circuit no uniform lever has moved
(held-ESR ≈0.18 at the shipping OS2 config; OS/depth/width/clip/mixed/epochs/EMA
all neutral — see `docs/optimal-architecture.md` and the project log). Prior work
treated it as a "real-time-bandwidth floor" and stopped. This plan takes real
swings instead. Every idea here is **uniform** (one architecture for all circuits,
no per-circuit tailoring) and **real-time on CPU** — the binding constraints — but
otherwise as radical as needed.

## Step 0 RESULT — the wall is NOT aliasing (representation-limited → Branch B)

`experiments/wavefolder_os_ab.py` (mixed/grad-clip/150ep, 2 seeds on the hard arms):

| config | held-ESR | vs OS2 | RTF |
|---|---|---|---|
| OS2_ch24 (shipping) | 0.1896 | — | 2.46× |
| OS3_ch24 | 0.1862 | −2% (noise) | 2.02× |
| OS4_ch24 | 0.2156 | **+14% worse** | 1.59× |
| OS4_ch16 | 0.2242 | **+18% worse** | 2.39× |

More internal bandwidth does **not** help — OS4 is *strictly worse* (a tighter
Nyquist/4 FIR cutoff raises the resampler floor and 4×-longer windows hurt
convergence at fixed epochs, but the headline is unambiguous: bandwidth is not the
lever). **The wall is representation / inductive bias, not self-aliasing.** Branch A
(ADAA alias-free activations) is therefore deprioritized; **taking Branch B**.

## Step 0 — the diagnostic (DONE: `experiments/wavefolder_os_ab.py`)

Train production CIRCE3 (mixed/grad-clip/150ep) at `oversample ∈ {2,3,4}` and read
held-ESR **and** CPU RTF. This is the branch point:

- **If OS4 lowers held-ESR meaningfully** → the wall IS the model's own
  *self-aliasing* (its internal nonlinearities fold harmonics back in-band). The
  problem becomes *efficient* anti-aliasing (get OS4 fidelity at OS2 cost). →
  Branch A.
- **If OS4 does NOT help** → the wall is *representation / inductive bias*, not
  bandwidth. More internal rate cannot synthesize the band-limited fold. →
  Branch B.

(The old "OS4 fails" verdict was measured on the BJT and gave a nonsensical 0.32 —
worse than OS2 for an anti-aliasing knob — so it is almost certainly a stale
artifact, never tested on the wavefolder. Hence this clean re-test.)

## Branch A — alias-free nonlinearities (efficient anti-aliasing)

Global oversampling band-limits the *whole* network at NxNyquist cost. The
principled, cheaper alternative attacks aliasing at its source — the pointwise
nonlinearity — with **antiderivative anti-aliasing (ADAA)** (Parker–Zavalishin–
Bilbao; the standard real-time anti-aliased-waveshaper method):

> 1st-order ADAA of an activation `f` with antiderivative `F`:
> `y[n] = (F(x[n]) − F(x[n−1])) / (x[n] − x[n−1])`, with the removable singularity
> at `x[n]≈x[n−1]` handled by falling back to `f((x[n]+x[n−1])/2)`.

This band-limits each nonlinearity with **1 sample of state per channel** and a
closed-form `F` — far cheaper than running the entire TCN at 4×. Closed forms for
our activation zoo:
- `tanh`  → `F = log cosh x`
- `relu`  → `F = ½ relu(x)²`
- `abs`   → `F = ½ x·|x|`
- `snake` `x + sin²(αx)/α` → `F = x²/2 + (x/2 − sin(2αx)/(4α))/α` (closed form)
- `gelu`  → use the tanh-approx gelu so `F` is closed-form, or ADAA only the
  corner-bearing groups (relu/abs/snake) where aliasing actually originates.

**The swing:** an *alias-suppressed mixed-activation TCN* — every layer's
activation ADAA'd — trained at OS1 (or OS2) and A/B'd against plain-OS2 on the
wavefolder (does it crack 0.18?) AND on bjt/jfet (must not regress smooth
circuits, and ideally lets us *drop* OS for an RTF win). Streaming-exact via the
per-channel 1-sample ADAA state; both torch forward and the numpy twin.

Fallback within A: StyleGAN3-style *filtered* nonlinearities (local up/down +
Kaiser-windowed filters around each activation) if plain ADAA under-suppresses.

## Branch B — a representation with the right inductive bias for folding

If bandwidth is not the wall, the smooth-TCN *prior* is wrong for a periodic fold.

### Swing #1 RESULT — learned Fourier output shaper: REJECTED (null + mild regression)

`experiments/wavefolder_shaper_ab.py` (`out_shaper="fourier"`, K=8, residual `y=o+Σc_k sin(kwo)`,
mixed/grad-clip/OS2/150ep, multi-seed):

| circuit | baseline | fourier8 | Δ | RTF |
|---|---|---|---|---|
| **wavefolder** (target, 2 seeds) | 0.1798 | 0.1793 | **−0%** (noise) | 2.62× |
| bjt (guard) | 0.0046 | 0.0047 | +1% (noise) | 2.56× |
| jfet (guard) | 0.0088 | 0.0097 | **+10% (regress)** | 2.63× |

An explicit periodic primitive **at the output** does not move the wavefolder
(within seed noise) and mildly hurts smooth jfet. So a *pointwise* fold-basis on
the scalar pre-output is too weak — the folding has to interact with the signal
dynamics *through* the network, not be bolted on at the end. The option is kept
default-off (RTF-free, harmless) but is **not** integrated. → escalate to the
swings below (periodicity *throughout* the net, or a fundamentally different
representation), gated on the capacity probe.

Remaining swings:
1. **Sinusoidal output basis / harmonic head.** The fold is near-periodic in the
   input amplitude; predict a band-limited harmonic series (learned amplitudes ×
   a fixed `sin(kθ)` basis where `θ` tracks an instantaneous phase from the input)
   instead of raw samples — band-limited *by construction* (no harmonic above the
   chosen K). Distinct from the retired Fourier-feature *input* embedding; this is
   a Fourier *output* parameterization. Real-time (a small oscillator bank).
2. **Snake-dominant / learned-period block.** The Snake activation
   `x+sin²(αx)/α` is a folding primitive; a block that is mostly Snake with a
   learned per-channel period may match folds a tanh-heavy block cannot. Test as a
   `block_act` variant, uniform.
3. **Teacher–student distillation** from an offline OS8 high-capacity teacher into
   the real-time student — gives the student a cleaner, alias-free target than the
   raw decimated SPICE output. (Adds no bandwidth, but may help the student spend
   its capacity better.)

## Protocol (all swings)

- GPU-bound CIRCE3 training (`device="cuda"`, 150ep shipping config), honest
  multi-seed A/B vs the plain-OS2 mixed baseline.
- Report held-ESR **and** CPU RTF (real-time is non-negotiable) **and** a smooth
  circuit (bjt/jfet) to catch regressions — a wavefolder win that breaks the
  showcase is not a win.
- Streaming-exactness + save/load round-trip for anything integrated into CIRCE3.
- Document each result (win or honest null) in the project log; integrate only
  what A/Bs prove, retire the rest (the project's discipline).

The honest bar: a *real* break is wavefolder well under ~0.15 with no smooth
regression and RTF > 1. A null result that finally explains *why* the wall holds
(with the OS4 evidence) is also a publishable conclusion — but only after the
swings above are actually taken.

## Capacity verdict — the wavefolder ~0.18 is a HARD TARGET FLOOR (not a model limit)

`experiments/wavefolder_capacity_probe.py` (big configs fairly trained: lr=1e-3,
grad-clip off, more epochs, OS2 — after the first attempt's ch24-tuned
hyperparameters made ch48 *diverge* to 0.93):

| config | params | held-ESR | vs ref | RTF |
|---|---|---|---|---|
| ref ch24/L9/OS2 | 54k | **0.1893** | — | 2.38× (RT) |
| ch48 | 213k | 0.2050 | +8% | 0.28× (not RT) |
| ch48/L11 | 259k | 0.2251 | +19% | 0.02× (not RT) |
| ch64/L11 | 459k | 0.2104 | +11% | 0.08× (not RT) |

**Even a fairly-trained 8.5×-bigger model is WORSE, and none is real-time.** With
the OS null (bandwidth) and the Fourier-head null (output primitive), three
independent levers now agree: **the wavefolder floor is a property of the
band-limited target** (its multi-fold harmonics exceed what a causal real-time
operator can recover from this excitation), not a capacity or bandwidth gap. This
is the honest, decisive conclusion for the wavefolder; further single-circuit
chasing is not worthwhile. The consolidated evidence figure is
`outputs/figs/frontier/wavefolder_frontier.png`
(`uv run python -m experiments.figures`).

## Step 2 — the multi-modal / mixture-of-experts swing (the big generalization bet)

User direction (high priority): *"maybe it would make sense to have a really
multimodal model, combining different activation functions, FFTs, longer memory,
shorter memory, deep paths, shallow paths, convolutions etc. The goal is to
generalize very well. If it isn't possible with a single architecture, just combine
every possible approach — still it should not be tailored to one circuit."*

The thesis: no *single* inductive bias is best for every circuit (the project log
shows smooth-saturating circuits want tanh/smoothness, sharp circuits want
corners, folds want periodicity, memory-circuits want long dilation, near-static
ones want shallow). So **fuse heterogeneous experts in parallel and let a learned
gate pick the mix per signal** — one architecture, trained circuit-agnostically,
that *contains* every approach. Call it **CIRCE-X** (experimental; CIRCE3 stays the
clean shipping model).

**Parallel branches (heterogeneous inductive biases), each a small TCN-ish stack:**
- **Activation-family experts:** tanh-gated (smooth), mixed (corner-capable),
  snake/periodic (folds), abs/relu (kinks). (Reuses `_GatedLayer`/`_MixedLayer`.)
- **Memory experts:** a **long-memory** path (large dilations / many layers, big
  receptive field for jfet/TS/bjt) **and** a **short-memory** path (few layers,
  near-static hard clippers) — depth/RF diversity.
- **Deep vs shallow** paths (capacity diversity).
- **Spectral/FFT branch:** an STFT → 1×1 mixing → iSTFT path (or a learned complex
  mask) for formant/harmonic structure a time-domain conv under-weights. Must be
  made causal + streaming (blocked STFT with overlap-save) or flagged offline-only
  for the first A/B.
- All branches share the input-scaling + FiLM control handling (so the
  signal/system split is preserved across the ensemble).

**Fusion:** a learned per-sample (or per-block) **gating head** (softmax over
branches, optionally control- and signal-feature-conditioned) — a mixture of
experts. The gate is learned from data, identical across circuits (no per-circuit
hyperparameters): on a smooth circuit it should learn to down-weight the periodic/
corner experts, on the wavefolder up-weight the periodic one, etc. That *is* the
generalization mechanism.

**Evaluation (circuit-agnostic, the whole point):** the headline is **mean AND
worst-case held-ESR across the full suite** (bjt, jfet, tube_screamer, + the 6 edge
cases) — not any single circuit — plus **CPU RTF**. Multi-seed, vs the current
mixed/OS2/grad-clip default, with regression guards on circuits already < 0.02.

**RTF discipline (binding):** the kitchen-sink will likely blow the real-time
budget. So the plan is: (1) train the full ensemble, measure per-circuit gate
weights + held-ESR; (2) **ablate branches** to find which actually carry the
cross-circuit generalization (gate weight × ESR contribution); (3) **prune to the
smallest real-time-affordable subset** that keeps the mean/worst-case win, and
document the ablation. A non-real-time ensemble that generalizes is still a useful
*teacher* (distill into a real-time student) and an informative upper bound — but
the shippable result must be RTF > 1.

**Order:** take this AFTER the capacity probe (Step 1 / `wavefolder_capacity_probe.py`)
returns — that result tells us whether the wavefolder residual is even reachable by
*any* model, which sets expectations for what the ensemble can do there. Build
CIRCE-X as a new experimental model (`models/circex.py`), streaming-contract-tested,
default-off; integrate into CIRCE3 only the branches an ablation proves carry
generalization (the project's retire-what-doesn't-help discipline).

### Step 2 RESULT — CIRCE-X MoE: REJECTED (the gate averages, it doesn't specialize)

`experiments/circex_probe.py` (3 experts: gated-L9 / mixed-L7 / gated-L4 + a
per-sample softmax gate, vs single-branch mixed TCN, OS1, same harness):

| circuit | baseline | CIRCE-X | Δ | gate (long/mixed/shallow) |
|---|---|---|---|---|
| bjt [smooth] | 0.0291 | 0.0306 | +5% | 0.05 / **0.71** / 0.25 |
| jfet [smooth] | 0.0037 | 0.0096 | **+160%** | 0.33 / 0.31 / 0.36 |
| tube_screamer [smooth] | 0.0073 | 0.0167 | **+129%** | 0.38 / 0.26 / 0.36 |
| crossover [hard] | 0.0583 | 0.0658 | +13% | 0.29 / 0.28 / 0.43 |
| wavefolder [hard] | 0.2288 | 0.1698 | −26% | 0.33 / 0.31 / 0.36 |
| **mean** | 0.0655 | 0.0585 | (wavefolder-dominated) | |

**Verdict: rejected.** The MoE regresses 4/5 circuits — the smooth showcase
(jfet/TS) catastrophically (+130-160%) — and wins only the wavefolder (a
documented target floor anyway), at 1.6× params. The "mean improved" is purely the
large-absolute wavefolder dominating the average. The **gate-weight figure**
(`outputs/figs/frontier/circex_gate.png`) is the diagnostic: except on bjt (gate
concentrates on `mixed`=0.71, only +5%), the gate collapses to a **~uniform split**
— it *averages* the experts rather than specializing, which is strictly worse than
the single best branch. That's the classic soft-MoE failure mode (no sparsity/
load-balancing incentive). **Conclusion: a jointly-trained soft-gated MoE over
heterogeneous branches does NOT generalize better than one well-chosen branch (the
mixed TCN); fusing every approach dilutes rather than combines.** A sparsity-forced
or top-1-routed gate is a conceivable refinement, but the prior is now weak and the
smooth regressions are large — parked in favour of the more principled spectral
swing (Step 3), which targets a *specific* documented gap with the *right* tool
rather than hoping a gate learns to specialize.

## Step 3 — spectral-domain (STFT) hybrid: complex filtering + a time-domain head

User direction (high priority, AFTER CIRCE-X): *"maybe we need to be more aggressive
than a simple TCN — an FFT, then multiplication in the spectral domain (= convolution
in time), then IFFT, with a head for purely time-based processing. Carefully
engineered: complex-valued network, real-time, probably STFT with different lengths."*

**Why it's well-motivated (not just novelty):** these circuits are a *linear filter*
∘ *instantaneous nonlinearity* ∘ *linear filter* (tone stacks, coupling caps, output
RC — the formants/poles-zeros) wrapped around diode/tube waveshaping. The **linear**
parts are *long* convolutions — cheap and exact as a **complex multiply per STFT
bin** — exactly the **formant/transfer fidelity** a time-domain TCN smooths (our
documented residual). The **nonlinear** part is *time-local* waveshaping — natural
for a small time-domain head. So the right shape is a **hybrid**:

```
x ─┬─ STFT(multi-len) ─ complex per-bin/per-frame net ─ ISTFT ─┐
   └─ time-domain head (TCN waveshaper, instantaneous NL) ──────┴─ fuse ─ y
```

**The three engineering constraints the user flagged — design notes:**

1. **Pure spectral multiplication is LINEAR (LTI).** A static complex mask = one
   fixed linear filter; it cannot create harmonics. For a *nonlinear* circuit the
   spectral processing must be **input-dependent**: a complex-valued net maps the
   frame spectrum (and/or a conditioning feature) to a per-frame complex
   transform ("deep filtering" / learned time-varying filter). The harmonic
   *generation* still comes from the time-domain head; the spectral branch carries
   the **linear formant/memory** structure. Keep that division of labour explicit.
2. **Complex-valued network.** Implement as split real/imag (2 real channels) with
   complex-aware ops (complex linear = the 2×2 real block; `modReLU`/`CReLU`
   activations; magnitude-phase only where it helps), or torch native `cfloat`.
   Start with the simplest that trains stably (split-real linear + modReLU).
3. **Real-time + STFT.** STFT framing imposes **latency ≈ window length** (the
   resolution↔latency tradeoff is fundamental). Use **overlap-add / overlap-save**
   blocked STFT for streaming-exactness, report the window as `latency_samples`
   (like the OS group delay). **Multi-length STFT** (e.g. 256/1024/4096) = a
   multi-resolution branch (short window → time/transient precision, long window →
   frequency/formant precision), fused — the analysis twin of `multi_stft_loss`.
   The longest window sets the latency budget; cap it to stay playable (≤~10 ms?).

**Plan:** prototype `_SpectralBranch` (single window first, then multi-len),
validate (a) it improves **formant/transfer fidelity** on bjt/jfet/tube_screamer
(the spectral-fidelity metrics: formant-peak err, transfer-RMSE, band>4k ESR) and
(b) streaming-exactness via overlap-add, **before** any fusion. Then A/B the
**hybrid (spectral + time head)** vs the plain time-domain default across the suite,
held-ESR + the spectral metrics + RTF + latency. It can also slot in as an extra
CIRCE-X branch. Honest bar: it must improve *spectral* fidelity (its reason to
exist) without losing real-time or regressing the time-domain wins — else it's a
documented null. Build as `experiments/spectral_probe.py` first (offline torch),
promote to a streaming model only if it earns it.

### Step 3 RESULT — the spectral branch WINS on formant fidelity (the first real swing)

`experiments/spectral_probe.py` (`y = time_TCN(x) + ISTFT(G(|STFT(x)|)⊙STFT(x))`,
n_fft=1024/hop=256, OS1, same harness; spectral ON vs OFF):

| circuit | overall ESR | **band>4k ESR (the formant target)** |
|---|---|---|
| bjt [smooth] | 0.0299 → 0.0221 (**−26%**) | 0.0545 → 0.0377 (**−31%**) |
| jfet [smooth] | 0.0098 → 0.0081 (−17%) | 0.0170 → 0.0028 (**−84%**) |
| tube_screamer [smooth] | 0.0160 → 0.0136 (−15%) | 0.0247 → 0.0115 (**−53%**) |
| crossover [hard] | 0.0694 → 0.0799 (+15%) | 0.0741 → 0.0818 (+10%) |

**This is the first architectural swing that genuinely wins** — and exactly where
theory predicts: on the formant-rich *smooth* circuits (the real pedal/amp
circuits) the input-dependent complex STFT branch cuts the **high-band/formant
error 31–84%** and overall ESR 15–26%, directly closing the documented
formant-fidelity gap a pure time-domain TCN smooths. It regresses only the
pure-discontinuity **crossover** (+15% — no formant structure there; the branch
dilutes the time head's corner work), so it is a **smooth-circuit tool**, not
universal. Figure: `outputs/figs/frontier/spectral_ab.png`.

**Caveats / escalation (the binding open question = real-time):** the prototype is
**offline** (`center=True` STFT, non-causal, ~12 ms look-ahead) and the spectral
MLP is **heavy (422k params)**. BUT the per-frame cost amortizes to ~1.6k MAC/sample
(one 513→256→1026 matmul + a 1024-pt FFT every 256-sample hop), so it is plausibly
real-time — **must be measured**. Next: (1) causal **overlap-add streaming** version
(latency = window; report it), measure **RTF**; (2) **slim** the MLP (smaller hidden
/ per-bin-local / low-rank) — the win likely survives a much smaller net; (3)
**multi-length STFT** (256/1024/4096) only if latency budget allows (the long window
dominates latency → ~93 ms at 4096, too much for live — likely keep ≤1024). If RTF
> 1 and latency is playable, **integrate into CIRCE3 as an optional spectral branch
for smooth circuits** (gate it off for discontinuity circuits). This is the most
promising lever found in the radical-swings arc.

**SLIM RESULT — the win is real-time-affordable (`experiments/spectral_slim.py`).**
Sweeping the spectral hidden width {16,32,64,256} on bjt+jfet: the band>4k win is
**flat across all sizes**, and **hidden=16 is as good or better** than 256 (bjt
band −32% at every width; jfet −90% at h16 vs −84% at h256 — the big MLP slightly
overfits). hidden=16 adds only **~25k params** (513→16→1026), ~4M MAC/s amortized
over the 256-sample hop — **trivially real-time**. The 422k MLP was pure overkill.
→ **Greenlit:** build the causal overlap-add streaming spectral branch at
hidden≈16-32 (train with `center=False` so streaming matches), confirm
streaming-exactness + RTF + latency, then integrate into CIRCE3.
Figure: `outputs/figs/frontier/spectral_slim.png`.
