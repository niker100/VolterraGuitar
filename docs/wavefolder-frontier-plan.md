# Cracking the wavefolder wall (~0.18) — radical-swing roadmap

The multi-fold wavefolder is the one circuit no uniform lever has moved
(held-ESR ≈0.18 at the shipping OS2 config; OS/depth/width/clip/mixed/epochs/EMA
all neutral — see `docs/optimal-architecture.md` and the project log). Prior work
treated it as a "real-time-bandwidth floor" and stopped. This plan takes real
swings instead. Every idea here is **uniform** (one architecture for all circuits,
no per-circuit tailoring) and **real-time on CPU** — the binding constraints — but
otherwise as radical as needed.

## Step 0 RESULT — the wall is NOT aliasing (representation-limited → Branch B)

`wavefolder_os_ab.py` (mixed/grad-clip/150ep, 2 seeds on the hard arms):

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

## Step 0 — the diagnostic (DONE: `wavefolder_os_ab.py`)

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

`wavefolder_shaper_ab.py` (`out_shaper="fourier"`, K=8, residual `y=o+Σc_k sin(kwo)`,
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
