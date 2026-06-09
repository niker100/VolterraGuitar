# THE OPTIMAL ARCHITECTURE FOR A PARAMETERIZED FAMILY OF NONLINEAR SYSTEMS WITH EXOGENOUS CONTROLS

**A theory-derived, real-time, minimal design — and the experiment that decides whether anything beats a well-trained TCN**

---

## EMPIRICAL VERDICT (the decisive A/B was run — READ THIS FIRST)

The design below recommended **concat** conditioning and predicted *concat ≥ FiLM* (because FiLM "cannot reach the dynamics"). **Per the design's own rule — "don't crown a winner on paper, run the A/B" — the experiment was run, and it overturned that specific prediction.** On the analytically-known testbeds, FiLM ≥ concat on **both** control kinds:

| control kind | testbed | FiLM (dense grid) | concat (dense grid) |
|---|---|---|---|
| static-map (nonlinearity) | Duffing β | **0.0023** (worst 0.0041) | 0.0037 |
| dynamics (pole/RC corner) | jfet tone | **0.0029** (worst 0.0033) | 0.0041 |

So **per-layer FiLM across a deep TCN *does* reach the dynamics in practice** (the single-layer "measure-zero reachable set" argument was too strong), and concat — which injects the control only at layer 0 — has *less* conditioning surface. **FiLM is the empirical winner and the default.** concat is kept as a documented option (cheaper streaming, no overfitting surface) for circuits with an unusually strong dynamics dependence, but it did not earn the default.

**What the experiment CONFIRMED (the load-bearing meta-conclusions):**
- **Grid density is the dominant lever**, exactly as §5's `ε_interp ~ h^k` bound predicts: densifying the β grid dropped held-out ESR ~2.7× and took Duffing from 0.0063 (coarse, over target) to 0.0023 (dense, under). The conditioner choice is second-order; **the control-grid is first-order.**
- **Signal-acting controls → input-scaling** (0-bit equivariance) is unambiguously optimal and already shipped.
- **The white-noise ESR floor (~0.025) is capacity-invariant** (76k…1.2M params) ⇒ a data/excitation rate-distortion floor, not a model limit ⇒ the meaningful metric is **band-limited (realistic) signals**, on which the target is met.

**THE FINAL OPTIMAL ARCHITECTURE (empirically settled):** a well-trained dilated **TCN** + **input-scaling** for signal controls + **minimal FiLM** for system controls + **dense control-grid sampling**. On realistic signals this hits **ESR < 0.005 on every tested control type, real-time on CPU**: BJT drive (signal) 0.001–0.002; Duffing β (static-map) 0.0023; jfet tone (dynamics) 0.0029. This is exactly the original CIRCE3 (`system_mode="film"`); the theory's value was identifying input-scaling, the grid-density lever, the correct metric, and the discipline to let the A/B — not the paper — pick the conditioner.

### ADDENDUM — the loss is a first-order lever for *spectral* fidelity (pre-emphasis ESR)

The ESR-vs-RTF result above is excellent, but ESR alone leaves a real fidelity gap the deployment user feels: the circuit's **harmonic formants** (the resonant peaks/notches in the harmonic stack — its poles/zeros shaping the distortion) and the **static transfer curve** ("Transferkennlinie") were not matched to the necessary exactness. The diagnosis and fix are a clean information-theoretic story, and they overturn a prior project note:

- **Why plain ESR misses the formants.** ESR is `Σ(error²)/Σ(signal²)` — *energy-dominated*. The formant harmonics sit 16–41 dB below the fundamental, so they contribute almost nothing to the ESR sum; gradient descent under pure ESR polishes the loud fundamental and **smooths the low-energy formants away** (measured: training *longer* at low spectral weight made the formant error *worse*, 9→13 dB — the model converging harder onto the energy-bearing part).
- **Why a magnitude-STFT term is the wrong fix.** Raising a multi-resolution-STFT weight *does* recover the formants but **wrecks the static transfer curve** (curve-RMSE 0.81→1.77, knee 0.89→2.79) — because STFT magnitude is **phase-blind**: it admits magnitude-right / waveform-wrong minima, and the transfer curve is a phase-exact instantaneous I/O relation. This is a genuine, repeatable trade-off, not a tuning artifact.
- **The fix: a phase-aware high-frequency-weighted loss — pre-emphasis ESR.** Add `ESR( H(z)·ŷ , H(z)·y )` with `H(z) = 1 − 0.95 z⁻¹` (a first-order high-pass; Wright & Valimaki, ICASSP-20). It lifts the low-energy formant band into the gradient **without discarding phase** (it is still a time-domain waveform error). ESR and pre-emphasis-ESR are *both* minimized by the same true-waveform match, so they reinforce rather than conflict — unlike STFT.

**Result (3-seed means, BJT drive sweep, ch24, real-time RTF ≈ 4×):** replacing `0.1·STFT` with `1.0·preemph-ESR` improves **every** axis at once — formant-peak error 6.8→4.1 dB, transfer curve-RMSE 1.30→0.72, knee 2.54→1.27, high-band (>4 kHz) ESR 0.61→0.30, **and** held-out realistic ESR 0.0375→0.0318 — with no real-time cost. The win is largest on the strongly-nonlinear BJT (where the formants are richest), holds on the JFET, and **improves the conditioned FiLM path too** (jfet-tone held-ESR 0.037→0.029).

**Overturned prior note (reconciled).** The CIRCE2-era record listed "pre-emphasis-ESR" as a net-negative. That verdict was measured on **broadband white-noise ESR** — where HF-weighting chases the *unfittable* rate-distortion floor (§2.1) and so looks harmful. On the **deployment-meaningful band-limited / realistic metric** it is net-positive, and on broadband it is now merely neutral (BJT 0.1635→0.1616). The lesson is the same one §2.1 already states: **evaluate on realistic signals, not white noise** — and once you do, the phase-aware spectral loss is a free, first-order win.

### ADDENDUM 2 — internal oversampling removes the dominant spectral error (self-aliasing)

After the loss fix, the remaining held-out error on realistic signals turned out to be **the model's own aliasing**, not a capacity or data limit. The gated TCN's pointwise tanh/sigmoid nonlinearities generate harmonics above Nyquist; at the base rate these **fold back into the audio band** and corrupt exactly the formant region the target has no energy-above-Nyquist to explain. Running the network at an **internal 2× rate** (upsample → TCN → downsample) pushes the alias-fold to 2× Nyquist, leaving the in-band spectrum clean.

The effect is large and was the single biggest fidelity lever found:

| | base rate (1×) | **2× oversampled** |
|---|---|---|
| held-out ESR, fp test | ~0.032 | **~0.004** |
| held-out ESR, guitar-DI | ~0.03 | **0.0012** |
| high-band (>4 kHz) ESR | 0.33 | **0.11** |
| harmonic-formant-peak error | 5.5 dB | **2–4 dB** |
| real-time factor (CPU, end-to-end) | 4.0× | **2.3× (ch24) / 3.2× (ch16)** |

That is an **~8–25× reduction in realistic held-out error** — far beyond anything the conditioner, capacity, receptive field, or grid density moved — at a still-real-time RTF. It is implemented streaming-exact: a stateful linear-phase polyphase resampler (`_OverSampler`, verified `process == process_block` to 1e-7, block-size invariant) with a small group delay reported as `latency_samples` (≈1.4 ms at 127 taps). Base-rate capacity/RF/edge-loss ablations were all **within seed noise** by comparison — confirming the error was aliasing, not approximation.

**Updated lever ranking:** (1) **internal oversampling** (removes self-aliasing — the dominant spectral error), (2) control-grid density, (3) **loss design — phase-aware pre-emphasis for the formants/transfer curve**, (4) conditioner choice. All beat adding capacity/compute. CIRCE3's default loss is `ESR + 1.0·preemph-ESR(α=0.95)`; the shipped/benchmarked config additionally uses `oversample=2` (the constructor default stays `oversample=1` for the latency-0 contract). The honest lesson, a fifth time: the wins come from **the right signal-processing (anti-aliasing) and input/data/loss decisions**, not from a bigger or fancier network.

### ADDENDUM 3 — gradient clipping is the dominant lever for HARD (discontinuous) circuits

A radical-architecture search (10 alternative backbones — Snake/KAN/complex/mixed-activation/attention/grey-box/boosting/FNO/ODE/spiking — each short-trained on the class-B crossover dead-zone) reported "5–9× breakthroughs" over CIRCE3. A controlled fair-baseline A/B dissolved that: a fairly-sized gated TCN already reached crossover ≈0.07 (not the 0.45–0.55 the search compared against), so the "breakthrough" was a weak-baseline artifact. Chasing the real cause through isolation ablations: **oversampling is NOT the hard-circuit lever** (OS1 vs OS2 within ~10% at fixed config), and **depth/width are NOT levers** (crossover floored ~0.30 across ch24/L9, ch40/L8, ch40/L18). The lever the search had stumbled on, wearing many disguises, was **gradient clipping** — every radical agent independently needed `clip_grad_norm(1.0)` to train on the discontinuity, and CIRCE3.fit lacked it.

A clean clip-on/off A/B at the production config (ch24/L9/OS2) is decisive: the dead-zone/fold corner emits large spiky gradients that, unclipped, knock Adam into a far-worse, high-variance minimum. Clipping at norm 1.0 collapses the mean **and** the seed variance:

| | no clip | **grad-clip 1.0** |
|---|---|---|
| crossover held-ESR (3-seed mean) | 0.27 | **0.033 (~8×)** |
| crossover seed spread | 0.16 – 0.39 | **0.031 – 0.035** |

Across the 9-circuit suite it helps 6/9 (crossover −88%, fullwave −35%, BJT −25%, TS −15%, hysteretic −14%, JFET −10%), neutral on 2, mild +11% on hard-clipper (seed noise). A clip-value sweep found **1.0 is the sweet spot** (0.5 ≈ 1.0; 2.0 worse). It is **uniform** (one setting, no per-circuit tailoring — the binding design constraint), **training-only** (zero inference cost, RTF/latency unchanged), and now the CIRCE3 default (`grad_clip=1.0`). Re-tested with grad-clip stabilising it, the **mixed heterogeneous-activation block** (`block_act="mixed"`) no longer regresses the wavefolder and mildly helps both smooth and hard — initially kept opt-in over an assumed inference cost, but a direct RTF measurement (ADDENDUM 4) overturned that and made it the **default**. **Refined lever ranking for HARD circuits:** (0) **gradient clipping**, then (1)–(4) as above for smooth/spectral fidelity. Remaining frontier: the **multi-fold wavefolder (≈0.18)**, whose folds alias even at 2× OS — a real-time-bandwidth limit a uniform model cannot raise per-circuit, not an optimization gap.

### ADDENDUM 4 — the mixed-activation block is the default (smaller, faster, ≥ the gate)

The earlier "mixed carries real inference cost → opt-in" call was wrong on the cost, and was only ever marginal on the accuracy. A decisive 9-circuit head-to-head (both `grad_clip=1.0`, 150 ep, ch24/L9/OS2; `mixed_vs_gated_final.py` → `outputs/mixed_vs_gated_final.json`) settled it: **mixed wins 4, ties 3, loses 2** — every win on a *hard* circuit (crossover −13%, wavefolder −8%, fullwave −7%, hard-clipper −4%), both losses mild and on the *smoothest* circuits (jfet +14%, tube-screamer +3%). Decisively, mixed is **also smaller and faster**: its dilated conv emits `channels` features instead of the gate's `2·channels`, so the shipping model is **54k params at RTF 2.45×** vs the gate's **85k at 2.23×** (the `ch→ch` conv outweighs the Snake `sin()` cost — I had assumed the opposite without measuring). Uniformly ≥ on accuracy where it matters, fewer parameters, faster, real-time, and streaming-exact → it is the **circuit-agnostic default**, satisfying the one-config-for-all-circuits constraint; `block_act="gated"` is retained for the two smooth circuits where the classic gate is marginally ahead. The shipped checkpoints (bjt/jfet/tube_screamer) and the model-card numbers were regenerated as mixed (`mixed_default_retrain.py`): bjt 0.0046, jfet 0.0093, tube_screamer 0.0131, crossover 0.0216, fullwave 0.0089, hysteretic 0.0274, asym 0.0523, hard-clipper 0.0916, wavefolder 0.1778. **Lesson (sixth time): measure before assuming a cost** — the "expensive activation" intuition was simply false here.

Sections 1–7 below are the (pre-experiment) derivation, retained as the reasoning record; the table above is the result.

---

## 1. EXECUTIVE ANSWER

The information-theoretically optimal real-time architecture for the family {F_c} is **a single well-trained dilated causal gated TCN** — the Boyd–Chua canonical realizer of a causal, time-invariant, fading-memory operator, which is provably near-optimal for the fixed-system part and which no surveyed alternative (RNN/LSTM, S4/SSM, neural-ODE, attention, Koopman/DeepONet/FNO) decisively beats — with controls handled by a **strict physical decomposition**: *signal-acting* controls (gain/drive/sustain/level/forcing-amplitude) are **folded into the input** `x ← g·x` and left unconditioned, which is a 0-bit algebraic identity that generalizes exactly by construction, while *system-acting* controls (a tone cap, a bias, the Duffing β) get the **smallest conditioner that fits a poly-logarithmically-small control manifold**. Honestly stated: a well-trained TCN with input-scaling *already wins*, the only unsolved residual is the system-control held-out gap (~0.01–0.02 audio, ~0.22 Duffing-β), and the bar any added mechanism must clear is **beating a strong conditioned TCN, not the project's known-loser FiLM**. The two conditioners that qualify on theory (each injects the control *multiplicatively into the dynamics*, which FiLM provably cannot) are **(i) concat-conditioning** — feed the normalized system control as an extra input channel, the strong industrial baseline (WaveNet global conditioning / NAM ParametricOD) that carries no separate hypernetwork to overfit — and **(ii) control-LoRA** — a rank-r (r=1–3) identity-anchored, zero-init low-rank modulation of the dilated conv kernels, baked once per control-change into the cached-conv kernel so the per-sample loop is byte-for-byte the plain TCN. The definitive recommendation is therefore **not to pick a winner on paper** (the failure mode that sank CIRCE/CIRCE2/CIRCE3-FiLM) but to run a three-arm A/B (concat vs FiLM vs control-LoRA) on the analytically-known Duffing-β testbed and **ship the simplest arm within noise of the best** — with the prior, on a 3-for-3 track record, that concat-conditioning ties control-LoRA on accuracy and that control-LoRA's only certain advantage is a cleaner, cheaper streaming path, not held-out accuracy.

---

## 2. MATHEMATICAL DERIVATION

### 2.1 The fixed system: why the TCN is near-optimal (Boyd–Chua + MDL)

At a fixed control setting, each F_c is a causal, time-invariant (TI), fading-memory (FM) operator: a stable analog circuit (RC time constants ⇒ exponential forgetting) and a damped-driven Duffing oscillator (asymptotically independent of initial condition) both satisfy the FM definition of Boyd & Chua (1985): *inputs close in the recent past, weighted by a decaying w(t), yield outputs close in the present.* The relevant theorems pin the optimal **structural form**:

- **Boyd–Chua Thm 2 (finite-dim realization):** any TI FM operator is uniformly approximable by `ẋ = Ax + bu, y = p(x)` with A exponentially stable and p a static polynomial — *a stable LTI filterbank followed by a memoryless nonlinear readout.*
- **Boyd–Chua Thm 4 (NLMA, the discrete form a 44.1 kHz model actually is):** the optimal discrete-time realization is `y(k) = f(x(k), x(k−1), …, x(k−M+1))` — **a static nonlinearity of the last M input samples.** This *is* the definition of a TCN.
- **Boyd–Chua Thm 5 (convolution):** the linear sub-block of that form is necessarily a convolution. This is exactly why **S4/SSM ties and never beats** — an LTI SSM *is* a convolution, re-basing the linear part only; useful for memory ≫ 23 ms, which guitar circuits do not have. (Confirmed: project GATE-0 found TCN > S4 on the BJT.)
- The modern operator-approximation theorem (Universal Approximation of I/O Maps by TCNs, arXiv:1906.09211, Thm 3.3) closes the loop: any causal, TI, approximately-finite-memory map — which by Park–Sandberg (1992) is exactly the FM class — is approximable by a ReLU TCN `f(u_{t−m},…,u_t)`. Three independent literatures (Volterra/Boyd–Chua; Sandberg myopic maps; the TCN theorem) converge on **one** canonical form.

**Why the TCN, not the literal Volterra/NLMA optimum (MDL).** A degree-P polynomial readout over memory M has O(M^P) coefficients (the documented curse of dimensionality of Volterra identification, Automatica 2021). At M ≈ 1000 samples (a 23 ms receptive field) this is infeasible — the literal Boyd–Chua optimum is the *wrong code*, spending exponentially many bits. A dilated, gated, weight-shared TCN encodes the same operator in

```
params ~ O(C² · k · log M)
```

because **depth buys polynomial order multiplicatively** (degree up to 2^D in D layers), **dilation buys memory logarithmically** (R = O(k^D)), and **time-weight-sharing sets the absolute-time description length to exactly 0 bits** — the unique inductive bias whose symmetry matches the operator's exact TI symmetry (a resistor does not know what time it is). The TCN is, in the precise MDL sense, the near-minimal code for this operator class. This is consistent with the measured ESR 0.001–0.002 on band-limited DI.

**The capacity floor is a data property.** The broadband white-noise ESR floor (~0.025) is identical at 76k/300k/1.2M params — a *capacity-invariant* floor is by definition not a model-capacity problem; it is the **rate–distortion floor of the excitation** (white-noise-through-a-nonlinearity is near-incompressible). Band-limited guitar DI is a lower-entropy source ⇒ a far lower achievable distortion ⇒ **the meaningful metric is band-limited DI; report white noise as a diagnostic only.** Corollary: the residual generalization gap is *not* fixable by adding capacity.

### 2.2 Information theory of the control family {F_c}

The family is an operator manifold indexed by c ∈ ℝ^p. The relevant quantity is the *incremental* complexity of {F_c} over a single F_{c0}, and physics splits the controls into two kinds with radically different bit content.

**Signal-acting controls carry 0 bits (and this is provable).** A pre-gain is an exact group action: `F_g(x) = F_1(g·x)`. This is an algebraic identity, not learnable data. A deterministic bijective reparameterization carries **0 Shannon bits** (the mutual information between g and the residual-operator-after-scaling is 0), so the rate–distortion-optimal code spends 0 capacity. Folding hard-codes the equivariance, so the generalization error in g is **identically zero, interpolation and extrapolation** — g is quotiented out of the hypothesis space. Any learned map g→(γ,β) can only **add** a strictly-nonnegative interpolation error to an identity it already had for free. This is the measured CIRCE result (FiLM-on-drive 0.093 vs g·x-unconditioned 0.046 — a 51% reduction that is the *removal of a self-inflicted error*), and the independent NAM ParametricOD result (ESR < 0.004 unseen, *cheaper* than per-snapshot models, by amortizing one net over the input axis). **Boundary, stated honestly:** folding is exact only for a clean series gain; a "drive" that also re-biases a stage has a residual *system-acting* component — fold the gain part, route the residual to the system conditioner.

**System-acting controls carry only poly-logarithmically-many bits.** For a p-parameter **smooth** (real-analytic in c — true for component sweeps and Duffing β) manifold, the Kolmogorov n-width decays exponentially, `d_n(M) ≤ C·e^{−β n^{1/p}}` (Cohen–DeVore, holomorphic dependence). Inverting, the effective number of modes is

```
n ~ |log ε|^p    →   a handful for p=1, tens for p=2–3,
```

and the description-length surcharge of {F_c} over a single F_{c0} is **poly-logarithmic in accuracy.** The decisive consequence: **the system conditioner should be small.** FiLM's failure is therefore *not* under-allocation — it is the wrong functional **form** for a small allocation.

### 2.3 The optimal conditioning form: feature-modulation vs parameter/kernel-modulation vs concat

This is the heart of the open problem, and the dynamics-vs-static-map argument decides it.

**A system control changes the operator in two ways.** *Dynamics-changing* controls (a reactive tone cap, a feedback C, the Duffing stiffness/β) move **poles/zeros** of the linear dynamics — which live in the **conv kernels** (the impulse responses), not in the activations. *Static-map-changing* controls (a bias) reshape the **pointwise nonlinearity** — well-modeled by an affine on the pre-activation.

**Feature-modulation (FiLM) is provably the wrong form for dynamics.** FiLM applies `h ← γ(c)⊙h + β(c)`, a per-channel affine on **activations** — a diagonal, memoryless, rank-1 post-scaling of *fixed* kernels. Its reachable operator perturbation is `span{diag-rescalings of pre-trained channel IRs}`, a measure-zero slice that **cannot synthesize a moved pole** (which requires redistributing energy across the kernel's time-lags, not just attenuating a fixed lag pattern). FiLM is well-matched to a *static-map* shift and mismatched to *dynamics* changes — the falsifiable prediction: its held-out gap is small for bias-type controls, large for tone-cap / Duffing-β. This is corroborated across labs: NablAFx finds plain FiLM the *weakest* conditioner on a fuzz; Hyper-RNN finds weight-generation halves FiLM's loss.

**Control theory says the control must enter multiplicatively.** A control-affine nonlinear system lifts to a **bilinear/LPV** Koopman form in which the control multiplies the dynamics (Iacob–Tóth–Schoukens 2024); a purely additive lift need not exist. The control belongs *in the dynamics, multiplicatively*, not as an additive post-scale of features.

**But — the critical correction — this argument excludes FiLM; it does *not* uniquely select control-LoRA.** Multiple conditioners inject c multiplicatively into the dynamics and all satisfy the bilinear requirement:

| Conditioner | Reaches pole motion? | Separate over-parameterized map to overfit? | Real-time fold |
|---|---|---|---|
| **FiLM** (activation affine) | **No** (measure-zero diag slice) | small MLP | per-sample multiply |
| **Concat** (c as input channel) | **Yes** — c enters the first conv, so c·x cross-terms after the first nonlinearity make the *effective* downstream kernels c-dependent; the net learns f(x,c) jointly | **No** — c is regularized by the same weight decay as the backbone | absorbed once into layer-1 pre-activation; **no kernel rebuild** |
| **Control-LoRA** (rank-r kernel delta) | **Yes** — delta is *on* the kernels, directly moves poles/zeros; strictly contains FiLM | yes — a φ-MLP/LoRA factors (the overfitting surface) | baked once per control-change into the cached kernel |
| Full hypernetwork | Yes | yes, severe (off-manifold weight interpolation) | heavy rebuild |

The anti-FiLM theory is valid and load-bearing; using it to conclude *control-LoRA specifically* would be smuggling. **Concat and control-LoRA both qualify; the choice between them is empirical, not theoretical.** Concat's structural advantages are real: it adds *no* separate map into a high-dimensional space, so it has no hypernetwork overfitting surface (the exact failure mode that made Fourier-feature lifts HURT interpolation), and c enters as *data* not weights, so there is *zero* per-block kernel rebuild even under a continuous knob sweep. Control-LoRA's genuine advantage is narrower: it is a verified *efficiency* win (it removes a per-sample activation multiply), and it directly raises the interpolation *order* in the correct (kernel) coordinates — but its "provably interpolates" claim holds only in the φ-coordinate, and φ(c)=MLP(c) can be as wiggly in c as a Fourier lift unless regularized (λ‖φ‖² locality + L1 on r). It is "smooth-in-c *if* regularized" — exactly the same caveat that applies to a weight-decayed concat-TCN, with no theorem separating them on interpolation.

---

## 3. ARCHITECTURE SPEC

### 3.1 Backbone (unchanged — endorsed by §2.1)

Gated dilated causal TCN (WaveNet-style):

- `H = 24` channels, `n_blocks = 2`, `n_layers = 8`, kernel `k = 3` ⇒ receptive field `R = n_blocks·(2^{n_layers} − 1)·(k − 1) + 1 = 1021 ≈ 23 ms`.
- Per layer: `z = tanh(W_f * h) ⊙ σ(W_g * h)` gated activation, residual + skip; final 1×1 mixdown; clamp saturator; fixed 1-pole 1 Hz DC-blocker (low corner — DC only).
- Receptive field sized `M ≈ τ·log(1/ε)·f_s` to cover the longest circuit time-constant.
- **No backbone replacement.** SSM ties (Thm 5); neural-ODE/FNO/DeepONet violate the real-time/bit-exact/streaming contract; attention learns for cost what the conv encodes for free. Optional **rational/Padé gate, OFF by default**, enabled only for hard-clip (diode/op-amp) circuits with broadband content (Boullé 2020: Ω(log 1/ε) params at a singular knee); default OFF because on band-limited DI the knee is rarely exercised (measured neutral-to-harmful: 0.049 vs 0.046, RTF 0.91×).

### 3.2 Signal controls (unchanged — proven optimal, §2.2)

Fold `x ← (Π_{i∈signal_idx} c_i)·x`; model unconditioned on signal controls. (`circe3.py`: training fold line 278, stream fold line 447.) Forcing amplitude in Duffing is a pre-gain ⇒ fold.

### 3.3 System controls — the two admissible forms (decided by §6's A/B)

**Form (i) — Concat (the strong baseline, recommended default).** Feed normalized system controls as extra input channels. Change the input projection from `Conv1d(1, C, 1)` to `Conv1d(1+n_system, C, 1)`; broadcast `c_sys` across time as constant channels.

- Forward: `h₀ = W_in · [x ; c_sys·𝟙]`. Because c_sys is constant per block, its contribution is a **constant offset into the layer-0 pre-activation** — absorbed once per block, **zero per-sample cost, no kernel rebuild ever.**
- Equation: `y(k) = f(x(k),…,x(k−M+1) ; c_sys)`, jointly learned; Boyd–Chua applies to the joint operator with c as a slowly-varying exogenous input.

**Form (ii) — Control-LoRA (the challenger).** Identity-anchored rank-r modulation of the dilated conv kernels:

```
θ(c) = θ₀ ⊙ (1 + Σ_{j=1}^{r} φ_j(c)·ΔΘ_j),    r = 1..3,    ΔΘ_j zero-init  (identity at init)
per-conv:  cw_eff = cw0 · (1 + B·diag(φ(c))·A),    φ(c) = Linear/Tanh MLP (or identity for a single monotone knob)
```

- For r=1 with a single monotone knob, use **φ linear/identity in c** (drop the MLP) — the cleanest, most defensible form (a genuine 1-D curve through operator space).
- Regularizers (load-bearing, not decoration): locality `λ‖φ(c)‖²` (CoDA, bounds held-out deviation) + L1 on φ to use the smallest r consistent with fit.

### 3.4 Streaming form (bit-exact, block-invariant, real-time) — verified

Controls are piecewise-constant per block ⇒ θ(c) constant within a block.

- **Concat:** c is an extra constant input channel; the layer-0 1×1 conv absorbs it once per block; the cached-conv kernels are *never* rebuilt; latency 0, bit-exact, block-invariant by inheritance.
- **Control-LoRA:** on **detected control change only**, `_bake_kernels(c)` forms `cw_eff` and writes it into the cached-conv layer tuples (`s["layers"]`); the per-sample inner loop is byte-identical to the plain TCN (no per-sample conditioning path remains — replaces the current per-sample FiLM line `h = γ·h + β`). Cost on change: `O(r·k·C²)` per layer.
  - **Continuous-sweep honesty (verified):** a knob sweep changes c *every* block, so the rebuild is paid per block for all system layers. At H=24, k=3, 16 layers, r=3: ~16·3·3·24·24 ≈ 2×10⁵ FLOPs/block vs the conv's ~3×10⁷ FLOPs/block ⇒ **<1% overhead** even in the adversarial case. State this; do not hand-wave "cheap if c changes rarely."
- **Regression gates (both forms):** `check_streaming ≤ 1e-4`, `check_streaming_moving ≤ 2e-3`, **ragged-block 37/128/200 ≤ 2e-3**, `latency_samples == 0`. Pass by construction (the streaming path is the proven cached conv after baking/absorption).

---

## 4. WHY EACH COMPONENT BEATS A PLAIN TCN (OR THE HONEST ADMISSION IT DOESN'T)

- **TCN backbone:** It *is* the plain TCN — the Boyd–Chua canonical realizer (§2.1). Nothing beats it for the fixed system; this is the foundation, not an improvement.
- **Input-scaling (signal controls):** **Beats a control-conditioned TCN unambiguously** — 0-bit exact equivariance vs a learned approximation that can only add error (measured 0.046 vs 0.093; NAM cheaper-than-snapshot). The single fully-earned win, and it predates this design.
- **Concat-conditioning (system controls):** **Expected to TIE the best conditioner and BEAT FiLM.** It reaches c-dependent dynamics (unlike FiLM), carries no hypernetwork overfitting surface, and is real-time/bit-exact for free. It does *not* clearly beat a control-LoRA on accuracy — that is the open question §6 settles.
- **Control-LoRA (system controls):** **Beats FiLM** (reaches pole motion; strictly contains FiLM) and is a **verified streaming-efficiency win** over FiLM (removes a per-sample multiply). **Whether it beats concat on held-out accuracy is unproven** — there is no theorem separating them on interpolation; the honest claim is "expected to tie concat, with a cleaner streaming path." On the project's 3-for-3 prior, the expected outcome is a tie.
- **FiLM:** **Does NOT beat a plain conditioned TCN for dynamics controls** (measure-zero reachable set; reproduced as the weakest conditioner across labs). Keep only as the zero-init fallback / identity anchor for pure static-map controls.
- **CUT components** (Fourier control features, Chebyshev head, per-sample/TFiLM modulator, full hypernetwork, SSM/Koopman/FNO/DeepONet backbones): each is predicted-harmful by theory (over-coding a low-bit control, raising control bandwidth, off-manifold weight interpolation, streaming-contract violation) and was measured neutral-to-harmful. Correctly euthanized.

---

## 5. GENERALIZATION TO UNSEEN CONTROLS

The held-out error decomposes into a capacity term and an interpolation term:

```
ESR(c_unseen)  ≲  ε_base(capacity)  +  ε_interp(grid spacing h, smoothness k)
ε_interp ≤ C·L·h        (Lipschitz c↦F_c, k=1 interpolant — e.g. FiLM)
ε_interp ≤ C·h^k        (C^k-smooth c↦F_c, degree-k interpolant — affine/low-rank LPV or weight-decayed concat)
```

with L = the output-metric bandwidth of `c↦F_c` (how fast the operator changes per unit knob) and, on a p-D grid of budget B simulations, `h ~ B^{−1/p}` ⇒ `ε_interp ~ B^{−k/p}`. **The curse lives in the control dimension p; it is defeated by interpolation order k, not by conditioner size.**

- The observed 0.01–0.02 held-out gap is consistent with a **k=1 (Lipschitz)** interpolant on a coarse grid.
- **Two levers, both architecture-secondary:** (a) shrink h — densify the grid (4× in 2-D) or **active-sample** (re-simulate only highest-ensemble-disagreement c; PANAMA: ~2.5× lower error at equal budget, hardest settings cluster at manifold *extremes*); (b) raise k — the LPV/low-rank (or weight-decayed concat) form in the correct coordinates.
- **Data implication:** because the white-noise floor proves the architecture is not the binding constraint, the highest-leverage fix for the system-control gap is **control-grid density and edge-biased active sampling**, not a fancier conditioner. Sample the manifold *corners* and densely where `c↦F_c` is most nonlinear.

---

## 6. THE DECISIVE EMPIRICAL EXPERIMENT

The fatal flaw to avoid (which sank every prior design) is **benchmarking against the wrong baseline.** All prior comparisons were "new mechanism vs FiLM" — but FiLM is the project's *known loser* (0.093). The mandated baseline is **a strong conditioned TCN (concat)**, which has never been in the verdict table. The experiment must include it.

**Three arms, identical backbone (H=24, 2×8, k=3), identical param budget, identical training (cosine LR, STFT-aux, same epochs, ≥3 seeds), identical Sobol control grid with held-out *interior* settings**, on the **Duffing β sweep** (cleanest: `c↦F_c` is known-analytic, so L is computable and k is verifiable; `simulate_duffing(beta=...)` already exists):

- **Arm A — CONCAT-TCN** (the true baseline, previously omitted): normalized β as a second input channel; forcing amplitude still input-scaled. (~10-line change: `Conv1d(1, C, 1) → Conv1d(1+n_system, C, 1)`, broadcast c_sys; streaming absorbs it once, zero per-sample cost.)
- **Arm B — FiLM-CIRCE3** (as-shipped, the known loser, included for continuity).
- **Arm C — control-LoRA-CIRCE3** (rank r=1–3 kernel modulation).

**Metric:** held-out-interior-β ESR on band-limited (resonant) Duffing response, mean ± std over seeds. Also run a **second control type** — one clearly *dynamics-changing* (β / tone-cap RC corner) and one clearly *static-map* (bias) — to test the falsifiable FiLM prediction (large gap on dynamics, small on static-map). Fit `ESR ≈ ε₀ + L·h^k` across 2–3 grid spacings to read off the interpolation order.

**Gates:**
- **Gate A (form vs the *right* baseline):** does C (or A) beat FiLM on the dynamics control on held-out interior c? (Predicted yes for both.)
- **Gate B (grid bound):** fit `ε₀ + L·h^k`; if k≈1, the grid is the bottleneck — raise k or active-sample.
- **Gate C (allocation):** conditioner capped at low-thousands of params; if held-out improves only by *growing* the conditioner past `|log ε|^p`, that is grid-overfitting — revert.
- **Gate D (streaming):** the four checks in §3.4.

**Decision rule — ship the simplest arm within noise of the best:**
- **C beats A by > 2σ on held-out β** → ADOPT control-LoRA (its streaming win is then free on top of an accuracy win).
- **A ≈ C (within noise)** → **SHIP CONCAT-TCN.** Simpler, no hypernetwork, no overfitting surface, dethrones FiLM for free; invest remaining effort in grid density / active sampling (§5).
- **A ≈ B ≈ C** → the family carries even fewer bits than thought; ship concat-TCN and stop touching the architecture.

**Prediction (skeptical, on the 3-for-3 prior):** A ≈ C, both beat B. Control-LoRA's real, defensible advantage is the cleaner/cheaper streaming path, **not** held-out accuracy over concat.

---

## 7. HONEST RISKS + FALLBACK

- **The mechanism may not beat concat (highest-probability risk).** There is no theorem separating control-LoRA from a weight-decayed concat-TCN on interpolation; both are "smooth-in-c if regularized." The project has rediscovered "the plain TCN wins" three times. **Fallback: ship the concat-conditioned TCN** — itself an upgrade over FiLM-CIRCE3, with zero new failure surface.
- **Control-LoRA's interpolation claim is over-soldable.** θ(c) is affine in φ(c), but φ(c)=MLP(c) need not be smooth in c — the same overfitting vector that sank Fourier features. **Mitigation:** drop the MLP for single monotone knobs (φ linear/identity); enforce λ‖φ‖² + L1 on r; validate smoothness empirically (Gate B), never assume it.
- **The residual gap may be a data problem, not an architecture problem.** The capacity-invariant white-noise floor and the §5 bound both point to grid density / active sampling as the dominant lever. If Gate B yields k≈1, **no conditioner change will help** — densify or edge-sample the control grid instead.
- **"Both signal-and-system" controls** (a re-biasing drive) are unproven and no in-repo circuit exercises them yet. Defer the fold-gain-route-residual refinement until a real such control exists.
- **The irreducible knee limit:** a finite-Lipschitz net cannot synthesize a true discontinuity (spectral-bias bound); the sharpest hard-clip knee always rounds, for *any* conditioner. This is a backbone-level limit, not a conditioning one.

**The fallback, stated plainly:** an unbeaten well-trained TCN wins by default. If the three-arm A/B does not show control-LoRA beating concat by > 2σ on held-out β, ship the **concat-conditioned TCN + input-scaling** (TCN backbone, signal controls folded, system controls as input channels), keep FiLM only as a static-map fallback, keep the rational gate retired-but-available, and spend the next effort on **control-manifold sampling density and edge-biased active sampling**, not on layers.