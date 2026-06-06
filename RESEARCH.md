# Research notes & design rationale

A condensed, cited summary of the literature review that shaped this rebuild.
Full verification: 30 sources fetched, 25 claims adversarially checked (22
confirmed). Dates flag where the field has moved.

## Why the v1 prototype failed

| v1 symptom | Root cause |
|---|---|
| "Volterra never worked" | Kernels were "extracted" from a tanh MLP's raw weights — mathematically invalid (ignores the activation's Taylor coefficients). Also, **unregularized** Volterra least-squares is ill-conditioned and high-variance (curse of dimensionality: parameters grow as `Mᵈ`). |
| "NN blew up for nonlinear circuits" | (a) per-term `max`-normalization destroyed the gain structure; (b) **aliasing** — a nonlinearity creates harmonics above Nyquist that fold back as inharmonic garbage at 44.1 kHz; (c) **sample-rate mismatch** (Micro-Cap 11025 Hz export vs ~400 kHz sim step vs 44.1 kHz inference). |
| Stepped sine sweeps only | Insufficient amplitude/spectral coverage → models diverge outside the trained range. |

## Model landscape (the broad survey)

* **Neural is the accuracy/latency winner.** In a 2025 head-to-head of LSTM, TCN,
  GCN and S4 on nonlinear audio effects, **structured state-space (S4) is best
  overall**, with TCN/GCN close; but there is a hard real-time tradeoff — LSTMs
  hit real-time at any block size, while the best large S4 missed `RTF > 1`
  (Comunità, Steinmetz, Reiss, *Frontiers in Signal Processing* 2025,
  [arXiv:2502.14405](https://arxiv.org/abs/2502.14405)).
* **WaveNet/TCN is directly validated for this exact pipeline**: a feedforward
  WaveNet emulating a tube amp trained on **SPICE-simulated data**
  (Damskägg, Juvela, Välimäki, ICASSP 2019, [arXiv:1811.00334](https://arxiv.org/abs/1811.00334)).
  Reinforced by Wright et al. real-time RNN amp models (*Applied Sciences* 2020)
  and the GuitarML ecosystem.
* **Block-oriented (Wiener-Hammerstein)** is accurate for *mild* distortion
  (ESR < 0.1) but degrades badly on heavy clipping — ESR > 1.0 on a distorted
  Marshall (Eichas/Möller/Zölzer, DAFx-17). So for hard overdrive, favor neural
  or richer (parallel-cascade) models.

## Volterra, done right (the "for fun" track, revived)

* **Regularized / RKHS kernel estimation** beats the curse of dimensionality by
  treating kernels as Gaussian-process draws with smoothness/decay priors —
  accurate from limited data (Birpoutsoukis et al., *Automatica* 2017; Dalla
  Libera, Carli, Pillonetto, *Automatica* 2021 — MPK / SED-MPK kernels).
* **Parallel-cascade identification** (Korenberg 1991): any finite-memory,
  finite-order Volterra system equals a finite sum of parallel LN cascades
  (linear filter → static polynomial), identifiable from a single I/O record —
  Volterra-equivalent but cheap to run (1-D convolutions).
* **Exponential swept-sine (ESS)** identification yields per-order impulse
  responses separated in time by `Δtₘ = L·ln m`, convertible to diagonal Volterra
  kernels (Farina 2000; Novak, *JAES* 2015) — one measurement gives data *and* a
  Volterra model.

> Open question the literature does **not** settle: whether order-2/3 Volterra
> can hit a live 44.1 kHz budget in Python. That's exactly what this repo's
> benchmark measures (RTF column).

## Aliasing

A real threat for nonlinear real-time at 44.1 kHz. **Antiderivative
antialiasing (ADAA)** for static nonlinearities gives suppression equivalent to
much higher oversampling: ~2–4× ADAA ≈ 6–12× plain (Bilbao, Esqueda, Parker,
Välimäki, *IEEE SPL* 2017; Albertini et al., DAFx-20). Implemented in
`vguitar.nonlinear.adaa`; used by the Wiener-Hammerstein static block and the
realtime oversampling path. (ADAA's *cost* advantage was not robustly confirmed —
measure, don't assume.)

## How this informs the build

* Single canonical sample rate end-to-end (fixes the v1 rate mismatch).
* Amplitude-rich, broadband excitation (noise + multisine + ESS + guitar DI
  across drive levels) — `vguitar.signals`.
* Anti-aliased decimation of SPICE output makes the *target* band-limited.
* A uniform `Model` contract so Volterra, block-oriented, and neural models are
  benchmarked identically on ESR / THD / RTF — and so we can later try our own
  hybrid (e.g. swept-sine-identified linear filters + a small neural or
  regularized-Volterra nonlinear core).
