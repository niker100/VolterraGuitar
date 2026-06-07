# CIRCE — model card

**CIRCE** (Conditioned Interpolatable Real-time Circuit Emulator) is the project's
SOTA model: a small FiLM-conditioned dilated TCN that emulates a nonlinear analog
guitar circuit **in real time** while exposing the circuit's analog controls
(drive, tone, …) as live, interpolatable knobs.

See [`CIRCE-design.md`](CIRCE-design.md) for the (adversarially reviewed)
architecture and the milestone log; this card summarizes what ships and how it does.

## Architecture

- **Backbone:** dilated gated WaveNet-style TCN (the `_GatedLayer` of
  `models/tcn.py`); default **24 channels, 2 blocks, 8 layers** (~23 ms receptive
  field), **~89k params** (raised from 12ch/7layers to match the circuits' high
  harmonics — see "High-frequency accuracy" below).
- **Conditioning:** per-block **FiLM** (`h ← γ(c)·h + β(c)` before each gated
  layer) from a tiny zero-init conditioner (untrained == plain TCN). Controls are
  normalized; γ is bounded `1+tanh` (centred at 1) → smooth interpolation.
- **Output stage:** a fixed clamp saturator of last resort (`±A`, A≈1.2× trained
  peak) + a fixed **1-pole ~5 Hz DC-blocker** (the real stages are AC-coupled;
  the low corner is transparent across the guitar band and settles sub-audibly).
  Optional **ADAA** hard-clip saturator (`saturator="adaa1"|"adaa2"`) for cleaner
  extreme-drive aliasing (inference-only; training keeps the differentiable clamp).
- **Streaming:** Fast-WaveNet cached incremental convolution + per-block FiLM, all
  numpy, `latency_samples = 0`. `process` (offline) and `process_block` (streamed)
  agree to ~2e-5 even under a moving knob.

## Training data

SPICE parametric **control sweeps** (`spice.runner.make_control_dataset` +
`spice.sampling.control_grid`): for each control setting the model input is the
*dry* excitation and the target is the circuit's response. **~⅓ of training
segments are real guitar-DI** windows (`DataConfig.di_mix=0.34`) run through the
circuit, so the model sees real playing, not only synthetic excitation. Loss =
ESR + 0.2·multi-resolution STFT (FFT windows 256/512/1024/2048; the 256 window
resolves clipping edges); Adam; everything seeded.

## High-frequency accuracy (harmonics & clipping transitions)

The distortion's *character* lives in the high-order harmonics and the sharp
clipping transitions, but plain ESR is energy-weighted and barely sees them, so an
early CIRCE matched overall ESR while missing the bright detail. Addressed by:

- **Capacity** (12→24 channels, +1 layer): the clean win. On the BJT it shrinks the
  harmonic-6 error from a 13–19 dB miss to ~8 dB and the harmonic-9 miss from ~25 dB
  to ~10 dB, lifting the whole high-harmonic stack — at ~3.7× RTF (still real-time).
- **More training epochs** (cheap on a GPU): the strong-nonlinearity high harmonics
  need the budget; high-harmonic level error drops ~10→6 dB by 300 epochs.
- **Metrics to measure it** (`metrics.py`): `harmonic_level_error` (full + high
  band), `pre_emph_esr`, `band_esr`, `knee_region_esr`, `slew_weighted_esr`,
  `transfer_critical_error`, `crest_factor_error_db` — so HF error is tracked, not
  hidden behind overall ESR. The validation report prints high-harmonic mean |err|.
- **Pre-emphasis ESR** (`CIRCE(pre_emph=)`, opt-in, OFF by default): the literature's
  go-to was a *net negative here* — it games the on-a-tone harmonic metric but hurts
  the clipping-knee match and roughly doubles broadband ESR. Internal
  oversampling/ADAA was rejected too (it *damps* the HF harmonics we want).

**Honest residual limits:** the single *sharpest* transfer-curve knee (a near-
instantaneous clipping corner) still rounds — a true step needs unbounded HF, which
a small smooth real-time TCN can't synthesize; and the *conditioned* model spreads
capacity across the whole 5–160 mV range, so at the extreme drive it's less accurate
than a single-operating-point specialist. Further gains (more capacity/epochs/data,
per-operating-point weighting) trade real-time headroom or training cost.

## Results — BJT overdrive (drive knob), the flagship

Trained settings 5–160 mV (clean→hard-clip) + 3 held-out; from `vguitar circe
--circuit bjt`:

| metric | value |
|---|---|
| interpolation ESR | trained 0.088 · **held-out mean 0.079** · worst 0.118 · p95 0.113 |
| streaming equivalence | constant 1.4e-6 · **moving-knob 1.9e-6** (turning it mid-stream is exact) |
| real-time factor | **~6.6× constant and moving-knob** (conditioning ~free; load-sensitive) |
| idle (zero-input) | steady-state **−102…−128 dBFS** (5 Hz DC-blocker) |
| hot input (4× amplitude) | **saturates** at the ±8.82 V bound (finite, offline = streamed) |
| held-out real guitar-DI | ESR 0.02–0.06 (0.24 at the quietest 5 mV; was 0.30 before DI-mix) |

Figures: `outputs/figs/bjt_circe_*.png` (ESR-by-drive, THD/peak/interp-vs-distance,
knob harmonics & waveforms, training curve, drive×freq heatmap, held-out-DI
compares); A/B wavs in `outputs/audio/`.

### vs the other methods (fixed-point shootout)

From `vguitar shootout` (every method on identical data/test at each circuit's
nominal operating point; ESR, lower = better):

| circuit | CIRCE | best baseline | classical best |
|---|---|---|---|
| diode (easy, memoryless) | 0.064 | tcn 0.012 / **volterra 0.009** | volterra 0.009 |
| **bjt** | **0.092** | tcn 0.099 | volterra 0.217 |
| **jfet** | **0.019** | tcn 0.051 | volterra 0.092 |
| tube_screamer | 0.061 | **tcn 0.060** | volterra 0.162 |

On the nonlinear circuits the feedforward neural models (CIRCE, tcn) beat classical
Volterra/WH by 2–8×. A well-trained unconditioned TCN matches CIRCE's *fixed-point*
accuracy — but CIRCE delivers it at 22.7k params, real-time, **and** with the live
interpolatable knob the baselines structurally cannot provide (the interpolation
table above is CIRCE's edge). On the easy near-memoryless diode, classical Volterra
rightly wins. (Training auto-uses the GPU; inference/RTF stay on CPU.)

## Circuit roster

| circuit | controls | nonlinearity |
|---|---|---|
| `diode` | drive | symmetric diode soft-clip |
| `bjt` | drive | 2N3904 common-emitter (exponential) |
| `jfet` | drive, tone | 2N5457 common-source (square-law, asymmetric) |
| `tube_screamer` | drive, tone | op-amp + anti-parallel feedback diodes (in-loop) |
| `big_muff` | sustain, tone, level | two cascaded op-amp clippers + tone stack |

`drive`/`sustain` are input pre-gains (giving amplitude coverage); `tone`/`level`
are real component changes via a parameterized netlist (`Circuit.netlist_for`).
Per-circuit and cross-circuit results: `vguitar validate --circuits …`.

## Limitations (honest)

- **Hard-clip THD shortfall.** At the most extreme drive the clamp saturator
  under-produces harmonics (BJT 160 mV: CIRCE THD 0.28 vs circuit 0.66). The ADAA
  saturator reduces aliasing but does not fully close the gap; a teacher-student
  alias-free fine-tune (GATE-5) is the planned remedy.
- **DI-vs-synthetic trade-off.** `di_mix` improves real-DI generalization at a
  small cost to synthetic held-out ESR (it is a tunable `DataConfig` knob).
- **Nonlinear feedback is out of scope.** A feedforward cascade provably can't
  represent bias-shifting / NFB loops; the benchmark *measures* the gap, the fix is
  deferred.
- **High-dimensional control is untested at scale.** ≤3 axes are validated
  (factorial / Sobol grids); >3 axes need active-learning data generation (GATE-6).

## Use

```python
from vguitar.models.circe import CIRCE
m = CIRCE.load("assets/checkpoints/bjt.circe.model")   # shipped checkpoint
y = m.process(x, c=[0.08])            # emulate at a drive setting (incl. unseen)
y = m.process_block(block, c=[g])     # real-time; g changeable per block
```

CLI: `vguitar circe --circuit <c>` (validate), `vguitar validate --circuits …`
(compare), `vguitar live --circuit <c> --model circe --control drive=0.08,tone=0.6`
(play). Checkpoints load from `runs/` or `assets/checkpoints/` if present.
