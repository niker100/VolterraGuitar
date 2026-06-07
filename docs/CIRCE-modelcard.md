# CIRCE — model card

**CIRCE** (Conditioned Interpolatable Real-time Circuit Emulator) is the project's
SOTA model: a small FiLM-conditioned dilated TCN that emulates a nonlinear analog
guitar circuit **in real time** while exposing the circuit's analog controls
(drive, tone, …) as live, interpolatable knobs.

See [`CIRCE-design.md`](CIRCE-design.md) for the (adversarially reviewed)
architecture and the milestone log; this card summarizes what ships and how it does.

## Architecture

- **Backbone:** dilated gated WaveNet-style TCN (the `_GatedLayer` of
  `models/tcn.py`); default **12 channels, 2 blocks, 7 layers** (~509-sample
  receptive field, ~11 ms), **22.7k params**.
- **Conditioning:** per-block **FiLM** (`h ← γ(c)·h + β(c)` before each gated
  layer) from a tiny zero-init conditioner (untrained == plain TCN). Controls are
  normalized; γ is bounded `1+tanh` (centred at 1) → smooth interpolation.
- **Output stage:** a fixed clamp saturator of last resort (`±A`, A≈1.2× trained
  peak) + a fixed **1-pole ~20 Hz DC-blocker** (the real stages are AC-coupled).
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
ESR + 0.1·multi-resolution STFT; Adam, 80 epochs; everything seeded.

## Results — BJT overdrive (drive knob), the flagship

Trained settings 5–160 mV (clean→hard-clip) + 3 held-out; from `vguitar circe
--circuit bjt`:

| metric | value |
|---|---|
| interpolation ESR | trained 0.095 · **held-out mean 0.145** · worst 0.206 · p95 0.198 |
| vs best unconditioned real-time model | TCN 0.236 (CIRCE is more accurate **and** adds a live knob) |
| streaming equivalence | constant 3.7e-6 · **moving-knob 2.0e-5** (turning it mid-stream is exact) |
| real-time factor | **2.2× constant, 2.2× moving-knob** (conditioning ~free) |
| idle (zero-input) | steady-state **≤ −140 dBFS** (DC-blocker; ~8 ms startup transient aside) |
| hot input (4× amplitude) | **saturates** at the ±8.82 V bound (finite, offline = streamed) |
| held-out real guitar-DI | ESR 0.06–0.10, 0.23 at the quietest 10 mV (was 0.30 before DI-mix) |

Figures: `outputs/figs/bjt_circe_*.png` (ESR-by-drive, THD/peak/interp-vs-distance,
knob harmonics & waveforms, training curve, drive×freq heatmap, held-out-DI
compares); A/B wavs in `outputs/audio/`.

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
