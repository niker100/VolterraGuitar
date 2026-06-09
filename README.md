# vguitar

Real-time digital emulation of **nonlinear analog guitar circuits**, trained on
SPICE simulations. Build a circuit out of real analog components, simulate it,
learn a fast model of its behavior, and play guitar through it live.

The project is a **benchmark of approaches** — Volterra series, block-oriented
(Wiener / Hammerstein / parallel-cascade), and modern neural models (TCN /
WaveNet, RNN, state-space) — measured on the same circuits with the same
metrics, so we can pick (or invent) the best accuracy-vs-latency tradeoff.

> This is a from-scratch rebuild of an earlier prototype (archived in
> [`legacy/`](legacy/)). See [`RESEARCH.md`](RESEARCH.md) for the literature
> review and design rationale, and [`docs/`](docs/) for the architecture.

## Pipeline

```
ngspice/PySpice  ->  resample  ->  Dataset  ->  train N models  ->  benchmark  ->  live app
 (Circuit)          (sim_sr->sr)   (x, y @sr)   (Model.fit)        (ESR/THD/RTF)  (sounddevice)
```

Three small contracts hold it together (`vguitar.circuits.base.Circuit`,
`vguitar.data.Dataset`, `vguitar.models.base.Model`) so every approach is
interchangeable.

## Quickstart

```bash
uv sync                              # create .venv, install deps
uv run vguitar gen   --circuit bjt   # simulate + build dataset
uv run vguitar train --model tcn     # train one model
uv run vguitar bench --circuit bjt   # train + compare all models
uv run vguitar live  --model tcn     # play guitar through it
```

## CIRCE3 — the SOTA model

**CIRCE3** (`vguitar.models.circe3`) is the project's SOTA model, derived from
approximation + information theory and confirmed by experiment (see
[`docs/optimal-architecture.md`](docs/optimal-architecture.md)). It is a
well-trained dilated **TCN** — the Boyd–Chua canonical realizer of a causal,
time-invariant, fading-memory operator, unbeaten by any other backbone — with a
**heterogeneous mixed-activation block** by default (tanh/gelu/relu/abs/Snake
units per layer, so corners are synthesizable where sharp circuits need them;
`block_act="gated"` keeps the classic WaveNet gate) and exogenous controls handled
by their **physical kind**:

- **signal-acting** controls (drive / gain / sustain / level) are folded **directly
  into the input** as a gain, so the model reproduces *any* setting exactly by
  construction (interpolation *and* extrapolation), with zero conditioning
  parameters and no interpolation error;
- **system-acting** controls (a tone cap, a bias) drive a **minimal per-block
  FiLM** — the conditioner that, in the A/B, matched or beat every heavier scheme
  (concat, rational gate, Chebyshev head, Fourier features, hypernetwork), all of
  which were retired.

Two signal-processing decisions then nail the **spectral fidelity** (the harmonic
formants and the static transfer curve), each a bigger lever than network size:

- a **phase-aware pre-emphasis-ESR** training loss lifts the low-energy formant
  band into the gradient without the phase-blindness of a magnitude-STFT term, so
  it sharpens the formants *and* the transfer curve at once;
- **internal 2× oversampling** (`oversample=2`) removes the per-layer
  nonlinearities' self-aliasing — the dominant residual error — dropping realistic held-out ESR
  **~8–25× (to ≈0.001 on guitar-DI)**, streaming-exact and still real-time.

On **realistic (band-limited) signals** the shipped oversampled model reaches
**ESR ≈ 0.001–0.004 real-time on every tested control kind** — signal (BJT drive),
static-map (Duffing β), dynamics (JFET tone). The dominant levers for held-out
accuracy are **anti-aliasing, loss design, and control-grid density** — not the
conditioner or more layers. Full results in
[`docs/CIRCE3-modelcard.md`](docs/CIRCE3-modelcard.md).

```python
from vguitar.circuits import get_circuit
from vguitar.spice.runner import make_drive_dataset
from vguitar.models.circe3 import CIRCE3

ds = make_drive_dataset(get_circuit("bjt"), [0.005, 0.02, 0.04, 0.08, 0.16])  # sweep a "drive" knob
m = CIRCE3(n_control=1, signal_idx=(0,)); m.fit(*ds.split()[:2])
y = m.process(x, c=[0.06])          # any drive (incl. unseen) — exact by input-scaling
y = m.process_block(block, c=[g])   # real-time, knob g changeable per block
```

### Circuits (increasing complexity)

| name | controls | nonlinearity |
|---|---|---|
| `diode` | drive | symmetric diode soft-clip |
| `bjt` | drive | 2N3904 common-emitter (exponential) |
| `jfet` | drive, tone | 2N5457 common-source (square-law, asymmetric) |
| `tube_screamer` | drive, tone | op-amp + anti-parallel feedback diodes (in-loop) |
| `big_muff` | sustain, tone, level | two cascaded op-amp clippers + tone stack |

`drive` is an input pre-gain (and gives amplitude coverage); `tone`/`level` are
real component changes rendered through a **parameterized netlist**
(`Circuit.netlist_for`, `ControlSpec`). Generate a multi-control dataset with
`spice.runner.make_control_dataset` + `spice.sampling.control_grid` (full-factorial
for ≤2 knobs, Sobol for ≥3).

### Benchmark — CIRCE3 vs the other architectures (the clean head-to-head)

```bash
uv run vguitar benchmark --circuits bjt,jfet     # + figures + CSV/JSON
```

`benchmark` is the project's single, insightful comparison. Per circuit it trains
CIRCE3 and every other deployable architecture (tcn, rnn, wiener-hammerstein,
volterra) on identical data and shows, by eye and by number, the two things that
make CIRCE3 the right choice:

1. **Accurate *and* real-time** — the **ESR-vs-RTF** plane (`compare_<circuit>_esr_rtf.png`)
   puts CIRCE3 in the good corner, beating the recurrent / state-space / classical
   models and matching the strong TCN backbone it builds on.
2. **Generalizes across the control knob** — `compare_<circuit>_generalization.png`:
   one CIRCE3 stays accurate at *every* drive (seen and unseen) by input-scaling,
   while a plain TCN trained at one operating point degrades as the knob moves — the
   capability the unconditioned architectures structurally lack.

3. **Nails the spectral properties** — a suite of fidelity overlays vs the real
   circuit: `transfer` (output-vs-input curve + residual panel), `harmonics`
   (stack + per-harmonic error), `spectrum` (the Welch **formant envelope** +
   error, log-freq), `spectrogram` (circuit / CIRCE3 / dB-difference), and
   `transfer_family` (the Transferkennlinie across drives). The oversampling win is
   shown directly in `circe3_oversample_aliasing.png`
   (`uv run python make_oversample_figure.py`).

Plus a cross-circuit ESR matrix (`compare_esr_matrix.png`) +
`outputs/benchmark.{csv,json}`.

### Play it — turn the knobs live

```bash
# offline render at fixed knob settings
uv run vguitar live --circuit tube_screamer --model circe3 \
    --in di.wav --out wet.wav --control drive=0.2,tone=0.6
# knob automation (JSON breakpoints, linearly interpolated over time)
uv run vguitar live --circuit bjt --model circe3 --in di.wav --out wet.wav \
    --automation knobs.json     # [{"t":0,"drive":0.02},{"t":3,"drive":0.16}]
# real-time (needs an audio device): omit --in/--out, set the knobs with --control
uv run vguitar live --circuit bjt --model circe3 --control drive=0.08
```

Trained checkpoints are loaded from `runs/` or, if absent, from
`assets/checkpoints/<circuit>.circe3.model` (written by `vguitar benchmark`), so
the live commands run without retraining.

## Reproducibility

All randomness is seeded (`TrainConfig.seed`, `DataConfig.seed`, per-segment
offsets in dataset generation). Datasets and models are cached under `data/` and
`runs/`; pass `--regen` to re-simulate or `--retrain` to refit. A single audio
rate (`vguitar.AUDIO_SR`) is shared end-to-end (simulate → dataset → train → live)
— the mismatch that broke v1.

**GPU.** Neural training auto-uses CUDA when available (`models.base.pick_device`),
then moves the model to the CPU for inference so streaming/RTF reflect the CPU
deployment target (`to_inference_cpu`). On Windows the CUDA torch wheel is pulled
via the `[tool.uv]` `pytorch-cu126` index; benchmark *numbers* are device-
independent (a CIRCE fit is ~16× faster on an RTX 4090, but the results are the
same). ngspice data generation is CPU-only (no GPU path).

## Dev

```bash
uv run ruff format . && uv run ruff check --fix .
uv run ty check
uv run pytest
```

ngspice is installed as a shared library via PySpice
(`uv run python -m PySpice.Scripts.install` / `pyspice-post-installation
--install-ngspice-dll` on Windows).

**Troubleshooting ngspice.** If a circuit command errors with "ngspice shared
library is not available", run `uv run pyspice-post-installation
--force-install-ngspice-dll`. On Windows only `ngspice.dll` ships, so the runner
pins `ngspice_id=0` (one process-wide instance — the sim cannot be parallelized
in-process). Everything that needs ngspice (`gen`, `bench`, `benchmark`,
`selftest`'s circuit rows) degrades gracefully or SKIPs when it is absent; the
model/plotting/streaming tests run without it.
