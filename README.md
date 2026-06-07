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

## CIRCE — the conditioned, interactive model

**CIRCE** (`vguitar.models.circe`) is the project's SOTA model: a small dilated
TCN with a real-time cached-streaming kernel, conditioned by **per-block FiLM**
so analog controls (potentiometers, switches, slow drift) are first-class — you
can turn the knobs live. It's trained on SPICE **parametric sweeps** of the
control, and **interpolates to control settings never simulated**. See
[`docs/CIRCE-design.md`](docs/CIRCE-design.md) for the (adversarially reviewed)
architecture.

```python
from vguitar.circuits import get_circuit
from vguitar.spice.runner import make_drive_dataset
from vguitar.models.circe import CIRCE

ds = make_drive_dataset(get_circuit("bjt"), [0.005, 0.02, 0.04, 0.08, 0.16])  # sweep a "drive" knob
m = CIRCE(n_control=1); m.fit(*ds.split()[:2])
y = m.process(x, c=[0.06])          # emulate at a drive setting (incl. unseen ones)
y = m.process_block(block, c=[g])   # real-time, knob g changeable per block
```

On a BJT overdrive drive-sweep this reaches ESR ~0.08 across the whole
clean→hard-clip range and interpolates to held-out knob settings. See
[`docs/CIRCE-modelcard.md`](docs/CIRCE-modelcard.md) for the full results,
controls, and honest limitations.

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

### Validate one circuit, or compare across all of them

```bash
uv run vguitar circe --circuit tube_screamer        # train + validate (per-axis + 2-D figures)
uv run vguitar validate --circuits bjt,diode,jfet,tube_screamer,big_muff
```

`circe` writes a report (per-setting ESR/THD, interpolation worst-case/p95,
moving-knob streaming error, real-time factor, stability) plus house-style figures
to `outputs/figs/` and A/B wavs to `outputs/audio/`. `validate` aggregates CIRCE
across circuits into a cross-circuit summary and a circuit×model ESR matrix (add
`--models circe,tcn,volterra` to include unconditioned baselines).

### Play it — turn the knobs live

```bash
# offline render at fixed knob settings
uv run vguitar live --circuit tube_screamer --model circe \
    --in di.wav --out wet.wav --control drive=0.2,tone=0.6
# knob automation (JSON breakpoints, linearly interpolated over time)
uv run vguitar live --circuit bjt --model circe --in di.wav --out wet.wav \
    --automation knobs.json     # [{"t":0,"drive":0.02},{"t":3,"drive":0.16}]
# real-time (needs an audio device): omit --in/--out, set the knobs with --control
uv run vguitar live --circuit bjt --model circe --control drive=0.08
```

Trained checkpoints are loaded from `runs/` or, if absent, from
`assets/checkpoints/<circuit>.circe.model` (the repo ships `bjt`), so the commands
above run without retraining.

## Reproducibility

All randomness is seeded (`TrainConfig.seed`, `DataConfig.seed`, per-segment
offsets in dataset generation). Datasets and models are cached under `data/` and
`runs/`; pass `--regen` to re-simulate or `--retrain` to refit. A single audio
rate (`vguitar.AUDIO_SR`) is shared end-to-end (simulate → dataset → train → live)
— the mismatch that broke v1.

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
in-process). Everything that needs ngspice (`gen`, `bench`, `circe`, `validate`,
`selftest`'s circuit rows) degrades gracefully or SKIPs when it is absent; the
model/plotting/streaming tests run without it.
