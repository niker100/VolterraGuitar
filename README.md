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
clean→hard-clip range and interpolates to held-out knob settings (see
`outputs/figs/bjt_circe_*.png`).

## Dev

```bash
uv run ruff format . && uv run ruff check --fix .
uv run ty check
uv run pytest
```

ngspice is installed as a shared library via PySpice
(`uv run python -m PySpice.Scripts.install` / `pyspice-post-installation
--install-ngspice-dll` on Windows).
