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

## Dev

```bash
uv run ruff format . && uv run ruff check --fix .
uv run ty check
uv run pytest
```

ngspice is installed as a shared library via PySpice
(`uv run python -m PySpice.Scripts.install` / `pyspice-post-installation
--install-ngspice-dll` on Windows).
