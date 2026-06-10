# vguitar

A **physical analog-circuit emulator** for guitar audio: build a realistic
circuit out of real electrical components (R, C, diodes, BJTs, JFETs, op-amps,
pots) in Python, compile it into a real-time solver, validate it against SPICE,
and play guitar through it live — with audio indistinguishable from the actual
circuit, and the whole build-to-playable pipeline in seconds.

**How it works (the new core, in progress):**

```
component API  ->  DK compiler          ->  real-time emulator   ->  validation gate
(CircuitGraph)     (MNA -> state-space      (per-sample Newton /     (ngspice reference vs
                    + DC operating point)    LUT, numba kernels)      emulator: null depth,
                                                                      harmonics, ESR)
```

- **No training.** The emulator is the discretized physics of the circuit
  (DK method: trapezoidal nodal state-space + implicit nonlinear solve), so
  "fitting" is compile + optional LUT bake — well under a minute, no GPU.
- **ngspice is the validator, not the trainer**: a short reference simulation is
  compared against the emulator through an indistinguishability gate
  (A-weighted null depth, worst-case ESR over the control grid, harmonic match)
  with rendered listening WAVs.
- **Realistic circuits only**: components with physical models, no behavioral
  B-sources, no infinite dv/dt.

> **The neural benchmarking era (2025–2026) is frozen.** Years of Volterra /
> block-oriented / neural benchmarking (CIRCE3, a TCN, reached held-ESR < 0.005
> on 4/9 circuits with one uniform config) are consolidated in
> [`outputs/knowledge/findings.json`](outputs/knowledge/findings.json) — the
> machine-readable record of every experiment, verdict, and number — with
> summary plots beside it. The full decision log lives in
> [`docs/archive/`](docs/archive/), the code under
> `src/vguitar/models/archive/` (`uv sync --extra neural` to run it), and the
> archived experiments under `experiments/archive/`. See
> [`RESEARCH.md`](RESEARCH.md) for the literature review.

## Quickstart (current state)

```bash
uv sync                                # create .venv, install deps (no torch)
uv run vguitar list                    # registered circuits
uv run vguitar gen --circuit bjt       # simulate a dataset via ngspice
uv run vguitar plots --circuit bjt     # circuit diagnostics (transfer, harmonics)
uv run vguitar selftest                # smoke-test models + every circuit netlist
```

The DK-emulator commands (`build` / `validate` / `live`) arrive with the new
core — see the phase plan in the repo history.

## Circuits

Realistic-component netlists (diode clipper, BJT/JFET stages, Tube Screamer,
Big Muff, class-B crossover, …) registered in `vguitar.circuits`; each is a raw
SPICE netlist today and becomes a `CircuitGraph` build as the component API
lands. Controls (drive, tone, …) are declared per circuit (`ControlSpec`) and
map to input pre-gain or parameterized component values.

## Dev

```bash
uv run ruff format . && uv run ruff check --fix .
uv run ty check
uv run pytest               # active suite (torch-free)
uv sync --extra neural && uv run pytest tests/archive   # frozen neural-era tests
```

ngspice is installed as a shared library via PySpice
(`uv run pyspice-post-installation --install-ngspice-dll` on Windows).

**Troubleshooting ngspice.** If a circuit command errors with "ngspice shared
library is not available", run `uv run pyspice-post-installation
--force-install-ngspice-dll`. On Windows only `ngspice.dll` ships, so the runner
pins `ngspice_id=0` (one process-wide instance). Everything that needs ngspice
(`gen`, `plots` probes, `selftest`'s circuit rows) degrades gracefully or SKIPs
when it is absent.
