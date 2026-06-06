# Architecture

Three small contracts hold the whole project together. Everything else is
written against them, so Volterra, block-oriented, and neural approaches are
interchangeable and directly comparable.

## The spine

| Contract | File | What it is |
|---|---|---|
| `Circuit` | `circuits/base.py` | A raw ngspice **netlist** with a driven input source `Vin` (node `in`) and output node `out`. Plain text = readable & portable. |
| `Dataset` | `data.py` | Aligned `(x, y)` mono float32 signals at one sample rate; `.split()`, `.save()/.load()` (npz). |
| `Model` | `models/base.py` | `fit(train, val, cfg)`, offline `process(x)`, **streaming** `reset()` + `process_block(x)`, `save/load`, `num_params()`. |

The **streaming invariant**: `process(x)` and a sequence of `process_block`
calls over the same `x` must agree within `latency_samples`
(`check_streaming` enforces it; every model is tested). This is what lets the
live engine and the benchmark trust block-by-block inference.

Circuits and models **self-register** via decorators and are auto-discovered, so
adding one is just dropping a file in the package — no central edit.

## The pipeline

```
 ngspice/PySpice      resample            Dataset           N models            benchmark         live
  (Circuit.netlist)   sim_sr -> sr        x, y @ sr         Model.fit/process   ESR·THD·RTF       sounddevice
        │                  │                  │                   │                   │              │
  spice/runner.py    resample.py          data.py          models/*.py        benchmark/run.py  realtime/engine.py
```

* **One sample rate everywhere** (`vguitar.AUDIO_SR = 44100`) — the fix for v1's
  rate-mismatch bug. The raw SPICE output (irregular adaptive timestep) is put on
  a uniform `sim_sr` grid then **anti-alias-decimated** to `sr`, making the target
  band-limited and alias-free.
* **SPICE driving** (`spice/runner.py`): the input is fed sample-by-sample into an
  *external* ngspice source via `NgSpiceShared.get_vsrc_data`. One process-wide
  ngspice instance is reused (its cffi binding is a per-process singleton).
* **Excitation** (`signals.py`): broadband noise + Schroeder multisine +
  exponential swept-sine + guitar DI, swept across drive levels for amplitude
  coverage.
* **Anti-aliasing** (`nonlinear/adaa.py`): antiderivative antialiasing for static
  nonlinearities, used by Wiener-Hammerstein and the realtime oversampling path.

## Models (benchmark contestants)

`fir` (linear floor) · `volterra` (regularized) · `volterra_pc` (parallel
cascade) · `wh` (Wiener-Hammerstein + ADAA) · `tcn` (WaveNet) · `rnn` (LSTM/GRU)
· `ssm` (diagonal state-space). See [`RESEARCH.md`](../RESEARCH.md) for why each
is here.

## CLI

`vguitar list | gen | train | bench | live | selftest` — see the
[README](../README.md). `selftest` is the fast health check (every model's
streaming + save/load contract, plus an ngspice smoke test).
