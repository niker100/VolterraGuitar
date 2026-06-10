"""Driven Duffing oscillator — a NON-AUDIO nonlinear system with exogenous inputs.

The Duffing oscillator is the canonical nonlinear dynamical system in physics and
control theory. We use a cubic-hardening-spring form driven by an exogenous
forcing ``u(t)``:

.. math::

    \\ddot{x} + \\delta\\,\\dot{x} + \\omega_0^2\\,x + \\omega_0^2\\,\\beta\\,x^3
        = \\omega_0^2\\,A\\,u(t)

with resonance ``\\omega_0 = 2\\pi f_0 / sr``, damping ``\\delta = 2\\zeta\\omega_0``,
cubic-nonlinearity strength ``\\beta``, and forcing amplitude ``A``. The
low-frequency gain is ~unity (``x + \\beta x^3 \\approx A u``), so ``x`` is the
nonlinear response of a resonant system to ``A u`` — a clean fading-memory
nonlinear-with-exogenous-input testbed.

It maps onto the project's control taxonomy exactly like an audio circuit:

* **forcing amplitude ``A``** is **signal-acting** — it scales the input, so a
  model fed ``A·u`` and left unconditioned on ``A`` reproduces any amplitude
  exactly (input-scaling). This is the direct analogue of a guitar *drive* knob.
* **nonlinearity strength ``\\beta``** is **system-acting** — it changes the
  system's dynamics, not the signal, so it must be conditioned on (minimal FiLM).
  This is the analogue of a *tone/topology* control.

So :class:`vguitar.models.archive.circe3.CIRCE3` emulates the Duffing oscillator by the
identical mechanism it uses for circuits, demonstrating domain-generality.

Integration is fixed-step RK4 (the recurrence is sequential, like the audio
streaming kernels). The cubic's harmonics of a resonant ``f0`` stay well below
Nyquist and are attenuated by the resonance, so base-rate integration is alias-
safe; ``oversample`` is available for stiffer settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vguitar.circuits.base import ControlSpec
    from vguitar.data import Dataset


def simulate_duffing(
    forcing: np.ndarray,
    sr: int,
    *,
    f0: float = 200.0,
    zeta: float = 0.3,
    beta: float = 1.0,
    oversample: int = 1,
) -> np.ndarray:
    """Integrate the driven Duffing oscillator for an exogenous ``forcing`` (= ``A·u``).

    Args:
        forcing: the (already amplitude-scaled) exogenous forcing ``A·u``, ``(N,)``.
        sr: sample rate of ``forcing`` and the returned response.
        f0: resonance frequency in Hz.
        zeta: damping ratio.
        beta: cubic-nonlinearity strength (the system-acting control).
        oversample: integer integration oversampling factor (1 = base rate).

    Returns:
        The displacement response ``x(t)``, float32, exactly ``len(forcing)``.
    """
    f = np.ascontiguousarray(forcing, dtype=np.float64).reshape(-1)
    n = f.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    sim_sr = max(1, int(oversample)) * sr
    if oversample > 1:
        import soxr

        fs = np.ascontiguousarray(soxr.resample(f, sr, sim_sr), dtype=np.float64)
    else:
        fs = f
    m = fs.shape[0]

    w0 = 2.0 * np.pi * f0 / sim_sr  # rad per integration step
    delta = 2.0 * zeta * w0
    w2 = w0 * w0
    w2b = w2 * beta

    def dv(xx: float, vv: float, fr: float) -> float:
        return w2 * fr - delta * vv - w2 * xx - w2b * xx * xx * xx

    x = 0.0
    v = 0.0
    out = np.empty(m, dtype=np.float64)
    for i in range(m):
        fr = fs[i]
        frn = fs[i + 1] if i + 1 < m else fs[i]
        frm = 0.5 * (fr + frn)  # midpoint forcing for RK4
        k1x, k1v = v, dv(x, v, fr)
        k2x, k2v = v + 0.5 * k1v, dv(x + 0.5 * k1x, v + 0.5 * k1v, frm)
        k3x, k3v = v + 0.5 * k2v, dv(x + 0.5 * k2x, v + 0.5 * k2v, frm)
        k4x, k4v = v + k3v, dv(x + k3x, v + k3v, frn)
        x += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        v += (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        out[i] = x

    if oversample > 1:
        import soxr

        y = np.ascontiguousarray(soxr.resample(out, sim_sr, sr), dtype=np.float64)
    else:
        y = out
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    return np.ascontiguousarray(y[:n], dtype=np.float32)


def make_duffing_dataset(
    grid: np.ndarray,
    control_specs: list[ControlSpec],
    *,
    sr: int = 44_100,
    seg_dur_s: float = 2.0,
    seed: int = 0,
    f0: float = 200.0,
    zeta: float = 0.3,
    oversample: int = 1,
) -> Dataset:
    """Generate a conditioned :class:`Dataset` over Duffing control settings.

    ``grid`` rows are ``(amplitude, beta)`` settings; ``control_specs`` mark which
    column is the signal-acting amplitude (``mode="pregain"``) and which is the
    system-acting ``beta`` (``mode="netlist"``). The model input is the **dry**
    unit-amplitude forcing ``u``; the target is the Duffing response to ``A·u`` at
    that ``beta`` — exactly mirroring :func:`vguitar.spice.runner.make_control_dataset`
    for circuits, so the same CIRCE3 trains on it unchanged.
    """
    from dataclasses import replace

    from vguitar.config import DataConfig
    from vguitar.data import Dataset
    from vguitar.signals import build_training_excitation

    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim == 1:
        grid = grid.reshape(-1, 1)
    pregain_idx = [i for i, s in enumerate(control_specs) if s.mode == "pregain"]
    beta_idx = [i for i, s in enumerate(control_specs) if s.mode != "pregain"]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    bounds: list[int] = [0]
    values: list[list[float]] = []
    for j, row in enumerate(grid):
        dcfg = replace(DataConfig(), sr=sr, duration_s=seg_dur_s, drive_levels=(1.0,), seed=seed + j)
        u = build_training_excitation(dcfg)  # unit-amplitude dry forcing
        amp = float(np.prod([row[i] for i in pregain_idx])) if pregain_idx else 1.0
        beta = float(row[beta_idx[0]]) if beta_idx else 0.0
        y = simulate_duffing(amp * u, sr, f0=f0, zeta=zeta, beta=beta, oversample=oversample)
        xs.append(u.astype(np.float32))
        ys.append(y)
        values.append([float(v) for v in row])
        bounds.append(bounds[-1] + len(u))

    meta: dict[str, Any] = {
        "system": "duffing",
        "f0": f0,
        "zeta": zeta,
        "controls": [s.name for s in control_specs],
        "modes": [s.mode for s in control_specs],
    }
    return Dataset.from_segments(
        np.concatenate(xs),
        np.concatenate(ys),
        sr,
        bounds,
        np.asarray(values, dtype=np.float32),
        name="duffing",
        meta=meta,
        control_names=[s.name for s in control_specs],
        control_kinds=[s.kind for s in control_specs],
    )
