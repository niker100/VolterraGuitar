"""Non-audio nonlinear dynamical systems with exogenous inputs.

These exist to demonstrate that the project's emulation approach — and in
particular CIRCE3's control decomposition (signal-acting controls folded into the
input, system-acting controls via minimal FiLM) — is **domain-general**, not
audio-specific. A guitar circuit and a Duffing oscillator are both causal,
fading-memory nonlinear systems driven by an exogenous input with exogenous
parameters; the same model handles both by the same mechanism.
"""

from __future__ import annotations

from vguitar.models.archive.systems.duffing import make_duffing_dataset, simulate_duffing

__all__ = ["make_duffing_dataset", "simulate_duffing"]
