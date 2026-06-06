"""vguitar — real-time digital emulation of nonlinear analog guitar circuits.

The package is organized around three small contracts (see the ``base`` modules):

* :class:`vguitar.circuits.base.Circuit` — a SPICE netlist with a driven input
  source ``Vin`` and an output node ``out``.
* :class:`vguitar.data.Dataset` — aligned ``(x, y)`` mono float32 signals at a
  single canonical sample rate.
* :class:`vguitar.models.base.Model` — anything that can ``fit`` a Dataset and
  ``process`` audio, offline and block-by-block (for the live engine).

Everything else — the SPICE runner, excitation design, metrics, the benchmark,
and the live app — is written against those three contracts so that Volterra,
block-oriented, and neural approaches are interchangeable and directly
comparable.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Canonical audio rate used for datasets, training, and benchmarking.
# Oversampling for anti-aliasing at inference is a separate, opt-in concern
# handled by the realtime engine and the ADAA helpers (see vguitar.nonlinear).
AUDIO_SR = 44_100

__all__ = ["AUDIO_SR", "__version__"]
