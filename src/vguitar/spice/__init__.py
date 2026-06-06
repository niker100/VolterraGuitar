"""ngspice-backed simulation of :class:`~vguitar.circuits.base.Circuit` netlists.

See :mod:`vguitar.spice.runner` for the (carefully documented) mechanism that
streams an arbitrary input signal through ngspice via the external-source
callback and captures the output.
"""

from __future__ import annotations

from vguitar.spice.runner import make_dataset, simulate

__all__ = ["make_dataset", "simulate"]
