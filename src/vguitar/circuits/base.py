"""The :class:`Circuit` contract.

A circuit is defined by a **raw ngspice netlist string** (not a PySpice object).
Plain text is the most explainable, portable, and version-control-friendly
representation — you can read exactly what is being simulated.

Netlist conventions (the SPICE runner relies on these):

* Ground is node ``0``.
* The input is an independent voltage source named ``Vin`` connected between
  node ``in`` and ``0``. Its ``dc 0`` value is overridden sample-by-sample by
  the runner (via PySpice's ``NgSpiceShared`` external-source callback), so the
  netlist value is just a placeholder.
* The output is taken at node ``out``.
* Any device ``.model`` cards the circuit needs are included in the string.

Keep netlists self-contained and free of analysis commands (``.tran`` etc.);
the runner adds those.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar


class Circuit(ABC):
    """A nonlinear analog circuit to be emulated."""

    name: ClassVar[str]
    description: ClassVar[str] = ""
    #: Suggested peak input drive (volts) at which the stage clips noticeably.
    #: Used to scale excitation and as a sane default for listening tests.
    nominal_drive_v: ClassVar[float] = 0.5

    @abstractmethod
    def netlist(self) -> str:
        """Return the self-contained ngspice netlist (see module docstring)."""
        ...


# --- registry -------------------------------------------------------------
_REGISTRY: dict[str, type[Circuit]] = {}

_CircuitT = TypeVar("_CircuitT", bound="Circuit")


def register_circuit(cls: type[_CircuitT]) -> type[_CircuitT]:
    """Class decorator: register a Circuit under its ``name``."""
    key = cls.name.lower()
    if key in _REGISTRY:
        raise ValueError(f"duplicate circuit name: {cls.name!r}")
    _REGISTRY[key] = cls
    return cls


def get_circuit(name: str) -> Circuit:
    try:
        return _REGISTRY[name.lower()]()
    except KeyError:
        raise KeyError(f"unknown circuit {name!r}; have {sorted(_REGISTRY)}") from None


def all_circuits() -> dict[str, type[Circuit]]:
    return dict(_REGISTRY)
