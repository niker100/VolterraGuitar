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
from dataclasses import dataclass
from typing import ClassVar, TypeVar


@dataclass(frozen=True)
class ControlSpec:
    """One exogenous control axis a circuit exposes (a knob, switch, or drift).

    A control either scales the input (``mode="pregain"`` -- the drive knob of a
    real overdrive pedal, modelled as a pre-gain into the stage) or substitutes a
    value into the netlist (``mode="netlist"`` -- a tone capacitor, a feedback
    resistor, a bias/supply voltage). Netlist-mode controls correspond to a
    ``{name}`` placeholder rendered by :meth:`Circuit.netlist_for`.

    Attributes:
        name: control identifier; also the netlist placeholder for netlist mode.
        kind: one of :data:`vguitar.data.CONTROL_KINDS`
            (``continuous`` | ``discrete`` | ``drift``).
        lo, hi: the swept range of the control value.
        default: the value used when the control is unspecified (e.g. the netlist
            returned by :meth:`Circuit.netlist`).
        mode: ``"pregain"`` (scale the input) or ``"netlist"`` (substitute value).
    """

    name: str
    kind: str = "continuous"
    lo: float = 0.0
    hi: float = 1.0
    default: float = 0.0
    mode: str = "pregain"


class Circuit(ABC):
    """A nonlinear analog circuit to be emulated."""

    name: ClassVar[str]
    description: ClassVar[str] = ""
    #: Suggested peak input drive (volts) at which the stage clips noticeably.
    #: Used to scale excitation and as a sane default for listening tests.
    nominal_drive_v: ClassVar[float] = 0.5
    #: Exogenous control axes this circuit exposes. Empty (the default) means the
    #: circuit has no declared controls; the only conditioning then available is
    #: the implicit input pre-gain ("drive"). Circuits with a tone knob / switch
    #: declare ``netlist``-mode controls and override :meth:`netlist_for`.
    controls: ClassVar[tuple[ControlSpec, ...]] = ()

    @abstractmethod
    def netlist(self) -> str:
        """Return the self-contained ngspice netlist (see module docstring).

        For a parameterized circuit this returns the netlist at the controls'
        *default* values; per-setting variants come from :meth:`netlist_for`.
        """
        ...

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        """Render the netlist for a set of ``netlist``-mode control values.

        The default ignores ``params`` and returns :meth:`netlist` -- correct for
        circuits with no netlist-mode controls (the two original circuits are
        untouched). Parameterized circuits override this to substitute ``params``
        into a template; :meth:`_resolve_netlist_params` fills defaults and
        rejects unknown keys.
        """
        return self.netlist()

    def _resolve_netlist_params(self, params: dict[str, float] | None) -> dict[str, float]:
        """Merge ``params`` over the netlist-mode control defaults; reject typos.

        Returns a complete ``{name: value}`` dict covering every ``netlist``-mode
        control, with provided ``params`` overriding the declared defaults. Raises
        ``ValueError`` for any key that is not a declared netlist-mode control, so
        a placeholder typo fails loudly at generation time.
        """
        specs = {c.name: c for c in self.controls if c.mode == "netlist"}
        resolved = {name: float(spec.default) for name, spec in specs.items()}
        if params:
            unknown = sorted(set(params) - set(specs))
            if unknown:
                raise ValueError(
                    f"circuit {self.name!r}: unknown netlist control(s) {unknown}; "
                    f"have {sorted(specs)}"
                )
            for key, value in params.items():
                resolved[key] = float(value)
        return resolved


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
