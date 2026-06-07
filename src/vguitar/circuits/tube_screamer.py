"""Tube Screamer-style overdrive: op-amp gain stage with feedback diode clipping.

The clipping core of the Ibanez Tube Screamer / countless "green pedal" clones:
a non-inverting op-amp whose feedback path contains a pair of **anti-parallel
diodes**. Below a diode drop the stage is a clean ``1 + Rf/Rg`` amplifier; once
the output swings beyond ~one diode drop above the inverting input the diodes
conduct and *softly* clamp the gain, so the output rides on the dry signal —
the dynamic, "transparent" overdrive character. This is a genuinely new topology
for the validation set: a high-gain **feedback** nonlinearity (the diodes sit
inside the loop), not a feedforward clamp.

The op-amp is an ideal VCVS (high gain + a small output series resistance so the
diode loop is well-posed); the split-supply virtual ground is taken as node ``0``
(the input is AC-coupled), which keeps the DC operating point at zero and the
transient solve robust. Controls:

* ``drive`` — input pre-gain into the fixed-gain stage (clean at low settings,
  clipped at high settings), giving amplitude coverage.
* ``tone`` — normalized ``0..1`` knob mapped (in :meth:`netlist_for`) to the
  post-clip low-pass capacitor (bright to dark).
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# 1N4148-class clipping diode (sets the ~0.6 V soft knee).
_DCLIP_MODEL = ".model DCLIP D(IS=2.52n N=1.752 RS=0.568 CJO=4p VJ=0.75 M=0.333)"

_TONE_C_LO = 4.7e-9  # brightest (tone=0): ~34 kHz corner with Rt=1k
_TONE_C_HI = 100.0e-9  # darkest (tone=1): ~1.6 kHz corner


@register_circuit
class TubeScreamer(Circuit):
    """Non-inverting op-amp + anti-parallel feedback diodes (TS-808 clipping core).

    Topology (ground/virtual-ground ``0``, input ``Vin``, output ``out``)::

        in -C1- np -+(op-amp)+- oa  ; Eop = 1e5*(np-nm), Roa series
        oa -Rf- nm -Rg- 0          ; gain 1+Rf/Rg
        oa =D1/D2= nm              ; anti-parallel feedback clipping diodes
        oa -Rt- out -Ctone- 0      ; post-clip tone low-pass
    """

    name = "tube_screamer"
    description = (
        "Tube Screamer-style overdrive: non-inverting op-amp with anti-parallel "
        "feedback diodes (in-loop soft clipping); drive (pre-gain) + tone (low-pass)."
    )
    nominal_drive_v = 0.2
    controls = (
        ControlSpec("drive", "continuous", 0.02, 1.0, 0.2, "pregain"),
        ControlSpec("tone", "continuous", 0.0, 1.0, 0.5, "netlist"),
    )

    _TEMPLATE = (
        "* Tube Screamer-style op-amp + feedback diode clipper\n"
        "Vin in 0 dc 0\n"
        "C1 in np 0.047u\n"
        "Rbias np 0 500k\n"
        "Eop oaraw 0 np nm 1e5\n"
        "Roa oaraw oa 100\n"
        "Rg nm 0 4.7k\n"
        "Rf oa nm 51k\n"
        "D1 oa nm DCLIP\n"
        "D2 nm oa DCLIP\n"
        "Rt oa out 1k\n"
        "Ctone out 0 {tone_c:g}\n"
        "Rload out 0 100k\n" + _DCLIP_MODEL + "\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        p = self._resolve_netlist_params(params)
        tone_c = _TONE_C_LO + p["tone"] * (_TONE_C_HI - _TONE_C_LO)
        return self._TEMPLATE.format(tone_c=tone_c)
