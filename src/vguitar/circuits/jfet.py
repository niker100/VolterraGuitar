"""JFET common-source gain stage (2N5457) — square-law overdrive with tone.

A self-biased N-channel JFET common-source amplifier, the "Class-A" front end of
many boutique boosters/overdrives (e.g. the AMZ Mosfet booster family, countless
JFET pre-stages). Its nonlinearity is *square-law*: the drain current follows
``Id ~ (Vgs - Vto)^2`` in saturation, which clips softly and **asymmetrically**
(cutoff when ``Vgs < Vto`` on one half-cycle, drain saturation on the other) —
a mechanism distinct from the BJT's exponential ``I_C(V_BE)`` and the diode pair's
Shockley clamp, so it broadens the nonlinearity coverage of the validation set.

It converges trivially in ngspice (no exponential feedback loop), which makes it
the low-risk first complex circuit. Controls:

* ``drive`` — input pre-gain (how hard the stage is driven), giving the amplitude
  coverage the model needs.
* ``tone`` — a normalized ``0..1`` knob mapped (in :meth:`netlist_for`) to the
  output low-pass capacitor, rolling the highs from bright to dark.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# Standard 2N5457 N-JFET parameters (Vgs(off) ~ -1.5 V, IDSS ~ 1 mA).
_JFET_MODEL = (
    ".model J2N5457 NJF(VTO=-1.526 BETA=4.6e-4 LAMBDA=3.7e-3 RD=10 RS=10 "
    "IS=1e-14 CGS=4.5p CGD=4p PB=1 FC=0.5)"
)

_TONE_C_LO = 1.0e-9  # brightest (tone=0): ~16 kHz corner
_TONE_C_HI = 47.0e-9  # darkest (tone=1): ~0.7 kHz corner


@register_circuit
class JfetCS(Circuit):
    """Self-biased 2N5457 common-emitter... common-*source* JFET overdrive stage.

    Topology (ground ``0``, input ``Vin``, output ``out``)::

        in -C1- gate -+- J1(drain,gate,src) ; Rd to Vcc, self-bias Rs+Cs at src
                      Rg to 0               ; C2 couples drain -> out
        out: Rload to 0, Ctone to 0 (tone)
    """

    name = "jfet"
    description = (
        "Self-biased 2N5457 JFET common-source overdrive stage: square-law, "
        "asymmetric soft clipping; drive (pre-gain) + tone (output low-pass)."
    )
    nominal_drive_v = 0.3
    controls = (
        ControlSpec("drive", "continuous", 0.05, 1.5, 0.3, "pregain"),
        ControlSpec("tone", "continuous", 0.0, 1.0, 0.5, "netlist"),
    )

    _TEMPLATE = (
        "* JFET common-source gain stage (2N5457)\n"
        "Vin in 0 dc 0\n"
        "Vcc vcc 0 dc 9\n"
        "C1 in gate 0.1u\n"
        "Rg gate 0 1meg\n"
        "J1 drain gate src J2N5457\n"
        "Rd vcc drain 10k\n"
        "Rs src 0 2.2k\n"
        "Cs src 0 22u\n"
        "C2 drain out 0.1u\n"
        "Rload out 0 1meg\n"
        "Ctone out 0 {tone_c:g}\n" + _JFET_MODEL + "\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        p = self._resolve_netlist_params(params)
        tone_c = _TONE_C_LO + p["tone"] * (_TONE_C_HI - _TONE_C_LO)
        return self._TEMPLATE.format(tone_c=tone_c)
