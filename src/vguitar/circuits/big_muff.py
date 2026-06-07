"""Big Muff-style multi-stage fuzz: cascaded clippers + tone stack + level.

A two-stage cascade modelled on the Electro-Harmonix Big Muff Pi sustain section:
two high-gain op-amp clipping stages in series (each with anti-parallel feedback
diodes), followed by the famous passive **tone stack** (a low-pass path and a
high-pass path blended by one pot, giving the scooped-mids voice) and a **level**
(volume) pot. Cascaded clipping is what makes the fuzz so harmonically dense and
so different from a single overdrive stage — and the multi-stage feedback +
passive interstage make it the convergence stress-test of the validation set.

This is also the first **three-control** circuit, exercising the multi-axis
dataset path (Sobol sampling) and 2-D control validation visuals:

* ``sustain`` — input pre-gain into the cascade (the "sustain"/fuzz amount),
  giving amplitude coverage.
* ``tone`` — normalized ``0..1`` knob blending the low-pass and high-pass paths
  of the tone stack (dark/bassy to bright/trebly), computed in :meth:`netlist_for`.
* ``level`` — normalized ``0..1`` output volume pot (a near-linear post-gain),
  also computed in :meth:`netlist_for`.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

_DCLIP_MODEL = ".model DCLIP D(IS=2.52n N=1.752 RS=0.568 CJO=4p VJ=0.75 M=0.333)"

_POT = 100_000.0  # tone & level pot resistance (ohms)


@register_circuit
class BigMuff(Circuit):
    """Two cascaded op-amp diode clippers + passive tone stack + level pot.

    Topology (ground ``0``, input ``Vin``, output ``out``)::

        in -C1- stage1(op-amp+D1/D2) -C2- stage2(op-amp+D3/D4) -> o2
        o2 -> [LPF node] and [HPF node] -> tone pot wiper -> level pot -> out
    """

    name = "big_muff"
    description = (
        "Big Muff-style two-stage fuzz: cascaded op-amp diode clippers + passive "
        "tone stack + level; sustain (pre-gain) + tone (blend) + level (volume)."
    )
    nominal_drive_v = 0.1
    controls = (
        ControlSpec("sustain", "continuous", 0.02, 1.0, 0.3, "pregain"),
        ControlSpec("tone", "continuous", 0.0, 1.0, 0.5, "netlist"),
        ControlSpec("level", "continuous", 0.0, 1.0, 0.7, "netlist"),
    )

    _TEMPLATE = (
        "* Big Muff-style two-stage fuzz (cascaded clippers + tone + level)\n"
        "Vin in 0 dc 0\n"
        "C1 in n1p 0.047u\n"
        "Rb1 n1p 0 500k\n"
        "Eo1 o1raw 0 n1p n1m 1e5\n"
        "Ro1 o1raw o1 100\n"
        "Rg1 n1m 0 2.2k\n"
        "Rf1 o1 n1m 47k\n"
        "D1 o1 n1m DCLIP\n"
        "D2 n1m o1 DCLIP\n"
        "C2 o1 n2p 0.01u\n"
        "Rb2 n2p 0 500k\n"
        "Eo2 o2raw 0 n2p n2m 1e5\n"
        "Ro2 o2raw o2 100\n"
        "Rg2 n2m 0 2.2k\n"
        "Rf2 o2 n2m 47k\n"
        "D3 o2 n2m DCLIP\n"
        "D4 n2m o2 DCLIP\n"
        "Rlp_in o2 lpnode 39k\n"
        "Clp lpnode 0 10n\n"
        "Chp o2 hpnode 10n\n"
        "Rhp_in hpnode 0 22k\n"
        "Rt_lp lpnode wiper {r_lp:g}\n"
        "Rt_hp hpnode wiper {r_hp:g}\n"
        "Rvtop wiper out {r_vtop:g}\n"
        "Rvbot out 0 {r_vbot:g}\n"
        "Rload out 0 1meg\n" + _DCLIP_MODEL + "\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        p = self._resolve_netlist_params(params)
        tone = p["tone"]
        level = p["level"]
        # Tone pot: smaller resistance to a path = more of that path in the blend.
        # tone=0 -> low-pass dominates (bassy); tone=1 -> high-pass dominates.
        r_lp = tone * _POT + 1.0
        r_hp = (1.0 - tone) * _POT + 1.0
        # Level pot as a voltage divider: out/wiper ~ level.
        r_vtop = (1.0 - level) * _POT + 1.0
        r_vbot = level * _POT + 1.0
        return self._TEMPLATE.format(r_lp=r_lp, r_hp=r_hp, r_vtop=r_vtop, r_vbot=r_vbot)
