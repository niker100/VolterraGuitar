"""Single-stage 2N3904 common-emitter (CE) overdrive gain stage.

A faithful port of the legacy Micro-Cap ``circuit1.cir``: the textbook
voltage-divider-biased common-emitter amplifier that is the workhorse front
end of countless guitar overdrive/booster pedals (e.g. the Electra Distortion
and Tube Screamer-style input stages).

Topology
--------
* ``C1`` (10uF) AC-couples the guitar signal onto the base.
* ``R1`` (10k) / ``R3`` (1k) form the voltage divider that sets the DC base
  bias (~0.82 V) from the 9V rail (``Vcc``); with the 470R emitter resistor
  this lands the quiescent collector roughly mid-rail (~4-5 V). (The legacy
  Micro-Cap values biased the stage into saturation; the divider is corrected
  here so the transistor sits in its active region and actually amplifies.)
* ``R4`` (10k) is the collector load; ``R2`` (470R) is the emitter
  degeneration resistor, bypassed for AC gain by ``C3`` (20uF).
* ``C2`` (0.1uF) AC-couples the collector to ``out``, loaded by ``R5`` (100k).

Behaviour
---------
With the emitter resistor fully bypassed the small-signal voltage gain is
approximately ``-g_m * R_load`` (~30-40 dB, *inverting*). The stage clips
early and asymmetrically: it is the transistor's large-signal
Ebers-Moll/Gummel-Poon behaviour -- exponential ``I_C(V_BE)`` plus cutoff and
saturation of the output swing against the rails -- that supplies the
nonlinearity we want to emulate, so ``nominal_drive_v`` is deliberately small.

The 2N3904 ``.model`` card uses the widely published ON Semiconductor
Gummel-Poon parameter set for the part.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, register_circuit

# Standard ON Semiconductor 2N3904 Gummel-Poon NPN parameters.
_Q2N3904_MODEL = (
    ".model Q2N3904 NPN("
    "IS=6.734f XTI=3 EG=1.11 VAF=74.03 BF=416.4 NE=1.259 ISE=6.734f "
    "IKF=66.78m XTB=1.5 BR=.7371 NC=2 ISC=0 IKR=0 RC=1 CJC=3.638p "
    "MJC=.3085 VJC=.75 FC=.5 CJE=4.493p MJE=.2593 VJE=.75 TR=239.5n "
    "TF=301.2p ITF=.4 VTF=4 XTF=2 RB=10)"
)


@register_circuit
class BjtCE(Circuit):
    """2N3904 common-emitter overdrive stage (port of legacy ``circuit1.cir``).

    A ~30-40 dB inverting gain stage whose distortion comes from the BJT
    large-signal (Gummel-Poon) behaviour; it clips early, hence the small
    :attr:`nominal_drive_v`.
    """

    name = "bjt"
    description = (
        "Single-stage 2N3904 common-emitter amplifier: a ~30-40 dB inverting "
        "guitar overdrive gain stage; nonlinearity from BJT large-signal behaviour."
    )
    nominal_drive_v = 0.1

    def netlist(self) -> str:
        """Return the self-contained ngspice netlist (see :mod:`base`)."""
        return "\n".join(
            (
                "* 2N3904 common-emitter overdrive stage (port of circuit1.cir)",
                "Vin in 0 dc 0",  # input source (overridden per-sample)
                "Vcc vcc 0 dc 9",  # 9 V supply rail
                "C1 in base 10u",  # input coupling (low corner, passes guitar lows)
                "R1 vcc base 10k",  # bias divider (top)
                "R3 base 0 1k",  # bias divider (bottom): V_base ~ 0.82 V
                "R4 vcc coll 10k",  # collector load
                "Q1 coll base emit Q2N3904",
                "R2 emit 0 470",  # emitter degeneration
                "C3 emit 0 20u",  # emitter bypass -> high AC gain
                "C2 coll out 0.1u",  # output coupling
                "R5 out 0 100k",  # output load (next-stage input impedance)
                _Q2N3904_MODEL,
            )
        )
