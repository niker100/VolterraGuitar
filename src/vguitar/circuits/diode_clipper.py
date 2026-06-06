"""Symmetric soft-clipping diode stage — the canonical overdrive/distortion core.

A series resistor feeds a node clamped to ground by two anti-parallel diodes.
As the input rises, each diode's exponential I-V curve (Shockley equation)
softly limits the output swing to roughly +/- one diode drop, producing the
smooth, harmonic-rich clipping of pedals like the Ibanez Tube Screamer / MXR
Distortion+. A small load capacitor adds a single real pole, gently rolling off
the highest harmonics.

Why it is here: electrically this is a *near-memoryless* nonlinearity (the diode
current depends on the instantaneous node voltage) followed by one linear pole.
That makes it the cleanest possible smoke-test circuit — it converges trivially
in SPICE — and a textbook target for antiderivative anti-aliasing (ADAA,
Parker et al., DAFx-16, "Reducing the Aliasing of Nonlinear Waveshaping Using
Continuous-Time Convolution"), in clear contrast to the stateful BJT stage.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, register_circuit


@register_circuit
class DiodeClipper(Circuit):
    """Two anti-parallel 1N4148 diodes clamping a resistor-fed node to ground.

    Topology (spine conventions; ground = ``0``, input ``Vin``, output ``out``)::

        in --[ R1 1k ]-- out --+-- D1 (out->0)
                               |-- D2 (0->out)   anti-parallel pair
                               +-- C1 10n -- 0   (1-pole low-pass, ~16 kHz)

    The diode pair is symmetric, so positive and negative half-cycles clip
    identically, yielding predominantly odd harmonics.
    """

    name = "diode"
    description = "Symmetric soft-clipping diode pair (overdrive/distortion core) + 1-pole LPF."
    nominal_drive_v = 0.7

    def netlist(self) -> str:
        # 1N4148 small-signal switching diode, standard published SPICE params
        # (IS/N set the soft-knee voltage ~0.6 V; RS the bulk resistance; the
        # junction-capacitance terms CJO/VJ/M and transit time TT shape the
        # tiny high-frequency / reverse-recovery behaviour).
        return (
            "* Symmetric diode clipper (soft clipping + 1-pole LPF)\n"
            "Vin in 0 dc 0\n"
            "R1 in out 1k\n"
            "D1 out 0 D1N4148\n"
            "D2 0 out D1N4148\n"
            "C1 out 0 10n\n"
            ".model D1N4148 D(IS=2.52n N=1.752 RS=0.568 "
            "CJO=4p VJ=0.75 M=0.333 TT=20n BV=100 IBV=100u)\n"
        )
