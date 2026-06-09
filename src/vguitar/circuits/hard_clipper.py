"""Near-square-wave hard clipper: a high-gain op-amp slammed into a +/-1 V clamp.

The harshest waveshaper in the validation set. A guitar-level input is multiplied
by a very large open-loop-style gain (an ideal op-amp VCVS configured as a fixed
high-gain non-inverting stage) so that even a few-millivolt input swings the raw
output far past the supply. That over-driven signal is then **hard-clamped to
about +/- 1 V**: below the clamp the transfer is an almost-vertical line, beyond
it the output is dead flat. The result is a near-**square wave** with extremely
sharp knees -- essentially a soft comparator. A single output RC pole then
band-limits the edges so they are steep but not literally vertical (the runner
oversamples 8x, so the band-limited edge still carries very high harmonics).

Why this is in the set: every other circuit clips *softly*. The diode pair, the
Tube Screamer feedback loop, the BJT/JFET stages and even the Big Muff cascade
all have a curved knee a smooth network can approximate with a low-order
polynomial / tanh. A near-square clamp is the opposite extreme: the static
transfer is essentially a clipped *step*, whose derivative is a pair of near-delta
spikes at the knees and zero elsewhere. That is the worst case for a smooth
(Lipschitz-bounded, low-curvature) neural emulator -- it must reproduce a vertical
transition and a perfectly flat top without overshoot or rounding, the regime
where ordinary regression and antiderivative anti-aliasing both struggle most.

Robustness: the op-amp is an ideal VCVS with a small output series resistor; the
clamp is a smooth-but-steep behavioral ``B``-source (a scaled ``tanh`` reinforced
by a hard ``min``/``max`` rail) referenced to ground (node ``0`` = virtual
ground), so the DC operating point sits exactly at 0 V and the transient solve is
well-posed. Every node has a DC path to ground.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# Clamp rail (volts): the flat top/bottom of the near-square output.
_VCLAMP = 1.0
# Edge steepness of the behavioral clamp (1/V inside tanh). Large -> sharper knee.
_KSHARP = 60.0
# Output low-pass: R * C sets the edge rise time / harmonic roll-off.
# 1k * 1.5n ~ 1.5 us -> ~106 kHz corner: steep edges, just shy of vertical.
_ROUT = 1_000.0
_COUT = 1.5e-9

# Drive pre-gain range. Even the low end clips hard (the stage gain is huge);
# higher settings only sharpen / widen the square plateau.
_DRIVE_LO = 0.05
_DRIVE_HI = 1.0
_DRIVE_DEFAULT = 0.3


@register_circuit
class HardClipper(Circuit):
    """High-gain op-amp + hard +/-1 V clamp + RC edge limiter (near-square wave).

    Topology (ground / virtual ground ``0``, input ``Vin``, output ``out``)::

        in -C1- np -+(op-amp, gain 1+Rf/Rg ~ 1001)+- oa     ; raw over-driven swing
        np -Rbias- 0                                          ; DC path for + input
        oa --B(clamp to +/-1 V via tanh + min/max)--> clamp   ; hard square clamp
        clamp -Rout- out -Cout- 0                             ; band-limit the edges
        out -Rload- 0                                         ; DC path / load

    The op-amp is a non-inverting amplifier with so much gain that any audible
    input is driven well past the +/-1 V rail; the behavioral clamp then squares
    it off. The transfer is therefore a clipped near-step centered on 0 V.
    """

    name = "hard_clipper"
    description = (
        "Near-square-wave hard clipper: high-gain op-amp slammed into a +/-1 V "
        "clamp (soft-comparator) + RC edge limiter; sharpest-knee stress case. "
        "drive (pre-gain)."
    )
    # A few tens of mV already clips; this peak drives it deep into the square.
    nominal_drive_v = 0.05
    controls = (ControlSpec("drive", "continuous", _DRIVE_LO, _DRIVE_HI, _DRIVE_DEFAULT, "pregain"),)

    _TEMPLATE = (
        "* Near-square-wave hard clipper (high-gain op-amp + hard +/-1 V clamp)\n"
        "Vin in 0 dc 0\n"
        # AC-couple in; bias the + input to virtual ground (node 0) for a DC path.
        "C1 in np 0.047u\n"
        "Rbias np 0 1meg\n"
        # Ideal op-amp as a non-inverting stage with very high gain (1 + Rf/Rg).
        "Eop oaraw 0 np nm 1e5\n"
        "Roa oaraw oa 100\n"
        "Rg nm 0 1k\n"
        "Rf oa nm 1meg\n"
        # Hard clamp to +/-VCLAMP: a steep tanh squares the knee, and a min/max
        # rail guarantees a perfectly flat, bounded plateau (no runaway, no UIC).
        "Bclamp clamp 0 "
        "V=max(-{vclamp:g}, min({vclamp:g}, {vclamp:g}*tanh({ksharp:g}*V(oa))))\n"
        "Rcs clamp cs 100\n"  # tiny series R after the behavioral source
        # Output RC pole: steep-but-not-vertical edges (band-limited).
        "Rout cs out {rout:g}\n"
        "Cout out 0 {cout:g}\n"
        "Rload out 0 100k\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        # No netlist-mode controls; resolve to validate/reject stray params.
        self._resolve_netlist_params(params)
        return self._TEMPLATE.format(
            vclamp=_VCLAMP, ksharp=_KSHARP, rout=_ROUT, cout=_COUT
        )
