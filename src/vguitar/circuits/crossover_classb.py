"""Class-B push-pull output stage: crossover (dead-zone) distortion.

A complementary emitter-follower output stage run with **little or no idle bias**
-- the classic source of *crossover distortion*. An ideal op-amp buffer drives the
joined bases of an NPN/PNP pair whose emitters are tied together at the output and
whose collectors go to the split supply rails. Because each transistor only turns
on once its base-emitter junction is forward-biased (~0.6 V), there is a **dead
zone** around zero: for small input swings *neither* device conducts and the output
sits pinned near 0 V; once the drive exceeds a V_be drop the corresponding device
switches on and the output follows. The transfer curve therefore has a flat dead
band bracketed by two **sharp slope discontinuities (kinks)** where one transistor
hands off to the other -- the textbook hard-to-model crossover artifact.

This is electrically distinct from the rest of the validation set: the diode/
JFET/BJT/Tube-Screamer/Big-Muff stages all *clip the peaks* (saturate at large
amplitude), whereas this stage *kills the centre* (a dead band at small amplitude
with linear follow outside it). It is also a **push-pull / complementary** topology
-- two devices alternating -- which none of the others use.

Robustness notes (the DC op + transient solve must converge with no UIC):

* The op-amp is an ideal VCVS plus a small output series resistor.
* The input is AC-coupled and the supplies are symmetric, so with zero idle bias
  both transistors are off at DC and the output operating point is exactly 0 V
  (virtual ground = node ``0``); ``Rload`` gives the output its DC path.
* Every node has a DC path to ground; gains/voltages stay O(1-10 V).
* An output RC forms the post-stage low-pass pole.

Controls:

* ``drive`` -- input pre-gain into the stage. Low settings sit almost entirely in
  the dead zone (tiny, heavily-mangled output); high settings push well past the
  crossover kinks (output follows with the gap carved out of the centre).
* ``bias`` -- normalized ``0..1`` idle-bias knob mapped (in :meth:`netlist_for`) to
  a pair of small base batteries that pre-bias the junctions. ``bias=0`` is pure
  class-B (widest dead zone / sharpest crossover); higher values shrink the dead
  zone toward class-AB (softer, narrower kink). It never fully closes the gap, so
  the nonlinearity stays sharp across the whole range.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# Generic complementary small-signal pair (2N3904 / 2N3906-class).
_NPN_MODEL = ".model QNPN NPN(IS=6.7e-15 BF=200 VAF=100 RB=10 RC=1 RE=0.5 CJE=4p CJC=4p TF=0.3n)"
_PNP_MODEL = ".model QPNP PNP(IS=1.0e-14 BF=200 VAF=100 RB=10 RC=1 RE=0.5 CJE=4p CJC=4p TF=0.3n)"

# Idle-bias mapping: half-battery (volts) inserted on EACH base.
# bias=0 -> 0 V (pure class-B, full ~1.2 V dead band); bias=1 -> ~0.5 V per side
# (class-AB, narrow residual kink -- never closes fully so the curve stays sharp).
_BIAS_HALF_MAX = 0.5


@register_circuit
class CrossoverClassB(Circuit):
    """Class-B complementary emitter-follower -> crossover / dead-zone distortion.

    Topology (ground/virtual-ground ``0``, input ``Vin``, output ``out``)::

        in -C1- np                      ; AC-couple, Rin references np to 0
        Eop oaraw 0 np oa 1e5 ; Roa     ; ideal op-amp, LOCAL unity buffer -> oa
        bp = oa + vb_half               ; Vbp: +half-bias battery to NPN base
        bn = oa - vb_half               ; Vbn: -half-bias battery to PNP base
        QN  vp bp out  QNPN             ; NPN follower, collector to +rail
        QP  vn bn out  QPNP             ; PNP follower, collector to -rail
        out -Rload- 0 , out -Rl- on -Cl- 0   ; DC path + output RC pole

    The op-amp is closed *locally* (its inverting input is ``oa`` itself) as a
    clean unity buffer of the input; the push-pull pair is then driven
    **open-loop** so the crossover dead zone appears undisguised at ``out`` (a
    closed loop around the output stage would use its gain to erase the very
    artifact we want to model). This also keeps the only feedback loop a simple,
    unconditionally stable buffer -- the nonlinear push-pull is pure feedforward,
    which is what makes the DC op and transient solve converge easily.
    """

    name = "crossover_classb"
    description = (
        "Class-B complementary push-pull emitter-follower: a dead zone around zero "
        "(no idle bias) with sharp crossover kinks; drive (pre-gain) + bias (class-AB)."
    )
    # Peak drive where the stage clearly works the crossover region: a ~1 V peak
    # spends a meaningful fraction of each cycle inside the ~1.2 V dead band.
    nominal_drive_v = 1.0
    controls = (
        ControlSpec("drive", "continuous", 0.1, 4.0, 1.0, "pregain"),
        ControlSpec("bias", "continuous", 0.0, 1.0, 0.0, "netlist"),
    )

    _TEMPLATE = (
        "* Class-B push-pull crossover (dead-zone) distortion stage\n"
        "Vin in 0 dc 0\n"
        "* split supply rails (symmetric -> 0 V output operating point)\n"
        "Vcc vp 0 dc 9\n"
        "Vee vn 0 dc -9\n"
        "* AC-couple input; reference the op-amp + input to virtual ground\n"
        "C1 in np 0.1u\n"
        "Rin np 0 1meg\n"
        "* ideal op-amp as a LOCAL unity buffer of the input (oa follows np)\n"
        "Eop oaraw 0 np oa 1e5\n"
        "Roa oaraw oa 100\n"
        "* idle-bias batteries (per-side); 0 V => pure class-B dead band\n"
        "Vbp bp oa dc {vb_half:g}\n"
        "Vbn oa bn dc {vb_half:g}\n"
        "* complementary push-pull emitter-followers (emitters joined at out)\n"
        "QN vp bp out QNPN\n"
        "QP vn bn out QPNP\n"
        "* output DC path + RC low-pass pole\n"
        "Rload out 0 100k\n"
        "Rl out on 1k\n"
        "Cl on 0 10n\n"
        "Rleak_bp bp 0 1meg\n"
        "Rleak_bn bn 0 1meg\n" + _NPN_MODEL + "\n" + _PNP_MODEL + "\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        p = self._resolve_netlist_params(params)
        vb_half = p["bias"] * _BIAS_HALF_MAX
        return self._TEMPLATE.format(vb_half=vb_half)
