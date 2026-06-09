"""Strongly *asymmetric* diode clipper — a 2:1 hard-clip with a rectifying DC shift.

A series resistor feeds a node that is clamped to ground by **two stacked diodes
on the positive half** and a **single diode on the negative half**. The positive
excursion therefore clips at roughly *two* diode drops (~+1.2 V) while the
negative excursion clips at only *one* drop (~-0.6 V). The two halves of the
waveform are limited at very different levels, which is the whole point:

* The transfer curve ``out = f(in)`` is markedly **non-odd** (``f(-x) != -f(x)``),
  so the stage generates strong **even** harmonics on top of the odd ones — the
  octave-up "gated/rectifier" timbre of classic asymmetric fuzz/overdrive
  (e.g. germanium clippers, the asymmetric clipping stage many tube-screamer
  mods add by stacking a diode on one leg only).
* Because the clip levels differ, a symmetric (zero-mean) input produces an
  output with a **nonzero mean**: the stage partially *rectifies*. A slow,
  signal-dependent DC offset rides under the audio — exactly the behaviour that
  stresses any DC handling downstream.

Why it is a genuinely *harder* case than the existing set:

* The symmetric ``diode`` clipper, the BJT/JFET single-ended stages, the
  tube-screamer feedback-diode loop and the big-muff cascade are all close to
  odd-symmetric (or smoothly biased) around their operating point. This stage is
  deliberately **break-point asymmetric**: there are two distinct, sharp knees at
  different input levels, and a *kink* near zero where the active clamp switches
  from the single diode to the stacked pair. A smooth neural emulator (a Volterra
  / TCN / RNN trained on bandlimited data) has to fit two different hard
  saturations *and* reproduce a level-dependent DC term — a poor fit shows up as
  the wrong even-harmonic balance and a drifting baseline.

Topology (spine conventions; ground = ``0``, input ``Vin``, output ``out``)::

    in --[ R1 ]-- out --+-- D1a (out->m) --D1b (m->0)   ; +half: 2 stacked diodes
                        |-- D2 (0->out)                  ; -half: 1 diode
                        |-- C1 -- 0                       ; 1-pole low-pass
                        +-- Rload -- 0                    ; DC path to ground

Convergence: every node has a resistive DC path to ground (``R1`` from the input
node, ``Rload`` from ``out``, ``Rmid`` from the inter-diode node ``m``), so the
DC operating point at ``Vin = 0`` is the trivial all-zero solution and the
transient solve starts from a clean, robust bias. Standard 1N4148 small-signal
diode models — no ideal switches, no hysteretic primitives — keep the solve well
behaved despite the sharp knees.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# 1N4148-class small-signal switching diode (sets the ~0.6 V soft knee per diode).
_D_MODEL = ".model DCLIP D(IS=2.52n N=1.752 RS=0.568 CJO=4p VJ=0.75 M=0.333 TT=20n)"


@register_circuit
class AsymClipper(Circuit):
    """2:1 asymmetric diode clipper (stacked-pair positive / single negative).

    The positive half clips at ~two diode drops, the negative half at ~one, so
    the transfer curve is strongly non-odd: strong even *and* odd harmonics plus
    a level-dependent rectifying DC shift.

    Controls:

    * ``drive`` — input pre-gain into the fixed clipper (clean at low settings,
      hard-clipped at high settings), giving amplitude coverage.
    * ``bias`` — optional symmetry trim. A normalized ``0..1`` knob mapped (in
      :meth:`netlist_for`) to a small current injected at ``out`` through a large
      resistor from a fixed rail, sliding the clip window so the even/odd balance
      and the DC-shift sign can be swept. ``0.5`` is the centred (most
      asymmetric-as-drawn) setting.
    """

    name = "asym_clipper"
    description = (
        "Strongly asymmetric diode clipper: two stacked diodes clamp the positive "
        "half (~2 drops) and one diode the negative half (~1 drop), giving strong "
        "even+odd harmonics and a rectifying DC shift; drive (pre-gain) + bias trim."
    )
    # ~1.2 V peak drives the positive half well past its 2-drop knee while the
    # negative half is already hard-clipped at one drop -> obvious asymmetry.
    nominal_drive_v = 1.2

    controls = (
        ControlSpec("drive", "continuous", 0.05, 1.0, 0.5, "pregain"),
        ControlSpec("bias", "continuous", 0.0, 1.0, 0.5, "netlist"),
    )

    # Bias trim: a current injected into `out` via a 2.2 Meg resistor from a rail
    # swept over +/-3 V. At bias=0.5 the rail is 0 V (no net injection -> the
    # symmetric-as-drawn 2:1 clipper). The large resistor keeps the injected
    # current tiny (~1 uA) so it only nudges the clip window / DC point.
    _BIAS_RAIL_RANGE_V = 3.0

    _TEMPLATE = (
        "* Asymmetric diode clipper (2:1 stacked-pair / single, rectifying DC shift)\n"
        "Vin in 0 dc 0\n"
        "R1 in out 2.2k\n"
        # Positive half: out clamps up through TWO stacked diodes to ground.
        "D1a out m DCLIP\n"
        "D1b m 0 DCLIP\n"
        # Negative half: out clamps down through ONE diode to ground.
        "D2 0 out DCLIP\n"
        # Inter-diode node needs its own DC path to ground (else it floats).
        "Rmid m 0 10meg\n"
        # 1-pole low-pass (~7 kHz with R1) rolls off the sharpest harmonics.
        "C1 out 0 10n\n"
        # Output DC path to ground (keeps the op point well-posed at Vin=0).
        "Rload out 0 1meg\n"
        # Optional bias trim: tiny current into `out` from a swept rail.
        "Vbias vb 0 dc {bias_v:g}\n"
        "Rbias vb out 2.2meg\n"
        + _D_MODEL + "\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        p = self._resolve_netlist_params(params)
        # Map bias 0..1 -> rail -range..+range, centred (no injection) at 0.5.
        bias_v = (p["bias"] - 0.5) * 2.0 * self._BIAS_RAIL_RANGE_V
        return self._TEMPLATE.format(bias_v=bias_v)