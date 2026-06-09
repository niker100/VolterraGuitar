"""Full-wave rectifier / octave-up stage — Green Ringer / Octavia core.

The harmonic heart of an *octave-up* fuzz (Tycobrahe Octavia, Dan Armstrong
Green Ringer): a **full-wave rectifier** that folds both halves of the waveform
to the same polarity, so the output approximates ``|gain * input|``. Folding the
signal about zero doubles the apparent pitch — a strong **octave-up** — and,
because ``abs(.)`` is an *even* function, the spectrum is dominated by **even**
harmonics (2f, 4f, 6f, ...), the opposite harmonic signature of the symmetric
soft-clippers (diode, Tube Screamer) which produce mostly odd harmonics. The
defining feature is the razor-sharp **cusp** at every zero crossing: the transfer
curve ``|v|`` has a derivative discontinuity at the origin (slope flips from
``-g`` to ``+g`` instantaneously), an infinitely sharp corner that no analytic
real circuit clipper has.

Implementation. Rather than a literal diode bridge (which adds two diode drops of
dead-zone and a fragile floating mid-node), the rectifier is a **behavioral
precision-rectifier**: a single ``B``-source evaluating ``abs(g * V(in))`` — the
idealized op-amp precision full-wave rectifier of the real pedal, with a perfect
cusp and no diode dead-band. One real **RC pole** after it supplies the analog
"memory" (it slightly rounds the cusp and rolls off the highest fold harmonics,
exactly as the real circuit's stray capacitance does). Because ``abs(.)`` has a
large positive DC component (the mean of a rectified sine is ``2/pi`` of its
peak), the stage is **AC-coupled** at the output (series cap + leak resistor) so
the operating point and the delivered signal are both centred on the virtual
ground (node ``0``); at the DC solve ``Vin = 0`` so ``abs(0) = 0`` and every node
sits at exactly 0 V, making the op-point trivial and the transient robust.

Why this is a hard case for a smooth neural emulator. A neural net (TCN / GRU /
the project's smooth folds + FiLM) is a composition of *smooth, Lipschitz*
maps, so its input->output transfer is everywhere differentiable. The rectifier's
cusp is a derivative discontinuity: to reproduce it the net must synthesize an
arbitrarily sharp corner from smooth pieces, which (a) needs very high effective
bandwidth right where the signal spends the most time (near the zero crossing of
a guitar note's decaying tail), and (b) generates strong, slowly-decaying even
harmonics that alias badly under the smooth model's limited internal sample rate.
The even-harmonic / octave-up signature is also categorically different from the
odd-harmonic clippers, so a model tuned on those generalizes poorly here.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# --- fixed network constants ------------------------------------------------
# Memory low-pass corner after the rectifier. Kept high (~7.2 kHz) so the cusp
# stays sharp / the octave-up harmonics survive, while still adding one real
# analog pole of state.  C_mem * R_mem = 22 nF * 1 kOhm -> f_c ~ 7.2 kHz.
_R_MEM = 1.0e3
_C_MEM = 22.0e-9

# Output AC-coupling high-pass. abs(.) injects a large DC term; this removes it
# so the output is centred on 0 V. C_hp * R_hp = 1 uF * 10 kOhm -> f_c ~ 16 Hz,
# safely below the 2f octave-up fundamental of even the lowest guitar notes.
_C_HP = 1.0e-6
_R_HP = 10.0e3

# Drive defaults / range (pre-gain into the rectifier core).
_DRIVE_LO = 0.1
_DRIVE_HI = 4.0
_DRIVE_DEFAULT = 1.0


@register_circuit
class FullWaveRectifier(Circuit):
    """Behavioral precision full-wave rectifier (octave-up) + 1 RC pole + AC out.

    Topology (ground / virtual ground ``0``, input ``Vin``, output ``out``)::

        Vin in 0                         ; runner overrides dc per sample
        Rin in 0 1meg                    ; DC path for the input node
        Brect rraw 0 V = abs(G * V(in))  ; precision full-wave rectifier (the cusp)
        Rs   rraw rect 100               ; tiny series R -> well-posed B-source
        Rmem rect mem  1k                ; }
        Cmem mem  0    22n               ; } one real RC memory pole (~7 kHz)
        Chp  mem  out  1u                ; AC-couple: strip the abs() DC bias
        Rhp  out  0    10k               ; leak: DC path + high-pass return (~16 Hz)

    ``G`` is the pre-gain set by the ``drive`` control. The rectified node carries
    a large positive DC offset (~``2/pi`` of the peak); the output high-pass
    removes it, so ``out`` is a centred octave-up signal swinging about 0 V.
    """

    name = "fullwave_rectifier"
    description = (
        "Full-wave rectifier / octave-up stage (Green Ringer / Octavia core): "
        "behavioral precision rectifier abs(g*in) with a sharp zero-crossing cusp "
        "and even-harmonic / octave-up spectrum, + one RC memory pole; AC-coupled. "
        "drive = pre-gain into the rectifier."
    )
    # abs(.) is hard everywhere, but the octave-up character is most musical when
    # the input is a healthy fraction of a volt; ~0.5 V peak gives a clear,
    # well-folded octave with the default drive.
    nominal_drive_v = 0.5
    controls = (
        ControlSpec("drive", "continuous", _DRIVE_LO, _DRIVE_HI, _DRIVE_DEFAULT, "pregain"),
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        # No netlist-mode controls (drive is a pregain); validate/ignore params.
        self._resolve_netlist_params(params)
        return (
            "* Full-wave rectifier / octave-up (precision abs() + 1 RC pole, AC out)\n"
            "Vin in 0 dc 0\n"
            "Rin in 0 1meg\n"
            # Precision full-wave rectifier: the |.| folds both half-cycles to the
            # same sign -> octave-up, even harmonics, sharp cusp at every zero.
            "Brect rraw 0 V = abs(1.0 * V(in))\n"
            "Rs rraw rect 100\n"
            # One real RC pole of analog memory (gently rounds the cusp / rolls off
            # the very top fold harmonics).
            f"Rmem rect mem {_R_MEM:g}\n"
            f"Cmem mem 0 {_C_MEM:g}\n"
            # AC-couple the output to strip the large abs() DC term -> centred on 0 V.
            f"Chp mem out {_C_HP:g}\n"
            f"Rhp out 0 {_R_HP:g}\n"
        )