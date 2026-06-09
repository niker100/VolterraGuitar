"""West-coast (Serge/Lockhart-style) sine wavefolder - a folding waveshaper.

Where every other circuit in the set *clips* (the transfer curve flattens to a
plateau once the signal exceeds a threshold), a wavefolder does the opposite: as
the amplitude grows the transfer curve **turns back on itself**, again and again.
This is the defining gesture of West-coast / Buchla-Serge synthesis: a clean tone
fed in at low level passes nearly untouched, but as drive rises the waveform is
repeatedly *folded*, each new fold injecting a fresh layer of high-order
harmonics. The result is a harmonic stack whose richness and bandwidth grow
**dramatically and monotonically** with input level.

We model the canonical *sine* folder. The classic analog Lockhart/Serge cell is a
transistor-pair shaper whose large-signal transfer is very close to a sum of
sines of the (scaled) input; using a behavioral ``B``-source ``V = sum A_k *
sin(k * V_in)`` reproduces the multi-fold characteristic exactly while being
unconditionally convergent (it is a smooth, bounded, memoryless map - no diode
loops or stiff junctions to stall the Newton solve). Two sine terms at different
"rates" give a denser, less pure fold than a single sinusoid. A gentle output RC
pole (the ``tone`` control) tames the highest folded harmonics so the
band-limited target stays alias-free.

Conventions: the input is AC-coupled (``Cin`` + ``Rbias`` to node ``0``) so the
DC operating point is exactly ``sin(0) = 0`` - silence in gives silence out,
output centered on the virtual ground ``0``. Every node has a DC path to ground
(``Rbias``, ``Rload``), so the operating-point solve is trivial.

Controls:

* ``drive`` - input pre-gain into the folder. Low settings stay near-linear (one
  or two harmonics); high settings drive several folds (a tall harmonic stack),
  giving the dataset its amplitude coverage.
* ``tone`` - normalized ``0..1`` knob mapped (in :meth:`netlist_for`) to the
  output low-pass capacitor: bright/raw at ``0`` to dark/smooth at ``1``.

Why it is a hard case for a smooth neural emulator: a folder is the antithesis of
a saturating clipper. Its static transfer is **non-monotonic with many sign
reversals of slope**, and the number of those reversals (hence the harmonic
order) keeps increasing with amplitude, so the model cannot lean on the usual
"soft-saturation"/tanh-shaped inductive bias every clipper in the set rewards.
It must learn a high-curvature oscillatory map and, crucially, extrapolate how
*new* harmonics appear as drive grows past the levels seen in training - a
genuinely sharper, higher-order nonlinearity than diode/BJT/JFET/op-amp clipping.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# Output low-pass cap range (with the 1k series resistor Rs):
#   tone=0 -> 1 nF   (~1.6 MHz corner: effectively bypassed, fully bright/raw)
#   tone=1 -> 47 nF  (~3.4 kHz corner: dark, the highest folds rolled off)
_TONE_C_LO = 1.0e-9
_TONE_C_HI = 47.0e-9


@register_circuit
class WaveFolder(Circuit):
    """Behavioral sine wavefolder (multi-fold West-coast waveshaper) + tone pole.

    Topology (ground / virtual ground ``0``, input ``Vin``, output ``out``)::

        in -Cin- np -Rbias- 0                 ; AC-couple, DC path to ground
        Bfold f 0  V = 0.45*sin(3.4*V(np))    ; sum-of-sines fold characteristic
                     + 0.18*sin(9.0*V(np))    ;   (folds repeatedly as |np| grows)
        f -Rs- out -Cout- 0                    ; output series R + tone low-pass pole
        out -Rload- 0                          ; DC path / load

    Small-signal slope through the origin is ``0.45*3.4 + 0.18*9.0 ~= 3.15``, so
    low levels pass with mild gain and almost no harmonics; the first fold (output
    turnaround) occurs near ``|V(np)| ~ 0.5 V`` and successive folds follow every
    further ``~0.9 V``, stacking ever-higher harmonics with drive.
    """

    name = "wavefolder"
    description = (
        "West-coast (Serge/Lockhart-style) sine wavefolder: a sum-of-sines folding "
        "waveshaper whose harmonic stack grows with drive (folds, not clips); "
        "drive (pre-gain) + tone (output low-pass)."
    )
    # Peak input (volts) at which folding is clearly audible (a couple of folds).
    nominal_drive_v = 0.6
    controls = (
        ControlSpec("drive", "continuous", 0.05, 1.0, 0.6, "pregain"),
        ControlSpec("tone", "continuous", 0.0, 1.0, 0.5, "netlist"),
    )

    _TEMPLATE = (
        "* West-coast sine wavefolder (multi-fold waveshaper + tone pole)\n"
        "Vin in 0 dc 0\n"
        "Cin in np 1u\n"
        "Rbias np 0 1meg\n"
        "Bfold f 0 V=0.45*sin(3.4*V(np)) + 0.18*sin(9.0*V(np))\n"
        "Rs f out 1k\n"
        "Cout out 0 {tone_c:g}\n"
        "Rload out 0 100k\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        p = self._resolve_netlist_params(params)
        tone_c = _TONE_C_LO + p["tone"] * (_TONE_C_HI - _TONE_C_LO)
        return self._TEMPLATE.format(tone_c=tone_c)
