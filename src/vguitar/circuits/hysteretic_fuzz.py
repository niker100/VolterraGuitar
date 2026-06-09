"""Bias-starved, strongly HYSTERETIC fuzz (Fuzz Face / gated-fuzz character).

The hardest member of the stress-test set: a fuzz whose static transfer curve is
**not a curve at all** but a wide, drive-dependent **hysteresis loop**. On a real
germanium Fuzz Face (and on "dying-battery"/gated fuzzes) the second stage runs
*bias-starved* — its operating point is set by a slow RC that charges from the
signal's own envelope. As the signal gets louder the stage robs its own bias,
the operating point sags, and the clipping thresholds drift; when the signal
quiets the bias recovers only slowly. The result is **path dependence**: the
output for a given instantaneous input depends on what the input was doing
*milliseconds ago*, so rising and falling halves of a cycle trace different
clipping limits and the transfer plot opens into a loop. Near silence the
starved bias collapses the gain entirely — the "gating"/sputter that makes these
fuzzes spit and stutter on note decays.

Topology (ground / virtual-ground ``0``, input ``Vin``, output ``out``)::

    in -C1- np                      ; AC-couple onto the non-inverting input
    np -Rbias- 0                    ; DC path for the input node
    Eop = 1e5*(np - nm)             ; ideal op-amp (VCVS) + Roa series R
    oa -Rf- nm -Rg- 0               ; non-inverting gain 1 + Rf/Rg (high)
    Benv env = tanh(|oa|)           ; saturated rectified envelope of the signal
    Bchg -> vb (one-way) + Cb + Rb  ; fast-attack / slow-release SLOW bias node
    Bclip out0 = gate(vb)*asym(oa,vb); gate + asymmetric squash both keyed to the
                                      SLOW, lagging bias node vb  <-- the loop
    out0 -Cdc- out -Rload- 0        ; DC-block so 'out' is centred on 0 V

Why the slow node makes a *loop* and not just a curve: ``Bclip`` saturates the
positive half of ``oa`` symmetrically but squashes the negative half harder as
the bias node ``vb`` rises (a ``1/(1 + k*vb)`` factor), and multiplies the whole
thing by a smooth gate ``tanh(vb/thr)`` that only opens once the bias has
charged. Because ``vb`` is a fast-attack / slow-release envelope follower it
*lags* the signal: at a fixed ``oa`` the negative-half squash and the gate differ
depending on whether ``vb`` is still rising (attack) or decaying (release) — the
defining signature of hysteresis. ``vb``'s steady level itself is set by the
charge/leak balance ``Rb/(Rb + Rcharge)``, so the ``bias`` knob also dials how
*starved* the stage runs.

Convergence / DC-op notes:

* Every node has a DC path to ``0`` (``Rbias``, ``Rg``, ``Rb``, ``Rload``).
* At the DC operating point the input is 0, the rectified envelope is 0, so
  ``vb = 0`` and ``out0 = 0`` — the solve starts cleanly at zero, no UIC needed.
* All nonlinearities are *smooth* behavioral ``tanh``/``sqrt`` expressions (no
  ideal switches or hysteretic primitives that stall); the macroscopic
  hysteresis is an emergent property of the slow RC **state**, not of any
  per-instant discontinuity, so the solver stays well-posed.
* High-gain ``B``/``E`` sources are followed by a tiny series resistor.

Controls:

* ``drive`` — input pre-gain into the starved stage. Low: nearly clean and
  gated-off; high: fully into the asymmetric clip + wide loop. Gives amplitude
  coverage and, because the bias node tracks the envelope, also *reshapes* the
  loop with level (a genuinely level-dependent memory effect).
* ``bias`` — normalized ``0..1`` knob mapped to the bias-recovery resistor
  ``Rb`` (and thus the RC time constant): ``bias=0`` is healthy/fast-recovery
  (tighter loop, less gating), ``bias=1`` is "dying-battery" slow recovery
  (very wide loop, heavy sputter/gating). This directly dials the *amount* of
  memory, which is the whole point of the circuit.
"""

from __future__ import annotations

from vguitar.circuits.base import Circuit, ControlSpec, register_circuit

# Bias-recovery (leak) resistor range (ohms), in balance with the Rchg=1M charge
# path and the Cb=22n bias cap. Two things scale with this resistor:
#   * the RELEASE time constant Rb*Cb (how long the loop remembers):
#       bias=0 -> 1M*22n   ~ 22 ms  (fast recovery, tighter loop, healthy)
#       bias=1 -> 20M*22n  ~ 440 ms (slow recovery, wide loop, dying-battery)
#   * the STEADY bias level vb = Rb/(Rb+Rchg): bias=0 -> ~0.50 (starved, gates
#     hard / sputters), bias=1 -> ~0.95 (conducting, deep asymmetric clip).
# So "bias" dials both the amount of memory AND how starved the stage runs.
_RB_HEALTHY = 1_000_000.0
_RB_STARVED = 20_000_000.0


@register_circuit
class HystereticFuzz(Circuit):
    """Bias-starved fuzz with a wide, drive-dependent hysteresis loop.

    A high-gain op-amp stage whose asymmetric hard clip and low-level gate are
    modulated by a SLOW envelope-tracking bias node. The lag of that node behind
    the signal envelope makes the input->output map path-dependent (a hysteresis
    loop, not a curve) and produces gating/sputter near silence — a memoryful,
    sharply nonlinear target. Controls: ``drive`` (pre-gain) and ``bias``
    (recovery time / amount of memory).
    """

    name = "hysteretic_fuzz"
    description = (
        "Bias-starved Fuzz Face-style fuzz: high-gain stage with asymmetric hard "
        "clipping and a low-level gate modulated by a slow envelope-tracking bias "
        "node, giving a wide drive-dependent hysteresis loop + sputter/gating; "
        "drive (pre-gain) + bias (recovery time / memory amount)."
    )
    # Strong stage gain (~ x47), so it clips and starves its bias at small inputs.
    nominal_drive_v = 0.15
    controls = (
        ControlSpec("drive", "continuous", 0.05, 1.0, 0.3, "pregain"),
        ControlSpec("bias", "continuous", 0.0, 1.0, 0.6, "netlist"),
    )

    _TEMPLATE = (
        "* Bias-starved hysteretic fuzz (Fuzz Face / gated-fuzz)\n"
        "Vin in 0 dc 0\n"
        # --- input coupling + high-gain op-amp stage ----------------------------
        "C1 in np 0.047u\n"
        "Rbias np 0 1meg\n"  # DC path for the non-inverting input
        "Eop oaraw 0 np nm 1e5\n"  # ideal op-amp
        "Roa oaraw oa 100\n"  # series R so the feedback loop is well-posed
        "Rg nm 0 1k\n"  # gain set: 1 + Rf/Rg = 1 + 47 = ~48x
        "Rf oa nm 47k\n"
        # --- SLOW envelope-tracking bias node (the memory / state) --------------
        # Rectify the amplified signal and SATURATE it to a bounded 0..1 drive
        # (tanh of |oa|) so the bias node stays O(1 V) at any input level -- the
        # asymmetry it controls must be a bounded *shift*, never an unbounded
        # gain. The bias node is an ENVELOPE FOLLOWER with FAST ATTACK / SLOW
        # RELEASE: a one-way behavioral charge current (only when env > vb)
        # charges Cb quickly through Rcharge, while the bias-recovery resistor Rb
        # discharges it slowly. So vb snaps up on a note attack but RECOVERS
        # slowly afterwards -> it LAGS the falling envelope by tens of ms ->
        # path dependence. Rb (the bias knob) sets the recovery time = how long
        # the loop remembers. max(0,.) is smooth-enough for the solver; Rb gives
        # vb a DC path to ground.
        "Benv env 0 V=tanh(sqrt(V(oa)*V(oa) + 1e-4)/0.5)\n"
        "Bchg 0 vb I=max(0, V(env)-V(vb))/1meg\n"  # one-way attack charge (Rchg=1M)
        "Cb vb 0 22n\n"  # bias storage cap (the slow state); attack tau ~ 22 ms
        "Rb vb 0 {rb:g}\n"  # bias-recovery (leak) resistor: sets memory + starve
        # --- asymmetric hard clip + gate, MODULATED by the slow bias node -------
        # Symmetric saturation core (+/-1 V via tanh), then a BOUNDED, DEEP
        # asymmetry: the negative half is squashed hard as the bias node vb rises
        # (divisor 1 + 2.5*vb), so the lower excursion collapses while the bias
        # is charged and slowly returns as vb recovers. Because vb LAGS the
        # envelope (fast attack / slow release), the rising and falling halves of
        # the note see different vb -> the input->output transfer plot opens into
        # a wide LOOP, and the half-wave asymmetry itself is path-dependent.
        # The GATE is driven by the SLOW node vb (gate = tanh(vb/0.45)): the
        # stage only conducts once the bias has charged, so on a note attack it
        # SPUTTERS on and on decay it gates off *late* (lingering) -- memoryful
        # gating, the dying-battery character. tanh/ternary keep every instant
        # smooth & convergent; the macroscopic hysteresis is emergent from vb's
        # lag, not from any switch. Output is bounded to ~1 V at any drive.
        "Bclip out0 0 V="
        "tanh(V(vb)/0.45) * "
        "( (V(oa) >= 0) ? "
        "tanh(V(oa)/0.8) : "
        "(tanh(V(oa)/0.8) / (1 + 2.5*V(vb))) )\n"
        "Rc out0 oc 100\n"  # tiny series R after the high-gain B-source
        "Rocl oc 0 1meg\n"  # DC path for the clip-output node
        # --- DC block so 'out' is centred on 0 V (~16 Hz corner) ----------------
        # The asymmetry injects a small DC term; a 0.1uF/100k blocker (~16 Hz)
        # removes it well within an audio window while passing the guitar band.
        "Cdc oc out 0.1u\n"
        "Rload out 0 100k\n"
    )

    def netlist(self) -> str:
        return self.netlist_for(None)

    def netlist_for(self, params: dict[str, float] | None = None) -> str:
        p = self._resolve_netlist_params(params)
        # bias 0..1 -> recovery (leak) resistor. Linear over this range is
        # monotone and converges fine; larger Rb = slower recovery = wider loop.
        rb = _RB_HEALTHY + p["bias"] * (_RB_STARVED - _RB_HEALTHY)
        return self._TEMPLATE.format(rb=rb)
