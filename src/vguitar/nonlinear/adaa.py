"""Antiderivative anti-aliasing (ADAA) for memoryless nonlinearities.

When a static nonlinearity ``f`` is applied sample-by-sample it folds the
harmonics it creates back below Nyquist as audible aliasing. The classic cure is
to oversample heavily, run ``f``, then decimate — but the oversampling factor has
to be large (and therefore expensive) to push aliases below the noise floor.

ADAA instead replaces the pointwise evaluation with a short, closed-form average
of ``f`` over the segment between consecutive (interpolated) input samples, which
it computes from the *antiderivatives* of ``f``. This continuous-domain averaging
acts as an anti-imaging filter built into the nonlinearity itself. The headline
empirical result is that **first-order ADAA at 2x oversampling, and second-order
ADAA at ~4x, suppress aliasing about as well as 6x-12x plain oversampling** — i.e.
comparable aliasing for a fraction of the cost.

References:

* J. Parker, V. Zavalishin, E. Le Bivic, "Reducing the aliasing of nonlinear
  waveshaping using continuous-time convolution," DAFx-16 (first-order ADAA).
* S. Bilbao, F. Esqueda, J. Parker, V. Valimaki, "Antiderivative antialiasing
  for memoryless nonlinearities," IEEE Signal Processing Letters 24(7), 2017
  (the general k-th order scheme; second-order used here).

ADAA only applies to **static (memoryless) nonlinearities** ``y = f(x)``. In this
package it is the waveshaper used inside the Wiener-Hammerstein block-oriented
model and the realtime processing path, where the linear filters carry the memory
and ADAA cleans up the saturating nonlinearity in between.

The ``ADAAProcessor`` keeps the inter-block state so streaming ``process_block``
calls produce exactly the same result as one offline pass over the concatenated
signal (the model streaming contract).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import spence  # real dilogarithm: spence(z) = Li2(1 - z)

#: Below this |dx| the divided-difference forms below are numerically unstable
#: (catastrophic cancellation), so we fall back to a direct/midpoint evaluation.
#: float32 audio has ~1e-7 resolution; this threshold sits safely above it.
_TINY = 1e-5


@dataclass(frozen=True)
class NL:
    """A static nonlinearity ``f`` bundled with its first two antiderivatives.

    All callables must be vectorized over numpy arrays and act elementwise.

    Attributes:
        name: Short identifier (e.g. ``"tanh"``).
        f: The nonlinearity ``f(x)``.
        F1: First antiderivative, ``F1' = f`` (used by first-order ADAA).
        F2: Second antiderivative, ``F2' = F1`` (used by second-order ADAA).
    """

    name: str
    f: Callable[[np.ndarray], np.ndarray]
    F1: Callable[[np.ndarray], np.ndarray]
    F2: Callable[[np.ndarray], np.ndarray]


# --- predefined nonlinearities -------------------------------------------


def _tanh_F1(x: np.ndarray) -> np.ndarray:
    """First antiderivative of tanh: ``log(cosh x)``.

    Computed via ``|x| + log1p(exp(-2|x|)) - log 2`` to stay finite for large
    ``|x|`` (``cosh`` overflows; this form does not).
    """
    a = np.abs(x)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


#: Constant making the closed-form tanh F2 below satisfy F2(0) = 0.
#: F2(0) = 0 - 0 + (1/2)·Li2(-1); subtract it so antiderivatives pass through 0.
_TANH_F2_C = 0.5 * spence(1.0 - (-1.0))  # = 0.5 * Li2(-1) = -pi^2/24


def _tanh_F2(x: np.ndarray) -> np.ndarray:
    """Second antiderivative of tanh, in closed form via the dilogarithm.

    Integrating ``F1(x) = log(cosh x) = x - log 2 + log(1 + e^{-2x})`` term by
    term and using ``∫ log(1 + e^{-2x}) dx = (1/2)·Li2(-e^{-2x}) + C`` gives

        ``F2(x) = x^2/2 - x·log 2 + (1/2)·Li2(-e^{-2x})``  (then shifted so
        ``F2(0) = 0``),

    where the real dilogarithm ``Li2(w) = spence(1 - w)``. This form is exact and
    numerically stable for large ``|x|`` (it grows like ``x^2/2``), unlike a
    direct ``cosh`` evaluation which would overflow.
    """
    li2 = spence(1.0 + np.exp(-2.0 * x))  # Li2(-e^{-2x}) = spence(1 - (-e^{-2x}))
    return 0.5 * x * x - np.log(2.0) * x + 0.5 * li2 - _TANH_F2_C


TANH = NL(name="tanh", f=np.tanh, F1=_tanh_F1, F2=_tanh_F2)


def _hardclip_f(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


def _hardclip_F1(x: np.ndarray) -> np.ndarray:
    """First antiderivative of clip(x,-1,1).

    Quadratic ``x^2/2`` in the linear region |x|<=1, linear ``|x|-1/2`` outside
    (continuous and C1 at the breakpoints).
    """
    a = np.abs(x)
    inside = a <= 1.0
    return np.where(inside, 0.5 * x * x, a - 0.5)


def _hardclip_F2(x: np.ndarray) -> np.ndarray:
    """Second antiderivative of clip(x,-1,1) (continuous value and slope).

    ``f`` is odd, so ``F1`` is even and ``F2`` is odd. Integrating ``F1``:

    * ``|x| <= 1``: ``F2 = x^3/6``  (matches ``F2(1) = 1/6``).
    * ``x >  1``:   ``F2 = x^2/2 - x/2 + 1/6``  (continuous at ``x = 1``).
    * ``x < -1``:   ``F2 = -F2(-x)`` by odd symmetry.

    The outer branch is written with ``sign(x)`` so both ``|x| > 1`` cases use one
    expression: ``sign(x)·(x^2/2 - |x|/2 + 1/6)``.
    """
    a = np.abs(x)
    inside = a <= 1.0
    inside_val = x * x * x / 6.0
    outside_val = np.sign(x) * (0.5 * a * a - 0.5 * a + 1.0 / 6.0)
    return np.where(inside, inside_val, outside_val)


HARDCLIP = NL(name="hardclip", f=_hardclip_f, F1=_hardclip_F1, F2=_hardclip_F2)


# Asymmetric soft clip standing in for a single shunt diode to ground: the two
# polarities saturate at different rates, so the curve is asymmetric and produces
# the even-harmonic-rich character of a one-diode shaper. Rather than the literal
# Shockley exp() (which overflows and has no elementary antiderivative), we use a
# two-gain tanh: f(x) = tanh(k_p·x) for x >= 0 and tanh(k_n·x) for x < 0. tanh is
# smooth, bounded, and its antiderivatives are the same stable closed forms as
# TANH above (scaled by 1/k and 1/k^2), keeping ADAA cheap and exact. The gentler
# positive gain and steeper negative gain set the asymmetry.
_KP = 1.0  # positive-side gain (gentle saturation)
_KN = 3.0  # negative-side gain (steeper -> more asymmetry)


def _diode_f(x: np.ndarray) -> np.ndarray:
    """Asymmetric soft clip: gentle tanh on +x, steeper tanh on -x.

    The two polarities saturate at different rates, giving the even-harmonic-rich,
    asymmetric character of a single-diode shaper. Continuous and smooth at 0
    (both branches and their slopes agree there up to the gain ratio).
    """
    pos = np.tanh(_KP * x)
    neg = np.tanh(_KN * x)
    return np.where(x >= 0.0, pos, neg)


def _diode_F1(x: np.ndarray) -> np.ndarray:
    """First antiderivative of the asymmetric tanh shaper (piecewise log cosh).

    Each branch integrates ``tanh(k x)`` to ``log(cosh(k x))/k``; the constant
    is fixed so the two branches meet at ``x = 0`` (both give 0 there).
    """
    pos = _tanh_F1(_KP * x) / _KP
    neg = _tanh_F1(_KN * x) / _KN
    return np.where(x >= 0.0, pos, neg)


def _diode_F2(x: np.ndarray) -> np.ndarray:
    """Second antiderivative of the asymmetric shaper.

    Integrates each ``log(cosh(k x))/k`` branch using the stable dilogarithm
    form from :func:`_tanh_F2` (scaled by ``1/k^2``). The branches and an
    additive constant are matched so the result is continuous at ``x = 0``;
    second-order ADAA uses only differences of ``F2`` so any residual constant
    cancels, but we match anyway for cleanliness.
    """
    pos = _tanh_F2(_KP * x) / (_KP * _KP)
    neg = _tanh_F2(_KN * x) / (_KN * _KN)
    # Match branch values at x=0 so the piecewise F2 is continuous there.
    pos0 = _tanh_F2(np.array(0.0)) / (_KP * _KP)
    neg0 = _tanh_F2(np.array(0.0)) / (_KN * _KN)
    return np.where(x >= 0.0, pos - float(pos0), neg - float(neg0))


DIODE = NL(name="diode", f=_diode_f, F1=_diode_F1, F2=_diode_F2)


#: All predefined nonlinearities, keyed by ``name`` for convenient lookup.
NONLINEARITIES: dict[str, NL] = {nl.name: nl for nl in (TANH, HARDCLIP, DIODE)}


# --- first-order ADAA -----------------------------------------------------


def adaa1(nl: NL, x: np.ndarray, x_prev: float) -> tuple[np.ndarray, float]:
    """First-order ADAA over a block (Parker et al., DAFx-16).

    For each sample the output is the divided difference of the first
    antiderivative,

        ``y[n] = (F1(x[n]) - F1(x[n-1])) / (x[n] - x[n-1])``,

    which equals the average of ``f`` over ``[x[n-1], x[n]]`` and removes most
    of the aliasing energy a direct ``f(x[n])`` would create. Where the input
    barely moves (``|x[n] - x[n-1]| < _TINY``) the divided difference loses all
    precision, so we fall back to evaluating ``f`` at the segment midpoint, its
    exact limit.

    Args:
        nl: The nonlinearity (needs ``f`` and ``F1``).
        x: Input block, 1-D.
        x_prev: Last input sample of the previous block (the streaming state).

    Returns:
        ``(y, new_x_prev)`` where ``y`` is the float64 output block (same length
        as ``x``) and ``new_x_prev`` is ``x[-1]`` for the next call. An empty
        ``x`` returns unchanged ``x_prev``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy(), x_prev
    x0 = np.empty_like(x)
    x0[0] = x_prev
    x0[1:] = x[:-1]  # x[n-1] for every n
    dx = x - x0
    small = np.abs(dx) < _TINY
    # Safe divisor avoids 0/0 warnings; masked-out lanes are overwritten below.
    safe_dx = np.where(small, 1.0, dx)
    y = (nl.F1(x) - nl.F1(x0)) / safe_dx
    y = np.where(small, nl.f(0.5 * (x + x0)), y)
    return y, float(x[-1])


# --- second-order ADAA ----------------------------------------------------


def adaa2(
    nl: NL, x: np.ndarray, state: tuple[float, float]
) -> tuple[np.ndarray, tuple[float, float]]:
    """Second-order ADAA over a block (Bilbao et al., IEEE SPL 2017, eq. for k=2).

    The second-order kernel forms a symmetric divided difference of the second
    antiderivative ``F2`` across three consecutive samples ``x[n-2..n]``:

        ``y[n] = 2/(x[n] - x[n-2]) · ( (F2(x[n]) - F2(x[n-1]))/(x[n]-x[n-1])
                                       - (F2(x[n-1]) - F2(x[n-2]))/(x[n-1]-x[n-2]) )``.

    This is the discrete analogue of a second-order continuous-time convolution
    kernel and rolls aliasing off faster than first order, so ~4x oversampling
    with ``adaa2`` rivals heavy plain oversampling. The output is aligned so that
    a sequence of blocks reproduces an offline pass exactly; with the two-sample
    memory it carries one sample of group delay relative to ``adaa1``.

    Degenerate cases — any pair of the three samples within ``_TINY`` — are
    handled by falling back to the first-order divided difference (and, when both
    gaps vanish, to the midpoint value), the analytic limits of the formula.

    Args:
        nl: The nonlinearity (needs ``f``, ``F1`` and ``F2``).
        x: Input block, 1-D.
        state: ``(x[-2], x[-1])`` from the previous block — the two most recent
            past inputs, oldest first.

    Returns:
        ``(y, new_state)`` with ``y`` the float64 output (same length as ``x``)
        and ``new_state`` the last two samples for the next call.
    """
    x = np.asarray(x, dtype=np.float64)
    x_m2_last, x_m1_last = state
    if x.size == 0:
        return x.copy(), state

    # Prepend the two past samples, then take three shifted views so that for
    # every n: x2 = x[n], x1 = x[n-1], x0 = x[n-2]. Works for any block size.
    padded = np.concatenate(([x_m2_last, x_m1_last], x))
    x2 = padded[2:]
    x1 = padded[1:-1]
    x0 = padded[:-2]

    y = _adaa2_kernel(nl, x2, x1, x0)
    new_state = (float(padded[-2]), float(padded[-1]))  # last two of (state + x)
    return y, new_state


def _first_diff(nl: NL, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """``(F2(xa) - F2(xb))/(xa - xb)`` with the midpoint-of-F1 fallback.

    This is the inner divided difference of the second-order kernel; where the
    two samples coincide it tends to ``F1`` evaluated at the shared point.
    """
    d = xa - xb
    small = np.abs(d) < _TINY
    safe = np.where(small, 1.0, d)
    val = (nl.F2(xa) - nl.F2(xb)) / safe
    return np.where(small, nl.F1(0.5 * (xa + xb)), val)


def _adaa2_kernel(nl: NL, x2: np.ndarray, x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Second-order ADAA output from three aligned sample vectors.

    Implements the symmetric divided difference of ``F2``; falls back to
    first-order ADAA between the outer samples when the outer gap collapses.
    """
    outer = x2 - x0
    small_outer = np.abs(outer) < _TINY
    safe_outer = np.where(small_outer, 1.0, outer)
    hi = _first_diff(nl, x2, x1)
    lo = _first_diff(nl, x1, x0)
    y = 2.0 * (hi - lo) / safe_outer
    # When x2 ~= x0 the kernel is undefined; fall back to first-order ADAA
    # across the outer pair, itself bottoming out at the midpoint value.
    d_o = x2 - x0
    small_d = np.abs(d_o) < _TINY
    safe_d = np.where(small_d, 1.0, d_o)
    fo = (nl.F1(x2) - nl.F1(x0)) / safe_d
    fo = np.where(small_d, nl.f(0.5 * (x2 + x0)), fo)
    return np.where(small_outer, fo, y)


# --- stateful streaming processor -----------------------------------------


class ADAAProcessor:
    """Stateful ADAA waveshaper for block-by-block (realtime) processing.

    Wraps :func:`adaa1` / :func:`adaa2` and carries the small inter-block memory
    so that ``reset()`` followed by a sequence of ``process_block`` calls over a
    signal yields exactly the same output as one offline ADAA pass (the streaming
    contract used throughout the package).

    Args:
        nl: The static nonlinearity to apply.
        order: ADAA order, ``1`` (DAFx-16) or ``2`` (Bilbao 2017). Higher order
            suppresses aliasing more for a given oversampling factor at the cost
            of one extra sample of memory.
    """

    def __init__(self, nl: NL, order: int = 1) -> None:
        if order not in (1, 2):
            raise ValueError(f"ADAA order must be 1 or 2, got {order!r}")
        self.nl = nl
        self.order = order
        self._x_prev: float = 0.0
        self._state2: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        """Clear inter-block state (treat the next sample as following silence)."""
        self._x_prev = 0.0
        self._state2 = (0.0, 0.0)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Apply ADAA to one block, advancing state; returns same-length float32."""
        if self.order == 1:
            y, self._x_prev = adaa1(self.nl, x, self._x_prev)
        else:
            y, self._state2 = adaa2(self.nl, x, self._state2)
        return y.astype(np.float32)
