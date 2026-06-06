"""Static-nonlinearity tools: antiderivative anti-aliasing (ADAA).

This subpackage holds the memoryless waveshapers and their alias-suppressing
evaluation used by the block-oriented (Wiener-Hammerstein) model and the realtime
path. See :mod:`vguitar.nonlinear.adaa` for the method and references.
"""

from __future__ import annotations

from vguitar.nonlinear.adaa import (
    DIODE,
    HARDCLIP,
    NL,
    NONLINEARITIES,
    TANH,
    ADAAProcessor,
    adaa1,
    adaa2,
)

__all__ = [
    "DIODE",
    "HARDCLIP",
    "NL",
    "NONLINEARITIES",
    "TANH",
    "ADAAProcessor",
    "adaa1",
    "adaa2",
]
