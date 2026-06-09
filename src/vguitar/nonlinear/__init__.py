"""Static-nonlinearity tools: ADAA (antiderivative-antialiased) waveshapers.

This subpackage holds the memoryless waveshapers and their alias-suppressing
evaluation used by the block-oriented (Wiener-Hammerstein) model and the realtime
path (see :mod:`vguitar.nonlinear.adaa`).
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
