"""Circuit registry + auto-discovery (see :mod:`vguitar.models` for the pattern)."""

from __future__ import annotations

import importlib
import pkgutil
import warnings

from vguitar.circuits.base import (
    Circuit,
    all_circuits,
    get_circuit,
    register_circuit,
)

__all__ = ["Circuit", "all_circuits", "get_circuit", "register_circuit"]

_SKIP = {"base"}


def _autodiscover() -> None:
    for info in pkgutil.iter_modules(__path__):
        if info.name in _SKIP or info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{info.name}")
        except Exception as exc:
            warnings.warn(f"could not import circuit module {info.name!r}: {exc}", stacklevel=2)


_autodiscover()
