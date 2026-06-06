"""Model registry + auto-discovery.

Any module dropped in this package that calls ``@register_model`` on a
:class:`Model` subclass is picked up automatically — no central edit needed.
A module that fails to import (e.g. work in progress) is skipped with a warning
so a single broken model can't take down the whole benchmark.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

from vguitar.models.base import (
    FitReport,
    Model,
    all_models,
    check_streaming,
    get_model,
    register_model,
)

__all__ = [
    "FitReport",
    "Model",
    "all_models",
    "check_streaming",
    "get_model",
    "register_model",
]

_SKIP = {"base"}


def _autodiscover() -> None:
    for info in pkgutil.iter_modules(__path__):
        if info.name in _SKIP or info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{info.name}")
        except Exception as exc:
            warnings.warn(f"could not import model module {info.name!r}: {exc}", stacklevel=2)


_autodiscover()
