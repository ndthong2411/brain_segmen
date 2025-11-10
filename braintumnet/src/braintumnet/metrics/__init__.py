"""Metrics package exposing unified metric helpers."""

from __future__ import annotations

import types

from . import base as _base
from . import regions as _regions
from . import multiclass as _multiclass


def _export(module: types.ModuleType) -> list[str]:
    exported: list[str] = []
    names = getattr(module, "__all__", None) or dir(module)
    for name in names:
        if name.startswith("_"):
            continue
        value = getattr(module, name, None)
        if isinstance(value, types.ModuleType):
            continue
        globals()[name] = value
        exported.append(name)
    return exported


__all__ = []
for _module in (_base, _regions, _multiclass):
    __all__.extend(_export(_module))
