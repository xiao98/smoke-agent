"""Solver-level target selection.

Foam-Agent selects an OpenFOAM *variant* through ``openfoam_target``.  This
module lifts that choice one level up: a *solver* target.  Targets whose name
starts with ``fds`` are handled by :mod:`fds.target`; everything else is routed
unchanged to :mod:`openfoam_target`, so existing OpenFOAM behaviour is
untouched.

Config field: ``solver_target`` (env ``FOAMAGENT_SOLVER_TARGET``).  Empty means
"use the OpenFOAM path exactly as before".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import openfoam_target as _of

FDS_611 = "fds-6.11"
FDS_TARGETS = frozenset({FDS_611})


def normalise_solver_target(value: str | None) -> str:
    target = (value or "").strip().lower()
    if not target:
        return ""
    if target in FDS_TARGETS:
        return target
    # Anything else must be a valid OpenFOAM explicit target.
    return _of.normalise_openfoam_target(target)


def configured_solver_target(config: Any) -> str:
    return normalise_solver_target(getattr(config, "solver_target", ""))


def runtime_target(config: Any) -> str:
    """Active target name: an fds-* name, or the OpenFOAM runtime target."""
    target = configured_solver_target(config)
    if target:
        return target
    return _of.runtime_openfoam_target(config)


def is_fds(config: Any) -> bool:
    return configured_solver_target(config).startswith("fds")


def database_path(config: Any) -> Path:
    if is_fds(config):
        configured_path = getattr(config, "database_path", None)
        if not configured_path:
            configured_path = Path(__file__).resolve().parent.parent / "database"
        return Path(configured_path).expanduser().resolve() / configured_solver_target(config)
    return _of.database_path_for_config(config)


def require_corpus(config: Any) -> None:
    if is_fds(config):
        from fds.target import require_corpus as _require_fds

        _require_fds(config)
        return
    _of.require_target_corpus(config)
