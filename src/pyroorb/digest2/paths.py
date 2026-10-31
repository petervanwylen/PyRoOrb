"""Default file location discovery for the population model and
observatory codes table, mirroring digest2's own "look in a few
sensible places" behavior (``d2cli.c``'s ``-p``/``CPspec`` plus this
package's bundled ``data/`` directory as a first-class source).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PKG_DATA_DIR = Path(__file__).parent / "data"


def _first_existing(*candidates: Optional[Path]) -> Optional[Path]:
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None


def find_model_path(search_dir: Optional[str] = None) -> str:
    """Locate digest2.model.csv (preferred) or a cached .npz model.

    Search order: $PYDIGEST2_MODEL, an explicit search_dir, the bundled
    package data/ directory, then the current working directory.
    """
    env = os.environ.get("PYDIGEST2_MODEL")
    if env and Path(env).is_file():
        return env

    dirs = [Path(search_dir)] if search_dir else []
    dirs += [_PKG_DATA_DIR, Path.cwd()]

    for d in dirs:
        found = _first_existing(d / "digest2.model.csv", d / "digest2.model.csv.npz")
        if found:
            return str(found)

    raise FileNotFoundError(
        "Cannot find digest2.model.csv. Set PYDIGEST2_MODEL or pass --model/-m, "
        "or place the file in the current directory."
    )


def find_obscodes_path(search_dir: Optional[str] = None) -> str:
    env = os.environ.get("PYDIGEST2_OBSCODES")
    if env and Path(env).is_file():
        return env

    dirs = [Path(search_dir)] if search_dir else []
    dirs += [_PKG_DATA_DIR, Path.cwd()]

    for d in dirs:
        found = _first_existing(d / "digest2.obscodes")
        if found:
            return str(found)

    raise FileNotFoundError(
        "Cannot find digest2.obscodes. Set PYDIGEST2_OBSCODES or pass --obscodes/-o, "
        "or place the file in the current directory."
    )


def find_mpc_config_path(search_dir: Optional[str] = None) -> Optional[str]:
    dirs = [Path(search_dir)] if search_dir else []
    dirs += [_PKG_DATA_DIR, Path.cwd()]
    found = _first_existing(*(d / "MPC.config" for d in dirs))
    return str(found) if found else None
