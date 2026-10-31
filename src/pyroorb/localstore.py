"""The ONE place local persistent storage lives: ``$LOCAL_PERSISTENT_FOLDER``.

By design, anything PyRoOrb keeps on local disk that must SURVIVE a
restart goes under one folder named by one project-wide env var, so we stop
inventing a new ``PYROORB_<FEATURE>_PATH`` for every feature that needs a file.
Each feature takes a subfolder (``persistent_dir("monitor")``) and owns what is
inside it.

  * On Railway that folder must be a mounted VOLUME (e.g. ``/data``) -- without
    one a redeploy wipes it exactly like RAM, and nothing here can tell.
  * Unset, it defaults to ``datadirs.persistent_default()``: a development
    checkout's existing ``.local_persistent/``, else the per-user data directory.

What does NOT belong here: caches that are meant to be thrown away (the fit
payload cache ``web/fitdisk.py`` refuses this folder for that reason), and the
large downloaded data (the ephemeris, debias tables and fetch caches), which are
re-derivable and live under ``datadirs.cache_dir()``.
"""
import os
from pathlib import Path

ENV = "LOCAL_PERSISTENT_FOLDER"
from pyroorb import datadirs

DEFAULT = datadirs.persistent_default()


def root():
    """The persistent root as a resolved Path (not created)."""
    try:
        from pyroorb.itf.dbio import load_dotenv      # same .env convention as every flag
        load_dotenv()
    except Exception:
        pass
    want = os.environ.get(ENV, "").strip()
    return Path(os.path.realpath(want)) if want else DEFAULT


def persistent_dir(feature):
    """``<root>/<feature>``, created on demand. Raises OSError when it cannot be
    created -- a caller that can live without persistence catches it and says so."""
    if not feature or "/" in feature or feature.startswith("."):
        raise ValueError(f"bad feature name {feature!r}")
    d = root() / feature
    d.mkdir(parents=True, exist_ok=True)
    return d
