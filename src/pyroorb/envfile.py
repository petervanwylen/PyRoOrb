"""The one settings file: a ``.env`` read by every PyRoOrb command at start-up.

Every setting PyRoOrb reads from the environment (data folders such as ``PYROORB_EPHEM_DIR``,
database credentials, feature flags) can live in one ``.env`` file of ``KEY=value`` lines.
Each command loads it FIRST, before any engine module is imported, because several settings
are read once at import (the ephemeris and debias folders among them): loaded any later,
they would be silently ignored.

Where the file is found, first match wins:

1. ``PYROORB_ENV_FILE`` -- an explicit path;
2. ``.env`` at the root of a source checkout;
3. ``.env`` in the folder the command runs from, or in any folder above it.

A variable already set in the real environment always wins over the file, so a host's own
settings (a container, a CI job) are never overridden. Blank lines and ``#`` comments are
skipped; values may be quoted.

Importing ``pyroorb`` as a library reads no ``.env`` up front, so a program using the Python
API keeps control of its own environment; only the database and persistent-storage helpers
still load the file on first use, for the credentials they need.
"""
import os
from pathlib import Path

ENV_FILE_VAR = "PYROORB_ENV_FILE"
_CHECKOUT = Path(__file__).resolve().parents[2]   # src/pyroorb/envfile.py -> checkout root
_LOADED = {"path": None, "done": False}


def candidates(cwd=None):
    """The files searched, in order (they need not exist)."""
    out = []
    explicit = os.environ.get(ENV_FILE_VAR, "").strip()
    if explicit:
        out.append(Path(explicit).expanduser())
    out.append(_CHECKOUT / ".env")
    here = Path(cwd or Path.cwd()).resolve()
    while True:
        out.append(here / ".env")
        if here.parent == here:
            break
        here = here.parent
    return out


def parse(text):
    """``KEY=value`` pairs from the file's text, in order."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        pairs.append((key.strip(), val.strip().strip('"').strip("'")))
    return pairs


def load_env_file(cwd=None, force=False):
    """Load the first ``.env`` found (see the module docstring) into ``os.environ`` without
    overriding variables already set. Returns the file's path, or None when there is none.
    Loads once per process unless ``force``."""
    if _LOADED["done"] and not force:
        return _LOADED["path"]
    found = None
    for path in candidates(cwd):
        if path.is_file():
            for key, val in parse(path.read_text()):
                os.environ.setdefault(key, val)
            found = path
            break
    _LOADED.update(path=found, done=True)
    return found
