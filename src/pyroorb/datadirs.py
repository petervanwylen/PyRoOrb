"""Where PyRoOrb keeps files on disk.

An installed package lives in a read-only directory that an upgrade replaces, so nothing
may be written beside the source. Two kinds of files are written, and each has one home:

* **Re-derivable downloads and caches** -- the DE440 ephemeris, the debias tables, fetched
  observations, published orbits, the MPCORB catalogue, the station table, the parsed
  digest2 model. They live under :func:`cache_dir` and can be deleted at any time.
* **State that must survive a restart** -- under :func:`persistent_default`, which
  ``localstore`` uses when ``LOCAL_PERSISTENT_FOLDER`` is unset.

Each location is chosen in this order:

1. its explicit environment variable (``PYROORB_CACHE_DIR`` for the cache, plus the
   per-table overrides ``PYROORB_EPHEM_DIR`` / ``PYROORB_DEBIAS_DIR`` /
   ``PYROORB_OBSCODES_PATH`` read by their modules);
2. a source checkout that already holds a reference table or the persistent state at its
   root (``<repo>/ephem``, ``<repo>/debias``, ``<repo>/.local_persistent``) keeps using
   it, so an existing checkout neither re-downloads nor loses anything;
3. the platform's per-user directory: ``$XDG_CACHE_HOME/pyroorb`` (default
   ``~/.cache/pyroorb``) and ``$XDG_DATA_HOME/pyroorb`` (``~/.local/share/pyroorb``) on
   Linux; ``~/Library/Caches/pyroorb`` and ``~/Library/Application Support/pyroorb`` on
   macOS; ``%LOCALAPPDATA%\\pyroorb\\Cache`` and ``%APPDATA%\\pyroorb`` on Windows.

Nothing here creates a directory; the writer does, when it first writes (cache writers go
through :func:`write_cache`, which does).
"""
import os
import sys
from pathlib import Path

CACHE_ENV = "PYROORB_CACHE_DIR"
APP = "pyroorb"

# The source checkout root (src/pyroorb/datadirs.py -> the repository). In an installed
# package none of the locations below exists there, so rule 2 never fires.
_CHECKOUT = Path(__file__).resolve().parents[2]


def _home():
    return Path(os.path.expanduser("~"))


def user_cache_dir(platform=None, environ=None):
    """The platform's per-user cache directory for PyRoOrb (not created)."""
    platform = platform or sys.platform
    env = os.environ if environ is None else environ
    if platform.startswith("win"):
        base = env.get("LOCALAPPDATA") or str(_home() / "AppData" / "Local")
        return Path(base) / APP / "Cache"
    if platform == "darwin":
        return _home() / "Library" / "Caches" / APP
    base = env.get("XDG_CACHE_HOME") or str(_home() / ".cache")
    return Path(base) / APP


def user_data_dir(platform=None, environ=None):
    """The platform's per-user data directory for PyRoOrb (not created)."""
    platform = platform or sys.platform
    env = os.environ if environ is None else environ
    if platform.startswith("win"):
        base = env.get("APPDATA") or str(_home() / "AppData" / "Roaming")
        return Path(base) / APP
    if platform == "darwin":
        return _home() / "Library" / "Application Support" / APP
    base = env.get("XDG_DATA_HOME") or str(_home() / ".local" / "share")
    return Path(base) / APP


def _checkout_dir(*parts):
    """``<checkout>/<parts>`` when that directory already exists, else None."""
    p = _CHECKOUT.joinpath(*parts)
    return p if p.is_dir() else None


def cache_dir():
    """The root for re-derivable downloads and caches (fetched observations, published
    orbits, MPCORB, the station table, radar astrometry)."""
    want = os.environ.get(CACHE_ENV, "").strip()
    return Path(want) if want else user_cache_dir()


def table_dir(name):
    """Where a large downloaded table set lives (``"ephem"`` for DE440, ``"debias"`` for
    the debias tables), when its own environment variable is unset."""
    want = os.environ.get(CACHE_ENV, "").strip()
    if want:
        return Path(want) / name
    return _checkout_dir(name) or (user_cache_dir() / name)


CACHE_MAX_MB = float(os.environ.get("PYROORB_CACHE_MAX_MB") or 2000)
# Astrometry caches by file-name prefix: deleted once expired, and first to go when the
# cache is over its cap. Everything else in the cache (the MPCORB catalogue, the station
# table, the radar table) is reference data that is replaced in place, never trimmed.
_ASTROMETRY_PREFIXES = ("ades_", "obs80_", "neocp_", "mpcorb_")
_TRIM_EVERY_S = 600.0
_LAST_TRIM = {"t": None}


def trim_cache(now=None, max_mb=None, root=None):
    """Delete expired astrometry caches, then the oldest ones until the cache directory is
    under ``max_mb`` (default CACHE_MAX_MB). Returns ``(n_deleted, bytes_after)``. Never
    raises: a file another process removed first is simply skipped."""
    import time
    from pyroorb.obs80 import ASTROMETRY_CACHE_TTL_S
    now = time.time() if now is None else now
    cap = (CACHE_MAX_MB if max_mb is None else max_mb) * 1e6
    root = Path(root) if root is not None else cache_dir()
    files, total, deleted = [], 0, 0
    try:
        entries = list(os.scandir(root))
    except OSError:
        return 0, 0
    for e in entries:
        try:
            if not e.is_file(follow_symlinks=False):
                continue
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if e.name.startswith(_ASTROMETRY_PREFIXES) and now - st.st_mtime > ASTROMETRY_CACHE_TTL_S:
            try:
                os.unlink(e.path)
                deleted += 1
            except OSError:
                pass
            continue
        total += st.st_size
        if e.name.startswith(_ASTROMETRY_PREFIXES):
            files.append((st.st_mtime, st.st_size, e.path))
    for _mtime, size, path in sorted(files):
        if total <= cap:
            break
        try:
            os.unlink(path)
            deleted += 1
            total -= size
        except OSError:
            pass
    return deleted, total


def note_cache_write():
    """Called after a cache write: trims the cache at most once per _TRIM_EVERY_S."""
    import time
    now = time.monotonic()
    last = _LAST_TRIM["t"]
    if last is not None and now - last < _TRIM_EVERY_S:
        return
    _LAST_TRIM["t"] = now
    try:
        trim_cache()
    except Exception:                       # pragma: no cover - a trim never fails a fetch
        pass


def write_cache(path, data):
    """Write ``data`` (str or bytes) to the cache file ``path``, creating its folder.

    A cache is an optimisation and must never fail the fetch that fills it: on a fresh
    machine the cache folder does not exist yet, and a bare ``write_text`` there raised
    FileNotFoundError after the MPC had already answered -- reported as "the MPC did not
    answer". Returns True when the file was written; an OSError is swallowed (False)."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            p.write_text(data)
    except OSError:
        return False
    note_cache_write()
    return True


def persistent_default():
    """Where restart-surviving state goes when ``LOCAL_PERSISTENT_FOLDER`` is unset."""
    return _checkout_dir(".local_persistent") or (user_data_dir() / "persistent")
