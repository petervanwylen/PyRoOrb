"""Ephemeral on-disk cache of finished fit payloads.

WHAT IT IS FOR -- and what it deliberately is NOT
-------------------------------------------------
This is a CAPACITY tier, not a durability one.  It is meant to live on
container-local scratch space and to VANISH on restart or redeploy, which is
a deliberate requirement: a fit is only valid for the observations it
was computed from, and a new deployment may compute a different orbit from the
same arc, so a cache that outlived the process would be a cache that serves
answers the running code did not produce.

What it buys, then, is not "survives a restart" but:

  * **Reach.** RAM holds a few dozen fits before it is competing with the
    searches and sessions that actually need the memory; a scratch disk holds
    thousands.  On a launch spike the interesting property is how many DISTINCT
    objects stay warm, and that is a disk number, not a RAM number.
  * **Relief.** A payload parked on disk is a payload not sitting in the
    process.  The in-memory job table can then keep only what is live, which is
    what makes `fitqueue`'s eviction cheap and the container's RSS boring.

The stored object is the SERIALISED PAYLOAD only -- the JSON the browser reads.
The live template session a coalesced fit hands out for interaction stays in
memory and is not written here: it holds a rebound simulation, which does not
serialise, and re-deriving it from disk would be a re-fit by another name.  So
a disk hit serves the ~96% of viewers who look and leave at full speed, and the
few who then interact pay one fresh fit -- exactly the trade chosen by caching
the response rather than the state.

SAFETY
------
The cache directory must be scratch.  :func:`_resolve_dir` REFUSES any path
under a persistent volume (``/data`` and friends) rather than quietly filling
it: this cache is designed to be thrown away, and a store that is designed to
be thrown away must never be pointed at storage that is designed to be kept.
"""
import os
import json
import time
import errno
import shutil
import tempfile
import threading

# Directory paths that are PERSISTENT on this deploy and must never hold an
# ephemeral cache.  Railway mounts its volume at /data; the others are the
# usual suspects a well-meaning override might reach for.
_PERSISTENT_ROOTS = ("/data", "/mnt/data", "/var/lib/data")

# Total bytes the cache may occupy.  A fit payload is tens of KB (a large arc
# with per-observation residuals reaches a few hundred), so 256 MB is thousands
# of objects -- far past any launch spike -- while staying small against a
# container disk.
BUDGET_BYTES = int(os.environ.get("PYROORB_FIT_CACHE_MB") or 256) * 1024 * 1024

# Never let the cache be the reason the disk fills: below this much free space
# the cache stops writing (and prunes), because a full disk breaks logging, the
# ephemerides and the process itself, none of which are worth a cache hit.
MIN_FREE_MB = float(os.environ.get("PYROORB_FIT_CACHE_MIN_FREE_MB") or 512)

# How long an entry may be served.  The engine revision is already in the key,
# so this is about the ASTROMETRY going stale, not the code.  It is far LONGER
# than the astrometry cache (obs80.ASTROMETRY_CACHE_TTL_S, 3 h), which is fine
# for a published object -- new astrometry of one arrives over days -- and wrong
# for a NEOCP / temporary id, whose arc can double in an hour.  Such a fit's key
# is served under the caller's shorter ``max_age`` (fitqueue.submit), never this.
TTL_SECS = float(os.environ.get("PYROORB_FIT_CACHE_TTL") or 6 * 3600)

_LOCK = threading.Lock()
_DIR = None
_DISABLED_REASON = None


def _resolve_dir():
    """The cache directory, created on first use, or None when unusable.

    Refuses a persistent location outright -- see the module docstring.  Every
    other failure (read-only fs, no space, a file where the directory should be)
    disables the cache rather than raising: a cache is an optimisation, and an
    optimisation that can take the server down is a liability.
    """
    global _DIR, _DISABLED_REASON
    if _DIR is not None or _DISABLED_REASON is not None:
        return _DIR
    want = os.environ.get("PYROORB_FIT_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), "pyroorb-fitcache")
    real = os.path.realpath(want)
    roots = list(_PERSISTENT_ROOTS)
    try:                                   # the project's own persistent folder too
        from pyroorb import localstore
        roots.append(str(localstore.root()))
    except Exception:
        pass
    for root in roots:
        if real == root or real.startswith(root.rstrip("/") + "/"):
            _DISABLED_REASON = (
                f"refusing to cache in {real}: {root} is a persistent volume and "
                f"this cache is meant to be discarded on restart")
            return None
    try:
        os.makedirs(real, exist_ok=True)
        probe = os.path.join(real, ".writable")
        with open(probe, "w") as fh:
            fh.write("1")
        os.unlink(probe)
    except OSError as ex:
        _DISABLED_REASON = f"cache directory unusable ({real}): {ex}"
        return None
    _DIR = real
    return _DIR


def _free_mb(path):
    try:
        st = shutil.disk_usage(path)
        return st.free / (1024.0 * 1024.0)
    except OSError:
        return None


def _path(d, key):
    return os.path.join(d, f"{key}.json")


def _entries(d):
    """(path, size, mtime) for every cache file, newest last.  Never raises."""
    out = []
    try:
        with os.scandir(d) as it:
            for e in it:
                if not e.name.endswith(".json"):
                    continue
                try:
                    st = e.stat()
                except OSError:
                    continue
                out.append((e.path, st.st_size, st.st_mtime))
    except OSError:
        return []
    out.sort(key=lambda r: r[2])
    return out


def _prune(d, need_bytes=0):
    """Evict oldest-first until the budget (minus ``need_bytes``) is met.
    Caller holds the lock."""
    rows = _entries(d)
    total = sum(r[1] for r in rows)
    limit = max(0, BUDGET_BYTES - need_bytes)
    for path, size, _mtime in rows:
        if total <= limit:
            break
        try:
            os.unlink(path)
            total -= size
        except OSError:
            total -= size          # already gone; stop counting it either way


def get(key, max_age=None):
    """The cached payload for ``key``, or None.

    ``max_age`` (seconds) tightens TTL_SECS for this read -- a fit of an object
    whose astrometry is still arriving may only be served while that astrometry
    would be.  An entry older than it is a miss but is NOT deleted: the TTL, not
    one caller's horizon, decides when a file is garbage.

    A corrupt or truncated file is DELETED and reported as a miss -- this
    filesystem tears large writes, and a half-written payload must not be able
    to poison a key for the life of the container.
    """
    d = _resolve_dir()
    if d is None:
        return None
    p = _path(d, key)
    try:
        os.stat(p)
    except OSError:
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        try:
            os.unlink(p)
        except OSError:
            pass
        return None
    if not isinstance(rec, dict) or "payload" not in rec:
        try:
            os.unlink(p)
        except OSError:
            pass
        return None
    # AGE FROM `stored`, NEVER FROM MTIME.  The LRU touches mtime on every read
    # (below), so a TTL measured from mtime is reset by the very act of serving
    # the entry: any object read at least once per TTL window would serve its
    # FIRST fit forever and never see new astrometry -- and the objects read
    # that often are exactly the popular ones a launch is about.  `stored` is
    # written once, at put(), and is the only clock that means "how old is this
    # answer".  A record without it (there are none, but a torn or foreign file
    # could lack it) is treated as expired rather than immortal.
    age = time.time() - float(rec.get("stored") or 0)
    if age > TTL_SECS:
        try:
            os.unlink(p)
        except OSError:
            pass
        return None
    if max_age is not None and age > max_age:
        return None
    try:                                   # LRU: reading is using
        os.utime(p, None)
    except OSError:
        pass
    return rec


def put(key, payload, desig=None):
    """Store ``payload`` under ``key``.  Best-effort: any failure is a silent
    no-op, because a cache write must never break the request that produced it.

    Written to a temp file and renamed, so a reader never sees a partial file
    even though this filesystem tears large writes.
    """
    d = _resolve_dir()
    if d is None or not isinstance(payload, dict):
        return False
    free = _free_mb(d)
    if free is not None and free < MIN_FREE_MB:
        with _LOCK:
            _prune(d)                      # make room, but do not write into a full disk
        return False
    try:
        blob = json.dumps({"payload": payload, "desig": desig,
                           "stored": time.time()})
    except (TypeError, ValueError):
        return False                       # not serialisable: not cacheable
    data = blob.encode("utf-8")
    with _LOCK:
        _prune(d, need_bytes=len(data))
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, _path(d, key))
            return True
        except OSError as ex:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            if ex.errno == errno.ENOSPC:
                _prune(d)
            return False


def clear():
    """Drop every entry (tests, and the /memory page's manual purge)."""
    d = _resolve_dir()
    if d is None:
        return
    with _LOCK:
        for path, _s, _m in _entries(d):
            try:
                os.unlink(path)
            except OSError:
                pass


def stats():
    """What the cache is holding, for /healthz and the memory page."""
    d = _resolve_dir()
    if d is None:
        return {"enabled": False, "reason": _DISABLED_REASON}
    rows = _entries(d)
    return {"enabled": True, "dir": d, "entries": len(rows),
            "bytes": sum(r[1] for r in rows), "budget_bytes": BUDGET_BYTES,
            "free_mb": _free_mb(d), "min_free_mb": MIN_FREE_MB,
            "ttl_secs": TTL_SECS}
