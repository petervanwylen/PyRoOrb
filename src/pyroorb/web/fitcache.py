"""Tiny in-process cache of completed fits, so the export endpoints can reuse a
``FittedOrbit`` the user already produced instead of re-fitting on every download.

Deliberately minimal and self-contained (no engine_api edits): the read-only
viewer's ``run_fit`` and the paste loader both deal in ``FittedOrbit`` objects;
this module just holds the last few of them keyed by a short token. It is NOT the
stateful Session abstraction from WEB_UI_PLAN §5.1 (toggle/full-step/undo) -- that
is a separate, larger piece. This only memoises whole fits for export.
"""
from pyroorb import obs_sources
import os
import sys
import time
import hashlib
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
import pyroorb.fit_object as fo  # noqa: E402

_MAX = 16                       # keep the last N fits; plenty for a single-user session
_CACHE = OrderedDict()          # key -> FittedOrbit
_STORED = {}                    # key -> time.time() at put(); ages get_by_desig's fits
# A by-designation fit of a PUBLISHED object may be re-served this long (the disk
# fit tier's own lifetime); one naming a NEOCP / temporary id only as long as its
# astrometry cache (obs_sources.fit_reuse_max_age).  Token-keyed get() never ages:
# a token names one specific fit the user already has on screen.
DESIG_TTL_SECS = float(os.environ.get("PYROORB_FIT_CACHE_TTL") or 6 * 3600)


def _evict():
    while len(_CACHE) > _MAX:
        k, _ = _CACHE.popitem(last=False)
        _STORED.pop(k, None)


def put(fit, key=None):
    """Store a FittedOrbit and return its cache key. A stable key is derived from
    the object designation + epoch so re-exporting the same object hits the cache."""
    if key is None:
        seed = f"{getattr(fit, 'designation', '?')}|{getattr(fit, 'epoch_tdb', '?')}"
        key = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    _CACHE[key] = fit
    _STORED[key] = time.time()
    _CACHE.move_to_end(key)
    _evict()
    return key


def get(key):
    fit = _CACHE.get(key)
    if fit is not None:
        _CACHE.move_to_end(key)
    return fit


def _desig_key(desig, force):
    norm = ",".join(sorted(s.strip() for s in str(desig).replace("\n", ",").split(",") if s.strip()))
    return "desig:" + hashlib.sha1(f"{norm}|{bool(force)}".encode("utf-8")).hexdigest()[:12]


def get_by_desig(desig, obscodes, force=False):
    """Return a cached fit for a designation (or linkage), fetching + fitting on a
    cache miss. Used by the export endpoints, which receive a designation rather
    than a session handle."""
    key = _desig_key(desig, force)
    desigs = [s.strip() for s in str(desig).replace("\n", ",").split(",") if s.strip()]
    max_age = obs_sources.fit_reuse_max_age(desigs) or DESIG_TTL_SECS
    fit = get(key)
    if fit is not None and time.time() - _STORED.get(key, 0.0) <= max_age:
        return fit
    if not desigs:
        raise ValueError("no designation given")
    obs, meta = obs_sources.fetch_observations(desigs, obscodes)
    # A fetch that FAILED is not an object that does not exist -- and a partial failure
    # is worse than either, since the rows that did arrive would be fitted and cached as
    # this designation's answer.  See obs_sources.fetch_observations.
    failed = (meta or {}).get("fetch_errors") or {}
    if failed:
        raise ValueError(f"the MPC did not answer for {', '.join(sorted(failed))} "
                         f"— try again in a moment")
    if not obs:
        raise ValueError(f"no observations found for {desigs}")
    fit = fo.fit_orbit(obs, obscodes, desig=meta["designation"], meta=meta,
                       verbose=False, force=force,
                       mpc_seed=fo.fetch_mpc_seed(obs, meta))
    put(fit, key=key)
    return fit
