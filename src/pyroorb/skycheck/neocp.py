"""Orbits of the objects on the MPC's NEO Confirmation Page, for the sky check.

The NEOCP lists newly reported objects that have no designation yet. For each the MPC
publishes a nominal orbit and a set of variant orbits sampling what its few observations
allow (``showobsorbs.cgi?orb=y``); a short arc's variants can spread over degrees within days.
The sky check propagates every variant, and reports an object when any of them falls in the
field -- with the nominal's offset and how widely the variants spread.

The page answers one object per request (~0.6 s each), so the whole set (typically 50-150
objects) is fetched a few at a time and cached for ``MAX_AGE_S``.
"""
import html
import json
import math
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from pyroorb import datadirs
from pyroorb.skycheck import catalogue

LIST_URL = "https://minorplanetcenter.net/iau/NEO/neocp.txt"
ORBITS_URL = "https://cgi.minorplanetcenter.net/cgi-bin/cgipy/showobsorbs.cgi"
MAX_AGE_S = 1800.0          # the page changes through the day; half an hour is current enough
WORKERS = 4                 # polite: a few requests at a time to the MPC
RELEVANT_DAYS = 30.0        # NEOCP objects are days old: a field older than this skips them

_FLOAT = re.compile(r"^-?\d+(\.\d*)?$")


def cache_path():
    return datadirs.cache_dir() / "skycheck_neocp.json"


def _get(url, params=None, timeout=60):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_listing(text):
    """``neocp.txt`` -> the temporary designations it lists (first column)."""
    out = []
    for ln in text.splitlines():
        tid = ln[:7].strip()
        if tid:
            out.append(tid)
    return out


def parse_orbits(text):
    """One object's ``showobsorbs.cgi?orb=y`` page -> list of orbit dicts (nominal first).

    Row: ``Obj H G Epoch M Peri Node Incl e n a NObs NOpp Arc rms OrbitID``; H (and G) can be
    blank, so the packed epoch (5 characters) is located rather than counted to."""
    body = html.unescape(re.sub(r"<[^>]+>", "", text))
    out = []
    for ln in body.splitlines():
        t = ln.split()
        if len(t) < 12 or t[0] == "Object":
            continue
        H = G = math.nan
        if _FLOAT.match(t[1]) and _FLOAT.match(t[2]) and len(t[3]) == 5:
            H, G, k = float(t[1]), float(t[2]), 3
        elif _FLOAT.match(t[1]) and len(t[2]) == 5:
            G, k = float(t[1]), 2
        elif len(t[1]) == 5:
            k = 1
        else:
            continue
        epoch = catalogue.unpack_epoch(t[k])
        try:
            M, peri, node, incl, e, _n, a = (float(x) for x in t[k + 1:k + 8])
        except (ValueError, IndexError):
            continue
        if epoch is None or not (a > 0) or not (0 <= e < 1):
            continue
        out.append({"tid": t[0], "H": H, "G": G, "epoch": epoch, "M": M, "peri": peri,
                    "node": node, "i": incl, "e": e, "a": a, "orbit_id": t[-1]})
    return out


def _fetch_one(tid):
    try:
        return tid, parse_orbits(_get(ORBITS_URL, {"Obj": tid, "orb": "y"}))
    except Exception:                                                # noqa: BLE001
        return tid, []


def _read_cache():
    try:
        return json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cached():
    """The last fetched NEOCP orbits whatever their age, without touching the network
    (empty when none were ever fetched)."""
    c = _read_cache()
    return c["objects"] if c else {}


def fetch(max_age_s=MAX_AGE_S, verbose=False):
    """``{temporary designation: [orbit, ...]}`` for every object on the NEOCP (cached). A failed
    refresh keeps the last good copy; with none it returns an empty mapping."""
    cached = _read_cache()
    if cached and time.time() - cached.get("fetched", 0) < max_age_s:
        return cached["objects"]
    try:
        ids = parse_listing(_get(LIST_URL))
        with ThreadPoolExecutor(WORKERS) as ex:
            objects = {tid: orbs for tid, orbs in ex.map(_fetch_one, ids) if orbs}
        if ids and not objects:         # every orbit page failed (throttled): not "empty"
            raise OSError(f"no orbits for any of the {len(ids)} NEOCP objects")
    except Exception as ex:                                          # noqa: BLE001
        if verbose:
            print(f"NEOCP fetch failed ({type(ex).__name__}: {ex})", flush=True)
        return cached["objects"] if cached else {}
    datadirs.write_cache(cache_path(), json.dumps({"fetched": time.time(), "objects": objects}))
    if verbose:
        print(f"NEOCP: {len(objects)} objects, "
              f"{sum(len(v) for v in objects.values())} orbits", flush=True)
    return objects


def as_catalogue(objects):
    """The NEOCP orbits as a :class:`catalogue.Catalogue` (one row per orbit; ``desig`` the
    temporary designation, ``name`` the orbit id, NOMINAL first within each object)."""
    from pyroorb.skycheck import conic
    rows = []
    for tid, orbs in objects.items():
        for o in orbs:
            T = float(conic.perihelion_time(o["a"], o["e"], o["M"], o["epoch"]))
            cols = {"q": o["a"] * (1 - o["e"]), "e": o["e"], "i": o["i"], "node": o["node"],
                    "peri": o["peri"], "T": T, "epoch": o["epoch"], "H": o["H"],
                    "G": o["G"], "U": math.nan, "nopp": 1.0}
            rows.append((tid, o.get("orbit_id") or "", cols, catalogue.KIND_ASTEROID))
    return catalogue.Catalogue.from_rows(rows) if rows else None
