"""Observation & observer-position I/O: everything that turns a designation (or a pasted
astrometry blob) into the internal obs dicts the fitter consumes, plus the observatory
parallax table and the heliocentric observer positions.

Factored out of fit_object so the engine file holds orbit-determination logic, not data
plumbing. Three concerns live here:
  * the MPC observatory parallax table (ObsCodes.htm: download/cache + code->position),
  * fetching + parsing astrometry (OBS80 from the published-obs API, ADES/NEOCP via ades.py,
    or a pasted file), with designation packing/unpacking, and
  * the batched ITRS->GCRS observer geometry that feeds the force model.

Leaf module: it imports the obs parsers (obs80), weights, the propagator backends, and
astropy -- never fit_object -- so importing it can't cycle. ades is imported lazily inside
the fetch/read entry points (ades imports back into this module's observer geometry, so a
module-level import would cycle). fit_object re-exports these names for compatibility.
"""
import json, os, re, sys, threading, urllib.request
from pathlib import Path
import numpy as np
from pyroorb import orbit_common                 # the shared "same physical detection" predicate
from astropy.time import Time
from astropy.coordinates import (get_body_barycentric, ITRS, GCRS, CartesianRepresentation)
import astropy.units as u

from pyroorb.obs80 import (mpc80_to_radec, mpc80_magnitude, mpc80_catalog, mpc80_satellite_offset,
                   mpc80_roving_itrs_au, cache_stem, cache_fresh, astrometry_ttl, NEOCP_CACHE_TTL_S)

# Pin ONE Earth-orientation table once, auto-download off (`eop.py`): IERS-B through its
# end, then the bundled Bulletin A rapid+predicted values. Unpinned, astropy re-parsed the
# IERS file on EVERY ITRS<->GCRS transform (~1 s per observation). The transforms live in
# observer_positions_batch below, so the pin is made where they are. Measured: the old B-only
# pin cost 17-25 m of site (<=0.02") past its 2026-05-02 end, where it took UT1-UTC=0.
from pyroorb import eop
from pyroorb import datadirs
eop.pin()

class ObsParseError(ValueError):
    """The supplied astrometry could not be read -- a MALFORMED-INPUT error.

    The point of having a type for this is to separate "the user pasted something we
    can't parse" from "we have a bug". A handler that catches every exception around a
    parse call reports both as the former: an internal TypeError (a call that drifted
    from its signature, say) comes back to the user as "could not parse pasted obs",
    which is both wrong and invisible in tests. Handlers catch THIS; anything else is
    ours and must propagate to the caller's error boundary.

    Raised at the parsing boundary (web/loader._parse), which translates the malformed-
    input exceptions the underlying readers raise -- bad XML, an unreadable number or
    timestamp -- into this one."""


ROOT = Path(__file__).parent
KM_PER_AU = 1.495978707e8                 # IAU 2012 astronomical unit (km)
A_EARTH_AU = 6378.137 / KM_PER_AU         # Earth equatorial radius in au
OBS_API = "https://data.minorplanetcenter.net/api/get-obs"

import pyroorb.forward_model as _reb


def _force_backend(fit=None):
    """Pick the force-model backend. ASSIST (JPL DE440 ephemeris + general
    relativity + separate Earth/Moon, and 16 asteroids if that ephemeris is
    present) when its planet ephemeris file exists; otherwise the rebound
    8-planet point-mass model. ASSIST tracks the true orbit ~450x better on long
    arcs (RH69: 284" -> 0.6"); the models agree on short arcs.

    `fit` is for the callers that resolve a backend for an ALREADY-FITTED orbit (an
    ephemeris, an impact scan): when that fit's own backend is gravity-pinned -- the object
    IS one of the 16 asteroid perturbers, so it may not be integrated in a field containing
    it -- hand back that backend and keep the product in the force model the orbit was
    determined in. Whole-fit +16ast mode is a process-wide switch, so without this an
    ephemeris for (1) Ceres would be computed in exactly the model its own fit had to
    refuse: degrees off within weeks."""
    pinned = (getattr(fit, "_ctx", None) or {}).get("backend") if fit is not None else None
    if getattr(pinned, "_gravity_pinned", False):
        return pinned, getattr(fit, "force_model", "?")
    try:
        import pyroorb.force_assist as _fa
        if Path(_fa.PLANETS_PATH).exists():
            ast_on = bool(_fa.ASTEROIDS_PATH and Path(_fa.ASTEROIDS_PATH).exists())
            # Whole-fit +16ast mode, but THIS object is one of the 16 (force_assist's
            # self-perturber context): hand back a backend with the perturbers pinned off, so
            # the pin travels on the fit's ctx and outlives the context that set it -- an
            # interactive step, a re-anchor or an ephemeris built later stays in the force
            # model the orbit was determined in. Nothing to pin when the perturbers are off
            # anyway (the engine default), so an ordinary fit is untouched.
            if ast_on and _fa.suppressed_perturber() is not None:
                return _fa.GravityOnlyForce(_fa), "ASSIST(DE440+GR, +16ast off: self-perturber)"
            tag = "ASSIST(DE440+GR" + ("+16ast)" if ast_on else ")")
            return _fa, tag
    except Exception as ex:
        print(f"(ASSIST unavailable: {ex}; using rebound 8-planet model)")
    return _reb, "rebound(8-planet)"


# ---------------------------------------------------------------------------
# MPC observatory parallax table  (ObsCodes.htm)
#
# A copy of the MPC table ships inside the package (BUNDLED_OBSCODES), so a fresh install,
# CI and the test suite need no download. A newer copy in the cache (OBSCODES_CACHE;
# PYROORB_OBSCODES_PATH overrides it) wins when present. A download is tried only when it
# can matter:
#   * a MISS -- an observation names a station the table lacks: one validated download,
#     at most once per process per OBSCODES_MISS_INTERVAL, then the fetch resolves again
#     (`refresh_on_miss`); offline, the observation is dropped and reported as before;
#   * the web server's periodic re-check (`refresh_obscodes`,
#     PYROORB_OBSCODES_REFRESH_HOURS, default 12);
#   * PYROORB_OBSCODES_REFRESH=1, which forces one at first use.
# PYROORB_OBSCODES_REFRESH=never turns every download off (tests and CI use it, so a
# station-table change can never change a test answer).
# Sources: the MPC's own page, then Bill Gray's find_orb mirror (byte-identical below the
# <pre>, 2026-10-06) only when the MPC fails. A download is VALIDATED before it replaces
# the cache (enough parseable stations, not a sudden shrink), so an error page or a
# truncated transfer can never cost us the table. The gitignored find_orb/ reference clone
# is never read at runtime.
# ---------------------------------------------------------------------------
OBSCODES_URLS = (
    "https://www.minorplanetcenter.net/iau/lists/ObsCodes.html",          # the source
    "https://raw.githubusercontent.com/Bill-Gray/find_orb/refs/heads/master/ObsCodes.htm",
)
OBSCODES_CACHE = Path(os.environ.get("PYROORB_OBSCODES_PATH")
                      or datadirs.cache_dir() / "ObsCodes.htm")
BUNDLED_OBSCODES = Path(__file__).resolve().parent / "package_data" / "ObsCodes.htm"
OBSCODES_MISS_INTERVAL = 3600.0   # seconds between miss-triggered downloads, per process
_OBSCODES_MISS_AT = None          # time.monotonic() of the last miss-triggered attempt
_OBSCODES_RESOLVED = None     # memoised path: resolve/download at most once per process
OBSCODES_MIN_STATIONS = 1500  # 2729 on 2026-10-06; fewer parseable rows = not the table
OBSCODES_MAX_SHRINK = 0.95    # a download listing < 95% of the cached stations is refused
_OBSCODES_LOCK = threading.Lock()
_OBSCODES_TOKEN = None        # content hash of the cached table (`obscodes_token`)

# Ground observations dropped because their station is absent from ObsCodes.htm,
# {obscode: count}. A non-empty record after a fetch is the cue to prompt the user.
_UNKNOWN_OBSCODES = {}


def _station_codes(text):
    """The station codes a table text lists (any row with a code, space-based included)."""
    return {line[:3] for line in text.splitlines()
            if len(line) >= 4 and line[:3].strip() and not line.startswith("<")
            and line[:3] != "Cod"}


def _download_obscodes(verbose=False, prior=None):
    """Fetch the table, MPC first, find_orb's mirror only when the MPC fails or serves
    something that is not the table. Returns (bytes, url). `prior` (a set of codes) arms the
    shrink guard. Raises when every source fails."""
    errors = []
    for url in OBSCODES_URLS:
        try:
            if verbose:
                print(f"  downloading ObsCodes from {url} ...")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            codes = _station_codes(data.decode("utf-8", errors="ignore"))
            if len(codes) < OBSCODES_MIN_STATIONS:
                raise ValueError(f"only {len(codes)} stations listed")
            if prior and len(codes) < OBSCODES_MAX_SHRINK * len(prior):
                raise ValueError(f"{len(codes)} stations vs {len(prior)} cached")
            return data, url
        except Exception as ex:
            errors.append(f"{url}: {ex}")
    raise RuntimeError("ObsCodes download failed: " + "; ".join(errors))


def _write_obscodes(data):
    global _OBSCODES_TOKEN
    OBSCODES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OBSCODES_CACHE.with_name(OBSCODES_CACHE.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(OBSCODES_CACHE)             # atomic: a reader sees the old or the new file
    _OBSCODES_TOKEN = _content_token(data)  # callers hold _OBSCODES_LOCK


def _content_token(data):
    import hashlib
    return hashlib.sha1(data).hexdigest()[:12]


def _obscodes_downloads_allowed():
    """False under PYROORB_OBSCODES_REFRESH=never: no download of any kind."""
    return os.environ.get("PYROORB_OBSCODES_REFRESH", "").strip().lower() != "never"


def _current_obscodes_path():
    """The table in use: the cached copy when there is one, else the bundled one."""
    return OBSCODES_CACHE if OBSCODES_CACHE.exists() else BUNDLED_OBSCODES


def ensure_obscodes_file(verbose=False):
    """Return a path to ObsCodes.htm (memoised per process): the cached copy when there is
    one, else the copy bundled with the package -- no download is needed to start.
    PYROORB_OBSCODES_REFRESH=1 forces one validated download first; when it fails, the
    cached or bundled copy is used. The gitignored find_orb/ clone is never a fallback."""
    global _OBSCODES_RESOLVED
    if _OBSCODES_RESOLVED is not None:
        return _OBSCODES_RESOLVED
    with _OBSCODES_LOCK:
        if _OBSCODES_RESOLVED is not None:
            return _OBSCODES_RESOLVED
        if os.environ.get("PYROORB_OBSCODES_REFRESH", "0").strip() == "1":
            current = _current_obscodes_path()
            try:
                prior = (_station_codes(current.read_text(errors="ignore"))
                         if current.exists() else None)
                data, _url = _download_obscodes(verbose, prior)
                _write_obscodes(data)
            except Exception as ex:
                if verbose:
                    print(f"  ObsCodes.htm download failed ({ex}); using {current}")
        path = _current_obscodes_path()
        if not path.exists():
            raise RuntimeError(f"no observatory-code table: neither {OBSCODES_CACHE} nor "
                               f"the bundled {BUNDLED_OBSCODES} exists")
        _OBSCODES_RESOLVED = path
    return _OBSCODES_RESOLVED


def refresh_on_miss(codes, verbose=False):
    """Observations named stations the table lacks (`codes`): try ONE validated download
    -- at most once per process per OBSCODES_MISS_INTERVAL, never under
    PYROORB_OBSCODES_REFRESH=never -- and return the subset of `codes` the refreshed
    table now lists (empty when nothing changed, the download failed or was not tried).
    The caller resolves those observations again."""
    global _OBSCODES_MISS_AT, _OBSCODES_RESOLVED
    import time as _time
    codes = set(codes or ())
    if not codes or not _obscodes_downloads_allowed():
        return set()
    with _OBSCODES_LOCK:
        now = _time.monotonic()
        if _OBSCODES_MISS_AT is not None and now - _OBSCODES_MISS_AT < OBSCODES_MISS_INTERVAL:
            return set()
        _OBSCODES_MISS_AT = now
        current = _current_obscodes_path()
        prior = _station_codes(current.read_text(errors="ignore")) if current.exists() else set()
        try:
            data, _url = _download_obscodes(verbose, prior or None)
        except Exception as ex:
            if verbose:
                print(f"  ObsCodes.htm refresh for {sorted(codes)} failed ({ex})")
            return set()
        found = codes & _station_codes(data.decode("utf-8", errors="ignore"))
        if found:
            _write_obscodes(data)
            _OBSCODES_RESOLVED = OBSCODES_CACHE
        return found


def refresh_obscodes(verbose=False):
    """Re-download the table and replace the cache when the content changed (the web
    server's periodic check). Never raises: returns {changed, source, n, added, removed,
    error}. A failed or implausible download keeps the cached table untouched.
    Under PYROORB_OBSCODES_REFRESH=never nothing is downloaded."""
    global _OBSCODES_RESOLVED
    ensure_obscodes_file()
    with _OBSCODES_LOCK:
        current = _current_obscodes_path()
        old = current.read_bytes() if current.exists() else b""
        prior = _station_codes(old.decode("utf-8", errors="ignore"))
        if not _obscodes_downloads_allowed():
            return {"changed": False, "source": None, "n": len(prior), "added": [],
                    "removed": [], "error": "downloads off (PYROORB_OBSCODES_REFRESH=never)"}
        try:
            data, url = _download_obscodes(verbose, prior)
        except Exception as ex:
            return {"changed": False, "source": None, "n": len(prior), "added": [],
                    "removed": [], "error": str(ex)}
        new = _station_codes(data.decode("utf-8", errors="ignore"))
        changed = data != old
        if changed:
            _write_obscodes(data)
            _OBSCODES_RESOLVED = OBSCODES_CACHE
        return {"changed": changed, "source": url, "n": len(new),
                "added": sorted(new - prior), "removed": sorted(prior - new), "error": None}


def obscodes_token():
    """A short hash of the cached table's content. Changes exactly when a refresh changed
    the table, so in-process caches of `load_obscodes()` (and the web fit cache's key)
    can tell a stale copy from a current one."""
    global _OBSCODES_TOKEN
    tok = _OBSCODES_TOKEN
    if tok is None:
        path = ensure_obscodes_file()
        with _OBSCODES_LOCK:                # never races a refresh's write
            if _OBSCODES_TOKEN is None:
                _OBSCODES_TOKEN = _content_token(path.read_bytes())
            tok = _OBSCODES_TOKEN
    return tok


def load_obscodes():
    codes = {}
    for line in ensure_obscodes_file().read_text(errors="ignore").splitlines():
        if len(line) < 30 or not line[:3].strip() or line.startswith("<"):
            continue
        try:
            lon = float(line[3:13]); cos = float(line[13:21]); sin = float(line[21:30])
        except ValueError:
            continue
        codes[line[:3]] = (lon, cos, sin)
    return codes


def load_obscode_names():
    """Map MPC observatory code -> human-readable name.

    Same fixed-width ObsCodes.htm table as ``load_obscodes`` (the name is the
    free-text tail from column 30 onward). Kept separate so the parallax loader
    stays a lean ``code -> (lon, cos, sin)`` map; this is a display-only lookup.
    """
    names = {}
    for line in ensure_obscodes_file().read_text(errors="ignore").splitlines():
        code = line[:3]
        if len(line) <= 30 or not code.strip() or line.startswith("<") or code == "Cod":
            continue
        name = line[30:].strip()
        if name:
            names[code] = name
    return names


def record_unknown_obscode(code):
    """Note a ground observation dropped because its station `code` is absent from
    ObsCodes.htm. Called from both obs parsers (OBS80 ``parse_obs`` and ADES
    ``ades.records_to_obs``); drained by ``pop_unknown_obscodes``."""
    _UNKNOWN_OBSCODES[code] = _UNKNOWN_OBSCODES.get(code, 0) + 1


def pop_unknown_obscodes():
    """Return {obscode: count} of ground observations dropped since the last call because
    their station is missing from ObsCodes.htm, and clear the record. A non-empty result
    is the cue to tell the user the table may be out of date and to refetch it
    (PYROORB_OBSCODES_REFRESH=1)."""
    global _UNKNOWN_OBSCODES
    u, _UNKNOWN_OBSCODES = _UNKNOWN_OBSCODES, {}
    return u


_GEOCENTRIC_MEMO = {}           # (jds bytes hash, n) -> Earth-Sun (n,3) au
_GEOCENTRIC_MEMO_MAX = 6        # a handful of grids; each ~n*24 bytes


_DE440_OBSERVER = os.environ.get("PYROORB_DE440_OBSERVER", "1") != "0"


def _earth_sun_eq(T):
    """Heliocentric equatorial GEOCENTRE positions (n,3 au) at astropy Time `T` -- from the
    SAME DE440 ephemeris the integration uses (force_assist) whenever it is available.

    astropy's builtin analytic Earth (erfa epv00) is 2-3 km off DE440. That is invisible at
    survey distances (~0.003" at 0.3 au) and was measured as exactly that in July's radar work
    ("invisible optically") -- but it scales as 1/range, and on a deep-Earth-encounter arc it is
    CATASTROPHIC: at 2004 FH's 0.003-0.01 au the offset reads as 0.5-1.5" of astrometry growing
    toward perigee, which inflated the near-encounter residuals (the sigma-gate clipped the 11 obs
    with the most b-plane leverage), biased the encounter threading, and let the post-encounter
    semimajor-axis error masquerade as a Yarkovsky: auto delivered A2 = -3.4e-13 where JPL and
    find_orb hold +1.7e-13, with our own optical rms 0.857" against the 0.497" the same data fit
    once the geometry is right. The radar path fixed this same defect at its own door
    (`radar._station_posvel`); this is the optical half, in the one place observer positions are
    built. Falls back to the builtin Earth when ASSIST / the DE440 kernel is absent -- which
    MATCHES the force model actually integrating on such a box (the rebound fallback builds its
    planets from the same astropy ephemeris), so each backend stays self-consistent.
    PYROORB_DE440_OBSERVER=0 restores the old behavior for A/B."""
    if _DE440_OBSERVER:
        try:
            import pyroorb.force_assist as _fa
            ephem = _fa.get_ephem()
            tdb = np.atleast_1d(np.asarray(T.tdb.jd, dtype=float))
            out = np.empty((len(tdb), 3))
            for i, t in enumerate(tdb):
                e = ephem.get_particle("Earth", float(t) - ephem.jd_ref)
                sunp, _ = _fa._sun_bary(ephem, float(t))
                out[i] = (e.x - sunp[0], e.y - sunp[1], e.z - sunp[2])
            return out
        except Exception:
            pass
    earth = get_body_barycentric("earth", T.tdb).xyz.to(u.au).value.T
    sun = get_body_barycentric("sun", T.tdb).xyz.to(u.au).value.T
    return earth - sun


def _geocentric_positions(jds):
    """Heliocentric equatorial position (n,3 au) of the GEOCENTRE at ``jds``.

    The geocentre needs no ITRS->GCRS rotation at all -- its site offset is
    exactly the zero vector, and rotating zero yields zero -- so this skips the
    (expensive: erfa c2i06a + astropy representation machinery) transform the
    general path performs, and is EXACT rather than approximate.

    Memoized on the epoch grid: the result depends only on ``jds``, never on the
    orbit, and the ITF corridor sweeps the SAME 1997->now grid once per
    recursion depth and once per LOV variant -- otherwise every sweep recomputes
    an identical array (erfa epv00/dtdb, ~0.6 s per 3600-node grid)."""
    jds = np.asarray(jds, dtype=float)
    key = (hash(jds.tobytes()), jds.shape)
    hit = _GEOCENTRIC_MEMO.get(key)
    if hit is None:
        hit = _earth_sun_eq(Time(jds, format="jd", scale="utc"))
        if len(_GEOCENTRIC_MEMO) >= _GEOCENTRIC_MEMO_MAX:
            _GEOCENTRIC_MEMO.pop(next(iter(_GEOCENTRIC_MEMO)))
        _GEOCENTRIC_MEMO[key] = hit
    return hit


def observer_positions_batch(jds, codes, obscodes, sat_pos=None, site_itrs=None):
    """Heliocentric ICRS-equatorial positions (n,3 au) for many (jd, obscode) pairs
    in ONE ITRS->GCRS transform + one ephemeris lookup. ~10000x faster than calling
    observer_helio_eq per observation (which re-triggered astropy's IERS handling).

    Two optional per-obs overrides (each a list aligned with jds; None entries fall back to the
    obscode ground site) handle observers that have no fixed station:
      `site_itrs` -- an Earth-FIXED geocentric ITRS position (au), e.g. a ROVING observer (247)
        from its geodetic lon/lat/alt. It replaces the ground site BEFORE the ITRS->GCRS rotation,
        so Earth's rotation at the obs time is applied just as for a fixed station.
      `sat_pos`   -- an INERTIAL geocentric equatorial position (au), e.g. a SATELLITE (C51/WISE)
        from its ADES/OBS80 vector. It is already in the GCRS-aligned frame, so it replaces the
        site offset AFTER the transform (no Earth-rotation applied).
    Such obs carry an obscode not in `obscodes`, so the ground term falls back to a geocenter
    placeholder and is overwritten."""
    # Pure-geocentre fast path (the ITF corridor's site "500", the ephemeris
    # pane's default): every site offset is exactly zero and no override moves
    # it, so the whole ITRS->GCRS step is a rotation of the zero vector.
    if sat_pos is None and site_itrs is None and all(
            obscodes.get(c, (0.0, 0.0, 0.0))[1] == 0.0
            and obscodes.get(c, (0.0, 0.0, 0.0))[2] == 0.0 for c in set(codes)):
        return _geocentric_positions(jds)
    T = Time(np.asarray(jds, dtype=float), format="jd", scale="utc")
    lam = np.radians([obscodes[c][0] if c in obscodes else 0.0 for c in codes])
    rcp = np.array([obscodes[c][1] if c in obscodes else 0.0 for c in codes])
    rsp = np.array([obscodes[c][2] if c in obscodes else 0.0 for c in codes])
    sites = A_EARTH_AU * np.column_stack([rcp*np.cos(lam), rcp*np.sin(lam), rsp])  # ITRS au
    if site_itrs is not None:
        for i, si in enumerate(site_itrs):
            if si is not None:
                sites[i] = si                                       # Earth-fixed ITRS site (au)
    itrs = ITRS(CartesianRepresentation((sites * u.au).T), obstime=T)
    site_eq = itrs.transform_to(GCRS(obstime=T)).cartesian.xyz.to(u.au).value.T    # (n,3)
    if sat_pos is not None:
        for i, sp in enumerate(sat_pos):
            if sp is not None:
                site_eq[i] = sp                                     # inertial geocentric au
    return _earth_sun_eq(T) + site_eq


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------
def fetch_obs80(desig, cache=None):
    raw = None
    if cache_fresh(cache, astrometry_ttl(desig)):
        try:
            raw = json.loads(Path(cache).read_text())
        except (json.JSONDecodeError, ValueError):
            Path(cache).unlink()        # corrupt/truncated cache -> drop & re-fetch
    if raw is None:
        body = json.dumps({"desigs": [desig], "output_format": ["OBS80"]}).encode()
        req = urllib.request.Request(OBS_API, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="GET")
        raw = json.loads(urllib.request.urlopen(req, timeout=60).read())
        if cache:
            datadirs.write_cache(cache, json.dumps(raw))
    return raw[0]["OBS80"]            # do NOT strip (preserves col-1 leading spaces)


# ---------------------------------------------------------------------------
# Cross-listed duplicates: one detection reported under two designations
# ---------------------------------------------------------------------------
# The MPC files the same observation under more than one name whenever it has
# identified them: a NEOCP tracklet (P12pwHF) is re-published under the designation
# it received (2013 PY26), a cross-listed survey tracklet rides under both linked
# provisional designations (HZ280 = BV65).  A fetch of both names -- or of one name
# whose family the MPC has already joined -- concatenates the two copies, and the
# fit then carries that tracklet TWICE: double weight, sigma halved, an O-C table
# showing the same line back to back.  So the merge keeps ONE copy per detection.
#
# Identity is the MPC's own: same station, same instant, same sky position.  The
# match is by TOLERANCE, not by exact fields, because the two copies need not be
# byte-identical -- a NEOCP record arrives at full ADES precision while the
# published OBS80 line truncates the time to ~0.9 s and the position to 0.01" --
# yet the tolerances sit far under the cadence of two genuinely distinct
# detections from one station (seconds to minutes apart, and a survey exposure
# CAN hold two objects at the same instant, which is why the position is
# load-bearing: see ident_check._duplicate_excluded and itf.obs80_fields.obs_unique_id
# for the same rule in the identification and ITF paths).
DUP_TIME_DAYS = 1.0 / 86400.0      # 1 s
DUP_POS_DEG = 1.0 / 3600.0         # 1 arcsec

# Unpacked comet / interstellar / A-object forms the asteroid packer does not cover:
# "1P", "73P-B", "C/2020 F3", "P/2019 LD2", "A/2017 U1", "C/2019 Q4-A".
_COMET_UNPACKED_RE = __import__("re").compile(
    r"^(\d+[PDCXI](-[A-Z]{1,2})?|[PDCXIA]/\d{4} [A-Z]{1,2}\d*(-[A-Z]{1,2})?)$")


def is_published_designation(desig):
    """True for a designation the MPC has PUBLISHED: a permanent number ("433"), a
    provisional ("2013 PY26", packed "K13P26Y", extended "_PO00Tq"), a survey form
    ("2040 P-L"), or a comet / interstellar form ("1P", "C/2020 F3", "A/2017 U1").
    False for the temporary ids astrometry travels under before that -- a NEOCP id
    ("P12pwHF", "W023204"), a submitter's trksub, an ITF tracklet id -- which is the
    class the dedup must always lose to a published copy of the same detection."""
    s = str(desig or "").strip()
    if not s:
        return False
    from pyroorb import packing
    if packing.pack(s) is not None or packing.unpack(s) is not None:
        return True
    return bool(_COMET_UNPACKED_RE.match(s))


def fit_reuse_max_age(desigs):
    """How long (seconds) a finished fit of ``desigs`` may be served again, or None
    for "the cache's own lifetime".

    A fit is only as fresh as its astrometry.  A published object's arc grows over
    days, so the web's fit caches may keep its answer for hours; a NEOCP /
    temporary id's arc can double in an hour, so a fit naming ANY such id -- alone
    or in a linkage with published designations -- lives exactly as long as the
    astrometry cache that fed it (``NEOCP_CACHE_TTL_S``).  Without this a pair
    like "P22r4rx P/2015 X3" kept showing the 4 observations its first fit saw
    while "P22r4rx" alone showed all 8."""
    names = [desigs] if isinstance(desigs, str) else list(desigs or [])
    if any(str(d).strip() and not is_published_designation(d) for d in names):
        return NEOCP_CACHE_TTL_S
    return None


def _dup_index(obs):
    """Bucket obs by (obscode, integer JD) so duplicate lookup is O(n), not O(n^2)."""
    idx = {}
    for o in obs:
        jd = float(o.get("jd_utc") or 0.0)
        idx.setdefault((o.get("obscode"), int(jd)), []).append(o)
    return idx


def find_duplicate(o, idx):
    """The already-indexed obs that is the SAME detection as ``o`` (same site, within
    DUP_TIME_DAYS, within DUP_POS_DEG on both axes), or None.  Checks the integer-JD
    bucket and both neighbours so a near-midnight pair straddling a day boundary still
    matches.

    The comparison itself is ``orbit_common.same_detection`` -- shared with
    ``ident_check._duplicate_excluded``, which asks the same question in the
    identification path.  RA is differenced ON THE CIRCLE there, so a pair
    straddling RA 0h (359.99998 and 0.00001) matches like any other.  It is
    deliberately NOT cos(dec)-scaled (``great_circle=False``): the offset this
    tolerance absorbs is the OBS80 round-trip's truncation of the RA COORDINATE
    (HH MM SS.dd -> at most ~0.2" of RA-degrees), which is the same size at every
    declination, so degrees of RA is the currency the tolerance belongs in."""
    jd = float(o.get("jd_utc") or 0.0)
    ra = float(o.get("ra_deg") or 0.0)
    dec = float(o.get("dec_deg") or 0.0)
    oc = o.get("obscode")
    for dd in (-1, 0, 1):
        for b in idx.get((oc, int(jd) + dd), ()):
            if orbit_common.same_detection(
                    jd, ra, dec,
                    float(b.get("jd_utc") or 0.0), float(b.get("ra_deg") or 0.0),
                    float(b.get("dec_deg") or 0.0),
                    time_tol_days=DUP_TIME_DAYS, pos_tol_deg=DUP_POS_DEG,
                    great_circle=False):
                return b
    return None


def dedup_observations(obs):
    """Drop every cross-listed copy of a detection, keeping one.  Returns
    ``(kept, dropped)``: ``kept`` in time order, ``dropped`` one record per removed
    copy -- ``{"designation", "kept_designation", "obscode", "jd_utc"}`` -- so the
    caller can SAY what happened rather than silently shrink the arc.

    Which copy survives is a RULE, not luck: a PUBLISHED designation
    (:func:`is_published_designation` -- number, provisional, survey, comet) always
    beats a temporary one (NEOCP id, trksub, ITF tracklet id), so a fit of
    "P12pwHF 2013 PY26" keeps the 2013 PY26 lines and zaps the NEOCP copies even when
    the temporary id would sort first.  Two copies of the same class fall back to the
    alphabetically-earliest designation -- the convention that names a combined
    object's primary -- so the choice is deterministic.  The designation itself is
    never a MATCH criterion: two copies under two names IS the case this exists for.  Distinct detections at one station within the
    tolerances (a double exposure re-measured?) would collapse too; none is known,
    and the ITF/identification paths apply the same rule.

    GROUP FIRST, THEN CHOOSE.  The rule is applied to a whole group of matched copies,
    never to the order they are visited in -- the first version sorted by ``jd_utc``
    FIRST and kept whichever copy it reached first, so the published-vs-temporary
    preference only ever decided an exact-JD tie.  Two copies of one detection never
    tie on JD: the OBS80 line truncates the time to ~0.9 s while the ADES record keeps
    it in full, which is precisely why the match is by TOLERANCE.  So the documented
    rule was decided by which source happened to round downward -- with the packed
    NEOCP id "P12pwHF" beating "2013 PY26" whenever the NEOCP copy carried the earlier
    timestamp.  Grouping is unchanged (still by tolerance against the earliest copy
    seen, in time order); only the choice within a group is now the rule."""
    if len(obs) < 2:
        return list(obs), []
    def _by_time(o):
        return float(o.get("jd_utc") or 0.0)
    def _rank(o):
        # PUBLISHED beats temporary; same class falls back to the alphabetically-earliest
        # designation (the combined-primary convention).  No time term: see the docstring.
        d = str(o.get("designation") or o.get("packed") or "")
        return (0 if is_published_designation(d) else 1, d)
    idx, groups = {}, {}
    for o in sorted(obs, key=_by_time):
        b = find_duplicate(o, idx)
        if b is not None:
            groups[id(b)].append(o)        # joins the group its match represents
            continue
        groups[id(o)] = [o]                # a new detection: it represents its own group
        idx.setdefault((o.get("obscode"), int(_by_time(o))), []).append(o)
    kept, dropped = [], []
    for g in groups.values():
        keeper = min(g, key=_rank) if len(g) > 1 else g[0]
        kept.append(keeper)
        for o in g:
            if o is not keeper:
                dropped.append({"designation": o.get("designation"),
                                "kept_designation": keeper.get("designation"),
                                "obscode": o.get("obscode"),
                                "jd_utc": _by_time(o)})
    kept.sort(key=_by_time)                # the keeper need not be the group's earliest copy
    return kept, dropped


def fetch_observations(desigs, obscodes, cache_dir=None, prefer_ades=True):
    """Default observation-fetching entry point.

    **Defaults to ADES** (the modern format that carries per-obs rmsRA/rmsDec) and
    only falls back to OBS80 if ADES is unavailable. Returns (obs, meta); always
    auto-combines multiple designations into one object. Radar astrometry — which
    the MPC obs API serves in NEITHER format — is attached as meta['radar'] from
    the cached JPL radar dump (radar.attach_radar; [] for the ~99% of objects with
    none, at the cost of a dict lookup once the dump is cached)."""
    before = dict(_UNKNOWN_OBSCODES)
    obs, meta = _fetch_observations(desigs, obscodes, cache_dir=cache_dir,
                                    prefer_ades=prefer_ades)
    # A station the table lacks may simply be newer than the table: one refresh, and
    # when it lists any of them, update the caller's table in place and resolve again.
    missed = {c for c, n in _UNKNOWN_OBSCODES.items() if n > before.get(c, 0)}
    if missed and refresh_on_miss(missed):
        obscodes.update(load_obscodes())
        _UNKNOWN_OBSCODES.clear()
        _UNKNOWN_OBSCODES.update(before)
        obs, meta = _fetch_observations(desigs, obscodes, cache_dir=cache_dir,
                                        prefer_ades=prefer_ades)
    # One copy per detection (see dedup_observations): the MPC republishes a NEOCP
    # tracklet under the designation it earned and cross-lists linked survey
    # tracklets, so a combined fetch concatenates the same lines twice.  Done here,
    # on the one entry point, so the CLI, the web fit and the ITF seed all fit the
    # same arc.  The record of what was dropped rides on the meta for the reports.
    if obs:
        obs, dropped = dedup_observations(obs)
        if dropped:
            meta["duplicates_dropped"] = dropped
    # WHICH designations the caller asked for, not just how many (n_requested).
    # The identification layer needs the names: a fetch of "A B" comes back with
    # A's already-acknowledged aliases mixed in among A and B, and only the
    # REQUESTED pair is the identification anyone may submit. Stamped here, on the
    # one entry point, so every fetch path (ADES batch, OBS80 batch, per-designation
    # NEOCP fallback) carries it identically.
    # UNPACKED, because that is the spelling every obs carries: a caller who typed
    # the packed form must still be recognisable as having requested it.
    meta.setdefault("requested_designations",
                    [ensure_unpacked(str(d))
                     for d in ([desigs] if isinstance(desigs, str)
                               else list(desigs))])
    if obs and not meta.get("radar"):
        from pyroorb.radar import attach_radar
        attach_radar(meta, cache_dir=cache_dir or (datadirs.cache_dir()))
    return obs, meta


def fetch_failure_message(meta):
    """One plain sentence when the fetch RECORDED transport failures, else None.

    ``meta["fetch_errors"]`` (set by ``_fetch_observations``) is per-designation proof
    that the MPC did not ANSWER -- a statement about the network, never about the
    object.  Callers that go on to FIT the rows must refuse when it is non-empty,
    because the dangerous case is the PARTIAL one: a linkage where one designation's
    fetch failed still returns rows, and fitting them silently answers from half the
    requested evidence while reporting success.  The web routes phrase their own
    refusals from the same record (engine_api.run_fit, fitcache.get_by_desig); this is
    the shared plain-text form for the CLI, --ident-check and the ITF seed fetch."""
    failed = (meta or {}).get("fetch_errors") or {}
    if not failed:
        return None
    names = sorted(failed)
    detail = str(failed[names[0]]).split(";")[0].strip()
    return (f"the MPC did not answer for {', '.join(names)} ({detail}) -- a transport "
            f"failure, not a statement about the object; fitting what did arrive would "
            f"answer from less astrometry than requested. Try again in a moment.")


def _fetch_observations(desigs, obscodes, cache_dir=None, prefer_ades=True):
    from pyroorb.ades import fetch_combined_ades, fetch_combined_neocp, is_neocp_desig
    cache_dir = cache_dir or (datadirs.cache_dir())
    desig_list = [desigs] if isinstance(desigs, str) else list(desigs)
    # Fast path: a single whole-batch fetch from the published-obs API. This is the
    # common case (every designation is a published object) and returns immediately,
    # byte-for-byte as before.
    if prefer_ades:
        try:
            obs, meta = fetch_combined_ades(desigs, obscodes, cache_dir=cache_dir)
            if obs:
                return obs, meta
        except Exception as ex:
            print(f"(ADES batch unavailable: {ex}; falling back to OBS80)")
    # The published-obs API (ADES + OBS80) shares one backend that HTTP-500s on a
    # designation it can't find, so guard the OBS80 attempt: a failure here is the
    # signal to try the NEOCP, not a reason to abort.
    try:
        obs, meta = fetch_combined(desigs, obscodes, cache_dir=cache_dir)
        if obs:
            meta["format"] = "OBS80"
            return obs, meta
    except Exception as ex:
        print(f"(OBS80 batch unavailable: {ex})")
    # Per-designation fallback. The whole-batch fetch above is all-or-nothing: a
    # single designation the published API can't find -- typically an unconfirmed
    # NEOCP / temporary tracklet -- HTTP-500s the entire request, taking its published
    # partner's observations down with it. So fetch each designation independently,
    # routing it to the published API and, only when that has nothing, to the NEOCP
    # CGI (ADES XML), exactly like find_orb. This is what makes a MIXED published +
    # NEOCP pair (e.g. "P12nCge 1997 OH") -- or a pure-NEOCP one -- fittable.
    # WHY a designation came back empty is not a detail -- it is the whole answer.
    # Every fetch below is best-effort, and an exception here means the MPC did not
    # ANSWER (a timeout, a throttle, a TLS failure, a response shape we could not
    # read), which is a statement about the network, not about the object.  Reporting
    # both as "no observations found" tells a user their object does not exist on the
    # strength of a dropped connection -- a confident claim we have no evidence for,
    # and one that reads as authoritative rather than as the outage it is.  So the
    # transport failures are KEPT, per designation, and travel out on the meta for the
    # caller to phrase.  A designation with no error and no rows really is empty.
    merged, used_neocp, used_sbn = [], False, False
    fetch_errors = {}
    for d in desig_list:
        one, why = [], []
        if prefer_ades:
            try:
                one, _ = fetch_combined_ades([d], obscodes, cache_dir=cache_dir)
            except Exception as ex:
                why.append(f"ADES: {type(ex).__name__}: {ex}")
                one = []
        if not one:
            try:
                one, _ = fetch_combined([d], obscodes, cache_dir=cache_dir)
            except Exception as ex:
                why.append(f"OBS80: {type(ex).__name__}: {ex}")
                one = []
        if not one and is_neocp_desig(d):
            try:
                one, _ = fetch_combined_neocp([d], obscodes, cache_dir=cache_dir)
                if one:
                    used_neocp = True
                    print(f"(fetched {len(one)} obs for {d} from the NEOCP)")
            except Exception as ex:
                why.append(f"NEOCP: {type(ex).__name__}: {ex}")
                print(f"(NEOCP fetch failed for {d}: {ex})")
        if not one:
            one = _fetch_from_sbn(d, obscodes, why)
            if one:
                used_sbn = True
        if not one:
            if why:
                fetch_errors[d] = "; ".join(why)
                print(f"(could not reach the MPC for {d}: {why[0]})")
            else:
                print(f"(no observations found for {d})")
        merged += one
    if not merged:
        meta = {"designation": desig_list[0] if desig_list else str(desigs),
                "designations": [], "combined": False, "other_designations": [],
                "fetch_errors": fetch_errors,
                "format": "OBS80"}
        return merged, meta
    merged.sort(key=lambda o: o["jd_utc"])
    present = sorted({o["designation"] for o in merged})
    meta = {"designation": present[0],
            "designations": present,
            "combined": len(present) > 1,
            "n_requested": len(desig_list),   # how many desigs the CALLER asked for (see fit_object)
            "other_designations": present[1:],
            # Carried on the SUCCESS path too, and deliberately: a linkage of two
            # designations where one fetch failed still returns rows, so without this
            # the fit silently proceeds on half its evidence and reports success --
            # a wrong answer that looks like a right one.
            "fetch_errors": fetch_errors,
            "format": "ADES"}
    if used_neocp:
        meta["source"] = "NEOCP"
    elif used_sbn:
        meta["source"] = "SBN"
    return merged, meta


def _fetch_from_sbn(desig, obscodes, why):
    """One designation's published observations from the SBN replica, when
    the MPC Observation API returned none (see sbn_obs).  [] when no replica
    is configured or it has nothing; a replica failure is appended to ``why``
    with the API's own failures."""
    from pyroorb import sbn_obs
    from pyroorb.ades import records_to_obs
    if not sbn_obs.available():
        return []
    try:
        one = records_to_obs(sbn_obs.fetch_records(desig), obscodes)
    except Exception as ex:
        why.append(f"SBN: {type(ex).__name__}: {ex}")
        return []
    if one:
        print(f"(fetched {len(one)} obs for {desig} from the SBN replica)")
    return one


_COMET_TYPE_LETTERS = "CPDXIA"


def obs80_line_id(line):
    """The packed id an OBS80 line is filed under.

    Cols 1-5 hold a permanent number, cols 6-12 a provisional designation or trksub --
    EXCEPT on a comet line, where col 5 is the orbit-type letter and cols 1-4 the periodic
    number. A provisional comet ('    CK26B030' = C/2026 B3) leaves 1-4 blank, so reading
    cols 1-5 as the number named the object 'C'. Such a line's id is the type letter plus
    the core ('CK26B030', the form `obs80.pack_designation` produces); a numbered comet
    ('0073P') keeps cols 1-5 as before."""
    num, prov = line[0:5], line[5:12].strip()
    if prov and num[4:5] and num[4:5] in _COMET_TYPE_LETTERS + "S" and not num[:4].strip():
        return num[4] + prov          # 'S' = a natural-satellite provisional ('    SK21J070')
    return num.strip() or prov


def parse_obs(obs80_blob, obscodes, want_packed=None, radar_out=None):
    """Parse OBS80 lines, tagging each obs with its embedded packed designation.

    By default ALL designations present are kept: an MPC response for one object
    may already contain several designations it has identified as the same body,
    and we honour that linkage (see fetch_combined). Pass `want_packed` to filter
    to a single designation. Radar two-line 'R'/'r' pairs are not optical obs; with
    `radar_out` (a list) they are collected as radar-obs dicts (radar.py), else
    skipped as before."""
    obs = []
    lines = obs80_blob.split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]; i += 1
        if len(line) < 80:
            continue
        obstype = line[14]
        if obstype in "svr":          # a lowercase companion seen alone (no primary) -> skip
            continue
        if obstype == "R":            # radar primary (range/Doppler two-line pair, not optical)
            if i < n and len(lines[i]) >= 80 and lines[i][14] == "r":
                if radar_out is not None:
                    packed = obs80_line_id(line)
                    if not (want_packed and packed != want_packed):
                        from pyroorb.radar import parse_obs80_radar_pair
                        radar_out.extend(parse_obs80_radar_pair(line, lines[i], obscodes))
                i += 1
            continue
        # A two-line record's COMPANION is the next line (col 15 's' = satellite geocentric vector,
        # 'v' = roving-observer geodetic position) -- identified by lookahead, not by the primary's
        # marker, and required to share the primary's packed desig (cols 1-12). The observer then
        # comes from that companion (a spacecraft / mobile site has no obscode ground position), so
        # the obs is kept even though its obscode (C51/WISE, 247) is not a fixed station.
        sat_pos = site_itrs = None
        companion = None
        nxt = lines[i] if i < n else ""
        if len(nxt) >= 61 and nxt[0:12] == line[0:12]:
            if nxt[14] == "s":
                sat_pos = mpc80_satellite_offset(nxt)
                if sat_pos is not None:
                    companion = nxt.rstrip("\n")
                    i += 1
            elif nxt[14] == "v":
                site_itrs = mpc80_roving_itrs_au(nxt)
                if site_itrs is not None:
                    companion = nxt.rstrip("\n")
                    i += 1
        if obstype in "SV" and sat_pos is None and site_itrs is None:
            continue                  # space/roving primary whose companion didn't parse -> drop
        # embedded packed designation: cols 1-5 (numbered) or 6-12 (provisional), comet-aware
        packed = obs80_line_id(line)
        if want_packed and packed != want_packed:
            continue                  # optional guard: keep one designation only
        try:
            jd, ra, dec, code = mpc80_to_radec(line)
        except ValueError:
            continue
        if code not in obscodes and sat_pos is None and site_itrs is None:
            # ground obs need a known station; space/roving are exempt. An unknown ground
            # code means ObsCodes.htm may be stale -- record it so the caller can prompt.
            record_unknown_obscode(code)
            continue
        mag, band = mpc80_magnitude(line)
        o = {"jd_utc": jd, "ra_deg": ra, "dec_deg": dec, "obscode": code,
             "packed": packed, "obs80_line": line.rstrip("\n"),
             "astcat": mpc80_catalog(line),       # col 72; keyed on by the debiasing below
             "mag": mag, "band": band, "mag_sigma": None}
        if sat_pos is not None:
            o["sat_pos_au"] = sat_pos
        if site_itrs is not None:
            o["site_itrs_au"] = site_itrs
        if companion is not None:
            o["obs80_line2"] = companion          # original 's'/'v' companion, kept verbatim
        obs.append(o)
    obs.sort(key=lambda o: o["jd_utc"])
    # Star-catalog debiasing (EFCC18), exactly as the ADES path does it (ades.records_to_obs):
    # subtract the per-catalog systematic offset so every obs is on the Gaia-DR2 frame before
    # fitting. This ran on ADES obs only until now, so the SAME astrometry read as OBS80 was
    # fit undebiased -- a coherent, per-sky-region offset (median 0.26", up to 1.2" for
    # USNO-A2) that is not noise the fit averages away but a bias it absorbs into the
    # elements. The `obs80_line` we keep is the observer's VERBATIM record and stays raw
    # (as does ra_raw/dec_raw); only the fitted ra_deg/dec_deg move. No-op when the bias
    # table is absent, and per-obs for a catalog already on Gaia-DR2/DR3 or unnamed.
    try:
        from pyroorb import debias
        debias.debias_obs(obs)
    except Exception as ex:
        # A missing bias table is handled INSIDE debias_obs (a clean no-op), so reaching
        # here means the debias step itself broke -- and silently shipping raw positions
        # would quietly reintroduce the catalog bias this call exists to remove.
        print(f"[debias] step failed, positions left raw: {type(ex).__name__}: {ex}",
              file=sys.stderr)
    return obs


# Base-62 high digit of a packed permanent number ('A0000' == 100000): any letter, since
# the digits 0-9 are covered by the all-numeric case.
_B62_HIGH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _looks_packed(s):
    """True when `s` has the SHAPE of an MPC packed designation, so handing it to the
    unpacker is safe. This is a shape test, NOT a validity test -- `packing.unpack`
    still decides, and returns None for a lookalike it can't read.

    The point of the guard is the converse: a token that is ALREADY unpacked, or is a
    survey/NEOCP temporary id, must never be run through the unpacker (an unguarded
    unpack turns the comet fragment 'A3' into '103'). Three shapes qualify:

      * a PROVISIONAL packing -- 7 chars opening with a century letter or the extended
        marker ('K16UL0A', '_PO00Tq');
      * a permanent NUMBER packing -- 5 chars, either all digits ('00433'), a base-62
        high digit + 4 digits ('A0000' == 100000), or '~' + 4 base-62 digits ('~0000');
      * a zero-padded / bare catalogue number of 5-8 digits ('00134340', '134340').

    A packed designation never contains a space, so any token with one is already
    unpacked and fails here immediately."""
    if not s or " " in s:
        return False
    if len(s) == 7 and s[0] in "IJK_":
        return True
    if len(s) == 5 and (s.isdigit() or s[0] == "~"
                        or (s[0] in _B62_HIGH and s[1:].isdigit())):
        return True
    return 5 <= len(s) <= 8 and s.isdigit()


def ensure_unpacked(desig):
    """Return the UNPACKED MPC designation for `desig`, accepting EITHER form.

    A packed designation ('K16UL0A', '_PO00Tq', '00433', 'D4340') is unpacked to its
    readable form ('2016 UA210', '2025 OA694', '433', '134340'); an already-unpacked
    designation -- or anything the unpacker can't parse -- is returned unchanged. This
    is the central helper so every entry point (CLI, --pairs batches, the OBS80
    per-obs designation) takes packed and unpacked designations interchangeably.

    WHICH tokens are offered to the unpacker is `_looks_packed`'s call -- see there for
    why the shape guard exists. It used to read `len(s) != 7 or s[0] not in "IJK_"`,
    which excluded every packed NUMBER form: 'D4340' (Pluto) came back unchanged, so
    guards.is_pluto could not recognise pasted Pluto astrometry, and OBS80-sourced
    numbered objects carried a packed '00433' as their designation into MPCORB/SOF/the
    identification JSON while their ADES twins carried '433'."""
    s = str(desig).strip()
    from pyroorb import packing
    # Comet packings (`obs80_line_id`): 'CK26B030' -> 'C/2026 B3', '0073P' -> '73P'.
    m = re.fullmatch(r"([%s])([IJK]\d\d[A-Z][0-9A-Za-z]\d[0-9a-z])" % _COMET_TYPE_LETTERS, s)
    if m:
        core = packing.unpack_comet(m.group(2)) or packing.unpack(m.group(2))
        if core:
            frag = m.group(2)[6]
            return f"{m.group(1)}/{core}" + (f"-{frag.upper()}" if frag.islower() else "")
    # Natural-satellite provisional (`obs80.pack_designation`): 'SK21J070' -> 'S/2021 J 7'.
    m = re.fullmatch(r"S([IJK]\d\d[JSUN][0-9A-Za-z]\d0)", s)
    if m:
        core = packing.unpack_comet(m.group(1))
        if core:
            year, rest = core.split()
            return f"S/{year} {rest[0]} {rest[1:]}"
    m = re.fullmatch(r"(\d{4})([%s])([a-z]{0,2})" % _COMET_TYPE_LETTERS, s)
    if m and int(m.group(1)):
        return f"{int(m.group(1))}{m.group(2)}" + (f"-{m.group(3).upper()}" if m.group(3) else "")
    if not _looks_packed(s):
        return s
    try:
        native = packing.unpack(s)       # numbered/provisional/extended-'_' forms
    except Exception:                    # malformed lookalike -> treat as already unpacked
        return s
    return native if native is not None else s     # unparseable -> return unchanged


def _unpack(packed):
    return ensure_unpacked(packed)


def fetch_combined(desigs, obscodes, cache_dir=None):
    """Fetch one or more designations and merge their observations into a single
    'combined object', exactly as if the MPC API had returned one object whose
    response is a mash-up of several designations.

    The combined object's primary name is the alphabetically-earliest UNPACKED
    designation (string sort); the rest become `other_designations`. This handles
    both cases identically: (a) a single query that comes back with >1 designation
    because MPC already identified them, and (b) several designations we pass in
    under the hypothesis that they are the same body.

    Returns (obs, meta). Each obs carries its source `designation`."""
    if isinstance(desigs, str):
        desigs = [desigs]
    cache_dir = Path(cache_dir or (datadirs.cache_dir()))
    obs = []
    radar_rows = []
    for d in desigs:
        blob = fetch_obs80(d, cache=cache_dir / f"obs80_{cache_stem(d)}.json")
        obs += parse_obs(blob, obscodes, radar_out=radar_rows)   # ALL designations present
    obs.sort(key=lambda o: o["jd_utc"])
    for o in obs:
        o["designation"] = _unpack(o["packed"])
    for r in radar_rows:
        r["desig"] = _unpack(r["desig"])
    present = sorted({o["designation"] for o in obs})     # string sort
    meta = {"designation": present[0],
            "designations": present,
            "combined": len(present) > 1,
            "n_requested": len(desigs),     # how many desigs the CALLER asked for (see fit_object)
            "other_designations": present[1:]}
    if radar_rows:
        meta["radar"] = radar_rows
    return obs, meta


# ---------------------------------------------------------------------------
# Astrometry-file readers (OBS80 / ADES PSV/XML), identification-agnostic grouping
# ---------------------------------------------------------------------------
def _detect_obs_format(text):
    """Sniff an astrometry blob: 'ades' (PSV or XML) vs 'obs80'. ADES XML opens with
    an <ades>/<optical> tag; ADES PSV's first non-comment line is a '|'-delimited
    header naming permID/provID/trkSub/obsTime. Everything else is OBS80."""
    head = text.lstrip()[:8000]
    low = head.lower()
    if low.startswith("<?xml") or "<ades" in low or "<optical" in low:
        return "ades"
    # ADES PSV: a '|'-delimited header naming permID/provID/trkSub/obsTime, possibly
    # preceded by '#'/'!' metadata directives. OBS80 lines are fixed 80-col with no '|',
    # so scanning for that header can't false-positive on OBS80.
    for ln in head.splitlines():
        s = ln.strip()
        if not s or s[0] in "#!":
            continue                       # ADES PSV '# version' / '! mpcCode' directives
        if "|" in s and any(tok in s for tok in ("permID", "provID", "trkSub", "obsTime")):
            return "ades"
    return "obs80"


def read_obs_file(text, obscodes, in_format=None, radar_out=None):
    """Parse an astrometry FILE (OBS80 or ADES PSV/XML) into internal obs dicts.
    Auto-detects the format unless `in_format` ('obs80'/'ades') forces it. Returns
    (obs, format_label). Every obs is tagged with its embedded `designation`.
    `radar_out` (a list) collects any radar records the file carries (OBS80 'R'/'r'
    pairs or ADES delay/doppler records), each tagged with its `desig`."""
    fmt = (in_format or _detect_obs_format(text)).lower()
    if fmt == "ades":
        from pyroorb import ades
        obs = ades.parse_ades_file(text, obscodes, radar_out=radar_out)
        return obs, "ADES"
    obs = parse_obs(text, obscodes, radar_out=radar_out)   # OBS80; tags `packed`
    for o in obs:
        o.setdefault("designation", _unpack(o.get("packed", "")))
    if radar_out:
        for r in radar_out:
            r["desig"] = _unpack(r["desig"]) or r["desig"]
    return obs, "OBS80"


def _group_by_designation(obs):
    """Group obs by their embedded designation (identification-agnostic: each distinct
    designation is its own object; no cross-designation linking). Returns an ordered
    dict {designation: [obs, ...]} sorted by designation."""
    groups = {}
    for o in obs:
        d = o.get("designation") or _unpack(o.get("packed", ""))
        o["designation"] = d
        groups.setdefault(d, []).append(o)
    return {d: groups[d] for d in sorted(groups)}
