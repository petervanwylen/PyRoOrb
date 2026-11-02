"""Fetch a published orbit from the MPC orbits API and convert it to the engine's
heliocentric-equatorial state vector.

This is the seed source for the well-observed-object FAST PATH in fit_object.py: an
object with hundreds of observations across many oppositions already has a published
MPC orbit far better than any IOD triangle, so for those objects we skip the
IOD/subarc/extend cascade and seed the differential corrector straight from here (the
normal drop/reclaim rejection still runs). See fit_object._fit_from_mpc_seed.

API (https://www.minorplanetcenter.net/mpcops/documentation/orbits-api/):

    GET https://data.minorplanetcenter.net/api/get-orb   body: {"desig": "<name>"}

returns a JSON list; the element carrying ``mpc_orb`` holds a one-element list whose
dict has CAR / COM / KEP coordinate blocks, ``epoch_data`` and ``system_data``.

The CAR block is heliocentric, ICRF, **ecliptic** of J2000 (system_data.refsys ==
"Ecliptic"), positions in au and velocities in au/day -- so converting to the engine's
heliocentric **equatorial** state is a single obliquity rotation about x. The epoch is
MJD in TDT (== TT, which differs from TDB by < 2 ms -- negligible for a seed), so the
TDB Julian date is simply MJD + 2400000.5. Both the frame conversion and the units were
validated against the published KEP elements (433 Eros reproduces a/e/i/node/argperi/M
exactly after the rotation).
"""

import json
import math
import ssl
import urllib.request

import numpy as np

ORB_API = "https://data.minorplanetcenter.net/api/get-orb"
_CTX = ssl.create_default_context()

# J2000 mean obliquity -- identical to orbital_elements.OBLIQ so the round-trip through
# element_vector is exact (kept local to avoid importing fit_object, which imports us).
_OBLIQ = math.radians(23.43928111)
_MJD_TO_JD = 2400000.5


def _ecl_to_eq(v):
    """Rotate an ecliptic-J2000 3-vector to the equatorial (ICRS) frame about +x by the
    obliquity -- the inverse of orbital_elements._eq_to_ecl."""
    c, s = math.cos(_OBLIQ), math.sin(_OBLIQ)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def _extract_mpc_orb(payload):
    """Pull the single ``mpc_orb`` dict out of the API's list-of-dicts response, or None."""
    if not isinstance(payload, list):
        return None
    for el in payload:
        if isinstance(el, dict) and el.get("mpc_orb"):
            blk = el["mpc_orb"]
            return blk[0] if isinstance(blk, list) and blk else blk
    return None


def _get_orb_raw(desig, cache=None, timeout=60):
    """Raw get-orb JSON list for `desig`, using `cache` (a path) if present.

    Shared by fetch_mpc_orbit (the fast-path seed) and fetch_designation_info (the
    number/name lookup): both hit the SAME endpoint with the SAME body, so a single
    cached response serves either. Genuine network/SSL errors propagate."""
    from pathlib import Path

    if cache and Path(cache).exists():
        try:
            return json.loads(Path(cache).read_text())
        except (json.JSONDecodeError, ValueError):
            Path(cache).unlink()                 # corrupt/truncated cache -> re-fetch
    body = json.dumps({"desig": str(desig)}).encode()
    req = urllib.request.Request(ORB_API, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="GET")
    raw = json.loads(urllib.request.urlopen(req, context=_CTX, timeout=timeout).read())
    if cache:
        # A CACHE IS AN OPTIMISATION AND MUST NEVER FAIL THE FETCH.  This wrote
        # straight through, so on a container whose data directory did not exist yet
        # the write raised FileNotFoundError AFTER the MPC had already answered --
        # turning a successful lookup into an error, and (via fetch_mpc_seed's
        # catch-all) silently demoting the published-orbit fast path back to IOD
        # until something else happened to create the directory first.
        from pyroorb import datadirs
        datadirs.write_cache(cache, json.dumps(raw))
    return raw


def fetch_designation_info(desig, cache=None, timeout=60):
    """Number + name for `desig` from the MPC orbits API, or None if unavailable.

    Returns a dict:
        number  -- MP number as a string (e.g. '279'), or None if unnumbered
        name    -- IAU name (e.g. 'Thule'), or None if unnamed
        aliases -- set of every designation form MPC records for this same body
                   (primary/secondary provisionals, the number, the name); lets a
                   caller resolving a list of designations recognise that several
                   entries are one object and skip the redundant lookups.

    Display-only (drives the web UI's "(279) Thule" object label). Never raises on a
    missing/malformed orbit -- returns None so the caller falls back to the raw
    designation; genuine network/SSL errors propagate (the caller wraps the call)."""
    m = _extract_mpc_orb(_get_orb_raw(desig, cache=cache, timeout=timeout))
    dd = (m or {}).get("designation_data") or {}
    number = (str(dd.get("permid") or "").strip()) or None
    name = (str(dd.get("name") or "").strip()) or None
    if not (number or name):
        return None
    aliases = set()
    primary = dd.get("unpacked_primary_provisional_designation")
    if primary:
        aliases.add(str(primary).strip())
    for a in (dd.get("unpacked_secondary_provisional_designations") or []):
        aliases.add(str(a).strip())
    if number:
        aliases.add(number)
    if name:
        aliases.add(name)
    return {"number": number, "name": name, "aliases": aliases}


def fetch_mpc_orbit(desig, cache=None, timeout=60):
    """Return the published MPC orbit for `desig` as a seed dict, or None if unavailable.

    The dict has:
        state_eq   -- np.array([x,y,z,vx,vy,vz]) heliocentric EQUATORIAL, au & au/day
        epoch_tdb  -- TDB Julian date of the state
        n_opp      -- number of oppositions in the MPC fit (orbit_fit_statistics.nopp)
        nobs_total -- total observations in the MPC fit
        nongrav    -- True if the MPC orbit carries a non-gravitational model
        comet_nongrav -- {"A1","A2","A3"} in au/day^2 when that model is the engine's
                      water-ice comet g(r) (see _comet_nongrav_params), else None
        raw        -- the full mpc_orb dict (for diagnostics)

    Accepts packed or unpacked designations. Caches the raw JSON to `cache` (a path),
    mirroring the OBS80/ADES fetchers. Returns None -- never raises -- on a missing
    orbit or a malformed response, so the caller falls back to IOD; genuine network/SSL
    errors propagate (the caller wraps the call in try/except)."""
    m = _extract_mpc_orb(_get_orb_raw(desig, cache=cache, timeout=timeout))
    if m is None:
        return None
    try:
        car = m["CAR"]
        names = car["coefficient_names"]
        vals = car["coefficient_values"]
        cv = dict(zip(names, vals))
        pos = _ecl_to_eq(np.array([cv["x"], cv["y"], cv["z"]], float))
        vel = _ecl_to_eq(np.array([cv["vx"], cv["vy"], cv["vz"]], float))
        ep = m["epoch_data"]
        if str(ep.get("timeform", "MJD")).upper() != "MJD":
            return None                          # only the documented MJD form is handled
        epoch_tdb = float(ep["epoch"]) + _MJD_TO_JD
    except (KeyError, TypeError, ValueError):
        return None
    stats = m.get("orbit_fit_statistics", {}) or {}
    ng = bool((m.get("non_grav_booleans") or {}).get("non_gravs"))
    return {"state_eq": np.concatenate([pos, vel]),
            "epoch_tdb": epoch_tdb,
            "n_opp": stats.get("nopp"),
            "nobs_total": stats.get("nobs_total"),
            "nongrav": ng,
            "comet_nongrav": _comet_nongrav_params(m, cv) if ng else None,
            "raw": m}


def _comet_nongrav_params(m, cv):
    """The published orbit's out-gassing coefficients as the engine's comet model would carry
    them -- ``{"A1": .., "A2": .., "A3": ..}`` in physical au/day^2 (absent axes 0.0) -- or
    None when the published model is not the engine's water-ice g(r).

    The MPC states the model in ``non_grav_booleans.non_grav_model``: ``marsden`` (Marsden
    et al. 1973) and ``yc`` (Yeomans & Chodas 1989, the same g(r) with an optional
    perihelion offset DT) both drive A1/A2/A3 on the water-ice g(r) that nongrav.COMET_GR
    holds, and their CAR block lists the coefficients in au/day^2 beside the state -- so
    the values seed refine_nongrav's parameter vector directly. A DT (asymmetric) solution,
    the Yabushita (CO) g(r), and the SRP / Yarkovsky inverse-square family are NOT the same
    force law, and a seed from them would start the comet LM at a point that means something
    else; those return None and the fast path treats the orbit as a gravity seed. A model
    whose coefficient flags are set but whose CAR block carries no values (seen on some
    older solutions) also returns None rather than seeding zeros."""
    nb = m.get("non_grav_booleans") or {}
    kinds = nb.get("non_grav_model") or {}
    coeffs = nb.get("non_grav_coefficients") or {}
    if not (kinds.get("marsden") or kinds.get("yc")):
        return None
    if coeffs.get("DT") or kinds.get("srp") or kinds.get("yarkovsky") or kinds.get("yabushita"):
        return None
    out = {}
    for ax in ("A1", "A2", "A3"):
        if coeffs.get(ax):
            v = cv.get(ax)
            if v is None:
                return None
            try:
                out[ax] = float(v)
            except (TypeError, ValueError):
                return None
        else:
            out[ax] = 0.0
    if not any(out.values()):
        return None
    return out


def fetch_published_elements(desig, cache=None, timeout=60):
    """The MPC's OWN published orbit for `desig`, display-ready, or None.

    This is not our fit and must never be presented as one -- every field here was
    computed by the Minor Planet Center, and the caller is responsible for saying so
    (the payload carries ``source: "MPC"`` so a UI cannot render it unattributed by
    accident).  It exists because a well-observed object's published orbit is available
    in one cheap request while FITTING that object's whole arc takes minutes, and a
    visitor who typed "433" is better served by the known orbit now, with our fit
    arriving after it, than by ten minutes of nothing.

    Returns None -- never raises for a missing or malformed orbit -- so a caller can
    treat absence as "we have nothing to show yet"; genuine network/SSL errors
    propagate, exactly as in :func:`fetch_mpc_orbit`, so a fetch FAILURE is
    distinguishable from an object the MPC has no orbit for.
    """
    m = _extract_mpc_orb(_get_orb_raw(desig, cache=cache, timeout=timeout))
    if m is None:
        return None
    kep = m.get("KEP") or {}
    names, vals = kep.get("coefficient_names"), kep.get("coefficient_values")
    if not names or not vals or len(names) != len(vals):
        return None
    el = dict(zip(names, vals))
    ep = m.get("epoch_data") or {}
    epoch_jd = None
    if str(ep.get("timeform", "")).upper() == "MJD":
        try:
            epoch_jd = float(ep["epoch"]) + _MJD_TO_JD
        except (KeyError, TypeError, ValueError):
            epoch_jd = None
    stats = m.get("orbit_fit_statistics") or {}
    mag = m.get("magnitude_data") or {}
    dd = m.get("designation_data") or {}
    return {
        "source": "MPC",
        "designation": dd.get("iau_designation") or dd.get("orbfit_name") or str(desig),
        "name": dd.get("iau_name") or dd.get("name"),
        "epoch_jd": epoch_jd,
        "elements": {"a": el.get("a"), "e": el.get("e"), "incl": el.get("i"),
                     "node": el.get("node"), "argperi": el.get("argperi"),
                     "mean_anomaly": el.get("mean_anomaly")},
        "H": mag.get("H"), "G": mag.get("G"),
        "n_oppositions": stats.get("nopp"),
        "n_observations": stats.get("nobs_total"),
        "arc": stats.get("arc_length_total"),
        "nongrav": bool((m.get("non_grav_booleans") or {}).get("non_gravs")),
    }
