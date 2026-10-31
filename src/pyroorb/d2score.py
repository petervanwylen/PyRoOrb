"""digest2 class scores for an arc that does not determine its orbit.

When `constraint.assess` withholds the nominal orbit, the panel used to say only what the arc
could NOT do. digest2 (the MPC's short-arc classifier; vendored as `pydigest2/`, see its
VENDORED.md) answers the question an undetermined arc CAN answer: across every bound orbit this
astrometry admits, what share of the modelled Solar System population looks like an NEO, a
Mars-crosser, an inner-belt object...? Each class score D2 runs 0-100 and is digest2's own
number, the one the MPC posts to the NEOCP on (NEO >= 65) -- so a user can read it the way they
already read the NEOCP.

Design choices: the RAW flavour (the full modelled population), not no-ID (the
population minus already-identifiable objects) -- the panel describes the object, not the
odds that it is new. The scores are DISPLAY ONLY. They come from a population prior, so they
must never decide a hazard, identification, ITF or NEOCP question: digest2 would answer those
by assuming the population, and an NEO at low elongation near the ecliptic is exactly the case
it mistakes for a main-belt object (Keys et al. 2019, sec. 4).

Classes are NOT exclusive and do not sum to 100: NEO contains N22 and N18, and every class is
scored against everything outside it. Report them as digest2 does, one score per class.
"""
import math
import threading

_ENGINE = None
_LOCK = threading.Lock()
_ARCSEC_RAD = math.pi / (180.0 * 3600.0)

# Which classes the panel shows, in digest2's own column order, and what it calls them. "Int"
# (MPC-interesting) is a union of the others plus large-q/high-e/i objects, which reads as
# noise beside the dynamical classes, so it is left out.
SHOWN = ("NEO", "N22", "N18", "MC", "Hun", "Pho", "MB1", "Pal", "Han", "MB2", "MB3",
         "Hil", "JTr", "JFC")
MIN_SHOWN = 1          # a class scoring below this is not listed (digest2 prints 0s)


def engine():
    """The one loaded digest2 engine (population model + its obscodes), built on first use.
    Double-checked: the cached read is lock-free, only the one-time load serialises (the
    get_ephem lesson -- concurrent web fits must not each parse the 4 MB model)."""
    global _ENGINE
    eng = _ENGINE
    if eng is not None:
        return eng
    with _LOCK:
        if _ENGINE is None:
            from pyroorb.digest2 import Digest2Engine
            from pyroorb.digest2.classes import N_CLASSES
            from pyroorb.digest2.paths import find_mpc_config_path
            # The MPC's own config: per-station astrometric errors (obserr703=0.54, ...) in
            # place of a flat 1". Without it our NEO score sat 13-16 points under the score the
            # MPC posts for the same tracklet (P22qGuz 52.5 -> 80.1, posted 80; P22qGGJ
            # 55.7 -> 71.5, posted 76; 2026-10-02 NEOCP, no-ID flavour). The config also limits
            # the computed classes to NEO alone -- all of them are wanted here.
            e = Digest2Engine.load(config_path=find_mpc_config_path())
            e.config.class_compute = list(range(N_CLASSES))
            # A page must not flicker between reloads: digest2's search jitters its midpoints
            # with a PRNG, and `repeatable` fixes the seed per tracklet.
            e.config.repeatable = True
            _ENGINE = e
        return _ENGINE


def _vmag(o):
    """V-band magnitude for digest2, 0.0 when none was reported. Converted with OUR band table
    (photometry.band_shift, keyed on the full ADES band) rather than digest2's one-letter one:
    the engine's own table is the researched one."""
    m = o.get("mag")
    if m is None:
        return 0.0
    try:
        m = float(m)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(m) or m <= 0:
        return 0.0
    from pyroorb import photometry
    return m + photometry.band_shift(o.get("band"))


def to_d2(o):
    """One engine observation dict -> a pydigest2 Observation (MJD UTC, radians, V)."""
    from pyroorb.digest2 import Observation
    d = Observation(mjd=float(o["jd_utc"]) - 2400000.5,
                    ra=math.radians(float(o["ra_deg"])), dec=math.radians(float(o["dec_deg"])),
                    vmag=_vmag(o), obscode=str(o.get("obscode") or "500"))
    # Space-based and roving observers carry their own geocentric position (au, equatorial),
    # which is what digest2's 's'/'v' second lines give it.
    pos = o.get("sat_pos_au")
    if pos is None:
        pos = o.get("site_itrs_au")
    if pos is not None:
        try:
            d.earth_observer = tuple(float(x) for x in pos)
            d.spacebased = True
        except (TypeError, ValueError):
            pass
    ra_s, dec_s = o.get("ades_rms_ra"), o.get("ades_rms_dec")
    if ra_s and dec_s:
        d.rms_ra = float(ra_s) * _ARCSEC_RAD
        d.rms_dec = float(dec_s) * _ARCSEC_RAD
    return d


def scores(obs):
    """digest2 class scores for `obs` (engine observation dicts), BOTH flavours, or None when
    digest2 declines it (fewer than two observations, no motion) or fails.

    ``{"flavour": "raw", "scores": {abbr: D2}, "shown": [[abbr, heading, D2], ...],
       "noid_scores": {...}, "shown_noid": [...], "rms": great-circle rms (arcsec), "n_obs": int}``

    RAW is the full modelled population (what kind of object could this be); NO-ID is the
    population minus objects already identifiable (what kind of NEW object could this be) and is
    the flavour the MPC posts on the NEOCP. The page shows raw by default and toggles;
    `flavour` names the default.
    """
    obs = sorted((o for o in obs if o.get("jd_utc") is not None), key=lambda o: o["jd_utc"])
    if len(obs) < 2:
        return None
    try:
        from pyroorb.digest2.classes import CLASS_ABBR, CLASS_HEADING
        olist = [to_d2(o) for o in obs]
        is_ades = all(o.rms_ra > 0 and o.rms_dec > 0 for o in olist)
        res = engine().score(olist, is_ades=is_ades)
    except Exception:
        return None
    heading = dict(zip(CLASS_ABBR, CLASS_HEADING))

    def table(sc):
        d = {CLASS_ABBR[i]: float(v) for i, v in sc.items()}
        shown = [[a, heading[a], round(d[a])] for a in SHOWN
                 if a in d and math.isfinite(d[a]) and d[a] >= MIN_SHOWN]
        return d, shown

    raw, shown = table(res.raw_scores)
    noid, shown_noid = table(res.noid_scores)
    return {"flavour": "raw", "scores": raw, "shown": shown,
            "noid_scores": noid, "shown_noid": shown_noid,
            "rms": float(res.rms), "n_obs": len(olist)}


# --- tracklet by tracklet ---------------------------------------------------------------------
TRACKLET_MAX_HOURS = 4.0     # a tracklet without an id is one station's obs
                             # within 4 hours of its first


def tracklets(obs):
    """Group `obs` (any order) into tracklets: ``[[index, ...], ...]`` (indices into `obs`, each
    group in time order, groups in order of their first observation).

    An observation that carries its submitter's tracklet id (ADES trkSub, kept on NEOCP obs as
    `trksub`) is grouped by (station, trkSub) -- the observer's own statement of which
    detections are one tracklet. Every other observation is grouped by station, a tracklet
    closing TRACKLET_MAX_HOURS after its first observation."""
    by_id, by_station = {}, {}
    order = sorted(range(len(obs)), key=lambda k: obs[k].get("jd_utc") or 0.0)
    groups = []
    for k in order:
        o = obs[k]
        code = str(o.get("obscode") or "")
        tid = o.get("trksub")
        if tid:
            key = (code, str(tid))
            if key not in by_id:
                by_id[key] = []
                groups.append(by_id[key])
            by_id[key].append(k)
            continue
        cur = by_station.get(code)
        if cur is None or (o["jd_utc"] - obs[cur[0]]["jd_utc"]) * 24.0 > TRACKLET_MAX_HOURS:
            cur = [k]
            by_station[code] = cur
            groups.append(cur)
        else:
            cur.append(k)
    return groups


def tracklet_scores(obs):
    """digest2 scores per tracklet of `obs`: ``[{"first": index, "indices": [...], "station",
    "n", **scores(...)}, ...]`` -- one entry per tracklet digest2 could score (a lone observation
    cannot be, and is left out). Indices are into `obs`."""
    out = []
    for g in tracklets(obs):
        if len(g) < 2:
            continue
        sc = scores([obs[k] for k in g])
        if not sc:
            continue
        out.append({"first": g[0], "indices": g, "station": obs[g[0]].get("obscode"),
                    "n": len(g), **sc})
    return out
