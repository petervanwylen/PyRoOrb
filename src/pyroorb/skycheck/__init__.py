"""Sky check: which known objects are in a field, or which known object a tracklet is.

The MPC's MPChecker answers these two questions; this is an independent implementation over the
MPC's own published orbits (``MPCORB.DAT``, ``CometEls.txt``, the NEO Confirmation Page):

* :func:`check_field` -- every catalogued object within a radius of a sky position at a time,
  as seen from an observatory: offsets, magnitude, motion;
* :func:`identify` -- an astrometry file grouped into tracklets (one station, one object, a few
  hours), each checked once: which catalogued object, if any, fits ALL its positions.

How the positions are computed (a two-body quick pass with measured margins, then numerical
integration of the survivors) is in :mod:`pyroorb.skycheck.check`. Positions are best within a
year of the catalogue's epoch (MPCORB re-osculates its orbits about twice a year): the quick
pass's margins, measured to 450 days, grow beyond that, and the near-Earth ephemeris covers 400
days before the catalogue was built to 60 days after.
"""
import math
import os
from dataclasses import dataclass, field

import numpy as np

from pyroorb.skycheck import catalogue as _catalogue
from pyroorb.skycheck import check as _check

DEFAULT_RADIUS_ARCMIN = 15.0        # MPChecker's default search radius
TRACKLET_HOURS = 4.0                # a tracklet without an id: one station's obs of one object
                                    # within this long of its first
# Matching, measured on 18,249 published tracklets of 11 stations over a year (every one
# correct after the MPC's later identifications, none wrongly identified): the true object's
# mean offset is 0.17" median, 1.4" at the 99th percentile; its motion rms 0.18" median, 2.1"
# at the 99.9th, and above 4" only for two close (0.014 au), fast (44-67"/min) NEOs whose
# catalogued orbits mispredict the RATE by 15-30%.
MATCH_ARCSEC = 30.0                 # a tracklet whose mean offset is within this ...
MOTION_ARCSEC = 4.0                 # ... and whose positions follow the predicted motion to
                                    # this (rms about the mean offset) is that object
FAST_RATE = 10.0                    # arcsec/min: a fast mover's motion tolerance also allows
FAST_FRACTION = 0.3                 # its rate to be off by this fraction (the tolerance grows
                                    # by this fraction of the predicted track's own rms)
COMET_MATCH_ARCSEC = 120.0          # comets: the published elements carry no non-gravitational
                                    # terms, so an active comet drifts (11P: 50" off, 0.2" rms)

# How far from the catalogue's epoch a check may look, in days: the cost of a check grows
# steeply with that distance, because the quick pass's margin grows with it. Over the 1.58 M
# orbits of the catalogue one tracklet keeps ~1,000 candidates at 200 days, ~110,000 at 500,
# ~495,000 at 800 and every orbit from ~1,400 days; 20 years back that is ~3 hours of ASSIST
# integration. A caller may cap it (``max_days=`` of check_field / identify); the library sets no
# cap of its own. The /skycheck web page caps what its visitors may ask with MAX_DAYS_ENV (unset:
# no limit), so that no request can occupy a public server for long; the server's own callers,
# the command line and the Python API are not capped by it.
MAX_DAYS_ENV = "PYROORB_SKYCHECK_MAX_DAYS"


class TooFarFromEpoch(ValueError):
    """A check asked for a time farther from the catalogue's epoch than its ``max_days`` allows."""


def max_days():
    """The web page's reach limit: the farthest (days) a time its visitors ask about may be from
    the catalogue's epoch, read from the ``PYROORB_SKYCHECK_MAX_DAYS`` environment variable at
    each call; None when unset or empty (no limit). Only the web add-on applies it (as the
    ``max_days`` of :func:`check_field` / :func:`identify`). Anything but a positive number is a configuration error (ValueError): a server
    that meant to set a limit must not silently run without one."""
    raw = os.environ.get(MAX_DAYS_ENV, "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        v = math.nan
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{MAX_DAYS_ENV}={raw!r}: expected a positive number of days")
    return v


def _too_far(cat, times_utc, limit):
    """Why ``times_utc`` may not be checked against ``cat`` under ``limit`` (days), or None."""
    ep = cat.meta.get("main_epoch")
    if limit is None or ep is None:
        return None
    far = float(np.max(np.abs(np.asarray(times_utc, float) - ep)))
    if far <= limit:
        return None
    return (f"the time is {far:.0f} days from the catalogue's epoch (JD {ep:.1f}); this "
            f"check is limited to times within {limit:g} days of it")


# --------------------------------------------------------------------------- time / position
def julian_date(when):
    """UTC Julian date from a float JD, a ``datetime`` (naive = UTC), or an ISO / MPC-style
    string ('2026-11-14T03:21:00', '2026 11 14.1396')."""
    from astropy.time import Time
    import datetime as _dt
    if isinstance(when, (int, float, np.floating)):
        return float(when)
    if isinstance(when, _dt.datetime):
        if when.tzinfo is not None:
            when = when.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return float(Time(when, scale="utc").jd)
    s = str(when).strip()
    parts = s.split()
    if len(parts) == 3 and all(p.replace(".", "", 1).isdigit() for p in parts):
        y, m, d = int(parts[0]), int(parts[1]), float(parts[2])
        return float(Time(f"{y:04d}-{m:02d}-01T00:00:00", scale="utc").jd) + d - 1.0
    try:
        return float(s)
    except ValueError:
        return float(Time(s.replace(" ", "T") if "T" not in s and ":" in s else s,
                          scale="utc").jd)


def angle_deg(value, hours=False):
    """Degrees from a number or a sexagesimal string ('12 34 56.7', '-05:06:07');
    ``hours`` reads the sexagesimal form as hours (right ascension)."""
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    s = str(value).strip().replace(":", " ")
    parts = s.split()
    if len(parts) == 1:
        return float(parts[0])
    neg = parts[0].startswith("-")
    vals = [abs(float(p)) for p in parts] + [0.0, 0.0]
    deg = vals[0] + vals[1] / 60.0 + vals[2] / 3600.0
    deg = -deg if neg else deg
    return deg * 15.0 if hours else deg


def sexagesimal(ra_deg, dec_deg):
    """'HH MM SS.ss', '+DD MM SS.s' (the layout of an MPC report)."""
    h = (ra_deg % 360.0) / 15.0
    hh, mm = int(h), int((h - int(h)) * 60)
    ss = ((h - hh) * 60 - mm) * 60
    if ss >= 59.995:
        ss, mm = 0.0, mm + 1
    if mm >= 60:
        mm, hh = 0, (hh + 1) % 24
    sign = "-" if dec_deg < 0 else "+"
    d = abs(dec_deg)
    dd, dm = int(d), int((d - int(d)) * 60)
    ds = ((d - dd) * 60 - dm) * 60
    if ds >= 59.95:
        ds, dm = 0.0, dm + 1
    if dm >= 60:
        dm, dd = 0, dd + 1
    return f"{hh:02d} {mm:02d} {ss:05.2f}", f"{sign}{dd:02d} {dm:02d} {ds:04.1f}"


# --------------------------------------------------------------------------- results
@dataclass
class FieldCheck:
    """The answer of :func:`check_field`. ``objects`` holds one dict per object, nearest first:
    ``designation``, ``name``, ``kind`` ('asteroid', 'comet', 'NEOCP'), ``ra``/``dec`` (deg,
    astrometric J2000), ``offset_ra``/``offset_dec`` (arcsec, object minus field centre, RA
    times cos dec), ``separation`` (arcsec), ``mag`` (V, or a comet's total magnitude),
    ``rate_ra``/``rate_dec`` (arcsec/min), ``r``/``delta`` (au), ``phase``/``elongation``
    (deg), ``U`` (MPC uncertainty parameter, asteroids), ``oppositions``; a NEOCP object adds
    ``variants`` (orbits the MPC lists), ``in_field`` (how many of them fall inside the radius)
    and ``spread`` (arcsec: the farthest variant from the nominal)."""
    time_utc: float
    ra: float
    dec: float
    radius_arcmin: float
    obscode: str
    objects: list
    notes: list = field(default_factory=list)
    catalogue: dict = field(default_factory=dict)

    def table(self):
        """The result as a fixed-width text table (the CLI's output)."""
        ra_s, dec_s = sexagesimal(self.ra, self.dec)
        out = [f"Field {ra_s} {dec_s}  radius {self.radius_arcmin:g}'  station "
               f"{self.obscode}  JD {self.time_utc:.5f} UTC",
               f"{len(self.objects)} object(s)"]
        out += [f"note: {n}" for n in self.notes]
        if self.objects:
            out.append(_HEADER)
            out += [_row_text(o) for o in self.objects]
        return "\n".join(out)


@dataclass
class TrackletCheck:
    """One tracklet of :func:`identify`: its observations (``indices`` into the input),
    station, time span, and ``candidates`` -- dicts as :class:`FieldCheck` objects plus
    ``mean_offset`` (arcsec, observed minus predicted), ``motion_rms`` (arcsec: how far the
    positions stray from the predicted motion once that offset is removed) and ``match``
    (True when both are within the matching tolerances), best first. ``identification`` is
    the designation of the matching object, or None: no match, or several with none clearly
    nearest (then ``ambiguous`` is True). ``refused`` is the reason a tracklet was NOT checked
    (its time is beyond the ``max_days`` the caller set), None when it was: a refused tracklet has no
    candidates and is not a "no match"."""
    indices: list
    obscode: str
    t_first: float
    t_last: float
    label: str
    candidates: list
    identification: str = None
    ambiguous: bool = False
    refused: str = None

    def table(self, limit=8):
        """The tracklet and its best ``limit`` candidates (None: all) as text."""
        n = len(self.indices)
        verdict = (f"not checked: {self.refused}" if self.refused else
                   self.identification or ('ambiguous' if self.ambiguous else 'no match'))
        head = (f"Tracklet {self.label or '?'}  {self.obscode}  {n} obs  "
                f"JD {self.t_first:.5f}-{self.t_last:.5f}  -> {verdict}")
        rows = [head]
        for c in self.candidates[:limit]:
            rows.append(f"  {'MATCH' if c['match'] else '     '} {c['designation']:<16} "
                        f"offset {c['mean_offset']:9.1f}\"  motion rms {c['motion_rms']:7.2f}\"  "
                        f"V {c['mag']:5.1f}  {c['kind']}")
        return "\n".join(rows)


# The arcsecond labels are built outside the f-string: a backslash inside an f-string's braces
# is a SyntaxError before Python 3.12.
_DRA, _DDEC, _SEP, _RATE = 'dRA"', 'dDec"', 'sep"', '"/min RA'
_HEADER = (f"{'designation':<18}{'RA':>12}{'Dec':>12}{_DRA:>9}{_DDEC:>9}{_SEP:>9}"
           f"{'mag':>6}{_RATE:>10}{'Dec':>7}{'r':>7}{'delta':>7}{'elong':>7}  kind")


def _row_text(o):
    ra_s, dec_s = sexagesimal(o["ra"], o["dec"])
    mag = f"{o['mag']:6.1f}" if o["mag"] is not None and math.isfinite(o["mag"]) else "     -"
    kind = o["kind"]
    if o.get("variants"):
        kind += f" ({o['in_field']}/{o['variants']} variants in field, spread {o['spread']:.0f}\")"
    elif o.get("U") is not None and math.isfinite(o["U"]):
        kind += f" U={o['U']:.0f}"
    return (f"{o['designation'][:17]:<18}{ra_s:>12}{dec_s:>12}{o['offset_ra']:9.1f}"
            f"{o['offset_dec']:9.1f}{o['separation']:9.1f}{mag}{o['rate_ra']:10.2f}"
            f"{o['rate_dec']:7.2f}{o['r']:7.3f}{o['delta']:7.3f}{o['elongation']:7.1f}  {kind}")


# --------------------------------------------------------------------------- shared machinery
def _obscodes(obscodes):
    if obscodes is not None:
        return obscodes
    from pyroorb import obs_sources
    return obs_sources.load_obscodes()


def _neocp_catalogue(times_utc, include, refresh=True):
    """The NEOCP orbits when asked for (``include``) and every time is within the page's
    horizon: a NEOCP orbit describes an object observed in the last few days, never an older
    observation. Without ``refresh`` only the cached copy is used (the web server's requests,
    which a background thread keeps current, never wait on the MPC's pages)."""
    from pyroorb.skycheck import neocp
    import time as _time
    now_jd = _time.time() / 86400.0 + 2440587.5
    if not include or any(now_jd - t > neocp.RELEVANT_DAYS for t in times_utc):
        return None
    try:
        return neocp.as_catalogue(neocp.fetch() if refresh else neocp.cached())
    except Exception:                                                  # noqa: BLE001
        return None


def _object_dict(cat, p, kind_label, centre=None):
    k = p.row
    o = {"designation": cat.designation(k), "name": cat.display_name(k), "kind": kind_label,
         "ra": p.ra_mid, "dec": p.dec_mid, "mag": None if not math.isfinite(p.mag) else p.mag,
         "rate_ra": p.rate_ra, "rate_dec": p.rate_dec, "r": p.r, "delta": p.delta,
         "phase": p.phase, "elongation": p.elong,
         "U": float(cat.cols["U"][k]) if math.isfinite(cat.cols["U"][k]) else None,
         "oppositions": (int(cat.cols["nopp"][k]) if math.isfinite(cat.cols["nopp"][k])
                         else None)}
    if centre is not None:
        dx, dy = _check.offsets_arcsec(centre[0], centre[1], p.ra_mid, p.dec_mid)
        o["offset_ra"], o["offset_dec"] = float(dx), float(dy)
        o["separation"] = float(math.hypot(dx, dy))
    return o


def _kind_label(cat, k):
    return "comet" if cat.kind[k] == _catalogue.KIND_COMET else "asteroid"


def _run(queries, cat, neo, comets):
    """Predictions per query from the main catalogue and (separately) the NEOCP orbits."""
    kinds_ok = None if comets else (np.asarray(cat.kind) != _catalogue.KIND_COMET)
    cand = _check._quick(cat, queries, kinds_ok=kinds_ok)
    preds = _check._precise(cat, queries, cand)
    neo_preds = [[] for _ in queries]
    if neo is not None:
        # an object one of whose variants is near: ALL its variants (the nominal, and the
        # spread reported, are over the whole set -- two-body, so cheap)
        tids = np.char.rstrip(np.asarray(neo.desig), b"\0")
        ncand = [np.flatnonzero(np.isin(tids, np.unique(tids[c]))) for c in
                 _check._quick(neo, queries)]
        neo_preds = _check._precise(neo, queries, ncand, two_body=True)
    return preds, neo_preds


def _neocp_objects(neo, preds, inside, centre):
    """One dict per NEOCP object from its variants' predictions: the NOMINAL's position (the
    first orbit the MPC lists), how many variants satisfy ``inside`` and how far they spread."""
    by_tid = {}
    for p in preds:
        by_tid.setdefault(neo.designation(p.row), []).append(p)
    out = []
    for tid, ps in by_tid.items():
        ok = [p for p in ps if inside(p)]
        if not ok:
            continue
        nominal = min(ps, key=lambda p: p.row)
        o = _object_dict(neo, nominal, "NEOCP", centre)
        o["name"] = tid
        o["variants"] = len(ps)
        o["in_field"] = len(ok)
        o["spread"] = float(max(_check.separation_arcsec(nominal.ra_mid, nominal.dec_mid,
                                                         p.ra_mid, p.dec_mid) for p in ps))
        out.append(o)
    return out


def _notes(cat, t_utc):
    notes = []
    ep = cat.meta.get("main_epoch")
    if ep is not None and abs(t_utc - ep) > 365.0:
        notes.append(f"the time is {abs(t_utc - ep):.0f} days from the catalogue's epoch: "
                     "positions are less certain and the check slower than within a year")
    t0, nd = cat.meta.get("near_t0"), cat.meta.get("near_days")
    if t0 is not None and not (t0 <= t_utc <= t0 + (nd or 0)):
        notes.append("outside the near-Earth ephemeris window: every near-Earth object is "
                     "integrated for this check")
    return notes


# --------------------------------------------------------------------------- public API
def check_field(ra, dec, time, *, radius_arcmin=DEFAULT_RADIUS_ARCMIN, obscode="500",
                mag_limit=None, comets=True, neocp=False, obscodes=None, refresh=True,
                max_days=None):
    """Every catalogued object within ``radius_arcmin`` of (ra, dec) at ``time``.

    Parameters
    ----------
    ra, dec : float or str
        Field centre: degrees, or sexagesimal strings (RA in hours: '12 34 56.7').
    time : float, str or datetime
        UTC: a Julian date, an ISO string or 'YYYY MM DD.ddddd'.
    radius_arcmin : float
        Search radius (default 15').
    obscode : str
        MPC observatory code the field is seen from (default '500', the geocentre).
    mag_limit : float, optional
        Faintest magnitude to report.
    comets : bool
        Include comets (default True).
    neocp : bool
        Also check the objects on the NEO Confirmation Page (each variant orbit the MPC lists),
        for a time within the last 30 days. Default False: the published catalogue only.
    obscodes : dict, optional
        An observatory table to use instead of the bundled MPC table.
    refresh : bool
        Bring the catalogue up to date first (downloads the MPC's files when a day old), and
        the NEOCP orbits when half an hour old. False: both as last fetched, no network.
    max_days : float, optional
        Refuse a time farther than this (days) from the catalogue's epoch: the cost grows
        steeply with that distance. None (default): no limit.

    Returns
    -------
    FieldCheck

    Raises
    ------
    TooFarFromEpoch
        When ``time`` is farther from the catalogue's epoch than ``max_days``.
    """
    t = julian_date(time)
    ra_d, dec_d = angle_deg(ra, hours=isinstance(ra, str)), angle_deg(dec)
    limit = max_days
    cat = _catalogue.load(refresh=refresh)
    why = _too_far(cat, [t], limit)
    if why:
        raise TooFarFromEpoch(why)
    qy = _check.Query(np.array([t]), np.array([ra_d]), np.array([dec_d]), str(obscode),
                      radius_arcmin / 60.0, mag_limit=mag_limit)
    _check.prepare([qy], _obscodes(obscodes))
    neo = _neocp_catalogue([t], neocp, refresh)
    preds, neo_preds = _run([qy], cat, neo, comets)
    rad = radius_arcmin * 60.0
    objects = []
    for p in preds[0]:
        o = _object_dict(cat, p, _kind_label(cat, p.row), (ra_d, dec_d))
        if o["separation"] <= rad and (mag_limit is None or o["mag"] is None
                                       or o["mag"] <= mag_limit):
            objects.append(o)
    if neo is not None:
        inside = lambda p: (_check.separation_arcsec(ra_d, dec_d, p.ra_mid, p.dec_mid) <= rad)
        objects += _neocp_objects(neo, neo_preds[0], inside, (ra_d, dec_d))
    objects.sort(key=lambda o: o["separation"])
    return FieldCheck(t, ra_d, dec_d, radius_arcmin, str(obscode), objects,
                      _notes(cat, t), _catalogue_info(cat, neo))


def _catalogue_info(cat, neo):
    m = cat.meta
    return {"orbits": len(cat), "epoch_jd": m.get("main_epoch"), "built": m.get("built"),
            "neocp_orbits": len(neo) if neo is not None else 0}


def tracklets(records, max_hours=TRACKLET_HOURS):
    """Group observation records into tracklets: ``[[index, ...], ...]``, each in time order.

    One tracklet = one station's observations of one object: grouped by (station, the
    submitter's tracklet id) when the record carries one (ADES trkSub), else by (station,
    designation), a group closing ``max_hours`` after its first observation."""
    order = sorted(range(len(records)), key=lambda k: records[k]["jd_utc"])
    open_, groups = {}, []
    for k in order:
        o = records[k]
        code = str(o.get("obscode") or "")
        if o.get("trksub"):
            key = (code, "trk", str(o["trksub"]))
            g = open_.get(key)
            if g is None:
                g = open_[key] = []
                groups.append(g)
            g.append(k)
            continue
        key = (code, "obj", str(o.get("designation") or o.get("packed") or ""))
        g = open_.get(key)
        if g is None or (o["jd_utc"] - records[g[0]]["jd_utc"]) * 24.0 > max_hours:
            g = open_[key] = []
            groups.append(g)
        g.append(k)
    return groups


def identify(observations, *, radius_arcmin=DEFAULT_RADIUS_ARCMIN, match_arcsec=MATCH_ARCSEC,
             motion_arcsec=MOTION_ARCSEC, comets=True, neocp=False, obscodes=None,
             refresh=True, max_days=None):
    """Which catalogued object each tracklet of ``observations`` is.

    The observations are grouped into tracklets (:func:`tracklets`) and each tracklet is
    checked ONCE: every catalogued object predicted within ``radius_arcmin`` of it is compared
    with ALL its positions. A candidate MATCHES when its mean offset (observed minus
    predicted) is within ``match_arcsec`` (for a comet at least COMET_MATCH_ARCSEC) and the
    positions follow its predicted motion to ``motion_arcsec`` rms once that offset is removed
    (plus FAST_FRACTION of the predicted track for a fast mover) -- a poorly known orbit is
    off by a nearly constant amount over a tracklet's few hours, a different object moves
    differently. Of several matches the nearest is the identification only when the next is
    clearly farther (:func:`_choose`).

    Parameters
    ----------
    observations : Observations, list of dict, str or path
        What :func:`pyroorb.read_observations` returns, its records, or a file / text to read.
    radius_arcmin, match_arcsec, motion_arcsec : float
        Search radius, and the two matching tolerances.
    comets, neocp, obscodes, refresh, max_days
        As for :func:`check_field`.

    Returns
    -------
    list of TrackletCheck
        One per tracklet, in time order; a tracklet farther from the catalogue's epoch than
        ``max_days`` is returned unchecked, with ``refused`` saying why.
    """
    records = _records(observations, obscodes)
    records = [o for o in records if o.get("jd_utc") is not None and o.get("ra_deg") is not None]
    limit = max_days
    every = tracklets(records)
    cat = _catalogue.load(refresh=refresh)
    refused = {}
    for n, g in enumerate(every):
        why = _too_far(cat, [records[k]["jd_utc"] for k in g], limit)
        if why:
            refused[n] = why
    groups = [g for n, g in enumerate(every) if n not in refused]
    queries = []
    for g in groups:
        obs = [records[k] for k in g]
        t = np.array([o["jd_utc"] for o in obs], float)
        ra = np.array([o["ra_deg"] for o in obs], float)
        dec = np.array([o["dec_deg"] for o in obs], float)
        c = _check.unit(ra, dec).sum(axis=0)
        cra, cdec = _check.radec(c)
        extent = float(np.max(_check.separation_arcsec(cra[0], cdec[0], ra, dec))) / 3600.0
        queries.append(_check.Query(t, ra, dec, str(obs[0]["obscode"]),
                                    radius_arcmin / 60.0 + extent,
                                    sat_pos=[o.get("sat_pos_au") for o in obs],
                                    site_itrs=[o.get("site_itrs_au") for o in obs]))
    out = [_refused_check(records, g, refused[n]) for n, g in enumerate(every)
           if n in refused]
    if not queries:
        return sorted(out, key=lambda tr: tr.t_first)
    _check.prepare(queries, _obscodes(obscodes))
    neo = _neocp_catalogue([q.t_utc[0] for q in queries], neocp, refresh)
    preds, neo_preds = _run(queries, cat, neo, comets)
    for g, qy, ps, nps in zip(groups, queries, preds, neo_preds):
        cands, row_of = [], {}
        for p in ps:
            c = _tracklet_candidate(cat, p, qy, _kind_label(cat, p.row), match_arcsec,
                                    motion_arcsec)
            if c["mean_offset"] <= radius_arcmin * 60.0:
                cands.append(c)
                row_of[id(c)] = p.row
        if neo is not None:
            by_tid = {}
            for p in nps:
                c = _tracklet_candidate(neo, p, qy, "NEOCP", match_arcsec, motion_arcsec)
                best = by_tid.get(c["designation"])
                if best is None or ((not c["match"], c["mean_offset"])
                                    < (not best["match"], best["mean_offset"])):
                    by_tid[c["designation"]] = c
            cands += [c for c in by_tid.values() if c["mean_offset"] <= radius_arcmin * 60.0]
        cands.sort(key=lambda c: (not c["match"], c["mean_offset"]))
        ident, ambiguous = _choose(_merge_twins(cat, [c for c in cands if c["match"]],
                                                row_of))
        obs = [records[k] for k in g]
        label = _label(obs[0])
        out.append(TrackletCheck(list(g), qy.obscode, float(qy.t_utc.min()),
                                 float(qy.t_utc.max()), str(label), cands, ident, ambiguous))
    return sorted(out, key=lambda tr: tr.t_first)


def _label(o):
    return str(o.get("trksub") or o.get("designation") or o.get("packed") or "")


def _refused_check(records, g, why):
    """The answer for a tracklet that is not checked: no candidates, ``refused`` = ``why``."""
    obs = [records[k] for k in g]
    t = [float(o["jd_utc"]) for o in obs]
    return TrackletCheck(list(g), str(obs[0]["obscode"]), min(t), max(t), _label(obs[0]), [],
                         refused=why)


CLEAR_FACTOR = 3.0                  # the best match must be this many times nearer ...
CLEAR_ARCSEC = 2.0                  # ... and this much nearer than the next one


def _choose(matches):
    """(identification, ambiguous) from the matching candidates, nearest first. One match is
    the identification; of several, the nearest when the next is CLEAR_FACTOR times and
    CLEAR_ARCSEC farther (a catalogued orbit predicts its object to well under an arcsecond
    to a few arcseconds, so a second object several times farther is a field neighbour);
    otherwise none, flagged ambiguous."""
    if not matches:
        return None, False
    if len(matches) == 1:
        return matches[0]["designation"], False
    a, b = matches[0]["mean_offset"], matches[1]["mean_offset"]
    if b >= CLEAR_FACTOR * a and b - a >= CLEAR_ARCSEC:
        return matches[0]["designation"], False
    return None, True


def _same_orbit(cat, a, b):
    """Rows ``a`` and ``b`` carry one orbit (osculating at nearby epochs): the same body."""
    c = cat.cols
    dnode = abs((float(c["node"][a]) - float(c["node"][b]) + 180.0) % 360.0 - 180.0)
    return (abs(float(c["q"][a]) / float(c["q"][b]) - 1.0) < 0.01
            and abs(float(c["e"][a]) - float(c["e"][b])) < 0.005
            and abs(float(c["i"][a]) - float(c["i"][b])) < 0.2 and dnode < 1.0)


def _merge_twins(cat, matches, row_of):
    """Drop a comet match that is a matching asteroid's own orbit: a dual-status object
    ((118401) = 176P/LINEAR) sits in both MPCORB and CometEls, and its two rows would
    otherwise make every tracklet of it ambiguous. The comet row stays among the
    candidates, marked ``same_as`` the asteroid."""
    asts = [c for c in matches if c["kind"] == "asteroid" and id(c) in row_of]
    out = []
    for c in matches:
        twin = None
        if c["kind"] == "comet" and id(c) in row_of:
            twin = next((a for a in asts if _same_orbit(cat, row_of[id(a)], row_of[id(c)])),
                        None)
        if twin is not None:
            c["same_as"] = twin["designation"]
        else:
            out.append(c)
    return out


def _records(observations, obscodes):
    if isinstance(observations, (list, tuple)):                   # Observations is a list
        return list(observations)
    import pyroorb
    return list(pyroorb.read_observations(observations, obscodes=obscodes))


def _tracklet_candidate(cat, p, qy, kind, match_arcsec, motion_arcsec):
    centre = _check.radec(qy.centre)
    o = _object_dict(cat, p, kind, (float(centre[0][0]), float(centre[1][0])))
    dx, dy = _check.offsets_arcsec(p.ra, p.dec, qy.ra, qy.dec)
    mx, my = float(np.mean(dx)), float(np.mean(dy))
    o["mean_offset"] = float(math.hypot(mx, my))
    o["offset_ra"], o["offset_dec"] = mx, my
    o["motion_rms"] = float(np.sqrt(np.mean((dx - mx) ** 2 + (dy - my) ** 2)))
    # A rate wrong by a fraction f leaves residuals drifting with rms f x (the predicted track's
    # rms about its mean), whatever the sampling -- so that rms, not the track length, scales it.
    rate = math.hypot(p.rate_ra, p.rate_dec)                      # arcsec/min
    tx, ty = _check.offsets_arcsec(p.ra_mid, p.dec_mid, p.ra, p.dec)
    track_rms = float(np.sqrt(np.mean((tx - tx.mean()) ** 2 + (ty - ty.mean()) ** 2)))
    o["motion_tolerance"] = motion_arcsec + (FAST_FRACTION * track_rms
                                             if rate >= FAST_RATE else 0.0)
    o["match_tolerance"] = (max(match_arcsec, COMET_MATCH_ARCSEC)
                            if cat.kind[p.row] == _catalogue.KIND_COMET else match_arcsec)
    o["match"] = bool(o["mean_offset"] <= o["match_tolerance"]
                      and o["motion_rms"] <= o["motion_tolerance"])
    return o


__all__ = ["check_field", "identify", "tracklets", "FieldCheck", "TrackletCheck",
           "julian_date", "angle_deg"]
