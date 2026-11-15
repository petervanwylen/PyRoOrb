"""The sky check's engine: which catalogued objects are where an observer looked.

A QUERY is one look at the sky from one observer: a field (a centre and a radius at one time)
or a tracklet (the positions one station measured of one object over a few hours). Every query
is answered in two passes.

1. **Quick pass** (:func:`_quick`). Every catalogue orbit is propagated two-body to the query
   time and kept when it falls inside the query's cone plus a MARGIN: the two-body error the
   orbit can have accumulated since its osculation epoch. The margin is a position error, in
   au, relative to the object's heliocentric distance r and growing with |t - epoch|
   (:func:`relative_margin`); divided by the object's distance from the observer it becomes an
   angle, so a nearby object is searched for over a correspondingly wider patch of sky. The
   table is the largest error measured, times 1.5, between two-body and ASSIST over 125,000
   asteroids and every comet at 15-day steps out to 450 days (with the near-Earth objects
   below excluded); no measured case exceeds it.

   Orbits that pass within 0.2 au of the Earth near the query time carry a stored daily
   ASSIST ephemeris (``catalogue.near_earth_states``): two-body is meaningless across an
   encounter, so the quick pass interpolates those states instead, with a fixed 1e-4 au
   margin. Outside the stored window they are always passed on to the precise pass.

   Queries are grouped in BIN_DAYS bins; the catalogue is propagated once per bin and carried
   to each query linearly (heliocentric), the acceleration term bounding that step's error.

2. **Precise pass** (:func:`_precise`). The survivors are integrated with ASSIST (DE440
   planets, Moon, GR) from their epoch -- or from the nearest stored state -- to every time the
   query needs, and placed as seen by the observer with light-time correction (astrometric
   J2000 positions, the frame of reported astrometry).

Times in are UTC Julian dates; the catalogue's TT and ASSIST's TDB differ from each other by
milliseconds, which no position here resolves.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from pyroorb.forward_model import MU_SUN
from pyroorb.skycheck import catalogue as _catalogue
from pyroorb.skycheck import conic

C_AU_DAY = 173.1446326846693        # speed of light, au/day
RAD_ARCSEC = 180.0 / math.pi * 3600.0
BIN_DAYS = 0.25                     # queries within a quarter day share one propagation
RATE_DT = 1.0 / 24.0                # the sky motion is the difference over one hour
NEAR_MARGIN_AU = 1e-4               # interpolated daily ASSIST states: cubic error bound
MARGIN_FLOOR_AU = 2e-6
_CHUNK = 1500                       # particles per ASSIST simulation (cost grows with count)
PRECISE_GROUP = 32                  # queries integrated together: bounds the (rows, times, 6)
                                    # state array to tens of MB however large the input

# Two-body vs ASSIST position error relative to r, max over the measured sample x 1.5, at
# |t - epoch| in days; linear between the knots (above the measured curve, which is convex),
# growing as dt^2 beyond the last one.
_MARGIN_DT = np.array([0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 165.0, 210.0, 300.0, 450.0])
_MARGIN_REL = np.array([0.0, 9.6e-5, 8.3e-4, 2.5e-3, 4.7e-3, 5.0e-3, 9.2e-3, 1.5e-2,
                        7.2e-2, 0.14])


def relative_margin(dt_days):
    """The quick pass's position margin, as a fraction of the heliocentric distance, for an
    orbit propagated two-body ``dt_days`` from its osculation epoch."""
    a = np.abs(np.asarray(dt_days, float))
    inside = np.interp(a, _MARGIN_DT, _MARGIN_REL)
    beyond = _MARGIN_REL[-1] * (a / _MARGIN_DT[-1]) ** 2
    return np.where(a <= _MARGIN_DT[-1], inside, beyond)


# --------------------------------------------------------------------------- queries
@dataclass
class Query:
    """One look at the sky. ``t_utc``/``ra``/``dec`` are the measured positions (a field has
    one: its centre); ``radius_deg`` is how far from ``centre`` an object may be."""
    t_utc: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    obscode: str
    radius_deg: float
    sat_pos: list = None            # per time: geocentric inertial position (au) or None
    site_itrs: list = None          # per time: Earth-fixed position (au) or None
    mag_limit: float = None
    # filled by prepare()
    t_tdb: np.ndarray = None
    R: np.ndarray = None            # observer, heliocentric equatorial (au), per time
    t_mid_tdb: float = None
    R_mid: np.ndarray = None
    R_rate: np.ndarray = None       # observer at t_mid + RATE_DT
    centre: np.ndarray = None


def unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def radec(v):
    v = np.atleast_2d(v)
    n = np.linalg.norm(v, axis=1)
    return (np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0,
            np.degrees(np.arcsin(np.clip(v[:, 2] / n, -1.0, 1.0))))


def prepare(queries, obscodes):
    """Observer positions and TDB times for every query, in one batched transform."""
    from astropy.time import Time
    from pyroorb import obs_sources
    jds, codes, sats, sites = [], [], [], []
    for qy in queries:
        mid = 0.5 * (qy.t_utc.min() + qy.t_utc.max())
        ts = list(qy.t_utc) + [mid, mid + RATE_DT]
        jds += ts
        codes += [qy.obscode] * len(ts)
        sp = list(qy.sat_pos) if qy.sat_pos is not None else [None] * len(qy.t_utc)
        si = list(qy.site_itrs) if qy.site_itrs is not None else [None] * len(qy.t_utc)
        # a satellite or roving observer's mid-point: its nearest measured position
        k = int(np.argmin(np.abs(qy.t_utc - mid)))
        sats += sp + [sp[k], sp[k]]
        sites += si + [si[k], si[k]]
    any_sat = any(s is not None for s in sats)
    any_site = any(s is not None for s in sites)
    R = obs_sources.observer_positions_batch(jds, codes, obscodes,
                                             sat_pos=sats if any_sat else None,
                                             site_itrs=sites if any_site else None)
    tdb = Time(np.asarray(jds, float), format="jd", scale="utc").tdb.jd
    i = 0
    for qy in queries:
        k = len(qy.t_utc)
        qy.R, qy.t_tdb = R[i:i + k], tdb[i:i + k]
        qy.R_mid, qy.t_mid_tdb = R[i + k], float(tdb[i + k])
        qy.R_rate = R[i + k + 1]
        c = unit(qy.ra, qy.dec).sum(axis=0)
        qy.centre = c / np.linalg.norm(c)
        i += k + 2
    return queries


# --------------------------------------------------------------------------- geometry
def apparent(x, v, R):
    """Light-time-corrected topocentric vectors (n, 3) of heliocentric states (x, v) at the
    observation time, seen from R: the object is placed where it was when the light left it,
    x(t - tau) = x - v tau - mu x / (2 r^3) tau^2 (tau of minutes to hours: the omitted
    terms are below 1e-12 au)."""
    x = np.atleast_2d(x)
    v = np.atleast_2d(v)
    r3 = np.linalg.norm(x, axis=1)[:, None] ** 3
    acc = -MU_SUN * x / r3
    d = x - R
    for _ in range(2):
        tau = (np.linalg.norm(d, axis=1) / C_AU_DAY)[:, None]
        d = x - v * tau + 0.5 * acc * tau * tau - R
    return d


def magnitude(H, G, kind, r, delta, sun_obs):
    """Predicted magnitude: asteroids V on the H,G system (Bowell et al. 1989; G 0.15 when
    unknown), comets total magnitude m = H + 5 log delta + 2.5 G log r (G 4 when unknown,
    the MPC's default slope)."""
    H, G, r, delta = (np.asarray(x, float) for x in (H, G, r, delta))
    cos_a = np.clip((r * r + delta * delta - sun_obs * sun_obs) / (2 * r * delta), -1.0, 1.0)
    alpha = np.arccos(cos_a)
    t = np.tan(alpha / 2.0)
    phi1 = np.exp(-3.33 * t ** 0.63)
    phi2 = np.exp(-1.87 * t ** 1.22)
    Ga = np.where(np.isfinite(G), G, 0.15)
    ast = H + 5 * np.log10(r * delta) - 2.5 * np.log10(np.maximum((1 - Ga) * phi1 + Ga * phi2,
                                                                  1e-30))
    Gc = np.where(np.isfinite(G), G, 4.0)
    com = H + 5 * np.log10(delta) + 2.5 * Gc * np.log10(r)
    return np.where(np.asarray(kind) == _catalogue.KIND_COMET, com, ast), np.degrees(alpha)


# --------------------------------------------------------------------------- quick pass
def _hermite(states, s):
    """Cubic Hermite position and velocity between two daily states at fraction s (0..1)."""
    p0, v0, p1, v1 = states[:, 0, :3], states[:, 0, 3:], states[:, 1, :3], states[:, 1, 3:]
    h00, h10, h01, h11 = (2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2)
    pos = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    d00, d10, d01, d11 = (6*s**2 - 6*s, 3*s**2 - 4*s + 1, -6*s**2 + 6*s, 3*s**2 - 2*s)
    vel = d00 * p0 + d10 * v0 + d01 * p1 + d11 * v1
    return pos, vel


def near_window(cat, t):
    """Index into the stored daily states for time t (TT/TDB), or None outside the window."""
    t0 = cat.meta.get("near_t0")
    nd = cat.near_states.shape[1] if cat.near_states.ndim == 3 else 0
    if t0 is None or not len(cat.near_index) or nd < 2:
        return None
    f = t - t0
    if f < 0 or f > nd - 1:
        return None
    return f


def catalogue_at(cat, t, rows=None):
    """Two-body (or stored near-Earth) heliocentric states of ``rows`` (all by default) at
    time t, with each row's position margin (au) for the quick pass."""
    sel = slice(None) if rows is None else rows
    q, e, T = (np.asarray(cat.cols[c][sel]) for c in ("q", "e", "T"))
    P, Q = np.asarray(cat.P[sel]), np.asarray(cat.Q[sel])
    pos, vel = conic.state_at(q, e, T, P, Q, t)
    r = np.linalg.norm(pos, axis=1)
    margin = relative_margin(t - np.asarray(cat.cols["epoch"][sel])) * r + MARGIN_FLOOR_AU
    if len(cat.near_index):
        all_rows = np.arange(len(cat)) if rows is None else np.asarray(rows)
        where = np.searchsorted(cat.near_index, all_rows)
        where = np.clip(where, 0, len(cat.near_index) - 1)
        is_near = cat.near_index[where] == all_rows
        if is_near.any():
            f = near_window(cat, t)
            k = np.flatnonzero(is_near)
            if f is None:
                margin[k] = np.inf                                   # always looked at closely
            else:
                d = min(int(f), cat.near_states.shape[1] - 2)
                st = np.asarray(cat.near_states[where[k], d:d + 2, :])
                p, v = _hermite(st, f - d)
                pos[k], vel[k] = p, v
                margin[k] = NEAR_MARGIN_AU
    return pos, vel, margin


INDEX_NSIDE = 64                    # HEALPix cells of ~0.9 deg for the quick pass's index
WIDE_DEG = 1.0                      # rows that may sit farther than this from their indexed
                                    # direction are tested against every query instead


def _bin_index(cat, t_ref, queries, idx):
    """Everything one time bin's queries share: the catalogue's states at t_ref, each row's
    margin, its geocentric direction and how far (``reach``, rad) any query of the bin can see
    it from that direction -- the margin, its motion relative to the Earth over the bin, the
    observers' offsets from the geocentre -- and a HEALPix index of the rows whose reach is
    under WIDE_DEG."""
    from pyroorb import healpix, obs_sources
    pos, vel, margin = catalogue_at(cat, t_ref)
    r = np.linalg.norm(pos, axis=1)
    ep = obs_sources._geocentric_positions(np.array([t_ref - 0.01, t_ref, t_ref + 0.01]))
    earth, earth_v = ep[1], (ep[2] - ep[0]) / 0.02
    dt_max = max(abs(queries[i].t_mid_tdb - t_ref) for i in idx)
    off = max(float(np.linalg.norm(queries[i].R_mid - earth)) for i in idx)
    geo = pos - earth
    dist = np.linalg.norm(geo, axis=1)
    geo = pos - vel * (dist / C_AU_DAY)[:, None] - earth
    dist = np.linalg.norm(geo, axis=1)
    slack = (margin + np.linalg.norm(vel - earth_v, axis=1) * dt_max + off
             + MU_SUN * dt_max * dt_max / (2.0 * r * r))
    with np.errstate(invalid="ignore", divide="ignore"):
        reach = np.where(np.isfinite(slack), slack / np.maximum(dist - slack, 1e-12), np.inf)
    narrow_mask = reach < math.radians(WIDE_DEG)
    wide = np.flatnonzero(~narrow_mask)
    narrow = np.flatnonzero(narrow_mask)
    ra, dec = radec(geo[narrow])
    pix = healpix.ang2pix(INDEX_NSIDE, ra, dec, nest=True)
    order = np.argsort(pix, kind="stable")
    return {"pos": pos, "vel": vel, "margin": margin, "r": r, "wide": wide,
            "rows": narrow[order], "pix": pix[order]}


def _quick(cat, queries, kinds_ok=None):
    """Candidate rows per query: inside its cone plus each row's margin, and possibly bright
    enough when the query has a magnitude limit (V >= H + 5 log(r delta) whatever the phase,
    so a row is dropped only when that bound, with delta shrunk by the margin, is fainter).

    The rows a query tests are those its bin's sky index puts within the query radius plus
    WIDE_DEG, and every wide row; the test itself is exact for each."""
    from pyroorb import healpix
    out = [None] * len(queries)
    bins = {}
    for i, qy in enumerate(queries):
        bins.setdefault(round(qy.t_mid_tdb / BIN_DAYS) * BIN_DAYS, []).append(i)
    H_all = np.asarray(cat.cols["H"])
    kind_all = np.asarray(cat.kind)
    for t_ref, idx in sorted(bins.items()):
        ix = _bin_index(cat, t_ref, queries, idx)
        for i in idx:
            qy = queries[i]
            disc = healpix.query_disc_nest(INDEX_NSIDE, qy.centre,
                                           math.radians(qy.radius_deg + WIDE_DEG))
            lo = np.searchsorted(ix["pix"], disc, side="left")
            hi = np.searchsorted(ix["pix"], disc, side="right")
            sub = [ix["rows"][a:b] for a, b in zip(lo, hi) if b > a]
            sub = np.concatenate(sub + [ix["wide"]]) if sub else ix["wide"]
            if not len(sub):
                out[i] = sub
                continue
            pos, vel, r = ix["pos"][sub], ix["vel"][sub], ix["r"][sub]
            dt = qy.t_mid_tdb - t_ref
            x = pos + vel * dt
            m = ix["margin"][sub] + MU_SUN * dt * dt / (2.0 * r * r)
            d = x - qy.R_mid
            dist = np.linalg.norm(d, axis=1)
            d = x - vel * (dist / C_AU_DAY)[:, None] - qy.R_mid
            dist = np.linalg.norm(d, axis=1)
            cosang = (d @ qy.centre) / dist
            ang = np.arccos(np.clip(cosang, -1.0, 1.0))
            with np.errstate(invalid="ignore"):
                reach = math.radians(qy.radius_deg) + np.minimum(
                    m / np.maximum(dist - m, 1e-9), math.pi)
            keep = (ang <= reach) | ~np.isfinite(m)
            if kinds_ok is not None:
                keep &= kinds_ok[sub]
            if qy.mag_limit is not None:
                H, kind = H_all[sub], kind_all[sub]
                near_d = np.maximum(dist - m, 1e-6)
                with np.errstate(invalid="ignore", divide="ignore"):
                    bright = np.where(kind == _catalogue.KIND_COMET,
                                      H + 5 * np.log10(near_d) - 10.0,   # a comet's slope
                                      H + 5 * np.log10(np.maximum(r - m, 1e-3) * near_d))
                keep &= ~(bright > qy.mag_limit) | ~np.isfinite(H)
            out[i] = np.unique(sub[keep])
    return out


# --------------------------------------------------------------------------- precise pass
def _start_states(cat, rows, t_ref):
    """Integration start for each row: its stored near-Earth state on the day nearest t_ref
    when inside the window, else its osculating state at its epoch. Returns (t0, states)."""
    t0 = np.asarray(cat.cols["epoch"][rows], float).copy()
    q, e, T = (np.asarray(cat.cols[c][rows]) for c in ("q", "e", "T"))
    P, Q = np.asarray(cat.P[rows]), np.asarray(cat.Q[rows])
    states = np.empty((len(rows), 6))
    for epv in np.unique(t0):
        k = np.flatnonzero(t0 == epv)
        p, v = conic.state_at(q[k], e[k], T[k], P[k], Q[k], epv)
        states[k, :3], states[k, 3:] = p, v
    f = near_window(cat, t_ref)
    if f is not None and len(cat.near_index):
        where = np.clip(np.searchsorted(cat.near_index, rows), 0, len(cat.near_index) - 1)
        is_near = cat.near_index[where] == rows
        if is_near.any():
            day = int(round(f))
            k = np.flatnonzero(is_near)
            states[k] = np.asarray(cat.near_states[where[k], day, :])
            t0[k] = cat.meta["near_t0"] + day
    return t0, states


def integrate(cat, rows, times, t_ref, two_body=False):
    """Heliocentric states (len(rows), len(times), 6) of catalogue ``rows`` at ``times``
    (sorted TDB), by ASSIST -- or two-body when ``two_body`` (orbits that are themselves
    two-body snapshots of the present, such as the NEOCP's)."""
    rows = np.asarray(rows)
    times = np.asarray(times, float)
    out = np.empty((len(rows), len(times), 6))
    if two_body:
        q, e, T = (np.asarray(cat.cols[c][rows]) for c in ("q", "e", "T"))
        P, Q = np.asarray(cat.P[rows]), np.asarray(cat.Q[rows])
        for j, t in enumerate(times):
            p, v = conic.state_at(q, e, T, P, Q, t)
            out[:, j, :3], out[:, j, 3:] = p, v
        return out
    from pyroorb import force_assist
    t0, states = _start_states(cat, rows, t_ref)
    for tv in np.unique(t0):
        k = np.flatnonzero(t0 == tv)
        for s in range(0, len(k), _CHUNK):
            kk = k[s:s + _CHUNK]
            st = force_assist.helio_states_multi(float(tv), list(states[kk]), times,
                                                 epsilon=force_assist.IAS15_EPSILON,
                                                 asteroids=False)
            out[kk] = np.transpose(st, (1, 0, 2))
    return out


@dataclass
class Prediction:
    """Where catalogue row ``row`` is for one query: per measured time (``ra``, ``dec``), at
    the query's mid-time (``ra_mid``, ``dec_mid``), its motion, distances and magnitude."""
    row: int
    ra: np.ndarray
    dec: np.ndarray
    ra_mid: float
    dec_mid: float
    rate_ra: float                  # arcsec/min, along RA (times cos dec)
    rate_dec: float                 # arcsec/min
    r: float
    delta: float
    phase: float
    elong: float
    mag: float
    extra: dict = field(default_factory=dict)


def _precise(cat, queries, cand, two_body=False):
    """Prediction lists per query, for the candidate rows ``cand[i]`` of each query."""
    preds = [[] for _ in queries]
    bins = {}
    for i, qy in enumerate(queries):
        if cand[i] is not None and len(cand[i]):
            bins.setdefault(round(qy.t_mid_tdb / BIN_DAYS) * BIN_DAYS, []).append(i)
    groups = []
    for t_ref, idx in sorted(bins.items()):
        # neighbours on the sky share candidates; a group's states are (rows, times, 6), so
        # it is kept to PRECISE_GROUP queries (a busy station-night is thousands)
        idx = sorted(idx, key=lambda i: tuple(np.round(queries[i].centre, 1)))
        groups += [(t_ref, idx[s:s + PRECISE_GROUP]) for s in range(0, len(idx), PRECISE_GROUP)]
    for t_ref, idx in groups:
        rows = np.unique(np.concatenate([cand[i] for i in idx]))
        times = np.unique(np.concatenate(
            [np.concatenate([queries[i].t_tdb, [queries[i].t_mid_tdb,
                                                queries[i].t_mid_tdb + RATE_DT]])
             for i in idx]))
        states = integrate(cat, rows, times, t_ref, two_body=two_body)
        pos_of = {int(rw): k for k, rw in enumerate(rows)}
        for i in idx:
            qy = queries[i]
            k = np.array([pos_of[int(rw)] for rw in cand[i]])
            tj = np.searchsorted(times, qy.t_tdb)
            jm = np.searchsorted(times, qy.t_mid_tdb)
            jr = np.searchsorted(times, qy.t_mid_tdb + RATE_DT)
            ra_obs = np.empty((len(k), len(tj)))
            dec_obs = np.empty((len(k), len(tj)))
            for c, j in enumerate(tj):
                d = apparent(states[k, j, :3], states[k, j, 3:], qy.R[c])
                ra_obs[:, c], dec_obs[:, c] = radec(d)
            dm = apparent(states[k, jm, :3], states[k, jm, 3:], qy.R_mid)
            dr = apparent(states[k, jr, :3], states[k, jr, 3:], qy.R_rate)
            ram, decm = radec(dm)
            rar, decr = radec(dr)
            per_min = RATE_DT * 1440.0
            rate_ra = (((rar - ram + 180.0) % 360.0) - 180.0) * np.cos(np.radians(decm)) \
                * 3600.0 / per_min
            rate_dec = (decr - decm) * 3600.0 / per_min
            x = states[k, jm, :3]
            r = np.linalg.norm(x, axis=1)
            delta = np.linalg.norm(dm, axis=1)
            sun_obs = np.linalg.norm(qy.R_mid)
            rows_k = np.asarray(cand[i])
            mag, phase = magnitude(np.asarray(cat.cols["H"][rows_k]),
                                   np.asarray(cat.cols["G"][rows_k]),
                                   np.asarray(cat.kind[rows_k]), r, delta, sun_obs)
            cos_el = (dm @ (-qy.R_mid)) / (delta * sun_obs)
            elong = np.degrees(np.arccos(np.clip(cos_el, -1.0, 1.0)))
            for c, rw in enumerate(rows_k):
                preds[i].append(Prediction(int(rw), ra_obs[c], dec_obs[c], float(ram[c]),
                                           float(decm[c]), float(rate_ra[c]),
                                           float(rate_dec[c]), float(r[c]), float(delta[c]),
                                           float(phase[c]), float(elong[c]), float(mag[c])))
    return preds


def offsets_arcsec(ra_pred, dec_pred, ra_obs, dec_obs):
    """(dRA cos dec, dDec) in arcsec of observed minus predicted, on the tangent plane at the
    prediction."""
    a0, d0 = np.radians(ra_pred), np.radians(dec_pred)
    a, d = np.radians(ra_obs), np.radians(dec_obs)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return xi * RAD_ARCSEC, eta * RAD_ARCSEC


def separation_arcsec(ra1, dec1, ra2, dec2):
    u1, u2 = unit(ra1, dec1), unit(ra2, dec2)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u1, u2), axis=-1),
                                 (u1 * u2).sum(-1))) * 3600.0
