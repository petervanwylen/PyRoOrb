"""ITF search from ONE tracklet (or an arc too short to fix a distance): statistical ranging.

A tracklet measures an ATTRIBUTABLE -- two angles and two angular rates at its mean time.
Every heliocentric state through it is fixed by two more numbers, the topocentric distance
``rho`` and radial velocity ``rhodot``.  An ITF tracklet measures an attributable too, so the
seed plus one candidate is 8 numbers for the 6 of a state: the best two-body orbit through
both leaves a chi2 with 2 degrees of freedom.  That is the linkage test, and the search is
built around it.  A seed of two or three nights (an arc too short to range) gives its
attributable from ONE night, its busiest station-night (:func:`seed_night`): a straight line
has no curvature to carry a position across a day; the other nights are judged by the
engine fit of every proposal.

1. **Pull the window.**  Every index tracklet within ``window_days`` of the seed, whole sky,
   in one query (a 120-day window is ~100k rows in 2-3 s).
2. **Family.**  A structured ``n_rho`` x ``n_v`` grid of BOUND two-body orbits through the
   attributable: ``rho`` geometric, ``rhodot`` uniform across each distance's bound
   interval.  No population is assumed.
3. **Screen, one frame per day.**  The family is carried to the frame (one vectorised
   conic solve).  Each candidate takes its ``knn`` nearest members in POSITION AND RATE --
   at one sky position the family folds many members with very different rates on top of
   each other, so the nearest by position alone are often the wrong ones -- and each pair
   is scored against the candidate's attributable with the seed's own noise propagated as
   a covariance (four 1-sigma perturbed families) and the best sub-grid offset from the
   member's grid Jacobian: a genuine 2-dof chi2 instead of one inflated by the grid.
4. **Refine** the best ``refine_top`` candidates over all six parameters (the seed
   attributable under its covariance, log rho, rhodot) with a batched Levenberg-Marquardt
   in a UNIVERSAL conic: a short pair's best orbit often sits at e ~ 1-1.6, and an elliptic
   wall there lost true links.
5. **Rank** by chi2 + ``w_vague`` x vagueness + ``w_prior`` x (capped) population prior +
   a magnitude penalty.  Vagueness is 2 ln(sigma_rate_e sigma_rate_n): a two-observation
   tracklet's rate is so loose that any of them fits a weak seed at chi2 ~ 0, and the
   likelihood's normalisation over the two rate dimensions is what tells it from a sharp
   match.  The prior (rho^4 x ``popprior.density``) chooses where to look, never what is
   accepted; it is capped because a short pair's best orbit is often just unbound, where the
   population model has no density at all.
6. **Triplets.**  Pairs of candidates on different nights whose solutions are compatible
   are refitted with the seed as ONE orbit (12 numbers, 6 dof).
7. **Engine fits** of the top proposals (``fitter.fit_astrometry``: ASSIST, every
   observation) and acceptance: every observation kept, chi2/dof <= ``accept_chi2_dof``,
   max |resid/sigma| <= ``accept_max_z``.  A triplet is preferred only when its own 6-dof
   chi2 is small (``trip_chi2_max``); otherwise the first passing PAIR in ranking order --
   never the pair with the lowest engine chi2/dof, which is not evidence: seed + one
   tracklet is 6-8 observations against 6 parameters, and wrong pairs fit as well as true
   ones there.

Measured on 480 harness objects and 47 objects whose real ITF tracklets were later
identified: ~97% / ~89% correct link, ~1% / ~6% wrong.  A link from this search is
therefore a HYPOTHESIS: it seeds the regular search (:func:`sr_then_itf`), whose own
gates -- and, for anything sent, a chimera guard -- decide.  With nothing of the object in the ITF, the best wrong candidate passes acceptance
for over half of all seeds.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from pyroorb import photometry
from pyroorb.orbit_common import count_nights, night_index
from pyroorb.skycheck import conic

MU = 2.959122082855911e-4
C_AU_DAY = 173.1446326846693
DEG = math.pi / 180.0
ARCSEC = DEG / 3600.0
UNIX_JD0 = 2440587.5
#: TDB-UTC, constant to ~5 s since 1998: far below what two-body prediction can tell
TT_UTC_DAYS = 69.184 / 86400.0


@dataclass
class SrConfig:
    window_days: float = 60.0      # +-60 d: +-90 d added no correct real link, doubled wrong ones
    min_dt_days: float = 0.5       # a same-night tracklet gives no range leverage
    n_rho: int = 200               # 140x140 kept every accepted link but ranked truths lower
    n_v: int = 200
    rho_lo: float = 0.002          # au
    rho_hi: float = 6.0
    knn: int = 8                   # neighbours per candidate (position + rate)
    knn_cut_deg: float = 4.0
    rate_tau_days: float = 2.0     # a rate mismatch x tau = its weight as a position mismatch
    model_pos_floor_arcsec: float = 5.0     # the screen's linearisation error floor
    model_rate_floor_degday: float = 0.01
    sub_grid_clip: float = 2.0     # largest sub-grid step (index units) the screen takes
    pair_chi2_max: float = 50.0    # screen cut before refinement
    refine_top: int = 400
    seed_sigma_floor: float = 0.3  # arcsec
    cand_pos_floor: float = 0.5    # arcsec
    cand_rate_floor: float = 0.002  # deg/day
    mag_tol: float = 1.5           # |dmag| past this costs rank
    vague_ref_degday: float = 0.004
    w_vague: float = 0.5
    w_prior: float = 0.5
    prior_cap: float = 8.0
    trip_top: int = 60             # best pair solutions combined into triplets
    trip_md_max: float = 50.0      # compatibility (Mahalanobis in log rho, rhodot)
    # a triplet is proposed AND accepted only at or under this 6-dof chi2: one above it can
    # never be picked, so proposing it only takes an engine-fit slot from a pair
    trip_chi2_max: float = 9.0
    pair_propose_chi2: float = 12.0
    fits: int = 6                  # engine fits of the top proposals
    fit_max_sec: float = 20.0
    accept_chi2_dof: float = 2.0
    accept_max_z: float = 4.0
    extra: dict = field(default_factory=dict)


@dataclass
class SrLink:
    """An accepted link: the candidate tracklets and the evidence for them."""
    trkids: list
    kind: str                      # "pair" | "triplet"
    lines: dict                    # trkid -> [obs80 line]
    sr_chi2: float
    dts: list                      # days from the seed, per tracklet
    fit: dict                      # engine-fit summary (rms, chi2_dof, max_z, used, count)
    # {obs_unique_id: trksub} from the database column, the name a document gives each
    # tracklet (empty for a source that does not know it)
    trksubs: dict = field(default_factory=dict)


@dataclass
class SrResult:
    link: SrLink | None = None
    reason: str = ""
    n_window: int = 0
    n_refined: int = 0
    proposals: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- geometry


def topo_attr(D, Ddot):
    """Angles (rad) and rates (rad/day: dRA cos dec, dDec) of topocentric vectors ``D``
    (..., 3) moving at ``Ddot``, and the distance."""
    x, y, z = D[..., 0], D[..., 1], D[..., 2]
    vx, vy, vz = Ddot[..., 0], Ddot[..., 1], Ddot[..., 2]
    rxy2 = x * x + y * y
    rxy = np.sqrt(rxy2)
    r2 = rxy2 + z * z
    ra = np.arctan2(y, x) % (2 * math.pi)
    dec = np.arctan2(z, rxy)
    ra_dot = (x * vy - y * vx) / rxy2
    dec_dot = (vz * rxy2 - z * (x * vx + y * vy)) / (r2 * rxy)
    return ra, dec, ra_dot * np.cos(dec), dec_dot, np.sqrt(r2)


def tangent_offsets(ra, dec, ra0, dec0):
    """Small sky offsets (rad): (dRA cos dec0, dDec), RA wrapped."""
    dra = (ra - ra0 + math.pi) % (2 * math.pi) - math.pi
    return dra * np.cos(dec0), dec - dec0


def unit_and_rate(x):
    """Unit vector and its time derivative from attributables ``x`` (..., 4)."""
    a, d, ad, dd = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    ca, sa, cdd, sd = np.cos(a), np.sin(a), np.cos(d), np.sin(d)
    L = np.stack([cdd * ca, cdd * sa, sd], -1)
    La = np.stack([-cdd * sa, cdd * ca, np.zeros_like(a)], -1)
    Ld = np.stack([-sd * ca, -sd * sa, cdd], -1)
    return L, ad[..., None] * La + dd[..., None] * Ld


def states(x, rho, rd, R0, V0):
    """Heliocentric states (at emission, tdb0 - rho/c) through attributable(s) ``x``."""
    L, Ldot = unit_and_rate(np.asarray(x))
    rho = np.asarray(rho, float)[..., None]
    rd = np.asarray(rd, float)[..., None]
    return R0 + rho * L, V0 + rho * Ldot + rd * L


def universal_elements(pos, vel, t):
    """Perihelion-form ``(q, e, T, P, Q)`` for elliptic AND hyperbolic states (the conic
    propagator is universal-variable); exactly parabolic rows come back NaN."""
    pos = np.atleast_2d(np.asarray(pos, float))
    vel = np.atleast_2d(np.asarray(vel, float))
    r = np.linalg.norm(pos, axis=1)
    h = np.cross(pos, vel)
    hn = np.linalg.norm(h, axis=1)
    ev = np.cross(vel, h) / MU - pos / r[:, None]
    e = np.linalg.norm(ev, axis=1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        P = ev / e[:, None]
        Q = np.cross(h / hn[:, None], P)
        q = hn * hn / (MU * (1.0 + e))
        nu = np.arctan2((Q * pos).sum(1) / r, (P * pos).sum(1) / r)
        a = q / (1.0 - e)
        ell = e < 1.0
        E = 2.0 * np.arctan(np.sqrt(np.abs((1.0 - e) / (1.0 + e))) * np.tan(nu / 2.0))
        F = 2.0 * np.arctanh(np.clip(np.sqrt(np.abs((e - 1.0) / (e + 1.0))) * np.tan(nu / 2.0),
                                     -1 + 1e-15, 1 - 1e-15))
        M = np.where(ell, E - e * np.sin(E), e * np.sinh(F) - F)
        T = np.asarray(t, float) - M / np.sqrt(MU / np.abs(a) ** 3)
    bad = ~np.isfinite(T) | (np.abs(e - 1.0) < 1e-9) | (e < 1e-10)
    q, e, T = (np.where(bad, np.nan, v) for v in (q, e, T))
    P = np.where(bad[:, None], np.nan, P)
    Q = np.where(bad[:, None], np.nan, Q)
    return q, e, T, P, Q


def _safe_elements(q, e, T, P, Q):
    ok = np.isfinite(q)
    q, e, T = np.where(ok, q, 1.0), np.where(ok, e, 0.5), np.where(ok, T, 0.0)
    P, Q = np.nan_to_num(P), np.nan_to_num(Q)
    P[~ok] = [1, 0, 0]
    Q[~ok] = [0, 1, 0]
    return (q, e, T, P, Q), ok


# ---------------------------------------------------------------------- observers


def fast_observers(jd, codes, obscodes):
    """Heliocentric observer positions AND velocities for many (UTC jd, code): the
    geocentre interpolated on a 0.01-day grid (a h^2/8 ~ 4e-9 au) plus the station rotated by
    the Earth rotation angle under one celestial-to-intermediate matrix (precession-
    nutation moves ~12" over 90 days: nothing on a one-Earth-radius offset).  The exact
    transform per row costs minutes on a 100k-row window; this agrees to 5e-9 au."""
    import erfa

    from pyroorb.obs_sources import A_EARTH_AU, _geocentric_positions
    jd = np.asarray(jd, float)
    h = 0.01
    grid = np.arange(float(jd.min()) - 2 * h, float(jd.max()) + 3 * h, h)
    Eg = np.asarray(_geocentric_positions(list(grid)))
    E = np.stack([np.interp(jd, grid, Eg[:, k]) for k in range(3)], 1)
    Edg = np.gradient(Eg, h, axis=0)
    Edot = np.stack([np.interp(jd, grid, Edg[:, k]) for k in range(3)], 1)
    lam = np.radians([obscodes.get(c, (0, 0, 0))[0] for c in codes])
    rcp = np.array([obscodes.get(c, (0, 0, 0))[1] for c in codes])
    rsp = np.array([obscodes.get(c, (0, 0, 0))[2] for c in codes])
    C2I = erfa.c2i06a(float(np.median(jd)) + TT_UTC_DAYS, 0.0)
    omega = 2 * math.pi * 1.00273781191135448
    ang = 2 * math.pi * ((0.7790572732640 + 1.00273781191135448 * (jd - 2451545.0)) % 1.0) + lam
    cirs = A_EARTH_AU * np.stack([rcp * np.cos(ang), rcp * np.sin(ang), rsp], 1)
    cirs_dot = A_EARTH_AU * omega * np.stack([-rcp * np.sin(ang), rcp * np.cos(ang),
                                              np.zeros_like(ang)], 1)
    return E + cirs @ C2I, Edot + cirs_dot @ C2I


def _ground(code, obscodes):
    rec = obscodes.get(code)
    return rec is not None and (code == "500" or any(float(v) != 0.0 for v in rec[:3]))


# ---------------------------------------------------------------------- the seed


def seed_night(obs, obscodes=None):
    """The seed observations the attributable is built from: the busiest station-night of
    the seed (ties: the longer night, then the later), over ground stations only when
    ``obscodes`` is given.  A straight line through two nights a day apart sits in the
    daytime gap and misses both: on a synthetic F51 object it missed by 2" at 0.6 au and
    40" at 0.03 au while claiming a 0.2"/day rate sigma, and the true tracklet then screened
    at chi2 1000-12000 against a cut of 50.  The other nights stay in the seed and are
    judged by every engine fit of a proposal.  A station-night is a station's observations
    in one night (``orbit_common.night_index``).  Empty when no observation qualifies."""
    by_stn = {}
    for o in obs:
        if obscodes is None or _ground(o["obscode"], obscodes):
            by_stn.setdefault(o["obscode"], []).append(o)
    nights = []
    for mine in by_stn.values():
        mine.sort(key=lambda o: o["jd_utc"])
        grouped = {}
        for o, k in zip(mine, night_index([o["jd_utc"] for o in mine])):
            grouped.setdefault(k, []).append(o)
        nights.extend(grouped.values())
    if not nights:
        return []
    return max(nights, key=lambda n: (len(n), round(n[-1]["jd_utc"] - n[0]["jd_utc"], 6),
                                      n[0]["jd_utc"]))


def seed_night_problem(obs, obscodes):
    """Why the seed has no night to range from, or None: the attributable is a straight
    line fitted per axis, which needs two observations at different times from one ground
    station on one night.  A space-based observer's parallax over a multi-hour tracklet
    (WISE: ~90" at 0.1 au) needs a per-distance correction the line does not make, so its
    nights never count; its observations still join every engine fit."""
    night = seed_night(obs, obscodes)
    if len({round(o["jd_utc"], 8) for o in night}) < 2:
        return ("no ground station observed the seed twice in one night: a single position "
                "fixes no motion to range from, and space-based seeds are not supported")
    return None


def seed_attributable(obs, obscodes, cfg):
    """The attributable of the seed's busiest ground station-night (:func:`seed_night`);
    the caller checks :func:`seed_night_problem` first."""
    return line_attributable(seed_night(obs, obscodes), obscodes, cfg)


def line_attributable(obs, obscodes, cfg):
    """Weighted linear attributable of ``obs`` (the most-observed station's, if several) at
    the weighted mean time: x = (ra, dec, ra_dot, dec_dot) (rad, rad/day), its covariance,
    the station's heliocentric state, and the mean V.  The line has no curvature and its
    covariance does not grow with the misfit, so ``obs`` must be one night."""
    stn = Counter(o["obscode"] for o in obs).most_common(1)[0][0]
    obs = sorted((o for o in obs if o["obscode"] == stn), key=lambda o: o["jd_utc"])
    jd = np.array([o["jd_utc"] for o in obs])
    sr = np.array([max(o.get("sigma_ra") or 1.0, cfg.seed_sigma_floor) for o in obs]) * ARCSEC
    sd = np.array([max(o.get("sigma_dec") or 1.0, cfg.seed_sigma_floor) for o in obs]) * ARCSEC
    w_r, w_d = 1 / sr ** 2, 1 / sd ** 2
    t0 = float(np.sum(jd * (w_r + w_d)) / np.sum(w_r + w_d))
    ra = np.unwrap(np.radians([o["ra_deg"] for o in obs]))
    dec = np.radians([o["dec_deg"] for o in obs])
    cdec = math.cos(float(np.mean(dec)))
    A = np.vstack([np.ones_like(jd), jd - t0]).T

    def wls(y, w):
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        return cov @ (A.T @ (w * y)), cov
    pa, ca = wls(ra, w_r * cdec ** 2)
    pd, cd = wls(dec, w_d)
    cov = np.zeros((4, 4))
    cov[np.ix_([0, 2], [0, 2])] = ca
    cov[np.ix_([1, 3], [1, 3])] = cd
    R, V = fast_observers(np.array([t0]), [stn], obscodes)
    mags = [o["mag"] + photometry.band_shift(o.get("band") or "")
            for o in obs if o.get("mag") is not None]
    return {"t0": t0, "tdb0": t0 + TT_UTC_DAYS, "x": np.array([pa[0], pd[0], pa[1], pd[1]]),
            "cov": cov, "station": stn, "R0": R[0], "V0": V[0], "n": len(obs),
            "V": float(np.mean(mags)) if mags else None}


def family(att, cfg):
    """The structured grid of bound members: flat arrays + grid shape + validity mask."""
    L, Ldot = unit_and_rate(att["x"])
    rho = np.geomspace(cfg.rho_lo, cfg.rho_hi, cfg.n_rho)
    r = att["R0"][None, :] + rho[:, None] * L[None, :]
    rn = np.linalg.norm(r, axis=1)
    vt = att["V0"][None, :] + rho[:, None] * Ldot[None, :]
    b = vt @ L
    disc = b * b - (np.einsum("ij,ij->i", vt, vt) - 2 * MU / rn)
    ok_rho = disc > 0
    sq = np.sqrt(np.where(ok_rho, disc, 0.0))
    f = np.linspace(0, 1, cfg.n_v + 2)[1:-1]
    rd = (-b - sq)[:, None] + (2 * sq)[:, None] * f[None, :]
    RHO = np.repeat(rho[:, None], cfg.n_v, 1)
    pos, vel = states(att["x"], RHO.ravel(), rd.ravel(), att["R0"], att["V0"])
    return {"rho": RHO.ravel(), "rd": rd.ravel(), "pos": pos, "vel": vel,
            "te": att["tdb0"] - RHO.ravel() / C_AU_DAY,
            "valid": np.repeat(ok_rho[:, None], cfg.n_v, 1).ravel(),
            "shape": (cfg.n_rho, cfg.n_v),
            "drd": np.repeat((2 * sq / (cfg.n_v + 1))[:, None], cfg.n_v, 1).ravel()}


# ---------------------------------------------------------------------- candidates


def candidate_arrays(rows, obscodes):
    """Index rows -> the arrays the scorer reads (observer state at each mid time; the
    geocentre for a space or unknown station)."""
    jd = np.array([r["epoch_mid_sec"] / 86400.0 + UNIX_JD0 for r in rows])
    codes = [r["stn"] if _ground(r["stn"], obscodes) else "500" for r in rows]
    O, Odot = fast_observers(jd, codes, obscodes) if len(rows) else (np.zeros((0, 3)),) * 2

    def g(k, d):
        return np.array([(r.get(k) if r.get(k) is not None else d) for r in rows], float)
    return {"n": len(rows), "jd": jd, "tdb": jd + TT_UTC_DAYS, "O": O, "Odot": Odot,
            "ra": np.radians(g("center_ra", 0)), "dec": np.radians(g("center_dec", 0)),
            "ra_rate": np.radians(g("dra_cosdec_rate", 0)),
            "dec_rate": np.radians(g("ddec_rate", 0)),
            "sp_e": g("sig_pos_e_arcsec", 1.0), "sp_n": g("sig_pos_n_arcsec", 1.0),
            "sv_e": g("sig_vel_e_degday", 0.004), "sv_n": g("sig_vel_n_degday", 0.004),
            "mag": g("mag_mean", np.nan), "band": [r.get("band") or "" for r in rows],
            "id": [r["tracklet_id"] for r in rows]}


# ---------------------------------------------------------------------- the screen


def _elements_for(x, fam, att):
    pos, vel = states(np.broadcast_to(x, (len(fam["rho"]), 4)), fam["rho"], fam["rd"],
                      att["R0"], att["V0"])
    with np.errstate(invalid="ignore", divide="ignore"):
        return _safe_elements(*conic.elliptic_elements(pos, vel, fam["te"]))


def _diff4(a, b):
    de, dn = tangent_offsets(b[0], b[1], a[0], a[1])
    return np.stack([de, dn, b[2] - a[2], b[3] - a[3]], -1)


def _grid_jac(A, shape, valid):
    """d(east, north, east-rate, north-rate)/d(index) along both grid axes."""
    nr, nv = shape
    G = [x.reshape(nr, nv) for x in A]
    V = valid.reshape(nr, nv)
    J = np.zeros((nr, nv, 4, 2))
    for ax in (0, 1):
        def shifted(arr, s):
            out = np.roll(arr, -s, axis=ax).copy()
            idx = [slice(None), slice(None)]
            idx[ax] = slice(arr.shape[ax] - s, None) if s > 0 else slice(0, -s)
            out[tuple(idx)] = False if arr.dtype == bool else arr[tuple(idx)]
            return out
        up, dn = tuple(shifted(g, 1) for g in G), tuple(shifted(g, -1) for g in G)
        vu, vd = shifted(V, 1), shifted(V, -1)
        dU, dD = _diff4(G, up), _diff4(G, dn)
        J[..., ax] = np.where((vu & vd)[..., None], (dU - dD) / 2,
                              np.where(vu[..., None], dU, np.where(vd[..., None], -dD, 0.0)))
    return J.reshape(nr * nv, 4, 2)


def pair_screen(att, fam, cand, cfg, obscodes):
    """Best (2-dof chi2, member, sub-grid offset) per candidate."""
    t0 = att["tdb0"]
    dt = cand["tdb"] - t0
    keep = (np.abs(dt) >= cfg.min_dt_days) & (np.abs(dt) <= cfg.window_days)
    frames = np.round(dt).astype(int)
    n = cand["n"]
    best_chi, best_m, best_delta = np.full(n, np.inf), np.full(n, -1), np.zeros((n, 2))
    ks = np.unique(frames[keep])
    if not len(ks):
        return best_chi, best_m, best_delta
    E_k, Edot_k = fast_observers(att["t0"] + ks, ["500"] * len(ks), obscodes)
    el0, ok0 = _elements_for(att["x"], fam, att)
    lam, vec = np.linalg.eigh(att["cov"])
    pert = [_elements_for(att["x"] + math.sqrt(max(lam[i], 0.0)) * vec[:, i], fam, att)
            for i in range(4)]
    valid = fam["valid"] & ok0
    idx_m = np.flatnonzero(valid)
    if not len(idx_m):
        return best_chi, best_m, best_delta
    pos_floor = cfg.model_pos_floor_arcsec * ARCSEC
    rate_floor = cfg.model_rate_floor_degday * DEG
    tau = cfg.rate_tau_days
    kn = cfg.knn
    for ki, k in enumerate(ks):
        ci = np.flatnonzero(keep & (frames == k))
        if not len(ci):
            continue
        tk = t0 + k
        pos, vel = conic.state_at(*el0, tk)
        A0 = topo_attr(pos - E_k[ki], vel - Edot_k[ki])
        Jg = _grid_jac(A0[:4], fam["shape"], valid)
        u = np.stack([np.cos(A0[1]) * np.cos(A0[0]), np.cos(A0[1]) * np.sin(A0[0]),
                      np.sin(A0[1]), tau * A0[2], tau * A0[3]], 1)
        tree = cKDTree(u[idx_m])
        dtc = tk - cand["tdb"][ci]
        ra_k = cand["ra"][ci] + cand["ra_rate"][ci] * dtc / np.cos(cand["dec"][ci])
        dec_k = cand["dec"][ci] + cand["dec_rate"][ci] * dtc
        uc = np.stack([np.cos(dec_k) * np.cos(ra_k), np.cos(dec_k) * np.sin(ra_k),
                       np.sin(dec_k), tau * cand["ra_rate"][ci], tau * cand["dec_rate"][ci]], 1)
        dd, ii = tree.query(uc, k=kn, distance_upper_bound=cfg.knn_cut_deg * DEG)
        dd, ii = np.atleast_2d(dd), np.atleast_2d(ii)
        hit = np.isfinite(dd)
        if not hit.any():
            continue
        pc = np.repeat(ci, kn).reshape(len(ci), kn)[hit]
        pm = idx_m[ii[hit]]
        # the seed-noise Jacobian, only for the members actually paired
        um, inv = np.unique(pm, return_inverse=True)
        Jx = np.zeros((len(um), 4, 4))
        vx = np.ones(len(um), bool)
        A0u = tuple(a[um] for a in A0[:4])
        for i, (el_i, ok_i) in enumerate(pert):
            p_i, v_i = conic.state_at(*(np.asarray(v)[um] for v in el_i), tk)
            Ai = topo_attr(p_i - E_k[ki], v_i - Edot_k[ki])
            Jx[:, :, i] = _diff4(A0u, Ai[:4])
            vx &= ok_i[um]
        Jx[~vx] = 0.0
        Cs = np.einsum("nij,nkj->nik", Jx, Jx)
        # exact topocentric prediction at the candidate's own time (Taylor from the frame)
        dtau = cand["tdb"][pc] - tk
        acc = -MU * pos[pm] / np.linalg.norm(pos[pm], axis=1)[:, None] ** 3
        P = pos[pm] + vel[pm] * dtau[:, None] + 0.5 * acc * dtau[:, None] ** 2
        Vv = vel[pm] + acc * dtau[:, None]
        Dt = P - cand["O"][pc]
        Dt = Dt - Vv * (np.linalg.norm(Dt, axis=1) / C_AU_DAY)[:, None]
        ra, dec, rar, decr, _ = topo_attr(Dt, Vv - cand["Odot"][pc])
        de, dn = tangent_offsets(cand["ra"][pc], cand["dec"][pc], ra, dec)
        r = np.stack([de, dn, cand["ra_rate"][pc] - rar, cand["dec_rate"][pc] - decr], 1)
        S = Cs[inv].copy()
        S[:, 0, 0] += np.hypot(np.maximum(cand["sp_e"][pc], cfg.cand_pos_floor) * ARCSEC,
                               pos_floor) ** 2
        S[:, 1, 1] += np.hypot(np.maximum(cand["sp_n"][pc], cfg.cand_pos_floor) * ARCSEC,
                               pos_floor) ** 2
        S[:, 2, 2] += np.hypot(np.maximum(cand["sv_e"][pc], cfg.cand_rate_floor) * DEG,
                               rate_floor) ** 2
        S[:, 3, 3] += np.hypot(np.maximum(cand["sv_n"][pc], cfg.cand_rate_floor) * DEG,
                               rate_floor) ** 2
        dsc = 1.0 / np.sqrt(np.einsum("nii->ni", S))
        Si = np.linalg.inv(S * dsc[:, :, None] * dsc[:, None, :])
        rw = r * dsc
        J = Jg[pm] * dsc[:, :, None]
        JtSi = np.einsum("nij,nik->njk", J, Si)
        delta = np.linalg.solve(np.einsum("njk,nkl->njl", JtSi, J) + 1e-12 * np.eye(2),
                                np.einsum("njk,nk->nj", JtSi, rw)[:, :, None])[:, :, 0]
        delta = np.clip(delta, -cfg.sub_grid_clip, cfg.sub_grid_clip)
        res = rw - np.einsum("nij,nj->ni", J, delta)
        chi = np.einsum("ni,nij,nj->n", res, Si, res)
        order = np.lexsort((chi, pc))
        pcs, first = np.unique(pc[order], return_index=True)
        sel = order[first]
        better = chi[sel] < best_chi[pcs]
        best_chi[pcs[better]] = chi[sel][better]
        best_m[pcs[better]] = pm[sel][better]
        best_delta[pcs[better]] = delta[sel][better]
    return best_chi, best_m, best_delta


# ---------------------------------------------------------------------- refinement


def _predict_batch(P, att, O, Odot, t_tdb, e_max=2.0):
    """Predicted (ra, dec, rates, distance) of 6-parameter rows P at times ``t_tdb``,
    hyperbolic states included up to ``e_max``; plus an ok mask and (q, e)."""
    lr = P[:, 4]
    inr = (lr > math.log(1e-4)) & (lr < math.log(100.0))
    rho = np.exp(np.clip(lr, math.log(1e-4), math.log(100.0)))
    pos, vel = states(P[:, :4], rho, P[:, 5], att["R0"], att["V0"])
    el, ok = _safe_elements(*universal_elements(pos, vel, att["tdb0"] - rho / C_AU_DAY))
    ok &= inr & (el[1] < e_max)
    tt = np.array(t_tdb, float, copy=True)
    for _ in range(2):
        ps, vs = conic.state_at(*el, tt)
        tt = t_tdb - np.linalg.norm(ps - O, axis=1) / C_AU_DAY
    ra, dec, rar, decr, dist = topo_attr(ps - O, vs - Odot)
    return ra, dec, rar, decr, dist, ps, ok, el[0], el[1]


def _lm(resid, P, steps, iters):
    """Batched Levenberg-Marquardt: one damping factor per row, a step kept only where
    it lowers that row's chi2.  ``resid(P) -> (r (K, m), ok (K,))``."""
    r, ok = resid(P)
    chi = np.einsum("ij,ij->i", r, r)
    lam = np.full(len(P), 1e-3)
    J = np.zeros(r.shape + (6,))
    for _ in range(iters):
        for j in range(6):
            Pj = P.copy()
            Pj[:, j] += steps[j]
            J[:, :, j] = (resid(Pj)[0] - r) / steps[j]
        A = np.einsum("kij,kil->kjl", J, J)
        g = np.einsum("kij,ki->kj", J, r)
        D = np.einsum("kjj->kj", A)
        improved = False
        for _try in range(4):
            Al = A + lam[:, None, None] * np.einsum("kj,jl->kjl", D + 1e-30, np.eye(6))
            try:
                dP = -np.linalg.solve(Al, g[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                dP = np.zeros_like(P)
            rn, okn = resid(P + dP)
            chin = np.einsum("ij,ij->i", rn, rn)
            better = chin < chi
            P[better] += dP[better]
            r[better], chi[better], ok[better] = rn[better], chin[better], okn[better]
            lam[better] *= 0.3
            lam[~better] *= 10.0
            improved |= bool(better.any())
            if better.all():
                break
        if not improved:
            break
    for j in range(6):
        Pj = P.copy()
        Pj[:, j] += steps[j]
        J[:, :, j] = (resid(Pj)[0] - r) / steps[j]
    try:
        C = np.linalg.inv(np.einsum("kij,kil->kjl", J, J) + 1e-30 * np.eye(6))
    except np.linalg.LinAlgError:
        C = np.full((len(P), 6, 6), np.nan)
    return P, r, chi, ok, C


def refine(att, cand, CI, P0, cfg, iters=25):
    """ONE two-body orbit (seed attributable + log rho + rhodot) per row of ``CI`` (K, m)
    through the seed and that row's m candidates.  Returns P, chi2, per-candidate chi2
    (K, m), ok, covariance."""
    CI = np.atleast_2d(np.asarray(CI, int))
    K, m = CI.shape
    Lc = np.linalg.cholesky(np.linalg.inv(att["cov"]))
    x0 = att["x"]

    def f(key):
        return np.asarray(cand[key])[CI]
    O, Odot, t = cand["O"][CI], cand["Odot"][CI], cand["tdb"][CI]
    sp_e = np.maximum(f("sp_e"), cfg.cand_pos_floor) * ARCSEC
    sp_n = np.maximum(f("sp_n"), cfg.cand_pos_floor) * ARCSEC
    sv_e = np.maximum(f("sv_e"), cfg.cand_rate_floor) * DEG
    sv_n = np.maximum(f("sv_n"), cfg.cand_rate_floor) * DEG
    cra, cdec, crr, cdr = f("ra"), f("dec"), f("ra_rate"), f("dec_rate")

    def resid(P):
        out = np.empty((K, 4 + 4 * m))
        out[:, :4] = (P[:, :4] - x0) @ Lc
        okall = np.ones(K, bool)
        for j in range(m):
            ra, dec, rar, decr, _d, _ps, ok, _q, _e = _predict_batch(P, att, O[:, j],
                                                                     Odot[:, j], t[:, j])
            de, dn = tangent_offsets(ra, dec, cra[:, j], cdec[:, j])
            out[:, 4 + 4 * j:8 + 4 * j] = np.stack(
                [de / sp_e[:, j], dn / sp_n[:, j], (rar - crr[:, j]) / sv_e[:, j],
                 (decr - cdr[:, j]) / sv_n[:, j]], 1)
            okall &= ok
        out[~okall, 4:] = 1e4
        return out, okall

    steps = np.concatenate([np.sqrt(np.diag(att["cov"])) * 1e-2, [1e-4, 1e-6]])
    P, r, chi, ok, C = _lm(resid, np.array(P0, float, copy=True), steps, iters)
    per = np.stack([np.einsum("ij,ij->i", r[:, 4 + 4 * j:8 + 4 * j], r[:, 4 + 4 * j:8 + 4 * j])
                    for j in range(m)], 1)
    return P, chi, per, ok, C


def _starts(att, fam, cand, cfg, cis, ms, deltas):
    """LM starts: each member moved by its sub-grid offset -- back to the member itself
    where that step crosses the bound/unbound edge (an unusable start would silently lose
    the candidate)."""
    dlog = math.log(cfg.rho_hi / cfg.rho_lo) / (cfg.n_rho - 1)
    P = np.concatenate([np.repeat(att["x"][None, :], len(cis), 0),
                        (np.log(fam["rho"][ms]) + deltas[:, 0] * dlog)[:, None],
                        (fam["rd"][ms] + deltas[:, 1] * fam["drd"][ms])[:, None]], 1)
    ok = _predict_batch(P, att, cand["O"][cis], cand["Odot"][cis], cand["tdb"][cis])[6]
    P[~ok, 4] = np.log(fam["rho"][ms][~ok])
    P[~ok, 5] = fam["rd"][ms][~ok]
    return P


def _dmag(att, P, cand, ci):
    """Candidate magnitude minus the one the seed's V predicts through orbit ``P``."""
    if att["V"] is None or not np.isfinite(cand["mag"][ci]):
        return None
    ra, dec, rar, decr, d1, ps, ok, q, e = _predict_batch(
        P[None, :], att, cand["O"][ci][None, :], cand["Odot"][ci][None, :],
        np.array([cand["tdb"][ci]]))
    if not ok[0]:
        return None
    rho0 = math.exp(P[4])
    pos0, _ = states(P[None, :4], [rho0], [P[5]], att["R0"], att["V0"])

    def mag_term(r, d, R):
        cosph = min(1.0, max(-1.0, (r * r + d * d - R * R) / (2 * r * d)))
        tt = math.tan(math.acos(cosph) / 2)
        ph = 0.85 * math.exp(-3.33 * tt ** 0.63) + 0.15 * math.exp(-1.87 * tt ** 1.22)
        return 5 * math.log10(r * d) - 2.5 * math.log10(max(ph, 1e-6))
    H = att["V"] - mag_term(float(np.linalg.norm(pos0[0])), rho0, float(np.linalg.norm(att["R0"])))
    vpred = H + mag_term(float(np.linalg.norm(ps[0])), float(d1[0]),
                         float(np.linalg.norm(cand["O"][ci])))
    return float(cand["mag"][ci] + photometry.band_shift(cand["band"][ci]) - vpred)


def _prior_logw(att, P):
    from pyroorb import popprior
    rho = np.exp(P[:, 4])
    pos, vel = states(P[:, :4], rho, P[:, 5], att["R0"], att["V0"])
    d = popprior.density(np.hstack([pos, vel]), att["R0"], vmag=att["V"])
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(d, 1e-300)) + 4 * np.log(rho)


def rank(att, cand, cfg, obscodes):
    """Screen + refine + score: ranked pair solutions (dicts, best first)."""
    fam = family(att, cfg)
    best_chi, best_m, best_delta = pair_screen(att, fam, cand, cfg, obscodes)
    vague = 2 * np.log(np.maximum(cand["sv_e"], cfg.cand_rate_floor)
                       * np.maximum(cand["sv_n"], cfg.cand_rate_floor) / cfg.vague_ref_degday ** 2)
    pre = np.where(best_chi < cfg.pair_chi2_max, best_chi + cfg.w_vague * vague, np.inf)
    order = np.argsort(pre)
    order = np.array([i for i in order[:cfg.refine_top] if np.isfinite(pre[i])], int)
    if not len(order):
        return []
    P0 = _starts(att, fam, cand, cfg, order, best_m[order], best_delta[order])
    P, chi, _per, ok, C = refine(att, cand, order[:, None], P0, cfg)
    out = []
    for k, ci in enumerate(order):
        if not ok[k]:
            continue
        dmag = _dmag(att, P[k], cand, ci)
        pen = (abs(dmag) - cfg.mag_tol) ** 2 * 4 if dmag is not None and abs(dmag) > cfg.mag_tol \
            else 0.0
        out.append({"ci": int(ci), "chi2": float(chi[k]), "vague": float(vague[ci]),
                    "dmag": dmag, "magpen": pen, "p": P[k].copy(), "cov": C[k]})
    if out:
        lw = _prior_logw(att, np.array([o["p"] for o in out]))
        lmax = float(np.max(lw))
        for o, w in zip(out, lw):
            o["score"] = (o["chi2"] + o["magpen"] + cfg.w_vague * o["vague"]
                          + 2 * cfg.w_prior * min(lmax - float(w), cfg.prior_cap))
    out.sort(key=lambda o: o["score"])
    return out


def triplets(att, cand, ranked, cfg):
    """Seed + two compatible candidates (different nights) refitted as one orbit."""
    R = ranked[:cfg.trip_top]
    if len(R) < 2:
        return []
    cis = np.array([r["ci"] for r in R])
    P = np.array([r["p"] for r in R])
    C1 = np.array([r["cov"] for r in R])
    chi1 = np.array([r["chi2"] for r in R])
    I, Jj = np.triu_indices(len(R), 1)
    sel = np.abs(cand["tdb"][cis[I]] - cand["tdb"][cis[Jj]]) > cfg.min_dt_days
    d = P[I][:, 4:] - P[Jj][:, 4:]
    Cs = C1[I][:, 4:, 4:] + C1[Jj][:, 4:, 4:] + np.diag([0.05 ** 2, 1e-6])
    with np.errstate(invalid="ignore"):
        try:
            md = np.einsum("ni,nij,nj->n", d, np.linalg.inv(Cs), d)
        except np.linalg.LinAlgError:
            md = np.full(len(I), np.inf)
    sel &= np.nan_to_num(md, nan=np.inf) < cfg.trip_md_max
    I, Jj = I[sel], Jj[sel]
    if not len(I):
        return []
    starts = np.where((chi1[I] <= chi1[Jj])[:, None], P[I], P[Jj])
    Pt, chit, _per, okt, _C = refine(att, cand, np.stack([cis[I], cis[Jj]], 1), starts, cfg,
                                     iters=30)
    out = [{"ci": (int(cis[a]), int(cis[b])), "chi2": float(c)}
           for a, b, c, o in zip(I, Jj, chit, okt) if o]
    out.sort(key=lambda r: r["chi2"])
    return out


# ---------------------------------------------------------------------- acceptance


def _fit_summary(seed_lines, cand_lines, obscodes, max_sec):
    """Engine fit of seed + candidates: ok, used/count, rms, chi2/dof and the largest
    |residual/sigma| over the included observations."""
    from .fitter import fit_astrometry
    tr = fit_astrometry("\n".join(seed_lines + cand_lines), obscodes, max_sec=max_sec)
    out = {"ok": bool(tr.ok), "used": tr.used, "count": tr.count,
           "rms": float(tr.rms) if tr.ok else None, "dropped": []}
    if not tr.ok or tr.fit is None:
        return out
    try:
        out["dropped"] = sorted(_obs_key(r) for r in tr.fit.excluded_observations())
        rows = tr.fit.residuals(included_only=True)
        z = [(r["d_ra_arcsec"] / r["sigma_ra"], r["d_dec_arcsec"] / r["sigma_dec"])
             for r in rows if r.get("sigma_ra") and r.get("sigma_dec")]
        if z:
            out["chi2_dof"] = sum(a * a + b * b for a, b in z) / max(2 * len(z) - 6, 1)
            out["max_z"] = max(max(abs(a), abs(b)) for a, b in z)
    except Exception:                                        # noqa: BLE001
        pass
    return out


def obs80_lines(obs):
    """The 80-column lines of observation dicts, a satellite / roving record's second line
    (``obs80_line2``) after its first."""
    out = []
    for o in obs:
        out.append(o["obs80_line"])
        if o.get("obs80_line2"):
            out.append(o["obs80_line2"])
    return out


def _own_detections(seed_lines):
    from .search import own_detections
    return own_detections("\n".join(seed_lines))


def _own_lines(lines, own):
    """The candidate lines that are the seed's own detections: the MPC keeps an accepted ITF
    detection in the ITF table, under its trkSub, while it sits on a NEOCP object, so a seed
    can be offered its own tracklet back (the regular search refuses it the same way,
    ``search.own_detections``)."""
    from .search import own_lines
    return own_lines(lines, own) if own else []


def _obs_key(row):
    """An observation's identity in a fit's residual rows: station + time to the second."""
    return row["obscode"], round(row["jd_utc"] * 86400.0)


def _seed_rejects(seed_lines, obscodes, max_sec):
    """The seed observations the seed's own fit rejects."""
    from .fitter import fit_astrometry
    tr = fit_astrometry("\n".join(seed_lines), obscodes, max_sec=max_sec)
    if not tr.ok or tr.fit is None:
        return frozenset()
    return frozenset(_obs_key(r) for r in tr.fit.excluded_observations())


def _fits_but_drops(fit, cfg, n_expected):
    """Every test of :func:`_passes` met except that the fit rejected observations."""
    return (fit["ok"] and fit["count"] >= n_expected and fit["used"] < fit["count"]
            and fit.get("chi2_dof") is not None and fit["chi2_dof"] <= cfg.accept_chi2_dof
            and fit.get("max_z", 99.0) <= cfg.accept_max_z)


def _passes(fit, cfg, n_expected, pardoned=frozenset()):
    """Every observation kept -- the seed's own flawed ones (``pardoned``: a seed of two or
    more nights whose own fit rejects them) excepted: such a seed, often from several
    stations, carries points like that, and a true link that confirms the rejection is not
    to blame for it (a real two-night seed's every true proposal failed on one) --
    chi2/dof <= ``accept_chi2_dof`` and max |z| <= ``accept_max_z``."""
    kept = fit["used"] == fit["count"] or (
        fit["count"] - fit["used"] == len(fit.get("dropped") or ())
        and set(fit.get("dropped") or ()) <= pardoned)
    return (fit["ok"] and fit["count"] >= n_expected and kept
            and fit.get("chi2_dof") is not None and fit["chi2_dof"] <= cfg.accept_chi2_dof
            and fit.get("max_z", 99.0) <= cfg.accept_max_z)


def sr_search(seed_obs, source, obscodes, cfg=None, log=None):
    """The single-tracklet ITF search: ``seed_obs`` (engine observation dicts carrying
    ``obs80_line``) against ``source`` (``window_rows`` + ``tracklet_obs80``).  Returns an
    :class:`SrResult`; ``result.link`` is the accepted link or None."""
    cfg = cfg or SrConfig()
    log = log or (lambda *a, **k: None)
    res = SrResult()
    t_all = time.time()
    seed_obs = [o for o in seed_obs if o.get("jd_utc") is not None]
    problem = seed_night_problem(seed_obs, obscodes)
    if problem:
        res.reason = problem
        return res
    att = seed_attributable(seed_obs, obscodes, cfg)
    mid = (att["t0"] - UNIX_JD0) * 86400.0
    rows = source.window_rows(mid - cfg.window_days * 86400, mid + cfg.window_days * 86400)
    res.n_window = len(rows)
    res.seconds["pull"] = round(time.time() - t_all, 2)
    if not rows:
        res.reason = "no ITF tracklet in the window"
        return res
    cand = candidate_arrays(rows, obscodes)
    t = time.time()
    ranked = rank(att, cand, cfg, obscodes)
    res.n_refined = len(ranked)
    trips = triplets(att, cand, ranked, cfg)
    res.seconds["search"] = round(time.time() - t, 2)
    props, seen = [], set()
    for tr in trips:
        if tr["chi2"] > cfg.trip_chi2_max:
            break
        key = tuple(sorted(cand["id"][c] for c in tr["ci"]))
        if key not in seen:
            seen.add(key)
            props.append(("triplet", key, tr["chi2"]))
    for r in ranked:
        key = (cand["id"][r["ci"]],)
        if r["chi2"] <= cfg.pair_propose_chi2 and key not in seen:
            seen.add(key)
            props.append(("pair", key, r["chi2"]))
    props = props[:cfg.fits]
    if not props:
        res.reason = "no candidate fits the seed"
        res.seconds["total"] = round(time.time() - t_all, 2)
        return res
    t = time.time()
    db_trksubs = {}
    lines = source.tracklet_obs80(sorted({k for _kind, key, _c in props for k in key}),
                                  trksubs=db_trksubs)
    seed_lines = obs80_lines(seed_obs)
    own = _own_detections(seed_lines)
    idx = {cid: i for i, cid in enumerate(cand["id"])}
    # a single night cannot fix an orbit, so its own fit's rejections mean nothing
    multi_night = _nights(sorted(o["jd_utc"] for o in seed_obs)) >= 2
    passing, pardoned = [], None
    for kind, key, chi2 in props:
        cl = [ln for k in key for ln in lines.get(k, [])]
        why = ("observations not found" if any(len(lines.get(k, [])) < 2 for k in key)
               else "the seed's own detections" if _own_lines(cl, own) else None)
        if why:
            res.proposals.append({"kind": kind, "trkids": list(key), "sr_chi2": chi2,
                                  "fit": None, "why": why})
            continue
        fit = _fit_summary(seed_lines, cl, obscodes, cfg.fit_max_sec)
        n_exp = len(seed_obs) + len(cl)
        if multi_night and pardoned is None and _fits_but_drops(fit, cfg, n_exp):
            pardoned = _seed_rejects(seed_lines, obscodes, cfg.fit_max_sec)
        ok = _passes(fit, cfg, n_exp, pardoned or frozenset())
        res.proposals.append({"kind": kind, "trkids": list(key), "sr_chi2": chi2, "fit": fit,
                              "passes": ok})
        if ok:
            passing.append((kind, key, chi2, fit))
    res.seconds["fits"] = round(time.time() - t, 2)
    trips_ok = [p for p in passing if p[0] == "triplet" and p[2] <= cfg.trip_chi2_max]
    pairs_ok = [p for p in passing if p[0] == "pair"]
    pick = trips_ok[0] if trips_ok else (pairs_ok[0] if pairs_ok else None)
    res.seconds["total"] = round(time.time() - t_all, 2)
    if pick is None:
        res.reason = f"none of {len(props)} proposals passed the engine fit"
        return res
    kind, key, chi2, fit = pick
    res.link = SrLink(trkids=list(key), kind=kind, lines={k: lines[k] for k in key},
                      sr_chi2=chi2, fit=fit, trksubs=db_trksubs,
                      dts=[round(float(cand["tdb"][idx[k]] - att["tdb0"]), 2) for k in key])
    res.reason = f"linked {len(key)} tracklet(s) ({kind})"
    log(f"SR search: {res.reason} in {res.seconds['total']:.1f}s")
    return res


# ---------------------------------------------------------------------- the hand-off


def itfitf_check(combined_obs80, trial):
    """``(ok, reasons)``: would the MPC accept a single-tracklet find as an ITF-to-ITF
    identification?  :func:`mpcrules.check` on the whole arc -- >= 3 nights over >= 3 days,
    a 3-night arc within 15 days, the apparition profile, the rms / orbit-quality test --
    with ``trial`` (an :class:`itf.fitter.TrialFit` of that arc) supplying the fit and rms,
    as the regular search's own check does.  A find that passes is submittable like any
    linkage; one that does not is a find for review."""
    from . import mpcrules
    if trial is None or getattr(trial, "fit", None) is None:
        return mpcrules.check(combined_obs80 or "")
    return mpcrules.check(combined_obs80 or "", fit=trial.fit, rms_arcsec=trial.rms)


#: a seed longer than this is never fitted for routing: it always fixes a distance (every
#: seed was determined from 4 days on), and a many-apparition fit can outrun the budget
ROUTE_MAX_FIT_ARC_DAYS = 180.0
ROUTE_FIT_MAX_SEC = 20.0


def seed_determined(obs80_text, obscodes, max_sec=ROUTE_FIT_MAX_SEC):
    """Does the seed's own fit fix a distance?

    True when its fit is determined (``FittedOrbit.determinacy``); False when it is not, or
    when the fit ran and no orbit fits the seed (nothing fixes a distance then either);
    None when there is no answer -- the fit ran out of ``max_sec`` or raised, or it has no
    readable covariance to judge it by.  A missing answer says nothing about the seed: the
    budget depends on the machine's load."""
    from .fitter import fit_astrometry
    try:
        tr = fit_astrometry(obs80_text, obscodes, max_sec=max_sec)
        if tr.timed_out:
            return None
        if not tr.ok or tr.fit is None:
            return False
        cov = np.asarray(tr.fit.covariance_matrix("state"), float)
        if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
            return None
        return tr.fit.determinacy() is None
    except Exception:                                        # noqa: BLE001
        return None


def needs_sr(obs, obs80_text, obscodes):
    """``(use_sr, why)``: does this seed go to the single-tracklet search, or to the regular
    ITF search?  By whether the seed fixes a distance, never by its length: one night cannot
    be fitted at all, and a longer seed goes to SR only when its own fit leaves the distance
    open or no orbit fits it at all; a fit that gives no answer (out of time, or an error)
    leaves it to the regular search.  Measured on 80 real short seeds (discovery tracklet + the
    follow-up night): 49 were already determined, and there the regular search linked
    48 / 1 wrong / 0 none against SR's 43 / 2 / 4, with no false link where nothing of the
    object was in the ITF (SR 2); on the 31 undetermined ones the two found as much
    (27 vs 26 correct, none wrong) and SR took a quarter of the time where there was nothing
    to find (median 14 s vs 56 s).  Past 2 days nearly every seed is
    determined and the regular search is at least as good in every arc bin to 15 days."""
    jds = sorted(o["jd_utc"] for o in obs if o.get("jd_utc") is not None)
    if not jds:
        return False, "no timed observations"
    if _nights(jds) <= 1:
        return True, "a single night fixes no distance"
    arc = jds[-1] - jds[0]
    if arc > ROUTE_MAX_FIT_ARC_DAYS:
        return False, f"the seed arc ({arc:.0f} d) fixes a distance"
    det = seed_determined(obs80_text, obscodes)
    if det is None:
        # no answer is no evidence: the regular search fits the seed again on its own budget
        return False, f"the seed's fit ({arc:.2f} d) gave no answer in time"
    if det:
        return False, f"the seed's fit ({arc:.2f} d) fixes a distance"
    return True, f"the seed's fit ({arc:.2f} d) leaves the distance open"


def _nights(jds):
    """Observing nights in ``jds`` -- the tree's one night (``orbit_common.count_nights``: a
    12-hour window opened by the night's first observation), the unit the MPC's rules and
    the chimera rule count in.  Not a gap rule: an object followed around the clock by
    stations at several longitudes never leaves a 12-hour gap, yet it is seen on several
    nights, not one."""
    return count_nights(jds)


def apparition_config(cfg, seed_jd, half_days=180.0):
    """A copy of ``cfg`` whose sweep covers only the seed's apparition (+-``half_days``),
    with the pre-1997 archive pass off: the same-apparition stage of an SR hand-off."""
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone
    start = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=seed_jd - UNIX_JD0 - half_days)
    return replace(cfg, ephem_start=start.strftime("%Y-%m-%d"),
                   ephem_steps=int(2 * half_days / cfg.ephem_step_days), archive_start="")


def sr_then_itf(seed_obs, source, obscodes, primary_desigs, itf_cfg, sr_cfg=None,
                log=None, apparition_only=False, **search_kw):
    """The single-tracklet search, then -- when it links -- the regular ITF search seeded
    with the seed AND the linked tracklets: over every apparition, or over the seed's alone
    (``apparition_only``, :func:`apparition_config`: a first step whose multi-apparition
    search waits until the MPC has accepted what the first one sent).  Returns
    ``(sr_result, itf_result)``; ``itf_result`` is None when the SR stage found nothing.
    The SR link rides in the regular search's SEED, where its gates do not judge it, and
    ``itf_result.itf_obs80`` lists only what that search ADDED: the caller judges the
    whole find -- :func:`link_lines` plus the added tracklets, by
    :func:`chimera_warnings_short_seed` with ``core_text`` the SR link -- so the regular
    search attaches no chimera warnings of its own (``chimera_check=False``) and never
    transmits (``do_submit`` is refused here: nothing would judge the document first).  The
    SR link is named to that search as unvouched seed (``unvouched_obs80``): a caution about
    one of its tracklets stays a caution."""
    from .search import itf_search
    if search_kw.pop("do_submit", False):
        raise ValueError("sr_then_itf never transmits: judge the find "
                         "(chimera_warnings_short_seed), then send it")
    sr = sr_search(seed_obs, source, obscodes, sr_cfg, log=log)
    if sr.link is None:
        return sr, None
    sr_lines = link_lines(sr.link)
    lines = obs80_lines(seed_obs) + sr_lines
    cfg = (apparition_config(itf_cfg, min(o["jd_utc"] for o in seed_obs
                                          if o.get("jd_utc") is not None))
           if apparition_only else itf_cfg)
    itf = itf_search(source, seed_obs80="\n".join(lines), primary_desigs=list(primary_desigs),
                     cfg=cfg, obscodes=obscodes, verbose=False,
                     log=log or (lambda *a, **k: None), chimera_check=False,
                     unvouched_obs80="\n".join(sr_lines), do_submit=False, **search_kw)
    return sr, itf


def sr_then_apparition(seed_obs, source, obscodes, primary_desigs, itf_cfg, sr_cfg=None,
                       log=None, **search_kw):
    """:func:`sr_then_itf` over the seed's apparition only."""
    return sr_then_itf(seed_obs, source, obscodes, primary_desigs, itf_cfg, sr_cfg=sr_cfg,
                       log=log, apparition_only=True, **search_kw)


def link_lines(link):
    """The 80-column lines of an :class:`SrLink`'s tracklets, in link order."""
    return [ln for k in link.trkids for ln in link.lines.get(k, ())]


# ------------------------------------------------------- the 2-3 night chimera guard
#
# A link this search finds rides in the regular search's SEED, where nothing judges it, and
# a weak or bent seed can absorb a foreign tracklet that fits as well as a true one: from
# wrong seeds, 36 chimeras came back LINKED in 418 hand-offs.  The 2-3 NIGHT CHIMERA GUARD
# judges linkages of at most three nights -- never more (see :func:`chimera_warnings`) --
# with two independent tests, both of which must pass:
#
# * ARC CAPS on the whole linked astrometry: a linkage resting on 2 nights must span
#   <= GUARD_MAX_ARC_2_NIGHTS days, on 3 nights <= GUARD_MAX_ARC_3_NIGHTS (the MPC's own
#   three-night limit for an ITF-to-ITF identification).  Every 2- and 3-night chimera
#   measured spanned 8-228 days; a short span leaves too little room for a foreign tracklet
#   to fit by chance.  One night is never a linkage.
# * LEAVE-OUT on every judged tracklet: refit WITHOUT it (``leaveout.measure``) and the rest
#   must PREDICT it -- the prediction informative (1-sigma <= GUARD_MAX_SIGMA_ARCSEC) and the
#   tracklet within GUARD_MISS_SIGMAS of it (floor GUARD_MISS_FLOOR_ARCSEC), from >= 2
#   observations.
#
# Why not at 4+ nights: on 95 saved hand-offs the leave-out kept 27 of 44 true 4+ night
# linkages where the MPC minimums and the fit's residual verdict kept 40, for ONE chimera
# those let through (and perhaps not even that: its "foreign" tracklet sits ~3" from the
# object's own orbit, so it may be the object's own high-residual measurements); on thousands of selected chimeras (a real core plus every foreign
# tracklet inside its 3-sigma prediction) residuals and photometry let 1 of 2382 four-night
# chimeras through.  A 4+ night linkage is judged by the search's own gates and the link
# scorers (:mod:`itf.linkscore`) alone.
#
# What a guard objection costs is the caller's: it is a WARNING on the linkage (shown,
# acknowledged before a person submits), and the NEOCP monitor never auto-submits a linkage
# carrying one.

GUARD_MAX_ARC_2_NIGHTS = 4.0
GUARD_MAX_ARC_3_NIGHTS = 15.0
GUARD_MAX_SIGMA_ARCSEC = 10.0
GUARD_MISS_SIGMAS = 2.0
GUARD_MISS_FLOOR_ARCSEC = 1.5
#: one station's observations closer than this (days) are one tracklet (:func:`guard_units`)
GUARD_NIGHT_GAP_DAYS = 0.5


def _guard_lines(text):
    return [ln for ln in (text or "").splitlines() if ln.strip()]


def _is_position_record(line):
    """A satellite / roving observer's second line (column 15 's', 'v' or 'r')."""
    return len(line) > 14 and line[14] in "svr"


def _line_jd(line):
    from pyroorb.obs80 import mpc80_to_radec
    return mpc80_to_radec(line)[0]


def _ut_label(jd_utc):
    """``YYYY-MM-DD hh:mm UT`` for a message a person reads -- never a raw JD."""
    from datetime import datetime, timedelta
    t = datetime(1858, 11, 17) + timedelta(days=float(jd_utc) - 2400000.5)
    t = (t + timedelta(seconds=30)).replace(second=0, microsecond=0)
    return t.strftime("%Y-%m-%d %H:%M UT")


def linkage_shape(obs80_text):
    """``(nights, arc_days)`` of 80-column astrometry: nights as :func:`_nights` counts
    them; arc = last - first observation."""
    jds = []
    for ln in _guard_lines(obs80_text):
        if _is_position_record(ln):
            continue
        try:
            jds.append(_line_jd(ln))
        except Exception:                                    # noqa: BLE001
            continue
    if not jds:
        return 0, 0.0
    jds.sort()
    return _nights(jds), jds[-1] - jds[0]


def caps_problem(nights, arc_days):
    """None when the linkage's shape passes the arc caps, else one sentence."""
    if nights <= 1:
        return "the linkage rests on a single night"
    if nights == 2 and arc_days > GUARD_MAX_ARC_2_NIGHTS:
        return (f"two nights spanning {arc_days:.1f} d (a two-night linkage must span "
                f"<= {GUARD_MAX_ARC_2_NIGHTS:g} d)")
    if nights == 3 and arc_days > GUARD_MAX_ARC_3_NIGHTS:
        return (f"three nights spanning {arc_days:.1f} d (a three-night linkage must span "
                f"<= {GUARD_MAX_ARC_3_NIGHTS:g} d)")
    return None


def guard_units(obs80_text):
    """80-column lines grouped into tracklets: one station, observations closer than
    ``GUARD_NIGHT_GAP_DAYS``; a position record rides with the line before it."""
    rows = []
    for ln in _guard_lines(obs80_text):
        if _is_position_record(ln):
            if rows:
                rows[-1][2].append(ln)
            continue
        try:
            jd = _line_jd(ln)
        except Exception:                                    # noqa: BLE001
            continue
        rows.append((ln[77:80].strip(), jd, [ln]))
    rows.sort(key=lambda r: (r[0], r[1]))
    out, last = [], None
    for stn, jd, lns in rows:
        if last is not None and last[0] == stn and jd - last[1] < GUARD_NIGHT_GAP_DAYS:
            out[-1].extend(lns)
        else:
            out.append(list(lns))
        last = (stn, jd)
    return out


def leaveout_confirmed(m):
    """The relaxed leave-out verdict on one ``leaveout.measure`` result."""
    smaj, miss = m.get("smaj"), m.get("miss")
    if smaj is None or miss is None or m.get("error"):
        return False
    if m.get("n_min", 2) < 2:
        return False                         # one observation: its rate cannot be checked
    return smaj <= GUARD_MAX_SIGMA_ARCSEC and miss <= max(GUARD_MISS_FLOOR_ARCSEC,
                                                           GUARD_MISS_SIGMAS * smaj)


def _leaveout_why(m):
    """Why ``leaveout_confirmed(m)`` is False, worded for a person: which test failed.
    A loose prediction is not a miss -- the rest of the linkage simply cannot vouch."""
    smaj, miss = m.get("smaj"), m.get("miss")
    if m.get("error") or smaj is None or miss is None:
        return f"could not be predicted from the rest of the linkage ({m.get('error') or 'no prediction'})"
    if m.get("n_min", 2) < 2:
        return "is a single observation: the rest of the linkage cannot check its motion"
    if smaj > GUARD_MAX_SIGMA_ARCSEC:
        return (f"cannot be vouched for by the rest of the linkage: its prediction there is too "
                f"uncertain (1-sigma {smaj:.1f}\", the limit is {GUARD_MAX_SIGMA_ARCSEC:g}\"; "
                f"it fell {miss:.1f}\" from the prediction)")
    return (f"is not predicted by the rest of the linkage (missed by {miss:.1f}\", beyond "
            f"{max(GUARD_MISS_FLOOR_ARCSEC, GUARD_MISS_SIGMAS * smaj):.1f}\" = "
            f"max({GUARD_MISS_FLOOR_ARCSEC:g}\", {GUARD_MISS_SIGMAS:g} x 1-sigma {smaj:.1f}\"))")


def chimera_guard(combined, unit_text, obscodes, what="ITF tracklet"):
    """``(ok, why)`` of the 2-3 night chimera guard: the arc caps on ``combined`` (all the
    linked astrometry), then the relaxed leave-out (``leaveout.measure``) on every tracklet
    of ``unit_text``.  Callers ask :func:`chimera_warnings`, which asks this only of 1-3
    night linkages."""
    from . import leaveout
    if not (combined or "").strip() or not (unit_text or "").strip():
        return False, "no linked astrometry to judge"
    why = caps_problem(*linkage_shape(combined))
    if why:
        return False, why
    units = guard_units(unit_text)
    if not units:
        return False, f"no {what} to judge"
    for u in units:
        m = leaveout.measure(u, combined, obscodes)
        if not leaveout_confirmed(m):
            stn, jd = u[0][77:80].strip(), _line_jd(u[0])
            return False, f"the {what} ({stn}, {_ut_label(jd)}) {_leaveout_why(m)}"
    return True, None


# ------------------------------------------------- one chimera rule, by night count

#: Every chimera warning starts with this.
CHIMERA_WARNING = "possible chimera: "
#: The 2-3 night chimera guard's objection, ahead of its reason.
CHIMERA_GUARD_WARNING = (CHIMERA_WARNING + "the 2-3 night chimera guard does not confirm "
                         "this linkage: ")
#: A chimera check that could not run (the guard or a link scorer raised, a scorer would not
#: load, the whole check failed) has not confirmed the linkage: a warning that says so. It
#: says nothing against the linkage, so a sender that trusts a link by default (the
#: identification queue) does not count it (:func:`not_run`).
CHECK_NOT_RUN = (CHIMERA_WARNING + "not confirmed, a check could not run: ")


def not_run(warning):
    """True for a :data:`CHECK_NOT_RUN` warning: a check that did not run, not an objection."""
    return str(warning).startswith(CHECK_NOT_RUN)


def chimera_warnings(combined, unit_text, obscodes, what="ITF tracklet", unvouched=False):
    """The chimera WARNINGS on a linkage (a list, empty when nothing objects), one rule
    for every surface:

    * 1-2 nights: the 2-3 night chimera guard (:func:`chimera_guard`);
    * 3 nights: the guard AND the link scorers (:mod:`itf.linkscore`);
    * 4+ nights: the link scorers alone -- the search's own gates judge the rest.

    ``unvouched`` -- no search gate judged this linkage (the single-tracklet search's link,
    which rides in the regular search's seed).  At 4+ nights the scorers still judge it
    alone when one of them gives a verdict; when none can score it (outside every scorer's
    scope, :data:`itf.linkscore.NO_VERDICT`, or none installed) the guard judges it, so
    such a link is never left unjudged.  (Its arc caps are 2-3 night caps; at 4+ nights it
    is the leave-out on every tracklet.)

    ``combined`` is all the linked astrometry, ``unit_text`` the tracklets the guard asks
    to be predicted.  A check that cannot run -- the guard, or a link scorer
    (:func:`itf.linkscore.judge`) -- has not confirmed the linkage: a :data:`CHECK_NOT_RUN`
    warning.  A warning is shown with the linkage and acknowledged before a person submits;
    it never refuses the linkage by itself."""
    nights, _arc = linkage_shape(combined)
    out = []

    def guard():
        try:
            ok, why = chimera_guard(combined, unit_text, obscodes, what=what)
        except Exception as ex:                              # noqa: BLE001 -- unconfirmed
            out.append(CHECK_NOT_RUN + f"the 2-3 night chimera guard ({type(ex).__name__})")
        else:
            if not ok:
                out.append(CHIMERA_GUARD_WARNING + str(why))

    if nights <= 3:
        guard()
    if nights >= 3:
        from . import linkscore
        verdicts = []
        objections, failed = linkscore.judge(combined, obscodes, verdicts=verdicts)
        out += [CHIMERA_WARNING + w for w in objections]
        out += [CHECK_NOT_RUN + w for w in failed]
        if nights > 3 and unvouched and not verdicts:
            guard()
    return out


def chimera_warnings_result(result, obscodes):
    """:func:`chimera_warnings` for a regular search result: the whole linked astrometry,
    the guard asking only the tracklets the search ADDED (its seed is vouched for by the
    designations it carries).  The search's own judgement (``vetting["chimera_warnings"]``)
    is reused.  No ITF astrometry, nothing to judge: no warning."""
    cached = (getattr(result, "vetting", None) or {}).get("chimera_warnings")
    if cached is not None:
        return list(cached)
    combined = getattr(result, "combined_obs80", "") or ""
    itf = getattr(result, "itf_obs80", "") or ""
    if not itf.strip() or not combined.strip():
        return []
    return chimera_warnings(combined, itf, obscodes)


def chimera_warnings_short_seed(seed_text, itf_text, obscodes, core_text=None):
    """:func:`chimera_warnings` for a link grown from a seed too short to vouch for anything
    (this search's hand-off): the caps on the whole linkage and the leave-out on EVERY
    tracklet, the seed's own included.  A short seed cannot predict an outsider, and a chain
    of another object's tracklets can vouch for itself (seeded with the wrong first
    tracklet and nothing of the object in the ITF, the search linked further tracklets of
    ONE other object); only asking the rest to predict the SEED exposes the outsider.

    ``core_text`` -- the tracklets the single-tracklet search linked, when a regular search
    then added more (``itf_text`` holds both).  The seed plus that core is judged as a
    linkage of its own as well: the regular search's gates never judged it (it rode in that
    search's seed), so the rule that skips the guard on a 4+ night linkage -- whose
    tracklets those gates did judge -- does not cover it: it is judged ``unvouched``
    (:func:`chimera_warnings`), so at 4+ nights a link no scorer can score gets the guard.
    When ``core_text`` is all of ``itf_text`` (the regular search added nothing), the whole
    linkage is that link."""
    if not (itf_text or "").strip():
        return [CHIMERA_GUARD_WARNING + "no ITF tracklet to judge"]
    # blank lines dropped, every line kept whole: an OBS80 line is read by column, and an
    # unnumbered or ITF line starts with five blank columns that a .strip() would eat
    combined = "\n".join(_guard_lines(seed_text) + _guard_lines(itf_text)) + "\n"
    core = _guard_lines(core_text)
    whole_is_core = bool(core) and set(core) == set(_guard_lines(itf_text))
    out = chimera_warnings(combined, combined, obscodes, what="tracklet",
                           unvouched=whole_is_core)
    if core and not whole_is_core:
        first = "\n".join(_guard_lines(seed_text) + core) + "\n"
        for w in chimera_warnings(first, first, obscodes, what="tracklet", unvouched=True):
            if w not in out:
                out.append(w)
    return out
