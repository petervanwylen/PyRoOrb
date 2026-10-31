"""Bound orbits across a multi-year gap, by revolution count.

`fit_object._prefer_bound_ecc` weighs a hyperbolic free fit against the best BOUND orbit it can
find. On a short arc that orbit is looked for across the admissible region
(`hyperbolic.bound_seeds`); past `LONG_ARC_DAYS` the only comparator used to be a penalised walk
from the hyperbolic optimum. On a sparse arc of two or three apparitions years apart the free fit
is a wrong-revolution hyperbola joining them, and no local move from it reaches the bound orbit
that fits the same observations at 0.3" (107474's first three nights: two 1961 plates and one 1999
night, e 1.44 delivered, the walk stuck at 24.6").

What is unknown on such an arc is discrete as well as continuous: WHICH revolution count joins the
apparitions, and at what distance each was seen. So this module ENUMERATES them. Two apparitions
A and B (the two richest), a trial range at each (the object's heliocentric position at each
reference epoch along its observed line of sight), and for every revolution count N the elliptic
Lambert transfers between the two positions -- Lagrange's time-of-flight equation, both branches,
both ways round. Every solution is a bound orbit through both lines of sight by construction; it
is scored against every included observation with a two-body propagation, and the best few, one
per (N, branch, way), are returned as seeds for the capped least-squares fit in
`fit_object._bridged_bound_candidates`. Nothing here decides anything: the station-night model
(`_weigh`) weighs whatever bound orbit the seeds lead to, exactly as it weighs the walk's.

Why not the linkage bridges (`arc_extend._bridge_far_multirev`: period sweep, a-secant, phase
scan, LOV slide)? They start from an ANCHOR ORBIT and slide it along the direction the anchor
leaves free, which presumes the anchor has a shape. Measured on the four #76 arcs (the capped
swarm sweep from each apparition's own admissible bound fits): 2026 DV62 and 1999 KK17 are
reached, 107474 and 2010 TA192 are not -- a two-night 1961 anchor, and a one-night anchor of a
distant object across N = 0, have no period to slide. The two-point boundary problem needs no
anchor orbit, and the revolution count is an explicit loop instead of a zero-crossing a slide may
land on the wrong side of.

Textbook two-body mechanics (Lagrange's equation and the Prussing & Conway velocity form, e.g.
Prussing & Conway, *Orbital Mechanics*, ch. 4; Battin 1987) -- not ported from find_orb, which has
no multi-revolution Lambert.
"""
import math
import numpy as np

from pyroorb.forward_model import MU_SUN, C_AU_DAY

RANGE_GRID_AU = (3.0e-3, 150.0)   # trial ranges at each reference epoch (au, log-spaced)
RANGE_GRID_N = 40                 # per apparition: 40 x 40 range pairs, ~31% apart
A_SCAN_X = (1.0e-7, 1.0e4)        # the a scan, as (a - a_m) / a_m, log-spaced: the upper branch's
A_SCAN_N = 80                     # two roots can sit a hair above the minimum-energy a_m (the object's
                                  # own orbit near a minimum-energy transfer), so the scan is dense there
BISECT_ITERS = 32                 # log-a bisection per bracketed root (widest bracket 0.32 in ln a -> 7e-11)
MAX_REVS = 200                    # revolution-count ceiling (a guard; the geometry sets the real one)
SCORE_OBS = 16                    # observations a candidate is scored on (spread over the arc)
SEEDS_PER_KEY = 1                 # best solutions kept per (N, branch, way)
MAX_SEEDS = 8                     # seeds handed to the capped fit
KEPLER_ITERS = 40
MAX_CANDIDATES = 400_000          # scored in chunks below this; the grid never approaches it in practice


def oppositions(times, idx, gap_days):
    """Split the (included) observation indices `idx` into apparitions: runs separated by more
    than `gap_days`, in time order. Returns a list of index arrays."""
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    idx = idx[np.argsort(np.asarray(times)[idx], kind="stable")]
    t = np.asarray(times)[idx]
    cuts = np.flatnonzero(np.diff(t) > gap_days) + 1
    return [g for g in np.split(idx, cuts) if g.size]


def _unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def lambert_multirev(r1, r2, dt, mu=MU_SUN, max_revs=MAX_REVS):
    """Every ELLIPTIC Lambert solution between position arrays `r1`, `r2` (P x 3, au) in `dt`
    (days, scalar or P): all revolution counts N, both Lagrange branches, both ways round.

    Returns a dict of flat arrays: ``pair`` (index into P), ``N``, ``branch`` (0 = alpha_0,
    1 = 2 pi - alpha_0), ``way`` (0 = transfer angle < pi, 1 = > pi), ``a``, ``v1``, ``v2``.
    Roots are bracketed on a log-a scan from the minimum-energy a_m (Lagrange's t(a) is not
    monotone on the upper branch for N >= 1, which has two roots or none) and bisected."""
    r1 = np.atleast_2d(np.asarray(r1, float)); r2 = np.atleast_2d(np.asarray(r2, float))
    P = r1.shape[0]
    dt = np.broadcast_to(np.asarray(dt, float), (P,)).copy()
    r1n = np.linalg.norm(r1, axis=1); r2n = np.linalg.norm(r2, axis=1)
    d = r2 - r1
    c = np.linalg.norm(d, axis=1)
    s = 0.5 * (r1n + r2n + c)
    am = 0.5 * s
    with np.errstate(all="ignore"):
        cos_th = np.clip(np.einsum("ij,ij->i", r1, r2) / (r1n * r2n), -1.0, 1.0)
    th_short = np.arccos(cos_th)
    ok = (dt > 0) & (c > 0) & (r1n > 0) & (r2n > 0) & np.isfinite(s)
    Pmin = 2.0 * math.pi * np.sqrt(am ** 3 / mu)
    nmax = np.where(ok, np.minimum(np.floor(dt / Pmin), max_revs), -1).astype(int)
    g = 1.0 + np.geomspace(A_SCAN_X[0], A_SCAN_X[1], A_SCAN_N)

    def tof(a, N, branch, way, sel):
        """Time of flight at semi-major axis `a` (shape (len(sel), K) or (len(sel),))."""
        ss, cc = s[sel], c[sel]
        if a.ndim == 2:
            ss, cc = ss[:, None], cc[:, None]
        with np.errstate(all="ignore"):
            a0 = 2.0 * np.arcsin(np.sqrt(np.clip(ss / (2.0 * a), 0.0, 1.0)))
            b0 = 2.0 * np.arcsin(np.sqrt(np.clip((ss - cc) / (2.0 * a), 0.0, 1.0)))
        al = a0 if branch == 0 else 2.0 * math.pi - a0
        be = b0 if way == 0 else -b0
        return np.sqrt(a ** 3 / mu) * (2.0 * math.pi * N + (al - np.sin(al)) - (be - np.sin(be)))

    out = {k: [] for k in ("pair", "N", "branch", "way", "a")}
    for N in range(0, int(nmax.max(initial=-1)) + 1):
        sel = np.flatnonzero(nmax >= N)
        if sel.size == 0:
            break
        A = am[sel][:, None] * g[None, :]
        for branch in (0, 1):
            if N == 0 and branch == 1:
                pass                     # the upper N=0 branch is the long-time half: keep it
            for way in (0, 1):
                f = tof(A, N, branch, way, sel) - dt[sel][:, None]
                sg = np.sign(f)
                br = np.argwhere((sg[:, :-1] * sg[:, 1:] < 0) & np.isfinite(f[:, :-1]) & np.isfinite(f[:, 1:]))
                if br.size == 0:
                    continue
                rows, cols = br[:, 0], br[:, 1]
                lo = np.log(A[rows, cols]); hi = np.log(A[rows, cols + 1])
                flo = f[rows, cols]
                psel = sel[rows]
                for _ in range(BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    fm = tof(np.exp(mid), N, branch, way, psel) - dt[psel]
                    left = np.sign(fm) == np.sign(flo)
                    lo = np.where(left, mid, lo); flo = np.where(left, fm, flo)
                    hi = np.where(left, hi, mid)
                out["pair"].append(psel); out["a"].append(np.exp(0.5 * (lo + hi)))
                out["N"].append(np.full(psel.size, N)); out["branch"].append(np.full(psel.size, branch))
                out["way"].append(np.full(psel.size, way))
    if not out["pair"]:
        return None
    sol = {k: np.concatenate(v) for k, v in out.items()}
    p, a = sol["pair"], sol["a"]
    with np.errstate(all="ignore"):
        a0 = 2.0 * np.arcsin(np.sqrt(np.clip(s[p] / (2.0 * a), 0.0, 1.0)))
        b0 = 2.0 * np.arcsin(np.sqrt(np.clip((s[p] - c[p]) / (2.0 * a), 0.0, 1.0)))
        al = np.where(sol["branch"] == 0, a0, 2.0 * math.pi - a0)
        be = np.where(sol["way"] == 0, b0, -b0)
        k = np.sqrt(mu / (4.0 * a))
        Acoef = k / np.tan(0.5 * al)
        Bcoef = k / np.tan(0.5 * be)
    u1 = r1[p] / r1n[p][:, None]; u2 = r2[p] / r2n[p][:, None]; uc = d[p] / c[p][:, None]
    sol["v1"] = (Bcoef + Acoef)[:, None] * uc + (Bcoef - Acoef)[:, None] * u1
    sol["v2"] = (Bcoef + Acoef)[:, None] * uc - (Bcoef - Acoef)[:, None] * u2
    good = np.all(np.isfinite(sol["v1"]), axis=1) & np.isfinite(a)
    return {k: v[good] for k, v in sol.items()}


def kepler_states(r0, v0, dts, mu=MU_SUN):
    """Two-body states of M ELLIPTIC orbits (r0, v0: M x 3) after `dts` (M x T days):
    positions and velocities, each (M, T, 3), NaN where an orbit is not elliptic.
    Eccentric-anomaly Newton from the standard 0.85 e starter, vectorised."""
    r0 = np.asarray(r0, float); v0 = np.asarray(v0, float)
    dts = np.asarray(dts, float)
    r0n = np.linalg.norm(r0, axis=1)
    v2 = np.einsum("ij,ij->i", v0, v0)
    with np.errstate(all="ignore"):
        a = 1.0 / (2.0 / r0n - v2 / mu)
        sig = np.einsum("ij,ij->i", r0, v0) / math.sqrt(mu)
        ecosE = 1.0 - r0n / a
        esinE = sig / np.sqrt(a)
        e = np.hypot(ecosE, esinE)
        E0 = np.arctan2(esinE, ecosE)
        n = np.sqrt(mu / a ** 3)
    ell = (a > 0) & (e < 1.0) & np.isfinite(a) & np.isfinite(e)
    a = np.where(ell, a, 1.0); e = np.where(ell, e, 0.0); n = np.where(ell, n, 0.0)
    E0 = np.where(ell, E0, 0.0); esinE = np.where(ell, esinE, 0.0)
    M = (E0 - esinE)[:, None] + n[:, None] * dts
    Mw = np.mod(M + math.pi, 2 * math.pi) - math.pi       # solve on (-pi, pi], unwrap after
    ee = e[:, None]
    E = Mw + 0.85 * ee * np.sign(np.sin(Mw))
    for it in range(KEPLER_ITERS):
        dE = np.clip((E - ee * np.sin(E) - Mw) / (1.0 - ee * np.cos(E)), -1.0, 1.0)
        E = E - dE
        if it % 4 == 3 and not np.any(np.abs(dE) > 1e-12):
            break
    dE = (E + (M - Mw)) - E0[:, None]
    A = a[:, None]
    rn = A + (r0n[:, None] - A) * np.cos(dE) + (sig[:, None] * np.sqrt(A)) * np.sin(dE)
    f = 1.0 - A / r0n[:, None] * (1.0 - np.cos(dE))
    g = dts - np.sqrt(A ** 3 / mu) * (dE - np.sin(dE))
    fd = -np.sqrt(mu * A) / (rn * r0n[:, None]) * np.sin(dE)
    gd = 1.0 - A / rn * (1.0 - np.cos(dE))
    pos = f[..., None] * r0[:, None, :] + g[..., None] * v0[:, None, :]
    vel = fd[..., None] * r0[:, None, :] + gd[..., None] * v0[:, None, :]
    pos[~ell] = np.nan; vel[~ell] = np.nan
    return pos, vel


def _reference(ev, grp):
    """Reference observation of an apparition: the one nearest its time-median."""
    t = ev.times_tdb[grp]
    return int(grp[int(np.argmin(np.abs(t - np.median(t))))])


def _score(ev, r0, v0, t0, score_idx):
    """chi2 of M candidate states (at `t0`) on the observations `score_idx`, two-body with one
    light-time iteration, in the fit's own sigmas. (M,) array; inf where not scorable."""
    tt = ev.times_tdb[score_idx]
    R = np.array([ev.Robs[i] for i in score_idx])                       # (T,3)
    dts = np.broadcast_to(tt[None, :] - t0, (r0.shape[0], tt.size))
    pos, _ = kepler_states(r0, v0, dts)
    rho = np.linalg.norm(pos - R[None], axis=2)
    pos, _ = kepler_states(r0, v0, dts - rho / C_AU_DAY)
    rel = pos - R[None]
    with np.errstate(all="ignore"):
        ra = np.degrees(np.arctan2(rel[..., 1], rel[..., 0]))
        dec = np.degrees(np.arcsin(rel[..., 2] / np.linalg.norm(rel, axis=2)))
    dra = (np.mod(ra - ev.ra[score_idx][None] + 180.0, 360.0) - 180.0) * ev.cosd[score_idx][None] * 3600.0
    ddec = (dec - ev.dec[score_idx][None]) * 3600.0
    chi2 = np.sum((dra / ev.fit_sig_ra[score_idx][None]) ** 2 + (ddec / ev.fit_sig_dec[score_idx][None]) ** 2, axis=1)
    return np.where(np.isfinite(chi2), chi2, np.inf)


def bridge_seeds(ev, groups, max_seeds=MAX_SEEDS):
    """Bound seeds joining the two richest apparitions of `groups` (index arrays into `ev`) over
    every revolution count. ``(t_ref, states, info)``: a common heliocentric epoch (the anchor's
    reference observation time), an (m, 6) array of states there, and one dict per seed (N,
    branch, way, a, two-body chi2). ``(None, None, [])`` when there is nothing to join."""
    if len(groups) < 2:
        return None, None, []
    order = sorted(groups, key=lambda g: (-g.size, -float(np.ptp(ev.times_tdb[g]))))
    gA, gB = order[0], order[1]
    iA, iB = _reference(ev, gA), _reference(ev, gB)
    tA, tB = float(ev.times_tdb[iA]), float(ev.times_tdb[iB])
    if tB == tA:
        return None, None, []
    uA = _unit(ev.ra[iA], ev.dec[iA]); uB = _unit(ev.ra[iB], ev.dec[iB])
    RA, RB = np.asarray(ev.Robs[iA], float), np.asarray(ev.Robs[iB], float)
    rho = np.geomspace(RANGE_GRID_AU[0], RANGE_GRID_AU[1], RANGE_GRID_N)
    ra_, rb_ = np.meshgrid(rho, rho, indexing="ij")
    ra_, rb_ = ra_.ravel(), rb_.ravel()
    # positions at the EMISSION epochs: the Lambert time of flight is between those
    p1 = RA[None] + ra_[:, None] * uA[None]
    p2 = RB[None] + rb_[:, None] * uB[None]
    t1 = tA - ra_ / C_AU_DAY; t2 = tB - rb_ / C_AU_DAY
    early = t1 < t2
    r1 = np.where(early[:, None], p1, p2); r2 = np.where(early[:, None], p2, p1)
    sol = lambert_multirev(r1, r2, np.abs(t2 - t1))
    if sol is None or sol["pair"].size == 0:
        return None, None, []
    p = sol["pair"]
    # the state at the ANCHOR's emission epoch, whichever end of the transfer it is
    isA_first = early[p]
    rA = np.where(isA_first[:, None], r1[p], r2[p])
    vA = np.where(isA_first[:, None], sol["v1"], sol["v2"])
    tAe = t1[p]
    # score everything on a spread of included observations (both apparitions' own included)
    allidx = np.sort(np.concatenate(groups))
    k = min(SCORE_OBS, allidx.size)
    score_idx = allidx[np.unique(np.linspace(0, allidx.size - 1, k).round().astype(int))]
    score_idx = np.union1d(score_idx, [iA, iB])
    # shift every candidate to the common epoch tA (a light-time's worth of two-body motion)
    pos_c, vel_c = kepler_states(rA, vA, (tA - tAe)[:, None])
    pos_c, vel_c = pos_c[:, 0, :], vel_c[:, 0, :]
    chi2 = np.full(p.size, np.inf)
    for s0 in range(0, p.size, 50_000):
        sl = slice(s0, s0 + 50_000)
        chi2[sl] = _score(ev, pos_c[sl], vel_c[sl], tA, score_idx)
    fin = np.isfinite(chi2) & np.all(np.isfinite(pos_c), axis=1) & np.all(np.isfinite(vel_c), axis=1)
    if not np.any(fin):
        return None, None, []
    keys = sol["N"] * 4 + sol["branch"] * 2 + sol["way"]
    order = np.flatnonzero(fin)[np.argsort(chi2[fin], kind="stable")]
    picked, per_key = [], {}
    for j in order:
        kk = int(keys[j])
        if per_key.get(kk, 0) >= SEEDS_PER_KEY:
            continue
        per_key[kk] = per_key.get(kk, 0) + 1
        picked.append(j)
        if len(picked) >= max_seeds:
            break
    states = np.array([np.concatenate([pos_c[j], vel_c[j]]) for j in picked])
    info = [{"N": int(sol["N"][j]), "branch": int(sol["branch"][j]), "way": int(sol["way"][j]),
             "a": float(sol["a"][j]), "rho_anchor": float(ra_[p[j]]), "rho_far": float(rb_[p[j]]),
             "chi2_twobody": float(chi2[j])} for j in picked]
    return tA, states, info
