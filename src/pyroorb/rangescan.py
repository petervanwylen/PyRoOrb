"""Range profile for a short arc: is the line-of-sight distance actually MEASURED?

A tracklet's angles fix four of the six orbital degrees of freedom; what they leave open is
the line-of-sight range and its rate. So the honest question about a one-night orbit is not
"how tight is the covariance ellipsoid" -- the fit can land in a wrong range basin whose local
curvature is perfectly tight, and then the covariance is confidently wrong -- but "over what
range of distances can this astrometry be fitted at all?".

This module answers that directly, by profiling: walk a log grid of trial ranges across the
admissible interval and, at each one, RE-FIT the remaining five degrees of freedom (the two sky
angles at the epoch and the three velocity components). The result is chi2(rho), the profile
likelihood in the one direction that matters, and the set of ranges within Delta-chi2 of its
minimum is the honest answer -- narrow when the range is measured, and a factor of many when
it is not.

Logic openly ported from find_orb (GPL-2.0; no source copied). find_orb has
every piece of this: `find_sr_ranges` (sr.cpp) bounds the admissible interval, `find_trial_orbit`
(orb_func.cpp) builds a transfer at a trial range, and `adjust_herget_results` ->
`extended_orbit_fit(..., FIT_FIXED_DISTANCES)` is a least-squares refit with the distances HELD
-- a profile at one range. What is different here is that the walk is EXHAUSTIVE and ordered
rather than a random draw: a gate must not depend on which samples the RNG happened to take,
and a grid of the same size covers the axis rather than clumping on it.

Two things the physics contributes, without which the profile says every arc is undetermined:

  * an UNBOUND solution fits a one-hour arc trivially (over an hour a distant hyperbolic body
    draws the same straight line through the data), so a pure chi2 profile is flat toward large
    rho for every tracklet ever taken. What excludes those is not the astrometry, it is that a
    solar-system object is bound -- Milani's admissible region; find_orb's score does the same
    job through its e > 1.01 penalty. Carried here as a smooth barrier residual (so the
    least-squares walk is steered rather than falling off a cliff it cannot climb back), and as
    a per-node `bound` flag the interval reader drops.
  * a sun-grazing trial is refused outright: not because it fits badly but because integrating
    it is where IAS15 shrinks its step to nothing.

The sigmas are the observations' OWN, never rescaled to make the best fit come out at
chi2/dof = 1. Rescaling would be the covariance's overconfidence by another door: a five-point
tracklet that happens to fit at chi2/dof = 0.06 would earn an interval 16x too tight, which is
exactly the failure this module exists to catch.
"""
import math

import numpy as np
from astropy.time import Time
from scipy.optimize import least_squares

import pyroorb.sr as _sr
import pyroorb.obs_sources as _os
from pyroorb.forward_model import predict_radec_batch, MU_SUN, los_unit
from pyroorb.orbit_common import radec_residuals_arcsec
from pyroorb.twobody import quick_aeq
from pyroorb.iod import build_base_sim

# --- the scan ------------------------------------------------------------------------------
N_RHO = 17                 # log-spaced trial ranges across the admissible interval
N_FINE = 11                # ...and a second, fine pass bracketing the minimum (see `profile`)
MAX_ZOOM = 6               # further fine passes while the minimum's valley holds < ZOOM_NODES nodes
ZOOM_NODES = 3             # ...the fewest nodes inside min + DCHI2 that describe a valley's width
N_V = 9                    # cold range-rate seeds per node (find_orb's vparam in [-1, 1])
MAX_PROFILE_OBS = 10       # observations used per node; a dense tracklet is thinned in TIME
Q_MIN_AU = 0.03            # a trial closer to the Sun than this is not scored (integration cost)
BOUND_BARRIER = 1000.0     # weight of the |v|/v_escape - 1 barrier residual
BOUND_TOL = 1e-3           # ...and how far past escape still counts as bound (see `profile`)

# --- the verdict ---------------------------------------------------------------------------
DCHI2 = 9.0                # 3-sigma in one parameter: the allowed set is chi2 <= min + this
MAX_REL_WIDTH = 0.5        # allowed interval, as a fraction of the best range: wider = not measured
MAX_CHI2_PER_DOF = 4.0     # ...and SOMETHING has to fit, or "narrow" only means "uniformly bad"


def _thin(sub, times, rhohats, Robs, sig, keep=MAX_PROFILE_OBS):
    """Evenly-in-time subset, endpoints always kept. The range signal is the parallax /
    curvature signature ACROSS the tracklet, not the number of points on it: 51 observations of
    2004 FH cost 50 s of profile and say what 12 say in 4."""
    n = len(sub)
    if n <= keep:
        return sub, times, rhohats, Robs, sig
    idx = sorted(set(np.linspace(0, n - 1, keep).round().astype(int).tolist()))
    return [sub[i] for i in idx], times[idx], rhohats[idx], Robs[idx], sig[idx]


def profile(sub, sigmas, n_rho=N_RHO, n_v=N_V, n_fine=N_FINE, r_max=_sr.SR_MAX_DIST,
            seed_state=None):
    """chi2 against `sub` (parsed observation dicts) as a function of trial range.

    Returns ``{"rows": [{rho, chi2, a, e, q, bound, state}], "r_lo", "r_hi", "r_max", "capped",
    "n_obs", "dof"}``. When there is nothing to profile the same dict comes back with NO rows
    and a `failure` sentence saying why (see `failed`) -- never None, because to the caller a
    profile that could not be run and a profile that found nothing wrong must not look alike.
    `capped` says the admissible interval ran into `r_max`, the scan's own outer
    limit, rather than closing on the escape geometry.

    `seed_state` is a known solution -- the delivered fit's heliocentric state at the epoch of
    the first observation in `sub`, TDB. Its distance joins the grid and its state seeds that
    node, so the profile can never report a minimum WORSE than an orbit already in hand. Without
    it the profile's minimum is only as good as its grid, and a node that misses a narrow valley
    reads as a valley somewhere else (see the zoom below)."""
    times = np.array([o["jd_utc"] for o in sub])
    rhohats = np.array([los_unit(o["ra_deg"], o["dec_deg"]) for o in sub])
    Robs = np.array(list(_os.observer_positions_batch(
        [o["jd_utc"] for o in sub], [o["obscode"] for o in sub], _os.load_obscodes())))
    sig = np.asarray(sigmas, float)
    sub, times, rhohats, Robs, sig = _thin(sub, times, rhohats, Robs, sig)

    t0 = Time(times[0], format="jd", scale="utc").tdb.jd
    base = build_base_sim(t0)
    tt = Time(times, format="jd", scale="utc").tdb.jd
    ra = np.array([o["ra_deg"] for o in sub])
    dec = np.array([o["dec_deg"] for o in sub])
    cosd = np.cos(np.radians(dec))
    R0, L0, Rl, Ll = Robs[0], rhohats[0], Robs[-1], rhohats[-1]
    dt = float(tt[-1] - tt[0])
    if abs(dt) < 1e-6:
        return failed("the observations span no time", n_obs=len(sub))
    r_floor = _sr._minimum_sr_distance(max(float(times[-1] - times[0]), 1.0 / 24.0))
    r_lo, r_hi = _sr._admissible_range(R0, L0, Rl, Ll, dt, r_floor, r_max)
    if r_hi is None:
        return failed(f"no bound orbit reaches this arc from any distance under {r_max:g} au",
                      n_obs=len(sub), r_lo=r_lo, r_max=r_max)
    n_resid = 2 * len(sub) + 1

    def bound_excess(st):
        """|v| / v_escape - 1: positive when the trial is unbound (see the module docstring)."""
        r = float(np.linalg.norm(st[:3]))
        if r <= 0:
            return 0.0
        return float(np.linalg.norm(st[3:])) / math.sqrt(2.0 * MU_SUN / r) - 1.0

    def resid(st):
        a, e, q = quick_aeq(st)
        if not math.isfinite(q) or q < Q_MIN_AU:
            return None
        try:
            pred = predict_radec_batch(base, t0, st, tt, Robs)
        except Exception:
            return None
        dra, ddec = radec_residuals_arcsec(pred, ra, dec, cosd)
        r = np.concatenate([dra / sig[:, 0], ddec / sig[:, 1],
                            [BOUND_BARRIER * max(0.0, bound_excess(st))]])
        return r if np.all(np.isfinite(r)) else None

    def state_at(rho, p):
        d = dec[0] + p[1] / 3600.0
        a_ = ra[0] + p[0] / 3600.0 / max(math.cos(math.radians(dec[0])), 1e-6)
        return np.concatenate([R0 + rho * los_unit(a_, d), p[2:]])

    def fit_at(rho, p0):
        def f(p):
            r = resid(state_at(rho, p))
            return r if r is not None else np.full(n_resid, 1e3)
        try:
            # trf + x_scale="jac": two sky offsets in ARCSEC beside three velocities in au/day
            # are four orders of magnitude apart, and an unscaled LM walks that badly.
            sol = least_squares(f, np.asarray(p0, float), method="trf", x_scale="jac",
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
        except Exception:
            return None
        c = float(np.sum(sol.fun ** 2))
        return (c, np.asarray(sol.x, float)) if math.isfinite(c) else None

    def cold(rho):
        """Best bare transfer at this range over the range-rate freedom (find_trial_orbit)."""
        best = None
        for v in np.linspace(-1.0, 1.0, n_v):
            st = _sr._trial_orbit(R0, L0, Rl, Ll, rho, float(v), dt, 0.0)
            if st is None:
                continue
            r = resid(st)
            if r is None:
                continue
            c = float(np.sum(r * r))
            if best is None or c < best[0]:
                best = (c, np.concatenate([[0.0, 0.0], st[3:]]))
        return best

    grid = [float(x) for x in np.geomspace(r_lo, r_hi, n_rho)]
    best_at = {}

    # The known solution, as this profile's own parameters at its own distance: sky offsets of
    # its line of sight from the first observation (arcsec) and its velocity.
    seed_at = None
    if seed_state is not None:
        st = np.asarray(seed_state, float)
        los = st[:3] - R0
        rho_s = float(np.linalg.norm(los))
        if math.isfinite(rho_s) and r_lo <= rho_s <= r_hi:
            u = los / rho_s
            ra_s = math.degrees(math.atan2(u[1], u[0])) % 360.0
            dec_s = math.degrees(math.asin(max(-1.0, min(1.0, u[2]))))
            dra = ((ra_s - ra[0] + 180.0) % 360.0) - 180.0
            seed_at = (rho_s, np.concatenate([[dra * 3600.0 * math.cos(math.radians(dec[0])),
                                               (dec_s - dec[0]) * 3600.0], st[3:]]))
            grid = sorted(set(grid) | {rho_s})

    def sweep(order):
        """One pass along the grid. Every node's least-squares starts from the BETTER of a cold
        transfer seed and its neighbour's solution -- never the neighbour alone: a continuation
        that loses the valley walks a smooth, entirely fictitious plateau afterwards, and a
        smooth chi2 is the one thing this profile is read for. Both directions are swept and the
        per-node minimum kept, so no node inherits one pass's bad luck."""
        warm = None
        for rho in order:
            seeds = []
            c0 = cold(rho)
            if c0 is not None:
                seeds.append(c0[1])
            if warm is not None:
                seeds.append(warm)
            if seed_at is not None and rho == seed_at[0]:
                seeds.append(seed_at[1])
            got = None
            for p0 in seeds:
                r = fit_at(rho, p0)
                if r is not None and (got is None or r[0] < got[0]):
                    got = r
            if got is None:
                got = c0
                warm = None
            else:
                warm = got[1]
            if got is None:
                continue
            if rho not in best_at or got[0] < best_at[rho][0]:
                best_at[rho] = got

    sweep(grid)
    sweep(list(reversed(grid)))

    # REFINE around the minimum. The coarse grid spans the whole admissible interval in 25 log
    # steps -- adjacent nodes differ by ~50% -- so any genuinely measured range collapses to a
    # single node and its width is reported as zero, which is the grid's resolution and not the
    # data's. A determined arc is exactly the case that needs the fine steps, so bracket the
    # minimum with its two neighbours and walk that decade properly.
    #
    # ...and KEEP walking while the valley holds fewer than ZOOM_NODES nodes. One fine pass
    # steps ~8% at a close approach, and a two-station tracklet of an object at 0.008 au has a
    # valley +/-0.4% wide: one node lands on its WALL, the true minimum is never sampled, and
    # the verdict then reads "the delivered orbit sits outside the distances the data support"
    # for a fit that is the better solution on the profile's own observations (C46MAH1,
    # C46MDH1, C46M9D1 on the 2026-10-02 NEOCP -- thinned chi2 10.5 at the fit's distance
    # against 17.0 at the node). Each pass brackets the current minimum by its neighbours, so
    # the step shrinks ~5x per pass; a valley the coarse grid already resolves never zooms --
    # and neither does a profile whose minimum FITS NOWHERE: its "valley" is one node at any
    # resolution and the verdict ("no orbit fits this arc at any distance") cannot change, so
    # every pass is wasted (PAIR_2016FS162, a mislinked tracklet: 62 s -> 243 s unguarded).
    dof = max(1, 2 * len(sub) - 6)
    for level in range(1 + MAX_ZOOM):
        if len(best_at) < 3:
            break
        have = sorted(best_at)
        lo_c = min(have, key=lambda r: best_at[r][0])
        if level > 0:
            if best_at[lo_c][0] / dof > MAX_CHI2_PER_DOF:
                break
            lim = best_at[lo_c][0] + DCHI2
            if sum(1 for r in have if best_at[r][0] <= lim) >= ZOOM_NODES:
                break
        i = have.index(lo_c)
        lo_b = have[max(0, i - 1)]
        hi_b = have[min(len(have) - 1, i + 1)]
        if not hi_b > lo_b * (1.0 + 1e-9):
            break
        fine = [float(x) for x in np.geomspace(lo_b, hi_b, n_fine) if float(x) not in best_at]
        if not fine:
            break
        sweep(fine)
        sweep(list(reversed(fine)))
        grid = sorted(set(grid) | set(fine))

    rows = []
    for rho in grid:
        got = best_at.get(rho)
        if got is None:
            continue
        st = state_at(rho, got[1])
        a, e, q = quick_aeq(st)
        rows.append({"rho": rho, "chi2": got[0], "a": a, "e": e, "q": q,
                     # <= the escape speed, not < it: the barrier holds a node whose free
                     # optimum would be unbound EXACTLY at the boundary, and that solution is
                     # the best bound fit at this distance -- the thing the interval is asking
                     # about. Read as unbound, a 5-hour main-belt tracklet reported "the
                     # delivered orbit sits outside the distances the data support" for a fit
                     # that was right to 0.2%, because its own distance had been filtered out.
                     # The tolerance is in speed, i.e. ~e = 1.001: the profile's subject is the
                     # DISTANCE, and a solution a tenth of a percent past escape is not a
                     # different answer to that question.
                     "bound": bool(bound_excess(st) <= BOUND_TOL), "state": st})
    if not rows:
        # Every node's seeding failed -- 1998 VD32's 7 observations over 8 minutes: the Lambert
        # transfer that seeds each node degenerates over so short a span (every trial is None
        # or plunges through the Sun), though the delivered orbit fits the tracklet at 0.33".
        # A statement about the seeding, not about the data; the caller must not read it as
        # either "determined" or "nothing to object to".
        return failed(f"no trial orbit could be fitted at any of the {len(grid)} distances "
                      f"scanned between {r_lo:.2g} and {r_hi:.3g} au",
                      n_obs=len(sub), r_lo=r_lo, r_hi=r_hi, r_max=r_max)
    return {"rows": rows, "r_lo": r_lo, "r_hi": r_hi, "r_max": r_max,
            "capped": bool(r_hi >= r_max * 0.999), "n_obs": len(sub),
            "dof": max(1, 2 * len(sub) - 6),
            # Where the node states live: heliocentric equatorial at t0 (TDB), seen from R0.
            # popprior needs both -- the line of sight to put a radial velocity on, and the
            # observer's distance from the Sun for the magnitude -> H reduction.
            "t0_tdb": float(t0), "R0": [float(x) for x in R0]}


def failed(why, n_obs=0, r_lo=None, r_hi=None, r_max=_sr.SR_MAX_DIST):
    """A profile that could not be run, in the profile's own shape: no rows, and `failure`
    saying why in the words `verdict` will report. One constructor so that every exit that
    cannot judge -- inside `profile` and in the callers that never reach it (too few
    observations, an exception) -- is the same kind of object."""
    return {"rows": [], "failure": str(why), "r_lo": r_lo, "r_hi": r_hi, "r_max": r_max,
            "capped": False, "n_obs": int(n_obs), "dof": max(1, 2 * int(n_obs) - 6)}


def allowed(prof, dchi2=DCHI2, bound_only=True):
    """``([(lo, hi), ...], chi2_min)`` -- the connected range intervals within `dchi2` of the
    profile minimum. Unbound nodes are dropped by default: they fit a short arc trivially and
    are excluded by physics, not by the data."""
    rows = [r for r in prof["rows"] if r.get("bound", True)] if bound_only else prof["rows"]
    if not rows:
        rows = prof["rows"]
    c = np.array([r["chi2"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    lim = c.min() + dchi2
    ok = c <= lim

    def cross(i, j):
        """Where chi2 crosses `lim` between nodes i and j, linear in (log rho, chi2). The grid
        is coarse by design -- ending the interval AT the first node that fails puts the edge
        up to a whole log step out, which on a determined arc is most of the width."""
        c0, c1 = float(c[i]), float(c[j])
        if not math.isfinite(c0) or not math.isfinite(c1) or c1 == c0:
            return float(rho[i])
        t = (lim - c0) / (c1 - c0)
        t = min(1.0, max(0.0, t))
        return float(math.exp(math.log(rho[i]) + t * (math.log(rho[j]) - math.log(rho[i]))))

    out, start = [], None
    for i, f in enumerate(ok):
        if f and start is None:
            start = float(rho[i]) if i == 0 else cross(i, i - 1)
        if start is not None and not f:
            out.append((start, cross(i - 1, i)))
            start = None
    if start is not None:
        out.append((start, float(rho[-1])))
    return out, float(c.min())


def verdict(prof, rho_fit=None):
    """Does this arc MEASURE the range? ``{"determined": bool, "reason": str, ...}``.

    Four things have to hold, and each of them caught a real case in the calibration:

      * ONE interval. Two separated distance solutions is the aliasing an orbit cannot be
        published through, and it is what a local covariance structurally cannot see.
      * NARROW -- the interval, as a fraction of the best range, under MAX_REL_WIDTH.
      * SOMETHING FITS: chi2/dof at the minimum under MAX_CHI2_PER_DOF. Without this, a profile
        that fits NOWHERE reports a single node inside Delta-chi2 and reads as width zero --
        "uniformly terrible" wearing the costume of "sharply determined" (107474, 162385).
      * NOT AT THE EDGE of the admissible interval: an interval that runs into the boundary is
        one the scan could not close, not one the data closed (13923, off by 281%).

    `rho_fit` (the delivered orbit's own geocentric range) is checked when given: a determined
    profile whose interval does not contain the fit means the fit is in a different basin from
    the one the data support, which is a finding about the fit, not a licence to show it.

    A profile with no rows (`failed`) is a fifth outcome and gets its own verdict: NOT
    determined, `profiled` False, the reason being the failure itself. "Could not measure the
    distance" is a reason to withhold the orbit exactly as "measured it and it is free" is; what
    it must never become is None, which the caller would read as "no objection" and fall back
    on the covariance -- the very method this gate replaced.
    """
    if not prof.get("rows"):
        return {"determined": False, "profiled": False, "rho_lo": None, "rho_hi": None,
                "rho_best": None, "rel_width": None, "n_intervals": 0, "edge": None,
                "chi2_min": None, "chi2_per_dof": None, "dof": prof.get("dof"),
                "n_obs": prof.get("n_obs"), "a_best": None, "e_best": None,
                "capped": False,
                "reason": ("the distance could not be measured: "
                           + str(prof.get("failure") or "the range profile did not run"))}
    ivs, cmin = allowed(prof)
    lo = min(a for a, _ in ivs)
    hi = max(b for _, b in ivs)
    rows = [r for r in prof["rows"] if r.get("bound", True)] or prof["rows"]
    best = min(rows, key=lambda r: r["chi2"])
    rel = (hi - lo) / max(best["rho"], 1e-12)
    at_floor = bool(lo <= prof["r_lo"] * 1.001)
    at_ceiling = bool(hi >= prof["r_hi"] * 0.999)
    edge = at_floor or at_ceiling
    capped = bool(prof.get("capped")) and at_ceiling
    out = {"determined": False, "profiled": True, "rho_lo": lo, "rho_hi": hi,
           "rho_best": best["rho"], "rel_width": rel, "n_intervals": len(ivs), "edge": edge,
           "capped": capped, "chi2_min": cmin, "chi2_per_dof": cmin / prof["dof"],
           "dof": prof["dof"], "n_obs": prof["n_obs"],
           "a_best": best["a"], "e_best": best["e"], "reason": None}
    if len(ivs) > 1:
        out["reason"] = f"{len(ivs)} separated distance solutions fit this arc"
    elif cmin / prof["dof"] > MAX_CHI2_PER_DOF:
        out["reason"] = "no orbit fits this arc at any distance"
    elif edge:
        # Say which edge, and whose. The outer limit is the SCAN's (r_max, a sampling constant
        # for the NEO/MBA regime) far more often than it is physics: 95 of 187 fixture
        # tracklets run into the 6 au cap, nearly all of them at 0.01-1 au. Only when the
        # admissible interval closed short of the cap did a bound orbit's reach close it -- and
        # even then the escape test is noise-blind, so "physically possible" was never the
        # right claim. The sentence names the limit it actually hit.
        if capped:
            out["reason"] = (f"the fitting distances run out to the {prof['r_max']:g} au limit "
                             f"of the scan, so the data did not close the interval")
        elif at_ceiling:
            out["reason"] = (f"the fitting distances run to the outer edge of where a bound "
                             f"orbit can reach this arc ({prof['r_hi']:.3g} au)")
        else:
            out["reason"] = (f"the fitting distances run down to the inner edge of the scan "
                             f"({prof['r_lo']:.2g} au)")
    elif rel > MAX_REL_WIDTH:
        out["reason"] = (f"distances from {lo:.4g} to {hi:.4g} au fit equally well "
                         f"({rel:.0f}x the range itself)" if rel >= 1 else
                         f"the distance is only pinned to +/-{100 * rel / 2:.0f}%")
    elif rho_fit is not None and not (lo <= rho_fit <= hi):
        out["reason"] = "the delivered orbit sits outside the distances the data support"
    else:
        out["determined"] = True
    return out
