"""Post-fit basin audit: notice a single-apparition fit that converged into a WRONG basin, and
climb out of it.

Why this exists. Rejection makes a wrong minimum self-consistent: once the observations that
contradict it are thrown out, the included residuals look acceptable and no later stage asks
again. On a multi-apparition arc the recovery cascade notices a stranded or discordant
apparition; a single apparition has no such symptom, so a fit seeded into the wrong range
basin used to be delivered as it was. The case that showed it: a NEOCP object (40 obs over
2.2 days, a retrograde near-parabolic orbit, q 0.67 au, i 124 deg) whose seed was a prograde
Atira; the fit converged there, rejected the whole discovery night at ~100 sigma and delivered
27/40 at 3.9" with nothing flagged. The right orbit keeps 35/40 at 0.64".

Why a range scan and not a random shake. The wrong and right orbits of a short arc differ
mainly in topocentric RANGE (and range-rate) along the same lines of sight: far apart in
elements, nearly identical on the sky. Perturbing the state at random almost never crosses to
the other basin; starts spread along the range axis do. Measured on the regression fixtures
with a wrong-start harness: every wrong basin a plain fit converged to was reached only from
seeds at small range (<~1 au geocentric), and seeds at larger range, or from heliocentric
distances 1.1-5 au, came back right.

How. Gated on symptoms only a converged fit shows (`triggered`): normalized rms well above 1,
a large rejected fraction, or a whole tracklet thrown out far beyond any noise. Then:
  1. range-spread seeds (Herget at fixed geocentric ranges, Vaisala at fixed heliocentric
     distances) over the IOD's geometric sub-arc, each corrected by a short ROBUST
     least-squares (Cauchy loss on the normalized residuals) over ALL observations, rejected
     ones included -- so a few blunders cannot dominate and the comparison is not rigged in
     favour of whichever basin did the rejecting;
  2. the delivered state corrected the same way, and the distinct basins whose robust cost
     clearly beats the delivered orbit's, are fitted for real -- full correction from the obs
     the robust state fits, plus the ordinary reject/reclaim rounds, so a genuinely bad night
     stays rejected under the new basin;
  3. a candidate is adopted only when it keeps a strict majority of the arc AND clearly more
     observations at a comparable normalized rms (or the delivered fit is catastrophic and
     the candidate is not); the richest adoptable candidate wins.
The bound/unbound decision stays with the fit's own later rules: an unbound seed is handed on
just inside the parabola, as the IOD does.

PYROORB_BASIN_AUDIT=0 turns it off."""
import math
import os

import numpy as np
from scipy.optimize import least_squares

from pyroorb.iod import SEED_RERANK_MAX_ARC
from pyroorb.orbit_common import OPP_GAP_DAYS
from pyroorb.twobody import quick_aeq

# --- trigger ------------------------------------------------------------------------------------
TRIGGER_RMS_N = 1.5          # normalized rms of the included obs
TRIGGER_REJ_FRAC = 0.2       # rejected fraction of the arc
TRIGGER_TRACKLET_SIGMA = 7.0  # a whole tracklet (>= 2 obs, one station, one night) rejected,
#                              every obs beyond this many sigma -- short arcs only (`triggered`)
TRACKLET_GAP_DAYS = 0.3      # same-station obs further apart than this start a new tracklet

# --- scan ---------------------------------------------------------------------------------------
RHO_GEO = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)   # Herget seeds, geocentric au
R_HELIO = (1.1, 1.6, 2.4, 3.5, 5.0)                 # Vaisala seeds, heliocentric au
Q_MIN_AU = 0.03              # a seed plunging closer to the Sun is not tried
ROBUST_SCALE = 3.0           # Cauchy scale, in sigmas
ROBUST_NFEV = 60             # robust correction budget per seed
COST_GAIN = 0.9              # a basin is fitted for real only below this x the delivered cost...
COST_MIN_DROP = 2.0          # ...and at least this much below it (Cauchy units)
MAX_CANDIDATES = 3           # distinct basins fitted for real, best robust cost first
HANDOFF_SIGMA = 3.0 * ROBUST_SCALE  # a candidate's real fit starts from the obs within this
#                              many sigma of its robust state (beyond it the Cauchy loss has
#                              already discounted them)
SAME_Q_REL = 0.05            # two corrected states with q this close...
SAME_POLE_DEG = 3.0          # ...and orbit poles this close are the same basin

# --- adoption -----------------------------------------------------------------------------------
ADOPT_MIN_GAIN = 3           # obs: a candidate must keep at least this many more...
ADOPT_MIN_GAIN_FRAC = 0.05   # ...and at least this fraction more
ADOPT_RMS_N_RATIO = 1.2      # ...at a normalized rms no worse than this x the delivered one


def enabled():
    return os.environ.get("PYROORB_BASIN_AUDIT", "1") != "0"


def single_apparition(times):
    t = np.sort(np.asarray(times, float))
    return t.size < 2 or float(np.max(np.diff(t))) <= OPP_GAP_DAYS


def _tracklets(rows):
    """Index groups of the residual rows: one station, consecutive obs within
    TRACKLET_GAP_DAYS."""
    by = {}
    for k, r in enumerate(rows):
        by.setdefault(r["obscode"], []).append(k)
    out = []
    for ks in by.values():
        ks = sorted(ks, key=lambda k: rows[k]["jd_utc"])
        cur = [ks[0]]
        for a, b in zip(ks, ks[1:]):
            if rows[b]["jd_utc"] - rows[a]["jd_utc"] > TRACKLET_GAP_DAYS:
                out.append(cur)
                cur = []
            cur.append(b)
        out.append(cur)
    return out


def symptoms(fit):
    """The trigger's inputs, read off a converged fit: ``{rms_n, rej_frac, tracklet_sigma}``,
    the last being the smallest |residual| (sigmas) of the worst whole-rejected tracklet of
    >= 2 obs (0 when none is)."""
    rows = fit.residuals()
    n = len(rows)
    rej = sum(1 for r in rows if not r["included"])
    worst = 0.0
    for g in _tracklets(rows):
        if len(g) >= 2 and all(not rows[k]["included"] for k in g):
            m = min(rows[k].get("resid_sigmas", 0.0) for k in g)
            if math.isfinite(m):
                worst = max(worst, m)
    t = [r["jd_utc"] for r in rows]
    return {"rms_n": float(fit.rms_normalized()), "rej_frac": rej / max(n, 1),
            "tracklet_sigma": worst, "span_days": float(max(t) - min(t)) if t else 0.0}


def triggered(fit):
    """The reasons this fit should be audited (empty = leave it alone)."""
    s = symptoms(fit)
    why = []
    if not math.isfinite(s["rms_n"]) or s["rms_n"] > TRIGGER_RMS_N:
        why.append(f"rms_n {s['rms_n']:.2f}")
    if s["rej_frac"] > TRIGGER_REJ_FRAC:
        why.append(f"{100 * s['rej_frac']:.0f}% rejected")
    # A whole rejected tracklet is read only on a SHORT arc -- the IOD's own range-ambiguity
    # bound. A wrong basin that hides behind one rejected tracklet at a few sigma (P/2026 U1's
    # southern station at 7.3) is a range-ambiguity symptom, and range ambiguity is a short-arc
    # phenomenon; on the fixtures every wrong basin over a long arc was catastrophic (caught by
    # rms_n / the rejected fraction), while right fits reject a whole tracklet at 7.0-7.8 sigma
    # as ordinary noise -- a 128-day comet among them, where the audit costs 3 s for nothing.
    if (s["tracklet_sigma"] > TRIGGER_TRACKLET_SIGMA
            and s.get("span_days", 0.0) <= SEED_RERANK_MAX_ARC):
        why.append(f"a whole tracklet rejected at >= {s['tracklet_sigma']:.0f} sigma")
    return why


# --- seeds and robust correction ----------------------------------------------------------------
def range_seeds(obs, robs, t0_tdb, rho_geo=RHO_GEO, r_helio=R_HELIO):
    """``[(label, state_at_t0)]``: two-body transfers between the ends of the IOD's geometric
    sub-arc with both ends at a fixed geocentric range (Herget) or heliocentric distance
    (Vaisala), carried to `t0_tdb` and handed on bound."""
    from pyroorb.iod import (select_subarc, _pos_at_distance, lambert, kepler_propagate,
                             _utc_to_tdb, _inside_parabola)
    from pyroorb.forward_model import los_unit
    times = np.array([o["jd_utc"] for o in obs])
    rh = [los_unit(o["ra_deg"], o["dec_deg"]) for o in obs]
    s, e = select_subarc(times, rh)
    if e <= s:
        s, e = 0, len(obs) - 1
    if e <= s:
        return []
    t1, t2 = _utc_to_tdb(times[s]), _utc_to_tdb(times[e])
    out = []
    for from_sun, grid, tag in ((False, rho_geo, "herget"), (True, r_helio, "vaisala")):
        for d in grid:
            r1 = _pos_at_distance(robs[s], rh[s], d, from_sun)
            r2 = _pos_at_distance(robs[e], rh[e], d, from_sun)
            if r1 is None or r2 is None:
                continue
            try:
                v1 = lambert(r1, r2, t2 - t1)
                if v1 is None:
                    continue
                st = kepler_propagate(np.concatenate([r1, v1]), t0_tdb - t1)
            except Exception:
                continue
            if st is None or not np.all(np.isfinite(st)):
                continue
            q = quick_aeq(st)[2]
            if not (math.isfinite(q) and q >= Q_MIN_AU):
                continue
            out.append((f"{tag} {d:g} au", np.asarray(_inside_parabola(st), float)))
    return out


def robust_cost(ev, state):
    """Cauchy cost of `state` over EVERY observation of the evaluator (sigma-normalized)."""
    dra, ddec = ev.resid_arcsec(np.asarray(state, float))
    z = np.concatenate([dra / ev.sig_ra, ddec / ev.sig_dec]) / ROBUST_SCALE
    z = z[np.isfinite(z)]
    return float(np.sum(np.log1p(z * z)))


def _cauchy(r):
    """Residuals whose sum of squares is ROBUST_SCALE^2 x the Cauchy cost, and d(out)/d(r).
    Done by hand rather than through least_squares(loss=...) so the bound barrier row below
    stays a plain quadratic: a robust loss would flatten the barrier too."""
    c = ROBUST_SCALE
    u = r / c
    l1 = np.log1p(u * u)
    g = np.sign(u) * c * np.sqrt(l1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.abs(u) / ((1.0 + u * u) * np.sqrt(l1))
    d = np.where(np.abs(u) < 1e-6, 1.0, d)
    return g, d


def _bound_row(state):
    """The bound barrier (rangescan's): BOUND_BARRIER x (|v| / v_escape - 1) when positive, and
    its gradient. Without it a free correction over a short arc runs off to an unbound straight
    line, which fits a few nights trivially and is excluded by physics, not by the data."""
    from pyroorb.forward_model import MU_SUN
    from pyroorb.rangescan import BOUND_BARRIER
    r = np.asarray(state[:3], float)
    v = np.asarray(state[3:6], float)
    rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    if rn <= 0 or vn <= 0:
        return 0.0, np.zeros(6)
    k = math.sqrt(rn / (2.0 * MU_SUN))
    x = vn * k - 1.0
    if x <= 0:
        return 0.0, np.zeros(6)
    grad = np.concatenate([vn * k / (2.0 * rn) * r / rn, k * v / vn])
    return BOUND_BARRIER * x, BOUND_BARRIER * grad


def robust_correct(ev, seed, max_nfev=ROBUST_NFEV):
    """A short Cauchy-loss correction of `seed` over all observations (thinned as the fit
    thins), held bound. Returns the corrected state, or None."""
    from pyroorb.arc_extend import _radec_resid, _radec_jac, _clamp
    idx = ev._thin(np.arange(ev.n))
    tt = ev.times_tdb[idx]
    Rs = [ev.Robs[i] for i in idx]
    ra, dec, cosd = ev.ra[idx], ev.dec[idx], ev.cosd[idx]
    sra, sdec = ev.sig_ra[idx], ev.sig_dec[idx]
    seed = np.asarray(seed, float)
    cache = {}

    def comp(s):
        k = s.tobytes()
        if k not in cache:
            cache.clear()
            if ev._predict_jac is not None:
                pred, J = ev._predict_jac(ev.base_sim, ev.t0, s, tt, Rs)
                cache[k] = (pred, _radec_jac(J, cosd, sra, sdec))
            else:
                cache[k] = (ev._predict(ev.base_sim, ev.t0, s, tt, Rs), None)
        return cache[k]

    def fun(s):
        r = _clamp(_radec_resid(comp(s)[0], ra, dec, cosd, sra, sdec))
        g, _ = _cauchy(r)
        return np.concatenate([g, [_bound_row(s)[0]]])

    def jac(s):
        r = _clamp(_radec_resid(comp(s)[0], ra, dec, cosd, sra, sdec))
        _, d = _cauchy(r)
        return np.vstack([comp(s)[1] * d[:, None], _bound_row(s)[1][None, :]])
    try:
        r = least_squares(fun, seed, jac=(jac if ev._predict_jac is not None else "2-point"),
                          method="trf", x_scale="jac", max_nfev=max_nfev)
    except Exception:
        return None
    return r.x if np.all(np.isfinite(r.x)) else None


def _pole(state):
    h = np.cross(state[:3], state[3:])
    n = float(np.linalg.norm(h))
    return h / n if n > 0 else h


def same_basin(s1, s2):
    q1, q2 = quick_aeq(s1)[2], quick_aeq(s2)[2]
    if not (math.isfinite(q1) and math.isfinite(q2)) or abs(q1 - q2) > SAME_Q_REL * max(q1, q2):
        return False
    c = float(np.clip(np.dot(_pole(s1), _pole(s2)), -1.0, 1.0))
    return math.degrees(math.acos(c)) <= SAME_POLE_DEG


def adopt(cand, cur):
    """Whether `cand` replaces the delivered fit `cur`.

    Never a candidate that keeps only half the arc or less: a robust fit is answerable for a
    MAJORITY of the observations (its breakdown point), and below that a short arc's few kept
    obs fit almost anything -- an 8-obs arc with one tracklet shifted 30" offered 4/8 at 0.13"
    that kept both shifted obs and dropped four good ones."""
    from pyroorb.fit_object import CATASTROPHIC_RMS_N
    if 2 * cand.n_observations() <= len(cand._obs):
        return False
    rn_c, rn_d = cand.rms_normalized(), cur.rms_normalized()
    if not math.isfinite(rn_c) or rn_c > CATASTROPHIC_RMS_N:
        return False
    if not math.isfinite(rn_d) or rn_d > CATASTROPHIC_RMS_N:
        return True                     # the delivered fit is not an orbit; this one is
    n_c, n_d = cand.n_observations(), cur.n_observations()
    need = max(ADOPT_MIN_GAIN, math.ceil(ADOPT_MIN_GAIN_FRAC * n_d))
    return n_c >= n_d + need and rn_c <= ADOPT_RMS_N_RATIO * rn_d


# --- the audit ----------------------------------------------------------------------------------
def audit(fit, verbose=False):
    """Return `fit`, or a better-basin fit found by the range scan (see the module docstring).
    Never raises: an audit that cannot run leaves the fit as it was."""
    try:
        if not enabled():
            return fit
        ctx = getattr(fit, "_ctx", None)
        if not isinstance(ctx, dict) or ctx.get("ev") is None:
            return fit
        obs = fit._obs
        if not single_apparition([o["jd_utc"] for o in obs]):
            return fit
        why = triggered(fit)
        if not why:
            return fit
        return _audit(fit, ctx, why, verbose)
    except Exception as ex:
        if verbose:
            print(f"  basin audit skipped: {type(ex).__name__}: {ex}")
        return fit


def _audit(fit, ctx, why, verbose):
    from pyroorb.fit_object import _reject_settle, _build_orbit, _hsnap
    ev = ctx["ev"]
    cur_state = np.asarray(fit._state, float)[:6]
    # The delivered state, robustly corrected on EVERY observation, is the first candidate: the
    # rejection rounds can pin a fit where its few included obs agree while all the observations
    # together pull elsewhere -- to another basin (2017 SN33 from a near-Earth start, 68/235 at
    # 246") or to the right one with the mask left behind (a 9/80 fit in the right basin).
    # Range seeds must then beat the better of the two costs.
    cur_cost = robust_cost(ev, cur_state)
    base_cost = cur_cost
    seen, found = [cur_state], []
    base = robust_correct(ev, cur_state)
    if base is not None:
        c = robust_cost(ev, base)
        seen.append(base)
        if c < COST_GAIN * cur_cost and c < cur_cost - COST_MIN_DROP:
            found.append((c, "the delivered orbit, corrected on every observation", base))
        base_cost = min(base_cost, c)
    if verbose:
        print(f"[basin audit] {'; '.join(why)} -- delivered {fit.n_observations()}/"
              f"{len(fit._obs)} @ {fit.rms_residual():.3f}\", robust cost {cur_cost:.1f}"
              f" ({base_cost:.1f} corrected)", flush=True)
    for label, seed in range_seeds(fit._obs, ctx["Robs"] if "Robs" in ctx else ev.Robs, ev.t0):
        st = robust_correct(ev, seed)
        if st is None or any(same_basin(st, s) for s in seen):
            continue
        seen.append(st)
        c = robust_cost(ev, st)
        if math.isfinite(c) and c < COST_GAIN * base_cost and c < base_cost - COST_MIN_DROP:
            found.append((c, label, st))
    found.sort(key=lambda x: x[0])
    adoptable = []
    for c, label, st in found[:MAX_CANDIDATES]:
        try:
            # Hand on from the observations the robust state actually fits: a plain correction
            # over ALL of them lets the few it discounted drag it straight back out of the basin
            # (an apparition carrying 3 foreign obs went from the right basin to a 470" orbit at
            # near-zero range). The reject/reclaim rounds then redeem anything left out wrongly.
            # Only when that is a MAJORITY, though: fewer means the short robust correction has
            # not converged yet (a long arc: 6 of 214 within reach of a half-way state), and the
            # plain correction over everything is what finishes it.
            z = ev.resid_sigmas(np.asarray(st, float))
            mask = np.isfinite(z) & (z <= HANDOFF_SIGMA)
            if 2 * int(mask.sum()) <= len(fit._obs):
                mask = np.ones(len(fit._obs), dtype=bool)
            history = list(fit.history)
            st2, res = ev.fit(st, mask)
            st2, res = _reject_settle(ev, st2, mask, res, history, orig_n=int(mask.sum()))
            cand = _build_orbit(ctx, st2, mask, res, history)
        except Exception:
            continue
        ok = adopt(cand, fit)
        if verbose:
            print(f"  [basin audit] {label}: robust cost {c:.1f} -> {cand.n_observations()}/"
                  f"{len(fit._obs)} @ {cand.rms_residual():.3f}\" "
                  f"({cand.rms_normalized():.2f} sigma){'' if ok else ' -- not adopted'}",
                  flush=True)
        if ok:
            adoptable.append((cand, st2, mask))
    if not adoptable:
        return fit
    # Every candidate is judged against the DELIVERED fit, and the richest adoptable one wins
    # (then the lower normalized rms): adopting the first that passes would let a thin fit
    # block a richer one behind it.
    cand, st2, mask = max(adoptable, key=lambda x: (x[0].n_observations(),
                                                    -x[0].rms_normalized()))
    cand.history.append(_hsnap({"stage": "basin-audit", "n_obs": cand.n_observations(),
                                "rms": cand.rms_residual(), "score": cand.rms_normalized(),
                                "kept": True}, st2, mask))
    if verbose:
        print(f"[basin audit] adopted {cand.n_observations()}/{len(fit._obs)} @ "
              f"{cand.rms_residual():.3f}\"", flush=True)
    return cand
