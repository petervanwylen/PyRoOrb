"""Statistical Ranging (SR) for short / single-night arcs.

Logic ported from find_orb (GPL-2.0; logic openly ported, no source copied):
`get_sr_orbits` / `find_nth_sr_orbit` / `find_trial_orbit` (orb_func.cpp),
`find_median_orbit` + `compute_sr_sigmas` (orb_func.cpp), `find_sr_ranges`
(sr.cpp).

Why SR: an arc too short to curve (a single night, or a few days) does not pin
the line-of-sight RANGE — Gauss/Herget then return an essentially arbitrary orbit.
SR samples the (range, range-rate) plane with a low-discrepancy (Halton) sequence,
builds a two-body trial orbit for each sample, keeps the ones that fit the observed
angles, and reports the MEDIAN orbit as the seed and the SPREAD of the kept orbits
as the (large, honest) uncertainty.

find_orb mapping:
  - range bounds      : we sample r1 in [floor, SR_MAX_DIST] and reject per-trial on
                        the solar-escape geometry (find_orb roots a 6th-degree
                        polynomial to pre-bound the interval; the per-trial escape
                        test is the same physics, just applied at sample time).
  - (r1, vparam)->orbit: find_trial_orbit -> _trial_orbit below (two-body Lambert
                        instead of find_orb's integrate-and-correct transfer).
  - sampling          : Halton base-2 (range, biased low) / base-3 (range-rate).
  - score/accept      : evaluate_initial_orbit -> sr_score below: the SIGMA-WEIGHTED
                        per-axis rms halved, plus e/2 and find_orb's hyperbolic /
                        high-inclination penalties, accepted below SR_ACCEPT_SCORE
                        (0.7). Absolute and in sigma units -- a family is the set of
                        orbits that FIT THE DATA, never "the best we drew".
  - combine           : component-wise median state, nearest sample to it.

Two corrections (2026-08-29, found on 2021 NX4 -- see constraint.py). (1) The trial
transfer is built between the EMISSION epochs, `t_obs - range/c` at each end, exactly as
find_orb's find_transfer_orbit does (`obs->jd - obs->r / AU_PER_DAY`); the scorer, like
the real fit, evaluates the orbit at emission time, so a trial pinned to the line of sight
at the observation instant misses the endpoints by the object's motion over one
light-time -- ~5" at 1 au. The best of 316 trials then misfit an 8-obs 6-day arc by 11.6"
(the fit: 0.098"), and the trial misfit was FLAT across half the range axis, which read
as an unconstrained range. With light-time in, the same arc has a sharp valley (0.17" at
the true range, ~1" at +/-3%). (2) Acceptance used to be `max(2.0", 3 x best)`, relative to
the family's own best; when nothing fits, that converts "the model cannot fit" into "the
data do not constrain" (108 of 316 kept, a = 1.8..222 au). It is now find_orb's absolute
sigma-scaled score, and a draw that fits nothing returns NO family -- the loosen-to-best-10
fallback is gone for the same reason. find_orb also Herget-adjusts each trial over the
interior observations (adjust_herget_results); that step is NOT ported -- a light-timed
endpoint transfer already reaches ~1.5x the converged rms at the true range.
"""
import math
import numpy as np
from astropy.time import Time

from pyroorb.forward_model import MU_SUN, C_AU_DAY, predict_radec_batch
from pyroorb.twobody import lambert, kepler_propagate, quick_aeq
from pyroorb.iod import select_subarc
from pyroorb.orbit_common import radec_residuals_arcsec, unphysical_penalty
from pyroorb.orbital_elements import OBLIQ
from pyroorb.weights import sigma_for

AU_IN_KM = 1.495978707e8

MAX_SR_SPAN = 20.0      # days; SR only applies to arcs at most this long (find_orb MAX_SR_SPAN)
SR_MAX_ORBITS = 400     # trial samples to draw (find_orb default max 500)
SR_MIN_ACCEPT = 10      # need at least this many good orbits to trust SR (find_orb: > 10)
SR_MAX_DIST = 6.0       # au; upper range bound sampled (NEO/MBA regime; raise for distant)
SR_ACCEPT_SCORE = 0.7   # find_orb initial_orbit: an SR orbit is "reasonable" below this score
SR_EXTRA_DRAWS = 2      # find_orb redraws while 3 < n_reasonable <= 10; we allow this many redraws


def _haltonize(idx, radix):
    """Halton low-discrepancy value in [0,1) for index `idx`, base `radix`
    (find_orb haltonize, orb_func.cpp)."""
    rval, divisor = 0, 1
    while idx:
        rval = rval * radix + idx % radix
        idx //= radix
        divisor *= radix
    return rval / divisor


def _minimum_sr_distance(span_days):
    """find_orb minimum_sr_distance: a floor on the sampled range so SR never lands on
    an Earth-grazing (geosat-distance) artifact. Larger of a geosynchronous-scaled
    distance and a 1 km/s slow-mover distance over the arc."""
    geosat = 42164.0 / AU_IN_KM
    geo_orbit = geosat * (span_days * 3.0) ** (2.0 / 3.0)
    slow = span_days * 1.0 * 86400.0 / AU_IN_KM      # 1 km/s lower speed limit
    return max(slow, geo_orbit)


def _trial_orbit(R0, L0, Rl, Ll, r1, vparam, dt, t0_off):
    """find_trial_orbit: build a state at the epoch from a first-obs range `r1` and a
    range-rate parameter `vparam` in [-1, 1].

    `R0,L0` / `Rl,Ll` are the heliocentric observer position and LOS unit vector at the
    first / last sub-arc obs; `dt` is the first->last span of OBSERVATION epochs (TDB
    days); `t0_off` is (t0_tdb - tdb(t_first)). Returns a heliocentric-equatorial state6 at
    the epoch, or None if the sample is hyperbolic-only / degenerate.

    Light-time: the object was at `p1` when the light LEFT it, `r1 / c` before the first
    observation, and at `p2` `r2 / c` before the last. The Lambert transfer therefore spans
    the emission epochs (find_orb find_transfer_orbit: `jd - r / AU_PER_DAY`), and the
    state is carried from the first EMISSION epoch to t0."""
    p1 = R0 + r1 * L0                                  # heliocentric position at first obs
    r2_los = float(np.dot(p1 - Rl, Ll))                # LOS range at last obs (straight-line)
    if r2_los <= 0:
        return None
    p2_los = Rl + r2_los * Ll
    solar_r = min(float(np.linalg.norm(p1)), float(np.linalg.norm(p2_los)))
    if solar_r <= 0:
        return None
    escape2 = 2.0 * MU_SUN / solar_r * dt * dt         # (max sub-escape displacement)^2, au^2
    disp2 = float(np.dot(p2_los - p1, p2_los - p1))
    if disp2 > escape2:                                # only a hyperbolic orbit reaches obs2
        return None
    r2 = r2_los + vparam * math.sqrt(escape2 - disp2)  # range-rate freedom about the LOS value
    if r2 <= 0:
        return None
    p2 = Rl + r2 * Ll
    tau1, tau2 = r1 / C_AU_DAY, r2 / C_AU_DAY          # light-times at the two ends (days)
    dt_emit = dt - tau2 + tau1                         # span between the EMISSION epochs
    if dt_emit <= 0:
        return None
    v1 = lambert(p1, p2, dt_emit)
    if v1 is None:
        return None
    st0 = kepler_propagate(np.concatenate([p1, v1]), t0_off + tau1)
    return st0 if np.all(np.isfinite(st0)) else None


def _ecliptic_inclination_deg(state):
    """Inclination to the ecliptic (deg) of an equatorial-J2000 heliocentric state -- the
    quantity find_orb's evaluate_initial_orbit penalises. Rotate the angular momentum
    from equatorial to ecliptic about x by the obliquity."""
    h = np.cross(state[:3], state[3:])
    c, s_ = math.cos(OBLIQ), math.sin(OBLIQ)
    hz_ecl = -s_ * h[1] + c * h[2]
    hn = float(np.linalg.norm(h))
    if hn <= 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, hz_ecl / hn))))


def sr_score(wrms, n_obs, a, e, q, incl_deg):
    """find_orb evaluate_initial_orbit (orb_func.cpp), heliocentric branch, ported as a
    score in which lower is better and < SR_ACCEPT_SCORE is "reasonable":

        score = wrms / 2                       (x (n_obs - 2) / 4 when n_obs < 6: with very
                                                few obs the rms deserves less weight)
              + e / 2                          (prior against high eccentricity)
              + 1000 x (e - 1.01)  if e > 1.01 (hyperbolics strongly discouraged)
              + high-inclination penalty       (i > 0.5 rad and e < 0.8: 0.2 x (i - 0.5),
                                                capped at 0.2, tapered to 0 as e -> 0.8)
              + 10000              if q < solar radius (an orbit through the Sun)

    `wrms` is the per-axis SIGMA-NORMALISED rms (compute_weighted_rms). find_orb also
    SUBTRACTS an (a, i) main-belt likelihood read from a scattergram table
    (adjustment_for_orbit_likelihood); that table is not ported, so our score is slightly
    STRICTER on main-belt members than find_orb's, never looser. The planet-centric branch
    (find_best_fit_planet) is not ported either: an SR trial here is always heliocentric,
    and unbound / sun-plunging trials are already rejected by unphysical_penalty before
    scoring."""
    rval = wrms / 2.0
    if n_obs < 6:
        rval *= (n_obs - 2) / 4.0
    rval += e / 2.0
    if e > 1.01:
        rval += (e - 1.01) * 1000.0
    incl = math.radians(incl_deg)
    if incl > 0.5 and e < 0.8:
        pen = min((incl - 0.5) * 0.2, 0.2)
        if e > 0.6:
            pen *= (0.8 - e) / 0.2
        rval += pen
    if e < 1.0 and q < 0.00465:                        # solar radius in au
        rval += 10000.0
    return rval


def obs_sigmas(obs):
    """(n, 2) per-obs (sigma_RA, sigma_Dec) in arcsec for the SR weighting: a pre-attached
    per-obs sigma (ADES rmsRA/rmsDec, or a fixture's) when present and non-None, else the
    station model (weights.sigma_for) -- the same rule fit_object uses to weight the fit."""
    out = []
    for o in obs:
        if o.get("sigma_ra") is not None and o.get("sigma_dec") is not None:
            out.append((float(o["sigma_ra"]), float(o["sigma_dec"])))
        else:
            out.append(sigma_for(o.get("obscode", ""), o.get("astcat")))
    return np.asarray(out, float)


def _admissible_range(R0, L0, Rl, Ll, dt, r_lo, r_hi, n_scan=240):
    """The interval of first-obs ranges from which SOME bound orbit reaches the last LOS.

    find_orb roots a 6th-degree polynomial for this (find_sr_ranges, sr.cpp) and samples
    only inside the roots. We apply the same escape-geometry test _trial_orbit uses -- can a
    sub-escape object cover the displacement from p1 to the last ray in dt -- on a
    log-spaced scan of the range (no integration; microseconds per point) and return the
    [min, max] admissible range. Without this the Halton draw is spread over the whole
    [floor, SR_MAX_DIST]: a fast-moving close NEO (2023 DW, 9 obs over 4 h at 0.07 au)
    admits only ranges under ~0.1 au, so 97% of the draws died on the geometry test and the
    ~30 survivors could not populate a family. Returns (r_lo, None) when nothing is
    admissible."""
    rs = np.geomspace(max(r_lo, 1e-6), r_hi, n_scan)
    ok = []
    for r1 in rs:
        p1 = R0 + r1 * L0
        r2_los = float(np.dot(p1 - Rl, Ll))
        if r2_los <= 0:
            continue
        p2_los = Rl + r2_los * Ll
        solar_r = min(float(np.linalg.norm(p1)), float(np.linalg.norm(p2_los)))
        if solar_r <= 0:
            continue
        escape2 = 2.0 * MU_SUN / solar_r * dt * dt
        if float(np.dot(p2_los - p1, p2_los - p1)) <= escape2:
            ok.append(r1)
    if not ok:
        return r_lo, None
    # pad by one scan step so the true edges (between scan points) stay inside
    lo_i = max(0, int(np.searchsorted(rs, ok[0])) - 1)
    hi_i = min(len(rs) - 1, int(np.searchsorted(rs, ok[-1])) + 1)
    return float(rs[lo_i]), float(rs[hi_i])


def _median_state(states):
    """find_median_orbit: the sample whose POSITION is nearest the component-wise
    median position (a robust central orbit, not a mean that an outlier could drag)."""
    pos = states[:, :3]
    med = np.median(pos, axis=0)
    d2 = np.sum((pos - med) ** 2, axis=1)
    return states[int(np.argmin(d2))]


def statistical_ranging(obs, times, rhohats, Robs, base_sim, t0_tdb, subarc=None,
                        max_orbits=SR_MAX_ORBITS, verbose=False, sigmas=None):
    """Run SR over the (sub-)arc. Returns (median_state_at_t0, accepted_states (k,6),
    n_accepted) or (None, None, 0) if SR can't muster enough good orbits.

    `accepted_states` are the kept trial orbits at the epoch — their spread is the SR
    uncertainty (see sr_element_sigmas). `sigmas` is an optional (n, 2) array of per-obs
    (sigma_RA, sigma_Dec) in arcsec aligned with `obs` (the fit's own weights); when None
    the per-obs sigma is derived from the obs records (obs_sigmas). Acceptance is
    find_orb's: a trial is kept when sr_score < SR_ACCEPT_SCORE, so the family is the set
    of trials that fit the astrometry to within its sigma -- an arc no trial can fit
    returns NO family rather than a loose one."""
    s_idx, e_idx = subarc if subarc is not None else select_subarc(times, rhohats)
    span = float(times[e_idx] - times[s_idx])
    sub = slice(s_idx, e_idx + 1)
    sub_obs = obs[sub]
    sub_R = Robs[sub]
    sub_L = rhohats[sub]
    R0, L0 = np.asarray(sub_R[0]), np.asarray(sub_L[0])
    Rl, Ll = np.asarray(sub_R[-1]), np.asarray(sub_L[-1])
    # Precompute the sub-arc's TDB epochs and observed angles ONCE -- scoring a trial is
    # then a single n-body sweep, with no per-trial astropy time conversion (which would
    # otherwise dominate the 400-sample loop).
    sub_tt = Time(np.asarray([o["jd_utc"] for o in sub_obs]), format="jd", scale="utc").tdb.jd
    ra_obs = np.asarray([o["ra_deg"] for o in sub_obs])
    dec_obs = np.asarray([o["dec_deg"] for o in sub_obs])
    cosd = np.cos(np.radians(dec_obs))
    sig = np.asarray(sigmas, float)[sub] if sigmas is not None else obs_sigmas(sub_obs)
    sig_ra, sig_dec = sig[:, 0], sig[:, 1]
    n_sub = len(sub_obs)
    dt = sub_tt[-1] - sub_tt[0]
    t0_off = t0_tdb - sub_tt[0]
    r_floor = _minimum_sr_distance(max(span, 1.0 / 24.0))   # never below ~1 hr-equiv floor

    if abs(dt) < 1e-6:        # all obs at one instant: no baseline at all -> SR cannot help
        return None, None, 0
    r_lo, r_hi = _admissible_range(R0, L0, Rl, Ll, dt, r_floor, SR_MAX_DIST)
    if r_hi is None:
        if verbose:
            print(f"   SR: no bound orbit reaches the last LOS from any range in "
                  f"[{r_floor:.4f}, {SR_MAX_DIST}] au -> no SR family")
        return None, None, 0

    def _score(st):
        """(find_orb score, raw arcsec rms) of one trial; (1e9, 1e9) when it cannot be scored."""
        a, e, q = quick_aeq(st)
        if unphysical_penalty(a, e, q):       # unbound / sun-plunging -> reject pre-integration
            return 1e9, 1e9
        try:
            pred = predict_radec_batch(base_sim, t0_tdb, st, sub_tt, sub_R)
        except Exception:
            return 1e9, 1e9
        dra, ddec = radec_residuals_arcsec(pred, ra_obs, dec_obs, cosd)
        raw = math.sqrt(float(np.mean(dra ** 2 + ddec ** 2)))
        wrms = math.sqrt(float(np.mean((dra / sig_ra) ** 2 + (ddec / sig_dec) ** 2)))
        if not (math.isfinite(raw) and math.isfinite(wrms)):
            return 1e9, 1e9
        return sr_score(wrms, n_sub, a, e, q, _ecliptic_inclination_deg(st)), raw

    states, scores, raws = [], [], []
    n_finite = 0
    # find_orb draws max_orbits, cuts to the reasonable ones, and REDRAWS while the count is
    # marginal (3 < n <= 10); it never loosens the bar. A draw that leaves fewer than 4 is a
    # failure, not a hint to try harder.
    for draw in range(1 + SR_EXTRA_DRAWS):
        lo = draw * max_orbits
        for k in range(lo + 1, lo + max_orbits + 1):
            rparam = _haltonize(k, 2)
            rparam = rparam * (1.0 + rparam) / 2.0          # find_orb bias toward smaller ranges
            vparam = 2.0 * _haltonize(k, 3) - 1.0           # range-rate in [-1, 1]
            r1 = r_lo + rparam * (r_hi - r_lo)
            st = _trial_orbit(R0, L0, Rl, Ll, r1, vparam, dt, t0_off)
            if st is None:
                continue
            sc, raw = _score(st)
            if sc >= 1e5:
                continue
            n_finite += 1
            if sc < SR_ACCEPT_SCORE:
                states.append(st)
                scores.append(sc)
                raws.append(raw)
        if len(states) >= SR_MIN_ACCEPT or len(states) <= 3:
            break

    if len(states) < SR_MIN_ACCEPT:
        if verbose:
            print(f"   SR: only {len(states)} of {n_finite} trial orbits fit the {span:.2f} d "
                  f"arc (score < {SR_ACCEPT_SCORE}); need {SR_MIN_ACCEPT} -> no SR family")
        return None, None, 0

    kept = np.asarray(states)
    median = _median_state(kept)
    if verbose:
        print(f"   SR: {len(kept)}/{n_finite} orbits fit the {span:.2f} d arc "
              f"(best raw rms {min(raws):.2f}\", best score {min(scores):.2f}); "
              f"median a={quick_a(median):.3f} au")
    return median, kept, len(kept)


def quick_a(state):
    """Semi-major axis (au) of a heliocentric state — for verbose/diagnostic only."""
    r = float(np.linalg.norm(state[:3]))
    v2 = float(np.dot(state[3:], state[3:]))
    energy = 0.5 * v2 - MU_SUN / r
    return -MU_SUN / (2.0 * energy) if energy < 0 else float("inf")


def sr_element_sigmas(kept_states, epoch_tdb, element_vector_fn, elem_names):
    """find_orb compute_sr_sigmas: the SR uncertainty is the SPREAD of the kept-orbit
    ELEMENTS, not a covariance from a Jacobian. Convert each kept state to elements and
    take the per-element standard deviation (angles unwrapped at ±180°). Returns
    {name: (median_value, sigma)} -- a robust median plus the (often large) SR sigma.

    `element_vector_fn` / `elem_names` are passed in to avoid importing fit_object here
    (it imports us) -- the caller hands us orbital_elements.element_vector and ELEM_NAMES."""
    rows = []
    for st in kept_states:
        try:
            v = element_vector_fn(st, epoch_tdb)
        except Exception:
            continue
        if np.all(np.isfinite(v)):
            rows.append(v)
    if len(rows) < 3:
        return None
    rows = np.asarray(rows)
    angle_idx = {i for i, nm in enumerate(elem_names)
                 if nm in ("i", "asc_node", "arg_per", "M")}
    out = {}
    for j, nm in enumerate(elem_names):
        col = rows[:, j]
        med = float(np.median(col))
        if j in angle_idx:                       # unwrap about the median before sigma
            d = (col - med + 180.0) % 360.0 - 180.0
            sig = float(np.std(d))
        else:
            sig = float(np.std(col))
        out[nm] = (med, sig)
    return out
