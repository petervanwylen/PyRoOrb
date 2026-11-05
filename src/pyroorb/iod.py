"""Initial Orbit Determination — a close port of find_orb's `initial_orbit()` cascade.

⚠️ PROVENANCE: The *control flow and method-selection logic* in this module was originally an almost
exact port of Bill Gray's find_orb `initial_orbit()` (find_orb/orb_func.cpp:4217) and its
helpers, read from the find_orb source. We follow find_orb deliberately and credit it here.
find_orb is GPL-2.0; this project is GPL-3.0 (see LICENSE.md) — because we
reproduce find_orb's *approach/structure* (not copied source lines) and GPL-3.0 is the
umbrella required by our REBOUND import, this is compatible.

find_orb's strategy that we mirror:
  1. work on a bounded, well-conditioned SUB-ARC (find_orb caps the search arc at 730 d and
     calls look_for_best_subarc); IOD on the whole multi-opposition arc is avoided.
  2. try several preliminary-orbit methods and KEEP THE BEST BY SCORE (evaluate_initial_orbit):
       - method of Gauss (>=3 obs), tried for its multiple polynomial roots
       - Vaisala  pass: assume a heliocentric ("from the Sun") distance      [arc < 230 d]
       - Herget   pass: assume a topocentric ("from the observer") distance
     find_orb encodes the two passes with the SIGN of `pseudo_r` (negative = from Sun); we
     keep that idea explicitly as `from_sun`.
  3. the winning seed is then handed to full differential correction (find_orb: full_improvement).

NOTE (honest deviations from find_orb, to be closed later):
  - We do NOT yet implement Statistical Ranging (SR), which find_orb prefers for very short
    arcs, nor the full progressive arc-extension across oppositions.
  - find_orb solves the 2-body transfer internally; here Herget/Vaisala get their seed velocity
    from a standard universal-variable Lambert solver, and Gauss states are moved between
    epochs with a universal-variable Kepler step (both are textbook, not from find_orb).
"""
import math
import numpy as np

import sys
from pathlib import Path
from pyroorb.forward_model import MU_SUN, predict_radec_batch, build_base_sim
from pyroorb.twobody import lambert, kepler_propagate, quick_aeq
from pyroorb.orbit_common import (OPP_GAP_DAYS as OPP_GAP, NIGHT_GAP_DAYS, radec_residuals_arcsec,
                          unphysical_penalty)

ARC_CAP = 730.0                 # find_orb: two-year maximum search arc
VAISALA_MAX_ARC = 230.0         # find_orb: max_arg_length_for_vaisala
SR_SINGLE_NIGHT_MAX = 0.7       # days; an arc this short with no second night has no
                                # curvature -> Statistical Ranging instead of Gauss/Herget
SR_SHORT_ARC_MAX = 400.0        # days; a DISTANT object on a short single-opposition arc
                                # (a~100 au TNO/damocloid seen over weeks) leaves the
                                # line-of-sight range as undetermined as a single night, so
                                # Gauss/Herget land in a spurious near-Earth basin. When their
                                # seed is catastrophic on such an arc, fall back to SR over the
                                # WHOLE arc (its full baseline). Gated on a bad score, so every
                                # cleanly-seeded object (the whole suite) skips it.
ACCEPTABLE_SCORE = 5.0          # find_orb: acceptable_score_limit (loop stops at/below this)
IOD_ARC_SHRINK = 0.7            # find_orb: arclen *= .7 when the seed score is unacceptable
IOD_MIN_ARCLEN = 0.3            # days; floor for the shrink loop (a dense single tracklet)
IOD_RESCUE_TRIGGER = 100.0      # arcsec; only a catastrophic (revolution-wrapped) seed enters
                                # the shrink loop. find_orb scores its seed AFTER a quick
                                # attempt_improvements so a decent raw seed lands under
                                # ACCEPTABLE_SCORE; we score the RAW seed (~tens of arcsec for a
                                # good one), so without this gate the loop would shrink to the
                                # floor on every object. A wrapped cross-apparition seed misses
                                # by ~degrees (thousands of "), cleanly separated from this.
IOD_RESCUE_STALLS = 2           # stop the rescue after this many non-improving shrinks
# Range-ambiguity guard. Two DISTINCT orbital basins (a~1 vs a~3) can both roughly fit a short
# arc's angles, so the RAW seed rms cannot tell them apart -- but the wrong basin can't hold the
# data once corrected (the 3-night MBA above: a=3.13 seed -> 11/11 @ 0.10", a=1.03 seed -> a=0.84,
# 7/11 @ 1.02"). find_orb compares CORRECTED solutions; so when the best raw seed is mediocre we
# differentially-correct the best of each top basin and adopt a DIFFERENT basin only if it fits
# better. Unbiased (no MBA/Vaisala prior -- that would suppress short-arc NEOs); changes nothing
# for unambiguous objects (same basin wins). Gated above this raw score (a clean seed is right).
SEED_RERANK_TRIGGER = 2.0       # arcsec; below this the best raw seed is already the right basin
SEED_RERANK_K = 4               # distinct orbital basins to correct and compare
SEED_BASIN_RATIO = 1.25         # |a1/a2| within this == same basin
SEED_RIVAL_FACTOR = 3.0         # only re-rank when a DIFFERENT-basin seed is this competitive on
                                # raw score (a clear single basin -> skip; keeps the suite fast).
                                # The raw score is coarse -- the Vaisala distance grid steps 1.4x --
                                # so the right basin can sit well behind: P/2026 U1 (9-day arc) raw
                                # 4.96" at a=0.78 (wrong, geocentric 1.9 au) vs 12.58" at a=2.9
                                # (right, 3.3 au), ratio 2.54; corrected, 1.03" vs 0.36". At 2.0
                                # the fit rejected the one southern station (W68, -3.5" in Dec:
                                # the parallax a wrong distance leaves) and kept q=0.05.
SEED_RERANK_MAX_ARC = 90.0      # days; rerank only a SHORT SINGLE-APPARITION arc -- range
                                # ambiguity is a short-arc thing; multi-apparition objects
                                # disambiguate by extension, so they skip it (and stay fast)
SEED_REANCHOR_DAYS = 100.0      # if the shrink loop settles on a sub-arc this far (one apparition
                                # gap, cf. OPP_GAP) from the preliminary epoch, re-seed it at its
                                # OWN epoch -- a far epoch makes the Cartesian-state LM hopelessly
                                # ill-conditioned. Below it the epoch is already within the data, so
                                # the common within-apparition case stays byte-identical.

# --- cross-apparition range-sweep link for a degenerate single-night ANCHOR ---
LINK_ANCHOR_MAX_SPAN = 1.5      # days; the anchor sub-arc must be ~one night (no curvature, so its
                                # heliocentric RANGE is undetermined and the geometric seed lands at
                                # the wrong distance). Longer anchors carry their own range.
LINK_WHOLE_FAIL = 1.0e4         # arcsec; the geometric seed must MISS the whole arc by ~degrees
                                # (every other apparition ~180deg off) to enter this rescue. A
                                # cleanly-linked object scores far below this, so the suite never
                                # enters -- this is keyed on a catastrophe, like the SR fallbacks.
LINK_ACCEPT = 5.0e3             # arcsec; only adopt the linked seed if it actually reproduces the
                                # whole arc this well -- a coarse two-body Vaisala link of a distant
                                # low-e orbit lands within ~hundreds of " (LM/extension refine it),
                                # so a result still in the thousands means no real link was found.
LINK_DIST_MIN = 2.0             # au; assumed-heliocentric-distance sweep bounds. Spans the MBA/
LINK_DIST_MAX = 200.0           # au; Centaur/TNO/inner-Oort regime the geometric grid (<=6 au) omits.
LINK_DIST_COARSE = 1.4          # au; coarse log step (~13 samples 2->200 au) to find the basin,
LINK_DIST_FINE = 1.05           # then a fine step refines one coarse cell around the best -- a
                                # coarse-then-fine search keeps the sweep ~30 evals/target so the
                                # rare distant-anchor objects that already fit don't slow down.


def gauss_iod(times_jd, rhohats, Rs):
    """Gauss's method of preliminary orbit determination (Curtis Alg. 5.5,
    heliocentric): three angles-only observations -> heliocentric state (r, v) at
    the middle epoch. Standard textbook celestial mechanics, not from find_orb."""
    t1, t2, t3 = times_jd
    tau1, tau3 = t1 - t2, t3 - t2
    tau = t3 - t1
    L1, L2, L3 = rhohats
    R1, R2, R3 = Rs
    p1, p2, p3 = np.cross(L2, L3), np.cross(L1, L3), np.cross(L1, L2)
    D0 = np.dot(L1, p1)
    D = np.array([[np.dot(R1, p1), np.dot(R1, p2), np.dot(R1, p3)],
                  [np.dot(R2, p1), np.dot(R2, p2), np.dot(R2, p3)],
                  [np.dot(R3, p1), np.dot(R3, p2), np.dot(R3, p3)]])
    A = (1/D0) * (-D[0,1]*tau3/tau + D[1,1] + D[2,1]*tau1/tau)
    B = (1/(6*D0)) * (D[0,1]*(tau3**2 - tau**2)*tau3/tau
                      + D[2,1]*(tau**2 - tau1**2)*tau1/tau)
    E = np.dot(R2, L2)
    R2sq = np.dot(R2, R2)
    a = -(A**2 + 2*A*E + R2sq)
    b = -2*MU_SUN*B*(A + E)
    c = -(MU_SUN**2)*(B**2)
    roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    r2 = max([x.real for x in roots if abs(x.imag) < 1e-9 and x.real > 0])
    num1 = 6*(D[2,0]*tau1/tau3 + D[1,0]*tau/tau3)*r2**3 + MU_SUN*D[2,0]*(tau**2 - tau1**2)*tau1/tau3
    den1 = 6*r2**3 + MU_SUN*(tau**2 - tau3**2)
    rho1 = (1/D0)*(num1/den1 - D[0,0])
    rho2 = A + MU_SUN*B/r2**3
    num3 = 6*(D[0,2]*tau3/tau1 - D[1,2]*tau/tau1)*r2**3 + MU_SUN*D[0,2]*(tau**2 - tau3**2)*tau3/tau1
    den3 = 6*r2**3 + MU_SUN*(tau**2 - tau1**2)
    rho3 = (1/D0)*(num3/den3 - D[2,2])
    r1v = R1 + rho1*L1
    r2v = R2 + rho2*L2
    r3v = R3 + rho3*L3
    # Lagrange f,g (first order) to get velocity at t2
    f1 = 1 - 0.5*MU_SUN/r2**3*tau1**2
    f3 = 1 - 0.5*MU_SUN/r2**3*tau3**2
    g1 = tau1 - (1/6)*MU_SUN/r2**3*tau1**3
    g3 = tau3 - (1/6)*MU_SUN/r2**3*tau3**3
    v2 = (-f3*r1v + f1*r3v) / (f1*g3 - f3*g1)
    return r2v, v2


# --- geometry + scoring (mirrors find_orb evaluate_initial_orbit) -----------
def _pos_at_distance(R, L, dist, from_sun):
    """Heliocentric position of the target along LOS L from observer R."""
    if from_sun:                              # |R + rho*L| = dist  (find_orb: pseudo_r<0)
        b = float(np.dot(R, L)); disc = b*b - (float(np.dot(R, R)) - dist*dist)
        if disc < 0:
            return None
        rho = -b + math.sqrt(disc)
    else:                                     # rho = dist           (from the observer)
        rho = dist
    return R + rho * L


SCORE_SAMPLE = 25     # max obs to evaluate a candidate seed against (it's only a
                      # ranking, not the fit) — keeps IOD O(1) in arc density.


def score_orbit(state, base_sim, t0_tdb, obs, Robs):
    """Lower is better; ~RMS in arcsec plus penalties for unphysical orbits.
    Analogue of find_orb's evaluate_initial_orbit(). Evaluated against at most
    SCORE_SAMPLE evenly-spaced obs (this is only a seed *ranking*, so a sample is
    enough and avoids one propagation per obs on dense arcs)."""
    if not np.all(np.isfinite(state)):
        return 1e9
    a, e, q = quick_aeq(state)
    penalty = unphysical_penalty(a, e, q)                # unbound / sun-plunging / absurd-a
    if penalty:
        # A degenerate candidate (unbound / sun-plunging / absurd a) is doomed regardless, and
        # INTEGRATING it makes the adaptive stepper crawl toward dt->0 for minutes inside C
        # (uninterruptible) -- the IOD multi-start on a chimera pair proposes exactly these. Reject
        # it on the cheap two-body screen WITHOUT integrating.
        return penalty
    n = len(obs)
    idx = (np.linspace(0, n - 1, SCORE_SAMPLE).round().astype(int) if n > SCORE_SAMPLE
           else np.arange(n))
    # ONE batched integration sweep over the sampled obs instead of re-integrating
    # from the epoch (and re-converting the time) per obs — this scoring dominated the
    # IOD cost on otherwise-fast objects (e.g. LH21 5s -> ~2s).
    jds = np.array([obs[k]["jd_utc"] for k in idx])
    tt = Time(jds, format="jd", scale="utc").tdb.jd
    Rs = [Robs[k] for k in idx]
    ra_obs = np.array([obs[k]["ra_deg"] for k in idx])
    dec_obs = np.array([obs[k]["dec_deg"] for k in idx])
    try:
        pred = predict_radec_batch(base_sim, t0_tdb, state, tt, Rs)
    except Exception:
        return 1e9
    dra, ddec = radec_residuals_arcsec(pred, ra_obs, dec_obs, np.cos(np.radians(dec_obs)))
    s = math.sqrt(float(np.mean(dra**2 + ddec**2))) + penalty
    return s if math.isfinite(s) else 1e9


COS_ARC_CAP = 0.70710678   # cos(45°): find_orb's max angular sub-arc length for IOD
FIVE_DEG_CHORD = 0.0872     # chord ≈ 5° between unit vectors (find_orb caps span here)


def _score_orbit_arc(vects):
    """find_orb's score_orbit_arc (orb_func.cpp:3512): how good an IOD arc the obs
    can yield, = departure from a great circle × angular span (capped at 5°). `vects`
    are the LOS unit vectors of the candidate window. Curvature is what carries range
    information, so a single night (zero curvature) scores ~0 and a well-curved
    multi-week arc scores high. Uses ONLY the unit vectors — no propagation."""
    n = len(vects)
    if n < 2:
        return -1.0
    xprod = np.cross(vects[0], vects[-1])
    max_dist = float(np.linalg.norm(vects[0] - vects[-1]))
    maximum = minimum = 0.0
    if n > 2:
        dots = vects[1:-1] @ xprod
        maximum = max(0.0, float(dots.max()))
        minimum = min(0.0, float(dots.min()))
        if max_dist < FIVE_DEG_CHORD:       # short arc: use widest intervening sep
            d = np.linalg.norm(vects[1:-1] - vects[0], axis=1)
            d2 = np.linalg.norm(vects[1:-1] - vects[-1], axis=1)
            max_dist = max(max_dist, float(d.max()), float(d2.max()))
    max_dist = min(max_dist, FIVE_DEG_CHORD)
    return (maximum - minimum) * max_dist


def select_subarc(times, rhohats=None, max_arc_len=ARC_CAP):
    """Pick the BEST sub-arc for IOD — a close port of find_orb's
    look_for_best_subarc() (orb_func.cpp:3562).

    find_orb slides a window whose ANGULAR length stays under 45° (beyond that,
    Gauss/Herget reliably diverge) and scores each candidate by departure from a
    great circle (`_score_orbit_arc`). This is what makes IOD robust on fast movers
    and long arcs: it never hands the preliminary-orbit methods a window so long
    (in sky-motion) that they blow up — the failure mode that produced non-finite
    seeds on NEOs and 100-yr arcs. arc_extend then grows OUTWARD from the seed arc.

    `max_arc_len` caps the window's time span (find_orb's `arclen` argument); the
    initial_orbit shrink loop lowers it from the full arc when a long cross-apparition
    window seeds a revolution-wrapped garbage orbit (orb_func.cpp:4328,4481).

    `rhohats` (per-obs LOS unit vectors) enables the geometric criterion; without
    them we fall back to the older nights×3+span heuristic. Returns (start, end)."""
    times = np.asarray(times)
    n = len(times)
    if n == 1:
        return 0, 0

    if rhohats is not None:
        V = np.asarray(rhohats)
        # WHY THE CONFINEMENT IS CONDITIONAL. What actually breaks a gap-spanning IOD window is
        # UNOBSERVED REVOLUTIONS -- Gauss and Herget cannot represent a piece of trajectory with a
        # whole orbit of unseen motion inside it -- and how many revolutions a gap hides depends on
        # the object's PERIOD, which the 100-day OPP_GAP does not know. It is Earth's synodic
        # bookkeeping, not the target's. For 162385 (an Aten, P = 0.64 yr) a 692-day window hides
        # ~3 revolutions and is fatal. For 2001 KQ76 (a TNO at 38.7 au, P ~ 240 yr) the same gap is
        # a tenth of a percent of one orbit and spanning it is harmless -- and NECESSARY, because
        # that object has 12 observations across 14 years and confining the window to a single
        # apparition starves the seed (measured: 12/12 @ 0.101" over 3 apparitions collapses to
        # 7/12 @ 0.057" over 2).
        #
        # We cannot know the period before the IOD, so use the observable that stands in for it:
        # how much DATA a single apparition affords. Confine first, and fall back to the
        # gap-spanning search only when confinement cannot find a window with enough observations
        # to seed. A dense inner-system object always clears the bar (162385 has 57 obs in one
        # apparition); a sparse distant one never does, and keeps exactly the behaviour it had.
        best = (0, 0); best_score = -999.0
        for confine in (True, False):
            best, best_score = _scan_windows(V, times, n, max_arc_len, confine)
            if best_score > 0 and (best[1] - best[0] + 1) >= SUBARC_MIN_OBS:
                break
        if best_score > 0:
            return best
        # geometric scoring found nothing usable (e.g. all single nights) -> heuristic

    return _subarc_heuristic(times, n, max_arc_len)


SUBARC_MIN_OBS = 5      # a confined (single-apparition) IOD window needs at least this many obs
#                         before we accept it over a gap-spanning one -- see select_subarc


def _scan_windows(V, times, n, max_arc_len, confine):
    """The sliding geometric window search. `confine` keeps each window inside one apparition."""
    best = (0, 0); best_score = -999.0
    j = 0
    for i in range(n - 1):
        if j < i:
            j = i
        # Extend the window while it stays within max_arc_len days, within 45 deg of sky, AND
        # (when `confine` -- see below) WITHIN ONE APPARITION. The last constraint is not a refinement, it is what makes the
        # window a sub-ARC: Gauss and Herget both fit a short continuous piece of trajectory,
        # and a window carrying an opposition-sized gap is two pieces with a revolution of
        # unobserved motion in between -- precisely the wrapped-orbit case they cannot
        # represent. Without it the geometric score, which rewards departure from a great
        # circle, actively PREFERS such windows: the object moves furthest across a gap.
        # 162385 (15 apparitions, 26 yr) was seeded on 11 obs spanning 692 DAYS at rms 1078",
        # and every downstream stage then worked from that: thirteen Tier-2 bridge cascades
        # rejected in a row before the recovery re-anchored and solved it in four extension
        # passes. Confining the window costs nothing -- an apparition-length window is still
        # available, it is just no longer allowed to jump the gap.
        while (j < n - 1 and times[j + 1] - times[i] <= max_arc_len
               and (not confine or times[j + 1] - times[j] <= OPP_GAP)
               and float(V[i] @ V[j + 1]) > COS_ARC_CAP):
            j += 1
        e = j if times[j] > times[i] else i      # avoid a zero-span window
        if e > i:
            sc = _score_orbit_arc(V[i:e + 1])
            if sc > best_score:
                best_score, best = sc, (i, e)
    return best, best_score


def _subarc_heuristic(times, n, max_arc_len):
    """Legacy fallback: split into oppositions, reward nights*3 + span."""
    breaks = [0] + [i + 1 for i in range(n - 1) if times[i + 1] - times[i] > OPP_GAP] + [n]
    best = (0, n - 1); best_q = -1.0
    for b in range(len(breaks) - 1):
        s, e = breaks[b], breaks[b + 1] - 1
        while e > s and times[e] - times[s] > max_arc_len:
            e -= 1
        span = times[e] - times[s]
        nights = 1 + sum(1 for k in range(s, e) if times[k + 1] - times[k] > 0.5)
        q = nights * 3 + span
        if q > best_q:
            best_q, best = q, (s, e)
    return best


def _correct_seed(state, base_sim, t0_tdb, obs, Robs, max_nfev=10):
    """A few bounded Levenberg-Marquardt steps on a seed (no rejection), so a candidate is
    judged by how well its BASIN actually fits -- find_orb compares corrected solutions, not
    raw seeds. Evaluated against at most SCORE_SAMPLE evenly-spaced obs (this only has to tell
    distinct basins apart, like score_orbit, so a sample keeps it cheap on dense/long sub-arcs).
    Returns the corrected state, or None on failure/divergence."""
    from scipy.optimize import least_squares
    n = len(obs)
    idx = (np.linspace(0, n - 1, SCORE_SAMPLE).round().astype(int) if n > SCORE_SAMPLE
           else np.arange(n))
    jds = np.array([obs[k]["jd_utc"] for k in idx])
    tt = Time(jds, format="jd", scale="utc").tdb.jd
    ra = np.array([obs[k]["ra_deg"] for k in idx])
    dec = np.array([obs[k]["dec_deg"] for k in idx])
    cosd = np.cos(np.radians(dec))
    Rs = [Robs[k] for k in idx]

    def resid(s):
        pred = predict_radec_batch(base_sim, t0_tdb, s, tt, Rs)
        dra, ddec = radec_residuals_arcsec(pred, ra, dec, cosd)
        return np.concatenate([dra, ddec])

    try:
        r = least_squares(resid, np.asarray(state, float), method="lm", max_nfev=max_nfev)
        return r.x if np.all(np.isfinite(r.x)) else None
    except Exception:
        return None


def _rerank_by_corrected_fit(scored, best, base_sim, t0_tdb, sub_obs, sub_R):
    """Resolve range ambiguity on the DATA: correct the best raw seed of each top distinct-a
    basin and, if a DIFFERENT basin fits better once corrected, adopt it. Returns the (state,
    method, score) to use -- the incumbent ``best`` unchanged when its own basin still wins (so
    unambiguous objects are untouched). No population/MBA prior -- the fit decides.
    ``best`` and the return are both (state, method, score)."""
    best_state, best_method, best_score = best
    best_a = quick_aeq(best_state)[0]
    basins = []                                          # best raw rep per distinct a-basin
    for sc, m, st in scored:
        if sc >= best_score * SEED_RIVAL_FACTOR:
            break                                        # only correct COMPETITIVE basins (cost)
        a = quick_aeq(st)[0]
        if not (a and np.isfinite(a) and a > 0):
            continue
        if any(1.0 / SEED_BASIN_RATIO < a / ab < SEED_BASIN_RATIO for _, _, _, ab in basins):
            continue
        basins.append((sc, m, st, a))
        if len(basins) >= SEED_RERANK_K:
            break
    if len(basins) < 2:
        return best                                      # only one basin -> nothing to resolve
    rer = []
    for sc, m, st, a in basins:
        cst = _correct_seed(st, base_sim, t0_tdb, sub_obs, sub_R)
        if cst is not None:
            rer.append((score_orbit(cst, base_sim, t0_tdb, sub_obs, sub_R), m, cst))
    if not rer:
        return best
    rer.sort(key=lambda x: x[0])
    winner_score, winner_method, winner_state = rer[0]
    winner_a = quick_aeq(winner_state)[0]
    different_basin = not (best_a and winner_a and 1.0 / SEED_BASIN_RATIO < winner_a / best_a < SEED_BASIN_RATIO)
    if winner_score < best_score and different_basin:
        return winner_state, winner_method, winner_score
    return best


def _seed_for_subarc(s_idx, e_idx, obs, times, rhohats, Robs, base_sim, t0_tdb, rerank=False):
    """Best Gauss / Vaisala / Herget seed over obs[s_idx..e_idx], expressed at t0_tdb.

    Returns (state_at_t0, method, score) -- score is ~rms arcsec (lower is better),
    or (None, "none", 1e9) if no finite candidate survives. With ``rerank`` (the primary
    geometric pick only, NOT the shrink loop / per-apparition probes), a mediocre best seed is
    re-ranked by CORRECTED fit across distinct basins -- the range-ambiguity guard."""
    sub = slice(s_idx, e_idx + 1)
    sub_obs, sub_R, sub_L = obs[sub], Robs[sub], rhohats[sub]
    sub_times = times[sub]
    t_first, t_last = times[s_idx], times[e_idx]
    arclen = t_last - t_first

    candidates = []   # (method, state_at_t0)

    # --- 1. Method of Gauss (find_orb tries Gauss first when >=3 obs) ---------
    if len(sub_obs) >= 3:
        idx = [0, len(sub_obs)//2, len(sub_obs)-1]
        try:
            r2, v2 = gauss_iod([sub_times[i] for i in idx],
                               [sub_L[i] for i in idx], [sub_R[i] for i in idx])
            t_mid = sub_times[idx[1]]
            st0 = kepler_propagate(np.concatenate([r2, v2]), t0_tdb - _utc_to_tdb(t_mid))
            candidates.append(("gauss", st0))
        except Exception:
            pass

    # --- 2. Vaisala ("from Sun") then Herget ("from observer") passes --------
    # find_orb distinguishes them by the sign of pseudo_r; we use `from_sun`.
    for from_sun in (True, False):
        if from_sun and arclen >= VAISALA_MAX_ARC:   # find_orb caps Vaisala at 230 d
            continue
        # scan an assumed distance geometrically (find_orb scans pseudo_r similarly)
        dist = 1.1 if from_sun else max(0.05, 0.004 * arclen**0.6667)   # find_orb heuristic
        hi = 6.0 if from_sun else 5.0
        while dist < hi:
            r1 = _pos_at_distance(sub_R[0], sub_L[0], dist, from_sun)
            r2 = _pos_at_distance(sub_R[-1], sub_L[-1], dist, from_sun)
            if r1 is not None and r2 is not None:
                v1 = lambert(r1, r2, _utc_to_tdb(t_last) - _utc_to_tdb(t_first))
                if v1 is not None:
                    st1 = np.concatenate([r1, v1])         # state at first sub-arc obs
                    st0 = kepler_propagate(st1, t0_tdb - _utc_to_tdb(t_first))
                    candidates.append(("vaisala" if from_sun else "herget", st0))
            dist *= 1.4

    # keep only finite, usable candidate states (a divergent Lambert/Kepler step can
    # return NaN/inf; scoring or fitting those crashes the LM downstream).
    candidates = [(m, st) for m, st in candidates
                  if st is not None and np.all(np.isfinite(st))]
    if not candidates:
        return None, "none", 1e9
    scored = [(score_orbit(st, base_sim, t0_tdb, sub_obs, sub_R), m, st) for m, st in candidates]
    scored = [c for c in scored if math.isfinite(c[0])]
    if not scored:
        return None, "none", 1e9
    scored.sort(key=lambda x: x[0])
    best_score, best_method, best_state = scored[0]
    # Re-rank ONLY when genuinely ambiguous: a mediocre best seed AND a competitive seed from a
    # DIFFERENT basin (raw score within SEED_RIVAL_FACTOR). A clear single-basin object -- the
    # whole suite -- skips the corrections entirely, so this adds cost only to range-ambiguous
    # arcs. (Cheap two-body a from each candidate; scored is sorted, so the first different-basin
    # entry is the strongest rival.)
    if rerank and best_score > SEED_RERANK_TRIGGER:
        best_a = quick_aeq(best_state)[0]
        for sc, m, st in scored[1:]:
            if sc >= best_score * SEED_RIVAL_FACTOR:
                break                                     # no competitive rival remains
            a = quick_aeq(st)[0]
            if a and best_a and not (1.0 / SEED_BASIN_RATIO < a / best_a < SEED_BASIN_RATIO):
                return _rerank_by_corrected_fit(scored, (best_state, best_method, best_score),
                                                base_sim, t0_tdb, sub_obs, sub_R)
    return best_state, best_method, best_score


def _shrink_for_best_seed(obs, times, rhohats, Robs, base_sim, t0_tdb, init_arclen, *,
                          offset=0, incumbent=None,
                          max_stalls=None, max_iters=None, reset_to_span=False):
    """find_orb's arc-length shrink (orb_func.cpp:4322-4348): descend the search arclen by
    IOD_ARC_SHRINK, reseeding the geometric-best window (select_subarc) of each length and
    keeping the best-scoring Gauss/Herget seed. Stops at ACCEPTABLE_SCORE, the IOD_MIN_ARCLEN
    floor, `max_stalls` consecutive non-improvements, or `max_iters` passes. Windows map back to
    GLOBAL indices via `offset`. `incumbent` is an existing (state, method, score, sub) to try to
    BEAT (the geometric pick) -- same shape as the return, so the output can be fed straight back
    in; None starts fresh. Returns the best (state, method, score, sub) found.

    The one shrink loop shared by both IOD anchor searches; they differ only in how arclen steps:
      - initial_orbit's whole-arc rescue: offset 0, seeded with the geometric pick as incumbent,
        stall-bounded, FIXED decay (arclen *= 0.7) -- pulls a revolution-wrapped cross-apparition
        seed back inside one apparition.
      - best_seed_subarc's per-apparition anchor: an obs[s..e] slice with offset=s, iteration-
        bounded, reset_to_span (arclen = current-window-span * 0.7, so it provably shrinks even
        when select_subarc keeps returning a window shorter than the cap) -- shortens a long
        single-apparition window that would seed a wrapped orbit. This stepping is load-bearing
        for the multi-rev recovery (e.g. 2023 VM6), hence kept exactly, not collapsed."""
    best_state, best_method, best_score, best_sub = incumbent or (None, None, float("inf"), None)
    arclen = init_arclen
    prev = best_sub                                  # don't re-seed the incumbent window
    stalls = it = 0
    while arclen >= IOD_MIN_ARCLEN and best_score > ACCEPTABLE_SCORE:
        if max_iters is not None and it >= max_iters:
            break
        it += 1
        gs, ge = select_subarc(times, rhohats, max_arc_len=arclen)
        if (gs, ge) != prev:                         # skip re-seeding an identical window
            prev = (gs, ge)
            try:
                st, m, sc = _seed_for_subarc(gs, ge, obs, times, rhohats, Robs, base_sim, t0_tdb)
            except Exception:
                sc, st, m = float("inf"), None, None
            if math.isfinite(sc) and sc < best_score:
                best_state, best_method, best_score = st, m, sc
                best_sub = (offset + gs, offset + ge)
                stalls = 0
            elif max_stalls is not None:
                stalls += 1
                if stalls >= max_stalls:
                    break
        arclen = (float(times[ge] - times[gs]) if reset_to_span else arclen) * IOD_ARC_SHRINK
    return best_state, best_method, best_score, best_sub


RESCUE_MIN_NIGHTS = 3   # a rescue window needs curvature: a one- or two-night window seeds a low
#                         score trivially (its range is free) and a wrong orbit -- on C/2013 A1 the
#                         un-floored loop collapsed to one night at 15" and a = 0.84 au.


def rescue_window(obs, times, rhohats, Robs, sub, score):
    """A shorter sub-arc to re-seed from when the fit GROWN from `sub` failed outright, or None.

    find_orb's own initial_orbit shrinks the search arc for ANY bad seed (orb_func.cpp: `while
    (best_score > acceptable_score_limit)`, `arclen *= .7`). We gate that loop on an opposition gap
    (see initial_orbit) because a bad RAW seed is routine -- 57 of 195 regression fixtures seed past
    IOD_RESCUE_TRIGGER on a gap-free window and extension repairs every one. The case this exists for
    is the one extension could not repair: a long, gap-free, slow-moving window Gauss / Vaisala /
    Herget cannot represent at all. C/2013 A1 (Siding Spring) seeded a 468-day window at 247531",
    grew nothing, and delivered 603 of 3248 obs at 279863" with q = 2e-5 au; its 9-27 day windows
    before perihelion seed q = 1.365-1.397 au (JPL 1.39976). So the caller (fit_object._fit_from_start)
    asks only when the seed window is longer than VAISALA_MAX_ARC, straddles no gap, and its OWN
    differential correction is catastrophic, and this walks find_orb's shrink with two guards measured on that object:
    windows of fewer than RESCUE_MIN_NIGHTS nights are skipped, and the walk stops at the first seed
    that is no longer catastrophic (<= IOD_RESCUE_TRIGGER) rather than chasing ACCEPTABLE_SCORE down
    to a single night. Each window is seeded at its own epoch, as initial_orbit's forced-subarc path
    will re-seed it. Only a window that beats the failed seed's own `score` is returned."""
    times = np.asarray(times)
    best, best_score = None, score
    prev = tuple(sub)
    arclen = float(times[sub[1]] - times[sub[0]]) * IOD_ARC_SHRINK
    while arclen >= IOD_MIN_ARCLEN:
        gs, ge = select_subarc(times, rhohats, max_arc_len=arclen)
        nights = 1 + int(np.sum(np.diff(times[gs:ge + 1]) > NIGHT_GAP_DAYS))
        if (gs, ge) != prev and nights >= RESCUE_MIN_NIGHTS:
            prev = (gs, ge)
            t0w = _utc_to_tdb(times[gs])
            try:
                _st, _m, sc = _seed_for_subarc(gs, ge, obs, times, rhohats, Robs,
                                               build_base_sim(t0w), t0w)
            except Exception:
                sc = float("inf")
            if math.isfinite(sc) and sc < best_score:
                best, best_score = (gs, ge), sc
                if sc <= IOD_RESCUE_TRIGGER:
                    break
        arclen = float(times[ge] - times[gs]) * IOD_ARC_SHRINK
    return best


def _link_distant_anchor(obs, times, rhohats, Robs, base_sim, t0_tdb, ws, we):
    """Link a degenerate single-night ANCHOR of a distant multi-apparition
    object by sweeping its assumed heliocentric distance.

    A one-night anchor of a 40+ au object has no curvature, so its line-of-sight RANGE is
    undetermined; Gauss/Herget (and the geometric Vaisala grid, capped at 6 au) seed it at
    the wrong distance and the orbit predicts every OTHER apparition ~180deg off-sky. SR
    doesn't apply -- the WHOLE arc spans years, not one night, so the curvature SR needs is
    absent here and present (as the multi-apparition baseline) only ACROSS the gaps.

    The fix, extending SR's idea to a multi-apparition link: place the anchor's first obs at
    an assumed heliocentric distance, place a LATER-apparition obs at the same distance
    (the Vaisala "from the Sun" pseudo-r geometry -- exact for a circular orbit, a good seed
    for the low-e distant objects this strands), and solve the two-body Lambert transfer
    between them over the long, well-conditioned inter-apparition baseline. A one-night
    Lambert has no baseline (the chord is below the noise, so the velocity is garbage at
    every distance); months-to-years of arc give a real velocity, and only the correct
    distance reproduces the rest of the arc. Sweep distance x later-obs and keep the pair
    whose orbit best fits the WHOLE arc. Returns (state_at_t0, whole_arc_score, target_j),
    where target_j is the linked later-apparition obs (so the caller can span the initial
    fit mask across the gap and stop the extension collapsing back onto the lone anchor
    night), or (None, 1e9, None) if nothing linked.
    """
    times_arr = np.asarray(times)
    R0 = np.asarray(Robs[ws]); L0 = np.asarray(rhohats[ws])
    t_anchor = _utc_to_tdb(times[ws])
    # ONE representative per later apparition (the first obs across each opposition gap) keeps the
    # sweep cheap; a low-e distant orbit links from any of them, and the whole-arc score still
    # discriminates against all the other apparitions.
    targets = [k for k in range(we + 1, len(times))
               if times_arr[k] - times_arr[k - 1] > OPP_GAP
               and times_arr[k] - times_arr[ws] > OPP_GAP]

    def _trial(j, dist):
        r1 = _pos_at_distance(R0, L0, dist, True)
        r2 = _pos_at_distance(np.asarray(Robs[j]), np.asarray(rhohats[j]), dist, True)
        if r1 is None or r2 is None:
            return None
        v1 = lambert(r1, r2, _utc_to_tdb(times[j]) - t_anchor)
        if v1 is None:
            return None
        st0 = kepler_propagate(np.concatenate([r1, v1]), t0_tdb - t_anchor)
        return st0 if np.all(np.isfinite(st0)) else None

    def _score(j, dist):
        st = _trial(j, dist)
        if st is None:
            return float("inf"), None
        return score_orbit(st, base_sim, t0_tdb, obs, Robs), st

    best_state, best_score, best_j = None, float("inf"), None
    for j in targets:
        # coarse log pass to find the basin, then a fine pass over the one best coarse cell.
        coarse = []
        d = LINK_DIST_MIN
        while d < LINK_DIST_MAX:
            coarse.append((d, *_score(j, d))); d *= LINK_DIST_COARSE
        d0 = min(coarse, key=lambda c: c[1])[0]
        d = d0 / LINK_DIST_COARSE
        hi = d0 * LINK_DIST_COARSE
        while d <= hi:
            sc, st = _score(j, d)
            if sc < best_score:
                best_score, best_state, best_j = sc, st, j
            d *= LINK_DIST_FINE
    return best_state, best_score, best_j


def initial_orbit(obs, times, rhohats, Robs, subarc=None):
    """Port of find_orb's initial_orbit(): score Gauss / Vaisala / Herget seeds, keep best.

    `subarc` optionally forces the (start,end) sub-arc to seed from (used by --force
    mode to try each apparition); otherwise select_subarc picks the best AND a
    find_orb-style shrink loop retries with a shorter search arc when the seed is bad.

    Owns the EPOCH: the seed must live at its own sub-arc (find_orb: epoch within the arc),
    so we anchor the working epoch on the chosen window and build the IOD base there. The
    caller builds its evaluator at the RETURNED epoch -- there is no separate epoch guess to
    drift out of sync with the arc the shrink loop settles on.

    Returns (seed_state_at_epoch, method_name, score, (s_idx, e_idx), epoch_tdb)."""
    # --- 0. Statistical Ranging for a SINGLE-NIGHT arc (find_orb prefers SR here) ----
    # When the whole sub-arc is one night it has no curvature, so the line-of-sight RANGE
    # is undetermined and Gauss/Herget return an essentially arbitrary orbit. SR samples
    # the (range, range-rate) plane and returns a robust median (+ a family whose spread
    # IS the large, honest uncertainty). Gated to a genuine single night, so every
    # multi-night object -- the entire regression suite -- takes the Gauss/Herget path
    # below unchanged.
    # Gate on the WHOLE arc, not the sub-arc: a multi-night object whose IOD sub-arc
    # happens to be one night still extends to its other nights and must take the normal
    # path -- only a genuinely single-night-ONLY object (nothing to extend to) wants SR.
    s_idx, e_idx = subarc if subarc is not None else select_subarc(times, rhohats)
    # Anchor the working epoch on the chosen sub-arc and build the (rebound 8-planet) IOD base
    # there; the rest of this function seeds/scores at this epoch.
    t0_tdb = _utc_to_tdb(times[s_idx])
    base_sim = build_base_sim(t0_tdb)
    whole_nights = 1 + int(np.sum(np.diff(np.asarray(times)) > 0.5))
    whole_span = float(times[-1] - times[0])
    if whole_nights == 1 and (e_idx - s_idx + 1) >= 2 and whole_span <= SR_SINGLE_NIGHT_MAX:
        from pyroorb import sr   # lazy import: sr imports from iod, so import only when needed
        med, kept, n_acc = sr.statistical_ranging(obs, times, rhohats, Robs, base_sim, t0_tdb,
                                                  subarc=(s_idx, e_idx))
        if med is not None and n_acc >= sr.SR_MIN_ACCEPT:
            sub = slice(s_idx, e_idx + 1)
            return med, "sr", score_orbit(med, base_sim, t0_tdb, obs[sub], Robs[sub]), (s_idx, e_idx), t0_tdb

    # A forced sub-arc (--force per-apparition probe) seeds exactly that window, once. Its
    # epoch is already the forced arc's start, so no re-anchor is needed.
    if subarc is not None:
        st, m, sc = _seed_for_subarc(s_idx, e_idx, obs, times, rhohats, Robs, base_sim, t0_tdb)
        return st, m, sc, (s_idx, e_idx), t0_tdb

    # The geometric best sub-arc over the full arc -- exactly the old single-shot seed, but with
    # the range-ambiguity guard (rerank): a mediocre best seed is re-judged by CORRECTED fit
    # across distinct basins, so a short multi-night arc cannot lock into a spurious near-Earth
    # basin when a different basin actually fits the data (the 3-night MBA case). Gated to a
    # SHORT SINGLE-APPARITION arc only: range ambiguity is a short-arc phenomenon -- a multi-
    # apparition object resolves its range by cross-apparition extension, so it needs no rerank
    # (and the correction would just cost time on the whole suite's harder objects).
    whole_gap0 = float(np.max(np.diff(np.asarray(times)))) if len(times) > 1 else 0.0
    do_rerank = whole_span <= SEED_RERANK_MAX_ARC and whole_gap0 <= OPP_GAP
    best_state, best_method, best_score = _seed_for_subarc(
        s_idx, e_idx, obs, times, rhohats, Robs, base_sim, t0_tdb, rerank=do_rerank)
    best_sub = (s_idx, e_idx)

    # --- find_orb initial_orbit() shrink loop (orb_func.cpp:4322,4481) -----------------
    # Only a sub-arc that STRADDLES AN OPPOSITION GAP can hand Herget a revolution-wrapped
    # seed: the object swings away and returns to near its discovery position, so the
    # geometric window scores high yet spans ~1 revolution (e.g. 2024 CL1's 365 d window,
    # rms ~7300"). A within-apparition window that merely fits poorly is hard, not wrapped
    # -- rescuing it would only perturb a tuned extension/bridge path. So gate the loop on
    # (internal gap > OPP_GAP) AND a bad seed; then find_orb's arclen *= 0.7 shrink pulls
    # the seed back inside one apparition. Stop at ACCEPTABLE_SCORE or IOD_RESCUE_STALLS.
    sub_t = np.asarray(times[s_idx:e_idx + 1])
    straddles_gap = bool(sub_t.size > 1 and np.max(np.diff(sub_t)) > OPP_GAP)
    if straddles_gap and best_score > IOD_RESCUE_TRIGGER:
        best_state, best_method, best_score, best_sub = _shrink_for_best_seed(
            obs, times, rhohats, Robs, base_sim, t0_tdb,
            min(whole_span, ARC_CAP) * IOD_ARC_SHRINK,
            incumbent=(best_state, best_method, best_score, best_sub),
            max_stalls=IOD_RESCUE_STALLS)

    # --- SR fallback for a DISTANT object on a short single-opposition arc -------------
    # If Gauss/Herget still gave a catastrophic seed and the arc is short with no
    # opposition gap (so its endpoints aren't a revolution apart), the range is
    # undetermined just as for a single night -- scan it with SR over the WHOLE arc,
    # whose full baseline gives the parallax the geometric sub-arc lacked (e.g.
    # 2017 SN33: a~280 au, e=0.99 over 67 d -- Herget -> a=0.74 Aten @ 246", SR -> 0.51").
    whole_max_gap = float(np.max(np.diff(np.asarray(times)))) if len(times) > 1 else 0.0
    if (subarc is None and best_score > IOD_RESCUE_TRIGGER
            and whole_span <= SR_SHORT_ARC_MAX and whole_max_gap <= OPP_GAP):
        from pyroorb import sr
        med, kept, n_acc = sr.statistical_ranging(obs, times, rhohats, Robs, base_sim,
                                                  t0_tdb, subarc=(0, len(times) - 1))
        if med is not None and n_acc >= sr.SR_MIN_ACCEPT:
            sc = score_orbit(med, base_sim, t0_tdb, obs, Robs)
            if sc < best_score:
                return med, "sr", sc, (0, len(times) - 1), t0_tdb

    # --- Cross-apparition range-sweep link for a degenerate single-night ANCHOR ---
    # The geometric seed fits its single-night anchor (any distance does -- no curvature) yet
    # predicts every OTHER apparition ~180deg off, so its WHOLE-arc score is catastrophic while
    # its anchor score looks fine. SR above never fires (the whole arc is years, not one night).
    # Sweep the anchor's heliocentric distance, linking it across an apparition gap, and adopt the
    # result only if it both reproduces the whole arc (LINK_ACCEPT) and beats the geometric seed
    # there. Gated on (single-night anchor + an apparition gap + a degrees-off whole-arc seed), so
    # a cleanly-linked object -- the entire regression suite -- never enters.
    ws0, we0 = best_sub if best_sub is not None else (s_idx, e_idx)
    anchor_span = float(times[we0] - times[ws0])
    has_app_gap = bool(np.any(np.diff(np.asarray(times)) > OPP_GAP))
    if (subarc is None and best_state is not None and has_app_gap
            and anchor_span <= LINK_ANCHOR_MAX_SPAN):
        whole_seed = score_orbit(best_state, base_sim, t0_tdb, obs, Robs)
        if whole_seed > LINK_WHOLE_FAIL:
            st, sc, j = _link_distant_anchor(obs, times, rhohats, Robs, base_sim, t0_tdb, ws0, we0)
            if st is not None and sc < LINK_ACCEPT and sc < whole_seed:
                # Span the seed sub-arc from the anchor to the linked later-apparition obs:
                # a one-night init mask lets the first LM fit overfit the anchor and the
                # keep-best then prefers that tiny-rms collapse over the correct whole-arc
                # orbit. A gap-spanning mask is well-determined and holds the link.
                return st, "vaisala-link", sc, (ws0, j), t0_tdb

    # Epoch ownership: the seed must live AT its sub-arc (find_orb: epoch within the arc).
    # The working epoch was anchored on select_subarc's preliminary pick, but the shrink loop
    # above may have settled on an arc an apparition away; a Cartesian-state seed expressed
    # that far from its data is catastrophically ill-conditioned downstream (the velocity
    # Jacobian columns scale with the epoch->obs leverage ~Δt, so a multi-year gap makes them
    # ~1e4× the position columns and the LM crawls into a false minimum, grinding to its nfev
    # cap -- 740715: 8.5s @ a false 26"). Re-seed the winning arc at ITS OWN epoch so the orbit
    # handed back is well-conditioned. Gated on an apparition-away gap, so within-apparition
    # picks (the common case) keep the preliminary epoch and stay byte-identical.
    ws, we = best_sub
    t0_window = _utc_to_tdb(times[ws])
    if abs(t0_window - t0_tdb) > SEED_REANCHOR_DAYS:
        base_window = build_base_sim(t0_window)
        st, m, sc = _seed_for_subarc(ws, we, obs, times, rhohats, Robs, base_window, t0_window)
        if st is not None and math.isfinite(sc):
            best_state, best_method, best_score, t0_tdb = st, m, sc, t0_window
    return best_state, best_method, best_score, best_sub, t0_tdb


def best_seed_subarc(obs, times, rhohats, Robs, base_sim, t0_tdb, s, e):
    """find_orb's arc-shrink (orb_func.cpp:4322-4348) RESTRICTED to apparition obs[s..e]:
    descend the search arclen, reseeding the geometric best sub-arc WITHIN the apparition each
    pass, and return the GLOBAL (start, end) of the window whose Gauss/Herget seed scores best.

    Why: a long single-apparition arc -- e.g. a ~70 d window near a slow NEO's stationary loop,
    which stays < 45deg on the sky yet spans dozens of days -- hands Gauss a revolution-wrapped
    seed (raw rms ~thousands of arcsec) that strands the OTHER apparition many revolutions away.
    A short window inside the same apparition seeds a Gauss-friendly orbit that differential-
    correction + extension then carry across the gap. This is the per-apparition anchor used by
    the force/recovery multi-start; it is NOT on the default fit path (that stays byte-identical).
    Shares the shrink loop with initial_orbit via _shrink_for_best_seed (same find_orb port),
    here scoped to the obs[s..e] slice. Returns (s, e) unchanged when too short to subdivide."""
    sub_obs = obs[s:e + 1]
    sub_t = np.asarray(times[s:e + 1])
    sub_r = list(rhohats[s:e + 1])
    sub_R = Robs[s:e + 1]
    if sub_t.size < 3 or float(sub_t[-1] - sub_t[0]) < IOD_MIN_ARCLEN:
        return (s, e)
    _, _, _, best_sub = _shrink_for_best_seed(
        sub_obs, sub_t, sub_r, sub_R, base_sim, t0_tdb, ARC_CAP,
        offset=s, max_iters=30, reset_to_span=True)
    return best_sub if best_sub is not None else (s, e)


# small helper kept local to avoid import cycles
from astropy.time import Time
def _utc_to_tdb(jd_utc):
    return Time(jd_utc, format="jd", scale="utc").tdb.jd
