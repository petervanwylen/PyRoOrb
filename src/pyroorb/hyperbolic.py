"""Unbound (e > 1) orbits: when one is believed, and what a clearly-unbound one still does not prove.

Design rules. Nothing here special-cases a designation: 1I, 2I and 3I earn their
orbits the way a future 4I pasted in as ITF tracklets must.

1. BOUND FIRST (applied in `fit_object._prefer_bound_ecc`; the damage test is `bound_damaged` here).
   A free fit at e > 1 is refitted with e held just below 1, and the bound orbit is delivered
   unless the data refuse it at 3 sigma (a scaled Delta chi2 for the one restriction, not a percent rise in rms, which cannot see how
   many observations stand behind it). A near-parabolic comet shown as "hyperbolic" reads to most viewers as "another interstellar!".

2. LINK BAR (`unbound_arc_ok`). A wrong linkage is exactly what fits hyperbolic, so an unbound
   orbit is only believed on an arc of >= MIN_NIGHTS nights, or MIN_NIGHTS_LONG_ARC once the arc
   spans more than LONG_ARC_DAYS ("4+ nights generally means not a mislinkage") -- UNLESS the
   data refuse the bound orbit outright (`bound_refused`: 1I's first 3 nights
   fit at 0.21" free and 0.39" held bound, and showing the bound one was wrong). Holding e at the
   cap is one restriction, so the test is the BIC bar the non-grav selection already uses
   (`NONGRAV_BIC_MARGIN`), on the chi2 the station-night error model assigns (`nightmodel.py`):
   survey sigmas run ~2x pessimistic within a night (1I's 3 nights: normalised rms 0.41) and the
   nights share offsets the model prices once each, so neither a raw chi2 (which would hide that
   doubling) nor a per-observation one (which would multiply the offsets by
   the observations) is the currency. The rms-damage rule (rule 1) is still required, so a long
   arc's many observations never buy a marginal hyperbola.

3. INBOUND REMINDER (`inbound_reminder`). e clearly above 1 (e - 1 > ECC_SIGNIFICANCE sigma) is
   unbound NOW, in osculating heliocentric elements -- not necessarily unbound on the way in, and
   the body may still lose energy on the way out. A planet can supply that difference. A
   patched-conic flyby of a planet on a circular orbit changes the heliocentric specific energy by
   at most v_P * dv, dv = 2 v_rel / (1 + b v_rel^2 / GM_P) (b the miss distance), maximised here
   over the approach geometry of a near-parabolic body. The miss is bounded below by the MOID to
   the planet's orbit, so evaluating the kick there bounds it above. When |1/a| lies within
   max(PERTURBATION_FLOOR, the largest giant-planet kick) the reminder says to read the elements
   at an epoch well before and after.

   Calibration: Jupiter at 0.05 / 0.5 / 1 / 2 / 5 au allows |d(1/a)| <= 0.117 / 0.017 / 0.009 /
   0.0045 / 0.0018 au^-1. C/1980 E1 (Bowell) -- bound inbound, osculating 1/a ~ -0.017 after
   passing 0.228 au from Jupiter, where the bound is 0.036 -- gets the reminder; 1I / 2I / 3I
   (1/a = -0.79 / -1.17 / -3.79) are far outside it. The floor covers what needs no encounter at
   all: osculating heliocentric e of long-period comets reaches ~1.004 while the most hyperbolic
   ORIGINAL (barycentric, pre-entry) orbit known is C/1987 W1 at 1/a = -2e-4 au^-1 (Krolikowska &
   Dybczynski 2019, MNRAS, Table 1).
"""
import math

import numpy as np

from pyroorb import orbit_common

MIN_NIGHTS = 4
MIN_NIGHTS_LONG_ARC = 5
LONG_ARC_DAYS = 30.0
ECC_SIGNIFICANCE = 3.0
PERTURBATION_FLOOR = 2.0e-3        # au^-1

GM_SUN_KM = 1.32712440018e11       # km^3 s^-2
AU_KM = 1.495978707e8
# (name, GM km^3 s^-2, equatorial radius km, mean orbital radius au)
GIANTS = (("Jupiter", 1.26686534e8, 71492.0, 5.2044),
          ("Saturn", 3.7931187e7, 60268.0, 9.5826),
          ("Uranus", 5.793939e6, 25559.0, 19.2184),
          ("Neptune", 6.836529e6, 24764.0, 30.1104))


def count_nights(jds):
    """Distinct observing nights in a set of observation times -- the tree's ONE
    definition of a night, :func:`orbit_common.count_nights` (the 12-hour window
    anchored on each night's first observation).

    It was a consecutive gap of 0.3 d here, the only site in the tree
    with that value: two exposures from one station 8 h apart -- a long winter
    night, or an evening and a pre-dawn revisit -- were TWO nights to this bar
    and one to everything else, and this is the bar that keeps a mislinkage
    (which fits hyperbolic trivially) from earning an unbound delivery, so a rule
    that split a night made it EASIER to clear.  The canonical rule is never
    easier: over the 198 regression fixtures the count moves on 45 (all long,
    dense arcs) and crosses the 4/5-night bar on none."""
    return orbit_common.count_nights(jds)


def unbound_arc_ok(jds):
    """Rule 2: ``(ok, nights, arc_days)`` -- whether this arc is one an unbound orbit may be
    believed on."""
    ts = [float(t) for t in jds]
    if not ts:
        return False, 0, 0.0
    nights, arc = count_nights(ts), max(ts) - min(ts)
    need = MIN_NIGHTS_LONG_ARC if arc > LONG_ARC_DAYS else MIN_NIGHTS
    return nights >= need, nights, arc


FREE_PARAMS = 6                    # a gravity-only orbit's state vector
# Rule 1's "damage": holding e at the cap is ONE restriction, so how badly the data
# refuse it is a one-degree-of-freedom Delta chi2, and 9 is 3 sigma. It replaced "the rms rises by more
# than max(10%, 0.02")", which measured the wrong thing: a percent of rms says nothing about how many
# observations stand behind it. A pasted 8-day 2I arc (68 obs, 11 nights) rose only 6% (0.651" ->
# 0.689") while the e profile put e <= 1 at 4.5 sigma -- so the bound orbit was delivered, landed ON
# e = 1, and constraint's bound fraction then read the cloud straddling it as ~50/50 undetermined.
BOUND_DAMAGE_DCHI2 = 9.0


# THE UNIT OF EVIDENCE IS THE STATION-NIGHT, AND WHAT IT IS WORTH IS MEASURED.
# Residuals inside one station-night share that night's catalogue, timing and centroiding error. A
# free six-parameter fit to TWO nights absorbs both nights' offsets outright, its residuals shrink to
# the within-night scatter, and any other orbit -- the object's TRUE one included -- then "fails" by
# the night-to-night systematics multiplied by however many observations each night happened to
# hold: 2010 LP68's two nights (17 obs) read its own 36 au orbit as refused at 9 sigma. An earlier
# rule answered that by ASSERTING a station-night worth "four measurements" (a position and a rate per
# axis), capping the dof at 4 per night and moving the bars to F(1, dof) quantiles, and then
# converted the statistic back to "sigma" for the page. Neither was derived from an error model, and
# the numerator kept the offsets' full cost, so the number had no textbook null.
#
# The model is now written down and the statistic is computed UNDER it (`nightmodel.py`): per
# station-night and axis the covariance is k^2 D + tau^2 11^T + tau_rate^2 t t^T -- within-night
# scatter, a shared offset, a shared rate error -- with (k, tau, tau_rate) MEASURED per station on
# the full-arc residuals of the regression fixtures. Both the free fit and the capped fit are
# re-optimised under that covariance (`fit_object._prefer_bound_ecc`, whitened LM), and the Delta
# chi2 of the one restriction between two such optima IS a chi2(1) variate: the rules' bars apply
# as written, 9 for damage and margin + ln N for refusal, on two nights as on two hundred, and
# nothing is converted for the page. What a night is worth falls out instead of going in: the Kish
# effective sample size n_b / (1 + (n_b - 1) rho), rho = tau^2 / (tau^2 + k^2 sigma^2), is 1.7
# independent positions per axis at F51 and 703/704 and 3.3 at G96/T08/T05 -- the retired "4 per
# night" was right for the first group and half the truth for the second -- and its sum over blocks
# and axes is the N in the refusal rule's BIC (`nightmodel.NightWhitener.n_eff`).
#
# `count_station_nights` stays as the REPORTED count (the web note's "on N independent station-
# nights", `unbound_check["station_nights"]`): the unit is the station-night rather than the
# <= 120-minute tracklet because two tracklets one station took three hours apart share the
# catalogue, the clock and the seeing that make up the systematic -- one look, not two. The module
# answers "how many nights" (count_nights, the 12-hour window, for the MIN_NIGHTS bar) and "how many
# independent looks" (this, a 0.5 d consecutive gap PER STATION) on the same arc; two exposures
# from one station 8 h apart are one night AND one look under both.
STATION_NIGHT_GAP_DAYS = orbit_common.NIGHT_GAP_DAYS


def count_station_nights(jds, stations, gap_days=STATION_NIGHT_GAP_DAYS):
    """Independent looks in a set of observations: one station, consecutive gaps <= `gap_days`
    (a station-night)."""
    by = {}
    for t, stn in zip(jds, stations):
        by.setdefault(stn, []).append(float(t))
    n = 0
    for ts in by.values():
        ts.sort()
        n += 1 + sum(1 for t0, t1 in zip(ts, ts[1:]) if t1 - t0 > gap_days)
    return n


# HOW FAR THE REPORTED SIGNIFICANCE IS ALLOWED TO GO.  Past about 10 sigma the tail probability is
# 1e-23 and the number has stopped being a statement about astrometry: what separates 10 from 40
# there is the assumption that the residuals are Gaussian, not the data (the CORRELATION is
# modelled now; the shape of the tails is not).  So the reported evidence saturates at a round,
# documented ceiling and the caller is told it saturated (`saturated_evidence`), which is a floor on
# the significance rather than a measurement of it.  The RAW statistic is kept on the record beside
# it for anyone who wants to see how far past the ceiling it went.  The verdicts never see the cap:
# it sits far above both bars.
EVIDENCE_SIGMA_CAP = 10.0
MAX_EVIDENCE_CHI2 = EVIDENCE_SIGMA_CAP ** 2


def reported_evidence(dchi2):
    """`dchi2` as the record carries it: floored at 0, capped at :data:`MAX_EVIDENCE_CHI2`."""
    return min(max(float(dchi2), 0.0), MAX_EVIDENCE_CHI2)


def saturated_evidence(evidence):
    """Is `evidence` the ceiling rather than a measurement? (:data:`EVIDENCE_SIGMA_CAP`)"""
    return evidence is not None and evidence >= MAX_EVIDENCE_CHI2


def bound_dchi2(chi2_free, chi2_bound, n_resid, n_params=FREE_PARAMS):
    """The PER-OBSERVATION statistic, kept as a diagnostic (`unbound_check["dchi2_raw"]`): how far
    holding e at the cap moves the fit's own chi2, in the residual scale the fit shows --
    (chi2_bound - chi2_free) / (chi2_free / (n_resid - n_params)). It treats every residual as
    independent, which is what makes it inflate on a few nights (module comment above), so it
    decides nothing; it is what old records and A/B studies are read against.
    None without degrees of freedom or a finite chi2."""
    dof = n_resid - n_params
    if dof <= 0 or not (chi2_free > 0) or not math.isfinite(chi2_bound):
        return None
    return (chi2_bound - chi2_free) / (chi2_free / dof)


# AN ESTIMATED SCALE MAKES THE STATISTIC AN F, AND THE F's DOF ARE THE ESTIMATE's. When the arc's own
# within-night scatter re-scales k (`nightmodel.within_scale`), the Delta chi2 is divided by a
# variance known on `scale_dof` = (within-night dof + the table's prior worth) degrees of freedom,
# and a chi2(1) over an independent chi2(nu)/nu is F(1, nu): on 20 dof the 3-sigma bar sits at 11.7
# rather than 9, on 5 at 30. So the bars are converted to the F(1, nu) quantile of the SAME tail
# probability and the reported evidence is the one-dof chi2 with the statistic's tail probability
# (`_equivalent_chi2`), so `damaged` is exactly `sqrt(evidence) >= 3` on any arc. This is Student's
# correction for an estimated variance -- the same mechanics an earlier rule used with a dof that was
# ASSERTED (4 per station-night); here nu is what the scale was actually estimated on. With no
# per-fit scale (`scale_dof` None: the table's k stands) the statistic is the chi2(1) it is meant to be.
def _f_bar(chi2_bar, dof):
    """The F(1, dof) value with the tail probability a one-dof chi2 has at `chi2_bar`."""
    from scipy.stats import chi2 as _chi2, f as _f
    return float(_f.isf(_chi2.sf(chi2_bar, 1), 1, dof))


def _equivalent_chi2(d, dof):
    """`d`, an F(1, dof) variate, as the one-dof chi2 with the SAME tail probability (:func:`_f_bar`
    run the other way). Underflow of the tail reports :data:`MAX_EVIDENCE_CHI2`."""
    from scipy.stats import chi2 as _chi2, f as _f
    p = float(_f.sf(d, 1, dof))
    if not p > 0.0:
        return MAX_EVIDENCE_CHI2
    return min(float(_chi2.isf(p, 1)), MAX_EVIDENCE_CHI2)


def block_damaged(dchi2, scale_dof=None):
    """Rule 1 on the block-model statistic: ``(damaged, evidence)`` -- the bound orbit is refused by
    the data at >= 3 sigma. With `scale_dof` None the statistic is chi2(1) and `evidence` is it;
    with the per-fit scale estimated on `scale_dof` dof the statistic is F(1, scale_dof), decided
    against the F quantile of the same tail probability and reported as the equivalent one-dof
    chi2 -- either way ``damaged`` is exactly ``sqrt(evidence) >= 3``. No verdict (False, None)
    without a finite statistic."""
    if dchi2 is None or not math.isfinite(dchi2):
        return False, None
    if scale_dof is None or not (scale_dof > 0):
        return dchi2 >= BOUND_DAMAGE_DCHI2, float(dchi2)
    if dchi2 <= 0.0:
        return False, float(dchi2)
    return dchi2 >= _f_bar(BOUND_DAMAGE_DCHI2, scale_dof), _equivalent_chi2(dchi2, scale_dof)


def block_refused(dchi2, n_eff, margin, scale_dof=None):
    """Rule 2's escape on the block-model statistic: ``(refused, dbic)`` -- a BIC for the one-
    parameter restriction e <= cap, ``dbic = evidence - ln(n_eff)`` with `n_eff` the independent
    measurements the arc holds under the model (`NightWhitener.n_eff`) and `evidence` as in
    :func:`block_damaged`, refused when ``dbic >= margin`` (the non-grav selection's
    NONGRAV_BIC_MARGIN), always. No verdict without a finite statistic or a positive count."""
    if dchi2 is None or not math.isfinite(dchi2) or not (n_eff > 0):
        return False, None
    ln_n = math.log(float(n_eff))
    if scale_dof is None or not (scale_dof > 0) or dchi2 <= 0.0:
        dbic = float(dchi2) - ln_n
        return dbic >= margin, dbic
    dbic = _equivalent_chi2(dchi2, scale_dof) - ln_n
    return dchi2 >= _f_bar(margin + ln_n, scale_dof), dbic


# --- the bound COMPARATOR ---------------------------------------------------------------------
# Rules 1 and 2 weigh the free unbound fit against "the best bound orbit", so they are only as good
# as the bound orbit they are shown. It used to be a penalised walk STARTED AT the unbound optimum.
# On a short arc the data fix the attributable (where the object is on the sky and how it moves)
# and leave the range nearly free, so chi2 has a long flat valley in range with the free optimum
# somewhere along it; dragging e down from there leaves the valley sideways and stops at 8-136"
# -- for objects whose own orbit fits the same nights at 0.2-0.5". That is a statement about where
# the walk started, and it was read as "the data refuse a bound orbit".
#
# So bound orbits are looked for where they can be: across the ADMISSIBLE REGION (Milani et al.
# 2004), the set of ranges at which a heliocentric-bound orbit through the observed lines of sight
# exists at all. For a trial range rho1 along the first included line of sight, the point on the
# LAST line of sight nearest to it gives the shortest chord and so the SLOWEST -- most bound --
# orbit through both; if even that is unbound, no bound orbit lives at that range and the seed is
# dropped. It needs no ephemeris, no Lambert solve and no observer velocity: two observed
# directions, two observer positions, one dot product per range.
# WHERE the seeds go is read off the admissible region itself, not a fixed grid: along that
# slowest-orbit family the energy is a smooth function of rho1, and the ranges where it is negative
# can be one interval or two and can be NARROW -- 2010 LP68 (a 36 au TNO, two nights) is bound only
# between ~30 and ~48 au, which a grid stepping 25 -> 50 walks straight over. So the energy is
# tabulated on a fine grid (arithmetic only) and the seeds are spread INSIDE each bound interval.
BOUND_SCAN_RANGE_AU = (1.0e-3, 300.0)
BOUND_SCAN_N = 600
BOUND_SEED_MAX_FACTOR = 3.0   # neighbouring seeds in one bound interval are at most this far apart
BOUND_SEED_EDGE = 0.12        # ...and the outermost sit this log-fraction in from the interval's ends
GM_SUN_AU = 2.959122082855911e-4   # au^3 day^-2
# The chord is a second-order stand-in for the velocity, good while the arc is short against the
# orbital period; past this the curvature of the arc pins the range itself and the local walk is
# the right comparator (and the arcs are the long, expensive ones). Same number as rule 2's
# LONG_ARC_DAYS on purpose: it is where the link bar already says an arc stops being "short".
BOUND_SEED_MAX_ARC_DAYS = LONG_ARC_DAYS


def _slowest_through(rho1, t1, obs1, u1, t2, obs2, u2):
    """``(mid position, velocity, energy)`` arrays for the slowest orbit through line of sight 1 at
    each `rho1` and line of sight 2 -- energy NaN where the nearest point of line 2 is behind the
    observer."""
    p1 = obs1[None, :] + rho1[:, None] * u1[None, :]
    rho2 = (p1 - obs2[None, :]) @ u2                  # nearest point of line 2 to each p1
    p2 = obs2[None, :] + rho2[:, None] * u2[None, :]
    v = (p2 - p1) / (t2 - t1)
    mid = 0.5 * (p1 + p2)
    r = np.linalg.norm(mid, axis=1)
    with np.errstate(all="ignore"):
        energy = 0.5 * np.einsum("ij,ij->i", v, v) - GM_SUN_AU / r
    return mid, v, np.where((rho2 > 0) & (r > 0), energy, np.nan)


def bound_seeds(t1, obs1, u1, t2, obs2, u2, scan=BOUND_SCAN_RANGE_AU, n_scan=BOUND_SCAN_N,
                max_factor=BOUND_SEED_MAX_FACTOR):
    """Heliocentric 6-states ``[(t_mid, state, rho1), ...]`` spread across the admissible region:
    the most-bound orbit through the two lines of sight, at a few ranges inside every interval of
    range where that orbit IS bound. [] when no bound orbit threads them at any range.

    `obs1`/`obs2` are the heliocentric observer positions (au) at `t1`/`t2` (days), `u1`/`u2` the
    observed unit vectors. Light-time and the arc's curvature are ignored: these are starting
    points for a least-squares fit, which supplies both."""
    obs1, u1, obs2, u2 = (np.asarray(v, float) for v in (obs1, u1, obs2, u2))
    t1, t2 = float(t1), float(t2)
    if not t2 - t1 > 0:
        return []
    grid = np.geomspace(scan[0], scan[1], int(n_scan))
    _, _, energy = _slowest_through(grid, t1, obs1, u1, t2, obs2, u2)
    bound = np.isfinite(energy) & (energy < 0)
    picks, k = [], 0
    while k < len(grid):                             # each run of consecutive bound grid points
        if not bound[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(grid) and bound[j + 1]:
            j += 1
        lo, hi = math.log(grid[k]), math.log(grid[j])
        if j == k:
            picks.append(grid[k])
        else:                  # >= 3 seeds, in from the ends, no two more than max_factor apart
            a, b = lo + BOUND_SEED_EDGE * (hi - lo), hi - BOUND_SEED_EDGE * (hi - lo)
            m = max(3, int(math.ceil((b - a) / math.log(max_factor))) + 1)
            picks += [math.exp(a + (b - a) * i / (m - 1)) for i in range(m)]
        k = j + 1
    if not picks:
        return []
    rho = np.array(sorted(set(picks)))
    mid, v, energy = _slowest_through(rho, t1, obs1, u1, t2, obs2, u2)
    return [(0.5 * (t1 + t2), np.concatenate([mid[i], v[i]]), float(rho[i]))
            for i in range(len(rho)) if np.isfinite(energy[i]) and energy[i] < 0]


def encounter_kick(gm, radius_km, a_planet_au, miss_au):
    """Upper bound on |Delta(1/a)| (au^-1) from one patched-conic flyby at `miss_au` of a planet on
    a circular orbit of radius `a_planet_au`, maximised over approach geometry for a body moving at
    the local parabolic speed."""
    v_p = math.sqrt(GM_SUN_KM / (a_planet_au * AU_KM))
    v_par = math.sqrt(2.0) * v_p
    b = max(float(miss_au) * AU_KM, radius_km)
    th = np.linspace(0.0, math.pi, 181)
    v_rel = np.sqrt(v_par**2 + v_p**2 - 2.0 * v_par * v_p * np.cos(th))
    dv = 2.0 * v_rel / (1.0 + b * v_rel**2 / gm)
    return float(np.max(2.0 * v_p * dv / GM_SUN_KM * AU_KM))


def planet_ecliptic_elements(name, epoch_tdb):
    """A planet's osculating heliocentric ecliptic elements at `epoch_tdb`, as an element dict."""
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric_posvel
    import astropy.units as u
    from pyroorb.orbital_elements import element_vector, ELEM_NAMES, _ecl_to_eq  # noqa: F401
    t = Time(epoch_tdb, format="jd", scale="tdb")
    p, v = get_body_barycentric_posvel(name.lower(), t)
    ps, vs = get_body_barycentric_posvel("sun", t)
    pos = (p - ps).xyz.to(u.au).value
    vel = (v - vs).xyz.to(u.au / u.day).value
    # get_body_barycentric_posvel is ICRS (equatorial); element_vector takes equatorial.
    return dict(zip(ELEM_NAMES, element_vector(np.concatenate([pos, vel]), epoch_tdb).tolist()))


def kick_reach(elem, epoch_tdb, planets=None):
    """``(reach_au^-1, nearest)`` -- the largest |Delta(1/a)| any giant planet could have given an
    orbit with these elements, floored at PERTURBATION_FLOOR; `nearest` is ``(planet, moid_au)`` for
    the planet setting it, or None when the floor does. `planets` maps name -> element dict (tests
    pass fixed ones)."""
    from pyroorb.hazard import moid
    reach, nearest = PERTURBATION_FLOOR, None
    for name, gm, radius, a_p in GIANTS:
        pel = (planets or {}).get(name) or planet_ecliptic_elements(name, epoch_tdb)
        d = moid(elem, pel)
        k = encounter_kick(gm, radius, a_p, d)
        if k > reach:
            reach, nearest = k, (name, d)
    return reach, nearest


def inbound_reminder(elem, sigma_e, epoch_tdb, planets=None):
    """Rule 3: the reminder dict, or None.

    Fires only when the orbit is CLEARLY unbound now (e - 1 > ECC_SIGNIFICANCE sigma_e) and its
    excess energy is within planetary reach. An orbit compatible with e = 1 never gets it (it is
    rule 1's business), nor does one whose sigma is unknown, nor one no planet could explain."""
    try:
        e, a = float(elem["e"]), float(elem["a"])
        s = float(sigma_e)
    except (TypeError, ValueError, KeyError):
        return None
    if not (math.isfinite(e) and math.isfinite(a) and math.isfinite(s)) or e <= 1.0 or a >= 0:
        return None
    if e - 1.0 <= ECC_SIGNIFICANCE * s:
        return None
    inv_a = 1.0 / a
    reach, nearest = kick_reach(elem, epoch_tdb, planets)
    if abs(inv_a) > reach:
        return None
    where = (f"its orbit passes within {nearest[1]:.2f} au of {nearest[0]}'s"
             if nearest else "no close planetary approach is even needed")
    text = (f"Unbound in its present osculating elements (e = {e:.5f} ± {s:.5f}), but the excess "
            f"(1/a = {inv_a:.5f} au⁻¹) is small enough for the planets to have supplied "
            f"({where}). That does not make it interstellar: it may have arrived bound, and it "
            f"could still lose energy on the way out. Check by moving the elements epoch a "
            f"century or more before and after perihelion.")
    return {"e": e, "sigma_e": s, "inv_a": inv_a, "reach": reach,
            "planet": nearest[0] if nearest else None,
            "moid_au": nearest[1] if nearest else None, "text": text}
