"""Detect an *essentially-unconstrained* orbit for the web UI.

A short arc can fit the astrometry beautifully (low RMS) yet leave the orbit almost free:
the line-of-sight range is undetermined, so the whole *scale* of the orbit floats. The
classic too-short-arc degeneracy. find_orb signals this through a saturated uncertainty
parameter U; here we turn it into a brief heads-up so the UI stops presenting
`a = 229650 +/- 2.4e10 au` as if it were a real measurement.

This is a *leaf* module: it depends only on a FittedOrbit's public API (`state`,
`covariance_matrix`, `geocentric_range`, `bound_fraction`, `elements`, `sr_uncertainty`, ...)
+ numpy, imports nothing from the engine, and never mutates the fit. `assess(fit)` returns
``None`` for a normally-determined orbit (no banner) or a small JSON-safe dict the browser
renders as a warning banner above the elements table.

Which view decides depends on ONE thing: whether the arc has a second night.

  * A SINGLE TRACKLET is decided by the RANGE PROFILE alone (`_range_verdict` ->
    rangescan.py): chi2 as a function of the trial distance, each node re-fitted over the other
    five degrees of freedom. There the covariance cannot be trusted in either direction -- the
    fit can settle in a wrong range basin whose local curvature is tight, and 17 of the 24
    silent tracklets in the fixture census were wrong by more than 6% in distance, one of them
    by 34x at 57 sigma of its own error. The profile asks the
    question the tracklet actually poses: over what distances can this astrometry be fitted?

  * Every LONGER arc is decided by the covariance geometry, as before, with SR consulted for
    range aliasing:

  * _cov_assess -- the covariance geometry ITSELF. 'Essentially unconstrained' means the
    normal matrix is near-singular: the covariance ellipsoid is hugely elongated along one
    combination of the state (almost always the range / orbit scale). We measure that
    DIRECTLY -- non-dimensionalise the 6x6 state covariance by the orbit's own length (|r|)
    and speed (|v|) and take sqrt(lambda_max), the 1-sigma extent of the worst-determined
    direction as a fraction of the orbit's own size. One dimensionless number; no per-element
    threshold pile. Works at any arc length. When the range is genuinely free the Jacobian is
    rank-deficient along it and this eigenvalue is enormous (or the covariance is not finite
    at all, which is itself a banner) -- the local covariance does NOT hide a free range.

  * _sr_modes -- Statistical Ranging, consulted ONLY on a genuine single-tracklet arc
    (<= SR_SINGLE_TRACKLET_MAX days, one apparition, nominal range <= SR_TRUST_RANGE) and
    ONLY for the one thing a local covariance cannot see: the admissible range plane
    splitting into two or more SEPARATED distance solutions (range aliasing). Its SPREAD
    never fires and never overrides a silent covariance verdict. (On a single tracklet the
    range profile now answers this question too, and better -- SR returns no family at all for
    a close object, whose narrow admissible window its blind draw never lands in.)

Why SR was demoted (2026-08-29). SR's family used to be allowed to upgrade a silent
covariance verdict on any single-apparition arc up to 6 days, on the premise that a
short arc's local covariance can sit in an overconfident valley. That premise was never
demonstrated on a real case: the trial builder in sr.py ignored light-time, so its best
trial misfit every short arc by 10-13" regardless of range (flat = "unconstrained"), and
a family-relative acceptance kept a third of the draws anyway. 2021 NX4 (8 obs, 5.96 d,
rms 0.098", a = 2.527 +/- 0.0018 au, s_max 7e-6) was banned as "essentially
unconstrained" by that family. With light-time corrected, NX4, the 3-night MBA fixture
(a = 3.17 +/- 0.15) and the single-night 2023 DW arc (range 0.072 au, pinned by diurnal
parallax over 4 h) all show a sharp misfit valley whose width agrees with the covariance.
SR earns its keep as a seed-finder (iod.py) and as a multi-modality detector; it is not a
better uncertainty estimator than the covariance of a converged fit.

The displayed text (headline + detail) is deliberately brief and generic: it says the orbit
isn't determined, NOT which individual elements are more or less trustworthy.

What the verdict COSTS (by design). It is not a caption over the numbers any
more: when it fires the web UI shows no nominal orbit at all -- no element values, no sigmas,
and no orbit export (elements / MPCORB / SOF / state / pseudo-MPEC all refuse) -- and in their
place the per-element interval this module supplies in `ranges`. The reasoning is that a
nominal value here is not a weak measurement, it is not a measurement: the reader has no way to
tell it from the ordinary well-determined case, it will not reproduce against find_orb or
anyone else (every code lands in a different spot of the same free valley), and an MPCORB line
carrying it travels further than any banner attached to it. The one product that survives is
the ephemeris, which is a sky prediction rather than an orbit and is what a short-arc object is
actually wanted for.
"""
import math

import numpy as np

# --- covariance-geometry determinacy (the always-available, principled core) ----------------
# sqrt(lambda_max) of the scaled state covariance is the worst-determined direction's 1-sigma
# as a FRACTION of the orbit's own size/speed. At/above S_UNCONSTR the orbit's scale is free.
# This single dimensionless threshold replaces the old per-element pile (UNCONSTR_REL/_E/_I/...).
# NOT rescaled when the s2 covariance floor made every sigma honest (2026-09-22),
# unlike the U-keyed gates that were.  This is the DEFINITION of the condition,
# not a constant tuned against a population: scaling it to hold the old verdicts
# would assert that an orbit is determined when its worst direction is 90% of
# its own scale, which is not a claim anyone would write down.  The old
# behaviour was over-confidence; the extra suppression is the correction.
#
# Measured, because the worry was a wave of newly-hidden orbits: over 17
# regression fixtures refit both ways, ZERO change verdict.  s_max is strongly
# BIMODAL -- a determined orbit sits at ~0.0000 and an undetermined one at 200+
# (2002 TD301: 200.4 -> 466.2, suppressed either way) -- so a ~1.8x scale change
# moves almost nothing across 0.5 because almost nothing lives near it.  The
# widest mover stayed far clear (2001 DV108, 0.1428 -> 0.2794).
#
# Where it DOES flip, it flips right: RANGEAMBIG_3NIGHT_MBA (11 obs over 5.08 d)
# crosses into suppressed, and a parametric bootstrap of that arc puts it at
# a = 3.31 +- 6.65 (201%, p5-p95 1.64-11.69 au); the old "loose but
# DETERMINED" reading (sa/a < 0.1) was the over-confidence.
S_UNCONSTR = 0.5         # worst-direction 1-sigma >= 50% of the orbit's own scale -> size/shape free
BOUND_FRAC_SEVERE = 0.90 # fraction of covariance clones that stay bound; below this bound-vs-unbound is open

# --- Statistical-Ranging mode count (single-tracklet arcs only) ---------------------------
# SR is asked one question, on one regime: does the admissible (range, range-rate) plane of a
# SINGLE TRACKLET split into separated distance solutions? A tracklet has no curvature, so
# the range is set by the rest of the family's physics (bound, non-sun-plunging, fits the
# angles to sigma) and can alias; past one night the arc has parallax/curvature and the
# covariance already reports the honest answer. The window mirrors iod.SR_SINGLE_NIGHT_MAX,
# the regime SR seeds.
SR_SINGLE_TRACKLET_MAX = 0.7   # days; SR consulted only when the INCLUDED arc is this short
SR_MIN_TRUST = 15              # accepted orbits required before SR's family is trustworthy (floor is 10)
SR_TRUST_RANGE = 5.0           # au; max nominal geocentric distance for which SR's family is reliable

# --- the replacement product: an interval per element ---------------------------------------
# When the banner fires the web UI shows NO nominal orbit and NO orbit export -- a value and a
# linear sigma are not a measurement when the scale is free, and publishing one invites exactly
# the "why doesn't this match find_orb" that neither number deserves. What it shows instead is
# this: the central RANGE_PCT of the family of orbits consistent with the astrometry.
RANGE_PCT = (5.0, 95.0)


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def assess(fit, n_mc=300, rng_seed=0):
    """Describe an essentially-unconstrained orbit, or return None if well determined.

    The dict (when not None) is JSON-safe and shaped for the banner:
        {
          "level":    "unconstrained",   # the only level (a merely-loose orbit is silent)
          "headline": str,               # bold one-liner
          "detail":   str,               # brief, generic plain-language sentence(s)
          "method":   "range" | "cov" | "sr-modes",   # which view produced the banner
          "s_max":    float | None,      # worst-direction 1-sigma / orbit scale (cov view)
          "bound_fraction": float | None,
          "n_modes":  int | None,        # separated range solutions (single-tracklet SR)
          "ranges":   {elem: [lo, hi]} | None,   # what the UI shows INSTEAD of a nominal orbit
          "ranges_method": "mc" | "sr" | None,
          "ranges_pct": [5.0, 95.0],
          "U": float | None,
          "digest2":  {...} | None,      # d2score.scores: digest2 RAW class scores, display only
          "unconstrained": [...],        # internal; NOT rendered (kept for tests/debugging)
        }

    An orbit chosen by the population prior (``optin_source`` set by popprior.most_probable_fit)
    answers for its ARC: the verdict is the arc's own fit's, never the chosen point's.
    """
    source = getattr(fit, "optin_source", None)
    if source is not None:
        fit = source
    rv = _range_verdict(fit)
    if rv is not None:
        # On a single tracklet the RANGE PROFILE is the whole verdict -- see _range_verdict.
        banner = None if rv["determined"] else _banner(fit, "range", rv=rv)
        if banner is not None:
            rngs, how = _ranges(fit, None)
            banner["ranges"] = rngs
            banner["ranges_method"] = how
            banner["ranges_pct"] = list(RANGE_PCT)
            banner["digest2"] = _digest2(fit)
            banner["detail"] += POPULATION_SENTENCE
        return banner
    banner = _cov_assess(fit)
    n_modes = _sr_modes(fit)
    if n_modes is not None and n_modes >= 2:
        if banner is None:
            # The one SR verdict that can stand alone: a single tracklet whose admissible
            # range plane has two or more separated solutions. A local covariance sees only
            # the one the fit landed in. NOTE: SR is consulted only on arcs the profile
            # answers for, and the profile never abstains inside its regime,
            # so on a real fit this branch is not reached -- the profile's own "separated
            # distance solutions" clause is what fires. Kept for the SR-family interval below.
            banner = _banner(fit, "sr-modes", n_modes=n_modes)
        else:
            banner["n_modes"] = n_modes
            banner["detail"] = _detail(n_modes)
    if banner is not None:
        rngs, how = _ranges(fit, n_modes)
        banner["ranges"] = rngs
        banner["ranges_method"] = how
        banner["ranges_pct"] = list(RANGE_PCT)
        # _cov_assess skips the bound-fraction draw once the scale is already known to be free,
        # which is cheap and right for the VERDICT -- but the interval it now ships is built
        # from the clones that converted to real elements, i.e. the bound ones, so how much of
        # the family that is has become part of what the panel has to say. A 300-clone draw.
        if how == "mc" and banner.get("bound_fraction") is None:
            try:
                banner["bound_fraction"] = _finite(fit.bound_fraction())
            except Exception:
                pass
        banner["detail"] = _detail(banner.get("n_modes"), banner.get("bound_fraction"))
        banner["digest2"] = _digest2(fit)
        if banner["digest2"]:
            banner["detail"] += POPULATION_SENTENCE_SCORES
    return banner


# The population prior is SAID, and credited, by design: one sentence on the banner
# wherever it is used. On a single tracklet it weighs the fitting orbits (popprior.py: the fan's
# shading and the "show an orbit anyway" choice); elsewhere it supplies the class scores, and
# "show an orbit anyway" states its own use on the page when it replaces an orbit the population
# all but rules out.
POPULATION_SENTENCE = (" PyRoOrb goes further than the astrometry alone: it weighs the orbits that "
                       "fit by a population prior -- digest2's model of the Solar System (Keys et "
                       "al. 2019, via PyDigest2) -- to find the most probable one.")
POPULATION_SENTENCE_SCORES = (" PyRoOrb also scores which kinds of object fit, using digest2's "
                              "population model of the Solar System (Keys et al. 2019, via PyDigest2).")


def _digest2(fit):
    """digest2's RAW class scores for the arc as fitted (d2score.py), or None. What an arc
    that cannot determine its orbit CAN say: across every bound orbit it admits, which kinds
    of object the modelled population puts there. Display only -- a population prior must
    never decide a hazard or identification question (see d2score's docstring)."""
    try:
        from pyroorb import d2score
        return d2score.scores(fit.included_observations())
    except Exception:
        return None


# --- the interval that replaces the nominal orbit -------------------------------------------
def _ranges(fit, n_modes=None):
    """``({element: [lo, hi]} | None, "mc" | "sr" | None)`` -- the central RANGE_PCT of the
    orbits this arc admits, per element.

    Two families can supply it and the choice is not a preference, it is which one can see the
    answer. The covariance clone cloud (Monte Carlo) is LOCAL: it samples the valley the fit
    landed in. That is the right family for the ordinary free-scale case, where the valley IS
    the ambiguity. It is the wrong one when SR has found the admissible range plane split into
    separated distance solutions -- the cloud then describes one mode and says nothing about
    the others -- so a multi-modal tracklet takes SR's family instead. Either may decline
    (an all-but-hyperbolic cloud converts to no finite elements); the other is then tried, and
    a banner with no interval at all is honest too: the UI shows the element names and no
    numbers.

    SR's family is characterised at the first observation's epoch rather than the fit's. Over
    the <= 0.7 d tracklet SR is consulted on, that costs nothing measurable in a/e/q/Q/i/node/
    peri and it is not worth re-anchoring a family this wide for M/Tp.
    """
    if n_modes and n_modes >= 2:
        got = _sr_ranges(fit)
        if got:
            return got, "sr"
    try:
        mc_rng = fit.element_ranges(pct=RANGE_PCT)
    except Exception:
        mc_rng = None
    if mc_rng:
        return {k: [float(v[0]), float(v[1])] for k, v in mc_rng.items()}, "mc"
    got = _sr_ranges(fit)
    return (got, "sr") if got else (None, None)


def _sr_ranges(fit):
    """SR's per-element percentile ranges, or None. Goes through the SAME applicability gate as
    the mode count (`_sr_family`) -- never widen it here: SR on an arc it does not apply to is
    not merely a wrong answer, it is a slow one (a family drawn over a multi-apparition span
    integrates every trial across that span; 11 s per web payload, for an answer then discarded)."""
    rng = (_sr_family(fit) or {}).get("ranges") or {}
    out = {}
    for k, v in rng.items():
        lo, hi = _finite(v[0]), _finite(v[1])
        if lo is not None and hi is not None:
            out[k] = [lo, hi]
    return out or None


# --- covariance-geometry view --------------------------------------------------------------
def _cov_assess(fit):
    """Principled determinacy straight from the covariance ellipsoid (see module docstring).

    Non-dimensionalise the 6x6 state covariance by the orbit's own length (|r|) and speed (|v|),
    eigendecompose, and take sqrt(lambda_max): the 1-sigma extent of the worst-determined
    direction as a fraction of the orbit's own size/speed. >= S_UNCONSTR -> the scale is free.
    A degenerate state/covariance (non-finite, zero scale) is itself a banner. The honest
    bound-vs-unbound fraction (energy sign over the covariance cloud) is a second physical
    trigger the scale eigenvalue can miss right at e=1."""
    try:
        state = np.asarray(fit.state(), float)
        cov = np.asarray(fit.covariance_matrix("state"), float)
    except Exception:
        return None
    if state.shape != (6,) or cov.shape != (6, 6):
        return None                               # nongrav-augmented (7-state) or no covariance
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(cov))):
        return _banner(fit, "cov")            # degenerate fit -> the numbers aren't real

    L = float(np.linalg.norm(state[:3]))      # heliocentric distance scale (au)
    V = float(np.linalg.norm(state[3:]))      # speed scale (au/day)
    if not (L > 0 and V > 0):
        return _banner(fit, "cov")

    d = np.array([L, L, L, V, V, V])
    Cs = cov / np.outer(d, d)                 # dimensionless scaled covariance
    Cs = 0.5 * (Cs + Cs.T)                     # symmetrise before eigvalsh
    try:
        lam = np.linalg.eigvalsh(Cs)
    except Exception:
        return _banner(fit, "cov")
    s_max = math.sqrt(max(float(lam[-1]), 0.0))
    scale_free = s_max >= S_UNCONSTR

    # The bound-fraction is a 300-clone covariance draw; it only ever ADDS the e~=1 trigger the
    # scale eigenvalue can miss, so it is skipped once the scale is already known to be free.
    # The test is symmetric about e = 1: a nominal on either side is open when a real share of
    # its own cloud lies on the OTHER side. An unbound nominal is not by itself undetermined --
    # 1I, 2I and 3I are unbound on arcs that pin every element -- and it used to be banner-ed
    # (its orbit suppressed, its exports refused) for that alone. One whose cloud cannot be
    # drawn is not waved through.
    bf = None
    if scale_free:
        unbound = False
    else:
        e_nom = _finite(fit.elements().get("e"))
        try:
            bf = fit.bound_fraction()         # energy-sign fraction over the covariance cloud
        except Exception:
            bf = None
        if e_nom is not None and e_nom >= 1.0:
            unbound = bf is None or bf > 1.0 - BOUND_FRAC_SEVERE
        else:
            unbound = bf is not None and bf < BOUND_FRAC_SEVERE
    if not (scale_free or unbound):
        return None
    return _banner(fit, "cov", bf=bf, s_max=s_max)


# --- Statistical-Ranging mode count (single-tracklet arcs) ---------------------------------
def _sr_family(fit):
    """SR's family for this fit, or None when SR is not applicable / not trustworthy here (long
    arc, multi-apparition, distant, too few accepted orbits, or the nominal orbit outside the
    family). The one gate: both things constraint asks SR (the mode count, and the interval that
    replaces a suppressed nominal orbit) come through here, so neither can drift wider than the
    other. `sr_uncertainty` caches on the fit's ctx, so asking twice costs one family."""
    arc = _finite(fit.arc_length_days()) or 0.0
    nopp = fit.n_oppositions() or 0
    if arc > SR_SINGLE_TRACKLET_MAX or nopp > 1:
        return None
    nom_range = fit.geocentric_range()        # SR's family is spurious beyond its sampled range
    if nom_range is None or nom_range > SR_TRUST_RANGE:
        return None
    try:
        sru = fit.sr_uncertainty()
    except Exception:
        sru = None
    if not sru or not sru.get("sigmas"):
        return None
    if sru.get("n", 0) < SR_MIN_TRUST:
        return None
    a_lo, a_hi = sru.get("ranges", {}).get("a", (None, None))
    a_nom = _finite(fit.elements().get("a"))
    if a_lo is not None and a_nom is not None and not (a_lo <= a_nom <= a_hi):
        return None
    return sru


def _sr_modes(fit):
    """Number of separated distance solutions SR finds for a single-tracklet arc, or None when
    SR does not apply here or found one. Never a spread."""
    n_modes = (_sr_family(fit) or {}).get("n_modes")
    return int(n_modes) if n_modes else None


# --- banner construction (brief, generic text) ---------------------------------------------
def _range_verdict(fit):
    """The range-profile verdict for a SINGLE-TRACKLET arc, or None when it does not apply.

    Why this outranks the covariance here, and only here. A tracklet's angles fix four of the
    six degrees of freedom; the range is left to parallax and curvature to measure, and when
    that measurement is weak the fit can settle in a WRONG range basin whose local curvature is
    perfectly tight. The covariance then reports a determined orbit, confidently, and is wrong:
    over the regression fixtures 24 of 187 best tracklets came back silent and 17 of those were
    wrong by more than 6% in distance -- 2001 UQ16 by 34x, at 57 sigma of its own reported error.
    Statistical Ranging cannot referee it either: for a close object
    its blind draw never lands inside the narrow admissible window, so it returns no family at
    all, which is exactly the case that needed one.

    The profile asks the question directly -- over what distances can this astrometry be fitted
    at all -- so on this regime it REPLACES the covariance rather than annotating it. Outside it
    (any arc with a second night, or a second apparition) nothing here applies: curvature has
    pinned the range and the covariance is the honest report.

    None therefore means ONE thing: this is not a single tracklet. Inside the regime a profile
    that could not be run comes back as a verdict too -- not determined, `profiled` False, the
    reason naming the failure -- because the alternative was a None that `assess` read as "no
    objection" and answered from the covariance, the method whose record on this population
    (24 shown, 17 wrong) is why the profile exists. It was reached: 1998 VD32's 7 observations
    over 8 minutes seed no profile node at all.
    """
    arc = _finite(fit.arc_length_days()) or 0.0
    nopp = fit.n_oppositions() or 0
    if arc > SR_SINGLE_TRACKLET_MAX or nopp > 1:
        return None
    from pyroorb import rangescan
    try:
        rv = fit.range_verdict()
    except Exception as ex:
        return rangescan.verdict(rangescan.failed(f"the range profile raised {type(ex).__name__}"))
    return rv if rv is not None else rangescan.verdict(rangescan.failed("the range profile did not run"))


def _banner(fit, method, bf=None, s_max=None, n_modes=None, unconstrained=None, rv=None):
    arc = _finite(fit.arc_length_days()) or 0.0
    nopp = fit.n_oppositions() or 0
    return {
        "level": "unconstrained",
        "headline": _headline(arc, nopp),
        "detail": _detail(n_modes, bf, rv),
        # The range profile that decided a single-tracklet verdict: the distances that fit this
        # arc, and the plain-language reason it is not one distance. None off that path.
        "range": None if rv is None else {k: rv.get(k) for k in
                                          ("rho_lo", "rho_hi", "rho_best", "rel_width",
                                           "n_intervals", "edge", "capped", "chi2_per_dof",
                                           "profiled", "reason")},
        "method": method,
        "s_max": _finite(s_max),
        "bound_fraction": _finite(bf),
        "n_modes": int(n_modes) if n_modes else None,
        "U": _finite(_safe_U(fit)),
        # The scale elements SR found free. NOT rendered (the banner text never ranks individual
        # elements); exposed only for tests/debugging -- test_constraint asserts "a" is here.
        "unconstrained": unconstrained if unconstrained is not None else ["a", "e", "Q", "P"],
    }


def _headline(arc, nopp):
    opp = f"{nopp}-opposition" if nopp else "single-opposition"
    return f"Essentially unconstrained — {_days(arc)}-day, {opp} arc"


def _detail(n_modes=None, bf=None, rv=None):
    """Brief, neutral sentences. They say the orbit isn't determined and what is shown instead;
    they do NOT rank individual elements as more or less reliable than one another."""
    msg = ("The arc is too short to determine the orbit, so no orbit is shown -- only the range "
           "each element could take across the solutions that fit this astrometry. Orbit exports "
           "are unavailable until the arc constrains a solution.")
    if rv is not None and rv.get("reason"):
        # The profile knows exactly WHY, in distances -- more use than "the arc is short".
        msg += f" Measured against the astrometry: {rv['reason']}."
    if n_modes and n_modes >= 2:
        msg += (f" The data admit {n_modes} separate distance solutions, so the object "
                f"could be on any of them.")
    bf = _finite(bf)
    if bf is not None and bf < BOUND_FRAC_SEVERE:
        # The ranges come from the clones that convert to real elements, i.e. the BOUND ones.
        # Saying so is the difference between a wide orbit and no orbit at all.
        msg += (f" Only {bf * 100:.0f}% of those solutions are gravitationally bound; the ranges "
                f"below describe that part.")
    return msg


def _days(arc):
    return f"{arc:.0f}" if arc >= 1 else f"{arc:.1f}"


def _safe_U(fit):
    try:
        return fit.uncertainty_parameter()
    except Exception:
        return None


# --- the family the 3D views draw instead of one orbit ---------------------------------------
FAMILY_N = 12          # orbits drawn faintly when the arc does not determine one


def orbit_family(fit, n=FAMILY_N, rng_seed=0):
    """Up to `n` heliocentric states the astrometry ADMITS, for the 3D views to draw as faint
    lines where no single orbit is shown (a blank scene reads as a bug, and
    "it is one of these" is what the arc actually says). None when there is nothing to draw.

    ``{"source": "range" | "covariance", "epoch_tdb": float,
       "members": [{"state": [6], "weight": float}, ...]}``

    On a single tracklet the members are RANGE-PROFILE nodes -- bound, inside the allowed
    interval (chi2 <= min + DCHI2), spread evenly in log distance -- so the fan spans exactly
    the distances the banner's interval names; each is weighted by the population posterior
    (popprior.py) for shading, uniformly when that cannot be formed. Elsewhere they are bound
    clones of the covariance. Weights are for SHADING only: the population prior is display."""
    rv = _range_verdict(fit)
    if rv is not None:
        if not rv.get("profiled"):
            return None
        from pyroorb import rangescan
        prof = fit.range_profile()
        rows = [r for r in prof.get("rows") or [] if r.get("bound") and r.get("state") is not None]
        if not rows:
            return None
        lim = min(r["chi2"] for r in rows) + rangescan.DCHI2
        rows = sorted((r for r in rows if r["chi2"] <= lim), key=lambda r: r["rho"])
        weight = {}
        try:
            from pyroorb import popprior
            post = popprior.posterior(prof, popprior.tracklet_vmag(fit.included_observations()))
            if post:
                weight = dict(zip(post["rho"], post["weight"]))
        except Exception:
            weight = {}
        idx = sorted({int(round(x)) for x in np.linspace(0, len(rows) - 1, min(n, len(rows)))})
        pick = [rows[i] for i in idx]
        w = np.array([weight.get(r["rho"], 1.0) for r in pick], float)
        if not np.any(w > 0):
            w = np.ones(len(pick))
        w = w / w.max()
        return {"source": "range", "epoch_tdb": prof.get("t0_tdb"),
                "members": [{"state": [float(x) for x in r["state"]], "weight": float(wi)}
                            for r, wi in zip(pick, w)]}
    try:
        state = np.asarray(fit.state(), float)
        cov = np.asarray(fit.covariance_matrix("state"), float)
    except Exception:
        return None
    if state.shape != (6,) or cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
        return None
    rng = np.random.default_rng(rng_seed)
    try:
        draws = rng.multivariate_normal(state, 0.5 * (cov + cov.T), size=4 * n,
                                        check_valid="ignore")
    except Exception:
        return None
    from pyroorb.forward_model import MU_SUN
    keep = []
    for st in draws:
        r, v = np.linalg.norm(st[:3]), np.linalg.norm(st[3:])
        if r > 0 and 0.5 * v * v - MU_SUN / r < 0:
            keep.append(st)
        if len(keep) >= n:
            break
    if not keep:
        return None
    return {"source": "covariance", "epoch_tdb": float(fit.epoch_tdb),
            "members": [{"state": [float(x) for x in st], "weight": 1.0} for st in keep]}
