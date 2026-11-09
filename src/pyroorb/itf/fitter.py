"""PyRoOrb fitting adapter for the ITF search.

Where the find_orb-era pipeline wrote ``obs.txt``, invoked ``fo`` and parsed
``elem_short.json``/``elements.json`` back, the native search hands an OBS80
blob straight to :func:`fit_object.fit_orbit` and reads the numbers off the
:class:`FittedOrbit`.  ``TrialFit`` is the compact per-attempt summary the
search's control flow consumes (used/count/rms/U/q/arc), with the full fit
attached for ephemerides and residual work.

RMS CURRENCY -- the codebase standard
-------------------------------------
An arcsec rms of astrometric residuals can be averaged two ways, and they
differ by exactly sqrt(2) for identical residuals:

* **per-axis** -- ``sqrt(sum(d_ra^2 + d_dec^2) / (2n))``.  RA and dec treated
  as two separate samples of the same measurement error, so the number
  estimates sigma itself: what an observer quotes, what ADES rmsRA/rmsDec
  hold, what chi^2/dof normalises by (each obs contributes 2 dof).
* **per-observation** -- ``sqrt(sum(d_ra^2 + d_dec^2) / n)``.  The rms 2-D
  offset of a position; estimates sigma*sqrt(2), which is nobody's stated
  uncertainty.

**Reported is per-axis.  Always.**  It is what find_orb prints
(``compute_rms``, orb_func.cpp -- which divides by ``n_included * 2``, revised
to do so on 2010 Nov 4 "so that RMS residuals are computed as root-mean-square
of RA and dec treated separately ... (Gareth Williams kindly steered me the
right way on this)"), what it writes into an MPCORB line and its JSON
``rms_residual``, and the convention the MPC, JPL and OrbFit share.  Ours must
be directly comparable to theirs.  Every engine producer is already per-axis:
``FittedOrbit.rms_residual()`` / ``rms_normalized()``, ``TrialFit.rms``,
``arc_extend._Evaluator.rms_arcsec`` / ``rms_norm``.

Per-observation exists in exactly one place -- :func:`subset_rms` and its two
companions below -- because the seed-degradation gates compare a fit's rms on
one SUBSET of obs before and after a link, and computing both sides here keeps
them in one currency (see :func:`itf.search._seed_pre_rms` for the sqrt(2)
handicap that mixing them caused until 2026-07-29).  Those gates are ratios,
so the currency cancels and the thresholds are unaffected by it.

The rule that follows: **per-observation may not escape into anything a person
reads.**  Convert with :func:`per_axis` at the display boundary, with a comment
saying so.  Three deliberate exceptions, each documented at its site:
``itf.gatestats`` rows (a study feed whose stored rows must stay comparable
with those already collected), ``itf.index``'s ``fit_rms_arcsec`` (a written
DB column, ML feature and SQL prefilter term -- re-basing it would mean
re-deriving the index and retraining every ranker), and a SINGLE observation's
|O-C| or a night's mean displacement, which are 2-D offsets rather than
averages over components and so have no currency question to answer.
"""

from dataclasses import dataclass

import numpy as np

from .obs80_fields import obs_unique_id, split_obs80_long_lines
from .opp_profile import SPAN_DAYS, spans_of


@dataclass
class TrialFit:
    """Outcome of one engine fit over an astrometry pile."""
    ok: bool
    fit: object = None                 # FittedOrbit when ok
    used: int = 0                      # obs included after rejection
    count: int = 0                     # obs total
    rms: float = 0.0                   # unweighted arcsec (find_orb-style)
    U: float = 10.0                    # MPC uncertainty parameter (10 = unknown)
    q: float = 0.0                     # perihelion distance, au
    # Eccentricity, and NaN when the engine could not give one.  Carried
    # because `q` alone does not say what kind of orbit it came off: a
    # range-degenerate short arc happily converges to a hyperbola whose
    # perihelion is a fraction of an au, and any gate reading `q < 1.3` as
    # "this is a NEO" is fooled by exactly that (see search._neo_orbit_waiver).
    e: float = float("nan")
    earliest_jd: float = 0.0
    latest_jd: float = 0.0
    error: str | None = None

    @property
    def arclength_days(self):
        return self.latest_jd - self.earliest_jd


def _meta_for(obs, fmt="OBS80"):
    """A combined-object meta for an arbitrary obs pile.  ``n_requested`` stays
    1 so the engine treats this as one joint fit rather than entering the
    multi-designation identification-diligence mode.  ``fmt`` labels where the
    observation DATA came from: "ADES" once an :class:`itf.obsreg.AdesRegistry`
    has put reported sigmas / ADES provenance back on the obs, even though the
    lines they were parsed from are 80-column."""
    from pyroorb.obs_sources import _group_by_designation
    groups = _group_by_designation(obs)
    desigs = list(groups)
    primary = desigs[0] if desigs else "object"
    return {
        "designation": primary,
        "designations": desigs,
        "combined": len(desigs) > 1,
        "n_requested": 1,
        "other_designations": [d for d in desigs if d != primary],
        "format": fmt,
    }, primary


def fit_astrometry(obs80_text, obscodes, verbose=False, registry=None,
                   nongrav=False, max_sec=None, identification=False, nongrav_ref=None,
                   perturbers=None):
    """Fit an OBS80 blob (published + candidate ITF lines mixed) with the
    PyRoOrb engine.  Never raises: engine failures come back ``ok=False``
    (the search treats that like the old "fo died").

    ``registry`` is an :class:`itf.obsreg.AdesRegistry`: the lines carry the
    identity, and the registry puts back the ADES fields 80 columns cannot hold
    (reported rmsRA/rmsDec -> the per-obs sigma the fit weights by, rmsMag,
    logSNR, obsID).  Omit it and this is exactly the OBS80-only fit it always
    was.

    ``max_sec`` bounds this ONE fit's wall clock (:mod:`deadline`, enforced in
    the propagator); a fit that runs out comes back ``ok=False`` like any other
    engine failure.  The search passes ``cfg.trial_fit_max_sec`` for its
    candidate trial fits -- see that setting for why.  None = unbounded.

    ``identification=True`` fits the pile as an IDENTIFICATION HYPOTHESIS
    (``n_requested`` >= 2, what the web UI and the CLI run when two designations
    are typed together): each designation is seeded alone and the engine must
    BRIDGE them, instead of fitting one body from its best sub-arc, which is free
    to bend one designation's arc to swallow the other's.  The NEOCP monitor's
    auto-submit gate asks it of a pair that would otherwise auto-submit
    (an add-on's own check).

    ``nongrav_ref`` (with ``nongrav``) is a fit whose non-gravitational model is
    already established: the observations are fitted under that force held
    fixed, warm-started from its orbit (:func:`fit_object.fit_under_known_nongrav`),
    instead of a full fit that re-discovers the model.  Falls back to the full
    fit when that path cannot seed.

    ``perturbers`` (a :class:`perturbers.Field`, the 16 plus extra asteroids) fits under that
    asteroid field: the search passes the field its seed needed (:func:`seed_perturbers`), so
    every trial is judged in the force model the seed is -- a tracklet across a deflection
    (2016 RM33's 2005 night, past (19) Fortuna's 2011 pass) fits only there.  None = as before."""
    from pyroorb.obs_sources import read_obs_file
    from pyroorb.fit_object import fit_orbit, fit_under_known_nongrav
    from pyroorb import deadline

    text = split_obs80_long_lines(obs80_text)
    if not text.strip():
        return TrialFit(ok=False, error="empty astrometry")
    try:
        obs, _fmt = read_obs_file(text, obscodes, in_format="obs80")
    except Exception as e:
        return TrialFit(ok=False, error=f"parse: {e}")
    if not obs:
        return TrialFit(ok=False, error="no parseable observations")
    obs.sort(key=lambda o: o["jd_utc"])
    n_ades = registry.enrich(obs) if registry is not None else 0
    # Judge OUTLIERS by the station model, weight by the reported sigma (the fit
    # honours this per obs -- fit_object._observer_geometry).  A trial fit exists to
    # answer "does this tracklet belong to this object?", and the gates that read
    # its answer already floor their sigmas; letting the fit clip an observation on a
    # self-reported sigma the station cannot actually deliver made the two disagree,
    # and on a 3-obs tracklet one such clip is fatal (2021 CP63).
    for o in obs:
        o["reject_sigma_floor"] = True
    meta, primary = _meta_for(obs, fmt="ADES" if n_ades else "OBS80")
    if identification:
        meta["n_requested"] = max(2, len(meta["designations"]))
    try:
        # auto_nongrav=False: keep every ITF trial fit purely gravitational (a
        # 6-element state).  The corridor/variant/ephemeris machinery is all 6-D
        # -- a leaked 7-element nongrav state crashes the analytic ephemeris
        # jacobian (_jac_batch perturbs exactly 6 particles) -- and a
        # non-gravitational model is never justified on the short single-/few-
        # opposition arcs the search fits anyway, so discovering one is only a
        # spurious over-fit of a weak variant orbit (and an expensive one:
        # nongrav discovery dominates a slow fit).  Same guard fit_object uses
        # internally to keep a nongrav state from leaking to 6-D callers.
        # ``nongrav`` opts a SINGLE fit back into the discovery: the ITF
        # search turns it on only after a low-q linkage has been rescued by it
        # (see search._nongrav_rescue).  Everything above still applies to the
        # default path -- and the corridor now tolerates the 7-element state it
        # produces (ephemeris._nongrav_pair).
        import contextlib
        from pyroorb import force_assist as _fa
        field_ctx = (_fa.perturbers_on_for("itf", perturbers) if perturbers
                     else contextlib.nullcontext())
        with deadline.deadline(max_sec, label="ITF trial fit"), field_ctx:
            fit = (fit_under_known_nongrav(obs, obscodes, nongrav_ref, desig=primary,
                                           meta=meta)
                   if nongrav and nongrav_ref is not None else None)
            if fit is None:
                fit = fit_orbit(obs, obscodes, desig=primary, meta=meta,
                                verbose=verbose,
                                auto_nongrav=(True if nongrav else False))
    except deadline.DeadlineExceeded as e:
        # BaseException by design (deadline.py), so it is caught BY NAME here,
        # at the scope that set it -- never by the engine's defensive handlers.
        return TrialFit(ok=False, error=f"fit: {e}")
    except Exception as e:
        return TrialFit(ok=False, error=f"fit: {e}")
    if fit is None:
        return TrialFit(ok=False, error="fit returned None")
    return trial_from_fit(fit)


def seed_perturbers(trial):
    """Screen a search's seed fit for asteroid perturbers beyond the 16.

    Returns ``(field, trial)``: the perturbers.Field the seed's refit kept
    (fit_object.refine_perturbers -- picks at dchi2 >= 1 on ALL its observations, kept only
    on a same-obs chi2 gain) and the refined TrialFit, or ``(None, trial)`` unchanged when
    there is none -- a seed that needs only the 16 stays gravity, as every trial fit is by
    design. Never raises: a screen that cannot run leaves the search as it was."""
    if trial is None or not getattr(trial, "ok", False) or getattr(trial, "fit", None) is None:
        return None, trial
    try:
        from pyroorb.fit_object import refine_perturbers
        g = refine_perturbers(trial.fit)
    except Exception:
        return None, trial
    field = getattr(g, "asteroids", None)
    if g is trial.fit or not (isinstance(field, frozenset) and field):
        return None, trial
    return field, trial_from_fit(g)


def refine_trial(trial):
    """An identification TrialFit with the asteroid perturbers its observations call for
    (fit_object.refine_perturbers: the 16 and any extra body, kept only on a same-obs chi2
    gain); the trial unchanged when nothing is added or the screen cannot run. What the
    identification services (the queue's assessment, the NEOCP monitor's identification fit)
    judge, as the web and the command line already do."""
    if trial is None or not getattr(trial, "ok", False) or getattr(trial, "fit", None) is None:
        return trial
    try:
        from pyroorb.fit_object import refine_perturbers
        g = refine_perturbers(trial.fit)
    except Exception:
        return trial
    return trial if g is trial.fit else trial_from_fit(g)


def trial_from_fit(fit):
    """The :class:`TrialFit` of an already-fitted ``FittedOrbit`` -- the tail of
    :func:`fit_astrometry`, on its own so a fit that was NOT made here (the web
    session's live orbit after a Full step) can be judged by the same gates the
    search judges its own trial fits with (:func:`itf.search.opp_profile_acceptable`)."""
    mask = np.asarray(fit.is_included(), bool)
    jds = np.array([o["jd_utc"] for o in fit._obs], float)
    inc_jds = jds[mask] if mask.any() else jds
    try:
        U = fit.uncertainty_parameter()
        if U is None:
            U = 10.0
    except Exception:
        U = 10.0
    try:
        q = fit.perihelion_distance()[0]
    except Exception:
        q = 0.0
    try:
        e = float(fit.eccentricity()[0])
    except Exception:                   # unavailable, or None -> TypeError
        e = float("nan")
    return TrialFit(
        ok=True, fit=fit,
        used=int(fit.n_observations()),
        count=int(fit.n_total_observations()),
        rms=float(fit.rms_residual()),
        U=float(U), q=float(q), e=e,
        earliest_jd=float(inc_jds.min()),
        latest_jd=float(inc_jds.max()),
    )


SCREEN_PLAUSIBLE = "plausible"   # run the cold fit (the screen only skips work)
SCREEN_REJECT = "reject"         # the warm step destroyed the parent arc
SCREEN_SKIP = "skip"             # not screened (disabled, high-U seed, or unusable)


def screen_candidate(seed_fit, cand_obs80, obscodes, cfg, seed_U=None,
                     registry=None):
    """Warm-start REJECTION screen for one trial: does adding ``cand_obs80`` to
    ``seed_fit``'s arc destroy that arc?

    Rather than converge a joint orbit (0.08 s when the tracklet is true, but
    5-7 s when it is not -- diverging is the expensive case), take a BOUNDED LM
    step from the parent's own converged state:

      * evaluator built at the PARENT'S OWN epoch (``seed_fit.epoch_tdb``), NOT
        the joint arc's mid-epoch.  A candidate years away re-anchors the joint
        arc ~1000 d outside the data that constrained the seed, and then the LM
        diverges for EVERY ob, seed included -- the screen would reject the true
        tracklet;
      * state seeded at the parent's converged state, mask = all obs;
      * ``cfg.screen_nfev`` LM evaluations, then the median sigma-normalised
        residual over the PARENT obs alone.

    Measured separation is four orders of magnitude (true tracklets leave the
    parent at 0.5-1.1 sigma, incompatible ones at 6 500-31 400) in 14-44 ms.

    The screen is CONSERVATIVE BY CONSTRUCTION: it can only skip work, never
    accept a linkage, and it defers (``SCREEN_PLAUSIBLE``) wherever it is not
    trustworthy -- a high-U parent (``cfg.screen_max_U``, where a bounded warm
    step cannot represent the joint solution at all and destroys true tracklets
    too), an unusable parent, or ANY exception.

    Returns ``(verdict, seed_sigma, seconds)`` with verdict one of
    ``SCREEN_REJECT`` / ``SCREEN_PLAUSIBLE`` / ``SCREEN_SKIP``."""
    import time
    t0 = time.process_time()

    def done(verdict, sigma=0.0):
        return verdict, float(sigma), time.process_time() - t0

    if not getattr(cfg, "screen_trials", False):
        return done(SCREEN_SKIP)
    if seed_fit is None or getattr(seed_fit, "_state", None) is None:
        return done(SCREEN_SKIP)
    max_U = getattr(cfg, "screen_max_U", None)
    if max_U is not None and seed_U is not None and seed_U >= max_U:
        return done(SCREEN_SKIP)          # opt-in valve; see cfg.screen_max_U
    try:
        from pyroorb.obs_sources import read_obs_file
        from pyroorb.fit_object import _observer_geometry, _build_evaluator_at

        seed_obs = list(seed_fit._obs)
        if not seed_obs:
            return done(SCREEN_SKIP)
        text = split_obs80_long_lines(cand_obs80)
        if not text.strip():
            return done(SCREEN_SKIP)
        cand_obs, _fmt = read_obs_file(text, obscodes, in_format="obs80")
        if not cand_obs:
            return done(SCREEN_SKIP)
        # the seed obs come off the parent fit already ADES-enriched; the
        # candidate lines were just parsed, so give them their sigmas back too
        if registry is not None:
            registry.enrich(cand_obs)
        # The predictor integrates forward through `times_tdb`, so the joint
        # pile must be time-ordered; carry a seed/candidate tag alongside so the
        # residual split survives the sort.
        tagged = ([(o["jd_utc"], True, o) for o in seed_obs]
                  + [(o["jd_utc"], False, o) for o in cand_obs])
        tagged.sort(key=lambda t: t[0])
        obs = [t[2] for t in tagged]
        is_seed = np.array([t[1] for t in tagged], bool)

        geom = _observer_geometry(obs, obscodes)
        ev, _ctx = _build_evaluator_at(geom, obs, float(seed_fit.epoch_tdb))
        mask = np.ones(len(obs), bool)
        state, _res = ev.fit(np.asarray(seed_fit._state, float).copy(), mask,
                             max_nfev=cfg.screen_nfev)
        rs = np.asarray(ev.resid_sigmas(state), float)[is_seed]
        rs = rs[np.isfinite(rs)]
        if rs.size == 0:                       # nothing to judge on -> defer
            return done(SCREEN_PLAUSIBLE)
        seed_sigma = float(np.median(rs))
    except Exception:
        return done(SCREEN_PLAUSIBLE)          # never let the screen break a search
    if seed_sigma > cfg.screen_seed_sigma_max:
        return done(SCREEN_REJECT, seed_sigma)
    return done(SCREEN_PLAUSIBLE, seed_sigma)


def residuals_by_obs_id(fit):
    """Map ``obs_unique_id(line) -> |O-C| arcsec`` for every obs of a fit.
    Satellite second lines share their first line's id, so a two-line obs is
    kept or dropped as a unit downstream."""
    out = {}
    result = fit.residuals(included_only=False)
    for o, r in zip(fit._obs, result):
        line = o.get("obs80_line")
        if not line:
            continue
        out[obs_unique_id(line)] = float(np.hypot(r["d_ra_arcsec"],
                                                  r["d_dec_arcsec"]))
    return out


def _subset_obs(fit, obs_ids, *, included_only=True):
    """Yield ``(obs_id, residual_row, obs80_line)`` for the obs of ``fit`` whose
    ``obs_unique_id`` is in ``obs_ids`` -- the one iteration every subset
    statistic below shares.  ``included_only`` skips obs THIS fit rejected."""
    result = fit.residuals(included_only=False)
    for o, r, inc in zip(fit._obs, result, fit.is_included()):
        if included_only and not inc:
            continue
        line = o.get("obs80_line")
        if not line:
            continue
        oid = obs_unique_id(line)
        if oid in obs_ids:
            yield oid, r, line


def _obs_sigmas(r, line, station_floor):
    """The residual row's per-axis sigma (arcsec), raised to the obs' Veres et
    al. (2017) station floor when ``station_floor`` -- so an over-tight
    self-reported sigma cannot fake a good normalised fit.  None when the fit
    assigned no sigma."""
    sra, sdec = r.get("sigma_ra"), r.get("sigma_dec")
    if not (sra and sdec):
        return None
    if station_floor:
        from pyroorb.weights import VERES_2017, DEFAULT_SIGMA
        vf = VERES_2017.get(line[77:80].strip(), DEFAULT_SIGMA)
        sra, sdec = max(sra, vf), max(sdec, vf)
    return float(sra), float(sdec)


SQRT2 = float(np.sqrt(2.0))


def per_axis(rms_per_obs):
    """A per-observation rms (:func:`subset_rms`, :func:`subset_sigma_floor`)
    converted to the PER-AXIS currency everything reported is quoted in --
    ``/ sqrt(2)``, exact, see this module's docstring.

    Call this at the display boundary and nowhere else: a gate that compares
    two per-observation numbers must keep both of them per-observation (the
    ratio is identical either way, so converting buys nothing and risks
    converting only one side).  ``None`` passes through, so a caller can hand
    over an unmeasured value without guarding."""
    return None if rms_per_obs is None else float(rms_per_obs) / SQRT2


def subset_rms(fit, obs_ids):
    """Unweighted arcsec rms of ``fit``'s INCLUDED obs restricted to
    ``obs_ids`` (a set of ``obs_unique_id`` keys) -- e.g. the seed obs, so the
    combined orbit's fit to exactly the pre-search observations can be compared
    to what it was before the ITF link. Returns (rms, n).

    CURRENCY WARNING -- this is a PER-OBSERVATION total,
    ``sqrt(mean(d_ra^2 + d_dec^2))`` over n obs.  ``TrialFit.rms`` /
    ``FittedOrbit.rms_residual()`` is a PER-AXIS rms, ``sqrt(mean(r**2))`` over
    the 2n axis components, which for the same residuals is smaller by exactly
    sqrt(2).  The two are NOT interchangeable: comparing a subset_rms against a
    ``.rms`` silently applies a 1.414x handicap (it did, for the seed-degradation
    gates, until 2026-07-29).  To compare a fit's seed-subset rms before and
    after a link, call THIS function on both fits (see
    :func:`itf.search._seed_pre_rms`).  :func:`subset_sigma_floor` and
    :func:`residuals_by_obs_id` are in the same per-observation currency as this
    one; :func:`subset_chi2` is dimensionless."""
    sq = [r["d_ra_arcsec"] ** 2 + r["d_dec_arcsec"] ** 2
          for _oid, r, _line in _subset_obs(fit, obs_ids)]
    if not sq:
        return 0.0, 0
    return float(np.sqrt(np.mean(sq))), len(sq)


def subset_included(fit, obs_ids):
    """``(included_ids, residual_by_id)`` for the obs of ``fit`` whose
    ``obs_unique_id`` is in ``obs_ids``: which ones this fit still INCLUDES, and
    every one's |O-C| in arcsec (rejected ones included -- that residual is the
    evidence for calling the observation bad).

    An id in ``obs_ids`` that is absent from both returned collections never
    reached the fit at all (dropped from the astrometry upstream, e.g. by
    :func:`drop_offending`); the caller counts that as clipped too."""
    included, resid = set(), {}
    for oid, r, _line in _subset_obs(fit, obs_ids, included_only=False):
        resid[oid] = float(np.hypot(r["d_ra_arcsec"], r["d_dec_arcsec"]))
    for oid, _r, _line in _subset_obs(fit, obs_ids, included_only=True):
        included.add(oid)
    return included, resid


def subset_sigma_floor(fit, obs_ids):
    """The chi^2 = 1 expected raw-arcsec rms of ``fit``'s INCLUDED obs
    restricted to ``obs_ids``: ``sqrt(mean(sigma_ra^2 + sigma_dec^2))`` -- the
    rms EXACTLY these observations would show if every residual sat at its
    assigned 1-sigma astrometric uncertainty.  It is the astrometric NOISE
    FLOOR of the subset (a station-691 arc has a higher floor than an all-F51
    one), in the same ``d_ra^2 + d_dec^2`` currency as :func:`subset_rms`, so
    the two are directly comparable.  Returns ``(sigma_floor, n)``."""
    sq = []
    for _oid, r, line in _subset_obs(fit, obs_ids):
        sig = _obs_sigmas(r, line, station_floor=False)
        if sig is not None:
            sq.append(sig[0] ** 2 + sig[1] ** 2)
    if not sq:
        return 0.0, 0
    return float(np.sqrt(np.mean(sq))), len(sq)


def sigma_map_for(fit, obs_ids, *, station_floor=True):
    """``obs_unique_id -> (sigma_ra, sigma_dec)`` in ARCSEC for the INCLUDED
    obs of ``fit`` whose id is in ``obs_ids`` -- the per-axis astrometric
    uncertainties this fit assigned each observation.  Handed to
    :func:`subset_chi2` so a pre/post comparison can normalise BOTH fits'
    residuals by the SAME (original) sigma, i.e. judge the re-fit residuals in
    the units the seed fit itself used.  ``station_floor`` raises each sigma to
    its Veres et al. (2017) station value (see :func:`_obs_sigmas`)."""
    out = {}
    for oid, r, line in _subset_obs(fit, obs_ids):
        sig = _obs_sigmas(r, line, station_floor)
        if sig is not None:
            out[oid] = sig
    return out


def subset_chi2(fit, obs_ids, *, sigma_map=None, station_floor=True,
                included_only=False):
    """Sigma-normalised chi-square of ``fit``'s residuals over the obs whose
    ``obs_unique_id`` is in ``obs_ids``.  Each obs contributes
    ``(dRA/sRA)^2 + (dDec/sDec)^2`` -- TWO chi-square degrees of freedom -- so a
    subset that sits exactly at its assigned 1-sigma noise scores ``chi2 ~=
    2*n``.  Returns ``(chi2, n_obs)``.

    Normalisation sigma, per axis:
      * ``sigma_map`` (``obs_id -> (sRA, sDec)`` arcsec) when given -- hold the
        uncertainties FIXED at one fit's assessment, so a re-fit's residuals are
        judged in the ORIGINAL sigma units (an inflated post-fit sigma can't
        mask a degraded seed obs);
      * otherwise the fit's own per-obs sigma, raised to its Veres station floor
        when ``station_floor`` (an over-tight self-reported sigma can't fake a
        good normalised fit).

    ``included_only=False`` counts an obs even when THIS fit rejected it -- its
    full (large) residual then enters chi2, so a dropped obs reads as the
    degradation it is (used for the seed-preservation and new-tracklet gates,
    which must see rejected obs)."""
    chi2 = 0.0
    n = 0
    for oid, r, line in _subset_obs(fit, obs_ids, included_only=included_only):
        sig = (sigma_map.get(oid) if sigma_map is not None
               else _obs_sigmas(r, line, station_floor))
        if sig is None:
            continue
        chi2 += (r["d_ra_arcsec"] / sig[0]) ** 2 + (r["d_dec_arcsec"] / sig[1]) ** 2
        n += 1
    return float(chi2), n


def worst_mag_residual(fit, jd_subset=None):
    """Largest |observed - computed| magnitude residual (mag) over the fit's
    INCLUDED photometry, optionally restricted to obs whose ``jd_utc`` (rounded
    to 1e-5 d) is in ``jd_subset``. The computed magnitude is the fit's H/G
    phase-reduced prediction, so a linked tracklet ~1.5 mag off the object's
    brightness (a different body) stands out. Returns (dmag, obs_mag) of the
    worst obs, or (0.0, None) when no photometry is available."""
    try:
        rows = fit.magnitude_residuals()
    except Exception:
        return 0.0, None
    worst = 0.0
    worst_mag = None
    for r in rows:
        if r.get("used") is False:
            continue
        if jd_subset is not None and round(r["jd_utc"], 5) not in jd_subset:
            continue
        dm = r.get("dmag")
        if dm is None:
            continue
        if abs(dm) > worst:
            worst = abs(dm)
            worst_mag = r.get("obs_mag")
    return worst, worst_mag


def _mag_tracklets(rows, jd_subset=None):
    """Group photometric residual rows into tracklets -- (station, SPAN), the
    granularity the submission document itself uses (:func:`_linked_tracklet_count`)
    -- applying the gate's span convention (the first row >= SPAN_DAYS after the
    CURRENT span's start opens a new one; see :func:`opp_profile.spans_of`).  Rejected
    obs and obs without a magnitude are dropped; ``jd_subset`` (jd rounded to
    1e-5 d) restricts to a subset as :func:`worst_mag_residual` does."""
    by_station = {}
    for r in rows:
        if r.get("used") is False or r.get("dmag") is None:
            continue
        if jd_subset is not None and round(r["jd_utc"], 5) not in jd_subset:
            continue
        by_station.setdefault(r.get("obscode"), []).append(r)
    out = []
    for station, obs in by_station.items():
        obs.sort(key=lambda r: r["jd_utc"])
        group, start = [], obs[0]["jd_utc"]
        for r in obs:
            if r["jd_utc"] - start >= SPAN_DAYS:
                out.append((station, group))
                group, start = [], r["jd_utc"]
            group.append(r)
        out.append((station, group))
    return out


def mag_outlier_tracklet(fit, floor_mag, p_max=0.05, jd_subset=None):
    """Is any TRACKLET's photometry significantly off the object's predicted
    brightness?  Returns the worst offender as a dict (or ``None``).

    The question a single-observation threshold cannot ask.  One ob 0.8 mag
    high says almost nothing -- survey photometry scatters that far routinely --
    whereas a whole tracklet sitting 0.8 mag high says the light curve does not
    belong to this body.  So a tracklet warns only when BOTH hold:

      * its mean |dmag| reaches ``floor_mag`` -- the effect is large enough to
        matter photometrically (a tight survey with a 0.13 mag zero-point
        offset is statistically certain and physically uninteresting), and
      * a two-tailed one-sample t-test of its residuals against zero gives
        ``p < p_max`` -- the offset is real, not one bright ob dragging a mean.
        Residuals of 0.9/0.2/0.1 mag average 0.4 but scatter far too widely to
        claim the tracklet is off at all (p = 0.25), so they stay quiet.

    No t-test is attempted where there is nothing to test.  A lone observation
    -- or a real tracklet carrying only ONE magnitude, since ``n`` counts
    MAGNITUDES and not observations -- has no scatter and zero degrees of
    freedom, so it falls back to the flat ``floor_mag`` threshold strictly,
    which is exactly the test that applied to every ob before.  Ranking is by
    significance first (p), then by size, so the strongest evidence surfaces
    rather than the biggest number.

    Keys: ``station``, ``jd``, ``n`` (magnitudes, not obs), ``mean``, ``sd``,
    ``p`` (None when n == 1), ``worst``."""
    try:
        rows = fit.magnitude_residuals()
    except Exception:
        return None
    from scipy import stats
    best = None
    for station, group in _mag_tracklets(rows, jd_subset):
        dm = np.array([float(r["dmag"]) for r in group])
        mean = float(dm.mean())
        if abs(mean) < floor_mag:
            continue
        if dm.size > 1:
            sd = float(dm.std(ddof=1))
            if sd == 0.0:               # identical residuals: infinitely significant
                p = 0.0
            else:
                t = mean / (sd / np.sqrt(dm.size))
                p = float(2.0 * stats.t.sf(abs(t), dm.size - 1))
            if p >= p_max:
                continue
        else:
            sd, p = 0.0, None           # no dof: the flat |dmag| >= floor test
        cand = {"station": station, "jd": float(group[0]["jd_utc"]),
                "n": int(dm.size), "mean": mean, "sd": sd, "p": p,
                "worst": float(np.abs(dm).max())}
        # most significant first (a single ob, p None, ranks below any tested
        # tracklet), then largest offset
        key = (0.0 if p is None else -1.0, p or 0.0, -abs(mean))
        if best is None or key < best[0]:
            best = (key, cand)
    return None if best is None else best[1]


def drop_offending(fit, obs80_text, thresh_arcsec, *, sigma_mult=None,
                   min_tracklet_obs=0):
    """Return ``obs80_text`` with every line whose residual against ``fit``
    exceeds ``thresh_arcsec`` removed (the native replacement for the old
    residualExamine.dropOffendingObsThresh).  Lines the fit doesn't know
    (never parsed) are kept.

    ``sigma_mult`` adds the test the flat threshold cannot make: an observation is
    only clipped if it ALSO misses by more than ``sigma_mult`` times ITS OWN
    uncertainty (per-axis sigma, raised to the Veres et al. (2017) station
    floor so an over-tight self-reported sigma cannot manufacture a rejection).

    Why both, and why AND.  A flat arcsec cut asks the wrong question of a
    noisy station: 0.5" is a blunder from a 0.15"-sigma survey and utterly
    ordinary from a 1"-sigma one, so the flat test alone clips perfectly good
    observations -- which is what happened to 2024 PQ7, whose linkage is real
    (the MPC accepted it) and which nonetheless lost three of its nine W76
    observations to a 0.5" cut.  Requiring BOTH tests keeps this strictly more
    permissive than the flat threshold alone: nothing that survives today can
    be clipped because of this argument, and the over-tight-sigma failure mode
    (where a pure sigma test would clip MORE) cannot bite either.

    ``min_tracklet_obs`` clips by TRACKLET rather than by loose point: whatever
    a station+night group is cut down to below it goes whole.  A tracklet
    pruned to one observation is not a thinner tracklet -- one point fixes no
    motion, so it can neither be checked against the corridor that found it nor
    submitted as an identification, and 2024 RL188 came back as a whole night's
    linkage on the strength of the single survivor of a three-obs tracklet.
    Grouping by station AND span (the shared opp_profile.SPAN_DAYS unit) means a junk
    tracklet still clips without touching its healthy siblings.  Pass 0 -- the
    default, and what the seed side always uses -- to clip points only.
    """
    result = fit.residuals(included_only=False)
    resid, sigmas = {}, {}
    for o, r in zip(fit._obs, result):
        line = o.get("obs80_line")
        if not line:
            continue
        oid = obs_unique_id(line)
        resid[oid] = float(np.hypot(r["d_ra_arcsec"], r["d_dec_arcsec"]))
        if sigma_mult:
            sig = _obs_sigmas(r, line, True)
            if sig is not None:
                sigmas[oid] = float(np.hypot(*sig))
    kept = []
    for line in split_obs80_long_lines(obs80_text).splitlines():
        if not line.strip():
            continue
        oid = obs_unique_id(line)
        r = resid.get(oid)
        if r is None or r <= thresh_arcsec:
            kept.append(line)
            continue
        sig = sigmas.get(oid)
        if sigma_mult and sig and r <= sigma_mult * sig:
            kept.append(line)          # ordinary noise for THIS observation
    if min_tracklet_obs > 1:
        whole = set()
        by_station = {}
        for line in kept:
            by_station.setdefault(line[77:80], []).append(line)
        for group in by_station.values():
            for span in spans_of(group):
                if len(span) >= min_tracklet_obs:
                    whole.update(obs_unique_id(l) for l in span)
        kept = [l for l in kept if obs_unique_id(l) in whole]
    return "\n".join(kept) + ("\n" if kept else "")
