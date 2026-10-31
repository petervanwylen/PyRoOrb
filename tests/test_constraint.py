"""Offline tests for the unconstrained-orbit banner (constraint.py).

The banner is decided by the covariance geometry alone (s_max); Statistical Ranging is
consulted only on a single-TRACKLET arc and only for its mode count. Guarded here:
  * mc._bound_fraction counts UNBOUND state clones (energy sign) rather than dropping them
    at element conversion -- the bug where a 90%-unbound family reported ~100% bound.
  * a genuine single tracklet (5 obs over 2 h of a main-belt object) fires via the
    COVARIANCE -- the local covariance does not hide a free range, it reports it -- and the
    (light-timed, sigma-accepted) SR family of that tracklet is honestly wide.
  * a short CURVED arc is silent: 2021 NX4 (8 obs / 6 d, a = 2.527 +/- 0.0018) and the
    3-night MBA (a = 3.17 +/- 0.15) were both banned "essentially unconstrained" by an SR
    family whose trial builder ignored light-time (2026-08-29); neither may fire again.
  * the full multi-opposition 2023 DW arc is well determined -> no banner.
  * SR ranges the INCLUDED observations only, and its family is cached across the fresh
    FittedOrbit each interactive re-solve builds (the 11-s-per-web-click bug).
  * _sr_modes' branch logic: a 2-mode single tracklet fires alone; a 2-mode result on a
    multi-day arc is never consulted; a wide 1-mode family never fires (spread is not a verdict).
"""
import pytest
import os, glob
import numpy as np
from _fixtures import load_fixture, FIX
import pyroorb.fit_object as fo
from pyroorb import obs_sources
from pyroorb import constraint
from pyroorb import mc

# constrained refits of a cached fixture
pytestmark = pytest.mark.slow

_OC = None
def _oc():
    global _OC
    if _OC is None:
        _OC = obs_sources.load_obscodes()
    return _OC


def _fixture(key):
    return load_fixture([p for p in glob.glob(os.path.join(FIX, "*.json")) if key in p][0])


def test_bound_fraction_counts_unbound_states():
    """mc._bound_fraction is the energy-sign fraction over the raw STATE clones: an unbound
    clone must be COUNTED as unbound, not silently dropped (the old element-row estimate
    reported ~1.0 for a family that was mostly hyperbolic)."""
    k = 0.01720209895
    mu = k * k
    r = 1.0
    v_circ = (mu / r) ** 0.5
    bound = [r, 0, 0, 0, 0.9 * v_circ, 0]      # below circular speed -> eps < 0 (bound)
    unbound = [r, 0, 0, 0, 2.0 * v_circ, 0]    # above escape speed   -> eps >= 0 (unbound)
    states = np.array([bound, bound, bound, unbound])   # 3 bound, 1 unbound
    bf = mc._bound_fraction(states)
    assert abs(bf - 0.75) < 1e-9, f"bound_fraction {bf} != 0.75"
    # an all-hyperbolic family must read 0.0, not ~1.0
    assert mc._bound_fraction(np.array([unbound, unbound])) == 0.0


def _first_night(fx, window_days):
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    return [o for o in obs if o["jd_utc"] <= obs[0]["jd_utc"] + window_days]


def test_single_tracklet_fires_via_the_range_profile_and_sr_family_is_wide():
    """One 2-hour tracklet (5 obs) of a main-belt object: no curvature, the range is free.

    The verdict on a single tracklet is the RANGE PROFILE's (rangescan.py) since 2026-09-08 --
    the covariance is not evidence in either direction there, see test_rangescan.py -- and it
    must refuse this one, in distances. The SR family, asked separately, must still be honestly
    wide (sigma_a of order 1 au) with many accepted orbits: the regime SR exists for."""
    fx = _fixture("RANGEAMBIG_3NIGHT_MBA")
    night = _first_night(fx, 0.5)
    assert 4 <= len(night) <= 8 and (night[-1]["jd_utc"] - night[0]["jd_utc"]) < 0.2, len(night)
    fit = fo.fit_orbit(night, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    c = constraint.assess(fit)
    assert c is not None, "single-tracklet arc should be flagged"
    assert c.get("method") == "range", f"expected a range-profile banner, got {c.get('method')!r}"
    assert c["range"] and c["range"]["reason"], c
    assert c["range"]["rel_width"] > constraint_rangescan().MAX_REL_WIDTH, c["range"]
    assert "a" in c["unconstrained"], c["unconstrained"]
    sru = fit.sr_uncertainty()
    assert sru and sru["n"] >= constraint.SR_MIN_TRUST, sru and sru["n"]
    a_med, a_sig = sru["sigmas"]["a"]
    assert a_sig > 0.5, f"SR family of a free-range tracklet should be wide: a = {a_med} +/- {a_sig}"


def constraint_rangescan():
    from pyroorb import rangescan
    return rangescan


def test_full_arc_is_not_flagged():
    """The full multi-opposition 2023 DW arc IS determined -> no banner (and SR isn't even
    attempted: the arc is long / multi-apparition)."""
    fx = _fixture("2023_DW")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    c = constraint.assess(fit)
    assert c is None, f"well-determined arc should not get a banner: {c}"


def test_loose_but_determined_3night_arc_is_silent():
    """The 3-night MBA (11 obs over 5.08 d) is LOOSE -- a = 3.17 +/- 1.52 (48%), U ~ 11.4 --
    but the banner stays SILENT, which is the subject of this test: `assess` asks whether the
    orbit's SCALE is free (the worst-determined direction as a fraction of the orbit's own
    size), and a loose semimajor axis is not that. It was the fixture that "proved" SR must
    upgrade silent verdicts on arcs up to 6 days; that family (a free by 22x) was the
    light-time artefact. Silent now, by design.

    The sigma below was +/- 0.15 (4.7%) until 2026-09-22, and the OLD number was the wrong
    one: `FittedOrbit._param_inverse` scaled the covariance by the fit's own chi2/dof with no
    floor, and this arc's s2 is ~0.01, so its uncertainty was shrunk ~10x. Flooring it agrees
    with what the astrometry actually allows -- a parametric bootstrap of these same 11
    observations, resampled at their own reported sigma, gives a = 3.31 +/- 6.65 (201%,
    p5-p95 1.64-11.69 au), i.e. WIDER still than the floored covariance. 48% is the honest
    reading of a 5-day arc, and the docstring's own "+/-10% of the true RANGE" is not in
    tension with it: a short arc can pin the distance and still say nothing about the period.

    Kept as a drift guard on the fixture, re-baselined rather than removed."""
    fx = _fixture("RANGEAMBIG_3NIGHT_MBA")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    a, sa = fit.semimajor_axis()
    assert 0.3 < sa / a < 0.7, f"fixture drifted: a = {a} +/- {sa}"
    assert constraint.assess(fit) is None, constraint.assess(fit)


def test_short_curved_arc_2021_nx4_is_silent():
    """2021 NX4: 8 obs over 5.96 days at 1.03 au, rms ~0.1", a = 2.527 +/- 0.0018 (0.07%),
    s_max ~ 7e-6. Reported defect: U = 7.2 yet banned "essentially
    unconstrained". The SR family that fired had a best trial 11.6" off the data. No banner."""
    fx = _fixture("2021_NX4")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    a, sa = fit.semimajor_axis()
    assert fit.rms_residual() < 0.3 and sa / a < 0.01, (fit.rms_residual(), a, sa)
    assert constraint.assess(fit) is None, constraint.assess(fit)


def test_range_profile_is_computed_once_per_mask_on_the_shared_ctx():
    """The single-tracklet verdict now runs a range PROFILE, which carries the same hazard SR
    did: every interactive re-solve builds a FRESH FittedOrbit, so a per-instance cache never
    hits and the profile would re-run (seconds of integration) on every web click. Drive the
    web path -- a damped step on a single-night mask, twice -- and require exactly one."""
    fx = _fixture("2023_DW")
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    fit = fo.fit_orbit(obs, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    ctx = fit._ctx
    t = np.array([o["jd_utc"] for o in ctx["obs"]])
    mask = t <= t.min() + 0.3
    from pyroorb import rangescan
    seen, real = [], rangescan.profile
    rangescan.profile = lambda *a, **kw: (seen.append(1) or real(*a, **kw))
    try:
        state, result = ctx["ev"].step(np.asarray(fit._state, dtype=float), mask)
        step1 = fo._build_orbit(ctx, state, mask, result, fit._history)
        constraint.assess(step1)
        state2, res2 = ctx["ev"].step(np.asarray(step1._state, dtype=float), mask)
        step2 = fo._build_orbit(ctx, state2, mask, res2, fit._history)
        constraint.assess(step2)
    finally:
        rangescan.profile = real
    assert len(seen) == 1, f"the range profile ran {len(seen)} times for one mask"


def test_sr_ranges_the_included_obs_only_and_caches():
    """SR must be shown the arc the CALLER gated on -- the INCLUDED observations -- and must not
    re-run for every fresh FittedOrbit.

    The web UI deselecting one apparition of a two-apparition object is the case that broke:
    the banner code asked about the included arc (one night, 1 opposition) while sr_uncertainty ranged
    ``ctx["obs"]`` -- ALL the astrometry, spanning 15 years. Every two-body trial was then
    integrated across the full span (~11 s per interactive payload, on every click, because each
    re-solve builds a new FittedOrbit and the cache was per-instance), and the resulting family
    was too small to pass SR_MIN_TRUST -- so the banner was silent about an arc that genuinely
    is unconstrained. Here: a full multi-opposition fit, re-solved on a single-night mask exactly
    as the web Full step does, must hand SR the 9 included obs and cache the answer.
    """
    fx = _fixture("2023_DW")
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    fit = fo.fit_orbit(obs, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    ctx = fit._ctx
    t = np.array([o["jd_utc"] for o in ctx["obs"]])
    mask = t <= t.min() + 0.3                      # the discovery night only
    n_incl = int(mask.sum())
    assert 3 <= n_incl < len(t), f"fixture mask is degenerate: {n_incl}/{len(t)}"

    import pyroorb.sr as _sr
    seen, real = [], _sr.statistical_ranging
    def spy(sub, times, *a, **kw):
        seen.append(len(sub))
        return real(sub, times, *a, **kw)
    _sr.statistical_ranging = spy
    try:
        # the web path: one damped step on the client's mask, then a fresh FittedOrbit
        state, result = ctx["ev"].step(np.asarray(fit._state, dtype=float), mask)
        step1 = fo._build_orbit(ctx, state, mask, result, fit._history)
        step1.sr_uncertainty()
        # ...and the NEXT click, a brand-new FittedOrbit on the same mask
        state2, res2 = ctx["ev"].step(np.asarray(step1._state, dtype=float), mask)
        step2 = fo._build_orbit(ctx, state2, mask, res2, fit._history)
        step2.sr_uncertainty()
        c = constraint.assess(step2)
    finally:
        _sr.statistical_ranging = real

    assert seen, "SR was never consulted on a single-night masked arc"
    assert all(n == n_incl for n in seen), \
        f"SR ranged {seen} observations, expected the {n_incl} INCLUDED ones"
    assert len(seen) == 1, f"SR re-ran on the second interactive re-solve ({len(seen)} calls)"
    # The DW discovery night is a close (0.07 au) NEO whose 4 h of diurnal parallax PIN the
    # range (a = 0.834 +/- 0.004): the covariance is rightly silent and SR, with an honest
    # sigma-scaled acceptance, finds no family it can trust -- so no banner. What this test
    # guards is the plumbing above: SR asked once, on the included obs.
    assert c is None or c.get("method") in ("cov", "sr-modes"), c


def test_sr_cache_is_keyed_by_mask_on_one_instance_too():
    """A SECOND question on the same FittedOrbit must not be answered with the FIRST one's
    family.  ``_mask`` is not frozen -- the rejection loop mutates it in place -- and the
    shared per-ctx cache is keyed by that mask, so an unkeyed per-INSTANCE fallback beside it
    would silently return the previous mask's ranging: the shared lookup misses on the new key,
    then the instance cache hits.  Ask twice on one object, changing the mask between, and
    require the two answers to describe the two different arcs."""
    fx = _fixture("2023_DW")
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    fit = fo.fit_orbit(obs, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    t = np.array([o["jd_utc"] for o in fit._obs])
    night = t <= t.min() + 0.3                      # the discovery night alone
    assert 3 <= int(night.sum()) < len(t)

    import pyroorb.sr as _sr
    seen, real = [], _sr.statistical_ranging
    _sr.statistical_ranging = lambda sub, times, *a, **kw: (
        seen.append(len(sub)) or real(sub, times, *a, **kw))
    try:
        fit._mask = night
        fit.sr_uncertainty()
        fit._mask = np.ones(len(t), dtype=bool)     # the mask CHANGES under the same object
        fit.sr_uncertainty()
    finally:
        _sr.statistical_ranging = real

    assert len(seen) == 2, \
        f"the second mask reused the first mask's cached family (SR calls: {seen})"
    assert seen[0] != seen[1], \
        f"both calls ranged the same {seen[0]} obs -- the mask was not honoured"


class _FakeFit:
    """Minimal FittedOrbit stand-in exercising _sr_modes' branch logic without a real fit.
    Only the handful of methods _sr_modes / _banner / _cov_assess touch are implemented;
    state()/covariance_matrix() raise, so _cov_assess returns None (silent covariance).
    `range_verdict` is the profile's answer when given; with none the stub RAISES, which is
    what an unrunnable profile looks like to `assess`."""
    def __init__(self, n_modes, sr_sigmas=None, a_nom=2.5, arc=0.1, n=30, bf=1.0,
                 range_verdict=None):
        self._modes, self._a, self._arc, self._n, self._bf = n_modes, a_nom, arc, n, bf
        self._sig = sr_sigmas or {"a": (2.5, 0.3), "e": (0.2, 0.05)}
        self._rv = range_verdict
    def range_verdict(self):
        if self._rv is None:
            raise RuntimeError("no profile on this stub")
        return self._rv
    def arc_length_days(self): return self._arc
    def n_oppositions(self): return 1
    def geocentric_range(self): return 1.2
    def elements(self): return {"a": self._a}
    def uncertainty_parameter(self): return 5.0
    def state(self): raise RuntimeError("no state")
    def covariance_matrix(self, kind): raise RuntimeError("no covariance")
    def sr_uncertainty(self):
        return {"sigmas": self._sig, "n": self._n, "bound_fraction": self._bf,
                "n_modes": self._modes, "ranges": {"a": (self._a * 0.5, self._a * 3.0)}}


def test_a_tracklet_the_profile_cannot_judge_is_suppressed_through_the_profile():
    """This stub used to reach the SR-modes / covariance branches BECAUSE its profile
    raised: `_range_verdict` swallowed the exception into None and `assess` read None as "no
    objection". On a single tracklet the profile is the whole verdict, and "could not run" is
    one of its answers -- not-determined, `profiled` False, the cause in the detail -- never a
    hand-off to the two methods it replaced. (Which also means the SR-modes-alone banner is no
    longer reachable on a real fit: SR is consulted only on arcs the profile answers for.)"""
    c = constraint.assess(_FakeFit(n_modes=2))
    assert c is not None and c["method"] == "range", c and c["method"]
    assert c["range"]["profiled"] is False and c["range"]["n_intervals"] == 0, c["range"]
    assert "could not be measured" in c["detail"] and "RuntimeError" in c["detail"], c["detail"]


def test_sr_is_not_consulted_past_a_tracklet():
    """A multi-day arc has curvature; the covariance answers and SR is not asked, so even a
    2-mode SR result (which sr_uncertainty would not be called to produce) cannot fire."""
    assert constraint._sr_modes(_FakeFit(n_modes=2, arc=4.0)) is None
    assert constraint.assess(_FakeFit(n_modes=2, arc=4.0)) is None


def test_wide_sr_spread_never_fires():
    """A wide SR family (a free by 6x) with ONE mode is not a verdict: the spread of a
    two-body sampler is not an uncertainty estimate and never overrides a silent covariance.
    (Before 2026-08-29 exactly this fired the banner.)"""
    sig = {"a": (2.5, 6.0), "Q": (3.5, 8.0), "P": (4.0, 40.0), "e": (0.27, 0.23), "q": (1.9, 2.0)}
    from pyroorb import rangescan
    sharp = [dict(rho=r, chi2=c, a=1.0, e=0.3, q=0.7, bound=True, state=None)
             for r, c in [(0.01, 900.0), (1.1, 100.0), (1.2, 1.0), (1.3, 2.0), (1.5, 400.0)]]
    measured = rangescan.verdict({"rows": sharp, "r_lo": 0.001, "r_hi": 6.0, "n_obs": 8,
                                  "dof": 10}, rho_fit=1.2)
    assert measured["determined"]
    assert constraint.assess(_FakeFit(n_modes=1, sr_sigmas=sig, range_verdict=measured)) is None


def test_suppressed_orbit_ships_ranges_and_no_nominal():
    """By design, a banner is not a caption over the numbers -- when it fires
    there ARE no numbers. The one 2-hour tracklet whose range is free must come back with
    (a) an interval per element on the verdict, (b) element rows carrying that interval and a
    NULL value/sigma in the web payload, and (c) every orbit export refusing. What it must NOT
    do is ship a nominal a with a linear sigma beside it: nobody can tell that from a measured
    one, and it will not reproduce against find_orb -- both codes land somewhere different in
    the same free valley."""
    import pyroorb.web.engine_api as engine_api
    import pyroorb.web.server as web_server

    fx = _fixture("RANGEAMBIG_3NIGHT_MBA")
    # 2.5 d, not 0.5 d (changed 2026-09-22).  The point of this test is that a
    # SUPPRESSED orbit still ships the replacement product, so it has to run on a
    # family that can BE characterised.  Once the covariance s2 floor made these
    # clouds honest the 0.5 d slice stopped qualifying -- at 5 obs over one night
    # its cloud is 99.9% unbound (14 of 20000 clones convert), and an interval
    # over those 14 would describe the extreme tail of a family that is somewhere
    # else entirely.  _ranges says so itself: "a banner with no interval at all is
    # honest too".  That case is now pinned by the test below; this one keeps
    # testing what its name says, on a slice that still has a family (bound
    # fraction 0.197, 11 element ranges).
    night = _first_night(fx, 2.5)
    fit = fo.fit_orbit(night, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])

    c = constraint.assess(fit)
    assert c is not None and c["ranges_method"] in ("mc", "sr"), c
    rngs = c["ranges"]
    assert rngs and "a" in rngs, rngs
    lo, hi = rngs["a"]
    assert lo < hi, rngs["a"]
    assert c["ranges_pct"] == list(constraint.RANGE_PCT)
    # the interval has to be an interval -- a free range that came back as a hair-width band
    # would be the covariance's overconfidence wearing the honest product's clothes
    assert (hi - lo) / max(abs(hi), abs(lo)) > 0.05, f"suspiciously tight range: {rngs['a']}"

    payload = engine_api.serialise_fit(fit)
    assert payload["orbit_suppressed"] is True
    by_name = {r["name"]: r for r in payload["elements"]}
    assert by_name, payload["elements"]
    for name, row in by_name.items():
        assert row["value"] is None and row["sigma"] is None, (name, row)
    assert by_name["a"]["range"] == rngs["a"]

    refusal = web_server._orbit_export_refusal(fit)
    assert refusal and "export refused" in refusal, refusal

    # ...and everything DERIVED from the nominal goes with it: a "PHA, MOID 0.003 au" badge or
    # a drawn ellipse is the same refused claim in a more confident-looking form.
    ph = payload["physical"]
    for k in ("class", "pha", "moid_earth_au", "tisserand_jupiter", "U", "H", "H_sigma"):
        assert ph[k] is None, (k, ph[k])
    assert payload["orbit3d"] is None
    assert ph["n_mags"] is not None and ph["frame"], ph   # what the observations say survives


def test_an_all_but_hyperbolic_cloud_ships_the_banner_with_no_interval():
    """No nominal AND no interval is an honest verdict, not a hole.

    `_ranges` reserves it in as many words -- "a banner with no interval at all is
    honest too: the UI shows the element names and no numbers" -- and the covariance
    s2 floor made it reachable: a 5-obs single night whose cloud is 99.9% unbound
    (14 of 20000 clones convert to finite elements) cannot honestly be given an
    interval, because those few are the far tail of a family that is overwhelmingly
    somewhere else.  What the user gets instead is the bound fraction, which is the
    real statement about such an arc.

    Pinned because the tempting "fix" is to force an interval out of the handful of
    clones that did convert -- that would be exactly the fake precision the floor
    exists to remove.
    """
    fx = _fixture("RANGEAMBIG_3NIGHT_MBA")
    fit = fo.fit_orbit(_first_night(fx, 0.5), _oc(),
                       desig=fx["meta"]["designation"], meta=fx["meta"])
    c = constraint.assess(fit)
    assert c is not None and c["level"] == "unconstrained", c
    assert c["ranges_method"] is None and not c.get("ranges"), c
    assert c.get("detail"), "the banner must still say something about the family"


def test_determined_orbit_keeps_its_orbit_and_its_exports():
    """The other side of the rule: a determined orbit is untouched -- values, sigmas, no
    suppression flag, and the exports do not refuse. (Guards the guard: `assess` runs on every
    payload and on every export, so a broadening of it silently takes the orbit away from
    every well-determined object.)"""
    import pyroorb.web.engine_api as engine_api
    import pyroorb.web.server as web_server

    fx = _fixture("2023_DW")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    payload = engine_api.serialise_fit(fit)
    assert payload["orbit_suppressed"] is False
    by_name = {r["name"]: r for r in payload["elements"]}
    assert by_name["a"]["value"] is not None and "range" not in by_name["a"], by_name["a"]
    assert payload["physical"]["class"] and payload["orbit3d"]["path"], payload["physical"]
    assert web_server._orbit_export_refusal(fit) is None


def test_a_suppressed_tracklet_says_what_kind_of_object_it_could_be():
    """The banner on an undetermined orbit carries digest2's RAW class scores (d2score.py,
    vendored pydigest2): what an arc that cannot fix its orbit CAN say. On a main-belt
    tracklet the main-belt classes must outscore the NEO class -- the population prior doing
    the one job it is allowed (display), on the case it is built for."""
    fx = _fixture("RANGEAMBIG_3NIGHT_MBA")
    night = _first_night(fx, 0.5)
    fit = fo.fit_orbit(night, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    c = constraint.assess(fit)
    assert c is not None
    d2 = c.get("digest2")
    assert d2 and d2["flavour"] == "raw" and d2["n_obs"] == fit.n_observations(), d2
    s = d2["scores"]
    assert all(0.0 <= v <= 100.0 for v in s.values()), s
    mb = max(s.get(k, 0.0) for k in ("MB1", "MB2", "MB3", "Pho", "Pal", "Han", "Hun"))
    assert mb > s["NEO"], s
    assert all(row[2] >= 1 for row in d2["shown"]), d2["shown"]



def test_the_banner_credits_the_population_prior():
    """Say -- in one sentence -- that the orbits are weighed by a population
    prior, and credit digest2 for it."""
    fx = _fixture("RANGEAMBIG_3NIGHT_MBA")
    night = _first_night(fx, 0.5)
    fit = fo.fit_orbit(night, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    c = constraint.assess(fit)
    assert constraint.POPULATION_SENTENCE in c["detail"]
    assert "digest2" in c["detail"] and "population prior" in c["detail"]
