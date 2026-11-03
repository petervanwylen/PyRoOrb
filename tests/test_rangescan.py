"""The range profile (rangescan.py) and the single-tracklet exception it decides.

The measured claim behind all of this: on a SINGLE TRACKLET the
covariance is not evidence in either direction -- 24 of 187 fixture tracklets came back silent
and 17 of those were wrong by more than 6% in distance, one by 34x at 57 sigma of its own
error. The profile asks the question the arc actually poses -- over what distances can this
astrometry be fitted at all -- and only a narrow, single, interior, actually-fitting answer is
allowed to show an orbit.
"""
import glob
import os

import numpy as np
import pytest

from _fixtures import load_fixture, FIX
import pyroorb.fit_object as fo
from pyroorb import obs_sources
from pyroorb import constraint
from pyroorb import rangescan


def _fixture(key):
    return load_fixture([p for p in glob.glob(os.path.join(FIX, "*.json")) if key in p][0])


def _tracklets(obs, max_min=120.0):
    o = sorted(obs, key=lambda r: r["jd_utc"])
    groups, cur = [], [o[0]]
    for r in o[1:]:
        if r.get("obscode") != cur[-1].get("obscode") or \
           (r["jd_utc"] - cur[-1]["jd_utc"]) * 1440.0 > max_min:
            groups.append(cur)
            cur = [r]
        else:
            cur.append(r)
    groups.append(cur)
    return groups


def _best_tracklet(obs):
    return max(_tracklets(obs), key=lambda t: (len(t), t[-1]["jd_utc"] - t[0]["jd_utc"]))


def _fit_best_tracklet(key):
    fx = _fixture(key)
    tr = _best_tracklet(fx["obs"])
    return fo.fit_orbit(tr, obs_sources.load_obscodes(),
                        desig=fx["meta"]["designation"], meta=fx["meta"]), tr


# --- the verdict logic itself: no fit, no integration -------------------------------------
def _prof(rows, r_lo=0.001, r_hi=6.0, dof=10):
    return {"rows": [dict(rho=r, chi2=c, a=1.0, e=0.3, q=0.7, bound=b, state=None)
                     for r, c, b in rows],
            "r_lo": r_lo, "r_hi": r_hi, "n_obs": 8, "dof": dof}


def test_verdict_needs_one_narrow_interior_fitting_interval():
    """Each clause refuses on its own, and each of them refused a real object in the census."""
    sharp = [(0.001, 900, True), (0.01, 100, True), (0.1, 1.0, True),
             (0.11, 2.0, True), (0.2, 400, True), (1.0, 900, True)]
    assert rangescan.verdict(_prof(sharp))["determined"] is True

    # two separated distance solutions -- the aliasing a covariance structurally cannot see
    two = [(0.01, 1.0, True), (0.05, 500, True), (0.1, 1.5, True), (1.0, 900, True)]
    v = rangescan.verdict(_prof(two))
    assert not v["determined"] and "separated" in v["reason"]

    # nothing fits anywhere: one node inside delta-chi2 reads as width ZERO, which is
    # "uniformly terrible" wearing the costume of "sharply determined" (107474, 162385)
    bad = [(0.01, 900, True), (0.1, 800, True), (1.0, 1200, True)]
    v = rangescan.verdict(_prof(bad))
    assert not v["determined"] and "no orbit fits" in v["reason"]

    # an interval that runs into the admissible boundary was not closed by the data (13923)
    edge = [(0.001, 1.0, True), (0.01, 2.0, True), (1.0, 900, True)]
    v = rangescan.verdict(_prof(edge, r_lo=0.001))
    assert not v["determined"] and "edge" in v["reason"]

    # ...and a delivered orbit outside the distances the data support is a finding about the
    # fit, not a licence to show it
    v = rangescan.verdict(_prof(sharp), rho_fit=0.5)
    assert not v["determined"] and "outside" in v["reason"]
    assert rangescan.verdict(_prof(sharp), rho_fit=0.105)["determined"] is True


def test_unbound_nodes_do_not_widen_the_interval():
    """An unbound solution fits a one-hour arc trivially, so a pure chi2 profile is flat toward
    large rho for EVERY tracklet. Those nodes are excluded by physics, not by the data."""
    rows = [(0.01, 1.0, True), (0.02, 2.0, True), (0.1, 1.2, False), (1.0, 1.5, False)]
    v = rangescan.verdict(_prof(rows))
    assert v["rho_hi"] <= 0.02, v
    assert rangescan.allowed(_prof(rows), bound_only=False)[0][0][1] > 0.02


def test_a_profile_that_could_not_run_is_a_verdict_and_not_a_silence():
    """`None` from the profile meant "I could not judge this" and `constraint.assess`
    read it as "no objection", falling back on the covariance -- the method whose record on
    this population (24 shown, 17 wrong) is why the profile exists. A failed profile is now a
    profile with no rows, and its verdict is NOT determined, in the failure's own words."""
    v = rangescan.verdict(rangescan.failed("no trial orbit could be fitted", n_obs=7))
    assert v["determined"] is False and v["profiled"] is False
    assert "could not be measured" in v["reason"] and "no trial orbit" in v["reason"]
    assert v["n_intervals"] == 0 and v["rho_lo"] is None and v["rho_best"] is None
    assert v["n_obs"] == 7 and v["dof"] == 8
    sharp = [(0.001, 900, True), (0.01, 100, True), (0.1, 1.0, True), (1.0, 900, True)]
    assert rangescan.verdict(_prof(sharp))["profiled"] is True


def test_the_edge_sentence_names_the_limit_it_actually_hit():
    """"The edge of what is physically possible" was said for the 6 au SCAN CAP -- a sampling
    constant for the NEO/MBA regime -- on 95 of 187 fixture tracklets, nearly all of them
    objects at 0.01-1 au. The sentence now says which limit closed the interval: the scan's
    own cap, the reach of a bound orbit (the escape geometry, short of the cap), or the inner
    floor."""
    lo = [(0.001, 1.0, True), (0.01, 2.0, True), (1.0, 900, True)]
    v = rangescan.verdict(_prof(lo, r_lo=0.001))
    assert v["edge"] and not v["capped"] and "inner edge" in v["reason"], v["reason"]

    hi = [(0.001, 900, True), (0.1, 2.0, True), (6.0, 1.0, True)]
    prof = _prof(hi, r_hi=6.0)
    prof.update(r_max=6.0, capped=True)          # the admissible interval ran into r_max
    v = rangescan.verdict(prof)
    assert v["edge"] and v["capped"] and "6 au limit of the scan" in v["reason"], v["reason"]
    assert "physically" not in v["reason"]

    prof.update(r_max=300.0, capped=False)       # ...or closed on its own, short of the cap
    v = rangescan.verdict(prof)
    assert v["edge"] and not v["capped"] and "bound orbit can reach" in v["reason"], v["reason"]


# --- end to end, on real astrometry --------------------------------------------------------
@pytest.mark.slow
def test_a_tracklet_the_profile_cannot_seed_is_refused_through_the_profile():
    """1998 VD32: 7 observations over 8 minutes at 0.0046 au. The Lambert transfer that seeds
    each profile node degenerates over so short a span -- every trial is None or plunges
    through the Sun -- so a COLD profile has no rows, though the delivered orbit fits the
    tracklet at 0.33". The one fixture of 187 that reached the `None` exit.

    Since the profile takes the delivered orbit as a known solution (2026-10-02) the fit's own
    profile has a node to grow from and is RUN -- and still refuses, now on the measurement
    (distances from ~0.0003 to ~0.6 au fit) instead of on the seeding. Both halves are pinned:
    the cold profile's failure must still be named, and the fit must still be refused through
    the profile, never handed to the covariance."""
    import numpy as np
    fit, tr = _fit_best_tracklet("1998_VD32")
    assert len(tr) == 7 and (tr[-1]["jd_utc"] - tr[0]["jd_utc"]) * 1440.0 < 10.0
    mask = np.asarray(fit._mask, bool)
    sub = [o for o, k in zip(fit._obs, mask) if k]
    sig = np.column_stack([np.asarray(fit._ctx["sigma_ra"])[mask],
                           np.asarray(fit._ctx["sigma_dec"])[mask]])
    cold = rangescan.verdict(rangescan.profile(sub, sig))
    assert cold["profiled"] is False and cold["determined"] is False
    assert "no trial orbit" in cold["reason"], cold["reason"]

    v = fit.range_verdict()
    assert v is not None and v["profiled"] is True and v["determined"] is False, v
    c = constraint.assess(fit)
    assert c is not None and c["method"] == "range", c and c["method"]
    assert c["range"]["profiled"] is True


@pytest.mark.slow
def test_a_close_neo_tracklet_can_determine_its_orbit():
    """2004 FH: 51 observations over 96 minutes of an object 0.0026 au away. The diurnal
    parallax over that baseline PINS the distance, the profile says so with a single narrow
    interior interval, and the orbit is shown -- a = 0.7778 against the full arc's 0.7805.
    This is the exception the whole instrument exists to allow."""
    fit, tr = _fit_best_tracklet("2004_FH")
    assert len(tr) > 20 and (tr[-1]["jd_utc"] - tr[0]["jd_utc"]) * 1440 > 60
    v = fit.range_verdict()
    assert v is not None and v["determined"], v
    assert v["n_intervals"] == 1 and not v["edge"]
    assert v["rho_lo"] <= fit.geocentric_range() <= v["rho_hi"]
    assert v["rel_width"] < rangescan.MAX_REL_WIDTH
    assert constraint.assess(fit) is None, "a determined tracklet must show its orbit"


@pytest.mark.slow
def test_a_wrong_basin_tracklet_is_refused_where_the_covariance_was_silent():
    """2001 UQ16: 6 observations over 94 minutes. The fit converges to 0.011 au with sigma_a
    0.037 and the covariance calls it determined; the full arc says 0.38 au -- 34x out, 57
    sigma, i 5.6 deg against 30.8. The profile finds every distance from 0.005 to 0.43 au fits
    as well, and refuses."""
    fit, _ = _fit_best_tracklet("2001_UQ16")
    v = fit.range_verdict()
    assert v is not None and not v["determined"], v
    assert v["rel_width"] > 1.0, v
    c = constraint.assess(fit)
    assert c is not None and c["method"] == "range", c
    assert "distances" in (c["range"]["reason"] or ""), c["range"]
    assert c["range"]["reason"] in c["detail"], "the reason has to reach the reader"


@pytest.mark.slow
def test_the_profile_is_not_consulted_past_a_single_tracklet():
    """Scope, and cost: a second night pins the range by curvature, the covariance is the
    honest report there, and a profile per web payload would be seconds of integration for a
    question nobody asked."""
    fx = _fixture("2023_DW")
    fit = fo.fit_orbit(fx["obs"], obs_sources.load_obscodes(),
                       desig=fx["meta"]["designation"], meta=fx["meta"])
    calls = []
    real = fo.FittedOrbit.range_profile
    fo.FittedOrbit.range_profile = lambda self, *a, **k: calls.append(1) or real(self, *a, **k)
    try:
        assert constraint.assess(fit) is None
    finally:
        fo.FittedOrbit.range_profile = real
    assert not calls, "the range profile was run on a multi-apparition arc"


@pytest.mark.slow
def test_a_narrow_valley_is_found_and_the_fit_is_inside_it():
    """C46MAH1 (NEOCP, 2026-10-02): 15 observations from Catalina (703) and Siding Spring (Q06)
    of an object 0.0085 au away -- two-station parallax pins the distance to a fraction of a
    percent. The profile's one fine pass stepped ~8%, so a single node landed on the WALL of a
    +/-0.4% valley, its interval came out 0.02% wide beside the delivered orbit, and the banner
    said "the delivered orbit sits outside the distances the data support" for a fit that was
    the BETTER solution on the profile's own observations (thinned chi2 10.5 against 17.0).

    Two fixes, both pinned: the profile zooms until the valley holds several nodes, and it takes
    the delivered orbit as a known solution, so its minimum can never be worse than the fit's."""
    import json, os
    import numpy as np
    d = json.load(open(os.path.join(os.path.dirname(__file__), "range_fixtures", "C46MAH1.json"), encoding="utf-8"))
    fit = fo.fit_orbit(d["obs"], obs_sources.load_obscodes(), desig="C46MAH1", meta=d["meta"])
    v = fit.range_verdict()
    assert v["determined"], v["reason"]
    assert v["rho_lo"] <= fit.geocentric_range() <= v["rho_hi"]
    assert 1e-3 < v["rel_width"] < 0.05, v["rel_width"]          # resolved, not one node wide
    prof = fit.range_profile()
    lim = v["chi2_min"] + rangescan.DCHI2
    assert sum(1 for r in prof["rows"] if r["chi2"] <= lim) >= rangescan.ZOOM_NODES
    assert constraint.assess(fit) is None, "a distance this well measured must show its orbit"
