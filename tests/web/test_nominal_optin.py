"""'Show an orbit anyway' on an undetermined arc (engine_api.nominal_optin, POST /api/nominal).

Previously the banner hid every orbit, including in the 3D views. Now a suppressed payload
carries the family of orbits the arc admits (drawn faintly), and on request the session shows
ONE orbit -- on a single tracklet the most probable one given the known population
(popprior.most_probable_fit) -- with the banner kept, hazard figures blank, and every export
working for THAT orbit only; "back to the ranges" restores the suppressed fit, and so does any
re-solve, because the request lives on the one FittedOrbit.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import pytest

import _fixtures as FX                       # noqa: E402
import pyroorb.web.engine_api as E                       # noqa: E402
from pyroorb.web import server                                # noqa: E402
import pyroorb.fit_object as fo                      # noqa: E402
from pyroorb import obs_sources                           # noqa: E402

pytestmark = pytest.mark.slow


def _night_session():
    fx = FX.load_fixture(os.path.join(FX.FIX, "RANGEAMBIG_3NIGHT_MBA.json"))
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    night = [o for o in obs if o["jd_utc"] <= obs[0]["jd_utc"] + 0.5]
    fit = fo.fit_orbit(night, obs_sources.load_obscodes(), desig=fx["meta"]["designation"],
                       meta=fx["meta"])
    return fit, E._store_session(fit)


def test_a_suppressed_payload_carries_the_family_and_no_orbit():
    fit, _ = _night_session()
    p = E.serialise_fit(fit)
    assert p["orbit_suppressed"] and p["orbit3d"] is None and not p["nominal_optin"]
    fam = p["orbit_family"]
    assert fam and fam["source"] == "range" and 2 <= len(fam["orbits"]) <= 12
    assert all(len(o["path"]) > 100 and 0.0 <= o["weight"] <= 1.0 for o in fam["orbits"])
    assert max(o["weight"] for o in fam["orbits"]) == pytest.approx(1.0)
    assert server._orbit_export_refusal(fit)


def test_asking_shows_the_most_probable_orbit_and_unlocks_exports_for_it_only():
    fit, sid = _night_session()
    p = E.nominal_optin(sid)
    assert p["ok"] and not p["orbit_suppressed"] and p["constraint"], "the banner must stay"
    info = p["nominal_optin"]
    assert info["source"] == "population"
    lo, med, hi = info["rho_pct"]["5"], info["rho_med"], info["rho_pct"]["95"]
    assert 0 < lo <= med <= hi
    a = next(r for r in p["elements"] if r["name"] == "a")
    assert a["value"] is not None, a
    phys = p["physical"]
    assert phys["pha"] is None and phys["moid_earth_au"] is None and phys["tisserand_jupiter"] is None
    assert phys["class"] is not None and phys["H"] is not None
    assert p["orbit3d"] and p["orbit3d"]["path"] and p["orbit_family"]
    shown = E.session_fit(sid)
    assert shown is not fit and getattr(shown, "nominal_optin", None)
    assert abs(shown.geocentric_range() - med) / med < 0.02
    assert server._orbit_export_refusal(shown) is None          # every export, by design
    assert server._orbit_export_refusal(fit)                    # ...for THIS orbit only
    assert E.nominal_optin(sid)["nominal_optin"] == info       # idempotent

    back = E.nominal_optin(sid, on=False)
    assert back["orbit_suppressed"] and not back["nominal_optin"]
    assert E.session_fit(sid) is fit and not getattr(fit, "nominal_optin", None)


def test_a_re_solve_returns_the_page_to_the_ranges():
    fit, sid = _night_session()
    E.nominal_optin(sid)
    p = E.full_step(sid, None)
    assert p.get("ok", True) and p["orbit_suppressed"] and not p["nominal_optin"]


def test_a_determined_orbit_has_nothing_to_ask_for():
    fx = FX.load_fixture(os.path.join(FX.FIX, "2004_FH.json"))
    from _fixtures import best_tracklet
    tr = best_tracklet(fx["obs"])
    fit = fo.fit_orbit(tr, obs_sources.load_obscodes(), desig="2004 FH", meta=fx["meta"])
    sid = E._store_session(fit)
    p = E.nominal_optin(sid)
    assert not p["ok"] and "already determined" in p["error"]
    assert E.serialise_fit(fit)["orbit_family"] is None


def _two_night_session():
    """The first two nights of a main-belt object, which fit a near-parabola (a ~ 9700)."""
    fx = FX.load_fixture(os.path.join(FX.FIX, "PAIR_2014HH101.json"))
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    gaps = [i for i in range(1, len(obs)) if obs[i]["jd_utc"] - obs[i - 1]["jd_utc"] > 0.5]
    fit = fo.fit_orbit(obs[:gaps[1]], obs_sources.load_obscodes(),
                       desig=fx["meta"]["designation"], meta=fx["meta"])
    return fit, E._store_session(fit)


def test_a_near_parabolic_multi_night_arc_shows_the_population_orbit_and_its_residuals():
    fit, sid = _two_night_session()
    before = E.serialise_fit(fit)
    assert before["orbit_suppressed"]
    p = E.nominal_optin(sid)
    assert p["ok"] and not p["orbit_suppressed"] and p["constraint"], "the banner must stay"
    assert p["nominal_optin"]["source"] == "population"
    a = next(r for r in p["elements"] if r["name"] == "a")
    assert 2.0 < a["value"] < 3.2, a
    assert p["physical"]["pha"] is None and p["physical"]["moid_earth_au"] is None
    moved = [abs(x["d_ra_arcsec"] - y["d_ra_arcsec"]) for x, y in
             zip(before["residuals"], p["residuals"])]
    assert max(moved) > 1e-3, "the obs pane must follow the orbit on screen"
    back = E.nominal_optin(sid, on=False)
    assert back["orbit_suppressed"] and E.session_fit(sid) is fit


def test_an_asked_for_orbit_is_judged_by_its_arc_never_by_itself(monkeypatch):
    """A prior-chosen orbit can pass the covariance gate alone; the arc still determines nothing,
    so the banner stays and no hazard figure appears."""
    fit, sid = _two_night_session()
    real = E.constraint.assess
    monkeypatch.setattr(E.constraint, "assess", lambda f, *a, **k: real(f, *a, **k) if f is fit else None)
    p = E.nominal_optin(sid)
    assert p["constraint"] and p["nominal_optin"]
    assert p["physical"]["pha"] is None and p["physical"]["moid_earth_au"] is None
    assert p["orbit_family"]
