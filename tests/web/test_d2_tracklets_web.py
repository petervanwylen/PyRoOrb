"""digest2 on the web payload: a top-level `digest2` block on EVERY single
tracklet, determined or not, and the tracklet-by-tracklet scores for the observations table
(engine_api.d2_tracklets), offered for q < 1.8 only."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import pytest

_MVP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _fixtures as FX                       # noqa: E402
import pyroorb.web.engine_api as E                       # noqa: E402
import pyroorb.fit_object as fo                      # noqa: E402
from pyroorb import obs_sources                           # noqa: E402

pytestmark = pytest.mark.slow


def _c46_fit():
    d = json.load(open(os.path.join(_MVP, "tests", "range_fixtures", "C46MAH1.json")))
    return fo.fit_orbit(d["obs"], obs_sources.load_obscodes(), desig="C46MAH1", meta=d["meta"])


def test_a_determined_tracklet_still_carries_digest2():
    """C46MAH1's distance IS measured, so there is no banner -- the case that used to show no
    digest2 at all."""
    p = E.serialise_fit(_c46_fit())
    assert not p["orbit_suppressed"] and p["constraint"] is None
    d2 = p["digest2"]
    assert d2 and d2["shown"] and d2["shown_noid"] and d2["scores"]["NEO"] > 90
    assert p["d2_tracklets"]["eligible"]             # q ~ 0.71


def test_tracklet_scores_endpoint():
    fit = _c46_fit()
    sid = E._store_session(fit)
    r = E.d2_tracklets(sid)
    assert r["ok"] and not r["truncated"]
    assert [(t["first"], t["station"], t["n"]) for t in r["tracklets"]] == [(0, "703", 12), (12, "Q06", 3)]
    assert all("noid_scores" in t and "shown" in t for t in r["tracklets"])
    assert E.d2_tracklets("no-such-session")["expired"]


def test_a_long_arc_main_belt_orbit_gets_neither():
    fx = FX.load_fixture(os.path.join(FX.FIX, "RANGEAMBIG_3NIGHT_MBA.json"))
    fit = fo.fit_orbit(fx["obs"], obs_sources.load_obscodes(), desig="x", meta=fx["meta"])
    p = E.serialise_fit(fit)
    if not p["orbit_suppressed"]:
        assert p["digest2"] is None
    assert not p["d2_tracklets"]["eligible"]
    r = E.d2_tracklets(E._store_session(fit))
    assert not r["ok"] and "q < 1.8" in r["error"]
