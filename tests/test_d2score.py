"""d2score.py: digest2 on the engine's observations -- both flavours, and tracklet by tracklet.

The grouping rule: one station's observations in one night (a 12 h window from the first);
the submitter's tracklet id (ADES trkSub) never matters."""
import json, os

from pyroorb import d2score


def _o(jd, code="F51", trk=None, ra=10.0, dec=5.0, mag=20.0):
    o = {"jd_utc": jd, "obscode": code, "ra_deg": ra, "dec_deg": dec, "mag": mag, "band": "G"}
    if trk:
        o["trksub"] = trk
    return o


def test_a_tracklet_is_one_station_in_one_night():
    h = 1.0 / 24.0
    obs = [_o(0.0), _o(1 * h), _o(5 * h), _o(11.9 * h),      # F51: one 12 h night
           _o(0.5 * h, "703"), _o(1.0 * h, "703"),            # another station, same night
           _o(1.0), _o(1.0 + h)]                              # F51 the next night
    g = d2score.tracklets(obs)
    assert g == [[0, 1, 2, 3], [4, 5], [6, 7]], g


def test_trksub_never_matters():
    """1997 AU24: one trkSub ('MH102') on three nights at 807 is three tracklets, and an arc
    of them gets no whole-object digest2."""
    nights = [2450453.8415, 2450453.8491, 2450454.7735, 2450454.8476,
              2450458.8479, 2450458.8686]
    obs = [_o(t, "807", trk="MH102") for t in nights]
    assert d2score.tracklets(obs) == [[0, 1], [2, 3], [4, 5]]
    assert d2score.arc_scores(obs) is None
    h = 1.0 / 24.0                                 # different ids, one station-night: one
    assert d2score.tracklets([_o(0.0, trk="A1"), _o(h, trk="B2")]) == [[0, 1]]


def test_input_order_does_not_matter():
    h = 1.0 / 24.0
    obs = [_o(2 * h), _o(0.0), _o(1 * h)]
    assert d2score.tracklets(obs) == [[1, 2, 0]]


def _c46():
    path = os.path.join(os.path.dirname(__file__), "range_fixtures", "C46MAH1.json")
    return json.load(open(path, encoding="utf-8"))["obs"]


def test_both_flavours_come_back():
    s = d2score.scores(_c46())
    assert s and s["flavour"] == "raw"
    for k in ("scores", "noid_scores"):
        assert set(s[k]) >= {"NEO", "MC", "MB1"} and all(0 <= v <= 100 for v in s[k].values())
    assert all(r[2] >= d2score.MIN_SHOWN for r in s["shown"] + s["shown_noid"])


def test_tracklet_scores_skip_a_lone_observation_and_index_into_the_input():
    obs = _c46()                                   # 12 x 703 then 3 x Q06, one night
    rows = d2score.tracklet_scores(obs)
    assert [(r["station"], r["n"]) for r in rows] == [("703", 12), ("Q06", 3)]
    assert all(obs[r["first"]]["obscode"] == r["station"] for r in rows)
    lone = d2score.tracklet_scores(obs[:1])
    assert lone == []


def test_arc_scores_score_one_tracklet_and_never_several_as_one():
    """digest2 is a tracklet classifier: two tracklets (here two stations on one night) are
    never scored as one object; each alone is."""
    obs = _c46()                                   # 12 x 703 then 3 x Q06, one night
    assert d2score.arc_scores(obs) is None
    one = [o for o in obs if o["obscode"] == "703"]
    s = d2score.arc_scores(one)
    assert s and s["n_obs"] == len(one) and s["scores"] == d2score.scores(one)["scores"]
