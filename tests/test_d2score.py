"""d2score.py: digest2 on the engine's observations -- both flavours, and tracklet by tracklet.

The grouping rule: the submitter's own tracklet id (ADES trkSub)
where an observation carries one, else one station's observations within 4 hours of the first."""
import json, os

from pyroorb import d2score


def _o(jd, code="F51", trk=None, ra=10.0, dec=5.0, mag=20.0):
    o = {"jd_utc": jd, "obscode": code, "ra_deg": ra, "dec_deg": dec, "mag": mag, "band": "G"}
    if trk:
        o["trksub"] = trk
    return o


def test_without_ids_a_tracklet_is_one_station_within_four_hours():
    h = 1.0 / 24.0
    obs = [_o(0.0), _o(1 * h), _o(3.9 * h), _o(4.1 * h),      # F51: closes 4 h after its first
           _o(0.5 * h, "703"), _o(1.0 * h, "703"),            # another station, same night
           _o(1.0), _o(1.0 + h)]                              # F51 the next night
    g = d2score.tracklets(obs)
    assert g == [[0, 1, 2], [4, 5], [3], [6, 7]], g


def test_a_trksub_groups_regardless_of_time_and_beats_the_four_hour_rule():
    h = 1.0 / 24.0
    obs = [_o(0.0, trk="A1"), _o(5 * h, trk="A1"), _o(1 * h), _o(2 * h),
           _o(1 * h, "Q06", trk="A1")]                         # same id, other station: its own
    g = d2score.tracklets(obs)
    assert [0, 1] in g and [2, 3] in g and [4] in g and len(g) == 3, g


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
