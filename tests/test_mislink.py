"""Mislinkage score (src/pyroorb/mislink): the frozen model, its explanations, the committed
watch list, and frozen objects scored end to end from their obs_sbn rows."""
import json
import math
import os
import sys

import pytest

from pyroorb.mislink import score as S  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mislink_data")


def test_model_has_the_feature_columns_in_order():
    assert S.model_problem() is None
    assert S._model().n_features == len(S.FEATURES) == 26
    assert set(S.DESCRIPTIONS) == set(S.FEATURES)


def test_explanation_adds_up_to_the_score():
    f = {"gap_n_nights": 3, "fit_e": 0.45, "fit_a": 2.7, "fit_q": 1.5, "lo_D": 0.6,
         "iso_gap": 28.0, "fit_rms_n": 0.8}
    ex = S.explain(f)
    total = ex["base"] + sum(c["shap"] for c in ex["contributions"])
    assert abs(1 / (1 + math.exp(-total)) - ex["score"]) < 2e-3
    # sorted by size, every feature present once, missing ones as None
    sizes = [abs(c["shap"]) for c in ex["contributions"]]
    assert sizes == sorted(sizes, reverse=True)
    assert sorted(c["feature"] for c in ex["contributions"]) == sorted(S.FEATURES)
    assert next(c for c in ex["contributions"] if c["feature"] == "chain_max_D")["value"] is None


def test_tiers_grade_the_whole_range():
    assert S.verdict(0.99)[0] == "very-likely"
    assert S.verdict(0.97)[0] == "very-likely"
    assert S.verdict(0.95)[0] == "likely" and S.verdict(S.THRESHOLD)[0] == "likely"
    assert S.verdict(0.848) == ("possible", "possible mislinkage")
    assert S.verdict(0.5) == ("small", "small chance of mislinkage")
    assert S.verdict(0.27)[0] == "small"
    assert S.verdict(0.1) == ("remote", "very remote chance of mislinkage")
    assert S.verdict(0.02)[0] == "remote"
    assert S.verdict(0.0199)[0] == "none" and S.verdict(0.0)[0] == "none"
    assert S.verdict(0.01)[1] == "no sign of a mislinkage"
    bounds = [t[0] for t in S.TIERS]
    assert bounds == sorted(bounds, reverse=True) and bounds[-1] == 0.0


def test_domain():
    assert S.in_domain({"fit_a": 2.5, "fit_q": 1.9})
    assert not S.in_domain({"fit_a": 2.5, "fit_q": 1.1})       # NEO
    assert not S.in_domain({"fit_a": 6.0, "fit_q": 4.0})       # beyond the Hildas
    assert not S.in_domain({})


def test_too_few_rows_is_a_status_not_a_score():
    rows = [{"obstime": "2026-03-01T01:00:00.000Z", "ra": 10.0, "dec": 1.0, "stn": "G96"}]
    r = S.score_records(rows, "2026 AA1", obscodes={})
    assert r["status"] == "too-few-obs" and "score" not in r


def _cadence_rows(minutes, nights=(0, 1, 2), stn="W84"):
    """Detections of a slow main-belt mover, ``minutes`` apart, four a night."""
    import datetime as dt
    t0 = dt.datetime(2014, 3, 3, 2, tzinfo=dt.timezone.utc)
    rows = []
    for n in nights:
        for k in range(4):
            t = t0 + dt.timedelta(days=n, minutes=k * minutes)
            h = (t - t0).total_seconds() / 3600
            rows.append({"obstime": t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                         "ra": 171.36 - 0.0055 * h, "dec": -1.55 + 0.0019 * h,
                         "stn": stn, "mag": 21.5, "band": "g"})
    return rows


def _tracklets(rows, **kw):
    from pyroorb.mislink.tracklets import build_tracklets, clean_record, epoch_sec
    recs = []
    for r in rows:
        r = clean_record(r)
        r["obj"], r["nopp"], r["_es"] = "2014 ET119", 1, epoch_sec(r["obstime"])
        recs.append(r)
    return build_tracklets(recs, **kw)


def test_two_hour_cadence_forms_station_night_tracklets():
    """DECam W84 in 2014 revisited each field every 2h12m-2h30m: under the 120-min
    rule every detection is alone and no tracklet forms; per station-night each night
    is one four-detection tracklet."""
    rows = _cadence_rows(135)
    assert _tracklets(rows) == []
    trk = _tracklets(rows, station_night=True)
    assert [t["n_obs"] for t in trk] == [4, 4, 4]
    # a 120-min cadence night is the same either way
    assert [t["n_obs"] for t in _tracklets(_cadence_rows(30))] == [4, 4, 4]


def test_lone_detection_nights_are_still_refused():
    rows = [r for k, r in enumerate(_cadence_rows(135, nights=(0, 1, 2, 3))) if k % 4 == 0]
    rows += _cadence_rows(135, nights=(5,))
    r = S.score_records(rows, "2014 ET119", obscodes={})
    assert r["status"] == "single-detection-nights" and r["tracklet_rule"] == "station-night"


def test_committed_watch_list_is_consistent():
    data = S.candidates()
    cands = data["candidates"]
    assert cands and data["threshold"] == S.THRESHOLD
    assert set(data["descriptions"]) == set(S.FEATURES)
    names = [c["designation"] for c in cands]
    assert len(names) == len(set(names))
    for c in cands:
        assert c["score"] >= S.THRESHOLD
        assert 3 <= c["nights"] <= 5
        assert {k for k, _v, _s in c["explanation"]["contributions"]} == set(S.FEATURES)
        assert c["label"] in (None, "reviewed chimera", "known mislinkage")


def test_watch_list_explanations_reproduce_their_model_scores():
    """Each entry's explanation is the committed model's: base + contributions = its score."""
    for c in S.candidates()["candidates"][:40]:
        e = c["explanation"]
        total = e["base"] + sum(s for _k, _v, s in e["contributions"])
        assert abs(1 / (1 + math.exp(-total)) - e["model_score"]) < 2e-3, c["designation"]
        assert abs(S.explain({k: v for k, v, _s in e["contributions"]})["score"]
                   - e["model_score"]) < 1e-3, c["designation"]


@pytest.mark.slow
@pytest.mark.parametrize("desig,rule", [("2026 EC104", "120min"), ("2026 EH15", "120min"),
                                        ("2014 ET119", "station-night")])
def test_frozen_object_scores_end_to_end(desig, rule):
    """Published rows -> tracklets -> 26 features -> score, as the census scored them
    (2026 EC104 a flagged chimera, 2026 EH15 a clean arc; 2014 ET119's three DECam
    nights are detections ~2 h apart, scored on station-night tracklets, not refused)."""
    from pyroorb.obs_sources import load_obscodes
    frozen = json.load(open(os.path.join(_DATA, "frozen_rows.json")))[desig]
    r = S.score_records([dict(x) for x in frozen["rows"]], desig, load_obscodes())
    assert r["status"] == "ok" and r["in_domain"] and r["tracklet_rule"] == rule
    assert abs(r["score"] - frozen["score"]) < 2e-3
    assert r["likely"] is (frozen["score"] >= S.THRESHOLD)


@pytest.mark.slow
def test_page_observations_score_like_the_published_rows():
    """A fit's own observations (what a paste becomes) converted back to records score as
    the published rows they came from: the page path judges the same astrometry."""
    from pyroorb import ades
    from pyroorb.obs_sources import load_obscodes
    oc = load_obscodes()
    frozen = json.load(open(os.path.join(_DATA, "frozen_rows.json")))["2026 EC104"]
    obs = ades.records_to_obs([dict(x) for x in frozen["rows"]], oc)
    rows, skipped = S.records_from_obs(obs, "2026 EC104")
    assert skipped == 0 and len(rows) == len(obs)
    r = S.score_observations(obs, "2026 EC104", oc)
    assert r["status"] == "ok" and r["source"] == "page"
    assert abs(r["score"] - frozen["score"]) < 2e-3


def test_space_based_observations_are_left_out():
    obs = [{"jd_utc": 2460000.5, "ra_deg": 1.0, "dec_deg": 2.0, "obscode": "C51",
            "sat_pos_au": [1e-5, 0, 0]},
           {"jd_utc": 2460000.6, "ra_deg": 1.1, "dec_deg": 2.0, "obscode": "F51", "band": "w"}]
    rows, skipped = S.records_from_obs(obs, "2026 AA1")
    assert skipped == 1 and [r["stn"] for r in rows] == ["F51"]
    assert rows[0]["obstime"].startswith("2023-02-25T02:24:00") and rows[0]["band"] == "w"


def test_page_records_carry_the_reported_position_not_the_debiased_one():
    """A fit's observations are already star-catalogue debiased; scoring debiases its
    records itself, so the records must carry the REPORTED position -- passing the
    corrected one on corrected it twice (a 691/USNO-B1 position moved by twice its bias)."""
    from pyroorb import ades, debias
    if not debias.available():
        pytest.skip("no star-catalogue bias tables")
    row = {"stn": "691", "ra": 150.123456, "dec": 12.345678, "astcat": "USNOB1",
           "obstime": "2012-03-04T05:06:07.000Z", "provid": "2012 AB1", "mode": "CCD"}
    oc = {"691": (0.0, 0.0, 0.0)}
    once = ades.records_to_obs([dict(row)], oc)[0]
    assert once.get("debiased") is True                     # USNO-B1 carries a bias here
    rows, skipped = S.records_from_obs([once], "2012 AB1")
    assert skipped == 0
    assert (rows[0]["ra"], rows[0]["dec"]) == (row["ra"], row["dec"])
    again = ades.records_to_obs(rows, oc)[0]
    assert abs(again["ra_deg"] - once["ra_deg"]) < 1e-12
    assert abs(again["dec_deg"] - once["dec_deg"]) < 1e-12


def test_roving_observations_are_left_out():
    """A roving observer (247) carries its site in ``site_itrs_au``: left out, not scored as
    a ground station the fit never placed."""
    obs = [{"jd_utc": 2460000.5, "ra_deg": 1.0, "dec_deg": 2.0, "obscode": "247",
            "site_itrs_au": [4e-5, 0, 0]},
           {"jd_utc": 2460000.6, "ra_deg": 1.1, "dec_deg": 2.0, "obscode": "F51"}]
    rows, skipped = S.records_from_obs(obs, "2026 AA1")
    assert skipped == 1 and [r["stn"] for r in rows] == ["F51"]
