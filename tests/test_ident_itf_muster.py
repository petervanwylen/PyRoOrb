"""An identification the ITF search would refuse as a linkage is not a clean identification.

The ITF search submits a linkage only when its observations meet the MPC's night profile (two
apparitions need one of >= 3 nights and the other >= 2) and its own anchor floor (one apparition
with >= 3 tracklets and >= 3 nights); a NEO is waived. linkage_verdict asked neither, so a real
4-night apparition joined to a ONE-night tracklet under a second designation came back 'y',
submittable -- a shape no fit can refute: the one apparition does not predict a tracklet years
away tightly enough to reject it. The verdict now asks the same floor of the kept observations.
"""
import os

import pytest

import _fixtures as FX
import pyroorb.fit_object as fo
from pyroorb import ident_check
from pyroorb.web import engine_api as E

pytestmark = pytest.mark.slow          # real engine fits


def _ident_fit(obs, designations):
    meta = {"designation": designations[0], "designations": list(designations), "n_requested": 2}
    return fo.fit_orbit([dict(o) for o in obs], E._obscodes(), desig=designations[0], meta=meta)


def test_a_one_night_side_is_not_a_clean_identification():
    # 2014 EE314's 4-night 2014 apparition plus one 4-obs 2016 W84 night filed under 2026 GV5:
    # the first two time-clusters of the PAIR_2026GV5 fixture. The MPC's own orbit misses that
    # night by ~38,000", yet the joint fit takes it at 0.13".
    fx = FX.load_fixture(os.path.join(FX.FIX, "PAIR_2026GV5.json"))
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    first_gv5 = min(o["jd_utc"] for o in obs if o["designation"] == "2026 GV5")
    two = [o for o in obs if o["designation"] == "2014 EE314" or o["jd_utc"] < first_gv5 + 1.0]
    assert {o["designation"] for o in two} == {"2014 EE314", "2026 GV5"}
    fit = _ident_fit(two, ["2014 EE314", "2026 GV5"])
    why = ident_check.itf_muster_problem(fit)
    assert why and "2 apparitions need one with >=3 nights and the other >=2" in why, why
    v = ident_check.linkage_verdict(fit)
    assert v["code"] == "y-examine" and not v["clean"] and not v["submittable"], v
    assert "ITF linkage" in v["reason"], v["reason"]


def test_a_neo_is_waived():
    # 2026 SA12's kept arc would not meet the non-NEO floor; as a NEO it stays a clean 'y'.
    fx = FX.load_fixture(os.path.join(FX.FIX, "PAIR_2026SA12.json"))
    fit = _ident_fit(fx["obs"], fx["meta"]["designations"])
    assert ident_check.itf_muster_problem(fit) is None
    v = ident_check.linkage_verdict(fit)
    assert v["code"] == "y" and v["submittable"], v


def test_a_preexisting_exception_leaves_the_floor_to_decide():
    # 2025 OA537's link excepts one tracklet, the 3-obs 2019-03-08 T09 tracklet of 2025 PA77,
    # which that object's own record rejects when fitted alone: a fault already in the MPC
    # record, not counted against the link (a note, not a y2). What decides it is then the ITF
    # floor -- apparitions of 1 and 7 nights -- so it is offered for review, never submitted.
    fx = FX.load_fixture(os.path.join(FX.FIX, "PAIR_2025OA537.json"))
    fit = _ident_fit(fx["obs"], fx["meta"]["designations"])
    v = ident_check.linkage_verdict(fit)
    assert v["code"] == "y-examine" and not v["submittable"], v
    assert "floor an ITF linkage must meet" in v["reason"], v["reason"]
    assert any("Not counted as an exception" in n for n in v["notes"]), v["notes"]


def test_the_profile_rule_is_one_rule():
    # the counts form and the OBS80 form are the same rule
    from pyroorb.itf import mpcrules
    assert mpcrules.opp_profile_counts_ok([4, 1])[0] is False
    assert mpcrules.opp_profile_counts_ok([3, 2])[0] is True
    assert mpcrules.opp_profile_counts_ok([2, 2, 2, 2, 2])[0] is True
    assert mpcrules.opp_profile_counts_ok([2, 2, 2])[0] is False
