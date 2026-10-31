"""Offline tests for the cross-listed observation dedup (obs_sources.dedup_observations).

The MPC republishes a NEOCP tracklet under the designation it earns (P12pwHF -> 2013 PY26)
and cross-lists linked survey tracklets, so a combined fetch concatenates the same
detection twice and the fit carries it at double weight.  One copy per detection, judged
by station + instant + sky position -- NEVER by designation, since two names is the case.
"""
import os
import sys

import pyroorb.obs_sources as osrc  # noqa: E402

OBS80 = (
    "     W023204  C2026 08 12.89938700 48 36.377+15 12 30.46         20.65rVNEOCPO18\n"
    "     K13P26Y  C2026 08 12.89938700 48 36.377+15 12 30.46         20.65rV     O18\n"
    "     K13P26Y  C2026 08 12.90938700 48 36.977+15 12 33.46         20.65rV     O18\n"
)


def _obs(desig, jd, ra, dec, code="O18"):
    return {"designation": desig, "obscode": code, "jd_utc": jd, "ra_deg": ra, "dec_deg": dec}


def test_identical_lines_under_two_designations_collapse_to_one():
    obs = osrc.parse_obs(OBS80, osrc.load_obscodes())
    for o in obs:
        o["designation"] = osrc._unpack(o["packed"])
    assert len(obs) == 3
    kept, dropped = osrc.dedup_observations(obs)
    assert len(kept) == 2 and len(dropped) == 1
    d = dropped[0]
    # the PUBLISHED designation survives; the NEOCP id is the one zapped
    assert d["kept_designation"] == "2013 PY26" and d["designation"] == "W023204"
    assert d["obscode"] == "O18"
    assert [o["jd_utc"] for o in kept] == sorted(o["jd_utc"] for o in kept)


def test_same_station_same_instant_different_position_is_not_a_duplicate():
    # a survey exposure carries two objects at once (F51: 2014 HZ280 vs 2018 BV65, ~28' apart)
    a = _obs("2014 HZ280", 2457000.5, 10.0, 5.0, "F51")
    b = _obs("2018 BV65", 2457000.5, 10.0, 5.0 + 28 / 60.0, "F51")
    kept, dropped = osrc.dedup_observations([a, b])
    assert len(kept) == 2 and not dropped


def test_different_station_same_instant_and_position_is_not_a_duplicate():
    a = _obs("A", 2457000.5, 10.0, 5.0, "F51")
    b = _obs("B", 2457000.5, 10.0, 5.0, "F52")
    kept, dropped = osrc.dedup_observations([a, b])
    assert len(kept) == 2 and not dropped


def test_obs80_time_truncation_still_matches_the_ades_copy():
    # the published OBS80 line truncates the time to 5 decimal days (~0.9 s) and the
    # position to 0.01"; the NEOCP/ADES copy is full precision
    a = _obs("2013 PY26", 2461265.399387, 12.151570833, 15.208461111)
    b = _obs("P12pwHF", 2461265.39939, 12.151570833 + 0.005 / 3600, 15.208461111)
    kept, dropped = osrc.dedup_observations([a, b])
    assert len(kept) == 1 and kept[0]["designation"] == "2013 PY26"
    assert dropped[0]["designation"] == "P12pwHF"


def test_straddling_a_day_boundary_still_matches():
    a = _obs("A", 2457000.4999999, 10.0, 5.0)
    b = _obs("B", 2457000.5000001, 10.0, 5.0)
    kept, dropped = osrc.dedup_observations([a, b])
    assert len(kept) == 1 and len(dropped) == 1


def test_fetch_observations_dedups_and_records_it(monkeypatch):
    two = [_obs("P12pwHF", 2461265.399387, 12.15, 15.2),
           _obs("2013 PY26", 2461265.399387, 12.15, 15.2),
           _obs("2013 PY26", 2461265.409387, 12.16, 15.21)]
    meta = {"designation": "2013 PY26", "designations": ["2013 PY26", "P12pwHF"],
            "combined": True, "other_designations": ["P12pwHF"], "n_requested": 2}
    monkeypatch.setattr(osrc, "_fetch_observations", lambda *a, **k: (list(two), dict(meta)))
    from pyroorb import radar
    monkeypatch.setattr(radar, "attach_radar", lambda meta, cache_dir=None: None)
    obs, meta = osrc.fetch_observations(["P12pwHF", "2013 PY26"], {})
    assert len(obs) == 2
    assert len(meta["duplicates_dropped"]) == 1
    assert meta["duplicates_dropped"][0]["designation"] == "P12pwHF"


def test_nothing_dropped_leaves_meta_untouched(monkeypatch):
    two = [_obs("A", 2461265.399387, 12.15, 15.2), _obs("A", 2461265.409387, 12.16, 15.21)]
    monkeypatch.setattr(osrc, "_fetch_observations",
                        lambda *a, **k: (list(two), {"designation": "A", "designations": ["A"],
                                                      "combined": False, "other_designations": []}))
    from pyroorb import radar
    monkeypatch.setattr(radar, "attach_radar", lambda meta, cache_dir=None: None)
    obs, meta = osrc.fetch_observations(["A"], {})
    assert len(obs) == 2 and "duplicates_dropped" not in meta


def test_published_designation_wins_even_when_the_temporary_id_sorts_first():
    # "0abc12" (a digits-first trksub) and "A10xyz" both sort before "2013 PY26" or "433"
    # in plain string order; the published copy must still be the keeper.
    for temp in ("0abc12", "A10xyz", "P12pwHF", "ZTF0abc"):
        for pub in ("2013 PY26", "433", "2040 P-L", "C/2020 F3", "1P"):
            a = _obs(temp, 2457000.5, 10.0, 5.0)
            b = _obs(pub, 2457000.5, 10.0, 5.0)
            kept, dropped = osrc.dedup_observations([a, b])
            assert kept[0]["designation"] == pub, (temp, pub)
            assert dropped[0]["designation"] == temp and dropped[0]["kept_designation"] == pub


def test_two_published_copies_fall_back_to_the_alphabetical_primary_rule():
    a = _obs("2018 BV65", 2457000.5, 10.0, 5.0)
    b = _obs("2014 HZ280", 2457000.5, 10.0, 5.0)
    kept, dropped = osrc.dedup_observations([a, b])
    assert kept[0]["designation"] == "2014 HZ280" and dropped[0]["designation"] == "2018 BV65"


def test_the_keeper_rule_survives_the_time_truncation_it_exists_for():
    # THE REGRESSION.  The two tests above give both copies the SAME jd_utc, which is the
    # one case the original implementation got right: it sorted by (jd, published, desig)
    # and kept the first copy visited, so the published-vs-temporary preference only ever
    # broke an exact-JD tie.  Two copies of one detection never tie -- the OBS80 line
    # truncates the time to ~0.9 s while the ADES record keeps it, which is exactly why
    # the match is by tolerance -- so the documented rule was decided by which source
    # happened to round downward.  Here the TEMPORARY copy carries the earlier timestamp.
    for dt in (-1e-5, 0.0, +1e-5):                       # temp earlier / tied / later
        a = _obs("P12pwHF", 2457000.5 + dt, 10.0, 5.0)
        b = _obs("2013 PY26", 2457000.5, 10.0, 5.0)
        for order in ([a, b], [b, a]):                   # and independent of input order
            kept, dropped = osrc.dedup_observations(order)
            assert len(kept) == 1 and kept[0]["designation"] == "2013 PY26", (dt, order)
            assert dropped[0]["kept_designation"] == "2013 PY26"


def test_three_cross_listed_copies_collapse_to_the_one_published_name():
    # A NEOCP id, a submitter's trksub and the published designation, in three different
    # instants inside the tolerance: one survivor, and it is the published one.
    obs = [_obs("P12pwHF", 2457000.500000, 10.0, 5.0),
           _obs("W023204", 2457000.500004, 10.0, 5.0),
           _obs("2013 PY26", 2457000.500011, 10.0, 5.0)]
    kept, dropped = osrc.dedup_observations(obs)
    assert [o["designation"] for o in kept] == ["2013 PY26"]
    assert {d["designation"] for d in dropped} == {"P12pwHF", "W023204"}
    assert all(d["kept_designation"] == "2013 PY26" for d in dropped)


def test_kept_stays_in_time_order_when_the_keeper_is_not_the_earliest_copy():
    # The survivor of a group need not be the copy that opened it, so the output is
    # re-sorted; a downstream fit indexes obs positionally and assumes time order.
    obs = [_obs("P12pwHF", 2457000.500000, 10.0, 5.0),
           _obs("2013 PY26", 2457000.500011, 10.0, 5.0),
           _obs("2013 PY26", 2456999.500000, 20.0, 5.0),
           _obs("2013 PY26", 2457002.500000, 30.0, 5.0)]
    kept, _ = osrc.dedup_observations(obs)
    jds = [o["jd_utc"] for o in kept]
    assert jds == sorted(jds) and len(kept) == 3


def test_a_pair_straddling_ra_zero_still_matches():
    # A plain RA subtraction reads 359.99999 and 0.00001 as 360 degrees apart and
    # silently declines to match; the difference is taken on the circle.
    a = _obs("P12pwHF", 2457000.5, 359.999990, 5.0)
    b = _obs("2013 PY26", 2457000.5, 0.000005, 5.0)
    kept, dropped = osrc.dedup_observations([a, b])
    assert len(kept) == 1 and kept[0]["designation"] == "2013 PY26" and len(dropped) == 1
    # ... and a genuine 10-arcsec separation across the wrap is still two detections
    c = _obs("2013 PY26", 2457000.5, 0.003, 5.0)
    assert len(osrc.dedup_observations([a, c])[0]) == 2


def test_both_duplicate_paths_share_one_predicate():
    # The fetch-time dedup and the identification check ask the same question and
    # used to answer it with two independent expressions, one of which was missing
    # the circle wrap.  Both now call orbit_common.same_detection; this pins that
    # they do, so a future fix to one cannot be forgotten for the other.
    import inspect

    from pyroorb import ident_check
    from pyroorb import orbit_common

    assert ident_check.same_detection is orbit_common.same_detection
    assert "same_detection" in inspect.getsource(osrc.find_duplicate)
    assert "same_detection" in inspect.getsource(ident_check._duplicate_excluded)


def test_the_identification_check_matches_a_duplicate_across_ra_zero():
    # An excluded obs cross-listed from a KEPT one is a benign duplicate, not a
    # suspect tracklet drop -- and it stays one when the pair straddles RA 0h.
    # Plain subtraction read that as 360 degrees apart, so the copy was reported
    # as a real drop (a 'y2' exception) purely for where it sits on the sky.
    import numpy as np

    from pyroorb import ident_check

    times = np.array([2457000.5, 2457000.5])
    codes = ["F51", "F51"]
    ras = np.array([359.99999, 0.00001])
    decs = np.array([5.0, 5.0])
    excluded = np.array([True, False])
    assert ident_check._duplicate_excluded(times, codes, ras, decs, excluded)[0]
    # ...while a genuinely different object in the same exposure still is not one
    far = np.array([359.99999, 0.5])
    assert not ident_check._duplicate_excluded(times, codes, far, decs, excluded)[0]


def test_is_published_designation_shapes():
    yes = ["433", "134340", "2013 PY26", "K13P26Y", "_PO00Tq", "2040 P-L", "1P", "73P-B",
           "C/2020 F3", "P/2019 LD2", "A/2017 U1", "C/2019 Q4-A"]
    no = ["P12pwHF", "W023204", "0abc12", "A10xyz", "ZTF0abc", "", None]
    assert all(osrc.is_published_designation(d) for d in yes), \
        [d for d in yes if not osrc.is_published_designation(d)]
    assert not any(osrc.is_published_designation(d) for d in no), \
        [d for d in no if osrc.is_published_designation(d)]
