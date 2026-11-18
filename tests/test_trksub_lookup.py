"""Offline tests for typing a submitter's trksub (an ITF tracklet) in place of a designation.

``H005311 K16FN7V`` -- an isolated 568 tracklet and 2016 FV237 -- is how a user proposes
linking a tracklet the MPC never designated.  The trksub is not a designation, so it is
pre-screened out of the published-obs batch (which would only HTTP-500 on it) and fetched
temporary-first: the NEOCP, then the SBN replica's ``obs_sbn.trksub``, the API last.  A
trksub is the submitter's own label, so it can name several tracklets (the user picks one
or all) or one already designated (the user is told which designation).

Nothing here touches the network or a database: fetchers and the cursor are replaced.
"""
import datetime

import pytest

import pyroorb.obs_sources as osrc
from pyroorb import sbn_obs


# --- the pre-screen ---------------------------------------------------------------------

@pytest.mark.parametrize("token, temporary", [
    ("H005311", True),             # a 568 trksub (the motivating case)
    ("P12pwHF", True),             # a NEOCP id
    ("8N62001", True),
    ("H005311@568", True),         # qualified forms are trksubs by construction
    ("H005311@568/2015-01-26", True),
    ("K16FN7V", False),            # packed provisional
    ("J99T43V", False),
    ("I98A00A", False),            # 1800s packed: a "starts with J or K" rule gets this wrong
    ("PLS2040", False),            # packed survey designations, likewise
    ("T1S3138", False),
    ("_PO00Tq", False),            # extended packing, likewise
    ("H0053", False),              # packed number (170053), likewise
    ("~0000", False),
    ("0073P", False),
    ("433", False),
    ("2016 FV237", False),
    ("C/2023 A3", False),
    ("Eros", False),               # letters only: far likelier a name, stays API-first
    ("Ceres", False),
])
def test_the_prescreen_uses_the_packing_rules_not_a_prefix(token, temporary):
    assert osrc.routes_as_temporary(token) is temporary


def test_split_and_qualify_round_trip():
    assert sbn_obs.split_trksub("H005311") == ("H005311", None, None)
    assert sbn_obs.split_trksub("H005311@568") == ("H005311", "568", None)
    assert sbn_obs.split_trksub("H005311@568/2015-01-26") == ("H005311", "568", "2015-01-26")
    assert sbn_obs.split_trksub("H005311@568/20150126") == ("H005311", "568", "2015-01-26")
    assert sbn_obs.split_trksub("2016 FV237") is None
    assert sbn_obs.split_trksub("TOOLONG12") is None
    t = {"trksub": "H005311", "stn": "568", "date": "2015-01-26"}
    assert sbn_obs.qualified_token(t) == "H005311@568/2015-01-26"


def test_a_qualified_token_is_requested_as_its_bare_trksub():
    """The obs are filed under the trksub, so the linkage builder (requested_primaries)
    must see the bare id or the typed tracklet drops out of the document."""
    assert osrc.requested_id("H005311@568/2015-01-26") == "H005311"
    assert osrc.requested_id("H005311") == "H005311"
    assert osrc.requested_id("K16FN7V") == "2016 FV237"


# --- the replica lookup -----------------------------------------------------------------

def _t(y, m, d, h=0):
    return datetime.datetime(y, m, d, h)


# obs_sbn rows by tracklet: (trkid, stn, first, last, n, provid, permid)
TRACKLETS = {
    "H005311": [("00000Ev3wV", "568", _t(2015, 1, 26, 7), _t(2015, 1, 26, 9), 5, None, None)],
    "ZZ00001": [("t1", "954", _t(2009, 6, 20), _t(2009, 6, 20, 1), 3, None, None),
                ("t2", "I41", _t(2010, 8, 21), _t(2010, 8, 21, 1), 2, None, None),
                ("t3", "W88", _t(2022, 1, 25), _t(2022, 1, 25, 1), 3, "2022 AB1", None)],
    "gb00884": [("g1", "G19", _t(2024, 5, 13, 21), _t(2024, 5, 13, 21), 6, "2024 HP", None)],
}


class TrkCursor:
    def __init__(self):
        self._rows = []

    def execute(self, sql, args):
        s = " ".join(sql.split())
        assert "from obs_sbn" in s and "trksub = %s" in s, s
        trksub, statuses = args[0], args[1]
        assert set(statuses) == {"I", "P", "p"}
        stn = args[2] if "stn = %s" in s else None
        self._rows = [r for r in TRACKLETS.get(trksub, []) if stn is None or r[1] == stn]

    def fetchall(self):
        return list(self._rows)


def test_a_trksub_resolves_to_its_tracklets():
    trk = sbn_obs.trksub_tracklets(TrkCursor(), "H005311")
    assert [(t["stn"], t["date"], t["n_obs"], t["designation"]) for t in trk] == \
        [("568", "2015-01-26", 5, None)]


def test_station_and_date_narrow_the_lookup():
    cur = TrkCursor()
    assert len(sbn_obs.trksub_tracklets(cur, "ZZ00001")) == 3
    assert [t["trkid"] for t in sbn_obs.trksub_tracklets(cur, "ZZ00001@I41")] == ["t2"]
    assert [t["trkid"] for t in sbn_obs.trksub_tracklets(cur, "ZZ00001@954/2009-06-20")] == ["t1"]
    assert sbn_obs.trksub_tracklets(cur, "ZZ00001@954/2009-06-21") == []


def test_a_designated_tracklet_carries_its_designation():
    (t,) = sbn_obs.trksub_tracklets(TrkCursor(), "gb00884")
    assert t["designation"] == "2024 HP"


# --- routing through _fetch_observations ------------------------------------------------

FV237 = [{"jd_utc": 2457470.0 + i, "packed": "K16FN7V", "designation": "2016 FV237",
          "obscode": "F51"} for i in range(3)]


def _h005311_obs():
    return [{"jd_utc": 2457048.81 + i * 0.02, "packed": "H005311", "designation": "H005311",
             "obscode": "568"} for i in range(5)]


@pytest.fixture
def routes(monkeypatch):
    """Every source replaced by a recorder; tests fill in what each one answers."""
    from pyroorb import ades
    calls = []
    answers = {"api": {}, "neocp": {}, "sbn": {}}

    def api(desigs, obscodes, cache_dir=None, **k):
        ds = [desigs] if isinstance(desigs, str) else list(desigs)
        calls.append(("api", tuple(ds)))
        if any(d not in answers["api"] for d in ds):
            raise RuntimeError("HTTP Error 500")      # what the API does on an unknown id
        return [o for d in ds for o in answers["api"][d]], {"designations": ds}

    def neocp(desigs, obscodes, cache_dir=None):
        calls.append(("neocp", tuple(desigs)))
        return [o for d in desigs for o in answers["neocp"].get(d, [])], {}

    def lookup(token):
        calls.append(("sbn", token))
        return answers["sbn"].get(token, ([], []))

    monkeypatch.setattr(ades, "fetch_combined_ades", api)
    monkeypatch.setattr(osrc, "fetch_combined", api)
    monkeypatch.setattr(ades, "fetch_combined_neocp", neocp)
    monkeypatch.setattr(ades, "records_to_obs", lambda recs, oc: list(recs))
    monkeypatch.setattr(sbn_obs, "available", lambda: True)
    monkeypatch.setattr(sbn_obs, "lookup_trksub", lookup)
    monkeypatch.setattr(sbn_obs, "fetch_records", lambda d: [])
    return calls, answers


def test_a_trksub_pair_skips_the_doomed_batch_and_finds_the_tracklet(routes):
    calls, answers = routes
    answers["api"]["2016 FV237"] = FV237
    trk = [{"trksub": "H005311", "stn": "568", "date": "2015-01-26", "n_obs": 5,
            "designation": None, "trkid": "x"}]
    answers["sbn"]["H005311"] = (trk, _h005311_obs())
    obs, meta = osrc._fetch_observations(["H005311", "2016 FV237"], {})
    assert ("api", ("H005311", "2016 FV237")) not in calls, "the batch would only HTTP-500"
    assert not any(c[0] == "api" and "H005311" in c[1] for c in calls), \
        "a trksub the replica answered must not also be sent to the API"
    assert calls.index(("neocp", ("H005311",))) < calls.index(("sbn", "H005311"))
    assert sorted(meta["designations"]) == ["2016 FV237", "H005311"]
    assert meta["source"] == "SBN"
    assert not meta["fetch_errors"] and "trksub_unresolved" not in meta
    assert osrc.fetch_failure_message(meta) is None


def test_a_live_neocp_object_is_answered_by_the_neocp_not_the_replica(routes):
    """The replica holds a NEOCP object's trksub too -- but only the submitter's own rows
    (A11JXBy: 3 obs from 734), or another tracklet under the same label (gb00884 is a
    2024 HP tracklet) -- so the NEOCP must be asked first and win."""
    calls, answers = routes
    answers["neocp"]["A11JXBy"] = [{"jd_utc": 2461000.0, "designation": "A11JXBy"}] * 7
    obs, meta = osrc._fetch_observations(["A11JXBy"], {})
    assert len(obs) == 7 and meta["source"] == "NEOCP"
    assert ("sbn", "A11JXBy") not in calls


def test_published_designations_alone_still_take_the_batch(routes):
    calls, answers = routes
    answers["api"]["2016 FV237"] = FV237
    obs, meta = osrc._fetch_observations(["2016 FV237"], {})
    assert calls == [("api", ("2016 FV237",))]
    assert len(obs) == 3


def test_a_name_stays_api_first(routes):
    calls, answers = routes
    answers["api"]["Eros"] = [{"jd_utc": 2451000.0, "designation": "433"}]
    osrc._fetch_observations(["Eros"], {})
    assert calls[0] == ("api", ("Eros",))
    assert not any(c[0] in ("neocp", "sbn") for c in calls)


def test_a_qualified_token_asks_only_the_replica(routes):
    calls, answers = routes
    trk = [{"trksub": "H005311", "stn": "568", "date": "2015-01-26", "n_obs": 5,
            "designation": None, "trkid": "x"}]
    answers["sbn"]["H005311@568"] = (trk, _h005311_obs())
    obs, _meta = osrc._fetch_observations(["H005311@568"], {})
    assert len(obs) == 5
    assert calls == [("sbn", "H005311@568")]


def test_an_unknown_temporary_id_still_reaches_the_api_last(routes):
    calls, answers = routes
    answers["api"]["Q123456"] = [{"jd_utc": 2451000.0, "designation": "Q123456"}]
    obs, _meta = osrc._fetch_observations(["Q123456"], {})
    assert len(obs) == 1
    assert [c[0] for c in calls] == ["neocp", "sbn", "api"]


def _ambiguous_trk():
    return [{"trksub": "ZZ00001", "stn": "954", "date": "2009-06-20", "n_obs": 3,
             "designation": None, "trkid": "t1"},
            {"trksub": "ZZ00001", "stn": "I41", "date": "2010-08-21", "n_obs": 2,
             "designation": None, "trkid": "t2"}]


def test_several_tracklets_are_a_question_not_a_fit(routes):
    calls, answers = routes
    answers["api"]["2016 FV237"] = FV237
    answers["sbn"]["ZZ00001"] = (_ambiguous_trk(), [])
    obs, meta = osrc._fetch_observations(["ZZ00001", "2016 FV237"], {})
    assert not any(c[0] == "api" and "ZZ00001" in c[1] for c in calls), \
        "a trksub the replica knows is not then guessed at by the API"
    assert not meta["fetch_errors"], "a question is not a transport failure"
    q = osrc.trksub_question(meta)["ZZ00001"]
    assert [c["token"] for c in q["choices"]] == ["ZZ00001@954/2009-06-20",
                                                  "ZZ00001@I41/2010-08-21"]
    msg = osrc.fetch_failure_message(meta)
    assert msg and "ZZ00001@954/2009-06-20" in msg, \
        "every caller of fetch_failure_message (CLI, ident-check, ITF seed) must refuse"


def test_an_already_designated_tracklet_names_its_designation(routes):
    calls, answers = routes
    trk = [{"trksub": "gb00884", "stn": "G19", "date": "2024-05-13", "n_obs": 6,
            "designation": "2024 HP", "trkid": "g1"}]
    answers["sbn"]["gb00884"] = (trk, [])
    obs, meta = osrc._fetch_observations(["gb00884"], {})
    assert obs == []
    msg = osrc.fetch_failure_message(meta)
    assert "gb00884" in msg and "is already 2024 HP" in msg


def test_long_lists_are_capped_with_their_totals():
    trk = ([{"trksub": "00001", "stn": f"{i:03d}", "date": "2020-01-01", "n_obs": 2,
             "designation": None, "trkid": f"f{i}"} for i in range(40)]
           + [{"trksub": "00001", "stn": "F51", "date": "2020-01-01", "n_obs": 2,
               "designation": f"2020 A{i}", "trkid": f"d{i}"} for i in range(20)]
           + [{"trksub": "00001", "stn": "F52", "date": "2020-01-02", "n_obs": 2,
               "designation": "2020 A0", "trkid": "dup"}])
    q = osrc.trksub_question({"trksub_unresolved": {"00001": trk}})["00001"]
    assert len(q["choices"]) == osrc.TRKSUB_MAX_CHOICES and q["n_choices"] == 40
    assert len(q["designated"]) == osrc.TRKSUB_MAX_DESIGNATED
    assert q["n_designated"] == 20, "one entry per designation, not per tracklet"
    assert "add @station" in osrc.trksub_message({"trksub_unresolved": {"00001": trk}})


# --- the web answer ---------------------------------------------------------------------

def test_the_web_fit_answers_with_the_question(monkeypatch):
    from pyroorb.web import engine_api as ea
    meta = {"trksub_unresolved": {"ZZ00001": _ambiguous_trk()}, "fetch_errors": {}}
    monkeypatch.setattr(osrc, "fetch_observations", lambda d, oc: (list(FV237), meta))
    monkeypatch.setattr(ea, "_obscodes", lambda: {})
    p = ea.run_fit("ZZ00001 K16FN7V")
    assert p["ok"] is False
    assert p["typed"] == ["ZZ00001", "2016 FV237"]
    assert [c["token"] for c in p["trksub_question"]["ZZ00001"]["choices"]] == \
        ["ZZ00001@954/2009-06-20", "ZZ00001@I41/2010-08-21"]


def test_the_pickers_rewritten_box_parses_back_to_the_same_ids():
    """trksubpick.js rewrites the box as comma-joined ids; the server must read back
    exactly those ids (the date's '/' and '-' must not split or join anything)."""
    from pyroorb.web.engine_api import parse_designations
    assert parse_designations("ZZ00001@954/2009-06-20, ZZ00001@I41/2010-08-21, 2016 FV237") == \
        ["ZZ00001@954/2009-06-20", "ZZ00001@I41/2010-08-21", "2016 FV237"]
    assert [osrc.ensure_unpacked(d) for d in parse_designations("H005311 K16FN7V")] == \
        ["H005311", "2016 FV237"]


def test_the_export_cache_refuses_the_same_way(monkeypatch):
    from pyroorb.web import fitcache
    meta = {"trksub_unresolved": {"ZZ00001": _ambiguous_trk()}, "fetch_errors": {}}
    monkeypatch.setattr(osrc, "fetch_observations", lambda d, oc: (list(FV237), meta))
    with pytest.raises(ValueError, match="ZZ00001@954/2009-06-20"):
        fitcache.get_by_desig("ZZ00001, 2016 FV237", {})
