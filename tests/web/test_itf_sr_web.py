"""The web ITF search picks the single-tracklet (SR) search by arc, offline.

``engine_api.itf_search_web`` runs ``itf.srsearch.sr_then_apparition`` when the seed is too
short to fix a distance (``srsearch.short_arc``: one night, or nights spanning under 2 days)
and the regular search otherwise.  Both searches are replaced by the ``_sr_searcher`` /
``_searcher`` seams and the page's refit by a stub, so no database, network or engine fit is
touched: these test the routing and the payload, not the search.

    pytest tests/web/test_itf_sr_web.py
"""
import types

import pytest

import pyroorb.web.engine_api as E
from pyroorb.itf import srsearch
from pyroorb.itf.search import ItfSearchResult
from pyroorb.itf.srsearch import SrLink, SrResult
from pyroorb.itf.verdict import SearchVerdict

JD0 = 2460000.5


def _line(desig, jd, stn="F51", ra="10 00 00.00", dec="+10 00 00.0"):
    """A column-correct OBS80 line (designation cols 6-12, date 16-32, station 78-80)."""
    from datetime import datetime, timedelta
    dt = datetime(1970, 1, 1) + timedelta(days=jd - 2440587.5)
    frac = dt.day + (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400
    date = f"{dt.year:04d} {dt.month:02d} {frac:08.5f}"
    line = ("     " + f"{desig:<7}" + "  C" + f"{date:<17}" + f"{ra:<12}" + f"{dec:<12}"
            + " " * 9 + " 21.0G" + " " + "     " + f"{stn:>3}")
    assert len(line) == 80
    return line


def _seed(jds, stn="F51"):
    """Seed observation dicts as a session holds them, each with its OBS80 rendering."""
    return [{"designation": "ZZ00001", "jd_utc": jd, "ra_deg": 150.0, "dec_deg": 10.0,
             "mag": 21.0, "band": "G", "obscode": stn, "sigma_ra": 0.5, "sigma_dec": 0.5,
             "_line": _line("ZZ00001", jd, stn)} for jd in jds]


@pytest.fixture
def web(monkeypatch):
    """A session holding ``state["obs"]``; the page's paste-style refit stubbed to open a new
    session without fitting, recording what it was handed."""
    state = {"obs": [], "refits": []}
    monkeypatch.setattr(E, "base_obs_for", lambda session=None, fit_key=None:
                        (list(state["obs"]), None))
    monkeypatch.setattr(E, "_fit_for", lambda session=None, fit_key=None: None)
    monkeypatch.setattr(E.obs80, "obs80_base_line", lambda o: o["_line"])
    # the chimera guard refits per tracklet; it passes here unless a test says otherwise
    monkeypatch.setattr(srsearch, "chimera_guard_short_seed", lambda *a, **k: (True, None))

    def refit(obs, meta, obscodes, meta_extra=None):
        sid = f"sr-test-{len(state['refits'])}"
        E._SESSIONS[sid] = {"fit": None}
        state["refits"].append(obs)
        return {"ok": True, "session": sid}
    monkeypatch.setattr(E, "fit_obs_list", refit)
    yield state
    for k in [k for k in E._SESSIONS if str(k).startswith("sr-test-")]:
        E._SESSIONS.pop(k, None)


def _no_regular(*a, **k):
    raise AssertionError("the regular search ran on a short arc")


def _no_sr(*a, **k):
    raise AssertionError("the single-tracklet search ran on a long arc")


def test_short_arc_routes_to_the_single_tracklet_search(web):
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(4)])
    assert srsearch.short_arc(web["obs"])
    seen = {}

    def sr(seed_obs, primaries, cfg, log):
        seen.update(seed=seed_obs, primaries=primaries)
        return SrResult(reason="no candidate fits the seed"), None
    out = E.itf_search_web("s", _searcher=_no_regular, _sr_searcher=sr)
    assert out["ok"] and out["itf"]["mode"] == "single_tracklet"
    assert [o["obs80_line"] for o in seen["seed"]] == [o["_line"] for o in web["obs"]]
    assert seen["primaries"] == ["ZZ00001"]


def test_two_nights_under_two_days_is_short(web):
    web["obs"] = _seed([JD0, JD0 + 0.02, JD0 + 1.0, JD0 + 1.02])
    out = E.itf_search_web("s", _searcher=_no_regular,
                           _sr_searcher=lambda *a: (SrResult(reason="x"), None))
    assert out["itf"]["mode"] == "single_tracklet"


def test_long_arc_runs_the_regular_search(web):
    web["obs"] = _seed([JD0, JD0 + 0.02, JD0 + 3.0, JD0 + 3.02, JD0 + 9.0])
    assert not srsearch.short_arc(web["obs"])
    ran = []

    def regular(seed_obs80, primaries, cfg):
        ran.append(seed_obs80)
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")
    out = E.itf_search_web("s", _searcher=regular, _sr_searcher=_no_sr)
    assert ran and out["itf"]["found"] is False
    assert out["itf"]["mode"] == "regular" and "mode_note" not in out["itf"]


def test_sr_no_link_is_the_no_linkage_shape_with_the_sr_reason(web):
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    out = E.itf_search_web("s", _searcher=_no_regular,
                           _sr_searcher=lambda *a: (SrResult(
                               reason="none of 4 proposals passed the engine fit",
                               n_window=812), None))
    itf = out["itf"]
    assert out["ok"] is True and itf["found"] is False and itf["suppressed"] is False
    assert itf["reason"] == "single-tracklet search: none of 4 proposals passed the engine fit"
    assert itf["mode"] == "single_tracklet" and itf["sr"]["n_window"] == 812
    assert "session" not in out


def test_a_space_based_seed_is_refused_with_its_reason(web):
    """The real hand-off, with no source: a seed from orbit stops at ``seed_station_ok``
    before any database is asked, and the page says why."""
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)], stn="C57")

    def sr(seed_obs, primaries, cfg, log):
        return srsearch.sr_then_apparition(seed_obs, None, E._obscodes(), primaries, cfg,
                                           log=log)
    out = E.itf_search_web("s", _searcher=_no_regular, _sr_searcher=sr)
    assert out["ok"] is True and out["itf"]["found"] is False
    assert "not on the ground" in out["itf"]["reason"]


def _sr_link(jd):
    sr_lines = {101: [_line("SR00001", jd + 20.0, "G96"), _line("SR00001", jd + 20.02, "G96")],
                102: [_line("SR00002", jd + 41.0, "T05"), _line("SR00002", jd + 41.02, "T05")]}
    return SrResult(link=SrLink(trkids=[101, 102], kind="triplet", lines=sr_lines,
                                sr_chi2=3.2, dts=[20.0, 41.0],
                                fit={"ok": True, "rms": 0.31, "chi2_dof": 0.8, "max_z": 2.1,
                                     "used": 7, "count": 7}),
                    reason="linked 2 tracklet(s) (triplet)", n_window=900, n_refined=40)


def _judge(monkeypatch, ok, why=()):
    """Stub the ITF-to-ITF minimums (and the whole-arc refit they judge)."""
    import pyroorb.itf.fitter as F
    monkeypatch.setattr(F, "fit_astrometry", lambda text, oc, **k: types.SimpleNamespace(
        ok=True, fit=object(), rms=0.3))
    monkeypatch.setattr(srsearch, "itfitf_check", lambda text, trial: (ok, list(why)))


def test_sr_link_then_apparition_lists_both_stages(web, monkeypatch):
    _judge(monkeypatch, False, ["3 nights spanning 41.0 d (> 15)"])
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    seed_text = "".join(o["_line"] + "\n" for o in web["obs"])
    sr = _sr_link(JD0)
    sr_text = "".join(ln + "\n" for k in (101, 102) for ln in sr.link.lines[k])
    itf_text = _line("PR00009", JD0 + 70.0, "F51") + "\n" + _line("PR00009", JD0 + 70.02) + "\n"
    itf = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="linked", fit=object(),
                          linkage={"links": {"link_0": {}}},
                          combined_obs80=seed_text + sr_text + itf_text, itf_obs80=itf_text,
                          telemetry={"obs_count": 2})
    out = E.itf_search_web("s", _searcher=_no_regular, _sr_searcher=lambda *a: (sr, itf))
    blk = out["itf"]
    assert blk["found"] is True and blk["mode"] == "single_tracklet"
    assert blk["mode_note"] == ("single-tracklet search: linked 2 tracklet(s) (triplet), "
                                "then same-apparition search: linked 1 more tracklet(s)")
    stage = {t["trksub"]: t.get("stage") for t in blk["tracklets"]}
    assert stage == {"SR00001": "single-tracklet", "SR00002": "single-tracklet",
                     "PR00009": None}
    # the SR lines are linked ITF astrometry: in itf_obs80, the obs count and the refit
    assert sr_text in blk["itf_obs80"] and itf_text in blk["itf_obs80"]
    assert blk["added_obs"] == 6
    refit = web["refits"][-1]
    assert {o["obs80_line"][5:12].strip() for o in refit} >= {"SR00001", "SR00002", "PR00009"}
    # short of the ITF-to-ITF minimums: a possible with no document, saying which rule
    assert blk["possible"] is True and blk["submittable"] is False and blk["linkage"] is None
    assert blk["reason"].startswith(E.SR_REVIEW_ONLY) and "41.0 d" in blk["reason"]
    pend = E._SESSIONS[out["session"]]["itf_pending"]
    assert pend["submittable"] is False and "review_only" not in pend
    ids = {E.obs_unique_id(l) for l in sr_text.splitlines()}
    assert ids <= set(pend["itf_ids"])
    assert blk["sr"]["kind"] == "triplet" and blk["sr"]["apparition"]["added_tracklets"] == 1


def test_sr_link_stands_alone_when_the_apparition_adds_nothing(web, monkeypatch):
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    sr = _sr_link(JD0)
    import pyroorb.itf.fitter as F
    fitted = []
    monkeypatch.setattr(F, "fit_astrometry", lambda text, oc, **k: (
        fitted.append(text) or types.SimpleNamespace(ok=True, fit=object(), rms=0.3)))
    itf = ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")
    out = E.itf_search_web("s", _searcher=_no_regular, _sr_searcher=lambda *a: (sr, itf))
    blk = out["itf"]
    assert blk["found"] is True and blk["possible"] is True
    assert blk["mode_note"].endswith("then same-apparition search: nothing further "
                                     "(no likely matches)")
    assert sorted(t["trksub"] for t in blk["tracklets"]) == ["SR00001", "SR00002"]
    assert all(t["stage"] == "single-tracklet" for t in blk["tracklets"])
    assert len(fitted) == 1 and fitted[0].count("\n") == 7      # 3 seed + 4 SR lines


def test_sr_find_meeting_the_itfitf_minimums_is_submittable(web, monkeypatch):
    """A single-tracklet find that meets the ITF-to-ITF submission
    minimums is submittable like any linkage, its tracklets named by the DB trksub."""
    _judge(monkeypatch, True)
    import pyroorb.itf.linkage as L
    built = []
    monkeypatch.setattr(L, "build_linkage", lambda combined, cfg, **k: (
        built.append(k) or ({"links": {"link_0": {}}}, None)))
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    sr = _sr_link(JD0)
    sr.link.trksubs = {"x1": "dbname1"}
    itf = ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")
    out = E.itf_search_web("s", _searcher=_no_regular, _sr_searcher=lambda *a: (sr, itf))
    blk = out["itf"]
    assert blk["found"] is True and blk["possible"] is False
    assert blk["sr"]["submittable"] is True
    pend = E._SESSIONS[out["session"]]["itf_pending"]
    assert pend["submittable"] is True
    assert built and built[0]["trksubs"] == {"x1": "dbname1"}
    assert built[0]["itf_obs_out"].count("\n") == 4                 # the 4 SR lines


def _submittable_sr(web, monkeypatch, guard):
    """A single-tracklet find meeting the ITF-to-ITF minimums, the chimera guard stubbed."""
    _judge(monkeypatch, True)
    import pyroorb.itf.linkage as L
    monkeypatch.setattr(L, "build_linkage", lambda combined, cfg, **k: (
        {"links": {"link_0": {}}}, None))
    asked = []
    monkeypatch.setattr(srsearch, "chimera_guard_short_seed",
                        lambda seed, itf, oc, **k: (asked.append((seed, itf)) or guard))
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    itf = ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")
    out = E.itf_search_web("s", _searcher=_no_regular,
                           _sr_searcher=lambda *a: (_sr_link(JD0), itf))
    return out, asked


def test_sr_find_the_chimera_guard_refuses_carries_a_caution(web, monkeypatch):
    """The guard that keeps automated senders from sending a find warns the web user: it is
    shown, and the find is not submitted until the user acknowledges it."""
    out, asked = _submittable_sr(web, monkeypatch, (False, "the tracklet (G96, JD 2460020.50) "
                                                    "is not predicted by the rest"))
    blk = out["itf"]
    # the seed and both SR tracklets were put to the guard
    assert asked and asked[0][0].count("\n") == 3 and asked[0][1].count("\n") == 4
    cautions = [w for w in blk["warnings"] if "possible chimera" in w]
    assert len(cautions) == 1 and "G96" in cautions[0]
    assert blk["possible"] is False                     # still a find meeting the minimums
    # stored as one of the cautions itf_submit_web makes the user acknowledge
    assert cautions[0] in E._SESSIONS[out["session"]]["itf_warnings"]


def test_sr_find_the_chimera_guard_passes_carries_no_caution(web, monkeypatch):
    out, asked = _submittable_sr(web, monkeypatch, (True, None))
    assert asked
    assert not [w for w in out["itf"]["warnings"] if "possible chimera" in w]


def test_itfitf_check_is_the_mpc_minimums():
    """One night of seed plus one SR tracklet two days later is 2 nights: refused."""
    a = _line("SEED001", JD0)
    b = _line("SR00001", JD0 + 2.0, "G96")
    ok, why = srsearch.itfitf_check(a + "\n" + b + "\n", None)
    assert not ok and any("distinct nights" in w for w in why)


def test_stop_between_the_stages_is_honoured(web):
    """The hand-off's log is a stop point: a Stop pressed during the single-tracklet stage
    ends the search before the apparition stage starts."""
    import threading
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    stop = threading.Event()
    later = []

    def sr(seed_obs, primaries, cfg, log):
        stop.set()                         # the user presses Stop during the SR stage
        log("SR search: linked 2 tracklet(s) (triplet)")
        later.append("apparition stage")
        return _sr_link(JD0), None
    with pytest.raises(E._deadline.Cancelled):
        with E._deadline.cancellable(stop, label="ITF search"):
            E.itf_search_web("s", _searcher=_no_regular, _sr_searcher=sr)
    assert later == []


def test_the_live_hand_off_owns_its_connection_and_never_submits(monkeypatch):
    """``_sr_search_live`` through the real ``sr_then_apparition``: the database and the two
    searches stubbed, the arguments the apparition search receives checked."""
    import pyroorb.itf.candidates as C
    import pyroorb.itf.dbio as D
    import pyroorb.itf.search as S
    closed, got = [], {}
    conn = types.SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(D, "connect_pub", lambda cfg: conn)
    monkeypatch.setattr(C, "DbSource", lambda c, cfg, reconnect=None: ("source", c))
    monkeypatch.setattr(srsearch, "sr_search", lambda seed, source, oc, cfg, log=None:
                        _sr_link(JD0))

    def itf_search(source, **kw):
        got.update(kw, source=source)
        return ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches")
    monkeypatch.setattr(S, "itf_search", itf_search)
    seed = [{**o, "obs80_line": o["_line"]} for o in _seed([JD0 + 0.01 * i for i in range(3)])]
    from pyroorb.itf.config import ItfConfig
    sr, itf = E._sr_search_live(seed, ["ZZ00001"], ItfConfig(), lambda m: None,
                                max_uncert_deg=2.0, uncert_mult=1.5)
    assert sr.link is not None and itf.reason == "no likely matches"
    assert got["source"] == ("source", conn) and closed == [True]
    assert got["do_submit"] is False and got["primary_desigs"] == ["ZZ00001"]
    assert got["max_uncert_deg"] == 2.0 and got["uncert_mult"] == 1.5
    assert got["seed_obs80"].count("SR0000") == 4      # the SR link rides in the seed


def test_an_unreachable_database_is_reported_not_raised(web, monkeypatch):
    import pyroorb.itf.dbio as D

    def down(cfg):
        raise OSError("connection refused")
    monkeypatch.setattr(D, "connect_pub", down)
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    out = E.itf_search_web("s", _searcher=_no_regular)
    assert out == {"ok": False, "error": "ITF database unreachable"}


def _grown_sr(web, monkeypatch, verdict, reason):
    """A single-tracklet find meeting the ITF-to-ITF minimums whose same-apparition search
    added one tracklet and returned ``verdict``."""
    _judge(monkeypatch, True)
    import pyroorb.itf.linkage as L
    monkeypatch.setattr(L, "build_linkage", lambda combined, cfg, **k: (
        {"links": {"link_0": {}}}, None))
    web["obs"] = _seed([JD0 + 0.01 * i for i in range(3)])
    seed_text = "".join(o["_line"] + "\n" for o in web["obs"])
    sr = _sr_link(JD0)
    sr_text = "".join(ln + "\n" for k in (101, 102) for ln in sr.link.lines[k])
    itf_text = _line("PR00009", JD0 + 3.0) + "\n" + _line("PR00009", JD0 + 3.02) + "\n"
    itf = ItfSearchResult(verdict=verdict, reason=reason, fit=object(),
                          linkage={"links": {"link_0": {}}}
                          if verdict == SearchVerdict.LINKED else None,
                          combined_obs80=seed_text + sr_text + itf_text, itf_obs80=itf_text,
                          telemetry={"obs_count": 2})
    return E.itf_search_web("s", _searcher=_no_regular, _sr_searcher=lambda *a: (sr, itf))


def test_sr_find_the_apparition_search_called_possible_carries_its_reason(web, monkeypatch):
    """A find the same-apparition search judged a possible (0.6) is still judged by the
    ITF-to-ITF minimums, but the regular search's own reason joins the warnings: the user
    reads and acknowledges it before the find can be submitted."""
    why = "seed observations fit 1.9x worse under the link"
    out = _grown_sr(web, monkeypatch, SearchVerdict.POSSIBLE, why)
    blk = out["itf"]
    assert blk["found"] is True and blk["possible"] is False
    cautions = [w for w in blk["warnings"] if w.startswith(E.SR_APPARITION_CAUTION)]
    assert cautions == [E.SR_APPARITION_CAUTION + why]
    assert cautions[0] in E._SESSIONS[out["session"]]["itf_warnings"]


def test_a_linked_apparition_result_carries_no_possible_caution(web, monkeypatch):
    out = _grown_sr(web, monkeypatch, SearchVerdict.LINKED, "linked")
    assert out["itf"]["possible"] is False
    assert not [w for w in out["itf"]["warnings"] if w.startswith(E.SR_APPARITION_CAUTION)]
