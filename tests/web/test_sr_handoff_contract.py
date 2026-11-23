"""What the single-tracklet hand-off delivers to the web page, end to end and offline.

``srsearch.sr_then_itf`` (the SR stage, then the full ITF search seeded with the seed and
the link) feeds ``engine_api._sr_web_result``, which turns the hand-off into ONE search
result: the SR tracklets listed with the linked ones, the find judged whole by the chimera
rule (seed included, exactly once), the ITF-to-ITF minimums deciding submittability.  The SR
stage, the regular search, the fits and the document builder are stubbed: these pin the
contract, not the search.

    pytest tests/web/test_sr_handoff_contract.py
"""
import types

import pytest

import pyroorb.web.engine_api as E
from pyroorb.itf import srsearch
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.search import ItfSearchResult
from pyroorb.itf.srsearch import SrLink, SrResult
from pyroorb.itf.verdict import SearchVerdict

JD0 = 2460000.5
PARTIAL = srsearch.CHIMERA_WARNING + "PARTIAL judgement of the added tracklets"
WHOLE = srsearch.CHIMERA_WARNING + "WHOLE judgement of the find"


def _line(desig, jd, stn="F51"):
    from datetime import datetime, timedelta
    dt = datetime(1970, 1, 1) + timedelta(days=jd - 2440587.5)
    frac = dt.day + (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400
    date = f"{dt.year:04d} {dt.month:02d} {frac:08.5f}"
    line = ("     " + f"{desig:<7}" + "  C" + f"{date:<17}" + f"{'10 00 00.00':<12}"
            + f"{'+10 00 00.0':<12}" + " " * 9 + " 21.0G" + " " + "     " + f"{stn:>3}")
    assert len(line) == 80
    return line


def _seed_obs():
    return [{"jd_utc": JD0 + 0.01 * i, "obscode": "F51", "ra_deg": 150.0, "dec_deg": 10.0,
             "obs80_line": _line("ZZ00001", JD0 + 0.01 * i)} for i in range(3)]


def _text(lines):
    return "".join(ln + "\n" for ln in lines)


def _sr():
    lines = {101: [_line("SR00001", JD0 + 6.0, "G96"), _line("SR00001", JD0 + 6.02, "G96")],
             102: [_line("SR00002", JD0 + 9.0, "I41"), _line("SR00002", JD0 + 9.02, "I41")]}
    return SrResult(link=SrLink(trkids=[101, 102], kind="triplet", lines=lines, sr_chi2=3.2,
                                dts=[6.0, 9.0], fit={"rms": 0.3}, trksubs={"u1": "db1"}),
                    reason="linked 2 tracklet(s) (triplet)", n_window=50, n_refined=10)


ADDED = [_line("PR00009", JD0 + 20.0, "703"), _line("PR00009", JD0 + 20.02, "703")]


@pytest.fixture
def stubs(monkeypatch):
    """The regular search replaced by a fake that, like the real one, attaches its own
    (partial) chimera judgement unless the hand-off asks it not to; the whole-find chimera
    rule, the fit, the ITF-to-ITF check and the document builder stubbed and recorded."""
    rec = {"search_kw": [], "whole": [], "itfitf_ok": False, "apparition": None}

    def fake_search(source, **kw):
        rec["search_kw"].append(kw)
        res = rec["apparition"]
        if res is not None and kw.get("chimera_check", True):
            for r in (res, res.alternative):
                if r is not None:
                    r.warnings = list(r.warnings) + [PARTIAL]
        return res
    import pyroorb.itf.search as S
    monkeypatch.setattr(S, "itf_search", fake_search)
    monkeypatch.setattr(srsearch, "sr_search", lambda *a, **k: _sr())

    def whole(seed_text, itf_text, obscodes, **k):
        rec["whole"].append((seed_text, itf_text))
        return [WHOLE]
    monkeypatch.setattr(srsearch, "chimera_warnings_short_seed", whole)
    import pyroorb.itf.fitter as F
    monkeypatch.setattr(F, "fit_astrometry", lambda text, oc, **k: types.SimpleNamespace(
        ok=True, fit="FIT(" + str(text.count("\n")) + ")", rms=0.3))
    monkeypatch.setattr(srsearch, "itfitf_check", lambda combined, trial: (
        (True, []) if rec["itfitf_ok"] else (False, ["3 nights spanning 41.0 d"])))
    import pyroorb.itf.linkage as L
    monkeypatch.setattr(L, "build_linkage", lambda combined, cfg, **k: (
        {"doc": True, "itf": k["itf_obs_out"], "trksubs": k["trksubs"]}, None))
    return rec


def _handoff(rec, apparition):
    rec["apparition"] = apparition
    seed = _seed_obs()
    sr, itf = srsearch.sr_then_itf(seed, object(), {}, ["ZZ00001"], ItfConfig())
    seed_text = _text(o["obs80_line"] for o in seed)
    return E._sr_web_result(sr, itf, seed_text, {}, primaries=["ZZ00001"], cfg=ItfConfig())


SR_TEXT = _text(_sr().link.lines[101] + _sr().link.lines[102])
SEED_TEXT = _text(o["obs80_line"] for o in _seed_obs())


def test_the_apparition_search_is_seeded_with_the_seed_and_the_link(stubs):
    stubs["apparition"] = None
    sr, itf = srsearch.sr_then_apparition(_seed_obs(), object(), {}, ["ZZ00001"], ItfConfig(),
                                          do_submit=False)
    (kw,) = stubs["search_kw"]
    assert kw["seed_obs80"].splitlines() == SEED_TEXT.splitlines() + SR_TEXT.splitlines()
    assert kw["primary_desigs"] == ["ZZ00001"] and kw["do_submit"] is False
    assert kw["cfg"].archive_start == ""                       # the apparition window only
    assert kw["chimera_check"] is False          # the caller judges the whole find, once
    assert sr.link.trkids == [101, 102] and itf is None


def test_outside_the_neocp_monitor_the_link_seeds_the_full_search(stubs):
    # only the NEOCP monitor's first step is bounded to the seed's apparition
    stubs["apparition"] = None
    cfg = ItfConfig()
    srsearch.sr_then_itf(_seed_obs(), object(), {}, ["ZZ00001"], cfg)
    (kw,) = stubs["search_kw"]
    assert kw["cfg"] is cfg and kw["chimera_check"] is False
    assert kw["seed_obs80"].splitlines() == SEED_TEXT.splitlines() + SR_TEXT.splitlines()


def test_no_sr_link_means_no_apparition_search(monkeypatch, stubs):
    monkeypatch.setattr(srsearch, "sr_search", lambda *a, **k: SrResult(reason="nothing"))
    sr, itf = srsearch.sr_then_apparition(_seed_obs(), object(), {}, ["ZZ00001"], ItfConfig())
    assert itf is None and stubs["search_kw"] == []


def test_a_grown_find_is_judged_whole_exactly_once(stubs):
    stubs["itfitf_ok"] = True
    app = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="linked", fit="APPFIT",
                          linkage={"app": 1}, combined_obs80=SEED_TEXT + SR_TEXT + _text(ADDED),
                          itf_obs80=_text(ADDED), warnings=["seed degraded"],
                          itf_trksubs={"u9": "db9"}, telemetry={"t": 1})
    out, info = _handoff(stubs, app)
    assert out.verdict is SearchVerdict.LINKED
    assert out.itf_obs80 == SR_TEXT + _text(ADDED)
    assert out.warnings == ["seed degraded", WHOLE]           # the partial one replaced
    assert stubs["whole"] == [(SEED_TEXT, SR_TEXT + _text(ADDED))]
    assert out.linkage["itf"] == SR_TEXT + _text(ADDED)
    assert out.itf_trksubs == {"u1": "db1", "u9": "db9"}
    assert out.telemetry == {"t": 1, "obs_count": 6}
    assert out.alternative is None
    assert info["submittable"] is True and info["review_reason"] == ""
    assert info["follow_up"] == {"outcome": "linked", "reason": "linked", "added_tracklets": 1}
    assert info["note"] == ("single-tracklet search: linked 2 tracklet(s) (triplet), then "
                            "the full ITF search: linked 1 more tracklet(s)")


def test_the_alternative_is_judged_whole_too(stubs):
    alt = ItfSearchResult(verdict=SearchVerdict.POSSIBLE, reason="thin", fit="ALTFIT",
                          combined_obs80=SEED_TEXT + SR_TEXT + _text(ADDED),
                          itf_obs80=_text(ADDED), warnings=[])
    app = ItfSearchResult(verdict=SearchVerdict.LINKED, reason="linked", fit="APPFIT",
                          linkage={"app": 1}, combined_obs80=SEED_TEXT + SR_TEXT,
                          itf_obs80=_text(ADDED[:1]), warnings=[], alternative=alt)
    out, _info = _handoff(stubs, app)
    assert out.alternative is not None
    assert out.alternative.itf_obs80 == SR_TEXT + _text(ADDED)
    assert out.alternative.warnings == [WHOLE]
    assert out.warnings == [WHOLE]
    assert out.alternative.verdict is SearchVerdict.POSSIBLE
    assert out.alternative.reason.startswith(E.SR_REVIEW_ONLY)


def test_a_possible_apparition_result_meeting_the_minimums_keeps_its_reason(stubs):
    stubs["itfitf_ok"] = True
    app = ItfSearchResult(verdict=SearchVerdict.POSSIBLE, reason="seed fit 1.9x worse",
                          fit="APPFIT", combined_obs80=SEED_TEXT + SR_TEXT + _text(ADDED),
                          itf_obs80=_text(ADDED), warnings=[])
    out, _info = _handoff(stubs, app)
    assert out.verdict is SearchVerdict.LINKED
    assert out.warnings == [WHOLE, E.SR_SEARCH_CAUTION + "seed fit 1.9x worse"]


@pytest.mark.parametrize("ok", [False, True])
def test_the_find_on_its_own_when_the_apparition_adds_nothing(stubs, ok):
    stubs["itfitf_ok"] = ok
    app = ItfSearchResult(verdict=SearchVerdict.NONE, reason="no likely matches",
                          diagnostics=["d1"], incomplete_sweep={"windows": 2})
    out, info = _handoff(stubs, app)
    assert out.verdict is (SearchVerdict.LINKED if ok else SearchVerdict.POSSIBLE)
    assert out.fit == "FIT(7)"                                  # seed + SR lines, fitted once
    assert out.combined_obs80 == SEED_TEXT + SR_TEXT and out.itf_obs80 == SR_TEXT
    assert out.warnings == [WHOLE]
    assert stubs["whole"] == [(SEED_TEXT, SR_TEXT)]
    assert out.telemetry == {"obs_count": 4}
    assert out.diagnostics == ["d1"] and out.incomplete_sweep == {"windows": 2}
    assert out.itf_trksubs == {"u1": "db1"} and out.alternative is None
    if ok:
        assert out.linkage["itf"] == SR_TEXT
        assert out.reason == "single-tracklet find meeting the ITF-to-ITF submission minimums"
    else:
        assert out.linkage is None
        assert out.reason == E.SR_REVIEW_ONLY + "3 nights spanning 41.0 d)"
    assert info["submittable"] is ok
    assert info["follow_up"] == {"outcome": "none", "reason": "no likely matches",
                                  "added_tracklets": 0}
    assert info["note"].endswith("then the full ITF search: nothing further "
                                 "(no likely matches)")


def test_a_find_whose_arc_does_not_fit_is_no_find(stubs, monkeypatch):
    import pyroorb.itf.fitter as F
    monkeypatch.setattr(F, "fit_astrometry", lambda text, oc, **k: types.SimpleNamespace(
        ok=False, fit=None, rms=None))
    out, info = _handoff(stubs, ItfSearchResult(verdict=SearchVerdict.NONE, reason="none"))
    assert out.verdict is SearchVerdict.NONE
    assert info["note"].endswith("but the combined arc did not fit")


def test_a_chimera_check_that_cannot_run_is_a_caution(monkeypatch):
    """The leave-out crashing on a single-tracklet find leaves it unvouched: a chimera
    caution the user must acknowledge, never a silent pass."""
    import pyroorb.itf.leaveout as LO

    def boom(*a, **k):
        raise RuntimeError("leave-out exploded")
    monkeypatch.setattr(LO, "measure", boom)
    seed = _text(o["obs80_line"] for o in _seed_obs())
    ws = E._sr_chimera_cautions(seed, SR_TEXT, {})
    assert len(ws) == 1 and ws[0].startswith(srsearch.CHIMERA_WARNING)
    assert "could not run" in ws[0]
