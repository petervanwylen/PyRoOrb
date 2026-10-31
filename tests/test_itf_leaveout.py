"""Leave-out confirmation of data-quality cautions (itf/leaveout.py).

Offline: the refit is replaced by a canned measurement, so these pin the RULES --
what is in scope, which unit a caution names, what moves where, and that a
three-thin (1.75 -> 3) linkage is never touched."""
from types import SimpleNamespace

import pytest

from pyroorb.itf import leaveout
from pyroorb.itf.residual_trend import review_warnings
from pyroorb.itf.residual_trend import TREND_NOTE, is_trend_note


def _line(trk, stn, date, ra="10 00 00.00", dec="+10 00 00.0"):
    """An 80-column ITF line (trksub in 6-12, date 16-32, station 78-80)."""
    head = f"     {trk:<7}  C{date:<17}{ra:<12}{dec:<12}"
    return (head.ljust(65) + "20.5 V").ljust(77) + stn


ITF = "\n".join([
    _line("AAA0001", "G96", "2019 03 10.30000"), _line("AAA0001", "G96", "2019 03 10.31000"),
    _line("AAA0001", "G96", "2019 03 10.32000"),
    _line("BBB0002", "F51", "2022 01 05.40000"), _line("BBB0002", "F51", "2022 01 05.41000"),
    _line("BBB0002", "F51", "2022 01 05.42000"),
]) + "\n"
SEED = "\n".join(_line("K24A00A", "T08", f"2024 02 0{d}.50000") for d in (1, 2, 3)) + "\n"
COMBINED = SEED + ITF

PHOT_LINKED = ("a tracklet of a linked tracklet (G96, 2019-03-10, 3 obs, p=0.002) averages "
               "+0.9 mag from the object's predicted brightness (possible different body)")
PHOT_OWN = ("a tracklet of the object's OWN published arc (T08, 2024-02-01, 3 obs, p=0.01) "
            "averages -0.8 mag from the object's predicted brightness")
OFFSET = ("a linked night is systematically offset: the 2022-01-05 tracklet sits +0.91\" off "
          "in RA (7x the pre-link residual level, n=3) -- a coherent offset, not noise")
TREND = (f"{TREND_NOTE} a linked tracklet: the residuals of the 2019-03-10 G96 tracklet "
         f"(3 obs, 29 min) walk steadily in BOTH RA (+2.1σ) and Dec (+2.4σ)")
SEED_BAD = "the object's own observations fit worse after the link (0.10\" -> 0.30\")"

GOOD = {"confirmed": True, "smaj": 0.3, "miss": 0.4, "D": 1.1, "walk": 0.3}
BAD = {"confirmed": False, "smaj": 0.3, "miss": 0.4, "D": 1.1, "walk": 2.8}


@pytest.fixture
def canned(monkeypatch):
    """measure() -> per-trksub canned result; records what was left out."""
    calls, table = [], {}

    def fake(lines, combined, obscodes, registry=None, max_sec=None):
        keys = {l[5:12].strip() for l in lines}
        calls.append(keys)
        res = [table.get(k, GOOD) for k in keys]
        return BAD if any(not r["confirmed"] for r in res) else GOOD
    monkeypatch.setattr(leaveout, "measure", fake)
    return calls, table


def test_scope_and_targets():
    assert leaveout.in_scope(PHOT_LINKED) == "photometry"
    assert leaveout.in_scope(OFFSET) == "offset"
    assert leaveout.in_scope(TREND) == "trend"
    assert leaveout.in_scope(SEED_BAD) is None
    assert leaveout.is_own_arc(PHOT_OWN) and not leaveout.is_own_arc(PHOT_LINKED)
    assert leaveout.target(PHOT_LINKED)[0] == "G96"
    assert [k for k, _ in leaveout.named_tracklets(PHOT_LINKED, ITF)] == [("AAA0001", "G96")]
    # an offset names only a date: every linked tracklet that night
    assert [k for k, _ in leaveout.named_tracklets(OFFSET, ITF)] == [("BBB0002", "F51")]
    # photometry takes the search's own (station, jd) when it has one
    mt = {"station": "F51", "jd": 2459584.9}
    assert [k for k, _ in leaveout.named_tracklets(PHOT_LINKED, ITF, mt)] == [("BBB0002", "F51")]


def test_a_confirmed_caution_becomes_a_diagnostic_the_review_layer_ignores(canned):
    kept, diags, rec = leaveout.confirm_cautions([PHOT_LINKED, SEED_BAD], [], ITF, COMBINED, {})
    assert kept == [SEED_BAD]                       # out of scope: untouched
    assert len(diags) == 1 and diags[0].startswith(leaveout.LEAVEOUT_NOTE)
    assert not is_trend_note(diags[0])
    assert review_warnings(kept, diags) == [SEED_BAD]
    assert rec[0]["confirmed"] and rec[0]["unit"] == "AAA0001@G96"


def test_an_unconfirmed_caution_stays_a_warning(canned):
    calls, table = canned
    table["AAA0001"] = BAD                          # drifts at the wrong rate
    kept, diags, rec = leaveout.confirm_cautions([PHOT_LINKED, TREND], [], ITF, COMBINED, {})
    assert kept == [PHOT_LINKED, TREND] and diags == []
    assert [r["confirmed"] for r in rec] == [False, False]


def test_a_two_axis_trend_warning_is_demoted_when_confirmed(canned):
    kept, diags, _ = leaveout.confirm_cautions([TREND], [], ITF, COMBINED, {})
    assert kept == [] and diags[0].startswith(leaveout.LEAVEOUT_NOTE)
    assert review_warnings(kept, diags) == []


def test_a_trend_note_in_the_diagnostics_is_rewritten(canned):
    kept, diags, _ = leaveout.confirm_cautions([], [TREND], ITF, COMBINED, {})
    assert kept == [] and len(diags) == 1 and diags[0].startswith(leaveout.LEAVEOUT_NOTE)
    assert review_warnings(kept, diags) == []       # no longer counted for review
    calls, table = canned
    table["AAA0001"] = BAD
    _k, diags2, _ = leaveout.confirm_cautions([], [TREND], ITF, COMBINED, {})
    assert diags2 == [TREND]                        # unconfirmed: still a review warning


def test_own_arc_caution_needs_every_linked_apparition(canned):
    calls, table = canned
    kept, diags, rec = leaveout.confirm_cautions([PHOT_OWN], [], ITF, COMBINED, {})
    assert kept == [] and diags and rec[0]["unit"] == "every linked apparition"
    assert {frozenset(c) for c in calls} == {frozenset({"AAA0001"}), frozenset({"BBB0002"})}
    table["BBB0002"] = BAD
    kept, diags, _ = leaveout.confirm_cautions([PHOT_OWN], [], ITF, COMBINED, {})
    assert kept == [PHOT_OWN] and diags == []


def test_a_caution_naming_no_linked_tracklet_stays(canned):
    w = OFFSET.replace("2022-01-05", "2024-02-02")  # a seed night
    kept, diags, rec = leaveout.confirm_cautions([w], [], ITF, COMBINED, {})
    assert kept == [w] and rec[0]["why"] == "names no linked tracklet"


def test_tracklet_stats_measures_the_drift():
    flat = [(0.0, 0.3, 0.1), (0.01, 0.3, 0.1), (0.02, 0.3, 0.1)]
    s = leaveout.tracklet_stats("G96", flat, [[0.04, 0], [0, 0.04]])
    assert s["walk"] == pytest.approx(0.0) and s["miss"] == pytest.approx(0.316, abs=1e-3)
    ramp = [(0.0, -1.0, 0.0), (0.01, 0.0, 0.0), (0.02, 1.0, 0.0)]
    assert leaveout.tracklet_stats("G96", ramp, [[0.04, 0], [0, 0.04]])["walk"] == pytest.approx(2.0)


def test_three_thin_confirmed_linkage_is_never_touched(monkeypatch):
    from pyroorb.itf import search
    called = []
    monkeypatch.setattr(leaveout, "confirm_cautions", lambda *a, **k: called.append(1) or ([], [], []))
    ctx = SimpleNamespace(cfg=SimpleNamespace(leaveout_confirm=True, trial_fit_max_sec=30.0),
                          obscodes={}, log=lambda *a: None)
    result = SimpleNamespace(warnings=[PHOT_LINKED], diagnostics=[])
    search._leaveout_cautions(ctx, result, {}, ITF, COMBINED, None, promoted_175=True)
    assert called == [] and result.warnings == [PHOT_LINKED]
    monkeypatch.setattr(search, "_ctx_registry", lambda ctx: None)
    search._leaveout_cautions(ctx, result, {}, ITF, COMBINED, None, promoted_175=False)
    assert called == [1]
    ctx.cfg.leaveout_confirm = False
    search._leaveout_cautions(ctx, result, {}, ITF, COMBINED, None, promoted_175=False)
    assert called == [1]


def test_two_observations_are_a_rate_one_is_not():
    cov = [[0.04, 0], [0, 0.04]]
    two = [(0.0, -0.9, 0.0), (0.02, 0.9, 0.0)]
    assert leaveout.tracklet_stats("G96", two, cov)["walk"] == pytest.approx(1.8)
    one = [(0.0, 0.1, 0.1)]
    assert leaveout.tracklet_stats("G96", one, cov)["walk"] == 0.0


def test_a_single_observation_is_never_confirmed(monkeypatch):
    from pyroorb.itf import leaveout as L
    rows = iter([])
    monkeypatch.setattr(L, "_sky_cov", lambda row: [[0.01, 0], [0, 0.01]])

    class TF:
        ok, error, fit = True, None, object()
    monkeypatch.setattr("pyroorb.itf.fitter.fit_astrometry", lambda *a, **k: TF())

    def eph(fit, obscodes, jd, jd2, step, site="500", full=False):
        from pyroorb.obs80 import mpc80_to_radec
        return [{"ra_deg": 150.0, "dec_deg": 10.0, "unc_smaj_arcsec": 0.1,
                 "unc_smin_arcsec": 0.1, "unc_pa_deg": 0.0}]
    monkeypatch.setattr("pyroorb.ephemerides.compute_ephemeris", eph)
    single = [_line("CCC0003", "W84", "2014 04 25.36204")]
    m = L.measure(single, COMBINED + single[0] + "\n", {})
    assert m["n_min"] == 1 and not m["confirmed"] and "single observation" in m["why"]
    pair = [_line("CCC0003", "W84", "2014 04 25.36204"), _line("CCC0003", "W84", "2014 04 25.37204")]
    m2 = L.measure(pair, COMBINED + "\n".join(pair) + "\n", {})
    assert m2["n_min"] == 2 and m2["confirmed"]
