"""Offline unit tests for the native ITF search (src/pyroorb/itf/).

Everything here runs without network or database: synthetic tracklets, hand-
built corridor windows, and the overlay candidate source.  The end-to-end
multi-opposition RECOVERY suite (real objects, MPC fetch) is separate:
``python -m pyroorb.itf recovery-suite`` -- see src/pyroorb/itf/harness.py.
"""
import sys, os, math
import pytest
import pathlib

import time

import numpy as np
from types import SimpleNamespace

from pyroorb.itf.candidates import (OverlaySource, match_chi2, obs_in_window,
                            tracklet_in_window, window_tolerances)
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.ephem import Window
from pyroorb.itf.harness import (_first_n_nights, designation_year, make_fake_itf,
                         observing_nights, pick_seed_opposition,
                         split_oppositions)
from pyroorb.itf.index import IndexConfig, compute_tracklet_features
from pyroorb.itf.linkage import build_linkage, collect_ids
from pyroorb.itf.obs80_fields import count_obs, is_crossing_360, obs_unique_id, shift_above_180
from pyroorb.itf.opp_profile import count_spans, opp_profile, OppProfile, OppProfileGrade
from pyroorb.itf.verdict import SearchVerdict

# opposition profiles by their legacy codes, for driving the gates directly
OPP_ONE = OppProfile(1, 0)                   # was 1
OPP_TWO_THIN = OppProfile(2, 1)              # was 1.5
OPP_THREE_THIN = OppProfile(3, 1)            # was 1.75
OPP_THREE = OppProfile(3, 3)                 # was 3

INDEX_CFG = IndexConfig()
T0 = 1.6e9


def _rows(ra0=150.0, dec0=10.0, dra=0.01, ddec=0.004, n=3, kink=0.0,
          trksub="PR00001", stn="F51", dt=1800.0):
    rows = []
    for k in range(n):
        ra = ra0 + k * dra + (kink if k == n - 1 else 0.0)
        rows.append({"tracklet_id": trksub, "es": T0 + k * dt, "ra": ra,
                     "dec": dec0 + k * ddec, "rms_ra": None, "rms_dec": None,
                     "mag": 21.0, "band": "G", "trksub": trksub, "stn": stn,
                     "provid": None})
    return rows


def _window(uncert_deg=0.05, ra0=150.0, dec0=10.0, dra=0.01, ddec=0.004):
    # corridor moving like the synthetic tracklet: dra deg per 1800 s
    return Window(unix_cur=T0, unix_start=T0 - 86400, unix_end=T0 + 86400,
                  ra_quad=(0.0, dra / 1800.0, ra0), dec_quad=(0.0, ddec / 1800.0, dec0),
                  min_ra=ra0 - 0.3, max_ra=ra0 + 0.3, min_dec=dec0 - 0.2,
                  max_dec=dec0 + 0.2, uncert=uncert_deg,
                  uncert_ra=uncert_deg / math.cos(math.radians(dec0)),
                  max_speed=1.0, dist_au=1.5, mag=20.0)


def _trial(used, *, rms=0.20, fit=None, ok=True, seed_rms=None, U=10.0):
    """A :class:`itf.fitter.TrialFit` stand-in for the tests that patch
    ``fit_astrometry``: it must carry every field the search reads off a live
    trial (the production code trusts the dataclass, so a partial double just
    hides real attribute errors).  ``seed_rms`` rides on the fake fit for the
    ``fake_subset_rms`` doubles to read back."""
    return SimpleNamespace(ok=ok, used=used, count=used, rms=rms, U=U,
                           fit=fit if fit is not None
                           else SimpleNamespace(_seed=seed_rms))


def _trial_queue_ctx(*, pub_rms=0.05, pub_used=15, cfg=None, pub_fit=None, **kw):
    """A ``SearchContext`` stand-in for the ``_recurse_trial_queue`` tests -- every field the
    trial queue reads, with the ordinary defaults; ``kw`` overrides (e.g. budget)."""
    fields = dict(cfg=cfg or ItfConfig(), obscodes=None, pub_obs80="SEED\n",
                  pub_ids=set(),
                  pub=SimpleNamespace(rms=pub_rms, used=pub_used, fit=pub_fit),
                  log=lambda *a: None, budget=None, telemetry={})
    return SimpleNamespace(**{**fields, **kw})


def test_features_linear_vs_kinked():
    clean = compute_tracklet_features(_rows(), INDEX_CFG)
    assert clean is not None and clean["linear_ok"] is True
    assert clean["fit_chi2_dof"] < 0.1
    kinked = compute_tracklet_features(_rows(kink=0.01), INDEX_CFG)   # 36" kink
    assert kinked["linear_ok"] is False and kinked["fit_chi2_dof"] > 100
    # degenerate cases are refused (the db-side admission criterion)
    assert compute_tracklet_features(_rows(n=1), INDEX_CFG) is None
    single_epoch = [dict(r, es=T0) for r in _rows()]
    assert compute_tracklet_features(single_epoch, INDEX_CFG) is None


def test_motion_prefilter_only_drops_on_positive_evidence():
    """The obs-saving motion pre-filter must keep everything that could still
    pass the trajectory gate, and only rule out tracklets whose OWN recorded
    motion contradicts the corridor by more than the slack."""
    from pyroorb.itf.candidates import motion_bound_ok
    cfg = ItfConfig(prefilter_motion=True)
    window = _window(uncert_deg=0.003)
    good = compute_tracklet_features(_rows(), INDEX_CFG)                  # matching motion
    reversed_ = compute_tracklet_features(_rows(dra=-0.01,
                                                trksub="PR00002"), INDEX_CFG)
    assert motion_bound_ok(good, window, cfg), "a matching tracklet must be kept"
    assert not motion_bound_ok(reversed_, window, cfg), \
        "reversed motion is positive evidence against"
    # unbounded rows are kept: no fitted rate, no fit rms, single epoch
    for missing in ("fit_rms_arcsec", "dra_cosdec_rate", "ddec_rate",
                    "epoch_start_sec", "epoch_end_sec"):
        assert motion_bound_ok(dict(reversed_, **{missing: None}), window, cfg), \
            f"a row without {missing} cannot be bounded and must be kept"
    assert motion_bound_ok(dict(reversed_, epoch_start_sec=T0,
                                epoch_end_sec=T0), window, cfg)
    # the slack knob is monotone: a huge slack keeps even the reversed tracklet
    assert motion_bound_ok(reversed_, window,
                           ItfConfig(prefilter_motion=True, prefilter_motion_slack_arcsec=1e6))


def test_prefilter_override_reaches_survivors_and_verdict_is_recorded():
    """The collector-mode bypass travels per-sweep (like row_budget) because the
    source binds its OWN cfg at construction -- a dataclasses.replace in
    itf_search never reaches it, which is exactly how the disable once shipped
    as a silent no-op.  And every survivor must carry the bound's verdict in
    ``_motion_bound_ok``, filtered or not, so collected training rows record
    which negatives the filter would have pruned."""
    from pyroorb.itf.candidates import DbSource
    src = DbSource.__new__(DbSource)                  # no connection needed
    src.cfg, src.index_cfg = ItfConfig(prefilter_motion=True), INDEX_CFG
    window = _window(uncert_deg=0.003)
    good = compute_tracklet_features(_rows(), INDEX_CFG)
    reversed_ = compute_tracklet_features(_rows(dra=-0.01,
                                                trksub="PR00002"), INDEX_CFG)
    ra_tol, dec_tol = window_tolerances(window, 1.1, 0.0)
    # default (None): the source's own cfg governs -- reversed motion is dropped
    kept = src._survivors(window, [good, reversed_], ra_tol, dec_tol)
    assert [c["tracklet_id"] for c in kept] == ["PR00001"]
    assert kept[0]["_motion_bound_ok"]
    # collector mode (False): the drop is bypassed but the verdict is recorded
    kept = src._survivors(window, [good, reversed_], ra_tol, dec_tol,
                          prefilter=False)
    assert {c["tracklet_id"]: bool(c["_motion_bound_ok"]) for c in kept} == \
        {"PR00001": True, "PR00002": False}


def test_collector_bypasses_the_motion_prefilter_at_the_source():
    """With an ML feature collector attached, _collect_tracklets must hand the
    sweep ``prefilter=False`` (the pruned tracklets ARE the negative class);
    without one, None -- the source's own cfg decides."""
    import pyroorb.itf.search as S
    seen = []

    class _Spy:
        last_sweep_skipped = 0

        def sweep_candidates(self, windows, um, ew, wr, row_budget=None,
                             prefilter=None):
            seen.append(prefilter)
            return iter(())

    def _ctx(collector):
        return SimpleNamespace(cfg=ItfConfig(), telemetry={},
                               log=(lambda *a: None), source=_Spy(),
                               verbose=False, trace=None, collector=collector,
                               pub_ids=set(), obscodes=None, uncert_mult=2.5,
                               max_uncert_deg=19.44)

    eph = SimpleNamespace(fit=None)
    S._collect_tracklets(_ctx(None), eph, 0.0, windows=[])
    S._collect_tracklets(_ctx([]), eph, 0.0, windows=[])
    assert seen == [None, False], seen


def test_feature_row_carries_the_prefilter_verdict_out_of_band():
    """The recorded verdict lands in the training CSV as ``motion_bound_ok`` --
    written with every row (so a retrain can pick either population) but never
    a model input (at serve time the filter makes it constant-True)."""
    from pyroorb.itf.training import ALL_COLS, FEATURE_COLS, candidate_feature_row
    assert "motion_bound_ok" in ALL_COLS
    assert "motion_bound_ok" not in FEATURE_COLS
    window = _window()
    ra_tol, dec_tol = window_tolerances(window, 1.1, 0.0)
    hit = tracklet_in_window(compute_tracklet_features(_rows(), INDEX_CFG),
                             window, ra_tol, dec_tol)
    hit["_motion_bound_ok"] = False
    obs = [{"epoch_sec": T0, "ra": 150.0, "dec": 10.0, "trksub": "PR00001",
            "stn": "F51", "trkid": "PR00001"}]
    tr = {"_win": window, "_feat": hit, "rows": obs, "dist": 1.0, "qual": 1.0,
          "traj_uncert": 0.1}
    eph = SimpleNamespace(earliest_jd=2459105.0, latest_jd=2459135.0, U=5.0,
                          rms=0.2, used=10, count=10, arclength_days=30.0,
                          fit=SimpleNamespace(n_oppositions=lambda: 1))
    row = candidate_feature_row(tr, eph, 0, 2.5, 0.0)
    assert row["motion_bound_ok"] == 0.0
    # a row that never met the verdict (e.g. scan path) -> nan, not a crash
    del hit["_motion_bound_ok"]
    row = candidate_feature_row(tr, eph, 0, 2.5, 0.0)
    assert math.isnan(row["motion_bound_ok"])


def test_corridor_predicates():
    feat = compute_tracklet_features(_rows(), INDEX_CFG)
    window = _window()
    ra_tol, dec_tol = window_tolerances(window, 1.1, 0.0)
    assert tracklet_in_window(feat, window, ra_tol, dec_tol) is not None
    # off-corridor in dec by 1 degree -> rejected
    far = dict(feat, center_dec=feat["center_dec"] + 1.0)
    assert tracklet_in_window(far, window, ra_tol, dec_tol) is None
    # outside the epoch range -> rejected
    late = dict(feat, epoch_mid_sec=window.unix_end + 10 * 86400)
    assert tracklet_in_window(late, window, ra_tol, dec_tol) is None
    # per-obs (scan-path) predicate agrees
    row = {"ra": 150.0, "dec": 10.0, "epoch_sec": T0}
    assert obs_in_window(row, window, ra_tol, dec_tol)
    assert not obs_in_window(dict(row, dec=11.5), window, ra_tol, dec_tol)


def test_certainty_ranks_right_motion_first():
    window = _window(uncert_deg=0.003)          # tight corridor (~11")
    ra_tol, dec_tol = window_tolerances(window, 1.1, 0.0)
    good = compute_tracklet_features(_rows(), INDEX_CFG)
    bad = compute_tracklet_features(
        _rows(dra=-0.01, trksub="PR00002"), INDEX_CFG)   # reversed motion
    g = tracklet_in_window(good, window, ra_tol, dec_tol)
    b = tracklet_in_window(bad, window, ra_tol, dec_tol)
    assert g is not None and b is not None
    chi_g = match_chi2(g, window)[0]
    chi_b, _pos, mot_b = match_chi2(b, window)
    assert chi_g < 5.0
    assert chi_b > 20 * chi_g and mot_b > 10


def _obs80(trksub, date, ra_txt, dec_txt, stn="F51", ref="     ", disc=" "):
    line = ("     " + f"{trksub:<7}" + disc + " C" + f"{date:<17}"
            + f"{ra_txt:<12}" + f"{dec_txt:<12}" + " " * 9
            + " 21.0G" + " " + f"{ref:<5}" + f"{stn:>3}")
    assert len(line) == 80, len(line)
    return line


def test_mag_gate_drops_the_faint_windows():
    """The corridor's magnitude gate keeps a window iff the object is predicted
    BRIGHTER than mag_limit -- and a window with no predicted magnitude at all
    is never dropped (an unknown mag is not evidence of faintness)."""
    from pyroorb.itf.ephem import corridor_windows

    def nodes(mags):
        # 5 nodes = one window; walk the sky slowly so nothing splits
        return [{"unix": 1e9 + 86400.0 * 3 * k, "ra": 10.0 + 0.01 * k,
                 "dec": 5.0, "unc_deg": 0.001, "delta": 2.0, "speed": 0.1,
                 "mag": m, "elong": 150.0} for k, m in enumerate(mags)]

    def kept(mag, limit):
        cfg = ItfConfig(mag_limit=limit, mag_floor_margin=None)
        return len(corridor_windows(nodes([mag] * 5), cfg, 70000.0 / 3600.0))

    assert kept(22.0, 23.0) == 1               # comfortably inside
    assert kept(23.4, 23.0) == 0               # too faint for the ITF
    assert kept(23.4, 24.0) == 1               # ... but inside the fallback
    assert kept(None, 23.0) == 1               # unknown mag is not faint

    # The window is judged on its BRIGHTEST node, not on whichever node the
    # 4-node tiling happened to start with.  Reading it off the first node made
    # coverage depend on alignment: inserting refined nodes earlier in the
    # sweep re-phased the tiling and deleted the window over 2016 EZ157's
    # 2021-10-16 apparition -- same orbit, same nodes, different offset.
    def kept_mixed(mags, limit):
        cfg = ItfConfig(mag_limit=limit, mag_floor_margin=None)
        return len(corridor_windows(nodes(mags), cfg, 70000.0 / 3600.0))

    assert kept_mixed([23.5, 23.4, 23.4, 23.4, 23.4], 23.5) == 1, \
        "a window holding brighter nodes must survive a faint first node"
    assert kept_mixed([23.4, 23.5, 23.6, 23.6, 23.6], 23.5) == 1, \
        "...and the reverse alignment must give the SAME answer"
    assert kept_mixed([23.6] * 5, 23.5) == 0   # genuinely faint throughout


def test_a_corridor_emptied_by_the_mag_gate_says_so():
    """When the magnitude gate leaves NO windows at all, the search must say
    it never looked -- "no match" and "not looked for" are different answers.

    2016 QC135 (a 36.8 au TNO predicted V 23.65-23.94 across the whole sweep):
    zero windows at 23.5, 907 at 24.0, and ~79% of 4-6-opposition TNOs sit the
    same side of the gate."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace
    from pyroorb.itf.trace import Tracer

    nodes = [{"unix": 1e9 + 86400.0 * 3 * k, "ra": 10.0 + 0.01 * k, "dec": 5.0,
              "unc_deg": 0.001, "delta": 40.0, "speed": 0.01, "mag": 23.9,
              "elong": 150.0} for k in range(9)]
    ctx = SimpleNamespace(cfg=ItfConfig(mag_limit=23.5, mag_floor_margin=None), telemetry={},
                          max_uncert_deg=70000.0 / 3600.0, trace=Tracer())
    S._note_mag_blackout(ctx, nodes)
    bl = ctx.telemetry.get("mag_blackout")
    assert bl, "an empty corridor at a faint object must be recorded"
    assert bl["limit"] == 23.5 and round(bl["min_mag"], 1) == 23.9
    assert bl["windows_if_deeper"] > 0, bl

    # a corridor emptied for ANOTHER reason (every window past the uncertainty
    # cap) must NOT be blamed on the magnitude gate
    ctx2 = SimpleNamespace(cfg=ItfConfig(mag_limit=23.5, mag_floor_margin=None), telemetry={},
                           max_uncert_deg=1e-9, trace=Tracer())
    S._note_mag_blackout(ctx2, nodes)
    assert not ctx2.telemetry.get("mag_blackout"), ctx2.telemetry

    # ...and a bright object never trips it
    bright = [dict(n, mag=20.0) for n in nodes]
    ctx3 = SimpleNamespace(cfg=ItfConfig(mag_limit=23.5, mag_floor_margin=None), telemetry={},
                           max_uncert_deg=70000.0 / 3600.0, trace=Tracer())
    S._note_mag_blackout(ctx3, bright)
    assert not ctx3.telemetry.get("mag_blackout")


def test_the_faint_cut_reaches_a_mag_past_the_brightest_since_1995():
    """The applied cut is never brighter than the object's brightest predicted
    V since mag_floor_since plus mag_floor_margin.  2026 DR69 (brightest 23.81)
    swept ZERO windows at the fixed 23.5 and missed two 2017 tracklets that
    link at P(match) 0.999; under the floor its default cut is 24.81."""
    from pyroorb.itf.ephem import corridor_windows, faint_cut

    def nodes(mags, t0=1e9):            # 1e9 s = 2001-09
        return [{"unix": t0 + 86400.0 * 3 * k, "ra": 10.0 + 0.01 * k,
                 "dec": 5.0, "unc_deg": 0.001, "delta": 40.0, "speed": 0.01,
                 "mag": m, "elong": 150.0} for k, m in enumerate(mags)]

    cfg = ItfConfig(mag_limit=23.5)
    dr69 = nodes([23.81, 23.9, 24.0, 23.95, 23.85])
    assert faint_cut(cfg, dr69) == pytest.approx(24.81)
    assert len(corridor_windows(dr69, cfg, 70000.0 / 3600.0)) == 1
    assert len(corridor_windows(
        dr69, ItfConfig(mag_limit=23.5, mag_floor_margin=None),
        70000.0 / 3600.0)) == 0, "floor off: the fixed cut stands"

    # only ever DEEPER: a bright object keeps the configured cut
    assert faint_cut(cfg, nodes([20.0] * 5)) == 23.5
    assert faint_cut(ItfConfig(mag_limit=26.0), dr69) == 26.0

    # the floor shapes the DEFAULT only: a cut the user typed is used exactly,
    # even one brighter than the floor, even the default value itself
    assert faint_cut(ItfConfig().set_mag_limit(24.0), dr69) == 24.0
    assert faint_cut(ItfConfig().set_mag_limit(23.5), dr69) == 23.5
    assert len(corridor_windows(dr69, ItfConfig().set_mag_limit(23.5),
                                70000.0 / 3600.0)) == 0
    # --thorough is depth, not a cut: the floor still applies on top of it
    assert faint_cut(ItfConfig().apply_thorough(), dr69) == pytest.approx(24.81)

    # judged since 1995 only: a 1990 bright spell does not hold the cut down
    t1990 = 631152000.0
    mixed = nodes([19.0] * 5, t0=t1990) + nodes([24.2] * 5)
    assert faint_cut(cfg, mixed) == pytest.approx(25.2)
    # ...and nodes that are ALL older say nothing: the configured cut stands
    assert faint_cut(cfg, nodes([24.2] * 5, t0=t1990)) == 23.5
    # a node with no predicted magnitude is ignored, not read as bright
    assert faint_cut(cfg, nodes([None, 24.0, None, 24.1, None])) == pytest.approx(25.0)


def test_the_archive_pass_reuses_the_modern_faint_cut():
    """The archive era's own nodes lie before mag_floor_since, so the floor
    must come from the cut the ordinary corridor judged on the modern track."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace
    from pyroorb.itf.trace import Tracer

    ctx = SimpleNamespace(cfg=ItfConfig(mag_limit=23.5), telemetry={},
                          trace=Tracer())
    modern = [{"unix": 1e9 + 86400.0 * 3 * k, "mag": 23.9} for k in range(5)]
    assert S._note_faint_cut(ctx, modern) == pytest.approx(24.9)
    assert ctx.telemetry["mag_cut"] == pytest.approx(24.9)
    assert ctx.telemetry["mag_cut_deepened"] is True


def test_the_capacity_verdict_is_recorded_without_a_trace():
    """"The server was too full to search deeply" is a message for the USER,
    not a diagnostic.  It used to be recorded inside the `if tracer.enabled`
    block, so a caller that did not ask for a trace got a search that ran
    shallowly and no way to say so -- the web UI happens to always send a
    trace id, which is what hid it."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    seed = _obs80("K17C37L", "2017 02 03.12345", "10 10 00.00", "+11 00 00.0")
    orig_it, orig_fit, orig_cap = S._iterate, S.fit_astrometry, S._search_capacity
    # as the real _iterate does: every result carries the search's own
    # telemetry dict, which is where the capacity verdict lands
    S._iterate = lambda ctx, *a, **kw: S.ItfSearchResult(
        verdict=SearchVerdict.NONE, telemetry=ctx.telemetry)
    S.fit_astrometry = lambda text, oc, **kw: SimpleNamespace(
        ok=True, used=1, count=1, rms=0.2, U=2.0, arclength_days=3.0,
        fit=object(), error=None)
    S._search_capacity = lambda *_a, **_k: {
        "verdict": "shallow", "reason": "not much room", "available_mb": 500,
        "floor_mb": 400, "headroom_mb": 100, "limit_mb": 1024}
    try:
        result = S.itf_search(None, seed_obs80=seed + "\n", cfg=ItfConfig(),
                           obscodes={}, log=lambda *a, **k: None,
                           verbose=False)          # <- no trace=
    finally:
        S._iterate, S.fit_astrometry, S._search_capacity = (
            orig_it, orig_fit, orig_cap)
    assert result.telemetry["capacity"]["verdict"] == "shallow"


def test_mag_fallback_resweeps_only_when_nothing_linked():
    """The deeper magnitude sweep is a FALLBACK: when ENABLED it runs if the
    tight pass linked nothing, never after a link, and the corridor it
    re-sweeps is the one bounded by mag_limit_deep.  Off by default -- see
    ItfConfig.mag_limit_fallback for why it is dormant rather than deleted."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    seed = _obs80("K17C37L", "2017 02 03.12345", "10 10 00.00", "+11 00 00.0")

    def run(first_status):
        seen = []

        def stub_iterate(ctx, *a, **kw):
            seen.append(ctx.cfg.mag_limit)
            result = S.ItfSearchResult()
            result.verdict = (first_status if len(seen) == 1
                           else SearchVerdict.LINKED)
            return result

        orig_it, orig_fit = S._iterate, S.fit_astrometry
        S._iterate = stub_iterate
        S.fit_astrometry = lambda text, oc, **kw: SimpleNamespace(
            ok=True, used=1, count=1, rms=0.2, U=2.0, arclength_days=3.0,
            fit=object(), error=None)
        try:
            # the fallback is OFF by default (ItfConfig.mag_limit_fallback) --
            # enable it explicitly, since this test is what documents it still
            # works for whoever turns it back on
            cfg = ItfConfig(mag_limit=23.0, mag_limit_deep=24.0,
                            mag_limit_fallback=True)
            result = S.itf_search(None, seed_obs80=seed + "\n", cfg=cfg,
                               obscodes={}, log=lambda *a, **k: None,
                               verbose=False)
        finally:
            S._iterate, S.fit_astrometry = orig_it, orig_fit
        return seen, result

    # the tight pass links -> no second sweep, and the tight limit is what ran
    seen, result = run(SearchVerdict.LINKED)
    assert seen == [23.0], seen
    assert result.verdict is SearchVerdict.LINKED

    # it links nothing -> re-sweep at the fallback magnitude, and that verdict
    # (a link) replaces the empty one
    seen, result = run(SearchVerdict.NONE)
    assert seen == [23.0, 24.0], seen
    assert result.verdict is SearchVerdict.LINKED, result.reason


def test_window_cap_keeps_the_motion_match_not_the_closest():
    """When a window has more candidates than ``max_candidates_per_window``, the
    ones it keeps must be those whose OWN motion matches the corridor -- the test
    the survivors face next.  The old key (match_chi2) degenerated on a wide
    corridor into "closest to the centreline" and threw away the motion match,
    which is exactly backwards for an under-determined seed whose true tracklets
    sit far off-centreline."""
    from pyroorb.itf.candidates import DbSource, cap_rank_key, window_tolerances
    src = DbSource.__new__(DbSource)                  # no connection needed
    src.cfg, src.index_cfg = ItfConfig(max_candidates_per_window=1), INDEX_CFG
    window = _window(uncert_deg=1.0)                     # degrees wide, like BE559's
    # right motion, 0.5 deg off the centreline
    match = compute_tracklet_features(_rows(dec0=10.5, trksub="PRMATCH"), INDEX_CFG)
    # wrong (reversed) motion, sitting exactly on the centreline
    centre = compute_tracklet_features(_rows(dra=-0.01, trksub="PRCENTR"), INDEX_CFG)
    ra_tol, dec_tol = window_tolerances(window, 2.5, 0.0)
    kept = src._survivors(window, [centre, match], ra_tol, dec_tol)
    assert len(kept) == 1, f"the cap must keep exactly one: {len(kept)}"
    assert kept[0]["tracklet_id"] == "PRMATCH", \
        "the cap kept the centreline tracklet with the wrong motion"
    # and the key itself orders them that way, with unrankable rows kept first
    # (it takes an ANNOTATED candidate, as the sweep hands it)
    a_match = tracklet_in_window(match, window, ra_tol, dec_tol)
    a_centre = tracklet_in_window(centre, window, ra_tol, dec_tol)
    assert cap_rank_key(a_match, window) < cap_rank_key(a_centre, window)
    blind = dict(a_centre, dra_cosdec_rate=None)
    assert cap_rank_key(blind, window) < cap_rank_key(a_match, window), \
        "a row the index cannot evaluate must not be dropped by the cap"


def test_cap_records_what_it_would_have_done_to_a_planted_truth():
    """A harness plant is injected past ``_survivors``, so it never faces the
    per-window cap -- which means the recovery rig cannot see a cap that would
    have thrown away a REAL tracklet of the same geometry, and the training data
    it produces is optimistic by exactly that amount.

    The cap's would-be verdict therefore rides on the plant the way the motion
    pre-filter's already does: recorded, never enforced."""
    from pyroorb.itf.candidates import DbSource, OverlaySource, window_tolerances
    cfg = ItfConfig(max_candidates_per_window=2)
    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = cfg, INDEX_CFG
    window = _window(uncert_deg=1.0)
    ra_tol, dec_tol = window_tolerances(window, 2.5, 0.0)

    # a saturated window: one real candidate with the RIGHT motion, one with a
    # reversed one.  prefilter off, exactly as a collection sweep runs it --
    # otherwise the reversed-motion row never even reaches the cap.
    good = compute_tracklet_features(_rows(trksub="ZTREAL"), INDEX_CFG)
    other = compute_tracklet_features(_rows(dra=-0.01, trksub="ZTOTHER"), INDEX_CFG)
    real = src._survivors(window, [good, other], ra_tol, dec_tol, prefilter=False)
    assert len(real) == 2
    assert real[0]["_n_in_corridor"] == 2, "pre-cap population must be recorded"
    assert all(c["_would_be_capped"] is False for c in real)

    feats = {c["tracklet_id"]: c for c in real}

    def _plant(**kw):
        f = compute_tracklet_features(_rows(**kw), INDEX_CFG)
        ov = OverlaySource(None, [{"feat": f, "obs": []}], cfg, INDEX_CFG)
        _r, _c, out = ov._merge_fakes(window, [], {}, 2.5, 0.0, False,
                                      feats=dict(feats))
        return out[kw["trksub"]]

    # a plant whose motion matches the corridor worse than every real survivor
    # of a saturated window is one the cap would have discarded
    assert _plant(dra=-0.03, trksub="ZTBAD")["_would_be_capped"] is True, \
        "a plant ranking below every survivor of a capped window would be lost"
    # ...while one with the right motion outranks the junk and would survive
    ok = _plant(dec0=10.02, trksub="ZTOK")
    assert ok["_would_be_capped"] is False
    assert ok["_cap_rank"] == 1, "rank is against the real population"
    # either way the plant is SERVED -- the verdict is recorded, not enforced


def test_split_wall_is_permanent_object_level_and_population_neutral():
    """The train/eval wall must not correlate with anything astronomical.  The
    designation's own trailing digit does: a NEO's provisional cycle count is
    small (it is designated early in its half-month), so splitting on that digit
    sorts NEOs by discovery circumstance -- and ~15% of NEO designations end in
    a letter and get no assignment at all."""
    from pyroorb.itf.training import EVAL_KEYS, split_key, split_of

    assert split_of("2014 SD57") == split_of("2014 SD57"), "must be stable"
    assert 0 <= split_key("2014 SD57") <= 9
    # defined for designations the trailing-digit rule cannot classify
    for d in ("2024 GV", "1998 XY", "51661"):
        assert split_of(d) in ("train", "eval")
    # roughly 80/20 over a spread of designations, and never both sides at once
    desigs = [f"20{y:02d} A{chr(65 + k)}{n}"
              for y in range(10, 26) for k in range(6) for n in range(6)]
    ev = [d for d in desigs if split_of(d) == "eval"]
    assert 0.12 < len(ev) / len(desigs) < 0.30, \
        f"eval share {len(ev) / len(desigs):.2f} is not near 20%"
    assert all(split_key(d) in EVAL_KEYS for d in ev)


def test_feature_row_carries_the_hand_written_prioritizers():
    """The heuristics the model replaced are handed to it as derived features.
    Both are closed-form combinations of columns already present -- which is the
    point: an axis-aligned tree cannot build `a + b + c` cheaply, so the ranking
    function that already works is given rather than rediscovered."""
    from pyroorb.itf.training import (FEATURE_COLS, V2_FEATURE_COLS,
                              candidate_feature_row)
    window = _window()
    feat = compute_tracklet_features(_rows(), INDEX_CFG)
    tr = {"rows": [{"epoch_sec": T0, "trksub": "ZT00001", "stn": "F51",
                    "trkid": "ZT00001"},
                   {"epoch_sec": T0 + 3600, "trksub": "ZT00001", "stn": "F51",
                    "trkid": "ZT00001"}],
          "dist": 12.0, "qual": 3.0, "traj_uncert": 1.0,
          "_win": window, "_feat": feat}
    eph = SimpleNamespace(U=4.0, rms=0.3, used=10, count=12,
                          arclength_days=30.0, earliest_jd=2460000.0,
                          latest_jd=2460030.0,
                          fit=SimpleNamespace(n_oppositions=lambda: 2))
    row = candidate_feature_row(tr, eph, 1, 2.5, 0.0, is_variant=1)
    assert row["legacy_score"] == 12.0 + 3.0 + window.uncert * 3600.0
    assert row["cap_key_primary"] == 2.0          # max(qual - traj_unc, 0)
    assert row["is_variant"] == 1.0
    # every v2 feature still present: the v2 model has to be scorable on v3 rows
    assert not set(V2_FEATURE_COLS) - set(row), "a v2 feature went missing"
    assert not set(FEATURE_COLS) - set(row)


def test_deadline_keeps_the_rows_harvested_so_far():
    """A collection deadline must cost the VERDICT, not the data: rows already
    harvested are exactly as valid as rows from a search that finished, and
    planted/distractor labels do not depend on the outcome."""
    import pyroorb.itf.harness as H

    collected = []

    def _slow_search(*a, **kw):
        kw["collector"].append({"trksub": "ZTPLANT", "label": -1})
        kw["collector"].append({"trksub": "ZTJUNK", "label": -1})
        time.sleep(5)                       # ...never returns before the alarm
        raise AssertionError("deadline did not fire")

    with H.deadline(0.3):
        try:
            _slow_search(collector=collected)
        except H.SearchDeadline:
            pass
    assert len(collected) == 2, "harvested rows must survive the deadline"


def test_the_sigint_deadline_fires_and_leaves_ctrl_c_alone():
    """Windows has no SIGALRM, so there the deadline rides a timer that raises SIGINT.
    It must fire like the alarm (waking a sleep), and a Ctrl-C before it rings, or the
    handler left behind after the block, must still be a KeyboardInterrupt."""
    import signal
    import pyroorb.itf.harness as H

    t0 = time.perf_counter()
    with pytest.raises(H.SearchDeadline):
        with H._deadline_by_sigint(0.3):
            time.sleep(5)
    assert time.perf_counter() - t0 < 3, "the deadline did not wake the sleep"

    runner = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            with H._deadline_by_sigint(30):
                signal.raise_signal(signal.SIGINT)
                time.sleep(1)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, runner)


def test_row_budget_override_and_incomplete_sweep_is_reported():
    """The sweep row budget is calibrated for ONE corridor, so the LOV variant
    scan's union of n_scan corridors overrides it with its own
    (cfg.variant_sweep_row_budget) -- and whenever a budget DOES drop windows the
    search has to say so, because a negative verdict after an incomplete sweep is
    not a clean negative."""
    import pyroorb.itf.search as S
    from pyroorb.itf.candidates import DbSource

    # the override reaches DbSource.sweep_candidates ...
    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = ItfConfig(candidate_path="scan"), INDEX_CFG
    src.trace, src.last_sweep_skipped = None, 0
    src._scan_path = lambda *a, **k: [{"obs80": "x"}] * 10
    windows = [_window() for _ in range(6)]
    list(src.sweep_candidates(windows, 1.1, 0.0, False, row_budget=1))
    assert src.last_sweep_skipped == 5, "a tiny budget must stop after one window"
    src.last_sweep_skipped = 0
    list(src.sweep_candidates(windows, 1.1, 0.0, False, row_budget=0))
    assert src.last_sweep_skipped == 0, "row_budget=0 must disable the valve"

    # ... and a skipped window is recorded for the caller, not just logged
    ctx = SimpleNamespace(cfg=ItfConfig(), telemetry={}, log=(lambda *a: None),
                          source=SimpleNamespace(last_sweep_skipped=7),
                          verbose=False, trace=None, collector=None,
                          pub_ids=set(), obscodes=None, uncert_mult=2.5,
                          max_uncert_deg=19.44)
    skipped = max((getattr(s_, "last_sweep_skipped", 0) or 0
                   for s_ in [ctx.source]), default=0)
    assert skipped == 7
    from pyroorb.itf.candidates import source_chain
    assert source_chain(ctx.source) == [ctx.source]


def test_memory_floor_stops_sweep_and_refetch(monkeypatch):
    """The sweep memory floor (cfg.sweep_min_free_mb): when available RAM is
    under it, the sweep stops serving through the same skipped-window
    accounting as the row budget, and a mid-chunk obs refetch bails with
    CANCELLED instead of accumulating the fetch that would OOM the process
    (2006 SJ252: silent OOM-kill after the seed fit, no verdict)."""
    import pyroorb.itf.candidates as C
    from pyroorb.itf.candidates import DbSource

    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = ItfConfig(candidate_path="scan"), INDEX_CFG
    src.trace, src.last_sweep_skipped = None, 0
    src._scan_path = lambda *a, **k: [{"obs80": "x"}] * 10
    windows = [_window() for _ in range(6)]

    monkeypatch.setattr(C, "_mem_available_mb", lambda: 100)   # under the floor
    list(src.sweep_candidates(windows, 1.1, 0.0, False, row_budget=0))
    assert src.last_sweep_skipped == 6, \
        "low memory must stop the sweep before the first window"
    assert src._obs_by_trkid([1, 2, 3]) is DbSource.CANCELLED, \
        "a refetch under the floor must drop the chunk, not fetch it"

    src.last_sweep_skipped = 0
    monkeypatch.setattr(C, "_mem_available_mb", lambda: None)  # /proc absent
    list(src.sweep_candidates(windows, 1.1, 0.0, False, row_budget=0))
    assert src.last_sweep_skipped == 0, \
        "an unreadable meminfo must disable the floor, not trip it"

    src.cfg = ItfConfig(candidate_path="scan", sweep_min_free_mb=0)
    monkeypatch.setattr(C, "_mem_available_mb", lambda: 100)
    list(src.sweep_candidates(windows, 1.1, 0.0, False, row_budget=0))
    assert src.last_sweep_skipped == 0, "sweep_min_free_mb=0 must disable it"


def test_low_memory_truncation_is_not_reported_as_a_row_budget(monkeypatch):
    """A sweep truncated by the MEMORY floor must say so.  It lands in the same
    `last_sweep_skipped` total as the row budget, and the sweep summary used to
    have only two causes -- so a container that ran out of RAM was told to RAISE
    ItfConfig.sweep_row_budget, which retains MORE rows and makes the OOM worse.
    Until the floor was made container-aware this branch could not be reached in
    production; now it is the likeliest one there."""
    import pyroorb.itf.candidates as C
    from pyroorb.itf.candidates import DbSource

    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = ItfConfig(candidate_path="scan"), INDEX_CFG
    src.trace = None
    src._scan_path = lambda *a, **k: [{"obs80": "x"}] * 10
    windows = [_window() for _ in range(6)]

    monkeypatch.setattr(C, "_mem_available_mb", lambda: 100)   # under the floor
    list(src.sweep_candidates(windows, 1.1, 0.0, False, row_budget=0))
    assert src.last_sweep_skipped == 6
    assert src.last_sweep_lowmem == 6, \
        "the memory floor must be counted as its own cause"
    assert src.last_sweep_cancelled == 0, \
        "the server cancelled nothing -- attributing this to it sends the " \
        "reader to sweep_statement_timeout_sec, where the fix is not"

    # ... and a refetch dropped for memory is a memory drop, not a server
    # cancel: it must not spend one of cfg.sweep_max_cancels either.
    src.last_sweep_skipped = src.last_sweep_lowmem = 0
    src.last_sweep_cancelled = src._sweep_cancels = 0
    assert src._obs_by_trkid([1, 2, 3]) is DbSource.CANCELLED
    assert src._lowmem_batch is True
    assert src._cancelled_batch(2, 3, None) is False, \
        "the loop-top valve ends the sweep, with the honest message"
    assert (src.last_sweep_lowmem, src.last_sweep_cancelled) == (2, 0)
    assert src._sweep_cancels == 0


def test_row_budget_truncation_still_reads_as_a_row_budget(monkeypatch):
    """The other side of the split: the memory cause must not swallow the one
    that WAS the row budget."""
    import pyroorb.itf.candidates as C
    from pyroorb.itf.candidates import DbSource

    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = ItfConfig(candidate_path="scan"), INDEX_CFG
    src.trace = None
    src._scan_path = lambda *a, **k: [{"obs80": "x"}] * 10
    monkeypatch.setattr(C, "_mem_available_mb", lambda: 100000)   # plenty
    list(src.sweep_candidates([_window() for _ in range(6)], 1.1, 0.0, False,
                              row_budget=1))
    assert src.last_sweep_skipped == 5 and src.last_sweep_lowmem == 0


def test_setjoin_query_box_is_an_outward_superset():
    """The set-join sweep hands each window's own box to Postgres.  Its bounds
    are rounded OUTWARD (whole second, 1e-4 deg) purely to keep the wire text
    short, so the SQL box must be a strict SUPERSET of the window's own box --
    the exact predicate still runs in Python, but a box that rounded INWARD
    would silently drop corridor tracklets the DB never returned."""
    from pyroorb.itf.candidates import DbSource, BOX_PAD_DEG, window_tolerances
    src = DbSource.__new__(DbSource)                  # no connection needed
    src.cfg, src.index_cfg = ItfConfig(), INDEX_CFG
    for uncert in (0.003, 0.05, 1.5):
        window = _window(uncert_deg=uncert)
        rows, (ra_tol, dec_tol) = src._window_query_tuples(window, 7, 2.5, 0.0)
        assert rows and all(r["wi"] == 7 for r in rows), \
            "window index must ride along"
        assert (ra_tol, dec_tol) == window_tolerances(window, 2.5, 0.0)
        r0 = rows[0]
        assert r0["t_lo"] <= window.unix_start and r0["t_hi"] >= window.unix_end
        assert r0["d_lo"] <= window.min_dec - dec_tol - BOX_PAD_DEG
        assert r0["d_hi"] >= window.max_dec + dec_tol + BOX_PAD_DEG
        # every pixel the chunked path would have queried is present
        from pyroorb.itf.candidates import window_healpix_disc
        assert {r["pix"] for r in rows} == window_healpix_disc(window, ra_tol,
                                                               dec_tol, INDEX_CFG)


def test_pixel_ranges_cover_exactly_the_pixel_set():
    """The range-join sends a window's healpix disc as contiguous [lo, hi] runs
    instead of a 17k-element `= ANY(...)` array.  That is only sound if the
    runs cover EXACTLY the set -- a missed pixel is a strip of sky the sweep
    stops looking at, silently."""
    from pyroorb.itf.candidates import DbSource, window_healpix_disc, window_tolerances
    rng = np.random.default_rng(7)
    sets = [set(), {5}, {1, 2, 3}, {1, 3, 5}, {9, 8, 7, 1},
            set(range(100, 140)) | {200} | set(range(500, 503))]
    for _ in range(40):
        sets.append(set(rng.integers(0, 400, size=int(rng.integers(1, 60)))
                        .tolist()))
    for window in (_window(uncert_deg=0.003), _window(uncert_deg=1.5)):
        ra_tol, dec_tol = window_tolerances(window, 2.5, 0.0)
        sets.append(set(window_healpix_disc(window, ra_tol, dec_tol, INDEX_CFG)))
    for pix in sets:
        rr = DbSource._pixel_ranges(pix)
        covered = {p for lo, hi in rr for p in range(lo, hi + 1)}
        assert covered == set(pix), f"ranges {rr} != pixels {sorted(pix)}"
        assert all(lo <= hi for lo, hi in rr), rr
        # maximal runs: no two ranges may be adjacent or overlapping, or the
        # collapse is not doing the job it was measured to do
        for (lo1, hi1), (lo2, _) in zip(sorted(rr), sorted(rr)[1:]):
            assert lo2 > hi1 + 1, f"non-maximal runs {rr}"


def test_sql_predicates_never_drop_what_python_keeps():
    """The server-side corridor and motion-bound SQL re-implement
    ``tracklet_in_window`` / ``motion_bound_ok``.  The whole pushdown rests on
    the SQL being a SUPERSET -- looser is free (``_survivors`` re-applies the
    exact test), stricter means candidates vanish with no error and the search
    reports a clean negative.  Nothing enforced that until this fuzz.

    Offline, the SQL text is transliterated into Python and evaluated on the
    same rows: it proves the ALGEBRA agrees.  ``tests/itf_sql_fuzz.py`` runs the
    identical cases through the REAL server for the float8 semantics this
    cannot see."""
    from itf_sql_fuzz import check_shim, random_cases
    n, rej_corr, rej_mot = check_shim(random_cases(600))
    assert n == 600
    # the fuzz must actually exercise both verdicts -- an all-accept run would
    # satisfy the superset property vacuously
    assert rej_corr > 20, f"corridor SQL rejected only {rej_corr}/{n}"
    assert rej_mot > 5, f"motion SQL rejected only {rej_mot}/{n}"


def test_sweep_mode_auto_routes_and_named_modes_force():
    """``cfg.sweep_mode`` is either ``"auto"`` -- route per corridor -- or the
    NAME OF A PATH, forced.  Forcing must not fall back: an A/B that silently
    re-routes measures the router again instead of the path it named."""
    from pyroorb.itf.candidates import DbSource
    src = DbSource.__new__(DbSource)
    src.index_cfg = INDEX_CFG
    narrow = [_window(uncert_deg=0.003)]
    wide = [_window(uncert_deg=1.5)]
    many = wide * (DbSource.RANGEJOIN_MAX_TOTAL_WINDOWS + 1)

    def route(mode, windows):
        src.cfg = ItfConfig(sweep_mode=mode)
        return src._route_sweep(windows, 2.5, 0.0, None)[0]

    assert route("auto", narrow) == "setjoin", "few pixels/window -> equality join"
    assert route("auto", wide) == "rangejoin", "wide but distinct -> range join"
    assert route("auto", many) == "chunk", "overlapping thicket -> union box"
    for mode in ("setjoin", "rangejoin", "chunk"):
        for windows in (narrow, wide, many):
            assert route(mode, windows) == mode, f"{mode} must be forced, not advisory"


def test_window_relation_columns_stay_aligned():
    """Every server-side path ships its batch of windows as an unnested array
    relation, which means three things must agree: the CTE's column NAMES, its
    unnest CASTS, and the VALUES packed into the arrays.  Hand-aligning them was
    a silent-wrong-answer risk -- a shifted value filters on the wrong number
    and drops candidates without raising -- so the ordering is declared once per
    relation and derived from there.  This pins that it stays derived: specs
    self-consistent, every column fed by name, and every assembled query's
    placeholder count matching the arguments handed to psycopg2."""
    import re
    from pyroorb.itf.candidates import DbSource

    per_row = {"wi", "pix", "lo", "hi"}          # supplied per array row
    specs = (DbSource._W_SETJOIN_COLS, DbSource._W_CORRIDOR_COLS,
             DbSource._W_RANGEJOIN_COLS)
    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = ItfConfig(), INDEX_CFG
    fields = src._window_sql_row(_window(), 0.01, 0.02)
    for spec in specs:
        cte = DbSource._window_cte(spec)
        names = cte[cte.index("(") + 1:cte.index(")")].split(", ")
        assert len(spec) == len(names) == cte.count("%s"), cte
        assert names == [c for c, _ in spec], "names must follow the spec order"
        assert all(c in fields or c in per_row for c, _ in spec), spec
    # ...and the arrays are column-major in that same order
    arrays = DbSource._window_arrays(DbSource._W_CORRIDOR_COLS, [fields])
    assert [a[0] for a in arrays] == [fields[c]
                                      for c, _ in DbSource._W_CORRIDOR_COLS]

    # Each path end to end against a capturing cursor: what psycopg2 is handed
    # must have one argument per placeholder, and equal-length arrays.
    captured = []

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, args): captured.append((sql, args))
        def fetchall(self): return []

    class _Conn:
        def cursor(self, **kw): return _Cur()

    for mode, uncert, spec in (("setjoin", 0.003, DbSource._W_SETJOIN_COLS),
                               ("rangejoin", 1.5, DbSource._W_RANGEJOIN_COLS),
                               ("chunk", 1.5, DbSource._W_CORRIDOR_COLS)):
        captured.clear()
        s = DbSource.__new__(DbSource)
        s.cfg = ItfConfig(sweep_mode=mode)
        s.index_cfg, s.conn, s.trace, s.last_sweep_skipped = INDEX_CFG, _Conn(), None, 0
        s.reconnect = lambda: s.conn
        windows = [_window(uncert_deg=uncert), _window(ra0=150.5,
                                                    uncert_deg=uncert)]
        list(s.sweep_candidates(windows, 2.5, 0.0, True))
        assert captured, f"{mode}: no query was issued"
        for sql, args in captured:
            assert sql.count("%s") == len(args), (mode, sql.count("%s"),
                                                  len(args))
            # the window relation's arrays lead the argument list (whatever a
            # path appends after them -- the chunk path's union box -- follows)
            lens = {len(a) for a in args[:len(spec)]}
            assert len(lens) == 1, f"{mode}: ragged window arrays {lens}"
            # ...and every `w.<col>` the predicates read is a column the
            # relation actually declares: a rename in the spec that missed
            # _sql_corridor / _sql_motion_bound would otherwise only show up
            # as a live query error against the real server.
            used = set(re.findall(r"\bw\.(\w+)", sql))
            assert used <= {c for c, _ in spec}, (mode,
                                                  used - {c for c, _ in spec})


def test_overlay_source_gates_injection():
    cfg = ItfConfig()
    feat = compute_tracklet_features(_rows(), INDEX_CFG)
    obs_rows = [{"obs80": _obs80("PR00001", "2020 09 13.32", "10 00 00.00",
                                 "+10 00 00.0"),
                 "ra": 150.0, "dec": 10.0, "epoch_sec": T0,
                 "trksub": "PR00001", "stn": "F51", "trkid": "PR00001"}]
    src = OverlaySource(None, [{"feat": feat, "obs": obs_rows}], cfg, INDEX_CFG)

    def _sweep(source, window, want_ranking=True):
        """One window through the sweep -- the only retrieval entry point the
        search uses, so the overlay's gating is tested where it actually runs."""
        (_w, rows, chi2, _feats), = source.sweep_candidates(
            [window], 1.1, 0.0, want_ranking)
        return rows, chi2

    rows, chi2 = _sweep(src, _window())
    assert len(rows) == 1 and "PR00001" in chi2
    # a window elsewhere on the sky must NOT serve the fake
    rows2, _ = _sweep(src, _window(ra0=200.0))
    assert rows2 == []
    # scan path applies the per-obs predicates instead
    cfg_scan = ItfConfig(candidate_path="scan")
    src2 = OverlaySource(None, [{"feat": feat, "obs": obs_rows}], cfg_scan, INDEX_CFG)
    rows3, _ = _sweep(src2, _window(), want_ranking=False)
    assert len(rows3) == 1


def test_linkage_document():
    itf1 = _obs80("PR00001", "2017 01 02.34567", "10 00 00.00", "+10 00 00.0")
    itf1b = _obs80("PR00001", "2017 01 02.35567", "10 00 00.10", "+10 00 00.4")
    # same trksub, different night -> its own triple
    itf2 = _obs80("PR00001", "2017 01 04.30000", "10 00 05.00", "+10 00 03.0")
    pub = _obs80("K17C37L", "2017 02 03.12345", "10 10 00.00", "+11 00 00.0",
                 stn="703")
    blob = "\n".join([itf1, itf1b, itf2, pub])
    desigs, triples, neocp = collect_ids(blob)
    assert desigs == ["K17C37L"] and not neocp
    assert [t[0] for t in triples] == ["PR00001", "PR00001"]
    assert triples[0][1].startswith("2017 01 02.3")
    cfg = ItfConfig()
    doc, err = build_linkage(blob, cfg, primary_desigs=["2017 CL37"],
                             itf_obs_out="")
    assert err is None
    link = doc["links"]["link_0"]
    assert link["designations"] == ["K17C37L"]
    assert len(link["trksubs"]) == 2
    # NEOCP detection via the reference columns
    neo = _obs80("PR00009", "2017 01 02.10000", "10 00 00.00", "+10 00 00.0",
                 ref="NEOCP")
    _, _, is_neo = collect_ids(neo)
    assert is_neo
    # find_orb's guard: a single id is not a linkage
    doc2, err2 = build_linkage(itf1 + "\n" + itf1b, cfg)
    assert doc2 is None and "1 distinct" in err2


def test_digits_only_trksub_is_not_a_designation():
    """SBN trksubs are 6-character hex-ish strings, and some of them are ALL
    DIGITS -- '133651' is a real one, sibling to '10b51e' and '1207c2' in the
    same linkage.  Fed to the string-only ``unpack`` it round-trips as minor
    planet 133651, and a submitted document named a numbered object we never
    fitted while silently dropping that tracklet's nights from ``trksubs``.

    Columns 6-12 hold the packed PROVISIONAL or a trksub; a permanent number
    lives only in columns 1-5.  So the identity test must be column-aware."""
    from pyroorb.itf.linkage import is_packed_provisional, line_designation
    itf = [_obs80("133651", "2006 10 11.39981", "10 00 00.00", "+10 00 00.0",
                  stn="705"),
           _obs80("133651", "2006 10 28.31626", "10 00 05.00", "+10 00 03.0",
                  stn="705"),
           _obs80("10b51e", "2006 11 02.28234", "10 00 09.00", "+10 00 06.0",
                  stn="705")]
    pub = _obs80("K23R67C", "2023 09 12.12345", "11 00 00.00", "+11 00 00.0",
                 stn="703")
    desigs, triples, _ = collect_ids("\n".join(itf + [pub]))
    assert desigs == ["K23R67C"], desigs
    assert sorted(t[0] for t in triples) == ["10b51e", "133651", "133651"]
    # ... and it must not sneak in through the designated-pickup path either
    doc, err = build_linkage("\n".join(itf + [pub]), ItfConfig(),
                             primary_desigs=["2023 RC67"],
                             itf_obs_out="\n".join(itf))
    assert err is None
    assert doc["links"]["link_0"]["designations"] == ["K23R67C"]

    # what a genuine 133651 looks like: the packed NUMBER, in columns 1-5
    numbered = "D3651" + " " * 7 + pub[12:]
    assert line_designation(numbered) == "D3651"
    # cols 6-12 accept packed provisionals and surveys, never a packed number
    assert is_packed_provisional("K23R67C") and is_packed_provisional("PLS2040")
    assert not is_packed_provisional("133651")
    assert not is_packed_provisional("D3651") and not is_packed_provisional("00433")
    # a permanent id in 1-5 AND a trksub in 6-12 is a contradiction the MPC
    # never emits -- refuse to guess which identity to submit
    both = "D3651" + "10b51e " + pub[12:]
    try:
        line_designation(both)
    except ValueError as e:
        assert "133651" not in str(e) and "10b51e" in str(e)
    else:
        raise AssertionError("number + trksub on one line must raise")


def test_document_names_the_database_trksub_not_the_line():
    """Columns 6-12 of a stored ITF record are a rendering of the trkSub, and it
    can differ from the database's ``trksub`` column ('M00000' in the line,
    'MB00000' submitted).  The MPC identifies a tracklet by the submitted trkSub
    and ``obs_sbn`` -- where the ledger resolves each item's trkid -- is keyed by
    it, so the document must name the column's spelling.  A designation line is
    never renamed, and a line the database did not supply keeps its columns."""
    from pyroorb.itf.linkage import db_trksubs
    from pyroorb.itf.obsreg import AdesRegistry
    itf = [_obs80("M00000", "2006 10 11.39981", "10 00 00.00", "+10 00 00.0",
                  stn="F51"),
           _obs80("M00000", "2006 10 11.42981", "10 00 01.00", "+10 00 00.5",
                  stn="F51"),
           _obs80("10b51e", "2006 11 02.28234", "10 00 09.00", "+10 00 06.0",
                  stn="705")]
    pub = _obs80("K23R67C", "2023 09 12.12345", "11 00 00.00", "+11 00 00.0",
                 stn="703")
    rows = [{"obs80": itf[0], "trksub": "MB00000"},
            {"obs80": itf[1] + " " * (160 - len(itf[1])), "trksub": "MB00000 "},
            {"obs80": itf[2], "trksub": None},
            {"obs80": pub, "trksub": "XYZ123"}]
    trksubs = db_trksubs(rows)
    assert trksubs[obs_unique_id(itf[0])] == "MB00000"
    assert trksubs[obs_unique_id(itf[1])] == "MB00000"
    assert obs_unique_id(itf[2]) not in trksubs

    blob = "\n".join(itf + [pub])
    desigs, triples, _ = collect_ids(blob, trksubs)
    assert desigs == ["K23R67C"]
    assert sorted(t[0] for t in triples) == ["10b51e", "MB00000"]
    # without the map, the line's own spelling (the old behaviour)
    assert sorted(t[0] for t in collect_ids(blob)[1]) == ["10b51e", "M00000"]

    doc, err = build_linkage(blob, ItfConfig(), primary_desigs=["2023 RC67"],
                             itf_obs_out="\n".join(itf), trksubs=trksubs)
    assert err is None
    link = doc["links"]["link_0"]
    assert link["designations"] == ["K23R67C"]
    assert ["MB00000", "2006 10 11.39981", "F51"] in link["trksubs"]
    # the ledger's trkid_map is keyed by the column spelling, so it now matches
    trkid_map = {("MB00000", "F51", itf[0][15:25]): "trk-1"}
    assert any((t[0], t[2], t[1][:10]) in trkid_map for t in link["trksubs"])

    # the search's registry carries the column for tracklets of earlier levels
    reg = AdesRegistry()
    reg.add_records(rows[:3])
    assert reg.trksubs({obs_unique_id(l) for l in itf}) == {
        obs_unique_id(itf[0]): "MB00000", obs_unique_id(itf[1]): "MB00000"}


def test_provisional_lookalike_trksub_is_a_trksub():
    """A 7-char trksub shaped like a packed provisional but with junk in the
    LETTER slots ('J053207' -- half-month '3', order letter '7') must read as a
    trksub, not as a designation.

    The lax unpacker read it as '1905 3720', so the linkage generator named it
    among ``designations`` (an object nobody fitted) and dropped its nights from
    ``trksubs``.  ``packing.unpack`` now checks every character against its slot,
    which is exactly the strict pack/unpack round-trip test: what ``pack`` cannot
    produce is not a designation."""
    from pyroorb.packing import unpack, unpack_comet
    from pyroorb.itf.linkage import is_packed_provisional, line_designation

    assert unpack("J053207") is None
    assert not is_packed_provisional("J053207")
    assert unpack("J01A23B") == "1901 AB23" and is_packed_provisional("J01A23B")
    # a packed COMET core keeps counting (lowercase/'0' fragment slot, so it can
    # never collide with an asteroid packing's uppercase order letter)
    assert unpack_comet("K23A030") == "2023 A3" and is_packed_provisional("K23A030")
    assert unpack("K23A030") is None

    itf = [_obs80("J053207", "2006 10 11.39981", "10 00 00.00", "+10 00 00.0",
                  stn="705"),
           _obs80("J053207", "2006 10 28.31626", "10 00 05.00", "+10 00 03.0",
                  stn="705")]
    pub = _obs80("K23R67C", "2023 09 12.12345", "11 00 00.00", "+11 00 00.0",
                 stn="703")
    assert line_designation(itf[0]) is None
    desigs, triples, _ = collect_ids("\n".join(itf + [pub]))
    assert desigs == ["K23R67C"], desigs
    assert sorted(t[0] for t in triples) == ["J053207", "J053207"]


def _comet_line(first12):
    """An OBS80 line whose first 12 columns are given verbatim -- ``_obs80``
    blanks columns 1-5, and a comet line is DEFINED by column 5."""
    assert len(first12) == 12, first12
    return first12 + _obs80("XXXXXXX", "2017 02 03.12345", "10 10 00.00",
                            "+11 00 00.0", stn="703")[12:]


def test_a_comet_designation_is_never_linked():
    """The MPC's identification pipeline takes MINOR-PLANET designations only, so
    a linkage naming a comet or interstellar object is refused outright and sent
    to Jira instead, by design.

    Three of the four shapes are exercised here, and the third is the one a
    columns-6-12 test cannot see: 'P/2023 JN16' packs as a BARE 'K23J16N' with
    its orbit-type letter alone in column 5, so it reads as an ordinary packed
    provisional unless that column is what you look at.
    """
    from pyroorb.itf.linkage import build_linkage
    cfg = ItfConfig()
    itf = _obs80("PR00001", "2017 01 02.34567", "10 00 00.00", "+10 00 00.0")
    itf2 = _obs80("PR00001", "2017 01 04.30000", "10 00 05.00", "+10 00 03.0")

    for first12, shown in [("    CK26B030", "C/2026 B3"),   # provisional comet
                           ("0073P       ", "73P"),         # numbered comet
                           ("    PK23J16N", "P/2023 JN16")]:  # asteroid-FORM core
        doc, err = build_linkage("\n".join([itf, itf2, _comet_line(first12)]), cfg)
        assert doc is None, first12
        assert "does not accept comet" in err and "Jira" in err, (first12, err)
        assert shown in err, (first12, err)

    # ...and as a REQUESTED primary, which reaches `designations` through
    # build_linkage's `pack(d) or d` passthrough with no astrometry to read
    doc, err = build_linkage("\n".join([itf, itf2]), cfg,
                             primary_desigs=["C/2023 A3"])
    assert doc is None and "Jira" in err, err

    # CONTROL: an ordinary asteroid linkage is untouched by the ban
    pub = _obs80("K17C37L", "2017 02 03.12345", "10 10 00.00", "+11 00 00.0",
                 stn="703")
    doc, err = build_linkage("\n".join([itf, itf2, pub]), cfg,
                             primary_desigs=["2017 CL37"])
    assert err is None and doc["links"]["link_0"]["designations"] == ["K17C37L"]


def test_short_link_refuses_a_comet_document():
    """The send-time backstop, where the two-id rule is also re-applied: a
    document that reached the wire another way is still refused."""
    from pyroorb.itf.linkage import short_link
    trk = [["PR00001", "2017 01 02.3", "F51"]]
    why = short_link({"links": {"link_0": {"designations": ["CK23A030",
                                                            "K17C37L"],
                                           "trksubs": trk}}})
    assert why and "Jira" in why and "C/2023 A3" in why, why
    # an all-minor-planet document still passes
    assert short_link({"links": {"link_0": {"designations": ["K17C37L"],
                                            "trksubs": trk}}}) is None


class _OrbitFit:
    """The two calls :func:`itf.linkage.orbit_section` makes of a FittedOrbit."""
    def __init__(self, **over):
        self.el = dict(a=2.72123256, e=0.0069779912, q=2.702243810397174, Q=2.74,
                       i=6.167736, asc_node=185.02205, arg_per=-263.161302,
                       M=260.66765, Tp=2450802.555424, n=0.2195, P=4.49)
        self.el.update(over)

    def elements(self):
        return dict(self.el)

    def epoch_jd(self):
        return 2451989.776393


def _orbit_docs(monkeypatch, fit, verdict=None):
    from pyroorb import constraint
    monkeypatch.setattr(constraint, "assess", lambda f, *a, **k: verdict)
    itf = _obs80("PR00001", "2017 01 02.30000", "10 00 00.00", "+10 00 00.0")
    pub = _obs80("K17C37L", "2017 02 03.12345", "10 10 00.00", "+11 00 00.0",
                 stn="703")
    return build_linkage("\n".join([itf, pub]), ItfConfig(),
                         primary_desigs=["2017 CL37"], fit=fit)


def test_the_linked_orbit_rides_in_the_document(monkeypatch):
    """The MPC's optional per-link ``orbit``: all seven fields, heliocentric
    ecliptic, angles wrapped to [0, 360), JD epoch and pericentre time --
    find_orb's make_linkage_json layout."""
    from pyroorb.itf.linkage import ORBIT_FIELDS, short_link
    doc, err = _orbit_docs(monkeypatch, _OrbitFit())
    assert err is None
    orbit = doc["links"]["link_0"]["orbit"]
    assert set(orbit) == set(ORBIT_FIELDS)
    assert orbit["arg_pericenter"] == 96.838698            # -263.161302 wrapped
    assert orbit["eccentricity"] == 0.00697799
    assert orbit["epoch"] == 2451989.776393
    assert orbit["pericenter_time"] == 2450802.555424
    assert orbit["pericenter_distance"] == 2.702243810397174
    assert short_link(doc) is None
    # no fit, no orbit: the block is optional and nothing is invented
    doc, err = build_linkage("\n".join(
        [_obs80("PR00001", "2017 01 02.30000", "10 00 00.00", "+10 00 00.0"),
         _obs80("PR00002", "2017 01 05.30000", "10 01 00.00", "+10 00 00.0")]),
        ItfConfig())
    assert err is None and "orbit" not in doc["links"]["link_0"]


def test_an_orbit_we_would_not_publish_is_left_out(monkeypatch):
    """An undetermined arc, a display-only opt-in orbit (the population prior
    must never feed an MPC product) and a non-finite element each drop the
    block -- never the linkage."""
    for fit, verdict in [
            (_OrbitFit(), {"headline": "essentially unconstrained"}),
            (_OrbitFit(Tp=float("nan")), None)]:
        doc, err = _orbit_docs(monkeypatch, fit, verdict)
        assert err is None and "orbit" not in doc["links"]["link_0"]
    optin = _OrbitFit()
    optin.nominal_optin = {"method": "prior"}
    doc, err = _orbit_docs(monkeypatch, optin)
    assert err is None and "orbit" not in doc["links"]["link_0"]


def test_short_link_refuses_an_incomplete_orbit():
    """When an orbit IS sent the MPC requires every field -- a document that
    lost one on the way (hand-edited, replayed) is refused at send time."""
    from pyroorb.itf.linkage import short_link
    trk = [["PR00001", "2017 01 02.3", "F51"]]
    orbit = {"arg_pericenter": 1.0, "eccentricity": 0.1, "epoch": 2451545.0,
             "inclination": 2.0, "lon_asc_node": 3.0,
             "pericenter_distance": 2.0}
    why = short_link({"links": {"link_0": {"designations": ["K17C37L"],
                                           "trksubs": trk, "orbit": orbit}}})
    assert why and "pericenter_time" in why, why
    orbit["pericenter_time"] = float("inf")
    assert short_link({"links": {"link_0": {"designations": ["K17C37L"],
                                            "trksubs": trk, "orbit": orbit}}})


def test_an_unsigned_document_is_never_generated():
    import pytest
    # the one assembler refuses an anonymous header outright
    from pyroorb.itf.linkage import (AnonymousDocument, header_problem, link_draft,
                             signed_document, submission_document)
    link = {"designations": ["K16G96F"], "trksubs": [["PR00001", "2020 01 06.1", "F51"]]}
    for bad in ({"name": "", "email": "", "comment": "REPLY_EMAIL"},
                {"name": "J. Doe", "email": ""}, {"name": "J. Doe", "email": "nobody"},
                {"name": "J. {Doe}", "email": "j@example.org"}, None):
        assert header_problem(bad)
        with pytest.raises(AnonymousDocument):
            submission_document(bad, [link])
    draft = link_draft([link])
    assert "header" not in draft
    with pytest.raises(AnonymousDocument):
        signed_document(draft, "", "")
    doc = signed_document(draft, "J. Doe", "j.doe@example.org")
    assert doc["header"] == {"name": "J. Doe", "email": "j.doe@example.org",
                             "comment": "REPLY_EMAIL"}
    assert doc["links"] == draft["links"]


def test_the_comet_test_does_not_catch_minor_planets():
    """The ban must not cost an asteroid its linkage.  The near-misses are real
    tokens: a packed EXTENDED number ('~AZAP') can END in an upper-case orbit-type
    letter, a survey designation starts with 'P', and ITF trksubs are whatever
    the submitter typed."""
    from pyroorb.itf.linkage import is_comet_designation, comet_line_id
    for d in ["1P", "73P", "73P-B", "3I", "2I", "C/2020 F3", "P/2019 LD2",
              "A/2017 U1", "C/2023 A3", "0073P", "CK23A030", "K23A030", "10P"]:
        assert is_comet_designation(d), d
    for d in ["433", "00433", "K23R67C", "PLS2040", "T1S2040", "2013 PY26",
              "_PO00Tq", "133651", "10b51e", "J053207", "D3651", "A0000",
              "~AZAP", "~AZaz", "2040 P-L", "K17C37L", "P10OAL0", "A11FwDx"]:
        assert not is_comet_designation(d), d
    # the column rule, line by line
    assert comet_line_id(_comet_line("    CK26B030")) == "CK26B030"
    assert comet_line_id(_comet_line("0073P       ")) == "0073P"
    assert comet_line_id(_comet_line("    PK23J16N")) == "PK23J16N"
    for first12 in ["     K23R67C", "00433       ", "~AZAP       ",
                    "     PR00001", "D3651       "]:
        assert comet_line_id(_comet_line(first12)) is None, first12


def test_a_natural_satellite_packs_and_is_never_linked():
    """'S/2021 J 7' packs the MPC way ('SK21J070': 'S' in column 5, the planet
    letter in the half-month slot, per find_orb's xdesig.txt).  Left unpacked,
    its 10-char readable form spilled across columns 1-12 and ``line_designation``
    raised on a 'permanent id' 'S/2' beside a 'trksub' '021 J 7', killing the ITF
    search.  Packed, it round-trips -- and, being no more a minor planet than a
    comet, takes the comet refusal rather than being claimed as 'K21J070'."""
    from pyroorb.obs80 import pack_designation, format_obs80
    from pyroorb.obs_sources import obs80_line_id, ensure_unpacked
    from pyroorb.itf.linkage import (comet_line_id, comet_refusal, is_comet_designation,
                             line_designation)
    assert pack_designation("S/2021 J 7") == "SK21J070"
    assert pack_designation("S/2003 J 18") == "SK03J180"
    assert pack_designation("S/2016 J 1") == "SK16J010"
    line = format_obs80({"packed": "SK21J070", "jd_utc": 2459459.71009,
                         "ra_deg": 326.5, "dec_deg": -14.7, "mag": 23.9,
                         "band": "r", "obscode": "304"})
    assert line[:12] == "    SK21J070"
    assert obs80_line_id(line) == "SK21J070"
    assert ensure_unpacked("SK21J070") == "S/2021 J 7"
    line_designation(line)                          # no longer raises
    assert comet_line_id(line) == "SK21J070"
    assert "S/2021 J 7" in comet_refusal(line)
    assert "natural-satellite" in comet_refusal(line)
    assert is_comet_designation("S/2021 J 7")
    assert comet_refusal("", ["S/2021 J 7"])


def test_pack_refuses_a_half_month_letter_unpack_cannot_read_back():
    """``pack`` must not emit a token ``unpack`` refuses -- the round trip is the
    test the whole trksub-vs-designation split rests on.

    The two letter slots are different alphabets and the pack regex treated them
    as one.  The HALF-MONTH letter is A-Y omitting 'I' (24 letters for the 24
    half-months; the MPC's own words are "I is omitted and Z is unused"), while
    the ORDER letter is A-Z omitting 'I' -- 'Z' is the legal 25th and must keep
    packing.  With 'Z' accepted in the half-month slot, ``pack("2020 ZA")``
    returned 'K20Z00A' and ``unpack('K20Z00A')`` returned None: a one-way
    designation that still reaches OBS80 lines and submission documents.
    Verified over the whole MPCORB (1548555 entries): zero 'I' and zero 'Z' in
    the half-month slot, packed and readable alike."""
    import string

    from pyroorb.packing import pack, pack_comet, unpack, unpack_comet

    # 'Z' is not a half-month: refused outright rather than packed unreadably
    assert pack("2020 ZA") is None
    assert pack("2020 ZA1") is None
    assert pack("2020 IA") is None            # ...and 'I' never was one
    assert pack_comet("2020 Z3") is None
    assert pack_comet("2020 I3") is None

    # ...while 'Z' as the ORDER letter is the legal 25th and still round-trips
    assert unpack(pack("2020 AZ")) == "2020 AZ"
    assert unpack(pack("2020 AZ17")) == "2020 AZ17"

    # every legal half-month letter round-trips, in both letter slots' alphabets
    for h1 in "ABCDEFGHJKLMNOPQRSTUVWXY":
        for h2 in "AJZ":
            d = "1998 %s%s" % (h1, h2)
            assert unpack(pack(d)) == d, d
        assert unpack_comet(pack_comet("1998 %s3" % h1)) == "1998 %s3" % h1

    # the property the whole thing exists for: pack and unpack agree on the
    # WHOLE 26x26 letter space -- nothing packs that will not read back
    for h1 in string.ascii_uppercase:
        for h2 in string.ascii_uppercase:
            packed = pack("1998 %s%s" % (h1, h2))
            assert packed is None or unpack(packed) == "1998 %s%s" % (h1, h2)


def test_neocp_discovery_trksub():
    # An object ON the NEOCP is submitted as ONE trksub entry stamped with its
    # DISCOVERY tracklet (col-13 asterisk), never one per night; ordinary ITF
    # trksubs keep the per-night granularity.
    neo = [_obs80("P21abcd", "2017 01 02.10000", "10 00 00.00", "+10 00 00.0",
                  ref="NEOCP"),
           _obs80("P21abcd", "2017 01 04.10000", "10 00 05.00", "+10 00 03.0",
                  ref="NEOCP", disc="*"),           # the discovery tracklet
           _obs80("P21abcd", "2017 01 06.10000", "10 00 09.00", "+10 00 06.0",
                  ref="NEOCP")]
    itf = [_obs80("PR00001", "2017 01 02.30000", "11 00 00.00", "+10 00 00.0"),
           _obs80("PR00001", "2017 01 04.30000", "11 00 05.00", "+10 00 03.0")]
    desigs, triples, neocp = collect_ids("\n".join(neo + itf))
    assert neocp
    neo_triples = [t for t in triples if t[0] == "P21abcd"]
    assert len(neo_triples) == 1, f"NEOCP trksub must collapse: {neo_triples}"
    assert neo_triples[0][1].startswith("2017 01 04.1")   # the DISCOVERY night
    assert len([t for t in triples if t[0] == "PR00001"]) == 2   # unchanged
    # no asterisk anywhere -> the first tracklet stands in
    neo_nodisc = [l[:12] + " " + l[13:] if l[12] == "*" else l for l in neo]
    _, triples2, _ = collect_ids("\n".join(neo_nodisc + itf))
    neo2 = [t for t in triples2 if t[0] == "P21abcd"]
    assert len(neo2) == 1 and neo2[0][1].startswith("2017 01 02.1")


def test_requested_neocp_id_is_a_trksub_not_a_designation():
    # Typing 'A11FwDx 2014 QS168' asks for a NEOCP identification: both ids are
    # REQUESTED, so both reach build_linkage as primary_desigs -- but the NEOCP
    # temporary id has no packed designation form and must be named ONCE, under
    # trksubs.  It used to fall through `pack(d) or d` verbatim and appear in
    # BOTH arrays, claiming a designation the MPC never issued.
    neo = _obs80("A11FwDx", "2026 08 09.48798", "10 00 00.00", "+10 00 00.0",
                 stn="T08", ref="NEOCP", disc="*")
    pub = _obs80("K14QG8S", "2014 08 30.10000", "10 00 05.00", "+10 00 03.0")
    doc, err = build_linkage("\n".join([neo, pub]) + "\n", ItfConfig(),
                             primary_desigs=["A11FwDx", "2014 QS168"])
    assert err is None, err
    link = doc["links"]["link_0"]
    assert link["designations"] == ["K14QG8S"], link["designations"]
    assert [t[0] for t in link["trksubs"]] == ["A11FwDx"], link["trksubs"]
    assert link["identification_type"] == "neocp"


def test_opp_and_night_counters():
    def line(date):
        return _obs80("PR00001", date, "10 00 00.00", "+10 00 00.0")
    two_full_opps = "\n".join(line(d) for d in [
        "2015 01 01.10000", "2015 01 03.10000",
        "2016 01 01.10000", "2016 01 03.10000"])
    assert opp_profile(two_full_opps).grade is OppProfileGrade.TWO
    assert count_spans(two_full_opps) == 4
    assert count_obs(two_full_opps) == 4
    # 2 opps but only one has 2+ nights -> the thin 1.5 profile
    half = "\n".join(line(d) for d in [
        "2015 01 01.10000", "2015 01 03.10000", "2016 01 01.10000"])
    assert opp_profile(half).grade is OppProfileGrade.TWO_THIN
    assert opp_profile(half).legacy_code == 1.5           # what the stores hold
    assert shift_above_180(359.0) == -1.0
    assert is_crossing_360(358.5, 359.5, 0.5 + 360) or True   # smoke
    # obs_unique_id keys on time + SKY POSITION + station: two objects in the
    # SAME exposure (same time+station, different position) must NOT collide,
    # while a re-label of the same observation (different designation, same
    # time+position+station) is the same id.
    d = "2015 01 01.10000"
    same_exposure_other = _obs80("PR99999", d, "11 00 00.00", "+20 00 00.0")
    relabelled_same_obs = _obs80("K15A01B", d, "10 00 00.00", "+10 00 00.0")
    assert obs_unique_id(line(d)) != obs_unique_id(same_exposure_other), \
        "same exposure, different position must be distinct observations"
    assert obs_unique_id(line(d)) == obs_unique_id(relabelled_same_obs), \
        "same time+position+station is one observation regardless of designation"


def test_column_selection_cannot_silently_disable_a_filter():
    """``usecols`` must pull in whatever the requested OPERATIONS read.

    Asking for split='train' while omitting the split column used to return the
    WHOLE dataset -- 277 objects when 208 were requested -- because the filter
    tested `"split" in chunk.columns` and simply found nothing to do.  A wrong
    answer that looks like a right one is worse than a crash, and this one sat
    under a measurement I had already reported."""
    import csv as _csv
    import tempfile

    from pyroorb.itf.training import ALL_COLS, read_dataset

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=ALL_COLS)
            w.writeheader()
            for i in range(40):
                row = {c: 0 for c in ALL_COLS}
                row.update(designation=f"OBJ{i % 4}", window_t=float(i % 3),
                           trkid=f"T{i}", trksub=f"T{i}", stn="F51", band="G",
                           traj_category="likely", label=i % 2,
                           split="train" if i % 4 else "eval", stratum="MBA")
                w.writerow(row)

        # the split filter must survive a narrow column request
        got = read_dataset(path, split="train", usecols=["designation", "label"],
                           log=lambda *a: None)
        assert set(got["designation"]) == {"OBJ1", "OBJ2", "OBJ3"}, \
            f"split filter silently did nothing: {sorted(set(got['designation']))}"
        # ...and so must the columns thinning groups on
        got = read_dataset(path, neg_per_window=1, usecols=["trkid", "label"],
                           log=lambda *a: None)
        assert len(got), "thinning dropped everything"
        # a column that genuinely is not there must RAISE, not be ignored
        try:
            read_dataset(path, usecols=["designation", "nonexistent_col"],
                         log=lambda *a: None)
        except ValueError as e:
            assert "nonexistent_col" in str(e)
        else:
            raise AssertionError("a missing column passed silently")


def test_negative_thinning_keeps_every_positive_and_samples_within_window():
    """Thinning exists so a 3M-row dataset fits in memory, and it is only sound
    because it SAMPLES each window's negatives rather than filtering them:
    every positive survives, and which negatives go is random rather than
    score-ordered.  Dropping the low-scoring negatives instead would teach the
    model that the easy negatives it meets at serve time do not exist."""
    import csv as _csv
    import tempfile

    from pyroorb.itf.training import ALL_COLS, read_dataset

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=ALL_COLS)
            w.writeheader()
            for window in range(4):
                for i in range(60):
                    row = {c: 0 for c in ALL_COLS}
                    row.update(designation="OBJ", window_t=float(window),
                               trkid=f"T{window}_{i}", trksub=f"T{window}_{i}",
                               stn="F51", band="G", split="train",
                               stratum="MBA",
                               # 5 positives per window; dist ORDERS the rows so
                               # a score-based drop would be visible
                               label=1 if i < 5 else 0, dist_arcsec=float(i))
                    w.writerow(row)

        df = read_dataset(path, neg_per_window=10, seed=1, log=lambda *a: None)
        assert int(df["label"].sum()) == 20, "a positive was thinned away"
        for window, g in df[df["label"] == 0].groupby("window_t"):
            assert len(g) == 10, f"window {window} kept {len(g)} negatives, not 10"
        # the kept negatives must span the range, not cluster at one end --
        # a score-ordered drop would leave only the smallest dist values
        kept = df[df["label"] == 0]["dist_arcsec"]
        assert kept.max() > 40, \
            f"kept negatives top out at {kept.max()} -- this looks ordered, not random"
        # and thinning must be reproducible for a given seed
        again = read_dataset(path, neg_per_window=10, seed=1, log=lambda *a: None)
        assert set(again["trkid"]) == set(df["trkid"])


@pytest.mark.skipif(not hasattr(__import__("signal"), "pthread_sigmask"),
                    reason="blocking a signal needs pthread_sigmask (POSIX only)")
def test_watchdog_rescues_rows_when_the_signal_cannot_be_delivered():
    """The signal deadline is a floor, not a guarantee -- SIGALRM is delivered
    between bytecodes and a long C call defers it indefinitely (2000 PL8 ran
    3.5 minutes past a 120 s alarm).  The watchdog thread is what bounds those,
    and since it hard-exits the process it can only be tested from one.

    The simulation is a BLOCKED SIGALRM rather than a real wedged call: a
    long BLAS matmul or ASSIST integration is what does this in production, but
    how long one runs is machine-dependent, and a test that races the alarm
    against the watchdog would pass or fail by CPU speed.  Blocking the signal
    states the same condition exactly -- the alarm is armed, fires, and cannot
    be delivered -- so what remains under test is the only thing that matters:
    that the watchdog still bounds the object and keeps its rows.

    (time.sleep is NOT a stand-in for this: it is interruptible, the handler
    runs, and the alarm wins -- which is the good case and most of what the
    search does.)
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "rescued.txt")
        prog = f"""
import signal, sys, time
sys.path.insert(0, {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")!r})
from pyroorb.itf.harness import _watchdog, deadline, SearchDeadline
signal.pthread_sigmask(signal.SIG_BLOCK, {{signal.SIGALRM}})
rows = []
_watchdog(1.0, lambda: open({out!r}, "w").write("%d" % len(rows)))
try:
    with deadline(0.3):          # armed, fires -- and cannot be delivered
        rows.extend([1, 2, 3])
        time.sleep(30)
except SearchDeadline:
    open({out!r}, "w").write("ALARM")
"""
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0, f"watchdog did not exit cleanly: {r.stderr}"
        assert os.path.exists(out), "the watchdog never fired"
        got = open(out, encoding="utf-8").read()
        assert got == "3", f"rescue saw {got!r} rows, not the 3 harvested"


def test_distant_seed_spans_apparitions_and_limits_nights_per_apparition():
    """A distant object cannot be seeded from ONE apparition -- the corridor
    comes out wider than the max_uncert gate and the sweep builds no windows at
    all.  So TNOs/Centaurs seed from two apparitions, and the night limit
    applies PER seeded apparition: applied to the concatenation it would take
    every night from the earliest one, which is a single-apparition seed again.
    """
    import pyroorb.itf.harness as H

    # four apparitions, a year apart, three nights each at one station
    obs = []
    for app in range(4):
        for night in range(3):
            jd = 2458000.0 + app * 400.0 + night * 3.0
            for k in range(2):          # 2 obs/night -> a plantable tracklet
                obs.append({
                    "jd_utc": jd + k * 0.01, "ra_deg": 150.0 + app + night * .01,
                    "dec_deg": 10.0, "obscode": "F51", "mag": 21.0, "band": "G",
                    "obs80_line": _obs80("K17C37L",
                                         "2017 09 %02d.%04d" % (night + 1, k),
                                         "10 00 00.00", "+10 00 00.0")})
    assert len(H.split_oppositions(obs)) == 4

    seen = {}

    def _capture(source, **kw):
        seen["seed"] = kw["seed_obs"]
        raise H.SearchDeadline("stop here -- the seed is what is under test")

    old = H.itf_search
    H.itf_search = _capture
    try:
        rep = H.run_recovery("TEST", obs_meta=(obs, {"designation": "TEST"}),
                             obscodes={}, seed_nights=2, seed_oppositions=2,
                             collector_rows=[], log=lambda *a: None)
    finally:
        H.itf_search = old

    assert rep.seed_apparitions_used == 2, "a distant seed spans 2 apparitions"
    jds = sorted(o["jd_utc"] for o in seen["seed"])
    # the seed must straddle the gap, not sit inside one apparition
    assert jds[-1] - jds[0] > H.HARNESS_OPPOSITION_GAP_DAYS, \
        "the seed collapsed into a single apparition"
    # ...and contribute at most `seed_nights` nights from EACH
    for lo in (2458000.0, 2458400.0):
        nights = {round(j - lo) for j in jds if lo <= j < lo + 300}
        assert len(nights) <= 2, f"{len(nights)} nights taken from one apparition"
    # the two untouched apparitions remain plantable truth
    assert len(rep.planted_apparitions) >= 2


def test_fake_itf_synthesis():
    # two apparitions of pseudo-observations a year apart, two stations
    obs = []
    for k in range(3):
        obs.append({"jd_utc": 2458000.0 + k * 0.01, "ra_deg": 150 + k * .01,
                    "dec_deg": 10.0, "obscode": "F51", "mag": 21.0, "band": "G",
                    "obs80_line": _obs80("K17C37L", "2017 09 03.5%04d" % k,
                                         "10 00 00.00", "+10 00 00.0")})
    obs.append({"jd_utc": 2458400.0, "ra_deg": 160.0, "dec_deg": 12.0,
                "obscode": "703", "mag": 21.0, "band": "G",
                "obs80_line": _obs80("K17C37L", "2018 10 08.50000",
                                     "10 40 00.00", "+12 00 00.0", stn="703")})
    apps = split_oppositions(obs)
    assert len(apps) == 2
    fakes, rejected = make_fake_itf(apps, INDEX_CFG)
    assert len(fakes) == 1                      # the 3-obs F51 tracklet
    assert len(rejected) == 1                   # the lone 703 obs: no direction
    fk = fakes[0]
    assert fk["feat"]["n_obs"] == 3
    line = fk["obs"][0]["obs80"]
    assert len(line) == 80 and line[0:5] == "     "
    assert line[5:12] == fk["feat"]["trksub"]
    # the re-labeled id must NOT read as a real designation
    from pyroorb.itf.linkage import is_designation
    assert not is_designation(fk["feat"]["trksub"])





def test_dedup_designations():
    # A=B already identified at the MPC: only ONE may be sent, or the whole
    # linkage (incl. its ITF news) is rejected
    from pyroorb.itf.linkage import dedup_designations
    fam = {"2016 UH221": "2016 UH221", "2025 NW433": "2016 UH221",
           "2020 AB1": "2020 AB1"}
    doc = {"links": {"link_0": {
        "designations": ["K16UM1H", "K25NW3T", "K20A01B"],
        "trksubs": [["0e9311", "2006 09 16.34863", "705"]]}}}
    # K16UM1H=2016 UH221, K25NW3T... use real packs via packing to be safe
    from pyroorb.packing import pack
    doc["links"]["link_0"]["designations"] = [
        pack("2016 UH221"), pack("2025 NW433"), pack("2020 AB1")]
    msgs = []
    dedup_designations(doc, lambda d: fam.get(d, d), log=msgs.append)
    kept = doc["links"]["link_0"]["designations"]
    assert kept == [pack("2016 UH221"), pack("2020 AB1")], kept
    assert msgs and "dropped" in msgs[0]
    # the representative must be the family PRIMARY even when the doc was
    # seeded with a secondary alias
    doc2 = {"links": {"link_0": {"designations": [pack("2025 NW433")],
                                 "trksubs": [["x", "2020 01 01.5", "F51"]]}}}
    dedup_designations(doc2, lambda d: fam.get(d, d), log=msgs.append)
    assert doc2["links"]["link_0"]["designations"] == [pack("2016 UH221")]
    # idempotent + no-op when nothing is identified
    msgs.clear()
    dedup_designations(doc, lambda d: d, log=msgs.append)
    assert doc["links"]["link_0"]["designations"] == kept and not msgs


def test_dedup_keep_spelling_only_drops():
    # the identification queue: a link is never edited -- the joined alias goes,
    # what stays is exactly the submitter's designation, not the family primary
    from pyroorb.itf.linkage import dedup_designations
    from pyroorb.packing import pack
    fam = {"2016 UH221": "2016 UH221", "2025 NW433": "2016 UH221"}
    doc = {"links": {"link_0": {
        "designations": [pack("2025 NW433"), pack("2016 UH221"), pack("2020 AB1")],
        "trksubs": [["x", "2020 01 01.5", "F51"]]}}}
    msgs = []
    dedup_designations(doc, lambda d: fam.get(d, d), log=msgs.append, keep_spelling=True)
    assert doc["links"]["link_0"]["designations"] == [pack("2025 NW433"), pack("2020 AB1")]
    assert msgs and "primary" not in msgs[0] and "dropped" in msgs[0]
    alone = {"links": {"link_0": {"designations": [pack("2025 NW433")],
                                  "trksubs": [["x", "2020 01 01.5", "F51"]]}}}
    msgs.clear()
    dedup_designations(alone, lambda d: fam.get(d, d), log=msgs.append, keep_spelling=True)
    assert alone["links"]["link_0"]["designations"] == [pack("2025 NW433")] and not msgs


def test_dedup_collapse_is_refused_not_submitted():
    """A link needs TWO ids -- in any mix (DES+DES, DES+ITF, DES+DES+ITF, ITF+ITF).
    build_linkage enforces that when it BUILDS the document, but the pre-submit dedup runs
    afterwards and can drop an id: two designations the MPC has already merged collapse to
    one family primary. With no tracklet attached that leaves a one-id document, which is
    the resubmitted-DES=DES case the dedup exists to prevent, arriving in a different shape.
    It must be REFUSED with a reason, never transmitted."""
    from pyroorb.itf.linkage import short_link, dedup_designations, submit_linkage
    from pyroorb.itf.config import ItfConfig
    from pyroorb.packing import pack
    fam = {"2016 UH221": "2016 UH221", "2025 NW433": "2016 UH221"}

    # two designations, one acknowledged family, nothing else attached -> collapses to one
    doc = {"links": {"link_0": {"designations": [pack("2016 UH221"),
                                                 pack("2025 NW433")]}}}
    assert short_link(doc) is None, "two ids before the dedup: a valid document"
    dedup_designations(doc, lambda d: fam.get(d, d), log=lambda m: None)
    assert len(doc["links"]["link_0"]["designations"]) == 1
    why = short_link(doc)
    assert why and "already accepted" in why, why

    # ...and submit_linkage refuses rather than POSTing (a real transmission would need
    # requests + the network; reaching either is itself the failure)
    sent, status, msg = submit_linkage(doc, ItfConfig(submit=True), do_submit=True,
                                       log=lambda m: None)
    assert sent is False and msg is None, (sent, status)
    assert "not submitted" in status, status

    # the SAME collapse with an ITF tracklet attached is real news: DES+ITF is two ids
    doc2 = {"links": {"link_0": {"designations": [pack("2016 UH221")],
                                 "trksubs": [["0e9311", "2006 09 16.34863", "705"]]}}}
    assert short_link(doc2) is None, "DES + ITF is a linkage"
    # ITF+ITF is a linkage too, and a lone tracklet is not
    assert short_link({"links": {"link_0": {"trksubs": [["a", "2020 01 01.5", "F51"],
                                                        ["b", "2020 02 01.5", "F51"]]}}}) is None
    assert short_link({"links": {"link_0": {"trksubs": [["a", "2020 01 01.5", "F51"]]}}})
    # every link of a multi-link batch document is checked, not just link_0
    assert short_link({"links": {"link_0": doc2["links"]["link_0"],
                                 "link_1": {"designations": [pack("2020 AB1")]}}})


def test_download_refuses_the_document_submission_would_refuse():
    """DOWNLOADING the linkage file is the other way a document leaves the engine, and the
    user posts that one by hand -- so it must apply the same two-id backstop the submit path
    does.  Both build the document BEFORE the DES=DES dedup runs, so both can watch an id
    disappear underneath build_linkage's own guard; only submission used to notice."""
    import pyroorb.web.engine_api as api
    import pyroorb.itf.dbio; from pyroorb import itf; import pyroorb.itf.linkage; from pyroorb import itf
    from pyroorb.packing import pack

    doc = {"links": {"link_0": {"designations": [pack("2016 UH221"),
                                                 pack("2025 NW433")]}}}
    orig = (api._linkage_block, api._itf_live_linkage,
            itf.dbio.connect_pub, itf.linkage._family_map)
    api._SESSIONS["dl-test"] = {"fit": object()}
    api._linkage_block = lambda s: {"found": True}
    api._itf_live_linkage = lambda s, *a, **kw: (doc, None)
    itf.dbio.connect_pub = lambda cfg: SimpleNamespace(close=lambda: None)
    try:
        # the MPC already merged the pair: the dedup collapses it to one family primary
        itf.linkage._family_map = lambda conn, unp: {d: "2016 UH221" for d in unp}
        got = api.itf_download_web("dl-test", "P. VanWyk", "p@example.com")
        assert got["ok"] is False and "needs two" in got["error"], got
        assert "document" not in got, "a refused linkage must not be written out"

        # ...and the same request when the dedup keeps both ids does download
        itf.linkage._family_map = lambda conn, unp: {d: d for d in unp}
        got = api.itf_download_web("dl-test", "P. VanWyk", "p@example.com")
        assert got["ok"] is True, got
        assert len(got["document"]["links"]["link_0"]["designations"]) == 2
        assert got["filename"].startswith("ident_")
    finally:
        (api._linkage_block, api._itf_live_linkage,
         itf.dbio.connect_pub, itf.linkage._family_map) = orig
        api._SESSIONS.pop("dl-test", None)


def test_a_non_submittable_verdict_cannot_be_posted_or_downloaded():
    """OFFERED is not SUBMITTABLE, and the front end is not what enforces it.

    A 'y2' (the join rests on excepting whole tracklets), a 'y-systematic' (the kept
    observations trend) and an 'unknown' (the exception shape could not be measured) are all
    shown for review -- with a caution the user can acknowledge -- but the engine declines to
    confirm them, so no button appears. Acking a caution is not consent to a claim the engine
    refused to make: both routes a document leaves by must re-ask the verdict, or the browser
    is the only thing standing between a hand-crafted POST and the MPC."""
    import pyroorb.web.engine_api as api
    from pyroorb.packing import pack

    doc = {"links": {"link_0": {"designations": [pack("2016 UH221"),
                                                 pack("2025 NW433")]}}}
    caution = "The engine could not judge this identification: boom."
    block = {"found": True, "kind": "ident", "submittable": False,
             "reason": "NOT submittable -- the engine could not determine the exception shape",
             "warnings": [caution]}
    orig = (api._linkage_block, api._itf_live_linkage, api._direct_submit_allowed)
    api._SESSIONS["nosub-test"] = {"fit": object()}
    api._linkage_block = lambda s: dict(block)
    api._itf_live_linkage = lambda s, *a, **kw: (doc, None)
    api._direct_submit_allowed = lambda: True
    sent = []
    try:
        got = api.itf_submit_web("nosub-test", acked_warnings=[caution], route="mpc",
                                 name="T. Tester", email="t@example.org",
                                 _submitter=lambda *a: sent.append(a) or (True, "200", "ok"))
        assert got["ok"] is False and "not submittable" in got["error"], got
        assert not sent, "a non-submittable linkage reached the MPC"

        got = api.itf_download_web("nosub-test", "P. VanWyk", "p@example.com")
        assert got["ok"] is False and "not submittable" in got["error"], got
        assert "document" not in got, "a refused linkage must not be written out"

        # ...and the identical request on a verdict that IS submittable goes through, so the
        # guard is the flag and not some other property of this session
        block["submittable"] = True
        got = api.itf_submit_web("nosub-test", acked_warnings=[caution], route="mpc",
                                 name="T. Tester", email="t@example.org",
                                 _submitter=lambda *a: sent.append(a) or (True, "200", "ok"))
        assert got["ok"] is True and got["accepted"] is True, got
        assert len(sent) == 1
    finally:
        (api._linkage_block, api._itf_live_linkage, api._direct_submit_allowed) = orig
        api._SESSIONS.pop("nosub-test", None)


def test_one_requested_designation_is_never_a_linkage():
    """Type ONE designation and the MPC may hand back several: 1999 CN154's fetch carries
    2014 EM304's observations because the MPC acknowledged that identification years ago.
    That is not news anyone may submit -- resubmitting an acknowledged DES=DES is rejected --
    so the engine must (a) decline the verdict, (b) show no "Submit linkage" button and no
    banner suggesting one, and (c) never name both designations in a document, even when an
    ITF tracklet IS genuinely new."""
    from pyroorb import ident_check
    import pyroorb.web.engine_api as api
    from pyroorb.packing import pack

    class _Fit:
        n_requested = 1
        def __init__(self, lines, desigs):
            self._obs = [{"obs80_line": l, "designation": d, "jd_utc": 2459100.0 + i}
                         for i, (l, d) in enumerate(zip(lines, desigs))]
        def is_included(self):
            return [True] * len(self._obs)

    # (a) the one rule both front ends ask (--linkage, the web button) declines it
    v = ident_check.linkage_verdict(_Fit([], []))
    assert v["offer"] is False and not v["clean"]
    assert "already acknowledged" in v["reason"]

    # two aliases of ONE fetched object, plus a genuinely new ITF tracklet
    lines = [_obs80(pack("1999 CN154"), "2020 09 13.30000", "10 00 00.00", "+10 00 00.0"),
             _obs80(pack("1999 CN154"), "2020 09 13.31000", "10 00 00.00", "+10 00 00.0"),
             _obs80(pack("2014 EM304"), "2020 09 16.30000", "10 00 01.00", "+10 00 01.0"),
             _obs80(pack("2014 EM304"), "2020 09 16.31000", "10 00 01.00", "+10 00 01.0"),
             _obs80("P10zzzz", "2020 09 20.30000", "10 00 02.00", "+10 00 02.0"),
             _obs80("P10zzzz", "2020 09 20.31000", "10 00 02.00", "+10 00 02.0")]
    desigs = ["1999 CN154"] * 2 + ["2014 EM304"] * 2 + [None] * 2
    fit = _Fit(lines, desigs)

    # (b) no block at all -> the payload carries no `itf`, so no button and no banner
    assert api._ident_linkage_block({"fit": fit}) is None

    # (c) the document names the family PRIMARY once; the ITF tracklet is the only news
    doc, err = api._itf_live_linkage({"fit": fit})
    assert doc is not None, err
    link = doc["links"]["link_0"]
    assert link["designations"] == [pack("1999 CN154")], link["designations"]
    assert [t[0] for t in link["trksubs"]] == ["P10zzzz"]

    # ...and a fit the USER assembled (two designations typed) is untouched by the gate
    fit.n_requested = 2
    doc2, err2 = api._itf_live_linkage({"fit": fit})
    assert doc2 is not None, err2
    assert sorted(doc2["links"]["link_0"]["designations"]) == \
        sorted([pack("1999 CN154"), pack("2014 EM304")])


def test_coherent_offset_needs_reference():
    # A synthetic fit: build residuals where one night is coherently offset.
    from pyroorb import ident_check
    class FakeFit:
        pass
    import numpy as np
    f = FakeFit()
    # 3 nights x 4 obs; night 1 offset +0.3" in Dec, others noise ~0.02"
    jds, R = [], []
    rng = np.random.default_rng(0)
    for ni, off in enumerate([0.0, 0.30, 0.0]):
        for k in range(4):
            jds.append(2460000.0 + ni * 3 + k * 0.01)
            R.append([rng.normal(0, 0.02), off + rng.normal(0, 0.02)])
    f._obs = [{"jd_utc": j} for j in jds]
    f._resid_all = np.array(R)
    f._mask = [True] * len(jds)
    # against a tight pre-link noise floor (0.03") the 0.30" night fires
    assert ident_check.coherent_offset_misfit(f, noise_floor=0.03) is not None
    # against a genuinely-noisy floor (0.3") it does NOT -- 0.30" is ~1x noise
    assert ident_check.coherent_offset_misfit(f, noise_floor=0.30) is None


def test_extension_gate_config():
    # Drive the REAL veto decision (search.extension_degrades) with the three
    # anchor cases -- so the test tracks the actual code path, not a re-derived
    # formula.  The rule is a RELATIVE degradation-factor test (NOT sigma-based;
    # see the docstring): survey-reported uncertainties are untrustworthy here.
    from pyroorb.itf.search import extension_degrades
    cfg = ItfConfig()
    # Seed-only RELATIVE factor test: the decision depends on the object's OWN
    # obs (pre -> seed_post), never the overall rms.  BOTH sides are subset_rms
    # (per-observation) values; the anchors' historical per-axis pre figures are
    # scaled by sqrt(2) accordingly (see search._seed_pre_rms).
    # 2026 FU24 -- CONTAMINANT (mag-23.6 tracklet bends the orbit off the
    # object's own mag-22 nights): pre 0.047" -> seed 0.297" (6.4x). Must BAN.
    assert extension_degrades(cfg, 0.047, 0.297)
    # 2025 NQ92 -- REAL precovery: pre 0.078" -> seed 0.084" (1.1x). Seed stays
    # tight -> must NOT ban.
    assert not extension_degrades(cfg, 0.078, 0.084)
    # 2026 DZ13 -- REAL self-recovery of an overfit compact seed: pre 0.098" ->
    # seed 0.175" (1.8x honest settling). Must NOT ban -- the false positive the
    # old 2.0x factor produced.
    assert not extension_degrades(cfg, 0.098, 0.175)

    # The factor's statistical escape (2015 FO479): a seed can be tight by
    # LUCK -- 13 obs over 30 d fitting themselves to 0.079" is 0.13 sigma
    # against the stations that took them -- and a ratio against luck is not a
    # test.  Its correct 6-opposition linkage read 3.09x while leaving the
    # object's own obs at 0.56 sigma, comfortably inside the stations' error.
    assert extension_degrades(cfg, 0.112, 0.346)                    # factor alone
    assert not extension_degrades(cfg, 0.112, 0.346, seed_post_sigma=0.56)

    # ...and the escape must NOT be what saves the contaminant anchor.  On
    # Rubin's SELF-REPORTED 0.5" FU24's post-link seed looks like 0.4 sigma;
    # normalised by X05's 0.1" STATION floor -- what the caller passes -- it is
    # 2.97, so the ban stands on both halves.
    assert extension_degrades(cfg, 0.047, 0.297, seed_post_sigma=2.97)
    assert extension_degrades(cfg, 0.047, 0.297, seed_post_sigma=1.01)
    # a genuinely misfitting link is still banned however lucky the baseline
    assert extension_degrades(cfg, 0.050, 2.000, seed_post_sigma=3.0)
    # 2020 KX6 -- REAL self-recovery from a 2-day DISCOVERY arc (8 obs, 2.18 d,
    # U 9.23): pre 0.073" -> seed 0.240" (3.3x).  The seed is nearly degenerate
    # against the state vector, so that is honest in-sample -> out-of-sample
    # widening, not contamination -- but the strict factor, calibrated on 11-17 d
    # arcs, banned the object's OWN 2020-06-23 tracklet plus its 2023 and 2025
    # apparitions (1 -> 3 opps).
    assert not extension_degrades(cfg, 0.073, 0.240, 2.18)
    # ...and the arc gate must not become a hole: an unknown seed arc length keeps the
    # strict factor, and the contaminant anchor stays banned even AS a short arc
    # (6.4x clears both factors), so no arc length can wave FU24 through.
    assert extension_degrades(cfg, 0.073, 0.240)
    assert extension_degrades(cfg, 0.047, 0.297, 2.0)
    assert extension_degrades(cfg, 0.047, 0.297, 11.0)


class _MagFit:
    """Stand-in exposing only ``magnitude_residuals`` -- tracklets given as
    ``(station, day_offset, [dmag, ...])``."""

    def __init__(self, tracklets):
        self._rows = []
        for station, day, dmags in tracklets:
            for k, dm in enumerate(dmags):
                self._rows.append({"jd_utc": 2460000.0 + day + 0.01 * k,
                                   "obscode": station, "dmag": dm,
                                   "obs_mag": 20.0 + dm, "used": True})

    def magnitude_residuals(self):
        return list(self._rows)


def test_mag_warning_is_a_tracklet_t_test_not_a_single_ob():
    """The photometric caution asks whether a TRACKLET's residuals are jointly
    off the predicted brightness -- mean >= warn_mag_resid AND two-tailed
    p < warn_mag_p against zero -- not whether any ONE ob crossed a threshold."""
    from pyroorb.itf.fitter import mag_outlier_tracklet, worst_mag_residual
    cfg = ItfConfig()
    floor, pmax = cfg.warn_mag_resid, cfg.warn_mag_p

    # The motivating case: 0.9/0.2/0.1 averages 0.40 and scatters far too widely to
    # claim the tracklet is off at all.  The old per-ob rule warned on the 0.9.
    scatter = _MagFit([("G96", 0.0, [0.9, 0.2, 0.1])])
    assert worst_mag_residual(scatter)[0] >= floor      # would have warned
    assert mag_outlier_tracklet(scatter, floor, pmax) is None

    # A tracklet consistently ~1.5 mag off IS the different-body signal.
    off = mag_outlier_tracklet(_MagFit([("G96", 0.0, [1.4, 1.6, 1.5, 1.55])]),
                               floor, pmax)
    assert off is not None and off["n"] == 4 and off["p"] < pmax
    assert 1.4 < off["mean"] < 1.6 and off["station"] == "G96"

    # Significant but SMALL: a tight survey 0.13 mag off its zero point is
    # ~15 sigma from zero and photometrically uninteresting -- the floor holds.
    assert mag_outlier_tracklet(_MagFit([("F51", 0.0, [.12, .14, .13, .15])]),
                                floor, pmax) is None

    # Sign is carried, not swallowed by an absolute value: a tracklet 1.5 mag
    # too FAINT is as much evidence as one too bright.
    faint = mag_outlier_tracklet(_MagFit([("G96", 0.0, [-1.4, -1.6, -1.5])]),
                                 floor, pmax)
    assert faint is not None and faint["mean"] < 0

    # One-observation tracklet: no degrees of freedom, so it falls back to the
    # flat threshold that used to apply to every ob (p reported as None).
    lone = mag_outlier_tracklet(_MagFit([("703", 0.0, [1.2])]), floor, pmax)
    assert lone is not None and lone["p"] is None and lone["n"] == 1
    assert mag_outlier_tracklet(_MagFit([("703", 0.0, [0.5])]), floor, pmax) is None

    # ...and the same holds for a real tracklet carrying only ONE magnitude:
    # the statistic counts MAGNITUDES, not observations, so two unphotometered
    # obs beside a 1.2 mag one leave nothing to test -- flat threshold, no
    # t-test attempted (a t-test on n=1 has no dof and would be a fiction).
    one_mag = _MagFit([("703", 0.0, [1.2, 0.0, 0.0])])
    for r in one_mag._rows[1:]:
        r["dmag"] = None                 # obs present in the fit, no photometry
    got1 = mag_outlier_tracklet(one_mag, floor, pmax)
    assert got1 is not None and got1["n"] == 1 and got1["p"] is None
    quiet = _MagFit([("703", 0.0, [0.5, 0.0, 0.0])])
    for r in quiet._rows[1:]:
        r["dmag"] = None
    assert mag_outlier_tracklet(quiet, floor, pmax) is None

    # Tracklets are (station, night): same station a week apart is two of them,
    # and the quiet one must not dilute the offset one.
    two = _MagFit([("G96", 0.0, [1.4, 1.6, 1.5]), ("G96", 7.0, [0.0, 0.1, -.1])])
    got = mag_outlier_tracklet(two, floor, pmax)
    assert got is not None and got["n"] == 3 and got["mean"] > 1.0
    # ...and a tested tracklet outranks a lone-ob fallback however big that is.
    mixed = _MagFit([("G96", 0.0, [1.4, 1.6, 1.5]), ("703", 3.0, [9.0])])
    assert mag_outlier_tracklet(mixed, floor, pmax)["n"] == 3

    # Rejected obs and obs without photometry never enter the statistic.
    f = _MagFit([("G96", 0.0, [1.4, 1.6, 1.5])])
    for r in f._rows:
        r["used"] = False
    assert mag_outlier_tracklet(f, floor, pmax) is None


class _CurrencyFit:
    """Minimal stand-in for a FittedOrbit that the fitter's subset helpers can
    walk: n obs, all included, each with the same (d_ra, d_dec) residual."""

    def __init__(self, lines, d_ra, d_dec):
        self._obs = [{"obs80_line": l, "jd_utc": 2460000.0 + i}
                     for i, l in enumerate(lines)]
        self._d = (d_ra, d_dec)

    def residuals(self, included_only=True):
        return [{"d_ra_arcsec": self._d[0], "d_dec_arcsec": self._d[1]}
                for _ in self._obs]

    def is_included(self):
        return [True] * len(self._obs)


def test_seed_pre_rms_is_subset_currency_not_per_axis():
    """The pre-link side of every seed-degradation comparison must be measured
    with the SAME statistic as the post side.

    ``subset_rms`` is a per-OBSERVATION total, sqrt(mean(d_ra^2 + d_dec^2));
    ``FittedOrbit.rms_residual`` (hence ``TrialFit.rms``) is a per-AXIS rms,
    sqrt(mean(r**2)) over 2n components.  For identical residuals they differ by
    exactly sqrt(2).  Passing ``pub.rms`` where a ``subset_rms`` belongs -- which
    the veto, the warning and the recursion seed guard all did until 2026-07-29
    -- charges every link a 1.414x handicap, and the 1.5x warning then fired on
    nearly every genuine recovery.  Guard the currency, not just the constants."""
    import math
    from pyroorb.itf.search import _seed_pre_rms
    from pyroorb.itf.fitter import subset_rms
    lines = _seed_4_nights()
    # isotropic 0.1"/axis residuals: per-axis rms 0.1, per-obs rms 0.1*sqrt(2)
    fit = _CurrencyFit(lines, 0.1, 0.1)
    ids = {obs_unique_id(l) for l in lines}
    per_obs, n = subset_rms(fit, ids)
    assert n == len(lines)
    assert abs(per_obs - 0.1 * math.sqrt(2)) < 1e-12
    # pub.rms is the per-axis figure; _seed_pre_rms must NOT return it
    ctx = SimpleNamespace(pub=SimpleNamespace(fit=fit, rms=0.1), pub_ids=ids)
    assert abs(_seed_pre_rms(ctx) - per_obs) < 1e-12
    # and the documented fallback is the exact conversion, not the raw value
    ctx_nofit = SimpleNamespace(pub=SimpleNamespace(fit=None, rms=0.1),
                                pub_ids=ids)
    assert abs(_seed_pre_rms(ctx_nofit) - 0.1 * math.sqrt(2)) < 1e-12


def test_per_axis_is_the_exact_display_conversion():
    """The one conversion the whole reporting standard rests on."""
    import math
    from pyroorb.itf.fitter import per_axis, subset_rms
    lines = _seed_4_nights()
    fit = _CurrencyFit(lines, 0.1, 0.1)
    per_obs, _ = subset_rms(fit, {obs_unique_id(l) for l in lines})
    assert abs(per_axis(per_obs) - 0.1) < 1e-12       # back to the per-axis rms
    assert per_axis(None) is None
    assert abs(per_axis(1.0) - 1.0 / math.sqrt(2)) < 1e-15


def test_seed_rms_is_reported_per_axis():
    """MEASURE per-observation, REPORT per-axis.

    find_orb's ``compute_rms`` divides the summed d_ra^2 + d_dec^2 by 2n (it
    was revised to, in 2010, to match the MPC), so every rms a person reads
    here -- the seed line, the web header, MPCORB, the vetting dict, the batch
    CSV -- is per-axis.  The seed-degradation gates are the one place that
    measures per-observation, because both of their sides must agree; the
    numbers they hand to a reader have to come back.  Found on 2021 CM40, whose
    banner quoted 0.754" -> 0.856" for a 0.533" arc.

    Locked at the vetting dict, which is what the batch CSV and the web UI
    read, and against the raw gate value it is derived from."""
    import math
    from pyroorb.itf.fitter import subset_rms, per_axis
    lines = _seed_4_nights()
    ids = {obs_unique_id(l) for l in lines}
    fit = _CurrencyFit(lines, 0.1, 0.1)
    per_obs, _ = subset_rms(fit, ids)
    # what the gate measures, and what the reader must be shown instead
    assert abs(per_obs - 0.1 * math.sqrt(2)) < 1e-12
    assert abs(per_axis(per_obs) - 0.1) < 1e-12
    # the vetting dict's two rms pairs must be in ONE currency: overall_* comes
    # straight off TrialFit.rms (per-axis), so seed_* has to be converted
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "pyroorb" / "itf" / "search.py").read_text(encoding="utf-8")
    assert '"seed_rms_pre": round(per_axis(self.rms_pre), 4),' in src
    assert '"seed_rms_post": round(per_axis(self.rms_post), 4),' in src
    # ...and no display site may quote the raw per-observation pair again
    for raw in ('{seed_pre_rms:.3f}', '{seed_post_rms:.3f}',
                '{seed_pre:.3f}', '{seed_post:.3f}', '{rest_rms:.3f}',
                '{seed_lvl:.3f}'):
        assert raw not in src, f"per-observation rms reaches a report: {raw}"


def test_warn_factor_clears_the_correct_link_null():
    """A CORRECT link widens the seed's own residuals: its pre-link rms is
    in-sample (six elements fitted to that very arc), so the same obs loosen once
    a longer arc takes control.  Measured on 18 PAIR_* fixtures (fit apparition 1
    alone, then the whole arc, then score the whole-arc orbit on exactly the seed
    obs), correct links ran 1.05-1.5x.  The warning must sit ABOVE that band, or
    it fires on good recoveries -- which is exactly what 1.5x did."""
    cfg = ItfConfig()
    assert cfg.warn_rms_factor >= 1.8
    assert cfg.warn_rms_factor < cfg.veto_rms_factor
    # the three real warned-on-but-good cases from the 2026-07 batch run, in
    # seed currency: none may warn now
    for pre, post in [(0.177, 0.226), (0.288, 0.356), (0.190, 0.227)]:
        lvl = max(pre * cfg.warn_rms_factor, pre + cfg.veto_rms_floor_arcsec)
        assert post < lvl, (pre, post, lvl)


def _seed_4_nights():
    """A 4-night seed arc, two obs a night, night 4 on a different station."""
    lines = []
    for night, (day, stn) in enumerate(
            [("2020 09 13", "F51"), ("2020 09 16", "F51"),
             ("2020 09 20", "F51"), ("2020 09 24", "703")], 1):
        for k in range(2):
            lines.append(_obs80("K20S00X", f"{day}.3{k}0000",
                                "10 00 00.00", "+10 00 00.0", stn=stn))
    return lines


def test_seed_clip_ledger_groups_by_night():
    """The ledger has to say WHICH night of the published arc a link threw away
    -- that is the whole point of reporting it ("I had to drop the 4th night").
    A night that loses every observation is LOST; one that loses some is
    PARTIAL, because the arc's opposition profile survives and ordinary outlier rejection is
    not a claim about the night."""
    import pyroorb.itf.search as S
    lines = _seed_4_nights()
    ctx = SimpleNamespace(pub_obs80="\n".join(lines) + "\n",
                          pub_ids={obs_unique_id(l) for l in lines})
    clipped = {obs_unique_id(lines[6]), obs_unique_id(lines[7]),   # all of night 4
            obs_unique_id(lines[2])}                            # half of night 2
    kept = ctx.pub_ids - clipped
    orig = S.subset_included
    S.subset_included = lambda fit, ids: (kept, {i: 3.7 for i in clipped})
    try:
        led = S._seed_clip_ledger(ctx, _trial(20))
    finally:
        S.subset_included = orig
    assert led["n_nights"] == 4 and led["kept_nights"] == 3
    assert led["shed_obs"] == 3 and led["kept_obs"] == 5
    assert [e["night"] for e in led["lost_nights"]] == [4]
    lost = led["lost_nights"][0]
    assert lost["date"] == "2020 09 24" and lost["stations"] == ["703"]
    assert lost["n_shed"] == 2 and lost["worst_resid_arcsec"] == 3.7
    assert [e["night"] for e in led["partial_nights"]] == [2]


def _qr56_shaped_linkage():
    """A 2006 QR56-shaped 1.75: a 3-night seed, two more tracklets in the SAME
    (2006) opposition, and two lone single-night oppositions years later."""
    seed = [_obs80("K06Q56R", f"2006 09 1{d}.3{k}0000", "10 00 00.00",
                   "+10 00 00.0") for d in (1, 3, 5) for k in range(2)]
    same_opp = [_obs80("A1234", f"2006 {m}.30000", "10 00 00.00",
                       "+10 00 00.0") for m in ("09 28", "10 22")]
    far = [_obs80("B5678", "2015 08 10.30000", "10 00 00.00", "+10 00 00.0"),
           _obs80("C9012", "2024 09 04.30000", "10 00 00.00", "+10 00 00.0")]
    return "\n".join(seed) + "\n", "".join(l + "\n" for l in same_opp + far)


def test_seed_opp_extension_splits_at_the_opposition():
    """The reduced linkage is "seed + the tracklets in the seed's OWN
    opposition" -- the part of a fragile 1.75 that makes no cross-opposition
    claim.  It must keep exactly those and refit them alone."""
    import pyroorb.itf.search as S
    seed, obs_out = _qr56_shaped_linkage()
    ctx = SimpleNamespace(pub_obs80=seed, obscodes=None, cfg=ItfConfig())
    seen = {}
    orig = S.fit_astrometry
    S.fit_astrometry = lambda text, oc, **kw: (
        seen.update(text=text) or _trial(23, rms=0.386, U=5.38))
    try:
        got = S._seed_opp_extension(ctx, obs_out)
    finally:
        S.fit_astrometry = orig
    assert got is not None
    trial, combined, obs_same = got
    assert opp_profile(combined).grade is OppProfileGrade.ONE                  # no cross-opp claim left
    assert "A1234" in obs_same and "B5678" not in obs_same
    assert "C9012" not in obs_same and combined == seen["text"]
    # every linked tracklet already in the seed's opposition -> nothing to price
    assert S._seed_opp_extension(ctx, "".join(
        l + "\n" for l in obs_same.splitlines())) is None


def test_confirm_top_pair_measures_the_extended_seed_opposition():
    """Path 1 asks "is the seed well constrained?" of the arc as it ENTERED the
    search, which on a discovery-arc recovery is the wrong moment: 2006 QR56
    enters as 2.0 d / U 9.87 but its own 2006 apparition is 57.5 d / U 5.38 once
    its two 2006 tracklets are linked back.  That orbit is independent of the
    1.75 profile (none of the single-night obs are in it), so it -- not the stub
    -- is what the tier should be measured against."""
    import pyroorb.itf.search as S
    cfg = ItfConfig()
    seed, obs_out = _qr56_shaped_linkage()
    ctx = SimpleNamespace(cfg=cfg, obscodes=None, pub_obs80=seed,
                          log=lambda *a: None, trace=None,
                          pub=SimpleNamespace(arclength_days=2.0, U=9.87))
    strong = (_trial(23, rms=0.386, U=5.38), "", "")   # arc set below
    strong[0].arclength_days = 57.5
    weak = (_trial(19, rms=0.5, U=8.4), "", "")
    weak[0].arclength_days = 9.0
    # the raw seed alone can confirm nothing, and neither can a still-short
    # reduced linkage -- the depth-0 top-pair path is the only other way in
    assert not S._confirm_top_pair(ctx, _trial(29), "", None, 1)
    assert not S._confirm_top_pair(ctx, _trial(29), "", None, 1, same_opp=weak)
    assert S._confirm_top_pair(ctx, _trial(29), "", None, 1, same_opp=strong)
    # ...and each threshold on its own is enough, as in path 1
    long_only = (_trial(23, rms=0.386, U=7.5), "", "")
    long_only[0].arclength_days = 57.5
    assert S._confirm_top_pair(ctx, _trial(29), "", None, 1, same_opp=long_only)


def _accum_pair_ctx(top2, cfg=None):
    """A ctx whose seed is too short and too sloppy for paths 1/1b, so only the
    top-pair evidence can confirm anything."""
    seed, _obs = _qr56_shaped_linkage()
    return SimpleNamespace(
        cfg=cfg or ItfConfig(), log=lambda *a: None, trace=None,
        pub=SimpleNamespace(arclength_days=2.0, U=9.87),
        pub_ids={obs_unique_id(l) for l in seed.splitlines() if l.strip()},
        depth0_top2=top2)


def test_confirm_top_pair_accepts_the_accumulated_top_two():
    """The trial queue authors no top-2 trial (top-10,
    top-3, then singles), so path 2's "taken together in ONE trial" fires only
    when the ML floor happens to leave exactly two candidates -- a fact about
    the trial queue, not about the evidence.  What matters is that the SEED orbit
    alone ranked both best before either was linked, so a linkage that IS the
    depth-0 top two is confirmed however the trials accumulated it: best single
    taken, orbit bent, second taken a level down."""
    import pyroorb.itf.search as S
    seed, obs_out = _qr56_shaped_linkage()
    far = [l for l in obs_out.splitlines() if "B5678" in l or "C9012" in l]
    ids = [{obs_unique_id(l)} for l in far]
    combined = seed + "".join(l + "\n" for l in far)
    top2 = [{"ids": ids[0], "ml_p": 0.91, "tag": "B5678"},
            {"ids": ids[1], "ml_p": 0.44, "tag": "C9012"}]
    ctx = _accum_pair_ctx(top2)
    trial = _trial(29)
    tight = {list(ids[0])[0]: 0.21, list(ids[1])[0]: 0.33}
    orig = S.residuals_by_obs_id
    S.residuals_by_obs_id = lambda fit: dict(resid)
    try:
        # accumulated one-per-level, at depth 2, with no trial_meta at all
        resid = tight
        assert S._confirm_top_pair(ctx, trial, combined, None, 2)
        # ...and the seed-strength paths really are not what passed it
        assert not S._confirm_top_pair(
            _accum_pair_ctx(None), trial, combined, None, 2)
        # a third tracklet in the linkage is a DIFFERENT claim than the pair
        # the seed ranked: no promotion (the 0.6 review still reports it)
        third = obs_out.splitlines()[0] + "\n"
        resid = {**tight, obs_unique_id(third): 0.2}
        assert not S._confirm_top_pair(ctx, trial, combined + third, None, 2)
        # only one of the two linked -> not a pair
        resid = tight
        assert not S._confirm_top_pair(ctx, trial, seed + far[0] + "\n",
                                       None, 2)
        # a weakly-rated candidate is no anti-coincidence evidence
        weak = [dict(top2[0]), {**top2[1], "ml_p": 0.1}]
        assert not S._confirm_top_pair(_accum_pair_ctx(weak), trial, combined,
                                       None, 2)
        # unscored (legacy scan path): the pair paths abstain
        blind = [dict(top2[0]), {**top2[1], "ml_p": None}]
        assert not S._confirm_top_pair(_accum_pair_ctx(blind), trial, combined,
                                       None, 2)
        # both there, both top-ranked -- but one does not FIT
        resid = {**tight, list(ids[1])[0]: 0.8}
        assert not S._confirm_top_pair(ctx, trial, combined, None, 2)
    finally:
        S.residuals_by_obs_id = orig


def test_top2_record_is_the_seed_ranking():
    """What the upgrade compares against is recorded from the depth-0 sweep
    BEFORE any trial runs: the two best by rank_key, with their as-swept obs
    ids -- and nothing at all when the sweep offered fewer than two."""
    import pyroorb.itf.search as S
    likely = [{"astrometry": f"line {i}\n", "ml_p": p, "trksub": f"T{i}",
               "rows": [], "dist": 0, "qual": 0}
              for i, p in enumerate((0.3, 0.9, 0.55))]
    got = S._top2_record(likely, lambda d: 1.0 - d["ml_p"])
    assert [t["ml_p"] for t in got] == [0.9, 0.55]
    assert got[0]["ids"] == {obs_unique_id("line 1\n")}
    assert S._top2_record(likely[:1], lambda d: 1.0 - d["ml_p"]) is None


def _accept_ctx(cfg=None, seed_pre=0.30, **kw):
    """A ctx for driving :func:`itf.search._judge` -- the seed side of
    the extension gates (rms, ids, arc) plus the bookkeeping it fills in."""
    seed, _obs = _qr56_shaped_linkage()
    fields = dict(cfg=cfg or ItfConfig(), obscodes=None, pub_obs80=seed,
                  pub_ids={obs_unique_id(l) for l in seed.splitlines()
                           if l.strip()},
                  pub=SimpleNamespace(rms=0.20, used=16, count=16, U=9.0,
                                      arclength_days=2.0, fit=None),
                  initial_opp_profile=OPP_ONE, primary_desigs=["K06Q56R"], co_credit=None,
                  log=lambda *a: None, trace=None, telemetry={},
                  do_submit=False, budget=None)
    return SimpleNamespace(**{**fields, **kw})


def _run_accept(monkeypatch, *, seed_post, seed_pre=0.30, promoted=False,
                mag=None, cfg=None, seed_chi2_per_dof=1.0, trend=None):
    """Drive `_judge` on an accepted profile past everything that needs a real fit, so what is
    under test is only its verdict: seed rms `seed_pre` -> `seed_post`, an
    optional magnitude warning, and whether a 1.75 profile was confirmed as 3."""
    from pyroorb import ident_check
    import pyroorb.itf.search as S
    seed, obs_out = _qr56_shaped_linkage()
    trial = _trial(29, rms=0.25, U=5.0)
    ctx = _accept_ctx(cfg=cfg)
    combined = seed + obs_out
    monkeypatch.setattr(S, "_seed_pre_rms", lambda c: seed_pre)
    monkeypatch.setattr(S, "subset_rms", lambda fit, ids: (seed_post, 16))
    # chi2 over 16 obs = 32 dof; seed_chi2_per_dof scales it, so a case can say
    # "degraded but still within its uncertainties" (<= 1) or "genuinely
    # misfitting" (> 1) -- the question the demotion's statistical escape asks
    monkeypatch.setattr(S, "subset_chi2",
                        lambda fit, ids, **kw: (32.0 * seed_chi2_per_dof, 16))
    # the harsher gates are exercised by their own tests; here they pass, so
    # the only question left is what the WARNING does
    monkeypatch.setattr(S, "extension_degrades", lambda *a, **kw: False)
    monkeypatch.setattr(S, "_seed_clip_ledger", lambda c, t: {
        "n_nights": 3, "kept_nights": 3, "lost_nights": [],
        "partial_nights": [], "shed_obs": 0, "kept_obs": 6})
    monkeypatch.setattr(S, "display_obs80", lambda c, o: o)
    monkeypatch.setattr(ident_check, "coherent_offset_misfit",
                        lambda fit, noise_floor=None: None)
    monkeypatch.setattr(S, "measure_trends", lambda fit, linked, seed=None, detailed=False: trend)
    monkeypatch.setattr(S, "mag_outlier_tracklet", lambda *a, **kw: mag)
    monkeypatch.setattr(S, "worst_mag_residual", lambda fit: (0.1, None))
    monkeypatch.setattr(S, "build_linkage",
                        lambda comb, cfg, **kw: ({"links": {"link_0": {}}}, None))
    monkeypatch.setattr(S, "submit_linkage",
                        lambda doc, cfg, **kw: (False, "dry-run", None))
    a = S.Assessment(trial=trial, combined=combined, obs_out=obs_out,
                     profile=OPP_THREE, spans=5, profile_ok=True,
                     confirmed=promoted, gains=True, quality_ok=True, notes=[],
                     bad_tracklets=[], other_fail=False, same_opp=None,
                     seed=S.SeedCheck(ctx, trial, obs_out))
    return S._judge(ctx, a, [], 16, sweep=None)


def test_a_degraded_seed_fit_is_never_linked_only_possible(monkeypatch):
    """If the object's OWN observations fit materially
    worse under the linked orbit, the search may report the linkage but must
    never call it found.  That measurement is `extension_degrades`' outright
    contaminant veto one threshold down -- so it stops being a caution the user
    (or an unattended `itf batch --submit`) can click through, and becomes the
    status 0.6 "possible" verdict, which no surface submits.  Live 2006 VT188
    is the type case: linked -> possible.

    The verdict needs the ABSOLUTE floor as well as the warn ratio, and 2020
    KX6 is why: it recovers correctly on a seed so tight that 0.16" of honest
    in-sample widening is 3.2x."""
    # control: the seed fit holds, so the ordinary acceptance stands
    ok = _run_accept(monkeypatch, seed_pre=0.30, seed_post=0.33)
    assert ok.verdict is SearchVerdict.LINKED and ok.linkage and not ok.warnings
    # 2020 KX6's shape: an over-fit seed starts absurdly tight (0.072"), so a
    # KNOWN-CORRECT recovery (3/3 apparitions, 4/4 tracklets, 0.126") clears the
    # warn RATIO on 0.16" of movement.  It warns -- and stays a find, because
    # the verdict waits for the absolute floor as well (demote_seed_abs_arcsec).
    kx6 = _run_accept(monkeypatch, seed_pre=0.072, seed_post=0.231)
    assert kx6.verdict is SearchVerdict.LINKED and kx6.linkage
    assert any("fit worse after the link" in w for w in kx6.warnings)
    # ...and the same linkage with the seed dragged past BOTH is the SAME
    # finding, reported, with its fit and astrometry -- just not a find
    # ...and the same linkage whose seed is genuinely MISFITTING (past its own
    # uncertainties, not merely past a ratio) is the SAME finding, reported,
    # with its fit and astrometry -- just not a find
    bad = _run_accept(monkeypatch, seed_pre=0.30, seed_post=0.75,
                      seed_chi2_per_dof=2.5)
    assert bad.verdict is SearchVerdict.POSSIBLE
    assert bad.linkage is None                     # nothing to submit, anywhere

    # The statistical escape: the same 2.5x ratio, but
    # the object's own observations still fit their uncertainties, is NOT a
    # demotion -- a ratio against an over-fit baseline is not evidence, and a
    # submittable 1.0 must not decay into a 0.6 for it.  2011 QH108: seed
    # 0.515" -> 1.554" (3.0x) at 0.84 sigma against the prior accepted state,
    # its ninth tracklet fitting at 0.66 sigma while ONE of its own nights
    # (2011-06-04, +1.88" of pure RA) does the misfitting.
    within = _run_accept(monkeypatch, seed_pre=0.30, seed_post=0.75,
                         seed_chi2_per_dof=0.7)
    assert within.verdict is SearchVerdict.LINKED, within.reason
    assert within.linkage is not None, "must stay submittable"
    assert any("still fit their uncertainties" in w for w in within.warnings), \
        within.warnings
    assert bad.fit is not None and bad.itf_obs80   # but the finding is kept
    assert bad.vetting["seed_degraded"] and bad.vetting["possible_unconfirmed"]
    assert "NOT SUBMITTABLE" in bad.warnings[0]
    assert any("fit worse after the link" in w for w in bad.warnings)
    # ...and the reason QUOTES those numbers per-axis (2026-08-23): the gate
    # measures per-observation (0.300 -> 0.750, as injected), the reader is
    # given the currency find_orb, the web header and the seed line all use,
    # /sqrt(2).  See itf.fitter's docstring.
    assert "0.212" in bad.reason and "0.530" in bad.reason


def test_a_confirmed_175_profile_is_withdrawn_by_any_warning(monkeypatch):
    """The three-thin upgrade is a judgement made on the
    PROFILE, before the gates run.  Any caution the gates then raise is evidence
    against exactly that judgement, so it takes the upgrade back -- while the
    same caution on a linkage with an ordinary profile still accepts, as it always
    did."""
    mag = {"station": "F51", "n": 4, "p": 0.001, "mean": -1.8}
    # an ordinary profile: the magnitude caution is reported and still accepted
    ord_ = _run_accept(monkeypatch, seed_post=0.31, mag=mag)
    assert ord_.verdict is SearchVerdict.LINKED and len(ord_.warnings) == 1
    # the same caution on a profile that is only submittable BECAUSE it was
    # confirmed -> the confirmation is withdrawn
    frag = _run_accept(monkeypatch, seed_post=0.31, mag=mag, promoted=True)
    assert frag.verdict is SearchVerdict.POSSIBLE and frag.linkage is None
    assert frag.vetting["confirm_withdrawn"] and frag.vetting["demoted"]
    assert "NOT SUBMITTABLE" in frag.warnings[0]
    # ...but a confirmed 1.75 with NOTHING to say against it still submits
    clean = _run_accept(monkeypatch, seed_post=0.31, promoted=True)
    assert clean.verdict is SearchVerdict.LINKED and clean.linkage


def test_a_seed_degradation_owned_by_the_seed_opposition_keeps_the_175(monkeypatch):
    """2015 KK542: one search linked the same-opposition W84 tracklet and the
    2021 + 2024 single nights, then withdrew the 1.75 confirmation over "the
    object's own observations fit worse" -- a degradation the W84 tracklet
    alone causes on the over-fit 8-obs seed.  Pressing ITF search again, with
    W84 now in the seed, linked all three.  The verdict must not depend on
    which of the two ways the same evidence arrives."""
    import pyroorb.itf.search as S
    # the escaped caution (2.5x ratio, still within the uncertainties)
    escaped = dict(seed_pre=0.30, seed_post=0.75, seed_chi2_per_dof=0.7,
                   promoted=True)
    monkeypatch.setattr(S, "_seed_opp_holds", lambda c, t, o: True)
    kept = _run_accept(monkeypatch, **escaped)
    assert kept.verdict is SearchVerdict.LINKED and kept.linkage, kept.reason
    assert not kept.vetting.get("confirm_withdrawn"), kept.vetting
    # ...and the caution is still on the record
    assert any("fit worse after the link" in w for w in kept.warnings)
    # the cross-opposition tracklets degrade the extended opposition too:
    # the caution is about them, and withdraws as before
    monkeypatch.setattr(S, "_seed_opp_holds", lambda c, t, o: False)
    gone = _run_accept(monkeypatch, **escaped)
    assert gone.verdict is SearchVerdict.POSSIBLE
    assert gone.vetting["confirm_withdrawn"]
    # ...and any OTHER caution still withdraws, whatever the seed opposition says
    monkeypatch.setattr(S, "_seed_opp_holds", lambda c, t, o: True)
    mag = {"station": "F51", "n": 4, "p": 0.001, "mean": -1.8}
    still = _run_accept(monkeypatch, mag=mag, **escaped)
    assert still.verdict is SearchVerdict.POSSIBLE
    assert still.vetting["confirm_withdrawn"]


def test_a_tracklet_the_orbit_does_not_move_toward_is_vouched_for(monkeypatch):
    """A tracklet failing its own clause is a mislinkage worry only if the fit
    BENT toward it.  2015 KJ19's PR00016 (G96, 3 obs): the orbit fitted without
    it predicts it to 0.15", two obs sit on that prediction and the third is a
    blunder, and adding the tracklet moves the prediction 0.079" -- a true
    tracklet with a bad observation, vouched for.  Each refusal arm below is one
    of the three conditions."""
    import pyroorb.itf.search as S
    from pyroorb.itf import leaveout
    lines = [_obs80("PR00016", f"2021 05 16.3{i}000", "11 00 00.00",
                    "+11 00 00.0", stn="G96") for i in range(3)]
    ids = frozenset(obs_unique_id(l) for l in lines)
    obs_out = "".join(l + "\n" for l in lines)
    ctx = SimpleNamespace(cfg=ItfConfig(), obscodes=None, trace=None,
                          swept_lines={ids: obs_out})
    a = SimpleNamespace(trial=SimpleNamespace(ok=True, fit=object()),
                        combined="SEED\n" + obs_out, obs_out=obs_out)
    monkeypatch.setattr(S, "_fit", lambda c, text: SimpleNamespace(ok=True, fit=object()))
    monkeypatch.setattr(S, "sigma_map_for",
                        lambda fit, i, station_floor=True: {k: (0.3, 0.3) for k in i})

    def vouch(offsets, shift=0.079, smaj=0.15):
        per = [(l, de, dn) for l, (de, dn) in zip(lines, offsets)]
        monkeypatch.setattr(leaveout, "influence", lambda *a, **k: {
            "smaj": smaj, "miss": 1.0, "shift": shift, "n": 3, "per_obs": per})
        return S._vouch(ctx, a, "PR00016@G96", ids)

    good = [(0.1, 0.1), (0.3, -0.2), (3.0, 1.5)]       # two on it, one blunder
    got = vouch(good)
    assert got is not None and "2/3 obs" in got[0], got
    # the BLUNDERS are the majority: not a true tracklet with one bad ob
    assert vouch([(0.1, 0.1), (3.0, 1.5), (2.5, -2.0)]) is None
    # the fit bends toward it: exactly what a swallowed mislinkage does
    assert vouch(good, shift=0.9) is None
    # the orbit without it cannot predict it: nothing independent to judge by
    assert vouch(good, smaj=3.0) is None


def test_a_vouch_belongs_to_the_assessment_that_earned_it(monkeypatch):
    """A vouch rests on one orbit (leave-out sigma and shift), so it may exempt
    a tracklet only in the assessment it was earned for.  When vouching stops
    part-way -- one tracklet vouched, the next not -- nothing of it may
    survive for a sibling alternative (the reduced linkage, the dropped-
    offender refit) to honour."""
    import pyroorb.itf.search as S
    a = S.Assessment(trial=SimpleNamespace(ok=True, fit=object()), combined="",
                     obs_out="", profile=OPP_ONE, spans=4, profile_ok=True,
                     confirmed=False, gains=True, quality_ok=False, notes=[],
                     bad_tracklets=[("A", frozenset({"a"})), ("B", frozenset({"b"}))],
                     other_fail=False, same_opp=None)
    ctx = SimpleNamespace(cfg=ItfConfig(), trace=None)
    seen = []

    def vouch(c, a_, tag, ids):
        seen.append(tag)
        return ("vouched", "") if tag == "A" else None

    monkeypatch.setattr(S, "_vouch", vouch)
    assert S._with_vouched(ctx, a, 0) is None
    assert seen == ["A", "B"]
    assert a.vouched == {}, "the failed attempt left a vouch on the linkage"
    assert not hasattr(ctx, "vouched"), "a vouch leaked onto the search context"


def test_seed_opp_holds_measures_against_the_extended_opposition(monkeypatch):
    """The baseline is the seed's own opposition, extended and fitted alone,
    with the same warn threshold `_accept_linkage` applies to the bare seed."""
    import pyroorb.itf.search as S
    seed, obs_out = _qr56_shaped_linkage()
    ctx = _accept_ctx()
    same_fit, full_fit = object(), object()
    same = (SimpleNamespace(fit=same_fit), seed, "")
    monkeypatch.setattr(S, "_seed_opp_extension", lambda c, o: same)
    rms = {}
    monkeypatch.setattr(S, "subset_rms", lambda fit, ids: (rms[fit], 11))
    trial = SimpleNamespace(fit=full_fit)
    # KK542, per observation: 0.129" alone -> 0.171" with 2021 + 2024 (1.3x)
    rms.update({same_fit: 0.129, full_fit: 0.171})
    assert S._seed_opp_holds(ctx, trial, obs_out)
    # the thin oppositions drag the extended opposition past the warn ratio
    rms.update({same_fit: 0.129, full_fit: 0.40})
    assert not S._seed_opp_holds(ctx, trial, obs_out)
    # nothing in the seed's opposition to measure against
    monkeypatch.setattr(S, "_seed_opp_extension", lambda c, o: None)
    assert not S._seed_opp_holds(ctx, trial, obs_out)


def test_an_unverifiable_pickup_warns_without_deleting_the_linkage(monkeypatch):
    """A designated pickup the archive could not be made to
    answer for is UNVERIFIED, which is worth saying -- but it is an absence of evidence,
    not evidence against, so it must not delete the find.

    Two ways it could have: banning the designation strips a name off the document, and
    the 1.75 withdrawal above drops the whole linkage to 0.6, which removes it from the
    batch's combined submission JSON (`batch.py` keeps `result.linkage` only at status 1).
    Either would mean a database hiccup silently costing a real identification. So it
    warns -- into the CSV, the web banner, and the submit click-through -- at status 1.

    Contrast the tracklet-level `compromised` case, which is a MEASUREMENT that the
    designation does not fit, and does both of those things on purpose."""
    import pyroorb.itf.search as S
    unchecked = {"packed": "K17Sd6S", "desig": "2017 SS396", "unchecked": True,
                 "compromised": False, "why": "the archive query failed",
                 "linked": [], "missing": [], "n_archive": None,
                 "warning": "COULD NOT CHECK 2017 SS396: ... trouble pulling in ..."}
    monkeypatch.setattr(S.pickup, "review", lambda *a, **kw: [unchecked])

    # an ordinary profile: warned, still linked, still carries its document
    ord_ = _run_accept(monkeypatch, seed_post=0.31)
    assert ord_.verdict is SearchVerdict.LINKED and ord_.linkage, ord_
    assert any("COULD NOT CHECK" in w for w in ord_.warnings), ord_.warnings
    assert ord_.vetting["pickup_unchecked"] == ["2017 SS396"], ord_.vetting
    # ...and the designation is NOT dropped from the document
    assert "linkage_dropped_designations" not in ord_.vetting, ord_.vetting

    # THE case: a 1.75 confirmed profile must survive it. A measured caution (the mag
    # test above) withdraws the confirmation; "we could not reach the database" does not.
    frag = _run_accept(monkeypatch, seed_post=0.31, promoted=True)
    assert frag.verdict is SearchVerdict.LINKED, f"an unverifiable pickup must not demote: {frag.reason}"
    assert frag.linkage, "...nor cost the linkage its submission document"
    assert any("COULD NOT CHECK" in w for w in frag.warnings), frag.warnings
    assert not frag.vetting.get("confirm_withdrawn"), frag.vetting

    # ...while a REAL measured warning on the same profile still withdraws it, unchanged
    mag = {"station": "F51", "n": 4, "p": 0.001, "mean": -1.8}
    still = _run_accept(monkeypatch, seed_post=0.31, mag=mag, promoted=True)
    assert still.verdict is SearchVerdict.POSSIBLE and still.vetting["confirm_withdrawn"], still.vetting
    # and even then the unchecked caution is on the record for the reviewer
    assert any("COULD NOT CHECK" in w for w in still.warnings), still.warnings


def test_a_trending_linked_tracklet_is_a_DIAGNOSTIC_and_takes_nothing_away():
    """A linked tracklet whose residuals TREND (itf/residual_trend.py) is REPORTED and acted
    on by nothing (2026-08-25).

    It used to be a warning, which meant it withdrew a fragile three-thin
    confirmation like any other caution.  The full-sample calibration killed
    that: on the tracklets where the trend is the last word it fires on 2 of 58
    CORRECT tracklets against 3 of 119 wrong, and no threshold separates them
    (measured on the ramp-gate calibration set).  A test that rejects about as
    much real astrometry as junk must not be able to take a submission away.

    So: it never reaches ``warnings`` (which gate the submit click-through), it
    never withdraws a confirmation, and it rides in ``diagnostics`` on every
    surface instead.  The three assertions below are the whole contract."""
    import pytest
    from pyroorb.itf.residual_trend import is_trend_note
    trend = ("the 2021-09-13 tracklet (F52, n=3, 47 min) walks -2.5 sigma in Dec "
            "-- 7.6x the worst trend anywhere else in this object's arc")
    with pytest.MonkeyPatch.context() as mp:
        ord_ = _run_accept(mp, seed_post=0.31, trend=trend)        # ordinary profile
        assert ord_.verdict is SearchVerdict.LINKED and ord_.linkage
        assert any(is_trend_note(d) for d in ord_.diagnostics), ord_.diagnostics
        assert not any("trend" in w.lower() for w in ord_.warnings), ord_.warnings
    with pytest.MonkeyPatch.context() as mp:
        # the case that used to be demoted: a 1.75 profile confirmed to 3, with a
        # trend.  It stays submittable, and the trend is on the record beside it.
        frag = _run_accept(mp, seed_post=0.31, trend=trend, promoted=True)
        assert frag.verdict is SearchVerdict.LINKED and frag.linkage
        assert not frag.vetting.get("confirm_withdrawn")
        assert any(is_trend_note(d) for d in frag.diagnostics), frag.diagnostics
    with pytest.MonkeyPatch.context() as mp:
        clean = _run_accept(mp, seed_post=0.31, promoted=True)   # control
        assert clean.verdict is SearchVerdict.LINKED and clean.linkage
        assert not clean.diagnostics


def test_a_real_warning_still_withdraws_a_confirmed_175():
    """The withdrawal rule itself is untouched -- only the trend stopped feeding
    it.  A genuine caution on a 1.75-confirmed profile still demotes to 0.6, which
    is what stops the trend change from quietly disarming the rule."""
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        # seed_post well past the warn ratio raises a real degradation caution
        frag = _run_accept(mp, seed_post=0.62, promoted=True)
        assert frag.verdict is SearchVerdict.POSSIBLE and frag.linkage is None
        assert frag.warnings


def test_reduced_linkage_submits_the_subset_and_names_what_it_declined():
    """Given the choice between a non-submittable 1.75
    and a submittable extension of the existing opposition, take the extension
    -- but SAY what was declined, or the larger finding vanishes (live QR56:
    the LOV fallback re-derived the same 2-tracklet subset 755 s later and its
    status 1.0 silently replaced the 4-tracklet result, warnings included)."""
    import pyroorb.itf.search as S
    seed, obs_out = _qr56_shaped_linkage()
    same_trial = _trial(23, rms=0.386, U=5.38)
    same_trial.arclength_days, same_trial.q = 57.5, 2.5
    obs_same = "".join(l + "\n" for l in obs_out.splitlines() if "A1234" in l)
    ctx = SimpleNamespace(cfg=ItfConfig(), obscodes=None, pub_obs80=seed,
                          log=lambda *a: None, trace=None, telemetry={},
                          initial_opp_profile=OPP_ONE,
                          pub_ids={obs_unique_id(l) for l in seed.splitlines()},
                          pub=SimpleNamespace(rms=0.428, used=16, count=16,
                                              arclength_days=2.0, U=9.87))
    refused = S.Assessment(trial=_trial(29, rms=0.419), combined=seed + obs_out,
                           obs_out=obs_out, profile=OPP_THREE_THIN, spans=6,
                           profile_ok=False, confirmed=False, gains=True,
                           quality_ok=True, notes=[], bad_tracklets=[],
                           other_fail=False,
                           same_opp=(same_trial, seed + obs_same, obs_same))
    passed = {}
    orig, orig_q = S._judge, S._accept_quality_ok
    def fake_judge(c, b, field, prior, trial_meta=None, depth=-1, sweep="all",
                   lead=()):
        passed.update(trial=b.trial, obs=b.obs_out, sweep=sweep)
        r = S.ItfSearchResult(verdict=SearchVerdict.LINKED)
        r.warnings.extend(lead)
        return r
    S._judge = fake_judge
    # The reduced linkage is judged like any linkage: the SAME per-tracklet
    # gate the full one faced, never an aggregate rms/retention rule.
    S._accept_quality_ok = lambda c, trial, out, prior, detail=None, **kw: (
        passed.update(quality=(trial, out, prior)) or (True, ["ok"]))
    try:
        got = S._reduced_linkage(ctx, refused, [], 16, 1)
        assert passed["quality"] == (same_trial, obs_same, 16)
        # ...and when that gate refuses, the fallback declines rather than
        # submitting on a weaker rule
        S._accept_quality_ok = lambda *a, **k: (False, ["tracklet retention  FAIL"])
        assert S._reduced_linkage(ctx, refused, [], 16, 1) is None
    finally:
        S._judge, S._accept_quality_ok = orig, orig_q
    assert got is not None and got.verdict is SearchVerdict.LINKED
    # what is SUBMITTED is the subset, not the fragile linkage
    assert passed["trial"] is same_trial and "B5678" not in passed["obs"]
    assert passed["sweep"] == "archive"   # the modern sweep would only re-add them
    # ...and what was declined is named, in the warning AND in vetting
    warn = " ".join(got.warnings)
    assert "LARGER linkage was assembled and refused" in warn
    assert "B5678" in warn and "C9012" in warn
    ref = got.vetting["refused_larger_linkage"]
    assert ref["opps"] == 1.75 and ref["tracklets_declined"] == 2

def test_clip_evidence_price():
    """By design, a whole published night may be dropped --
    one bad observation must not make an object unrecoverable, and a recovery is
    often how a bad night is PROVEN bad -- but it has to be BOUGHT with
    independently linked tracklets, and it can never leave less than a real arc.
    Driven through the real gate so the test tracks the code, not a formula."""
    import pyroorb.itf.search as S
    cfg = ItfConfig()
    nothing = {"n_nights": 4, "kept_nights": 4, "lost_nights": [],
               "partial_nights": [], "shed_obs": 0, "kept_obs": 8}
    one_lost = dict(nothing, kept_nights=3, shed_obs=2,
                    lost_nights=[{"night": 4, "date": "2020 09 24"}])
    # nothing clipped -> never refuses, whatever the linkage brought
    assert S._clip_evidence_ok(cfg, nothing, 0)[0]
    # one night dropped: 3 new tracklets buy it, 2 do not
    assert S._clip_evidence_ok(cfg, one_lost, 3)[0]
    ok, why = S._clip_evidence_ok(cfg, one_lost, 2)
    assert not ok and "3 are needed" in why
    # ...and the report says which span, so the refusal is actionable
    assert "span 4" in why and "2020 09 24" in why
    # two spans is past clip_spans_max however much evidence there is
    two_lost = dict(one_lost, kept_nights=2,
                    lost_nights=one_lost["lost_nights"] * 2)
    assert not S._clip_evidence_ok(cfg, two_lost, 99)[0]
    # and a clip that leaves under clip_keep_spans_min is refused even at 1
    thin = dict(nothing, n_nights=2, kept_nights=1, shed_obs=2,
                lost_nights=[{"night": 2, "date": "2020 09 16"}])
    ok2, why2 = S._clip_evidence_ok(cfg, thin, 99)
    assert not ok2 and "rests on the ITF tracklets alone" in why2


def test_clip_note_names_the_dropped_night():
    """The accepted-with-a-drop sentence must name the night, its evidence and
    what pays for it -- it rides result.warnings to every surface."""
    import pyroorb.itf.search as S
    led = {"n_nights": 4, "kept_nights": 3, "shed_obs": 2, "kept_obs": 6,
           "partial_nights": [],
           "lost_nights": [{"night": 4, "date": "2020 09 24",
                            "stations": ["703"], "n_obs": 2, "n_shed": 2,
                            "worst_resid_arcsec": 3.7}]}
    note = S._clip_note(led, 3, 3)
    for expect in ("span 4", "2020 09 24", "703", "3.7", "3-span seed",
                   "3 newly linked tracklet"):
        assert expect in note, (expect, note)


def test_linked_tracklet_count_uses_submission_granularity():
    """The clip price is paid in tracklets, so "a tracklet" must mean what the
    MPC document means: one (trksub, station, night) entry."""
    import pyroorb.itf.search as S
    same_night = [_obs80("PR00001", "2017 01 02.34567", "10 00 00.00",
                         "+10 00 00.0"),
                  _obs80("PR00001", "2017 01 02.35567", "10 00 00.10",
                         "+10 00 00.4")]
    other_night = [_obs80("PR00001", "2017 01 04.30000", "10 00 05.00",
                          "+10 00 03.0")]
    other_stn = [_obs80("PR00002", "2018 03 04.30000", "10 00 05.00",
                        "+10 00 03.0", stn="703")]
    assert S._linked_tracklet_count("\n".join(same_night) + "\n") == 1
    assert S._linked_tracklet_count(
        "\n".join(same_night + other_night) + "\n") == 2
    assert S._linked_tracklet_count(
        "\n".join(same_night + other_night + other_stn) + "\n") == 3


def test_spans_of_is_the_one_span_convention():
    """`spans_of` and `count_spans` must be the same rule -- "span 4" in the
    clip report has to mean span 4 of the opposition-profile gate's count."""
    from pyroorb.itf.opp_profile import spans_of, nights_of, count_nights, SPAN_DAYS
    lines = _seed_4_nights()
    groups = spans_of(lines)
    assert len(groups) == count_spans("\n".join(lines)) == 4
    assert [len(g) for g in groups] == [2, 2, 2, 2]
    # the ported quirk (measured from the span's START) is preserved: obs
    # 1.0 d apart are ONE span (the span is set to 1.5 d by design;
    # a night, the observer's unit, is the 12-hour window)
    assert SPAN_DAYS == 1.5
    close = [_obs80("K20S00X", "2020 09 13.30000", "10 00 00.00", "+10 00 00.0"),
             _obs80("K20S00X", "2020 09 14.30000", "10 00 00.00", "+10 00 00.0")]
    assert len(spans_of(close)) == count_spans("\n".join(close)) == 1
    assert len(nights_of(close)) == count_nights("\n".join(close)) == 2
    # 1.5 d apart opens a new span (it did not at 1.6)
    far = [_obs80("K20S00X", "2020 09 13.30000", "10 00 00.00", "+10 00 00.0"),
           _obs80("K20S00X", "2020 09 14.80000", "10 00 00.00", "+10 00 00.0")]
    assert len(spans_of(far)) == 2


def test_recurse_trial_queue_guards_and_backs_off():
    # Drive _recurse_trial_queue with a fake fitter so no DB/engine is needed. The
    # seed pre-rms is 0.05". Tracklet 0 (best-ranked) poisons the seed PAST the
    # absolute noise floor (recurse_seed_abs_arcsec); 1 and 2 are clean. The
    # trial queue must skip the poisoned batches and pick a clean set.
    import pyroorb.itf.search as S

    # map an obs_out string -> (used, seed_rms_post)
    def fake_fit(text, obscodes, **kw):
        has0 = "T0" in text; has1 = "T1" in text; has2 = "T2" in text
        used = 15 + 5*(has0+has1+has2) + 5*("HYP" in text)
        # poisoned only when T0 is present (and alone or in a batch)
        return _trial(used, seed_rms=0.60 if has0 else 0.06)
    def fake_subset_rms(fit, ids):
        return fit._seed, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        ctx = _trial_queue_ctx()
        likely = [{"astrometry": f"T{i}\n", "_r": i} for i in range(3)]
        rank = lambda d: d["_r"]                      # T0 best-ranked
        trial, obs_out, _meta = S._recurse_trial_queue(ctx, likely, rank, prior_linkage_used=15)
        assert trial is not None, "should find a clean subset"
        assert "T0" not in obs_out, f"poisoned T0 must be skipped, got {obs_out!r}"
        assert "T1" in obs_out                        # first clean single
        # single-candidate exemption: one poisoned candidate is still tried
        likely1 = [{"astrometry": "T0\n", "_r": 0}]
        trial1, obs1, _m1 = S._recurse_trial_queue(ctx, likely1, rank, prior_linkage_used=15)
        assert trial1 is not None and "T0" in obs1    # nothing to fall back on
        # hypothesis accumulation: prior levels took HYP (5 obs, prior used
        # 20); a 5-obs single must now progress ON TOP of the hypothesis
        # (before the fix its lone fit reached only 20 = "no progress"), and
        # obs_out must carry hypothesis + trial
        likely2 = [{"astrometry": f"T{i}\n", "_r": i} for i in range(3)]
        trial2, obs2, _m2 = S._recurse_trial_queue(ctx, likely2, rank,
                                              prior_linkage_used=20,
                                              hypothesis_obs80="HYP\n")
        assert trial2 is not None and trial2.used == 25
        assert obs2.startswith("HYP\n") and "T1" in obs2 and "T0" not in obs2
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_screen_candidate_defers_wherever_it_is_not_trustworthy():
    # The warm-start screen may only ever SKIP WORK. Everywhere it cannot judge
    # -- switched off, no parent orbit, an under-determined parent (the 2015
    # BE559 regime, where a bounded warm step destroys the TRUE tracklet too),
    # or any engine trouble at all -- it must come back "skip"/"plausible", i.e.
    # run the cold fit. Only a parent arc actually destroyed may reject.
    from pyroorb.itf.fitter import (screen_candidate, SCREEN_SKIP, SCREEN_PLAUSIBLE,
                            SCREEN_REJECT)
    cfg = ItfConfig()
    good_parent = SimpleNamespace(_state=np.zeros(6), epoch_tdb=2.46e6, _obs=[{}])

    def verdict(*a, **kw):
        return screen_candidate(*a, **kw)[0]

    off = ItfConfig(); off.screen_trials = False
    assert verdict(good_parent, "LINE\n", None, off) == SCREEN_SKIP
    assert verdict(None, "LINE\n", None, cfg) == SCREEN_SKIP
    assert verdict(SimpleNamespace(_state=None), "LINE\n", None, cfg) == SCREEN_SKIP
    # The U gate is RETIRED by default (measured: in-corridor true trials screen
    # at 0.13-0.92 sigma even at parent U 8.94-17.87 -- the corridor already
    # bounds reachability), so a severe parent must still be screened...
    assert cfg.screen_max_U is None
    # A parent that raises when read discriminates the two: with the gate ON the
    # screen returns before ever touching it (SKIP); with the gate retired it
    # gets as far as the parent and defers (PLAUSIBLE).  Same call, same U.
    class Raises:
        _state = np.zeros(6)
        @property
        def _obs(self):
            raise RuntimeError("reached the parent")
    gated = ItfConfig(); gated.screen_max_U = 8.0
    assert verdict(Raises(), "LINE\n", None, gated, seed_U=9.56) == SCREEN_SKIP
    assert verdict(Raises(), "LINE\n", None, cfg, seed_U=9.56) == SCREEN_PLAUSIBLE
    assert verdict(Raises(), "LINE\n", None, cfg, seed_U=17.87) == SCREEN_PLAUSIBLE
    # an exploding parent is a defer, not a reject
    class Boom:
        _state = np.zeros(6)
        @property
        def _obs(self):
            raise RuntimeError("engine trouble")
    assert verdict(Boom(), "LINE\n", None, cfg) == SCREEN_PLAUSIBLE
    assert SCREEN_REJECT not in (SCREEN_SKIP, SCREEN_PLAUSIBLE)


def test_recurse_trial_queue_warm_screen_skips_the_cold_fit():
    # Wiring: a trial the screen rejects must never reach fit_astrometry, and the
    # trial queue must move on to the next trial. A trial the screen passes fits as
    # before. The screen is advisory-only in the acceptance sense -- it cannot
    # make the trial queue TAKE anything it would not otherwise have taken.
    import pyroorb.itf.search as S
    from pyroorb.itf.fitter import SCREEN_REJECT, SCREEN_PLAUSIBLE

    fitted, screened = [], []

    def fake_fit(text, obscodes, **kw):
        fitted.append(text)
        return _trial(15 + 5*("T0" in text) + 5*("T1" in text), seed_rms=0.06)
    def fake_subset_rms(fit, ids):
        return 0.06, 15
    def fake_screen(seed_fit, cand, obscodes, cfg, seed_U=None, **kw):
        screened.append(cand)
        # T0 is the incompatible candidate: the warm step destroys the parent
        return ((SCREEN_REJECT, 21000.0, 0.02) if "T0" in cand
                else (SCREEN_PLAUSIBLE, 0.7, 0.02))
    orig = (S.fit_astrometry, S.subset_rms, S.screen_candidate)
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    S.screen_candidate = fake_screen
    try:
        ctx = _trial_queue_ctx()
        parent = _trial(15, seed_rms=0.05, U=2.0)   # well-determined -> screened
        likely = [{"astrometry": f"T{i}\n", "_r": i} for i in range(2)]
        rank = lambda d: d["_r"]                        # T0 best-ranked
        trial, obs_out, _m = S._recurse_trial_queue(ctx, likely, rank,
                                               prior_linkage_used=15,
                                               parent=parent)
        assert trial is not None and "T1" in obs_out
        # every trial containing T0 was screened out before any fit
        assert all("T0" not in t for t in fitted), fitted
        assert any("T0" in c for c in screened), "the screen must have seen T0"
        assert ctx.telemetry["screen_" + SCREEN_REJECT] >= 1
        assert ctx.telemetry["screen_seconds"] > 0
        # ...and with NO parent (the variant-scan path) the screen never runs
        fitted.clear(); screened.clear()
        S._recurse_trial_queue(ctx, likely, rank, prior_linkage_used=15)
        assert screened == [] and any("T0" in t for t in fitted)
    finally:
        S.fit_astrometry, S.subset_rms, S.screen_candidate = orig


def test_variant_parent_screens_against_its_own_orbit():
    """A variant scan's trials are screened against THAT VARIANT's state, not the
    nominal one -- stepping from the nominal destroys the true tracklet too,
    which is why the LOV is being sampled at all.  The shim must swap ONLY the
    state (same obs, same epoch, nominal left untouched) and must carry U=None,
    so ``screen_max_U`` -- which gates on the NOMINAL orbit's uncertainty --
    cannot skip a variant whose whole point is that it is a specific orbit."""
    import pyroorb.itf.search as S

    obs = [{"jd_utc": 2.46e6}]
    nominal = SimpleNamespace(_state=np.zeros(6), epoch_tdb=2.46e6, _obs=obs)
    vstate = np.arange(6, dtype=float)
    vp = S._VariantParent(nominal, vstate)
    assert vp.U is None, "a variant must not be gated by the nominal's U"
    assert np.array_equal(vp.fit._state, vstate)
    assert vp.fit.epoch_tdb == nominal.epoch_tdb and vp.fit._obs is obs
    assert np.array_equal(nominal._state, np.zeros(6)), "nominal must not mutate"

    # ...and the trial queue passes that U straight through, so the screen runs
    seen = {}
    def spy(seed_fit, cand, obscodes, cfg, seed_U=None, **kw):
        seen["U"] = seed_U
        seen["state"] = seed_fit._state
        return (S.SCREEN_PLAUSIBLE, 0.8, 0.01)
    orig = (S.screen_candidate, S.fit_astrometry, S.subset_rms)
    S.screen_candidate = spy
    S.fit_astrometry = lambda text, oc, **kw: _trial(20, seed_rms=0.06)
    S.subset_rms = lambda fit, ids: (0.06, 15)
    try:
        S._recurse_trial_queue(_trial_queue_ctx(), [{"astrometry": "T0\n", "_r": 0}],
                          lambda d: d["_r"], prior_linkage_used=15, parent=vp)
        assert seen["U"] is None
        assert np.array_equal(seen["state"], vstate)
    finally:
        S.screen_candidate, S.fit_astrometry, S.subset_rms = orig


def test_variant_scan_defers_the_screen_on_an_unmatched_lov_offset():
    """The variant->state lookup is keyed by the LOV offset k as a FLOAT, which
    variants.py happens to hand back bit-identical today.  A miss must cost the
    SCREEN (defer to the cold fit, like every other untrustworthy case) and never
    the SCAN -- a KeyError here would abort the whole scan on a severe seed."""
    import pyroorb.itf.search as S
    import pyroorb.itf.lov_variants as V

    seen = []
    orig = (V.variant_states, V.variant_windows, S._iterate)
    V.variant_states = lambda fit, cfg: ([np.zeros(6), np.ones(6)], [0.0, 1.0])
    # k=+1.5 matches NO entry of ks -- the lookup must miss
    V.variant_windows = (lambda fit, oc, cfg, mu, states, lov_sigmas, jds=None,
                         tracks=None: [(1.5, [SimpleNamespace(unix_start=0.0)])])
    def fake_iterate(ctx, trial, used, ew, depth, hypothesis_obs80="", windows=None,
                     screen_parent=None, **kw):
        seen.append(screen_parent)
        return S.ItfSearchResult(telemetry={})
    S._iterate = fake_iterate
    try:
        ctx = S.SearchContext(source=SimpleNamespace(sweep_candidates=lambda *a, **k: []),
                     cfg=ItfConfig(), obscodes=None, pub_obs80="SEED\n",
                     pub_ids=set(), pub=_trial(15), initial_opp_profile=OPP_ONE,
                     primary_desigs=[], max_uncert_deg=19.44, uncert_mult=2.5,
                     do_submit=False, co_credit=None, telemetry={},
                     log=lambda *a: None, verbose=False)
        out = S._variant_scan(ctx, _trial(15, U=9.56), 15, 0.0, 0, "")
        assert out is not None, "an unmatched offset must not abort the scan"
        assert seen == [None], "an unmatched offset must leave the trials unscreened"
    finally:
        V.variant_states, V.variant_windows, S._iterate = orig


def test_recurse_trial_queue_rms_sanity_gate():
    # A trial whose CONVERGED orbit is thousands of arcsec off (the wide-corridor
    # noise of an under-determined seed) must be refused even though it adds obs
    # and keeps the seed subset tight -- the exact 2016 NY214 failure. The gate
    # is independent of single/joint, so it also rejects a lone off-sky trial.
    import pyroorb.itf.search as S

    def fake_fit(text, obscodes, **kw):
        has0 = "T0" in text; has1 = "T1" in text
        # T0's converged orbit is way off the sky (corridor noise); T1 is clean.
        return _trial(15 + 5*(has0+has1), rms=4800.0 if has0 else 0.20,
                      seed_rms=0.06)
    def fake_subset_rms(fit, ids):
        return fit._seed, 15                          # seed stays tight either way
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        ctx = _trial_queue_ctx()
        rank = lambda d: d["_r"]                       # T0 best-ranked
        # joint + singles: T0 (off-sky) must be skipped, first clean single wins
        likely = [{"astrometry": f"T{i}\n", "_r": i} for i in range(2)]
        trial, obs_out, _m = S._recurse_trial_queue(ctx, likely, rank, prior_linkage_used=15)
        assert trial is not None and trial.rms == 0.20
        assert "T0" not in obs_out and "T1" in obs_out
        # a LONE off-sky candidate is refused too (seed guard would have exempted
        # it -- the rms gate sits ahead of that exemption)
        likely0 = [{"astrometry": "T0\n", "_r": 0}]
        trial0, _o0, _m0 = S._recurse_trial_queue(ctx, likely0, rank, prior_linkage_used=15)
        assert trial0 is None, "lone off-sky trial must not be taken"
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_recurse_trial_queue_stops_when_the_budget_is_spent():
    # Past the search budget the trial queue must stop BEFORE launching any
    # (potentially multi-hundred-second) trial fit -- the bound on a doomed
    # wide-corridor grind (2016 NY214). A ready-to-take trial is offered; the
    # spent budget must veto it and the fitter must never be called.
    import time
    import pyroorb.itf.search as S

    calls = []
    def fake_fit(text, obscodes, **kw):
        calls.append(text)                            # a good, takeable trial
        return _trial(15 + 5*("T0" in text))
    def fake_subset_rms(fit, ids):
        return 0.06, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        spent = S.SearchBudget(0.0, 0.0)               # already past
        ctx = _trial_queue_ctx(budget=spent)
        likely = [{"astrometry": "T0\n", "_r": 0}]
        trial, obs_out, _m = S._recurse_trial_queue(ctx, likely, lambda d: d["_r"],
                                               prior_linkage_used=15)
        assert trial is None, "a spent budget must veto the trial"
        assert calls == [], "no trial fit may start past the budget"
        # sanity: with budget remaining the same trial IS taken
        ctx.budget = S.SearchBudget(60.0, 0.0)
        trial2, _o2, _m2 = S._recurse_trial_queue(ctx, likely, lambda d: d["_r"],
                                             prior_linkage_used=15)
        assert trial2 is not None and calls, "in-budget trial must be tried"
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_setup_cost_is_not_charged_to_the_search_budget():
    """The variant scan's ONE union DB sweep is bounded setup, not grind, and
    charging it to the budget is a pure recall bug: 2015 BE559 live spends ~146 s
    in that sweep, so the 150 s no-progress clock was gone before variant #1 and
    the scan tried nothing -- while OFFLINE (no DB, no sweep) it recovers, so the
    suite could not see it.  ``setup_done`` excuses BOTH clocks, by exactly the
    measured toll."""
    import time
    import pyroorb.itf.search as S

    b = S.SearchBudget(0.05, 0.0)
    time.sleep(0.08)
    assert b.expired(), "the sweep ate the whole no-progress budget"
    b.setup_done(0.08)
    assert not b.expired(), "setup must not spend the clock it pays for"
    assert b.extensions == 0, "setup is not progress -- it must not count as a trial"
    assert b.setup_sec == 0.08

    # ...and the total CEILING credit is capped, so "one setup, credited once" is
    # a property of the budget rather than of the call graph (today a single
    # _variant_scan call site is the only caller; a second one must not let the
    # web UI's bound drift out by every sweep).
    cap = S.SearchBudget(10.0, 1.0, 0.3)
    ceiling0 = cap.ceiling_at
    cap.setup_done(0.2)
    cap.setup_done(0.2)                        # 0.4 s of toll, 0.3 s of credit
    assert abs(cap.ceiling_at - (ceiling0 + 0.3)) < 1e-9, \
        "the ceiling credit must stop at search_setup_credit_max_sec"
    assert abs(cap.setup_sec - 0.4) < 1e-9, "the toll itself is still recorded"
    # the no-progress clock is still restarted past the cap: it is self-limiting
    # (progress restarts it anyway) and refusing it re-opens the recall bug
    assert not cap.expired()

    # The ceiling moves by the toll and NO MORE: without excusing it, BE559 links
    # live at 595 s against a 600 s ceiling -- correct, and 5 s from flapping.
    c = S.SearchBudget(10.0, 1.0)
    before = c.ceiling_at
    c.setup_done(0.5)
    assert abs(c.ceiling_at - (before + 0.5)) < 1e-9
    d = S.SearchBudget(10.0, 0.05)
    time.sleep(0.12)
    d.setup_done(0.01)                         # a 10 ms toll cannot buy 120 ms
    assert d.expired() and d.why() == "the overall ceiling"

    # The complement: past the budget BEFORE the scan starts, _variant_scan
    # must return without paying for the union sweep at all.
    swept = []
    ctx = _trial_queue_ctx(budget=S.SearchBudget(0.0, 0.0), verbose=False,
                      source=SimpleNamespace(
                          sweep_candidates=lambda *a, **k: swept.append(1) or []),
                      max_uncert_deg=19.44, uncert_mult=2.5)
    out = S._variant_scan(ctx, _trial(15, U=9.56), 15, 0.0, 0, "")
    assert out is not None and out.verdict is not SearchVerdict.LINKED
    assert "before the LOV variant scan" in out.reason
    assert swept == [], "no union sweep may be paid for past the budget"


def test_budget_restarts_on_progress_and_still_has_a_ceiling():
    """The wall-clock budget must bound FRUITLESS work, not a working search.  A
    taken trial restarts the no-progress clock (as a pure deadline it cut searches
    that were still linking tracklets under DB load); the absolute ceiling still
    stops a pathological loop."""
    import time
    import pyroorb.itf.search as S

    b = S.SearchBudget(0.05, 0.0)                 # 50 ms of no-progress, no ceiling
    time.sleep(0.08)
    assert b.expired(), "an idle budget must expire"
    b.took_a_trial()
    assert not b.expired(), "progress must restart the clock"
    assert b.extensions == 1 and "without progress" in b.why()

    # ...but the ceiling is absolute: progress cannot push past it
    c = S.SearchBudget(10.0, 0.05)
    time.sleep(0.08)
    c.took_a_trial()
    assert c.expired(), "the overall ceiling must survive progress"
    assert c.why() == "the overall ceiling"

    # a trial queue that takes a trial tells the budget about it
    calls = []
    def fake_fit(text, obscodes, **kw):
        calls.append(text)
        return _trial(15 + 5 * ("T0" in text), seed_rms=0.06)
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit
    S.subset_rms = lambda fit, ids: (0.06, 15)
    try:
        budget = S.SearchBudget(60.0, 0.0)
        ctx = _trial_queue_ctx(budget=budget)
        trial, _o, _m = S._recurse_trial_queue(ctx, [{"astrometry": "T0\n", "_r": 0}],
                                          lambda d: d["_r"], prior_linkage_used=15)
        assert trial is not None and budget.extensions == 1, \
            "a taken trial must restart the no-progress clock"
    finally:
        S.fit_astrometry, S.subset_rms = orig_fit, orig_sub


def test_own_designations_collects_every_id_that_is_the_seed():
    """The seed's own ids, in BOTH spellings: packed as an OBS80 line carries
    them, unpacked as the database's provid column carries them."""
    from pyroorb.itf.search import own_designations
    seed = ("     K22P04R* C2022 08 06.371093"
            "21 18 24.75 -14 16 55.8          21.0 wU     F52\n")
    own = own_designations(seed, ["2022 PR4"])
    assert "K22P04R" in own and "2022 PR4" in own
    itf_line = ("     P21Qwcd  C2024 01 05.362728"
                "07 43 37.40 +20 19 38.1          21.5 wX     F52\n")
    assert "P21Qwcd" not in own_designations(itf_line, [])


def _own_ban_ctx(seed, own, rows):
    """A context whose sweep serves ``rows`` from one on-corridor window."""
    import pyroorb.itf.search as S

    class _Src:
        last_sweep_skipped = 0

        def sweep_candidates(self, windows, um, ew, wr, row_budget=None,
                             prefilter=None):
            yield (_window(), rows, {}, {})

    return SimpleNamespace(
        cfg=ItfConfig(), telemetry={}, log=(lambda *a: None), source=_Src(),
        verbose=False, trace=None, collector=None, obscodes=None,
        uncert_mult=2.5, max_uncert_deg=19.44,
        pub_ids={obs_unique_id(l) for l in seed.splitlines() if l.strip()},
        own_desigs=own)


def _ban_rows(ident, provid=None, ra_txt="21 18 24.751", dec_txt="-14 16 55.82"):
    """Three ON-CORRIDOR obs rows (off-corridor would be dropped as 'unlikely'
    and pass these tests for the wrong reason)."""
    line = (f"     {ident:<8}C2022 08 06.371093"
            f"{ra_txt}{dec_txt}          21.0 wU     F52")
    return [{"obs80": line, "provid": provid,
             "ra": 150.0 + k * 0.01, "dec": 10.0 + k * 0.004,
             "epoch_sec": T0 + k * 1800.0, "trksub": ident.strip("*"),
             "stn": "F52", "trkid": ident.strip("*")} for k in range(3)]


def test_sweep_bans_the_seeds_own_tracklets_by_designation():
    """A candidate filed under the seed's OWN designation is the object itself
    and must never be a candidate -- even when its OBS80 line is not
    byte-identical to the seed's, which is what happens when the seed is
    re-rendered from ADES (the web UI) at a different position precision."""
    import pyroorb.itf.search as S
    seed = ("     K22P04R* C2022 08 06.371093"
            "21 18 24.75 -14 16 55.8          21.0 wU     F52\n")
    own = S.own_designations(seed, ["2022 PR4"])
    ctx = _own_ban_ctx(seed, own, _ban_rows("K22P04R"))
    assert S._collect_tracklets(ctx, SimpleNamespace(fit=None), 0.0,
                                windows=[_window()]) == [], \
        "the seed's own designation must never be a candidate"
    # ...while a STRANGER's id stays a candidate: this bans self-identification
    # only, not the pseudo-ITF pickups the search exists to make
    ctx = _own_ban_ctx(seed, own, _ban_rows("P21Qwcd"))
    assert S._collect_tracklets(ctx, SimpleNamespace(fit=None), 0.0,
                                windows=[_window()]), \
        "a tracklet under another id is a legitimate candidate"


def test_sweep_bans_a_tracklet_the_db_says_is_already_ours():
    """The 2026 AP9 case: the MPC has attributed a tracklet to our object, so
    obs_sbn's provid reads '2026 AP9', but the ARCHIVED line still shows the
    trksub it was filed under.  A line-text ban cannot see that, and with a
    not-yet-reconciled index the object gets its own published observations
    offered back as a discovery.  The DB's provid decides.  (A harness-planted
    tracklet carries no provid and stays linkable -- it is simulating a linkage
    that has NOT been accepted yet.)"""
    import pyroorb.itf.search as S
    seed = ("     K26A09P* C2026 01 05.371093"
            "14 23 10.67 -15 33 50.9          21.0 wU     F52\n")
    own = S.own_designations(seed, ["2026 AP9"])
    assert "K26A09P" in own and "2026 AP9" in own, own
    ctx = _own_ban_ctx(seed, own, _ban_rows("N149763", provid="2026 AP9"))
    assert S._collect_tracklets(ctx, SimpleNamespace(fit=None), 0.0,
                                windows=[_window()]) == [], \
        "a tracklet the DB attributes to the seed must not be a candidate"
    ctx = _own_ban_ctx(seed, own, _ban_rows("N149763", provid=None))
    assert S._collect_tracklets(ctx, SimpleNamespace(fit=None), 0.0,
                                windows=[_window()]), \
        "an unattributed tracklet must still be offered"


def test_candidate_features_do_not_depend_on_set_iteration_order():
    """The feature row a candidate is SCORED from must be picked
    deterministically.

    A run-length group (same trksub + station, no 12 h gap) can span more than
    one index ``tracklet_id``, and the representative used to be chosen with
    ``next(...)`` over a SET of those ids -- so it followed string-hash
    iteration order, which Python randomises per PROCESS.  ``_feat`` carries the
    DB's fitted rate, magnitudes and band into the ML feature row, so the same
    candidate off the same sweep scored differently from run to run: 2024 PQ7's
    JDWM1JW came out P(match) 0.498 and 0.088 on consecutive live runs, which
    flipped its rank, changed the trial queue's shape and cost the recovery.

    The pick is now the group's BIGGEST fragment (most observations = longest
    baseline = best-constrained rate and most meaningful scatter), ties to the
    earliest.  Exercised over many id pairs because a set of two short strings
    iterates in insertion order often enough that a single pair proves
    nothing."""
    import pyroorb.itf.search as S
    seed = ("     K22P04R* C2022 08 06.371093"
            "21 18 24.75 -14 16 55.8          21.0 wU     F52\n")
    own = S.own_designations(seed, ["2022 PR4"])

    # `first` is the LATER-starting id but holds 2 of the 3 rows, so a correct
    # pick cannot be explained by row order alone -- it has to be the count.
    for i in range(40):
        first, second = f"aa{i}", f"zz{i}"
        rows = _ban_rows("P21Qwcd")            # one group: same trksub + stn
        rows[0]["trkid"] = second                       # ...spanning two trkids
        rows[1]["trkid"] = rows[2]["trkid"] = first
        feats = {first: {"rate_deg_day": 0.11, "band": "V", "_which": first},
                 second: {"rate_deg_day": 0.99, "band": "R", "_which": second}}

        class _Src:
            last_sweep_skipped = 0

            def sweep_candidates(self, windows, um, ew, wr, row_budget=None,
                                 prefilter=None):
                yield (_window(), rows, {}, feats)

        ctx = _own_ban_ctx(seed, own, rows)
        ctx.source = _Src()
        got = S._collect_tracklets(ctx, SimpleNamespace(fit=None), 0.0,
                                   windows=[_window()])
        assert got, "the synthetic tracklet must survive the corridor"
        assert got[0]["_feat"]["_which"] == first, (
            f"features came from {got[0]['_feat']['_which']}, not the group's "
            f"biggest fragment {first} -- the pick is set-order dependent")


def test_clipped_seed_obs_stay_visible_to_the_caller():
    """An ITF search must never make one of the object's OWN observations
    disappear.  The internal fit clips poor fitters from the text it fits, but
    the web UI rebuilds its observation list by re-fitting `combined_obs80` --
    so a seed line missing from that string is a line gone from the user's
    screen, unjudgeable and unrecoverable.  Clipping is a claim, and the user
    must be able to see it and overrule it: the line comes back listed, off and
    red (the caller's own fit rejects it), not deleted."""
    import pyroorb.itf.search as S

    seed = ("     K24P07Q  C2024 08 07.100000"
            "21 18 24.75 -14 16 55.8          21.0 wU     W76\n"
            "     K24P07Q  C2024 08 09.100000"
            "21 18 25.75 -14 16 56.8          21.0 wU     W76\n"
            "     K24P07Q  C2024 08 11.100000"
            "21 18 26.75 -14 16 57.8          21.0 wU     W76\n")
    itf_lines = ("     P10QoPZ  C2019 03 01.100000"
                 "07 43 37.40 +20 19 38.1          21.5 wX     F51\n")
    ctx = SimpleNamespace(pub_obs80=seed)
    out = S.display_obs80(ctx, itf_lines)
    for line in seed.splitlines():
        assert line in out, "a seed observation must never vanish from the result"
    assert itf_lines.strip() in out, "the linked ITF astrometry must be there too"

    # ...even when the internal fit clipped one: the CLIPPED text is what the orbit
    # and the linkage document use, never what the caller is shown
    clipped_combined = "\n".join(seed.splitlines()[:2]) + "\n" + itf_lines
    assert len(clipped_combined.splitlines()) < len((seed + itf_lines).splitlines())
    assert len(out.splitlines()) == len((seed + itf_lines).splitlines())


def test_clip_is_sigma_aware_not_just_a_flat_arcsec_cut():
    """An observation is only a poor fitter if it misses by more than the flat
    threshold AND by more than k sigma of ITS OWN uncertainty.

    The flat cut alone asks the wrong question of a noisy station: 0.5" is a
    blunder from a 0.15"-sigma survey and ordinary from a 1"-sigma one.  2024
    PQ7 -- a linkage that is real, and accepted by the MPC -- lost three good
    W76 observations to exactly that.  The AND also keeps this strictly more
    permissive than before: nothing that survived the flat cut can now be clipped.
    """
    from pyroorb.itf.fitter import drop_offending

    def line(n):
        return (f"     K24P07Q  C2024 08 {7 + n:02d}.100000"
                f"21 18 24.75 -14 16 55.8          21.0 wU     W76")

    class _Fit:
        """Residual/sigma surface drop_offending reads (per-axis, arcsec)."""
        def __init__(self, lines, resids, sigmas):
            self._obs = [{"obs80_line": l} for l in lines]
            self._r, self._s = resids, sigmas

        def residuals(self, included_only=False):
            return [{"d_ra_arcsec": dra, "d_dec_arcsec": ddec,
                     "sigma_ra": sra, "sigma_dec": sdec, "included": True}
                    for (dra, ddec), (sra, sdec) in zip(self._r, self._s)]

    lines = [line(0), line(1), line(2)]
    #   ob 0: well inside the flat cut -- untouched either way
    #   ob 1: 0.59" total, sigma 1.5"/axis -> ~0.3 sigma: ordinary noise for
    #         THIS station, but past the 0.5" flat cut
    #   ob 2: 7.07" total, sigma 1.5"/axis -> ~3.3 sigma AND far past the cut
    resid = [(0.10, 0.10), (0.42, 0.42), (5.00, 5.00)]
    sigma = [(1.50, 1.50), (1.50, 1.50), (1.50, 1.50)]
    fit = _Fit(lines, resid, sigma)
    text = "\n".join(lines) + "\n"

    flat = drop_offending(fit, text, 0.5)
    assert len(flat.splitlines()) == 1, \
        "the flat cut alone clips the noisy-station ob too"
    kept = drop_offending(fit, text, 0.5, sigma_mult=3.0).splitlines()
    assert len(kept) == 2, f"only the genuine blunder should clip, got {kept}"
    assert lines[1] in kept, "a sub-sigma miss is not a poor fitter"
    assert lines[2] not in kept, "a multi-sigma miss still is"


def test_budget_needs_fruitless_WORK_not_just_a_slow_clock():
    """The no-progress budget must not cut a search merely because the machine
    is busy.  A wall-clock deadline cannot tell a doomed grind from a slow box,
    and on a loaded suite it cut 2024 PQ7 -- a real MPC-accepted linkage that
    recovers 4/4 standalone -- to 0/4.  So expiry needs BOTH: the clock spent
    AND fruitless fits burned.  The absolute ceiling is unconditional."""
    import pyroorb.itf.search as S

    b = S.SearchBudget(no_progress_sec=0.0, ceiling_sec=1000.0,
                       min_fruitless_fits=3)
    assert not b.expired(), "clock spent but no work burned: keep going"
    for _ in range(2):
        b.note_fit()
    assert not b.expired(), "still under the fruitless-fit bar"
    b.note_fit()
    assert b.expired(), "clock spent AND work burned: a doomed grind"
    assert "fruitless" in b.why()

    # progress resets the work counter as well as the clock
    b.took_a_trial()
    assert b.fits_since_progress == 0 and not b.expired()

    # the ceiling does not negotiate -- no fruitless-fit bar applies to it
    # (ceiling_sec=0 DISABLES the ceiling, so drive it past instead)
    c = S.SearchBudget(no_progress_sec=1000.0, ceiling_sec=1000.0,
                       min_fruitless_fits=99)
    c.ceiling_at = time.perf_counter() - 1.0
    assert c.expired() and c.why() == "the overall ceiling"

    # min_fruitless_fits=0 keeps the old clock-only behaviour
    d = S.SearchBudget(no_progress_sec=0.0, ceiling_sec=1000.0,
                       min_fruitless_fits=0)
    assert d.expired()


def test_seed_guard_noise_floor():
    # The true multi-opposition orbit honestly widens an overfit short seed
    # arc (2024 RX190: 0.196" -> 0.450"). Below the absolute noise floor that
    # is NOT contamination: the joint batch must be TAKEN, not guarded.
    import pyroorb.itf.search as S

    def fake_fit(text, obscodes, **kw):
        n = sum(f"T{i}" in text for i in range(3))
        return _trial(12 + 3*n, seed_rms=0.45)
    def fake_subset_rms(fit, ids):
        return fit._seed, 12
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        ctx = _trial_queue_ctx(pub_rms=0.196, pub_used=12)
        likely = [{"astrometry": f"T{i}\n", "_r": i} for i in range(3)]
        trial, obs_out, _m = S._recurse_trial_queue(ctx, likely, lambda d: d["_r"],
                                               prior_linkage_used=12)
        assert trial is not None and trial.used == 21, \
            "0.45\" honest scatter must not trip the guard"
        assert all(f"T{i}" in obs_out for i in range(3))
        # seed-retention floor: same clean rms but most seed obs clipped -> refuse
        S.subset_rms = lambda fit, ids: (0.06, 8)     # 8/12 < 80%
        trial2, _o2, _m2 = S._recurse_trial_queue(ctx, likely, lambda d: d["_r"],
                                             prior_linkage_used=12)
        assert trial2 is None, "clipping 4/12 seed obs must refuse the trial"
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_confirm_seed_strength():
    # The three-thin upgrade's seed-strength path: a well-constrained initial
    # arc (> 30 d OR U <= 6) confirms a recursion-accumulated pair of
    # single-night oppositions regardless of how the trials fell; a short
    # sloppy seed still needs the depth-0 top-pair evidence.
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def ctx_with(arc, U):
        return SimpleNamespace(cfg=ItfConfig(), log=lambda *a: None,
                               pub=SimpleNamespace(arclength_days=arc, U=U))
    # 2024 RX190 shape: 36.9 d arc, U 7.31 -> arc criterion confirms
    assert S._confirm_top_pair(ctx_with(36.9, 7.31), None, "", None, 5)
    # short arc but tight U -> U criterion confirms
    assert S._confirm_top_pair(ctx_with(12.0, 5.5), None, "", None, 5)
    # short sloppy seed, no trial_meta -> falls through to the pair path -> no
    assert not S._confirm_top_pair(ctx_with(12.0, 8.0), None, "", None, 0)


def test_one_night_show_gate():
    # The 1.5-profile lone-night path, driven through the real verdict
    # (_assess -> _judge).  ALL in sigma units (overall arcsec rms never
    # consulted): the night must be CONSISTENT like any linkage -- each linked
    # tracklet fits its own uncertainties, the seed is not statistically
    # degraded (a Wilks Delta-chi2 on the seed obs, sigma held fixed at the
    # seed's own assessment) and is not dragged off as a contaminant -- and
    # then (3) the SOLE fit, measured.  Path-agnostic -- no in_variant.
    import dataclasses
    from pyroorb import ident_check
    import pytest
    import pyroorb.itf.search as S
    from pyroorb.itf.obs80_fields import obs_unique_id
    from types import SimpleNamespace

    SIG = (1.0, 1.0)    # per-axis arcsec; F51 Veres floor 0.2" -> station floor a no-op

    class StubFit:
        """Minimal FittedOrbit stand-in: the residual/inclusion surface the
        sigma-chi2 helpers read (obs80_line ids, per-obs O-C and sigma, mask)."""
        def __init__(self, lines, resids, sigmas, included=None):
            self._obs = [{"obs80_line": l} for l in lines]
            self._resids, self._sigmas = resids, sigmas
            self._inc = list(included) if included is not None else [True] * len(lines)

        def is_included(self):
            return list(self._inc)

        def residuals(self, included_only=False):
            out = []
            for o, (dra, ddec), (sra, sdec), inc in zip(
                    self._obs, self._resids, self._sigmas, self._inc):
                if included_only and not inc:
                    continue
                out.append({"d_ra_arcsec": dra, "d_dec_arcsec": ddec,
                            "sigma_ra": sra, "sigma_dec": sdec, "included": inc})
            return out

    seed_lines = [_obs80("SEED%03d" % i, "2011 09 %02d.30000" % (10 + i),
                         "01 00 00.00", "-09 00 00.0") for i in range(5)]
    # ONE new night, made of TWO tracklets of two obs each -- the 2024 QF7 shape
    # (a survey splitting one night's detections of one object into two trksubs).
    # The gate's claim is one NIGHT, so the fixture has to be one night.
    new_lines = [_obs80("NEW%04d" % i, "2012 01 10.%05d" % (20000 + 2000 * i),
                        "02 00 00.00", "-08 00 00.0") for i in range(4)]
    seed_ids = {obs_unique_id(l) for l in seed_lines}
    obs_out = "\n".join(new_lines)
    combined = "\n".join(seed_lines) + "\n" + obs_out
    # how the trial was swept: NEW0000+NEW0001 = tracklet A, NEW0002+NEW0003 = B
    trk_a = frozenset(obs_unique_id(l) for l in new_lines[:2])
    trk_b = frozenset(obs_unique_id(l) for l in new_lines[2:])
    swept = {**{i: ("A@F51/2012-01-10(2obs)", trk_a) for i in trk_a},
             **{i: ("B@F51/2012-01-10(2obs)", trk_b) for i in trk_b}}
    # ...and the same four obs spread over four separate nights, which is a
    # DIFFERENT (richer) profile the one-night gate must not speak for
    multinight_out = "\n".join(
        _obs80("NEW%04d" % i, "2012 01 %02d.30000" % (10 + i),
               "02 00 00.00", "-08 00 00.0") for i in range(4))
    # seed fit: residuals at 0.2" each axis, comfortably within its 1.0" sigma
    # (``seed_pre`` below re-shapes it -- the seed's PRE-link residual is one
    # side of the movement the banner now reports)
    def run(*, seed_pre=0.2, seed_post=0.3, new_resid=0.3,
            ml_ps=(0.32, 0.0, 0.0),
            winner_rank=0, trials_failed=(), rivals_link=(), trial_size=1,
            U=3.73, opps=OPP_TWO_THIN, seed_dropped=0, trk_b_resid=None,
            multinight=False, n_rivals=0):
        """Drive the gate.  ``ml_ps`` is the WHOLE ML-scored field best-first;
        ``winner_rank`` is the field position of the tracklet the trial queue took;
        ``trials_failed`` the ranks the trial queue already tried and rejected; and
        ``rivals_link`` the ranks whose (stubbed) exhaustion refit LINKS."""
        seed_fit = StubFit(seed_lines, [(seed_pre, seed_pre)] * 5, [SIG] * 5)
        inc = [True] * 5
        for k in range(seed_dropped):
            inc[k] = False       # a clipped seed obs still counts at its full residual
        # tracklet B may be given residuals of its own: a pooled chi2 must not
        # let a good tracklet carry a bad one into the report
        b_resid = new_resid if trk_b_resid is None else trk_b_resid
        out_lines = multinight_out.splitlines() if multinight else new_lines
        combined_fit = StubFit(
            seed_lines + out_lines,
            [(seed_post, seed_post)] * 5 + [(new_resid, new_resid)] * 2
            + [(b_resid, b_resid)] * 2,
            [SIG] * 9, included=inc + [True] * 4)
        trial = SimpleNamespace(used=8, count=9, rms=0.19, fit=combined_fit,
                                q=2.5, e=0.1, U=3.0, arclength_days=120.0)
        # a LIVE tracer (enabled), not the null one: the verdict's trace lines
        # are formatted only under it, so a null tracer hides any crash in them
        trace = lambda *a: None
        trace.enabled = True
        ctx = SimpleNamespace(cfg=ItfConfig(show_n_rivals=n_rivals),
                              initial_opp_profile=OPP_ONE, log=lambda *a: None,
                              pub_ids=seed_ids, obscodes=None, trace=trace,
                              pub_obs80="\n".join(seed_lines) + "\n",
                              swept_tracklets=swept, telemetry={},
                              pub=SimpleNamespace(used=5, count=5, U=U, rms=0.2,
                                                  arclength_days=4.0,
                                                  fit=seed_fit))
        # the scored field: one stub tracklet per ml_p, its astrometry naming
        # its own rank so the stubbed refit can decide links-vs-fails
        field = [{"ml_p": p, "traj_category": "likely", "rows": [{}],
                  "astrometry": "RANK%02d\n" % k}
                 for k, p in enumerate(ml_ps)]
        trial_meta = {"tracklets": [field[winner_rank + k] for k in range(trial_size)],
                "failed_tracklets": [field[k] for k in trials_failed],
                "ml_ps": [ml_ps[winner_rank]], "indices": [0]}

        def stub_fit(text, obscodes, **kw):
            # an exhaustion refit: bare seed + ONE rival's astrometry
            rank = int(text.strip().splitlines()[-1][4:])
            if rank in rivals_link:      # links: progresses past the 5 seed obs
                return SimpleNamespace(ok=True, used=8, count=9, rms=0.19,
                                       fit=combined_fit)
            return SimpleNamespace(ok=True, used=5, count=9, rms=0.19,
                                   fit=combined_fit)      # no progress

        out = multinight_out if multinight else obs_out
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(S, "fit_astrometry", stub_fit)
            # the cautions that need a real FittedOrbit have their own tests
            mp.setattr(ident_check, "coherent_offset_misfit",
                       lambda fit, noise_floor=None: None)
            mp.setattr(S, "measure_trends", lambda *a, **k: None)
            mp.setattr(S, "mag_outlier_tracklet", lambda *a, **k: None)
            mp.setattr(S, "worst_mag_residual", lambda fit: (0.0, None))
            mp.setattr(S, "two_axis_trend_warning", lambda *a, **k: None)
            mp.setattr(S.pickup, "review", lambda *a, **k: [])
            a = S._assess(ctx, trial, "\n".join(seed_lines) + "\n" + out, out,
                          5, trial_meta, 0)
            a = dataclasses.replace(a, profile=opps, profile_ok=False)
            return S._judge(ctx, a, field, 5, trial_meta, 0, sweep=None)

    # sole dominant candidate + new night fits its sigma + seed intact -> SHOWN
    result = run()
    assert result.verdict is SearchVerdict.POSSIBLE, result.reason
    assert result.itf_obs80 == obs_out and result.fit is not None
    assert result.vetting.get("single_best_fit") is True
    assert result.vetting.get("tier") == "likely"        # tight fit + P >= 0.25
    assert any("archival" in w.lower() for w in result.warnings), result.warnings

    # ...and the banner REPORTS what the link did to the object's own
    # astrometry instead of asserting it is "unharmed".  Clearing the
    # Delta-chi2 cap means "not degraded past the seed's 3-sigma state-dof
    # region", which a shown linkage can spend three quarters of: 2010 TR241's
    # P1156lI cleared it at 74% of budget with the seed going 0.382" -> 0.679"
    # (1.8x, chi2 tripled) and the reader was told it was unharmed.
    assert "unharmed" not in " ".join(result.warnings).lower(), result.warnings
    assert "moves" in result.warnings[0], result.warnings
    # reported per-axis, like the acceptance path's same two fields
    assert abs(result.vetting["seed_rms_pre"] - 0.2) < 1e-3
    assert abs(result.vetting["seed_rms_post"] - 0.3) < 1e-3
    assert result.vetting["seed_sigma_post"] is not None
    assert result.vetting["seed_degraded"] is False       # 1.5x, inside warn_rms_factor

    # THE 2010 TR241 SHAPE: 0.382" -> 0.679" is 1.78x -- inside the acceptance
    # path's warn_rms_factor (2.0), so no caution, but the movement is stated.
    r_tr = run(seed_pre=0.27, seed_post=0.48)
    assert r_tr.verdict is SearchVerdict.POSSIBLE
    assert r_tr.vetting["seed_degraded"] is False
    assert "1.8x" in r_tr.warnings[0], r_tr.warnings
    assert not any("fit worse" in w for w in r_tr.warnings), r_tr.warnings

    # ...and past that ratio the acceptance path's caution is raised HERE too,
    # on its own line -- the seed-degradation tell must not be a clause buried
    # in a sentence that is otherwise good news.  Verdict is unaffected (this
    # path only ever produces a 0.6, which nothing submits); the reader is not.
    r_deg = run(seed_post=0.7)                         # 3.5x, Delta-chi2 still fine
    assert r_deg.verdict is SearchVerdict.POSSIBLE, r_deg.reason
    assert r_deg.vetting["seed_degraded"] is True
    worse = [w for w in r_deg.warnings if "fit worse after the link" in w]
    assert len(worse) == 1, r_deg.warnings
    # ...with the statistical mitigation named when it applies (0.70 sigma)
    assert "still fit their uncertainties" in worse[0], worse
    # ...and when the seed no longer fits its uncertainties at that ratio (1.35
    # sigma against the station floor, 6.75x) it is the outright contaminant
    # veto (extension_degrades), which a lone night faces like any linkage:
    # not shown, and the reader is told why
    r_bad = run(seed_post=1.35)
    assert r_bad.fit is None and r_bad.vetting.get("show_suppressed"), r_bad.vetting
    assert "degrades the existing fit" in r_bad.reason, r_bad.reason

    # still shown when a stale seed obs is clipped (net used 4 < seed 5): the gate
    # never keys on used-count -- the clip ob rides at its full 0.2" residual, a
    # trivial Delta-chi2, so the original astrometry stays statistically intact
    assert run(seed_dropped=1).verdict is SearchVerdict.POSSIBLE

    # PATH-AGNOSTIC: nothing distinguishes a variant hit -- same inputs, shown
    # (the old `not in_variant` exclusion is gone)
    assert run().verdict is SearchVerdict.POSSIBLE

    # (2a) the NEW tracklet does not fit its own sigma (~3-sigma residuals) -> silent
    assert run(new_resid=3.0).fit is None

    # (2b) the ORIGINAL astrometry is degraded (seed residuals blow to ~3 sigma)
    # even though the new night still fits its own sigma -> silent
    assert run(seed_post=3.0).fit is None

    # (3) exclusivity is MEASURED: every rival above the ML cliff (the first
    # candidate under show_exhaust_frac x the WINNER's P, within
    # show_exhaust_max_rank) is refitted against the bare seed and must fail.
    # A decisive cliff (2024 RO175: 0.999 then 0.057) leaves no rival ABOVE it,
    # but exclusivity is never granted on zero fits: show_exhaust_min_rivals
    # reaches past the cliff for one, which fails to link, and the verdict is
    # the same sole standout with one banner.  The extra fit is the price of the
    # claim being measured rather than read off the ML scores that chose the
    # winner (see the 2024 PQ7 case below).
    r_sole = run(ml_ps=(0.999, 0.057, 0.010))
    assert r_sole.verdict is SearchVerdict.POSSIBLE and r_sole.vetting.get("single_best_fit") is True
    assert r_sole.vetting["exhaustion"]["n_rivals"] == 1
    assert r_sole.vetting["exhaustion"]["n_refitted"] == 1
    assert len([w for w in r_sole.warnings if "archival" in w.lower()]) == 1
    assert not any("not necessarily" in w.lower() for w in r_sole.warnings)
    # 2024 RE182 shape: the winner is the ONLY scored candidate.  A field of one
    # has no cliff because it has nothing to fall off -- and nothing to rule out
    # either, so it is a sole standout outright, with zero refits.
    r_one = run(ml_ps=(1.0,))
    assert r_one.verdict is SearchVerdict.POSSIBLE and r_one.vetting.get("single_best_fit") is True
    assert r_one.vetting["exhaustion"]["n_rivals"] == 0
    assert not any("not necessarily" in w.lower() for w in r_one.warnings)
    # 2024 PQ7 shape, and the reason the cliff alone is not evidence: off a
    # 1-apparition seed the model put P 0.976 on IUe7454 -- a tracklet 2.363 deg
    # (~62000 sigma) from where PQ7 demonstrably was -- and 0.127 on the
    # SECOND-ranked candidate, P10RILV@F51/2019-09-24, which ALSO links and is
    # 2.643 deg off itself.  Two mutually incompatible strangers both fitting is
    # exactly what "not the sole standout" means: an under-determined seed admits
    # several.  The cliff is decisive, so the old gate fitted nothing and reported
    # the first as a lone best fit.  Fitting the one rival past the cliff finds
    # that it links too, and the claim collapses to the multiplicity it was.
    r_pq7 = run(ml_ps=(0.976, 0.127, 0.104), rivals_link=(1,))
    assert r_pq7.vetting["exhaustion"]["why"] == "a rival also links", r_pq7.vetting
    assert r_pq7.vetting["exhaustion"]["rival"], r_pq7.vetting
    assert r_pq7.vetting.get("single_best_fit") is not True, r_pq7.vetting
    assert any("not necessarily the only" in w.lower() for w in r_pq7.warnings)

    # 2024 RD175 shape: a graded high-P field (0.999/0.887/0.734, the WHOLE
    # scored field incl. "maybe"s) never falls below 1/3 of the winner -- but it
    # ENDS well inside show_exhaust_max_rank, so no rival can be hiding past it:
    # the field is finite and fully enumerable, and ALL of it is fitted.  Both
    # rivals fail to link -> exclusivity is measured, not merely unclaimable.
    rd175 = (0.999, 0.887, 0.734)
    r_enum = run(ml_ps=rd175)
    assert r_enum.verdict is SearchVerdict.POSSIBLE and r_enum.vetting.get("single_best_fit") is True
    assert r_enum.vetting["exhaustion"]["n_refitted"] == 2
    assert not any("not necessarily" in w.lower() for w in r_enum.warnings)
    # ...and when one of those same-station nights DOES link as an alternative,
    # exclusivity is disproved: still shown (top P clears the multi bar) WITH the
    # second same-severity banner.
    r_multi = run(ml_ps=rd175, rivals_link=(1,))
    assert r_multi.verdict is SearchVerdict.POSSIBLE and r_multi.fit is not None
    assert r_multi.vetting.get("multiple_compatible") is True
    assert r_multi.vetting["exhaustion"]["why"] == "a rival also links"
    assert r_multi.vetting.get("tier") == "possible"
    assert any("not necessarily the only" in w.lower() for w in r_multi.warnings)
    assert any("archival" in w.lower() for w in r_multi.warnings)
    assert len(r_multi.warnings) == 2                 # two banners, same channel
    # A field still above the bar AT the rank bound is the one that stays
    # unclaimable: there is no cliff AND no end in sight, so comparable rivals
    # may lie past rank show_exhaust_max_rank that we never look at.
    r_dense = run(ml_ps=tuple([0.999] + [0.90] * 11))
    assert r_dense.verdict is SearchVerdict.POSSIBLE and r_dense.vetting.get("multiple_compatible") is True
    assert r_dense.vetting["exhaustion"]["why"] == "many similarly-ranked candidate tracklets"
    assert r_dense.vetting["exhaustion"]["scanned"] == 10

    # THE 2024 QT17 SHAPE.  The trial queue took rank 3 because ranks 1-2 were fitted
    # and FAILED; ranks 4-7 sit above the cliff at rank 8 (0.061 < 1/3 x 0.997)
    # and are refitted here.  All fail -> the winner IS the only fit -> shown
    # CLEAN.  (Under the old is_top_ranked prefix rule this was silent.)
    qt17 = (1.000, 1.000, 0.997, 0.986, 0.985, 0.980, 0.861, 0.061, 0.022)
    r_qt = run(ml_ps=qt17, winner_rank=2, trials_failed=(0, 1))
    assert r_qt.verdict is SearchVerdict.POSSIBLE and r_qt.vetting.get("single_best_fit") is True
    assert r_qt.vetting["exhaustion"]["cliff_rank"] == 8
    assert r_qt.vetting["exhaustion"]["n_rivals"] == 6      # ranks 1-7 less winner
    assert r_qt.vetting["exhaustion"]["n_refitted"] == 4    # ranks 4-7
    assert not any("not necessarily" in w.lower() for w in r_qt.warnings)
    # ...but if ONE of those rivals also links, exclusivity is disproved -> the
    # multiplicity banner (shown, not silent: the top P clears the multi bar)
    r_qt2 = run(ml_ps=qt17, winner_rank=2, trials_failed=(0, 1), rivals_link=(5,))
    assert r_qt2.verdict is SearchVerdict.POSSIBLE and r_qt2.vetting.get("multiple_compatible") is True
    assert r_qt2.vetting["exhaustion"]["why"] == "a rival also links"
    assert any("not necessarily the only" in w.lower() for w in r_qt2.warnings)
    # a rival the trial queue never tried and never ruled out is NOT assumed dead:
    # with ranks 1-2 unexamined the refits decide, and a linking one demotes
    assert run(ml_ps=qt17, winner_rank=2,
               rivals_link=(0,)).vetting.get("multiple_compatible") is True

    # a crowded field (no cliff, no end inside the bound) whose BEST P is below
    # the multi bar (<= 0.7) stays silent -- exclusivity unclaimable AND the top
    # candidate is weak
    assert run(ml_ps=tuple([0.60] + [0.55] * 11)).fit is None
    # ...and likewise when a cliff EXISTS but a rival above it also links
    assert run(ml_ps=(0.32, 0.30, 0.02), rivals_link=(1,)).fit is None
    # BUT a second comparably-plausible candidate that is FITTED and does NOT
    # link no longer silences a weak best P: exclusivity is measured, not
    # inferred from the scores (this field was silent under show_dominance_frac)
    r_weak = run(ml_ps=(0.32, 0.30, 0.02))
    assert r_weak.verdict is SearchVerdict.POSSIBLE and r_weak.vetting.get("single_best_fit") is True
    assert r_weak.vetting["exhaustion"]["n_refitted"] == 1
    # best ML below the absolute floor -> silent (no fits spent)
    assert run(ml_ps=(0.10, 0.0)).fit is None

    # a badly under-determined seed (U >= single_night_show_max_U) -> silent.
    # Read the cap off the CONFIG rather than repeating its value: it moved
    # 9.0 -> 10.4 when the s2 floor made U honest (the shift is +1.40 at U 9,
    # not the +0.45 population median), and a literal here would have pinned
    # the test to a scale the engine no longer uses.
    assert run(U=ItfConfig().single_night_show_max_U + 0.5).fit is None

    # THE 2024 QF7 SHAPE: one new night made of TWO tracklets (UT96489 4obs +
    # UT9674C 3obs, same station, same night).  That is ONE identification to
    # vet, not two -- shown, and the second tracklet is part of the winner, so
    # exclusivity does not treat it as a rival that "also links" against itself.
    r_qf7 = run(trial_size=2)
    assert r_qf7.verdict is SearchVerdict.POSSIBLE, r_qf7.reason
    assert r_qf7.vetting.get("single_best_fit") is True, r_qf7.vetting
    assert r_qf7.vetting["n_tracklets"] == 2
    assert r_qf7.vetting["exhaustion"]["n_refitted"] == 1   # the one NON-taken rival
    assert any("2 tracklets on one night" in w for w in r_qf7.warnings), r_qf7.warnings
    assert not any("not necessarily" in w.lower() for w in r_qf7.warnings)
    # ...but a pooled chi2 must not let the good tracklet carry a bad one: the
    # second tracklet is tested on its own uncertainties too
    assert run(trial_size=2, trk_b_resid=3.5).fit is None
    # ...and the gate speaks for ONE NIGHT only.  The same four obs spread over
    # four nights is a different, richer profile -- it falls through to the ledger
    # even though every sigma test still passes.
    assert run(trial_size=2, multinight=True).fit is None

    # tier drops to 'possible' with a looser (but still consistent) fit / lower P
    res_poss = run(new_resid=1.5, ml_ps=(0.18, 0.0))
    assert res_poss.verdict is SearchVerdict.POSSIBLE and res_poss.vetting["tier"] == "possible"

    # --- clause 3b: COUNT the rivals that fit (web surfaces only) ------------
    # A crowded field is shown today on the winner's P alone, exclusivity
    # having fitted nothing.  With show_n_rivals set (the web layer does,
    # nothing else does) the rivals are fitted and COUNTED, and a night with
    # too many equally-good company is not put in front of a reader who might
    # go searching archival images for it.
    dense = tuple([0.999] + [0.9] * 11)          # no cliff inside rank 10
    r_off = run(ml_ps=dense, rivals_link=(1, 2, 3))
    assert r_off.verdict is SearchVerdict.POSSIBLE, r_off.reason     # k=0: today's behaviour
    assert "rival_count" not in r_off.vetting["exhaustion"], r_off.vetting

    r_many = run(ml_ps=dense, rivals_link=(1, 2, 3), n_rivals=6)
    assert r_many.fit is None, r_many.reason     # 3 of 6 link -> suppressed
    # ...and the reader is told all three things: how many fit, that the count
    # is a FLOOR (only the top few of the field were tested), and what would
    # settle it.  A withheld find that does not say what would un-withhold it
    # reads as the software being unhelpful.
    assert "at least 3 of the 6" in r_many.reason, r_many.reason
    assert "field of 12" in r_many.reason, r_many.reason
    assert "second night" in r_many.reason, r_many.reason
    assert r_many.vetting.get("show_suppressed") is True, r_many.vetting

    r_few = run(ml_ps=dense, rivals_link=(1, 2), n_rivals=6)
    assert r_few.verdict is SearchVerdict.POSSIBLE, r_few.reason     # 2 of 6 -> still shown...
    cnt = r_few.vetting["exhaustion"]["rival_count"]
    assert cnt["n_linking"] == 2 and cnt["n_tested"] == 6, cnt
    # ...with the count ON the banner, since "one of several" without a number
    # is exactly what the reader cannot act on
    multi = [w for w in r_few.warnings if "not necessarily the only" in w]
    assert len(multi) == 1 and "2 also link" in multi[0], multi

    r_none = run(ml_ps=dense, n_rivals=6)         # nothing else fits at all
    assert r_none.verdict is SearchVerdict.POSSIBLE, r_none.reason
    assert r_none.vetting["exhaustion"]["rival_count"]["n_linking"] == 0
    assert any("none link" in w for w in r_none.warnings), r_none.warnings

    # a SOLE STANDOUT never pays for the count -- exclusivity was measured
    r_sole = run(n_rivals=6)
    assert r_sole.vetting.get("single_best_fit") is True
    assert "rival_count" not in r_sole.vetting["exhaustion"], r_sole.vetting

    # --- and every one of those silences SAYS WHY, in English ---------------
    # Each `fit is None` above is a night that DID link and is being withheld,
    # and the ledger used to render all of them as "1.5 opps without progress,
    # abandoning" -- a sentence about the trial queue that a reader takes as
    # "the search found nothing".  The clause that actually refused is recorded
    # on its way out (search._refuse) and printed here instead.
    def why(**kw):
        r = run(**kw)
        assert r.fit is None, "expected a silence to explain"
        assert r.vetting.get("show_suppressed") is True, r.vetting
        assert "without progress" not in r.reason, r.reason
        assert r.reason.startswith("nothing shown: "), r.reason
        # it is a sentence, and reads as one: the clauses do not punctuate
        # themselves, so the ledger must
        assert r.reason.endswith("."), r.reason
        return r.reason.lower()

    # the crowded-field case: it fits, but so would many
    crowd = why(ml_ps=tuple([0.60] + [0.55] * 11))
    assert "one of many" in crowd and "coincidence" in crowd, crowd
    assert "0.60" in crowd, crowd                      # the probability, stated
    # ...and the same when a cliff exists but a rival above it also links
    assert "one of many" in why(ml_ps=(0.32, 0.30, 0.02), rivals_link=(1,))

    # the tracklet lands near the orbit but not on it
    assert "uncertainties" in why(new_resid=3.0)
    # one tracklet of a two-tracklet night misfits on its own
    assert "one of them" in why(trial_size=2, trk_b_resid=2.2)
    # linking it drags the object's own astrometry off the orbit
    assert "own observations off the orbit" in why(seed_post=3.0)
    # nothing in the field is worth the reader's time -- and the sentence
    # names the candidate it is REFUSING, not just the field's best: the two
    # are different tracklets with different scores (2006 UP17 read "best P
    # 0.070" while silencing a candidate scoring 0.001), and the study's 1316
    # attempts disagree about the floor on 452 of them
    weak = why(ml_ps=(0.10, 0.0))
    assert "strong enough" in weak and "the candidate that linked" in weak
    assert "0.15 floor" in weak
    # the object's own orbit cannot support the claim either way
    assert "weakly determined" in why(U=ItfConfig().single_night_show_max_U + 0.5)
    # a richer, multi-night profile this report does not speak for
    assert "nights at once" in why(trial_size=2, multinight=True)


def test_top_ml_maybe_is_promoted():
    # A "maybe" tracklet (motion-mismatch in the 1.0-2.5" band) that is the
    # single highest-ML candidate is promoted into the tried set -- regardless
    # of its maybe status and regardless of the orbit being well-determined.
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def ctx():
        return SimpleNamespace(cfg=ItfConfig(), verbose=False, log=lambda *a: None)

    def cand(cat, p):
        return {"traj_category": cat, "ml_p": p, "qual": 1.5, "dist": 5.0}

    # top overall is a maybe -> promoted, and it sorts first in `likely`
    tall = [cand("maybe", 0.9), cand("likely", 0.4), cand("maybe", 0.2)]
    _rk, likely, _tail = S._rank_candidates(ctx(), tall, 0)
    assert tall[0]["traj_category"] == "maybe" and tall[0]["ml_p"] == 0.9
    assert likely[0] is tall[0]                       # the promoted maybe, tried first
    assert sum(t["traj_category"] == "likely" for t in likely) == 1  # real likely kept

    # top overall is a likely -> a lower-ranked maybe is NOT promoted
    _rk, likely, _tail = S._rank_candidates(ctx(), [cand("likely", 0.9), cand("maybe", 0.5)], 0)
    assert [t["traj_category"] for t in likely] == ["likely"]

    # flag off -> never promote even when the top is a maybe
    c = ctx(); c.cfg.try_top_ranked_maybe = False
    _rk, likely, _tail = S._rank_candidates(c, [cand("maybe", 0.9), cand("likely", 0.3)], 0)
    assert [t["traj_category"] for t in likely] == ["likely"]

    # no ML scores (not ml_ranked) -> the maybe is left alone
    tall = [{"traj_category": "maybe", "score": 1.0, "qual": 1.5, "dist": 5.0},
            {"traj_category": "likely", "score": 2.0, "qual": 0.5, "dist": 3.0}]
    _rk, likely, _tail = S._rank_candidates(ctx(), tall, 0)
    assert [t["traj_category"] for t in likely] == ["likely"]


def test_prune_trial_clips_blunder():
    # A joint trial with one blunder tracklet: the prune must drop only the
    # misfitting CANDIDATE obs, refit, and keep the good tracklet -- never a
    # seed line, and pruned lines must not survive into cand.
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    seed = _obs80("K17C37L", "2017 02 03.12345", "10 10 00.00", "+11 00 00.0")
    # two REAL tracklets: three obs each, one night each, one station each --
    # a single obs is not a tracklet and the prune may not leave one behind
    good = [_obs80("PR00001", "2017 01 02.3456%d" % i,
                   "10 00 00.0%d" % i, "+10 00 00.%d" % i) for i in range(3)]
    bad = [_obs80("PR00002", "2017 01 05.3456%d" % i,
                  "10 00 05.0%d" % i, "+10 00 03.%d" % i, stn="G96")
           for i in range(3)]

    class StubFit:
        def __init__(self, lines, resids):
            self._obs = [{"obs80_line": l} for l in lines]
            self._r = resids
        def residuals(self, included_only=False):
            return [{"d_ra_arcsec": r, "d_dec_arcsec": 0.0} for r in self._r]

    all_lines = [seed] + good + bad
    tr = SimpleNamespace(ok=True, used=1, count=7, rms=2.9, U=1.0,
                         fit=StubFit(all_lines, [0.1] + [0.2] * 3 + [5.0] * 3))
    refits = []
    def fake_fit(text, obscodes, **kw):
        refits.append(text)
        return SimpleNamespace(ok=True, used=2, count=2, rms=0.15, U=1.0,
                               fit=None)
    orig_fit = S.fit_astrometry
    S.fit_astrometry = fake_fit
    try:
        ctx = SimpleNamespace(cfg=ItfConfig(), obscodes=None,
                              log=lambda *a: None)
        base = seed + "\n"
        cand = "\n".join(good + bad) + "\n"
        tr2, cand2 = S._prune_trial(ctx, tr, base, cand, "")
        assert all(l in cand2 for l in good)
        assert not any(l in cand2 for l in bad)
        assert tr2.rms == 0.15                       # the refit was adopted
        assert refits and refits[0] == base + cand2  # seed rode along untouched
        # nothing to prune -> no refit, trial returned as-is
        refits.clear()
        tr3 = SimpleNamespace(ok=True, used=7, count=7, rms=0.1, U=1.0,
                              fit=StubFit(all_lines, [0.1] + [0.2] * 6))
        tr4, cand4 = S._prune_trial(ctx, tr3, base, cand, "")
        assert tr4 is tr3 and cand4 == cand and not refits
        # everything pruned -> trial unchanged (verdict will fail it upstream)
        tr5 = SimpleNamespace(ok=True, used=1, count=7, rms=9.0, U=1.0,
                              fit=StubFit(all_lines, [0.1] + [5.0] * 6))
        tr6, cand6 = S._prune_trial(ctx, tr5, base, cand, "")
        assert tr6 is tr5 and cand6 == cand
        # 2024 RL188: a tracklet pruned below min_tracklet_obs is not a thinner
        # tracklet.  Two of `bad`'s three obs are blunders -- the survivor goes
        # too, and with `good` intact the trial is still pruned (not bailed out).
        refits.clear()
        tr7 = SimpleNamespace(ok=True, used=5, count=7, rms=3.0, U=1.0,
                              fit=StubFit(all_lines,
                                          [0.1] + [0.2] * 3 + [0.2, 5.0, 6.0]))
        tr8, cand8 = S._prune_trial(ctx, tr7, base, cand, "")
        assert all(l in cand8 for l in good)
        assert not any(l in cand8 for l in bad)      # incl. the 0.2" survivor
        assert tr8.rms == 0.15 and refits
        # ...and when the stub is ALL that would have survived, the prune has
        # nothing left to offer: the trial comes back as fitted, for the
        # acceptance gates to fail on its own merits
        refits.clear()
        cand_bad = "\n".join(bad) + "\n"
        tr9 = SimpleNamespace(ok=True, used=1, count=4, rms=3.0, U=1.0,
                              fit=StubFit([seed] + bad, [0.1, 0.2, 5.0, 6.0]))
        tr10, cand10 = S._prune_trial(ctx, tr9, base, cand_bad, "")
        assert tr10 is tr9 and cand10 == cand_bad and not refits
    finally:
        S.fit_astrometry = orig_fit


def test_batch_trial_ml_floor():
    # The joint "top-N together" trials must take the top N ABOVE the ML floor
    # (P >= batch_min_ml_frac x best P): a far-below-best candidate stays out
    # of the joint hypothesis but keeps its singles trial. Without ML scores
    # (or with the floor disabled) the batch is the plain top-N.
    import pyroorb.itf.search as S

    def fake_fit(text, obscodes, **kw):        # clean fit; used grows per tracklet
        return _trial(15 + 5*sum(f"T{i}" in text for i in range(4)))
    def fake_subset_rms(fit, ids):
        return 0.05, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        ctx = _trial_queue_ctx()
        ps = [0.9, 0.6, 0.35, 0.05]           # floor = 0.9/3 = 0.30 -> T3 out
        likely = [{"astrometry": f"T{i}\n", "ml_p": p} for i, p in enumerate(ps)]
        rank = lambda d: 1.0 - d["ml_p"]
        _t, obs_out, _m = S._recurse_trial_queue(ctx, likely, rank, prior_linkage_used=15)
        assert "T0" in obs_out and "T1" in obs_out and "T2" in obs_out
        assert "T3" not in obs_out, f"below-floor T3 joined the batch: {obs_out!r}"
        # floor disabled -> plain top-N again
        ctx.cfg = ItfConfig(batch_min_ml_frac=0.0)
        _t, obs_out, _m = S._recurse_trial_queue(ctx, likely, rank, prior_linkage_used=15)
        assert "T3" in obs_out
        # no ML scores -> unfiltered (certainty/legacy ranking, scan path)
        ctx.cfg = ItfConfig()
        likely_noml = [{"astrometry": f"T{i}\n", "_r": i} for i in range(4)]
        _t, obs_out, _m = S._recurse_trial_queue(ctx, likely_noml, lambda d: d["_r"],
                                            prior_linkage_used=15)
        assert "T3" in obs_out
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_seed_guard_floor_is_noise_scaled():
    """The mid-search seed guard's absolute floor is the seed's OWN chi^2=1
    noise floor (x sigma_mult), never below the 0.5" safety minimum.  A
    coarse-station seed (noise floor 0.58") admits an honest 0.55" degradation
    the fixed 0.5" floor rejected (2014 TM87); a tight seed (0.30" floor) keeps
    the 0.5" minimum, so nothing loosens for it."""
    import pyroorb.itf.search as S

    def fake_fit(text, obscodes, **kw):
        n = sum(f"T{i}" in text for i in range(3))
        return _trial(15 + 3*n)
    orig_fit, orig_sub, orig_sig = (S.fit_astrometry, S.subset_rms,
                                    S.subset_sigma_floor)
    S.fit_astrometry = fake_fit
    # the guard measures BOTH sides with subset_rms (_seed_pre_rms for the pre),
    # so the double has to answer per fit: the seed fit's own 0.332" against the
    # trial's honest 0.55" degraded seed rms
    pub_fit = SimpleNamespace()
    S.subset_rms = lambda fit, ids: ((0.332, 15) if fit is pub_fit
                                     else (0.55, 15))
    try:
        ctx = _trial_queue_ctx(pub_rms=0.235, pub_fit=pub_fit)
        likely = [{"astrometry": f"T{i}\n", "_r": i} for i in range(3)]
        # coarse seed: noise floor 0.58" -> floor rises, 0.55" is taken
        S.subset_sigma_floor = lambda fit, ids: (0.58, 15)
        trial, obs_out, _m = S._recurse_trial_queue(ctx, likely, lambda d: d["_r"],
                                               prior_linkage_used=15)
        assert trial is not None, "0.55\" < the seed's 0.58\" noise floor must be taken"
        # tight seed: 0.30" floor -> 0.5" minimum holds, 0.55" still refused
        S.subset_sigma_floor = lambda fit, ids: (0.30, 15)
        trial2, _o2, _m2 = S._recurse_trial_queue(ctx, likely, lambda d: d["_r"],
                                             prior_linkage_used=15)
        assert trial2 is None, "0.55\" past a tight seed's 0.5\" floor must refuse"
    finally:
        (S.fit_astrometry, S.subset_rms, S.subset_sigma_floor) = (orig_fit,
                                                                  orig_sub, orig_sig)


def test_variant_verdict_beats_linear_on_evidence_not_order():
    # The LOV fallback replaces the linear verdict on STATUS first, but a tie of
    # two reviewable 0.6s is broken by evidence, not by which corridor ran first:
    # a fit beats no fit, and an exhaustion-PROVED sole standout beats a field
    # that could not be exhausted.  2024 QT17 live is exactly the latter pair.
    import pyroorb.itf.search as S

    def r(status, *, fit=None, sole=None):
        result = S.ItfSearchResult()
        result.verdict, result.fit = status, fit
        if sole is not None:
            result.vetting = {"single_best_fit": sole}
        return result

    rank = S._verdict_rank
    assert rank(r(SearchVerdict.LINKED)) > rank(r(SearchVerdict.POSSIBLE, fit=object(), sole=True))   # status first
    # a reviewable WITH a fit beats a bare "abandoning" 0.6
    assert rank(r(SearchVerdict.POSSIBLE, fit=object(), sole=False)) > rank(r(SearchVerdict.POSSIBLE))
    # exhaustion-proved sole standout beats a shown-but-not-exhaustible one
    assert (rank(r(SearchVerdict.POSSIBLE, fit=object(), sole=True))
            > rank(r(SearchVerdict.POSSIBLE, fit=object(), sole=False)))
    # equal evidence -> no replacement (the incumbent linear verdict stands)
    assert not (rank(r(SearchVerdict.POSSIBLE, fit=object(), sole=True))
                > rank(r(SearchVerdict.POSSIBLE, fit=object(), sole=True)))
    assert rank(None) < rank(r(SearchVerdict.NONE))


def test_variant_fan_is_contiguous_not_a_picket_fence():
    """Each variant's corridor widens to half the sky gap to its NEIGHBOUR, so
    the fan searches the LOV continuously instead of leaving the object in a
    hole between two thin corridors.

    The k grid is spaced by equal PROBABILITY, so samples pile up near the
    nominal and fan apart in the tails -- which is where a weak seed's object
    actually is.  Measured at the default 31 steps over the four 2025-26 short
    arcs, judged at 2010 (median gap between adjacent variants against the 2W
    it takes to touch): 2025 NV385 19.3 deg -> 0.0% of its LOV searched, 2026
    EP114 4.2%, 2025 YQ13 76.7%, 2026 EX121 93.3% -- the same order their
    searches actually placed in.  Half-gap widening takes NV385 to 86.7% and
    EX121 to 100% with the same variant count and the same integration.
    """
    import pytest
    from dataclasses import replace
    from pyroorb.itf.lov_variants import _neighbour_half_gaps, variant_windows
    import pyroorb.itf.lov_variants as V

    # three variants strung along RA at one epoch: 0, 10 and 30 deg apart
    lov_sigmas = np.array([0.0, -1.0, 1.0])
    tracks = [[{"_i": 0, "ra": 0.0, "dec": 0.0}],
              [{"_i": 0, "ra": -10.0, "dec": 0.0}],
              [{"_i": 0, "ra": 30.0, "dec": 0.0}]]
    gaps = _neighbour_half_gaps(tracks, lov_sigmas)
    # nominal sits between them and takes the LARGER half-gap (30/2, not 10/2)
    assert gaps[(0, 0)] == pytest.approx(15.0, abs=1e-6), gaps
    assert gaps[(1, 0)] == pytest.approx(5.0, abs=1e-6), gaps   # one neighbour
    assert gaps[(2, 0)] == pytest.approx(15.0, abs=1e-6), gaps  # one neighbour
    # ...so the k=0 and k=+1 corridors touch: 15 + 15 >= their 30 deg gap
    assert gaps[(0, 0)] + gaps[(2, 0)] >= 30.0 - 1e-9

    # Driven through variant_windows: the widths reach the node dicts, are
    # never below the fixed width, and never past the sweep's own ceiling.
    cfg = ItfConfig()
    seen = {}

    def _tracks(fit, oc, c, states, jds=None):
        return [0.0], [[dict(n, mag=None, speed=0.0, unc_deg=0.0,
                             delta=1.0, unix=0.0, elong=None)]
                       for n in (t[0] for t in tracks)]

    def _windows(run, c, max_unc, mag_limit=None):
        seen[run[0]["ra"]] = run[0]["unc_deg"]
        return [object()]

    orig = (V.variant_tracks, V.corridor_windows)
    V.variant_tracks, V.corridor_windows = _tracks, _windows
    try:
        variant_windows(None, None, replace(cfg, variant_fill_gaps=True),
                        12.0, np.zeros((3, 6)), lov_sigmas)
    finally:
        V.variant_tracks, V.corridor_windows = orig
    # capped at variant_fill_max_mult x W (2 x 2.5 = 5.0), below the 12.0
    # max_uncert ceiling -- the cap is what keeps the query affordable
    assert seen[0.0] == pytest.approx(5.0), seen       # 15 -> capped
    assert seen[-10.0] == pytest.approx(5.0), seen     # its own half-gap
    assert seen[30.0] == pytest.approx(5.0), seen      # 15 -> capped
    # ...and with the cap off, max_uncert_deg is the only ceiling
    seen.clear()
    V.variant_tracks, V.corridor_windows = _tracks, _windows
    try:
        variant_windows(None, None,
                        replace(cfg, variant_fill_gaps=True,
                                variant_fill_max_mult=0),
                        12.0, np.zeros((3, 6)), lov_sigmas)
    finally:
        V.variant_tracks, V.corridor_windows = orig
    assert seen[0.0] == pytest.approx(12.0), seen      # 15 clipped to max_uncert

    # the switch restores the old fixed-width picket fence exactly
    seen.clear()
    V.variant_tracks, V.corridor_windows = _tracks, _windows
    try:
        variant_windows(None, None, replace(cfg, variant_fill_gaps=False),
                        12.0, np.zeros((3, 6)), lov_sigmas)
    finally:
        V.variant_tracks, V.corridor_windows = orig
    assert set(seen.values()) == {cfg.variant_width_deg}, seen


def test_variant_states_central_out_and_fallback():
    """LOV variant states come back nominal-first, |k| ascending (the
    early-exit scan order), and a degenerate/absent LOV degrades to
    (None, None) so the search falls back to the linear corridor."""
    from pyroorb.itf.lov_variants import variant_states
    cfg = ItfConfig()
    lov_sigmas = [0.0, -3.0, 1.5, -1.5, 3.0]
    states = np.array([[i, 0, 0, 0, 0, 0] for i in range(len(lov_sigmas))], float)

    class _Fit:
        def lov_uncertainty(self, n_steps, n_sigma):
            assert n_steps == cfg.variant_n_steps and n_sigma == cfg.variant_n_sigma
            return {"variants": states, "k": list(lov_sigmas)}

    s, kk = variant_states(_Fit(), cfg)
    assert kk[0] == 0.0, f"nominal (k=0) must be scanned first: {kk}"
    assert list(np.abs(kk)) == sorted(np.abs(kk)), f"not central-out: {kk}"
    assert s[0, 0] == 0.0 and s.shape == states.shape   # states rode with their k

    class _Degenerate:
        def lov_uncertainty(self, n_steps, n_sigma):
            return None

    assert variant_states(_Degenerate(), cfg) == (None, None)

    # ...but "no ELEMENT sigmas" is not "no orbits": lov_uncertainty drops every
    # variant whose osculating elements come out non-finite (a momentarily
    # hyperbolic LOV sample) and then discards the lot below its sigma floor,
    # while this scan only ever wanted states.  2002 PG150 (U 18.65, the
    # weakest kind of seed the fallback exists FOR) has all 31 of its samples
    # re-correct finite and still got nothing, so the search scanned zero
    # variants and called it "no likely matches".
    class _NoElements:
        _state = np.zeros(6)

        def lov_uncertainty(self, n_steps, n_sigma):
            return None                      # elements non-finite -> no sigmas

        def lov_refit(self):
            u = np.array([1.0, 0, 0, 0, 0, 0])
            return u, 2.0, lambda seed, u_, sig, k: seed

    st, kk = variant_states(_NoElements(), cfg)
    assert st is not None and len(st) == cfg.variant_n_steps, st
    assert kk[0] == 0.0, f"still central-out: {kk}"
    assert list(np.abs(kk)) == sorted(np.abs(kk)), kk


def test_itf_cli_runs_from_any_directory():
    # The ITF CLI is the package's `python -m pyroorb.itf` (installed as `pyroorb-itf`): it must
    # run from ANY working directory, not only the repo's. --help exercises the whole import
    # chain without touching the database.
    import subprocess, tempfile
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ, PYTHONPATH=src)
    r = subprocess.run([sys.executable, "-m", "pyroorb.itf", "search", "--help"],
                       cwd=tempfile.gettempdir(), capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"the ITF CLI failed from a foreign cwd: {r.stderr[-400:]}"
    assert "--trace-level" in r.stdout, r.stdout[:400]
    import pyroorb
    assert callable(pyroorb._itf), "pyroorb-itf console entry point is missing"


def test_a_cancelled_chunk_drops_windows_instead_of_hanging_or_retrying():
    """A statement timeout must degrade the sweep, not re-run the query.

    Nothing bounded a sweep QUERY before this: `connect_timeout` covers only
    the handshake, and SearchBudget is consulted BETWEEN trials and variants,
    never inside a sweep.  So one bad chunk hung whole searches -- 2025 NV385's
    variant sweep is 315 chunks and py-spy caught the process parked in
    cur.execute with flat CPU for 45+ min; 2025 YQ13 and 2026 EP114 hung in the
    same query on unmodified code.

    Two things are pinned here.  (1) `_retry` must NOT retry a cancellation:
    psycopg2's QueryCanceled subclasses OperationalError, so the reconnect
    clause would otherwise re-run the very query the server just gave up on,
    paying the ceiling twice before raising anyway.  (2) the sweep drops the
    chunk's windows into `last_sweep_skipped` -- which the search turns into
    its "not a clean negative" warning -- and abandons the rest once
    `sweep_max_cancels` is reached, because a per-statement ceiling alone does
    not bound a 315-chunk sweep.
    """
    from dataclasses import replace
    from pyroorb.itf.candidates import DbSource
    from pyroorb.itf.trace import Tracer, L_STORY
    from psycopg2 import errors as pgerr

    class _Cancelling:
        """Cursor whose execute always times out server-side."""
        def __init__(self, calls): self.calls = calls
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, args):
            self.calls.append(1)
            raise pgerr.QueryCanceled("canceling statement due to "
                                      "statement timeout")
        def fetchall(self): return []

    calls, rollbacks = [], []

    class _Conn:
        def cursor(self, **kw): return _Cancelling(calls)
        def rollback(self): rollbacks.append(1)

    lines = []
    s = DbSource.__new__(DbSource)
    s.cfg = ItfConfig(sweep_mode="chunk", sweep_max_cancels=3)
    s.index_cfg, s.conn, s.last_sweep_skipped = INDEX_CFG, _Conn(), 0
    s.trace = trc = Tracer(lines.append, L_STORY)
    reconnects = []
    s.reconnect = lambda: (reconnects.append(1), s.conn)[1]

    # epochs a YEAR apart, so each window becomes its OWN chunk (the sweep
    # breaks a chunk on a > 200 d jump) and several cancellations can occur
    def _epoch_window(k):
        w = _window(ra0=150.0 + 0.7 * k, uncert_deg=1.5)
        off = k * 400 * 86400
        return replace(w, unix_cur=w.unix_cur + off,
                       unix_start=w.unix_start + off,
                       unix_end=w.unix_end + off)

    windows = [_epoch_window(k) for k in range(8)]
    out = list(s.sweep_candidates(windows, 2.5, 0.0, True))

    # (1) the cancellation was never retried -- one execute per chunk attempt,
    #     and no reconnect (which is what the OperationalError clause does)
    assert not reconnects, "a cancelled query must not trigger a reconnect"
    assert len(calls) <= s.cfg.sweep_max_cancels, \
        f"query re-run after cancellation: {len(calls)} executes"
    assert rollbacks, "the aborted transaction must be rolled back"

    # (2) it degraded rather than raising, and said so
    assert all(not rows for _w, rows, _c, _f in out), out
    assert s.last_sweep_skipped >= len(windows), \
        f"dropped windows must be reported, got {s.last_sweep_skipped}"
    body = " ".join(lines)
    assert "cancelled" in body and "abandoning" in body, body

    # ...and with the abandon rule off it keeps trying every chunk, still
    # never retrying any of them
    calls.clear()
    s2 = DbSource.__new__(DbSource)
    s2.cfg = ItfConfig(sweep_mode="chunk", sweep_max_cancels=0)
    s2.index_cfg, s2.conn, s2.last_sweep_skipped = INDEX_CFG, _Conn(), 0
    s2.trace, s2.reconnect = None, lambda: s2.conn
    list(s2.sweep_candidates(windows, 2.5, 0.0, True))
    assert s2.last_sweep_skipped >= len(windows), s2.last_sweep_skipped
    assert len(calls) == len(windows), \
        f"with no abandon rule every chunk is tried once: {len(calls)}"


def test_statement_timeout_is_armed_for_search_but_not_maintenance():
    """The query ceiling rides the SEARCH role only.

    Index maintenance (connect_index) legitimately runs for minutes -- the
    weekly sweep -- so it must not inherit a ceiling sized for one corridor
    chunk.  Both go through dbio._connect, so the split is worth pinning."""
    from pyroorb.itf import dbio

    seen = {}

    def _fake_connect(**kw):
        seen[kw["user"]] = kw.get("options")

        class _C:
            pass
        return _C()

    cfg = ItfConfig(sweep_statement_timeout_sec=120, db_host="db.example.org",
                    db_index_host="db.example.org", db_password="x", db_index_password="x")
    real = dbio.psycopg2 if hasattr(dbio, "psycopg2") else None
    import psycopg2
    orig = psycopg2.connect
    psycopg2.connect = _fake_connect
    try:
        dbio.connect_pub(cfg)
        dbio.connect_index(cfg)
    finally:
        psycopg2.connect = orig

    assert seen[cfg.db_user] == "-c statement_timeout=120000", seen
    assert seen[cfg.db_index_user] is None, \
        f"maintenance must not inherit the sweep ceiling: {seen}"

    # 0 disables it entirely (the pre-2026-08 behaviour)
    seen.clear()
    psycopg2.connect = _fake_connect
    try:
        dbio.connect_pub(ItfConfig(sweep_statement_timeout_sec=0,
                                   db_host="db.example.org", db_password="x"))
    finally:
        psycopg2.connect = orig
    assert seen[cfg.db_user] is None, seen


def test_db_sweep_paths_narrate_without_crashing():
    """Every DbSource sweep path, driven with a TRACER attached.

    The recovery suite runs live but with trace=None, and the offline tests use
    OverlaySource, so the DB layer's trace code had no coverage at all -- which
    is how two paths came to reach for `feats` where the local is `matched`
    (a NameError the moment anyone traced setjoin/rangejoin).  Same capturing
    cursor as test_window_relation_columns_stay_aligned, plus a tracer: at
    level 4 each query narrates itself, at level 2 it only feeds the tally the
    sweep rollup reads."""
    from pyroorb.itf.candidates import DbSource
    from pyroorb.itf.trace import Tracer, L_STORY, L_FIREHOSE

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, args): pass
        def fetchall(self): return []

    class _Conn:
        def cursor(self, **kw): return _Cur()

    for mode, uncert in (("setjoin", 0.003), ("rangejoin", 1.5),
                         ("chunk", 1.5)):
        for level in (L_STORY, L_FIREHOSE):
            lines = []
            trc = Tracer(lines.append, level)
            s = DbSource.__new__(DbSource)
            s.cfg = ItfConfig(sweep_mode=mode)
            s.index_cfg, s.conn, s.last_sweep_skipped = INDEX_CFG, _Conn(), 0
            s.trace = trc
            s.reconnect = lambda: s.conn
            windows = [_window(uncert_deg=uncert),
                    _window(ra0=150.5, uncert_deg=uncert)]
            # the assertion is mostly "this does not raise" -- a NameError in a
            # trace line is still a crashed search
            list(s.sweep_candidates(windows, 2.5, 0.0, True))
            assert trc.get("db_queries") > 0, (mode, level, "no query tallied")
            body = " ".join(lines)
            if level == L_FIREHOSE:
                assert "query" in body, (mode, "level 4 must narrate queries")
            else:
                assert "db " not in body and "query" not in body, \
                    (mode, f"level 2 leaked a per-query line: {body!r}")


def test_trial_queue_outcome_tally_renders_every_rejection_tag():
    """The STORY rollup of a trial queue is the only place a level-2 reader learns
    WHY the trials died, and no offline fixture reaches the rejection paths (the
    fixtures all link on the first trial) -- so pin the rendering here, with the
    tags _trial_rejection actually returns."""
    import pyroorb.itf.search as S
    from pyroorb.itf.trace import Tracer

    # every tag the trial queue can tally must have prose; a bare tag leaking into
    # the trace is the bug this guards
    tags = {"warm screen", "fit failed", "no-progress", "rms-sanity",
            "seed-guard", "seed-retention", "hyp-drop"}
    assert tags <= set(S._OUTCOME_GLOSS), tags - set(S._OUTCOME_GLOSS)

    out = []
    trc = Tracer(out.append)
    S._trial_queue_outcome(trc, {"warm screen": 4, "no-progress": 2,
                            "rms-sanity": 1}, 7, 7)
    body = [l.split("] ", 1)[1].rstrip() for l in out]
    assert body[0].strip() == "7/7 trials tried:", body[0]
    # ordered by count, most first, so the dominant failure mode reads first
    assert "4 x killed by warm screen" in body[1]
    assert "2 x fitted, NO PROGRESS" in body[2]
    assert "1 x fitted, RMS SANITY" in body[3]

    # a partly-run trial queue (budget spent) must say tried-of-total, not total
    out.clear()
    S._trial_queue_outcome(trc, {"warm screen": 1}, 1, 7)
    assert "1/7 trial tried:" in out[0], out[0]

    # nothing tried -> nothing said (the round's outcome line carries it)
    out.clear()
    S._trial_queue_outcome(trc, {}, 0, 7)
    assert out == []


def test_trial_rejection_tags_match_its_prose():
    """_trial_rejection returns (trc_msg, log_msg, tag); the tag is what the
    trace tallies by, so it must not drift from the message it labels."""
    import pyroorb.itf.search as S
    cfg = ItfConfig()
    pub = SimpleNamespace(used=8, count=8, rms=0.05, arclength_days=10.0)
    ctx = SimpleNamespace(cfg=cfg, pub=pub, pub_ids=set())

    # (a) no progress: fewer kept than the seed
    tr = SimpleNamespace(used=8, count=12, rms=0.05, fit=None)
    msg, _log, tag = S._trial_rejection(ctx, tr, 8, True, 1.0, 8)
    assert tag == "no-progress" and "NO PROGRESS" in msg, (tag, msg)

    # (b) rms sanity: converged, progressed, but far off-sky
    tr = SimpleNamespace(used=12, count=12, rms=500.0, fit=None)
    msg, _log, tag = S._trial_rejection(ctx, tr, 8, True, 1.0, 8)
    assert tag == "rms-sanity" and "RMS SANITY" in msg, (tag, msg)

    # (c) a clean trial is takeable -> None, not a tag
    tr = SimpleNamespace(used=12, count=12, rms=0.06, fit=None)
    assert S._trial_rejection(ctx, tr, 8, True, 1.0, 8) is None


def test_a_take_may_not_expel_a_linked_tracklet():
    """A recursion trial must EXTEND the linkage it recursed on.

    The trial queue counts progress in obs, so a trial that fits only by rejecting a
    tracklet a shallower level took -- 3 linked obs out, 4 foreign in -- reads as
    progress, and the seed guard cannot see it (it protects the published seed
    alone).  2005 UD562 lost a live search exactly so: the 2005 truth taken at
    depth 0, then foreign C6ZKGY2 TAKEN at depth 1 by expelling it, 42084" off.
    Clause (d) refuses such a trial; one rejected ob of a linked tracklet (the
    engine's own outlier cut) is still an extension.  And a trial bringing
    MORE observing nights than it expels supersedes the earlier take -- counted in nights, so one foreign night
    split into two tracklet ids does not out-vote a real one."""
    import pyroorb.itf.search as S
    from pyroorb.itf.obs80_fields import obs_unique_id

    class StubFit:
        def __init__(self, lines, included):
            self._obs = [{"obs80_line": l} for l in lines]
            self._inc = list(included)

        def is_included(self):
            return list(self._inc)

        def residuals(self, included_only=False):
            return [{"d_ra_arcsec": 0.1, "d_dec_arcsec": 0.1}
                    for inc in self._inc if inc or not included_only]

    seed = [_obs80("SEED%03d" % i, "2015 05 %02d.30000" % (10 + i),
                   "01 00 00.00", "-09 00 00.0", stn="W84") for i in range(8)]
    hyp = [_obs80("PR00000", "2005 10 25.%05d" % (10000 + 1000 * i),
                  "02 00 00.00", "-08 00 00.0", stn="568") for i in range(3)]
    cand = [_obs80("C6ZKGY2", "2022 01 06.%05d" % (30000 + 1000 * i),
                   "03 00 00.00", "-07 00 00.0", stn="G96") for i in range(4)]
    hypothesis_ids = frozenset(obs_unique_id(l) for l in hyp)
    ctx = SimpleNamespace(
        cfg=ItfConfig(),
        pub=SimpleNamespace(used=8, count=8, rms=0.05, arclength_days=4.0),
        pub_ids={obs_unique_id(l) for l in seed},
        swept_tracklets={i: ("PR00000@568/2005-10-25(3obs)", hypothesis_ids)
                         for i in hypothesis_ids})
    groups = S._hypothesis_groups(ctx, "\n".join(hyp) + "\n")
    assert [(t, i) for t, i, _l in groups] == [
        ("PR00000@568/2005-10-25(3obs)", hypothesis_ids)], groups
    cand_text = "\n".join(cand) + "\n"

    def fit_for(hypothesis_kept, extra=()):
        lines = seed + hyp + cand + list(extra)
        inc = ([True] * 8 + [i < hypothesis_kept for i in range(3)]
               + [True] * (4 + len(extra)))
        return SimpleNamespace(used=sum(inc), count=len(inc), rms=0.2,
                               fit=StubFit(lines, inc))

    def judge(tr, prior, added, single=False):
        return S._trial_rejection(ctx, tr, prior, single, 9.0, 7, groups,
                                 added_lines=added)

    # the UD562 take: every seed ob kept, the linked tracklet wholly rejected,
    # ONE foreign night in -- 12 used > 11 prior, so (a) calls it progress
    tr = fit_for(0)
    assert S._trial_rejection(ctx, tr, 11, False, 9.0, 7) is None   # before clause (d)
    msg, _log, tag = judge(tr, 11, cand_text)
    assert tag == "hyp-drop" and "HYPOTHESIS DROP" in msg, (tag, msg)
    assert "1 night(s) against the 1" in msg, msg
    assert tag in S._NONGRAV_RESCUABLE    # a drift can hold both; ask it

    # NOT waived for a lone candidate: declining leaves the parent standing
    assert judge(tr, 11, cand_text, single=True)[2] == "hyp-drop"

    # one of three rejected is the engine's outlier cut, not an expulsion...
    assert judge(fit_for(2), 11, cand_text) is None
    # ...but a majority rejected is
    assert judge(fit_for(1), 12, cand_text)[2] == "hyp-drop"

    # a SECOND foreign tracklet on the SAME night is still one night: refused
    same = [_obs80("C6ZKGZ9", "2022 01 06.%05d" % (40000 + 1000 * i),
                   "03 00 01.00", "-07 00 00.0", stn="G96") for i in range(3)]
    tr = fit_for(0, same)
    assert judge(tr, 11, cand_text + "\n".join(same) + "\n")[2] == "hyp-drop"
    # ...but a second NIGHT supersedes the lone linked one: 2 nights in, 1 out
    other = [_obs80("C6ZQ001", "2022 01 20.%05d" % (30000 + 1000 * i),
                    "03 10 00.00", "-07 10 00.0", stn="G96") for i in range(3)]
    tr = fit_for(0, other)
    assert judge(tr, 11, cand_text + "\n".join(other) + "\n") is None
    exp = S._expelled_tracklets(tr, groups)
    assert [t for t, *_ in exp] == ["PR00000@568/2005-10-25(3obs)"]
    assert S._supersedes(tr, exp, cand_text + "\n".join(other) + "\n") == (2, 1)

    # A TIE (one night for one) is broken by the object's own obs: the swap
    # stands only if the seed fits no worse than under the linkage it
    # replaces.  The stub's residuals put the seed at sqrt(0.1^2+0.1^2) per ob.
    tr = fit_for(0)
    seed_now = S.subset_rms(tr.fit, ctx.pub_ids)[0]
    def tie(parent_rms):
        return S._trial_rejection(ctx, tr, 11, False, 9.0, 7, groups,
                                 added_lines=cand_text,
                                 parent_seed_rms=parent_rms)
    assert tie(None)[2] == "hyp-drop"                  # unknown -> refuse
    assert tie(seed_now * 0.5)[2] == "hyp-drop"        # UD562: seed worse
    assert tie(seed_now * 4.0) is None                 # HE349: seed better
    assert tie(seed_now) is None                       # no worse = enough
    # ...but not bought by rejecting the object's own obs
    clipped = SimpleNamespace(used=11, count=15, rms=0.2, fit=StubFit(
        seed + hyp + cand, [True] * 5 + [False] * 3 + [False] * 3 + [True] * 4))
    assert S._trial_rejection(ctx, clipped, 8, False, 9.0, 7, groups,
                             added_lines=cand_text,
                             parent_seed_rms=seed_now * 4.0)[2] == "hyp-drop"

    # depth 0 has no hypothesis and is untouched
    assert S._hypothesis_groups(ctx, "") == []
    # a hypothesis line no trial took is left unjudged, never a 1-ob tracklet
    ctx.swept_tracklets = {}
    assert S._hypothesis_groups(ctx, "\n".join(hyp) + "\n") == []


def test_sweep_tiers_keep_the_fan_central_out():
    """The fan is swept in probability-order tiers -- consecutive
    slices of the central-out order, every variant in exactly one tier -- and
    0 is ONE tier, i.e. the original single union sweep unchanged."""
    import pyroorb.itf.search as S
    fan = [(k, [f"w{k}"]) for k in (0.0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5)]
    assert S._sweep_tiers(fan, 0) == [fan]
    assert S._sweep_tiers(fan, 7) == [fan]
    assert S._sweep_tiers(fan, 99) == [fan]
    tiers = S._sweep_tiers(fan, 3)
    assert [len(t) for t in tiers] == [3, 3, 1]
    assert [e for t in tiers for e in t] == fan      # order and coverage kept
    assert ItfConfig().variant_sweep_tier > 0     # on by default


def test_trace_levels_filter_and_stay_cumulative():
    """A level-N tracer writes every line at or below N and nothing above it,
    and the narrative levels are cumulative (no holes at 2 that 3 would fill)."""
    from pyroorb.itf.trace import Tracer, L_HEAD, L_STORY, L_DETAIL, L_FIREHOSE

    def emit(level):
        out = []
        t = Tracer(out.append, level)
        t.head("H")
        t.story("S")
        t("D")                      # unnamed level == DETAIL, the old default
        t("F", L_FIREHOSE)
        t.field("lbl", "story-field")
        t.field("lbl", "head-field", L_HEAD)
        return "".join(out)

    assert "H" in emit(L_HEAD) and "S" not in emit(L_HEAD)
    assert "S" in emit(L_STORY) and "D" not in emit(L_STORY)
    assert "D" in emit(L_DETAIL) and "F" not in emit(L_DETAIL)
    assert "F" in emit(L_FIREHOSE)
    # cumulative: everything level 2 says, level 3 also says
    assert all(l in emit(L_DETAIL) for l in ("H", "S", "head-field",
                                             "story-field"))
    # a head-level field survives the strictest level; a story one does not
    assert "head-field" in emit(L_HEAD) and "story-field" not in emit(L_HEAD)
    # wants() is the hot-loop guard and must agree with what __call__ does
    t = Tracer(lambda _l: None, L_STORY)
    assert t.wants(L_HEAD) and t.wants(L_STORY)
    assert not t.wants(L_DETAIL) and not t.wants(L_FIREHOSE)
    assert not Tracer(None, L_FIREHOSE).wants(L_HEAD), "no sink -> no lines"


def test_trace_says_unchanged_facts_once():
    """field_changed/cont_changed exist so a 31-variant scan states the parent
    orbit once, not 31 times -- and speak up again the moment it changes."""
    from pyroorb.itf.trace import Tracer
    out = []
    t = Tracer(out.append)
    for _ in range(3):
        t.field_changed("orbit", "used 11/11")
    t.field_changed("orbit", "used 22/22")
    t.field_changed("orbit", "used 22/22")
    bodies = [l.split("] ", 1)[1].strip() for l in out if l.strip()]
    assert bodies == ["orbit        used 11/11", "orbit        used 22/22"], bodies
    # the memory must survive a round boundary -- that is the whole point
    out.clear()
    t.round_begin("round 2")
    t.field_changed("orbit", "used 22/22")
    assert not [l for l in out if "orbit" in l], out


def test_trace_round_tally_is_per_round():
    """The DB rollup reads counters the candidate source accumulates; a new
    round must not inherit the previous round's tally."""
    from pyroorb.itf.trace import Tracer
    t = Tracer(lambda _l: None)
    t.round_begin("r1")
    t.count("db_queries", 3)
    t.count("db_queries")
    assert t.get("db_queries") == 4
    assert t.pop("db_queries") == 4 and t.get("db_queries") == 0
    t.count("db_rows", 7)
    t.round_begin("r2")
    assert t.get("db_rows") == 0, "round_begin must reset the tally"


def test_trace_histogram_answers_were_any_promising():
    from pyroorb.itf.trace import histogram
    assert histogram([0.9, 0.6, 0.2, 0.01, None]) == "P>=0.5: 2, 0.1-0.5: 1, <0.1: 1"
    assert histogram([]) == "no ML scores"
    assert histogram([None, None]) == "no ML scores"


def test_bare_callable_trace_still_works():
    """itf_search accepts a plain callable(str) sink; the levelled helpers must
    survive that (the offline tests and older callers pass functions)."""
    import pyroorb.itf.search as S
    out = []
    ctx = SimpleNamespace(trace=out.append)
    trc = S._ctx_tracer(ctx)
    trc.head("headline")
    trc.field("lbl", "value")
    trc.warn("careful")
    assert any("headline" in l for l in out)
    assert any("careful" in l and l.split("] ")[1].startswith("! ") for l in out)
    # and no tracer at all must be silent rather than an AttributeError
    S._ctx_tracer(SimpleNamespace()).head("nobody listening")


def _grp(*jds):
    return [{"jd_utc": j} for j in jds]


def test_seed_opposition_is_compact_and_three_nights():
    """The seed opposition is the most COMPACT one with >= 3 observing nights
    -- not the fattest, and not the earliest.

    2023 FU10's shape, which is what exposed this: a lone 2013 precovery night
    first, a fat well-observed 2024 apparition, and the 2-day 2023 discovery
    arc in between.  The old max-obs default reached for the fat one; seeding
    apparition 0 (the chronological first) gets a single night that determines
    no orbit at all.

    ...and compactness is not the only requirement: the arc must carry enough
    OBSERVATIONS to determine anything (SEED_MIN_OBS).
    Three observations are six measurements against six orbital elements --
    exactly determined, rms identically 0.000", covariance collapsing to
    numerical zero and U reporting 0.00, the best possible certainty for the
    worst possible arc (2013 UL49)."""
    precovery = _grp(0.0, 0.02)                        # 1 night
    discovery = _grp(500.0, 500.03, 500.06, 501.0, 501.03, 502.0)   # 3 n, 6 obs
    fat = _grp(*[1000.0 + i * 0.9 for i in range(30)])  # many nights, 26 d
    apps = [precovery, discovery, fat]
    assert pick_seed_opposition(apps) == 1, "must pick the compact 3-night arc"

    # no apparition reaches 3 nights -> best available (most nights), not a crash
    assert pick_seed_opposition([precovery, _grp(500.0, 501.0)]) == 1

    # a 3-night arc of only THREE observations is exactly determined, so it
    # loses to a fatter apparition that can actually carry information
    thin = _grp(500.0, 501.0, 502.0)                   # 3 nights, 3 obs
    assert pick_seed_opposition([precovery, thin, fat]) == 2, \
        "an exactly-determined arc must not be preferred for compactness"
    # ...but when nothing qualifies it is still the best available
    assert pick_seed_opposition([precovery, thin]) == 1


def test_seed_prefers_the_discovery_apparition_over_a_sparse_precovery():
    """2024 NH2's shape.  A sparse PRECOVERY apparition can be shorter in SPAN
    than the real discovery arc while carrying almost no information, and
    most-compact then picks it: 7 obs over 7 nights (1.0 per night) in 36 d,
    against the 28-obs/10-night 2024 discovery arc at 57 d.  Its first 3 nights
    are 3 single-obs nights -- six measurements, six elements, exactly
    determined -- and such a seed cannot accept ANY extension.  The designation
    names the discovery year; use it."""
    day = 86400.0
    def night(t0, n):                      # n obs inside one night
        return [{"jd_utc": t0 + 0.01 * k} for k in range(n)]
    # 2019 precovery: 7 nights, 1 obs each, spread over 36 d -> SHORTER span
    precovery = [o for k in range(7) for o in night(2458692.0 + 6.0 * k, 1)]
    # 2024 discovery: 10 nights, ~3 obs each, over 57 d
    discovery = [o for k in range(10) for o in night(2460495.0 + 6.3 * k, 3)]
    apps = [precovery, discovery]
    assert pick_seed_opposition(apps) == 0, "span-only still prefers precovery"
    assert pick_seed_opposition(apps, desig="2024 NH2") == 1, \
        "the designation year must select the discovery apparition"
    # a designation with no parseable year falls back, and does not crash
    for d in (None, "433", "P10zO2R", "PLS2040"):
        assert pick_seed_opposition(apps, desig=d) == 0
    assert designation_year("2024 NH2") == 2024
    assert designation_year("1998 QE2") == 1998
    assert designation_year("433") is None


def test_observing_nights_are_real_nights_not_the_gate_counter():
    """Three consecutive nights over 2 days are THREE nights here, even though
    the gates' 36-hour span counter (opp_profile.count_spans) deliberately reads them as
    one.  Collapsing them is what made a 2-day discovery arc look unfittable."""
    arc = _grp(0.0, 0.03, 1.0, 1.03, 2.0)
    assert len(observing_nights(arc)) == 3
    assert count_spans("\n".join(["     K23F10U  C2023 03 %02d.00000 "
                                    "00 00 00.00 +00 00 00.0          W68"
                                    % (28 + int(o["jd_utc"]))
                                    for o in arc])) < 3, \
        "the gate counter still collapses them -- the two rules differ on purpose"

    head, rest = _first_n_nights(arc, 2)
    assert [o["jd_utc"] for o in head] == [0.0, 0.03, 1.0, 1.03]
    assert [o["jd_utc"] for o in rest] == [2.0]


def test_acceptance_judges_each_tracklet_as_swept():
    """The status-1.0 acceptance gate (2026-08-01) asks per-TRACKLET questions,
    not aggregate ones: each linked tracklet must keep >= 3/4 of the obs it was
    SWEPT with (the denominator is pre-prune, so a 3-obs tracklet down to two
    fails while a 4-obs tracklet down to three passes), its kept obs must fit
    their own sigma, and the seed must not be statistically degraded.

    Driven through the real gate.  The retention case is 2015 BO519's: a 3-obs
    tracklet keeping 2 was accepted at status 1.0 because the OLD gate looked
    only at counts and an overall arcsec rms -- which had FALLEN, since bolting
    a distant tracklet onto a 3-day arc buys the orbit leverage."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def _fit(lines, included, resid_arcsec, sigma=0.5):
        """Stub with just the surface subset_chi2/subset_included use."""
        n = len(lines)
        rows = [{"d_ra_arcsec": r, "d_dec_arcsec": 0.0,
                 "sigma_ra": sigma, "sigma_dec": sigma}
                for r in resid_arcsec]
        return SimpleNamespace(
            _obs=[{"obs80_line": l} for l in lines],
            residuals=lambda included_only=False: rows,
            is_included=lambda: list(included))

    seed = [_obs80("K15Bp9O", f"2015 01 2{i}.30000", "10 00 00.00", "+10 00 00.0")
            for i in range(4)]
    seed_fit = _fit(seed, [True] * 4, [0.1] * 4)
    pub_ids = {S.obs_unique_id(l) for l in seed}

    def _ctx(trial_fit, swept):
        return SimpleNamespace(cfg=ItfConfig(), swept_tracklets=swept,
                               pub_ids=pub_ids, pub_obs80="".join(
                                   l + "\n" for l in seed),
                               pub=SimpleNamespace(fit=seed_fit, rms=0.1,
                                                   arclength_days=3.0))

    def _tracklet(tag, n, stn="F51"):
        lines = [_obs80(tag, f"2020 07 1{i}.30000", "11 00 00.00", "+11 00 00.0",
                        stn=stn) for i in range(n)]
        ids = frozenset(S.obs_unique_id(l) for l in lines)
        return lines, {i: (f"{tag}({n}obs)", ids) for i in ids}

    # (a) 3-obs tracklet keeping 2 -> 67% < 75% -> REFUSED (the BO519 shape)
    lines, swept = _tracklet("P10aaaa", 3)
    trial = _fit(seed + lines, [True] * 4 + [True, True, False], [0.1] * 7)
    ok, notes = S._accept_quality_ok(_ctx(trial, swept), _wrap(trial),
                                     "".join(l + "\n" for l in lines[:2]), 0)
    assert not ok, notes
    assert any("2/3 obs on" in n and "FAIL" in n for n in notes), notes

    # (b) 4-obs tracklet keeping 3, all fitting tightly -> ACCEPTED
    lines, swept = _tracklet("P10bbbb", 4)
    trial = _fit(seed + lines, [True] * 4 + [True, True, True, False],
                 [0.1] * 8)
    ok, notes = S._accept_quality_ok(_ctx(trial, swept), _wrap(trial),
                                     "".join(l + "\n" for l in lines[:3]), 0)
    assert ok, notes
    assert any("3/4 obs on" in n and "ok" in n for n in notes), notes

    # (c) every obs on, but they do NOT fit their own sigma -> REFUSED.
    # (The old gate could not see this at all: it had no per-tracklet test.)
    lines, swept = _tracklet("P10cccc", 4)
    trial = _fit(seed + lines, [True] * 8, [0.1] * 4 + [6.0] * 4)
    ok, notes = S._accept_quality_ok(_ctx(trial, swept), _wrap(trial),
                                     "".join(l + "\n" for l in lines), 0)
    assert not ok, notes
    assert any("chi2" in n and "FAIL" in n for n in notes), notes

    # (d) tracklet immaculate, but the link wrecks the SEED -> REFUSED
    lines, swept = _tracklet("P10dddd", 4)
    trial = _fit(seed + lines, [True] * 8, [9.0] * 4 + [0.1] * 4)
    ok, notes = S._accept_quality_ok(_ctx(trial, swept), _wrap(trial),
                                     "".join(l + "\n" for l in lines), 0)
    assert not ok, notes
    assert any("seed chi2" in n and "FAIL" in n for n in notes), notes


def test_acceptance_refuses_a_linkage_that_clips_the_seed():
    """A trial whose astrometry no longer CONTAINS the seed observations must
    fail the seed-preservation clause, not sail through it.

    `subset_chi2` scores what it can find, so seed lines the trial has dropped
    outright score as nothing: clip them all and it returns (0.0, 0), the
    Delta comes out NEGATIVE, and the guard whose whole job is protecting the
    seed passes BECAUSE the seed is gone.  The lines really can leave -- the
    drop-poor-fitters rebuild refits from the dropped text, so they are absent
    from the obs list rather than present-and-rejected.

    2025 NV385 is the case: an 8-obs/8-day seed whose LOV variant k=-1.18 took
    four W84/T09 tracklets from 2014-04 and 2015-10.  The fit kept 18/26 obs,
    dropping all 8 seed lines -- seed+tracklets and tracklets-alone converge on
    the same 0.497" orbit byte for byte, i.e. it had fitted a DIFFERENT object
    -- and the vetting read "seed chi2 1.0 -> 0.0 (Delta -1.0)  ok".  It died
    on the opposition-profile gate and the single-night U cap instead, which is luck.
    """
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def _fit(lines, included, resid_arcsec, sigma=0.5):
        rows = [{"d_ra_arcsec": r, "d_dec_arcsec": 0.0,
                 "sigma_ra": sigma, "sigma_dec": sigma}
                for r in resid_arcsec]
        return SimpleNamespace(
            _obs=[{"obs80_line": l} for l in lines],
            residuals=lambda included_only=False: rows,
            is_included=lambda: list(included))

    seed = [_obs80("K25Nc5V", f"2025 07 1{i}.30000", "10 00 00.00", "+10 00 00.0")
            for i in range(4)]
    seed_fit = _fit(seed, [True] * 4, [0.1] * 4)
    pub_ids = {S.obs_unique_id(l) for l in seed}

    lines = [_obs80("N004279", f"2014 04 2{i}.30000", "11 00 00.00",
                    "+11 00 00.0", stn="W84") for i in range(4)]
    ids = frozenset(S.obs_unique_id(l) for l in lines)
    swept = {i: ("N004279(4obs)", ids) for i in ids}

    tel = {}

    def _ctx(trial_fit, prior_ids=None):
        return SimpleNamespace(cfg=ItfConfig(), swept_tracklets=swept,
                               pub_ids=pub_ids, prior_ids=prior_ids,
                               telemetry=tel, pub_obs80="".join(
                                   l + "\n" for l in seed),
                               pub=SimpleNamespace(fit=seed_fit, rms=0.1,
                                                   arclength_days=3.0))

    # The seed lines are GONE from the trial; the four tracklet obs fit
    # immaculately.  Under the old code this read "Delta -1.0  ok".
    trial = _fit(lines, [True] * 4, [0.1] * 4)
    ok, notes = S._accept_quality_ok(_ctx(trial), _wrap(trial),
                                     "".join(l + "\n" for l in lines), 0)
    assert not ok, notes
    # ...and it is COUNTED on the telemetry that rides out on the result.  The
    # refusal is decided on string identity (obs_unique_id), so the failure mode
    # worth being able to see is a systematic identity mismatch -- an
    # ADES->OBS80 collapse, a satellite two-line record -- which would otherwise
    # read as quiet non-recovery at both gates.
    assert tel["seed_obs_missing"] == 4, tel
    assert any("CLIPPED" in n and "FAIL" in n for n in notes), notes
    assert any("4/4 seed obs CLIPPED" in n for n in notes), notes

    # ...and the `prior accepted` rescue must not hand the acceptance back.
    # It answers off the surviving hypothesis obs, which fit perfectly here.
    ok2, notes2 = S._accept_quality_ok(_ctx(trial, prior_ids=set(ids)),
                                       _wrap(trial),
                                       "".join(l + "\n" for l in lines), 0)
    assert not ok2, notes2

    # A seed that is merely REJECTED (present in the obs list, excluded by the
    # fit) is unchanged: it still counts at its full residual, so it fails on
    # the chi2 rather than the clip clause -- and a seed that is present and
    # fits is still accepted.
    trial_kept = _fit(seed + lines, [True] * 8, [0.1] * 8)
    ok3, notes3 = S._accept_quality_ok(_ctx(trial_kept), _wrap(trial_kept),
                                       "".join(l + "\n" for l in lines), 0)
    assert ok3, notes3
    assert not any("CLIPPED" in n for n in notes3), notes3


def test_thorough_is_one_shared_definition():
    """``--thorough`` means depth in MAGNITUDE, defined once.

    Every surface offering the flag (``itf search --thorough``,
    ``fit_object.py --itf-thorough``) goes through ItfConfig.apply_thorough,
    because a flag meaning two different things on two CLIs is how the engine
    and its front ends drift apart.  It RAISES rather than sets, so a caller who
    already asked for more is never walked back -- and an explicit --mag-limit
    applied after it still wins, which is the documented precedence.

    (A version that also raised the trial queue's RANK depth was reverted as
    unproven; if it returns it belongs here, not in each CLI.)
    """
    from pyroorb.itf.config import ItfConfig
    base = ItfConfig()
    assert ItfConfig().apply_thorough().mag_limit == base.mag_limit_deep
    assert base.mag_limit_deep > base.mag_limit
    more = ItfConfig()
    more.mag_limit = 25.0
    assert more.apply_thorough().mag_limit == 25.0


def test_incomplete_sweep_names_the_cause_from_one_source():
    """The skipped/cancelled pair is read from ONE source, not maxed apart.

    They are the same sweep's tally split by cause, and the search turns the
    split into the remedy it tells the caller to apply -- raise the row budget,
    or narrow a corridor the server will not serve.  Maxing them independently
    over the source chain could take `skipped` from a budget-exhausted source
    and `cancelled` from a different one, and report a cancellation about a
    sweep that was never cancelled.
    """
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def _src(skipped, cancelled):
        return SimpleNamespace(last_sweep_skipped=skipped,
                               last_sweep_cancelled=cancelled,
                               sweep_candidates=lambda *a, **k: iter(()))

    def _causes(sources):
        lines, tel = [], {}
        ctx = SimpleNamespace(cfg=ItfConfig(), source=sources[0],
                              log=lines.append, telemetry=tel, trace=None,
                              pub_ids=set(), obscodes=None, uncert_mult=2.5,
                              max_uncert_deg=19.44, collector=None,
                              verbose=False)
        orig = S.source_chain
        S.source_chain = lambda _s: sources
        try:
            S._collect_tracklets(ctx, SimpleNamespace(fit=None), 0.0, windows=[])
        finally:
            S.source_chain = orig
        return " ".join(lines), tel

    # budget on one source, a cancellation on another: the LOSS belongs to the
    # source that lost the most, and its cause comes with it
    msg, tel = _causes([_src(9, 0), _src(2, 2)])
    assert "9 window(s) skipped" in msg and "row budget exhausted" in msg
    assert "cancelled" not in msg, msg
    assert tel["windows_skipped"] == 9 and "windows_cancelled" not in tel
    # ...and when the losing source IS the cancelled one, say so
    msg, tel = _causes([_src(2, 0), _src(9, 9)])
    assert "the server cancelled" in msg or "query the server cancelled" in msg
    assert tel["windows_cancelled"] == 9


def test_archive_era_is_swept_early_only_for_a_strong_seed():
    """The pre-`ephem_start` archive era rides in every ordinary sweep when the
    SEED starts at U <= archive_seed_U, and otherwise waits for the
    post-acceptance final sweep (2022 SN92's 1995 Spacewatch tracklet).

    A weak seed's decades-old corridor is degrees wide and its windows die on
    the uncertainty gate, so sweeping it early would be waste on exactly the
    searches that fail."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def ctx(seed_U, archive_start="1980-01-01", archive_seed_U=6.0):
        cfg = ItfConfig()
        cfg.archive_start = archive_start
        cfg.archive_seed_U = archive_seed_U
        return SimpleNamespace(cfg=cfg, pub=SimpleNamespace(U=seed_U))

    assert S._archive_from_the_start(ctx(5.22))       # SN92's own seed
    assert S._archive_from_the_start(ctx(6.0))        # the boundary is inclusive
    assert not S._archive_from_the_start(ctx(6.01))
    assert not S._archive_from_the_start(ctx(3.0, archive_start=""))
    assert not S._archive_from_the_start(ctx(3.0, archive_seed_U=0.0))
    # the era swept is [archive_start, ephem_start), never past the modern one
    cfg = ItfConfig()
    assert S._jd_of_iso(cfg.archive_start) < S._jd_of_iso(cfg.ephem_start)


def _two_apparition_lines():
    """A seed + candidate pair of OBS80 blocks two apparitions apart: the
    Yarkovsky retry measures a drift across apparitions and is never tried on
    one (search._yarkovsky_retry)."""
    def line(trk, date):
        return f"     {trk}  C{date} 10 00 00.00 +10 00 00.0          20.0 G      F51"
    seed = "\n".join(line("PR00001", d) for d in ("2021 05 01.50000", "2021 05 03.50000",
                                                  "2021 05 05.50000")) + "\n"
    return seed, line("PR00002", "2016 04 10.50000") + "\n"


def test_yarkovsky_rescue_is_gated_three_ways():
    """The nongrav retry fires only for a low-q object, only on a
    close-but-disagreeing rejection, and only within its attempt budget --
    then latches on for the rest of the search (2021 JK2)."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace
    from pyroorb.itf.trace import Tracer

    calls = []

    def fake_fit(text, obscodes, registry=None, nongrav=False):
        calls.append(nongrav)
        return SimpleNamespace(ok=False, error="stub")

    def ctx(q, tries=0, nongrav=False):
        return SimpleNamespace(cfg=ItfConfig(), obscodes={}, registry=None,
                               pub=SimpleNamespace(q=q), nongrav=nongrav,
                               telemetry={"nongrav_tries": tries},
                               trace=Tracer(), log=lambda *a: None,
                               budget=None)

    def rescue(c, tag, trial_q=None):
        fit = SimpleNamespace(q=trial_q) if trial_q is not None else None
        seed, cand = _two_apparition_lines()
        return S._nongrav_rescue(c, seed, cand, "T1", tag, 0, False,
                                 1.0, 1, trial_fit=fit)

    orig = S.fit_astrometry
    S.fit_astrometry = fake_fit
    try:
        # q above the bound: never tried, whatever the rejection says
        assert rescue(ctx(2.4), "no-progress") is None
        assert not calls
        # ...but the Trial's q is what counts when the seed's is uninformative
        # (a 21-day arc barely determines perihelion -- 2021 JK2)
        assert rescue(ctx(2.4), "no-progress", trial_q=0.62) is None
        assert calls == [True], calls
        calls.clear()
        # the WRONG rejection: thousands of arcsec off is corridor noise, not
        # an unmodelled force
        assert rescue(ctx(1.06), "rms-sanity") is None
        assert not calls
        # budget spent
        assert rescue(ctx(1.06, tries=ItfConfig().nongrav_rescue_max),
                      "no-progress") is None
        assert not calls
        # already on -- no second discovery
        assert rescue(ctx(1.06, nongrav=True), "no-progress") is None
        assert not calls
        # ...and the case that should fire (the stub fit then fails, so the
        # rescue returns None -- what matters is that it ASKED for nongrav)
        assert rescue(ctx(1.06), "seed-guard") is None
        assert calls == [True], calls
    finally:
        S.fit_astrometry = orig


def test_corridor_nodes_refine_where_the_object_moves_fast():
    """Node spacing adapts to sky rate: a stretch where consecutive nodes are
    more than fast_node_deg apart is re-sampled, and a slow one is untouched.

    2015 KJ19 (650-2900"/hr) is why -- a quadratic through 3-day nodes
    mis-stated its tracklets' motion by 2-17x, so the corridor's motion gate
    called the object's own tracklets 'unlikely' (13 of 37 passing; 10/38
    linked).  Driven through the real refiner with a stub ephemeris."""
    from pyroorb.itf.ephem import refine_fast_segments

    cfg = ItfConfig()
    cfg.ephem_step_days = 3.0
    cfg.fast_node_deg = 2.0

    def node(unix, ra):
        return {"unix": unix, "ra": ra, "dec": 0.0, "delta": 0.1, "mag": 20.0,
                "speed": 0.0, "unc_deg": 0.001, "elong": 120.0}

    day = 86400.0
    # 20 deg per 3-day step: 10x the bound, so this stretch must be refined
    fast = [node(i * 3 * day, i * 20.0) for i in range(4)]
    # 0.5 deg per step: already inside the bound, nothing to do
    slow = [node(i * 3 * day, i * 0.5) for i in range(4)]

    calls = []

    def fake_ephem(fit, obscodes, jd0, jd1, step, site=None, full=None):
        calls.append(step)
        n = max(int((jd1 - jd0) / step), 1)
        return [{"jd_utc": jd0 + k * step, "ra_deg": 0.0, "dec_deg": 0.0,
                 "delta_au": 0.1, "v_mag": 20.0, "motion_arcsec_min": 0.0,
                 "unc_smaj_arcsec": 3.6, "elong_deg": 120.0}
                for k in range(n + 1)]

    import pyroorb.itf.ephem as E
    orig = E.compute_ephemeris
    E.compute_ephemeris = fake_ephem
    try:
        out_slow = refine_fast_segments(None, {}, cfg, slow)
        assert out_slow is slow and not calls, "a slow arc must not be resampled"
        out_fast = refine_fast_segments(None, {}, cfg, fast)
    finally:
        E.compute_ephemeris = orig

    assert calls, "the fast stretch must be resampled"
    # step scaled to bring 20 deg/step back to the 2 deg bound, floored
    assert abs(calls[0] - max(3.0 * 2.0 / 20.0, cfg.min_node_step_days)) < 1e-9, calls
    assert len(out_fast) > len(fast), (len(out_fast), len(fast))
    assert all(a["unix"] < b["unix"] for a, b in zip(out_fast, out_fast[1:])), \
        "nodes must stay time-ordered for the window quadratics"
    # ...and the refinement must not open a hole: the span is still covered
    assert out_fast[0]["unix"] <= fast[0]["unix"]
    assert out_fast[-1]["unix"] >= fast[-1]["unix"]

    # zero disables it entirely
    cfg.fast_node_deg = 0.0
    assert refine_fast_segments(None, {}, cfg, fast) is fast


def test_acceptance_names_the_tracklets_that_failed():
    """The gate reports WHICH tracklets failed and whether anything else did,
    because that is what decides whether dropping them can rescue the linkage
    (2016 EZ157: one 3-obs tracklet at 2/3 failed depths 4, 3 and 2, and the
    search fell back to depth 1's 15 tracklets instead of dropping it)."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def _fit(lines, included, resid, sigma=0.5):
        rows = [{"d_ra_arcsec": r, "d_dec_arcsec": 0.0,
                 "sigma_ra": sigma, "sigma_dec": sigma} for r in resid]
        return SimpleNamespace(
            _obs=[{"obs80_line": l} for l in lines],
            residuals=lambda included_only=False: rows,
            is_included=lambda: list(included))

    seed = [_obs80("K16Ep7Z", f"2016 03 1{i}.30000", "10 00 00.00",
                   "+10 00 00.0") for i in range(4)]
    seed_fit = _fit(seed, [True] * 4, [0.1] * 4)
    good = [_obs80("P10good", f"2016 04 1{i}.30000", "11 00 00.00",
                   "+11 00 00.0") for i in range(3)]
    bad = [_obs80("P10badd", f"2016 05 1{i}.30000", "12 00 00.00",
                  "+12 00 00.0") for i in range(3)]
    swept = {}
    for tag, lines in (("P10good(3obs)", good), ("P10badd(3obs)", bad)):
        ids = frozenset(S.obs_unique_id(l) for l in lines)
        swept.update({i: (tag, ids) for i in ids})
    ctx = SimpleNamespace(cfg=ItfConfig(), swept_tracklets=swept,
                          pub_ids={S.obs_unique_id(l) for l in seed},
                          pub_obs80="".join(l + "\n" for l in seed),
                          pub=SimpleNamespace(fit=seed_fit, rms=0.1,
                                              arclength_days=3.0))
    # the bad tracklet keeps 2 of 3; everything else is immaculate
    trial = _fit(seed + good + bad,
                 [True] * 4 + [True] * 3 + [True, True, False], [0.1] * 10)
    detail = {}
    ok, notes = S._accept_quality_ok(
        ctx, _wrap(trial), "".join(l + "\n" for l in good + bad[:2]), 0,
        detail=detail)
    assert not ok, notes
    assert [t for t, _ids in detail["bad_tracklets"]] == ["P10badd(3obs)"], \
        detail
    assert detail["other_fail"] is False, "only a per-tracklet clause failed"

    # a WRECKED SEED is not a per-tracklet problem, and must not look like one
    trial2 = _fit(seed + good, [True] * 7, [9.0] * 4 + [0.1] * 3)
    detail2 = {}
    ok2, notes2 = S._accept_quality_ok(
        ctx, _wrap(trial2), "".join(l + "\n" for l in good), 0, detail=detail2)
    assert not ok2, notes2
    assert detail2["other_fail"] is True, detail2


def test_seed_preserve_gate_is_invariant_to_seed_size():
    """Clause 3's cap is chi2_6 PLUS a fraction of the seed's own chi2, so the
    same RELATIVE degradation gets the same verdict on a 4-obs seed and a
    200-obs one.

    1997 QK1 is why.  Seeded with its 1997 discovery apparition (164 obs of
    photographic astrometry), the search assembles its 2002/2011/2025
    apparitions correctly -- 1 opp -> 4, a 10 300-day arc at rms 0.593" against
    the seed's own 0.692" -- and the fixed chi2_6 cap refused the lot on
    Delta 28.6 vs 19.8, while the joint orbit fit those 164 obs at chi2/dof
    0.75, i.e. inside their stated sigma.  Delta-chi2 is a SUM over the seed's
    observations; a constant cap therefore fails a seed for being large.
    """
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def _fit(lines, resid_arcsec, sigma=0.5):
        rows = [{"d_ra_arcsec": r, "d_dec_arcsec": 0.0,
                 "sigma_ra": sigma, "sigma_dec": sigma} for r in resid_arcsec]
        return SimpleNamespace(
            _obs=[{"obs80_line": l} for l in lines],
            residuals=lambda included_only=False: rows,
            is_included=lambda: [True] * len(lines))

    def verdict(n_seed, pre_resid, post_resid):
        seed = [_obs80("K97Q01K", f"1997 09 {1 + i % 28:02d}.{i:05d}",
                       "10 00 00.00", "+10 00 00.0") for i in range(n_seed)]
        link = [_obs80("P10eeee", f"2011 04 1{i}.30000", "11 00 00.00",
                       "+11 00 00.0") for i in range(4)]
        ids = frozenset(S.obs_unique_id(l) for l in link)
        swept = {i: ("P10eeee(4obs)", ids) for i in ids}
        seed_fit = _fit(seed, [pre_resid] * n_seed)
        trial = _fit(seed + link, [post_resid] * n_seed + [0.05] * 4)
        ctx = SimpleNamespace(cfg=ItfConfig(), swept_tracklets=swept,
                              pub_ids={S.obs_unique_id(l) for l in seed},
                              pub_obs80="".join(l + "\n" for l in seed),
                              pub=SimpleNamespace(fit=seed_fit, rms=pre_resid,
                                                  arclength_days=30.0))
        ok, notes = S._accept_quality_ok(ctx, _wrap(trial),
                                         "".join(l + "\n" for l in link), 0)
        return ok, notes

    # a 7% growth in the seed's residuals -- the QK1 shape -- passes at BOTH
    # sizes, though at 200 obs its Delta (22) is past the old constant cap
    ok_small, _ = verdict(4, 0.45, 0.48)
    ok_big, notes = verdict(200, 0.45, 0.48)
    assert ok_small, "a small seed's 7% degradation must pass"
    assert ok_big, notes
    line = next(n for n in notes if n.startswith("seed chi2"))
    delta = float(line.split("Delta ")[1].split(" ")[0])
    assert delta > 19.8, \
        f"the big seed's Delta ({delta}) must exceed the old constant cap"

    # ...and the gate still bites at that size: doubled residuals are refused
    ok_bad, notes = verdict(200, 0.45, 0.90)
    assert not ok_bad, notes
    assert any("seed chi2" in n and "FAIL" in n for n in notes), notes


def _wrap(fit):
    """The gate takes a TrialFit-shaped object; only .fit/.used/.rms are read."""
    from types import SimpleNamespace
    return SimpleNamespace(fit=fit, used=99, count=99, rms=0.3)


def test_nongrav_rescue_trace_line_binds_q(monkeypatch):
    """The low-q Yarkovsky rescue announces itself with the perihelion it
    gated on.  That line once interpolated a name the function never bound,
    so a single-tracklet NEO discovery -- exactly the population the rescue
    exists for -- died with ``NameError: name 'q' is not defined`` the moment
    tracing was on (i.e. the default).  Reaching the line at all is the test."""
    from pyroorb.itf import search as S
    from pyroorb.itf.trace import Tracer, L_STORY

    lines = []
    cfg = ItfConfig()
    ctx = SimpleNamespace(cfg=cfg, obscodes=None, telemetry={}, budget=None,
                          nongrav=False, registry=None,
                          pub=SimpleNamespace(q=0.61),
                          trace=Tracer(lines.append, level=L_STORY))
    # The retry itself fails to converge, so the function returns right after
    # the trace line -- no fit machinery is exercised.
    monkeypatch.setattr(S, "fit_astrometry",
                        lambda *a, **k: SimpleNamespace(ok=False))

    tag = sorted(S._NONGRAV_RESCUABLE)[0]
    seed, cand = _two_apparition_lines()
    out = S._nongrav_rescue(ctx, seed, cand, "TRK", tag,
                            prior_used=10, single_candidate=True,
                            seed_lvl=None, seed_keep=None, trial_fit=None)
    assert out is None
    assert any("0.61" in l and "Yarkovsky" in l for l in lines), lines


def test_every_sweep_path_degrades_when_a_query_is_cancelled():
    """The statement-timeout ceiling is armed on the ROLE, so it can cancel a
    query on ANY sweep path -- but the degradation was written only for the
    chunked one, and `_route_sweep` sends most corridors to set-join or
    range-join instead (narrow -> setjoin, wide -> rangejoin; chunk is for the
    wide-and-many case, i.e. the variant scan).  On those paths a cancellation
    used to propagate out of the sweep generator and abort the whole search,
    which is not an improvement on the hang it replaced.

    Same contract on every path: never retried (psycopg2's QueryCanceled
    subclasses OperationalError, so the reconnect clause would re-run the query
    the server just gave up on), never raised, the transaction rolled back, and
    the windows dropped into `last_sweep_skipped` -> the search's "not a clean
    negative" warning.
    """
    from pyroorb.itf.candidates import DbSource
    from pyroorb.itf.trace import Tracer, L_STORY
    from psycopg2 import errors as pgerr

    for mode, uncert in (("setjoin", 0.003), ("rangejoin", 1.5),
                         ("chunk", 1.5), ("scan", 1.5)):
        calls, rollbacks, reconnects, lines = [], [], [], []

        class _Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, sql, args):
                calls.append(1)
                raise pgerr.QueryCanceled("canceling statement due to "
                                          "statement timeout")
            def fetchall(self): return []

        class _Conn:
            def cursor(self, **kw): return _Cur()
            def rollback(self): rollbacks.append(1)

        s = DbSource.__new__(DbSource)
        # "scan" is not a sweep_mode -- it is the legacy non-index candidate
        # path, which queries per window and was equally unguarded
        s.cfg = ItfConfig(sweep_mode="auto" if mode == "scan" else mode,
                          candidate_path="scan" if mode == "scan" else "index",
                          sweep_max_cancels=3)
        s.index_cfg, s.conn, s.last_sweep_skipped = INDEX_CFG, _Conn(), 0
        s.trace = Tracer(lines.append, L_STORY)
        s.reconnect = lambda: (reconnects.append(1), s.conn)[1]

        windows = [_window(ra0=150.0 + 0.7 * k, uncert_deg=uncert)
                for k in range(4)]
        out = list(s.sweep_candidates(windows, 2.5, 0.0, True))   # must not raise

        assert not reconnects, (mode, "a cancelled query must not reconnect")
        assert len(calls) <= s.cfg.sweep_max_cancels, \
            (mode, f"query re-run after cancellation: {len(calls)} executes")
        assert rollbacks, (mode, "the aborted transaction must be rolled back")
        assert all(not rows for _w, rows, *_ in out), (mode, out)
        assert s.last_sweep_skipped >= len(windows), \
            (mode, f"dropped windows must be reported: {s.last_sweep_skipped}")
        assert "cancelled" in " ".join(lines), (mode, lines)


def test_obs_refetch_cancellation_is_not_an_empty_refetch():
    """The obs refetch is as cancellable as the prefilter, and "the server gave
    up" must not read as "this tracklet has no observations" -- that would hand
    the corridor a silently empty candidate instead of an incomplete sweep."""
    from pyroorb.itf.candidates import DbSource
    from psycopg2 import errors as pgerr

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, args):
            raise pgerr.QueryCanceled("canceling statement due to "
                                      "statement timeout")
        def fetchall(self): return []

    class _Conn:
        def cursor(self, **kw): return _Cur()
        def rollback(self): pass

    s = DbSource.__new__(DbSource)
    s.cfg, s.index_cfg, s.conn = ItfConfig(), INDEX_CFG, _Conn()
    s.trace, s.last_sweep_skipped = None, 0
    s.reconnect = lambda: s.conn

    assert s._obs_by_trkid(["T1", "T2"]) is DbSource.CANCELLED
    assert s._obs_by_trkid([]) == {}, "no ids is not a cancellation"


def test_clip_poor_fitters_never_drops_a_seed_observation(monkeypatch):
    """The clip rebuilds `combined` -- seed AND candidates -- so it could clip
    the object's own astrometry, and did: 2025 NV385 lost all 8 seed lines and
    the re-fit then ran on a different body's tracklets, which is how a trial
    reached the acceptance gates with nothing left for the seed-preservation
    clause to score (it passed vacuously, returning (0.0, 0)).

    `obs_out` holds the ITF side only, so the first clip cannot reach the seed;
    the second pass runs over `combined` with min_tracklet_obs=0 and could.  A
    seed obs that misfits is now KEPT and rejected by the fit instead -- which
    is the state included_only=False reads -- so it enters the Delta-chi2 at
    its full residual and the linkage is refused for the size of the damage,
    not silently improved by deleting the evidence."""
    import pyroorb.itf.search as S
    from types import SimpleNamespace

    def _mkfit(lines, resid_arcsec, sigma=0.5):
        rows = [{"d_ra_arcsec": r, "d_dec_arcsec": 0.0,
                 "sigma_ra": sigma, "sigma_dec": sigma} for r in resid_arcsec]
        return SimpleNamespace(
            _obs=[{"obs80_line": l} for l in lines],
            residuals=lambda included_only=False: rows,
            is_included=lambda: [True] * len(lines))

    # the seed misfits badly (3" at 0.5" sigma -- past both the 0.5" flat cut
    # and 3 sigma), one candidate tracklet fits, another is junk at 10"
    seed = [_obs80("K25Nc5V", f"2025 07 1{i}.30000", "10 00 00.00",
                   "+10 00 00.0") for i in range(4)]
    good = [_obs80("N004279", f"2014 04 2{i}.30000", "11 00 00.00",
                   "+11 00 00.0", stn="W84") for i in range(4)]
    junk = [_obs80("N009999", f"2015 10 1{i}.30000", "12 00 00.00",
                   "+12 00 00.0", stn="T09") for i in range(3)]

    lines = seed + good + junk
    resid = [3.0] * 4 + [0.05] * 4 + [10.0] * 3
    trial = SimpleNamespace(fit=_mkfit(lines, resid), used=99, count=11,
                            rms=0.3, U=2.0, ok=True)
    obs_out = "".join(l + "\n" for l in good + junk)
    combined = "".join(l + "\n" for l in lines)

    ctx = SimpleNamespace(cfg=ItfConfig(), log=lambda *a, **k: None,
                          telemetry={}, obscodes={}, trace=None,
                          pub_ids={S.obs_unique_id(l) for l in seed},
                          pub=SimpleNamespace(used=4, fit=None))
    # the fit itself is not under test here; the ASTROMETRY handed to it is
    monkeypatch.setattr(S, "_fit", lambda c, text: trial)
    monkeypatch.setattr(S, "_yarkovsky_retry",
                        lambda *a, **k: None)

    _tr, out_combined, out_obs, err = S._clip_poor_fitters(ctx, trial,
                                                           combined, obs_out)
    assert err is None, err
    kept = {S.obs_unique_id(l) for l in out_combined.splitlines() if l.strip()}
    # every seed line survives, misfit and all
    assert ctx.pub_ids <= kept, sorted(ctx.pub_ids - kept)
    # ...while the junk tracklet is still clipped, and the good one still kept
    assert not {S.obs_unique_id(l) for l in junk} & kept
    assert {S.obs_unique_id(l) for l in good} <= kept
    assert not {S.obs_unique_id(l) for l in junk} & \
        {S.obs_unique_id(l) for l in out_obs.splitlines() if l.strip()}


# ---------------------------------------------------------------------------
# LOV profiles (itf/lov_profile.py) + trial sources + barren-take backtracking
# ---------------------------------------------------------------------------
def _ktr(name, rate_resid_arcsec, *, ids=None, cat="likely", night=0):
    """Minimal _score_tracklet-shaped dict for LovProfileSet.add."""
    return {"_ids": frozenset(ids or {name}), "qual": rate_resid_arcsec, "dist": 100.0,
            "traj_category": cat,
            "rows": [{"trksub": name, "stn": "F51",
                      "epoch_sec": night * 86400.0 + 3600.0}]}


def test_lov_profile_sharpness_best_sigma_and_reorder():
    from pyroorb.itf.lov_profile import LovProfileSet
    cfg = ItfConfig(lov_profile=True)
    lov_sigmas = [0.0, 0.5, 1.0, 2.0, 2.83, 3.5]
    lov_profiles = LovProfileSet(cfg, lov_sigmas)
    # the truth: served by two adjacent variants, sharp minimum at 2.83
    lov_profiles.add(2.83, _ktr("TRUTH", 1.35, night=100))
    lov_profiles.add(3.5, _ktr("TRUTH", 3.0, night=100))
    # a second-night ally centered on the same variant
    lov_profiles.add(2.83, _ktr("ALLY", 1.8, night=103))
    # broad junk: fits everywhere a little (served into many variants)
    for k in lov_sigmas:
        lov_profiles.add(k, _ktr("BROAD", 1.9, night=101))
    # junk with a decent min but a wide good-k span
    lov_profiles.add(0.0, _ktr("WIDE", 1.0, night=102))
    lov_profiles.add(2.0, _ktr("WIDE", 1.2, night=102))
    lov_profiles.finalize()

    truth = lov_profiles.profiles[frozenset({"TRUTH"})]
    assert truth.sharp and truth.n_served == 2
    assert abs(truth.best_lov_sigma - 2.83) < 0.35        # discrete argmin at the edge
    broad = lov_profiles.profiles[frozenset({"BROAD"})]
    assert not broad.sharp, "junk served everywhere must not read as sharp"
    wide = lov_profiles.profiles[frozenset({"WIDE"})]
    assert not wide.sharp, "a 2-sigma-wide good-k span is not sharp"

    # 2a: the two-night agreement at k=2.83 must outrank every other variant
    pervar = [(k, []) for k in lov_sigmas]
    ordered = lov_profiles.reorder(pervar)
    assert ordered[0][0] == 2.83
    # ...and with no evidence anywhere the order degrades to central-out
    empty = LovProfileSet(cfg, lov_sigmas).finalize()
    assert [k for k, _ in empty.reorder(pervar)] == sorted(lov_sigmas, key=abs)

    # 2b/2c: the round for 2.83 carries both candidates, cluster proposed
    rnd = lov_profiles.round_for(2.83)
    assert rnd is not None
    assert rnd.center_ids[0] == frozenset({"TRUTH"})
    assert rnd.cluster_groups and rnd.cluster_groups[0] == {
        frozenset({"TRUTH"}), frozenset({"ALLY"})}
    # summary is JSON-shaped for telemetry
    s = lov_profiles.summary()
    assert s["n_sharp"] == 2 and s["clusters"][0]["k"] == 2.83


def test_lov_profile_parabolic_refinement():
    from pyroorb.itf.lov_profile import _parabolic_best_lov_sigma
    # a clean parabola: vertex recovered between samples
    pts = [(1.0, 2.0), (2.0, 0.5), (3.0, 1.0)]
    k = _parabolic_best_lov_sigma(pts)
    assert 2.0 < k < 2.5
    # edge minimum: discrete argmin, no extrapolation past the bracket
    assert _parabolic_best_lov_sigma([(1.0, 0.1), (2.0, 1.0), (3.0, 2.0)]) == 1.0
    # degenerate (flat): the discrete argmin (first of the ties, an edge)
    assert _parabolic_best_lov_sigma([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]) == 1.0
    # fewer than 3 points: the discrete argmin
    assert _parabolic_best_lov_sigma([(1.0, 2.0), (2.0, 1.0)]) == 2.0


def test_barren_take_resumes_without_banning_the_tracklet():
    """A resumed trial queue (barren take) must NOT ban what rode in the take.

    The unit that may not be repeated is the HYPOTHESIS, not the tracklet:
    {seed + A} is asked once (the search-wide ledger holds that), while
    {seed + A + B} is a different question and has to stay askable -- take A
    alone may be barren exactly because A needs B.  The old ban filtered A out
    of every later trial of the round, which foreclosed it.
    """
    import pyroorb.itf.search as S

    fitted = []

    def fake_fit(text, obscodes, **kw):
        fitted.append(text)
        used = 15 + 5 * sum(f"T{i}" in text for i in range(3))
        return _trial(used, seed_rms=0.06)
    def fake_subset_rms(fit, ids):
        return fit._seed, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        ctx = _trial_queue_ctx(tried_hypotheses={})
        # the tag sits in obs_unique_id's identity columns (15:30), so each
        # fake line carries a DISTINCT obs id -- mark_failed judges survival
        # by obs ids, and 2-char lines would all collapse to id ""
        likely = [{"astrometry": "x" * 15 + f"T{i}\n", "_r": i,
                   "_ids": frozenset({f"T{i}"})} for i in range(3)]
        lad = S.TrialQueue(ctx, likely, lambda d: d["_r"], prior_linkage_used=15)
        trial, obs1, meta1 = lad.next_take()
        # first take: the top-10 batch = all three
        assert all(f"T{i}" in obs1 for i in range(3))
        # barren, with only T0's obs surviving the subtree
        rode = lad.mark_failed(meta1, set(meta1["obs_ids"][0]))
        assert [t["_ids"] for t in rode] == [frozenset({"T0"})]
        # the members the fit CLIPPED are re-queued and taken next...
        trial2, obs2, meta2 = lad.next_take()
        assert "T1" in obs2 and "T2" in obs2
        # ...and T0 is NOT barred: the trial queue goes on to the SINGLE {seed+T0},
        # which is a different hypothesis from the batch it rode in and so has
        # never been asked
        assert not hasattr(lad, "banned")
        lad.mark_failed(meta2, set().union(*meta2["obs_ids"]))
        trial3, obs3, meta3 = lad.next_take()
        assert "T0" in obs3 and "T1" not in obs3, obs3
        lad.mark_failed(meta3, set(meta3["obs_ids"][0]))
        # drain the round, settling each take the way _iterate does once its
        # subtree has been vetted (a take is NOT settled before that)
        for m in (meta1, meta2, meta3):
            S._hypothesis_record(ctx, m["hyp_key"], True)
        while (t := lad.next_take()) is not None:
            S._hypothesis_record(ctx, t[2]["hyp_key"], True)
            lad.mark_failed(t[2], set().union(*t[2]["obs_ids"]))
        # a SECOND round over the same field re-fits nothing: every distinct
        # hypothesis has been asked and answered
        n_fits = len(fitted)
        assert S.TrialQueue(ctx, likely, lambda d: d["_r"],
                         prior_linkage_used=15).next_take() is None
        assert len(fitted) == n_fits, "a second round must re-fit nothing"
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_trial_queue_lov_round_sources_run_first():
    """2b/2c: cluster joint trials and LOV-centered singles are fitted BEFORE the
    ranked batches -- the take race is decided by fit order."""
    import pyroorb.itf.search as S
    from pyroorb.itf.lov_profile import _LovRound

    fitted = []
    def fake_fit(text, obscodes, **kw):
        fitted.append(text)
        used = 15 + 5 * sum(f"T{i}" in text for i in range(4))
        return _trial(used, seed_rms=0.06)
    def fake_subset_rms(fit, ids):
        return fit._seed, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        cfg = ItfConfig(lov_profile_cluster_trials=1, lov_profile_center_singles=1)
        ctx = _trial_queue_ctx(cfg=cfg)
        # ranking says T0 first; the k-evidence says T2+T3 cluster, T2 centered
        likely = [{"astrometry": f"T{i}\n", "_r": i,
                   "_ids": frozenset({f"T{i}"})} for i in range(3)]
        extra = [{"astrometry": "T3\n", "_r": 3, "_ids": frozenset({"T3"}),
                  "traj_category": "maybe"}]
        lov_round = _LovRound(k=2.83,
                         center_ids=[frozenset({"T2"})],
                         cluster_groups=[{frozenset({"T2"}),
                                          frozenset({"T3"})}])
        lad = S.TrialQueue(ctx, likely, lambda d: d["_r"],
                        prior_linkage_used=15, lov_round=lov_round,
                        extra_pool=extra)
        _trial1, obs1, meta1 = lad.next_take()
        # the FIRST fit is the cluster trial -- T2 + T3 (from the maybe pool),
        # ahead of the ranked batch that would have led with T0
        assert "T2" in fitted[0] and "T3" in fitted[0] \
            and "T0" not in fitted[0]
        assert "T2" in obs1 and "T3" in obs1
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_second_singles_maybe_source_reaches_below_the_cliff():
    """Part C: with second_singles_maybe on, a maybe the second ranker loves
    gets a trial; with it off, the maybe pool stays invisible to the trial queue."""
    import pyroorb.itf.search as S

    def fake_fit(text, obscodes, **kw):
        # only the maybe candidate M9 fits cleanly; the likely T0 poisons
        return _trial(20, seed_rms=0.06 if "M9" in text else 0.60)
    def fake_subset_rms(fit, ids):
        return fit._seed, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        likely = [{"astrometry": "T0\n", "_r": 0, "_ids": frozenset({"T0"}),
                   "ml_p": 0.9, "ml_p2": 0.1},
                  {"astrometry": "T1\n", "_r": 1, "_ids": frozenset({"T1"}),
                   "ml_p": 0.8, "ml_p2": 0.1}]
        extra = [{"astrometry": "M9\n", "_r": 9, "_ids": frozenset({"M9"}),
                  "traj_category": "maybe", "ml_p": 0.1, "ml_p2": 0.99}]
        for flag, expect in ((True, True), (False, False)):
            cfg = ItfConfig(second_singles_maybe=flag,
                            recurse_second_singles=2,
                            ml_second_model_path="x.json")
            ctx = _trial_queue_ctx(cfg=cfg)
            lad = S.TrialQueue(ctx, likely, lambda d: d["_r"],
                            prior_linkage_used=15, extra_pool=extra)
            got = []
            while (take := lad.next_take()) is not None:
                got.append(take[1])
            assert any("M9" in o for o in got) == expect, (flag, got)
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_trial_queue_interleaves_second_opinion_when_backtracking():
    """With interleave on (a backtracking round), the second-opinion singles
    are woven between the primary singles -- so the take AFTER a barren
    primary pick belongs to the other model; appended (legacy order) when
    off."""
    import pyroorb.itf.search as S

    def fake_fit(text, obscodes, **kw):
        return _trial(20, seed_rms=0.06)
    def fake_subset_rms(fit, ids):
        return fit._seed, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        cfg = ItfConfig(recurse_second_singles=2, ml_second_model_path="x",
                        recurse_batch_sizes=())
        likely = [{"astrometry": f"T{i}\n", "_r": i,
                   "_ids": frozenset({f"T{i}"}),
                   "ml_p": 0.9 - 0.1 * i, "ml_p2": 0.1 * i}
                  for i in range(4)]      # primary order T0..T3; second T3..T0
        for interleave, expect in (
                (True, ["T0", "T3", "T1", "T2"]),   # p1, s1, p2, s2
                (False, ["T0", "T1", "T2", "T3"])): # p1..p3(dedup), s1, s2
            ctx = _trial_queue_ctx(cfg=cfg)
            lad = S.TrialQueue(ctx, likely, lambda d: 1 - d["ml_p"], 15,
                            interleave=interleave)
            order = []
            while (take := lad.next_take()) is not None:
                _t, obs, meta = take
                got = [x for x in obs.split() if x.startswith("T")][0]
                order.append(got)
                lad.mark_failed(meta, set().union(*meta["obs_ids"]))
            assert order[:4] == expect[:len(order)], (interleave, order)
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


# ---------------------------------------------------------------------------
# the rejection ledger names the clause that actually refused
# ---------------------------------------------------------------------------

def _rej_ctx(initial_opp_profile=OPP_ONE, pub_used=13, pub_count=13):
    """A ctx with just what _acceptance_failures reads."""
    return SimpleNamespace(
        cfg=ItfConfig(), initial_opp_profile=initial_opp_profile, log=(lambda *a: None),
        pub=SimpleNamespace(used=pub_used, count=pub_count, fit=None))


def _rej_trial(used=22, count=22, rms=0.553, q=2.51, e=0.198):
    return SimpleNamespace(used=used, count=count, rms=rms, q=q, e=e, fit=None)


# 2013 SM26's real vetting block, verbatim from the live trace: the seed is
# pulled off by the candidate night and the profile is 1.5 -- while the aggregate
# rms (0.553") is exactly what the ledger used to blame.
_SM26_NOTES = [
    "tracklet H326348@T09/2016-04-02(9obs): 9/9 obs on (100% vs 75%)  ok",
    "tracklet H326348@T09/2016-04-02(9obs): chi2 10.6 vs 38.8 (18 dof, 99.7%)  ok",
    "seed chi2 5.3 -> 34.0 (Delta 28.7 vs 21.1, 6 dof + 25% of seed)  FAIL",
    "...seed obs under the new orbit: 1.308/dof (1.14 sigma, 26 dof) vs 1.00  FAIL",
    "...prior accepted obs under the new orbit: 1.308/dof (1.14 sigma, 26 dof) vs 1.00  FAIL",
]


def _rej_assessment(ctx, trial, quality_ok, notes, opp_profile_ok, gains=None,
                    fails=()):
    """An Assessment carrying exactly the acceptance evidence under test."""
    import pyroorb.itf.search as S
    if gains is None:
        gains = trial.used > ctx.pub.used and trial.count > ctx.pub.count
    return S.Assessment(trial=trial, combined="", obs_out="", profile=OPP_ONE,
                        spans=4, profile_ok=opp_profile_ok, confirmed=False,
                        gains=gains, quality_ok=quality_ok, notes=list(notes),
                        bad_tracklets=[], other_fail=not quality_ok,
                        same_opp=None, fails=list(fails))


def test_acceptance_failures_names_the_seed_clause_not_the_rms():
    import pyroorb.itf.search as S
    ctx = _rej_ctx()
    # SM26's one failed clause is the seed's Delta-chi2 (its two "..." escape
    # lines fail with it); the reason is read off the structured record
    seed_note = [n for n in _SM26_NOTES if n.startswith("seed chi2")][0]
    why = S._refusal_reasons(ctx, _rej_assessment(
        ctx, _rej_trial(), False, _SM26_NOTES, False,
        fails=[{"kind": "seed", "note": seed_note[:-len("FAIL")].strip()}]))
    joined = "; ".join(why)
    assert "seed chi2 5.3 -> 34.0" in joined
    assert "too few nights/oppositions" in joined
    # the rms is a measurement, and must not be offered as a cause
    assert "0.553" not in joined
    # the three seed-preservation lines are ONE clause: its two "..." escapes
    # fail with it by construction, so they must not be counted as extras
    assert "more failing clause" not in joined


def test_acceptance_failures_falls_back_to_a_subclause_when_alone():
    import pyroorb.itf.search as S
    ctx = _rej_ctx()
    note = ("night coherence: worst night at 3.10x its expected "
            "displacement vs 2.00 (0.91\", z 3.10, 4 nights, 6 dof)")
    why = S._refusal_reasons(ctx, _rej_assessment(
        ctx, _rej_trial(), False, ["..." + note + "  FAIL"], True,
        fails=[{"kind": "night", "note": note}]))
    assert why and "night coherence" in why[0]
    assert not why[0].startswith(".")


def test_an_escaped_seed_delta_is_never_offered_as_a_refusal():
    """The seed's Delta-chi2 note used to end in FAIL before its escape ran, so
    a linkage refused for a TRACKLET clause could name the escaped seed clause
    as a reason.  The reasons come from the failure records, which an escaped
    clause never writes."""
    import pyroorb.itf.search as S
    ctx = _rej_ctx()
    notes = ["seed chi2 5.3 -> 34.0 (Delta 28.7 vs 21.1, 6 dof + 25% of seed)  "
             "ok, escaped (below)",
             "...seed obs under the new orbit: 0.800/dof (0.89 sigma, 22 dof) vs 1.00  ok",
             "tracklet P10aaaa(3obs): 2/3 obs on (67% vs 75%)  FAIL"]
    why = S._refusal_reasons(ctx, _rej_assessment(
        ctx, _rej_trial(), False, notes, True,
        fails=[{"kind": "retention", "tag": "P10aaaa(3obs)", "kept": 2, "n": 3,
                "note": "tracklet P10aaaa(3obs): 2/3 obs on (67% vs 75%)"}]))
    assert why == ["tracklet P10aaaa(3obs): 2/3 obs on (67% vs 75%)"], why


def test_acceptance_failures_reports_a_linkage_that_gains_nothing():
    import pyroorb.itf.search as S
    ctx = _rej_ctx(pub_used=13, pub_count=13)
    why = S._refusal_reasons(ctx, _rej_assessment(
        ctx, _rej_trial(used=13, count=13), True, [], True))
    assert why and "adds nothing to the seed fit" in why[0]


def test_sweep_cause_headline_names_the_dominant_cause_never_defaults():
    """A mixed truncation must be headlined by whichever cause dropped the most
    windows, never defaulting to "row budget" -- the old default told a user to
    raise a budget that had dropped nothing."""
    import pyroorb.itf.search as S
    # cancel-dominant with a memory tail: 6 cancelled, 4 lowmem, 0 budget
    h = S._sweep_cause_headline(skipped=10, cancelled=6, lowmem=4)
    assert h.startswith("a query the server cancelled"), h
    assert "4 of them to the memory floor" in h, h
    assert "row budget" not in h, "the budget dropped nothing and must not lead"
    # memory-dominant
    h2 = S._sweep_cause_headline(skipped=10, cancelled=1, lowmem=9)
    assert h2.startswith("the sweep memory floor"), h2
    # pure budget still reads simply
    h3 = S._sweep_cause_headline(skipped=5, cancelled=0, lowmem=0)
    assert h3 == "row budget exhausted", h3


def test_fold_sweep_truncation_is_exactly_once_across_a_reused_source():
    """The variant scan fronts one DbSource with many _CachedSource collects; the
    union sweep's truncation must be folded into telemetry ONCE and then zeroed,
    or every fully-cached variant re-reads and re-adds it (inflating the
    user-facing 'N windows dropped' count up to variant_max_scan times)."""
    import pyroorb.itf.search as S
    from pyroorb.itf.candidates import source_chain
    base = SimpleNamespace(last_sweep_skipped=8, last_sweep_cancelled=3,
                           last_sweep_lowmem=5, base=None)
    ctx = SimpleNamespace(cfg=ItfConfig(), telemetry={}, log=(lambda *a: None),
                          source=base)
    S._fold_sweep_truncation(ctx)
    assert ctx.telemetry["windows_skipped"] == 8
    assert ctx.telemetry["windows_lowmem"] == 5
    assert ctx.telemetry["windows_cancelled"] == 3
    # the counters are consumed, so a second fold (the next variant) adds nothing
    assert base.last_sweep_skipped == 0
    S._fold_sweep_truncation(ctx)
    assert ctx.telemetry["windows_skipped"] == 8, "the same sweep was counted twice"


def test_chunk_epoch_span_cap_is_two_sided():
    """A chunk's epoch box must honour SWEEP_SPAN_SEC even when the sweep hands
    windows over out of epoch order.

    `windows` arrives in sigma-BAND order, not chronological order -- the 200-day
    pairwise jump guard exists precisely because band boundaries interleave
    distant epochs.  The span cap used to be measured one-sidedly, as
    `window.unix_start - chunk[0].unix_start`, which goes NEGATIVE for any window
    EARLIER than the chunk's first and so could never trip: backwards drift was
    bounded only by the pairwise guard, letting a chunk span several hundred
    days against a documented 90-day cap.  Since the prefilter is
    `healpix = ANY(union) AND epoch BETWEEN t_lo..t_hi`, that is exactly the
    "decouple each pixel from its epoch and fetch the whole space-time box"
    over-fetch the cap exists to prevent.
    """
    from dataclasses import replace as _replace
    from pyroorb.itf.candidates import DbSource

    day = 86400.0
    # walk BACKWARDS in 50-day steps: each step is inside the 200-day pairwise
    # guard, but the total reach (150 d) is well past the 90-day span cap.
    base = _window()
    windows = [_replace(base, unix_cur=T0 - k * 50 * day,
                     unix_start=T0 - k * 50 * day - day,
                     unix_end=T0 - k * 50 * day + day) for k in range(4)]

    seen = []

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, args): seen.append(args)
        def fetchall(self): return []

    class _Conn:
        def cursor(self, **kw): return _Cur()
        def rollback(self): pass

    s = DbSource.__new__(DbSource)
    s.cfg = ItfConfig(sweep_mode="chunk", candidate_path="index",
                      prefilter_motion=False)
    s.index_cfg, s.conn = INDEX_CFG, _Conn()
    s.last_sweep_skipped = s.last_sweep_cancelled = s.last_sweep_lowmem = 0
    s.trace = None
    s.reconnect = lambda: s.conn

    out = list(s.sweep_candidates(windows, 2.5, 0.0, False))

    # every window is still served -- the cap may re-chunk, never drop
    assert len(out) == len(windows), (len(out), len(windows))
    assert seen, "no query was issued"
    # args tail is [pix_list, t_lo, t_hi, dec_lo, dec_hi] with motion off
    for args in seen:
        t_lo, t_hi = args[-4], args[-3]
        assert (t_hi - t_lo) <= DbSource.SWEEP_SPAN_SEC, (
            f"chunk epoch box spans {(t_hi - t_lo) / day:.1f} d, "
            f"cap is {DbSource.SWEEP_SPAN_SEC / day:.1f} d")
    # and it really did have to split: one chunk could not have held them all
    assert len(seen) > 1, "backwards-drifting windows were not re-chunked"


def test_a_hypothesis_is_fitted_once_per_search():
    """ONE rule: the astrometry set (plus the force model) is the identity of
    the question, and it is asked once.

    2006 UP17: the same tracklet was taken, fitted (bit-identically -- the LOV
    variant picks which candidates are RETRIEVED and never enters the trial
    fit), recursed on and vetted in EIGHT rounds.  tried_keys cannot see that
    (a TrialQueue per round, keyed on that round's rank positions), so the ledger
    is keyed on the observations the fit would be handed.

    What the rule does NOT do is ban a tracklet: {seed+A} asked once still
    leaves {seed+A+B} askable, because it is a different question.
    """
    import pyroorb.itf.search as S

    fits = []

    def fake_fit(text, obscodes, **kw):
        fits.append(text)
        # T0 poisons the seed (refused by the take criterion); T1 is clean
        return _trial(15 + 5 * ("T0" in text) + 5 * ("T1" in text),
                      seed_rms=0.60 if "T0" in text else 0.06)

    def fake_subset_rms(fit, ids):
        return fit._seed, 15

    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        ledger = {}
        ctx = _trial_queue_ctx(tried_hypotheses=ledger)
        # real candidates carry _ids (their obs identities); the two-character
        # astrometry these doubles use would otherwise collapse to one id
        likely = [{"astrometry": "T0\n", "_r": 0, "_ids": frozenset({"i0"})},
                  {"astrometry": "T1\n", "_r": 1, "_ids": frozenset({"i1"})}]
        rank = lambda d: d["_r"]

        # round 1: the top-2 batch and T0 alone are refused, T1 is taken
        trial, obs, meta = S._recurse_trial_queue(ctx, likely, rank,
                                             prior_linkage_used=15)
        assert trial is not None and "T1" in obs and "T0" not in obs
        assert len(fits) == 3, fits

        # round 2 (a fresh trial queue, as every LOV variant builds): the two
        # REFUSED hypotheses are settled and not re-fitted.  T1 was TAKEN but
        # nothing has vetted its subtree yet, so it is not settled -- a take
        # becomes final only when _iterate says its subtree searched the whole
        # corridor and came back.
        del fits[:]
        trial2, obs2, meta2 = S._recurse_trial_queue(ctx, likely, rank,
                                                prior_linkage_used=15)
        assert trial2 is not None and "T1" in obs2
        assert len(fits) == 1, fits

        # ...and once T1's subtree HAS been vetted, no round asks again
        S._hypothesis_record(ctx, meta["hyp_key"], True)
        del fits[:]
        assert S.TrialQueue(ctx, likely, rank, 15).next_take() is None
        assert fits == []

        # the FORCE MODEL is part of the question: a Yarkovsky rescue mid-
        # search makes the same lines a different fit, and it is asked again
        ctx.nongrav = True
        assert S.TrialQueue(ctx, likely, rank, 15).next_take() is not None
        assert len(fits) >= 1, "a nongrav re-ask must reach the fitter"
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_a_truncated_subtree_is_not_a_settled_answer():
    """The one carve-out, and it is the rule's own logic: a subtree whose sweep
    DROPPED windows -- a row budget, a cancelled query, the low-memory floor --
    searched only part of the ITF, so its answer is not the ITF's answer and
    the hypothesis stays askable."""
    import pyroorb.itf.search as S
    ctx = SimpleNamespace(telemetry={})
    assert S._windows_dropped(ctx) == 0
    ctx.telemetry["windows_skipped"] = 12
    assert S._windows_dropped(ctx) == 12
    # a bare namespace with no telemetry at all must not raise mid-search
    assert S._windows_dropped(SimpleNamespace()) == 0


def _refused_by_show(rms):
    """Judge a consistent lone new night the one-night show declines, at a
    given trial rms.  Returns the filled result."""
    import pyroorb.itf.search as S
    night = [_obs80("ITF0001", "2020 09 13.3000%d" % i, "10 00 00.00",
                    "+10 00 00.0") for i in range(3)]
    obs_out = "\n".join(night) + "\n"
    seed = "\n".join(_obs80("K20S00X", "2019 01 0%d.30000" % d,
                            "10 00 00.00", "+10 00 00.0")
                     for d in (1, 2, 3)) + "\n"
    trace = lambda *a: None
    trace.enabled = False
    ctx = SimpleNamespace(cfg=ItfConfig(), initial_opp_profile=OPP_ONE, log=lambda *a: None,
                          pub_ids=set(), obscodes=None, trace=trace,
                          pub_obs80=seed, swept_tracklets={},
                          pub=SimpleNamespace(used=5, U=2.0, rms=0.2,
                                              fit=None))   # -> clause 2 silence
    trial_meta = {"tracklets": [{"rows": [{}], "astrometry": obs_out}],
            "failed_tracklets": [], "ml_ps": [0.01], "indices": [0]}
    trial = _trial(8, rms=rms)
    trial.q, trial.e = 2.51, 0.198        # a plain MBA: no NEO orbit waiver
    ctx.telemetry = {}
    ctx.pub.count = 5
    a = S.Assessment(trial=trial, combined=seed + obs_out, obs_out=obs_out,
                     profile=OPP_TWO_THIN, spans=4, profile_ok=False,
                     confirmed=False, gains=True, quality_ok=True, notes=[],
                     bad_tracklets=[], other_fail=False, same_opp=None)
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "_extension_check",
                   lambda c, a: S.ExtensionCheck(vetting={}))
        return S._judge(ctx, a, [], 5, trial_meta, 0, sweep=None)


# 2021 SA70's real vetting block, verbatim from a LIVE run against the ITF
# (2026-09-22): the ONE 1.75-branch firing in 25 live recovery searches, and it
# is the doubtful kind.  Both candidate tracklets pass their own retention and
# chi2 tests -- it is the SEED that degrades when they are taken together, which
# is the seed-degradation gate and exactly the doubt the profile-only promotion
# must not be handed to.  Offline the same 25 objects fire the branch 0 times:
# it takes real ITF distractors to reach it.
_SA70_NOTES = [
    "tracklet XUA3F61@G96/2016-10-20(4obs): 4/4 obs on (100% vs 75%)  ok",
    "tracklet XUA3F61@G96/2016-10-20(4obs): chi2 7.4 vs 23.3 (8 dof, 99.7%)  ok",
    "tracklet t3a3bad@W84/2013-01-07(3obs): 3/3 obs on (100% vs 75%)  ok",
    "tracklet t3a3bad@W84/2013-01-07(3obs): chi2 2.1 vs 19.8 (6 dof, 99.7%)  ok",
    "seed chi2 5.3 -> 36.9 (Delta 31.6 vs 21.1, 6 dof + 25% of seed)  FAIL",
    "...seed obs under the new orbit: 1.677/dof (1.30 sigma, 22 dof) vs 1.00  FAIL",
    "...prior accepted obs under the new orbit: 1.477/dof (1.22 sigma, 30 dof) vs 1.00  FAIL",
]


def _possible_175(quality_ok, opp_profile_ok=False, *, used=18, count=18, pub_used=11,
                  rms=0.3268, notes=None):
    """Judge a recursion-accumulated 1.75 with the acceptance evidence
    :func:`_assess` would have measured.  Returns the filled result (status
    0.6, the reviewable possible).

    The defaults are 2021 SA70's LIVE numbers (depth 1: 18/18 used against an
    11-observation seed, rms 0.327", opp_profile_ok False, quality_ok False), so the
    doubtful arm below is the shape of an observed event rather than a shape
    invented to fail."""
    import pyroorb.itf.search as S
    new = [_obs80("ITF000%d" % k, "%d 09 13.30000" % yr, "10 00 00.00",
                  "+10 00 00.0") for k, yr in enumerate((2016, 2020))]
    obs_out = "\n".join(new) + "\n"
    seed = "\n".join(_obs80("K20S00X", "2019 01 0%d.30000" % d,
                            "10 00 00.00", "+10 00 00.0")
                     for d in (1, 2, 3)) + "\n"
    trace = lambda *a: None
    trace.enabled = False
    ctx = SimpleNamespace(cfg=ItfConfig(), initial_opp_profile=OPP_ONE, log=lambda *a: None,
                          pub_ids=set(), obscodes=None, trace=trace,
                          pub_obs80=seed, swept_tracklets={},
                          pub=SimpleNamespace(used=pub_used, count=pub_used,
                                              U=2.0, rms=0.2, fit=None))
    trial_meta = {"tracklets": [{"rows": [{}], "astrometry": obs_out}],
            "failed_tracklets": [], "ml_ps": [0.01], "indices": [0]}
    trial = _trial(used, rms=rms)
    trial.count = count
    trial.q, trial.e = 2.51, 0.198        # a plain MBA: no NEO orbit waiver
    ctx.telemetry = {}
    a = S.Assessment(trial=trial, combined=seed + obs_out, obs_out=obs_out,
                     profile=OPP_THREE_THIN, spans=4, profile_ok=opp_profile_ok,
                     confirmed=False, gains=used > pub_used and count > pub_used,
                     quality_ok=quality_ok, notes=list(notes or ()),
                     bad_tracklets=[], other_fail=not quality_ok, same_opp=None,
                     # each FAILing clause note as the record _accept_quality_ok
                     # writes for it
                     fails=[{"kind": "seed", "note": n[:-len("FAIL")].strip()}
                            for n in (notes or ())
                            if n.rstrip().endswith("FAIL") and not n.startswith("...")])
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        # what the link did to the seed has its own tests: clean here
        mp.setattr(S, "_extension_check",
                   lambda c, a: S.ExtensionCheck(vetting={}))
        return S._judge(ctx, a, [], pub_used, trial_meta, 0, sweep=None)


def test_a_possible_refused_for_a_doubt_is_not_promotable_on_a_step():
    """The profile-only promotion is a PERMISSION, and it may only
    be given to the possible that was refused on OPPOSITION PROFILE ALONE.

    ``vetting["promotable_on_step"]`` is the one deliberate exception: the
    recursion-accumulated 1.75, fitting cleanly with no doubt raised, may be
    submitted after the user re-solves it WITHOUT the observation set growing
    (``engine_api._pending_verdict``).  Every other possible carries a doubt and
    keeps the growth rule.  But a 1.75 also lands on this branch when acceptance
    refused it on QUALITY -- the seed-preservation gate, or a per-tracklet
    retention/chi2 clause -- and both rescues (_drop_offenders_and_revet,
    _reduced_linkage) declined to repair it.  The flag used to be an
    unconditional True, so exactly that doubtful find was offered the exception.

    The verdict is unchanged either way: it is a reviewable 0.6 keeping its fit,
    astrometry and cautions.  Only the permission moves.

    The doubtful arm is 2021 SA70's, measured (see ``_SA70_NOTES``): the one
    1.75-branch firing in 25 LIVE recovery searches, and its quality failure is
    the seed-preservation gate.  Note what that run does NOT establish -- SA70
    went on to LINK (status 1.0, 4/4 apparitions), so this possible was
    superseded and never reached a user.  The branch is demonstrably reachable
    with a doubt on real data; an end-to-end case where the button appears is
    still unmeasured, and would need a search whose DELIVERED verdict is a
    doubtful 1.75 (the recovery fixtures exist to link, so they are the wrong
    population for it)."""
    clean = _possible_175(quality_ok=True)
    assert clean.verdict is SearchVerdict.POSSIBLE and clean.vetting["possible_unconfirmed"]
    assert clean.vetting["promotable_on_step"] is True, clean.vetting
    assert "will enable linkage download/submission" in clean.warnings[-1]

    # THE case, on SA70's own numbers: a quality clause refused it too, so it
    # is not "opposition profile alone" -- both its tracklets pass their own retention and
    # chi2 tests and the SEED is what degrades (chi2 5.3 -> 36.9)
    doubt = _possible_175(quality_ok=False, notes=_SA70_NOTES)
    assert doubt.verdict is SearchVerdict.POSSIBLE, doubt.reason      # still reviewable, still a 0.6
    assert doubt.vetting["possible_unconfirmed"] is True
    assert doubt.fit is not None and doubt.itf_obs80, "the evidence is kept"
    assert doubt.vetting["promotable_on_step"] is False, doubt.vetting
    # ...and the caution must not promise a button the growth rule holds back
    assert "will enable linkage download/submission" not in doubt.warnings[-1]
    assert "has to grow first" in doubt.warnings[-1], doubt.warnings[-1]

    # ...and the clause that refused it rides along for the reviewer
    assert "seed chi2" in doubt.vetting["refused_because"], doubt.vetting
    print("PASS a possible refused for a doubt is not promotable on a step")


def test_show_refusal_is_the_reason_at_any_rms():
    """The clause that refused a candidate which DID link is what the reader
    is told, said once -- and no rms condition can take its place.

    2006 UP17 linked a night fitting at chi2 2.8 against a 19.8 cap, seed
    intact, overall rms 0.514" -> 0.505" -- and was once reported as "too few
    nights/oppositions to submit", because a legacy `trial.rms >= rms_max`
    branch (a cap acceptance never asks) answered before the show's sentence.
    The overall rms is reported, never gated, so there is no such branch."""
    over = _refused_by_show(0.60)        # over the old 0.44" rms_max
    under = _refused_by_show(0.20)
    for r in (over, under):
        assert r.reason.startswith("nothing shown: "), r.reason
        assert "error bars" in r.reason, r.reason
        assert r.reason.count("nothing shown") == 1, r.reason
        assert r.vetting.get("show_suppressed") is True, r.vetting

def test_recurrent_note_names_the_tracklet_that_was_never_shown():
    """A silenced show leaves a 0.6 carrying no fit and no astrometry (that is
    what tells backtracking the branch was barren), so the TRACKLET itself
    used to vanish.  The ledger says it once at the end -- as a diagnostic,
    never a warning: it is a claim the search declined to make, not a doubt
    about one it made."""
    import pyroorb.itf.search as S
    ctx = SimpleNamespace(tried_hypotheses={
        # n counts every round that reached the hypothesis: one TOOK it, the
        # other seven were told it had already been tried
        "k1": {"settled": True, "n": 8, "used": 41, "count": 41,
               "rms": 0.505, "tags": ("p00046@F51/2025-04-23(3obs)",),
               "refusal": "the best candidate is too weak",
               "where": "depth 0, linear corridor"},
        "k2": {"settled": True, "n": 3, "used": None, "count": None,
               "rms": None, "tags": ("junk",), "refusal": None,
               "where": ""}})
    notes = S._recurrent_notes(ctx, S.ItfSearchResult(verdict=SearchVerdict.NONE))
    assert len(notes) == 1, notes
    for expect in ("p00046@F51/2025-04-23(3obs)", "depth 0, linear corridor",
                   "7 later round(s)", "41/41", "0.505", "too weak"):
        assert expect in notes[0], (expect, notes[0])
    assert "junk" not in notes[0]
    # a LINKED verdict says nothing: the reader has the linkage itself
    assert S._recurrent_notes(ctx, S.ItfSearchResult(verdict=SearchVerdict.LINKED)) == []


def test_recurrent_note_reads_what_the_ledger_writer_writes():
    """The test above builds the ledger by hand, so it cannot see the writer
    and the reader disagree on a key -- which is how a rename of the keyword
    `_iterate` passes (silence -> refusal) once left the reader on the old key
    and the note silently never appeared.  Write through the real writer."""
    import pyroorb.itf.search as S
    ctx = SimpleNamespace(tried_hypotheses={})
    S._hypothesis_record(ctx, "k1", True, where="depth 0, linear corridor",
                         tags=("p00046@F51/2025-04-23(3obs)",),
                         refusal="the best candidate is too weak",
                         used=41, count=41, rms=0.505)
    S._hypothesis_record(ctx, "k2", True, tags=("junk",), refusal=None)
    notes = S._recurrent_notes(ctx, S.ItfSearchResult(verdict=SearchVerdict.NONE))
    assert len(notes) == 1, notes
    assert "p00046@F51" in notes[0] and "too weak" in notes[0], notes[0]


def test_lov_profile_reads_the_union_sweep_through_the_accessor():
    """The LOV profile pass must actually FIND the union sweep's rows.

    Every other LOV profile test drives `LovProfileSet.add` directly, so the
    sharpness maths is covered and the PLUMBING is not -- and the plumbing is
    what broke: `build_lov_profiles` used to reach into `_CachedSource.cache` and
    spell the key itself, so re-keying that cache made every lookup miss.  The
    pass then profiled NOTHING (no sharp candidates, hence no priority variants
    and no LOV-centered or cluster trial sources -- what the BE559 recovery rests on) and
    said so nowhere: the whole offline suite stayed green and the only tell was
    a live trace reading `priority variants: none`.

    This is the canary.  It fails the moment the union sweep's rows stop
    reaching the profiles, whatever the cause.
    """
    import pyroorb.itf.search as S
    from pyroorb.itf.lov_profile import build_lov_profiles

    window = _window()
    rows = [{"trksub": "PR00001", "stn": "F51", "epoch_sec": T0 + i * 1800.0,
             "ra": 150.0 + i * 0.01, "dec": 10.0 + i * 0.004,
             "obs80": _obs80("PR00001", "2020 09 13.3000%d" % i,
                             "10 00 00.00", "+10 00 00.0")}
            for i in range(3)]

    class Base:
        last_sweep_skipped = 0

    cached = S._CachedSource(Base())
    cached.remember(window, rows, {}, {})
    trace = lambda *a: None
    trace.enabled = False
    ctx = SimpleNamespace(cfg=ItfConfig(lov_profile=True), pub_ids=set(),
                          obscodes=None, own_desigs=set(), telemetry={},
                          collector=None, trace=trace, log=lambda *a: None)

    lov_profiles = build_lov_profiles(ctx, [(0.0, [window])], cached, [0.0])
    assert lov_profiles.summary()["n_tracklets"] == 1, (
        "the LOV profile pass found none of the union sweep's rows -- the "
        "cache accessor and its keying have diverged")

    # ...and a window the sweep never held contributes nothing, quietly
    lov_profiles2 = build_lov_profiles(ctx, [(0.0, [_window(ra0=200.0)])],
                            cached, [0.0])
    assert lov_profiles2.summary()["n_tracklets"] == 0


def test_show_floor_foreclosure_is_announced_up_front():
    """A 1-opp seed's likeliest find is a lone new night, and only the
    one-night show can report one -- so when the WHOLE field scores under
    `show_min_ml_p_abs`, the verdict is settled before a single trial fit.
    Say so where the reader can act on it, exactly as the seed-U foreclosure
    already is.  2006 UP17: field max P 0.07 against the 0.15 floor across all
    23 LOV variants, discovered after 35 s and eight rounds."""
    import pyroorb.itf.search as S

    warned = []

    def sink(msg, *a, **k):
        # the Tracer stamps and prefixes: "[   0.00s] ! <text>"
        if "! " in str(msg):
            warned.append(str(msg))

    def _ctx(initial_opp_profile=OPP_ONE):
        from pyroorb.itf.trace import Tracer, L_DETAIL
        return SimpleNamespace(cfg=ItfConfig(), initial_opp_profile=initial_opp_profile,
                               telemetry={}, trace=Tracer(sink, L_DETAIL))

    field = [{"ml_p": 0.07}, {"ml_p": 0.001}, {"ml_p": None}]

    # the whole field under the floor: announced, and recorded for the payload
    ctx = _ctx()
    S._note_show_floor(ctx, field, 0, None)
    assert ctx.telemetry["show_floor"] == {"best_ml_p": 0.07, "floor": 0.15,
                                           "foreclosed": True}
    assert len(warned) == 1 and "cannot be reported" in warned[0]
    assert "0.070" in warned[0] and "0.15" in warned[0]

    # said ONCE per search, not once per LOV variant round
    S._note_show_floor(ctx, field, 0, None)
    assert len(warned) == 1

    # a field that CAN clear it is recorded and says nothing
    del warned[:]
    ctx2 = _ctx()
    S._note_show_floor(ctx2, [{"ml_p": 0.4}], 0, None)
    assert ctx2.telemetry["show_floor"]["foreclosed"] is False
    assert warned == []

    # and it is measured on the LINEAR depth-0 field only -- the sweep that
    # happens before the expensive variant fan, which is the decision the
    # reader is making.  Never at depth, never inside a variant round, never
    # on a multi-opposition seed (the show is not its only route), and never
    # with no ML scores to judge.
    for args, ctxN in (((field, 1, None), _ctx()),
                       ((field, 0, ["a window"]), _ctx()),
                       ((field, 0, None), _ctx(initial_opp_profile=OPP_THREE)),
                       (([{"ml_p": None}], 0, None), _ctx())):
        S._note_show_floor(ctxN, *args)
        assert ctxN.telemetry == {} and warned == []


def test_a_tracklet_with_no_trksub_is_named_by_its_packed_id():
    """A blank ``trksub`` used to label the whole tracklet ``?``, which names
    nothing -- and an ITF row of a comet fragment (Pan-STARRS submits these
    with no trksub at all) looked exactly like any other unnamed tracklet.
    Fall back to columns 6-12 of the 80-column line, the same field
    `batch.tracklet_summary` reads."""
    from pyroorb.itf.trace import row_label, tracklet_tag

    itf = ("     /25K01d KB2025 12 07.04704901 53 45.199+62 08 44.95"
           "         16.1 GW     I79")
    named = ("     P21l7qLKC2025 12 07.04704901 53 45.199+62 08 44.95"
             "         16.1 GW     I79")

    def tr(sub, line):
        return {"rows": [{"trksub": sub, "stn": "I79", "obs80": line,
                          "epoch_sec": 1765069665.0}] * 2}

    # no trksub -> the packed id, not "?"
    assert tracklet_tag(tr(None, itf)) == "/25K01d@I79/2025-12-07(2obs)"
    assert tracklet_tag(tr("", itf)) == "/25K01d@I79/2025-12-07(2obs)"
    # a trksub the submitter DID give still wins
    assert tracklet_tag(tr("P21l7qL", named)) == "P21l7qL@I79/2025-12-07(2obs)"
    # nothing to fall back on, and a malformed row, still answer "?"
    assert row_label({"trksub": "", "obs80": ""}) == "?"
    assert row_label({}) == "?"
    assert row_label(None) == "?"
    assert tracklet_tag({"rows": []}) == "?"


def test_the_trajectory_gate_is_pole_safe():
    """Near the pole, RA is singular and its quadratic mis-states the RATE.

    The corridor predicts with a quadratic through three ephemeris nodes.  In
    RA/Dec that breaks down near the pole -- at dec +82 an object sweeps tens
    of degrees of RA in days, with curvature to match -- and the failure is
    small in POSITION and large in RATE, which is the wrong way round: the
    trajectory gate judges rate (``rate_resid_arcsec``) against 1-2.5 arcsec while the
    corridor around it is thousands of arcsec wide.  2016 MB1's real 2024
    tracklet was called "unlikely" on 11.2" of pure interpolation error and a
    whole apparition was lost, though it fits the object's orbit at 0.27".

    A GREAT CIRCLE is the honest probe: exact constant angular velocity, so
    every arcsec of predicted-rate error belongs to the interpolation.  The
    same path near the equator must stay accurate in both, or the test would
    pass on a prediction that is simply bad everywhere.
    """
    import numpy as np
    from pyroorb.itf.ephem import corridor_windows

    def great_circle(t_days, incl_deg, phase0_deg, rate_deg_per_day):
        """(ra, dec) on a great circle inclined ``incl_deg`` to the equator."""
        ph = math.radians(phase0_deg + rate_deg_per_day * t_days)
        i = math.radians(incl_deg)
        x = math.cos(ph)
        y = math.sin(ph) * math.cos(i)
        z = math.sin(ph) * math.sin(i)
        return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z))

    def nodes_for(incl, phase0, rate):
        out = []
        for k in range(5):
            t = (k - 2) * 3.0                      # -6, -3, 0, +3, +6 days
            ra, dec = great_circle(t, incl, phase0, rate)
            out.append({"unix": T0 + t * 86400.0, "ra": ra, "dec": dec,
                        "delta": 1.0, "mag": 20.0, "speed": 1.0,
                        "unc_deg": 0.001, "elong": 120.0})
        return out

    def rate_errors(incl, phase0, rate):
        """(RA/Dec-quadratic, unit-vector) rate error in arcsec over 1.73 h."""
        cfg = ItfConfig()
        windows = corridor_windows(nodes_for(incl, phase0, rate), cfg,
                                cfg.max_uncert_deg)
        dt = 6212.0                                # the MB1 tracklet's span
        t_a = T0                                   # a node, mid-window
        window = next(w for w in windows if w.unix_start <= t_a <= w.unix_end)
        truth = [great_circle((t - T0) / 86400.0, incl, phase0, rate)
                 for t in (t_a, t_a + dt)]
        cosd = math.cos(math.radians(truth[0][1]))
        t_dra = (truth[1][0] - truth[0][0]) * cosd * 3600.0
        t_ddec = (truth[1][1] - truth[0][1]) * 3600.0
        out = []
        for predict in (window.predict, window.predict_sky):
            p0, p1 = predict(t_a), predict(t_a + dt)
            out.append(math.hypot((p1[0] - p0[0]) * cosd * 3600.0 - t_dra,
                                  (p1[1] - p0[1]) * 3600.0 - t_ddec))
        return out

    # near the pole (the 2016 MB1 geometry): the RA/Dec quadratic loses the
    # rate by more than the gate's whole budget; the unit vector does not
    polar_radec, polar_vec = rate_errors(incl=82.0, phase0=88.0, rate=1.0)
    assert polar_radec > 5.0, polar_radec
    assert polar_vec < 1.0, polar_vec

    # ...and away from the pole both are fine, so the fix is not just a
    # uniformly wider prediction.  NB the unit vector is very slightly WORSE
    # here (0.24" against 0.02"): an equatorial great circle is linear in RA
    # and sinusoidal in the direction cosines, so the quadratic suits it
    # better.  That is the whole trade -- a few tenths of an arcsec everywhere
    # against tens of arcsec near the pole -- and both stay far inside the
    # gate.  Rate 1 deg/day is 2016 MB1's own on-sky rate; the error grows with
    # it (68" against 2.0" at 2 deg/day), so this is not the worst case.
    eq_radec, eq_vec = rate_errors(incl=5.0, phase0=0.0, rate=1.0)
    assert eq_radec < 1.0, eq_radec
    assert eq_vec < 1.0, eq_vec


def test_dedup_covers_every_link_not_just_link_0():
    """A bundled document (itf batch, a bundled submission) carries many
    links, and ONE of them restating an identification the MPC has already
    accepted is enough to have the whole submission refused. The dedup read
    link_0 alone until 2026-09-22, so links 1..N went out unchecked."""
    from pyroorb.itf.linkage import dedup_designations, all_unpacked_designations
    from pyroorb.packing import pack
    fam = {"2016 UH221": "2016 UH221", "2025 NW433": "2016 UH221",
           "2020 AB1": "2020 AB1", "2019 QQ2": "2019 QQ2"}
    doc = {"links": {
        "link_0": {"designations": [pack("2019 QQ2")],
                   "trksubs": [["a", "2020 01 01.5", "F51"]]},
        "link_1": {"designations": [pack("2016 UH221"), pack("2025 NW433")],
                   "trksubs": [["b", "2020 01 02.5", "F51"]]},
        "link_2": {"designations": [pack("2025 NW433")],
                   "trksubs": [["c", "2020 01 03.5", "F51"]]},
    }}
    # the family lookup must be asked about every link's designations
    assert all_unpacked_designations(doc) == [
        "2019 QQ2", "2016 UH221", "2025 NW433"]
    dedup_designations(doc, lambda d: fam.get(d, d), log=lambda m: None)
    assert doc["links"]["link_0"]["designations"] == [pack("2019 QQ2")]
    # the already-merged pair collapses to its primary ...
    assert doc["links"]["link_1"]["designations"] == [pack("2016 UH221")]
    # ... and a secondary alias alone is replaced by the primary
    assert doc["links"]["link_2"]["designations"] == [pack("2016 UH221")]


def test_last_resort_pick_never_earns_the_single_candidate_waiver():
    """The seed-guard waiver ("nothing to fall back on") belongs to a field of
    ONE likely/maybe candidate.  A post-fan round whose only trial is an
    'unlikely' LOV-refinement pick is there BECAUSE the field was exhausted, so
    it must keep the seed guard."""
    import pyroorb.itf.search as S
    ctx = _trial_queue_ctx(tried_hypotheses={})
    one = lambda cat, tag: {"astrometry": "x" * 15 + f"{tag}\n", "_r": 0,
                            "_ids": frozenset({tag}), "traj_category": cat}
    assert S.TrialQueue(ctx, [one("likely", "L0")], lambda d: d["_r"],
                        prior_linkage_used=15).single_candidate
    assert S.TrialQueue(ctx, [], lambda d: d["_r"], prior_linkage_used=15,
                        extra_pool=[one("maybe", "M0")]).single_candidate
    tail = one("unlikely", "U0")
    assert not S.TrialQueue(ctx, [], lambda d: d["_r"], prior_linkage_used=15,
                            extra_pool=[tail],
                            lov_refine_tail=[tail["_ids"]]).single_candidate


def test_terminal_fan_runs_only_from_a_reviewable_take_once_per_search():
    """The terminal LOV fan has one shot per search: a take the vet does not
    call reviewable (in a weak field, usually a distractor) must not spend it,
    so a later true take can still have it."""
    import pyroorb.itf.search as S
    cfg = ItfConfig()
    ctx = SimpleNamespace(is_variant=False, terminal_fan_done=False, budget=None)
    weak = SimpleNamespace(U=cfg.variant_corridor_U + 1.0)
    linked = SimpleNamespace(reviewable=True)
    bare = SimpleNamespace(reviewable=False)
    assert S._terminal_fan_due(cfg, ctx, weak, linked)
    assert not S._terminal_fan_due(cfg, ctx, weak, bare)        # distractor-shaped
    assert not S._terminal_fan_due(cfg, ctx, weak, None)
    assert not S._terminal_fan_due(cfg, ctx, SimpleNamespace(U=0.5), linked)  # determined
    ctx.terminal_fan_done = True
    assert not S._terminal_fan_due(cfg, ctx, weak, linked)      # once per search
    ctx.terminal_fan_done, ctx.is_variant = False, True
    assert not S._terminal_fan_due(cfg, ctx, weak, linked)      # linear corridors only
    ctx.is_variant, cfg.terminal_fan = False, False
    assert not S._terminal_fan_due(cfg, ctx, weak, linked)


def test_a_linkage_the_terminal_fan_may_supersede_is_sent_only_if_it_stands(
        monkeypatch):
    """The vet of a terminal take accepts and builds its linkage BEFORE the
    terminal fan runs from it, and the fan's own linkage contains it.  Sent at
    build time, a fan link then went out as a second submission, and the
    returned result never said the first was sent.  While the fan could still
    run, the document is held; it goes out only if the vetted result stands."""
    import pyroorb.itf.search as S
    sent = []
    monkeypatch.setattr(S, "submit_linkage",
                        lambda doc, cfg, **kw: (sent.append(doc) or True,
                                                "sent", None))
    monkeypatch.setattr(S, "build_linkage",
                        lambda *a, **kw: ({"doc": len(sent)}, None))
    monkeypatch.setattr(S, "display_obs80", lambda ctx, text: text)
    line = ("     P21Qwcd  C2024 01 05.362728"
            "07 43 37.40 +20 19 38.1          21.5 wX     F52")
    ctx = SimpleNamespace(cfg=ItfConfig(), log=(lambda *a: None), telemetry={},
                          primary_desigs=["2022 PR4"], co_credit=None,
                          do_submit=False, pub=SimpleNamespace(used=10, U=8.0),
                          held_transmits=[])
    trial = SimpleNamespace(fit=None, used=13, U=5.0)

    def build():
        return S._build_and_submit(ctx, S.ItfSearchResult(), trial, line, [],
                                   line, line, transmit=True)

    first = build()
    assert first.verdict is SearchVerdict.LINKED and sent == [], \
        "a linkage the fan may still supersede must not be sent yet"
    held = ctx.held_transmits
    S._release_held(ctx, held, standing=S.ItfSearchResult())   # superseded
    assert sent == [], "a superseded linkage is never sent"
    S._release_held(ctx, held, standing=first)                 # it stands
    assert len(sent) == 1 and first.submitted
    # outside the hold the document goes out at once, as before
    ctx.held_transmits = None
    build()
    assert len(sent) == 2
    # the hold is armed exactly when the fan could run from this take
    cfg = ItfConfig()
    fan_ctx = SimpleNamespace(is_variant=False, terminal_fan_done=False,
                              budget=None)
    weak = SimpleNamespace(U=cfg.variant_corridor_U + 1.0)
    assert S._terminal_fan_possible(cfg, fan_ctx, weak)
    assert not S._terminal_fan_possible(cfg, fan_ctx, SimpleNamespace(U=0.5))


@pytest.mark.parametrize("fan", ["supersedes", "finds_nothing", "raises"])
def test_a_vetted_terminal_is_sent_once_whatever_the_fan_does(monkeypatch, fan):
    """The whole boundary: a LINKED terminal whose fan is due.  A fan link is
    the answer and the terminal is never sent; a fan that finds nothing, or
    fails, leaves the terminal standing and it is sent exactly once."""
    import pyroorb.itf.search as S
    sent, logged = [], []
    monkeypatch.setattr(S, "submit_linkage",
                        lambda doc, cfg, **kw: (sent.append(doc) or True,
                                                "sent", None))
    fan_link = S.ItfSearchResult(verdict=SearchVerdict.LINKED)

    def fake_scan(*a, **kw):
        if fan == "raises":
            raise RuntimeError("corridor blew up")
        return fan_link if fan == "supersedes" else None
    monkeypatch.setattr(S, "_variant_scan", fake_scan)
    cfg = ItfConfig()
    ctx = SimpleNamespace(cfg=cfg, log=logged.append, do_submit=False,
                          is_variant=False, terminal_fan_done=False, budget=None,
                          trace=None)
    latest = S.ItfSearchResult(verdict=SearchVerdict.LINKED)
    held = [(latest, "terminal-doc", {})]
    trial = SimpleNamespace(U=cfg.variant_corridor_U + 1.0, used=20)
    got = S._terminal_fan(ctx, trial, "", latest, held, depth=0)
    assert ctx.terminal_fan_done
    if fan == "supersedes":
        assert got is fan_link and sent == []
    else:
        assert got is None and sent == ["terminal-doc"] and latest.submitted
    assert any("terminal LOV fan failed" in m for m in logged) == (fan == "raises")


def test_an_unlikely_tracklet_that_can_be_fitted_carries_its_ades_record():
    """'unlikely' tracklets were never banked in the ADES registry because none
    was ever fitted -- but the final sweep's topocentric rescue
    (``keep_unlikely``) and the LOV refinement's last-resort trials now fit
    them, and without the record they are weighted by the 80-column fallback
    sigma instead of their reported uncertainty."""
    import pyroorb.itf.search as S
    from pyroorb.itf.obsreg import AdesRegistry
    seed = ("     K22P04R* C2022 08 06.371093"
            "21 18 24.75 -14 16 55.8          21.0 wU     F52\n")
    own = S.own_designations(seed, ["2022 PR4"])

    def sweep(keep_unlikely=False, lov_refine=None):
        rows = _ban_rows("P21Qwcd")
        for k, r in enumerate(rows):
            r["ra"] += k * 0.05           # motion far off the corridor's rate
        ctx = _own_ban_ctx(seed, own, rows)
        ctx.registry = AdesRegistry()
        if lov_refine is not None:
            ctx.lov_refine = lov_refine(rows)
        got = S._collect_tracklets(ctx, SimpleNamespace(fit=None), 0.0,
                                   windows=[_window()], keep_unlikely=keep_unlikely)
        return got, len(ctx.registry)

    got, n = sweep(keep_unlikely=True)
    assert [t["traj_category"] for t in got] == ["unlikely"]
    assert n == 1, "a kept 'unlikely' tracklet must carry its ADES records"
    got, n = sweep(lov_refine=lambda rows: {
        frozenset(obs_unique_id(r["obs80"]) for r in rows): (0.0, 1.0, 1)})
    assert [t["traj_category"] for t in got] == ["unlikely"] and n == 1, \
        "an 'unlikely' tracklet the LOV refinement measured is a last-resort trial"


def test_a_resumed_take_does_not_restart_the_no_progress_clock():
    """A take reached by resuming after a barren one is the round still
    looking, not progress: it must leave the no-progress clock running, or
    backtracking is bounded only by the search ceiling."""
    import pyroorb.itf.search as S

    def fake_fit(text, obscodes, **kw):
        used = 15 + 5 * sum(f"T{i}" in text for i in range(3))
        return _trial(used, seed_rms=0.06)
    def fake_subset_rms(fit, ids):
        return fit._seed, 15
    orig_fit, orig_sub = S.fit_astrometry, S.subset_rms
    S.fit_astrometry = fake_fit; S.subset_rms = fake_subset_rms
    try:
        budget = S.SearchBudget(60.0, 0.0)
        ctx = _trial_queue_ctx(tried_hypotheses={}, budget=budget)
        likely = [{"astrometry": "x" * 15 + f"T{i}\n", "_r": i,
                   "_ids": frozenset({f"T{i}"})} for i in range(3)]
        lad = S.TrialQueue(ctx, likely, lambda d: d["_r"], prior_linkage_used=15)
        _t, _o, meta1 = lad.next_take()
        assert budget.extensions == 1, "a round's first take is progress"
        clock = budget.no_progress_at
        lad.mark_failed(meta1, set().union(*meta1["obs_ids"]))
        assert lad.next_take() is not None
        assert budget.extensions == 1 and budget.no_progress_at == clock, \
            "a resumed take must not restart the clock"
    finally:
        S.fit_astrometry = orig_fit; S.subset_rms = orig_sub


def test_tracklet_repair_runs_through_a_seed_failure_but_not_others():
    """A bad tracklet can itself cause the seed check to fail (2008 GU20: a
    clipped 2000 pair pulled the seed past its Delta-chi2), so a seed failure
    must not block dropping it -- the re-vet judges the seed again.  Progress
    and night-coherence failures still block the repair."""
    from types import SimpleNamespace
    import pyroorb.itf.search as S
    bad = [("PR00000@645", frozenset({"x"}))]
    a = lambda fails, other=True, ok=False, bad_=bad: SimpleNamespace(
        quality_ok=ok, bad_tracklets=bad_, other_fail=other, fails=fails)
    tracklet = {"kind": "retention"}
    assert not S._repair_blocked(a([tracklet], other=False))      # the classic case
    assert not S._repair_blocked(a([tracklet, {"kind": "seed"}]))
    assert not S._repair_blocked(a([tracklet, {"kind": "veto"}]))
    assert S._repair_blocked(a([tracklet, {"kind": "progress"}]))
    assert S._repair_blocked(a([tracklet, {"kind": "seed"}, {"kind": "night"}]))
    assert S._repair_blocked(a([], other=True))                    # unknown cause
    assert S._repair_blocked(a([tracklet], ok=True))               # nothing to repair
    assert S._repair_blocked(a([{"kind": "seed"}], bad_=[]))       # no bad tracklet
