"""Offline tests for the --ident-check identification logic (ident_check.py).

Uses the cached PAIR fixtures: a clean pair must emit `A = B`; a pair whose drops are a
PARTIAL rejection of a multi-night apparition must emit nothing (the load-bearing rule);
a pair whose drops are a whole single-night tracklet must emit `A = B except:` + the
verbatim obs lines (we inject a synthetic obs80_line, since the ADES fixtures carry none).
"""
import pytest
import os
from _fixtures import load_fixture, FIX
import pyroorb.fit_object as fo
from pyroorb import cli
from pyroorb import obs_sources
from pyroorb import ident_check

# each test refits a PAIR fixture through the full engine
pytestmark = pytest.mark.slow

_OC = None
def _oc():
    global _OC
    if _OC is None:
        _OC = obs_sources.load_obscodes()
    return _OC


def _fix(label):
    return load_fixture(os.path.join(FIX, label + ".json"))


def _fit(fx, inject_line=False):
    obs = [dict(o) for o in fx["obs"]]
    if inject_line:
        for o in obs:                       # synthetic verbatim record (cols 1-5 blank)
            o["obs80_line"] = f"     {o.get('packed','XXXXXXX'):7} C2020 01 01.00000 ...{o['obscode']}"
    return fo.fit_orbit(obs, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])


def test_clean_pair_emits_identification():
    fit = _fit(_fix("PAIR_2002QT68"))
    txt = ident_check.identify(fit)
    assert txt is not None and "=" in txt and "except" not in txt, f"expected clean A = B, got {txt!r}"
    assert fit.n_excluded() == 0


def test_systematic_residual_pairs_flagged_not_y():
    # Three pairs whose kept obs fit at an acceptable OVERALL rms (so identify() emits a clean
    # 'A = B') but whose residuals TREND systematically across an apparition -- the tell of two arcs
    # the main linker forced together. They MUST be caught by systematic_misfit so the batch
    # downgrades them from 'y' to 'y-systematic' (be skeptical) and does not auto-confirm:
    #   * 2014 HV328 = 2025 OV408 -- bogus link (2014 nights trend +1.1 -> -1.0" in Dec @ 0.50");
    #   * 2014 EW228 = 2026 FQ21  -- bogus link;
    #   * 2025 OT173 = 2014 EN155 -- a CORRECT identification, but its two 2014 nights trend, so we
    #     still surface it for a human look (a clean overall rms can hide a real systematic).
    for label in ("PAIR_2014HV328", "PAIR_2014EW228", "PAIR_2025OT173"):
        fit = _fit(_fix(label), inject_line=True)   # verbatim lines so the except block can render
        assert ident_check.identify(fit) is not None, f"{label}: should still form a clean A = B"
        assert ident_check.systematic_misfit(fit), f"{label}: systematic residual must be detected"


def test_corroborated_subfloor_trend_flagged():
    # 2021 TJ205 = 2015 KS376: a forced link whose per-night RA means trend in BOTH apparitions
    # (0.45"/0.42" at ratio 4.8x/4.6x over ~0.09" scatter) -- each single arc alone fits to ~0.07".
    # Neither apparition reaches the strict 0.5" absolute floor, so it is caught only by the
    # corroborated-apparition relaxation (two independent trends unlock SYS_BETWEEN_CORROB). It
    # forms a clean 'A = B' at 0.34" overall, exactly the kind of clean-rms mislink the audit exists
    # to catch, so systematic_misfit MUST flag it (and name the corroboration).
    fit = _fit(_fix("PAIR_2021TJ205"))
    assert ident_check.identify(fit) is not None, "should still form a clean A = B"
    desc = ident_check.systematic_misfit(fit)
    assert desc and "corroborated" in desc, f"corroborated sub-floor trend must be detected, got {desc!r}"


def test_well_sampled_coherent_offset_flagged_by_significance():
    # 2016 UJ137: the guard for the SIGNIFICANCE arm of the dominance test (SYS_SE_RATIO). All six
    # of its nights sit coherently off the orbit in RA (3-8 sigma each) and its 2017 apparition
    # clears the strict 0.5" floor at 0.53" -- but its nights carry 4-18 obs, so the mean averages
    # far below the single-observation scatter and the within-scatter test scored only 2.19x
    # (below SYS_WITHIN_RATIO). Measured against the standard ERROR of those means it is 3.7x.
    # Without the significance arm this object is silent in the O-C pane despite an obviously
    # structured residual plot -- which is exactly how the gap was found.
    fit = _fit(_fix("SYS_2016UJ137"))
    desc = ident_check.systematic_misfit(fit)
    assert desc and "RA" in desc, f"well-sampled coherent offset must be detected, got {desc!r}"
    assert "the mean's own error" in desc, f"description should report significance, got {desc!r}"


def test_subfloor_statistical_trend_is_flagged():
    # By design, a trend is real below the arcsec floors when it is statistically one.
    # 2025 NF505 = 2015 HG272 keeps 32/32 at 0.29" -- each arc alone fits 0.08"/0.06" -- while the
    # 2025 Dec means walk -0.59 -> +0.68" over four nights tight to ~0.1" (z 14, 11x the night
    # scatter). Neither arcsec floor can see that (the whole residual budget is under both), so only
    # the trend arm does. It was submitted as a clean 'y' and the MPC rejected it.
    fit = _fit(_fix("PAIR_2025NF505"))
    tr = ident_check.significant_trend(fit)
    assert tr and tr["fires"] and tr["axis"] == "Dec", tr
    assert tr["z"] >= ident_check.SYS_TREND_Z and tr["ratio"] >= ident_check.SYS_TREND_MIN_SWING_RATIO
    desc = ident_check.systematic_misfit(fit)
    assert desc and "straight line" in desc, f"sub-floor trend must be detected, got {desc!r}"
    fit.n_requested = 2
    v = ident_check.linkage_verdict(fit)
    assert not v["submittable"] and not v["clean"], v


@pytest.mark.parametrize("label", ["GUARD_2014EF94", "PAIR_2002QT68", "PAIR_2003SR221"])
def test_trend_arm_leaves_flat_nights_alone(label):
    # The trend arm asks whether the nights WALK, not whether they are displaced. GUARD_2014EF94's
    # nights sit far off (34x their SE) but flat, and the station-night tau in the noise model is
    # the displacement a correct orbit leaves -- z 3.3, under the bar. QT68 / SR221: clean links.
    fit = _fit(_fix(label))
    tr = ident_check.significant_trend(fit)
    assert tr is None or not tr["fires"], (label, tr)


def test_dense_harmless_drift_is_not_a_trend():
    # 2015 CY86 = 2026 DO29 is a REAL identification. Its 2026 apparition -- 1482 X05 obs
    # over seven nights -- walks 0.16" in Dec, the light early nights trailing off the two dense
    # ones that pin the orbit: statistically a line (z ~6), but only ~4x the 0.04" scatter within a
    # night. The magnitude bar (SYS_TREND_MIN_SWING_RATIO) is what keeps it: a correct orbit's
    # nights drift a few single-obs errors, a forced link's drift many.
    fit = _fit(_fix("PAIR_2015CY86"))
    tr = ident_check.significant_trend(fit)
    assert tr and tr["z"] >= ident_check.SYS_TREND_Z, tr      # significant...
    assert not tr["fires"], tr                               # ...but small against the nights
    assert ident_check.systematic_misfit(fit) is None
    fit.n_requested = 2
    assert ident_check.linkage_verdict(fit)["submittable"]


@pytest.mark.parametrize("label,apps", [("162385", (4, 10)), ("2003_BN4", (0, 2))])
def test_trend_arm_closest_correct_links_stay_clean(label, apps):
    # The two CORRECT links nearest the magnitude bar in the trend-significance study: two apparitions
    # of one known object fitted as designations A and B (the synthetic null). Both clear z >= 5
    # (5.2) and walk 6.9x / 5.7x their night scatter -- under SYS_TREND_MIN_SWING_RATIO (8). If the
    # noise model or the bar moves, these are the first correct links to start firing.
    import numpy as np
    from pyroorb.orbit_common import OPP_GAP_DAYS
    fx = _fix(label)
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    t = np.array([o["jd_utc"] for o in obs])
    groups = np.split(np.arange(len(t)), np.where(np.diff(t) > OPP_GAP_DAYS)[0] + 1)
    sub = [dict(obs[k], designation="A") for k in groups[apps[0]]] + \
          [dict(obs[k], designation="B") for k in groups[apps[1]]]
    meta = dict(fx["meta"], designation="A", designations=["A", "B"], combined=True,
                n_requested=2, other_designations=["B"])
    fit = fo.fit_orbit(sub, _oc(), desig="A", meta=meta)
    tr = ident_check.significant_trend(fit)
    assert tr is not None and not tr["fires"], (label, tr)


def test_hard_astrometry_pair_is_examined():
    # 2025 MT356 = 243P, a COMET, and the ONLY fixture that exercises ident_check.kept_obs_fit's
    # sigma-normalized clause: every other pair fits under MAX_KEPT_RMS in arcsec (population max
    # 0.576"), so without this object the relaxation is untested.
    #
    # The joint fit lands at 0.647", over the flat 0.6" bar that used to be an outright veto -- so
    # --ident-check filed a REAL identification as a flat 'no'. The excess is 243P's own astrometry,
    # not the link: a coma centroid plus two decades of mixed astrometry cannot deliver 0.6" for ANY
    # set of designations. Asserted rather than asserted-about: the 243P fixture is the SAME comet
    # fitted ALONE, and it is no better.
    fit = _fit(_fix("PAIR_2025MT356"))
    solo = _fit(_fix("243P"))
    assert fit.rms_residual() >= ident_check.MAX_KEPT_RMS, \
        "this fixture exists to sit ABOVE the arcsec bar -- if it no longer does, it guards nothing"
    # The heart of it: the identification does not make the fit worse. (It is in fact marginally
    # BETTER -- 0.647" vs 0.649" -- but pin the direction, not the third decimal.)
    assert fit.rms_residual() <= solo.rms_residual() * 1.05, (
        f"the link must not be what costs the rms: joint {fit.rms_residual():.3f}\" vs "
        f"243P alone {solo.rms_residual():.3f}\"")
    # ...and 2025 MT356's own obs are the TIGHTEST part of the joint fit, not the loose part.
    import numpy as np
    mask = np.asarray(fit._mask, dtype=bool)
    mt = [i for i, o in enumerate(fit._obs) if o.get("designation") == "2025 MT356" and mask[i]]
    assert mt, "2025 MT356 must contribute kept obs"
    mt_rms = float(np.sqrt(np.mean(fit._resid_all[mt] ** 2)))
    assert mt_rms < fit.rms_residual(), \
        f"the new designation fits BETTER than the arc it joins ({mt_rms:.3f}\")"
    # The verdict: the sigma clause carries it, so it reaches a human as 'y-examine' rather than
    # being filed as a non-match. (identify() still declines to auto-confirm -- an 85-reject comet
    # arc trips its tracklet-drop gates -- which is the right outcome, not a workaround.)
    ok, note = ident_check.kept_obs_fit(fit)
    assert ok and "sigma" in note, f"hard astrometry must pass the fit gate, got {ok} {note!r}"
    code, reason = ident_check.examine_or_no(fit)
    assert code == "y-examine", f"a real identification must not be a flat 'no', got {code} ({reason})"
    # and the web UI / --linkage surface must OFFER it (with cautions), not hide it
    fit.n_requested = 2
    assert ident_check.linkage_verdict(fit)["offer"], "the linkage must be offered for review"


def test_elevated_rms_clean_pair_not_systematic():
    # GUARD against over-flagging: 2003 SR221 = 2025 QK106 has an elevated overall rms (~0.47") --
    # the highest of the genuine links -- but its residuals are random scatter, NOT a per-night
    # trend, so it must NOT be flagged systematic (stays a real 'y').
    fit = _fit(_fix("PAIR_2003SR221"))
    assert ident_check.systematic_misfit(fit) is None, "random-scatter elevated rms must not be flagged"


def test_single_night_tracklet_drop_emits_except_block():
    # PAIR_2026DV62 drops exactly one whole single-night tracklet -> valid `except` block.
    fx = _fix("PAIR_2026DV62")
    fit = _fit(fx, inject_line=True)
    assert fit.n_excluded() > 0
    txt = ident_check.identify(fit)
    assert txt is not None and "except:" in txt, f"expected an except block, got {txt!r}"
    body = txt.splitlines()
    assert body[0].endswith("except:") and len(body) - 1 == fit.n_excluded(), \
        f"except block should list all {fit.n_excluded()} dropped obs"


def test_sparse_opposition_stray_tracklet_is_excepted():
    # PAIR_2025NX399 = 2015 HU337: the 2021 opposition is just TWO thin tracklets 44 d apart --
    # a stray F51 night on 2021-08-17 (4 obs) and a good night that fits (3 obs). Only the stray
    # is dropped, so this is a clean 'A = B except <one tracklet>'. The cross-path-disagreement
    # guard must judge the gutting by NIGHTS, not raw obs: the obs-count majority test wrongly
    # rejected this (the dropped night had more obs than the kept one), reading 1-of-2 stray
    # tracklets as a gutted solid arc. By nights it's a 1-1 tie, not a majority loss -> excepted.
    fx = _fix("PAIR_2025NX399")
    fit = _fit(fx, inject_line=True)
    assert fit.n_excluded() > 0
    txt = ident_check.identify(fit)
    assert txt is not None and "except:" in txt, f"expected an except block, got {txt!r}"
    body = txt.splitlines()
    assert body[0].endswith("except:") and len(body) - 1 == fit.n_excluded(), \
        f"except block should list all {fit.n_excluded()} dropped obs"


def test_same_night_same_station_other_designation_tracklet_is_excepted():
    # PAIR_2015KR274 = 2015 KR274 = 2025 PY146: a W84 night carries 5 kept 2015 KR274 obs PLUS 3
    # stray 2025 PY146 obs that sit ~1.4 deg off the joint orbit (a 2015 detection mis-filed under
    # the 2025 designation), plus a separate lone 2023 V00 tracklet. Because the strays share the
    # night AND the station with the kept obs, the old (night, obscode) tracklet grouping read them
    # as a PARTIAL gutting of an 8-obs night -> rejected (identify returned None, classified only
    # 'y-examine'). Splitting tracklets by SUBMISSION -- (night, obscode, designation), since each
    # provisional designation is its own MPC tracklet -- makes the strays their own whole tracklet,
    # so this is a clean 'A = B except <2 tracklets>' (y2). The systematic-trend audit must stay
    # quiet (the kept obs are honest random scatter, not a forced-link trend).
    fx = _fix("PAIR_2015KR274")
    fit = _fit(fx, inject_line=True)
    assert fit.n_excluded() == 6, f"expected 6 dropped strays, got {fit.n_excluded()}"
    txt = ident_check.identify(fit)
    assert txt is not None and "except:" in txt, f"expected an except block, got {txt!r}"
    body = txt.splitlines()
    assert body[0].endswith("except:") and len(body) - 1 == fit.n_excluded(), \
        f"except block should list all {fit.n_excluded()} dropped obs"
    assert ident_check.systematic_misfit(fit) is None, "kept obs are clean scatter, not a trend"


def test_two_apparition_multinight_gutting_emits_nothing():
    # PAIR_2015HC310 = 2015 HC310 = 2026 EM61: a TWO-opposition link whose BOTH oppositions are
    # multi-night (2015: 4 nights; 2026: 2 nights), which converges only by dropping a whole night
    # of the 2026 opposition. Two solid multi-night arcs that "fit" only after one is gutted are
    # disagreeing -> NOT the same object. Must emit nothing even with verbatim lines available
    # (rejected on the DECISION, not for a missing obs80 line). Contrast 2005 JK163 = 2022 MT4 =
    # 2026 HV6 (test_known_identifications_still_match), where a stray night is dropped but the
    # link spans 3 oppositions, so the OTHER clean oppositions still anchor the orbit.
    fit = _fit(_fix("PAIR_2015HC310"), inject_line=True)
    assert fit.n_excluded() > 0
    assert ident_check.identify(fit) is None, "a two-apparition multi-night gutting must NOT identify"


def test_known_identifications_still_match():
    # Real identifications that must keep producing 'A = B [except]' under their fixtures:
    # the 3-way 2005 JK163 = 2022 MT4 = 2026 HV6 (one stray night), and the tracklet+outlier
    # 2014 UO269 = 2025 OY448. (PAIR_2013CK176 = 2026 DO7 needs its ident-mode ATTRIBUTION path to
    # identify -- see test_ident_attribution_relaxes_overconfident_night -- so it is guarded there,
    # not in this fixture-meta loop where n_requested is unset.)
    # PAIR_2022YL13 exercises obscode-aware tracklets: a dropped F52 tracklet shares a night
    # with a kept G96 tracklet (a "partial night" by time alone, a whole tracklet by station).
    # (PAIR_2025OS355 / PAIR_2014HZ280 -- the HZ280=BV65=OS355 3-way -- have their OWN test below
    # that locks the except block: their F51 "duplicate" is at a DIFFERENT sky position, so it is a
    # real drop, not a benign de-duplication. See test_cross_listed_diff_position_tracklet_is_excepted.)
    # The hand-picked coverage pairs (build_fixtures): a ~17-yr y2 carrying the advisory
    # magnitude flag (2024 WY87 = 2007 PQ8), a ~12-yr multi-rev-bridge y2 (2026 GV5 = 2014 EE314),
    # a 6-opposition y2 (2016 FS162 = 2025 YA14), and a clean 2-opp baseline (2015 FL474 = 2021
    # TY207). Each must keep producing 'A = B [except]' as the gates evolve.
    # PAIR_2015KP420 (2015 KP420 = 2025 NH546) is the DEGENERATE-GATE REGRESSION GUARD: its base fit
    # lands in a poisoned-anchor garbage minimum (463"), which made _ident_doomed_degenerate fire and
    # return 'no' -- a y2 that regressed vs 4/9 days ago. The gate now runs AFTER the cheap multi-start
    # re-anchor (which recovers 25/28 @ 0.106"), so the full pipeline must once again identify it.
    for lbl in ("PAIR_2022MT4", "PAIR_2025OY448", "PAIR_2022YL13",
                "PAIR_2024WY87", "PAIR_2026GV5", "PAIR_2016FS162", "PAIR_2015FL474",
                "PAIR_2015KP420"):
        fit = _fit(_fix(lbl), inject_line=True)
        txt = ident_check.identify(fit)
        assert txt and " = " in txt, f"{lbl} should identify, got {txt!r}"
    # 2024 WY87 = 2007 PQ8 also guards the advisory magnitude-residual flag (its arcs imply H ~1.4
    # mag apart): identify() must PREPEND a '# FLAG ... mag residual' comment, and the batch turns
    # that into the classify suffix '-mag'. (Still emitted -- the flag is advisory, not a rejection.)
    wy87 = ident_check.identify(_fit(_fix("PAIR_2024WY87"), inject_line=True))
    assert wy87.startswith("# FLAG") and "mag" in wy87.splitlines()[0], \
        f"2024 WY87 = 2007 PQ8 must carry the advisory mag flag, got {wy87.splitlines()[0]!r}"


def test_cross_listed_diff_position_tracklet_is_excepted():
    # PAIR_2014HZ280 / PAIR_2025OS355 -- the 2014 HZ280 = 2018 BV65 = 2025 OS355 3-way. An F51
    # tracklet on 2021-11-01 is submitted under BOTH K14HS0Z and K18B65V at IDENTICAL station and
    # timestamps, but the two submissions sit ~28' apart on the sky -- genuinely different measured
    # positions, NOT the same detection cross-listed. The joint fit keeps the K18B65V copy and drops
    # the K14HS0Z copy. Because de-duplication requires the sky POSITION to agree too (not just
    # station + time), those 4 drops are REAL -> a clean 'A = B except <one tracklet>' (a 'y2'),
    # not a benign de-dup that would hide them into a bare 'y'. Both fixtures are the same joint fit.
    for lbl in ("PAIR_2014HZ280", "PAIR_2025OS355"):
        fit = _fit(_fix(lbl), inject_line=True)
        assert fit.n_excluded() == 4, f"{lbl}: expected the 4 diff-position F51 drops, got {fit.n_excluded()}"
        txt = ident_check.identify(fit)
        assert txt is not None and "except:" in txt, f"{lbl}: expected an except block, got {txt!r}"
        # every dropped obs is the K14HS0Z copy of the F51 tracklet
        dropped = [o for i, o in enumerate(fit._obs) if not fit.is_included()[i]]
        assert all(o["obscode"] == "F51" for o in dropped), f"{lbl}: drops must all be F51, got {dropped}"
        n_drops = sum(1 for ln in txt.splitlines() if " = " not in ln and not ln.startswith("#"))
        assert n_drops == 4, f"{lbl}: except block must list the 4 drops (a y2), got {n_drops}"


def test_duplicate_excluded_requires_position_match():
    # Unit-pin the de-dup gate: same station + same instant is NOT enough -- the sky position must
    # also coincide (within DUP_TOL_ARCSEC). Two excluded obs, each with a kept same-station,
    # same-time twin: one twin at the SAME position (a true cross-listing -> deduped/benign), the
    # other ~28' away (a genuinely different detection -> a REAL drop, not deduped).
    import numpy as np
    t = 2459519.89016
    times = [t, t, t, t]
    codes = ["F51", "F51", "F51", "F51"]
    # idx 0 excluded @ (54.22, 15.94); idx 1 kept twin at SAME position -> benign duplicate.
    # idx 2 excluded @ (54.30, 15.95); its only same-station same-time kept obs (idx 3) sits ~28'
    # away, so it has NO position-matching twin -> a REAL drop, not a duplicate.
    ras = [54.2205, 54.2205, 54.3000, 54.6885]
    decs = [15.9359, 15.9359, 15.9500, 16.0261]
    excluded = np.array([True, False, True, False])
    dup = ident_check._duplicate_excluded(times, codes, ras, decs, excluded)
    assert bool(dup[0]) is True,  "same station+time+POSITION excluded obs must be a benign duplicate"
    assert bool(dup[2]) is False, "same station+time but ~28' offset must NOT be a duplicate (a real drop)"
    assert not dup[1] and not dup[3], "kept obs are never marked duplicate"


def test_two_opp_survivor_multinight_identifies():
    # The SHARPENED two-opposition chimera ban: a 2-opp both-multinight link that clips a whole
    # night is rejected ONLY when the drops collapse an opposition to a lone kept night (or clip
    # as many nights as they keep). An opposition that stays SOLIDLY multi-night after clipping a
    # verified-bad stray is still an independent arc, so identify() must auto-confirm (a y2):
    #   * 2015 OF201 = 2025 NA385 -- 8d/12d arcs ~10 yr apart at 0.10", drops 1 of 3 nights (the
    #     bad T09 night). Held at 'y-examine' under the old zero-tolerance ban;
    #   * 2014 HH101 = 2022 DQ27 -- drops 1 of 5 nights (a whole 8-obs F51 night) at 0.09".
    # (Every documented chimera collapses a side to ONE kept night -- see the boundary test below.)
    for lbl, ndrop in (("PAIR_2015OF201", 3), ("PAIR_2014HH101", 8)):
        fit = _fit(_fix(lbl), inject_line=True)
        txt = ident_check.identify(fit)
        assert txt is not None and "except:" in txt, f"{lbl}: must auto-confirm as a y2, got {txt!r}"
        n_drops = sum(1 for ln in txt.splitlines() if " = " not in ln and not ln.startswith("#"))
        assert n_drops == ndrop, f"{lbl}: expected {ndrop} excepted obs, got {n_drops}"


def test_two_opp_collapsed_side_still_rejected():
    # Chimera BOUNDARY guard for the sharpened ban: 2015 RC182 = 2026 ES49's 2026 opposition clips
    # 2 of its 3 nights -- collapsed to a lone kept night, so nothing on that side independently
    # confirms -> identify() must stay None (with PAIR_2015HC310's 2-night-clips-1 shape covered in
    # test_two_apparition_multinight_gutting_emits_nothing).
    fit = _fit(_fix("PAIR_2015RC182"), inject_line=True)
    assert fit.n_excluded() > 0
    assert ident_check.identify(fit) is None, "an opposition collapsed to a lone night must NOT identify"


def test_clipped_night_between_two_short_arcs_is_a_flat_no():
    # THE CLIP RULE (ident_check.clip_doubt + the 2-opposition retention gate). A 2-opposition link
    # between two short, unconstrained arcs that closes ONLY by discarding a whole night out of one
    # side's own coherent record is two DIFFERENT objects -- so it is a flat 'no', not a 'y-examine'
    # candidate. The years between the apparitions do NOT rescue it: one orbit can nearly always
    # thread two short arcs once it is allowed to clip a night, which is exactly the coincidence.
    # Hand-adjudicated pairs from the ident-check batch, one per route into the rule:
    #   * PAIR_2014EO135 (= 2015 HT386)  -- clips the 3rd of 3 CONSECUTIVE W84 nights at 2.5"
    #     (a soft drop: that night plainly belongs to its own sequence);
    #   * PAIR_2014EN102 (= 2022 TB28)   -- clips a night on BOTH sides, the 2014 one at 2.7";
    #   * PAIR_2025QL180 (= 2014 HA394)  -- an 18" GROSS clip, but clipping it leaves no opposition
    #     with even a 3 d baseline (COINCIDENCE_MIN_KEPT_ARC_DAYS);
    #   * PAIR_2016FH213 (= 2025 YS17)   -- the 2025 side's 23 d arc made it look 'secure', but the
    #     clipping side keeps only 8/13 obs -> the 2-opposition retention gate;
    #   * PAIR_2025DS43  (= 2026 ER100)  -- 3 nights clipped from 2025 DS43's coherent 7-night / 25 d
    #     O18 campaign (8/14 kept): a survey run like that does not contain foreign nights.
    for lbl in ("PAIR_2014EO135", "PAIR_2014EN102", "PAIR_2025QL180",
                "PAIR_2016FH213", "PAIR_2025DS43"):
        fit = _fit(_fix(lbl), inject_line=True)
        assert fit.n_excluded() > 0, f"{lbl}: fixture should still clip obs"
        assert ident_check.identify(fit) is None, f"{lbl}: a forced clip must NOT identify"
        code, reason = cli._examine_or_no(fit)
        assert code == "no", f"{lbl}: must be a flat 'no', got {code!r} ({reason})"


def test_three_opposition_link_never_flat_no_on_clip():
    # ...but a link that threads 3+ OPPOSITIONS coherently is exempt, however many nights it clips:
    # the clean oppositions still pin the orbit, so there is something real to investigate. Guards
    # that clip_doubt() stays 2-opposition-only (PAIR_2025NM2 clips a whole 82" night across 3 opps;
    # PAIR_2014HZ280 is the 3-designation / 7-opposition HZ280 = BV65 = OS355 with 4 obs dropped).
    for lbl in ("PAIR_2025NM2", "PAIR_2014HZ280"):
        fit = _fit(_fix(lbl), inject_line=True)
        assert ident_check.clip_doubt(fit) is None, f"{lbl}: 3+ oppositions must be exempt"
        assert cli._examine_or_no(fit)[0] == "y-examine", f"{lbl}: must stay reviewable"


def test_unclipped_35_day_side_anchors_the_link():
    # The second exemption (UNCLIPPED_ANCHOR_ARC_DAYS): a side that gave up NOT ONE night and still
    # spans 35+ days anchors the orbit itself, so a clip on the other side is no longer what makes
    # the join close -- the link keeps whatever it would otherwise be, never a flat 'no'. Synthetic
    # (no fixture has this shape): 'B' is a fully-kept 40 d / 4-night arc one opposition later, 'A'
    # a 3-night arc whose last night is clipped at a modest 3" -- the exact shape clip_doubt() rejects
    # when nothing anchors it, so this pins the exemption itself, not some other gate.
    import numpy as np
    jd0 = 2461000.0

    def _fit_with_anchor(span):
        """'A' = 3 nights, its 3rd clipped at 3"; 'B' = a fully-kept 4-night arc `span` days long,
        one opposition later."""
        obs, mask, resid = [], [], []
        def _night(desig, t0, keep, r):
            for k in range(3):
                obs.append({"jd_utc": t0 + 0.01 * k, "designation": desig, "obscode": "W84",
                            "ra_deg": 10.0 + 0.001 * len(obs), "dec_deg": 5.0})
                mask.append(keep); resid.append((r, 0.0))
        for d, keep in ((0.0, True), (1.0, True), (2.0, False)):
            _night("A", jd0 + d, keep, 0.2 if keep else 3.0)
        for frac in (0.0, 0.3, 0.65, 1.0):
            _night("B", jd0 + 900.0 + frac * span, True, 0.2)
        class _F:
            _obs = obs
            _mask = np.array(mask, dtype=bool)
            _resid_all = np.array(resid, dtype=float)
            def rms_residual(self): return 0.2
            def n_observations(self): return int(self._mask.sum())
            def n_total_observations(self): return len(obs)
            def n_excluded(self): return len(obs) - int(self._mask.sum())
        return _F()

    fit = _fit_with_anchor(40.0)
    assert ident_check.unclipped_anchor_arc(fit, ~fit._mask) > 35.0, "B is the unclipped 40 d anchor"
    assert ident_check.clip_doubt(fit) is None, "a 35+ d unclipped side must exempt the link"
    assert cli._examine_or_no(fit)[0] == "y-examine", "the anchored link must not be a flat 'no'"
    # ...and an anchor below the bar brings the doubt straight back (nothing else differs; 12 d is
    # also under SECURE_MIN_ARC_DAYS, so this really tests the anchor and not the older gate).
    short = _fit_with_anchor(12.0)
    assert ident_check.clip_doubt(short), "a 12 d unclipped side is no anchor -- doubt must return"
    assert cli._examine_or_no(short)[0] == "no", "...and the link is a flat 'no' again"


def test_two_opp_coincidence_soft_drop_downgraded():
    # The COINCIDENCE gate: 2015 XO232 = 2025 PD58 survives the collapse ban with the SAME
    # night-for-night shape as the confirmed 2015 OF201 = 2025 NA385 y2 (a 2-night ~1 d arc + a
    # 4-night ~13 d arc, one 2015 night dropped) -- so night count / arc length cannot separate them.
    # The discriminator is WHY the night was dropped: OF201 clips a 23" gross stray, XO232 clips a
    # night only ~4.5" off that plausibly belongs to its coherent 3-night F51 sequence. Neither side
    # is secure (no >=3 kept nights, no >20 d arc), so the ID rests on that soft drop -> identify()
    # must DECLINE (a real-doubt downgrade, not an auto-confirm).
    fit = _fit(_fix("PAIR_2015XO232"), inject_line=True)
    assert fit.n_excluded() > 0
    assert ident_check.identify(fit) is None, "a 2-opp link resting on a soft (non-blunder) drop must NOT identify"
    # ...and it is a flat 'no', not a human-review candidate: the 9.7 yr baseline does not rescue a
    # join that exists only because a night of 2015 XO232's own coherent 3-night F51 sequence was
    # discarded. (Adjudicated by hand: see test_clipped_night_between_two_short_arcs_is_a_flat_no.)
    code, _ = cli._examine_or_no(fit)
    assert code == "no", f"a clip-night coincidence must be a flat 'no', got {code!r}"


def test_two_opp_coincidence_gross_stray_drop_guts_baseline_rejected():
    # The COINCIDENCE gate's GROSS-STRAY side (COINCIDENCE_MIN_KEPT_ARC_DAYS): 2003 QC91 = 2006
    # QX180 -- two sparse ~1-day arcs three years apart. 2006 QX180 is 3 obs / 1.0 d; 2003 QC91's
    # own 5-obs / 58-day record collapses to a 1.0 d kept arc when its far 304 night (78" off the
    # joint orbit) is clipped. That night IS a gross stray, so OF201's hard-drop rule would trust it --
    # but it was 2003 QC91's OWN longer baseline, and after clipping it NO opposition keeps even a
    # multi-day arc (vs OF201/HH101's retained 9-12 d). One orbit threading two single-tracklet-pairs
    # is a coincidence, not an identification -> identify() must DECLINE...
    fit = _fit(_fix("PAIR_2003QC91"), inject_line=True)
    assert fit.n_excluded() > 0
    assert ident_check.identify(fit) is None, \
        "a 2-opp link whose gross-stray drop guts one side's only baseline must NOT identify"
    # ...and with 2003 QC91 retaining only 3/5 obs (60% < MPC_MIN_RETENTION) it is a clean 'no',
    # not merely a human-review 'y-examine' (contrast XO232, whose 2025 side keeps a real 13 d arc).
    code, _ = cli._examine_or_no(fit)
    assert code == "no", f"a gutted-baseline coincidence with sub-threshold retention must be 'no', got {code!r}"


def test_short_short_same_apparition_stays_no():
    # cli._examine_or_no: a SAME-apparition short+short (synthetic, two 3-obs arcs 5 days apart,
    # one arc partly dropped) has no baseline pinning the orbit, so it must stay 'no'.
    import numpy as np
    jd0 = 2461000.0
    fobs = ([{"jd_utc": jd0 + 0.01 * k, "designation": "A"} for k in range(3)]
            + [{"jd_utc": jd0 + 5.0 + 0.01 * k, "designation": "B"} for k in range(3)])
    class _F:
        _obs = fobs
        _mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)   # min retention 2/3, < 1.0
        def rms_residual(self): return 0.2
    code2, _ = cli._examine_or_no(_F())
    assert code2 == "no", f"a same-apparition short+short must stay 'no', got {code2!r}"



def test_interleaved_short_short_is_a_clean_identification():
    # PAIR_2026SA12: 2026 SW11 = 2026 SA12, two 0.9 d arcs from the SAME 22 hours (H36 on both
    # sides, I41/I52, ...), 41/48 at 0.49". The same-apparition short+short 'no' and the
    # mostly-clipped-tracklet refusal both read the drops as the price of the join; with the arcs
    # interleaved the joint orbit is checked against both at the same epochs, and every drop is
    # marginal (under MARGINAL_REJECT_SIGMA) -- two of them the 703 pair SW11's solo fit rejects.
    fit = _fit(_fix("PAIR_2026SA12"), inject_line=True)
    assert fit.n_excluded() > 0
    assert ident_check._interleaved(fit._obs, fit._mask)
    rsig = ident_check._resid_sigmas(fit)
    assert all(rsig[i] < ident_check.MARGINAL_REJECT_SIGMA for i in range(len(rsig)) if not fit._mask[i])
    txt = ident_check.identify(fit)
    assert txt is not None and "except" not in txt, f"expected a clean A = B, got {txt!r}"
    assert ident_check.linkage_verdict(fit)["code"] == "y"
    assert cli._examine_or_no(fit)[0] == "y-examine"


@pytest.mark.parametrize("offset,rate", [(1.0, 0.0), (0.0, 3.0), (2.0, 0.0)])
def test_interleaved_wrong_track_is_still_refused(offset, rate):
    # CONTROL for the interleave exemption (robust-solver rule): the SAME interleaved pair with
    # 2026 SA12 moved onto a track 2026 SW11 is not on -- a constant offset, or a drift across
    # the 0.9 d arc -- must still be a flat 'no'. The joint fit then drops obs well past
    # MARGINAL_REJECT_SIGMA (SA12's 8-obs I41 night loses 7 at a 1" offset), which neither the
    # scatter cap nor the tracklet clause excuses. Shifts below what the 0.5" astrometry can
    # resolve (0.5", 1-2"/day) are not different tracks and are not asserted here.
    import math
    fx = _fix("PAIR_2026SA12")
    t0 = min(o["jd_utc"] for o in fx["obs"])
    obs = [dict(o) for o in fx["obs"]]
    for o in obs:
        o["obs80_line"] = f"     {o.get('packed','XXXXXXX'):7} C2020 01 01.00000 ...{o['obscode']}"
        if o["designation"] == "2026 SA12":
            dd = (offset + rate * (o["jd_utc"] - t0)) / 3600.0
            o["dec_deg"] += dd
            o["ra_deg"] += dd / math.cos(math.radians(o["dec_deg"]))
    fit = fo.fit_orbit(obs, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    assert ident_check.identify(fit) is None, "a wrong interleaved track must not identify"
    assert cli._examine_or_no(fit)[0] == "no", "...and must not even be offered for review"


def test_interleaved_needs_overlapping_arcs():
    # _interleaved is about TIME: side-by-side arcs (the synthetic same-apparition pair above)
    # do not interleave, overlapping ones do, and a lone designation never does.
    import numpy as np
    jd0 = 2461000.0
    side = ([{"jd_utc": jd0 + 0.01 * k, "designation": "A"} for k in range(3)]
            + [{"jd_utc": jd0 + 5.0 + 0.01 * k, "designation": "B"} for k in range(3)])
    assert not ident_check._interleaved(side, np.ones(6, bool))
    mixed = [{"jd_utc": jd0 + 0.1 * k, "designation": "AB"[k % 2]} for k in range(8)]
    assert ident_check._interleaved(mixed, np.ones(8, bool))
    # ...judged on KEPT obs: B cut back to one instant has no arc left to interleave
    keep = np.array([1, 1, 1, 0, 1, 0, 1, 0], bool)
    assert not ident_check._interleaved(mixed, keep)
    assert not ident_check._interleaved(mixed[0::2], np.ones(4, bool))   # A alone

def test_big_arc_scatter_and_lone_deviant_night_is_y2():
    # PAIR_2025NM2 = 2025 NM2 = 2021 WL5, an MPC-CONFIRMED y2 (exception: one junk 126 tracklet at
    # ~76") that guards three relaxations at once:
    #   (a) the isolated-outlier cap scales with the arc (max(MAX_DROP_SCATTER, SCATTER_FRAC*n)):
    #       3 scatter drops among 62 obs is honest rejection noise, not a fishy link;
    #   (b) systematic_misfit is leave-worst-night-out robust in a 4+-night apparition: a lone L04
    #       night +1.2" off in Dec must not fake a link-level "trend" (a real mislink displaces
    #       multiple nights);
    #   (c) thin_arc_falsepos ignores blunder-level drops: clipping the ~76" 126 tracklet doesn't
    #       "collapse" NM2's arc -- the kept 2-night arc is everything real it has;
    #   (d) the except: ECHO lists only wholly-dropped >EXCEPT_MIN_RESID_ARCSEC tracklets: the
    #       fit also excludes 3 sub-1.25" isolated outliers (2 of I52's 7-obs night, 1 of W05's
    #       3), but those are bad OBSERVATIONS, not bad tracklets -- the emitted record must
    #       carry exactly the two 126 lines, matching the MPC-accepted identification.
    fit = _fit(_fix("PAIR_2025NM2"), inject_line=True)
    txt = ident_check.identify(fit)
    assert txt is not None and "except:" in txt, f"must emit the confirmed y2, got {txt!r}"
    assert ident_check.systematic_misfit(fit) is None, "a lone deviant night among 5 must not flag systematic"
    assert ident_check.thin_arc_falsepos(fit) is None, "blunder-only drops must not flag a collapsed arc"
    dropped = [o for i, o in enumerate(fit._obs) if not fit.is_included()[i]]
    assert sum(1 for o in dropped if o["obscode"] == "126") == 2, \
        f"the junk 126 tracklet must be dropped, got {[o['obscode'] for o in dropped]}"
    body = [ln for ln in txt.splitlines() if " = " not in ln and not ln.startswith("#")]
    assert len(body) == 2 and all(ln.endswith("126") for ln in body), \
        f"except block must list exactly the two 126 obs, got {body}"


def test_ident_attribution_relaxes_overconfident_night():
    # OVER-CONFIDENT-OBSCODE relaxation on the ATTRIBUTION path (fit_object._ident_attribute_to_anchor
    # + arc_extend._relax_overconfident_obscodes). 2013 CK176 = 2026 DO7 is a genuine 6-opposition y2
    # whose 2026 DO7 arc anchors the attribution. Its 3-obs D29 night reports over-tight ADES sigmas
    # (~0.1-0.2" vs the 0.5" default), so 2 of its 3 good sub-arcsec obs (0.5-0.9" raw) read as 4-5σ
    # outliers and the fresh attribution evaluator's reject rounds clipped them -> only the B74 blunder
    # night (~19") should drop, but the D29 night fell too and identify() returned 'no'. The path now
    # re-applies the relaxation AFTER rejecting and redeems the small-raw obs (and the excluded-side
    # trigger judges over-confidence on the code's TOTAL obs, so a 3-obs night qualifies). Guards that
    # ONLY the B74 night drops and the pair identifies. Uses n_requested=2 to arm the attribution path
    # (the fixture's own meta leaves it unset -- a linkage-only snapshot).
    fx = _fix("PAIR_2013CK176")
    obs = [dict(o) for o in fx["obs"]]
    for o in obs:                                    # verbatim lines so the except block renders
        o["obs80_line"] = f"     {o.get('packed','XXXXXXX'):7} C2020 01 01.00000 ...{o['obscode']}"
    meta = dict(fx["meta"]); meta["n_requested"] = 2
    fit = fo.fit_orbit(obs, _oc(), desig=meta["designation"], meta=meta, verbose=False)
    txt = ident_check.identify(fit)
    assert txt and " = " in txt, f"2013 CK176 = 2026 DO7 should identify via attribution, got {txt!r}"
    dropped_codes = {o["obscode"] for i, o in enumerate(fit._obs) if not fit.is_included()[i]}
    assert dropped_codes == {"B74"}, \
        f"only the B74 blunder night should drop; the over-confident D29 night must be kept, got {dropped_codes}"


def test_ident_attribution_keeps_genuine_match():
    # The ident-mode ATTRIBUTION fast path (fit_object._ident_attribute_to_anchor) must KEEP a real
    # match, not drop it. GUARD_2014EF94's obs are the genuine 3-designation identification
    # 2014 EF94 = 1999 TC333 = 2025 UO106; internal designation 2025 UO106 spans 10 oppositions, so
    # re-fitting with n_requested=3 anchors on that precise orbit and attributes the 1999 & 2014 obs.
    # All 111 must survive at the same ~0.31" the normal single-designation path gives.
    fx = _fix("GUARD_2014EF94")
    obs = [dict(o) for o in fx["obs"]]
    meta = dict(fx["meta"]); meta["n_requested"] = 3          # arm ident-mode -> attribution triggers
    fit = fo.fit_orbit(obs, _oc(), desig=meta["designation"], meta=meta, verbose=False)
    assert fit.n_observations() == len(obs), \
        f"attribution must keep all {len(obs)} genuine obs, kept {fit.n_observations()}"
    assert fit.rms_residual() < 0.5, f"genuine attribution should stay sub-arcsec, got {fit.rms_residual():.3f}"
    assert fit.n_oppositions() >= 10, f"the 10+-opp orbit must survive, got {fit.n_oppositions()}"


def test_ident_attribution_gate_requires_ident_mode_and_well_determined_member():
    # Guard the attribution TRIGGER conditions directly (no fit): only in ident_mode (n_requested>1)
    # AND only when a designation's own obs span >= IDENT_ATTR_MIN_OPP oppositions. A single-object
    # request, or an ident where every member is short-arc, must return None (normal IOD path).
    import numpy as np
    jd0 = 2451545.0
    # one designation with 6 oppositions (>=5), plus a short foreign arc
    obs = []
    for k in range(6):
        obs.append({"jd_utc": jd0 + 400.0 * k, "designation": "A"})       # 6 opps, ~400 d apart
    for k in range(3):
        obs.append({"jd_utc": jd0 + 0.03 * k, "designation": "B"})        # one short night
    # single-object (n_requested=1) -> never attributes, even with a 6-opp member
    assert fo._ident_attribute_to_anchor(obs, _oc(), "A", {"n_requested": 1}) is None
    # ident with NO >=5-opp member -> None (all short)
    short = [{"jd_utc": jd0 + 0.03 * k, "designation": "A"} for k in range(3)] + \
            [{"jd_utc": jd0 + 5.0 + 0.03 * k, "designation": "B"} for k in range(3)]
    assert fo._ident_attribute_to_anchor(short, _oc(), "A", {"n_requested": 2}) is None


def test_attr_low_q_fallback_gate():
    # The attribution LOW-q DILIGENCE FALLBACK (fit_object._attr_low_q_inconclusive): clipping an
    # entire requested designation is a trustworthy non-match verdict ONLY for a main-belt/outer
    # anchor (q >= IDENT_FAR_MIN_Q) -- a low-q anchor's back-prediction can miss a GENUINE match by
    # more than IDENT_ATTR_KEEP_ARCSEC (multi-rev phase ambiguity, encounter divergence: 2024 LW's
    # own tracklets ride kilo-arcsec off), so there the outcome is INCONCLUSIVE and attribution
    # returns None (the normal path grinds with full Tier-2 diligence). Pin every arm.
    import numpy as np
    obs = [{"designation": "A"}, {"designation": "A"}, {"designation": "B"}, {"designation": "B"}]
    clipped_B = np.array([1, 1, 0, 0], bool)            # B fully excluded (the attribution 'no')
    partial = np.array([1, 1, 1, 0], bool)           # B keeps an ob -> attached
    # _attr_fully_clipped: the state detector
    assert fo._attr_fully_clipped(obs, clipped_B) == "B"
    assert fo._attr_fully_clipped(obs, partial) is None
    # main-belt anchor + clipped -> verdict stands (fast path keeps the win; PAIR_2016UK176's regime)
    assert fo._attr_low_q_inconclusive(2.5, obs, clipped_B) is None
    # low-q anchor + clipped -> INCONCLUSIVE, names the clipped side (falls back to the full grind)
    assert fo._attr_low_q_inconclusive(0.9, obs, clipped_B) == "B"
    # exactly at the cut is main-belt (>= IDENT_FAR_MIN_Q)
    assert fo._attr_low_q_inconclusive(fo.IDENT_FAR_MIN_Q, obs, clipped_B) is None
    # undetermined q must be read conservatively -> INCONCLUSIVE (keep diligence)
    assert fo._attr_low_q_inconclusive(float("nan"), obs, clipped_B) == "B"
    # a low-q ident that genuinely ATTACHES keeps the fast path (attachment is its own proof)
    assert fo._attr_low_q_inconclusive(0.9, obs, partial) is None


def test_ident_far_bridge_skip_spares_two_night_anchors():
    # arc_extend._ident_far_bridge_skip: in an ident hypothesis a main-belt anchor (q >= 1.3) skips
    # the Tier-2 far bridge -- unless the anchor spans <= IDENT_BRIDGE_MAX_NIGHTS nights, whose far
    # prediction is too loose for a Tier-1 miss to mean anything (2015 XS276 = P12r4oH).
    # Pin every arm on a circular 2.5 au anchor.
    import numpy as np
    from types import SimpleNamespace
    from pyroorb import arc_extend
    from pyroorb.twobody import MU_SUN
    state = np.array([2.5, 0, 0, 0, np.sqrt(MU_SUN / 2.5), 0])           # q = 2.5
    low_q = np.array([1.0, 0, 0, 0, np.sqrt(MU_SUN / 1.0), 0])           # q = 1.0
    jd0 = 2457362.97
    two = [jd0, jd0 + 0.03, jd0 + 3.0, jd0 + 3.02]                       # 2 nights
    three = two + [jd0 + 6.0]                                            # 3 nights
    t = np.array(two + [jd0 + 3960.0, jd0 + 3960.1])                     # + the far tracklet
    t3 = np.array(three + [jd0 + 3960.0])
    ev = SimpleNamespace(ident_mode=True, times_tdb=t)
    ev3 = SimpleNamespace(ident_mode=True, times_tdb=t3)
    m2 = np.array([1, 1, 1, 1, 0, 0], bool)
    m3 = np.array([1, 1, 1, 1, 1, 0], bool)
    skip = arc_extend._ident_far_bridge_skip
    assert skip(ev, state, m2) is False                 # 2-night main-belt anchor -> bridge
    assert skip(ev3, state, m3) is True                 # 3-night main-belt anchor -> skip, as before
    assert skip(ev, state) is True                      # no mask -> the q rule alone (old behaviour)
    assert skip(ev3, low_q, m3) is False                # low q -> bridge regardless
    assert skip(SimpleNamespace(ident_mode=False, times_tdb=t3), state, m3) is False   # not ident


def test_ident_doomed_degenerate_gate():
    # The degenerate-non-match short-circuit (fit_object._ident_doomed_degenerate): a joint LM that
    # forced two designations' arcs together lands in a garbage minimum (included obs fit >>
    # IDENT_DEGEN_RMS). That is a settled non-match for a MAIN-BELT body, so the whole recovery cascade
    # is skipped (killed the slow-non-match tail: K14EO6K + K13B16O 78s -> <1s, code 'no'). Gate =
    # n_requested>1 AND rms>=IDENT_DEGEN_RMS AND main-belt q. The n_requested>1 scope is load-bearing:
    # _degenerate_anchor_q reads an INDEPENDENT q only when a SEPARATE designation exists; on a lone
    # designation it reproduces the same garbage fit, and a poisoned single-object garbage fit CAN be
    # cascade-rescued -- so a single object must KEEP the cascade. The q gate protects low-q NEO idents
    # (2021 WO). Guards below pin every arm.
    class _FakeFit:
        def __init__(self, rms): self._rms = rms
        def rms_residual(self): return self._rms
    IDENT = {"n_requested": 2}                                   # an explicit cross-designation request
    orig = fo._degenerate_anchor_q
    try:
        fo._degenerate_anchor_q = lambda obs, oc: 2.5           # main-belt
        assert fo._ident_doomed_degenerate(_FakeFit(7542.0), [], None, IDENT), \
            "a MAIN-BELT garbage IDENT fit must trip the degenerate gate"
        # #1 regression guard: the SAME garbage main-belt fit on a SINGLE object (n_requested<=1) must
        # NOT trip -- there is no independent designation, so the q is not a reliable discriminator and
        # a poisoned single fit is cascade-rescuable. This is the bug the n_requested>1 scope closes.
        assert not fo._ident_doomed_degenerate(_FakeFit(7542.0), [], None, {"n_requested": 1}), \
            "a SINGLE-object garbage fit is cascade-rescuable -- must NOT trip (no independent q)"
        assert not fo._ident_doomed_degenerate(_FakeFit(7542.0), [], None, None), \
            "absent meta defaults to a single object -- must NOT trip"
        assert not fo._ident_doomed_degenerate(_FakeFit(2.4), [], None, IDENT), \
            "a few-arcsec fit is not degenerate -- must NOT trip (below IDENT_DEGEN_RMS)"
        fo._degenerate_anchor_q = lambda obs, oc: 0.6           # low-q NEO
        assert not fo._ident_doomed_degenerate(_FakeFit(7542.0), [], None, IDENT), \
            "a q<1.3 NEO garbage ident fit is cascade-RESCUABLE -- must NOT trip"
        fo._degenerate_anchor_q = lambda obs, oc: None          # q undetermined
        assert not fo._ident_doomed_degenerate(_FakeFit(7542.0), [], None, IDENT), \
            "an undetermined q must be treated conservatively -- do NOT trip (keep the cascade)"
        # MEMOIZATION: the gate is consulted twice on the settled path (fit_orbit short-circuit,
        # then the _rescue_single_night_poison re-check) and _degenerate_anchor_q is a full solo
        # re-fit -- the SAME fit object must pay it exactly once (even when the answer is None).
        calls = {"n": 0}
        def counting_q(obs, oc):
            calls["n"] += 1
            return 2.5
        fo._degenerate_anchor_q = counting_q
        memo_fit = _FakeFit(7542.0)
        assert fo._ident_doomed_degenerate(memo_fit, [], None, IDENT)
        assert fo._ident_doomed_degenerate(memo_fit, [], None, IDENT)
        assert calls["n"] == 1, f"anchor-q re-fit must be memoized on the fit, ran {calls['n']}x"
    finally:
        fo._degenerate_anchor_q = orig


def test_thin_collapsed_arc_falsepos_flagged():
    # PAIR_2025QH122 = 2014 HV101: an emitted 'y2' that is really a THIN-COLLAPSED-ARC false
    # positive. 2014 HV101's 4-night 2014 apparition keeps 23/23, but its 2011 and 2023 nights are
    # dropped whole and 2025 QH122's own 3-night / 11-day arc collapses to a ~0.8 d (< 24 h) kept
    # sub-arc -- nothing on the 2025 side independently confirms the identity. identify() still emits
    # it (whole-tracklet drops, kept obs fit ~0.14"), but thin_arc_falsepos must flag it and name the
    # collapsed 2025 side (the batch turns this into the classify code 'y2-falsepos?').
    fit = _fit(_fix("PAIR_2025QH122"), inject_line=True)
    txt = ident_check.identify(fit)
    assert txt is not None and "except:" in txt, f"expected an emitted 'A = B except', got {txt!r}"
    desc = ident_check.thin_arc_falsepos(fit)
    assert desc and "2025 QH122" in desc, f"thin collapsed arc must be flagged & named, got {desc!r}"
    # the dropped X05 night (whole tracklet at 1.20-1.67", one obs UNDER 1.25") is a
    # near-threshold drop, not a provable stray -> excluded from the fit but NOT echoed as an
    # exception; the listed exceptions are the far-off G96 (~3000") and F52 (~10500") nights.
    body = [ln for ln in txt.splitlines() if " = " not in ln and not ln.startswith("#")]
    assert body and not any(ln.endswith("X05") for ln in body), \
        f"a near-threshold whole-tracklet drop must not be excepted, got {body}"


def test_genuine_full_retention_pair_not_falsepos():
    # GUARD against over-flagging: a clean identification that retains a real multi-hour+ arc on BOTH
    # sides must NOT be flagged a false positive. PAIR_2002QT68 keeps every obs (n_excluded == 0), so
    # neither side collapses -> thin_arc_falsepos stays quiet.
    fit = _fit(_fix("PAIR_2002QT68"))
    assert ident_check.identify(fit) is not None, "should be a clean A = B"
    assert ident_check.thin_arc_falsepos(fit) is None, "fully-retained pair must not be a falsepos"


def test_magnitude_residual_flags_large_dH():
    # PAIR_2014EW228 = 2026 FQ21 is a GENUINE identification (0.28" over two oppositions) but its
    # 2014 W84 photometry is junk, so the arcs imply absolute magnitudes ~1.8 mag apart. The match
    # must STILL be emitted (we never drop on photometry) but carry a '# FLAG' comment for review.
    fit = _fit(_fix("PAIR_2014EW228"))
    st = ident_check._mag_stats(fit)
    assert st is not None and st["tier"] is not None, f"expected a magnitude flag, got {st}"
    assert st["dH"] >= 1.0, f"expected a >=1 mag residual, got {st['dH']}"
    txt = ident_check.identify(fit)
    assert txt is not None and " = " in txt, f"flagged pair must still be identified, got {txt!r}"
    assert "# FLAG" in txt, f"expected a magnitude FLAG comment, got {txt!r}"
    # the flag is a leading comment; the bare identification line itself is unchanged
    assert not txt.splitlines()[-1].startswith("#"), "the identification line must not be a comment"


def test_consistent_photometry_no_flag():
    # PAIR_2002QT68 has matching brightness across oppositions (dH ~ 0.01 mag) -> no flag.
    fit = _fit(_fix("PAIR_2002QT68"))
    txt = ident_check.identify(fit)
    assert txt is not None and "FLAG" not in txt, f"consistent photometry must not be flagged, got {txt!r}"


class _FakeFit:
    """Minimal stand-in for the audit: opposition_mag_audit only reads _obs (jd_utc), _mask and
    _phot['per'][i]['H_i']. Two apparitions ~2 yr apart with the given per-obs H lists."""
    def __init__(self, h_2013, h_2015):
        jd0, jd1 = 2456600.0, 2457300.0            # ~700 d apart -> distinct apparitions
        self._obs, per = [], []
        for base, hs in ((jd0, h_2013), (jd1, h_2015)):
            for k, h in enumerate(hs):
                self._obs.append({"jd_utc": base + 0.03 * k, "designation": "2013 XX1"})
                per.append({"H_i": h})
        self._mask = [True] * len(self._obs)
        self._phot = {"per": per}


def test_opposition_mag_audit_flags_shifted_opposition():
    # One body imaged in two apparitions: the 2013 arc sits ~1 mag off the 2015 arc (each internally
    # tight). opposition_mag_audit must flag it, and locate the two apparitions.
    fit = _FakeFit([18.6, 18.7, 18.65, 18.72, 18.6],
                   [17.6, 17.7, 17.68, 17.66, 17.7])
    result = ident_check.opposition_mag_audit(fit)
    assert result is not None and len(result["apps"]) == 2, f"expected 2 apparitions, got {result}"
    assert result["tier"] is not None, f"a ~1 mag inter-apparition shift must flag, got {result}"
    assert result["worst"]["dH"] >= 0.8, f"expected a large dH, got {result['worst']}"


def test_opposition_mag_audit_consistent_no_flag():
    # Same body, same brightness both apparitions (dH ~ 0.02) -> no flag.
    fit = _FakeFit([18.60, 18.70, 18.65, 18.62, 18.68],
                   [18.61, 18.66, 18.63, 18.69, 18.64])
    result = ident_check.opposition_mag_audit(fit)
    assert result is not None and result["tier"] is None, f"consistent photometry must not flag, got {result}"


def test_opposition_mag_audit_single_opposition_returns_none():
    # Plenty of obs (>= 2*MAG_MIN_N) but all in ONE apparition -> nothing to compare -> None.
    fit = _FakeFit([18.6, 18.7, 18.65, 18.72, 18.6, 18.63, 18.68], [])
    assert ident_check.opposition_mag_audit(fit) is None, "single apparition must return None"


def test_obs80_magnitude_and_satellite_parsing():
    # Column-bookkeeping lock for the OBS80 reader (src/pyroorb/obs80.py): the magnitude (cols 66-71) and
    # the two-line SPACE-BASED satellite record (the 'S' position line + its 's' geocentric-XYZ
    # companion, e.g. C51/WISE). Literal lines, no network.
    from pyroorb import obs80; import math
    # ground CCD line with a magnitude (2002 GX31)
    G = "     K02G31X*1C2002 04 06.19898 14 19 02.38 -12 40 13.4          22.7 R f5154807"
    mag, band = obs80.mpc80_magnitude(G)
    assert mag == 22.7 and band == "R", f"mag/band {mag!r}/{band!r}"
    # WISE (C51) two-line satellite record for 2010 OA81: geocentric XYZ in km on the 's' line
    s = "     K10O81A  s2010 07 25.8115231 + 4303.6772 + 3189.2987 + 4351.0582   ~2gopC51"
    pos = obs80.mpc80_satellite_offset(s)
    assert pos is not None and len(pos) == 3, f"sat offset {pos!r}"
    r = math.sqrt(sum(c * c for c in pos))          # WISE geocentric radius ~6900 km ~ 4.6e-5 au
    assert 4.0e-5 < r < 5.0e-5, f"WISE geocentric radius {r:.2e} au out of range"
    # the 'S' position line is NOT the 's' companion -> no offset
    S = "     K10O81A  S2010 07 25.81152301 22 13.781+37 11 04.13               L~2gopC51"
    assert obs80.mpc80_satellite_offset(S) is None
    # roving-observer 'v' companion (code 247): geodetic lon/lat/alt -> Earth-fixed ITRS ~Earth R
    buf = list(" " * 80)
    def _put(c, s): buf[c - 1:c - 1 + len(s)] = list(s)
    _put(6, "K10O81A"); _put(15, "v"); _put(16, "2010 07 25.811523"); _put(33, "1")
    _put(35, f"{243.14:10.6f}"); _put(46, f"{34.20:+10.6f}"); _put(57, f"{1700:5d}"); _put(78, "247")
    rov = obs80.mpc80_roving_itrs_au("".join(buf))
    assert rov is not None and len(rov) == 3
    r_km = math.sqrt(sum(c * c for c in rov)) * obs80.KM_PER_AU
    assert 6300 < r_km < 6450, f"roving geocentric radius {r_km:.0f} km not ~Earth radius"


def test_ades_roving_matches_obs80_roving():
    # The ADES roving path (sys=WGS84, ctr=399, pos1/2/3 = lon/lat/alt) must yield the SAME
    # geocentric ITRS site as the OBS80 'v' line for the same geodetic triple (2006 OE10's 247
    # site). And an ICRF satellite record must NOT be misread as roving. Offline, no network.
    from pyroorb import ades; from pyroorb import obs80; import numpy as np
    rec = {"sys": "WGS84", "ctr": 399.0, "pos1": "289.14670", "pos2": "-30.52631", "pos3": "1710"}
    rov = ades._rov_geodetic_au(rec)
    assert rov is not None
    itrs_au, geodetic = rov          # (itrs_au, (lon, lat, alt)) -- the triple
    assert geodetic == (289.14670, -30.52631, 1710.0)
    assert np.allclose(itrs_au, obs80.geodetic_to_itrs_au(289.14670, -30.52631, 1710))
    # a space-based ICRF vector is a satellite, not roving
    assert ades._rov_geodetic_au({"sys": "ICRF_KM", "ctr": 399.0,
                                  "pos1": "4303.7", "pos2": "3189.3", "pos3": "4351.1"}) is None
    # ...and the satellite parser does not claim a WGS84 roving record
    assert ades._sat_geocentric_au(rec) is None


def test_ades_disc_flag_renders_col13_asterisk():
    # The ADES discovery flag (disc="*") must survive conversion to OBS80 as
    # the col-13 asterisk (0-based index 12, right after the 7-char provid) --
    # the identification pipeline stamps a NEOCP object by its discovery
    # tracklet, so losing the flag loses the discovery date. Offline.
    from pyroorb import ades; from pyroorb import obs80
    base = {"stn": "F51", "ra": "150.0", "dec": "10.0",
            "obstime": "2017-01-02T02:24:00Z", "trksub": "P21abcd",
            "mag": "21.0", "band": "G"}
    obscodes = {"F51": {}}
    disc, plain = dict(base, disc="*"), dict(base, disc="")
    o_disc = ades.records_to_obs([disc], obscodes)[0]
    o_plain = ades.records_to_obs([plain], obscodes)[0]
    assert o_disc["discovery"] is True and o_plain["discovery"] is None
    l_disc = obs80.obs80_base_line(o_disc).splitlines()[0]
    l_plain = obs80.obs80_base_line(o_plain).splitlines()[0]
    assert l_disc[12] == "*", repr(l_disc)
    assert l_plain[12] == " ", repr(l_plain)


def test_read_pairs_tab_and_comma():
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "_identcheck_pairs_test.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# comment\n2021 GD209\t2026 DB29\n2002 QT68, 2026 EO60\n\n")
    pairs = cli._read_pairs(p)
    assert pairs == [("2021 GD209", "2026 DB29"), ("2002 QT68", "2026 EO60")], pairs


def test_short_link_flagged_but_still_identified():
    # PAIR_1999KK17 = 2016 LL89: an identification that passes every gate we had -- 9/9 obs kept,
    # nothing dropped, 0.56" -- because all of those gates judge what the fit DISCARDED. With 5 + 4
    # observations the joint orbit has 18 residuals against 6 free parameters, so "it fits" is
    # nearly free and nothing here TESTS the identification. It must still be EMITTED (advisory,
    # never a rejection), carry a leading '# FLAG short link' comment in the written record, and
    # classify as 'y-short'.
    fit = _fit(_fix("PAIR_1999KK17"))
    assert fit.n_observations() < ident_check.THIN_LINK_MIN_OBS, "fixture must be the thin case"
    desc = ident_check.thin_link(fit)
    assert desc and "1999 KK17" in desc and "2016 LL89" in desc, \
        f"short link must be flagged and both sides counted, got {desc!r}"
    txt = ident_check.identify(fit)
    assert txt is not None and " = " in txt, f"the ID must still be emitted, got {txt!r}"
    flags = [ln for ln in txt.splitlines() if ln.startswith("# FLAG short link")]
    assert len(flags) == 1, f"written record must carry the short-link FLAG, got {txt!r}"
    v = ident_check.linkage_verdict(fit)
    assert v["offer"] and v["clean"], "a short link is still offered as a clean identification"
    assert any("Short link" in c for c in v["cautions"]), \
        f"linkage_verdict must caution about the short link, got {v['cautions']}"


def test_well_sampled_link_not_short():
    # GUARD against over-flagging: a normal-sized identification (PAIR_2002QT68, fully retained)
    # is well past THIN_LINK_MIN_OBS, so thin_link stays quiet and no FLAG reaches the record.
    fit = _fit(_fix("PAIR_2002QT68"))
    assert fit.n_observations() >= ident_check.THIN_LINK_MIN_OBS
    assert ident_check.thin_link(fit) is None, "a well-sampled link must not be called short"
    assert "# FLAG short link" not in (ident_check.identify(fit) or "")


def test_lone_tracklet_apparition_poison_is_set_aside_not_absorbed():
    # PAIR_2015HP527 = 2015 HP527 = 2025 PS30. 2025 PS30's record carries a 4-obs 2021-10-02 V00
    # night that is not this object -- and it is its OWN apparition, so no night beside it can
    # contradict it (leverage ~ 1). Absorbed into the fit, the joint orbit satisfies that lone
    # tracklet EXACTLY and pays for it across both real apparitions: 26/40 @ 0.43" with the 2015
    # and 2025 nights each sitting coherently off the orbit -- which systematic_misfit correctly
    # read as a link-level TREND, so the verdict was 'y-systematic' (not auto-confirmed, not
    # written). The fix is upstream, in the engine: _rescue_single_night_poison sets the lone-night
    # opposition aside and the remainder fits 32/36 @ 0.09", which its acceptance test used to
    # refuse on the >=90%-of-`reduced` link rate (32 vs a 32.4 bar) -- the 4 unlinked obs being a
    # SECOND junk tracklet (the 1-degree-off 2015-04-22 W84 night, also filed under PS30) that the
    # poisoned fit had rejected too. The DOMINANCE clause accepts it: the rescue gives up nothing
    # `best` was keeping, so it is strictly better on both axes. End to end this must now be a
    # clean y2 excepting exactly those two junk tracklets -- and NOT a systematic trend.
    fit = _fit(_fix("PAIR_2015HP527"), inject_line=True)
    assert fit.n_observations() == 32 and fit.rms_residual() < 0.15, \
        f"the poison must be set aside, got {fit.n_observations()}/40 @ {fit.rms_residual():.2f}\""
    assert ident_check.systematic_misfit(fit) is None, \
        "with the lone-tracklet apparition set aside the kept nights must not trend"
    txt = ident_check.identify(fit)
    assert txt is not None and "except:" in txt, f"must auto-confirm as a y2, got {txt!r}"
    body = [ln for ln in txt.splitlines() if " = " not in ln and not ln.startswith("#")]
    assert len(body) == 8, f"except block must list both junk tracklets (8 obs), got {body}"
    dropped = {(o["jd_utc"] // 1) for i, o in enumerate(fit._obs) if not fit.is_included()[i]}
    # 2015-04-22 06h and 2021-10-02 07h UTC -> JD 2457134.x and 2459489.x
    assert {2457134.0, 2459489.0} <= dropped, f"both junk nights must be excluded, got {sorted(dropped)}"


def test_short_short_systematic_is_a_flat_no():
    # 2025 OC732 + 2014 HJ409: the joint orbit keeps every observation (27/27 @ 0.57") and
    # identify() emits a clean 'A = B', but the Dec per-night means sit 0.78" apart at 27x their
    # own error -- and BOTH designations are 3 d arcs. A trend is normally a caution ('y-systematic':
    # unmodelled nongravs, a bad night, survey systematics, with the rest of the arc left for a
    # human to weigh). Two day-long arcs give a human nothing to weigh -- no curvature on either
    # side, no nongrav the fit would have missed at that size -- so the systematic simply means one
    # orbit does not thread both arcs: short_arc_systematic fires and the verdict is a flat 'no',
    # not offered and not written.
    fit = _fit(_fix("PAIR_2025OC732"))
    assert ident_check.identify(fit) is not None, "the joint fit does form a clean A = B"
    assert ident_check.systematic_misfit(fit), "its per-night means trend"
    desc = ident_check.short_arc_systematic(fit)
    assert desc and "short arcs" in desc, f"short+short trend must be fatal, got {desc!r}"
    v = ident_check.linkage_verdict(fit)
    assert not v["offer"] and not v["clean"], f"must not be offered for submission: {v}"


def test_long_arc_systematic_stays_reviewable():
    # GUARD on the other side: the short-arc rule must not swallow the ordinary 'y-systematic'.
    # 2025 OT173 = 2014 EN155 is a CORRECT identification whose 2014 nights trend; its arcs are far
    # longer than SYS_SHORT_ARC_DAYS, so the trend stays a caution a human reviews rather than a
    # refusal.
    fit = _fit(_fix("PAIR_2025OT173"), inject_line=True)
    assert ident_check.systematic_misfit(fit), "this pair's nights do trend"
    assert ident_check.short_arc_systematic(fit) is None, \
        "a trend on arcs longer than SYS_SHORT_ARC_DAYS is a caution, not a refusal"


def test_y2_is_offered_for_review_named_but_not_submittable():
    # PAIR_2015HP527 = 2015 HP527 = 2025 PS30 is a 'y2': the engine confirms the identification but
    # only by EXCEPTING junk tracklets. The web UI asks linkage_verdict, so the verdict must say
    # exactly that -- code 'y2', not clean, the excepted tracklet NAMED -- and must NOT fall
    # through to the 'ambiguous link worth a human look' wording, which quotes retention (a gate
    # this link passed) as though it were the problem. It must also refuse TRANSMISSION: the
    # identification document has no way to say "except these observations", so sending it would
    # claim the excepted tracklet as this object; a y2 goes to the MPC as a Jira ticket, and both
    # front ends say so rather than raising a submit button.
    fit = _fit(_fix("PAIR_2015HP527"), inject_line=True)
    v = ident_check.linkage_verdict(fit)
    assert v["offer"] and v["code"] == "y2" and not v["clean"], v
    assert v["submittable"] is False, v
    assert "EXCEPTIONS" in v["reason"] and "tracklet" in v["reason"], v["reason"]
    assert "Jira" in v["reason"], v["reason"]
    assert "ambiguous link" not in v["reason"], v["reason"]
    # ONE message: the exception IS the story, so it must not be split into a headline plus a
    # caution repeating it (the reader saw two banners saying the same thing).
    assert v["cautions"] == [], v["cautions"]


def test_y2_verdict_survives_an_ades_fit_with_no_obs80_lines():
    # The bug this split fixes: identify() renders its exceptions as VERBATIM 80-column records, so
    # a fit whose observations carry no `obs80_line` -- every ADES fetch, which is every fit the
    # web UI makes -- returned None for a link it had fully accepted. The UI then reported real y2
    # links as ambiguous 'y-examine' candidates and quoted their retention as the reason (2018 VL78
    # = 2026 OF5: "both arcs retain >=92% of their obs", a number that had PASSED). Judging the
    # shape rather than the rendering, the same fixture must give the same verdict with and
    # without the lines -- only `renderable` differs.
    with_lines = ident_check.linkage_verdict(_fit(_fix("PAIR_2015HP527"), inject_line=True))
    ades = _fit(_fix("PAIR_2015HP527"))              # no obs80_line injected
    assert ident_check.identify(ades) is None, "identify() cannot RENDER without verbatim lines"
    assert ident_check.exception_shape(ades)["renderable"] is False
    v = ident_check.linkage_verdict(ades)
    assert v["code"] == with_lines["code"] == "y2", (v, with_lines)
    assert v["reason"] == with_lines["reason"]
    assert v["submittable"] is with_lines["submittable"] is False
    assert "retain" not in v["reason"], v["reason"]


def test_an_unmeasurable_exception_shape_fails_closed(monkeypatch):
    """A shape check that CRASHES must not come back as a more permissive verdict than one that
    succeeded -- which is exactly what it did.

    `exception_shape` wrapped the whole of identify() in `except Exception: return None`, and
    linkage_verdict read None as "not an identification" -> no exceptions to worry about ->
    `submittable` True. So a y2 (not transmissible, Jira route) could be served as a y-examine
    with a live Submit button, on nothing but an internal fault. Before the split existed,
    linkage_verdict called identify() bare and the exception reached the web layer's own guard as
    "linkage check failed": loud, and closed. This restores that property.

    Offered, not hidden: we could not rule the identification OUT either, and losing a real one to
    a transient fault would be its own defect. Only transmission is withdrawn."""
    fit = _fit(_fix("PAIR_2015HP527"), inject_line=True)
    good = ident_check.linkage_verdict(fit)
    assert good["code"] == "y2", good                     # baseline: a real y2

    def _boom(_fit, _shape=False, _why=None):
        raise KeyError("jd_utc")                          # the shape of a real obs-dict fault

    monkeypatch.setattr(ident_check, "identify", _boom)
    # THE property, asserted first so a regression names the defect and not a detail: the same
    # fixture that is a non-submittable y2 when the check works must not become submittable
    # because the check broke.
    v = ident_check.linkage_verdict(fit)
    assert v["submittable"] is False, f"a crash must never grant submittability: {v}"
    assert v["offer"] is True, f"...but it must not hide the fit either: {v}"
    shape = ident_check.exception_shape(fit)
    assert shape["unknown"] is True and "KeyError" in shape["error"], shape
    assert v["clean"] is False and v["code"] == "unknown", v
    assert "could not determine" in v["reason"], v["reason"]
    assert any("KeyError" in c for c in v["cautions"]), v["cautions"]
    # the fault is NOT dressed up as a finding about the astrometry
    assert "retain" not in v["reason"], v["reason"]


def test_clean_y_is_clean_with_no_cautions():
    # The other side of the split, and the property the unification is aiming at: a bare 'y' (the
    # batch's clean code) is the ONLY thing that comes back clean with an empty caution list, so
    # "no warnings in the UI" and "'y' in --ident-check" mean the same thing.
    fit = _fit(_fix("PAIR_2002QT68"))
    v = ident_check.linkage_verdict(fit)
    assert v["offer"] and v["clean"] and v["code"] == "y" and v["submittable"], v
    assert v["cautions"] == [], v["cautions"]


def test_systematic_trend_is_neither_clean_nor_submittable():
    # The batch calls these 'y-systematic' and does NOT write them, so the UI must not paint them
    # as clean or raise a Submit button either. 2014 HV328 = 2025 OV408 is a known-bogus link that
    # identify() accepts at 0.50" -- it used to read "These fit as one object -- a clean
    # identification" over a live submit button, with the trend mentioned in a caution underneath.
    # Now: code 'y-systematic', clean False, submittable False, the trend stated as the reason.
    # 2014 EW228 = 2026 FQ21 comes with an unrelated photometric caution, which must SURVIVE (the
    # trend taking over the headline must not swallow the other warnings).
    for label in ("PAIR_2014HV328", "PAIR_2014EW228"):
        fit = _fit(_fix(label), inject_line=True)
        assert ident_check.identify(fit) is not None, f"{label}: still forms an A = B"
        v = ident_check.linkage_verdict(fit)
        assert v["offer"], f"{label}: stays on screen for review: {v}"
        assert v["code"] == "y-systematic" and not v["clean"], v
        assert v["submittable"] is False, f"{label}: must not be transmissible: {v}"
        assert "TREND" in v["reason"], v["reason"]
    v = ident_check.linkage_verdict(_fit(_fix("PAIR_2014EW228"), inject_line=True))
    assert any("Photometric" in c for c in v["cautions"]), v["cautions"]


class _TrkFit:
    """Stand-in for tracklet_mag_audit: it reads _obs (jd_utc, obscode, designation),
    _mask and _phot['per'][i]['H_i'].  ``tracklets`` is a list of (station, [H_i, ...]),
    each placed on its own night so ident_check._tracklets separates them.  A None in an
    H list is an observation the survey filed with no magnitude -- the common case, and
    the reason TMAG_MIN_N is 1."""
    def __init__(self, tracklets, desig="2014 XX1"):
        self._obs, per = [], []
        for n, (code, hs) in enumerate(tracklets):
            for k, h in enumerate(hs):
                self._obs.append({"jd_utc": 2456600.0 + 3.0 * n + 0.01 * k,
                                  "obscode": code, "designation": desig})
                per.append({"H_i": h})
        self._mask = [True] * len(self._obs)
        self._phot = {"per": per}


def test_tracklet_mag_audit_flags_the_foreign_tracklet():
    # Three tracklets of "one object"; the first implies H ~14.8 while the other two sit
    # at ~18.4 -- 2014 WP143's shape (a 246 tracklet against two F51 ones).
    fit = _TrkFit([("246", [14.75]), ("F51", [18.27, 18.47, 18.37]),
                   ("F51", [18.68, 18.78, 18.38])])
    result = ident_check.tracklet_mag_audit(fit)
    assert result is not None and len(result["trk"]) == 3, f"expected 3 tracklets, got {result}"
    assert result["worst"]["i"] == 0, f"the 14.8-mag tracklet must be the suspect, got {result['worst']}"
    assert result["tier"] == "!", f"a 3.6 mag gap must be strong, got {result['tier']}"
    assert result["trk"][1]["tier"] is None and result["trk"][2]["tier"] is None, \
        "the two consistent tracklets must not flag"


def test_tracklet_mag_audit_measures_a_single_magnitude_tracklet():
    # THE regression that matters: a survey files a magnitude on ONE observation of the
    # tracklet and leaves the rest blank (2014 WP143's 246 tracklet is 1 mag in 9). With
    # TMAG_MIN_N = 2 that tracklet dropped out of its own audit and the object read clean.
    fit = _TrkFit([("246", [14.75, None, None, None, None, None, None, None, None]),
                   ("F51", [18.27, 18.47, 18.37]), ("F51", [18.68, 18.78, 18.38])])
    result = ident_check.tracklet_mag_audit(fit)
    assert result is not None, "a 1-magnitude tracklet must still be measured"
    codes = [t["code"] for t in result["trk"]]
    assert "246" in codes, f"the 1-magnitude tracklet vanished from the audit: {codes}"
    assert result["worst"]["i"] == codes.index("246"), f"expected 246 as the suspect, got {result['worst']}"
    assert result["tier"] == "!", f"a 3.6 mag gap must be strong, got {result['tier']}"


def test_tracklet_mag_audit_consistent_arc_no_flag():
    fit = _TrkFit([("F51", [18.30, 18.40]), ("G96", [18.35, 18.28, 18.44]),
                   ("703", [18.38, 18.31])])
    result = ident_check.tracklet_mag_audit(fit)
    assert result is not None and result["tier"] is None, f"a consistent arc must not flag: {result}"


def test_tracklet_mag_audit_tight_photometry_cannot_fake_significance():
    # Two tracklets agreeing to 0.01 mag would claim a standard error of 0.01 without the
    # TMAG_OBS_SD floor, and a physically trivial 0.2 mag gap would read as certain. The
    # raw-dH gate (MAG_DH_MIN) and the floor must BOTH keep this clean.
    fit = _TrkFit([("F51", [18.30, 18.31]), ("G96", [18.50, 18.51]), ("703", [18.30, 18.29])])
    result = ident_check.tracklet_mag_audit(fit)
    assert result is not None and result["tier"] is None, \
        f"a 0.2 mag gap must never flag however tight the photometry: {result}"


def test_tracklet_mag_audit_two_tracklets_is_not_an_audit():
    # One-vs-rest needs a rest: with two tracklets each would flag the other symmetrically
    # and neither could be identified as the odd one.
    fit = _TrkFit([("F51", [14.7, 14.8]), ("G96", [18.5, 18.4])])
    result = ident_check.tracklet_mag_audit(fit)
    assert result is None, \
        f"two tracklets cannot be audited one-vs-rest, got {result}"


def test_implied_h_pvalue_min_sd_default_is_off():
    # The designation and apparition callers must be untouched by the min_sd addition.
    a, b = [18.6, 18.7, 18.65], [17.6, 17.7, 17.68]
    assert ident_check._implied_h_pvalue(a, b) == ident_check._implied_h_pvalue(a, b, min_sd=0.0)


def test_an_immaterial_partial_night_does_not_refuse_an_identification():
    # 2012 GC2 = CERV9W2 (frozen as the web user saw it): 109 obs over 8 nights in 2012 plus a
    # 16-obs 2026 NEOCP arc, 0.47". The fit rejects 4 outliers -- three of them off ONE 4-obs A48
    # night. identify() used to call that "most of a night, but not all" and refuse outright, on a
    # within-night proportion blind to the 106 kept obs around it, so the web UI showed "not a
    # clean identification". The escape is clip_doubt's own "secure on its own" test one level
    # down: that apparition keeps >= SECURE_MIN_KEPT_NIGHTS other nights, so the join does not
    # close because of the clip.
    fit = _fit(_fix("PAIR_2012GC2"), inject_line=True)
    import numpy as np
    m = np.asarray(fit._mask, bool)
    times = [o["jd_utc"] for o in fit._obs]
    partial = [tr for tr in ident_check._tracklets(times, [o["obscode"] for o in fit._obs],
                                                   [o.get("designation") for o in fit._obs])
               if len(tr) // 3 < sum(1 for i in tr if not m[i]) < len(tr)]
    assert partial, "the fixture must still carry a most-but-not-all night, or this proves nothing"
    why = {}
    assert ident_check.exception_shape(fit, _why=why) is not None, why
    v = ident_check.linkage_verdict(fit)
    assert v["code"] == "y" and v["clean"] and v["submittable"], v
    assert v["cautions"] == [], v["cautions"]


def test_an_examine_caution_names_the_refusal_and_labels_the_exculpation():
    # The banner used to read "Not a clean identification -- an ambiguous link worth a human look:
    # <examine_or_no's reason>", and a 'y-examine' reason is the grounds for NOT calling the link a
    # non-match ("a baseline that long pins the orbit") -- so it announced a doubt, argued the other
    # way, and never said what was wrong. 2025 MT356 is refused for clipping 23 whole tracklets.
    fit = _fit(_fix("PAIR_2025MT356"), inject_line=True)
    v = ident_check.linkage_verdict(fit)
    assert v["code"] == "y-examine", v
    [c] = [c for c in v["cautions"] if "not confirmed" in c]
    assert "refused it: the joint fit clips" in c, c
    assert "What keeps it a candidate rather than a non-match:" in c, c
    assert "ambiguous link worth a human look:" not in c, c


def test_a_load_bearing_partial_night_is_still_refused():
    # CONTROL for the escape above (robust-solver rule: a loosened gate must still refuse the bad
    # case). The same fit, but now MOST of the 8-obs 2026 G96 tracklet is gone -- and that
    # apparition has only one other night, so nothing shows the join stands without the clip.
    # The within-night refusal must still fire, and say why.
    fit = _fit(_fix("PAIR_2012GC2"), inject_line=True)
    import numpy as np
    obs = fit._obs
    times = [o["jd_utc"] for o in obs]
    [g96] = [tr for tr in ident_check._tracklets(times, [o["obscode"] for o in obs],
                                                 [o.get("designation") for o in obs])
             if len(tr) == 8 and obs[tr[0]]["obscode"] == "G96"
             and obs[tr[0]]["designation"] == "CERV9W2"]
    mask = np.array(fit._mask, dtype=bool)
    mask[g96] = False
    mask[g96[:2]] = True                           # keep 2 of 8: most, but not all, of the night
    fit._mask = mask
    why = {}
    assert ident_check.exception_shape(fit, _why=why) is None, "a load-bearing clip must refuse"
    assert "too few other nights" in why.get("reason", ""), why
    assert "G96" in why["reason"] and "CERV9W2" in why["reason"], why
