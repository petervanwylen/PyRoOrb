"""Intra-tracklet residual TREND detector (itf/residual_trend.py).

Offline and synthetic: the statistics are pure functions of (time, residual,
sigma), so the cases that matter -- a smooth walk, a single bad endpoint, a
tracklet that trends only relative to an immaculate arc -- are built directly
rather than fitted, and the numbers in the docstrings are the ones asserted.

The thresholds were calibrated on matched
known-good vs known-wrong pairs in the weak one-apparition regime the gate runs
in); what is locked here is that the gate implements the calibrated rule and
keeps the properties it was designed around.
"""

import numpy as np
import pytest

from pyroorb.itf import residual_trend


def _t(n, minutes=45.0):
    """n evenly spaced times over ``minutes``, in days."""
    return list(np.linspace(0.0, minutes / 1440.0, n))


def _stats(resid, sigma=1.0, minutes=45.0):
    n = len(resid)
    return residual_trend._trend_stats(_t(n, minutes), list(resid), [sigma] * n)


# --------------------------------------------------------------- the statistic

def test_swing_is_end_to_end_in_sigmas():
    # The plain-language number: a residual walking -1.2 -> -0.1 -> +1.3 at
    # sigma = 1 is "2.5 sigmas from the first observation to the last".
    d = _stats([-1.2, -0.1, 1.3])
    assert d["swing"] == pytest.approx(2.5, abs=0.01)
    assert d["mono"] == 1.0


def test_swing_scales_with_sigma_not_arcsec():
    # Same arcsec walk, twice the uncertainty -> half the swing.  The test is
    # about significance, not about angle.
    a = _stats([-1.2, -0.1, 1.3], sigma=1.0)
    b = _stats([-1.2, -0.1, 1.3], sigma=2.0)
    assert b["swing"] == pytest.approx(a["swing"] / 2, rel=1e-6)


def test_a_jump_is_not_a_trend():
    # One bad FIRST observation, the other two flat.  A straight line through
    # these reports a confident slope -- monotonicity is what refuses it.  The
    # tiny ``ref`` makes size and the relative ratio both pass, so shape is the
    # only thing left that can (and does) refuse.
    d = _stats([4.0, -0.2, 0.2])
    assert abs(d["swing_loo"]) >= residual_trend.TREND_REL_FLOOR_SIGMA   # size passes...
    assert d["mono"] == 0.5                                   # ...shape does not
    assert residual_trend._notable(d, ref=0.05) is None


def test_one_bad_endpoint_collapses_under_leave_one_out():
    # Four obs where only the LAST is bad: monotone, and a line fits it, but the
    # trend is one observation's doing.  This is the case the support test exists
    # for -- see the module docstring.
    d = _stats([-0.1, 0.0, 0.1, 2.5])
    assert d["loo_tested"] is True
    assert abs(d["swing"]) > 2.0          # whole-tracklet: looks like a trend
    assert abs(d["swing_loo"]) < 0.5      # drop that obs and there is nothing
    assert residual_trend._notable(d, ref=0.05) is None


def test_a_real_four_point_trend_survives_leave_one_out():
    d = _stats([-3.0, -1.0, 1.0, 3.0])
    assert d["loo_tested"] is True
    assert abs(d["swing_loo"]) >= 2.0     # weakened by the shorter baseline...
    assert d["mono_loo"] == 1.0           # ...but still a walk
    assert abs(d["swing"]) > abs(d["swing_loo"])


def test_three_obs_cannot_be_leave_one_out_tested():
    # Dropping one of three leaves two points, which define a line exactly.  The
    # loo values must therefore repeat the whole-tracklet ones and SAY so.
    d = _stats([-1.2, -0.1, 1.3])
    assert d["loo_tested"] is False
    assert d["swing_loo"] == d["swing"]
    assert d["t_loo"] == d["t"]


def test_flat_residuals_are_silent():
    # even against an immaculate arc: the absolute floor is what stops a clean
    # object turning its own noise into trends
    assert residual_trend._notable(_stats([0.1, -0.05, 0.02]), ref=0.01) is None
    assert residual_trend._notable(_stats([0.3, -0.2, 0.1, -0.1]), ref=0.01) is None


# --------------------------------------------------------------------- the gate

def test_no_yardstick_means_no_verdict():
    # The gate is RELATIVE and has exactly one clause, so a seed with no
    # measurable tracklet of its own gives nothing to compare against -- and a
    # trend of any size stays silent rather than falling back to an absolute
    # limit.  (The absolute limit was deleted: on the population where this test
    # is the last word it discriminated no better than chance, AUC 0.476, while
    # costing 2 of 39 good tracklets.)
    assert residual_trend._notable(_stats([-1.2, -0.1, 1.3]), ref=0.0) is None
    assert residual_trend._notable(_stats([-4.0, 0.0, 4.0]), ref=0.0) is None


def test_a_trend_far_outside_the_arcs_own_habits_fires():
    # ...but the same 2.5-sigma walk on an object whose own arc never trends past
    # 0.32 sigma is 7.7x anything it does.  This is 2026 OE3/P21l7qL.
    d = _stats([-1.2, -0.1, 1.3])
    why = residual_trend._notable(d, ref=0.32)
    assert why and "arc" in why
    assert residual_trend._notable(d, ref=1.5) is None      # a habitually trending arc: quiet


def test_relative_clause_needs_an_absolute_floor():
    # An immaculate arc must not turn noise into a trend: 0.5 sigma is 10x a
    # 0.05-sigma reference and still nothing at all.
    d = _stats([-0.25, 0.0, 0.25])
    assert abs(d["swing"]) < residual_trend.TREND_REL_FLOOR_SIGMA
    assert residual_trend._notable(d, ref=0.05) is None


def test_relative_clause_still_requires_the_walk():
    # 7.7x the reference, but it jumped rather than walked.
    d = _stats([2.0, -0.1, 0.1])
    assert residual_trend._notable(d, ref=0.32) is None


def test_thresholds_are_the_calibrated_ones():
    # Measurements, not preferences: the calibration measured this trio
    # together, in the weak-seed regime the gate runs in, at 0 of 39 good and 2
    # of 77 wrong tracklets.  Changing one means re-running that study -- and
    # note the ratio ALONE fires on 8.7% of good tracklets, so none of the three
    # is optional.
    assert residual_trend.TREND_MIN_MONOTONIC_FRAC == 1.0
    assert residual_trend.TREND_RELATIVE == 4.0
    assert residual_trend.TREND_REL_FLOOR_SIGMA == 2.0
    assert not hasattr(residual_trend, "RAMP_SWING_SIGMA"), "absolute clause was deleted"


# ------------------------------------------------------------- over a real fit

class _FakeFit:
    """Minimal FittedOrbit stand-in: tracklet_trends only reads _obs,
    residuals(), is_included() and _resid_all.

    ``included`` is a per-row mask, so a tracklet the fit TRIMMED can be built
    -- the case :func:`itf.residual_trend._notable` refuses."""

    def __init__(self, rows, included=None):
        self._rows = rows
        self._obs = [{"jd_utc": r["jd_utc"], "obs80_line": r.pop("line", None)}
                     for r in rows]
        self._resid_all = True
        self._included = ([True] * len(rows) if included is None else included)

    def residuals(self):
        return self._rows

    def is_included(self):
        return self._included


def _rows(t0, resid, code="F52"):
    return [{"jd_utc": t0 + t, "obscode": code, "sigma_ra": 0.3,
             "sigma_dec": 0.3, "d_ra_arcsec": 0.0, "d_dec_arcsec": r * 0.3}
            for t, r in zip(_t(len(resid)), resid)]


def test_tracklet_trends_splits_nights_and_reports_the_worse_axis():
    f = _FakeFit(_rows(2459000.5, [-1.2, -0.1, 1.3])
                 + _rows(2459010.5, [0.1, 0.0, -0.1]))
    out = residual_trend.tracklet_trends(f)
    assert len(out) == 2                        # two nights, not one tracklet
    assert out[0]["axis"] == "Dec"              # worst-first, and Dec is where
    assert out[0]["swing"] == pytest.approx(2.5, abs=0.05)
    assert abs(out[1]["swing"]) < 0.5


# ----------------------------------------------------- what counts as ONE unit

def test_two_stations_the_same_night_are_two_tracklets():
    # The first cut grouped by night gap alone, station-blind: two observatories
    # working the same object the same night became one "tracklet", and their
    # relative pointing offset became its trend.  Over the regression archive that
    # was 858 of 5548 groups.  One station, one tracklet.
    f = _FakeFit(_rows(2459000.50, [-1.2, -0.1, 1.3], code="F52")
                 + _rows(2459000.62, [0.1, 0.0, -0.1], code="703"))
    out = residual_trend.tracklet_trends(f)
    assert len(out) == 2
    assert {d["station"] for d in out} == {"F52", "703"}


def test_a_two_hour_gap_ends_the_tracklet():
    # Same station, same night, two visits three hours apart.  Chaining them
    # gives a 3-hour "tracklet", and over three hours the sky motion is no longer
    # the straight line the whole statistic assumes.
    f = _FakeFit(_rows(2459000.50, [-1.2, -0.1, 1.3])
                 + _rows(2459000.50 + 3.0 / 24.0, [0.1, 0.0, -0.1]))
    out = residual_trend.tracklet_trends(f)
    assert len(out) == 2
    assert all(d["span_min"] < 60 for d in out)


def test_a_normal_cadence_is_not_split():
    # ...and the split must not fire inside a real tracklet: 4 obs over 45 min
    # (p90 of the adjacent-pair gap in the archive is 32 min) stay one unit.
    f = _FakeFit(_rows(2459000.5, [-3.0, -1.0, 1.0, 3.0]))
    out = residual_trend.tracklet_trends(f)
    assert len(out) == 1 and out[0]["n"] == 4


def test_chaining_cannot_walk_across_nights():
    # Observations 10 hours apart chained under the old half-day rule, so a run
    # of them produced spans of up to 2.6 DAYS.  Each stands alone now.
    f = _FakeFit(sum((_rows(2459000.5 + k * 10.0 / 24.0, [-1.2, -0.1, 1.3])
                      for k in range(4)), []))
    out = residual_trend.tracklet_trends(f)
    assert len(out) == 4
    assert max(d["span_min"] for d in out) < 60


def test_trend_reference_is_the_arcs_own_worst():
    f = _FakeFit(_rows(2459000.5, [-1.2, -0.1, 1.3])
                 + _rows(2459010.5, [0.1, 0.0, -0.1]))
    ids = None
    ref = max(abs(d["swing_loo"]) for d in residual_trend.tracklet_trends(f, ids))
    assert ref == pytest.approx(2.5, abs=0.05)


def test_measure_trends_is_silent_without_linked_ids():
    f = _FakeFit(_rows(2459000.5, [-3.0, 0.0, 3.0]))
    assert residual_trend.measure_trends(f, set(), set()) is None


# ------------------------------------------------- the ratio's units, and trims

def test_the_ratio_is_raw_on_both_sides():
    # Leave-one-out shrinks a swing only where it can RUN.  Dividing an
    # unshrunk 3-observation candidate by a seed reference that leave-one-out
    # HAS shrunk inflates the ratio for arithmetic reasons -- so both sides are
    # the raw swing.  Here the seed's own 4-obs tracklet walks as hard as the
    # candidate does; whatever else is true, the candidate is not 4x the arc.
    seed = _FakeFit(_rows(2459000.5, [-3.0, -1.0, 1.0, 3.0]))
    ref = residual_trend.trend_reference(seed, None)
    cand = _stats([-3.0, 0.0, 3.0])
    assert ref == pytest.approx(abs(_stats([-3.0, -1.0, 1.0, 3.0])["swing"]),
                                rel=1e-6)          # raw, not the LOO value...
    assert ref > abs(_stats([-3.0, -1.0, 1.0, 3.0])["swing_loo"])   # ...which is smaller
    assert residual_trend._notable(cand, ref) is None


def test_a_trimmed_tracklet_that_cannot_be_supported_never_fires():
    # 408038's 2009-09-21 G96 in miniature: four observations, one of them
    # thrown out by the fit, leaving three -- so the support test cannot run AND
    # a bad observation is known to be present.  That combination is refused
    # whatever the trend looks like.
    rows = _rows(2459000.5, [-1.2, -0.1, 1.3, 9.0])
    f = _FakeFit(rows, included=[True, True, True, False])
    d = residual_trend.tracklet_trends(f)[0]
    assert d["n"] == 3 and d["n_rejected"] == 1 and d["trimmed"] is True
    assert d["loo_tested"] is False
    assert residual_trend._notable(d, ref=0.05) is None
    # ...and the very same three residuals, with nothing trimmed, DO fire:
    # it is the known-bad observation that withdraws the verdict, not the count.
    clean = _FakeFit(_rows(2459000.5, [-1.2, -0.1, 1.3]))
    dc = residual_trend.tracklet_trends(clean)[0]
    assert dc["trimmed"] is False
    assert residual_trend._notable(dc, ref=0.32) is not None


def test_an_untrimmed_tracklet_reports_no_rejections():
    f = _FakeFit(_rows(2459000.5, [-1.2, -0.1, 1.3]))
    d = residual_trend.tracklet_trends(f)[0]
    assert d["n_rejected"] == 0 and d["trimmed"] is False


def test_the_absolute_floor_carries_the_support_test():
    # The floor is applied to the LEAVE-ONE-OUT swing, which is what stops the
    # raw ratio from firing on a trend one observation is carrying.  Same case as
    # test_one_bad_endpoint_collapses_under_leave_one_out, stated as the reason.
    d = _stats([-0.1, 0.0, 0.1, 2.5])
    assert abs(d["swing"]) >= residual_trend.TREND_RELATIVE * 0.05      # ratio passes...
    assert abs(d["swing_loo"]) < residual_trend.TREND_REL_FLOOR_SIGMA   # ...floor does not
    assert residual_trend._notable(d, ref=0.05) is None


# ------------------------------------- the two-axis walk: a review WARNING
# 2013 RD21's live ITF search links C14PV12 (G96,
# 2019-10-08) with a photometric caution and -- until this rule -- no word
# about its residuals, which walk in BOTH axes across all four observations
# while the fit has rejected the first.  These are its real lines and its real
# residuals under the linked orbit (G96 sigma 0.5").

_C14PV12 = [
    ("     C14PV12  C2019 10 08.23661 01 05 49.17 -14 20 58.1          21.4 GV     G96",
     2458764.73661, -1.0906, -1.8058, False),
    ("     C14PV12  C2019 10 08.24141 01 05 48.93 -14 20 59.0          21.7 GV     G96",
     2458764.74141, -0.3780, -0.1698, True),
    ("     C14PV12  C2019 10 08.24621 01 05 48.66 -14 21 01.4          21.7 GV     G96",
     2458764.74621, -0.1002, -0.0348, True),
    ("     C14PV12  C2019 10 08.25102 01 05 48.41 -14 21 03.7          21.2 GV     G96",
     2458764.75102, 0.4780, 0.2045, True),
]


def _c14pv12():
    from pyroorb.itf.obs80_fields import obs_unique_id
    rows = [{"jd_utc": jd, "obscode": "G96", "sigma_ra": 0.5, "sigma_dec": 0.5,
             "d_ra_arcsec": ra, "d_dec_arcsec": de, "line": line}
            for line, jd, ra, de, _inc in _C14PV12]
    return (_FakeFit(rows, included=[c[4] for c in _C14PV12]),
            {obs_unique_id(c[0]) for c in _C14PV12})


def _both_axes(resid):
    """``_rows`` with the same walk in RA as in Dec."""
    rows = _rows(2459000.5, resid)
    for r in rows:
        r["d_ra_arcsec"] = r["d_dec_arcsec"]
    return rows


def test_c14pv12_is_silent_under_the_diagnostic_clause():
    # Why nothing was reported: the relative clause meets a TRIMMED 3-obs
    # remainder (1.7 sigma in RA, 0.8 in Dec) and declines to judge it.
    f, ids = _c14pv12()
    d = residual_trend.tracklet_trends(f, ids)[0]
    assert d["n"] == 3 and d["trimmed"] is True
    assert residual_trend._notable(d, ref=0.1) is None
    assert residual_trend.measure_trends(f, ids, set()) is None


def test_c14pv12_raises_the_two_axis_warning_with_its_rejection_back():
    f, ids = _c14pv12()
    d = residual_trend.two_axis_trend(f, ids)
    assert d is not None and d["with_rejected"] is True
    assert d["n"] == 4 and d["n_rejected_in"] == 1
    assert d["ra"]["swing"] == pytest.approx(2.99, abs=0.02)
    assert d["dec"]["swing"] == pytest.approx(3.70, abs=0.02)
    w = residual_trend.two_axis_trend_warning(f, ids)
    assert residual_trend.is_trend_note(w)                     # the review split sees it
    assert w.startswith("residual trend on a linked tracklet")
    assert "BOTH RA (+3.0σ) and Dec (+3.7σ)" in w
    assert "the 1 observation the fit rejected" in w
    assert "2019-10-08 G96" in w


def test_the_two_axis_warning_needs_both_axes():
    # A walk in ONE axis is the diagnostic's business, not this caution's:
    # either-axis fires on 10% of correct tracklets in the calibration sample.
    assert residual_trend.two_axis_trend(_FakeFit(_rows(2459000.5, [-1.5, 0.0, 1.5])),
                              None) is None
    assert residual_trend.two_axis_trend(_FakeFit(_both_axes([-1.5, 0.0, 1.5])),
                              None) is not None


def test_the_two_axis_warning_needs_a_walk_not_a_jump():
    assert residual_trend.two_axis_trend(_FakeFit(_both_axes([4.0, -0.2, 0.2])),
                              None) is None


def test_a_kept_only_walk_warns_without_mentioning_rejections():
    w = residual_trend.two_axis_trend_warning(_FakeFit(_both_axes([-1.5, 0.0, 1.5])), None)
    assert w is not None and "rejected" not in w


def test_the_two_axis_warning_needs_linked_ids():
    f, _ids = _c14pv12()
    assert residual_trend.two_axis_trend_warning(f, set()) is None


def test_two_axis_thresholds_are_the_measured_ones():
    # 2.0 sigma per axis, fully monotonic: 1 of 58 correct / 2 of 121 wrong on
    # the decisive calibration population (measured 2026-08-25).
    assert residual_trend.TREND_WARN_SWING == 2.0
    assert residual_trend.TREND_WARN_MONOTONIC_FRAC == 1.0


def test_a_stored_note_is_still_a_trend_note():
    """The note read "residual ramp on ..." until 2026-10-01, and stored batch results
    stores hold records written under that wording.  A re-cut of those records
    folds trend notes into the review warnings by this predicate, so it must
    know both -- while only the new wording is ever written."""
    assert residual_trend.TREND_NOTE == "residual trend on"
    assert residual_trend.is_trend_note("residual trend on a linked tracklet: ...")
    assert residual_trend.is_trend_note("residual ramp on a linked tracklet: ...")
    assert residual_trend.is_trend_note("residual ramp on the candidate night: ...")
    assert not residual_trend.is_trend_note("a candidate was not shown")
    assert not residual_trend.is_trend_note(None)
