"""Intra-tracklet residual TREND -- a DIAGNOSTIC, not a gate.

**It gates nothing (2026-08-25).**  It was built as a gate and ran as one, on
an operating point calibrated against ~40 correct tracklets that it never fired
on.  The full calibration then measured **2 false positives against 3 catches**
on the population where it is the last word, with no threshold separating them.
It is now computed, reported on every surface, and acted on by nothing in the
search -- though the batch CSVs count it as a REVIEW warning (see
``is_trend_note``).  The
argument, the evidence and the conditions for re-arming it are in the block
above the constants -- read that before changing anything here.

What follows describes what the statistic MEASURES, which is unchanged.

The acceptance gates judge a linked tracklet three ways: does it keep its
observations (retention), do those observations fit their uncertainties
(``search._accept_quality_ok`` clause 2, a chi2 test), and does the object's own
astrometry survive the link (``search.extension_degrades``).  All three measure
the SIZE of the residuals.  None of them looks at their ORDER.

A tracklet is a handful of observations spanning ~15-60 minutes.  Over that span
the object's apparent motion is, to any accuracy the astrometry can resolve, a
straight line -- so a correct orbit leaves that tracklet's residuals scattered
about zero with no time structure.  A residual that walks monotonically across
the tracklet means the orbit predicts the wrong RATE of motion there: it puts the
object in about the right PLACE at about the right time while moving at the wrong
speed.  That is the signature of a wrong linkage, and it is exactly what a chi2
test cannot see -- a +-1.25 sigma trend and a +-1.25 sigma scatter carry identical
chi2.

The candidate ranker already measures the PRE-fit version of this (the
``motion-mismatch`` feature: the tracklet's own rate against the corridor's
predicted rate), but only as an ML feature -- nothing gates on it, and after the
trial fit the evidence is discarded.  This module asks the post-fit question:
having bent the orbit to swallow the tracklet, does the tracklet still move at
the wrong rate?

WHAT COUNTS AS A TREND
---------------------
Three properties, all required, because each rules out a different innocent
explanation:

``swing`` -- SIZE.  The end-to-end change of a sigma-weighted straight line
    fitted to residual against time, divided by the tracklet's mean per-obs
    sigma.  The plain-language number: "2.5 sigmas in Dec from the first
    observation to the last".  Its companion ``t`` is the same swing divided by
    its OWN standard error, propagated from the per-observation sigmas rather
    than from the residual scatter (with three observations the scatter has one
    degree of freedom and estimates nothing).  ``swing`` says how big; ``t``
    says how sure.

``mono`` -- SHAPE.  The residuals must actually WALK, not jump.  Measured as the
    fraction of adjacent pairs that step the same way as the overall trend, so a
    smooth -1.2 -> -0.1 -> +1.3 scores 1.0 while a +2.0 -> -0.1 -> +0.1 (one bad
    first observation, which a line fit still reports as a steep slope) scores
    0.5 and is not a trend.

``loo`` -- SUPPORT.  The trend must be carried by the whole tracklet, not by one
    observation.  For n >= 4 the swing is recomputed with each single
    observation dropped in turn and the WEAKEST of those answers is the one that
    has to clear the absolute FLOOR.  Four observations where only the last is
    bad (-0.1, 0.0, +0.1, +2.5) fit a confident line and step monotonically --
    and collapse to nothing the moment that observation is removed, which is the
    difference between a bad measurement and a wrong rate.  A 3-observation
    tracklet cannot be tested this way (dropping one leaves two points, which
    define a line exactly), so it is judged on size and shape alone and its
    verdict is correspondingly weaker.

    Support constrains the FLOOR, not the ratio.  Leave-one-out shrinks a swing
    only where it can run, so a LOO ratio divides an unshrunk n == 3 numerator
    by a shrunk n >= 4 denominator and inflates it ~1.5x for arithmetic reasons
    -- see :func:`_notable`.  The ratio is therefore raw over raw and the floor
    carries the support requirement, which costs nothing measured and is the
    sharper statistic besides.

    The one place n == 3 is not merely weaker but WRONG is a tracklet the fit
    has TRIMMED: the count is of INCLUDED observations, so a 4-observation
    tracklet trimmed to 3 used to drop out of the support test and stay
    eligible -- and a trimmed observation is precisely the circumstance in which
    one bad point manufactures the trend (408038's 2009-09-21 G96 tracklet
    ratios 1.49 with four kept and 7.34 trimmed to three).  Those tracklets now
    do not fire; :func:`tracklet_trends` reports ``trimmed``/``n_rejected`` so
    the gate can see it.

``t`` (the slope's significance against its own propagated error) is computed and
    reported throughout, because it is what a reader wants to see -- but nothing
    gates on it: on the population that decides, it discriminated no better than
    chance (AUC 0.502).

Sigmas are Veres-2017 station-floored, like every other sigma-normalised test in
the search -- an over-tight reported sigma must not be able to manufacture a
trend, nor to hide one.

RELATIVE, NOT ABSOLUTE
----------------------
Real astrometry trends a little: differential refraction, an uncorrected trailing
loss, a slightly wrong exposure timestamp, a star catalogue gradient across the
field.  So a linked tracklet's trend is judged against what the OBJECT'S OWN
published tracklets do under the same orbit (:func:`trend_reference`).  A survey
whose own nights all trend 2 sigma has told us that 2 sigma is its noise; a linked
tracklet that trends well outside that arc's habits has told us something else.

That relative form is not a preference, it is the one that measured any power at
all -- see the calibration note above the constants.
"""

import numpy as np

from .obs80_fields import obs_unique_id

# A trend needs at least three points: two define a line exactly and leave
# nothing over to say whether it is a line or a coincidence.
TREND_MIN_OBS = 3

# --- WHAT IS ONE TRACKLET ---------------------------------------------------
# The whole statistic rests on "over this span the motion is a straight line",
# so the grouping has to produce TRACKLETS, not nights.  The first cut of this
# file grouped by NIGHT_GAP_DAYS (0.5 d) alone, station-blind, and chained: any
# two observations within half a day joined, and the chain carried on.  Over the
# 5548 such groups in the regression archive that produced 858 groups mixing two
# or more OBSERVATORIES and spans up to 3690 minutes -- 2.6 DAYS presented to a
# straight-line test.  A 2.6-day "tracklet" measures the object's real sky
# curvature and the two stations' relative offsets, neither of which is a rate
# error, and it inflates the reference (`trend_reference`) that every candidate
# is divided by.
#
# So: a tracklet is one STATION, and consecutive observations no further apart
# than `TRACKLET_GAP_DAYS`.  The threshold is measured on the same archive --
# adjacent-pair gaps within a night run p50 7.7 min, p90 32, p95 64, and then
# jump to p99 391.  120 minutes sits in that empty zone: it splits 3.1% of
# adjacent pairs (the ones that are not one tracklet) and leaves genuine
# tracklets whole -- resulting spans p50 33 min, p99 219, and the count of
# testable n>=3 tracklets goes UP (5548 -> 6419), because merged groups split
# into real ones rather than being lost.
TRACKLET_GAP_DAYS = 120.0 / 1440.0

# --- CALIBRATED OPERATING POINT -------------------------------------------
# ONE clause, and the reason there is only one is measured.
#
# The calibration study built matched pairs in the regime this gate
# actually runs in -- a WEAK one-apparition seed, the shape an ITF search always
# has.  Each pair holds out a real tracklet of object A (known good, it IS the
# object) and, from the same night and field, a DESIGNATED tracklet of some other
# object B (known wrong, by the MPC's own `provid`).  70 objects gave 60 good and
# 206 wrong tracklets that reached this test.
#
# Two findings decide the design:
#
#  1. 63% of wrong tracklets (129/206) are ALREADY rejected by retention, the
#     per-tracklet chi2 and the seed guard.  This test is only ever the last
#     word on the rest, so it must be judged on THOSE -- 39 good, 77 wrong.
#
#  2. On those, an ABSOLUTE sigma-swing threshold has no discriminating power
#     at all (AUC 0.476, i.e. chance) while the RELATIVE one does (AUC 0.815).
#     That is not a surprise once stated: the per-tracklet chi2 gate already
#     tests sigma-normalised residual SIZE, and an absolute swing is another
#     sigma-normalised size measure, so it carries nothing the gate upstream
#     has not already used.  Measured against the object's OWN arc it asks a
#     question no other gate asks.
#
# So the absolute clause was deleted.  It cost 2/39 good tracklets on that
# population (and 17/60 across everything reaching the test) to catch 2/77 wrong
# -- paying real recoveries for nothing.
#
# CALIBRATE IN THE RIGHT REGIME.  An earlier version of this file set its
# thresholds from full fits of 4+-opposition objects -- strong orbits, where
# everything fits -- and claimed 1 false positive in 1000.  The gate does not run
# there.  In the weak-seed regime `ratio >= 4.0` alone fires on 8.7% of GOOD
# tracklets, not 0.1%: two orders of magnitude out.
#
# --- WHAT THESE NUMBERS DO AND DO NOT ESTABLISH (2026-08-05) ---------------
# The operating point quoted above and below is **provisional**, on four counts
# found by re-reading the evidence file rather than the summary.  None of them
# has moved a threshold -- the constants are unchanged -- but none of them
# should be quoted as settled either.  `power.py --report` and
# `power.py --variants` print all four.
#
#  (a) THE UNIT -- SETTLED, and it is not the problem.  `pairs-2026-08-04.csv`
#      was collected before a tracklet became "one station, gaps <= 120 min"
#      (TRACKLET_GAP_DAYS above) and still contains 1429-minute groups, which
#      mattered because regrouping moves BOTH sides of the ratio.  The matched
#      re-run (`pairs-2026-08-05.csv`, FINDINGS.md Finding 4) settles it: on the
#      same 10 objects under both units the clause fires identically.  What that
#      re-run ALSO shows, and the summary does not say, is that on those 10
#      objects the decisive-population catch count is **ZERO under both units**
#      (0/17 old, 0/19 new).  Both of the catches behind "2 of 77" and "roughly
#      1 in 40" live in the 13 objects that were NOT re-run.  So the unit
#      question is closed and the DETECTION rate is still carried entirely by
#      un-re-run data.
#
#  (b) THE RATE IS OVER A POPULATION THAT IS MOSTLY INERT.  `_notable` returns
#      None on its first line when `ref <= 0`, and the seed has no measurable
#      tracklet of its own in 64% of the 08-04 rows.  "0 of 39 good, 2 of 77
#      wrong" is therefore diluted by 88 tracklets the clause could never fire
#      on; over the rows it can act on it is **0 of 15 good, 2 of 13 wrong**.
#      The re-run does not improve this -- on 08-05 the decisive population has
#      **9 good and 7 wrong** rows with a yardstick at all.  Whatever this gate
#      is, it is being judged on a couple of dozen tracklets.
#
#  (c) THE SHARPER STATISTIC IS NOW THE SHIPPED ONE (2026-08-25).  0.815 is
#      `ratio_raw` (swing / ref_raw); the clause used to test `swing_loo /
#      ref_loo`, which scores 0.785, and the re-run says the same more sharply
#      (0.857 vs 0.794).  Leave-one-out shrinks a swing by a median factor 0.67
#      where it can run and by exactly 1.00 where it cannot, so a LOO ratio
#      inflates ~1.5x whenever a small candidate is judged against a larger seed
#      tracklet -- an artefact of the arithmetic.  The ratio is therefore taken
#      raw over raw, and the support requirement moved to the absolute FLOOR,
#      which is still applied to `swing_loo`: a trend one observation is carrying
#      collapses under LOO and cannot clear 2.0 sigma, so the property the
#      module was designed around is kept while the artefact is dropped.
#      Measured both ways on both samples, FALSE POSITIVES ARE ZERO EITHER WAY
#      and the catch count only goes up: over everything reaching the test
#      17 -> 19 of 206 (08-04) and 10 -> 12 of 86 (08-05); on the decisive
#      population 2 -> 2 of 77 and 0 -> 0 of 19.  The decisive count is
#      unchanged but the ROWS are not, and that is the point: the design drops
#      408038's trimmed 2009-09-21 (see (d)) and gains 659414's 2008-03-01, a
#      four-observation tracklet with nothing trimmed that the LOO ratio missed
#      at 2.39 sigma against 4x0.67.
#
#      What was NOT adopted, and why: the bare `raw` ratio (no LOO anywhere)
#      catches one more decisive row on each sample, but it fires on
#      `-0.1, 0.0, +0.1, +2.5` -- one bad endpoint, the case the support test
#      exists for.  Its extra catch is a single tracklet (229521 2009-12-17,
#      swing 2.89 collapsing to 1.53 under LOO) and it is the same tracklet in
#      both files, i.e. one data point, which does not buy the removal of a
#      structural protection.  Requiring `loo_tested` outright was not adopted
#      either: it disqualifies every 3-observation tracklet, ~40% of everything
#      reaching the test, including the untrimmed triples the gate was built on
#      (2026 OE3's P21l7qL), and it costs catches on both samples (18 and 12
#      over everything reaching the test) for no measured gain.
#
#  (d) THE SUPPORT TEST IS NO LONGER SKIPPED WHERE THE FIT TRIMS A TRACKLET.
#      `loo_tested` follows the KEPT observation count, not `n`, so a
#      4-observation tracklet the fit trims to 3 lost the leave-one-out test
#      and stayed eligible to fire -- and a trimmed observation is exactly the
#      circumstance in which one bad point manufactures the trend.  408038's
#      2009-09-21 G96 tracklet is the illustration: with 4 obs kept it ratios
#      1.49 and stays silent, trimmed to 3 it ratios 7.34 and fires.
#      `tracklet_trends` now reports `trimmed`/`n_rejected` and `_notable` refuses
#      that combination.  On both calibration samples the clause is INERT --
#      the raw ratio of (c) already refuses 408038 (4.33 against 4 x 1.58), and
#      no other row is both trimmed and untestable and past the thresholds --
#      so it is kept as a guard against a case with a worked example rather
#      than as a measured filter, and it costs nothing.
#
# --- THE GATE WAS WITHDRAWN (2026-08-25).  READ THIS BEFORE RE-ARMING IT. ---
# Everything above is the record of an operating point set on ~40 correct
# tracklets, on which it never once fired.  The full 70-object run that the
# note above says "is what decides it" was collected on 2026-08-25
# (`pairs-2026-08-25.csv` + `-25b.csv`, 38 objects that survived the database,
# 441 rows) and it decided the other way:
#
#   on the population where this test is the LAST WORD, the clause fires on
#   **2 of 58 CORRECT tracklets and 3 of 119 wrong ones**.
#
# It rejects about as much real astrometry as junk.  The earlier "0 GOOD in
# every sample" was a small-sample artefact, and the earlier reading -- "not
# proven to help, but proven harmless" -- is false in both halves.
#
# And there is no threshold that rescues it.  Scanning the multiplier on the
# decisive population: 4x -> 2 false / 3 caught, 6x -> 2/2, 8x -> 1/1,
# 12x -> 0/0.  The false-positive rate reaches zero exactly where the catch
# rate does.  Object 518538 shows why in one line: it supplies a CORRECT
# tracklet trending 6.0x and a WRONG one trending 10.1x against the same 0.41
# sigma yardstick, so the two classes are not merely close, they are
# interleaved.
#
# The statistic is not worthless -- `ratio_raw` scores AUC 0.767 on that
# population, comfortably the best of the five measured, and the raw-vs-LOO
# and trimmed-tracklet fixes above are real improvements to it.  What it does
# not have is a usable operating point, and a gate is an operating point.
#
# So the clause still COMPUTES and is still REPORTED -- as
# `ItfSearchResult.diagnostics`, which the CLI prints as NOTE, the web UI
# renders as an info banner and the batch CSVs carry in their own
# columns -- and it decides nothing.  Both former consumers were disarmed:
# the lone-night show no longer silences a lone new night over it, and the
# accepted-linkage path no longer files a warning, which is what used to
# withdraw a fragile three-thin confirmation.
#
# A REVIEW WARNING in the CSVs, still not a gate.  "Acted
# on by nothing" was right for the SEARCH -- no veto, no silenced night, no
# withdrawn three-thin confirmation -- but it hid the trend from the review pile:
# a batch review filed a trending linkage "clean", beside no hint that it deserves the
# second look a slight photometric disagreement gets.  So the reporting layer
# (the batch CSV `warnings` column and a review's clean/warned
# split) counts a trend note as a warning, through `is_trend_note` below; the
# search's own `ItfSearchResult.warnings` -- which gate the submit click-through
# and withdraw a fragile confirmation -- still never carry it.
#
# To re-arm it you need an operating point measured on the decisive
# population with a false-positive rate you are willing to pay -- which this
# sample says does not exist at any multiplier that catches anything.  The
# thresholds below are kept UNCHANGED so that the diagnostic keeps describing
# the same event the study measured; they are no longer a gate's calibration.
TREND_MIN_MONOTONIC_FRAC = 1.0            # every adjacent step must go the trend's way (LOO)
TREND_RELATIVE = 4.0        # x the object's own worst other tracklet, RAW both
#                            sides -- see (c) and _notable
TREND_REL_FLOOR_SIGMA = 2.0 # ...and at least this big absolutely AFTER dropping
#                            the most helpful observation, so neither an
#                            immaculate arc nor one bad point makes a trend

# Every trend note the search files starts with this, so a reader of
# `ItfSearchResult.diagnostics` can tell a residual trend from the other
# diagnostics (a candidate that was not shown, ...) without matching free text.
TREND_NOTE = "residual trend on"
# What the note said before 2026-10-01.  Stored batch results already hold
# records whose diagnostics start with it (115 on the day of the change), and a
# re-cut of those records must still fold them into the review warnings
# (old words on disk are read, never rewritten) -- so this prefix
# is recognised and never written.
_TREND_NOTE_STORED = ("residual ramp on",)


def is_trend_note(text):
    """True for a diagnostic the search filed about a residual trend -- under
    today's prefix or the one already in stored records."""
    return isinstance(text, str) and text.startswith(
        (TREND_NOTE,) + _TREND_NOTE_STORED)


def _sigmas(r, obscode, station_floor=True):
    """The residual row's per-axis sigma (arcsec), raised to the observatory's
    Veres et al. (2017) floor when ``station_floor``.

    Same rule as :func:`itf.fitter._obs_sigmas`, but keyed on the OBSCODE rather
    than on an 80-column line, because this test also runs on fits whose
    observations came from ADES and carry no line (see
    :func:`tracklet_trends`)."""
    sra, sdec = r.get("sigma_ra"), r.get("sigma_dec")
    if not (sra and sdec):
        return None
    if station_floor:
        from pyroorb.weights import VERES_2017, DEFAULT_SIGMA
        vf = VERES_2017.get(obscode, DEFAULT_SIGMA)
        sra, sdec = max(sra, vf), max(sdec, vf)
    return float(sra), float(sdec)


def _wls_trend(t, r, s):
    """Sigma-weighted linear trend of ``r`` against ``t``.

    Returns ``(swing, tstat, sigma_mean)`` -- the end-to-end change of the
    fitted line in units of the mean per-obs sigma, that change divided by its
    own standard error, and the sigma it was normalised by."""
    t = np.asarray(t, float)
    r = np.asarray(r, float)
    s = np.asarray(s, float)
    span = float(t.max() - t.min())
    sig_mean = float(np.sqrt(np.mean(s ** 2))) if s.size else 0.0
    if span <= 0 or s.size == 0 or np.any(s <= 0):
        return 0.0, 0.0, sig_mean
    w = 1.0 / s ** 2
    dt = t - float(np.sum(w * t) / np.sum(w))
    sxx = float(np.sum(w * dt ** 2))
    if sxx <= 0:
        return 0.0, 0.0, sig_mean
    slope = float(np.sum(w * dt * r) / sxx)
    swing = slope * span
    # se(slope) from the ASSIGNED sigmas, not from the residual scatter: three
    # points leave one degree of freedom, and a scatter-based error bar on that
    # is noise pretending to be a measurement.
    se_swing = float(np.sqrt(1.0 / sxx)) * span
    return (swing / sig_mean if sig_mean > 0 else 0.0,
            swing / se_swing if se_swing > 0 else 0.0, sig_mean)


def _monotonicity(r, swing):
    """Fraction of adjacent steps that move the way the overall trend does.

    1.0 is a clean walk; 0.5 on three points means one observation jumped and
    the other stayed put, which a straight line reports as a slope but is not a
    trend.  Ties (an exactly repeated residual) count as half a step, so a flat
    pair neither confirms nor denies the trend."""
    r = np.asarray(r, float)
    if r.size < 2 or swing == 0:
        return 0.0
    d = np.diff(r) * np.sign(swing)
    return float(np.mean(np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))))


def _trend_stats(t, r, s):
    """Full trend description of one axis of one tracklet.

    ``swing``/``t``/``mono`` are the whole-tracklet numbers -- ``swing`` is what
    :func:`_notable` ratios against the object's own arc; ``swing_loo`` and
    ``t_loo`` are the WEAKEST values found by dropping each single observation
    in turn (n >= 4 only -- see the module docstring), i.e. what survives when
    the single most helpful observation is taken away, and ``swing_loo`` is what
    the absolute floor is applied to.  For n == 3 they repeat
    the whole-tracklet values, and ``loo_tested`` records that the support test
    could not be run."""
    swing, tstat, sig = _wls_trend(t, r, s)
    mono = _monotonicity(r, swing)
    n = len(t)
    if n >= TREND_MIN_OBS + 1:
        worst = None
        for k in range(n):
            keep = [j for j in range(n) if j != k]
            sw2, t2, _ = _wls_trend([t[j] for j in keep], [r[j] for j in keep],
                                   [s[j] for j in keep])
            m2 = _monotonicity([r[j] for j in keep], sw2)
            cand = (abs(sw2), abs(t2), m2, sw2, t2)
            if worst is None or cand[:2] < worst[:2]:
                worst = cand
        swing_loo, t_loo, mono_loo = worst[3], worst[4], worst[2]
        loo_tested = True
    else:
        swing_loo, t_loo, mono_loo, loo_tested = swing, tstat, mono, False
    return {"swing": swing, "t": tstat, "mono": mono, "sigma_mean": sig,
            "swing_loo": swing_loo, "t_loo": t_loo, "mono_loo": mono_loo,
            "loo_tested": loo_tested}


def _notable(d, ref):
    """Does this tracklet trend?  Returns ``None`` or a short reason.

    Four things together, and ``ref`` -- the object's own worst tracklet trend,
    :func:`trend_reference` -- is the scale the size of it is judged against.
    Without a reference there is no question to ask, so a seed with no
    measurable tracklet of its own never fires.

    The RATIO is taken on the raw swing, both sides.  Leave-one-out shrinks a
    swing only where it can run (n >= 4, median factor 0.67) and not at all at
    n == 3, so a LOO ratio silently multiplies a small candidate judged against
    a larger seed tracklet by ~1.5x -- an artefact of the arithmetic, not of the
    astrometry.  Raw over raw is the same measurement on both sides, and it is
    also the sharper statistic where it counts: on the decisive population it
    scores AUC 0.815 / 0.857 across the two calibration samples against the LOO
    ratio's 0.785 / 0.794.

    The SUPPORT test survives as the absolute floor, which is applied to the
    leave-one-out swing: a trend that collapses when its most helpful
    observation is dropped cannot clear it (``-0.1, 0.0, +0.1, +2.5`` swings
    2.6 sigma and 0.4 with the last point gone).  At n == 3 the LOO swing IS
    the swing, so an untrimmed triple is judged exactly as before.

    And where the fit has already TRIMMED an observation out of the tracklet,
    a swing that cannot be leave-one-out tested does not fire at all.  That is
    the one case where "no support test" and "a bad observation is known to be
    present" coincide, and firing there decides the gate on the single point
    LOO exists to discount (408038's 2009-09-21 G96 tracklet: 4 kept, silent;
    trimmed to 3, it fired)."""
    sw, sw_loo, mono = abs(d["swing"]), abs(d["swing_loo"]), d["mono_loo"]
    if ref <= 0 or mono < TREND_MIN_MONOTONIC_FRAC:
        return None                       # no yardstick, or it jumped not walked
    if d.get("trimmed") and not d["loo_tested"]:
        return None                       # a known bad obs, and no way to test it
    if sw >= TREND_RELATIVE * ref and sw_loo >= TREND_REL_FLOOR_SIGMA:
        return (f"{sw / ref:.1f}x the worst trend anywhere else in this object's "
                f"arc ({ref:.2f} sigma), past the {TREND_RELATIVE:g}x limit")
    return None


def split_tracklets(items, key):
    """Split time-sorted ``items`` into tracklets -- one station, consecutive
    observations no further apart than :data:`TRACKLET_GAP_DAYS`.

    ``key(item)`` returns ``(jd, station)``.  Shared with the calibration
    study's pair builder so the experiment and the gate cannot drift
    apart on what a tracklet IS: the calibration measures the statistic on
    whatever this function returns, so a second definition over there would
    calibrate one unit and ship another."""
    groups = []
    for it in items:
        jd, stn = key(it)
        if groups:
            pjd, pstn = key(groups[-1][-1])
            if stn == pstn and jd - pjd <= TRACKLET_GAP_DAYS:
                groups[-1].append(it)
                continue
        groups.append([it])
    return groups


def tracklet_trends(fit, obs_ids=None, *, station_floor=True, include_rejected=False):
    """Per-tracklet residual trend for ``fit``, INCLUDED observations only.

    ``include_rejected=True`` reads the rejected observations too, and exists for
    one specific question the default cannot answer: a fit that cannot reconcile
    a tracklet may resolve it by EJECTING that tracklet's worst observation and
    then fitting the remainder cleanly -- which removes the very point that made
    the tracklet walk, so the trend becomes unmeasurable exactly when it is the
    thing worth measuring.  (2023 QA131: the isolated tracklet loses one of three
    observations and the surviving pair says nothing.)  The ITF engine keeps the
    default: there a rejected observation is not part of the arc the search
    accepted, and its residual can be enormous.

    Groups the included obs into tracklets (:func:`split_tracklets` -- one
    station, gaps <= :data:`TRACKLET_GAP_DAYS`), optionally restricted to
    ``obs_ids`` (an ``obs_unique_id`` set -- e.g. just the linked tracklets, or
    just the seed's own).  Returns a list of dicts, worst-first, one per tracklet
    with at least :data:`TREND_MIN_OBS` observations::

        {"night": "2021-09-13", "station": "F52", "n": 3, "axis": "Dec",
         "swing": 2.53, "t": 1.79, "mono": 1.0, "sigma_mean": 0.42,
         "ra": {...}, "dec": {...}}

    ``axis`` names whichever axis trends harder and the top-level ``swing``/``t``/
    ``mono`` are that axis'; ``ra`` and ``dec`` carry both full stat dicts,
    because which axis holds the rate error is itself informative -- an
    along-track rate error lands in whichever axis the sky motion is aligned
    with."""
    from datetime import datetime, timedelta

    if fit is None or getattr(fit, "_resid_all", None) is None:
        return []
    try:
        rows = fit.residuals()
    except Exception:
        return []
    pts, rej = [], []
    for o, r, inc in zip(fit._obs, rows, fit.is_included()):
        # An ITF trial fit's obs were parsed from 80-column lines and carry
        # them; an ordinary ADES fetch's obs do not.  The line is only needed
        # to answer "is this one of the ids I was asked about" -- the station,
        # which is all the sigma floor wants, is on the residual row either way.
        line = o.get("obs80_line")
        if obs_ids is not None and (not line
                                    or obs_unique_id(line) not in obs_ids):
            continue
        code = (line[77:80] if line else (r.get("obscode") or "")).strip()
        if not inc and not include_rejected:
            # Not part of the arc -- but REMEMBER it.  A tracklet the fit has
            # trimmed is one where a bad observation is known to be present,
            # which is exactly the circumstance in which a leave-one-out test
            # that cannot run must not be waived (see :func:`_notable`).  Only
            # the time and the station are needed to attribute it to a
            # tracklet; its residual is meaningless and its sigma unused.
            rej.append((float(o["jd_utc"]), code))
            continue
        sig = _sigmas(r, code, station_floor)
        if sig is None:
            continue
        pts.append((float(o["jd_utc"]), float(r["d_ra_arcsec"]),
                    float(r["d_dec_arcsec"]), sig[0], sig[1], code,
                    0 if inc else 1))
    if len(pts) < TREND_MIN_OBS:
        return []
    pts.sort()
    groups = split_tracklets(pts, lambda p: (p[0], p[5]))

    out = []
    for g in groups:
        if len(g) < TREND_MIN_OBS:
            continue
        t = [p[0] for p in g]
        ra = _trend_stats(t, [p[1] for p in g], [p[3] for p in g])
        dec = _trend_stats(t, [p[2] for p in g], [p[4] for p in g])
        night = (datetime(1970, 1, 1)
                 + timedelta(days=t[0] - 2440587.5)).strftime("%Y-%m-%d")
        name, worst = (("Dec", dec) if abs(dec["swing"]) > abs(ra["swing"])
                       else ("RA", ra))
        # Rejected observations of THIS tracklet: same station, and inside its
        # span widened by one tracklet gap so an obs trimmed off either end
        # still belongs to it.
        j0, j1, stn = t[0], t[-1], g[0][5]
        n_rej = sum(1 for jd, c in rej if c == stn
                    and j0 - TRACKLET_GAP_DAYS <= jd <= j1 + TRACKLET_GAP_DAYS)
        out.append({"night": night, "station": g[0][5], "n": len(g),
                    "axis": name, "ra": ra, "dec": dec,
                    "n_rejected": n_rej, "trimmed": n_rej > 0,
                    # rejected obs measured INSIDE this group -- nonzero only
                    # under include_rejected, where `n_rejected` is 0 because
                    # nothing was set aside
                    "n_rejected_in": sum(p[6] for p in g),
                    # jd0/jd1 so a caller can match a trend back to ITS OWN
                    # tracklet by time containment.  ``night`` is a calendar
                    # date string and two tracklets of one station can share
                    # one -- keying on it silently hands one tracklet another's
                    # trend (2014 EO217, 2025 NO35).
                    "jd0": t[0], "jd1": t[-1],
                    "span_min": (t[-1] - t[0]) * 1440.0, **worst})
    out.sort(key=lambda d: -abs(d["swing"]))
    return out


def trend_reference(fit, seed_ids):
    """The object's OWN trend scale under ``fit``: the largest |swing| among the
    seed's published tracklets.  The yardstick a linked tracklet's trend is
    measured against -- see the module docstring.  ``0.0`` when the seed has no
    tracklet long enough to measure -- and with no yardstick nothing fires.

    RAW swing, matching the numerator :func:`_notable` divides by it: a reference
    built from ``swing_loo`` is shrunk by leave-one-out wherever the seed's own
    tracklets have four or more observations and not at all where they have
    three, so the same candidate would be measured against a yardstick that
    changes size with the seed's tracklet lengths."""
    return max((abs(d["swing"]) for d in tracklet_trends(fit, seed_ids)),
               default=0.0)


def describe(d):
    """One line of prose for a trend dict from :func:`tracklet_trends`."""
    loo = ("" if not d["loo_tested"] else
           f", and survives dropping any single observation "
           f"({abs(d['swing_loo']):.1f} sigma at worst)")
    if d.get("n_rejected"):
        loo += (f" [the fit rejected {d['n_rejected']} observation"
                f"{'s' if d['n_rejected'] > 1 else ''} of this tracklet]")
    return (f"the {d['night']} tracklet ({d['station']}, n={d['n']}, "
            f"{d['span_min']:.0f} min) walks {d['swing']:+.1f} sigma in "
            f"{d['axis']} from its first observation to its last "
            f"({abs(d['t']):.1f} sigma significant, monotonicity "
            f"{d['mono']:.2f}){loo}")


def measure_trends(fit, linked_ids, seed_ids=None, detailed=False):
    """Does a LINKED tracklet trend in a way the object's own arc does not?

    Returns a short description of the worst offender, or ``None``.  A linked
    tracklet is flagged when it walks monotonically AND trends
    :data:`TREND_RELATIVE` x harder than anything in the object's own published
    arc AND still swings :data:`TREND_REL_FLOOR_SIGMA` with its most helpful
    observation dropped.

    The default text is written for a READER -- the banner, the CLI ``NOTE:``,
    the batch CSV -- and carries the three numbers a person can act on (which
    tracklet, how far it drifts, what the rest of the arc manages).  It used to
    carry six, including the ``TREND_RELATIVE`` limit it had crossed, which sat
    in the same sentence as "does not count against the linkage" and read as a
    contradiction.  ``detailed=True`` returns the full statistic
    (:func:`describe` + :func:`_notable`) for the trace and the search log.

    NOTE: this is a DIAGNOSTIC, not a gate -- nothing gates on its answer (see the
    module header). The "0 of 39 good, 2 of 77 wrong" that used to be quoted here was
    the ~40-tracklet artefact the full calibration overturned: on 38 objects / 441 rows
    it fires on 2 of 58 CORRECT tracklets and 3 of 119 wrong, which is why the gate was
    withdrawn."""
    if not linked_ids:
        return None
    ref = trend_reference(fit, seed_ids or set())
    for d in tracklet_trends(fit, linked_ids):
        why = _notable(d, ref)
        if why:
            if detailed:
                return (f"{describe(d)} -- {why}. The orbit has this tracklet "
                        f"moving at the wrong RATE, not merely sitting in the "
                        f"wrong place")
            return (f"the residuals of the {d['night']} {d['station']} tracklet "
                    f"drift steadily in {d['axis']} over its {d['span_min']:.0f} "
                    f"minutes, about {abs(d['swing']):.1f}σ end to end, where "
                    f"nothing else in this arc drifts more than {ref:.1f}σ. "
                    f"That can mean the orbit has this tracklet's rate slightly "
                    f"wrong")
    return None


# --- THE TWO-AXIS WALK: a review WARNING, not the withdrawn gate -------------
# Motivating case: 2013 RD21's C14PV12 (G96 2019-10-08, four obs).
# The linked orbit leaves that tracklet walking in BOTH axes -- RA -1.09,
# -0.38, -0.10, +0.48", Dec -1.81, -0.17, -0.03, +0.20" -- and nothing said so:
# the fit had rejected the first observation, which carries much of both walks,
# so the relative clause above saw a 3-obs remainder (1.7 sigma in RA, 0.8 in
# Dec) and its trimmed-tracklet guard declined to judge it at all.  A tracklet
# whose residuals walk the same way in RA AND Dec is one the orbit has moving
# along the wrong sky VECTOR, which is what a different body looks like; that
# belongs in front of the reviewer as a caution, beside the photometric
# one, not as a NOTE.
#
# So this clause asks a different, simpler question than `_notable`: no seed
# yardstick, no leave-one-out -- does each axis walk monotonically, by at least
# `TREND_WARN_SWING` sigma end to end?  It is asked twice, of the KEPT
# observations and of the whole tracklet with the fit's rejections put back
# (`tracklet_trends(include_rejected=True)`), because a fit that cannot
# reconcile a walking tracklet resolves it by ejecting the end of the walk
# (2023 QA131 is the same shape) and the kept remainder then says nothing.
#
# WHAT IT MEASURES (calibration pairs 2026-08-25 + 25b, kept obs only --
# the rejected-obs variant was not collected there): on the population where a
# trend is the last word, both axes >= 2.0 sigma and monotonic fires on 1 of 58
# CORRECT tracklets and 2 of 121 wrong.  It does not discriminate -- nothing in
# that sample does, see above -- but it is RARE on correct links (~2%), which
# is what a review prompt needs.  That measurement is exactly why it is a
# warning that nothing in the search acts on: it is appended PAST the fragile
# 1.75 withdrawal and the seed-degradation demotion in `search.py`, like the
# unchecked-pickup caution, so it asks for a human's look and never takes a
# submission away.  Either axis alone (the tempting widening) fires on 10% of
# correct tracklets and would be noise.
TREND_WARN_SWING = 2.0      # sigma end to end, in EACH axis
TREND_WARN_MONOTONIC_FRAC = 1.0       # every adjacent step the trend's way, in each axis


def _walks_both(d):
    """Does tracklet ``d`` (a :func:`tracklet_trends` dict) walk in both axes?"""
    return all(d[ax]["mono"] >= TREND_WARN_MONOTONIC_FRAC
               and abs(d[ax]["swing"]) >= TREND_WARN_SWING for ax in ("ra", "dec"))


def two_axis_trend(fit, linked_ids):
    """The worst LINKED tracklet whose residuals walk in both RA and Dec, or
    ``None``.  Returns the :func:`tracklet_trends` dict plus ``with_rejected``
    (True when the walk is only complete with the fit's rejected observations
    put back -- the reader must be told that, since the orbit itself discounted
    those points).  See the block above for the rule and what it measures.
    ``linked_ids`` None measures every tracklet; an EMPTY set means nothing was
    linked, and answers None."""
    if linked_ids is not None and not linked_ids:
        return None
    best = None
    for with_rej in (False, True):
        for d in tracklet_trends(fit, linked_ids, include_rejected=with_rej):
            if with_rej and not d.get("n_rejected_in"):
                continue            # identical to the kept-only measurement
            if not _walks_both(d):
                continue
            size = min(abs(d["ra"]["swing"]), abs(d["dec"]["swing"]))
            if best is None or size > best[0]:
                best = (size, dict(d, with_rejected=with_rej))
    return best[1] if best else None


def two_axis_trend_warning(fit, linked_ids):
    """The reviewer-facing caution for :func:`two_axis_trend`, or ``None``.

    Starts with :data:`TREND_NOTE` so every consumer that already recognises a
    trend (``is_trend_note``: the batch review split) recognises this
    one."""
    d = two_axis_trend(fit, linked_ids)
    if d is None:
        return None
    rej = ""
    if d["with_rejected"]:
        k = d["n_rejected_in"]
        rej = (f", counting the {k} observation{'s' if k > 1 else ''} the fit "
               f"rejected from it")
    return (f"{TREND_NOTE} a linked tracklet: the residuals of the {d['night']} "
            f"{d['station']} tracklet ({d['n']} obs, {d['span_min']:.0f} min) "
            f"walk steadily in BOTH RA ({d['ra']['swing']:+.1f}σ) and Dec "
            f"({d['dec']['swing']:+.1f}σ) from its first observation to its "
            f"last{rej}. The linked orbit may have this tracklet moving at the "
            f"wrong rate (possible different body)")


def review_warnings(warnings, diagnostics):
    """The warnings a REVIEWER should see: the search's own ``warnings`` plus
    every residual-trend note among its ``diagnostics``.

    The trend was withdrawn as a GATE on 2026-08-25 (it flagged 2 of 58
    correct tracklets against 3 of 119 wrong) and
    stays withdrawn -- the search never lets it veto, silence a night or take a
    three-thin confirmation away, which is why it rides in ``diagnostics`` there.
    But a trending linkage is not a clean one either: like a slight photometric
    disagreement it is doubtful enough to deserve a look before submission.  So
    at the REPORTING layer -- a batch report's ``warnings`` column and any review
    that splits clean from warned on it -- it counts as one."""
    out = list(warnings or [])
    out += [d for d in (diagnostics or []) if is_trend_note(d) and d not in out]
    return out
