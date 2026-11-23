"""Mislinkage scan for SINGLE-OPPOSITION objects -- is one tracklet not this body?

A three- or four-tracklet single-opposition arc with a big gap in it is the
catalogue's weakest linkage: a discovery cluster of a couple of nights, then a
recovery tracklet a few weeks later that was attributed on an orbit nobody could
test.  Sometimes the recovery tracklet is a different object.  This scans for
that, and reports the evidence rather than a verdict.

WHY THE ASTROMETRY USUALLY CANNOT DECIDE, and why this tool has two arms.  Three
tracklets contribute six sky coordinates and an orbit costs six parameters, so
the fit threads them EXACTLY; four tracklets leave two degrees of freedom (the
dof note in :mod:`itf.night_coherence` states this for nights, and it is the same
arithmetic).  Every one of the five worked examples below fits at rms_normalized
0.39-0.81 -- the joint orbit reaches every tracklet well inside its own stated
uncertainties, mislinked or not.  There is no misfit to find.  A body's ABSOLUTE
MAGNITUDE is not subject to that arithmetic: one body has one H, however few
degrees of freedom its astrometry has left.  So the photometric arm carries most
of the detection and the astrometric arm catches the rest.

THE FIVE STATISTICS, each per tracklet, one against the rest of the arc:

  ``dH``          implied-H offset (:func:`ident_check.tracklet_mag_audit`), with
                  the same significance grid as ``--ident-check`` and
                  ``--mag-audit``.  The workhorse: 4 of the 5 worked examples.
  ``drop_ratio``  how much worse the OTHER tracklets fit because this one is in
                  the arc: the RETAINED observations under the orbit that
                  includes it, over the same observations under the orbit that
                  does not.  Same observations, two orbits.  Deliberately NOT a
                  whole-arc ratio -- that put the suspect's own residuals in the
                  numerator, so a noisy observatory scored as a foreign one, and
                  it compared two different observation sets.  See
                  :func:`_drop_one`.
  ``z``           the tracklet's per-axis, standard-error-normalised coherent
                  displacement -- the per-tracklet form of
                  :func:`itf.night_coherence.night_coherence`'s statistic, so the
                  offender can be NAMED (nightsys reports only the worst night,
                  and reads obs80 lines an ADES fetch does not carry).  Reported
                  RAW: nightsys's expected-worst-of-n divisor de-biases a
                  MAXIMUM and has no business on a per-tracklet number, where it
                  makes the threshold drift with tracklet count.  See
                  :data:`Z_FLAG`.
  ``ramp``        residuals walking across the tracklet's own 30-60 minutes --
                  :mod:`itf.residual_trend`, with the reference taken from the REST of
                  this object's arc.  The only astrometric lever that survives
                  at zero dof, because it lives inside a tracklet rather than
                  between them.  Fired on 1 of the 5 (2014 KH169, 2.0 sigma in
                  Dec = 6.7x the 0.3 the rest of the arc manages).
  ``dropobs``     the fit REJECTED observations, and every rejection in the arc
                  came out of this one tracklet.  The only arm that survives when
                  the fit resolves a bad tracklet by ejecting its worst point --
                  which silences ``z``, ``ramp`` and ``drop_ratio`` at the same
                  time, since all three read the KEPT observations.  Found on
                  2025 TY59.  See :func:`_dropobs`.
  ``iso``         gap to the nearest other tracklet over the span of the rest --
                  the TRIGGER SHAPE, never evidence.  All five examples are >= 6;
                  so are plenty of innocent recoveries.

WHAT IS DELIBERATELY NOT HERE.  Leave-one-tracklet-out PREDICTION -- fit the
rest, propagate to the suspect tracklet's station and epoch, compare (the
:mod:`itf.pickup` remainder test).  It is the natural test and it is inert in
this regime: dropping the isolated tracklet leaves a one-to-two day arc whose
prediction a month later is DEGREES wide, so 2015 FH479's 703 tracklet sits
252418" off a sky sigma of 880178" -- 0.29 sigma, a pass.  Measured on all five;
every one of them passes.  A gate that cannot fail is worse than no gate.

CALIBRATION (2026-08-23), and its limits.  Positives: the five worked examples
(2021 CM40, 2014 KH169, 2014 WP143, 2015 FH479, 2022 HK25).  Control: 240 objects
sampled at >= 5 oppositions -- where the linkage is trustworthy (a 1-3
opposition catalogue arc is itself sometimes a mislinkage) -- and truncated to their discovery apparition with
``--first-opposition``, of which 49 had the 3+ tracklet shape.  Object-level
maxima over that control:

    statistic     median   p90    max      the five positives
    dH             0.46   1.08   2.05      3.67 1.61 1.60 1.08 0.30
    drop_ratio     1.24   1.74   4.50      3.51 2.25 1.72 1.47 1.30
    z              0.31   0.86   1.52      1.58 0.08 0.02 0.01 0.00

Requiring the flagged tracklet to be the ISOLATED one gives **5 of 5 positives at
0 of 48 control suspects**, which is the operating point shipped.

Three things must be said plainly about that number.  First, only 12 of the 48
control objects have an isolated tracklet at all, so a zero here is a zero out of
twelve real opportunities, not out of forty-eight.
Second, the whole detection side rests on FIVE positives, all chosen by hand
because they looked like mislinkages -- there is no blind positive sample, and no
planted-truth experiment has been run yet.  Third, ``drop_ratio`` overlaps the control badly (control max
4.50 against a best positive of 3.51): it earns its place only because it is the
arm that catches 2021 CM40, which is photometrically silent, and it should not be
trusted alone on a single object.

The ``ramp`` arm is measured and reported but has never fired on a positive
(worst positive 1.29 sigma against a control max of 2.27).  It is kept because it
is the only astrometric statistic that survives at zero degrees of freedom, and
because a null that is measured and stated is worth more than one that is
quietly absent -- but nothing here should be read as evidence that it works.

Treat a flag as a candidate for a human, which is what it is: nothing in this
module submits, alters an orbit, or writes to the catalogue.

Run it::

    python -m pyroorb.mislink.scan_mislinkage "2014 WP143" "2022 HK25"
    python -m pyroorb.mislink.scan_mislinkage --list objects.txt --out scan.jsonl
    python -m pyroorb.mislink.scan_mislinkage --list objects.txt --workers 6

Re-measure the control after any threshold moves::

    python -c "from pyroorb import mpcorb; \
        print('\\n'.join(mpcorb.sample_asteroids(240, seed=4242, nopp=(5,None), H=(15.5,20.0))))" > control.txt
    python -m pyroorb.mislink.scan_mislinkage --list control.txt \
        --allow-multiopp --first-opposition --workers 5 --out control.jsonl

Multi-opposition objects are SKIPPED (``--allow-multiopp`` to override): a 4+
opposition linkage is essentially never a mislinkage, and the two- and
three-opposition cases want a different test than this one -- there the evidence
is a whole apparition sitting coherently off, which ``ident_check`` already has
in :func:`systematic_misfit` and :func:`coherent_offset_misfit`.
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


import pyroorb.fit_object as fo                                   # noqa: E402
from pyroorb import ident_check                                        # noqa: E402
from pyroorb import obs_sources                                        # noqa: E402
from pyroorb import orbital_elements                                   # noqa: E402
from pyroorb.itf import residual_trend as itf_trend                          # noqa: E402

# ---------------------------------------------------------------------------
# thresholds -- see CALIBRATION in the module docstring.  Every one is measured
# against the 49-object known-good control, and against OBJECT-LEVEL maxima
# rather than per-tracklet ones: an object carries ~5.5 tracklets, so a
# threshold set at a 5% per-tracklet false-positive rate flags a quarter of all
# objects.  The first version of this file made exactly that mistake and put 40%
# of a known-good control on the suspect list.
# ---------------------------------------------------------------------------

#: Isolation (gap to the nearest other tracklet over the span of the rest) at
#: which a tracklet is "the isolated one" -- the shape this scan is FOR.
#:
#: This is the single most load-bearing constant here, and it is not evidence:
#: it is scope.  The same evidence on a tracklet sitting inside a dense cluster
#: is reported, at a lower tier, because it is a different claim -- a foreign
#: tracklet three days from its neighbours would have been caught by the
#: discovery linkage itself, whereas one attributed across a 30-day gap was
#: attributed on an orbit nobody could test.  The control bears the distinction
#: out: an innocent tracklet's implied-H offset reaches 2.05 mag somewhere in
#: the arc but only 0.73 mag among the isolated ones.
#:
#: 2.75 rather than a round 3.0 at the owner's direction (2026-08-24).  Nothing
#: in the control distribution picks out either value -- the isolated sample is
#: 9-12 tracklets and cannot resolve that difference -- so this is a judgement
#: about what shape is worth looking at, and it is recorded as one.  It admits
#: 2010 JK165 (iso 2.97) and 2021 CE57 (2.95), both of which carry evidence.
ISO_TRIGGER = 2.75

#: An arc so thin that no tracklet needs to be isolated for the same problem to
#: apply: nights divided by arc length, in nights per day.  0.10 is one night per
#: ten days -- three nights over a month, the shape the whole scan is about.
#:
#: The SECOND way into the top tiers, OR-ed with :data:`ISO_TRIGGER`, because 696
#: of 1288 survey objects have no isolated tracklet at all and 50 of those carry
#: evidence that could never be acted on.  An arc of three tracklets spread
#: EVENLY over 40 days gives every tracklet a low ``iso`` (each sits ~13 d from
#: its neighbour while the rest span ~27 d) and is exactly as untestable as one
#: with the gap at the end.
#:
#: Like ``iso`` this is scope and NOT evidence -- and here that is measured
#: rather than asserted.  The five worked positives span 0.090-0.182 nights/day;
#: the known-good control's interquartile range is 0.086-0.188.  They are the
#: same distribution.  Sparsity describes survey CADENCE, not mislinkage, so
#: widening the trigger buys recall at a proportional cost in precision, and the
#: cost is real: it takes the control from 0 suspects to 1 (2%).
#:
#: MEASURED COST at 0.10, with DROPOBS requiring a second arm (see :func:`_rank`):
#: 2 of 50 control objects become suspects, from 0 -- so this trigger, not any of
#: the evidence, is what the tool's false-positive rate now rests on.  Letting
#: DROPOBS convict alone doubles that to 4 of 50.  A cliff to 0 exists at 0.07,
#: but only ~4 control objects reach that threshold at all, so it is a zero out
#: of four opportunities and not a strong one.
SPARSE_TRIGGER = 0.10

#: Implied-H offset (magnitudes, one tracklet against the pooled rest) that
#: counts as evidence, and as decisive evidence.  Control object-level maxima:
#: median 0.45, p90 1.03, max 2.14 over 49 objects.
MAG_DH_FLAG = 1.0
MAG_DH_STRONG = 1.5

#: ...and the significance the offset must also reach.  A raw magnitude gap on
#: two junk survey magnitudes is not a finding; the p comes from
#: :func:`ident_check._implied_h_pvalue` with its systematic floor.
MAG_P_FLAG = 0.05

#: How much worse the RETAINED tracklets fit because this one is in the arc --
#: the same observations under two orbits (see :func:`_drop_one`).  NOT a ratio
#: of whole-arc rms: that put the suspect tracklet's own residuals in the
#: numerator, so a noisy observatory scored as a foreign one, and it compared
#: two different observation sets.
#:
#: The corrected statistic lives on a far tighter scale and is far more
#: discriminating.  Control, 232 innocent tracklets: median 1.02, p95 1.38,
#: p99 1.82, max 2.83; of the 9 innocent ISOLATED tracklets, max 1.65.  At 1.8
#: no innocent isolated tracklet in the control reaches the flag at all, while
#: 2021 CM40 -- the one worked example with no photometric signal whatever --
#: sits at 1.96, above every innocent isolated tracklet and above the p99 of
#: all of them.
#:
#: What the correction cost, and it is worth stating: under the old whole-arc
#: form, four of the five worked examples showed "astrometric corroboration"
#: (2014 KH169 read DROP! 3.51x) that was an artifact of their own residuals.
#: On the corrected statistic they read 1.17-1.45 -- ordinary.  They are
#: photometric detections and always were; the old number was manufacturing
#: agreement between two arms that had not in fact agreed.
DROP_FLAG = 1.8
DROP_STRONG = 2.4

#: Coherent-displacement thresholds, on the PER-TRACKLET scale where
#: ``z**2 = n*(mean(dRA/sigma)**2 + mean(dDec/sigma)**2)/2`` and a tracklet
#: sitting exactly where its uncertainties say scores E[z**2] = 1.
#:
#: NO expected-worst-of-n divisor.  An earlier version divided every tracklet's z
#: by :func:`itf.night_coherence._expected_max` -- but that correction exists to
#: de-bias a MAXIMUM taken over n nights, and this scan reports every tracklet,
#: not the worst.  z is already shape-free (z**2 ~ Exp(1) for ANY observation
#: count), so the divisor did the very thing it exists to prevent: it made the
#: effective threshold drift with tracklet count.  Z_FLAG 1.5 on the deflated
#: scale demanded z >= 2.03 on a 3-tracklet arc and 2.47 on an 8-tracklet one --
#: a different test on every arc shape, which is nightsys's own documented
#: failure mode ("length bias, harshest on exactly the long arcs most likely to
#: be genuine") reintroduced.
#:
#: Control on the corrected scale, 259 innocent tracklets: median 0.32, p90 1.01,
#: p99 1.62, max 2.45.  At 2.0 exactly ONE control tracklet flags -- the same
#: count the deflated scale gave -- and 2021 CM40's 309 tracklet, the only z
#: detection in the five worked examples, reads 2.28 and is preserved.  Note WHICH tracklet: not the isolated suspect but the dense cluster
#: that absorbed the strain of reaching it, which is nightsys's own worked
#: example ("the fit had been dragged onto the new tracklet and the seed
#: absorbed all of the error").  So a fired z is evidence that the ARC is
#: strained, and the attribution comes from the other statistics.
Z_FLAG = 2.0
Z_STRONG = 2.5

#: How many observations one tracklet must lose -- and it must hold ALL of the
#: arc's losses -- before :func:`_dropobs` calls it evidence.  One is enough
#: BECAUSE the concentration test carries the weight: a single rejection that is
#: also the arc's only rejection is a different statement from one of five
#: scattered across it.
DROPOBS_MIN = 1

#: ...and its tracklet must WALK by at least this much, measured with the
#: rejected observations read back in.  Shares :data:`itf.residual_trend.TREND_REL_FLOOR_SIGMA`'s
#: value on purpose: below it there is no trend to have been disguised, and a
#: concentrated rejection is just a bad observation.
DROPOBS_TREND_MIN = 2.0

#: A tracklet whose misfit is NOT a coherent displacement -- the scatter part of
#: its normalised residual, ``walk = sqrt(chi**2 - z**2/n)``.  See :func:`_chi`.
#:
#: This is the rate-error statistic, and it is the one arm that works on a
#: TWO-OBSERVATION tracklet, where :mod:`itf.residual_trend` is structurally blind
#: (TREND_MIN_OBS is 3, and two points define a line exactly).  Those are not rare:
#: three of 2025 MP286's four tracklets have n=2.
#:
#: The decomposition is physically meaningful, not just convenient.  A tracklet
#: can be wrong by being DISPLACED (all of chi in z) or by MOVING AT THE WRONG
#: RATE (all of chi in walk), and the two say different things.  In both worked
#: examples the ISOLATED tracklet is a rate error and the cluster is a pure
#: displacement -- the signature of a foreign tracklet against an orbit dragged
#: off to reach one:
#:
#:     2025 MP286  06-21  chi 1.53  z 0.44  ->  walk 1.50   (isolated)
#:                 07-13  chi 1.54  z 2.14  ->  walk 0.28   (cluster)
#:     2021 CM40   01-16  chi 1.15  z 0.00  ->  walk 1.15   (isolated)
#:                 02-04  chi 0.72  z 2.28  ->  walk ~0     (cluster)
#:
#: Control: **0 of 259 innocent tracklets reach 1.0** (median 0.31, p99 0.90, max
#: 0.98); 2-observation tracklets max at 0.48, so short does not mean noisy.  Note
#: the margin above that maximum is THIN -- 1.0 against 0.98 -- and 2021 CM40
#: clears it by only 17%.  One more control draw could move this.
#:
#: Why the control's walk is ~0.3 rather than the ~1.0 a perfect orbit predicts:
#: these arcs OVER-fit (control rms_normalized median 0.47), so walk >= 1.0 means
#: "this tracklet's non-coherent misfit is at its full stated uncertainty while
#: the arc as a whole fits at half of one".
#: NOT A FLAG -- ``walk`` raises no flag of its own and cannot convict a tracklet
#: alone.  This is the SCALE UNIT in :func:`_anomaly` ("one unit of walk"), and
#: that distinction is the whole finding: as a flag at 1.0 it fired on **4.2% of
#: survey tracklets against 0.4% of control tracklets** and took the survey from
#: 76 suspects to 125, most at walk 1.00-1.19.
#:
#: The cause is a CONTROL-POPULATION MISMATCH, and it is the most important
#: caveat on this constant.  The control is multi-opposition objects truncated to
#: a first apparition -- older, denser, far fewer two-observation tracklets -- and
#: its walk distribution (median 0.31, p99 0.90) sits well below the survey's
#: (median 0.45, p99 1.24).  A threshold set at the control maximum is therefore
#: not a threshold at all in the field.  Two draws of the SAME control also gave
#: maxima of 0.98 and 1.28, so the margin is inside the sampling noise.
#:
#: Contributing to the additive score instead costs nothing measurable (control
#: suspects 2 -> 2) and adds 22 survey suspects, every one of which clears the
#: 2.0 additive bar on more than one line -- the standard the rest of the tool
#: already uses.  A properly matched control (single-opposition shapes, the same
#: station mix and tracklet lengths) is what a flag would need.
WALK_UNIT = 1.0

#: An arc whose residuals reach its observations' own stated uncertainties.  Not
#: a tracklet statistic and not attributable to one -- an OBJECT-level note that
#: the orbit fits nothing well.  0 of the 48 known-good control objects reach it
#: (their max is 0.91) and only 15 of 1288 survey objects do, so it is a
#: rare-event marker.  Short arcs normally fit themselves to a FRACTION of their
#: noise (control median 0.48), which is what makes reaching 1.0 notable.
STRAINED_RMS_N = 1.0

#: A tracklet needs this many INCLUDED observations before its mean residual is
#: a displacement rather than one measurement.
MIN_TRK_OBS = 2

#: Below this many tracklets there is nothing to compare a tracklet against --
#: one-vs-rest needs a rest.
MIN_TRACKLETS = 3

#: Refuse to refit a leave-one-out arc smaller than this.  Six observations is
#: three positions, the bare minimum an orbit can be fitted from at all, and a
#: ratio taken against a fit that barely converged is noise.
MIN_DROP_OBS = 6

#: Reported, never branched on: between-tracklet degrees of freedom, 2*n - 6.
#: At <= 0 a clean astrometric result means UNTESTED, not good -- the single
#: most important caveat in this whole module.
def evidence_dof(n_tracklets):
    """Between-tracklet degrees of freedom of an ``n``-tracklet arc."""
    return 2 * int(n_tracklets) - 6


def _date(jd):
    return (datetime(1970, 1, 1) + timedelta(days=jd - 2440587.5)).strftime("%Y-%m-%d")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _tracklet_groups(obs, mask=None):
    """Index groups over ``obs`` = tracklets, via :func:`ident_check._tracklets`
    (one night, one station, one submitted designation).  The engine's own
    definition is imported rather than re-rolled: grouping by night gap alone
    chains two observatories into one 'tracklet'."""
    keep = [i for i in range(len(obs)) if mask is None or mask[i]]
    times = [obs[i]["jd_utc"] for i in keep]
    codes = [obs[i].get("obscode") for i in keep]
    desigs = [obs[i].get("designation") for i in keep]
    return [[keep[k] for k in g] for g in ident_check._tracklets(times, codes, desigs)]


def _isolation(groups, obs, i):
    """``(gap, iso)`` for group ``i``: days to the nearest OTHER tracklet, and
    that over the span of all the others.  The trigger shape the owner
    described -- "3 tracklets in 3 days and then a 33 day gap" -- as one number.
    Never evidence: an isolated tracklet is where to LOOK, not what was found."""
    t0, t1 = obs[groups[i][0]]["jd_utc"], obs[groups[i][-1]]["jd_utc"]
    others = [(obs[g[0]]["jd_utc"], obs[g[-1]]["jd_utc"])
              for j, g in enumerate(groups) if j != i]
    if not others:
        return 0.0, 0.0
    gap = min(min(abs(t0 - b), abs(a - t1)) for a, b in others)
    span = max(b for _, b in others) - min(a for a, _ in others)
    return gap, gap / max(span, 1.0)


def _z_displacement(fit, group):
    """Per-axis, standard-error-normalised coherent displacement of one tracklet.

    ``z**2 = n * (mean(dRA/sigma_RA)**2 + mean(dDec/sigma_Dec)**2) / 2`` -- the
    mean over the TWO AXES of the squared SE-normalised displacement, so a
    tracklet sitting exactly where its uncertainties say scores 1.0 whatever its
    observation count.  Byte-for-byte the per-night quantity in
    :func:`itf.night_coherence.night_structure`; recomputed here only because that
    function groups by obs80 line (which an ADES fetch has none of) and reports
    the worst night rather than every one, and this scan has to NAME the
    offender.  Returns ``(z, mean_ra, mean_dec)`` or ``None``.
    """
    try:
        rows = fit.residuals()
    except Exception:                                      # noqa: BLE001
        return None
    zr = zd = mr = md = 0.0
    n = 0
    for i in group:
        if i >= len(rows):
            return None
        r = rows[i]
        o = fit._obs[i]
        s_ra = float(r.get("sigma_ra") or o.get("sigma_ra") or 1.0) or 1.0
        s_dec = float(r.get("sigma_dec") or o.get("sigma_dec") or s_ra) or s_ra
        d_ra, d_dec = float(r["d_ra_arcsec"]), float(r["d_dec_arcsec"])
        zr += d_ra / s_ra
        zd += d_dec / s_dec
        mr += d_ra
        md += d_dec
        n += 1
    if n < MIN_TRK_OBS:
        return None
    zr, zd, mr, md = zr / n, zd / n, mr / n, md / n
    return math.sqrt(n * (zr * zr + zd * zd) / 2.0), mr, md


def _chi(fit, group):
    """Per-axis sigma-normalised rms of one tracklet's own residuals, or None.

    The tracklet's TOTAL misfit, where :func:`_z_displacement` measures only its
    COHERENT part: ``chi**2 = z**2/n + within-tracklet scatter``.  Reported (not
    yet wired to any verdict) because it is the only residual-shape statistic
    that survives a TWO-OBSERVATION tracklet, and those are not rare -- three of
    2025 MP286's four X05 tracklets have n=2, so :mod:`itf.residual_trend` (which needs
    TREND_MIN_OBS=3) is structurally blind on the whole object.

    With n=2 nothing can separate a displaced tracklet from a walking one -- two
    points define a line exactly -- but nothing needs to: any difference between
    the two points IS a slope, so the scatter term captures the walk exactly.
    """
    try:
        rows = fit.residuals()
    except Exception:                                      # noqa: BLE001
        return None
    tot, n = 0.0, 0
    for i in group:
        if i >= len(rows):
            return None
        r, o = rows[i], fit._obs[i]
        s_ra = float(r.get("sigma_ra") or o.get("sigma_ra") or 1.0) or 1.0
        s_dec = float(r.get("sigma_dec") or o.get("sigma_dec") or s_ra) or s_ra
        tot += (float(r["d_ra_arcsec"]) / s_ra) ** 2 + (float(r["d_dec_arcsec"]) / s_dec) ** 2
        n += 1
    return math.sqrt(tot / (2.0 * n)) if n else None


def _trends_for_groups(fit, obs, groups):
    """``{group index: trend dict}`` from :func:`itf.residual_trend.tracklet_trends`.

    Matched by STATION plus time containment, never by a date string.  A date
    string is not an identity: ``tracklet_trends`` splits on ITF's own rule (one
    station, gaps <= 120 min) which is finer than a night, so one station can
    produce two entries carrying the same calendar date -- and keying on
    ``(date, station)`` then hands one tracklet the OTHER one's trend, silently.
    It affects 10 of 1283 survey objects (2014 EO217's two 2014-03-27 W84
    tracklets both received trend 0.20; 2025 NO35's two 2025-07-13 X05 tracklets
    both received 0.10).  This is the codebase's standing index-alignment bug
    class: reconstructing an identity instead of carrying one.

    Where the finer ITF rule split one of our tracklets in two, the
    harder-trending half is kept -- the question is whether ANY part of this
    tracklet walks."""
    spans = [(obs[g[0]]["jd_utc"], obs[g[-1]]["jd_utc"],
              obs[g[0]].get("obscode")) for g in groups]
    out = {}
    # include_rejected: a rejection is how a fit HIDES a trend -- it ejects the
    # observation that made the tracklet walk and then fits the remainder
    # cleanly, so the default (kept obs only) goes quiet exactly where the walk
    # matters.  2023 QA131's isolated tracklet loses one of three observations
    # and the surviving pair has nothing left to say.
    for d in itf_trend.tracklet_trends(fit, include_rejected=True):
        jd0 = d.get("jd0")
        i = _containing_group(spans, jd0, d["station"]) if jd0 is not None else None
        if i is None:
            continue
        if i not in out or abs(d["swing_loo"]) > abs(out[i]["swing_loo"]):
            out[i] = d
    return out


def _containing_group(spans, jd, code):
    """Index of the tracklet whose station matches and whose time window holds
    ``jd``, or None.  Tracklets of one station are disjoint in time, so this is
    unambiguous where a date string is not."""
    for i, (t0, t1, c) in enumerate(spans):
        if c == code and t0 - 1e-6 <= jd <= t1 + 1e-6:
            return i
    return None


def audit(desig, oc=None, *, allow_multiopp=False, timeout=None,
          first_opposition=False):
    """Audit one designation.  Never raises; returns a record dict.

    ``status``: ``ok`` | ``skip-multiopp`` | ``skip-shape`` | ``error`` |
    ``timeout``.  On ``ok`` the record carries ``trk`` (one entry per tracklet,
    time-ordered, each with its own evidence and ``flags``), ``verdict`` and
    ``worst``.

    ``first_opposition`` truncates the arc to its first opposition before
    fitting.  That is the CALIBRATION mode, not a scanning mode: paired with
    ``allow_multiopp`` it turns an object whose linkage is trustworthy (>= 4
    oppositions) into a known-good specimen of exactly
    the shape this scan runs on, which is the only way to find out what an
    innocent tracklet scores.  Using it on a real scan would be throwing away
    the evidence.
    """
    oc = oc or obs_sources.load_obscodes()
    rec = {"d": desig}
    if timeout:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
    t0 = time.time()
    try:
        obs, meta = obs_sources.fetch_observations([desig], oc)
        obs.sort(key=lambda o: o["jd_utc"])
        rec["n_opp_all"] = (orbital_elements.n_oppositions([o["jd_utc"] for o in obs])
                            if obs else 0)
        if first_opposition and obs:
            times = [o["jd_utc"] for o in obs]
            first = ident_check._split(times, ident_check.OPP_GAP_DAYS)[0]
            obs = [obs[i] for i in first]
            rec["truncated"] = True
        rec["n_obs"] = len(obs)
        rec["n_opp"] = orbital_elements.n_oppositions([o["jd_utc"] for o in obs]) if obs else 0
        rec["arc_days"] = round(obs[-1]["jd_utc"] - obs[0]["jd_utc"], 2) if obs else 0.0
        rec["desigs"] = sorted({o.get("designation") for o in obs if o.get("designation")})
        if rec["n_opp"] > 1 and not allow_multiopp:
            rec["status"] = "skip-multiopp"
            return rec
        # Only enough observations to FIT is required here.  Gating the whole
        # object on what the drop arm needs (a leave-one-out arc of MIN_DROP_OBS
        # on top of the tracklet being dropped) skipped objects the PHOTOMETRIC
        # arm could have judged perfectly well -- it never asks the orbit
        # anything.  2025 PG242 and 2025 QG193 are three tracklets in 7
        # observations each: auditable, and skipped unmeasured.  The drop arm
        # declines per tracklet in _drop_one, which is where that judgement
        # belongs.
        if len(obs) < MIN_DROP_OBS:
            rec["status"] = "skip-shape"
            rec["why"] = f"{len(obs)} observations -- too few to fit an orbit from"
            return rec
        fit = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta)
        rec.update(_evidence(fit, oc, desig, meta))
        rec["secs"] = round(time.time() - t0, 1)
    except _Timeout:
        rec["status"] = "timeout"
    except Exception as e:                                 # noqa: BLE001
        rec["status"] = "error"
        rec["err"] = f"{type(e).__name__}: {e}"
    finally:
        if timeout:
            signal.alarm(0)
    return rec


def _evidence(fit, oc, desig, meta):
    """The whole per-tracklet evidence table for a converged ``fit``."""
    mask = np.asarray(fit._mask, dtype=bool)
    obs = fit._obs
    # Group ALL observations, not just the ones the fit kept.  Grouping the
    # INCLUDED obs (which this did) makes a REJECTED observation vanish from the
    # record entirely -- and a rejection is not an absence of evidence, it is the
    # arc paying to include a tracklet.  See :func:`_dropobs`.
    groups = _tracklet_groups(obs, None)
    out = {"rms": round(fit.rms_residual(), 3), "rms_n": round(fit.rms_normalized(), 3),
           "n_kept": int(mask.sum()), "n_trk": len(groups),
           "evidence_dof": evidence_dof(len(groups)),
           "strained": bool(fit.rms_normalized() >= STRAINED_RMS_N)}
    # the SAME night rule the tracklet grouping uses, not the UTC calendar date.
    # Two definitions in one function disagree (2014 EO217: 3 calendar nights,
    # 4 gap-rule nights) and the bias has a direction -- fewer nights means lower
    # sparsity means MORE likely to clear SPARSE_TRIGGER, which is the trigger
    # already carrying the whole false-positive budget.
    nights = len(ident_check._split([o["jd_utc"] for o in obs],
                                    ident_check.NIGHT_GAP_DAYS))
    out["nights"] = nights
    out["sparsity"] = round(nights / max(obs[-1]["jd_utc"] - obs[0]["jd_utc"], 1.0), 3)
    if len(groups) < MIN_TRACKLETS:
        out["status"] = "skip-shape"
        out["why"] = f"{len(groups)} tracklets -- one-vs-rest needs at least {MIN_TRACKLETS}"
        return out
    out["status"] = "ok"

    # --- photometric arm ---------------------------------------------------
    audit_p = ident_check.tracklet_mag_audit(fit)
    # tracklet_mag_audit groups the INCLUDED obs the same way (it calls the same
    # _tracklets) but drops tracklets with too few usable magnitudes, so its
    # entries cannot be matched back by POSITION.  Match by station plus time
    # containment -- not by a (date, station) string, which is not an identity
    # and hands one tracklet another's photometry where a station has two
    # tracklets sharing a calendar date.  See :func:`_trends_for_groups`.
    spans = [(obs[g[0]]["jd_utc"], obs[g[-1]]["jd_utc"],
              obs[g[0]].get("obscode")) for g in groups]
    phot = {}
    for e in (audit_p or {}).get("trk", []):
        i = _containing_group(spans, e["jd0"], e["code"])
        if i is not None:
            phot[i] = e

    trends = _trends_for_groups(fit, obs, groups)

    entries = []
    for i, g in enumerate(groups):
        gap, iso = _isolation(groups, obs, i)
        key = (_date(obs[g[0]]["jd_utc"]), obs[g[0]].get("obscode"))
        kept = [k for k in g if mask[k]]
        e = {"date": key[0], "code": key[1], "n": len(kept), "n_all": len(g),
             "n_drop": len(g) - len(kept),
             "gap": round(gap, 2), "iso": round(iso, 2)}

        p = phot.get(i)
        if p and p.get("dH") is not None:
            e.update(H=round(p["H"], 2), dH=round(p["dH"], 2),
                     p=round(p["p"], 5), tier=p["tier"])

        c = _chi(fit, kept)
        if c is not None:
            e["chi"] = round(c, 2)
        z = _z_displacement(fit, kept)
        if c is not None:
            zz = (z[0] if z else 0.0)
            e["walk"] = round(math.sqrt(max(c * c - zz * zz / max(len(kept), 1), 0.0)), 2)
        if z:
            # raw, undivided: see the Z_FLAG note.  z_raw kept as an alias so a
            # record written before the fix and one written after can be compared
            e.update(z=round(z[0], 2), z_raw=round(z[0], 2),
                     mean_ra=round(z[1], 2), mean_dec=round(z[2], 2))

        r = trends.get(i)
        if r:
            e.update(ramp=round(abs(r["swing_loo"]), 2), ramp_axis=r["axis"],
                     ramp_mono=round(r["mono_loo"], 2))
        entries.append(e)

    # trend is judged against the REST of this object's arc -- the same relative
    # form itf.residual_trend uses against a seed, applied intramurally because here there
    # is no seed and no linked set, only tracklets of one designation.
    for i, e in enumerate(entries):
        others = [o.get("ramp", 0.0) for j, o in enumerate(entries) if j != i]
        ref = max(others) if others else 0.0
        if e.get("ramp") is not None and ref > 0:
            e["ramp_ratio"] = round(e["ramp"] / ref, 2)

    _drop_one(fit, obs, groups, entries, oc, desig, meta, out["rms_n"])
    _dropobs(entries)
    out["drop_unstable"] = _drop_unstable(entries)
    for e in entries:
        e["flags"] = _flags(e, drop_ok=not out["drop_unstable"])
        e["anomaly"], e["anomaly_parts"] = _anomaly(e, drop_ok=not out["drop_unstable"])
    out["trk"] = entries
    out["verdict"], out["worst"] = _verdict(
        entries, sparse=out["sparsity"] <= SPARSE_TRIGGER)
    return out


def _obs_key(o):
    """Identity of one observation, for matching the SAME observation across two
    different fits of overlapping arcs.

    Position is not identity here -- the two fits hold different observation
    lists and each applies its own outlier mask, so index k of one is not index
    k of the other.  (The codebase has a standing bug class for exactly this;
    see the index-alignment guards in the ITF search.)  Station + epoch + sky
    position is what a re-fit cannot change.
    """
    return (round(float(o["jd_utc"]), 8), o.get("obscode"),
            round(float(o["ra_deg"]), 7), round(float(o["dec_deg"]), 7))


def _subset_rms_norm(fit, wanted):
    """``(rms, n)`` -- per-axis, sigma-normalised rms over the INCLUDED
    observations of ``fit`` whose :func:`_obs_key` is in ``wanted``.

    Per-axis (divide the summed squares by 2n) because that is the codebase's
    reported currency (``FittedOrbit.rms_normalized``).  ``(None, 0)``
    when nothing matched.
    """
    try:
        rows = fit.residuals()
    except Exception:                                      # noqa: BLE001
        return None, 0
    tot, n = 0.0, 0
    for o, r, inc in zip(fit._obs, rows, fit.is_included()):
        if not inc or _obs_key(o) not in wanted:
            continue
        s_ra = float(r.get("sigma_ra") or o.get("sigma_ra") or 1.0) or 1.0
        s_dec = float(r.get("sigma_dec") or o.get("sigma_dec") or s_ra) or s_ra
        tot += (float(r["d_ra_arcsec"]) / s_ra) ** 2 + (float(r["d_dec_arcsec"]) / s_dec) ** 2
        n += 1
    return (math.sqrt(tot / (2.0 * n)) if n else None), n


def _drop_one(fit, obs, groups, entries, oc, desig, meta, joint_rms_n):
    """Refit without each tracklet and record ``drop_ratio`` -- **how much worse
    the OTHER tracklets fit because this one is in the arc.**

    THE ONE THING THIS MUST NOT BE is a ratio of whole-arc rms.  That was the
    first version and it is wrong twice over:

      * the suspect tracklet's OWN residuals sit in the numerator, so a tracklet
        from a noisier observatory inflates the ratio without having displaced
        anything.  Being noisy is not evidence of being foreign -- and the
        sigma-normalisation does not save it, because a station's real scatter
        and its REPORTED sigma are different numbers;
      * the two sides are measured over different observation SETS (the whole
        arc against the remainder), so it is not a before/after comparison at
        all.

    The question that actually discriminates is the one the ITF seed-degradation
    veto asks (``itf.fitter.subset_rms``, and see that module's docstring on
    measuring both sides of a comparison in the same currency): take the
    RETAINED observations, and compare how they fit under the orbit that
    includes the suspect tracklet against how the same observations fit under
    the orbit that does not.  Same observations, two orbits.

    Both sides must cover the SAME observations or the ratio is meaningless, so
    the counts are compared and a mismatch (either fit rejected an observation
    the other kept) declines to measure rather than reporting a number built
    from two different sets.

    ``drop_rms``/``drop_rms_n`` are kept as whole-arc diagnostics; nothing
    branches on them.  A refit that fails or has too few observations leaves
    ``drop_ratio`` absent, which reads as "not measured" downstream -- it is
    never silently filled with 1.0.
    """
    for i, g in enumerate(groups):
        drop = set(g)
        keep = [o for k, o in enumerate(obs) if k not in drop]
        if len(keep) < MIN_DROP_OBS:
            entries[i]["drop_why"] = f"only {len(keep)} obs would remain"
            continue
        try:
            f2 = fo.fit_orbit(keep, oc, desig=desig, meta=meta)
            entries[i]["drop_rms_n"] = round(f2.rms_normalized(), 3)
            entries[i]["drop_rms"] = round(f2.rms_residual(), 3)
            wanted = {_obs_key(o) for o in keep}
            before, n_before = _subset_rms_norm(fit, wanted)   # retained obs, WITH it
            after, n_after = _subset_rms_norm(f2, wanted)      # retained obs, WITHOUT it
            if before is None or after is None or n_before != n_after:
                entries[i]["drop_why"] = (f"retained-obs sets differ ({n_before} vs "
                                          f"{n_after}) -- not a like-for-like comparison")
                continue
            entries[i]["drop_n"] = n_before
            entries[i]["drop_before"] = round(before, 3)
            entries[i]["drop_after"] = round(after, 3)
            if after > 1e-9:
                entries[i]["drop_ratio"] = round(before / after, 2)
        except Exception as e:                             # noqa: BLE001
            entries[i]["drop_why"] = f"{type(e).__name__}"


def _dropobs(entries):
    """Record where the fit's REJECTED observations landed, and whether they are
    concentrated in one tracklet.

    Every other statistic here runs on the observations the fit KEPT, and that
    is exactly how this evidence gets lost.  When an orbit cannot reconcile a
    foreign tracklet with the rest, one way the fit resolves it is to eject the
    tracklet's worst observation and then fit the remainder cleanly -- so ``z``,
    ``ramp`` and ``drop_ratio`` all go quiet BECAUSE the rejection happened.  A
    rejection is not an absence of evidence; it is the arc paying to include a
    tracklet, and the receipt is the only thing left.

    Worked example, 2025 TY59: the isolated 2025-10-03 V00 tracklet has its first
    observation rejected at 0.72" against a 0.137" sigma -- about 5 sigma -- and
    the surviving three then fit at 0.04-0.12".  Before this, the scan reported
    that tracklet as ``n=3`` and never mentioned the fourth observation existed.

    The discriminating form is CONCENTRATION, not count.  Blunders happen
    everywhere -- a cosmic ray, a blended star, a timing slip -- and land in any
    tracklet, so "this arc dropped an observation" says little.  "Every
    observation this arc dropped came out of ONE tracklet" is the shape a foreign
    tracklet makes.  So a tracklet flags only when it holds ALL of the arc's
    rejections and at least :data:`DROPOBS_MIN` of them.

    AND THE TRACKLET MUST BE WALKING.  Concentration alone is not enough, because
    an observation can be intrinsically bad -- a cosmic ray, a blend, a timing
    slip -- and would be thrown out of any orbit it was ever fitted to.  What
    separates the two is the SHAPE of the tracklet it came from, and it needs no
    refit to see:

      * an intrinsically bad observation is one point off the straight line its
        own neighbours define.  2016 DA47's G96 tracklet loses its first
        observation at 2.03" (it carries no magnitude and a blank band at all)
        while the tracklet as a whole trends 0.73 sigma -- there is no walk.
      * a linkage-caused rejection is the far END of a coherent walk: the orbit
        could not reach the tracklet, so the tracklet's residuals trend and the
        fit resolves it by ejecting the extreme point.  2023 QA131's isolated
        tracklet trends 5.87 sigma once the rejected observation is read back in.

    So this is not an independent line of evidence at all -- **a rejection is a
    way of DISGUISING a trend**, and the only thing it adds is that the trend must
    be measured with the rejected observations included (which is why
    :func:`_trends_for_groups` passes ``include_rejected``).  A concentrated
    rejection on a tracklet that does not walk is a blunder.

    (An earlier version tried to test this by refitting without each other
    tracklet and asking whether the rejection survived.  That is confounded by
    the same thing the whole-arc drop ratio was: removing a tracklet SHORTENS the
    arc, so the remainder is under-determined and absorbs almost any bad point.
    2016 DA47's rejection is "rescued" by all three leave-one-out fits.)
    """
    total = sum(e["n_drop"] for e in entries)
    for e in entries:
        e["drop_frac"] = round(e["n_drop"] / e["n_all"], 2) if e["n_all"] else 0.0
        concentrated = bool(total and e["n_drop"] >= DROPOBS_MIN and e["n_drop"] == total)
        walks = (e.get("ramp") or 0.0) >= DROPOBS_TREND_MIN
        e["dropobs"] = bool(concentrated and walks)
        if concentrated and not walks:
            e["dropobs_why"] = (f"tracklet does not walk ({e.get('ramp') or 0:.2f} sigma "
                                f"< {DROPOBS_TREND_MIN}) -- one point off a line, a blunder")


def _drop_unstable(entries):
    """Do TWO OR MORE tracklets fit better out of the arc than in it?

    Then ``drop_ratio`` has stopped being an attribution.  "One tracklet is
    foreign" predicts exactly one tracklet improving the arc by leaving it, and
    that is what the one astrometric worked example shows (2021 CM40: 1.96 on the
    isolated tracklet against 1.07, 1.40, 1.34 on the rest).  Two or more is the OTHER
    thing a leave-one-out can mean, and on these arc shapes it is the likelier
    one: dropping any tracklet from a three-tracklet arc leaves TWO, and two
    tracklets are four coordinates against six parameters -- threadable exactly,
    whichever two they are.  The improvement is then the dof collapse, not a
    verdict about the tracklet that left.

    Found on the 2014 E* group: 2014 ET220 and 2014 EU140 are three W84 nights
    inside a TWO-day arc with no gap anywhere, every tracklet of which scored as
    an offender -- five separate objects from one survey run, all reading as
    mislinkages.  The guard predates the retained-obs correction to
    :func:`_drop_one` and fires on 1 of 49 control objects under it.

    The photometric arm is unaffected (it never asked the orbit anything) and
    keeps running; only the drop arm is withdrawn, and the record says so rather
    than going quiet.
    """
    measured = [e for e in entries if e.get("drop_ratio") is not None]
    return sum(1 for e in measured if e["drop_ratio"] >= DROP_FLAG) >= 2


#: Provisional weights for :func:`_anomaly`, on a scale where **1.0 is one arm at
#: exactly its flag threshold**.  They are a stand-in for coefficients a fitted
#: model would learn and are HAND-SET, not fitted -- there are five
#: hand-picked positives and a 50-object control, which cannot fit a ten-feature
#: model.  Read them as "these lines are commensurate and additive", not as
#: measured relative importance.
#:
#: A contribution saturates at :data:`ANOM_CAP` because past a point more of one
#: line stops being more certainty ABOUT MISLINKAGE: 2018 PZ138's dH of 8.93 is
#: 9x the flag threshold and should not out-vote every other line combined.
ANOM_CAP = 3.0

#: Total anomaly at which accumulated partial evidence reaches the top tier on
#: its own, with no single arm having flagged.  This is the additive route, and
#: it is what the gate structure could not do: 2021 CE57 carries dH 0.79 (flag is
#: 1.0) AND drop_ratio 2.29, two partial signals a gate either ignores or counts
#: once.  Calibrated on the control like every other threshold here.
ANOM_SUSPECT = 2.0


def _anomaly(e, drop_ok=True):
    """Additive anomaly score for one tracklet: the evidence lines SUMMED.

    Every arm is mapped to a common currency -- **1.0 means "at this arm's flag
    threshold"** -- and the contributions are added.  Two things follow that the
    gate structure could not do:

      * partial evidence accumulates.  A tracklet just under two thresholds is
        more suspicious than one just under a single threshold, and the gates
        scored both at zero.
      * the lines are commensurate.  Each contribution starts at 0 where the
        statistic sits at its NULL (drop_ratio 1.0, z 0, dH 0), not at some
        arbitrary offset, so summing them means something.

    Returns ``(score, {arm: contribution})``.  SCOPE terms (``iso``,
    ``sparsity``) are deliberately NOT summed in: they decide whether a tracklet
    is one this scan can judge, not how anomalous it is, and adding them would
    let a wide gap manufacture evidence out of nothing.

    The weights are provisional and hand-set; see :data:`ANOM_CAP`.  The fitted
    model in :mod:`pyroorb.mislink.score` is the calibrated successor.
    """
    parts = {}
    dH, pv = e.get("dH"), e.get("p")
    if dH is not None and pv is not None and pv < MAG_P_FLAG:
        # the significance gate is a PREREQUISITE, not a term: an unresolved
        # magnitude gap is not weak evidence, it is no evidence
        parts["mag"] = min(dH / MAG_DH_FLAG, ANOM_CAP)
    dr = e.get("drop_ratio") if drop_ok else None
    if dr is not None and dr > 1.0:
        parts["drop"] = min((dr - 1.0) / (DROP_FLAG - 1.0), ANOM_CAP)
    z = e.get("z")
    if z:
        parts["z"] = min(z / Z_FLAG, ANOM_CAP)
    wk = e.get("walk")
    if wk:
        parts["walk"] = min(wk / WALK_UNIT, ANOM_CAP)
    rr, rs, mono = e.get("ramp_ratio"), e.get("ramp"), e.get("ramp_mono")
    if rr is not None and rs is not None and (mono or 0) >= itf_trend.TREND_MIN_MONOTONIC_FRAC:
        # BOTH trend clauses must be satisfied to earn anything, so the weaker of
        # the two sets the contribution -- 2010 JK165's 2.90 sigma is 1.45x the
        # absolute floor but only 0.86x the relative one, and 0.86 is what a
        # near-miss on the binding clause is worth
        parts["ramp"] = min(rs / itf_trend.TREND_REL_FLOOR_SIGMA,
                            rr / itf_trend.TREND_RELATIVE, ANOM_CAP)
    if e.get("dropobs"):
        parts["dropobs"] = 1.0
    return round(sum(parts.values()), 2), {k: round(v, 2) for k, v in parts.items()}


def _flags(e, drop_ok=True):
    """The evidence against ONE tracklet: ``[(arm, severity, text), ...]``.

    ``severity`` is 2 (decisive) or 1 (evidence).  ``drop_ok`` False withdraws the
    drop arm for the whole object -- see :func:`_drop_unstable`.  Each text names its statistic,
    its value and the threshold it passed, so a record read six months from now
    says why it fired without needing this file open.
    """
    out = []
    dH, pv = e.get("dH"), e.get("p")
    if dH is not None and pv is not None and pv < MAG_P_FLAG:
        if dH >= MAG_DH_STRONG:
            out.append(("mag", 2, f"MAG! dH={dH:.2f} p={pv:.4f}"))
        elif dH >= MAG_DH_FLAG:
            out.append(("mag", 1, f"MAG dH={dH:.2f} p={pv:.4f}"))
    dr = e.get("drop_ratio") if drop_ok else None
    if dr is not None and dr >= DROP_STRONG:
        out.append(("drop", 2, f"DROP! {dr:.2f}x (arc fits {dr:.2f}x better without it)"))
    elif dr is not None and dr >= DROP_FLAG:
        out.append(("drop", 1, f"DROP {dr:.2f}x"))
    z = e.get("z")
    if z is not None and z >= Z_STRONG:
        out.append(("z", 2, f"OFFSET! z={z:.2f} "
                            f"({e.get('mean_ra'):+.2f},{e.get('mean_dec'):+.2f})\""))
    elif z is not None and z >= Z_FLAG:
        out.append(("z", 1, f"OFFSET z={z:.2f}"))
    if e.get("dropobs"):
        out.append(("dropobs", 1, f"DROPOBS {e['n_drop']}/{e['n_all']} obs rejected "
                                  f"-- every rejection in this arc"))
    rr = e.get("ramp_ratio")
    if (rr is not None and e.get("ramp", 0) >= itf_trend.TREND_REL_FLOOR_SIGMA
            and rr >= itf_trend.TREND_RELATIVE and e.get("ramp_mono", 0) >= itf_trend.TREND_MIN_MONOTONIC_FRAC):
        out.append(("ramp", 1, f"TREND {e['ramp']:.1f}sig in {e.get('ramp_axis')} "
                               f"= {rr:.1f}x the rest"))
    return out


def _rank(e, sparse=False):
    """Severity rank of one tracklet, 0-4.  The top two ranks are gated by SCOPE,
    not by evidence: either this tracklet is isolated (:data:`ISO_TRIGGER`) or the
    whole arc is thin (:data:`SPARSE_TRIGGER`, passed in as ``sparse`` because it
    is a property of the object and not of any one tracklet).

    There are two routes to the top tier: an arm FLAGGING, and accumulated
    partial evidence reaching :data:`ANOM_SUSPECT` with no arm having flagged at
    all (see :func:`_anomaly`).  The second exists because the first cannot add
    up -- 2021 CE57 carries dH 0.79 against a flag of 1.0 and drop_ratio 2.29,
    and a gate structure scores that the same as nothing.
    """
    fl = e["flags"]
    if not fl:
        # the ADDITIVE route: no single arm flagged, but the lines together did
        if e.get("anomaly", 0) >= ANOM_SUSPECT and (e["iso"] >= ISO_TRIGGER or sparse):
            return 3
        return 0
    arms = {a for a, _, _ in fl}
    # A concentrated rejection ALONE convicts only where it is ISOLATED, and that
    # split is measured rather than assumed.  DROPOBS fires on 7 of the 50
    # known-good control objects -- 14%, not a rare event -- so unaccompanied it
    # is usually an ordinary blunder (a cosmic ray, a blend, a timing slip).  But
    # all 7 of those fires sit at iso <= 0.9 and NOT ONE reaches ISO_TRIGGER: in
    # good data a rejection lands inside a dense cluster, where a blunder is the
    # whole explanation.  So the isolated route costs nothing measurable and
    # keeps the arm's power where it was built to work (2025 TY59's V00 tracklet
    # at iso 19.4), while the SPARSITY route -- which puts every tracklet of a
    # thin arc in scope, and is what promoted all 7 -- still demands a second arm.
    if arms == {"dropobs"}:
        return 3 if e["iso"] >= ISO_TRIGGER else (2 if sparse else 1)
    strong = any(sev == 2 for _, sev, _ in fl)
    isolated = e["iso"] >= ISO_TRIGGER or sparse
    if isolated:
        return 4 if (strong or len(arms) >= 2) else 3
    return 2 if (strong or len(arms) >= 2) else 1


_VERDICT = {0: "clean", 1: "advisory", 2: "possible", 3: "suspect", 4: "suspect"}


def _verdict(entries, sparse=False):
    """``(verdict, worst_tracklet_summary)`` for the object.

    ``suspect``   evidence against a tracklet the scan is SCOPED to judge: the
                  isolated one, or any tracklet of an arc thin enough that no
                  tracklet needs to be isolated.  5 of 5 worked examples; 1 of 49 known-good control
                  objects (and that 1 is 1 of the 12 control objects that had the
                  trigger shape at all -- the honest conditional rate).
    ``possible``  the same evidence, but against a tracklet that is not isolated:
                  real photometric or astrometric structure, a weaker claim.
    ``advisory``  one arm, at flag strength, on a non-isolated tracklet.
    ``clean``     nothing reached a threshold -- which at ``evidence_dof <= 0``
                  means the ASTROMETRY was never able to speak, not that it
                  cleared the arc.  The report prints the dof beside it.
    """
    best, best_rank = None, -1
    for e in entries:
        rank = _rank(e, sparse)
        # Isolation breaks TIES only.  A one-vs-rest audit on a 3-tracklet arc
        # split 2-against-1 lights up all three (2014 KH169: both W84 nights and
        # the F51 tracklet each read ~1.5 mag from "the rest"), and which side is
        # the intruder is exactly what dH cannot say.  The gap can.  This never
        # promotes a tracklet past better evidence -- only orders equals.
        # ties: the higher anomaly first (the evidence SUMMED, not counted), then
        # isolation -- which orders equals and never promotes past better evidence
        key = (rank, e.get("anomaly", 0.0), e["iso"])
        if best is None or key > (best_rank, best.get("anomaly", 0.0), best["iso"]):
            best, best_rank = e, rank
    if best_rank <= 0:
        return "clean", None
    return _VERDICT[best_rank], {"date": best["date"], "code": best["code"],
                                 "iso": best["iso"],
                                 "anomaly": best.get("anomaly", 0.0),
                                 "parts": best.get("anomaly_parts", {}),
                                 "flags": [t for _, _, t in best["flags"]]}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def format_record(rec):
    """One object's evidence table as text."""
    d = rec["d"]
    st = rec.get("status")
    if st != "ok":
        why = rec.get("why") or rec.get("err") or ""
        opp = rec.get("n_opp")
        extra = f" ({opp} oppositions, {rec.get('n_obs')} obs)" if st == "skip-multiopp" else (
            f" {why}" if why else "")
        return f"{d:<14} {st}{extra}"
    L = [f"{d:<14} {rec['verdict'].upper():<9} "
         f"{rec['n_trk']} tracklets / {rec['n_kept']} obs / {rec['arc_days']:.0f} d, "
         f"rms {rec['rms']:.3f}\" ({rec['rms_n']:.2f} sigma), "
         f"{rec.get('sparsity', 0):.3f} n/d"
         + ("*" if rec.get("sparsity", 9) <= SPARSE_TRIGGER else "")
         + f", between-tracklet dof {rec['evidence_dof']:+d}"
         + ("  [astrometry untestable]" if rec["evidence_dof"] <= 0 else "")
         + ("  [drop arm withdrawn: every tracklet fits better out than in -- "
            "under-determined arc]" if rec.get("drop_unstable") else "")
         + (f"  [STRAINED: fits at its own stated uncertainties, above every "
            f"known-good control object]" if rec.get("strained") else "")]
    L.append(f"    {'date':<12}{'stn':<5}{'n':>6} {'iso':>6} {'H':>6} {'dH':>6} "
             f"{'p':>8} {'drop':>6} {'z':>6} {'walk':>6} {'ramp':>6} {'anom':>6}")
    for e in rec["trk"]:
        def g(k, f="{:.2f}"):
            v = e.get(k)
            return "  --  " if v is None else f.format(v)
        n_txt = (f"{e['n']}" if not e.get("n_drop")
                 else f"{e['n']}-{e['n_drop']}")     # kept, and what was rejected
        L.append(f"    {e['date']:<12}{str(e['code']):<5}{n_txt:>6} "
                 f"{g('iso'):>6} {g('H'):>6} {g('dH'):>6} {g('p','{:.4f}'):>8} "
                 f"{g('drop_ratio'):>6} {g('z'):>6} {g('walk'):>6} {g('ramp'):>6} {g('anomaly'):>6}"
                 + (("   <-- " + "; ".join(t for _, _, t in e["flags"]))
                    if e["flags"] else ""))
    return "\n".join(L)


def _worker(args):
    desig, allow_multiopp, timeout, first_opposition = args
    return audit(desig, allow_multiopp=allow_multiopp, timeout=timeout,
                 first_opposition=first_opposition)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("designations", nargs="*")
    ap.add_argument("--list", dest="listfile", help="file of designations, one per line "
                                                    "(blank lines and # comments ignored)")
    ap.add_argument("--out", help="JSONL record per object")
    ap.add_argument("--report", help="text report (default: stdout)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300, help="seconds per object")
    ap.add_argument("--allow-multiopp", action="store_true",
                    help="audit multi-opposition objects too (skipped by default)")
    ap.add_argument("--first-opposition", action="store_true",
                    help="truncate each arc to its first opposition before fitting -- the "
                         "CALIBRATION mode, use with --allow-multiopp on trustworthy "
                         "objects to measure what an innocent tracklet scores")
    a = ap.parse_args(argv)

    desigs = list(a.designations)
    if a.listfile:
        with open(a.listfile, encoding="utf-8") as f:
            desigs += [ln.strip() for ln in f
                       if ln.strip() and not ln.lstrip().startswith("#")]
    seen, uniq = set(), []
    for d in desigs:                                       # a duplicate in the list is a
        if d not in seen:                                  # duplicate fetch and a duplicate
            seen.add(d)                                    # row; the owner's lists have them
            uniq.append(d)
    if not uniq:
        ap.error("no designations")
    if len(uniq) < len(desigs):
        print(f"# {len(desigs) - len(uniq)} duplicate designation(s) collapsed", flush=True)

    out = open(a.out, "a", buffering=1, encoding="utf-8") if a.out else None
    rep = open(a.report, "a", buffering=1, encoding="utf-8") if a.report else sys.stdout
    recs = []
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_worker, (d, a.allow_multiopp, a.timeout,
                                        a.first_opposition)): d for d in uniq}
            for fu in as_completed(futs):
                recs.append(fu.result())
                _emit(recs[-1], out, rep, len(recs), len(uniq))
    else:
        for i, d in enumerate(uniq, 1):
            recs.append(audit(d, allow_multiopp=a.allow_multiopp, timeout=a.timeout,
                              first_opposition=a.first_opposition))
            _emit(recs[-1], out, rep, i, len(uniq))

    order = {"suspect": 0, "possible": 1, "advisory": 2, "clean": 3}
    hits = sorted([r for r in recs if r.get("status") == "ok"
                   and r["verdict"] in ("suspect", "possible", "advisory")],
                  key=lambda r: (order[r["verdict"]],
                                 -(r["worst"].get("anomaly", 0) if r.get("worst") else 0)))
    print(f"\n{'=' * 78}\nSUMMARY  {len(recs)} objects in {time.time() - t0:.0f}s", file=rep)
    for k in ("ok", "skip-multiopp", "skip-shape", "error", "timeout"):
        n = sum(1 for r in recs if r.get("status") == k)
        if n:
            print(f"  {k:<16}{n}", file=rep)
    for v in ("suspect", "possible", "advisory", "clean"):
        n = sum(1 for r in recs if r.get("status") == "ok" and r.get("verdict") == v)
        if n:
            print(f"  {v:<16}{n}", file=rep)
    if hits:
        print(f"\nRANKED  (worst tracklet, strongest first)", file=rep)
        for r in hits:
            w = r["worst"]
            parts = "+".join(f"{k}{v:g}" for k, v in sorted(
                w.get("parts", {}).items(), key=lambda kv: -kv[1]))
            print(f"  {r['verdict']:<9}{r['d']:<14}{w['date']} {w['code']:<4} "
                  f"iso={w['iso']:<6.1f} anom={w.get('anomaly', 0):<5.2f} "
                  f"[{parts}] {'; '.join(w['flags'])}", file=rep)
    if out:
        out.close()
    return recs


def _emit(rec, out, rep, i, n):
    if out:
        out.write(json.dumps(rec) + "\n")
    print(f"[{i}/{n}] {format_record(rec)}", file=rep, flush=True)


if __name__ == "__main__":
    main()
