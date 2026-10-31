"""MPC-style identification output (`--ident-check`): decide whether a fitted pair is a clean
identification and, if so, format it.

The rule (the load-bearing part): an identification `A = B` is emitted ONLY when every
observation that does NOT fit the joint orbit is a clean tracklet exception -- a whole
stray single-night tracklet, or a bounded handful of isolated outliers. Such drops are fine
and listed under `except:`. If instead the drops GUT a solid multi-night apparition --
most-but-not-all of a night, or so many nights that an opposition no longer independently
confirms its side (in a 2-opposition link: collapsed to a lone kept night) -- that is a
genuine cross-path disagreement between two well-determined arcs, i.e. NOT the same object,
and nothing is emitted (see identify() for the exact night-level gates).

    A = B                          (no tracklet-level exception; the fit may still exclude a few
                                    isolated outlier obs -- bad observations, not bad tracklets)
    A = B except:                  (whole stray tracklet(s) dropped, every obs > 1.25" off)
         <verbatim OBS80 line>
         ...
    <nothing>                      (a gutted night / collapsed opposition / too many drops)

In a TWO-opposition link the clip itself is the whole question, because one orbit can nearly always
thread two short arcs once it is allowed to discard a night -- and a coherent survey run essentially
never contains a night that is not the object. So such a link must retain the bulk of BOTH sides'
own obs (MPC_MIN_RETENTION) and carry no `clip_doubt`: a join that exists only because a night of
one side's own coherent short arc was thrown away, with neither arc constrained enough to make that
clip non-load-bearing, is two DIFFERENT objects. Those are a flat 'no' -- not even a 'y-examine'
candidate, and the years between the apparitions do not rescue them. Two things do exempt a link
from both gates: spanning 3+ oppositions (an orbit that threads three apparitions coherently is
worth a human look however many nights it clips), or an UNCLIPPED side -- one that gave up not a single
night -- keeping an arc of UNCLIPPED_ANCHOR_ARC_DAYS or more, which anchors the orbit on its own. An
exempt link keeps whatever it would otherwise be: a clean y/y2, or the 'y-systematic' / 'y-examine'
its own residuals earn it (see clip_doubt() and examine_or_no()).

A separate, ADVISORY photometric check runs on every emitted identification: the linked arcs'
implied absolute magnitudes are compared on a dH x p grid (`_mag_stats`/`_mag_tier`) that gates
on BOTH the raw difference (physically meaningful?) and its statistical significance (resolved
above photometric noise, with a per-object systematic floor so tight survey photometry can't fake
certainty). When the brightness doesn't agree the match is STILL emitted -- we never drop on
photometry, only flag -- but a leading comment marks it for a closer look before submission, and
the classify code gains a suffix: `-mag?` (advisory), `-mag` (flag), or `-mag!` (strong):

    # FLAG mag residual 1.8 mag between oppositions (A H~7.2, B H~9.0; p=3e-05) -- examine before submitting
    A = B

A second ADVISORY flag (`thin_link`) rides along the same way: an identification supported by
fewer than THIN_LINK_MIN_OBS observations in total is barely over-determined, so the joint orbit
fitting says little -- every other gate here judges what the fit DROPPED, and a 9-obs link that
drops nothing sails through them all. It is still emitted, with a leading comment and a '-short'
suffix on the classify code (`y-short`).
"""
import math
from datetime import datetime, timedelta
import numpy as np
from pyroorb.orbit_common import NIGHT_GAP_DAYS, OPP_GAP_DAYS   # single source for the gap thresholds
from pyroorb.orbit_common import same_detection   # ...and for "these two records are one detection"


def _jd_year(jd):
    """Calendar year of a UTC Julian date (exact at the day level; labels an apparition)."""
    return (datetime(2000, 1, 1, 12) + timedelta(days=float(jd) - 2451545.0)).year
MAX_DROP_TRACKLETS = 4     # at most this many suspect single-night tracklets may be dropped
                           # (no cap on the TOTAL obs so dropped -- a whole stray night is a
                           # whole night however well it was sampled)
MAX_DROP_SCATTER = 2       # ...plus at most this many isolated outliers (a minority of an
                           # otherwise-kept night) dropped OUTSIDE any wholly-dropped tracklet
INTERLEAVE_MIN_FRAC = 0.5  # two designations' kept arcs INTERLEAVE (see _interleaved) when their
                           # common time window covers at least this share of the shorter arc
MARGINAL_REJECT_SIGMA = 4.5  # in an interleaved link a scattered drop under this many sigma is
                           # rejection noise and does not count toward the scatter cap: 1.5 x the
                           # 3-sigma REJECT_LIMIT, the first step of the rejection ramp
                           # (arc_extend.auto_reject_within_arc, find_orb's
                           # auto_reject_obs_within_arc) -- only the strictest level removes it
SCATTER_FRAC = 0.05        # ...where the scatter allowance scales with the arc: a big link clips
                           # more honest one-off outliers than a 12-obs pair ever could, so the
                           # cap is max(MAX_DROP_SCATTER, SCATTER_FRAC * total obs). (2025 NM2 =
                           # 2021 WL5: 3 isolated outliers among 62 obs is normal rejection noise,
                           # not a fishy link -- the flat cap of 2 was tuned on small arcs.)
MAX_KEPT_RMS = 0.6         # the kept obs must fit at least this well (ARCSEC) to be an ID...
# ...but an arcsec bar measures the ASTROMETRY's quality, not the identification's truth, and this
# one was calibrated on modern survey asteroid arcs. An object whose own record cannot reach 0.6"
# -- a COMET, whose coma centroid wanders, or a long arc carrying decades of photographic plates --
# misses it no matter WHICH designations are in the fit, so an absolute bar alone turns "this object
# has hard astrometry" into "these are two different objects". 243P = 2025 MT356 is the case that
# found it: 243P fitted ALONE lands at 0.649" (same per-night trend, same 85 rejects), the joint fit
# WITH 2025 MT356 at 0.647" -- marginally BETTER -- and MT356's own 6 obs sit at 0.036", the
# TIGHTEST part of the whole fit. The absolute bar filed that as a flat 'no'.
#
# So the gate is also asked in the currency that adapts to astrometric quality: SIGMA-NORMALIZED
# residuals, the same measure find_orb judges a fit by (`FittedOrbit.rms_normalized`, ~1.0 = the
# residuals are what the assigned per-obs uncertainties predict). An arc of 1"-uncertainty plates
# fitting at 0.65" is a GOOD fit; an arc of 0.1"-uncertainty survey astrometry fitting at 0.65" is
# not, and the normalized measure says so where the arcsec bar cannot.
#
# Measured on the 98 PAIR fixtures: normalized rms has median 0.52
# and a MAXIMUM of 1.31 across the whole population, 87 of which are clean identifications -- and
# not one of those fixtures' arcsec rms reaches 0.6" (max 0.576"), so this relaxation changed no
# existing verdict. Which also left it guarded by nothing, so the motivating object was fixtured:
# PAIR_2025MT356 is the ONE fixture above the arcsec bar (0.647", 0.75 sigma -- mid-population),
# and the 243P SOLO fixture beside it is what makes "the link is not what costs the rms" a
# comparison rather than a claim (test_ident_check.test_hard_astrometry_pair_is_examined).
MAX_KEPT_NRMS = 2.0        # sigma. The normalized bar that lets a fit over MAX_KEPT_RMS still count
                           # as fitting. 2.0 is ~1.5x the worst value any of the 98 pair fixtures
                           # reaches (1.31) and 2x "consistent with the stated uncertainties", so a
                           # fit admitted here is one whose residuals its own astrometry predicted.
RMS_CEILING = 1.5          # arcsec. ...but only up to an absolute ceiling, because sigma can be
                           # WRONG. Old astrometry is assigned 1-3" uncertainties, so a genuinely
                           # bad orbit through two ancient arcs could fit at 2" and still score
                           # ~1 sigma; past this the arcsec bar decides alone, as before. This is
                           # the one JUDGMENT constant here -- no population of same-object pairs
                           # that MISS 0.6" exists to calibrate it -- so it is set just clear of the
                           # hard case we have (243P's worst apparition needs 0.85") and left tight.
#
# NOT here, and deliberately: a PER-DESIGNATION version of this test ("is one side the reason for
# the excess?", each designation's own residuals against the pooled residuals of the others,
# leave-one-designation-out). It is the intuitive shape for catching a forced side, it was
# implemented first, and on the 98 pair fixtures it is a NULL: worst-ratio median 1.65x for the
# fixtures identify() accepts vs 1.78x for those it rejects in arcsec, 1.78x vs 1.79x normalized,
# and the population MAXIMUM (22.8x raw, 6.7x normalized) belongs to an ACCEPTED identification.
# It separates nothing. That is the expected answer on reflection: this gate was never the chimera
# discriminator -- rejection HIDES a mislink's misfit in the dropped obs rather than in the rms,
# which is why MPC_MIN_RETENTION is the decisive tell -- so the rms clause is only a backstop
# against a garbage fit, and it does not need a per-side refinement it cannot support.
EXCEPT_CLEAN_MAX_OBS = 1   # at most this many EXCEPTED observations still counts as a CLEAN
                           # match ('y'); beyond it the link is a 'y2' -- an identification whose
                           # join rests on a claim that whole tracklets are not this object. One
                           # stray observation is astrometry; a tracklet is a night somebody
                           # reported, and disowning it is a judgement a human has to make. This
                           # is the same split the batch has always printed (cli._ident_check_one
                           # counted the echoed lines), lifted here so the CLI code and the web
                           # UI's caution cannot drift apart.
EXCEPT_MIN_RESID_ARCSEC = 1.25  # an `except:` line is an MPC-facing claim that a TRACKLET does
                           # not belong to the object, so only a WHOLLY-dropped tracklet with
                           # EVERY obs beyond this total residual is listed. Partial-tracklet
                           # outliers and near-threshold whole tracklets are bad OBSERVATIONS,
                           # not bad tracklets -- excluded from the fit, never echoed.
DUP_TOL_DAYS = 1.0e-4      # ~9 s: an excluded obs this close in time to a KEPT obs from the
                           # SAME station AND at the same sky position is the same detection
                           # cross-listed under another designation -- dropping it is
                           # de-duplication, not a suspect drop
DUP_TOL_ARCSEC = 1.0       # ...and the two must also coincide on the sky to within this: a
                           # single station can image two DIFFERENT objects in the same exposure
                           # (F51 saw 2014 HZ280 and 2018 BV65 ~28' apart at the same instant),
                           # so same-station same-time is NOT enough -- the position must match too
# --- Photometric-mismatch flag (advisory, NEVER a rejection) -----------------------------------
# The match is STILL emitted (some survey photometry is junk, e.g. bad W84 magnitudes); the flag
# is an advisory '# FLAG' comment / classify-code suffix, never a rejection. A genuine same-object
# pair shares one absolute magnitude across oppositions; a chimera (two different bodies) shows the
# size/albedo gap as a persistent H offset. Rather than one hard |dH| threshold we combine TWO
# gates -- a raw-difference gate (is the gap physically meaningful?) and a significance gate (is it
# resolved above photometric noise?) -- on the dH x p grid in `_mag_tier`. This adapts to how tight
# each object's photometry actually is: a 1-mag gap between two internally-consistent arcs is far
# more telling than the same gap between two noisy ones.
MAG_SYS_FLOOR = 0.25       # mag. Irreducible per-object systematic (band->V, phase and station
                           # reduction error) that does NOT average down with more obs. Added in
                           # quadrature to the SEMs so a survey contributing many internally-tight
                           # points can't fake an arbitrarily small uncertainty on its implied H
                           # (a genuine same-object pair can still show ~0.3-0.5 mag of such offset).
MAG_MIN_N     = 3          # a designation needs this many usable implied-H obs for a meaningful
                           # mean/scatter; below it the p-value is undefined -> flag stays off.
MAG_DH_MIN    = 0.3        # mag. Below this the gap is within a single body's rotational lightcurve
                           # amplitude -- never a chimera signature, so NEVER flagged, however
                           # 'significant' ultra-tight photometry makes it look.
MAG_DH_MID    = 0.7        # mag. From here a 2-sigma (p<MAG_P_WEAK) resolution flags; below it a
                           # gap must clear the full 3-sigma (p<MAG_P_STRONG) bar.
MAG_DH_HIGH   = 1.5        # mag. A size/albedo gap no lightcurve explains: at least ADVISORY even
                           # when the photometry is too sparse to resolve it, STRONG once 3-sigma.
MAG_P_WEAK    = 0.05       # ~2-sigma: the difference is resolved above photometric noise.
MAG_P_STRONG  = 0.003      # ~3-sigma: strongly resolved.
SYS_BETWEEN_ARCSEC = 0.5   # arcsec. A correct orbit leaves zero-mean RANDOM residuals per night;
                           # a wrong linkage (or a contaminated apparition) leaves per-night MEAN
                           # residuals that trend across an apparition -- the orbit can't trace the
                           # true curvature, so each night sits coherently off it. FLAG when the
                           # RMS of per-night mean residuals within one apparition reaches this...
SYS_WITHIN_RATIO = 3.0     # ...AND is at least this many times the pooled within-night scatter
                           # (the nights are each internally TIGHT but collectively displaced --
                           # structure, not noise). Both gates together keep genuine links, whose
                           # per-night means are small and comparable to the scatter, off the flag.
SYS_SE_RATIO = 3.0         # ...OR is at least this many times the standard ERROR of those means.
                           # SYS_WITHIN_RATIO compares a night's MEAN against the scatter of a
                           # SINGLE observation, which silently demands more and more signal the
                           # better sampled the night is: a night of 16 obs averages its scatter
                           # down by 4x, so a mean that is 8-sigma resolved still scores ~1.5x and
                           # is thrown away as "noise". The mean's own uncertainty is SE =
                           # sd/sqrt(n), and between/SE is ~1.0 for a correct orbit BY
                           # CONSTRUCTION (verified: median 1.11 over 960 apparition x axis rows
                           # of the fixture suite), so this ratio means the same thing at every
                           # night length. OR-ed with -- never replacing -- the scatter test, so
                           # everything flagged before is still flagged. The ABSOLUTE floors above
                           # are what keep this honest and are deliberately unchanged: they are
                           # the gate that holds back the genuine identifications whose nights are
                           # significantly but harmlessly displaced (GUARD_2014EF94 sits at 34x
                           # its SE and 45x its scatter -- only the 0.5" floor keeps it a match).
                           # 2016 UJ137 is the motivating case: 6 nights 3-8 sigma off in RA, one
                           # apparition over the strict floor at 0.53", scored 2.19x and was lost.
SYS_MIN_NIGHT_OBS = 2      # a night needs this many obs for its mean/scatter to be meaningful
SYS_MIN_NIGHTS = 2         # and an apparition this many such nights to show a cross-arc trend
SYS_BETWEEN_CORROB = 0.35  # relaxed absolute floor used ONLY when >= SYS_MIN_CORROB_APPS separate
                           # apparitions EACH independently show a ratio-dominant trend. One
                           # apparition at 0.45" could be a fluke of one night's weighting; the SAME
                           # systematic signature in two independent apparitions is far stronger
                           # evidence, so we lower the absolute floor for the corroborated case. This
                           # catches links that trend ~0.4-0.5" in every apparition -- just under the
                           # strict 0.5" floor -- while each single arc alone fits to ~0.1" (e.g.
                           # 2021 TJ205 = 2015 KS376: RA between 0.45"/0.42" at ratio 4.8x/4.6x).
SYS_MIN_CORROB_APPS = 2    # how many apparitions must corroborate to unlock the relaxed floor
SYS_LOO_MIN_NIGHTS = 4     # with this many qualifying nights in an apparition, the between-night
                           # signal must SURVIVE removing the single worst night. A mislink trend
                           # displaces MULTIPLE nights (the orbit can't trace the curvature
                           # anywhere along the arc), but one deviant night among 4+ clean ones is
                           # just mediocre astrometry the fit rightly keeps -- not link-level
                           # structure (2025 NM2 = 2021 WL5: a lone L04 night +1.2" off in Dec in a
                           # 5-night apparition faked a 0.58" "trend"). A 2-3 night apparition keeps
                           # the conservative all-nights measure: there a lone night IS half the
                           # arc, and a genuine 2-night trend must stay flagged (2025 OT173).
SYS_SHORT_ARC_DAYS = 4.0   # ...and when BOTH designations are arcs no longer than this, a fired
                           # trend is a flat 'no', not a 'y-systematic' to be reviewed. The
                           # y-systematic verdict exists because a trend CAN have an innocent
                           # explanation -- unmodelled nongravs, a bad night, a survey's own
                           # systematics -- that a human can weigh against the rest of the arc.
                           # Two ~day-long arcs give a human nothing to weigh: there is no
                           # curvature on either side to say the orbit is right and the trend is
                           # noise, an unmodelled nongrav large enough to walk the residuals would
                           # have been auto-detected on a real object, and the joint fit is so
                           # weakly over-determined that a clear systematic simply means the one
                           # orbit does NOT thread both arcs. Nothing here could ever be called a
                           # fit, so it goes out with the non-matches (2025 OC732 + 2014 HJ409:
                           # 27/27 kept at 0.57" overall, yet the Dec per-night means sit 0.78"
                           # apart at 27x their own error, across two 1-2 d arcs).
                           # Judged on each designation's OWN arc (all its obs, kept or not), so a
                           # designation with a genuinely long record is never called short.
SYS_TREND_Z = 5.0          # A trend is a trend BELOW the arcsec floors when it is statistically one:
                           # the floors above measure how BIG the nightly
                           # offsets are, and a link whose whole residual budget is 0.3" cannot show
                           # a 0.5" trend however wrong it is -- 2025 NF505 = 2015 HG272 fits 32/32 at
                           # 0.29" (each arc alone 0.08"/0.06") while its 2025 Dec means walk
                           # -0.59 -> -0.22 -> +0.07 -> +0.68" over four nights each tight to 0.05".
                           # So a THIRD arm, OR-ed like the second (it only ever adds flags): per
                           # apparition and axis, a weighted line through the station-night MEAN
                           # residuals against time, each night carrying its within-night error
                           # (pooled scatter / sqrt(n)) PLUS the measured station-night offset tau
                           # (nightmodel_table) -- the displacement a CORRECT orbit leaves, which is
                           # why GUARD_2014EF94's harmlessly displaced nights stay unflagged (z 3.3)
                           # -- and the slope's sigma inflated by sqrt(chi2/dof) of that line when
                           # the nights scatter about it more than the model says (never deflated).
                           # Flag at |slope| / sigma >= this. Calibrated:
                           # 0 of 75 synthetic CORRECT two-apparition links (two apparitions of one
                           # known object fitted as two designations -- exactly the NF505 geometry)
                           # reach 5 (2.7% reach 4); of 77 single-object full arcs only comet 243P
                           # (a real unmodelled nongrav, already flagged); of 81 MPC-ACCEPTED PAIR
                           # fixtures one, 2015 CY86 (below). The per-test null is heavier-tailed
                           # than Gaussian (1.0% at 3, 0.25% at 5, n=800), which is why the bar is
                           # 5 and not 3.
SYS_TREND_MIN_SWING_RATIO = 8.0  # ...AND the walk must be LARGE against one night's precision:
                           # swing across the apparition >= this x the pooled within-night scatter.
                           # Significance alone flags a correct orbit's harmless drift wherever the
                           # nights are dense enough to measure it: 2015 CY86 = 2026 DO29, a real
                           # identification, walks 0.16" in Dec across seven X05 nights
                           # holding 1482 obs -- z 5.9, but only 3.7x its 0.04" night scatter, the
                           # early light nights trailing off the two dense ones (1136 obs) that pin
                           # the orbit. Catalogue zones, timing and weighting can move a correct
                           # orbit's nights a few single-obs errors; a forced link moves them many.
                           # Measured on every test
                           # with z >= 5: correct links -- 185 synthetic two-apparition links, 76
                           # single-object arcs, 80 MPC-accepted pairs -- top out at 6.9 (162385,
                           # 2003 BN4 synthetic, CY86 3.7); the forced links start at 8.8 (TJ205;
                           # DS43 17.7, HV328 15.9-40.6, OC732 39-45, NF505 11.5). 8 sits inside
                           # that gap. 2014 EW228 (5.0) falls under it and stays flagged by the
                           # arcsec arms, which it already fired.
SYS_TREND_MIN_BLOCKS = 3   # station-nights needed in an apparition: a line through two points
                           # leaves nothing to judge it by.


# --- Per-tracklet COHERENT-OFFSET flag (COMPARATIVE: needs a noise reference) -------------------
# The trend detector above needs MULTIPLE nights trending and drops the single worst night
# (SYS_LOO), so a lone contaminant survives it. coherent_offset_misfit catches a single night
# whose residuals are coherently offset from zero -- BUT only meaningfully when measured against
# a noise reference, because real old astrometry has genuinely deviant nights that look identical
# in a single fit. It is therefore used ONLY in the ITF-vetting context, where the seed's
# pre-search residual scale is the reference: a night offset many times the noise the orbit
# achieved BEFORE the link is the contaminant tell (FU24: seed fits to 0.03", the link then leaves
# the object's own nights 0.16-0.45" off -> 5-14x).  NOT wired into the general fit banner.
COH_MIN_OBS = 3            # a night needs this many obs for a meaningful mean
COH_RATIO = 4.0           # night mean must exceed this many times the noise_floor reference
COH_MEAN_FLOOR = 3.0      # AND dominate the night's own scatter (|mean| >= this x sd/sqrt(n))

# --- Thin-collapsed-arc FALSE-POSITIVE flag (advisory, NEVER a rejection) -----------------------
# An emitted 'A = B [except]' whose match rests on ONE designation collapsing to a single sub-day
# tracklet is a false-positive risk. A genuine identification threads ONE orbit through a strong
# arc on BOTH sides; when the joint fit can only keep a < 24 h sub-arc of one designation -- and
# gets there by DROPPING the rest of that designation's own multi-night arc -- the intra-arc
# curvature that would actually TEST the link has been thrown away, so the "match" is really the
# OTHER arc fitting itself with a loosely-attached tracklet. (K25QC2H = K14HA1V: 2014 HV101's solid
# 4-night 2014 arc keeps 23/23, but 2025 QH122's own 3-night/11-day arc collapses to a 0.8 d kept
# sub-arc -- its 08-17 night dropped -- and 2014 HV101's 2011 and 2023 apparitions are dropped
# whole; the tool calls it 'y2', but nothing on the 2025 side independently confirms the identity.)
FALSEPOS_ARC_DAYS = 1.0       # a side whose KEPT obs span less than this (< 24 h) is a lone tracklet
FALSEPOS_COLLAPSE_DAYS = 1.0  # ...and it is suspect ONLY if that side's DROPPED obs extend its arc
                              # this much past the kept window -- i.e. it was genuinely observed
                              # longer and got COLLAPSED. An inherently single-night tracklet linked
                              # to a known object (the common LEGITIMATE identification) keeps its
                              # whole arc (dropped ~= 0), so kept ~= full span and it is NOT flagged.
FALSEPOS_BLUNDER_ARCSEC = 10.0  # ...and only PLAUSIBLE dropped astrometry counts toward that full
                              # span: a dropped obs sitting this far off the orbit is a blunder that
                              # never testified about the arc's curvature, so clipping it collapses
                              # nothing (2025 NM2 = 2021 WL5: NM2's only drops are a junk 126
                              # tracklet at ~76" -- its kept 2-night arc is everything real it has).
# --- Two-opposition COINCIDENCE gate (a real-doubt DOWNGRADE, not an advisory flag) -------------
# A 2-opposition link that survives the collapse ban below can still be a coincidence: two short
# arcs a few years apart can almost always be threaded by ONE orbit if you are allowed to clip a
# night. The confirmed 2015 OF201 = 2025 NA385 (real) and the coincidental 2015 XO232 = 2025 PD58
# are night-for-night the SAME shape -- a 2-night ~1 d arc + a 4-night ~12 d arc, one night dropped
# -- so neither night count nor arc length separates them. What does is WHY the night was dropped:
# OF201's dropped 2015 night sits 23" off the joint orbit (an obvious stray that plainly is not this
# object), while XO232's dropped night is only ~4.5" off -- it plausibly belongs to its coherent
# 3-night PS1 sequence, so discarding it to force the join is a coincidence, not an identification.
SECURE_MIN_KEPT_NIGHTS = 3    # a 2-opp identity stands on its own -- no drop needed to believe it --
                              # when BOTH oppositions independently keep at least this many nights.
SECURE_MIN_ARC_DAYS = 20.0    # ...or when one side keeps an arc at least this long: a real baseline
                              # pins the orbit, so a clipped night on the other side is not load-bearing
                              # (2025 NM2 = 2021 WL5's 1459 d arc; 2024 WY87's own 23 d arc).
COINCIDENCE_HARD_DROP_ARCSEC = 10.0  # a not-secure link is trusted only when the clipped night is a
                              # gross stray, at least this far off the joint orbit (OF201's 23"
                              # night qualifies; XO232's 4.5" does not). Numerically equal to
                              # FALSEPOS_BLUNDER_ARCSEC today, but an independent calibration --
                              # retune either without moving the other.
COINCIDENCE_MIN_KEPT_ARC_DAYS = 3.0  # ...AND a gross-stray drop still cannot rescue a not-secure
                              # link if, after the clip, NO opposition keeps an arc even this long:
                              # what remains is two ~single-tracklet arcs that one orbit threads by
                              # coincidence, and the "stray" was the object's OWN longer baseline
                              # (its own record) whose removal is what let the join fit. A real y2
                              # keeps a genuine multi-day baseline on at least one side (OF201's
                              # 12 d / 4-night NA385 arc, HH101's 9 d DQ27 arc); the false
                              # K03Q91C = K06QI0X keeps only ~1 d on BOTH sides -- 2003 QC91's 58 d
                              # arc collapses to 1.0 d once its 78" 304 night is clipped, and 2006
                              # QX180 is itself 1.0 d. Well below any real keeper (>=9 d), well
                              # above the coincidence (~1 d).
UNCLIPPED_ANCHOR_ARC_DAYS = 35.0 # ...and NEITHER clip rule (clip_doubt, nor the 2-opposition retention
                              # gate) fires when the designation that clipped NOTHING keeps an arc this
                              # long. A side that gave up not one night and still spans 35+ days is
                              # a real anchor: it carries enough of its own curvature that the orbit
                              # is not free to be bent onto whatever the other arc happens to be, so
                              # the clip over there is no longer what makes the join close. Such a
                              # link keeps whatever it would otherwise be -- a clean y/y2, or the
                              # 'y-systematic' / 'y-examine' its own residuals earn it -- and stays
                              # in front of a human instead of being filed as a non-match. Requires
                              # a side with NO whole-night clip at all: when both sides clip there
                              # is no anchor, however long either arc is (2014 EN102 = 2022 TB28).


THIN_LINK_MIN_OBS = 12        # ...and an emitted identification supported by fewer than this many
                              # observations in TOTAL is flagged 'short' (advisory, never a
                              # rejection). An orbit has 6 free parameters and each observation
                              # contributes 2 residuals, so 12 obs leave only 18 - 6 = 12 degrees
                              # of freedom SPLIT ACROSS TWO designations -- barely over-determined.
                              # At that size "the joint orbit fits" is weak evidence: with so few
                              # constraints one orbit threads a wrong pairing nearly as easily as
                              # the right one, and none of the other gates can tell the difference
                              # (they all key on what was DROPPED, and a 9-obs link that drops
                              # nothing looks immaculate -- J99K17K = K16L89L, 9/9 @ 0.56").
                              # So the ID is still emitted and still classified y/y2, but the code
                              # gains a '-short' suffix and the written record a '# FLAG' comment,
                              # exactly like the photometric flag: a human looks before submitting.


def thin_link(fit):
    """Advisory SHORT-LINK flag: an identification resting on fewer than THIN_LINK_MIN_OBS
    observations. Returns a short description, or None.

    Judged on INCLUDED obs -- the dropped ones support nothing -- and counted across BOTH
    designations, since it is the joint orbit's total constraint that is thin. NEVER a rejection:
    the ID is emitted as usual (the batch suffixes the classify code with '-short'; the written
    record gets a '# FLAG' comment, as the photometric mismatch does), because a small link can of
    course be real -- it just has not been TESTED by anything, and a human should say so."""
    mask = np.asarray(fit._mask, dtype=bool)
    n_kept = int(mask.sum())
    if n_kept >= THIN_LINK_MIN_OBS:
        return None
    by = {}
    for i, o in enumerate(fit._obs):
        if mask[i]:
            d = o.get("designation")
            by[d] = by.get(d, 0) + 1
    parts = ", ".join(f"{d} {n}" for d, n in sorted(by.items(), key=lambda kv: -kv[1]))
    return (f"only {n_kept} obs ({parts}) support the joint orbit -- under {THIN_LINK_MIN_OBS}, "
            f"so it is barely over-determined")


def kept_obs_fit(fit):
    """Do the kept obs actually FIT well enough for the pair to be an identification?

    Returns (ok, note): `note` is a short phrase for the caller to print -- what failed when ok is
    False, and, when the sigma-normalized relaxation carried a fit that missed the arcsec bar, WHY
    it was accepted anyway. It is None only in the ordinary case of clearing the arcsec bar
    outright, so callers must branch on `ok`, never on the note.

    A fit passes when EITHER
      * its unweighted rms is under MAX_KEPT_RMS (arcsec) -- the original bar, unchanged; or
      * its SIGMA-NORMALIZED rms is under MAX_KEPT_NRMS and its arcsec rms is still under
        RMS_CEILING -- the residuals are what this astrometry's own stated uncertainties predict.

    The second clause exists because an arcsec bar asks about measurement quality, not identity: a
    comet's coma centroid or a 1970s photographic plate cannot deliver 0.6" for ANY set of
    designations, so the bar alone files hard astrometry as a non-match (243P = 2025 MT356: 0.647"
    jointly, 0.649" for 243P by ITSELF, 0.75 sigma). RMS_CEILING bounds the trust, because a wrong
    orbit through arcs assigned 1-3" uncertainties could also score ~1 sigma.

    This clause is a BACKSTOP against a garbage fit, not the chimera discriminator -- rejection
    hides a mislink's misfit in the DROPPED obs rather than in the rms of the kept ones, which is
    why MPC_MIN_RETENTION is the decisive tell. See the constants above for the per-designation
    refinement that was tried and measured to be a null."""
    rms = fit.rms_residual()
    if rms < MAX_KEPT_RMS:
        return True, None
    nrms = fit.rms_normalized()
    if rms >= RMS_CEILING:
        return False, (f"kept obs fit at {rms:.2f}\" -- past the {RMS_CEILING:.2f}\" ceiling, so "
                       f"the orbit is wrong rather than the astrometry hard")
    if nrms >= MAX_KEPT_NRMS:
        return False, (f"kept obs fit at {rms:.2f}\" = {nrms:.1f} sigma -- worse than the "
                       f"uncertainties this astrometry claims, not merely hard astrometry")
    return True, (f"fits at {rms:.2f}\" (over the {MAX_KEPT_RMS:.2f}\" bar) but at {nrms:.2f} "
                  f"sigma of its own stated uncertainties -- hard astrometry, not a misfit")


def _split(times, gap):
    """Index groups of ASCENDING `times`, split wherever a gap exceeds `gap`."""
    t = np.asarray(times)
    groups, start = [], 0
    for i in range(len(t) - 1):
        if t[i + 1] - t[i] > gap:
            groups.append(list(range(start, i + 1)))
            start = i + 1
    groups.append(list(range(start, len(t))))
    return groups


def _interleaved(obs, keep):
    """True when the designations' KEPT arcs overlap in time: their common window covers at least
    INTERLEAVE_MIN_FRAC of the shortest kept arc (and that arc has a duration at all).

    The "no baseline" objection to a thin join -- two short arcs that close only by dropping
    observations, so the drops bought the join -- assumes the arcs sit side by side and the joint
    orbit EXTRAPOLATES from one to the other. When they interleave, it does not: each side's
    observations fall inside the other's arc, so the joint orbit is checked against both at the
    same epochs, often from the same stations, and two different objects cannot track each other
    there to sub-arcsec. A partial drop in such a join is outlier trimming, not the price of the
    link. (2026 SW11 = 2026 SA12: two 0.9-day arcs from the same 22 hours at H36, I41/I52 and
    others, linked at 0.49" with 7 of 48 scattered obs trimmed -- two of them the 703 pair SW11's
    own solo fit rejects.)"""
    span = {}
    for i, o in enumerate(obs):
        if keep[i]:
            t = float(o["jd_utc"]); d = o.get("designation")
            lo, hi = span.get(d, (t, t))
            span[d] = (min(lo, t), max(hi, t))
    if len(span) < 2:
        return False
    shortest = min(hi - lo for lo, hi in span.values())
    common = min(hi for _, hi in span.values()) - max(lo for lo, _ in span.values())
    return shortest > 0 and common >= INTERLEAVE_MIN_FRAC * shortest


def _resid_sigmas(fit):
    """Per-obs residual in the fit's own sigmas, hypot(dRA*/sig_ra, dDec/sig_dec) -- the
    currency the outlier rejection judged each obs in (arc_extend.Evaluator.resid_sigmas).
    +inf where the fit does not carry its sigmas, so nothing reads as marginal."""
    R = getattr(fit, "_resid_all", None)
    sra, sdec = getattr(fit, "_sig_ra", None), getattr(fit, "_sig_dec", None)
    if R is None or sra is None or sdec is None:
        return np.full(len(fit._obs), np.inf)
    R = np.asarray(R, dtype=float)
    return np.hypot(R[:, 0] / np.asarray(sra, float), R[:, 1] / np.asarray(sdec, float))


def _tracklets(times, codes, desigs=None):
    """Index groups = TRACKLETS: one observatory, one night, one SUBMISSION. Split into nights
    (gap > NIGHT_GAP_DAYS), then group each night BY (OBSCODE, DESIGNATION) -- two stations that
    happen to image the same night are two distinct tracklets, so dropping one whole station's
    detections is a clean tracklet drop, not a 'partial night' (K22Y13L = K25NL8F: an F52
    tracklet shares a night with a kept G96 tracklet). For the SAME reason, two DESIGNATIONS
    submitted from the same station on the same night are two distinct tracklets -- each
    provisional designation is its own MPC submission -- so dropping one designation's obs is a
    clean tracklet drop, not a partial gutting of a shared night. (2015 KR274 = 2025 PY146: a
    W84 night carries 5 kept 2015 KR274 obs plus 3 stray 2025 PY146 obs 1.4 deg off the orbit;
    the strays are a whole separate tracklet, not a partial night.) When `desigs` is None the
    split is by OBSCODE alone (back-compat)."""
    out = []
    for night in _split(times, NIGHT_GAP_DAYS):
        by_key = {}
        for i in night:
            key = codes[i] if desigs is None else (codes[i], desigs[i])
            by_key.setdefault(key, []).append(i)
        out.extend(by_key.values())
    return out


def _duplicate_excluded(times, codes, ras, decs, excluded):
    """Boolean over obs: an EXCLUDED obs that coincides (same station, same instant, SAME sky
    position) with a KEPT obs -- the same physical detection cross-listed under another
    designation. De-duplicating it (keeping one copy) is benign, not a suspect tracklet drop.
    (HZ280 = BV65 = OS355: one F51 tracklet is submitted under two of the linked names.)

    The sky-position check is essential: a survey exposure images a wide field and CAN carry two
    different objects at the same instant. F51 submitted a genuinely different 2014 HZ280 tracklet
    and 2018 BV65 tracklet at identical timestamps but ~28' apart -- same station + same time but
    NOT the same detection, so those excluded obs are real drops (a 'y2'), not duplicates.

    The coincidence test is ``orbit_common.same_detection``, shared with the fetch-time dedup
    (``obs_sources.find_duplicate``) so the two cannot drift apart again. This path asks whether
    two records name one point on the SKY, so it scales by cos(dec) (``great_circle=True``) and
    keeps its own looser ~9 s window -- it compares across designations whose timestamps have
    been through different round trips. Sharing the arithmetic is what brought the circle wrap
    here: differencing RA by plain subtraction read a cross-listed pair straddling RA 0h as 360
    degrees apart, so such a pair was reported as a suspect drop rather than the benign
    duplicate it is."""
    t = np.asarray(times, float)
    ra = np.asarray(ras, float)
    dec = np.asarray(decs, float)
    pos_tol = DUP_TOL_ARCSEC / 3600.0                    # degrees
    kept = ~excluded
    dup = np.zeros(len(t), bool)
    for i in np.flatnonzero(excluded):
        for j in np.flatnonzero(kept):
            if codes[j] == codes[i] and same_detection(
                    t[i], ra[i], dec[i], t[j], ra[j], dec[j],
                    time_tol_days=DUP_TOL_DAYS, pos_tol_deg=pos_tol,
                    great_circle=True):
                dup[i] = True
                break
    return dup


def _packed_pair(fit):
    """(left, right) packed designations: the fit's primary on the left, the other
    designation(s) on the right. Packed forms are read from the obs (exactly the packed
    desig the MPC file carried). Returns (None, None) if it isn't a 2+-designation link."""
    dmap = {}
    for o in fit._obs:
        dmap.setdefault(o.get("designation"), o.get("packed"))
    primary = fit.designation
    others = sorted(d for d in dmap if d != primary and d is not None)
    if primary not in dmap or not others or any(dmap[d] is None for d in [primary] + others):
        return None, None
    return dmap[primary], " = ".join(dmap[d] for d in others)


_TIER_RANK = {None: 0, "?": 1, "": 2, "!": 3}          # severity order for picking the worst pair


def _implied_h_pvalue(a, b, min_sd=0.0):
    """Two-sided significance that two implied-H samples (a, b) come from different absolute
    magnitudes, with the per-object systematic floor folded in. Returns (dH, p): dH is the
    |median| difference (robust to junk mags) and p = P(|Z| > dH/se) under a normal, with
    se = sqrt(SEM_a^2 + SEM_b^2 + 2*MAG_SYS_FLOOR^2). Both samples need >= 2 points.

    ``min_sd`` (default 0.0, i.e. OFF -- the designation and apparition callers are unchanged)
    floors each sample's per-observation scatter before it becomes a standard error. It exists
    for :func:`tracklet_mag_audit`, whose samples are routinely ONE magnitude: a survey reports
    a brightness on one observation of a tracklet and leaves the rest blank (2014 WP143's 246
    tracklet is 1 magnitude in 9 observations; 2014 KH169's W84 nights are 1 in 4), so a sample
    variance is either undefined or measured from two numbers that came off the same frame.
    Flooring it at a real single-frame photometric error is what makes an n=1 sample usable at
    all, and it is the same argument as MAG_SYS_FLOOR one level down."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    dH = abs(float(np.median(a)) - float(np.median(b)))
    sa = max(float(np.std(a, ddof=1)) if a.size > 1 else 0.0, min_sd)
    sb = max(float(np.std(b, ddof=1)) if b.size > 1 else 0.0, min_sd)
    se = math.sqrt(sa * sa / a.size + sb * sb / b.size + 2.0 * MAG_SYS_FLOOR ** 2)
    z = dH / se if se > 0 else float("inf")
    return dH, math.erfc(z / math.sqrt(2.0))


def _mag_tier(dH, p):
    """The dH x p decision grid -> photometric-flag tier, one of:
        None  clean (no flag)      ''   flag       ('-mag')
        '?'   advisory ('-mag?')   '!'  strong     ('-mag!')

    Raw-difference AND significance gates BOTH apply. A physically trivial gap (dH < MAG_DH_MIN)
    is never flagged however significant tight photometry makes it; an unresolved gap
    (p >= MAG_P_WEAK) is at most advisory. See `_mag_stats` for dH / p."""
    if dH < MAG_DH_MIN:                                # within one body's lightcurve -> never flag
        return None
    if dH < MAG_DH_MID:                                # 0.3 - 0.7 mag: needs full 3-sigma to flag
        if p < MAG_P_STRONG:
            return ""
        return "?" if p < MAG_P_WEAK else None
    if dH < MAG_DH_HIGH:                               # 0.7 - 1.5 mag: 2-sigma flags, else advisory
        return "" if p < MAG_P_WEAK else "?"
    # dH >= 1.5 mag: a size/albedo gap -- strong once 3-sigma, flag at 2-sigma, else still advisory
    if p < MAG_P_STRONG:
        return "!"
    return "" if p < MAG_P_WEAK else "?"


def _mag_stats(fit):
    """Per-designation implied-H summary and the photometric-mismatch verdict for a linked pair,
    as a dict {dH, p, note, tier}, or None if photometry is insufficient (fewer than 2
    designations each carry >= MAG_MIN_N usable implied-H obs).

    Each obs's implied absolute magnitude H_i (photometry.fit_absolute_magnitude, stashed on
    fit._phot['per']) already folds in the band->V shift and the phase/distance reduction, so a
    genuine same-object pair shares ONE H across oppositions while a chimera shows the size/albedo
    gap as a persistent H offset between the arcs.

      dH   -- |median implied-H| difference between the faintest- and brightest-H designations
              (MEDIAN per designation over the INCLUDED obs, so a few junk survey mags can't
              move the point estimate).
      p    -- two-sided significance that the two differ, from z = dH / se with
              se = sqrt(SEM_a^2 + SEM_b^2 + 2*MAG_SYS_FLOOR^2). The MAG_SYS_FLOOR term is an
              irreducible per-object reduction systematic so a survey's internally-tight scatter
              can't fake an arbitrarily small uncertainty on its implied H.
      note -- short "desig H~x.x, ..." per-designation summary, faint-to-bright.
      tier -- the _mag_tier verdict on the (dH, p) pair.
    """
    per = (getattr(fit, "_phot", None) or {}).get("per")
    if not per:
        return None
    mask = np.asarray(fit._mask, dtype=bool)
    byd = {}
    for i, o in enumerate(fit._obs):
        h = per[i].get("H_i")
        if h is not None and mask[i]:
            byd.setdefault(o.get("designation"), []).append(h)
    grp = {d: np.asarray(v, float) for d, v in byd.items() if len(v) >= MAG_MIN_N}
    if len(grp) < 2:
        return None
    meds = {d: float(np.median(v)) for d, v in grp.items()}
    hi = max(meds, key=meds.get)                       # faintest (largest H)
    lo = min(meds, key=meds.get)                       # brightest (smallest H)
    dH, p = _implied_h_pvalue(grp[hi], grp[lo])
    note = ", ".join(f"{d} H~{m:.1f}" for d, m in sorted(meds.items(), key=lambda kv: kv[1]))
    return {"dH": dH, "p": p, "note": note, "tier": _mag_tier(dH, p)}


def opposition_mag_audit(fit):
    """Per-APPARITION implied-H audit for a SINGLE object (`--mag-audit`): split the INCLUDED obs
    into apparitions (gaps > OPP_GAP_DAYS) and compare each apparition's implied absolute
    magnitude. One body shares ONE H across its oppositions, so a significant brightness change
    between apparitions is worth a look -- a bad-photometry apparition, an unrecognised contaminant
    tracklet folded into the arc, or a genuinely interesting light-curve/activity change. Uses the
    SAME dH x p grid (`_mag_tier`) and per-object systematic floor as the cross-designation flag,
    only comparing apparitions of one designation instead of two designations.

    Returns None when fewer than 2 apparitions each carry >= MAG_MIN_N usable implied-H obs, else:
        {apps: [{jd0, jd1, n, H, sem}, ...],           # per apparition, time-ordered
         worst: {i, j, dH, p, tier},                   # most significant apparition PAIR
         tier}                                         # worst['tier'] (None if all clean)
    """
    per = (getattr(fit, "_phot", None) or {}).get("per")
    if not per:
        return None
    mask = np.asarray(fit._mask, dtype=bool)
    obs = fit._obs
    idx = [i for i in range(len(obs)) if i < len(mask) and mask[i] and per[i].get("H_i") is not None]
    if len(idx) < 2 * MAG_MIN_N:
        return None
    times = [obs[i]["jd_utc"] for i in idx]            # obs are time-sorted upstream
    hs = [per[i]["H_i"] for i in idx]
    apps = []
    for g in _split(times, OPP_GAP_DAYS):              # index groups INTO the idx/times/hs lists
        if len(g) < MAG_MIN_N:
            continue
        h = np.asarray([hs[k] for k in g], float)
        apps.append({"jd0": times[g[0]], "jd1": times[g[-1]], "n": len(h),
                     "H": float(np.median(h)), "sem": float(h.std(ddof=1) / math.sqrt(len(h))),
                     "_h": h})
    if len(apps) < 2:
        return None
    worst = None
    for i in range(len(apps)):
        for j in range(i + 1, len(apps)):
            dH, p = _implied_h_pvalue(apps[i]["_h"], apps[j]["_h"])
            tier = _mag_tier(dH, p)
            cand = {"i": i, "j": j, "dH": dH, "p": p, "tier": tier}
            if (worst is None or _TIER_RANK[tier] > _TIER_RANK[worst["tier"]]
                    or (_TIER_RANK[tier] == _TIER_RANK[worst["tier"]] and p < worst["p"])):
                worst = cand
    for a in apps:
        a.pop("_h", None)
    return {"apps": apps, "worst": worst, "tier": worst["tier"]}


#: A tracklet needs this many usable implied-H observations to be measured.  ONE, where
#: :data:`MAG_MIN_N` asks a designation for three -- and that is not a relaxation, it is
#: what the archive actually contains.  Surveys report a magnitude on ONE observation of
#: a tracklet and leave the rest blank: 2014 WP143's 246 tracklet carries 1 magnitude in
#: 9 observations, 2014 KH169's two W84 nights 1 in 4 each.  Requiring two silently
#: excluded the very tracklets this audit exists to judge -- WP143's 3.6 mag offender
#: dropped out of its own audit and the object read CLEAN -- which is the worst failure
#: a check of this kind can have, because it is invisible.  What makes n=1 legitimate is
#: that the uncertainty is then carried entirely by TMAG_OBS_SD and MAG_SYS_FLOOR rather
#: than by a sample variance measured from two numbers off the same frame.
TMAG_MIN_N = 1

#: ...and the REST of the arc needs this many, pooled.  One tracklet against one other
#: tracklet is not an audit -- there is no way to tell which of the two is the odd one,
#: and the pair would flag each other symmetrically.
TMAG_MIN_REST = 2

#: Fewer tracklets than this and there is no audit to run.  With TWO, each is "the rest"
#: of the other, so they flag each other with the identical dH and p and nothing in the
#: numbers says which one is the intruder -- a symmetric verdict dressed as a finding.
#: Three is the smallest arc where a majority can be the reference.
TMAG_MIN_TRACKLETS = 3

#: Floor on a tracklet's per-observation photometric scatter, in magnitudes, passed to
#: :func:`_implied_h_pvalue` as ``min_sd``.  A survey's single-frame photometry at the
#: detection limit is good to a few tenths and no better; without this floor a tracklet
#: whose two magnitudes happen to agree to 0.01 would claim a standard error of 0.01 and
#: make any difference look infinitely significant.  (ADES ``rmsMag`` is carried on the
#: observations for a minority of rows and would be the better number where it exists --
#: worth using once its coverage on this population is measured.)
TMAG_OBS_SD = 0.3


def tracklet_mag_audit(fit):
    """Per-TRACKLET implied-H audit: one tracklet against the pooled rest of the arc.

    :func:`opposition_mag_audit` asks the same question between OPPOSITIONS, and on a
    single-opposition object it returns None -- there is only one apparition to compare.
    That is precisely the population where a mislinked tracklet hides: three or four
    tracklets over a few weeks, one of them separated from the others by a gap, joined
    on an orbit that no amount of astrometry can test (three tracklets contribute six
    coordinates and an orbit costs six parameters, so the fit threads them exactly --
    see the dof note in :mod:`itf.night_coherence`).  A body has ONE absolute magnitude, and
    that constraint does not care how many degrees of freedom the astrometry has left.

    Measured examples (all still single-opposition catalogue objects): 2014 WP143's 246
    tracklet implies H 14.75 against 18.37/18.68 for the two F51 tracklets it is linked
    to -- a factor-30 size gap, not a lightcurve; 2015 FH479 1.7 mag; 2014 KH169 1.6;
    2022 HK25 1.05.  The known-good control (multi-opposition objects truncated to their
    discovery apparition) puts the worst innocent tracklet at a median dH of 0.31 and a
    p90 of 0.81, so the separation is real but the overlap at ~1 mag is not empty.

    ONE-VS-REST, not all-pairs, and deliberately so.  A mislinkage is one foreign
    tracklet among several of one body, so the rest of the arc is a legitimate pooled
    reference; comparing every pair instead would dilute a real offender against the
    partner it is being compared to, and on a 3-tracklet arc the worst PAIR is almost
    always two innocent tracklets at opposite ends of the same lightcurve.

    Returns None when fewer than :data:`TMAG_MIN_TRACKLETS` tracklets can be measured,
    else::

        {trk: [{i, jd0, jd1, n, code, H, sem, dH, p, tier}, ...],   # time-ordered
         worst: {i, dH, p, tier},                                   # the odd one out
         tier}                                                      # worst['tier']

    ``dH``/``p``/``tier`` on each entry are that tracklet against the pooled OTHERS, so
    they are directly comparable across the arc and the largest is the suspect.  Uses
    the same :func:`_implied_h_pvalue` significance and :func:`_mag_tier` grid as both
    siblings, so a '!' means the same thing on all three surfaces.  INCLUDED obs only.
    """
    per = (getattr(fit, "_phot", None) or {}).get("per")
    if not per:
        return None
    mask = np.asarray(fit._mask, dtype=bool)
    obs = fit._obs
    idx = [i for i in range(len(obs))
           if i < len(mask) and mask[i] and i < len(per) and per[i].get("H_i") is not None]
    if len(idx) < TMAG_MIN_N + TMAG_MIN_REST:
        return None
    times = [obs[i]["jd_utc"] for i in idx]             # obs are time-sorted upstream
    codes = [obs[i].get("obscode") for i in idx]
    desigs = [obs[i].get("designation") for i in idx]
    hs = np.asarray([per[i]["H_i"] for i in idx], float)
    groups = [g for g in _tracklets(times, codes, desigs) if len(g) >= TMAG_MIN_N]
    if len(groups) < TMAG_MIN_TRACKLETS:
        return None
    trk, pool = [], []
    for i, g in enumerate(groups):
        h = hs[g]
        trk.append({"i": i, "jd0": times[g[0]], "jd1": times[g[-1]], "n": len(g),
                    "code": codes[g[0]], "H": float(np.median(h)),
                    "sem": float(max(h.std(ddof=1), TMAG_OBS_SD) / math.sqrt(len(h)))
                    if len(g) > 1 else TMAG_OBS_SD})
        pool.append(h)
    worst = None
    for i, ent in enumerate(trk):
        rest = np.concatenate([p for j, p in enumerate(pool) if j != i])
        if rest.size < TMAG_MIN_REST:
            ent["dH"] = ent["p"] = ent["tier"] = None
            continue
        dH, p = _implied_h_pvalue(pool[i], rest, min_sd=TMAG_OBS_SD)
        ent["dH"], ent["p"], ent["tier"] = dH, p, _mag_tier(dH, p)
        cand = {"i": i, "dH": dH, "p": p, "tier": ent["tier"]}
        if (worst is None or _TIER_RANK[ent["tier"]] > _TIER_RANK[worst["tier"]]
                or (_TIER_RANK[ent["tier"]] == _TIER_RANK[worst["tier"]] and p < worst["p"])):
            worst = cand
    if worst is None:
        return None
    return {"trk": trk, "worst": worst, "tier": worst["tier"]}


def unclipped_anchor_arc(fit, excluded):
    """Longest KEPT arc (days) among the designations that clipped no whole night of their own -- the
    anchor a clip rule must respect. 0.0 when every designation clipped a night (then there is no
    anchor, whatever the arcs look like).

    'Clip a night' is judged on the designation's OWN record: its obs grouped into nights, any night
    wholly excluded. A partially-dropped night is an isolated outlier, not a clipped tracklet, and the
    designation still counts as unclipped (its curvature survives).  `excluded` is the dedup-aware drop
    mask (a cross-listed duplicate is not a drop)."""
    obs = fit._obs
    by = {}
    for i, o in enumerate(obs):
        by.setdefault(o.get("designation"), []).append(i)
    best = 0.0
    for idx in by.values():                       # obs are time-sorted upstream, so idx is too
        times = [obs[i]["jd_utc"] for i in idx]
        nights = _split(times, NIGHT_GAP_DAYS)
        if any(all(excluded[idx[k]] for k in ni) for ni in nights):
            continue                              # this side gave up a whole night -> not an anchor
        kept = [t for i, t in zip(idx, times) if not excluded[i]]
        if kept:
            best = max(best, max(kept) - min(kept))
    return best


def clip_doubt(fit):
    """Why a TWO-opposition link that only closes by clipping a whole night is not believable --
    a short reason string, or None when no such doubt applies. Not an advisory: a link with clipped
    doubt is not an identification at all (identify() declines) AND not a candidate worth a human
    look (examine_or_no() returns 'no') -- two short arcs one orbit threads only after a night is
    discarded are two different objects.

    Two short arcs a few oppositions apart can almost always be threaded by ONE orbit if you are
    allowed to clip a night, so the clip is the whole question. Neither night count nor arc length
    separates the confirmed 2015 OF201 = 2025 NA385 from the coincidental 2015 XO232 = 2025 PD58
    (both a 2-night ~1 d arc + a 4-night ~12 d arc, one night dropped). What separates them:

      * nothing is SECURE. The identity stands on its own -- no drop needed to believe it -- when
        both oppositions independently keep >= SECURE_MIN_KEPT_NIGHTS nights, or one keeps an arc
        longer than SECURE_MIN_ARC_DAYS. Then a clipped night is not load-bearing and there is no
        doubt to report;
      * ...and the clipped night is not obviously a stray. A night clipped at a MODEST residual
        (< COINCIDENCE_HARD_DROP_ARCSEC -- XO232's 4.5", 2014 EO135's 2.5") plausibly belongs to
        its own coherent same-survey sequence: discarding it to force the join is the tell. A GROSS
        stray (OF201's 23") plainly is not this object, so clipping it is legitimate...
      * ...unless clipping it left NO opposition with even a multi-day baseline
        (COINCIDENCE_MIN_KEPT_ARC_DAYS). Then both sides are ~single-tracklet arcs one orbit
        threads by coincidence, and the "stray" was really the object's OWN longer record
        (2025 QL180's 11 d arc collapsing to 0.9 d; 2003 QC91's 58 d arc collapsing to 1.0 d);
      * a not-secure link whose drops COLLAPSE an opposition (to a lone kept night, or clip as many
        nights as they keep) is doubted for the same reason -- nothing on that side confirms it.

    Two exemptions, both deliberate. A 3+-opposition link is exempt by construction: however many
    nights it clips, an orbit that threads three or more apparitions coherently is worth
    investigating (identify() applies its own night gates there; examine_or_no() routes it to review,
    not 'no'). And a link is exempt when the side that clipped NOTHING keeps an arc of at least
    UNCLIPPED_ANCHOR_ARC_DAYS -- an untouched 35+ day baseline anchors the orbit, so the clip on the
    other side is no longer what makes the join close."""
    obs = fit._obs
    R = getattr(fit, "_resid_all", None)         # no residuals -> nothing to judge the clip by
    if R is None:
        return None
    times = [o["jd_utc"] for o in obs]
    dropped = ~np.asarray(fit._mask, dtype=bool)
    excluded = dropped & ~_duplicate_excluded(          # a cross-listed duplicate is not a drop
        times, [o["obscode"] for o in obs], [o.get("ra_deg") for o in obs],
        [o.get("dec_deg") for o in obs], dropped)
    apps = _split(times, OPP_GAP_DAYS)
    app_nights = [_split([times[i] for i in app], NIGHT_GAP_DAYS) for app in apps]
    if len(apps) != 2 or not all(len(ns) >= 2 for ns in app_nights):
        return None                              # 3+ oppositions (or a single-night side) -> exempt
    if unclipped_anchor_arc(fit, excluded) >= UNCLIPPED_ANCHOR_ARC_DAYS:
        return None                              # an untouched 35+ d side anchors the orbit -> exempt
    kept_per_app, longest_kept_arc = [], 0.0
    for app, nights in zip(apps, app_nights):
        kept_t = [times[app[k]] for ni in nights for k in ni if not excluded[app[k]]]
        kept_per_app.append(sum(1 for ni in nights if any(not excluded[app[k]] for k in ni)))
        if kept_t:
            longest_kept_arc = max(longest_kept_arc, max(kept_t) - min(kept_t))
    if (min(kept_per_app) >= SECURE_MIN_KEPT_NIGHTS
            or longest_kept_arc > SECURE_MIN_ARC_DAYS):
        return None                              # secure on its own -- the clip is not load-bearing
    clipped_any = False
    for app, nights in zip(apps, app_nights):
        n_clipped = sum(1 for ni in nights if all(excluded[app[k]] for k in ni))
        if n_clipped and (len(nights) - n_clipped < 2 or n_clipped >= len(nights) - n_clipped):
            return (f"an opposition keeps only {len(nights) - n_clipped} of its {len(nights)} nights "
                    f"and nothing else anchors the orbit")
        for ni in nights:
            if all(excluded[app[k]] for k in ni):          # a whole night clipped from a multi-night opp
                clipped_any = True
                night_min = min(math.hypot(R[app[k], 0], R[app[k], 1]) for k in ni)
                if night_min < COINCIDENCE_HARD_DROP_ARCSEC:
                    return (f"the join only closes by clipping a whole night that sits just "
                            f"{night_min:.1f}\" off it -- part of its own coherent short arc, not a "
                            f"stray, and neither arc is constrained enough to override that")
    if clipped_any and longest_kept_arc < COINCIDENCE_MIN_KEPT_ARC_DAYS:
        return (f"clipping a night leaves no opposition with more than a {longest_kept_arc:.1f} d "
                f"baseline -- two ~single-tracklet arcs one orbit threads by coincidence")
    return None


def _except_label(obs, tr):
    """Human label for one EXCEPTED tracklet: how many obs, which night, which station -- what a
    reviewer needs to find it in the record and decide whether it really is not this object."""
    from pyroorb.egress import _jd_to_ymd            # the engine's one calendar conversion
    o = obs[tr[0]]
    yr, mo, day = _jd_to_ymd(float(o["jd_utc"]))
    desig = o.get("designation") or "?"
    return (f"{len(tr)}-obs {yr:04d}-{mo:02d}-{int(day):02d} {o.get('obscode', '???')} "
            f"tracklet of {desig}")


def exception_shape(fit, _why=None):
    """The EXCEPTION SHAPE of a link that :func:`identify` accepts: ``{"n_except_obs",
    "n_except_tracklets", "tracklets", "renderable"}``, or None when it is not an identification.

    Pass ``_why`` (a dict) to learn WHICH gate said no: on a None return it carries
    ``{"reason": ...}``, the plain-words defect :func:`identify` refused on. A caller that tells a
    human "not a clean identification" needs that sentence -- :func:`linkage_verdict` used to fill
    the gap with :func:`examine_or_no`'s reason for keeping the link a CANDIDATE, and so announced
    a doubt and then printed an exculpation.

    Split out of :func:`identify` because "is this a y2, and which tracklets does it except?" is a
    question about the LINK, while identify()'s None also covers a purely presentational failure:
    an exception is an MPC-facing claim echoed as the verbatim 80-column record, so an ADES-sourced
    fit (the web UI's, which carries no `obs80_line`) fails to RENDER an identification it fully
    earned. Judging the web verdict on identify() alone therefore reported real y2 links as
    ambiguous 'y-examine' candidates and quoted their retention as the reason -- a number that had
    passed (2018 VL78 = 2026 OF5). ``renderable`` is False in exactly that case.

    A THIRD answer, ``{"unknown": True, "error": ...}``, when the measurement itself failed.  It is
    kept separate from None on purpose: None means "measured, and it is not an identification",
    which is a FINDING, and folding a crash into it makes the engine state a conclusion it never
    reached.  Worse, it states the PERMISSIVE one -- :func:`linkage_verdict` reads not-an-
    identification as "no exceptions to worry about" and so as submittable, which is how a y2 (not
    transmissible, Jira route) could come back as a y-examine with a live Submit button.  Before
    this function existed :func:`linkage_verdict` called ``identify`` bare and a failure propagated
    to the web layer's own guard as "linkage check failed" -- loud, and closed.  The split restores
    that property without giving up the ADES fix above: an unknown shape is still OFFERED for
    review (we cannot rule the identification out either), it simply may not be transmitted."""
    try:
        return identify(fit, _shape=True, _why=_why)
    except Exception as ex:               # noqa: BLE001 -- see above: not a finding
        return {"unknown": True, "error": f"{type(ex).__name__}: {ex}"}


def identify(fit, _shape=False, _why=None):
    """Return the ident-check identification text for a fitted pair, or None when it is not a
    clean identification.

    With ``_shape=True`` the EXCEPTION SHAPE is returned instead of the text (see
    :func:`exception_shape`, the public entry): the same gates, the same exception set, but
    reported as counts rather than rendered as an MPC record -- so a caller that only needs to
    know "is this a y2, and of what" is not defeated by an ADES fit having no verbatim OBS80 line
    to echo. Returns None in both modes when the link is not an identification at all.

    The drops are judged at TRACKLET (single-night) granularity, since a mislink shows up
    as one or two stray tracklets, not a fraction of an apparition:
      * a WHOLE night dropped is a stray tracklet -- fine (<= MAX_DROP_TRACKLETS of them,
        with NO cap on the total obs so dropped);
      * a MINORITY of a night dropped is an isolated outlier -- fine (<= max(MAX_DROP_SCATTER,
        SCATTER_FRAC of the total obs), counted only OUTSIDE the wholly-dropped tracklets);
      * MOST-but-not-all of a night dropped is fishy -> reject;
      * a multi-night OPPOSITION that has a whole night dropped is two arcs disagreeing -> reject
        when EXACTLY two oppositions are both multi-night and the drops collapse one to a lone
        kept night or clip as many nights as they keep (the chimera ban -- nothing else anchors
        the orbit), or when the opposition loses the MAJORITY of its nights (a lone stray night
        in a 3+ opposition link is fine -- the clean oppositions still pin it);
      * both designations must keep some obs (else 'A = B' drops one side entirely);
      * a TWO-opposition link must retain >= MPC_MIN_RETENTION of EACH designation's own obs and
        must carry no :func:`clip_doubt`: a join bought by clipping a night from a coherent short arc
        with nothing securing it is not an identification at all. Both are waived for a 3+-opposition
        link (the clean oppositions anchor the orbit) and for one whose UNCLIPPED side keeps a
        >= UNCLIPPED_ANCHOR_ARC_DAYS arc (that side anchors it by itself);
      * the kept obs must fit (:func:`kept_obs_fit`: under MAX_KEPT_RMS in arcsec, or -- when the
        object's own astrometry cannot reach that -- within the uncertainties it claims).

    The `except:` list is stricter than the drop GATES above: it is an MPC-facing claim that a
    tracklet does NOT belong to the object, so it echoes only a WHOLLY-dropped tracklet whose
    every obs sits > EXCEPT_MIN_RESID_ARCSEC off the joint orbit. A partially-dropped tracklet
    (isolated outlier) or a near-threshold whole tracklet stays excluded from the FIT but is not
    an exception -- bad observations, not a bad tracklet."""
    def _no(reason):
        """Refuse, and SAY why. The refusal is a finding in its own right: 'not a clean
        identification' is what the front ends put in front of a human, and a caution that names
        no defect is a caution nobody can act on (the web UI used to print the reason
        :func:`examine_or_no` gives for keeping the link a CANDIDATE -- an exculpation -- under a
        heading announcing a doubt). ``_why`` is an out-dict, so the return contract is untouched:
        every caller still reads None as 'not an identification'."""
        if _why is not None:
            _why["reason"] = reason
        return None

    obs = fit._obs
    if fit.n_observations() < 6:
        return _no(f"only {fit.n_observations()} observations survive the fit -- too few to "
                   f"determine an orbit, let alone confirm an identification with it")
    _fits, _fitnote = kept_obs_fit(fit)
    if not _fits:
        return _no(f"the kept observations do not fit the joint orbit: {_fitnote}")
    mask = np.asarray(fit._mask, dtype=bool)
    times = [o["jd_utc"] for o in obs]           # obs are time-sorted upstream
    codes = [o["obscode"] for o in obs]
    ras = [o.get("ra_deg") for o in obs]
    decs = [o.get("dec_deg") for o in obs]
    desigs = [o.get("designation") for o in obs]
    # Ignore benign DUPLICATES: an excluded obs that is the same detection as a kept obs (same
    # station, same instant, SAME sky position) cross-listed under another designation --
    # de-duplication, not a drop. Position must match: same station + same time is NOT enough
    # (a survey exposure can carry two different objects ~arcmin apart at one instant).
    excluded = (~mask) & ~_duplicate_excluded(times, codes, ras, decs, ~mask)

    # both designations must still contribute -- 'A = B' is meaningless if one is fully dropped
    _gone = ({o["designation"] for o in obs}
             - {obs[i]["designation"] for i in range(len(obs)) if mask[i]})
    if _gone:
        return _no(f"the joint fit drops every observation of {', '.join(sorted(_gone))} -- "
                   f"nothing is left of that side to identify")

    apps = _split(times, OPP_GAP_DAYS)
    app_nights = [_split([times[i] for i in app], NIGHT_GAP_DAYS) for app in apps]

    def _secure_without(tr):
        """How many OTHER nights of this tracklet's own apparition still keep observations."""
        for app, nights in zip(apps, app_nights):
            if tr[0] in app:
                mine = next(ni for ni in nights if app.index(tr[0]) in ni)
                return sum(1 for ni in nights
                           if ni is not mine and any(not excluded[app[k]] for k in ni))
        return 0

    interleaved = _interleaved(obs, ~excluded)
    rsig = _resid_sigmas(fit)

    # classify drops per TRACKLET (one observatory, one night, one submitted designation)
    n_tracklets = n_scatter = 0
    for tr in _tracklets(times, codes, desigs):
        ne = sum(1 for i in tr if excluded[i])
        if ne == 0:
            continue
        if ne == len(tr):
            n_tracklets += 1                     # whole tracklet gone -> a stray
        elif ne <= max(1, len(tr) // 3):
            # a minority of the tracklet -> outlier(s). In an INTERLEAVED link a MARGINAL one
            # (under MARGINAL_REJECT_SIGMA) is rejection noise, not the price of the join: the
            # other arc checks the orbit at the same epochs (_interleaved). The cap below exists
            # for drops that bought a join, and a marginal drop in an interleaved link did not.
            # A wrong track still shows: SA12 moved 1" off 2026 SW11 drops four obs at 5.6-6.3
            # sigma; the real pair's seven all sit at 2.7-3.6.
            n_scatter += sum(1 for i in tr if excluded[i]
                             and not (interleaved and rsig[i] < MARGINAL_REJECT_SIGMA))
        elif _secure_without(tr) >= SECURE_MIN_KEPT_NIGHTS or (
                interleaved and all(rsig[i] < MARGINAL_REJECT_SIGMA for i in tr if excluded[i])):
            # Most of a tracklet gone, but it is not LOAD-BEARING: its own apparition keeps at
            # least SECURE_MIN_KEPT_NIGHTS other nights, so the join does not close BECAUSE of the
            # clipped -- the same "secure on its own" test clip_doubt applies to a whole night. The
            # within-night proportion alone is blind to the arc around it: 2012 GC2 = CERV9W2 was
            # refused because a 4-obs A48 night lost 3, out of 109 obs over 8 nights (0.47").
            # Counted as a stray, so the tracklet budget and every later gate still apply, and it
            # is never ECHOED as an exception (only a wholly-dropped tracklet is).
            # Nor is it load-bearing when the designations' arcs INTERLEAVE (_interleaved) and
            # every obs it lost is MARGINAL: the other arc tests the orbit at the same epochs, and
            # only the strictest rejection level removed them (2026 SW11's 703 pair, which its
            # own solo fit rejects too). A tracklet clipped by more than that is a track the joint
            # orbit does not follow -- SA12 moved 1" off loses 7 of its 8-obs I41 night.
            n_tracklets += 1
        else:
            # most-but-not-all of a tracklet, in an apparition too thin to carry it -> fishy
            return _no(f"the {_except_label(obs, tr)} loses {ne} of its {len(tr)} observations -- "
                       f"most of a night, but not all of it, in an apparition with too few other "
                       f"nights to show the join does not depend on it")
    if n_tracklets > MAX_DROP_TRACKLETS:
        return _no(f"the joint fit clips {n_tracklets} whole tracklets (at most "
                   f"{MAX_DROP_TRACKLETS} may be clipped)")
    _scatter_max = max(MAX_DROP_SCATTER, int(SCATTER_FRAC * len(obs)))
    if interleaved:
        # Marginal drops were left out of n_scatter (see the loop), so bound the TOTAL trimmed by
        # the chimera tell itself: each designation keeps MPC_MIN_RETENTION of its own record.
        for _d in sorted({o["designation"] for o in obs}):
            _mine = [i for i in range(len(obs)) if obs[i]["designation"] == _d]
            _kept = sum(1 for i in _mine if not excluded[i])
            if _kept / len(_mine) < MPC_MIN_RETENTION:
                return _no(f"the joint fit keeps only {_kept} of {_d}'s {len(_mine)} "
                           f"observations -- under the {MPC_MIN_RETENTION * 100:.0f}% an "
                           f"identification must retain")
    if n_scatter > _scatter_max:
        return _no(f"the joint fit drops {n_scatter} scattered observations (at most "
                   f"{_scatter_max} may be dropped on an arc this size)")

    # A multi-night opposition that has a WHOLE night dropped is suspect -- two well-determined
    # arcs disagreeing, not the same object. Judged by NIGHTS, not raw obs (partially-dropped
    # nights are scatter, bounded above, and count as KEPT). How much dropping is tolerated turns
    # on the link's shape:
    #   * EXACTLY two oppositions, BOTH multi-night -> reject when the drops COLLAPSE an
    #     opposition to a lone kept night, or clip as many nights as they keep (no strict
    #     majority). With no third opposition to anchor the orbit, a side reduced to a single
    #     tracklet no longer independently confirms anything -- and every documented chimera has
    #     exactly that shape (2015 HC310 = 2026 EM61, 2014 PN3 = 2019 QH83, 2013 TX15 = 2026 AE24:
    #     a 2-night opp clips 1; 2015 RC182 = 2026 ES49: a 3-night opp clips 2). But an opposition
    #     that stays SOLIDLY multi-night after clipping a verified-bad stray is still an
    #     independent multi-night arc, and the link is real (a 'y2'): 2015 OF201 = 2025 NA385
    #     drops 1 of 3 nights, 2014 HH101 = 2022 DQ27 drops 1 of 5;
    #   * otherwise (3+ oppositions, so the CLEAN ones still pin the orbit) -> reject only if an
    #     opposition loses the MAJORITY of its nights (2005 JK163 = 2022 MT4 = 2026 HV6 keeps it:
    #     one stray 2022 night, anchored by clean 2005 & 2026).
    # A whole single-night opposition dropped is exempt either way -- it is itself a stray
    # tracklet, and a one-night opposition isn't "multi-night" anyway.
    two_multinight = len(apps) == 2 and all(len(ns) >= 2 for ns in app_nights)
    for app, nights in zip(apps, app_nights):
        if len(nights) == 1:                                  # single-night opp -> stray, exempt
            continue
        dropped = sum(1 for ni in nights if all(excluded[app[k]] for k in ni))
        kept_nights = len(nights) - dropped
        if not dropped:
            continue
        if two_multinight:
            if kept_nights < 2 or dropped >= kept_nights:
                return _no(f"one of the two oppositions loses {dropped} of its {len(nights)} "
                           f"nights, leaving {kept_nights} -- with only two apparitions, a side "
                           f"cut back that far no longer independently confirms the identity")
        elif dropped > kept_nights:
            return _no(f"one opposition loses {dropped} of its {len(nights)} nights -- the "
                       f"majority of that apparition does not fit the joint orbit")

    # A TWO-opposition link must also RETAIN the bulk of each designation's own record (the same
    # MPC_MIN_RETENTION bar examine_or_no calls a clear 'no'). With no third opposition to anchor
    # the orbit, a join bought by discarding a third of one side's obs is the join being forced:
    # the clipped nights are not contaminants, they are the object's own record, and a coherent survey
    # sequence essentially never contains a night that isn't the object. (K25Y17S = K16FL3H: the
    # 2025 side's 23 d arc made the link look 'secure', but 2016 FH213 kept only 8/13 -- its third
    # W84 night clipped at 9.5". K26EA0R = K25D43S: 2025 DS43's coherent 7-night/25 d O18 campaign
    # loses its first THREE nights, 8/14 kept.) A 3+-opposition link is exempt: the clean
    # oppositions still pin the orbit, so a gutted apparition there is worth a human look. So is a
    # link whose UNCLIPPED side keeps a >= UNCLIPPED_ANCHOR_ARC_DAYS arc -- a side that gave up not one
    # night and still spans 35+ days anchors the orbit itself, and the link keeps whatever it would
    # otherwise be (a clean y/y2, or the 'y-systematic' its own residuals earn it).
    if len(apps) == 2 and unclipped_anchor_arc(fit, excluded) < UNCLIPPED_ANCHOR_ARC_DAYS:
        n_by_desig, kept_by_desig = {}, {}
        for i, o in enumerate(obs):
            d = o.get("designation")
            n_by_desig[d] = n_by_desig.get(d, 0) + 1
            if not excluded[i]:
                kept_by_desig[d] = kept_by_desig.get(d, 0) + 1
        _worst = min(n_by_desig, key=lambda d: kept_by_desig.get(d, 0) / n_by_desig[d])
        if kept_by_desig.get(_worst, 0) / n_by_desig[_worst] < MPC_MIN_RETENTION:
            return _no(f"{_worst} keeps only {kept_by_desig.get(_worst, 0)} of its "
                       f"{n_by_desig[_worst]} observations "
                       f"({100.0 * kept_by_desig.get(_worst, 0) / n_by_desig[_worst]:.0f}%, "
                       f"under the {MPC_MIN_RETENTION * 100:.0f}% an identification must retain) "
                       f"-- and with only two oppositions nothing else anchors the orbit")

    # COINCIDENCE gate (a real-doubt DECLINE): a 2-opposition link that survives the collapse ban
    # can still be a coincidence when the ONLY way to thread the joint orbit is to clip a whole
    # night out of an otherwise-coherent short arc and nothing independently secures the identity.
    # The rule lives in clip_doubt() because examine_or_no() needs the SAME judgement to route such
    # a link to a flat 'no' rather than to human review.
    R = fit._resid_all
    _clip = clip_doubt(fit)
    if _clip:
        return _no(_clip)

    left, right = _packed_pair(fit)
    if left is None:
        return _no("the link has no pair of packed MPC designations to name -- an identification "
                   "is a statement about two catalogued ids")
    header = f"{left} = {right}"
    # The `except:` echo is an MPC-facing claim that a TRACKLET does not belong: list only a
    # WHOLLY-dropped tracklet whose every obs sits > EXCEPT_MIN_RESID_ARCSEC off the joint orbit.
    # A partially-dropped tracklet is an isolated astrometric outlier (the tracklet itself
    # belongs: 2021 WL5's W05 night keeps 2 of 3), and a whole tracklet clipped at ~1" is plausible
    # astrometry the fit merely disfavors -- both stay excluded from the FIT but are NOT
    # exceptions. (2025 NM2 = 2021 WL5 lists exactly its junk ~76" 126 tracklet, matching the
    # MPC-accepted record; the I52/W05 sub-1.25" outliers are bad obs, not bad tracklets.)
    echo = np.zeros(len(obs), dtype=bool)
    excepted = []                                 # the excepted tracklets, as index lists
    for tr in _tracklets(times, codes, desigs):
        if all(excluded[i] for i in tr) and (R is None or all(
                math.hypot(R[i, 0], R[i, 1]) > EXCEPT_MIN_RESID_ARCSEC for i in tr)):
            excepted.append(list(tr))
            for i in tr:
                echo[i] = True
    if _shape:
        # Everything above passed: this IS an identification. Report WHAT it costs -- the
        # exception set -- and stop before rendering, which is the only step that needs a verbatim
        # OBS80 line and so the only step an ADES-sourced fit cannot complete.
        return {"n_except_obs": int(echo.sum()), "n_except_tracklets": len(excepted),
                "tracklets": [_except_label(obs, tr) for tr in excepted],
                "renderable": all(obs[i].get("obs80_line") for i in range(len(obs)) if echo[i])}
    if not echo.any():
        body = header
    else:
        lines = [header + " except:"]
        for i, o in enumerate(obs):
            if echo[i]:
                raw = o.get("obs80_line")
                if raw is None:
                    return None                  # no verbatim line to echo (e.g. ADES source)
                lines.append(raw)                 # verbatim 80-col record (already col-1 aligned)
        body = "\n".join(lines)

    # Photometric sanity FLAG (advisory, NEVER a rejection): when the linked arcs' implied absolute
    # magnitudes disagree on the dH x p grid (`_mag_stats`) the brightness doesn't agree across
    # oppositions -- could be a mislink, could be junk survey photometry -- so we still emit the
    # identification but prepend a '# FLAG' comment marking it for a closer look before submitting to
    # the MPC. The comment carries no ' = ', so downstream line counters (which key on ' = ') ignore it.
    st = _mag_stats(fit)
    if st and st["tier"] is not None:
        label = {"?": "possible mag residual", "": "mag residual",
                 "!": "LARGE mag residual"}[st["tier"]]
        body = (f"# FLAG {label} {st['dH']:.2f} mag between oppositions "
                f"({st['note']}; p={st['p']:.1g}) -- examine before submitting\n" + body)

    # SHORT-LINK FLAG (advisory, NEVER a rejection): an identification carried by fewer than
    # THIN_LINK_MIN_OBS observations. Every other gate here judges what the fit DROPPED, so a tiny
    # link that drops nothing passes them all while resting on almost no evidence -- flag it in the
    # written record the same way, so a human weighs it before submitting.
    thin = thin_link(fit)
    if thin:
        body = f"# FLAG short link: {thin} -- examine before submitting\n" + body
    return body


MPC_MIN_RETENTION = 0.65   # a candidate is worth a human look ('y-examine') only if the joint fit
                           # retains at least this fraction of EACH designation's obs. A genuine
                           # A = B threads a common orbit through a strong majority of both arcs;
                           # a chimera can only be linked by discarding the bulk of one side, and
                           # because rejection HIDES the misfit from the joint rms (the fit drops
                           # the non-threading obs until what survives looks clean), retention --
                           # not rms -- is the decisive tell. Below this, it is a clear 'no'.
                           # (2016 UA210 = 2014 HM410 keeps 24/24 UA210 but only 10/24 HM410 = 42%.)


def examine_or_no(fit):
    """Classify a NON-clean ident fit (:func:`identify` returned None) as an AMBIGUOUS link worth a
    human look ('y-examine') or a clear non-match ('no'). Returns (code, reason).

    Mind the POLARITY of ``reason``: on a 'no' it is the DEFECT, on a 'y-examine' it is the
    grounds for NOT calling the link a non-match -- an exculpation ("a baseline that long pins
    the orbit", "both arcs retain >=97% of their obs").  A caller putting it in front of a human
    must label which of the two it is holding, and must not present it as the reason the link is
    unclean: that reason belongs to :func:`identify`, which reports it through
    :func:`exception_shape`'s ``_why``.  Pure -- no I/O --
    so the batch prints it, the web UI gates its submit button on it, and tests assert it. Decision,
    in order:

      * only ONE designation contributed obs -> nothing was joined -> 'no' (the lone arc fits itself
        trivially; 2010 OA81 = 2015 MR211, where OA81 is WISE-only space obs we can't yet read);
      * per-designation RETENTION below MPC_MIN_RETENTION -> 'no'. A chimera links only by
        DISCARDING the bulk of one side, and rejection HIDES the misfit in dropped obs rather than
        rms, so we gate on the fraction of EACH designation's obs that survive (2016 UA210 =
        2014 HM410: keeps 24/24 but only 10/24 -> no);
      * the kept obs don't fit (:func:`kept_obs_fit`) -> 'no'. Reported SEPARATELY from retention:
        the two used to share one message that named retention, so a pair refused on its residuals
        was reported as a retention failure it had actually passed;
      * CLIP DOUBT (:func:`clip_doubt`) -> 'no'. A 2-opposition link that closes ONLY by clipping a
        whole night out of a coherent short arc, with neither side constrained enough to make the
        clip non-load-bearing, is two different objects -- not a candidate. The long baseline between
        the apparitions does NOT rescue it: two short arcs years apart are exactly what one orbit can
        thread once it may discard a night (2015 XO232 = 2025 PD58, 2014 EO135 = 2015 HT386,
        2025 QL180 = 2014 HA394, 2014 EN102 = 2022 TB28). Exempt: 3+ oppositions, or an UNCLIPPED side
        keeping a >= UNCLIPPED_ANCHOR_ARC_DAYS arc -- those keep whatever they would otherwise be;
      * SHORT+SHORT that only links by dropping obs -> 'no' ONLY when both arcs sit in the SAME
        apparition (total span <= OPP_GAP_DAYS): with no baseline, zapping a day to make the join fit
        proves they are different objects -- unless the arcs INTERLEAVE in time and every drop is
        MARGINAL (:func:`_interleaved`, MARGINAL_REJECT_SIGMA), where nothing was extrapolated and
        nothing bought the join -> 'y-examine'. When the arcs are in DIFFERENT apparitions and no whole
        night was clipped (the partial-drop case clip_doubt does not cover) the long baseline genuinely
        pins the orbit -> route to 'y-examine' for review. NOT auto-confirmed either way;
      * otherwise (both arcs retain the majority and fit sub-arcsec) -> 'y-examine'."""
    kept, n_by_desig, span = {}, {}, {}
    fmask = np.asarray(fit._mask, dtype=bool)
    for i, o in enumerate(fit._obs):
        d = o.get("designation"); t = o["jd_utc"]
        n_by_desig[d] = n_by_desig.get(d, 0) + 1
        if fmask[i]:
            kept[d] = kept.get(d, 0) + 1
        lo, hi = span.get(d, (t, t)); span[d] = (min(lo, t), max(hi, t))
    if len(n_by_desig) < 2:
        return "no", "only one designation contributed obs -- nothing joined"
    min_retention = min(kept.get(d, 0) / n_by_desig[d] for d in n_by_desig)
    if min_retention < MPC_MIN_RETENTION:
        return "no", f"min retention {min_retention*100:.0f}%"
    # The two clauses are reported SEPARATELY: they were one `not (a and b)` whose single message
    # named retention, so a pair refused for its RESIDUALS was reported as a retention failure
    # (243P + 2025 MT356 printed "min retention 89%" -- 89% passes; the 0.647" rms was the refusal).
    fits, note = kept_obs_fit(fit)
    if not fits:
        return "no", note
    doubt = clip_doubt(fit)
    if doubt:
        return "no", doubt
    arcs = sorted((hi - lo) for lo, hi in span.values())
    total_span = max(hi for _, hi in span.values()) - min(lo for lo, _ in span.values())
    short_short = len(arcs) >= 2 and arcs[-1] <= 15.0 and arcs[0] <= 10.0
    if short_short and min_retention < 1.0:
        n_all = sum(n_by_desig.values())
        n_dropped = n_all - sum(kept.values())
        _rsig = _resid_sigmas(fit)
        marginal = all(_rsig[i] < MARGINAL_REJECT_SIGMA for i in np.flatnonzero(~fmask))
        if total_span <= OPP_GAP_DAYS and not (marginal and _interleaved(fit._obs, fmask)):
            return "no", (f"two short arcs ({arcs[0]:.1f} d / {arcs[-1]:.1f} d) in one apparition "
                          f"link only by dropping observations ({n_dropped} of {n_all}) -- with no "
                          f"baseline between them, that is not the same object")
        if total_span <= OPP_GAP_DAYS:
            # Same apparition but INTERLEAVED (_interleaved) and every drop MARGINAL: no baseline,
            # and none needed -- the joint orbit is checked against both arcs at the same epochs,
            # and only the strictest rejection level trimmed anything, so the drops did not buy
            # the join. A drop past MARGINAL_REJECT_SIGMA keeps the flat 'no' above.
            return "y-examine", (f"the arcs are brief ({arcs[0]:.1f} d / {arcs[-1]:.1f} d) and "
                                 f"the joint fit drops {n_dropped} of {n_all} observations, all "
                                 f"marginal, and they interleave in time -- each is checked "
                                 f"against the other at the same epochs, not extrapolated to")
        # SAY WHAT WAS DROPPED. This clause fires on ANY drop at all, so on an ordinary arc it
        # can be four 5-sigma outliers out of 125 -- and a reader told only "short arcs" cannot
        # tell that from a join bought by clipping half a night.
        return "y-examine", (f"the arcs are brief ({arcs[0]:.1f} d / {arcs[-1]:.1f} d) and the "
                             f"joint fit drops {n_dropped} of {n_all} observations, but they sit "
                             f"{total_span/365.25:.1f} yr apart -- a baseline that long pins the "
                             f"orbit")
    # Carry the relaxation note into the reason a human reads: an rms over the absolute bar is the
    # kind of thing a reviewer must be told was excused, and on what grounds.
    ok = f"both arcs retain >={min_retention*100:.0f}% of their obs"
    return "y-examine", f"{ok}; {note}" if note else ok


def linkage_verdict(fit):
    """May the joint fit on screen be OFFERED to a human as an MPC identification, and with what
    cautions?  ``{"offer", "clean", "reason", "cautions"}``.

    This is the one rule both front ends ask.  ``--ident-check`` runs it over candidate pairs in
    bulk and prints the verdict; the web UI asks it of the fit the user assembled by typing two
    designations into the search box, and shows "Submit linkage" only when it says yes.  Same
    thresholds, same wording, one place to change them -- the alternative is a UI that offers
    submissions the batch would have thrown out.

    ``code`` mirrors what ``--ident-check`` prints ('y' / 'y2' / 'y-examine' / 'no'), so the two
    front ends cannot classify the same fit differently: a caution here corresponds to a suffix or
    a non-'y' code there, and only a bare 'y' comes back clean with an empty caution list.

    ``offer`` is deliberately not "confirmed", and it is a separate question from ``submittable``:

      * 'y'  -- a CLEAN identification: no caution, and the only thing that comes back clean;
      * 'y2' -- the join requires EXCEPTING whole tracklets. Offered for review with the exception
        named (disowning a night somebody reported is a claim a human makes, not the engine), but
        NOT submittable: the document has no way to say "except these observations", so it goes to
        the MPC as a Jira ticket;
      * 'y-systematic' -- accepted, but the kept residuals TREND. Offered for review, NOT
        submittable, matching the batch, which does not write one either. The trend can have an
        innocent cause and a real identification can show it (2025 OT173 = 2014 EN155), so the fit
        stays on screen -- but the engine does not put its name to it (2014 HV328 = 2025 OV408, a
        known-bogus link, used to read "a clean identification" over a live Submit button);
      * 'y-examine' -- an ambiguous-but-real candidate, offered WITH its reason as a caution;
      * 'no' -- not offered at all. The fit may be perfectly good as an ORBIT while the
        identification it implies is exactly the kind the batch files as a non-match.

    The remaining advisory flags (photometric dH, short link) ride along as cautions: they never
    veto an identification, they make a human look twice. A systematic trend between two SHORT arcs
    (:func:`short_arc_systematic`) is a flat refusal, there being no arc on either side against
    which a human could weigh it.

    ONE designation requested is never an identification, however many designations the fit
    carries.  Type ``1999 CN154`` alone and the MPC hands back 2014 EM304's observations with it
    -- not a discovery, an identification it acknowledged years ago, which is precisely WHY both
    arcs arrive from one query.  Resubmitting an acknowledged DES=DES is rejected (and the same
    rule already restricts the ITF search's document to the family PRIMARY --
    ``itf.linkage.build_linkage(primary_desigs=...)``), so there is nothing here to offer: no
    button, no banner.  A fit whose requested count is unknown (a paste, a fixture) is left as it
    was -- there the designations came from the user's own astrometry."""
    if getattr(fit, "n_requested", None) == 1:
        return {"offer": False, "clean": False, "code": "no", "submittable": False,
                "reason": "one designation requested -- its other designations came back from "
                          "that single fetch, so the MPC has already acknowledged this "
                          "identification and it must not be resubmitted",
                "cautions": []}
    # A trend across two short arcs is not a caution to be weighed, it is a non-match: neither
    # side constrains the orbit, so nothing here could be called a fit (short_arc_systematic).
    # Checked BEFORE cleanliness, because such a link can otherwise pass identify() outright.
    fatal_sys = short_arc_systematic(fit)
    if fatal_sys:
        return {"offer": False, "clean": False, "code": "no", "submittable": False,
                "reason": fatal_sys, "cautions": []}
    # An identification is judged by its SHAPE, not by whether the record could be rendered:
    # identify() also returns None when the link is fine but carries no verbatim OBS80 line to
    # echo its exceptions with, which is every ADES fetch -- i.e. every fit the web UI makes.
    why = {}
    shape = exception_shape(fit, _why=why)
    # ...and a shape we could not MEASURE is neither of those. It is offered (nothing here rules
    # the identification out, and hiding it would lose a real one to a transient fault) but never
    # transmitted, because the question the exception shape answers -- does this join rest on
    # disowning whole tracklets? -- is exactly the question that decides whether it may be sent.
    unknown = bool(shape) and bool(shape.get("unknown"))
    is_ident = shape is not None and not unknown
    n_except = shape["n_except_obs"] if is_ident else 0
    # 'y' vs 'y2', the batch's own split: at most ONE excepted observation is a clean match; more
    # than that, the join is bought by declaring whole tracklets not to be this object, and that
    # claim is the thing a human has to agree with before anything is submitted.
    y2 = is_ident and n_except > EXCEPT_CLEAN_MAX_OBS
    clean = is_ident and not y2
    submittable = not y2 and not unknown   # ...until a residual trend withdraws it below
    code_override = None
    code, reason = examine_or_no(fit)
    cautions = []
    if unknown:
        # Offered, never sent. The reason states its own case (so the web layer adds no
        # "identification candidate:" framing) and NAMES the fault, because the person reading it
        # is the only one who can pursue the identification by hand or report the bug.
        code_override = "unknown"
        cautions.append(f"The engine could not judge this identification: {shape['error']}. "
                        f"Everything below is the ordinary joint fit and is unaffected -- it is "
                        f"only the identification VERDICT that is missing.")
        if code != "y-examine":
            # ...and whatever examine_or_no did manage to say is worth keeping: it is measured on
            # retention and rms, which the shape failure did not touch.
            cautions.append(f"Independently of that fault, this link would not have been "
                            f"confirmed on its own: {reason}")
        reason = ("NOT submittable -- the engine could not determine whether this identification "
                  "depends on excepting whole tracklets, and that question is exactly what "
                  "decides whether an identification may be transmitted at all. Shown for review; "
                  "if you judge it real, take it to the MPC by hand.")
    elif y2:
        # ONE message, not a headline plus a caution repeating it: the exception IS the whole
        # story here, and it ends in an instruction (a Jira ticket) that must not be split off
        # into a second banner the reader might not connect to the first.
        trks = shape["tracklets"]
        what = " and ".join(trks) if trks else f"{n_except} observations"
        is_are, that = ("are", "those tracklets") if len(trks) != 1 else ("is", "that tracklet")
        reason = (f"an identification WITH EXCEPTIONS -- the join holds only if the {what} {is_are} "
                  f"not this object. Check {that} in the record before claiming anything. "
                  f"NOT submittable here: an identification document has no way to say 'except "
                  f"these observations', so an identification with exceptions goes to the MPC "
                  f"as a Jira ticket.")
    elif not is_ident:
        if code != "y-examine":
            return {"offer": False, "clean": False, "code": "no", "submittable": False,
                    "reason": reason, "cautions": []}
        # The DOUBT and the EXCULPATION are two different sentences and must be labelled as
        # such.  This caution used to read "Not a clean identification -- an ambiguous link
        # worth a human look: <examine_or_no's reason>", and examine_or_no's 'y-examine'
        # reasons are the grounds for NOT calling the link a non-match ("a long baseline pins
        # the orbit", "both arcs retain >=97% of their obs") -- reassurance, spliced in after a
        # colon that promised a defect.  So the banner announced a doubt it never named and
        # then argued the other way, on a fit whose real refusal (the ident gates' own, now
        # carried out of identify() in `_why`) went unmentioned.
        defect = why.get("reason")
        cautions.append(
            (f"Offered for review, not confirmed -- the identification gates refused it: "
             f"{defect}. What keeps it a candidate rather than a non-match: {reason}. "
             if defect else
             f"Offered for review, not confirmed. What keeps it a candidate rather than a "
             f"non-match: {reason}. ")
            + "The joint orbit fits, but the engine does not put its name to this "
              "identification.")
    thin = thin_link(fit)
    if thin:
        cautions.append(f"Short link: {thin}. A handful of observations spread over two "
                        f"designations barely over-determines a 6-parameter orbit, so a wrong "
                        f"pairing threads it almost as easily as the right one.")
    # SYSTEMATIC residual trend. On an accepted identification this is not an advisory -- it is the
    # batch's 'y-systematic', which is NOT written and NOT auto-confirmed, so the UI must not paint
    # it as clean or offer to transmit it either (2014 HV328 = 2025 OV408, a known-bogus link,
    # showed "a clean identification" over a live Submit button). It stays OFFERED, because the
    # fit and its residuals are exactly what a reviewer needs to see, and a real identification
    # can trend (2025 OT173 = 2014 EN155) -- but the engine will not back it.
    sysmis = systematic_misfit(fit)
    if sysmis and is_ident:
        if y2:                                    # both faults: the trend leads, the exception
            cautions.append(reason)               # stays as its own caution
        code_override = "y-systematic"
        clean = False
        submittable = False
        reason = (f"NOT a clean identification -- the kept observations fit at an acceptable "
                  f"overall rms but their residuals TREND: {sysmis}. Nights sitting coherently OFF "
                  f"the orbit rather than scattered around it is the tell of a wrong linkage that "
                  f"a clean rms hides. NOT submittable here: the engine does not confirm an "
                  f"identification whose own residuals trend -- review it, and take it to the MPC "
                  f"yourself if you are satisfied it is real.")
    elif sysmis:
        cautions.append(f"Systematic residual pattern: {sysmis}. Nights sitting coherently OFF the "
                        f"orbit (rather than scattered around it) is the tell of a wrong linkage "
                        f"that an acceptable overall rms hides.")
    st = _mag_stats(fit)
    if st and st["tier"] is not None:
        label = {"?": "possible", "": "significant", "!": "LARGE"}[st["tier"]]
        cautions.append(f"Photometric mismatch ({label}): implied H differs by {st['dH']:.2f} mag "
                        f"between apparitions ({st['note']}; p={st['p']:.1g}). One body has one "
                        f"absolute magnitude. This could represent a mislinkage (not the same "
                        f"object), could be a photometry issue, an H-G magnitude shortcoming, or "
                        f"(rarely) activity inflating magnitude. Research further and use good "
                        f"judgement before submitting.")
    return {"offer": True, "clean": clean,
            "code": code_override or ("y" if clean else ("y2" if y2 else code)),
            # ...offered for REVIEW, but neither a y2 nor a 'y-systematic' is transmissible: the
            # document cannot carry the exception a y2 depends on (sending it would claim the
            # excepted tracklet AS this object), and a trending link is one the batch refuses to
            # write at all. Both stay on screen with their reason; neither raises a button.
            "submittable": submittable,
            "reason": "clean identification" if clean else reason,
            "cautions": cautions}


def systematic_misfit(fit):
    """Detect a SYSTEMATIC within-apparition residual pattern in an otherwise clean-rms fit -- the
    tell of a wrong linkage (or a contaminated apparition) that the joint fit absorbs into an
    acceptable OVERALL rms by leaving each night coherently OFF the orbit rather than scattered
    around it. Returns a short description when found (so the batch can downgrade 'y'/'y2' to
    'y-systematic' and NOT auto-confirm the ID), else None.

    A correct orbit fits every night to ~measurement noise, so per-night MEAN residuals are small
    and comparable to the within-night scatter. A mislink can't trace the true curvature, so an
    apparition's per-night means TREND across the arc while each night stays internally tight. Per
    apparition (and axis) we therefore require the RMS of per-night means (between-night) to be
    large in absolute terms (>= SYS_BETWEEN_ARCSEC) AND to dominate the noise -- where "dominates"
    is satisfied EITHER way: >= SYS_WITHIN_RATIO x the pooled within-night scatter (the original
    test), OR >= SYS_SE_RATIO x the standard ERROR of those means. The two are OR-ed, so this can
    only ever add a flag, never remove one. The second test exists because the first scales wrongly
    with night length -- a well-sampled night averages its scatter down, so a hugely significant
    per-night offset scores as "noise" (see SYS_SE_RATIO). Examples: 2014 HV328 = 2025 OV408 -- the
    2014 nights trend +1.1 -> -1.0" in Dec while each night is tight to ~0.05" (scatter test);
    2016 UJ137 -- 2 nights +-0.53" apart in RA, only 2.19x the 0.24" scatter but 3.7x the mean's
    own error (significance test). A third arm (:func:`significant_trend`, SYS_TREND_Z) needs no arcsec
    floor at all: nightly means that WALK in a straight line at >= 5x the slope's own error, by
    >= 8x the scatter within a night, are a trend however small in arcsec (2025 NF505 = 2015 HG272,
    1.26" over four nights at z 14, 11x). Judged on
    INCLUDED obs only (rejected blunders are already gone)."""
    if fit._resid_all is None:
        return None
    m = np.asarray(fit._mask, dtype=bool)
    idx = [i for i in range(len(fit._obs)) if m[i]]
    if len(idx) < 6:
        return None
    idx.sort(key=lambda i: fit._obs[i]["jd_utc"])     # _obs is time-sorted, but be explicit
    times = [fit._obs[i]["jd_utc"] for i in idx]
    R = fit._resid_all
    worst = None                                      # best axis clearing the STRICT absolute floor
    corrob = []                                       # per-apparition best axis clearing the RELAXED floor
    for app in _split(times, OPP_GAP_DAYS):
        nights = [ni for ni in _split([times[k] for k in app], NIGHT_GAP_DAYS)
                  if len(ni) >= SYS_MIN_NIGHT_OBS]
        if len(nights) < SYS_MIN_NIGHTS:
            continue                                  # too sparse to tell a trend from noise
        app_best = None                               # this apparition's strongest ratio-dominant trend
        for ax in (0, 1):                             # 0 = dRA*cosδ, 1 = dDec
            vals = [np.array([R[idx[app[k]], ax] for k in ni]) for ni in nights]
            means = np.array([v.mean() for v in vals])
            within = float(np.mean([v.std() for v in vals]))
            # Each night's mean carries its OWN uncertainty, SE = sd/sqrt(n) -- the scale the
            # significance test measures against. (SYS_MIN_NIGHT_OBS keeps every night at n >= 2,
            # so ddof=1 is always defined; a 1-obs night would be uninformative, hence the nan.)
            ses = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
                            for v in vals])
            between = float(np.sqrt(np.mean(means ** 2)))
            if len(means) >= SYS_LOO_MIN_NIGHTS:      # robust: the signal must survive dropping
                drop = int(np.argmax(np.abs(means)))                    # the single worst night
                between = float(np.sqrt(np.mean(np.delete(means, drop) ** 2)))
                ses = np.delete(ses, drop)            # ...dropped from the noise scale too, so
                                                      # signal and noise describe the same nights
            se = float(np.sqrt(np.nanmean(ses ** 2))) if not np.all(np.isnan(ses)) else 0.0
            # Structure, not noise -- EITHER test suffices (OR-ed, so this only ever adds flags).
            scatter_dominant = within < 1e-6 or between / within >= SYS_WITHIN_RATIO
            signif_dominant = se > 1e-9 and between / se >= SYS_SE_RATIO
            if not (scatter_dominant or signif_dominant):
                continue
            axis = "RA" if ax == 0 else "Dec"
            # Written for the banner and the CLI line: WHICH apparition, which axis, and the
            # two scales side by side. The old "RA per-night mean 0.42\" vs within 0.18\"" had no
            # verb and named no apparition, and it was the harshest-coloured banner in the UI.
            desc = (between,
                    f"in the {_jd_year(times[app[0]])} apparition the nightly {axis} means "
                    f"drift by {between:.2f}\" while each night scatters only {within:.2f}\""
                    + (f", {between / se:.1f}x the mean's own error" if se > 1e-9 else ""))
            if between >= SYS_BETWEEN_ARCSEC and (worst is None or between > worst[0]):
                worst = desc                          # strict fire (absolute floor UNCHANGED)
            if between >= SYS_BETWEEN_CORROB and (app_best is None or between > app_best[0]):
                app_best = desc
        if app_best is not None:
            corrob.append(app_best)
    if worst is not None:
        return worst[1]
    # Corroborated relaxation: no single apparition reached the strict 0.5" floor, but if two or
    # more apparitions EACH independently trend (ratio-dominant, >= the relaxed floor) that repeated
    # signature is itself the evidence -- report the strongest such apparition.
    if len(corrob) >= SYS_MIN_CORROB_APPS:
        best = max(corrob, key=lambda t: t[0])
        return f"{best[1]} (corroborated across {len(corrob)} apparitions)"
    trend = significant_trend(fit)
    if trend is not None and trend["fires"]:
        return (f"in the {trend['year']} apparition the nightly {trend['axis']} means walk "
                f"{trend['swing']:.2f}\" across {trend['nights']} nights in a straight line -- "
                f"{trend['ratio']:.0f}x the {trend['within']:.2f}\" scatter within a night, and "
                f"{trend['z']:.1f}x the slope's own error against the stations' night-to-night "
                f"noise")
    return None


def significant_trend(fit):
    """The strongest within-apparition TREND in the station-night mean residuals, as
    ``{"z", "ratio", "fires", "axis", "year", "swing", "within", "nights"}``, or None when no
    apparition has SYS_TREND_MIN_BLOCKS station-nights of SYS_MIN_NIGHT_OBS+ included obs.

    ``z`` is |slope| / sigma of a weighted line through the nights' means (see SYS_TREND_Z for
    the noise model and the calibration): whether the nights WALK, not how far -- a displaced
    but flat apparition scores ~0. ``ratio`` is how far: the swing across the apparition over the
    pooled within-night scatter (SYS_TREND_MIN_SWING_RATIO). ``fires`` = both bars cleared; the
    test returned is the strongest (by z) that fires, else the strongest overall."""
    from pyroorb import nightmodel
    if fit._resid_all is None:
        return None
    m = np.asarray(fit._mask, dtype=bool)
    idx = sorted((i for i in range(len(fit._obs)) if m[i]), key=lambda i: fit._obs[i]["jd_utc"])
    if len(idx) < 6:
        return None
    R = fit._resid_all
    t = np.array([fit._obs[i]["jd_utc"] for i in idx])
    codes = [str(fit._obs[i].get("obscode") or "500") for i in idx]
    best = None
    for app in _split(list(t), OPP_GAP_DAYS):
        app = np.asarray(app)
        gid = nightmodel.station_night_index(t[app], [codes[k] for k in app])
        blocks = [app[gid == g] for g in np.unique(gid)]
        blocks = [b for b in blocks if len(b) >= SYS_MIN_NIGHT_OBS]
        if len(blocks) < SYS_TREND_MIN_BLOCKS:
            continue
        tb = np.array([t[b].mean() for b in blocks])
        nb = np.array([len(b) for b in blocks], float)
        for ax in (0, 1):
            vals = [np.array([R[idx[k], ax] for k in b]) for b in blocks]
            mb = np.array([v.mean() for v in vals])
            dof_w = sum(len(v) - 1 for v in vals)
            s2 = sum(((v - v.mean()) ** 2).sum() for v in vals) / dof_w
            tau = np.array([nightmodel.station_params(codes[b[0]])[2 + ax] for b in blocks])
            w = 1.0 / np.maximum(s2 / nb + tau ** 2, 1e-8)
            dt = tb - (w * tb).sum() / w.sum()
            sxx = (w * dt * dt).sum()
            if sxx <= 0:
                continue
            mbar = (w * mb).sum() / w.sum()
            slope = (w * dt * (mb - mbar)).sum() / sxx
            dof = len(blocks) - 2
            chi2 = float((w * (mb - mbar - slope * dt) ** 2).sum())
            infl = max(1.0, chi2 / dof) if dof > 0 else 1.0
            z = abs(slope) * np.sqrt(sxx / infl)
            swing = float(abs(slope) * (tb.max() - tb.min()))
            within = float(np.sqrt(s2))
            ratio = swing / max(within, 1e-3)
            cand = {"z": float(z), "ratio": ratio,
                    "fires": bool(z >= SYS_TREND_Z and ratio >= SYS_TREND_MIN_SWING_RATIO),
                    "axis": "RA" if ax == 0 else "Dec", "year": _jd_year(t[app[0]]),
                    "nights": len(blocks), "swing": swing, "within": within}
            if best is None or (cand["fires"], cand["z"]) > (best["fires"], best["z"]):
                best = cand
    return best


def short_arc_systematic(fit):
    """A fired :func:`systematic_misfit` on a link where BOTH designations are SHORT arcs
    (<= SYS_SHORT_ARC_DAYS each) -- returns a short reason, else None.

    Where a trend on a well-observed link is a caution ('y-systematic': it may be unmodelled
    nongravs, a bad night, survey systematics, and the rest of the arc lets a human judge), a trend
    across two day-long arcs is decisive. Neither side carries curvature of its own, so there is
    nothing to weigh the systematic against; a nongrav big enough to walk the residuals would have
    been auto-detected; and the joint orbit is so thinly over-determined that a coherent per-night
    offset just means it does not thread both arcs. There is no reading of it that is a fit, so the
    caller files it as a plain non-match rather than sending it to review.

    Arc lengths are each designation's OWN span over ALL its obs (dropping obs cannot make an arc
    'short'). Returns None when the fit carries fewer than two designations."""
    sysdesc = systematic_misfit(fit)
    if not sysdesc:
        return None
    span = {}
    for o in fit._obs:
        d = o.get("designation")
        t = float(o["jd_utc"])
        lo, hi = span.get(d, (t, t))
        span[d] = (min(lo, t), max(hi, t))
    if len(span) < 2:
        return None
    arcs = sorted(hi - lo for lo, hi in span.values())
    if arcs[-1] > SYS_SHORT_ARC_DAYS:                 # the LONGEST arc decides: one real baseline
        return None                                   # anywhere in the link is enough to judge by
    spans = " / ".join(f"{a:.1f} d" for a in arcs)
    return (f"systematic residual across short arcs ({spans}, all <= {SYS_SHORT_ARC_DAYS:.0f} d): "
            f"{sysdesc} -- no arc here constrains the orbit well enough for this to be anything "
            f"but a non-match")


def coherent_offset_misfit(fit, noise_floor):
    """Detect a single NIGHT/tracklet whose residuals are COHERENTLY offset from
    zero, measured against ``noise_floor`` (arcsec) -- the residual scale the
    orbit achieved BEFORE this obs set was in the fit. Per included night and
    axis (n >= COH_MIN_OBS): the night's |mean| must exceed COH_RATIO x
    noise_floor AND dominate the within-night scatter (>= COH_MEAN_FLOOR x
    sd/sqrt(n)). Used only for ITF vetting, where ``noise_floor`` is the seed's
    pre-search rms -- so a genuinely-noisy old fit does not fire (its floor is
    already large), while a link that displaces tight nights does. Returns the
    strongest offending night's description, else None. INCLUDED obs only."""
    if fit._resid_all is None:
        return None
    floor = max(float(noise_floor), 0.03)
    m = np.asarray(fit._mask, dtype=bool)
    idx = [i for i in range(len(fit._obs)) if m[i]]
    if len(idx) < COH_MIN_OBS:
        return None
    idx.sort(key=lambda i: fit._obs[i]["jd_utc"])
    times = [fit._obs[i]["jd_utc"] for i in idx]
    R = fit._resid_all
    worst = None
    for ni in _split(times, NIGHT_GAP_DAYS):
        if len(ni) < COH_MIN_OBS:
            continue
        n = len(ni)
        from datetime import datetime, timedelta
        d = (datetime(1970, 1, 1)
             + timedelta(days=times[ni[0]] - 2440587.5)).strftime("%Y-%m-%d")
        for ax in (0, 1):
            vals = np.array([R[idx[k], ax] for k in ni])
            mean = float(np.mean(vals))
            ratio = abs(mean) / floor
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            dominance = abs(mean) / (sd / np.sqrt(n) + 1e-9)
            if ratio >= COH_RATIO and dominance >= COH_MEAN_FLOOR:
                axis = "RA" if ax == 0 else "Dec"
                if worst is None or ratio > worst[0]:
                    worst = (ratio,
                             f"the {d} tracklet sits {mean:+.2f}\" off in {axis} "
                             f"({ratio:.0f}x the pre-link residual level, n={n}) "
                             f"-- a coherent offset, not noise")
    return worst[1] if worst else None


def thin_arc_falsepos(fit):
    """Advisory FALSE-POSITIVE flag: an emitted identification whose match rests on ONE side
    collapsing to a single sub-day tracklet. Returns a short description, or None.

    A genuine 'A = B' threads one orbit through a real arc on BOTH designations. This fires when
    the joint fit keeps only a < FALSEPOS_ARC_DAYS (24 h) sub-arc of some designation WHILE that
    designation was actually observed over a longer arc that got dropped (its dropped obs extend
    it >= FALSEPOS_COLLAPSE_DAYS past the kept window). Dropping that side's own extra nights
    throws away the very intra-arc curvature that would TEST the link, so the "identification" is
    really the OTHER arc fitting itself with a loosely-attached tracklet -- more likely a false
    positive than a real y/y2. NEVER a rejection: the ID is still emitted, only flagged for review
    (the batch suffixes the classify code with '-falsepos?'; the written record gets a '# FLAG'
    comment). An inherently single-night tracklet linked to a known object -- the common LEGITIMATE
    identification -- keeps its whole arc, so kept span ~= full span and it is NOT flagged.

    Judged on INCLUDED obs only. Returns None unless the pair actually forms an identification."""
    if identify(fit) is None:
        return None                                  # not an emitted ID -> nothing to flag
    mask = np.asarray(fit._mask, dtype=bool)
    obs = fit._obs
    by = {}
    for i, o in enumerate(obs):
        by.setdefault(o.get("designation"), []).append(i)
    if len(by) < 2:
        return None
    for d, idx in by.items():
        kept = [i for i in idx if mask[i]]
        if len(kept) < 2:
            continue
        kt = [obs[i]["jd_utc"] for i in kept]
        kept_span = max(kt) - min(kt)
        if kept_span >= FALSEPOS_ARC_DAYS:
            continue                                 # this side keeps a real (multi-hour+) arc
        # the full span is judged over the kept obs plus the PLAUSIBLE drops only -- a dropped
        # blunder (>= FALSEPOS_BLUNDER_ARCSEC off the orbit) never testified about the arc's
        # curvature, so clipping it doesn't collapse anything
        R = fit._resid_all
        informative = [i for i in idx if mask[i] or R is None
                       or math.hypot(R[i, 0], R[i, 1]) < FALSEPOS_BLUNDER_ARCSEC]
        allt = [obs[i]["jd_utc"] for i in informative]
        full_span = max(allt) - min(allt)
        if full_span - kept_span >= FALSEPOS_COLLAPSE_DAYS:
            return (f"{d} collapses to a {kept_span * 24:.1f} h kept sub-arc of its own "
                    f"{full_span:.1f} d / {len(informative)}-obs arc ({len(kept)} kept) -- the "
                    f"link rests on a single sub-day sub-arc, not an independently-confirmed arc")
    return None
