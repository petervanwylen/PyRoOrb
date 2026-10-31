"""Shared two-body / geometry primitives for the seeding and scoring paths
(iod, sr, arc_extend, ident_check).

A dependency *leaf* -- it imports only numpy -- so any of those modules can import
it without risking an import cycle. It centralizes small constants and expressions
that were previously copy-pasted (and in a couple of cases had silently drifted)
across those modules: the apparition/night gap thresholds, the unweighted (O-C)
astrometric residual, and the two-body plausibility penalty.
"""
import numpy as np

# --- gap constants: the time boundaries that define apparitions and nights ------
OPP_GAP_DAYS = 100.0    # a time gap larger than this separates oppositions/apparitions.
                        # Oppositions are >1 yr apart; obs within one apparition span weeks.
                        # The engine's one copy (arc_extend imports it); other opposition rules in the
                        # tree use different gaps and are not yet reconciled.
NIGHT_GAP_DAYS = 0.5    # obs separated by <= this many days are the SAME night.
                        # THE consecutive-gap rule: ident_check's _split, arc_extend's
                        # seed-clean split and excluded-night count, iod's rescue-window
                        # floor, hyperbolic's station-night count.
SINGLE_NIGHT_SPAN_DAYS = 0.5   # a cluster whose first-to-last SPAN is under this is one
                               # night (arc_extend._is_single_night) -- a span test, not
                               # a gap test, so it gets its own name.

# A NIGHT, as minor-planet astrometry means it -- NOT the same rule as
# NIGHT_GAP_DAYS above, which is a gap between CONSECUTIVE observations and so
# never ends a night while observing continues.  Here the first observation
# STARTS the night and opens a 12-hour window; everything inside belongs to it;
# the first observation past the window starts the next night.  An object
# observed around the clock therefore accumulates two nights per day, which is
# the behaviour the consecutive-gap rule cannot produce.
#
# This exists because the alternative that had spread through the codebase --
# ``round(jd - 0.5)`` -- puts the boundary at NOON UT, which falls in the middle
# of the observing night for every Hawaiian and Arizonan station (F51/F52,
# T05/T08/T09, 703/G96): one tracklet straddling it counted as TWO nights.
NIGHT_WINDOW_DAYS = 0.5


def night_index(times, window_days=NIGHT_WINDOW_DAYS):
    """``[night_number, ...]`` aligned to ``sorted(times)``, night 0 first.

    Use this rather than :func:`count_nights` when the nights of a SUBSET
    (say, the rejected observations) must be counted: indexing over the whole
    arc anchors each window on the night's true first observation, whereas
    counting the subset alone would anchor on the first member of the subset.
    """
    ts = sorted(float(t) for t in times)
    out, n, start = [], 0, None
    for t in ts:
        if start is None:
            start = t
        elif t - start >= window_days:
            n += 1
            start = t
        out.append(n)
    return out


def count_nights(times, window_days=NIGHT_WINDOW_DAYS):
    """Distinct observing nights spanned by ``times`` (see NIGHT_WINDOW_DAYS)."""
    ts = sorted(float(t) for t in times)
    if not ts:
        return 0
    n, start = 1, ts[0]
    for t in ts[1:]:
        if t - start >= window_days:
            n += 1
            start = t
    return n


# --- "the same physical detection, seen twice" ----------------------------------
# Three paths have to answer this question and each had its own arithmetic:
# ``obs_sources.find_duplicate`` (drops the extra copy at fetch time),
# ``ident_check._duplicate_excluded`` (classifies a cross-listed drop as benign
# rather than a suspect tracklet) and ``itf.obs80_fields.obs_unique_id`` (column
# identity on the raw 80-column line, exact, no tolerance -- it compares text,
# not sky, so it is NOT expressed here).  The first two are the same test, and
# one of them was missing the circle wrap: a cross-listed pair straddling RA 0h
# read as 360 degrees apart and was silently declined.
#
# What is shared is the GEOMETRY.  The TOLERANCES and the CURRENCY deliberately
# are not, and must stay the caller's to choose:
#
#   * fetch-time dedup measures the OBS80 round trip's truncation of the RA
#     COORDINATE (HH MM SS.dd -> at most ~0.2" of RA-degrees), which is the same
#     size at every declination -- so it compares degrees of RA, un-scaled, and
#     wants a tight ~1 s / 1" window.
#   * the identification check asks whether two records describe one point on
#     the SKY, which is an angle -- so it scales by cos(dec), and wants a looser
#     ~9 s window because it compares across designations whose timestamps have
#     been through different round trips.
#
# Collapsing those two into one number would be a false economy: each is right
# about the thing it measures.  Sharing the arithmetic is what stops a fix to
# one (the wrap) from having to be remembered for the other.


def ra_delta_deg(ra_a, ra_b):
    """Signed RA difference in degrees, taken ON THE CIRCLE (result in [-180, 180)).

    A plain subtraction reads 359.99998 and 0.00001 as 360 degrees apart, so any
    tolerance test built on it silently declines every pair straddling RA 0h."""
    return (float(ra_a) - float(ra_b) + 180.0) % 360.0 - 180.0


def same_detection(jd_a, ra_a, dec_a, jd_b, ra_b, dec_b,
                   *, time_tol_days, pos_tol_deg, great_circle):
    """True when two records are one detection: same instant to ``time_tol_days``
    and same place to ``pos_tol_deg`` on both axes.

    The caller owns "same station" -- the two callers index it differently (one
    buckets by (obscode, int(JD)), the other loops over a kept/excluded split) --
    and owns both tolerances; see the note above on why they differ.

    ``great_circle`` scales the RA offset by cos(dec) so the comparison is a true
    angular separation.  False compares degrees of RA as such, which is the right
    currency when the tolerance stands for a coordinate's rounding rather than a
    distance on the sky."""
    if abs(float(jd_a) - float(jd_b)) >= time_tol_days:
        return False
    d_ra = ra_delta_deg(ra_a, ra_b)
    if great_circle:
        d_ra *= np.cos(np.radians(float(dec_a)))
    return abs(d_ra) < pos_tol_deg and abs(float(dec_a) - float(dec_b)) < pos_tol_deg


# --- ident-hypothesis policy: shared by the ident-mode short-circuit family ------
# (arc_extend._ident_far_bridge_skip, fit_object's ident-doomed-degenerate skip and
# _ident_doomed_degenerate). One constant + one predicate so the four enforcement
# points can't drift; each site documents its OWN q source (they legitimately differ).
IDENT_FAR_MIN_Q = 1.3   # Perihelion cut (au) below which an ident hypothesis keeps FULL diligence.
                        # In an EXPLICIT ident-check hypothesis (n_requested>1 / ev.ident_mode, an
                        # UNCONFIRMED cross-designation link the engine is under no obligation to
                        # force), an anchor with q >= this is a main-belt / outer body: a GENUINE such
                        # identification reaches the far apparition through the cheap in-basin / Tier-1
                        # swarm (every known positive control -- 2001 FS222=2025 NW338,
                        # 2015 KE500=2025 MY29, and every PAIR_* fixture -- links with ZERO Tier-2), so
                        # the expensive Tier-2 multi-revolution bridge (period sweep / a-secant / phase
                        # / LOV) only ever grinds on a NON-match. Re-run from each recovery re-anchor,
                        # that grind is exactly what pushes a doomed pair to the 120 s pair_timeout.
                        # Below this q we are in the NEO / Mars-crosser regime where multi-revolution
                        # linkage genuinely arises (Earth-resonant returns, short arcs many revs apart:
                        # 2024 LW, 2003 LW2), so every short-circuit stays DISARMED -- there we DO bend
                        # over backwards. All uses fire ONLY under an explicit ident request, so a
                        # numbered object's self-link and every non-ident fit (CLI, web UI single
                        # object) keep full diligence and stay byte-identical.
IDENT_BRIDGE_MAX_NIGHTS = 2   # ...EXCEPT an anchor this thin: the "a genuine
                        # main-belt link goes through Tier-1" premise holds between DETERMINED arcs,
                        # and a 2-night anchor's prediction years out is uncertain by tens of degrees
                        # -- 2015 XS276 (2 F51 nights, q 1.7) = NEOCP P12r4oH sits 155 deg off its solo
                        # orbit and links 16/16 at 0.15" only through Tier-2. So arc_extend's far-bridge
                        # skip does not fire on an anchor of <= this many nights at any q. Priced on the
                        # 342-pair NEOCP replay: +1 true
                        # (2023 VN4), +1 wrong pair bridged (2010 VJ13), ~+7% CPU on wrong pairs; at 3
                        # nights +5 wrong for no more true -- the 3-night anchors are where wrong pairs bridge.


def ident_requested(meta):
    """True when the caller EXPLICITLY requested >= 2 designations -- an ident-check hypothesis
    ("is A = B?"), meta["n_requested"] > 1. The single reading of that field: a numbered object
    queried by one name expands to many alias designations for ONE body but n_requested == 1, so
    it is NOT an ident request; when the field is absent (an older fetch path), default to a
    single-object request (every ident gate stays off -- the safe, byte-identical direction)."""
    return (meta or {}).get("n_requested", 1) > 1


def radec_residuals_arcsec(pred, ra_obs, dec_obs, cosd):
    """(O-C) residuals in arcsec for a batch of predicted vs observed (RA, Dec).
    RA is wrapped to (-180, 180] and scaled by cos(dec); returns (dra, ddec) arrays.
    The single definition of an UNWEIGHTED astrometric residual for the seeding /
    scoring paths. (The weighted, sigma-divided LM residual lives separately in
    arc_extend._radec_resid, where it feeds least_squares.)"""
    dra = ((pred[:, 0] - ra_obs + 180) % 360 - 180) * cosd * 3600.0
    ddec = (pred[:, 1] - dec_obs) * 3600.0
    return dra, ddec


ABSURD_A_AU = 1.0e4   # au: no real bound orbit is this large; a > this is a near-parabolic
                      # numerical artifact whose integration makes the adaptive stepper crawl
                      # toward dt->0 (an effective hang), so reject it on the two-body screen.


def unphysical_penalty(a, e, q, *, big=1e6, allow_unbound=False):
    """Two-body plausibility penalty (0.0 if physical). `big` is added, separately, for an
    unbound orbit (a <= 0 or e >= 1), a sun-plunging one (q < 0.02), and an absurd
    semi-major axis (a > ABSURD_A_AU). Lets the IOD scorer, statistical ranging, and the
    extension scorer all reject a doomed candidate on the cheap two-body elements before
    handing it to the integrator.

    `allow_unbound` drops the unbound term only, for a caller that has already shown this arc
    admits no bound orbit on enough nights to believe an unbound one (arc_extend, hyperbolic.py);
    the sun-plunging and absurd-a terms still apply."""
    pen = 0.0
    if (a <= 0 or e >= 1.0) and not allow_unbound:
        pen += big
    if q < 0.02:
        pen += big
    if a > ABSURD_A_AU:
        pen += big
    return pen
