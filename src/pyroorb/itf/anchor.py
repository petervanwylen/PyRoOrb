"""The anchor apparition: an ITF linkage is submitted only when ONE of its
apparitions is well observed in its own right.

This is OUR floor, not the MPC's (:mod:`itf.mpcrules` holds theirs, unchanged):

* every linkage needs an apparition with at least :data:`MIN_TRACKLETS`
  tracklets;
* a non-NEO needs, in addition, an apparition with at least
  :data:`MIN_NIGHTS_NON_NEO` observing nights.

WHY.  An arc made only of thin apparitions -- a night here, two there --
fits almost any orbit family, because no single apparition pins the motion;
each thin piece is then "confirmed" by the others it was chosen to fit.
2013 EZ43 is the shape: one night in 2008, two tracklets on two nights in
2013, one night in 2014, published as one object and a mislinkage.  The
MPC's profile rule (:func:`itf.mpcrules.mpc_opp_profile_ok`) refuses that
shape at three or four apparitions, but passes it at five or more (two
apparitions of >= 2 nights suffice) and asks nothing of a single apparition;
this floor closes both.  A NEO may anchor on tracklets rather than nights
because its motion is fast enough that a few tracklets of one apparition
determine it, and its links are proposals worth sending (see
``itf.search._opp_profile_gate``); a slow object needs the nights.

The units are the MPC's: an apparition ends at a gap of more than
``mpcrules.MPC_OPPOSITION_GAP_DAYS`` (237 d) between observations, a night
is the 12-hour window of ``opp_profile.NIGHT_DAYS``, and a tracklet is one
station's observations within one such night (two stations on one night are
two tracklets; one station's two visits on one night count once -- the
conservative reading).
"""

from __future__ import annotations

from .mpcrules import MPC_OPPOSITION_GAP_DAYS
from .opp_profile import NIGHT_DAYS, obs_night_ordinal

#: Every linkage needs one apparition with at least this many tracklets.
MIN_TRACKLETS = 3

#: A non-NEO needs, in addition, one apparition with at least this many nights.
MIN_NIGHTS_NON_NEO = 3


def is_neo(cfg, q, e):
    """The search's NEO test: perihelion under ``cfg.neo_waiver_q`` AND a bound
    orbit (``e < cfg.neo_waiver_max_e``; a range-degenerate distant arc can
    converge on a hyperbola whose perihelion is an artefact).  A missing or NaN
    element fails the comparison, so an orbit we cannot read is a non-NEO and
    takes the stricter rule."""
    try:
        return bool(float(q) < cfg.neo_waiver_q and float(e) < cfg.neo_waiver_max_e)
    except (TypeError, ValueError):
        return False


def _windows(days, width):
    """Count windows of ``width`` days opened by the first value >= ``width``
    after the current window's start -- the rule of ``opp_profile.spans_of``."""
    n, start = 0, None
    for d in sorted(days):
        if start is None or d - start >= width:
            start, n = d, n + 1
    return n


def apparition_counts(points):
    """``[(tracklets, nights), ...]`` per apparition, in time order, for
    ``points`` = ``[(ordinal_day, station), ...]`` (any order)."""
    pts = sorted((float(d), str(s)) for d, s in points)
    if not pts:
        return []
    groups, cur = [], [pts[0]]
    for prev, p in zip(pts, pts[1:]):
        if p[0] - prev[0] > MPC_OPPOSITION_GAP_DAYS:
            groups.append(cur)
            cur = []
        cur.append(p)
    groups.append(cur)
    out = []
    for g in groups:
        by_stn = {}
        for d, s in g:
            by_stn.setdefault(s, []).append(d)
        tracklets = sum(_windows(ds, NIGHT_DAYS) for ds in by_stn.values())
        out.append((tracklets, _windows([d for d, _ in g], NIGHT_DAYS)))
    return out


def points_of(astrometry):
    """``[(ordinal_day, station), ...]`` from OBS80 text.  The second line of a
    two-line record (satellite / roving: a lower-case column 15) repeats its
    first line's instant and is skipped."""
    out = []
    for line in (astrometry or "").splitlines():
        if len(line) < 80 or not line.strip() or line[14] in "svr":
            continue
        try:
            out.append((obs_night_ordinal(line), line[77:80]))
        except ValueError:
            continue
    return out


def anchor_problem_points(points, *, neo):
    """Why ``points`` (see :func:`apparition_counts`) have no anchor
    apparition, or None when they do."""
    apps = apparition_counts(points)
    best_t = max((t for t, _ in apps), default=0)
    best_n = max((n for _, n in apps), default=0)
    profile = ", ".join(f"{t} tracklet(s)/{n} night(s)" for t, n in apps) or "no observations"
    if best_t < MIN_TRACKLETS:
        return (f"no apparition has {MIN_TRACKLETS}+ tracklets (per apparition: {profile}); "
                f"an arc of thin apparitions does not pin one orbit")
    if not neo and best_n < MIN_NIGHTS_NON_NEO:
        return (f"no apparition has {MIN_NIGHTS_NON_NEO}+ nights, which a non-NEO needs "
                f"(per apparition: {profile})")
    return None


def anchor_problem(astrometry, *, neo):
    """:func:`anchor_problem_points` on OBS80 text."""
    return anchor_problem_points(points_of(astrometry), neo=neo)
