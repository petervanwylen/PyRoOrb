"""The MPC's own acceptance rules for an ITF-to-ITF identification.

Ported from the MPC's published criteria:
https://www.minorplanetcenter.net/mpcops/documentation/identifications/additional/

WHY THIS IS SEPARATE from :func:`itf.search._opp_profile_gate`.  That gate answers a
different question -- "is this linkage strong enough that WE believe it?" -- and
it is deliberately conservative, because a search that accepts a wrong trial
loses the object its chance at the right one.  This module answers "would the
MPC accept the submission?", which is not ours to tighten OR to loosen.  A
linkage can pass here and still be one we decline to send (see the caller's own
gates); it can also fail here while being perfectly good science, in which case
sending it is simply pointless.

**One number differs from the rest of this package and it matters.**  The MPC
defines a new apparition as a gap between subsequent nights of more than **237
days**.  ``itf.opp_profile.opp_profile`` uses 275, and the 1.5/1.75 fractional
convention on top of that.  Those are OUR statements about arc strength; 237 is
THEIRS about bookkeeping, and an arc can be two oppositions by one rule and one
by the other.  This module counts the MPC's way, always, and never imports the
fractional convention.
"""

from __future__ import annotations

from .opp_profile import nights_of, obs_night_ordinal

#: A new apparition begins after a gap longer than this many days (MPC).
MPC_OPPOSITION_GAP_DAYS = 237

#: Fewer distinct nights than this and the submission is refused before any
#: orbit is fitted.
MIN_NIGHTS = 3

#: Shorter total arc than this (days) and it is refused before fitting.
MIN_ARC_DAYS = 3.0

#: A 3-night submission may not span more than this (days).  A 3-night arc
#: stretched over months is a linkage nobody can check.
MAX_ARC_DAYS_3_NIGHTS = 15.0

#: Above this rms (arcsec) the orbit quality test has to carry the submission.
RMS_ARCSEC = 0.25

#: Arc below which the rms/quality test applies, for 3 nights and for more.
QUALITY_ARC_DAYS_3_NIGHTS = 15.0
QUALITY_ARC_DAYS_MORE = 10.0

#: Element-uncertainty ceilings for the 3-night case (a and q in au, i in deg).
MAX_SIGMA_A_AU = 0.05
MAX_SIGMA_Q_AU = 0.05
MAX_SIGMA_I_DEG = 0.5
MAX_SIGMA_E = 0.05

#: And a 3-night orbit may not be more eccentric than this.
MAX_E_3_NIGHTS = 0.5


def night_ordinals(astrometry):
    """One ordinal-day float per observing NIGHT, in time order.

    NOT one per observation: ``obs_night_ordinal`` is a fractional ordinal day,
    so a tracklet's three exposures 20 minutes apart are three distinct values
    of it.  Taking those for "distinct nights" turns a 4-night linkage into a
    13-night one and every rule below reads the wrong number.  Nights are
    grouped by :func:`opp_profile.nights_of` at the CALENDAR gap -- what an
    observer, and the MPC, mean by another night.
    """
    lines = [s for s in (astrometry or "").splitlines() if s.strip()]
    return [min(obs_night_ordinal(l) for l in night)
            for night in nights_of(lines)]


def arc_days(astrometry):
    """First to last OBSERVATION, in days -- not first to last night."""
    lines = [s for s in (astrometry or "").splitlines() if s.strip()]
    if len(lines) < 2:
        return 0.0
    days = [obs_night_ordinal(l) for l in lines]
    return max(days) - min(days)


def oppositions(astrometry):
    """Nights per apparition, MPC-style: a gap > 237 d starts a new one.

    Returns a list of night counts, one per apparition, in time order.
    """
    days = night_ordinals(astrometry)
    if not days:
        return []
    out, n = [], 1
    for prev, cur in zip(days, days[1:]):
        if cur - prev > MPC_OPPOSITION_GAP_DAYS:
            out.append(n)
            n = 1
        else:
            n += 1
    out.append(n)
    return out


def mpc_opp_profile_ok(astrometry):
    """``(ok, reason)`` for the MPC's multi-opposition night requirements on OBS80
    text (:func:`opp_profile_counts_ok` on its :func:`oppositions`)."""
    return opp_profile_counts_ok(oppositions(astrometry))


def opp_profile_counts_ok(apps):
    """``(ok, reason)`` for the MPC's multi-opposition night requirements, from the
    nights per apparition in time order -- the rule itself, for callers holding
    observations rather than OBS80 text.

    * 1 apparition  -- handled by the night/arc rules, not here.
    * 2 apparitions -- one needs >= 3 nights, the other >= 2.
    * 3-4           -- at least one apparition with >= 3 nights.
    * 5+            -- one with >= 3 nights, OR two with >= 2 nights each.
    """
    apps = list(apps)
    n = len(apps)
    if n <= 1:
        return True, ""
    ge3 = sum(1 for a in apps if a >= 3)
    ge2 = sum(1 for a in apps if a >= 2)
    if n == 2:
        ok = ge3 >= 1 and ge2 >= 2
        return ok, "" if ok else (
            f"2 apparitions need one with >=3 nights and the other >=2; "
            f"got {apps}")
    if n <= 4:
        ok = ge3 >= 1
        return ok, "" if ok else (
            f"{n} apparitions need at least one with >=3 nights; got {apps}")
    ok = ge3 >= 1 or ge2 >= 2
    return ok, "" if ok else (
        f"{n} apparitions need one with >=3 nights or two with >=2; got {apps}")


def submittable_nights(astrometry):
    """``(ok, reason)`` -- the MPC's night requirement on the arc a document
    would carry, as the LAST word before any linkage is called submittable.

    Our own opposition-profile gate (``itf.search.opp_profile_gate``) decides
    whether WE believe a linkage, and it has waivers (the NEO waiver, the
    multi-opposition seed) and later re-judges (the web's promote-on-step and
    grown-set paths) that never ask the MPC's question.  The MPC's is
    bookkeeping: two tracklets count as two nights only when they are at least
    **12 hours** apart (:func:`night_ordinals`, the 12-hour window), and two
    apparitions need one of >= 3 nights and the other >= 2.  Two tracklets of
    one apparition taken three hours apart on 2019-06-03 are ONE night however
    well they fit (found on K26E25U).
    """
    ok, why = mpc_opp_profile_ok(astrometry)
    if ok:
        return True, ""
    return False, (f"the MPC would refuse it: {why} (tracklets less than 12 "
                   f"hours apart are one night)")


def check(astrometry, fit=None, rms_arcsec=None):
    """Would the MPC accept this identification?  ``(ok, reasons)``.

    ``reasons`` is every rule that refused it, so a caller can report all of
    them rather than the first.  ``fit`` is a ``FittedOrbit`` (for the 3-night
    element-uncertainty test); omit it and a 3-night case that NEEDS that test
    is refused as untested rather than passed.

    Nothing here is a judgement about whether the linkage is TRUE -- only about
    whether it is the sort of thing the MPC will take.
    """
    bad = []
    nights = len(night_ordinals(astrometry))
    arc = arc_days(astrometry)

    # ---- refused before any orbit is fitted --------------------------------
    if nights < MIN_NIGHTS:
        bad.append(f"{nights} distinct nights (< {MIN_NIGHTS})")
    if arc < MIN_ARC_DAYS:
        bad.append(f"arc {arc:.1f} d (< {MIN_ARC_DAYS:g})")
    if nights == 3 and arc > MAX_ARC_DAYS_3_NIGHTS:
        bad.append(f"3 nights spanning {arc:.1f} d (> {MAX_ARC_DAYS_3_NIGHTS:g})")

    ok_profile, why = mpc_opp_profile_ok(astrometry)
    if not ok_profile:
        bad.append(why)

    # ---- refused after fitting ---------------------------------------------
    if fit is None and rms_arcsec is None:
        bad.append("no fit: convergence and rms untested")
        return (not bad), bad

    quality_applies = ((nights == 3 and arc < QUALITY_ARC_DAYS_3_NIGHTS)
                       or (nights > 3 and arc < QUALITY_ARC_DAYS_MORE))
    if quality_applies and (rms_arcsec or 0) > RMS_ARCSEC:
        # The MPC's wording is "RMS > 0.25 AND orbit quality is not
        # sufficient", so a high rms alone does not refuse it -- the element
        # uncertainties get to rescue it.  Refuse only if BOTH fail.
        ok_q, why_q = _orbit_quality(fit)
        if not ok_q:
            bad.append(f"rms {rms_arcsec:.2f}\" > {RMS_ARCSEC} and {why_q}")

    if nights == 3:
        ok_q, why_q = _orbit_quality(fit)
        if not ok_q:
            bad.append(f"3-night orbit quality: {why_q}")

    return (not bad), bad


def _orbit_quality(fit):
    """The MPC's 3-night orbit-quality test.  ``(ok, reason)``."""
    if fit is None:
        return False, "no fit to test"
    try:
        a, sa = fit.semimajor_axis()
        q, sq = fit.perihelion_distance()
        i, si = fit.inclination()
        e, se = fit.eccentricity()
    except Exception as ex:                              # noqa: BLE001
        return False, f"elements unavailable ({type(ex).__name__})"
    bad = []
    if not (sa < MAX_SIGMA_A_AU):
        bad.append(f"sigma_a {sa:.4f} au")
    if not (sq < MAX_SIGMA_Q_AU):
        bad.append(f"sigma_q {sq:.4f} au")
    if not (si < MAX_SIGMA_I_DEG):
        bad.append(f"sigma_i {si:.4f} deg")
    if not (se < MAX_SIGMA_E):
        bad.append(f"sigma_e {se:.4f}")
    if not (e < MAX_E_3_NIGHTS):
        bad.append(f"e {e:.3f} >= {MAX_E_3_NIGHTS}")
    return (not bad), ", ".join(bad)
