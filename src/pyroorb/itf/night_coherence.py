"""Per-night residual COHERENCE: does the orbit actually fit the nights it links?

Every other quality clause asks how BIG the residuals are.  This one asks how
they are ARRANGED, which is a different question and catches a different
failure.

A correct orbit leaves residuals that scatter the same way WITHIN a night as
BETWEEN nights -- both are just measurement error.  A linkage that has taken the
wrong tracklet cannot fit every night at once, so the fit splits the difference:
each night ends up COHERENTLY displaced, its mean residual a real vector rather
than noise about zero, and the displacements alternate in sign as the orbit is
pulled between the nights it cannot reconcile.

The worked example is an ITF-only candidate whose four seed nights came out at
+1.04, -1.12, -0.70 and +0.44 arcsec while the newly linked night sat at 0.04:
the fit had been dragged onto the new tracklet and the seed absorbed all of the
error.  Every size-based clause passed it -- the arc's rms was 0.56" against a
1.0" station floor, i.e. "fits its uncertainties" -- because size was never the
problem.

Why this is not :mod:`itf.residual_trend`.  That gate measures residuals WALKING ACROSS a
tracklet's own 30-60 minutes.  A large part of the ITF (and nearly all of the
archival surveys) is two exposures ~70 seconds apart -- one position, no walk to
measure -- so the trend gate is structurally inert on exactly the population
where this failure lives.  This works BETWEEN nights, where the lever arm is.

**Read the dof note before trusting a pass.**  Structure can only be seen where
there is slack to see it in: each night contributes one position (two
coordinates) and an orbit costs six, so a four-night arc has two degrees of
freedom and will absorb almost any fourth point.  ``evidence_dof`` says how much
the arc could have tested itself; a clean result at 2 dof means untested, not
good.

------------------------------------------------------------------------------
THE SCALE.  Everything here is normalised PER AXIS, so 1.0 means "exactly as the
observations' own uncertainties say".  That is worth stating because the first
version of this module was not: it divided a 2-D radial residual
(``sqrt(dRA^2 + dDec^2)``) by a 1-AXIS sigma (the mean ``sigma_ra``; ``sigma_dec``
was never read), so a PERFECT arc scored sqrt(2) = 1.41 against a veto at 1.0.
It only ever passed because ITF sigmas carry a ~1.0" station floor against ~0.5"
of real scatter -- one error cancelling another, and neither one visible in the
number.  Both are fixed: residuals are divided by their OWN axis's sigma, and
the two axes are averaged rather than summed.
"""

import collections
import math

from .opp_profile import NIGHT_DAYS, spans_of

#: Fallback per-axis uncertainty when neither the residual row nor the
#: observation carries one.  Only reached for hand-built observations; the
#: engine always attaches the sigmas it actually fitted with.
DEFAULT_SIGMA_ARCSEC = 1.0


def _expected_max(n):
    """Expected value of the worst night's ``z**2`` under a PERFECT orbit.

    ``z**2`` for one night is (per axis, standard-error normalised) the mean of
    two squared standard normals, i.e. ``chi2_2 / 2``, which is Exponential(1).
    The expected maximum of ``n`` iid Exponential(1) draws is the harmonic
    number ``H_n`` exactly -- so dividing the worst night's ``z`` by
    ``sqrt(H_n)`` makes the statistic mean-1 for ANY number of nights.

    (The previous divisor, ``sqrt(2 ln n)``, is the asymptotic max of n
    half-normals -- the wrong distribution AND wrong at small n: it is 0 at
    n = 1 and 1.18 at n = 2, so it inflated short arcs.)
    """
    return math.sqrt(sum(1.0 / i for i in range(1, int(n) + 1))) if n >= 1 else 1.0


def night_structure(fit, obs=None, residuals=None, gap_days=NIGHT_DAYS):
    """Per-night residual structure of ``fit`` (a ``FittedOrbit``).

    Returns a dict; ``n_nights == 0`` means nothing could be measured, which is
    reported rather than treated as a pass.

    Nights are the codebase's nights -- :func:`itf.opp_profile.nights_of`, a GAP in
    the observation times -- not a UT calendar date.  Keying on ``line[15:25]``
    (which this did) splits any western-hemisphere night that crosses UT
    midnight into two half-nights: each half gets a smaller ``n_obs`` (so a
    bigger standard error) and the night COUNT is inflated (so a bigger
    max-correction), moving both terms of the statistic for a purely
    bookkeeping reason.

    ``gap_days`` defaults to the CALENDAR rule (0.5 d) rather than the ported
    1.6 d one on purpose: 1.6 d merges two genuinely consecutive nights into
    one, and merging two nights is precisely how a between-night displacement
    hides itself from a between-night statistic.  0.5 d sits in the measured
    trough between the within-night and next-night modes (see
    ``ItfConfig.span_days``).
    """
    try:
        # INCLUDED observations only.  The question here is whether the arc the
        # fit ACCEPTED is internally consistent; an observation the fit rejected
        # as an outlier is not part of that arc and its residual can be
        # enormous.  Reading all of them made an 8-night linkage fitting at
        # 0.153" report a night displaced 213 arcsec, because the rebuild had
        # swept in observations the search never used.
        #
        # (The seed-preservation clause in search.py deliberately does the
        # opposite -- there a rejected seed observation SHOULD count at full
        # residual, because the question is what the link did to the seed.)
        # residuals(included_only=False) stays aligned with fit._obs, so the
        # mask is applied here rather than asking for a shortened list
        resid_all = residuals if residuals is not None else fit.residuals()
        rows_all = obs if obs is not None else fit._obs
        mask = list(fit.is_included())
        # ``obs`` / ``residuals`` are the caller's, ``mask`` is the fit's, and the
        # three are combined POSITIONALLY below.  A caller list that is not the
        # fit's own arc would be zipped against the wrong mask entries -- silently,
        # since zip() just stops at the shortest -- and every night in the report
        # would be built from a mixture of two arcs.  Nothing measurable is better
        # than something wrong.
        if not (len(resid_all) == len(rows_all) == len(mask)):
            return {"n_nights": 0}
        resid = [r for r, keep in zip(resid_all, mask) if keep]
        rows = [o for o, keep in zip(rows_all, mask) if keep]
    except Exception:                                   # noqa: BLE001
        return {"n_nights": 0}

    # ---- gather (line, residual/sigma) per observation ---------------------
    per_line = collections.defaultdict(list)
    sigmas, lines = [], []
    for o, r in zip(rows, resid):
        line = o.get("obs80_line") or ""
        if len(line) < 25:
            continue
        # The residual row carries the sigmas the fit ACTUALLY used (timing
        # inflation, station floor, ADES rmsRA/rmsDec); the observation dict is
        # the fallback for hand-built inputs.
        s_ra = r.get("sigma_ra") or o.get("sigma_ra") or DEFAULT_SIGMA_ARCSEC
        s_dec = (r.get("sigma_dec") or o.get("sigma_dec")
                 or r.get("sigma_ra") or o.get("sigma_ra") or DEFAULT_SIGMA_ARCSEC)
        s_ra, s_dec = float(s_ra), float(s_dec)
        if not (s_ra > 0):
            s_ra = DEFAULT_SIGMA_ARCSEC
        if not (s_dec > 0):
            s_dec = DEFAULT_SIGMA_ARCSEC
        d_ra = float(r["d_ra_arcsec"])
        d_dec = float(r["d_dec_arcsec"])
        per_line[line].append((d_ra, d_dec, s_ra, s_dec))
        sigmas.extend((s_ra, s_dec))
        lines.append(line)
    if not lines:
        return {"n_nights": 0}

    # Group the UNIQUE lines: `per_line` already holds every measurement filed
    # under a line, so handing nights_of the raw list would count a repeated
    # line's observations once per repetition.
    groups = [[m for l in night for m in per_line[l]]
              for night in spans_of(list(per_line), gap_days)]
    groups = [g for g in groups if g]
    if not groups:
        return {"n_nights": 0}

    n = len(groups)
    # ---- per-night statistics ---------------------------------------------
    # z_night**2 = k * mean(z_ra)**2 + k * mean(z_dec)**2, halved: the mean over
    # the TWO AXES of the squared standard-error-normalised displacement, so a
    # night that sits exactly where its uncertainties say scores 1.0 whatever
    # its observation count.
    worst_z, worst_arcsec = 0.0, 0.0
    sq_z, within, n_obs = [], [], 0
    for g in groups:
        k = len(g)
        n_obs += k
        m_ra = sum(d_ra for d_ra, _, _, _ in g) / k
        m_dec = sum(d_dec for _, d_dec, _, _ in g) / k
        mz_ra = sum(d_ra / s_ra for d_ra, _, s_ra, _ in g) / k
        mz_dec = sum(d_dec / s_dec for _, d_dec, _, s_dec in g) / k
        z2 = k * (mz_ra * mz_ra + mz_dec * mz_dec) / 2.0
        sq_z.append(z2)
        if z2 > worst_z * worst_z:
            worst_z = math.sqrt(z2)
            worst_arcsec = math.hypot(m_ra, m_dec)
        within.extend((d_ra - m_ra) ** 2 + (d_dec - m_dec) ** 2
                      for d_ra, d_dec, _, _ in g)

    all_pairs = [m for g in groups for m in g]
    # Both arcsec rms values below are PER-AXIS (divide the summed
    # d_ra^2 + d_dec^2 by 2n, not n) -- the codebase standard for anything
    # reported, and the currency rms_sigma right beneath already used.  A
    # record whose two rms diagnostics sit sqrt(2) apart invites exactly the
    # comparison that cannot be made; see itf.fitter's docstring.
    rms_arcsec = math.sqrt(
        sum(d_ra * d_ra + d_dec * d_dec for d_ra, d_dec, _, _ in all_pairs)
        / (2.0 * len(all_pairs)))
    rms_sigma = math.sqrt(
        sum((d_ra / s_ra) ** 2 + (d_dec / s_dec) ** 2
            for d_ra, d_dec, s_ra, s_dec in all_pairs)
        / (2.0 * len(all_pairs)))

    return {
        # THE GATE.  See the module docstring and `night_coherence` below.
        "night_coherence": worst_z / _expected_max(n),
        "n_nights": n,
        "n_obs": n_obs,
        "evidence_dof": 2 * n - 6,
        # ---- diagnostics, for the notes string and for a human reading a
        # stored record.  Nothing branches on these.
        "worst_night_z": worst_z,
        "worst_night_arcsec": worst_arcsec,
        "rms_sigma": rms_sigma,
        "rms_arcsec": rms_arcsec,
        "sigma_arcsec": sum(sigmas) / len(sigmas),
        "within_night_arcsec": (math.sqrt(sum(within) / (2.0 * len(within)))
                                if within else 0.0),      # per-axis, as above
    }


def night_coherence(fit):
    """Just the number the acceptance clause reads, or ``None``.

    WHY THIS ONE, of the three the module used to compute.  All three ask "how
    far is the worst night displaced?"; they differ in what they divide by, and
    only this one divides by something that makes the threshold mean the same
    thing on two arcs of different SHAPE:

    * the raw displacement (``mean residual / sigma``) is biased by observations
      per night -- a 1-observation night scatters at a full sigma while a
      16-observation night scatters at a quarter of one, so the same threshold
      is four different tests on one arc;
    * dividing by the standard error (``sigma / sqrt(n_obs)``) fixes that but
      leaves the MAXIMUM over N nights, which grows with N under a perfect
      orbit -- length bias, harshest on exactly the long multi-opposition arcs
      most likely to be genuine.  This is how the first version of the clause
      came to veto 856215 (19 nights, 5 oppositions, 90/91 observations, 0.242"):
      not because that orbit fits its nights badly but because it has many;
    * dividing further by the expected maximum (:func:`_expected_max`) removes
      both, and leaves a number that is ~1.0 for ANY arc whose residuals are
      what its uncertainties claim.

    So the threshold is a statement about the orbit and not about the arc's
    shape, which is the only form in which a single constant can be shared
    between the engine's clause and the campaign's review.
    """
    st = night_structure(fit)
    return st.get("night_coherence") if st.get("n_nights") else None
