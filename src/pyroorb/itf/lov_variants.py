"""Variant-orbit corridor for UNDER-DETERMINED (short-arc, high-U) ITF seeds.

When a seed apparition is so short that its linear covariance is meaningless --
U ~ 9, the true position years away is off by *tens of degrees*, not a small
ellipse (2015 BE559 was the case) -- no amount of corridor
widening on the single nominal ephemeris can reach the object.  The right tool
is a set of **variant orbits** sampled along the Line of Variations (the
weakly-constrained period/along-track direction): each is a legitimate orbit,
and one of them sits near the truth.  We scan them one at a time (the proven
"try each variant with a narrow corridor" approach), earliest-exit on a clean
link, and let the normal recursion tighten the rest -- recovery only needs ONE
variant to reach ONE apparition, because linking it collapses U (BE559:
9.56 -> 3.96) and the remaining apparitions then fall in the ordinary corridor.

The cost that used to make this impractical -- one 27-year ephemeris sweep PER
variant -- is gone: all K variants ride a SINGLE ASSIST integration
(:func:`force_assist.helio_states_multi`, the same shared-integration trick the
analytic Jacobian and radar sweeps use), so K variant tracks over the whole ITF
era cost ~one sweep (41 variants in ~1.1 s, vs ~90 s naive).

This module builds the variant tracks + their thin corridor windows; the scan /
early-exit loop lives in :mod:`pyroorb.itf.search`.
"""

import math
from datetime import datetime, timezone

import numpy as np
from astropy.time import Time

from pyroorb import photometry
from pyroorb.forward_model import _radec_from_helio

from .ephem import corridor_windows, faint_cut
from .obs80_fields import unix_from_jd

DAY_MIN = 1440.0
C_AU_DAY = 173.1446326847


def _era_grid(cfg, fit, jd_start=None, jd_stop=None):
    """UTC-JD node grid across the ITF era (same span/step as sweep_nodes)."""
    if jd_start is None:
        dt = datetime.fromisoformat(cfg.ephem_start).replace(tzinfo=timezone.utc)
        jd_start = dt.timestamp() / 86400.0 + 2440587.5
    if jd_stop is None:
        if cfg.ephem_steps is not None:
            jd_stop = jd_start + cfg.ephem_steps * cfg.ephem_step_days
        else:
            now_jd = (datetime.now(timezone.utc).timestamp() / 86400.0
                      + 2440587.5)
            jd_stop = now_jd + 30.0
    return np.arange(jd_start, jd_stop + 1e-9, cfg.ephem_step_days)


def variant_states(fit, cfg):
    """LOV variant orbit states for a weak seed, ordered CENTRAL-OUT (most
    probable first, so the scan's early exit hits the likeliest variants
    first).  Returns ``(states, ks)`` -- (m,6) heliocentric-equatorial states
    at ``fit.epoch_tdb`` and their signed LOV sigma-offsets -- or
    ``(None, None)`` when the LOV is degenerate / unavailable (nongrav-augmented
    state, no live evaluator).  The nominal orbit is always included (k=0)."""
    lov = fit.lov_uncertainty(n_steps=cfg.variant_n_steps,
                              n_sigma=cfg.variant_n_sigma)
    if not lov or lov.get("variants") is None:
        return _states_without_elements(fit, cfg)
    states = np.asarray(lov["variants"], float)
    lov_sigmas = np.asarray(lov["k"], float)
    if len(states) != len(lov_sigmas) or not len(states):
        return None, None
    order = np.argsort(np.abs(lov_sigmas))          # central-out: |k| ascending
    return states[order], lov_sigmas[order]


def prior_variant(fit, obscodes):
    """The orbit digest2's population model expects on this LOV, as a one-variant fan
    ``(states, ks)``, or ``(None, None)``.  See :func:`pyroorb.popprior.lov_weights`.

    A short arc can fit a parabola with the truth a sliver of the LOV away, where the
    equal-probability fan cannot land: 2008 SZ347 (a 5.2 au, seed e 0.9999) has its truth at
    k 0.063, the peak here, between fan samples at 0 and 0.19, and a variant 2 deg off
    leaves its tracklets at P 0.000 below 280 live candidates.
    """
    from pyroorb import popprior
    if getattr(fit, "lov_refit", None) is None:
        return None, None
    lov = popprior.lov_weights(fit, obscodes)
    if lov is None:
        return None, None
    state = popprior.lov_peak_state(fit, lov["k_peak"])
    if state is None:
        return None, None
    return state[None, :], np.array([lov["k_peak"]])


def _states_without_elements(fit, cfg):
    """LOV states for a seed too weak for ``lov_uncertainty`` to report on.

    That function serves TWO products from one loop -- element sigmas and the
    variant states -- and keeps a variant only if its osculating ELEMENTS come
    out finite, then discards everything unless >= max(5, n_steps//3) survive.
    Sound for sigmas.  Wrong here: this scan wants ORBITS, and a sample that is
    momentarily hyperbolic (e > 1, so some element is undefined) is still a
    perfectly good state to sweep a corridor for.

    2002 PG150 is the case -- a 5-obs/30-day arc on a q 23.4 object, U 18.65,
    the weakest kind of seed the variant fallback exists FOR.  All 31 of its
    LOV samples re-correct to finite states; enough of them are hyperbolic that
    the element test leaves too few rows, so ``lov_uncertainty`` returns None
    and the fallback silently does nothing at all -- the search then reports
    "no likely matches" having scanned zero variants.

    Uses the same machinery (``lov_refit``: the LOV direction, its 1-sigma
    extent, and the pinned re-correction), the same k grid, and keeps every
    state that comes back finite.  Returns ``(None, None)`` when even that is
    unavailable (no live evaluator, degenerate covariance, nongrav state).
    """
    from pyroorb.lov import _equal_prob_sigmas

    refit = getattr(fit, "lov_refit", None)
    if refit is None:                    # a test double / a fit without the LOV
        return None, None
    u, sigma, refit_fn = refit()
    if u is None:
        return None, None
    state0 = np.asarray(fit._state, float)
    states, lov_sigmas = [], []
    for k in _equal_prob_sigmas(cfg.variant_n_steps, cfg.variant_n_sigma):
        try:
            v = np.asarray(refit_fn(state0 + (k * sigma) * u, u, sigma, k),
                           float)
        except Exception:
            continue
        if v.size >= 6 and np.all(np.isfinite(v[:6])):
            states.append(v[:6]); lov_sigmas.append(k)
    if not states:
        return None, None
    states = np.asarray(states, float); lov_sigmas = np.asarray(lov_sigmas, float)
    order = np.argsort(np.abs(lov_sigmas))          # central-out, as above
    return states[order], lov_sigmas[order]


def variant_tracks(fit, obscodes, cfg, states, jds=None):
    """Per-variant ephemeris tracks over the ITF era, all in ONE integration.

    ``states`` -- (m,6) heliocentric-equatorial variant states at
    ``fit.epoch_tdb``.  Returns ``(jds, tracks)`` where ``tracks`` is a list of
    m node-lists; each node dict matches :func:`pyroorb.itf.ephem._node`
    (``unix/ra/dec/delta/mag/speed/unc_deg/elong``) so it can be fed straight to
    :func:`corridor_windows`.  ``unc_deg`` is left at 0 here -- the caller sets
    the thin fixed width (the corridor's coverage comes from having many
    variant centrelines, not one wide ellipse).  NaN nodes (a variant that went
    hyperbolic / failed to integrate at some epoch) are dropped per variant.
    """
    from pyroorb.obs_sources import observer_positions_batch
    from pyroorb.force_assist import helio_states_multi

    if jds is None:
        jds = _era_grid(cfg, fit)
    n = len(jds)
    tt = Time(jds, format="jd", scale="utc").tdb.jd
    Robs = np.asarray(observer_positions_batch(jds, ["500"] * n, obscodes))
    H, _ = fit.absolute_magnitude()
    G = fit.slope_parameter()

    hs = helio_states_multi(fit.epoch_tdb, list(states), tt)   # (n, m, 6)
    m = hs.shape[1]
    earth_sun = np.linalg.norm(Robs, axis=1)                   # (n,)

    # RA/Dec + geometry per epoch, vectorized across the m variants
    ra = np.full((n, m), np.nan)
    dec = np.full((n, m), np.nan)
    delta = np.full((n, m), np.nan)
    rhelio = np.full((n, m), np.nan)
    for i in range(n):
        r = hs[i, :, :3]                                       # (m,3)
        v = hs[i, :, 3:]
        a, d = _radec_from_helio(r, v, Robs[i])
        ra[i] = a
        dec[i] = d
        rel = r - Robs[i]
        delta[i] = np.linalg.norm(rel, axis=1)
        rhelio[i] = np.linalg.norm(r, axis=1)

    uni = unix_from_jd(jds)                                    # (n,)
    tracks = []
    for k in range(m):
        nodes = []
        for i in range(n):
            if not np.isfinite(ra[i, k]) or not np.isfinite(delta[i, k]):
                continue
            # apparent magnitude (H,G) -- lets the mag_limit gate work
            mag = None
            if H is not None:
                red, _ = photometry.phase_reduction(
                    float(rhelio[i, k]), float(delta[i, k]),
                    float(earth_sun[i]), G)
                if red < 9000.0:
                    mag = H + red
            nodes.append({
                "unix": float(uni[i]), "ra": float(ra[i, k]),
                "dec": float(dec[i, k]), "delta": float(delta[i, k]),
                "mag": mag, "speed": 0.0, "unc_deg": 0.0, "elong": None,
                "_i": i,
            })
        _fill_speed(nodes)
        tracks.append(nodes)
    return jds, tracks


def _fill_speed(nodes):
    """On-sky speed (arcsec/min) per node from the adjacent-node finite
    difference (2-day baseline is ample for the corridor's fast-mover split and
    the 0.02*speed tolerance term -- the corridor needs the scale, not
    precision)."""
    for j, nd in enumerate(nodes):
        nb = nodes[j + 1] if j + 1 < len(nodes) else (nodes[j - 1] if j else None)
        if nb is None:                  # a lone node has no baseline
            continue
        cosd = math.cos(math.radians(nd["dec"]))
        dE = (((nb["ra"] - nd["ra"] + 180) % 360 - 180) * cosd) * 3600.0
        dN = (nb["dec"] - nd["dec"]) * 3600.0
        dt_min = abs(nb["unix"] - nd["unix"]) / 60.0
        if dt_min > 0:
            nd["speed"] = math.hypot(dE, dN) / dt_min


def _sky_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation (deg) between two sky points."""
    r1, d1, r2, d2 = (math.radians(x) for x in (ra1, dec1, ra2, dec2))
    c = (math.sin(d1) * math.sin(d2)
         + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _contiguous_runs(nodes):
    """Split node lists on a gap in the epoch-grid index ``_i`` so a corridor
    window is never built across a DROPPED (redundant near-seed) region -- its
    quadratic + time box would otherwise span the gap and query empty sky."""
    runs, run = [], []
    for nd in nodes:
        if run and nd["_i"] - run[-1]["_i"] > 1:
            runs.append(run)
            run = []
        run.append(nd)
    if run:
        runs.append(run)
    return runs


def _lov_prob_scale(cfg):
    """Factor mapping a k in the fan's rescaled sigma units to a STANDARD
    normal quantile, so a k-interval can be priced in probability.

    :func:`lov._equal_prob_sigmas` divides its quantiles by the largest one and
    multiplies by ``n_sigma``, so the fan's "5 sigma" edge is really the
    ``(n-0.5)/n`` quantile -- 2.13 sigma at the default 31 steps, not 5.  The
    rescale is a single monotone factor, and undoing it is what makes
    ``Phi(k*scale)`` an honest probability rather than a decorative one.

    SUPERSEDED (2026-09-22) unless ``cfg.variant_prob_legacy``: the premise
    above is wrong.  The rescale moves where the samples SIT, not what they
    mean -- each variant is re-corrected at ``state0 + k*sigma*u``, i.e. at k
    LOV sigmas, so the probability between two variants is
    ``Phi(k_b) - Phi(k_a)`` and the right factor is 1.  The legacy 0.426
    priced the tails ~2.35x too wide, which on the pre-floor covariance (too
    tight by a typical 2.5x) happened to cancel.  With the s2 floor the
    covariance is honest -- over all 26 ITF fixtures the truth sits at a
    median |k| of 0.67 LOV sigma (N(0,1): 0.674), max 2.72 -- so the legacy
    factor now double-counts, and the refinement spends its extra variants
    (and their union-sweep windows) in tails that hold ~1e-5 of the truth's
    probability instead of the core where it lives."""
    if not getattr(cfg, "variant_prob_legacy", False):
        return 1.0
    from scipy.special import erfinv

    n = max(int(cfg.variant_n_steps), 1)
    edge = math.sqrt(2.0) * float(erfinv(2.0 * ((n - 0.5) / n) - 1.0))
    return edge / cfg.variant_n_sigma if cfg.variant_n_sigma else 0.0


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _interval_costs(tracks, lov_sigmas, cfg, scale):
    """Price every adjacent-in-k interval by the probability it LOSES.

    Returns ``[(uncovered, i_lo, i_hi, lov_sigma_lo, lov_sigma_hi), ...]``, where ``uncovered``
    is ``dp * (1 - coverage)``: the chance the truth lies between these two
    variants AND at an epoch where neither corridor reaches it.

    Coverage is measured only over epochs the search would actually look at --
    both variants have a node there, and at least one is predicted brighter
    than ``mag_limit`` (a window fainter than the cut is dropped by
    :func:`~pyroorb.itf.ephem.corridor_windows`, so a hole at that epoch costs
    nothing that this fan can fix).  An interval nobody can measure -- no
    shared epochs -- is priced 0: unmeasured is not the same as uncovered, and
    guessing it wide would spend the budget on nothing."""
    W = cfg.variant_width_deg
    order = sorted(range(len(lov_sigmas)), key=lambda i: lov_sigmas[i])
    by_epoch = [{nd["_i"]: nd for nd in tr} for tr in tracks]
    out = []
    for pos in range(len(order) - 1):
        a, b = order[pos], order[pos + 1]
        dp = abs(_norm_cdf(lov_sigmas[b] * scale) - _norm_cdf(lov_sigmas[a] * scale))
        cov, n_ep = 0.0, 0
        for ei, nd in by_epoch[a].items():
            nb = by_epoch[b].get(ei)
            if nb is None:
                continue
            mags = [m for m in (nd["mag"], nb["mag"]) if m is not None]
            if mags and min(mags) >= cfg.mag_limit:
                continue
            gap = _sky_sep_deg(nd["ra"], nd["dec"], nb["ra"], nb["dec"])
            cov += 1.0 if gap <= 2.0 * W else (2.0 * W / gap)
            n_ep += 1
        if not n_ep:
            continue
        out.append((dp * (1.0 - cov / n_ep), a, b, float(lov_sigmas[a]), float(lov_sigmas[b])))
    out.sort(reverse=True)
    return out


def refine_variant_fan(fit, obscodes, cfg, states, lov_sigmas, jds=None):
    """Bisect the LOV fan where its corridors leave a HOLE the truth can hide in.

    Returns ``(states, ks, tracks, n_extra)`` -- the fan central-out as
    :func:`variant_states` hands it back, plus the tracks (so the caller need
    not integrate a third time) and how many variants were ADDED.

    The rule is the one in the config note: greedily split the interval with
    the largest *uncovered probability* ``dp * (1 - coverage)``, re-measure,
    repeat.  Bounded three ways -- ``variant_refine_max_extra`` variants,
    ``variant_refine_rounds`` passes (one shared ASSIST integration each), and
    a ``variant_refine_min_gain`` floor below which a split buys less than it
    costs.  A fan with no holes exits after the first measurement having added
    nothing, so a healthy seed pays one arithmetic pass over tracks it needed
    anyway.

    Needs ``fit.lov_refit`` (arbitrary-k re-correction); without it the fan is
    returned untouched, since ``lov_uncertainty`` can only produce the fixed
    equal-probability grid.

    MEASURED, uncovered probability before -> after (+variants added):

        2014 HY176  0.045 -> 0.032  (+3)     the miss this was built for
        2026 EX121  0.028 -> 0.015  (+2)
        2025 YQ13   0.151 -> 0.066  (+16)
        2026 EP114  0.544 -> 0.470  (+24, capped)
        2025 NV385  0.584 -> 0.435  (+24, capped)
        five healthy 4.5 <= U <= 6.3 seeds:  0.000, +0 each

    Two things to read there.  The severity order is the one the four short
    arcs' live searches actually ranked in (see the coverage table in
    ``variant_windows``), which is some evidence the metric measures the right
    thing.  And the NV385/EP114 class is CAPPED, not cured: a fan that sparse
    needs hundreds of samples, and 24 buys ~14 points of probability rather
    than closing the gate.  That is the deliberate trade -- the cap bounds the
    union sweep at +77% variants on the worst case, and a partial cure that
    cannot blow up the sweep is worth more than a whole one that can (the
    widening cure took NV385 from a completed 465 s to >5400 s unfinished).

    The coverage metric assumes the shipped fixed-width corridor; with
    ``variant_fill_gaps`` on it understates coverage, so the refinement spends
    more than it needs to rather than less."""
    budget = int(getattr(cfg, "variant_refine_max_extra", 0) or 0)
    refit = getattr(fit, "lov_refit", None)
    if budget <= 0 or refit is None:
        return states, lov_sigmas, None, 0
    # Everything below is an OPTIMISATION of an existing fan, so it degrades to
    # a no-op rather than propagating: a fan that cannot be measured is still a
    # fan the scan can run, and a refinement that raises would abort the whole
    # variant scan on exactly the severe seeds it exists to rescue.
    try:
        u, sigma, refit_fn = refit()
        if u is None:
            return states, lov_sigmas, None, 0
        jds, tracks = variant_tracks(fit, obscodes, cfg, states, jds=jds)
    except Exception:
        return states, lov_sigmas, None, 0

    state0 = np.asarray(fit._state, float)
    scale = _lov_prob_scale(cfg)
    try:
        return _refine_loop(fit, obscodes, cfg, states, lov_sigmas, tracks, jds,
                            state0, u, sigma, refit_fn, scale, budget)
    except Exception:
        return states, lov_sigmas, tracks, 0


def _refine_loop(fit, obscodes, cfg, states, lov_sigmas, tracks, jds, state0, u, sigma,
                 refit_fn, scale, budget):
    """The greedy bisection itself (see :func:`refine_variant_fan`)."""
    min_gain = float(getattr(cfg, "variant_refine_min_gain", 0.0) or 0.0)
    states = np.asarray(states, float)
    lov_sigmas = np.asarray(lov_sigmas, float)
    n_extra = 0
    for _round in range(max(int(getattr(cfg, "variant_refine_rounds", 0)), 0)):
        if budget <= 0:
            break
        costs = [c for c in _interval_costs(tracks, lov_sigmas, cfg, scale)
                 if c[0] > min_gain]
        if not costs:
            break
        new_states, new_lov_sigmas = [], []
        for _cost, _a, _b, lov_sigma_lo, lov_sigma_hi in costs[:budget]:
            lov_sigma_mid = 0.5 * (lov_sigma_lo + lov_sigma_hi)
            try:
                v = np.asarray(refit_fn(state0 + (lov_sigma_mid * sigma) * u, u, sigma,
                                        lov_sigma_mid), float)
            except Exception:          # this k will not re-correct; leave the gap
                continue
            if v.size >= 6 and np.all(np.isfinite(v[:6])):
                new_states.append(v[:6])
                new_lov_sigmas.append(lov_sigma_mid)
        if not new_states:
            break
        # ONE integration for the whole round's additions, on the SAME epoch
        # grid, so the new nodes carry the same ``_i`` epoch indices and drop
        # straight into the gap measurement above.
        _jds, new_tracks = variant_tracks(fit, obscodes, cfg,
                                          np.asarray(new_states, float), jds=jds)
        states = np.vstack([states, np.asarray(new_states, float)])
        lov_sigmas = np.concatenate([lov_sigmas, np.asarray(new_lov_sigmas, float)])
        tracks = list(tracks) + list(new_tracks)
        budget -= len(new_states)
        n_extra += len(new_states)

    order = np.argsort(np.abs(lov_sigmas))     # central-out, as variant_states promises
    tracks = [tracks[i] for i in order]
    return states[order], lov_sigmas[order], tracks, n_extra


def variant_windows(fit, obscodes, cfg, max_uncert_deg, states, lov_sigmas, jds=None,
                    tracks=None):
    """Thin corridor windows per variant, ordered central-out.

    Returns ``[(k, windows), ...]`` -- one entry per variant that produced any
    window, ``k`` its signed LOV sigma-offset.  Each variant's windows use the
    fixed half-width ``cfg.variant_width_deg`` (the union of many thin
    centrelines covers the along-track arc; a per-variant covariance ellipse
    would just re-inflate the corridor we are trying to avoid).

    FAR-EPOCH ONLY: the variants coincide with the nominal near the seed and
    diverge only far away, so an off-nominal variant's near-seed windows just
    duplicate the nominal corridor (already searched by the linear pass).  Each
    off-nominal variant therefore keeps only the windows where it diverges from
    the nominal track by >= one corridor width -- the fanned-out far tail that
    is its whole reason to exist.  This is the dominant lever on the union
    sweep's size (28k windows -> a few thousand): the nominal keeps its full
    track, the rest contribute only their divergent segments.
    """
    if tracks is None:
        jds, tracks = variant_tracks(fit, obscodes, cfg, states, jds=jds)
    W = cfg.variant_width_deg
    # nominal (|k| smallest -> index 0 by central-out order): epoch_idx -> radec
    nominal = ({nd["_i"]: (nd["ra"], nd["dec"]) for nd in tracks[0]}
               if tracks else {})
    half_gap = _neighbour_half_gaps(tracks, lov_sigmas) if cfg.variant_fill_gaps else {}
    out = []
    for vi, (k, nodes) in enumerate(zip(lov_sigmas, tracks)):
        # judged on the variant's WHOLE track: the runs below are slices of it
        cut = faint_cut(cfg, nodes)
        if vi != 0:
            nodes = [nd for nd in nodes
                     if (nb := nominal.get(nd["_i"])) is None
                     or _sky_sep_deg(nd["ra"], nd["dec"], nb[0], nb[1]) >= W]
        for nd in nodes:
            # The union of the variant corridors is only a SEARCH of the LOV
            # where the corridors touch.  A fixed half-width does not touch:
            # the k grid is spaced by equal PROBABILITY, so the samples pile up
            # near the nominal (redundant) and fan apart in the tails, which is
            # exactly where the object is when the seed is this weak.  Measured
            # over the four 2025-26 short arcs, at the default 31 steps and
            # judged at 2010 (median gap between adjacent variants vs the 2W it
            # takes to leave no hole):
            #
            #     2026 EX121  1.3 deg  -> 93.3% of the LOV covered
            #     2025 YQ13   2.0 deg  -> 76.7%
            #     2025 NV385 19.3 deg  ->  0.0%
            #     2026 EP114 28.5 deg  ->  4.2%
            #
            # NV385's scan was 31 thin slivers with ~19 deg of unsearched sky
            # between them; it cannot find the object because it is not looking
            # at it.  The coverage ranks the four in exactly the order their
            # searches actually did.
            #
            # Densifying the grid fixes it (481 steps takes NV385 to 97.9%) but
            # multiplies the union sweep ~15x against a SHARED server.  Widening
            # does it for free: give each node half the distance to its
            # neighbouring variant AT THAT EPOCH, and the union is contiguous by
            # construction with the same variant count and the same integration.
            # Never NARROWER than the fixed width (a dense region keeps its thin
            # corridor) and never past `max_uncert_deg`, which is the sweep's own
            # ceiling -- a gap wider than that is one no corridor may cover, and
            # `corridor_windows` would drop the window anyway.
            # ...bounded by variant_fill_max_mult (see the config note): the
            # widening is paid for in QUERY cost, and uncapped it took NV385's
            # union sweep from ~498 healpix pixels per window to ~7630 -- one
            # chunk query that used to take 465 s had not returned in 45 min.
            ceiling = max_uncert_deg
            if cfg.variant_fill_max_mult > 0:
                ceiling = min(ceiling, cfg.variant_fill_max_mult * W)
            nd["unc_deg"] = min(max(W, half_gap.get((vi, nd["_i"]), 0.0)),
                                ceiling)
        windows = []
        for run in _contiguous_runs(nodes):
            windows.extend(corridor_windows(run, cfg, max_uncert_deg, mag_limit=cut))
        if windows:
            out.append((float(k), windows))
    return out


def _neighbour_half_gaps(tracks, lov_sigmas):
    """``(variant_index, epoch_index) -> half the sky gap to the adjacent
    variant in LOV order`` -- the half-width that makes the variant fan a
    CONTIGUOUS corridor rather than a picket fence (see `variant_windows`).

    Adjacency is in ``k``, not in scan order: ``variant_states`` hands the
    states back central-out (most probable first) so the early-exit hits the
    likeliest variants first, and the neighbour on the sky is the neighbour in
    LOV coordinate.  A variant takes the LARGER of its two half-gaps, so the
    pair on either side of it is covered even when the fan is lopsided; the two
    end variants have one neighbour and use that.
    """
    order = sorted(range(len(lov_sigmas)), key=lambda i: lov_sigmas[i])
    by_epoch = [{nd["_i"]: nd for nd in tr} for tr in tracks]
    gaps = {}
    for pos, vi in enumerate(order):
        for adj in (order[pos - 1] if pos else None,
                    order[pos + 1] if pos + 1 < len(order) else None):
            if adj is None:
                continue
            for ei, nd in by_epoch[vi].items():
                nb = by_epoch[adj].get(ei)
                if nb is None:      # neighbour has no node at this epoch
                    continue
                h = 0.5 * _sky_sep_deg(nd["ra"], nd["dec"], nb["ra"], nb["dec"])
                if h > gaps.get((vi, ei), 0.0):
                    gaps[(vi, ei)] = h
    return gaps
