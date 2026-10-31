"""LOV profiles: mutual consistency before fitting (clustering-idea.md).

The variant scan (search._variant_scan) asks, once per variant, "does any
tracklet fit THIS point on the Line of Variations?" -- and discards the
cross-variant structure that falls out of asking.  This module inverts the
question over the union-sweep cache the scan already holds: every candidate
tracklet gets a **LOV profile** -- which LOV offset k* would make it true
(argmin of the motion-mismatch qual over the variants that served it), and how
sharply (how few variants, how narrow the good-k span).  Three consumers, all
flag-gated in ItfConfig and OFF by default:

- ``reorder``  (2a, ``lov_profile_order``): the PRIORITY variants -- homes of
  the global top ``lov_profile_priority_rounds`` k-informative sharp candidates
  by scored P -- are visited first; the rest stay central-out.  Every
  per-variant AGGREGATE weighting (candidate counts, decent-night P sums)
  was measured to lose to junk mass; a global top-N is the shape junk cannot
  out-vote, and it bounds the trial source spend to a handful of rounds.
- ``round_for(k).center_ids``  (2b, ``lov_profile_center_singles``): within a
  PRIORITY round, the candidates whose k* CENTERS on this variant earn
  front-of-trial queue singles trials -- the take race on a weak seed goes to
  whoever is fitted first, and a broad profile is the signature of junk that
  fits everywhere a little.
- ``round_for(k).cluster_groups``  (2c, ``lov_profile_cluster_trials``): >= 2
  sharp candidates on distinct nights sharing a k* are tried TOGETHER as one
  joint trial, best member per night (mutual consistency under the seed's own
  covariance, before any trial fit; the acceptance gates remain the
  arbiter).

The recall mechanism these compose with is barren-take backtracking
(``backtrack_per_round`` + the woven second-opinion trial source in
search.TrialQueue.next_take): the profiles keep a knife-edge candidate addressable
and the visit targeted, and the round survives junk takes until the take
that cascades.  See clustering-idea.md for the live-measured record.

Measured feasibility (2015 BE559, full instrumented live run, 55 variants):
the decisive 2024 truth is served by 2 of 55 variants with a clean minimum at
exactly the recovering k (+2.83 sigma, qual 1.35"); 3.6% of 103k junk
tracklets share that profile shape, 0.5% collide at the same k.  Cost is
bookkeeping only -- one pass over rows the union sweep already fetched, no new
DB queries, no new integrations.
"""

import math
from dataclasses import dataclass, field


@dataclass
class _Profile:
    """One tracklet's evidence across the variant fan."""
    ids: frozenset
    tag: str = ""
    night: int = 0                      # UTC day number of the tracklet
    by_lov_sigma: dict = field(default_factory=dict)   # k -> (qual, dist, category)
    tr: dict = None                     # best-qual candidate dict (transient)
    # finalized:
    best_lov_sigma: float = 0.0                  # refined argmin-qual LOV offset
    rate_resid_min: float = float("inf")
    n_served: int = 0
    lov_sigma_width: float = 0.0                # span of ks with qual <= 2*min + 0.5"
    sharp: bool = False
    # best P(match) over the configured ranker(s), scored against the profile's
    # own best window; None when scoring is unavailable (scan path, no model)
    ml_p: float = None
    # -- lov_refine.py (the continuous LOV offset) --------------------------
    # The tracklet's own endpoints, ``((epoch_sec, stn, ra, dec), ...)`` first
    # and last -- enough to re-measure it against ANY orbit without going back
    # to the database, which is what lets k be solved for rather than read off
    # the fan.  Kept (four small tuples) where ``tr`` is not: the scoring pass
    # drops the candidate dicts, and 30k of those is memory this search has
    # been bitten by before.
    ends: tuple = None
    best_served_lov_sigma: float = 0.0                 # variant that served it best (its qual)
    lov_refine_resid: float = float("inf")   # arcsec off the LOV curve at k*


@dataclass
class _LovRound:
    """What one PRIORITY variant's round consumes: this variant's k, the
    sharp candidates centered on it (best scored-P first), and the proposed
    cluster trials (each a set of member ids).  Only priority variants get a
    _LovRound at all -- the trial sources and the backtracking they license are spend,
    and spend must stay targeted (a trial source in all 55 rounds burned the whole
    search budget by variant 3, measured live)."""
    k: float
    center_ids: list = field(default_factory=list)     # [frozenset, ...]
    cluster_groups: list = field(default_factory=list) # [set(frozenset), ...]


def _parabolic_best_lov_sigma(pts):
    """Refined argmin over ``[(k, qual), ...]`` sorted by k: the vertex of the
    parabola through the discrete minimum and its two k-neighbours, clamped to
    that bracket; the discrete argmin when fewer than 3 points exist or the
    parabola degenerates (equal quals, minimum at an edge)."""
    j = min(range(len(pts)), key=lambda i: pts[i][1])
    if not (1 <= j < len(pts) - 1):
        return pts[j][0]
    (k0, q0), (k1, q1), (k2, q2) = pts[j - 1], pts[j], pts[j + 1]
    denom = (k1 - k0) * (q1 - q2) - (k1 - k2) * (q1 - q0)
    if abs(denom) < 1e-12:
        return k1
    k = k1 - 0.5 * ((k1 - k0) ** 2 * (q1 - q2)
                    - (k1 - k2) ** 2 * (q1 - q0)) / denom
    return min(max(k, k0), k2)


class LovProfileSet:
    """Per-tracklet LOV profiles over one variant scan's union sweep, plus the
    variant ordering and cluster proposals derived from them."""

    def __init__(self, cfg, lov_sigmas):
        self.cfg = cfg
        self.lov_sigmas = [float(k) for k in lov_sigmas]
        self._nominal_lov_sigma = min(self.lov_sigmas, key=abs) if self.lov_sigmas else None
        self.profiles = {}                 # ids -> _Profile
        self._rounds = {}                  # k -> _LovRound (priority ks only)
        self._priority_lov_sigmas = []             # best-candidate order

    def _lov_sigma_informative(self, p):
        """True when the profile carries actual LOV information: the
        candidate was NEVER served by the nominal variant's corridor, so its
        position is a real displacement from the nominal track.

        This is the gate that keeps the ranker's darlings out of the priority
        slots.  A candidate inside the nominal corridor reads as a plain
        continuation of the seed arc and scores the highest P of anything in
        the field (0.91-0.97 measured live) -- but the linear pass already
        swept the nominal corridor and failed, and such a candidate's k* is
        an accident of window overlap, not evidence.  Epoch distance is NOT
        the right gate: the BE559 truth is 24 days from its seed nights (a
        same-apparition extension whose take collapses U) yet served only by
        two high-k variants -- displacement from the nominal track, not time,
        is what makes a k* mean something."""
        return self._nominal_lov_sigma is not None and self._nominal_lov_sigma not in p.by_lov_sigma

    # -- build ---------------------------------------------------------------
    def add(self, lov_sigma, tr):
        """Record one scored candidate (search._score_tracklet dict) as served
        by the variant at ``lov_sigma``, keeping its best (qual, dist) per variant."""
        p = self.profiles.get(tr["_ids"])
        if p is None:
            from .trace import tracklet_tag
            rows = tr.get("rows") or ()
            night = int(rows[0]["epoch_sec"] // 86400.0) if rows else 0
            p = _Profile(ids=tr["_ids"], tag=tracklet_tag(tr), night=night)
            self.profiles[tr["_ids"]] = p
        lov_sigma = float(lov_sigma)
        prev = p.by_lov_sigma.get(lov_sigma)
        if prev is None or tr["qual"] < prev[0]:
            p.by_lov_sigma[lov_sigma] = (tr["qual"], tr["dist"], tr["traj_category"])
        if p.tr is None or tr["qual"] < p.tr["qual"]:
            p.tr = tr                  # best realization: the ML scoring input
            p.best_served_lov_sigma = lov_sigma
            rows = tr.get("rows") or ()
            # A row that cannot be re-measured (a test double, a source that
            # carries no sky position) simply forgoes the continuous LOV offset -- the
            # refinement skips it and the fan's k* stands.
            ends = tuple((r.get("epoch_sec"), r.get("stn"), r.get("ra"),
                          r.get("dec")) for r in (rows[0], rows[-1])) \
                if rows else ()
            p.ends = (ends if ends and all(v is not None
                                           for e in ends for v in e)
                      else None)

    def finalize(self):
        """Compute k*, width and sharpness per profile, then the per-variant
        rounds and visit-order scores."""
        self._finalize_stats()
        self._build_rounds()
        return self

    def _finalize_stats(self):
        cfg = self.cfg
        for p in self.profiles.values():
            pts = sorted((k, qd[0]) for k, qd in p.by_lov_sigma.items())
            p.n_served = len(pts)
            p.rate_resid_min = min(q for _k, q in pts)
            p.best_lov_sigma = _parabolic_best_lov_sigma(pts)
            # The good-k region is where the tracklet would be TRUE: qual both
            # near its own minimum AND small absolutely.  The absolute cap
            # matters -- without it a 2-variant profile whose second point is
            # merely mediocre (fan-spacing-limited serving, the BE559 truth's
            # own shape) reads as "wide" and loses its sharpness to the gap
            # between pickets rather than to any property of the tracklet.
            bound = min(2.0 * p.rate_resid_min + 0.5, cfg.lov_profile_rate_resid_max)
            good = [k for k, q in pts if q <= bound]
            p.lov_sigma_width = (max(good) - min(good)) if good else 0.0
            p.sharp = (p.rate_resid_min <= cfg.lov_profile_rate_resid_max
                       and p.n_served <= cfg.lov_profile_max_served
                       and p.lov_sigma_width <= cfg.lov_profile_width_max)

    def score_ml(self, ctx, eph_trial, extrawidth=0.0):
        """P(match) for every sharp profile's best realization, under the
        configured ranker(s) -- max over primary + second-opinion model, so a
        candidate one model is blind to still counts as decent.

        This is what keeps the k-evidence from being swamped on LIVE data:
        sharpness alone admits ~3.7k junk tracklets on a BE559-class corridor
        (the feasibility's 3.6% junk rate, in absolute numbers), so counting
        sharp candidates ranks variants by junk density.  The intended metric
        is "2+ DECENT tracklets agreeing" -- and decent means probability,
        which the profile pass can score from the feature rows the union sweep
        already fetched.  Degrades to a no-op (profiles keep ml_p None and the
        ordering falls back to quality mass) on the scan path or a missing
        model runtime."""
        cfg = self.cfg
        if not (cfg.ranking_mode == "ml" and cfg.candidate_path == "index"):
            return
        sharp = [p for p in self.profiles.values()
                 if p.sharp and p.tr is not None]
        if not sharp:
            return
        try:
            from .training import score_pair, feature_uncert_mult
            trs = [p.tr for p in sharp]
            probs, probs2 = score_pair(trs, eph_trial, 0,
                                       feature_uncert_mult(ctx),
                                       extrawidth, cfg, is_variant=True,
                                       v4ctx=(True,
                                              [p.best_served_lov_sigma for p in sharp],
                                              getattr(ctx, "lov_feat", None)))
            for p, pr in zip(sharp, probs):
                p.ml_p = max(p.ml_p or 0.0, float(pr))
            if probs2 is not None:
                for p, pr in zip(sharp, probs2):
                    p.ml_p = max(p.ml_p or 0.0, float(pr))
        except Exception as e:
            ctx.log(f"LOV-profile ML scoring unavailable "
                    f"({type(e).__name__}: {e}); ordering by quality only")

    def _nearest_lov_sigma(self, best_lov_sigma):
        return min(self.lov_sigmas, key=lambda k: abs(k - best_lov_sigma)) if self.lov_sigmas else None

    def _build_rounds(self):
        """Pick the PRIORITY variants and build their rounds.

        Two aggregate weightings were tried and both lost to junk mass,
        measured live on BE559: counting sharp centered candidates ranks
        variants by junk density (~3.7k sharp junk spread over the fan), and
        summing/thresholding scored P does too (junk that some model scores
        >= 0.15 is common; the junkiest variants collect the most "decent
        nights").  What junk mass CANNOT out-vote is a global top-N: the
        priority variants are the homes of the top ``lov_profile_priority_rounds``
        sharp candidates by scored P.  Those few rounds get the trial sources, the
        early visits and (in search._iterate) the backtracking; every other
        round runs at exactly baseline cost -- the 55x-round spend of a
        trial-sources-everywhere scan burned the entire search budget by variant 3.
        """
        cfg = self.cfg
        by_variant = {}                    # k -> [k-informative sharp profiles]
        for p in self.profiles.values():
            if not p.sharp or not self._lov_sigma_informative(p):
                continue
            k = self._nearest_lov_sigma(p.best_lov_sigma)
            if k is not None:
                by_variant.setdefault(k, []).append(p)
        scored = any(p.ml_p is not None
                     for ps in by_variant.values() for p in ps)
        # Best-first: probability, then quality.  A thin corridor is fitted
        # better by junk than by truth (0.02" junk vs the BE559 truth's
        # 1.35"), so qual alone hands the take race to the wrong candidate;
        # unscored (scan path / no model) it is all that is left.
        #
        # ...weighted by the LOV PRIOR at the candidate's own k*.  The ranker
        # scores how much a tracklet LOOKS like the object; it cannot know how
        # probable the LOV offset it would need is, and on a live corridor the
        # far tails are where the junk mass sits -- they sweep the most sky.
        # With the s2 covariance floor k is an honest Gaussian coordinate
        # (26 ITF fixtures: median |k*| 0.67, max 2.72), so P(truth at k*) is
        # ~exp(-k*^2/2): a k=4.7 home carries ~1e-5 of a core home's prior.
        # Unweighted, 2021 SA70 spent its priority rounds at k -4.72/-5.00/
        # -3.16/-4.02 (junk P ~0.6-0.9) while its truth sat at k -0.2.
        # Multiplying keeps the evidence ordering among comparable ks and only
        # demotes a nomination whose k the covariance says is implausible.
        psig = float(getattr(cfg, "lov_profile_prior_sigma", 0.0) or 0.0)
        prior = ((lambda p: math.exp(-0.5 * (p.best_lov_sigma / psig) ** 2))
                 if psig > 0 else (lambda p: 1.0))
        rank = lambda p: (-((p.ml_p or 0.0) * prior(p)), p.rate_resid_min, p.tag)
        top = sorted((p for ps in by_variant.values() for p in ps), key=rank)
        if scored:
            top = [p for p in top if (p.ml_p or 0.0) >= cfg.lov_profile_min_ml]
        top = top[:max(int(cfg.lov_profile_priority_rounds), 0)]
        # priority ks in best-candidate order (a variant hosting two top
        # candidates keeps its best position)
        priority_lov_sigmas = []
        for p in top:
            k = self._nearest_lov_sigma(p.best_lov_sigma)
            if k is not None and k not in priority_lov_sigmas:
                priority_lov_sigmas.append(k)
        self._priority_lov_sigmas = priority_lov_sigmas
        # A cluster trial is a JOINT hypothesis and faces the warm screen as a
        # whole: measured live, a 9-member cluster (8 junk k-colliders + the
        # BE559 truth) destroys the parent arc and is screened out before the
        # truth inside it is ever fitted.  Small clusters of the best members
        # are the only shape that can survive; the LOV-centered singles trial source
        # carries the rest of the membership individually anyway.
        max_members = 4
        self._rounds = {}
        for k in priority_lov_sigmas:
            ps = sorted(by_variant.get(k, ()), key=rank)
            rnd = _LovRound(k=k, center_ids=[p.ids for p in ps])
            # One cluster proposal per variant: the best member PER NIGHT,
            # >= 2 distinct nights.  Per-night, because the cluster's whole
            # evidence is nights voting for the same k -- and because a cap
            # applied to a mixed list can collapse to one night: measured
            # live, four same-night near-seed tracklets in one joint trial
            # made a degenerate fit that ground LM for 851 s.  The joint fit
            # and the unchanged gates dispose of chance collisions.
            best_by_night = {}
            for p in ps:                        # ps is best-first
                best_by_night.setdefault(p.night, p)
            members = sorted(best_by_night.values(), key=rank)[:max_members]
            if len(members) >= 2:
                rnd.cluster_groups.append({p.ids for p in members})
            self._rounds[k] = rnd

    # -- consume -------------------------------------------------------------
    def reorder(self, pervar):
        """2a: the PRIORITY variants first (best-candidate order), the rest
        central-out unchanged -- so a fan with no priority candidates keeps
        today's exact order."""
        pos = {k: i for i, k in enumerate(self._priority_lov_sigmas)}
        def key(entry):
            return (pos.get(float(entry[0]), len(pos)), abs(entry[0]))
        return sorted(pervar, key=key)

    def round_for(self, lov_sigma):
        """The _LovRound for the variant at ``lov_sigma`` (None when no sharp candidate centers
        there)."""
        return self._rounds.get(float(lov_sigma))

    # -- report --------------------------------------------------------------
    def summary(self, top=8):
        """Compact dict for the trace/telemetry: profile counts, the predicted
        best variants, the sharpest candidates, and the cluster proposals."""
        sharp = [p for p in self.profiles.values() if p.sharp]
        return {
            "n_tracklets": len(self.profiles),
            "n_sharp": len(sharp),
            "variants": [{"k": round(k, 3),
                          "n_centered": len(self._rounds[k].center_ids)}
                         for k in self._priority_lov_sigmas[:top]],
            "sharp": [{"tag": p.tag, "kstar": round(p.best_lov_sigma, 3),
                       "qual": round(p.rate_resid_min, 3), "served": p.n_served,
                       "width": round(p.lov_sigma_width, 3),
                       "ml_p": (round(p.ml_p, 4)
                                if p.ml_p is not None else None)}
                      for p in sorted(
                          sharp, key=lambda p: (-(p.ml_p or 0.0), p.rate_resid_min,
                                                p.tag))[:top]],
            "clusters": [{"k": round(k, 3),
                          "n_members": len(next(iter(r.cluster_groups))),
                          "tags": sorted(self.profiles[i].tag
                                         for i in next(iter(r.cluster_groups)))}
                         for k, r in sorted(self._rounds.items(),
                                            key=lambda kv: abs(kv[0]))
                         if r.cluster_groups][:top],
        }


def build_lov_profiles(ctx, pervar, cached, lov_sigmas, eph_trial=None, extrawidth=0.0):
    """One pass over the union-sweep cache -> a finalized :class:`LovProfileSet`.

    Reuses the scan's own :func:`~pyroorb.itf.search._score_tracklet` (lazy import
    -- search imports this module) so the profile's qual/dist/category can
    never drift from what the per-variant rounds will measure.  Serves every
    (variant k, window) pair from ``cached`` (the union sweep's in-memory
    rows); windows the sweep skipped simply contribute nothing.  With
    ``eph_trial`` the sharp profiles are ML-scored (:meth:`score_ml`) before
    the rounds and visit order are derived.
    """
    from .candidates import _mem_available_mb
    from .search import _group_rows, _score_tracklet

    cfg = ctx.cfg
    lov_profiles = LovProfileSet(cfg, lov_sigmas)
    n_windows = 0
    for k, vwins in pervar:
        for window in vwins:
            # The profile churn runs in whatever memory the union sweep left
            # behind, and on a monster union that can be nearly nothing (2006
            # SJ252: sweep ends ~550 MB free, this pass then OOM-killed the
            # process in seconds).  Bail onto the designed degrade path --
            # _variant_scan catches this and scans without profiles.
            n_windows += 1
            if n_windows % 512 == 0:
                from .memory import sweep_floor_mb
                floor = sweep_floor_mb(getattr(cfg, "sweep_min_free_mb", 0))
                free = _mem_available_mb() if floor else None
                if free is not None and free < floor:
                    raise MemoryError(
                        f"available memory {free} MB is under the {floor} MB "
                        f"floor with {n_windows} windows profiled")
            hit = cached.lookup(window)
            if hit is None:
                continue
            _window, rows, chi2_by_trkid, feats_by_trkid = hit
            for t in _group_rows(rows):
                tr = _score_tracklet(ctx, t, window, cfg.likely_arcsec,
                                     ctx.pub_ids, chi2_by_trkid,
                                     feats_by_trkid)
                if tr is None or tr["duplicate"]:
                    continue
                lov_profiles.add(k, tr)
    lov_profiles._finalize_stats()
    if eph_trial is not None:
        lov_profiles.score_ml(ctx, eph_trial, extrawidth)
    lov_profiles._build_rounds()
    # the per-candidate dicts have served their purpose (stats + scoring);
    # holding 30k of them for the scan's lifetime is memory the search has
    # been bitten by before
    for p in lov_profiles.profiles.values():
        p.tr = None
    return lov_profiles
