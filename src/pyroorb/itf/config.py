"""Configuration for the native ITF search.

One ``ItfConfig`` carries every switch and tunable of the search pipeline
(ephemeris corridor, candidate retrieval, scoring, acceptance gates, database,
submission identity).  The tracklet-index maintenance has its own
``IndexConfig`` (see :mod:`pyroorb.itf.index`).

Defaults mirror the proven find_orb-era pipeline (the ``ift outline`` notebooks)
where a value was validated there, and PyRoOrb-native choices elsewhere.
"""

import os
from dataclasses import dataclass, field

# the RETIRED pair's second-opinion artifact (training.default_second_model_path
# carries the same path; keep the two in step) -- not a default since v4
_V2_MODEL = os.path.join(os.path.dirname(__file__), "training_data",
                         "itf_match_v2.json")


def _env(name, fallback):
    """Env-overridable default (exported var first, gitignored ``.env`` as
    fallback -- :func:`itf.dbio.env_default`), evaluated at construction."""
    from .dbio import env_default
    return env_default(name, fallback)


PREFER_CHOICES = ("complete", "submittable")


def default_prefer(submit):
    """The ``prefer`` a command-line search uses when none is given: a search
    that transmits wants the linkage it can send (a 0.6 is never sent), any
    other wants the fuller result to review."""
    return "submittable" if submit else "complete"


@dataclass
class ItfConfig:
    # ---- ephemeris corridor ------------------------------------------------
    # Geocentric (obscode 500) ephemeris sweep for the corridor search. The
    # legacy pipeline used 5200 x 2d steps from 1997-01-02; steps=None extends
    # the sweep to ~30 days past today automatically.
    ephem_start: str = "1997-01-02"
    # 3 d (not the legacy 2 d): fewer nodes + windows speed EVERY sweep ~1.6x.
    # A window is a quadratic through 3 nodes, so 3 d = 12 d wide; the cubic
    # residual 6 d from the mid-node is arcsec-level for a smooth main-belt orbit
    # (negligible vs the >=10" corridor) and the 7deg fast-mover split refines the
    # high-curvature near-opposition windows.  A/B FULL 16/16 at 3 d and 4 d; kept
    # 3 d (4 d's 16 d windows stretch the quadratic across opposition reversals).
    # LIVE precision unmeasured offline.
    ephem_step_days: float = 3.0
    ephem_steps: int | None = None
    # The era BEFORE ephem_start, swept once at the post-acceptance final sweep
    # (search._archive_tracklets) -- otherwise no window ever exists there and
    # the 7198 tracklets our index holds before 1997 (back to 1902) are
    # unreachable by any orbit at any depth (2022 SN92's 1995 Spacewatch
    # precovery).  1980 is where the mass is: 6573 of those 7198, against 3x
    # the ephemeris nodes to reach the 570 that predate 1978.  "" disables.
    archive_start: str = "1980-01-01"
    # ...and how far back the ONE final sweep reaches.  Deeper than
    # archive_start because it is paid once per accepted linkage instead of on
    # every sweep: 1950 costs 1.76 s against 1980's 0.65 s and reaches 541 of
    # the 625 tracklets the index holds before 1980 (2020 OG4/756769 needs
    # 1974-06-22).  Earlier still buys only 84 more.  "" falls back to
    # archive_start.
    archive_deep_start: str = "1950-01-01"
    # ...and a seed that STARTS at or below this U does not wait for the
    # final sweep: its backward corridor is already thin, so the archive windows
    # ride in EVERY ordinary sweep and an old tracklet can link and be recursed
    # on like any other.  0 disables the early pass,
    # leaving the final sweep one.
    archive_seed_U: float = 6.48   # was 6.0; +0.48 at U 6.0 -- see U-SHIFT note
    # Skip a window when the predicted sky-plane uncertainty (deg, 1-sigma
    # semi-major axis + topocentric parallax allowance) exceeds this. The driver
    # passes the per-search value; this is the default.
    #
    # 40 deg, measured (the former 19.44 deg = 70000" had no basis).  Seeding 294
    # objects at 2-7 oppositions on their most compact 3 nights, the truth stays
    # inside the 1.5-sigma corridor 90-100% of the time in every sigma bin above
    # 10 deg once the seed's own apparition is fitted -- the linear sigma
    # overstates the miss (median 0.1-0.3 sigma) -- so geometry sets no ceiling;
    # cost and search behaviour do.  Share of objects with a cross-opposition
    # tracklet below the cap after that extension: 93% at 19.44, 95% at 30, 96% at
    # 40, 98% at 60.  Live end-to-end recoveries of 37 objects (17 the old cap
    # blocked): 24 full at 19.44, 28 at 35, 40 and 45, no tracklet lost, no foreign
    # link; recall plateaus from 35, cost rises with the cap (controls +27% at 40,
    # +32% at 45), so 40 is the plateau with margin.  On 40 fresh random objects
    # recall is identical to 19.44 at +8% wall.  Above ~40 the +-1.5-sigma box spans
    # more than +-60 deg and stops carrying information.
    max_uncert_deg: float = 40.0
    # Scale the 1-sigma corridor half-width by this before matching tracklets.
    # 2.5 (not 1.1): a single-opposition seed's linear covariance UNDER-predicts
    # its true multi-year extrapolation error, and the miss is ~100% along-track
    # (period/mean-anomaly is the weak direction). A 62-tracklet calibration on
    # 2-3-opp objects put the true position at required-mult median 0.54 but a
    # fat along-track tail (p90 1.44, max 2.60); 1.1 admits 79%, 2.5 admits 95%,
    # 3.0 admits 100%. Free for well-constrained seeds (their corridor is ~0.006deg,
    # so 2.5x is ~0.015deg -- nothing); the width only grows where the seed is weak,
    # exactly where we need it. The max_uncert_deg window gate + sweep_row_budget still
    # bound the absolute corridor. UNVALIDATED against live false-link rate.
    # 1.5 SINCE THE s2 COVARIANCE FLOOR (2026-09-22).  Everything above was
    # measured on the pre-floor covariance, which ran ~2-2.5x too tight on a
    # seed, so "2.5 admits 95%" meant ~1 HONEST sigma.  Re-measured on the
    # floored scale over 137 truth nights in 25 fixtures: needed multiple p95
    # 1.03, max 1.40.  2.5 honest sigma was ~2.9x the old corridor AREA (up to
    # 9x), all of it junk.  Same-day live suite, 26 fixtures incl. BE559/SA70:
    # 2.5 -> 25/26 in 3472 s with a FALSE status-1 link on SA70 (two foreign
    # O18 tracklets at 0.9"); 1.5 -> 25/26 in 2626 s, no foreign link, the one
    # miss (2014 HM543) silent.  Master (floored, 2.5, no LOV prior / ranker
    # scale fix) 23/26 in 5749 s.
    uncert_mult: float = 1.5
    # The multiple the rankers' corridor-WIDTH features are defined at -- the
    # v3 collection's 2.5 -- held apart from ``uncert_mult`` so the corridor can
    # be retuned without moving every model input (training.feature_uncert_mult).
    ml_feature_uncert_mult: float = 2.5
    # A window whose ephemeris nodes are further apart than this on either sky
    # axis is split in two for finer corridor fitting (fast movers).
    max_fallback_deg: float = 7.0
    # Rate-ADAPTIVE node spacing (ephem.refine_fast_segments): wherever
    # consecutive nodes are further apart on the sky than this, that stretch of
    # the ephemeris is recomputed finely enough to bring per-node motion back
    # under it.  A quadratic through three nodes cannot describe a fast mover
    # sampled every 3 days: 2015 KJ19 (650-2900"/hr) had its own tracklets'
    # motion mismatch inflated 2-17x by the interpolation and only 13 of 37
    # passed the corridor's motion gate; refined, 35+.  Costs nothing on a
    # main-belt object, whose nodes never breach the bound.  0 disables.
    # 6.0 (not 2.0): across the seven NEO/fast fixtures the two bounds recover
    # IDENTICALLY -- KJ19 38/38, 882845 33/34, the rest full -- while 6 costs
    # 40.8 s against 2's 69.8 s.  (4.0 came out at 37/38 on KJ19, which is the
    # measurement telling you the last tracklet is path-dependence noise, not a
    # bound effect.)  It also sits at the scale of max_fallback_deg's 7, the
    # window-splitting rule this complements.
    fast_node_deg: float = 6.0
    min_node_step_days: float = 0.05       # floor on the refined step (~72 min)
    fast_node_budget: int = 20000          # cap on nodes ADDED by refinement
    # Skip a window where the object is predicted FAINTER than this -- the ITF
    # cannot hold a detection there, so sweeping it is pure cost.
    #
    # 23.5, not the legacy 24.0.  Two measurements set it (2026-08-01):
    #
    #  * The ITF's OWN depth.  2.84M indexed tracklets carry a mean magnitude;
    #    the distribution is a sharp survey-depth spike at 21.5-22.5 (half of
    #    the whole file sits in that one magnitude) with a long faint tail --
    #    p50 21.98, p75 22.51, p90 23.48, p95 24.07, p99 25.14.  So 23.5 keeps
    #    ~90% of the file within reach and even 24.0 was never "everything"
    #    (5.5% of ITF tracklets are fainter than 24).
    #  * What the gate COSTS.  Corridor windows built for the 16 recovery
    #    fixtures with the gate lifted: 15018 windows ungated, 10604 at 24.0,
    #    6887 at 23.5, 3595 at 23.0, 2833 at 22.8, 2053 at 22.5.  The curve
    #    turns over right at 23 -- below it you stop sweeping the aphelion /
    #    quadrature stretches where an MBA spends most of its timeline and
    #    where the ITF has nothing to offer.  23.5 is ~65% of 24.0's windows.
    #
    # The margin is set by the ephemeris's own photometric error, not by the
    # ITF percentile: over the 143 true planted observations in those fixtures,
    # observed - predicted mag came out mean +0.00, sd 0.48 (p5..p95 -0.62,
    # +0.52), so a 23.5 gate reliably reaches ~22.5 -- past the ITF median --
    # at 2 sigma.  Predicted mag at the truth obs peaked at 23.30 (p95 22.38),
    # so 23.5 clears every one of them; 22.8 already clipped 3 (all on
    # 2015 BE559, the U 9.56 seed).
    #
    # RECALL was measured, not assumed: a live 16-fixture A/B at 24.0 vs 23.0
    # recovered IDENTICAL tracklets (45/71, the same five objects short) while
    # 23.0 ran 25% faster (878.2 s -> 658.5 s).  23.5 sits ABOVE the measured-
    # safe 23.0 and so sweeps a superset of its windows -- a more conservative
    # choice that trades some of that win for margin.  Don't drop below 23.0
    # without re-running the recall A/B: H is often poorly known for exactly
    # the short-arc seeds this search is aimed at, and the loss from a low gate
    # is CONCENTRATED (a faint object is faint at every window, so it finds
    # nothing at all rather than losing candidates piecemeal).
    mag_limit: float = 23.5
    # The deeper reach: what `--thorough` sweeps to, and where the (disabled,
    # see below) automatic fallback would re-sweep.  24.0 is the legacy limit.
    mag_limit_deep: float = 24.0
    # A FLOOR under mag_limit, per object: the cut actually applied is never
    # brighter than the brightest the object has been predicted since
    # mag_floor_since, plus mag_floor_margin.  A fixed cut does not trim a
    # faint object's corridor -- it deletes it, because a faint object is faint
    # at every window: 2026 DR69 (a 43 au TNO, brightest V 23.81 since 1997) got
    # ZERO windows at 23.5 while its two 2017 T09 tracklets (reported 23.6-24.5)
    # link at P(match) 0.999 once the corridor is swept to 24.0.  The floor only
    # ever DEEPENS the default (an object bright somewhere since 1995 sweeps
    # exactly as before), and it shapes the DEFAULT only: a cut the user typed
    # (set_mag_limit, mag_limit_explicit) is applied as given, even 23.5.
    # --thorough is a request for depth, not a cut, so the floor still applies
    # on top of it.  None turns the floor off.
    mag_floor_since: str = "1995-01-01"
    mag_floor_margin: float | None = 1.0
    mag_limit_explicit: bool = False

    def set_mag_limit(self, mag):
        """Apply a cut the USER chose (CLI --mag-limit, the web field, a
        campaign override): it is used exactly, with no per-object floor."""
        self.mag_limit = float(mag)
        self.mag_limit_explicit = True
        return self

    def apply_thorough(self):
        """Turn this config into the ``--thorough`` one, in place.

        ONE definition, shared by every surface that offers the flag
        (``itf search --thorough``, ``fit_object.py --itf-thorough``), because
        a flag meaning two different things on two CLIs is how the engine and
        its front ends drift apart.  RAISED rather than set, so a caller who
        already asked for more is never walked back; an explicit ``--mag-limit``
        applied AFTER this call still wins outright, which is the documented
        precedence.

        Thorough is depth in MAGNITUDE only.  A version of this also raised the
        trial queue's rank depth and was reverted as unproven -- if that comes back,
        it belongs here rather than in each CLI."""
        self.mag_limit = max(self.mag_limit, self.mag_limit_deep)
        return self

    # OFF, and deliberately kept as working-but-dormant code (2026-08-01).
    #
    # When enabled, a search that links NOTHING re-sweeps the corridor out to
    # mag_limit_deep before concluding anything -- the reasoning being that the
    # recall a low mag_limit gives up is CONCENTRATED rather than spread (a
    # faint object is faint at every window, so it finds nothing at all), which
    # would make retrying-only-on-empty free for everything that links.
    #
    # It is off because the measurement did not support it.  The recall it was
    # built to protect turned out not to be at risk at 23.0 (identical
    # tracklets recovered in a live 16-fixture A/B), so the retry never earned
    # its keep -- and it is not free: the branch that ran 23.0 WITH this
    # fallback took 1012.4 s against a 24.0 baseline of 974.8 s, ~4% SLOWER,
    # because an object that links nothing pays a second sweep on top of its
    # first.  The fixtures cannot settle the case for it either way, since none
    # of them is a faint object that goes dark.
    #
    # Left in place (not deleted) because the argument for it is sound and only
    # the population it protects is unmeasured: a batch over genuinely faint
    # objects is where it would pay.  Turn it on with `--mag-fallback` (any
    # surface) or by setting this True, and re-measure before trusting it.  It
    # runs BEFORE the variant-corridor fallback below -- "too faint for the
    # gate" is a more mundane explanation for an empty corridor than "the
    # ephemeris is wrong", and much the cheaper of the two to test.
    mag_limit_fallback: bool = False

    # ---- variant-orbit corridor (fallback for under-determined seeds) ------
    # The linear corridor is one nominal ephemeris, so an under-determined seed
    # retrieves NOTHING when its true position sits past the scaled corridor --
    # mildly (2014 SD57/2019 AE65 at ~2.5-3.5 sigma) or wildly (2015 BE559,
    # U 9.56, tens of degrees).  As a FALLBACK -- only after the linear pass
    # failed to LINK anything (see itf_search) -- we SCAN orbits sampled along
    # the Line of Variations (itf/lov_variants.py): each a legitimate orbit,
    # one near the truth, and linking one collapses U so the rest fall in the
    # ordinary corridor.  Gated on seed U >= this, so a well-constrained seed
    # (whose empty corridor means genuinely nothing) never pays.  0 disables.
    variant_corridor_U: float = 4.0
    # The same fan from the linear recursion's TERMINAL orbit.  A linear pass
    # that links only more of the seed's own apparition counts as LINKED, so
    # the fallback above never runs -- yet the grown orbit can still be
    # under-determined, with its linear corridor far over max_uncert_deg at the
    # other apparitions while the truth sits on its Line of Variations (2017
    # AM13 at k +0.18, 2014 TJ33 at k +0.14, 0.1-1.5 deg off the variant).  So
    # when the deepest linear level comes back without a deeper link on an
    # orbit still at U >= variant_corridor_U, the fan is scanned from THAT
    # orbit: once per search, never from inside a variant round.  Measured
    # live over 120 objects (3-night compact seeds): 97 -> 103 full
    # recoveries, 0 lost, 0 foreign links, median wall 47 -> 50 s.
    # Only from a take the terminal vet calls REVIEWABLE: fired from the first
    # take that came back, half the firings over those 120 objects started
    # from a take holding no true tracklet and spent the one shot, while every
    # fan that produced a link started from a reviewable take.  Gated, the 32
    # firing objects gain 2025 OZ629 (1/6 -> 6/6), lose nothing, and run 27%
    # faster (fans 32 -> 21).
    terminal_fan: bool = True
    # When the search holds BOTH a LINKED linkage and a reviewable 0.6 that
    # links strictly more of the ITF (the 0.6 adds oppositions too thin to
    # submit, e.g. two single-night ones -- 2014 GV99), it returns both: this
    # picks the primary and the other rides on ``ItfSearchResult.alternative``.
    # "complete" (the fuller 0.6) suits review -- the CLI, the recovery harness;
    # "submittable" (the linkage) suits a pipeline that submits -- automated senders and
    # the identification queue set it.  The web asks the user.
    prefer: str = "complete"
    # Variant COUNT is the lever on the scan's wall time (its ONE union DB sweep
    # is ~n_steps x windows-per-corridor).  31 with a wider 2.5deg width still
    # recovers SD57/AE65/BE559 while halving the sweep vs 61; max_scan bounds the
    # negative worst case (every variant swept, no early-exit).
    # LEFT AT 31 (2026-09-22), after measuring 31 -> 55 and reverting it.
    #
    # With the s2 covariance floor the fan's SPAN changes meaning: it samples
    # state0 + k*sigma*u over a fixed variant_n_sigma, so an honest (~1.8x
    # wider) sigma reaches ~1.8x further in physical terms for the same k.
    # That is what recovers 2014 HM543, whose truth sits ~8.9 sigma out on the
    # OLD scale -- i.e. ~4.9 on the new one, INSIDE the +-5 fan -- 0/5 -> 5/5.
    #
    # Raising the COUNT to hold density looks like the natural companion and is
    # NOT: the k grid is spaced by equal PROBABILITY, so extra samples land in
    # the CORE where the mass is, barely improve spacing at |k| ~ 3.5, and move
    # every picket. Measured live: 55 recovers 2015 BE559 (0/4 -> 4/4) and 2021
    # SA70 (0/5 -> 5/5) but BREAKS 2014 HY176 (3/3 -> 0/3, twice, in 30 s), whose
    # truth sits at k ~ -3.55 in the tail -- the case config's own refinement
    # note names. Raising variant_refine_max_extra to 48 does not rescue SA70
    # either (n_scan already rises with the extras, search.py:5204).
    #
    # So the fan cannot serve both reach and tail density at ANY single count
    # while the grid is equal-probability, and the honest fix is sigma-aware
    # SAMPLING (uniform-in-k or equal-sky-spacing in the tail), not a number.
    # Until that exists, 31 is kept: it is master's value, it keeps HY176, and
    # the floor's cost is the two objects named above -- documented, not hidden.
    #
    # SUPERSEDED DIAGNOSIS -- read this before acting on the paragraph above.  The next day showed the fan was NEVER why BE559 and
    # SA70 failed: ranker train/serve skew, un-weighted priority variants, the
    # 0.426 probability rescale and the 2.5 corridor were, and with those fixed
    # all four of HM543 / BE559 / SA70 / HY176 recover live on this 31-step grid.
    # The one fan failure found since (2005 UD562) was a CORE hole,
    # fixed by refinement budget (variant_refine_*), not a tail one.  Sigma-aware
    # sampling stays a research idea, not the known fix for a known miss.
    variant_n_steps: int = 31                  # LOV samples (central-out scan order)
    variant_n_sigma: float = 5.0               # LOV span sampled, in sigma
    variant_width_deg: float = 2.5             # thin fixed per-variant corridor half-width
    variant_max_scan: int = 31                 # cap on variants actually scanned (early-exit)
    # Sweep the fan in probability-order TIERS of this many variants (gaps
    # #73a): sweep a tier, profile everything swept so far, scan it, and only
    # then sweep the next -- so a link in the core no longer pays for the
    # tails.  A spent budget still sweeps the remaining tiers (as setup) so the
    # continuous LOV offset last resort reads the whole fan.  0 = ONE union sweep, the
    # behaviour before #73a.  24, on a paired live A/B (2026-09-25, one run at a
    # time, identical recoveries throughout): UD562 705 -> 180 s, HE349 564 ->
    # 330 s, BE559 406 -> 373 s, SA70 204 -> 176 s, HM543 322 -> 333 s, and
    # UM18 -- which links only through the continuous LOV offset retry, i.e. after the
    # whole fan -- still links (3/3, 1422 s).
    variant_sweep_tier: int = 24
    # Widen each variant's corridor to half the sky gap to its NEIGHBOURING
    # variant, so the fan is a contiguous search of the LOV instead of a picket
    # fence with the object between the pickets.  The k grid is spaced by equal
    # PROBABILITY, which piles samples up near the nominal and fans them apart
    # in the tails -- at the default 31 steps, judged at 2010, 2025 NV385's
    # adjacent variants sat a median 19.3 deg apart against the 5 deg it takes
    # to touch, i.e. 0% of its LOV was actually searched (2026 EP114 4.2%,
    # 2025 YQ13 76.7%, 2026 EX121 93.3% -- the four searches' outcomes rank in
    # exactly that order).  Densifying the grid instead costs a ~15x bigger
    # union sweep against a shared server for the same coverage.  Off restores
    # the fixed-width picket fence.  See itf/lov_variants._neighbour_half_gaps.
    # DEFAULT OFF, on measurement.  The coverage defect below is real, but the
    # cure costs more than the disease: NV385's variant sweep went from a
    # COMPLETED 465 s (fixed width) to >5400 s WITHOUT FINISHING even capped at
    # 2x -- 90 min, zero statement cancellations, so every query was individually
    # healthy and it is simply ~11x more work.  A corridor that cannot finish
    # recovers nothing, whatever its coverage.  Turn on deliberately (and expect
    # a long sweep); turning it on by default should wait until the chunk sweep
    # itself is cheaper.
    variant_fill_gaps: bool = False
    # ...but BOUNDED, because the widening is paid for in query cost.  Uncapped
    # (the first cut of this) each node could reach max_uncert_deg = 19.4 deg,
    # and NV385's union sweep went from ~498 healpix pixels per window to ~7630
    # -- a single chunk query that had taken 465 s had not returned after 45
    # MINUTES against the shared server.  The coverage/cost curve says take 2x:
    #
    #   cap    NV385 cov (2020/2010)   EP114            EX121      mean width
    #   1x W    26.7% /  0.0%          8.3% /  4.2%   96.7%/93.3%   x1.0
    #   2x W    60.0% / 30.0%         37.5% /  4.2%  100.0%/100.0%  x1.0-1.9
    #   4x W   100.0% / 50.0%         66.7% / 20.8%  100.0%/100.0%  x2.6-3.3
    #   19.4    100.0% / 86.7%        87.5% / 62.5%  100.0%/100.0%  x3.4-4.9
    #
    # 2x buys the whole EX121-class win for NOTHING (its gaps are small, so the
    # widths barely move: mean x1.0) and still lifts NV385 off zero, while the
    # tail of the curve is where the query blows up.  Raise deliberately for a
    # thorough run; 0 means no cap (the original unbounded behaviour).
    variant_fill_max_mult: float = 2.0
    # ---- adaptive fan refinement -------------------------------------------
    # The picket fence has two cures and they cost differently.  WIDENING each
    # corridor (above) is paid in healpix pixels PER WINDOW and blows up
    # superlinearly -- NV385 went 465 s -> >5400 s without finishing.  ADDING
    # variants is paid linearly in windows (481 steps = ~15x the sweep for
    # 15.5x the variants), so a BOUNDED number of extra samples is affordable
    # where a uniformly denser grid is not.  The question is only where to put
    # them, and equal-probability spacing answers it badly: every interval
    # carries the same ~1/n of the truth's probability while the sky gaps
    # between adjacent variants range over two orders of magnitude (1 deg in
    # the core, 23-70 deg in a close-approach tail).
    #
    # So place them by what a hole actually COSTS the search:
    #
    #     P(miss | interval) ~ dp_i * (1 - coverage_i)
    #     coverage_i = mean over searchable epochs of min(1, 2W / skygap_i(t))
    #
    # and greedily bisect the interval with the largest uncovered probability
    # (itf/lov_variants.refine_variant_fan).  Splitting halves an interval's
    # probability mass, so a monstrous but improbable gap stops earning splits
    # after a step or two while a modest gap carrying real probability keeps
    # them -- the budget lands where it buys the most recall per window.
    #
    # SELF-LIMITING BY CONSTRUCTION: a fan whose corridors already touch scores
    # zero uncovered probability everywhere, buys nothing, and spends nothing
    # -- no extra states, no extra integration, no extra windows.  The whole
    # cost falls on the seeds that are actually failing, which is the point.
    # 2014 HY176 is the type case: a 2014 seed (U 7.39) propagated back 14
    # years to a 2000 apparition at 0.11-0.15 au, where the truth sits at
    # k ~ -3.55 between samples at -3.27 and -3.88 that are 23 deg apart on
    # the sky against a 5 deg reach.  0 disables (the plain equal-probability
    # fan).
    #
    # 96 / 8 rounds (was 24 / 3 until 2026-09-23).  The s2 covariance floor
    # stretched the fan's SKY spacing by the floor factor, and a weak seed whose
    # fan used to touch now leaves most of its probability between pickets:
    # 2005 UD562 (12 DECam obs over 4 d at 0.047", floor factor 12.8, U 11.17)
    # left 74% uncovered, 65% after 24 extras, and its truth -- at k ~ 0.02, the
    # CORE, not a tail -- sat in the hole between k = 0 and +-0.095 on every
    # night of its 2005-2026 arc.  The linear pass cannot catch it (every window
    # past ~5 months exceeds max_uncert_deg), so the search spent its budget
    # linking foreign tracklets and ended 0/5.  At 96 the core pitch halves to
    # 0.024 and it links 5/5 at variant #2 (k = -0.02); 24 -> 48 does not move
    # the pitch (0.048 -> 0.047) because a round bisects every costly interval
    # at once, so it is ROUNDS that buy core density, not the cap alone.
    #
    # Self-limiting held on the fixture population: over all 26 ITF recovery
    # seeds the fan is BYTE-IDENTICAL at 24 and 96 for 23 of them (HY176, the
    # tail case, stops at 40 variants either way); only HM543 (x1.16 union-sweep
    # windows), BE559 (x1.36) and UD562 (x2.18) buy more, and all three recover
    # live.  The cost is concentrated on exactly the seeds whose fans leak.
    variant_refine_max_extra: int = 96          # hard cap on ADDED variants
    variant_refine_rounds: int = 8              # bisection passes (1 integration each)
    # Stop refining an interval once its uncovered probability falls below
    # this: below ~0.5% the next window buys less recall than it costs, and
    # without a floor the greedy loop would spend the whole budget shaving
    # slivers off the core where the corridors already overlap.
    variant_refine_min_gain: float = 0.005
    # A seed at U >= this first scans ONE variant at the population model's peak on the
    # LOV (lov_variants.prior_variant), before the fan's union sweep; a link there ends
    # the search, anything else leaves the fan to run as it would have.  Below U 9 the
    # peak is the nominal on nearly every seed measured, which the fan tries anyway.
    # Measured against 0 on the 43 ITF recovery fixtures: the variant forms on 7, links
    # first on 2 (2008 SZ347 0/15 -> 15/15; 2005 UD562 the fan's own link, 4.6x sooner
    # against the live ITF), and leaves the other 5 unchanged in tracklets and rms, live
    # distractors included, at 0-15% more time; no foreign tracklet either way.
    # 0 disables.
    variant_prior_min_U: float = 9.0
    # Price a hole with the pre-floor 0.426 k-rescale instead of the honest
    # Phi(k) (variants._lov_prob_scale explains why that factor was wrong and
    # why it used to cancel).  A/B switch only.
    variant_prob_legacy: bool = False

    # ---- untaken-candidate carry (search._carried_pool) -----------------------
    # A candidate is only ever seen through the corridor of the orbit that was
    # current when it was swept, and LINKING one tightens that corridor -- so a
    # tracklet the search retrieved, ranked and declined can become invisible
    # precisely because the search got better.  2016 TJ194: a 2015 seed (U 7.95)
    # served all 7 of its tracklets, linking the two 2016 ones took U to 4.64,
    # and the 2025/2007 tracklets then sat 782-1052" outside a +-394" corridor
    # (the 2-opposition covariance under-predicts a 9-year extrapolation by
    # 5.9x).  They fit that improved orbit perfectly -- 45/45 obs at 0.234".
    #
    # So untaken candidates ride down the recursion and are re-scored against
    # each improved orbit.  Free of database cost (the rows are already in
    # hand) and self-limiting (a taken tracklet leaves the pool as a duplicate),
    # but it must not become a memory cost: a wide corridor can retrieve tens of
    # thousands of candidates, so the pool is ranked best-first and truncated.
    # 0 disables the carry entirely.
    carried_max: int = 200

    # ---- LOV profiles: mutual consistency before fitting -----------------
    # (clustering-idea.md.)  The variant scan asks, per variant, "does any
    # tracklet fit THIS point on the LOV?" and throws the cross-variant
    # structure away.  Keeping it gives every candidate the union sweep served
    # a LOV PROFILE -- which LOV offset would make it true (k* = argmin of the
    # motion-mismatch qual over the variants that served it), and how sharply.
    # Measured on a full instrumented 2015 BE559 run: the decisive 2024 truth
    # is served by 2 of 55 variants with a sharp minimum at exactly the
    # recovering k (qual 1.35"), only 3.6% of 103k junk tracklets share that
    # profile SHAPE, and only 0.5% collide at the same k -- a ~200x pool
    # reduction computed from data the scan already holds (no new DB queries,
    # no new integrations; one dict pass over the cached union sweep).
    # Master switch: build the profiles and narrate them (report-only unless
    # the consumers below are also on).
    # ON BY DEFAULT since 2026-08-07, as the whole flag set below was
    # validated together: live matrix BE559/GL24/WY29 x (v2, v3e+v2) all
    # LINKED (BE559 under v3e+v2 for the first time ever); full 24-fixture
    # live suite 24/24 flags-on AND 24/24 flags-off (the trial queue refactor
    # regresses nothing).  BE559 under v3e-ONLY remains a miss -- a single
    # blind ranker leaves no signal to order by; reliability comes from
    # ranker diversity + backtracking (see ladder-backtracking.md).
    lov_profile: bool = True
    # 2a: visit variants where the k-evidence mass is, not central-out.  Each
    # variant is scored by the sharp, good-qual candidates whose k* centers on
    # it -- with agreement ACROSS DISTINCT NIGHTS ranked above any single
    # candidate's quality (two nights voting for the same k is far stronger
    # evidence than either fitting alone).  Pure
    # reprioritization of the existing scan: no take semantics change, no new
    # fits.  BE559's recovering k=+2.83 stops being visit #37 of 55.
    lov_profile_order: bool = True
    # 2b: within a round, candidates whose k* CENTERS on this variant outrank
    # ones that merely straddle it (a broad profile is the signature of junk
    # that fits everywhere a little).  Implemented as up to this many extra
    # singles trials at the FRONT of the trial queue -- ahead of the joint batches,
    # because the failure this exists for is the take RACE: on a weak seed
    # junk fits cleanly, so whoever is fitted first wins the round.  0 = off.
    lov_profile_center_singles: int = 2
    # 2c: where >= 2 sharp candidates on distinct nights put their k* on the
    # same variant, try them TOGETHER as one joint trial at the front of that
    # variant's trial queue (mutual consistency under the seed's own covariance,
    # before any trial fit; the acceptance gates remain the arbiter).  Up to
    # this many cluster trials per round.  0 = off.
    lov_profile_cluster_trials: int = 2
    # Profile-shape gates (the measured feasibility operating point): a
    # candidate is SHARP when its best qual clears lov_profile_rate_resid_max, it was
    # served into few enough variants (lov_profile_max_served), and the k-span of
    # its good region (qual <= 2*min + 0.5") stays inside lov_profile_width_max
    # LOV-sigma.  The 3.6% / 0.5% junk numbers above were measured at
    # qual <= 2", served <= 3, span <= 0.5 sigma.
    lov_profile_rate_resid_max: float = 2.0
    lov_profile_max_served: int = 3
    lov_profile_width_max: float = 0.5
    # PRIORITY variants: the homes of the global top-N sharp candidates by
    # scored P (max over primary + second model; lov_profile.score_ml, over
    # feature rows already in memory).  Only these rounds get the trial sources, the
    # early visits and the backtracking -- both aggregate weightings
    # (candidate counts and P sums/thresholds per variant) were measured to
    # lose to junk mass on live corridors (~3.7k sharp junk over the fan),
    # and trial sources in ALL rounds burned the entire search budget by variant 3.
    # A global top-N is the shape junk mass cannot out-vote, and it bounds
    # the spend to a handful of rounds.
    lov_profile_priority_rounds: int = 5
    # Only K-INFORMATIVE candidates may nominate priority variants or ride
    # the trial sources: never served by the nominal variant's corridor, so their
    # position is a real displacement from the nominal track.  A candidate
    # inside the nominal corridor reads as a continuation of the seed arc and
    # scores the highest P in the field (0.91-0.97 measured live) while
    # carrying zero LOV information -- the linear pass already swept the
    # nominal corridor and failed.  Epoch distance was tried and is the WRONG
    # gate: the BE559 truth is 24 days from its seed nights (a
    # same-apparition extension whose take collapses U) yet served only by
    # two high-k variants.  The gate is structural (lov_profile._lov_sigma_informative),
    # not a knob.  This floor, then, only screens P noise out of the top-N
    # nomination -- and it must stay near zero, because the whole population
    # a weak seed can hope to recover scores in the 1e-3 regime (the BE559
    # truth: 0.0042 under max(v3e, v2)).
    lov_profile_min_ml: float = 0.0
    # Priority nominations are ranked by P x exp(-k*^2 / 2 s^2) with s this
    # many LOV sigmas -- the covariance's own prior on where the truth sits
    # (see lov_profile._build_rounds).  1.0 is the honest Gaussian, which the
    # s2-floored covariance measures up to (26 fixtures, median |k*| 0.67);
    # 0 restores the pure-evidence order.
    lov_profile_prior_sigma: float = 1.0

    # ---- continuous LOV offset: the LOV coordinate as a MEASUREMENT (itf/lov_refine.py) --
    # The fan exists because RETRIEVAL costs a database sweep per variant.
    # SCORING costs no sweep at all, so k need not be read off the fan: a fine
    # re-corrected LOV, propagated in one shared integration, measures the k a
    # tracklet actually implies ~50x sharper than the fan can resolve.  Without
    # this the LOV profile's k* IS the serving variant's k, so "these agree on k"
    # means "these came from the same corridor" -- true of every candidate in
    # the round.  Measured on 2006 JE41 (U 8.80, five known tracklets 2016-2024
    # all served by the single variant k=-0.971, 12,761 candidates in that
    # round): by distance to the served centreline the truths rank 85/99/132/
    # 744/832; by the residual at a solved k they rank 9/3/2/33/38, and the top
    # multi-night k cluster holds all five.
    lov_refine: bool = True
    # The same geometry as RANKER FEATURES (search._lov_feature_pass): every
    # variant-scan candidate within lov_refine_resid_max of the LOV curve gets its
    # k*, residual and night support recorded for the model.  Only a ranker
    # trained on those columns wants this on; a decision-log collection turns
    # the pass on regardless (it records, it does not act).  The keep is large
    # because a feature must describe the candidate, not the crowd it came in.
    lov_features: bool = False
    lov_feature_keep: int = 50000
    # ...and its reach around each serving variant, in LOV sigma.  The retry's
    # lov_refine_halfwidth (0.15) under-reaches a 2.5 deg corridor's EDGE: measured
    # on the first v4 eval cases, true ex-ITF tracklets served by a variant
    # solved to k* pinned at exactly +/-0.15 with their residual inflated to
    # 111-483" (measured at the clipped edge, not at their own k).  A feature
    # must measure every candidate the corridor can serve.
    lov_feature_halfwidth: float = 0.35
    # Node spacing and the neighbourhood measured around each variant, in LOV
    # sigma.  A variant_width_deg corridor spans ~0.1-0.3 sigma of k at ITF-era
    # epochs, so the halfwidth covers a candidate's whole reach; the step only
    # has to bracket the minimum, which is then solved analytically from the
    # local gradient (one step is 300-800", far above what discriminates).
    lov_refine_step: float = 0.005
    lov_refine_halfwidth: float = 0.15
    # Setup ceiling on the node count -- a LOV re-correction is ~11 ms, so this
    # is ~16 s against a union sweep that runs to 146 s on a severe seed.  Over
    # budget the STEP coarsens; the span never truncates (dropping the tail
    # nodes would un-refine exactly the far-out variants the fan was widened
    # to reach).
    lov_refine_max_nodes: int = 1500
    # Time grid for the shared propagation.  States are cubic-Hermite
    # interpolated (position AND velocity, so the curve is the trajectory
    # rather than a chord), which is exact to well under an arcsecond at 4 d.
    lov_refine_grid_days: float = 4.0
    lov_refine_batch: int = 400
    # How many candidates survive the sweep, ranked by residual.  This is
    # the whole memory footprint of the pass: a record is an id set, a k, a
    # residual and a night -- no observations, no feature row -- so a severe
    # seed's 30k candidates cost what a quiet one's 300 do.  Ranking by
    # residual rather than by P is deliberate: it is what separates the
    # truth here (top 38 of 12,761), it needs no model, and an ML pre-gate
    # would risk discarding exactly the candidate the ranker is blind to,
    # which the BE559 record says is the normal case for a real recovery.
    lov_refine_keep: int = 2000
    # A candidate may join a cluster only if it sits within this many arcsec of
    # the LOV CURVE at its own k -- the cross-track miss no k can absorb.  The
    # 2006 JE41 truths sit at 25-126" against a 1st percentile of 386"; loosen
    # it to 1000-2000" and junk clusters start out-voting the real one.
    lov_refine_resid_max: float = 500.0
    # Cluster tolerance in LOV sigma, and the evidence bar.  The five 2006 JE41
    # truths span 0.0075 sigma across 8.6 years.  THREE distinct nights,
    # because two nights of one apparition is the cheapest coincidence there
    # is; the joint fit and the unchanged acceptance gates still dispose.
    lov_refine_cluster_dsigma: float = 0.005
    lov_refine_cluster_min_nights: int = 3
    # ...and the agreement must be IMPROBABLE, not merely local.  Tracklets
    # from one station in one observing season are neighbours on the sky and
    # therefore neighbours on the LOV; counting nights cannot tell that from a
    # real object.  2021 SA70 is the case: every cluster sat at k = 2.82-2.93,
    # near-sequential archive ids, essentially all station 705 -- and TIGHT, 4
    # nights at a median 30.4", better than the 2006 JE41 cluster that is
    # right.  Acting on it lost all 5 planted tracklets and linked 2 foreign
    # ones, where the pass switched off recovers 5/5.  No bar on nights or
    # residual separates those two; the BASELINE does (JE41's winning cluster
    # spans 2012-2024 across F51, G96, F52 and 703).
    lov_refine_cluster_min_span_days: int = 400      # more than one opposition
    lov_refine_cluster_min_stations: int = 2
    # One member per night, capped: a joint trial faces the warm screen as a
    # whole, and a big cluster is killed before the truth inside it is fitted
    # (measured live on BE559 with a 9-member proposal).
    lov_refine_cluster_max_members: int = 4
    lov_refine_priority_rounds: int = 5
    # Blend the measured geometry into the candidate RANKING, per tracklet,
    # rather than only into the variant order and the front trial sources:
    # "a combined score for each tracklet based on how
    # many nearby objects on LOV plus its ML score, then try tracklets in that
    # order".  It is the better shape for two reasons -- it plugs into the
    # ranked trial queue that already exists instead of adding trial sources, and it never
    # has to CHOOSE between clusters, which is the step that went wrong live
    # (a 10-night junk pile at 163" outranking the 8-night truth at 27", the
    # scan then spending its whole budget on the wrong variant).
    # A candidate with no company scores zero on the support term instead of
    # being unrankable.
    lov_refine_rank: bool = True
    # How many strongly-evidenced candidates may be PROMOTED to the front of a
    # round's ranked list.  A promotion, never a re-sort: the first cut sorted
    # the whole list by the combined score and cost 2015 BE559 its recovery
    # outright (804 s, 0/4, against 491 s and 4/4 without it).  That object is
    # on record as recovering BECAUSE of ranker diversity -- the v3e primary
    # plus the v2 second opinion, whose truth scores P 0.0042 against junk at
    # 0.9+ -- so a global re-ranking destroys the one ordering that works
    # there.  Inserting a bounded few at the front is the same shape the
    # validated LOV profile trial sources already use, and it cannot disturb what the
    # model had right.
    lov_refine_promote: int = 3
    # Reorder the VARIANT fan by cluster evidence.  Separate from the ranking
    # because it is a different risk: the ranking decides order WITHIN a round
    # and can only promote on measured evidence, while this decides WHICH
    # rounds are reached at all -- and a fan that sends the scan at the wrong
    # variants spends the whole budget there (measured twice: a 10-night junk
    # pile on 2006 JE41, and the BE559 loss above).
    lov_refine_order: bool = True
    # The re-scan's allowance, as a fraction of the search ceiling.  Pass 1
    # ends BY spending the budget, so without a slice of its own the retry is
    # refused before it starts; a slice rather than a raised ceiling, because
    # the whole point of the budget is that a doomed search stays bounded.
    lov_refine_retry_budget_frac: float = 0.5
    # Weights on the support and model terms; the residual term is the unit.
    # Both 1.0 until a multi-object A/B says otherwise -- with one tuned object
    # any other number is a guess wearing a decimal point.
    lov_refine_support_weight: float = 1.0
    lov_refine_ml_weight: float = 1.0

    # ---- barren-take backtracking (ladder-backtracking.md, part A) ---------
    # The trial queue is greedy and final: the first trial that improves the
    # fit is taken, and when the recursion below it comes back EMPTY the whole
    # round is abandoned -- the remaining trials are never fitted.  On a weak
    # seed that is a race the truth can lose by one position: junk fits a wide
    # corridor cleanly, so the take goes to whichever candidate the ranker put
    # first (2015 BE559 variant 37: v2 puts the truth #1 and recovers; v3e
    # puts a junk tracklet #1, the subtree dies, and the round forecloses with
    # the truth sitting in a trial that never ran).  With this > 0 a take is
    # PROVISIONAL: when the branch below it produces NOTHING (terminal status
    # 0 -- not a linkage, not a kept-for-review 0.6), the trial queue resumes at
    # the next untried trial with the barren take's surviving tracklets
    # excluded, up to this many resumes per round.  Cost lands only in rounds
    # that today END IN FAILURE; a round that takes correctly never pays.
    backtrack_per_round: int = 3
    # ...and part C: the second-opinion singles trial source draws from likely+maybe
    # instead of likely only.  The truth's category is knife-edge at variant
    # corridors (the SAME tracklet measured 1.3" = likely in one window
    # realization and 2.5" = maybe in another), and a maybe is invisible to
    # every trial source except the promote-top-1 rule -- so a category flip can hide
    # a candidate from BOTH rankers.  The promote rule already concedes that
    # model confidence overrides the maybe gate; this extends it to the
    # second-opinion trial source (same K, trial-fitted and gated like every trial).
    second_singles_maybe: bool = True

    # ---- candidate retrieval ----------------------------------------------
    # "index" = the precomputed itf.itf_tracklets fast path (healpix prefilter
    # + corridor + per-tracklet uncertainty); "scan" = the legacy per-window
    # obs_sbn/box+quadratic-corridor SQL, kept for A/B of candidate
    # completeness against the index.
    candidate_path: str = "index"              # "index" | "scan"
    # How the index path sweeps a whole corridor (no effect on candidate_path
    # "scan").  "auto" -- the default -- picks a path per corridor in
    # candidates.DbSource._route_sweep, which carries the measurements and the
    # reasoning; the other three values NAME A PATH and force it
    # unconditionally, for an A/B of that path (no fallback, or the A/B just
    # measures the router again).  The three paths:
    #
    #   "setjoin"   -- every window's OWN (healpix, epoch, dec) box goes to the
    #     server as one joined unnest() relation, so Postgres applies the exact
    #     per-window prefilter via the (healpix, epoch_mid_sec) index and
    #     returns only rows a window asked for: one query per ~20k (window,
    #     pixel) boxes instead of ~100 chunk round trips, and no union-box false
    #     positives to discard in Python (2024 RX190, 907 windows: 5.4s/4461
    #     rows -> 0.5s/321 rows, identical candidates after the corridor test).
    #     Needs few pixels per window: on a wide corridor the per-pixel tuples
    #     explode, which is why the router only sends narrow ones here.
    #   "rangejoin" -- each window's healpix pixels as contiguous RANGES plus
    #     its own epoch/dec box, corridor test and motion bound, ~48 windows
    #     batched into ONE query.  Measured on 2015 BE559's 272-window corridor,
    #     the cost was never the rows, the pixels or the index -- it was
    #     PER-QUERY overhead: a 25 ms round-trip floor, ~52 ms of planning
    #     driven by a 17k-element `= ANY(pixels)` array, against 2-26 ms of
    #     actual index execution.  NESTED healpix ids are spatially local, so
    #     those 17096 pixels are only 164 ranges.
    #       one query per window 272 queries 39.4s | batched 48  6 queries 16.0s
    #     -- 2.46x, 18165 survivors identical.
    #   "chunk"     -- the original coarse union-box chunker, one box per group
    #     of windows with the per-window test re-applied in Python
    #     (SWEEP_CHUNK / SWEEP_SPAN_SEC / SWEEP_PIX_BUDGET apply only to it).
    #     Still the right shape for thousands of mutually OVERLAPPING windows,
    #     where a per-window join would return each row once per window.
    #
    # No shape wins everywhere -- measured live, candidates identical in every
    # cell (this table is what the router encodes):
    #                       BE559 (wide, 272 win)   RX190 (narrow, 907 win)
    #     chunk                   31.2 s                   3.6 s
    #     setjoin                 33.0 s                   0.6 s
    #     rangejoin               14.6 s                   8.1 s
    sweep_mode: str = "auto"       # "auto" | "setjoin" | "rangejoin" | "chunk"
    # Try-best-first ordering of candidate tracklets: "ml" = the trained
    # tracklet-match classifier (itf/training_data; halves the try-order depth
    # of a true tracklet vs the legacy score); "certainty" = match-chi2 against
    # the tracklet's own fitted uncertainty + ephemeris sigma; "legacy" = the
    # dist+qual+uncert score. ml/certainty need the index path.
    ranking_mode: str = "ml"                   # "ml" | "certainty" | "legacy"
    # Override the trained PRIMARY model path (None = the deployed default,
    # training.default_model_path -- ONE model, v4, since 2026-09-27; see its
    # docstring)
    ml_model_path: str | None = None
    # SECOND-OPINION ranker: a different model artifact whose top singles join
    # the trial queue (woven between the primary's when backtracking is
    # armed -- search.TrialQueue.next_take).  It was the recall mechanism of the
    # v3e+v2 pair (2026-08-07 .. 2026-09-27): v3e ALONE lost 2015 BE559 and
    # 2022 GL24, the v2 trial source took them.  v4 alone recovers both (and all 27
    # fixtures), so the default is OFF.  Set a path (e.g. _V2_MODEL,
    # training.default_second_model_path) to re-arm a second trial source for an A/B.
    ml_second_model_path: str | None = None
    # ...and how many of the second ranker's top singles join each trial queue.
    recurse_second_singles: int = 2

    # Skip the OBS FETCH for a corridor tracklet whose own recorded motion
    # already rules it out of the trajectory gate below
    # (candidates.motion_bound_ok, applied in DbSource._survivors).  A
    # degrees-wide corridor is dominated by tracklets that pass the POSITION
    # corridor and then fail the MOTION test, and their obs are fetched, grouped
    # and thrown away: 2015 BE559's union sweep pulls 308283 obs rows for 87799
    # survivors, ~60 s of its ~130 s.  The bound keeps anything it cannot bound
    # (no rate, no rms, single epoch) and carries a paranoid slack -- 77% of
    # BE559's survivors are still ruled out at 10x rms, 65% even at 25x -- so a
    # tracklet is only ever dropped on POSITIVE evidence.  Being ahead of
    # max_candidates_per_window, it also lets that cap bite on noise rather than
    # on candidates.  Bypassed while an ML feature collector is attached (those
    # hard negatives ARE the training set's negative class) -- per-sweep, via
    # sweep_candidates(prefilter=False), NOT by editing this config: the source
    # binds its own cfg at construction, so a replace() in the search entry
    # point would never reach it.  Each collected row records the bound's
    # verdict in the out-of-band motion_bound_ok column (training.META_COLS),
    # so a retrain can choose either population without a fresh collection.
    #
    # NOT candidate-identical (search._group_rows groups obs by
    # trksub + station + 12 h, not by tracklet id), so the acceptance test is
    # RECALL -- the recovery suite live and offline -- not set identity.
    #
    # MEASURED live on BE559's union sweep (deterministic row counts, two
    # off-runs vs one on-run): corridor survivors 87799/87846 -> 23153 (-74%),
    # obs rows fetched 308283/308419 -> 77950 (-75%), sweep 126.6 s -> 108.4 s.
    # The time win is smaller than the row win (the fetch is not row-proportional
    # -- there is fixed per-query cost underneath), but it is real, and cutting
    # the payload 4x is worth having on its own against a shared server.  Recall
    # unchanged: recovery suite 8/8 live and 8/8 offline, every rms identical.
    prefilter_motion: bool = True
    prefilter_motion_slack_rms: float = 10.0    # x the tracklet's own fit rms
    prefilter_motion_slack_arcsec: float = 1.0  # ... plus this absolute floor

    # ---- corridor trajectory gates ----------------------------------------
    # A tracklet is "likely" when its corridor motion-mismatch (qual) is under
    # likely_arcsec + trajectory allowance; "maybe" under maybe_arcsec. With a
    # WELL-DETERMINED orbit (trial U <= relax_U, or >= relax_opps oppositions
    # in the trial arc) the corridor prediction is essentially exact, so any
    # residual qual is tracklet astrometric noise -- the likely gate widens to
    # the maybe bound and the trial fit's residuals become the real arbiter.
    likely_arcsec: float = 1.0
    maybe_arcsec: float = 2.5
    relax_U: float = 3.45          # was 3.0; +0.44 at U 3.0 -- see U-SHIFT note
    relax_opps: int = 4
    # ML override of the geometric gate: if the single highest-ML-ranked
    # candidate is only "maybe" (motion-mismatch in the 1.0-2.5" band), try it
    # anyway -- a confident ML match earns a trial fit regardless of the maybe
    # status AND of whether the orbit is well-determined (the relaxed widening).
    # Only the ONE overall-best maybe is promoted; the trial queue + acceptance
    # gates still vet it, so the cost is at most one extra trial fit. ML ranking
    # only (ranking_mode="ml", index path); no effect under certainty/legacy.
    try_top_ranked_maybe: bool = True

    # ---- extension-quality gates -----------------------------------------
    # An ITF link is REFUSED when it makes the object's OWN pre-search (seed)
    # observations meaningfully worse -- a contaminant the orbit bent to swallow
    # (2026 FU24: the object's own obs 0.047" -> 0.297").  Seed-only, and
    # RELATIVE to the seed's own achieved rms; the rationale for both choices is
    # in search.extension_degrades.  "Meaningfully worse" = rms above BOTH a
    # multiplicative and an additive floor.  3.0 (not 2.0): an overfit compact
    # seed settling onto its true orbit widens ~1.8x honestly (2026 DZ13
    # 0.098"->0.175", wrongly vetoed at 2.0x) while a contaminant blows up far
    # more (FU24 6.4x; the real 2025 NQ92 precovery only 1.1x) -- see
    # test_extension_gate_config.
    #
    # Both sides of the ratio are subset_rms (per-observation) values via
    # search._seed_pre_rms.  Until 2026-07-29 the pre side was pub.rms, a
    # PER-AXIS rms, so these factors were silently divided by sqrt(2): the veto
    # really fired at 2.12x and the warning at 1.06x -- i.e. at any degradation
    # at all, which is why the warning appeared on nearly every good link.  Never
    # feed a `.rms` in here; see fitter.subset_rms's currency warning.
    veto_rms_factor: float = 3.0               # post rms >= pre x this ...
    veto_rms_floor_arcsec: float = 0.08        # ... AND pre + this (both, to ban)
    # A near-degenerate SHORT-ARC seed honestly widens far past 3x.  The factor
    # above was calibrated on seeds of 11-17 d (FU24 18 obs/11.0 d, NQ92 15
    # obs/16.7 d); a discovery arc of a couple of days is nearly degenerate
    # against the state vector, and extension_degrades's own docstring records
    # the measurement -- correct links ran to 3.7x on a 22-obs seed and 19x on a
    # 4-obs one.  2020 KX6 is the case in hand: an 8-obs, 2.18 d, U 9.23 seed
    # fitting to 0.073", whose CORRECT link (1 -> 3 opps, 6 nights, the object's
    # own 2020-06-23 tracklet plus its 2023 and 2025 apparitions) settles it to
    # 0.240" -- 3.3x, honest in-sample -> out-of-sample widening, banned as a
    # contaminant by a threshold that never saw an arc this short.
    #
    # Only ratios BETWEEN the two factors change hands: FU24's 6.4x contaminant
    # is banned under either, so the arc gate cannot wave the anchor through.
    # UNVALIDATED against a live false-link rate (as veto_rms_factor itself is).
    # The factor's statistical escape: however big the multiple, a link that
    # leaves the object's OWN observations fitting their uncertainties has not
    # pulled the orbit off the object.  In sigma, normalised by the STATION
    # floor (never the survey's self-reported sigma -- that is what would let
    # 2026 FU24's Rubin contaminant through at "0.4 sigma"; on X05's 0.1" floor
    # it is 2.97).  2015 FO479 is why: a 0.13-sigma seed fit by luck, whose
    # correct 6-opposition linkage read as 3.09x while sitting at 0.56 sigma.
    # 0 disables the escape.
    veto_within_sigma: float = 1.0
    veto_short_arc_days: float = 5.0           # seed arc <= this = near-degenerate
    veto_rms_factor_short_arc: float = 5.0     # ... and gets this factor instead
    # ---- what a link may CLIP from the seed --------------------------------
    # The rms gate above compares only the seed obs the combined fit still
    # INCLUDES, so an orbit that bends far enough for the engine to REJECT the
    # object's own astrometry scores an improvement there: it discarded the
    # evidence.  The complement is the seed-clip ledger
    # (search._seed_clip_ledger / _clip_evidence_ok), and the policy is
    # deliberately NOT "never clip a seed ob" -- that would make ONE bad
    # published observation enough to render an object permanently
    # unrecoverable, and a successful recovery is often precisely how a bad
    # night gets PROVEN bad ("we found it, but the 4th night of the discovery
    # arc had to go").  So a clip is priced, bounded, and always reported:
    #
    #   * losing individual obs from nights that SURVIVE is ordinary outlier
    #     rejection -- not priced here (the overall min_retention still bounds
    #     it), because the arc's opposition profile is unchanged;
    #   * losing a WHOLE span (opp_profile.SPAN_DAYS, the gate's own 36-hour unit;
    #     these were called nights until 2026-09) is a claim about the
    #     published arc, and must be bought with clip_span_price_tracklets
    #     independently linked tracklets per span dropped.  3, because three tracklets in other oppositions
    #     agreeing on an orbit the dropped night contradicts is evidence, and
    #     two is a coincidence budget the rest of this search already spends
    #     (see the 1.75-opposition combinatorics caution);
    #   * at most clip_spans_max may go and at least clip_keep_spans_min must
    #     remain: below that the "object" the linkage attaches to has stopped
    #     being an arc and the identification rests on ITF tracklets alone --
    #     a different and much weaker claim than the one this search makes.
    #
    # UNVALIDATED against a live false-link rate (like veto_rms_factor before
    # it); the numbers are a design rule, chosen to be legible
    # rather than tuned.  Raising the price or lowering clip_spans_max is the
    # conservative direction.
    clip_spans_max: int = 1                    # whole seed spans that may go
    clip_keep_spans_min: int = 2               # ...and what must remain
    clip_span_price_tracklets: int = 3         # new tracklets per span dropped
    # An observation is only clipped as a poor fitter when it misses by more than
    # the flat arcsec threshold AND by more than this many times its OWN sigma
    # (Veres-floored).  The flat cut alone asks the wrong question of a noisy
    # station -- 0.5" is a blunder from a 0.15"-sigma survey and unremarkable
    # from a 1"-sigma one -- and it cost 2024 PQ7 three good W76 observations
    # on a linkage that is real.  3 sigma is the usual rejection bar; raising
    # it clips less, 0 restores the flat-threshold-only behaviour.
    clip_sigma_mult: float = 3.0
    # ...and the fewest observations a candidate tracklet may be pruned DOWN to
    # and still be a tracklet.  One surviving observation fixes no motion: it
    # cannot be checked against the corridor that found it, cannot be submitted
    # as an identification, and would be shown as a lone night's linkage on the
    # strength of a single dot (2024 RL188).  Anything left below this is
    # dropped whole, per station and night.
    min_tracklet_obs: int = 2
    # Softer thresholds that only WARN (surface a caution, not a refusal): a
    # notable-but-not-banning seed degradation, or a large linked-tracklet mag
    # residual (a different body is often a different brightness).
    #
    # 2.0, not 1.5.  A CORRECT link widens the seed's own residuals on its own:
    # the pre-link rms is in-sample (6 elements fitted to that arc), so the obs
    # loosen once the longer arc takes over, and 18 known-correct links measured
    # across the PAIR_* fixtures ran 1.05-1.5 with a tail to 2.0.  1.5 sits
    # inside that honest distribution and warns on good recoveries; see
    # search.extension_degrades for the measurement and the dof argument.
    warn_rms_factor: float = 2.0
    # A linkage the published object has fallen out of entirely (no natural
    # designation left in the fit) is an unrelated ITF-ITF link: it is built
    # for the record but only transmitted when it fits this cleanly (arcsec,
    # per-axis overall rms).
    pure_itf_max_rms: float = 0.15
    # ...and the ABSOLUTE floor under the VERDICT that warning now carries
    # (by design a seed that fits materially worse is never
    # `linked`, only a reviewable `possible` -- search.SeedCheck).  The
    # relative factor above is the right shape for a caution but cannot decide
    # a verdict on its own, because a short over-fit seed starts absurdly tight
    # and clears any ratio for nothing: 2020 KX6 recovers all 3 apparitions,
    # 4/4 tracklets at 0.126" -- a known-correct linkage -- while its seed goes
    # 0.072" -> 0.231", i.e. 3.2x for 0.16" of movement.  2006 VT188's
    # contaminant goes 0.189" -> 1.273".  0.5" separates them and is the same
    # noise floor recurse_seed_abs_arcsec already uses for the same reason (an
    # over-fit 3-night seed honestly widening to ~0.45" under the true
    # multi-opposition orbit is not contamination).  The WARNING is unchanged
    # and still fires below this; only the demotion waits for it.
    demote_seed_abs_arcsec: float = 0.5
    # The photometric warning is a TRACKLET test, not a per-observation one
    # (fitter.mag_outlier_tracklet): a tracklet warns when its MEAN residual
    # reaches warn_mag_resid AND a two-tailed t-test puts it below warn_mag_p
    # from zero.  One ob 0.8 mag high is ordinary survey scatter; a whole
    # tracklet 0.8 mag high is a different light curve.  0.7 is kept from the
    # old per-ob threshold on purpose -- nothing quiet today starts warning on
    # size grounds, the significance test only ever subtracts.  In 67k labelled
    # ITF rows (data/itf_train_v2.csv, itf_holdout_fresh.csv) tracklet mean
    # |residual| reached 0.7 for 4-5% of CONFIRMED links and 51-64% of
    # non-matches, and confirmed links never exceeded ~1.0 mag.
    #
    # The anchor: 2022 WY29's recovery fixture is a REAL (MPC-accepted) linkage
    # carrying one 1.7 mag photometric outlier.  The per-ob rule made every one
    # of its submissions a click-through; under the tracklet test no tracklet
    # is jointly off and it is silent -- which is what the fixture note ("the
    # mag warning must stay a caution, not a veto") was working around.
    warn_mag_resid: float = 0.7                # mag; mean tracklet |obs - computed|
    warn_mag_p: float = 0.05                   # two-tailed t-test vs zero
    # ---- partial pickup of a designated (pseudo-ITF) object (itf/pickup.py) --
    # A document names DESIGNATIONS; the search links TRACKLETS.  When it takes
    # only part of a designated object, the archive's other tracklets are
    # measured against the linked orbit at their own station and epoch, and a
    # leftover that does not fit says the DESIGNATION is itself a mislinkage --
    # not something an identification document can express (see itf/pickup.py).
    pickup_check: bool = True
    # "does not fit": past this many arcsec AND past this many formal sky
    # sigmas.  Blunt on purpose -- the case that motivated it reads 0.4" for the
    # tracklet that belongs and 3779"/3332" for the two that do not, on a
    # sigma of 0.08".  The floor is what decides in practice; the sigma term
    # only keeps a genuinely uncertain extrapolation from being accused.
    pickup_off_arcsec: float = 10.0
    # 5.0 -> 2.8 (2026-09-22), to hold this term where it was.  The s2 floor in
    # FittedOrbit._param_inverse made every sigma honest, so the ephemeris sky
    # sigma this multiplies is ~1.8x wider for the same physical uncertainty
    # (median over 25 JPL-matched objects; 5.0 / 1.8 = 2.8).  Left at 5.0 the
    # sigma arm would start protecting extrapolations it was never meant to
    # cover: the term bites once 5*sigma exceeds the 10" floor, i.e. at a true
    # sigma of 2" before and 1.1" after -- so a leftover that IS a mislinkage
    # gets excused for being "uncertain" when it is not.  This arm guards a
    # SAFETY check (a designation we would otherwise claim), and the direction
    # it drifts under an honest sigma is the wrong one.
    pickup_off_sigma: float = 2.8
    # Per-recursion guard (stricter than the final gate, so passing it implies
    # passing the gate): before recursing on a tracklet subset, refuse it if the
    # EXISTING (pre-inclusion) obs degrade by more than recurse_seed_factor-1 AND
    # by at least recurse_seed_floor_arcsec. A single lone candidate is exempt
    # (nothing to fall back on). The trial queue then tries a smaller set.
    # Same seed currency as the gate above (search._seed_pre_rms): before
    # 2026-07-29 this factor too was measured against a per-axis pub.rms and so
    # really bit at 1.17x -- masked for tight seeds by the absolute floor below,
    # which dominates until the seed is looser than ~0.5".
    recurse_seed_factor: float = 1.65          # >65% worse ...
    recurse_seed_floor_arcsec: float = 0.10    # ... AND >= this many arcsec worse
    # ... AND past an ABSOLUTE noise floor.  A short overfit seed arc (three
    # nights fit to ~0.2") honestly doubles its residuals once the true
    # multi-opposition orbit pins it -- ordinary survey scatter, not
    # contamination, and the guard must not reject the true orbit (2024 RX190:
    # every joint trial tripped at 0.450" against a 0.324" relative limit).
    # The floor is PRINCIPLED, not a fixed arcsec: it is the seed's OWN chi^2=1
    # astrometric noise floor (`subset_sigma_floor` = the rms the seed obs would
    # show if every residual sat at its assigned sigma) times sigma_mult -- so a
    # coarse-station arc (2014 TM87's app0 carries station 691, noise floor
    # 0.58") gets a higher floor than an all-Pan-STARRS one, and the guard never
    # calls a seed poisoned while its residuals are still within the
    # observations' assigned uncertainties.  recurse_seed_abs_arcsec is the
    # SAFETY MINIMUM (a tight-survey seed keeps 0.5", so this only ever RAISES
    # the floor -- it cannot reject a link the old fixed floor accepted).
    recurse_seed_abs_arcsec: float = 0.5
    recurse_seed_sigma_mult: float = 1.0       # x the seed's chi^2=1 noise floor
    # Seed retention under a trial fit: the engine's own outlier rejection may
    # clip a couple of seed obs in a trial, but a trial that drops more than
    # 1 - this fraction of the object's OWN (published) observations is bending
    # the orbit away from the object -- refused regardless of rms.
    recurse_seed_retention: float = 0.8
    # Trial queue: try the top-N tracklets together for each N here, then the
    # top singles one at a time (best-ranked first) -- each trial guarded above.
    recurse_batch_sizes: tuple = (10, 3)
    recurse_singles: int = 5
    # A FLAT field: the ranker rates no likely candidate a plausible match
    # (best P(match) under this -- the same 0.15 floor at which a one-night show
    # is foreclosed), so its order among them is near noise and "take the first
    # single that passes" takes whichever distractor happened to rank first.
    # On a LINEAR corridor (not a LOV variant round), the primary singles window
    # there widens to flat_field_singles and a take whose branch comes back
    # barren backtracks to the next passing single at ANY depth (normally depth
    # 0 only; cfg.backtrack_per_round resumes) -- the branch's corroboration is
    # the evidence the rank lacks, and the cost lands only on barren takes.
    # Measured on 2015 HU248's depth 1 (46811 candidates, best P 0.049): the
    # truth ranked #12 at P 0.009 behind same-station distractors, the third
    # passing single.  A resumed take does not restart the no-progress clock
    # (search.TrialQueue.resumed), which is what bounds these resumes in time;
    # over 120 live recoveries that changed no result and cut two fruitless
    # searches 200-300 s before the ceiling.  0 disables.
    flat_field_ml_p: float = 0.15
    flat_field_singles: int = 30
    # ML floor for the joint "top-N together" trials: a candidate whose P(match)
    # is under this fraction of the best-rated candidate's P adds noise, not
    # signal, to a joint hypothesis (it can trip the seed guard and burn the
    # whole trial) -- batch trials take the top N ABOVE this floor.  Weak
    # candidates still get their chance via the singles trials and the
    # re-sweeps of the recursion.  Applies only when ML scores are present
    # (index path, model loaded); 0 disables.
    batch_min_ml_frac: float = 1.0 / 3.0
    # Overall-trial rms sanity, independent of the seed-subset guard above: a trial
    # whose CONVERGED orbit sits far off the sky is an under-determined seed's wide
    # corridor accreting whole survey nights into a degenerate solution.  The seed
    # obs can stay tight while the new obs blow up (so the seed guard passes and
    # obs count reads as "progress"), and _prune_trial cannot rescue it when EVERY
    # candidate is offset.  Such a trial can never clear rms_max, so recursing on it
    # only snowballs corridor noise (2016 NY214: 20 depths at 2500-5900", 837 s).
    # The ceiling is GENEROUS -- far above any real intermediate, far below the
    # garbage -- and takes the larger of an absolute floor and a multiple of the
    # seed rms so a coarse-station seed is not held to a tight cap.
    recurse_rms_max_arcsec: float = 10.0
    recurse_rms_max_factor: float = 8.0
    # ---- warm-start rejection screen for trial fits ------------------------
    # Most trial fits test a candidate that is hopeless, and a cold fit costs
    # 5-7 s to say so (vs 0.08 s on a TRUE tracklet -- diverging is what is
    # expensive).  You do not need a converged orbit to reject: take a few LM
    # steps FROM THE PARENT ORBIT'S OWN converged state, at the PARENT'S OWN
    # epoch, over parent obs + candidate obs, and ask whether the step DESTROYS
    # THE PARENT ARC.  Measured over seven fixtures (2026-07-23) the separation
    # is four orders of magnitude: a true tracklet leaves the seed at 0.5-1.1
    # sigma, an incompatible one at 6 500-31 400 sigma, both in 14-44 ms.
    # The screen can only SKIP WORK -- it never accepts a linkage on its own,
    # and it defers to the cold fit wherever it is not trustworthy.
    screen_trials: bool = True
    screen_nfev: int = 15                      # LM evaluations for the warm step
    # Median sigma-normalised residual over the PARENT obs above which the warm
    # step is judged to have destroyed the parent arc.  Deliberately ~100x above
    # the true-link values and ~65x below the incompatible ones -- the gap is so
    # wide that the exact value is not load-bearing.
    screen_seed_sigma_max: float = 100.0
    # A U gate on the parent orbit USED to sit here (screen_max_U = 8), on one
    # datapoint: 2015 BE559 (U 9.56) had its TRUE tracklet screened at 3315
    # sigma, so a bounded warm step looked untrustworthy for a severe seed.
    # RETIRED on measurement -- that pair was forced by hand, and BE559's true
    # tracklets sit 26-56 deg off the corridor, i.e. they are never candidates
    # the trial queue sees.  An IN-CORRIDOR candidate is near the parent's own
    # prediction by construction, so the joint solution is reachable from the
    # parent whatever the parent's U: the corridor already does the bounding the
    # U gate was trying to do.  Probed by shortening fixture seeds until U
    # crossed 8 (the offline suite reaches no such case on its own): six objects
    # at parent U 8.94-17.87, every TRUE trial screening at 0.13-0.92 sigma
    # (2024 RX190 at U 17.87 -> 0.82).  None: no U gate.  Kept as a valve --
    # set a float to restore the old skip-above-U behaviour.
    screen_max_U: float | None = None
    # ...and the VARIANT scan, where the nominal-orbit screen is off by
    # construction (a bounded step from the nominal destroys the true tracklet
    # too -- that is why the LOV is being sampled).  Each LOV variant is itself a
    # legitimate orbit at the same epoch, so its OWN state can be the warm-step
    # parent (search._VariantParent).  This is where the time actually is on a
    # severe seed: 2015 BE559 live spends 307 s of its 562 s in variant-scan trial
    # fits, and 11 fits over 5 s are 238 s of that.  DEFAULT OFF until the
    # true-tracklet sigma distribution is established from a variant state --
    # a variant is not converged on the seed obs the way a trial fit is, so the
    # nominal screen's measured margin does NOT carry over.  MEASURED (shadow
    # mode, live BE559, 69 trials): it does carry over, and then some.  Both trials
    # containing a planted tracklet -- including the one that LINKS (24 obs, rms
    # 0.134") -- screen at 0.7-0.9 sigma from their variant's own state, while
    # wrong trials run to 20 405 sigma (median 461).  At the same 100 sigma
    # threshold that rejects 38 of 67 wrong trials, saving 254 s of the 619 s run
    # for 1.5 s of total screen cost.  ON.
    screen_variant_trials: bool = True
    # Wall-clock budget for a stretch of work with NO PROGRESS (seconds; 0
    # disables), plus an overall ceiling at search_ceiling_mult x that.  The
    # clock RESTARTS every time a trial is taken (search.SearchBudget), so it
    # bounds fruitless grinding rather than the whole search -- as a pure
    # deadline (which is what this knob was until 2026-07-23, and what its old
    # name no_link_deadline_sec said) it was a recall bug: under DB load it cut
    # searches that were still making progress (2015 BE559 returned ok=False at
    # 198 s while working).  An under-determined
    # seed's corridor is degrees wide, so every opposition offers hundreds of
    # unrelated tracklets that each cost a full trial fit to reject -- and if
    # NOTHING ever links, that grind yields nothing for many minutes (2016 NY214:
    # 900 s+, 98 doomed fits).  An accepted linkage early-exits on its own, so the
    # budget only ever bounds the hopeless case: past it, stop LAUNCHING trials
    # (_recurse_trial_queue) and variants (_variant_scan) and report the no-link result.
    # 150 s: the checks sit BETWEEN fits, so the real bound is 150 + one
    # in-flight fit (~25-40 s observed) -- under the ~200 s target, and ~35 s above
    # the slowest known real recovery (SD57 links live in ~114 s).  NOT a hard
    # ceiling: a single pathological fit needs per-fit nfev bounding.
    no_progress_sec: float = 150.0
    # ...and the absolute ceiling, as a multiple of it: a pathological
    # take-a-trial-forever loop still has to stop, which the interactive callers
    # (web UI) depend on.  0 = no ceiling.  NB the ceiling is wall-clock but not
    # a wall-clock GUARANTEE: SearchBudget.setup_done moves it out by the variant
    # scan's one-time union sweep (see search_setup_credit_max_sec).
    search_ceiling_mult: float = 4.0
    # Fruitless trial fits required BEFORE the no-progress clock may cut a
    # search.  The clock alone cannot tell a doomed grind from a slow machine,
    # and on a loaded box it cut 2024 PQ7 -- a real, MPC-accepted linkage that
    # recovers 4/4 standalone in ~155 s -- to nothing.  Fits are load-
    # independent: a grind burns them regardless of machine speed (2016 NY214:
    # 98 of them), a merely-slow search burns few.  20 sits well under any
    # observed grind and well above a working search's fruitless stretch.
    # 0 restores the clock-only behaviour.  (The absolute ceiling above is
    # unconditional -- the interactive bound does not negotiate.)
    min_fruitless_fits: int = 20
    # Wall-clock bound on ONE candidate trial fit (itf.fitter.fit_astrometry ->
    # deadline.py).  The budget above is checked BETWEEN fits, so a single
    # pathological fit could spend it alone: live 2014 HM543 lost two ~300 s
    # fits to junk V00 tracklets and hit the ceiling one variant short of its
    # truth.  Over 1435 trial fits in 18 live traces: median 0.5 s, p99 14 s,
    # and every one of the 11 over 20 s ended NO PROGRESS or refused -- none
    # was ever taken.  A fit that runs out is an ordinary failed fit.  The seed
    # fit and the non-grav rescues stay unbounded.  0/None disables.
    trial_fit_max_sec: float = 30.0
    # Leave-out confirmation (itf/leaveout.py): an accepted linkage's photometric / coherent-
    # offset / residual-trend caution whose data the orbit fitted WITHOUT it
    # predicts (sigma <= 2", miss <= 1.5", D <= 3, drift <= 1.5") becomes a
    # diagnostic -- noisy data of this object, not a doubt about the link.  Never
    # applied to a three-thin (1.75 -> 3) confirmed linkage.  False = off.
    leaveout_confirm: bool = True
    # Vouching (search._vouch): a linked tracklet that fails its own retention
    # or chi2 clause is still accepted when it does not PULL the orbit -- the
    # orbit fitted without it is determined there (1-sigma <= vouch_max_sigma_
    # arcsec) and adding it back moves the prediction at its epochs by at most
    # vouch_shift_frac of its miss (or the prediction's own sigma, if larger)
    # -- AND at least vouch_ok_frac of its observations (two or more) sit
    # within vouch_ok_sigma of that leave-out prediction, measured against
    # their own uncertainty.  Then the rest are blunders inside a true
    # tracklet, not a misidentification the fit absorbed.
    vouch_max_sigma_arcsec: float = 2.0
    vouch_shift_frac: float = 0.1
    vouch_ok_frac: float = 2.0 / 3.0
    vouch_ok_sigma: float = 5.0
    # Cap on the total one-time SETUP time both clocks will excuse
    # (SearchBudget.setup_done; seconds, 0 = uncapped).  The credit exists so a
    # slow DB's fixed toll is not charged to the scanning it pays for, and today
    # only ONE credit can ever be taken (a single _variant_scan call site, one
    # union sweep).  The cap is what keeps that a property of the BUDGET rather
    # than of the call graph: if a second setup site ever appears, the ceiling
    # the web UI depends on drifts out by every sweep instead of by one.  600 s
    # is ~4x the worst measured sweep (BE559 live: ~146 s).
    search_setup_credit_max_sec: float = 600.0

    # ---- trial-fit / acceptance gates -------------------------------------
    rms_max: float = 0.44                      # arcsec, unweighted find_orb-style
    min_retention: float = 0.8                 # accept only when at least this
                                               # fraction of ALL obs stay included
    # (rms_max / min_retention still gate the RECURSION's trials and the
    # post-acceptance final sweep.  They no longer gate ACCEPTANCE itself, which
    # since 2026-08-01 asks the three per-tracklet / sigma questions below --
    # see search._accept_quality_ok for why an aggregate arcsec rms is the
    # wrong currency for "is this linked tracklet real?".)
    accept_tracklet_min_retention: float = 0.75   # each linked tracklet must
                                               # keep this fraction of the obs
                                               # it was SWEPT with (pre-prune)
    accept_tracklet_conf: float = 0.997        # ~3 sigma: each linked tracklet's
                                               # kept obs vs their own sigma
    accept_seed_preserve_conf: float = 0.997   # ~3 sigma: seed Delta-chi2 region
    # ...plus this fraction of the seed's OWN chi2, because Delta-chi2 is a sum
    # over the seed's observations while the chi2_6 region above is a constant:
    # a big or systematically-biased seed fails a fixed cap for having many
    # observations, not for fitting badly (1997 QK1: 164 obs of 1997 photographic
    # astrometry, a correct 4-opposition linkage refused at Delta 28.6 vs 19.8
    # while the joint orbit fit those obs at chi2/dof 0.75).  0.25 is ~12% growth
    # in the seed's rms; see search.seed_preserve_cap.
    accept_seed_preserve_frac: float = 0.25
    # The clause's STATISTICAL escape, in reduced chi2 (chi2 per degree of
    # freedom, station-floored): protected observations that fit their own
    # uncertainties have not been pulled off the object, whatever multiple of a
    # lucky baseline the Delta represents.  Asked against the seed AND against
    # the prior accepted state -- either passing is enough.
    # 1.0 is the honest bar (residuals at their assigned sigma); it needs no
    # relaxing once the reference is right, which is the whole point: 2011
    # QH108 reads 1.243/dof against its 21-obs seed and 0.699/dof against the
    # 52-obs linkage a resumed search would start from.
    accept_seed_within_dof: float = 1.0
    # ---- A new opposition the seed cannot vouch for (search._new_opposition_unpredicted)
    # A single-opposition seed leaves its orbit loose along the line of
    # variations, so the FIRST tracklet taken in another opposition mostly picks
    # a point on that line, and a second one there may only agree with the first
    # -- two detections of one OTHER body. 2022 DF9: a 3-night seed (U 11.4) plus
    # two 2006 SDSS pairs each moving < 0.5" fit at 0.48" and passed every
    # per-tracklet clause, yet sit ~121,000" off DF9's real orbit. Such a
    # linkage is demoted to a reviewable possible when BOTH tests fail:
    #  * the identification penalty -- the seed's chi2 rise plus the new
    #    opposition's own chi2 under the joint orbit -- exceeds chi2_6 at this
    #    confidence (6 = the orbit's parameters; DF9: 23.8, cap 19.75);
    #  * some new-opposition tracklet, left out of a WARM refit (from the joint
    #    orbit, never a fresh solution), is predicted at p below this.
    # Measured on 520 MPC-resolved production new oppositions from
    # single-opposition seeds: the penalty alone refuses 9 correct links, the
    # leave-out alone 2; the pair refuses none (DF9: p 0.0033).
    new_opposition_penalty_conf: float = 0.997
    new_opposition_leaveout_p: float = 0.01
    # ---- Yarkovsky rescue (search._nongrav_rescue) -------------------------
    # A small NEO drifts: across a decade-plus gap the transverse (Yarkovsky)
    # acceleration exceeds the astrometry's own error, so a gravity-only fit
    # cannot hold both ends of the arc and throws the far tracklet's obs away.
    # 2021 JK2's 2011/2016 Pan-STARRS tracklets sit 14.7" and 6.2" off
    # gravity-only (F51 is good to ~0.1"); with Yarkovsky the 2011-2026 arc
    # fits 40/40 at 0.238".  So a trial rejected for being CLOSE BUT
    # DISAGREEING is retried with the model, on objects whose perihelion is
    # under this (1.4 is generous around the q < 1.3 NEO line -- Yarkovsky is
    # measurable exactly here; A2 is what JPL fits for hundreds of NEOs, while
    # radial A1 is not recoverable from optical astrometry at all).  0 disables.
    nongrav_rescue_q: float = 1.4
    nongrav_rescue_max: int = 3        # nongrav fits attempted per search
    # The NEO waiver of the opposition profile (see search._neo_orbit_waiver):
    # a linkage whose perihelion is under this may be accepted on a profile the
    # gate otherwise refuses -- a 1.5-opposition NEO identification is worth
    # publishing to the MPC recovery page as a PROPOSED link.  The waiver reads
    # the TRIAL FIT's perihelion, so it is only meaningful on a fit that has
    # one: an unbound solution's q is a mathematical artefact of a
    # range-degenerate arc, not a statement that the body comes near the Sun.
    # 2002 PS152 -- a catalogued 39.4 au TNO on 4 observations over 3 days --
    # linked one ITF night and fitted at e 6.81, a -0.02 au, "q" 0.11 au, which
    # bought it the NEO waiver and a `linked` verdict at 1.5 oppositions.
    # Hence `neo_waiver_max_e`: the waiver requires a BOUND orbit, and an
    # eccentricity the engine could not produce at all fails it (an
    # unverifiable orbit does not get NEO privileges).
    neo_waiver_q: float = 1.3
    neo_waiver_max_e: float = 1.0
    # Confirmed-pair upgrade of the thin 1.75-opposition profile: when the
    # DEPTH-0 sweep's top-2 tracklets by ML rank are what the search links --
    # either taken TOGETHER in one trial, or ending up as exactly the accepted
    # linkage however the trial queue accumulated them --
    # and both fit tightly, the two single-night extension oppositions are no
    # combinatoric coincidence: call it 3 oppositions and accept normally
    # (auto-submit allowed).  The ranking that names the pair is the SEED
    # orbit's, formed before either tracklet was linked, so neither one's
    # acceptance is what promoted the other.
    confirm_min_ml_p: float = 0.2              # both tracklets' P(match) >= this
    confirm_max_resid_arcsec: float = 0.5      # every linked obs |O-C| <= this
    # Seed-strength path of the same three-thin upgrade:
    # when the INITIAL arc is itself well-constrained -- longer than this many
    # days OR fit U at most this -- its corridor is tight enough that two
    # independently-matched single-night oppositions are no combinatoric
    # coincidence, however the recursion accumulated them: submittable.
    #
    # MEASURED 2026-08-19: planting one night in
    # each of two other apparitions of 40 known multi-opposition objects and
    # searching the live ITF from a 3-4 night seed, a CORRECT pair reached this
    # confirmation in 71 cases and 15 of them (21%) were refused -- every one
    # by this U cap, at U 6.02-8.89 with arcs of 2.7-28 d (i.e. the arc-days
    # path could not save them either).  Moving the cap, counted per case
    # against the pairs that carried a tracklet which is NOT the object:
    #
    #     U <= 6.0 (was)  +0/15 correct    +0/3 contaminated
    #     U <= 6.5        +4/15            +0/3
    #     U <= 7.0        +7/15            +0/3
    #     U <= 7.25 (now) +8/15            +0/3
    #     U <= 7.5        +8/15            +0/3   (buys nothing 7.25 did not)
    #     U <= 8.0       +12/15            +2/3   <- the cost starts here
    #     U <= 9.0       +15/15            +3/3
    #
    # 7.25 takes every refusal that is free -- it recovers the same 8 as 7.5
    # while keeping more margin to the nearest contaminated pair (U 7.95).  The
    # cost side rests on only THREE contaminated pairs, so this is a floor on
    # the price, not a measurement of it: re-run the study before going past
    # 7.5.
    confirm_seed_arc_days: float = 30.0
    # ALSO not shifted (2026-09-22): the one-night gate study priced this cap
    # per-case and says in as many words "re-run this study before going past
    # 7.5".  7.25 -> 7.75 crosses that line, and the nearest CONTAMINATED pair
    # in the study sits at U 7.95 -- a shifted cap cuts the margin from 0.70 to
    # 0.20.  The study's U values are on the old scale and do move up with
    # everything else, but by +0.50 at U 7.25 and +1.35 in the 7.5-9.0 band, so
    # which way the margin actually goes is exactly what the re-run must
    # measure.  A study-derived number gets re-derived, not arithmetic.
    confirm_seed_U_max: float = 7.25
    # Single-night extension (the 1.5 profile): shown to the user as a
    # possible/likely linkage -- never submittable (a second night outside the
    # initial opposition is required first).  ONE gate, three clauses (see
    # search._show_lone_night), all read off statistics the sweep already produced,
    # and path-agnostic: the dominance clause, not which corridor found the
    # tracklet, is the anti-coincidence guard.
    #
    # What it gates is ONE NIGHT, not one tracklet: a survey routinely
    # splits a night's detections of one object into two trksubs, and a trial
    # that takes both is a single identification to vet (2024 QF7's new night is
    # UT96489 + UT9674C, same station, 4+3 obs).  Every tracklet in the trial then
    # faces clause (a) on its own as well as pooled, and none of them is treated
    # as a RIVAL to the linkage it is part of when clause (b) measures
    # exclusivity.
    #
    # (a) IT FITS -- judged in SIGMA units, not overall arcsec rms (the overall
    #     rms may grow, e.g. 0.1"->0.2", when that growth is just a noisier new
    #     night).  The NEW night's residuals must be consistent with ITS OWN
    #     assigned uncertainties at this confidence (chi2 within the upper tail
    #     of its 2*n_obs dof), as a whole AND per tracklet -- a pooled chi2 can
    #     hide a bad tracklet behind a good one; and the ORIGINAL astrometry
    #     must not be made
    #     statistically worse -- re-fitting with the tracklet may move the seed
    #     obs' chi2 by no more than the seed's own STATE-dof confidence region at
    #     this level (a Wilks Delta-chi2 test, sigma held fixed at the seed's).
    show_new_tracklet_conf: float = 0.997      # ~3 sigma, new-tracklet own-fit
    show_seed_preserve_conf: float = 0.997     # ~3 sigma, seed Delta-chi2 region
    # (b) IT IS THE ONLY ONE THAT FITS -- exclusivity is MEASURED, not inferred
    #     from the ML scores.  Walk the ML-ranked field (likely + maybe, NOT just
    #     the "likely" geometric subset -- a high-P "maybe" is still a comparable
    #     fit) for the first candidate whose P(match) falls below
    #     show_exhaust_frac x the WINNER's P: the cliff.  Every rival ABOVE that
    #     cliff is then fitted as a competing hypothesis against the bare seed,
    #     and the winner is the sole standout only if none of them links.  A field
    #     that RUNS OUT of candidates before the bar needs no cliff -- it is finite
    #     and fully enumerable, so all of it is fitted (a field of one is a sole
    #     standout outright).  Only a field still above the bar AT rank
    #     show_exhaust_max_rank is one where exclusivity can never be claimed -- it
    #     takes the multiplicity banner without any fits.
    #     (Replaces the old show_dominance_frac COUNT of comparably-plausible
    #     candidates: a proxy for exclusivity, where this is evidence of it.)
    #     Rivals the trial queue already tried and rejected are not refitted.
    #     ...and AT LEAST show_exhaust_min_rivals of them are fitted whenever the
    #     field HAS that many, however sharp the cliff.  A cliff at rank 2 leaves
    #     no rival above it, so exclusivity used to be granted having fitted
    #     NOTHING -- "is this the only tracklet that fits?" answered by re-reading
    #     the very ML scores that chose the winner, which is not independent
    #     evidence of anything.  2024 PQ7 is why: off a 1-apparition seed (4.9 d,
    #     U 8.39, corridor out to 19 deg) the model gave P 0.976 to IUe7454 -- a
    #     tracklet 2.363 deg (8506", ~62000 sigma) from where PQ7 demonstrably was,
    #     the object's own 4-opposition arc being observed 26 days on either side
    #     -- against 0.127 for the SECOND-ranked candidate, P10RILV@F51/2019-09-24,
    #     which ALSO links.  (P10RILV is not PQ7 either: 2.643 deg / ~251000 sigma
    #     off the true orbit.  It shares a night and a telescope with PQ7's own
    #     2019-09-24 observations and sits 2.6 deg from them in the same wide
    #     Pan-STARRS field -- same night + same station is not identity, and an
    #     earlier note here claimed it was PQ7's own tracklet on exactly that bad
    #     inference.)  That two mutually incompatible strangers both link is the
    #     POINT: an under-determined seed has several compatible tracklets, so
    #     "sole standout" was false on its face and only an actual fit could say
    #     so.  Fitting that one rival collapses it into the multiplicity banner it
    #     should always have been.  The cost is
    #     bounded (one extra bare-seed fit per show attempt) and the suppression is
    #     correct by construction: a report is only withheld when a rival ACTUALLY
    #     LINKS, i.e. exactly when "sole" was false.
    show_exhaust_frac: float = 1.0 / 3.0
    show_exhaust_max_rank: int = 10
    show_exhaust_min_rivals: int = 1
    show_min_ml_p_abs: float = 0.15
    # ...and when NOT a sole standout (several comparable fits), still surface the
    # top candidate if its P(match) clears this higher bar -- but with a SECOND,
    # same-severity banner making clear it may be one of several compatible
    # tracklets.  Below this, a crowded field stays silent.
    show_multi_min_ml_p: float = 0.7
    # ...and BEFORE surfacing such a candidate, optionally COUNT how many of the
    # best-ranked rivals also link on their own (search.count_linking_rivals),
    # suppressing the show when too many do.  This is the only measurement the
    # multiplicity path has: when the ML field is too dense for a cliff inside
    # show_exhaust_max_rank, the exclusivity test fits NOTHING and the candidate
    # is shown on its P alone.  Measured over 103 shows:
    # precision 53/54 where exclusivity WAS
    # measured, 26/49 where it was not, and no ML-score density proxy separates
    # the two (~2:1 at best -- P is a ranker, not a fit test).
    #
    # Counting does separate them.  28 counting passes at k=6:
    #
    #     correct (planted truth)        0 x8, 1, 2
    #     real unlinked ITF detection    0, 0, 0, 1, 1, 1, 1
    #     wrong                          0, 1, 3, 3, 3, 3, 4
    #
    # so >= 3 linkers suppresses 5 of 7 wrong shows and 0 of 17 right ones,
    # with a one-count margin below the worst right show.  CAVEAT: all five of
    # those wrong shows are ONE object (702590), so this is a rare-event flag,
    # not a calibrated threshold -- and it is a COUNT, so a second crowded
    # object can move the cut without re-deriving anything.
    #
    # OFF by default (k = 0), because it costs k trial fits: median 12.4 s per
    # pass, +10% of a search's own runtime on the ~24% of searches that reach
    # this path, zero on the rest, ~+4-6.5 s amortised.  That price is
    # worth paying only where a HUMAN might act on
    # the banner and go dig through archival images, so the WEB layer turns it
    # on (web/engine_api.py) and every automated path -- `itf batch`,
    # the harness, the suite -- keeps today's behaviour and today's cost.
    show_n_rivals: int = 0                # 0 = never count
    show_rival_count_max_linkers: int = 3      # suppress at this many linkers
    # (c) Tier: 'likely' (vs the plainer 'possible') when the new tracklet fits
    #     tightly (reduced chi2 per dof at or under this) AND the best candidate
    #     clears a higher ml_p bar -- same statistics, one boundary.
    show_likely_new_redchi2: float = 1.5
    show_likely_min_ml_p: float = 0.25
    # ...and ONLY for a seed that is not itself badly under-determined: at U this
    # high the ML corridor calibration is meaningless and any single hit is
    # untrustworthy.  NB: distinct from variant_corridor_U (4.0), which merely
    # decides when the variant FALLBACK runs; a healthy single-opp seed routinely
    # sits at U~4-6 and must still be reportable.
    # ---- span bookkeeping --------------------------------------------------
    # The SPAN the acceptance opposition-profile gate counts in (itf/opp_profile.py:count_spans):
    # a window opened by its first observation, so two calendar nights 22 h
    # apart are ONE span.  The default is the ported notebook unit the DESITF
    # thresholds (spans >= 8 / 6 / 4 / 3) are calibrated on -- 1.6 d in the
    # notebook, 1.5 d (a 36-hour span) since 2026-09-22, the count
    # being identical on every recovery fixture at both.
    #
    # It is a knob because a 36-hour span is wrong for a population of ITF-only
    # linkage candidates.  Measured on OrbImp's 49 725 three-night candidates, the gaps
    # between consecutive observations are sharply bimodal: 285 k below 0.3 d
    # (within a night), a near-empty trough from 0.3 to 0.7 d (312 gaps in
    # total), then the next-night mode at 0.8-1.1 d.  The 1.6 d rule merges that
    # second mode into the first, so a third of the pool (15 783 candidates) is
    # counted as TWO nights where the linker -- and an observer -- sees three.
    # Those candidates would then need two new nights rather than one to clear
    # `nights >= 4`, and a correct fourth night would be refused.
    #
    # 0.5 sits in the middle of the measured trough.  Note this LOOSENS whatever
    # clause reads it (the night count is >= the span count for every arc),
    # which is why it is scoped to a config field rather than changed globally:
    # an ITF-to-ITF linking campaign sets it to a night (opp_profile.NIGHT_DAYS).
    span_days: float = 1.5

    # Per-night residual COHERENCE cap (see itf/night_coherence.py): a
    # linkage whose orbit cannot fit the nights it links leaves each night
    # coherently displaced rather than scattered.  Above this the linkage is
    # REFUSED at the quality gate, so the trial queue moves to the next trial instead
    # of stopping on a bogus one -- which is the point: a wrong trial that is
    # ACCEPTED costs the object its chance at the right trial.
    #
    # 0 DISABLES it, and that stays the shipped default: only an ITF-to-ITF pool
    # arms it, because that is the
    # population whose failure mode it was built for.
    #
    # ARMING IT IS NO LONGER A REGRESSION, and that is measured rather than
    # assumed.  The FIRST version of this clause did cost a real recovery at 2.0
    # -- 856215, a 19-night 5-opposition linkage of 90/91 observations fitting
    # at 0.242" -- but only because its statistic grew with the NUMBER of
    # nights; the current one divides that out (see itf/night_coherence.night_coherence
    # and _expected_max).  Re-measured 2026-08-16 at 2.0:
    #   * the 14-object five-opposition harness suite is 14/14, 856215 included
    #     (it was 13/14 under the old statistic, and 856215 was the loss);
    #   * all 24 offline recovery fixtures recover, pinned by
    #     tests/test_itf_recovery.py::test_recovers_with_the_night_coherence_clause_armed.
    # The margin is real but not wide: the worst fixture measures 1.84 (2015
    # KJ19, 30 nights) against the 2.0 cap.
    #
    # What the clause diagnoses is a SHORT, single-station arc that the orbit
    # cannot reconcile -- the ITF-to-ITF population, whose tracklets are typically
    # two exposures ~70 s apart, so itf/residual_trend.py (which needs residuals walking
    # across a tracklet) is structurally inert there and nothing else looks at
    # arrangement at all.  The seven ITF-to-ITF linkages whose tracklets are KNOWN
    # correct re-measure 0.044-0.929 on the current scale (median 0.126), so 2.0
    # keeps real margin there.  Its DETECTION side is unproven: the worked
    # failure the module was written from (Z001y6z) re-measures 0.743, i.e. this
    # threshold would not refuse it.
    max_night_systematic_sigma: float = 0.0

    # NOT shifted with the U scale, deliberately (2026-09-22).  The obvious
    # move when the s2 floor raised every U was to slide this cap up with it
    # (+1.40 at U 9).  It is not taken, on the study rather than on a guess:
    # the one-night gate study prices this cap over 40 objects of constructed
    # truth and finds it the SOLE cause of refusing 57 wrong one-night attempts
    # (325 fails in all), while the show withholds only 2 of 72 CORRECT nights
    # -- and neither of those two is refused by THIS clause.  So the gate was
    # not costing recall at 9.0, and a wider cap buys wrong shows and nothing
    # else.  A study-derived number gets re-derived, not arithmetic.
    #
    # NB what raising it did to 2015 BE559 (seed U 9.56) live: it stopped at a
    # 0.6 carrying a FOREIGN tracklet instead of staying silent.  That is the
    # cap doing its job.  It is NOT why BE559 lost its 4 tracklets -- measured:
    # reverting this cap alone still gives 0/4 -- so BE559's regression lies
    # elsewhere and is tracked separately.
    single_night_show_max_U: float = 9.0
    # Memory guard for very wide corridors (sloppy seeds, U >~ 9): keep at
    # most this many candidate tracklets per window, best match-chi2 first,
    # before their observations are fetched. A noise-dominated window's tail
    # is unrankable anyway; 0 disables.
    max_candidates_per_window: int = 1000
    # WORK budget for one corridor sweep: stop querying once this many index
    # feature rows have been corridor-tested.  Windows are swept narrowest
    # sky-sigma FIRST (sigma bands, chronological within a band), so on a normal
    # corridor a budget stop drops the degree-wide noise-dominated tail a
    # short-arc seed produces -- a healthy deep search tests ~150k rows and never
    # notices (2015 BE559's severe LINEAR corridor: 188k rows, nothing skipped).
    # CAVEATS, measured 2026-07-23: (a) the LOV variant scan sweeps a UNION of
    # n_scan corridors and so gets its OWN budget (variant_sweep_row_budget) --
    # against this single-corridor value it truncated 7361 of BE559's 19459
    # windows; (b) every variant window has the SAME fixed width, so the
    # sigma-band ordering is degenerate there and a stop drops a contiguous block
    # of epochs, not a noise tail; (c) whenever windows ARE dropped the search now
    # says so in `result.warnings` + `telemetry["windows_skipped"]`, because a
    # negative verdict after an incomplete sweep is not a clean negative.
    # 0 disables.
    sweep_row_budget: int = 500_000
    # ...and the same WORK budget for the variant scan's single union sweep,
    # which covers n_scan corridors in one pass and cannot be bounded by a
    # one-corridor number (caveat (a) above).  Deliberately its own knob rather
    # than `sweep_row_budget x n_scan`: this budget's job is politeness to a
    # SHARED server, and multiplying the polite number by up to variant_max_scan
    # is how a valve quietly stops being one.  4M = 8x the single-corridor value:
    # the largest measured union sweep (BE559, 31 variants over 19459 windows)
    # tests 502k rows -- which is why the single-corridor number truncated 38% of
    # its windows -- so 8x bounds a runaway with room to spare while not biting a
    # real scan.  0 disables.
    variant_sweep_row_budget: int = 4_000_000
    # ...and the MEMORY floor those budgets cannot express: the row budgets are
    # politeness to the server, counted in rows TESTED, but what kills the
    # process is rows RETAINED -- the variant scan caches every union-sweep
    # window's rows for the per-variant replays and the LOV profile pass, so a
    # U~10 seed's union (2006 SJ252: 55 windows and 1062 tracklets at the
    # nominal corridor alone, times the fan) grows the process until the OOM
    # killer ends the search with no verdict and no traceback.  When available
    # system memory falls under this floor, the sweep stops serving and the
    # remaining windows are counted in `last_sweep_skipped` -- the same honest
    # partial-sweep accounting the row budget uses, so the search still ends
    # in a verdict that SAYS the sweep was incomplete.  The floor covers more
    # than not-dying-mid-fetch: the sweep is SETUP, and the LOV profile pass and
    # variant rounds that follow do their work in whatever it leaves behind --
    # SJ252's union sweep finished at ~550 MB free under a 500 floor and the
    # LOV profile churn then ate the rest in seconds.  "Available" is measured by
    # `itf.memory.available_mb` and counts the CONTAINER's headroom as well as
    # the host's free memory -- reading /proc/meminfo alone (as this floor did
    # until 2026-08-19) leaves it inert on every containerised deploy, where
    # meminfo is the HOST's and the cgroup OOM killer arrives first.
    # 0 disables.
    sweep_min_free_mb: int = 1000
    # ---- noise / variant-orbit hack ----------------------------------------
    # When on, gaussian-jitter the KNOWN object's input astrometry (per the
    # per-observatory sigma table) to explore variant orbits across repeated
    # runs; ITF candidate tracklets are never touched.
    jitter: bool = False
    jitter_default_sigma: float = 0.26          # arcsec
    jitter_sigma_t: float = 0.6 / 86400         # days
    jitter_sigma_table: dict = field(default_factory=lambda: {
        "413": 0.3, "422": 0.3, "568": 0.1, "673": 0.1, "689": 0.1,
        "F51": 0.2, "F52": 0.2, "J95": 0.2, "T09": 0.1, "T12": 0.1,
        "T14": 0.1, "C51": 1.2,
    })

    # ---- database ----------------------------------------------------------
    # Hosts/identity come from the environment (SBN_DB_HOST, ITF_SUBMIT_NAME/EMAIL --
    # exported var or .env) so a server move or another operator needs no code
    # change.  There is no built-in host: every operator runs its own SBN database
    # clone, and an unset SBN_DB_HOST fails at connect with a message naming it.
    db_name: str = "sbn"
    db_host: str | None = field(default_factory=lambda: _env("SBN_DB_HOST", None))
    db_port: int = 5432
    db_connect_timeout: int = 15
    # SERVER-side ceiling on ONE sweep query, seconds (0 = none).  connect_timeout
    # above bounds only the handshake; nothing bounded a QUERY, and that is how a
    # single corridor chunk hung a whole search: 2025 NV385's variant sweep is 315
    # chunks and one of them sat in cur.execute for 45+ min (py-spy: parked in the
    # driver, flat CPU).  2025 YQ13 and 2026 EP114 hung in the same query on
    # unmodified code, so none of those three has ever produced a clean negative.
    # SearchBudget cannot catch it -- it is consulted BETWEEN trials and variants,
    # never inside a sweep.
    #
    # 120 s, not the 900 s first considered: honest chunk queries run in
    # milliseconds to a few seconds, so this is already ~30x headroom over real
    # work, while 900 s x 315 chunks is a 78-HOUR worst case.  Server-side
    # (statement_timeout) rather than a client socket timeout, so a cancelled
    # query stops costing the SHARED server instead of running on unwatched.
    # Applies to the search role only (connect_pub); index maintenance runs on
    # connect_index and legitimately takes minutes.
    sweep_statement_timeout_sec: int = 120
    # ...and a per-statement ceiling does not bound a SWEEP (315 x the ceiling),
    # so a run of cancellations abandons the rest of it: past this many, the
    # corridor is not slow, it is one this server will not serve.  The dropped
    # windows go to last_sweep_skipped -> the search's "corridor was NOT swept
    # completely ... not a clean negative" warning, so recall lost this way is
    # always reported.  0 = never abandon.
    sweep_max_cancels: int = 3
    db_user: str = "pub"                       # read-only search role
    db_password: str | None = None             # else from env var below / .env
    db_password_env: str = "SBN_PUB_PASSWORD"
    db_index_host: str | None = field(default_factory=lambda: _env("SBN_DB_HOST", None))
    db_index_user: str = "itf"                 # owns the itf schema
    db_index_password: str | None = None
    db_index_password_env: str = "ITF_DB_PASSWORD"

    # ---- submission (always dry-run unless explicitly enabled) -------------
    # The submitter identity goes to the MPC in the identification document.
    # NO built-in identity, by design: the web takes the name / e-mail the
    # person remembered in their browser, the queue passes the submitter's header
    # through, an automated monitor carries its own, and anything
    # else -- the CLI's --submit, itf batch -- needs ITF_SUBMIT_NAME /
    # ITF_SUBMIT_EMAIL, or submit_linkage refuses to send an anonymous document.
    submit: bool = False
    submit_url: str = ("https://minorplanetcenter.net/mpcops/submissions/"
                       "identifications/")
    submit_email: str = field(default_factory=lambda: _env("ITF_SUBMIT_EMAIL", ""))
    submit_name: str = field(default_factory=lambda: _env("ITF_SUBMIT_NAME", ""))
