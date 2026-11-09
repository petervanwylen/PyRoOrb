"""The core ITF search: corridor sweep -> candidate tracklets -> trial fits ->
recursion on improvement -> acceptance gates -> linkage.

A native port of the notebook pipeline's ``itf_search``, with the find_orb
engine replaced by PyRoOrb end to end:

* the seed object is fetched/fitted with :func:`fit_object.fit_orbit`;
* the ephemeris corridor comes from :mod:`pyroorb.itf.ephem` (our propagator +
  covariance sky-sigma, geocentric);
* candidate tracklets come from :mod:`pyroorb.itf.candidates` (itf_tracklets fast
  path or legacy scan, or the harness overlay);
* every trial linkage is a full PyRoOrb fit (:mod:`pyroorb.itf.fitter`), whose
  included/total counts play the role of find_orb's used/count;
* poorly-fitting obs are clipped by residual threshold
  (:func:`pyroorb.itf.fitter.drop_offending`), 0.5" for well-determined orbits
  (U < 3.5) else 0.74", AND beyond clip_sigma_mult times the obs' own sigma;
* the found linkage is assembled/submitted by :mod:`pyroorb.itf.linkage`
  (dry-run unless explicitly enabled twice).

Control flow, scoring, categories, backoff and the opposition/night
acceptance gates are ported faithfully -- they encode years of tuning against
real MPC submissions.  Status codes keep the legacy meanings: 1 = linkage
found (submitted or dry-run), 0 = nothing, 0.6 = a reviewable POSSIBLE, never
transmitted: a thin 1.5- or accumulated 1.75-opposition profile, a linkage
demoted by a seed-degradation or withdrawn-confirmation warning, or a partial
pickup whose reduced document would not stand.

How one search runs
-------------------
:func:`itf_search` fits the seed and calls :func:`_iterate` at depth 0; when
that does not link it may re-sweep deeper in magnitude and then scan LOV
variant orbits (:func:`_variant_scan`), each variant another ``_iterate``.

One ``_iterate`` call is one ROUND: sweep the orbit's corridor
(:func:`_collect_tracklets`), rank the candidates (:func:`_rank_candidates`),
and pull takes from a :class:`TrialQueue` -- tracklet subsets, each fitted,
PRUNED of blunder obs, and taken only if it progresses without degrading the
seed (:func:`_trial_rejection`).  A take becomes the HYPOTHESIS: its
tracklets ride in every deeper trial fit and leave the candidate pool, so the
next round asks only "of the remaining candidates, which next?".

The recursion UNWINDS through :func:`_vet_terminal`.  A child that LINKED is
returned as is.  Otherwise the parent's own take is the terminal candidate,
judged -- with its alternatives, stopping at the first LINKED -- by ONE
verdict, :func:`_judge`: the take itself; the take minus the tracklets that
failed their own clause (:func:`_repaired`); the non-fragile same-opposition
part of a 1.75 (:func:`_reduced_linkage`); the child's reviewable 0.6.
Without a LINKED answer the strongest verdict wins.  With backtracking on, a
depth-0 take whose whole subtree came back with nothing hands the round's
trial queue its next take.

How one linkage is judged
-------------------------
:func:`_assess` asks whether it is CONSISTENT: it gains astrometry, every
linked tracklet holds its observations and fits their uncertainties, and the
seed is still the object (:class:`SeedCheck`, every test of what the link did
to the object's own observations).  An inconsistent linkage is refused -- or,
on a thin profile off a one-opposition seed, reported as a distrusted 0.6.  A
consistent one is LINKED when its profile is submittable and nothing demotes
it (:func:`_accepted`: a materially degraded seed, a caution on a confirmed
1.75, the document rules); a thin profile is a 0.6 (:func:`_thin_possible`,
:func:`_lone_night`).
"""

import math
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from pyroorb import deadline
from pyroorb.packing import pack, unpack
from pyroorb.orbit_common import night_index

from . import candidates as _candidates
from .candidates import source_chain
from .memory import (breakdown as _mem_breakdown,
                     sweep_floor_mb as _sweep_floor_mb,
                     search_capacity as _search_capacity,
                     CAPACITY_OK as _CAPACITY_OK, CAPACITY_SHALLOW as _CAPACITY_SHALLOW)
from .memory_sampler import RssSampler
from .config import ItfConfig
from .verdict import SearchVerdict
from .ephem import sweep_nodes, corridor_windows, faint_cut
from .fitter import (fit_astrometry, seed_perturbers, drop_offending, residuals_by_obs_id,
                     subset_rms, subset_sigma_floor, subset_chi2, sigma_map_for,
                     subset_included, worst_mag_residual, mag_outlier_tracklet,
                     screen_candidate, per_axis,
                     SCREEN_REJECT, SCREEN_PLAUSIBLE)
from .linkage import (build_linkage, submit_linkage, collect_ids, db_trksubs,
                      line_designation, line_identifier, comet_refusal)
from . import anchor, gatelog, gatestats, mpcrules
from .obsreg import AdesRegistry
from . import pickup
from .residual_trend import TREND_NOTE, measure_trends, two_axis_trend_warning
from .obs80_fields import (obs_unique_id, obs_ids, drop_obs_ids, count_obs, split_obs80_long_lines,
                           n_observations)
from .opp_profile import (obs_night_ordinal, count_spans, count_nights, opp_profile, spans_of, nights_of, OppProfile, OppProfileGrade)
from .jitter import maybe_jitter
from .decisions import trksubs_of
from .trace import (Tracer, as_tracer, day_of, tracklet_tag, span_of,
                    histogram, L_HEAD, L_STORY, L_DETAIL, L_FIREHOSE)

_MAX_DEPTH = 20                 # recursion safety net (progress is required
                                # each level, so real searches stay shallow)

_NULL_TRACER = Tracer()         # disabled sink


def _ctx_registry(ctx):
    """The ctx's ADES registry (:class:`itf.obsreg.AdesRegistry`), or None for
    hand-built test ctx objects.  None is never an error: it means "fit the
    80-column data"."""
    return getattr(ctx, "registry", None)


def _ctx_tracer(ctx):
    """The ctx's tracer, tolerating test ctx objects with no tracer or a bare
    ``callable(str)`` sink.  Coerced to a :class:`Tracer` because callers use
    its levelled helpers (``head``/``field``/``count``)."""
    trace = getattr(ctx, "trace", None)
    if trace is None:
        return _NULL_TRACER
    return trace if isinstance(trace, Tracer) else as_tracer(trace)


class SearchBudget:
    """The search's wall-clock budget: ``no_progress_sec`` without a single
    tracklet being TAKEN, and an absolute ``ceiling_sec`` on the whole search.

    The no-progress clock RESETS on every take (:meth:`took_a_trial`) except
    one a trial queue reached by resuming after a barren take (a backtrack is
    the round still looking, not progress): a
    fruitless stretch is bounded, a productive but slow search is not cut off
    (2015 BE559 was, under DB load, on a plain deadline).  The ceiling keeps
    interactive callers safe from a take-forever loop.

    Shared by reference, so the variant scan's ``dataclasses.replace(ctx, ...)``
    copy and the parent frame bill against the SAME budget."""

    def __init__(self, no_progress_sec, ceiling_sec, setup_credit_max_sec=0.0,
                 min_fruitless_fits=0):
        self.no_progress_sec = no_progress_sec
        now = time.perf_counter()
        self.no_progress_at = now + no_progress_sec
        self.ceiling_at = now + ceiling_sec if ceiling_sec > 0 else float("inf")
        self.extensions = 0
        # Fits launched since the last progress.  A wall clock cannot tell a
        # doomed grind from a slow machine; a fit count can (see expired).
        # Needed for 2024 PQ7; on a busy machine its correct linkage was cut off.
        self.min_fruitless_fits = int(min_fruitless_fits or 0)
        self.fits_since_progress = 0
        self.ceiling_sec = float(ceiling_sec) if ceiling_sec > 0 else 0.0
        self.retry_granted = False
        self.setup_sec = 0.0            # one-time setup excused (see setup_done)
        self.setup_credit_max_sec = (float(setup_credit_max_sec)
                                     if setup_credit_max_sec > 0
                                     else float("inf"))

    def grant_retry(self, frac):
        """A BOUNDED extra allowance for the continuous LOV offset re-scan.

        Pass 1 of the variant scan ends by spending the budget, so the re-scan
        would be refused before it started.  Moves both clocks out by ``frac``
        of the original ceiling, at most ONCE -- never an open extension.
        Returns whether it was granted.
        """
        if self.retry_granted or not self.ceiling_sec or frac <= 0:
            return False
        self.retry_granted = True
        extra = self.ceiling_sec * float(frac)
        now = time.perf_counter()
        self.ceiling_at = (now + extra if self.ceiling_at == float("inf")
                           else self.ceiling_at + extra)
        self.no_progress_at = max(self.no_progress_at,
                                  now + min(extra, self.no_progress_sec))
        self.fits_since_progress = 0
        return True

    def took_a_trial(self):
        """A tracklet subset was accepted and the search is recursing on it --
        real progress, so the no-progress clock starts over."""
        self.no_progress_at = time.perf_counter() + self.no_progress_sec
        self.fits_since_progress = 0
        self.extensions += 1

    def note_fit(self):
        """One trial fit was launched and did not become progress."""
        self.fits_since_progress += 1

    def setup_done(self, seconds):
        """A ONE-TIME SETUP cost of ``seconds`` just finished -- the variant
        scan's single union DB sweep.  It is the price of admission to the work
        the clocks bound, not a fruitless grind; charging it let 2015 BE559's
        variant scan expire live having tried nothing.

        The no-progress clock restarts and the ceiling moves out by
        ``seconds``.  The ceiling credit is capped in total at
        ``setup_credit_max_sec`` (cfg.search_setup_credit_max_sec), so the web
        UI's bound holds whatever the call graph; past the cap only the
        self-limiting no-progress clock is restarted."""
        secs = max(0.0, float(seconds))
        self.no_progress_at = time.perf_counter() + self.no_progress_sec
        credit = max(0.0, min(secs, self.setup_credit_max_sec - self.setup_sec))
        self.ceiling_at += credit
        self.setup_sec += secs

    def expired(self):
        now = time.perf_counter()
        if now > self.ceiling_at:
            return True
        if now <= self.no_progress_at:
            return False
        # Wall clock spent -- but that alone describes the MACHINE.  Expire only
        # once the search has also burned fits without progress (a grind burns
        # them whatever the machine speed).  min_fruitless_fits 0 = clock-only.
        return self.fits_since_progress >= self.min_fruitless_fits

    def why(self):
        if time.perf_counter() > self.ceiling_at:
            return "the overall ceiling"
        return (f"{self.no_progress_sec:.0f}s and "
                f"{self.fits_since_progress} fruitless fits without progress")


def _budget_spent(ctx):
    """Is the search's budget spent (see :class:`SearchBudget`)?  Asked before
    LAUNCHING more work, so a grind stops within one in-flight fit.  No budget
    (None, or a test ctx without one) is never spent."""
    # The caller may have stopped the search (the web UI's Stop): this is the
    # point every launch of new work passes, so the search unwinds from here
    # rather than finishing a round nobody is waiting for.
    deadline.check_cancelled()
    budget = getattr(ctx, "budget", None)
    return budget is not None and budget.expired()


def _note_fit(ctx):
    """Tell the budget a trial fit was launched (see SearchBudget.note_fit)."""
    budget = getattr(ctx, "budget", None)
    if budget is not None:
        budget.note_fit()


def _note_progress(ctx):
    """Tell the budget a trial was taken, so its no-progress clock restarts."""
    budget = getattr(ctx, "budget", None)
    if budget is not None:
        budget.took_a_trial()


def _note_setup(ctx, seconds):
    """Tell the budget a one-time setup cost of ``seconds`` was paid (see
    :meth:`SearchBudget.setup_done`)."""
    budget = getattr(ctx, "budget", None)
    if budget is not None:
        budget.setup_done(seconds)


@dataclass
class ItfSearchResult:
    """Outcome of one search.  ``verdict``: LINKED (a submittable linkage),
    POSSIBLE (kept for review, never submittable) or NONE -- see
    itf/verdict.py; the legacy 1 / 0.6 / 0 is ``verdict.legacy_status``."""
    verdict: SearchVerdict = SearchVerdict.NONE
    reason: str = ""
    fit: object = None                  # final accepted FittedOrbit (LINKED)
    linkage: dict | None = None
    submitted: bool = False
    combined_obs80: str = ""            # published + accepted ITF astrometry
    itf_obs80: str = ""                 # the accepted ITF lines alone
    # {obs_unique_id: trksub} of ``itf_obs80`` from the DB column -- the name
    # the document gives each tracklet (linkage.db_trksubs); pass it back in
    # when rebuilding the document from the lines
    itf_trksubs: dict = field(default_factory=dict)
    telemetry: dict = field(default_factory=dict)
    # this level's sweep retrieved no likely candidate at all, so the parent
    # can skip its provably redundant final sweep (see _iterate)
    no_candidates: bool = False
    # how much an accepted link degraded the fit, for the caller to warn on
    # (a status 0 with reason "degrades…" was already refused)
    warnings: list = field(default_factory=list)
    vetting: dict = field(default_factory=dict)
    # ADES fields behind ``combined_obs80``, keyed by line identity, so a caller
    # re-fitting that text fits the same data the search did (itf/obsreg.py)
    registry: object = None
    # Set when part of the corridor was never queried: {"windows", "lowmem",
    # "cancelled", "budget", "message", "verdict_affected"}.  A structured twin
    # of the warning text so the web UI can show a banner WITHOUT it entering
    # the submit-acknowledgement gate: an incomplete sweep makes a found
    # linkage partial, not doubtful.
    incomplete_sweep: dict | None = None
    # Advisory measurements that are NOT doubts about the linkage (e.g. the
    # residual trend, see itf/residual_trend.py).  Kept out of ``warnings``,
    # which gate the submit click-through and withdraw a fragile confirmation;
    # shown by the CLI, web UI and batch CSV, acted on by nothing in the search.
    diagnostics: list = field(default_factory=list)
    # the seed fit's U, for DISPLAY; None if the search ended before a seed
    # fit.  The catalogue's ``u_param`` is 0 on many undetermined rows.
    seed_U: float | None = None
    # The other half of a submittable/complete pair (ItfConfig.prefer): when
    # the search holds both a LINKED linkage and a reviewable 0.6 that links
    # strictly more of the ITF, the one ``prefer`` did not pick rides here.
    alternative: "ItfSearchResult | None" = None

    @property
    def reviewable(self):
        """A verdict someone can actually REVIEW: a linkage, or a 0.6 with a fit
        and its ITF astrometry attached.  A bare 0.6 (e.g. "1.5 opps without
        progress") is a barren branch.  Backtracking's stop condition and the
        0.6-supersede path both use this and must never disagree."""
        return (self.verdict is SearchVerdict.LINKED
                or (self.verdict is SearchVerdict.POSSIBLE
                    and self.fit is not None
                    and bool(self.itf_obs80)))


def _verdict_rank(result):
    """Sort key comparing the linear pass and the LOV variant fallback on
    EVIDENCE, not on which corridor ran first.

    ``verdict`` dominates; ties (two 0.6s) go to the one carrying a fit, then
    to the one the exhaustion test proved exclusive (``single_best_fit``).
    Needed for 2024 QT17; without it, the linear pass's weaker verdict buried
    the variant scan's sole-standout one."""
    if result is None:
        return (-1, 0, 0)
    return (result.verdict.rank,
            1 if getattr(result, "fit", None) is not None else 0,
            1 if (result.vetting or {}).get("single_best_fit") else 0)


def _fuller_possible(linked, possible):
    """Is ``possible`` a reviewable 0.6 that links STRICTLY MORE of the ITF
    than the LINKED ``linked`` -- every observation ``linked`` took, and more?

    The verdict rank alone would let ``linked`` replace it, discarding
    whatever the 0.6 found beyond it (2014 GV99: a same-opposition tracklet,
    LINKED, against a 0.6 adding two single-night oppositions)."""
    if (linked is None or possible is None
            or linked.verdict is not SearchVerdict.LINKED
            or possible.verdict is not SearchVerdict.POSSIBLE
            or not possible.reviewable):
        return False
    mine = obs_ids(linked.itf_obs80 or "")
    theirs = obs_ids(possible.itf_obs80 or "")
    return bool(mine) and mine < theirs


def _pair(linked, possible):
    """Keep a fuller reviewable 0.6 (:func:`_fuller_possible`) on the LINKED
    linkage as its ``alternative``.  Inside the search the linkage always
    leads: every decision the search makes from here on (backtracking, the
    variant fallback fan) sees exactly the verdict it saw before the pair was
    kept.  Which one the caller gets first is decided once, at the end
    (:func:`_choose`)."""
    possible.alternative = None             # one pair, never a chain
    linked.alternative = possible
    return linked


def _choose(cfg, result):
    """The search's answer: when ``result`` is a linkage still carrying a
    fuller 0.6, ``cfg.prefer`` picks which one leads ("submittable" -> the
    linkage, "complete" -> the 0.6) and the other rides on it as
    ``alternative`` -- the consumer decides.  A pair that no longer holds
    (the linkage changed after it was formed) is dropped."""
    alt = getattr(result, "alternative", None)
    if alt is None:
        return result
    if not _fuller_possible(result, alt):
        result.alternative = None
        return result
    if getattr(cfg, "prefer", "complete") == "submittable":
        return result
    result.alternative = None
    alt.alternative = result
    return alt


def alternative_note(result):
    """One line naming the result the search held besides ``result`` (its
    ``alternative``), for a command-line report; "" when there is none."""
    alt = getattr(result, "alternative", None)
    if alt is None:
        return ""
    n = count_obs(alt.itf_obs80 or "")
    if alt.verdict is SearchVerdict.LINKED:
        return (f"ALSO: a submittable linkage (status {alt.verdict.legacy_status}) of "
                f"{n} ITF obs, fewer than shown -- rerun with prefer=submittable to lead "
                f"with it")
    return (f"ALSO: a fuller unconfirmed result (status {alt.verdict.legacy_status}) of "
            f"{n} ITF obs, more than shown -- rerun with prefer=complete to lead with it")


@dataclass
class SearchContext:
    source: object
    cfg: ItfConfig
    obscodes: dict
    pub_obs80: str
    pub_ids: set
    pub: object                          # TrialFit of the published seed
    initial_opp_profile: OppProfile
    primary_desigs: list
    max_uncert_deg: float
    uncert_mult: float
    do_submit: bool
    co_credit: str | None
    telemetry: dict
    log: object
    verbose: bool
    collector: list | None = None
    # set once a Yarkovsky rescue saved a trial; every later fit keeps the
    # model on (see _nongrav_rescue)
    nongrav: bool = False
    # ...and the rescued fit itself: later fits hold its force fixed and start
    # from its orbit (itf.fitter.fit_astrometry ``nongrav_ref``)
    nongrav_ref: object = None
    # the asteroid field the seed needed beyond the 16 (a perturbers.Field), or None: every
    # in-search fit runs in it (see fitter.seed_perturbers, cfg.seed_perturber_screen)
    perturbers: object = None
    # obs ids of the PRIOR ACCEPTED state at the depth being vetted (seed +
    # shallower takes), the second seed-preservation reference (see
    # _accept_quality_ok); None outside a vetting
    prior_ids: set | None = None
    # packed/unpacked designations that ARE the seed object (own_designations)
    own_desigs: set = field(default_factory=set)
    # ADES fields the 80-column rendering can't hold, keyed by line identity;
    # every fit in the search is enriched through it (itf/obsreg.py)
    registry: object = None
    # itf/lov_refine.py's measured geometry, tracklet id set -> (k*, residual
    # off the LOV curve, nights agreeing at k*); blended into the ranking by
    # _rank_candidates
    lov_refine: dict | None = None
    # PASSIVE decision recording for ranker training (itf/decisions.py);
    # nothing reads it back, so a logged search decides as an unlogged one
    dlog: object = None
    # LOV offset geometry kept only as ML FEATURES (shape of ``lov_refine``,
    # which is the one _rank_candidates acts on)
    lov_feat: dict | None = None
    # the LOV variant whose corridor the current depth-0 variant round sweeps
    variant_lov_sigma: float | None = None
    trace: Tracer = field(default_factory=Tracer)   # detail feed (see itf.trace)
    budget: "SearchBudget | None" = None    # the no-progress clock + overall
                                            # ceiling (None = unbounded)
    # every tracklet any trial took, EACH obs id -> its full pre-prune id set:
    # acceptance judges a tracklet by the obs it was SWEPT with, not what
    # survived pruning (see _accept_quality_ok); over-collection is harmless
    swept_tracklets: dict = field(default_factory=dict)
    # each such tracklet's lines as swept, keyed by its id set: what _vouch
    # restores when it accepts a tracklet the clip had cut down
    swept_lines: dict = field(default_factory=dict)
    # the DEPTH-0 sweep's two best-ranked candidates, recorded before any
    # trial: [{"ids", "ml_p", "tag"}, ...], None if fewer than two.  A linkage
    # of exactly these two is confirmed-pair evidence (_confirm_top_pair path
    # 3).  Each depth-0 sweep, including a LOV variant's, overwrites it.
    depth0_top2: list | None = None
    # True while the LOV variant scan drives: its corridors are fixed-width, so
    # the ML row records the regime (training.is_variant)
    is_variant: bool = False
    # set once the terminal LOV fan has run (ItfConfig.terminal_fan): it is
    # scanned at most once per search.  A declared field so that
    # dataclasses.replace -- how a variant round derives its context --
    # carries it into every round.
    terminal_fan_done: bool = False
    # Documents a vet built while the terminal fan could still supersede its
    # verdict: ``(result, doc, trkid_map)``, sent only if that result stands
    # (see _release_held).  None = transmit at once.
    held_transmits: list | None = None
    # why the one-night show last declined a night that DID link, printed by
    # the rejection ledger instead of "without progress"; None = not asked
    show_refusal: str | None = None
    # the one-night show's P(match) readings for its last trial:
    # {"best_in_field", "this_candidate", "floor"} -- the gate reads the first,
    # the reader is looking at the second
    show_ml: dict | None = None
    # SEARCH-WIDE ledger of hypotheses already fitted, keyed by
    # :func:`_hypothesis_key` (observations + force model): each is fitted ONCE
    # per search.  The TrialQueue's ``tried_keys`` cannot do this -- it is
    # rebuilt per round and keyed by that round's candidate indices.  Needed
    # for 2006 UP17; without it, one tracklet was re-fitted once per variant.
    tried_hypotheses: dict = field(default_factory=dict)
    # SEARCH-WIDE record of the seed obs each retention-refused trial clipped,
    # ``[(clipped seed obs ids, UT dates of the trial's candidate obs), ...]``:
    # the evidence :func:`_corroborated_seed_clips` reads.  Shared by every
    # round, since dataclasses.replace copies the reference.
    seed_clip_votes: list = field(default_factory=list)


def _pub_obs80_from_fetch(designations, obscodes, cfg, log=print,
                          registry=None):
    """Fetch the seed object(s) from the MPC and return (obs80 text, primary
    designations).

    ADES is requested and the obs handed to ``registry``, so their reported
    rmsRA/rmsDec, rmsMag, logSNR and obsID survive into the fit.  The search's
    bookkeeping and the MPC document are line-based, so every obs is RENDERED
    to 80 columns (obs80_base_line: verbatim if it carried a line, else
    synthesized -- which lets an ADES-only or NEOCP object seed at all); the
    registry puts back, by line identity, what that drops.

    Raises RuntimeError if any designation's fetch failed."""
    from pyroorb.obs_sources import fetch_observations, fetch_failure_message
    from pyroorb.obs80 import obs80_base_line
    obs, meta = fetch_observations(designations, obscodes)
    _fmsg = fetch_failure_message(meta)
    if _fmsg:
        # A partially-fetched seed aims the corridor by half the arc and would
        # answer "no match" with conviction; fail with the transport error.
        raise RuntimeError(f"seed fetch: {_fmsg}")
    lines, skipped = [], 0
    for o in sorted(obs, key=lambda o: o["jd_utc"]):
        try:
            parts = obs80_base_line(o).splitlines()
        except Exception:
            skipped += 1                    # counted below -- never silent
            continue
        if parts:
            line = maybe_jitter(parts[0], cfg)
            lines.append(line)
            lines.extend(parts[1:])
            if registry is not None:
                # register the line AS SEEDED: --jitter perturbs the position,
                # which is part of the line's identity
                registry.add_obs([{**o, "obs80_line": line}])
    if skipped:
        # a systematic synthesis failure would otherwise just shrink the seed
        log(f"seed: {skipped}/{len(obs)} obs could not be rendered as OBS80 "
            f"and were left out of the seed")
    primaries = (designations if isinstance(designations, (list, tuple))
                 else [designations])
    return "\n".join(lines) + "\n", [str(d) for d in primaries]


def own_designations(pub_obs80, primaries=()):
    """The designations that ARE the seed object -- every id its own astrometry
    is filed under, plus the family primaries we were asked to search for.

    A candidate carrying one of these is the object itself (its rows sit in
    ``obs_sbn`` as pseudo-ITF) and must never be offered as a linkage.  Only
    the seed's OWN family is banned -- a pseudo-ITF tracklet under a DIFFERENT
    designation is exactly the identification this search exists to make.

    Both spellings go in: packed (``K26A09P``, as an OBS80 line carries it) and
    unpacked (``2026 AP9``, as the DB's ``provid`` column carries it).
    """
    own = set()

    def add(ident):
        if not ident:
            return
        own.add(ident)
        other = unpack(ident) or pack(ident)
        if other:
            own.add(other)

    for line in pub_obs80.splitlines():
        if not line.strip():
            continue
        ident = line_designation(line)
        if ident:
            add(ident)
    for d in primaries or ():
        add(str(d).strip())
    return own


# ---------------------------------------------------------------------------
# Candidate collection + scoring for one corridor sweep
# ---------------------------------------------------------------------------
def _wrap_delta(a, b):
    """a - b in degrees, wrapped to (-180, 180] (seam-safe scoring)."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _group_rows(rows):
    """Group candidate obs rows into tracklets: same trksub + station, and no
    intra-tracklet gap over 12 h (the legacy run-length grouping)."""
    groups = []
    run = []
    last = None
    for r in rows:
        if last is not None and (r["trksub"] != last["trksub"]
                                 or r["stn"] != last["stn"]
                                 or r["epoch_sec"] > last["epoch_sec"] + 43200):
            groups.append(run)
            run = []
        run.append(r)
        last = r
    if run:
        groups.append(run)
    return groups


def _astro_text(rows):
    out = []
    for r in rows:
        txt = r["obs80"]
        if len(txt) == 160:
            out.append(txt[:80])
            out.append(txt[80:160])
        else:
            out.append(txt)
    return "\n".join(out) + "\n"


def _relaxed(cfg, trial):
    """True when the trial orbit is determined well enough that the corridor
    prediction can be trusted over the tracklet-noise-limited qual metric."""
    if trial is None or not trial.ok:
        return False
    if trial.U <= cfg.relax_U:
        return True
    try:
        return trial.fit.n_oppositions() >= cfg.relax_opps
    except Exception:
        return False


def _window_for(windows, epoch_sec):
    """The corridor window covering ``epoch_sec``, narrowest first -- or None.

    Re-scores a carried candidate against the current orbit, whose corridor
    may be too thin to RETRIEVE it.
    """
    hit = [w for w in windows if w.unix_start <= epoch_sec <= w.unix_end]
    return min(hit, key=lambda w: w.uncert) if hit else None


def _score_tracklet(ctx, rows, window, likely_bound, own_ids,
                    chi2_by_trkid=None, feats_by_trkid=None):
    """Score ONE candidate tracklet against ONE corridor window.

    A pure function of the rows and the window, so a candidate can be re-scored
    against a better orbit without the database (the ``carried`` pool in
    :func:`_collect_tracklets`).  Returns the candidate dict (``_ids`` is its
    identity) or None when the rows are not a tracklet.
    """
    if len(rows) < 2:
        return None
    cfg = ctx.cfg
    ids = frozenset(obs_unique_id(r["obs80"]) for r in rows)
    start, end = rows[0], rows[-1]
    # predict_SKY, not predict: near the pole the RA/Dec quadratic carries
    # arcsec of rate error of its own, enough to fail the rate gate below
    # (2016 MB1: a matching tracklet looked far off).  Retrieval keeps the
    # RA/Dec form; the JUDGEMENT uses the pole-safe one.
    pred_s = window.predict_sky(start["epoch_sec"])
    pred_e = window.predict_sky(end["epoch_sec"])
    cosd = math.cos(math.radians(start["dec"]))

    if start["epoch_sec"] == end["epoch_sec"]:
        rate_resid_arcsec = 9999.0
    else:
        # motion mismatch: (predicted displacement) - (observed
        # displacement) across the tracklet, on-sky arcsec
        d_ra = (_wrap_delta(pred_s[0], pred_e[0])
                - _wrap_delta(start["ra"], end["ra"])) * cosd
        d_dec = (pred_s[1] - pred_e[1]) - (start["dec"] - end["dec"])
        rate_resid_arcsec = float(np.hypot(d_ra, d_dec)) * 3600.0
    dist = float(np.hypot(_wrap_delta(start["ra"], pred_s[0]) * cosd,
                          start["dec"] - pred_s[1])) * 3600.0
    # how far a geocentric-speed object could drift over the tracklet:
    # earth-circumference au/day, small-angle, arcsec
    traj_unc = (0.00026 / 86400.0
                * (end["epoch_sec"] - start["epoch_sec"])
                / window.dist_au * 57.3 * 3600.0)
    if rate_resid_arcsec < likely_bound + traj_unc:
        category = "likely"
    elif rate_resid_arcsec < cfg.maybe_arcsec + traj_unc:
        category = "maybe"
    else:
        category = "unlikely"

    # "Ours" three ways.  By OBS80 identity -- the observation is already in
    # the seed.  By DESIGNATION on the line, because the byte-exact identity
    # test misses the seed's own pseudo-ITF rows once the seed is RE-RENDERED
    # (the web UI synthesizes OBS80 from ADES).  And by the DB's provid,
    # because an archived line keeps its trksub after the MPC attributes it,
    # so a stale index offers the object its own observations (2026 AP9).
    own_desig = any(k in ctx.own_desigs
                    for k in (line_identifier(r["obs80"]) for r in rows)
                    if k)
    if not own_desig:
        own_desig = any(p in ctx.own_desigs
                        for p in (r.get("provid") for r in rows) if p)
    tr = {
        "rows": rows,
        "astrometry": _astro_text(rows),
        "duplicate": any(i in own_ids for i in ids) or own_desig,
        "qual": rate_resid_arcsec,
        "dist": dist,
        "traj_uncert": traj_unc,
        "traj_category": category,
        "score": dist + rate_resid_arcsec + window.uncert * 3600.0,
    }
    tr["_win"] = window
    # ONE feature row per candidate, from its BIGGEST index tracklet.  A
    # run-length group occasionally spans two or more index tracklet_ids, each
    # with its own fitted rate, scatter and photometry.  The pick must be
    # deterministic -- picking from a set (hash order) made the same candidate's
    # rank flip run to run.  Most observations wins (best-constrained rate);
    # ties fall to the earliest.  trk_n_obs/trk_arc_min still describe the
    # whole group; making the row self-consistent would need a retrain.
    _n = {}
    for r in rows:
        _n[r.get("trkid")] = _n.get(r.get("trkid"), 0) + 1
    tks = sorted(dict.fromkeys(r.get("trkid") for r in rows),
                 key=lambda k: -_n[k])
    _feats = feats_by_trkid or {}
    tr["_feat"] = next((_feats[k] for k in tks if k in _feats), None)
    if chi2_by_trkid:
        tr["match_chi2"] = min(
            (chi2_by_trkid[k] for k in tks if k in chi2_by_trkid),
            default=float("inf"))
    tr["_ids"] = ids
    return tr


def _sweep_cause_headline(skipped, cancelled, lowmem):
    """The DOMINANT cause of a truncation, plus a breakdown of the rest.

    The headline is whichever cause dropped the most windows, never "row
    budget" by default -- that would send the reader to raise a budget that
    dropped nothing.  Smaller nonzero causes are appended.
    """
    budget = max(0, skipped - cancelled - lowmem)
    causes = [("the sweep memory floor", "the memory floor", lowmem),
              ("a query the server cancelled", "a cancelled query", cancelled),
              ("row budget exhausted", "the row budget", budget)]
    causes.sort(key=lambda c: c[2], reverse=True)
    head = causes[0][0]
    rest = [f"{n} of them to {short}" for _full, short, n in causes[1:] if n]
    return head + ("" if not rest else "; " + "; ".join(rest))


def _windows_dropped(ctx):
    """How many corridor windows this search has failed to query so far -- a
    row budget, a query the server cancelled, or the low-memory sweep floor.

    The running total :func:`_fold_sweep_truncation` keeps, read before/after a
    recursion so the hypothesis ledger records only subtrees that searched the
    WHOLE ITF as settled (see :func:`_hypothesis_key`)."""
    tel = getattr(ctx, "telemetry", None) or {}
    return tel.get("windows_skipped", 0)


def _fold_sweep_truncation(ctx):
    """Read the just-finished sweep's truncation off the source chain, fold it
    into telemetry, and ZERO the counters consumed -- so it can never be counted
    twice.  Returns the ``(skipped, cancelled, lowmem)`` it consumed, all zero
    when the sweep was complete.

    A caller that wants to report the truncation must use the returned values:
    the source now reads zero.

    The counters live on the DbSource, which the variant scan reuses across
    every variant's collect; without the reset the SAME union sweep's
    truncation would be re-added once per variant.

    The trio is read from ONE source (the worst by ``skipped``): it is one
    sweep's tally split by cause, and maxing each over the chain could report
    "the server cancelled" about a sweep that never was.
    """
    worst = max(source_chain(ctx.source),
                key=lambda s: getattr(s, "last_sweep_skipped", 0) or 0,
                default=None)
    skipped = getattr(worst, "last_sweep_skipped", 0) or 0
    if not skipped:
        return 0, 0, 0
    cancelled = getattr(worst, "last_sweep_cancelled", 0) or 0
    lowmem = getattr(worst, "last_sweep_lowmem", 0) or 0
    for attr in ("last_sweep_skipped", "last_sweep_cancelled",
                 "last_sweep_lowmem"):
        try:
            setattr(worst, attr, 0)
        except Exception:
            pass
    ctx.log(f"incomplete sweep: {skipped} window(s) skipped "
            f"({_sweep_cause_headline(skipped, cancelled, lowmem)})")
    # A negative after an INCOMPLETE sweep is not a clean negative; record it
    # in telemetry, since the log may be silenced (the recovery harness does).
    ctx.telemetry["windows_skipped"] = (
        ctx.telemetry.get("windows_skipped", 0) + skipped)
    if cancelled:
        ctx.telemetry["windows_cancelled"] = (
            ctx.telemetry.get("windows_cancelled", 0) + cancelled)
    if lowmem:
        ctx.telemetry["windows_lowmem"] = (
            ctx.telemetry.get("windows_lowmem", 0) + lowmem)
    return skipped, cancelled, lowmem


def _collect_tracklets(ctx, eph_trial, extrawidth, relaxed=False, depth=0,
                       own_ids=None, windows=None, carried=None,
                       keep_unlikely=False):
    """Sweep the corridor for one trial orbit and return the scored candidate
    tracklet dicts (deduplicated across windows, unsorted).

    ``relaxed`` widens the likely gate to the maybe bound (ItfConfig.relax_U /
    relax_opps).  ``own_ids``: obs ids already ours (default the published
    seed; the recursion adds its taken tracklets).  ``windows``: pre-built
    corridor windows (one LOV variant's); None sweeps ``eph_trial``'s own
    corridor.  ``carried``: untaken candidates from shallower depths, re-scored
    here.  With ``ctx.collector`` set, one ML feature row per candidate is
    recorded against THIS trial orbit, never the final one.
    ``keep_unlikely`` keeps the tracklets the motion gate calls 'unlikely'
    (the final sweep judges them on where their observations sit instead).
    """
    cfg = ctx.cfg
    own_ids = ctx.pub_ids if own_ids is None else own_ids
    likely_bound = cfg.maybe_arcsec if relaxed else cfg.likely_arcsec
    if windows is None:
        nodes = sweep_nodes(eph_trial.fit, ctx.obscodes, cfg)
        cut = _note_faint_cut(ctx, nodes)
        windows = corridor_windows(nodes, cfg, ctx.max_uncert_deg, mag_limit=cut)
        if not windows and nodes:
            _note_mag_blackout(ctx, nodes)
        if _archive_from_the_start(ctx):
            # a well-determined seed sweeps the pre-ephem_start era every
            # round, so an archive tracklet can LINK, not only be mopped up by
            # the final sweep
            windows = windows + _archive_windows(ctx, eph_trial)
    else:
        nodes = windows                       # only its length is logged below
    if cfg.sweep_row_budget:
        # spend the row budget narrowest-first by sigma BAND (powers of two of
        # 100"), chronological within a band: a budget stop drops the widest
        # windows, and chunk queries stay epoch-tight
        windows.sort(key=lambda w: (int(math.log2(max(w.uncert * 36.0, 1.0))),
                                 w.unix_start))
    want_rank = (cfg.ranking_mode == "certainty"
                 and cfg.candidate_path == "index")
    if ctx.verbose:
        ctx.log(f"corridor: {len(nodes)} nodes -> {len(windows)} searchable windows")
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.field("sweep", f"{len(windows)} windows ({len(nodes)} ephemeris nodes), "
                  f"{span_of(w.unix_start for w in windows)}")
        trc.cont_changed("gate",
                         f"likely gate {'RELAXED to ' if relaxed else ''}"
                         f"{likely_bound:.0f}\" + drift, corridor "
                         f"x{ctx.uncert_mult:g}"
                         + (f", extrawidth {extrawidth:g}deg"
                            if extrawidth else ""))
    n_windows_hit = n_rows_total = 0
    # corridor half-width range and densest windows, summarised for the trace
    unc_lo = unc_hi = None
    densest = []

    # A legacy ML collector needs the HARD NEGATIVES the motion pre-filter
    # would drop, so its sweeps bypass it (``_motion_bound_ok`` still records
    # the verdict).  Passed per sweep: the source holds its own cfg.
    prefilter = False if _legacy_collect(ctx) else None
    by_key = {}
    _t_sweep = time.perf_counter()
    for window, rows, chi2_by_trkid, feats_by_trkid in ctx.source.sweep_candidates(
            windows, ctx.uncert_mult, extrawidth, want_rank, prefilter=prefilter):
        if rows:
            n_windows_hit += 1
            n_rows_total += len(rows)
            if trc.enabled:
                arcsec = window.uncert * 3600.0
                unc_lo = arcsec if unc_lo is None else min(unc_lo, arcsec)
                unc_hi = arcsec if unc_hi is None else max(unc_hi, arcsec)
                densest.append((len(rows), day_of(window.unix_start)))
            if trc.wants(L_FIREHOSE):
                trc(f"  window {day_of(window.unix_start)}..{day_of(window.unix_end)} "
                    f"±{window.uncert * 3600.0:.0f}\" @{window.dist_au:.2f}au: "
                    f"{len(rows)} obs rows in corridor", L_FIREHOSE)
        for t in _group_rows(rows):
            tr = _score_tracklet(ctx, t, window, likely_bound, own_ids,
                                 chi2_by_trkid, feats_by_trkid)
            if tr is None:
                continue
            ids = tr["_ids"]
            prev = by_key.get(ids)
            if prev is None or tr["score"] < prev["score"]:
                by_key[ids] = tr
    # RE-OFFER what a shallower depth retrieved and did not take.  Linking a
    # tracklet TIGHTENS the corridor, sometimes past the truth (a short arc
    # extrapolated far under-predicts its error), so the next sweep may not
    # retrieve a true tracklet already in hand.  Needed for 2016 TJ194; without
    # it, correct tracklets were dropped the moment the orbit got better.
    #
    # RE-SCORED, never re-used: only tracklet properties (rows, index features)
    # are carried; every orbit-relative term, including the ML P(match) below,
    # is recomputed -- a P(match) from the weak seed corridor is near zero for
    # a true tracklet.  Costs no sweep: the rows are already in hand.
    for old in (carried or ()):
        ids = old.get("_ids")
        if ids is None or ids in by_key:
            continue                       # this sweep found it again -- fresher
        rows = old.get("rows") or []
        if not rows:
            continue
        window = _window_for(windows, rows[0]["epoch_sec"])
        if window is None:
            continue                       # no window here under this orbit
        tr = _score_tracklet(ctx, rows, window, likely_bound, own_ids)
        if tr is None:
            continue
        tr["_feat"] = old.get("_feat")     # index features: orbit-independent
        tr["_carried"] = True
        by_key[ids] = tr

    kept = [tr for tr in by_key.values() if not tr["duplicate"]]
    n_skip, n_cancel, n_lowmem = _fold_sweep_truncation(ctx)
    if trc.enabled:
        n_dup = sum(1 for tr in by_key.values() if tr["duplicate"])
        cats = {}
        for tr in kept:
            cats[tr["traj_category"]] = cats.get(tr["traj_category"], 0) + 1
        if unc_lo is not None:
            trc.cont(f"corridor ±{unc_lo:.0f}\" .. ±{unc_hi:.0f}\" on the "
                     f"{n_windows_hit} window(s) that had rows")
        # the DB tally the source accumulated (level 4 narrates it per query)
        q, qr = trc.pop("db_queries"), trc.pop("db_rows")
        qf = trc.pop("db_feature_rows")
        elapsed = time.perf_counter() - _t_sweep
        if q:
            trc.cont(f"db: {q} queries, {qf} feature rows tested, "
                     f"{qr} obs rows fetched, {elapsed:.1f}s")
        if by_key:
            trc.cont(f"{n_rows_total} obs rows -> {len(by_key)} tracklets "
                     f"({n_dup} already ours; of the rest: "
                     f"{cats.get('likely', 0)} likely, "
                     f"{cats.get('maybe', 0)} maybe, "
                     f"{cats.get('unlikely', 0)} unlikely)"
                     # name the real cause (see _sweep_cause_headline)
                     + (f"; SWEEP TRUNCATED, {n_skip} widest windows skipped "
                        f"({_sweep_cause_headline(n_skip, n_cancel, n_lowmem)})"
                        if n_skip else ""))
        if densest and trc.wants(L_DETAIL):
            densest.sort(reverse=True)
            trc.cont("densest: " + ", ".join(f"{d} ({n} rows)"
                                             for n, d in densest[:3]), L_DETAIL)
    if not _legacy_collect(ctx) and not keep_unlikely:
        # 'unlikely' tracklets are never fitted or ranked; dropping them keeps
        # wide-corridor sweeps inside memory.  The legacy collector keeps them
        # as hard negatives; a collector WITH a decision log sees the served field.
        # Exception: in the post-fan retry, the tracklets the LOV refinement
        # measured stay -- its last-resort trials are drawn from exactly these,
        # and they are 'unlikely' by construction (far off every picket's
        # centreline, close to the LOV curve).  ``lov_refine`` is set only in
        # that retry, and holds a bounded record set.
        lov_measured = getattr(ctx, "lov_refine", None) or {}
        kept = [tr for tr in kept if tr["traj_category"] != "unlikely"
                or tr["_ids"] in lov_measured]
    if (cfg.ranking_mode == "ml" and cfg.candidate_path == "index" and kept):
        try:
            # ml_p2 is the second-opinion ranker's, whose top singles join the
            # trial queue against ONE model's blind spot (training.score_pair)
            from .training import score_pair, feature_uncert_mult
            in_var = windows is not None and ctx.is_variant
            probs, probs2 = score_pair(kept, eph_trial, depth,
                                       feature_uncert_mult(ctx), extrawidth,
                                       cfg,
                                       is_variant=ctx.is_variant,
                                       v4ctx=(in_var,
                                              getattr(ctx, "variant_lov_sigma", None),
                                              getattr(ctx, "lov_feat", None)))
            for tr, p in zip(kept, probs):
                tr["ml_p"] = float(p)
            if probs2 is not None:
                for tr, p2 in zip(kept, probs2):
                    tr["ml_p2"] = float(p2)
            if trc.enabled:
                trc.field("scored", f"{len(kept)} candidates | "
                          + histogram([tr.get("ml_p") for tr in kept]))
        except (ImportError, OSError, FileNotFoundError) as e:
            # model runtime/artifact missing -- the designed fallback
            ctx.log(f"ml ranking unavailable ({e}); falling back to certainty")
        except Exception as e:
            # a real scoring bug must not hide behind the fallback
            import traceback
            ctx.log(f"ml ranking FAILED ({type(e).__name__}: {e}); falling "
                    f"back to certainty -- this is a bug, not a missing model")
            if trc.enabled:
                trc.warn("; ".join(
                    traceback.format_exc().strip().splitlines()[-3:]))
    # Bank the ADES payload (an obs_sbn row IS an ADES record) of the tracklets
    # that can still be FITTED, so every fit gets back the sigma / rmsMag /
    # obsID the 80-column rendering drops.  Only `kept`: holding ADES for every
    # raw row of a wide corridor costs real memory.
    # An 'unlikely' tracklet is fitted only when this sweep was asked to keep
    # them (the final sweep's topocentric rescue) or the LOV refinement
    # measured it (its last-resort trials); otherwise it never reaches a fit.
    _r = _ctx_registry(ctx)
    if _r is not None:
        lov_measured = getattr(ctx, "lov_refine", None) or {}
        for tr in kept:
            if (tr["traj_category"] != "unlikely" or keep_unlikely
                    or tr["_ids"] in lov_measured):
                _r.add_records(tr["rows"])
    if ctx.collector is not None:
        from .training import (candidate_feature_row, feature_uncert_mult,
                               decision_context)
        dlog = getattr(ctx, "dlog", None)
        for tr in kept:
            row = candidate_feature_row(
                tr, eph_trial, depth, feature_uncert_mult(ctx), extrawidth,
                is_variant=ctx.is_variant, pre_floor=dlog is None)
            if dlog is not None:
                row.update(decision_context(ctx, tr, eph_trial,
                                            in_variant=windows is not None
                                            and ctx.is_variant))
            ctx.collector.append(row)
    return kept


def _legacy_collect(ctx):
    """True for the LEGACY collector, which sweeps a wider field than the
    search serves (motion pre-filter bypassed, 'unlikely' kept).  A collector
    riding with a decision log (itf/decisions.py) records exactly the served
    field instead."""
    return (ctx.collector is not None
            and getattr(ctx, "dlog", None) is None)


# ---------------------------------------------------------------------------
# The recursion
# ---------------------------------------------------------------------------
def _tracklet_ids(t):
    """A swept tracklet's observation ids, read from the astrometry it adds:
    a two-line record's companion line is one of them, exactly as when a
    trial takes the tracklet (its DB row holds both lines in one record)."""
    return frozenset(obs_ids(t["astrometry"]))


def _linked_tracklets_as_swept(ctx, obs_out):
    """The tracklets a linkage is made of, each with the obs it was SWEPT
    with.  Returns ``[(tag, frozenset(all_ids)), ...]``, deduplicated.

    ``obs_out`` holds only the lines that survived pruning, so it cannot say
    how many obs a tracklet had.  ``ctx.swept_tracklets`` (filled when the
    trial was taken) is keyed by every one of a tracklet's ids, so ANY
    surviving line recovers it.  A line with no record (the harness, a
    hand-built obs_out) conservatively stands for itself."""
    out, seen = [], set()
    swept = getattr(ctx, "swept_tracklets", None) or {}
    for line in obs_out.splitlines():
        if not line.strip():
            continue
        oid = obs_unique_id(line)
        tag, ids = swept.get(oid, (None, frozenset([oid])))
        if ids in seen:
            continue
        seen.add(ids)
        out.append((tag or "tracklet", ids))
    return out


def _fit(ctx, text):
    """Every in-search fit goes through here so a Yarkovsky rescue STAYS on for
    the rest of the search (ctx.nongrav): the orbit that holds the old
    tracklets is the one every later trial, vetting and final sweep is judged
    against."""
    # getattr: unit tests use SimpleNamespace doubles with only the fields
    # under test
    return fit_astrometry(text, ctx.obscodes, registry=_ctx_registry(ctx),
                          nongrav=getattr(ctx, "nongrav", False),
                          nongrav_ref=getattr(ctx, "nongrav_ref", None),
                          max_sec=_trial_max_sec(ctx),
                          perturbers=getattr(ctx, "perturbers", None))


def _trial_max_sec(ctx):
    """The per-trial-fit wall-clock bound (``cfg.trial_fit_max_sec``); None on
    the test doubles that carry no config."""
    cfg = getattr(ctx, "cfg", None)
    return getattr(cfg, "trial_fit_max_sec", None) if cfg is not None else None


class SeedPreservation:
    """What ``trial`` did to the object's OWN published astrometry.

    The Delta-chi2 half of :class:`SeedCheck`, which sets the cap; the lone
    night's banner reports the same numbers.

    ``pre``  -- the seed's chi2 under its OWN fit, included obs only.
    ``post`` -- the same obs' chi2 under the trial, sigma held FIXED at the
        seed fit's assessment (``sigma_map``), ``included_only=False`` so a
        seed ob the re-fit REJECTED counts at its full residual.
    ``delta``   -- ``post - pre``, what the caps are set against.
    ``missing`` -- seed obs the trial's astrometry does not contain AT ALL.

    A non-zero ``missing`` is a flat fail: a linkage that
    discards the object's own astrometry is not a linkage OF that object, and
    a Delta-chi2 cannot see it -- ``subset_chi2`` scores only what it finds, so
    a clipped seed makes the Delta NEGATIVE and the guard pass BECAUSE the seed
    is gone.  Needed for 2025 NV385; without it, an orbit of a different
    object with every seed line dropped read as "seed ok".  It is a backstop:
    :func:`_clip_poor_fitters` protects seed lines, so an offending seed ob
    normally arrives present-and-rejected and the Delta judges its residual.
    """

    __slots__ = ("ids", "pre", "post", "n_post", "missing")

    def __init__(self, ctx, trial, dropped=frozenset()):
        smap = sigma_map_for(ctx.pub.fit, ctx.pub_ids, station_floor=True)
        # a night dropped by _drop_seed_night is priced by the clip ledger
        self.ids = set(smap) - set(dropped)
        self.pre, _ = subset_chi2(ctx.pub.fit, self.ids, sigma_map=smap,
                                  included_only=True)
        self.post, self.n_post = subset_chi2(trial.fit, self.ids,
                                             sigma_map=smap,
                                             included_only=False)
        self.missing = len(self.ids) - self.n_post
        if self.missing:
            # COUNT it: the flat fail is decided on STRING IDENTITY
            # (obs_unique_id), so a non-zero more likely means an identity
            # mismatch (ADES->OBS80 collapse, a satellite two-line record
            # round-tripping differently) than a genuine clip.  Both gates
            # refuse silently; on the telemetry a systematic mismatch shows.
            tel = getattr(ctx, "telemetry", None)
            if tel is not None:
                tel["seed_obs_missing"] = max(tel.get("seed_obs_missing", 0),
                                              self.missing)

    @property
    def delta(self):
        return self.post - self.pre


def _protected_fit(ctx, trial, seed_ids, included_only):
    """Do the PROTECTED observations fit their own uncertainties under
    ``trial``?  The statistical escape for a seed-degradation test whose
    baseline may be tight by luck.  Asked against the seed AND the prior
    accepted state (seed + what shallower depths took, which CONTAINS the
    seed), the state a resumed search starts from.  Needed for 2011 QH108;
    without it, the same final orbit passed or failed depending on how many
    sessions the search took.

    Returns station-floored chi2 per dof as ``[(reference, chi2/dof, n)]`` in
    that order, skipping a reference with nothing to score.
    ``included_only`` matches the test being escaped: the acceptance
    Delta-chi2 counts rejected seed obs (False); the rms-ratio tests in
    :class:`SeedCheck` use included obs only (True)."""
    out = []
    for ref, ids in (("seed", seed_ids),
                     ("prior accepted", getattr(ctx, "prior_ids", None))):
        if not ids:
            continue
        c2, n = subset_chi2(trial.fit, ids, station_floor=True,
                            included_only=included_only)
        if n:
            out.append((ref, c2 / (2.0 * n), n))
    return out


class SeedCheck:
    """What a linkage did to the object's OWN astrometry -- THE seed
    measurement, made once for every gate that asks.  The seed obs ARE the
    object, so whether they still fit is the faithful test of "did the orbit
    get pulled off it?".

    ``fail`` (structured; its clause note ends ``FAIL``) -- the linkage is not
    that object, the first of:

    * seed obs MISSING from its astrometry (:class:`SeedPreservation`): a
      flat refusal, since no residual can be scored for them;
    * a Delta-chi2 past :func:`seed_preserve_cap`, sigma held fixed at the
      seed fit's own assessment and rejected obs at full residual;
    * a dramatic rms widening, the outright contaminant veto
      (:func:`extension_degrades`).  The two size tests see different
      failures: an orbit that bends far enough to REJECT the seed obs it no
      longer fits improves its rms, and pays in Delta-chi2.

    The Delta-chi2 and the widening each have the statistical escape
    (:func:`_protected_fit`) -- the protected obs still fit their own
    uncertainties -- counted over the obs the escaped test counts.

    ``caution`` -- the seed fits worse than ``warn_rms_factor`` x its own fit;
    ``degraded`` -- and past ``demote_seed_abs_arcsec`` (an over-fit short seed
    is so tight any ratio clears, 2020 KX6), read as a MEASUREMENT: an rms
    over n observations carries a fractional standard error of about
    1/(2 sqrt(n)), so it is past the floor only when its one-sigma lower bound
    is.  A degraded seed DEMOTES an accepted linkage to a 0.6 unless the
    statistical escape applies -- and the escape never lifts a degradation
    the contaminant veto would refuse: a seed fitted far below its assigned
    sigma reads "within its uncertainties" at any widening (2022 LC14: 11x
    at 0.61 sigma).
    ``bad_night`` -- the seed obs of ONE coherently offset seed night that
    carries the degradation while the rest of the seed fits
    (:func:`_seed_offending_night`), or None.  Not an excuse but a REPAIR:
    :func:`_drop_seed_night` drops it and the clip ledger prices it.  The
    Delta-chi2 needs the seed's own fit; without it, it is skipped.

    ``unpaid`` -- a whole seed span the orbit rejected that the link has not
    PAID for (:func:`_seed_clip_ledger`, :func:`_clip_evidence_ok`), or None.
    That is a question of EVIDENCE, not of identity: a real recovery is often
    how a bad night is proven bad, so dropping one is priced in newly linked
    tracklets, and a lone new night can never pay.  It refuses a linkage that
    would be SUBMITTED (:meth:`Assessment.acceptable`); on a 0.6 it is a
    caution."""

    def __init__(self, ctx, trial, obs_out, dropped=frozenset()):
        cfg = ctx.cfg
        self.dropped = frozenset(dropped)
        self.notes, self.fail = [], None
        self.caution, self.degraded, self.sigma = None, False, None
        self.unpaid, self.bad_night = None, None
        # both subset_rms currency (per observation); reported per-axis
        self.rms_pre = _seed_pre_rms(ctx)
        self.rms_post, self.n = subset_rms(trial.fit, ctx.pub_ids)
        self.ledger = _seed_clip_ledger(ctx, trial)
        self.n_linked = _linked_tracklet_count(obs_out)
        if ctx.pub.fit is not None and trial.fit is not None:
            self._delta_chi2(ctx, trial)
        self._clip(cfg)
        if self.fail is None and self.n:
            self._widening(ctx, trial)

    def _delta_chi2(self, ctx, trial):
        cfg = ctx.cfg
        seed = SeedPreservation(ctx, trial, self.dropped)
        cap = seed_preserve_cap(cfg.accept_seed_preserve_conf,
                                cfg.accept_seed_preserve_frac, seed.pre)
        within = seed.delta <= cap and seed.missing == 0
        clipped = (f", {seed.missing}/{len(seed.ids)} seed obs CLIPPED from the fit"
                   if seed.missing else "")
        head = (f"seed chi2 {seed.pre:.1f} -> {seed.post:.1f} (Delta "
                f"{seed.delta:.1f} vs {cap:.1f}, {_STATE_DOF} dof + "
                f"{cfg.accept_seed_preserve_frac:.0%} of seed){clipped}")
        # A lucky-tight baseline makes a correct link look catastrophic: ask
        # the statistical question directly -- but never over a CLIPPED seed,
        # whose missing obs the escape cannot score.
        escapes = []
        escaped = False
        if not within and not seed.missing:
            for tag, red, n in _protected_fit(ctx, trial, seed.ids,
                                              included_only=False):
                fits = red <= cfg.accept_seed_within_dof
                escapes.append(f"...{tag} obs under the new orbit: "
                               f"{red:.3f}/dof ({math.sqrt(red):.2f} sigma, "
                               f"{2 * n} dof) vs {cfg.accept_seed_within_dof:.2f}"
                               f"  {'ok' if fits else 'FAIL'}")
                if fits:
                    escaped = True
                    break
        # the clause's verdict is written AFTER its escape: an escaped Delta
        # is not a failure, and must never read as one
        self.notes.append(f"{head}  " + ("ok" if within else
                                         "ok, escaped (below)" if escaped else "FAIL"))
        self.notes.extend(escapes)
        if not (within or escaped):
            self.fail = ({"kind": "seed_missing", "n": seed.missing,
                          "of": len(seed.ids), "note": head} if seed.missing
                         else {"kind": "seed", "note": head})

    def _clip(self, cfg):
        led = self.ledger
        if not led["lost_nights"]:
            return
        ok, why = _clip_evidence_ok(cfg, led, self.n_linked)
        self.notes.append(f"seed spans: {why} (refuses a submission)" if not ok
                          else f"seed spans: {len(led['lost_nights'])} of "
                          f"{led['n_nights']} dropped, paid for by "
                          f"{self.n_linked} linked tracklet(s)  ok")
        if not ok:
            self.unpaid = why

    def _widening(self, ctx, trial):
        cfg = ctx.cfg
        protected = _protected_fit(ctx, trial, ctx.pub_ids, included_only=True)
        red = min((r for _ref, r, _n in protected), default=None)
        self.sigma = math.sqrt(red) if red is not None else None
        pre, post = self.rms_pre, self.rms_post
        veto = extension_degrades(cfg, pre, post, ctx.pub.arclength_days,
                                  seed_post_sigma=self.sigma)
        shown = (f"{per_axis(pre):.3f}\" -> {per_axis(post):.3f}\"")
        ratio = f"{post / pre:.1f}x" if pre > 0 else "n/a"
        self.notes.append(f"seed rms {shown} ({ratio}; a caution at "
                          f"{cfg.warn_rms_factor:g}x, refused at "
                          f"{cfg.veto_rms_factor:g}x unless within its "
                          f"uncertainties"
                          + (f", {self.sigma:.2f} sigma" if self.sigma is not None
                             else "") + f")  {'FAIL' if veto else 'ok'}")
        if veto:
            self.fail = {"kind": "veto", "shown": shown,
                         "note": f"seed rms {shown} ({ratio}), past the "
                                 f"contaminant veto"}
            return
        warn_lvl = max(pre * cfg.warn_rms_factor, pre + cfg.veto_rms_floor_arcsec)
        if post < warn_lvl:
            return
        past_floor = (post * (1.0 - 1.0 / (2.0 * math.sqrt(self.n)))
                      >= cfg.demote_seed_abs_arcsec)
        veto_sized = past_floor and extension_degrades(cfg, pre, post,
                                                       ctx.pub.arclength_days)
        night = _seed_offending_night(ctx, trial, warn_lvl)
        self.bad_night = night[1] if night else None
        escape = None
        if (red is not None and cfg.accept_seed_within_dof > 0
                and red <= cfg.accept_seed_within_dof and not veto_sized):
            escape = (f"the object's own observations still fit their "
                      f"uncertainties ({self.sigma:.2f} sigma)")
        self.caution = (f"the object's own observations fit worse after the "
                        f"link ({shown})"
                        + (f", but {escape}" if escape else "")
                        + (f"; {night[0]}" if night else ""))
        self.degraded = escape is None and past_floor

    def vetting(self):
        return {
            # REPORTED per-axis like every displayed rms; gates read the raw values
            "seed_rms_pre": round(per_axis(self.rms_pre), 4),
            "seed_rms_post": round(per_axis(self.rms_post), 4),
            "seed_obs_compared": self.n,
            "seed_shed": self.ledger,
            "linked_tracklets": self.n_linked,
        }


def _accept_quality_ok(ctx, trial, obs_out, prior_linkage_used, detail=None,
                       vouched=None, dropped=frozenset()):
    """May this linkage be ACCEPTED, judged on the quality of what it links?
    Returns ``(ok, notes)`` -- ``notes`` are the per-clause trace lines.

    A progress precondition (link at least as many obs as the best prior
    linkage), then four clauses.  Deliberately NOT an aggregate arcsec rms
    cap: a distant tracklet gives a short arc new leverage and can LOWER the
    overall rms while itself fitting badly (2015 BO519).

      1. EACH linked tracklet keeps at least ``accept_tracklet_min_retention``
         of the obs it was SWEPT with: a 4-obs tracklet clipping one is fine,
         a 3-obs tracklet down to two is not.  A tracklet that cannot hold
         its own observations together is no evidence of anything.
      2. EACH linked tracklet's KEPT obs fit their own uncertainties -- chi2
         at ``accept_tracklet_conf`` on 2n dof, sigma floored per station
         (never the survey's reported sigma alone: see extension_degrades).
      3. The SEED is still that object (:class:`SeedCheck`): none of its
         obs missing, not statistically degraded, no unpaid span dropped, no
         contaminant-sized widening.
      4. Each linked night's residuals are not coherently displaced (opt-in,
         ``max_night_systematic_sigma``).

    ``detail["seed"]`` carries the :class:`SeedCheck`.  ``vouched`` -- the
    tracklets (id set -> note) this linkage's own repair vouched for
    (:func:`_with_vouched`): exempt from clauses 1 and 2.  ``dropped`` -- the
    seed night its repair dropped (:func:`_drop_seed_night`)."""
    from scipy.stats import chi2 as _chi2
    cfg = ctx.cfg
    notes = []
    ok = True
    # ``detail`` (optional) records WHICH tracklets failed and whether any
    # other clause did (what _repaired needs), and every failure as a
    # structured record (what _plain_refusal phrases for a reader).
    if detail is None:
        detail = {}
    detail.setdefault("bad_tracklets", [])
    detail["other_fail"] = False
    fails = detail.setdefault("fails", [])
    if not (trial.used >= prior_linkage_used and trial.rms > 0):
        detail["other_fail"] = True
        note = (f"progress: used {trial.used} < best prior linkage "
                f"{prior_linkage_used}, or a dead rms")
        fails.append({"kind": "progress", "note": note})
        return False, [note + "  FAIL"]

    groups = _linked_tracklets_as_swept(ctx, obs_out)
    detail["n_groups"] = len(groups)
    all_ids = set().union(*(ids for _t, ids in groups)) if groups else set()
    included, _ = subset_included(trial.fit, all_ids)
    vouched = vouched or {}
    for tag, ids in groups:
        if ids in vouched:
            notes.append(f"tracklet {tag}: {vouched[ids]}  ok")
            continue
        kept = {i for i in ids if i in included}
        n_obs = n_observations(ids)
        frac = len(kept) / n_obs if n_obs else 0.0
        good = frac >= cfg.accept_tracklet_min_retention
        note = (f"tracklet {tag}: {len(kept)}/{n_obs} obs on "
                f"({frac:.0%} vs {cfg.accept_tracklet_min_retention:.0%})")
        notes.append(f"{note}  {'ok' if good else 'FAIL'}")
        ok = ok and good
        if not good:
            detail["bad_tracklets"].append((tag, ids))
            fails.append({"kind": "retention", "tag": tag, "kept": len(kept),
                          "n": n_obs, "note": note})
        if not kept:
            continue
        c2, n = subset_chi2(trial.fit, kept, station_floor=True,
                            included_only=True)
        if not n:
            continue
        dof = 2 * n
        cap = float(_chi2.ppf(cfg.accept_tracklet_conf, dof))
        fits = c2 <= cap
        note = (f"tracklet {tag}: chi2 {c2:.1f} vs {cap:.1f} "
                f"({dof} dof, {cfg.accept_tracklet_conf:.1%})")
        notes.append(f"{note}  {'ok' if fits else 'FAIL'}")
        ok = ok and fits
        if not fits:
            detail["bad_tracklets"].append((tag, ids))
            fails.append({"kind": "chi2", "tag": tag, "red": c2 / dof, "note": note})

    seed = detail["seed"] = SeedCheck(ctx, trial, obs_out, dropped)
    notes.extend(seed.notes)
    if seed.fail is not None:
        ok = False
        detail["other_fail"] = True
        fails.append(seed.fail)

    # ---- clause 4: the orbit must FIT THE NIGHTS IT LINKS -------------------
    # Off unless configured (ItfConfig.max_night_systematic_sigma).  Asks how
    # residuals are ARRANGED, not how big: a wrong tracklet displaces each
    # night coherently while every size-based clause passes.
    cap = getattr(cfg, "max_night_systematic_sigma", 0.0)
    if cap > 0 and trial.fit is not None:
        from .night_coherence import night_structure
        st = night_structure(trial.fit)
        worst = st.get("night_coherence")
        if worst is not None:
            bad = worst >= cap
            note = (f"night coherence: worst night at "
                    f"{worst:.2f}x its expected displacement vs {cap:.2f} "
                    f"({st.get('worst_night_arcsec', 0):.2f}\", z "
                    f"{st.get('worst_night_z', 0):.2f}, "
                    f"{st['n_nights']} nights, {st['evidence_dof']} dof)")
            notes.append(f"...{note}  {'FAIL' if bad else 'ok'}")
            if bad:
                ok = False
                # NOT per-tracklet: other_fail stops _repaired from dropping
                # tracklets over it.
                detail["other_fail"] = True
                fails.append({"kind": "night", "note": note})
    return ok, notes


def _archive_tracklets(ctx, trial, deep_only=False, keep_unlikely=False):
    """Candidate tracklets from BEFORE ``cfg.ephem_start`` -- the era the
    ordinary sweep cannot reach at all.

    The corridor's ephemeris begins at ``ephem_start``, so no ordinary window
    exists before it -- yet an old precovery is the longest lever arm an orbit
    can get.  Needed for 2022 SN92; without it, its own 1995 tracklet was
    unreachable.

    Swept at the post-acceptance final sweep rather than by moving
    ``ephem_start`` back: the ephemeris is the whole price (the old era is
    sparse), paid once on an ACCEPTED linkage instead of on every sweep of
    every search; and a precovery is only findable once the orbit is good (a
    weak seed's decades-old corridor is degrees wide and dies on the
    uncertainty gate).  Being paid once, it reaches ``archive_deep_start``,
    below the ordinary sweeps' ``archive_start`` floor (2020 OG4 needs a 1974
    tracklet).  The node step stays ``ephem_step_days``: a window is a
    quadratic through 3 nodes, and a coarser grid breaks the corridor's
    tolerances.  Additions face the ordinary final sweep gates.

    A seed with U <= ``archive_seed_U`` sweeps these windows in every ordinary
    sweep (:func:`_archive_from_the_start`); this function covers a weaker
    seed and a run whose modern final sweep was skipped as redundant.
    """
    cfg = ctx.cfg
    start = cfg.archive_deep_start or cfg.archive_start
    # deep_only: the ordinary sweeps already covered [archive_start,
    # ephem_start), so add only the slice below that floor.
    stop = cfg.archive_start if deep_only else cfg.ephem_start
    windows = _archive_windows(ctx, trial, start=start, stop=stop)
    if not windows:
        return []
    return _collect_tracklets(ctx, trial, 0.0,
                              relaxed=_relaxed(ctx.cfg, trial),
                              depth=-1, windows=windows,
                              keep_unlikely=keep_unlikely)


def _archive_from_the_start(ctx):
    """Does the ARCHIVE era ride in every ordinary sweep for this search?

    Yes when the SEED is well-determined (U <= ``archive_seed_U``): its
    backward corridor is searchable at once, so an archive tracklet can be
    LINKED and recursed on, not merely mopped up by the final sweep (see
    :func:`_archive_tracklets` for why weaker seeds wait).
    """
    cfg = ctx.cfg
    return (bool(cfg.archive_start) and cfg.archive_seed_U > 0
            and ctx.pub is not None and ctx.pub.U is not None
            and ctx.pub.U <= cfg.archive_seed_U)


def _note_faint_cut(ctx, nodes):
    """The magnitude cut for this sweep (:func:`ephem.faint_cut`), recorded on
    the telemetry -- the archive pass reuses it -- and traced once when the
    object's own faintness deepened it past ``cfg.mag_limit``."""
    cut = faint_cut(ctx.cfg, nodes)
    ctx.telemetry["mag_cut"] = float(cut)
    if cut > ctx.cfg.mag_limit and not ctx.telemetry.get("mag_cut_deepened"):
        ctx.telemetry["mag_cut_deepened"] = True
        trc = _ctx_tracer(ctx)
        if trc.enabled:
            trc.field("mag cut", f"{cut:.2f} (deepened from {ctx.cfg.mag_limit:g}: "
                      f"{ctx.cfg.mag_floor_margin:g} mag fainter than the brightest "
                      f"predicted since {ctx.cfg.mag_floor_since[:4]})", L_STORY)
    return cut


def _note_mag_blackout(ctx, nodes):
    """Record that the corridor came back EMPTY because of the magnitude gate.

    A faint object is faint at every window, so ``mag_limit`` deletes its
    corridor and the search reports "no likely matches" as though it had
    looked (common for TNOs, e.g. 2016 QC135).  ``ephem.faint_cut`` prevents
    this for a DEFAULT cut; under a typed cut (or the floor off) it can still
    happen, and the search must SAY so rather than let silence read as
    absence.  Surfaced on ``ItfSearchResult.warnings``.
    """
    if ctx.telemetry.get("mag_blackout"):
        return
    mags = [n["mag"] for n in nodes if n.get("mag") is not None]
    if not mags:
        return
    # self-validating: only an ACTUALLY empty corridor is blamed
    if corridor_windows(nodes, ctx.cfg, ctx.max_uncert_deg):
        return
    deep = corridor_windows(nodes, replace(ctx.cfg, mag_limit=float("inf")),
                            ctx.max_uncert_deg)
    if not deep:
        return                       # empty for some other reason entirely
    cut = faint_cut(ctx.cfg, nodes)
    ctx.telemetry["mag_blackout"] = {"limit": float(cut),
                                     "min_mag": float(min(mags)),
                                     "windows_if_deeper": len(deep)}
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.warn(f"NO corridor windows were searched: the object is predicted "
                 f"fainter than mag_limit {cut:g} at every epoch "
                 f"(brightest {min(mags):.2f}); {len(deep)} windows exist with "
                 f"the gate lifted")


def _archive_windows(ctx, trial, start=None, stop=None):
    """Corridor windows over the archive era (``start`` .. ``ephem_start``) for
    one trial orbit; empty when disabled or unsearchable.

    ``start`` defaults to ``cfg.archive_start``, the ORDINARY sweeps' floor;
    the final sweep passes ``cfg.archive_deep_start`` (see
    :func:`_archive_tracklets`)."""
    cfg = ctx.cfg
    start = start or cfg.archive_start
    stop = stop or cfg.ephem_start
    if not start:
        return []
    try:
        start_jd = _jd_of_iso(start)
        stop_jd = _jd_of_iso(stop)
    except ValueError:
        return []
    if stop_jd <= start_jd:
        return []
    nodes = sweep_nodes(trial.fit, ctx.obscodes, cfg, jd_start=start_jd,
                        jd_stop=stop_jd)
    # the archive era lies (mostly) before mag_floor_since, so reuse the cut
    # the ordinary corridor judged on the modern track
    cut = ctx.telemetry.get("mag_cut")
    windows = corridor_windows(nodes, cfg, ctx.max_uncert_deg,
                               mag_limit=None if cut is None else max(cut, faint_cut(cfg, nodes)))
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.field_changed("archive", f"{len(windows)} searchable window(s) "
                          f"{start} .. {stop} -- the era the ordinary corridor "
                          f"cannot reach", L_STORY)
    return windows


def _jd_of_iso(text):
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0 + 2440587.5


def _prune_trial(ctx, tr, base_text, cand, names):
    """Blunder clip within one trial: drop CANDIDATE obs (never a seed or
    hypothesis line) whose residual against the trial's own orbit exceeds the
    legacy threshold, and refit on the survivors, so one junk tracklet does
    not burn a joint trial and pruned lines never ride into the hypothesis.
    Returns the possibly-updated ``(trial, cand)``; any pruning trouble (a
    test double, a refit failure) falls back to the trial as fitted."""
    try:
        # NB deliberately NOT shifted with the U scale: it decides which obs
        # survive (the linkage's composition), and nothing re-derives the 3.5.
        thresh = 0.5 if tr.U < 3.5 else 0.74
        cand2 = drop_offending(tr.fit, cand, thresh,
                               sigma_mult=ctx.cfg.clip_sigma_mult,
                               min_tracklet_obs=ctx.cfg.min_tracklet_obs)
    except Exception:
        return tr, cand
    if cand2 == cand or not cand2.strip():
        return tr, cand
    tr2 = _fit(ctx, base_text + cand2)
    if not tr2.ok:
        return tr, cand
    n_clipped = len(cand.splitlines()) - len(cand2.splitlines())
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc(f"trial [{names}]: pruned {n_clipped} misfitting candidate obs "
            f"(> {thresh}\") and refit -- used {tr2.used}/{tr2.count}, "
            f"rms {tr2.rms:.3f}\"")
    return tr2, cand2


class _VariantParent:
    """A screen parent standing in for ONE LOV variant orbit.

    Variant trials cannot be screened against the NOMINAL orbit (a bounded
    step from it destroys the true tracklet too -- why the LOV is sampled at
    all), but each variant is a legitimate orbit at the same epoch, so a warm
    step from ITS state is representable.  Wraps the nominal fit's obs and
    epoch with the variant's state.

    ``U`` is deliberately None: the opt-in ``screen_max_U`` gates on the
    parent's UNCERTAINTY, and a variant is a specific orbit.  Screening
    variants at all is ``cfg.screen_variant_trials``."""

    __slots__ = ("fit", "U")

    def __init__(self, nominal_fit, state):
        import copy as _copy
        shim = _copy.copy(nominal_fit)      # same obs/epoch, different state
        shim._state = state
        self.fit = shim
        self.U = None


def _screen_trial(ctx, parent, cand, names):
    """Warm-start rejection screen for one trial (:func:`fitter.screen_candidate`).

    ``parent`` is the orbit the trial queue's ``base_text`` was fitted on (the
    seed at the top level, the taken trial deeper down), so the warm step
    starts converged on every base obs.  Returns True to SKIP the trial (the
    step destroyed the parent arc); False -- also every untrustworthy case --
    means run the cold fit."""
    if parent is None:
        return False
    verdict, seed_sigma, secs = screen_candidate(parent.fit, cand, ctx.obscodes,
                                                 ctx.cfg, seed_U=parent.U,
                                                 registry=_ctx_registry(ctx))
    tel = ctx.telemetry
    tel["screen_seconds"] = tel.get("screen_seconds", 0.0) + secs
    tel["screen_" + verdict] = tel.get("screen_" + verdict, 0) + 1
    trc = _ctx_tracer(ctx)
    if verdict == SCREEN_REJECT:
        if trc.enabled:
            trc(f"trial [{names}]: WARM SCREEN rejects (parent arc goes to "
                f"{seed_sigma:.0f} sigma > {ctx.cfg.screen_seed_sigma_max:g} in "
                f"{secs * 1000:.0f} ms) -- no trial fit, next trial")
        return True
    if trc.enabled and verdict == SCREEN_PLAUSIBLE:
        trc(f"trial [{names}]: warm screen passes (parent arc stays "
            f"{seed_sigma:.2f} sigma, {secs * 1000:.0f} ms) -- fitting")
    return False


def _seed_pre_rms(ctx):
    """The seed's PRE-link rms on its own observations, in the same currency as
    :func:`subset_rms` -- the only valid left-hand side for every "did the link
    degrade the object's own obs?" comparison.

    Not ``ctx.pub.rms``: that is PER-AXIS (mean over 2n components) while
    ``subset_rms`` is PER-OBSERVATION (mean of d_ra^2 + d_dec^2 over n), a
    sqrt(2) difference that would charge every link a handicap and fire the
    rms-ratio warning on nearly every genuine recovery.  Using ``subset_rms``
    on both sides also matches the included sets.

    Falls back to ``pub.rms * sqrt(2)`` (the exact conversion) when the seed
    fit object is unavailable."""
    pub_fit = ctx.pub.fit
    if pub_fit is not None:
        rms, n = subset_rms(pub_fit, ctx.pub_ids)
        if n:
            return rms
    return ctx.pub.rms * math.sqrt(2.0)


def _seed_guard_levels(ctx):
    """The per-recursion seed-preservation levels ``(seed_lvl, seed_keep)``: the
    rms a trial may not push the seed obs past, and how many seed obs it must keep
    included.  Shared by the trial queue and the exhaustion test, so a rival
    is judged by EXACTLY the bar that ruled out the candidates above it.

    Absolute floor: a seed is not "poisoned" while its residuals match its
    assigned uncertainties -- the chi^2=1 noise rms times sigma_mult, never
    below ``recurse_seed_abs_arcsec``.  It only raises the bar (admitting a
    coarse-station arc), never rejects more.  All terms are in
    :func:`subset_rms` currency (see :func:`_seed_pre_rms`)."""
    cfg = ctx.cfg
    pub_fit = ctx.pub.fit
    seed_pre = _seed_pre_rms(ctx)
    seed_noise_floor = (subset_sigma_floor(pub_fit, ctx.pub_ids)[0]
                        if pub_fit is not None else 0.0)
    seed_abs = max(cfg.recurse_seed_abs_arcsec,
                   cfg.recurse_seed_sigma_mult * seed_noise_floor)
    seed_lvl = max(seed_pre * cfg.recurse_seed_factor,
                   seed_pre + cfg.recurse_seed_floor_arcsec,
                   seed_abs)
    seed_keep = math.ceil(cfg.recurse_seed_retention * ctx.pub.used)
    return seed_lvl, seed_keep


_OUTCOME_GLOSS = {
    "warm screen": "killed by warm screen (the trial destroys the parent arc)",
    "fit failed": "fit failed",
    "no-progress": "fitted, NO PROGRESS (every candidate obs rejected)",
    "rms-sanity": "fitted, RMS SANITY (converged far off-sky)",
    "seed-guard": "fitted, SEED GUARD (degrades the object's own obs)",
    "seed-retention": "fitted, SEED RETENTION (clips the object's own obs)",
    "hyp-drop": "fitted, HYPOTHESIS DROP (expels a tracklet a shallower level linked)",
}


def _trial_queue_outcome(trc, outcomes, n_tried, n_trials):
    """The trial queue's trial tally as prose -- what happened to the trials, in the
    order a reader cares about (cheap kills first, real fits last)."""
    if not outcomes:
        return
    trc.cont(f"{n_tried}/{n_trials} trial{'' if n_tried == 1 else 's'} tried:")
    for tag, n in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        trc.cont(f"  {n} x {_OUTCOME_GLOSS.get(tag, tag)}")


def _hypothesis_groups(ctx, hypothesis_obs80):
    """The tracklets shallower recursion levels TOOK, as ``[(tag, ids,
    lines), ...]`` -- only those with a ``ctx.swept_tracklets`` record
    (``lines``: those still in ``hypothesis_obs80``).  A line with no record
    is left unjudged: as a one-line tracklet, one rejected ob would read as a
    tracklet dropped and refuse a CORRECT take."""
    swept = getattr(ctx, "swept_tracklets", None) or {}
    by_ids, order = {}, []
    for line in hypothesis_obs80.splitlines():
        if not line.strip():
            continue
        rec = swept.get(obs_unique_id(line))
        if rec is None:
            continue
        if rec[1] not in by_ids:
            by_ids[rec[1]] = (rec[0], rec[1], [])
            order.append(rec[1])
        by_ids[rec[1]][2].append(line)
    return [by_ids[k] for k in order]


def _expelled_tracklets(tr, hypothesis_groups):
    """The hypothesis tracklets ``tr`` REJECTS -- fewer than half of their
    swept obs still included -- as ``[(tag, ids, lines, n_included), ...]``.

    Half, not all: the engine's own outlier rejection may legitimately take one
    ob of a correct tracklet, and a 2-ob tracklet keeping one is still in."""
    out = []
    for tag, ids, lines in hypothesis_groups:
        inc, _res = subset_included(tr.fit, ids)
        if 2 * len(inc) < n_observations(ids):
            out.append((tag, ids, lines, len(inc)))
    return out


def _supersedes(tr, expelled, added_lines):
    """``(nights_in, nights_out)``: the observing NIGHTS the trial's own
    INCLUDED candidate obs bring, against the nights of the linked tracklets it
    expels; strictly more supersedes them.  NIGHTS, not tracklets: a survey
    splits one night into several tracklet ids, and one foreign night must not
    out-vote a real one."""
    add_ids = obs_ids(added_lines)
    inc, _res = subset_included(tr.fit, add_ids) if add_ids else (set(), {})
    kept = [l for l in added_lines.splitlines()
            if l.strip() and obs_unique_id(l) in inc]
    gone = [l for _t, _i, lines, _n in expelled for l in lines]
    return len(nights_of(kept)), len(nights_of(gone))


def _seed_holds(ctx, tr, parent_seed_rms):
    """True when ``tr`` fits the object's own (published) obs no worse than
    the linkage it would replace did -- ``parent_seed_rms``, same
    :func:`subset_rms` currency.  False when that is unknown."""
    if parent_seed_rms is None or tr.fit is None:
        return False
    post, n = subset_rms(tr.fit, ctx.pub_ids)
    # the rms could be bought by rejecting seed obs: also hold the swap to
    # clause (c)'s retention share, which a lone candidate would skip
    keep = math.ceil(ctx.cfg.recurse_seed_retention * ctx.pub.used)
    return n >= keep and post <= parent_seed_rms


def _trial_rejection(ctx, tr, prior_used, single_candidate, seed_lvl, seed_keep,
                    hypothesis_groups=(), added_lines="", parent_seed_rms=None):
    """Judge an ALREADY-PRUNED trial fit by the trial queue's take criterion: it must
    (a) progress past ``prior_used`` while keeping the seed obs, (b) not sit
    absurdly off-sky, and (c) not poison or clip the seed arc -- (c) waived for a
    lone candidate (nothing to fall back on) -- and (d) not REJECT a tracklet a
    shallower level took (``hypothesis_groups``, see :func:`_hypothesis_groups`)
    UNLESS the trial's own candidates (``added_lines``) bring strictly more
    observing nights than the tracklets it expels (:func:`_supersedes`) -- or
    EQUAL nights while the object's own seed obs fit no worse than under the
    linkage being replaced (``parent_seed_rms``; the tie-break below).

    (d) makes "progress" mean EXTENSION: (a) counts obs, so a trial that
    expels linked obs for more foreign ones reads as progress, and (c) guards
    only the published seed.  Needed for 2005 UD562; without it, a foreign
    tracklet displaced a correct one and the search recursed on the wrong
    orbit.  Refusing such a COMPETING hypothesis hands the choice back to the
    trial queue above (barren-take backtracking).  Not waived for a lone
    candidate: the parent linkage still stands to fall back on.

    A trial carrying MORE nights than it expels is the better-supported
    hypothesis: it is judged by the other clauses, and the trial queue removes
    the expelled tracklets from the hypothesis it hands down
    (``TrialQueue.next_take``), so deeper levels and the document see the
    corrected linkage.  A TIE is broken by the object's own obs, the one
    ground truth a search has; it cannot simply be refused, because a
    one-for-one swap is how a search corrects an early wrong take (2014 HE349
    lost its whole recovery without it).  U does not separate a correction
    from a contamination; the seed's fit does.

    Returns ``None`` when the trial is TAKEABLE, else ``(trc_msg, log_msg,
    tag)`` naming the clause that rejected it (``log_msg`` None when the clause
    has no plain-log line; ``tag`` a short stable key the trace tallies trial
    outcomes by, so a round can be summarised without parsing prose)."""
    cfg = ctx.cfg
    if not (tr.used >= ctx.pub.used and tr.used > prior_used):
        return (f"used {tr.used}/{tr.count} rms {tr.rms:.3f}\" -- NO PROGRESS "
                f"(need >= {ctx.pub.used} seed obs kept and > {prior_used} "
                f"total)", None, "no-progress")
    # Overall-rms sanity, for single AND joint trials (the seed guard skips a
    # lone candidate and sees only the seed).  A trial thousands of arcsec off
    # is corridor noise; recursing on it only snowballs.
    trial_rms_cap = max(cfg.recurse_rms_max_arcsec,
                       ctx.pub.rms * cfg.recurse_rms_max_factor)
    if tr.rms > trial_rms_cap:
        return (f"RMS SANITY tripped (rms {tr.rms:.3f}\" > {trial_rms_cap:.3f}\" "
                f"cap)", None, "rms-sanity")
    expelled = _expelled_tracklets(tr, hypothesis_groups) if hypothesis_groups else []
    if expelled:
        n_in, n_out = _supersedes(tr, expelled, added_lines)
        if n_in == n_out and _seed_holds(ctx, tr, parent_seed_rms):
            pass            # tie broken by the object's own obs: a correction
        elif n_in <= n_out:
            tags = ", ".join(t for t, _i, _l, _n in expelled)
            kept = ", ".join(f"{n}/{len(i)}" for _t, i, _l, n in expelled)
            return (f"HYPOTHESIS DROP (rejects {tags}, {kept} obs kept, which a "
                    f"shallower level linked; brings {n_in} night(s) against "
                    f"the {n_out} it expels -- a competing hypothesis, not an "
                    f"extension)",
                    f"expels linked tracklet(s) {tags} for {n_in} night(s) "
                    f"against {n_out}",
                    "hyp-drop")
    if single_candidate:
        return None
    seed_post, n_seed = subset_rms(tr.fit, ctx.pub_ids)
    if n_seed and seed_post >= seed_lvl:
        seed_pre = _seed_pre_rms(ctx)
        # Compared per-observation; displayed per-axis like every other rms
        # in the trace (fitter.per_axis).  All three scale together.
        p_pre, p_post, p_lvl = (per_axis(seed_pre), per_axis(seed_post),
                                per_axis(seed_lvl))
        return (f"SEED GUARD tripped (seed obs rms {p_pre:.3f}\" -> "
                f"{p_post:.3f}\", limit {p_lvl:.3f}\")",
                f"existing obs degrade {p_pre:.3f}\"->{p_post:.3f}\"",
                "seed-guard")
    if n_seed and n_seed < seed_keep:
        conceded = _corroborated_seed_clips(ctx, tr, added_lines)
        if n_seed + len(conceded) >= seed_keep:
            trc = _ctx_tracer(ctx)
            if trc.enabled:
                trc.cont(f"seed retention {n_seed}/{ctx.pub.used} below {seed_keep}, but "
                         f"{len(conceded)} of the clipped seed obs (one night) were clipped "
                         f"by an independent trial too -- not counted against it")
            return None
        return (f"SEED RETENTION tripped ({n_seed}/{ctx.pub.used} seed obs "
                f"included, need >= {seed_keep})",
                f"only {n_seed}/{ctx.pub.used} seed obs stay included",
                "seed-retention")
    return None


def _corroborated_seed_clips(ctx, tr, added_lines):
    """The seed obs ``tr`` clips that an INDEPENDENT trial clipped as well: the
    set seed retention does not count against ``tr``.  Empty unless
    ``cfg.seed_clip_corroboration``.

    Seed retention refuses a trial that clips more of the object's own obs than
    ``recurse_seed_retention`` allows, because an orbit bending away from the
    object sheds its astrometry.  On a short seed that rule cannot tell such an
    orbit from a correct one exposing bad astrometry inside the seed, which a
    2-day arc absorbs by bending: every correct extension then clips the SAME
    seed obs and every one is refused.  Two trials whose candidate obs share no
    UT date are independent evidence, and when they clip the same seed obs the
    obs are what is wrong.  Two bounds keep it the clip ledger's exception
    rather than a loosening: the corroborated obs must lie on ONE seed night
    (the unit a linkage may drop, :func:`_clip_evidence_ok`), and the trial
    still faces every other clause here and the whole verdict, which prices and
    reports what the linkage clipped.

    Measured on a 3-night compact seed: 457768 (a 9-obs seed, two obs of its
    last night 3.9 sigma off the whole-arc orbit; eight different tracklets
    each clipped those two) went from 1/8 apparitions to 8/8, and 2010 VB1 (an
    11-obs seed whose 4-obs third night the whole-arc fit rejects outright)
    from 1/3 to 3/3, each linkage dropping exactly the obs the whole-arc fit
    rejects.

    Records ``tr``'s clip for the trials after it; a trial without candidate
    obs (a rival refit) is neither recorded nor corroborated."""
    if not ctx.cfg.seed_clip_corroboration:
        return set()
    days = set()
    for line in added_lines.splitlines():
        try:
            days.add(int(obs_night_ordinal(line)))
        except ValueError:          # a line without a date cannot vouch
            continue
    days = frozenset(days)
    if not days:
        return set()
    inc, _resid = subset_included(tr.fit, ctx.pub_ids)
    clipped = frozenset(i for i in ctx.pub_ids if i not in inc)
    corroborated = set()
    for other_clipped, other_days in ctx.seed_clip_votes:
        if other_days.isdisjoint(days):
            corroborated |= clipped & other_clipped
    ctx.seed_clip_votes.append((clipped, days))
    if not corroborated:
        return set()
    lines = [l for l in ctx.pub_obs80.splitlines()
             if l.strip() and obs_unique_id(l) in corroborated]
    return corroborated if len(nights_of(lines)) == 1 else set()


# Trial rejections meaning "this tracklet is CLOSE but disagrees" -- the shape
# a Yarkovsky drift makes.  An rms-sanity rejection is corridor noise thousands
# of arcsec off, never worth a nongrav fit.
_NONGRAV_RESCUABLE = ("no-progress", "seed-guard", "seed-retention",
                      # _prune_trial can clip a drifting tracklet away with
                      # no rejection at all (2021 JK2).
                      "pruned",
                      # what an unmodelled drift across a long gap looks like
                      "hyp-drop")


def _nongrav_rescue(ctx, base_text, cand, names, tag, prior_used,
                    single_candidate, seed_lvl, seed_keep, trial_fit=None,
                    hypothesis_groups=(), parent_seed_rms=None):
    """Retry ONE rejected trial with Yarkovsky (:func:`_yarkovsky_retry`)
    when the rejection is in ``_NONGRAV_RESCUABLE``.  The rescued fit is
    pruned and must clear the SAME take criterion; nothing is waived.
    Returns ``(trial, cand)`` on a rescue, else None."""
    if tag not in _NONGRAV_RESCUABLE:
        return None

    def _takeable(tr):
        tr2, cand2 = _prune_trial(ctx, tr, base_text, cand, names)
        rej = _trial_rejection(ctx, tr2, prior_used, single_candidate,
                               seed_lvl, seed_keep, hypothesis_groups,
                               added_lines=cand2,
                               parent_seed_rms=parent_seed_rms)
        return None if rej is not None else (tr2, cand2)

    # the rejected trial spans the whole candidate arc, so its q leads
    return _yarkovsky_retry(ctx, base_text + cand,
                            f"trial [{names}]: rejected ({tag})", _takeable,
                            f"RESCUED [{names}]", trial_fit, level=L_STORY)


def flat_field(cfg, cands, n_likely):
    """True when the ranker rates NO likely candidate a plausible match (best
    P(match) under ``cfg.flat_field_ml_p``): its order among them is then near
    noise, so the first passing single is a guess that only the branch below
    it can confirm (see ``cfg.flat_field_singles``)."""
    floor = getattr(cfg, "flat_field_ml_p", 0.0) or 0.0
    likely = cands[:n_likely]
    if floor <= 0 or n_likely < 2 or not all("ml_p" in t for t in likely):
        return False
    return max(t["ml_p"] for t in likely) < floor


def _terminal_fan_due(cfg, ctx, trial, latest):
    """Should the terminal LOV fan (``cfg.terminal_fan``) run from this take?

    Only on a linear corridor, once per search, while the take's orbit is
    still at U >= ``cfg.variant_corridor_U`` -- and only when the vet called
    the take REVIEWABLE.  The fan's single shot is otherwise spent on whichever
    take comes back first, and in a weak field that is often a distractor:
    over 120 live recoveries half the firings started from a take holding no
    true tracklet, and in 11 searches a later true take found the fan already
    spent.  Every take whose fan produced a link vetted reviewable."""
    return bool(_terminal_fan_possible(cfg, ctx, trial)
                and latest is not None and latest.reviewable)


def _terminal_fan_possible(cfg, ctx, trial):
    """Everything :func:`_terminal_fan_due` asks except the vet's verdict --
    known BEFORE the vet, so a document the vet builds can be held until the
    fan has decided whether it stands (see _release_held)."""
    return bool(cfg.terminal_fan and not ctx.is_variant
                and not ctx.terminal_fan_done
                and cfg.variant_corridor_U > 0
                and trial.U >= cfg.variant_corridor_U
                and not _budget_spent(ctx))


def _terminal_fan(ctx, trial, obs_out, latest, held, depth):
    """Run the terminal LOV fan from a vetted take when it is due, and settle
    the take's held documents.  Returns the fan's linkage when it supersedes
    the take (the held documents are dropped), else None (they are sent).

    The fan is an extra pass over a verdict already reached, so its failure
    degrades to that verdict rather than aborting the search -- as a failed
    variant corridor or LOV-profile pass does."""
    cfg = ctx.cfg
    trc = _ctx_tracer(ctx)
    if _terminal_fan_due(cfg, ctx, trial, latest):
        ctx.terminal_fan_done = True
        if trc.enabled:
            trc.head(f"<<< depth {depth}'s reviewable terminal is still at "
                     f"U {trial.U:.2f} (>= {cfg.variant_corridor_U:g}) -> "
                     f"scanning the LOV fan from it")
        try:
            fanned = _variant_scan(ctx, trial, trial.used, 0, depth + 1, obs_out)
        except Exception as e:                                   # noqa: BLE001
            ctx.log(f"terminal LOV fan failed ({type(e).__name__}: {e}); "
                    f"keeping the vetted terminal")
            if trc.enabled:
                trc.warn("; ".join(
                    traceback.format_exc().strip().splitlines()[-3:]))
            fanned = None
        if fanned is not None and fanned.verdict is SearchVerdict.LINKED:
            return fanned
    _release_held(ctx, held, latest)
    return None


def _release_held(ctx, held, standing):
    """Transmit the held documents whose result is ``standing`` -- the verdict
    this take ends on.  A linkage the terminal fan superseded is never sent:
    the fan's own linkage, which contains it, is the one transmitted."""
    for result, doc, trkid_map in held or ():
        if result is standing:
            _transmit(ctx, result, doc, trkid_map)


def _build_trials(cfg, cands, n_likely, ranked, lov_round, interleave, trc,
                 hypothesis_obs80, lov_refine_tail=(), flat=False):
    """The ordered trial list for one round's trial queue -- each trial a
    list of indices into ``cands``.  THE one place trial source eligibility lives:

    - the LOV profile trial sources (``itf.lov_profile``: cluster joint
      trials, then LOV-centered singles) go FIRST and address the WHOLE
      field: on a weak seed junk fits cleanly, so whoever is fitted first
      wins.  The profile is measured across variants, so one window's
      likely/maybe flip must not hide a candidate;
    - the ranked "top-N together" batches (``cfg.recurse_batch_sizes``) and
      primary singles (``cfg.recurse_singles``) draw ONLY from the first
      ``n_likely``.  Batches keep only candidates with P(match) >=
      ``batch_min_ml_frac`` of the best (a far-below-best one only pollutes a
      joint fit; it still gets its single);
    - the SECOND ranker's best singles, so a candidate the primary buried
      still earns a fit (whole field under ``cfg.second_singles_maybe``).
      With ``interleave`` (a backtracking round) they weave p1, s1, p2, ...:
      the primary's pick proved barren, so take the OTHER model's top next.

    Trials are deduped on their index tuple.
    """
    batch_pool = ranked
    # (a round can hold only last-resort trials and no ranked candidate)
    if (cfg.batch_min_ml_frac > 0 and ranked
            and all("ml_p" in cands[i] for i in ranked)):
        floor = max(cands[i]["ml_p"] for i in ranked) * cfg.batch_min_ml_frac
        batch_pool = [i for i in ranked if cands[i]["ml_p"] >= floor]

    trials = []
    n_cluster = n_center = 0
    if lov_round is not None:
        by_ids = {cands[i]["_ids"]: i for i in range(len(cands))}
        if cfg.lov_profile_cluster_trials > 0:
            for group in lov_round.cluster_groups[:cfg.lov_profile_cluster_trials]:
                idxs = sorted(by_ids[i] for i in group if i in by_ids)
                if len(idxs) >= 2:
                    trials.append(idxs)
                    n_cluster += 1
        if cfg.lov_profile_center_singles > 0:
            for ids in lov_round.center_ids:
                if n_center >= cfg.lov_profile_center_singles:
                    break
                i = by_ids.get(ids)
                if i is not None:
                    trials.append([i])
                    n_center += 1
    for n in cfg.recurse_batch_sizes:          # top-N together (above the floor)
        if n >= 1:
            trials.append(batch_pool[:min(n, len(batch_pool))])
    n_singles = (max(cfg.recurse_singles, cfg.flat_field_singles) if flat
                 else cfg.recurse_singles)
    p_singles = ([[ranked[i]] for i in range(min(n_singles, n_likely))]
                 if n_likely > 1 else [])
    s_singles = []
    if cfg.recurse_second_singles > 0 and any("ml_p2" in t for t in cands):
        n2 = len(cands) if cfg.second_singles_maybe else n_likely
        ranked2 = sorted(range(n2), key=lambda i: -cands[i].get("ml_p2", 0.0))
        s_singles = [[ranked2[i]]
                     for i in range(min(cfg.recurse_second_singles, n2))]
    if interleave and s_singles:
        woven = []
        for a, b in zip(p_singles, s_singles + [None] * len(p_singles)):
            woven.append(a)
            if b is not None:
                woven.append(b)
        woven.extend(s_singles[len(p_singles):])
        trials.extend(woven)
    else:
        trials.extend(p_singles)
        trials.extend(s_singles)
    # LAST RESORT: the LOV-refinement trials ride at the BACK, so they run only
    # where the ranked trials found nothing.  Do not promote them to the
    # front: on 2021 SA70 that linked foreign tracklets in place of a correct
    # recovery.
    if lov_refine_tail:
        by_ids = {cands[i]["_ids"]: i for i in range(len(cands))}
        for ids in lov_refine_tail:
            i = by_ids.get(ids)
            if i is not None:
                trials.append([i])

    seen, uniq = set(), []
    for r in trials:
        key = tuple(r)
        if r and key not in seen:
            seen.add(key)
            uniq.append(r)

    if trc.enabled:
        if len(batch_pool) < n_likely:
            trc(f"joint-trial ml floor: {len(batch_pool)}/{n_likely} candidates "
                f"have P(match) >= {cfg.batch_min_ml_frac:.2f} x best -- the "
                f"rest are singles-only")
        if flat:
            trc(f"flat field: best P(match) < {cfg.flat_field_ml_p:g} -- "
                f"{len(p_singles)} singles, a barren take backtracks to the next")
        if n_cluster or n_center:
            trc(f"LOV-profile trial sources at the front: {n_cluster} cluster joint "
                f"trial(s), {n_center} LOV-centered single(s)")
        shape = " -> ".join(f"top-{len(r)}" if len(r) > 1
                            else f"single #{r[0] + 1}" for r in uniq)
        hyp = (f" (+ {len(hypothesis_obs80.splitlines())}-line hypothesis in "
               f"every trial)" if hypothesis_obs80 else "")
        trc.field("trials", f"{len(uniq)} trial{'' if len(uniq) == 1 else 's'} "
                  f"over {n_likely} likely tracklet(s): {shape}{hyp}")
    return uniq


# Cap on ledger entries for a pathological search; past it, lookups still work
# and only new entries go unrecorded.
_HYPOTHESIS_LEDGER_MAX = 20000


def _hypothesis_key(ctx, base_ids, tracklets):
    """Identity of the SEARCH a trial would run: the observations that would be
    fitted, and the force model they would be fitted under.

    A hypothesis is fitted once per search: the same astrometry gives the
    same fit and so the same corridor sweep.  Rank positions and the LOV
    variant that retrieved it are facts about the round, not the question.

    ``base_ids`` is the queue's own frozenset (seed + hypothesis), shared by
    reference so an entry allocates only the trial's few ids.  ``ctx.nongrav``
    is IN the key: the same lines under gravity and under Yarkovsky are
    different questions (:func:`_nongrav_rescue`).
    """
    ids = frozenset()
    for t in tracklets:
        # fallback for the bare dicts the offline tests and the harness build
        ids |= frozenset(t.get("_ids") or obs_ids(t.get("astrometry", "")))
    return (bool(getattr(ctx, "nongrav", False)), base_ids, ids)


def _recurrent_notes(ctx, result, limit=2):
    """The candidates that LINKED and were never shown, as diagnostics.

    A silenced one-night show leaves a verdict with no fit or astrometry
    (deliberately not :attr:`ItfSearchResult.reviewable`: that marks the
    branch barren and keeps the night out of every submission path), so the
    reader never learns WHAT was refused (2006 UP17).  The ledger says it once
    at the end.  A DIAGNOSTIC, never a warning: warnings gate the submit
    click-through, and this is a claim the search declined to make.
    """
    ledger = getattr(ctx, "tried_hypotheses", None)
    if not ledger or result.verdict is SearchVerdict.LINKED:
        return []
    seen, out = set(), []
    for e in sorted((e for e in ledger.values() if e["refusal"]),
                    key=lambda e: (-e["n"], -(e["used"] or 0))):
        tags = ", ".join(e["tags"] or ()) or "an unnamed tracklet"
        if tags in seen:
            continue
        seen.add(tags)
        fit = (f"fitting {e['used']}/{e['count']} obs at {e['rms']:.3f}\""
               if e["used"] is not None and e["rms"] is not None
               else "fitting")
        again = (f"; {e['n'] - 1} later round(s) reached the same "
                 f"hypothesis and were told it had already been tried"
                 if e["n"] > 1 else "")
        out.append(f"A tracklet DID link and was deliberately not shown: "
                   f"{tags}, taken and vetted in "
                   f"{e['where'] or 'this search'} ({fit}){again}. It was not "
                   f"shown because {_sentence(e['refusal'])} Nothing here is "
                   f"submittable -- but it is real astrometry that fitted the "
                   f"orbit, so judge it yourself.")
        if len(out) >= limit:
            break
    return out


def _hypothesis_record(ctx, key, settled, **fields):
    """Record (or update) one ledger entry, and return it.

    ``settled`` True means this exact fit has been answered and no later
    round re-asks it.  A take is recorded UNsettled and settled by
    :func:`_iterate` only if its subtree searched the whole corridor (a
    truncated sweep's answer is not final).

    ``getattr``: the tests' bare namespace has no ledger, which just means
    nothing is remembered across rounds.
    """
    ledger = getattr(ctx, "tried_hypotheses", None)
    if ledger is None or key is None:
        return None
    prev = ledger.get(key)
    if prev is None:
        if len(ledger) >= _HYPOTHESIS_LEDGER_MAX:
            return None
        prev = ledger[key] = {"settled": False, "n": 0, "tags": (),
                              "refusal": None, "where": "", "used": None,
                              "count": None, "rms": None}
    prev["settled"] = prev["settled"] or settled
    prev["n"] += 1
    prev.update({k: v for k, v in fields.items() if v is not None})
    return prev


class TrialQueue:
    """One round's trial queue: the tracklet subsets to fit-and-recurse
    on (:func:`_build_trials`), each trial guarded against degrading the
    EXISTING (seed) observations.  The caller pulls

        take = queue.next_take()        # (trial, obs_out, meta), or None

    and recurses on it.  It comes back for another take ONLY when
    backtracking is on and that take's branch proved barren -- after saying
    so explicitly:

        rode = queue.mark_failed(meta, survived_obs_ids)

    and next_take() resumes at the next untried trial (see
    :meth:`mark_failed`).  A single next_take() with no mark_failed is the
    non-backtracking trial queue.

    ``hypothesis_obs80`` is every tracklet a shallower level already took.
    Those obs are IN every trial fit here, as if published, so singles
    accumulate instead of each having to out-count the whole prior linkage.
    The terminal clip/acceptance gates still vet every linked obs.
    """

    def __init__(self, ctx, likely, rank_key, prior_linkage_used,
                 hypothesis_obs80="", parent=None, lov_round=None, extra_pool=None,
                 interleave=False, round_label="", lov_refine_tail=(),
                 linkage_fit=None):
        cfg = ctx.cfg
        self.ctx = ctx
        self.rank_key = rank_key
        self.prior_linkage_used = prior_linkage_used
        self.hypothesis_obs80 = hypothesis_obs80
        self.parent = parent
        # the full addressable field: likely first, then the maybe pool for the
        # trial sources _build_trials lets reach below the likely cliff
        self.cands = list(likely) + list(extra_pool or ())
        self.n_likely = len(likely)
        self.ranked = sorted(range(self.n_likely),
                             key=lambda i: rank_key(self.cands[i]))
        # the seed-guard waiver ("nothing to fall back on") is judged over the
        # WHOLE field: a maybe-source trial is something to fall back on.  An
        # 'unlikely' last-resort pick (the LOV-refinement tail) is not part of
        # that field -- it is tried BECAUSE the field was exhausted -- so it
        # never earns the waiver.
        self.single_candidate = sum(
            1 for t in self.cands if t.get("traj_category") != "unlikely") == 1
        # a flat field on a LINEAR corridor widens the singles window (a LOV
        # variant round is already a breadth scan; widening it multiplies it)
        self.flat = (flat_field(cfg, self.cands, self.n_likely)
                     and not getattr(ctx, "is_variant", False))
        self.queue = _build_trials(cfg, self.cands, self.n_likely, self.ranked,
                                  lov_round, interleave, _ctx_tracer(ctx), hypothesis_obs80,
                                  lov_refine_tail=lov_refine_tail, flat=self.flat)
        self.qi = 0
        self.tried_keys = set()
        # singles TRIED and rejected: the exhaustion test's evidence (see
        # _refused)
        self.singles_failed = []
        # set once a barren take has resumed this queue (mark_failed)
        self.resumed = False
        # outcome tally for the STORY-level rollup; DETAIL narrates each trial
        self.outcomes = Counter()
        self.n_tried = self.n_takes = 0
        self.base_text = ctx.pub_obs80 + hypothesis_obs80
        # for the search-wide ledger: the fixed part of every trial's key, and
        # the round's name
        self.base_ids = frozenset(obs_ids(self.base_text))
        self.round_label = round_label
        # the decision log's id for this round (itf/decisions.py), or None
        self.rid = (ctx.dlog.cur_round if getattr(ctx, "dlog", None) is not None
                    else None)
        self.seed_lvl, self.seed_keep = _seed_guard_levels(ctx)
        # what a shallower level linked; a trial must EXTEND it
        # (_trial_rejection clause (d))
        self.hypothesis_groups = _hypothesis_groups(ctx, hypothesis_obs80)
        # the seed rms under the linkage being extended: clause (d)'s tie-break
        self.parent_seed_rms = (subset_rms(linkage_fit, ctx.pub_ids)[0]
                                if self.hypothesis_groups and linkage_fit is not None
                                else None)
        self._closed = False

    def _dl(self, r, outcome, t0=None, tr=None):
        """One trial's outcome into the decision log (a no-op without one)."""
        if self.rid is None:
            return
        self.ctx.dlog.trial(
            self.rid, trksubs_of([self.cands[i] for i in r]), outcome,
            fit_sec=(time.perf_counter() - t0) if t0 is not None else None,
            rms=getattr(tr, "rms", None), used=getattr(tr, "used", None))

    def _refused(self, r, outcome, t0=None, tr=None, logged=None):
        """Book one refused trial: the rollup tally, the exhaustion evidence (a
        refused SINGLE is a rival ruled out, not merely unexamined; a joint
        trial failing says nothing about its members) and the decision log."""
        self.outcomes[outcome] += 1
        if len(r) == 1:
            self.singles_failed.append(r[0])
        self._dl(r, logged or outcome, t0=t0, tr=tr)

    def _yarkovsky(self, cand, names, tag, tr):
        """:func:`_nongrav_rescue` on this queue's take criterion."""
        return _nongrav_rescue(self.ctx, self.base_text, cand, names, tag,
                               self.prior_linkage_used, self.single_candidate,
                               self.seed_lvl, self.seed_keep, trial_fit=tr,
                               hypothesis_groups=self.hypothesis_groups,
                               parent_seed_rms=self.parent_seed_rms)

    def next_take(self):
        """The first remaining trial that passes the take criterion, as
        ``(trial, obs_out, meta)`` -- or None when the trial queue is exhausted
        (or the search budget is spent).

        Each trial is fitted, PRUNED (:func:`_prune_trial`: a blunder
        tracklet's obs clipped rather than failing the trial) and taken when
        it passes :func:`_trial_rejection` (seed bar from
        :func:`_seed_guard_levels`), after a Yarkovsky retry on a rescuable
        refusal.  ``obs_out`` is the WHOLE hypothesis (prior + that trial);
        ``meta`` describes the take for the confirmed-pair upgrade, the
        exhaustion test and the ledger."""
        ctx = self.ctx
        cands = self.cands
        trc = _ctx_tracer(ctx)
        while self.qi < len(self.queue):
            r = self.queue[self.qi]
            self.qi += 1
            # no-progress budget spent: a doomed wide-corridor grind
            if _budget_spent(ctx):
                trc.warn(f"budget spent ({ctx.budget.why()}) -- stopping the "
                         f"trial queue with "
                         f"{len(self.queue) - self.n_tried} trial(s) untried")
                break
            key = tuple(r)
            if not r or key in self.tried_keys:
                continue
            self.tried_keys.add(key)
            self.n_tried += 1
            cand = "".join(cands[i]["astrometry"] for i in r)
            names = (" + ".join(tracklet_tag(cands[i]) for i in r)
                     if trc.enabled else "")
            # FREE rejection: has this SEARCH already asked this question?
            # (tried_keys is per-round, on rank positions; the ledger is on obs.)
            hkey = _hypothesis_key(ctx, self.base_ids, [cands[i] for i in r])
            prev = (getattr(ctx, "tried_hypotheses", None) or {}).get(hkey)
            if prev is not None and prev["settled"]:
                prev["n"] += 1
                # still exhaustion evidence: ruled out earlier is ruled out
                self._refused(r, "already tried this search",
                              logged="already tried")
                if trc.wants(L_DETAIL):
                    trc(f"trial [{names}]: this exact fit was already made in "
                        f"{prev['where'] or 'an earlier round'} -- not "
                        f"re-fitting")
                continue
            # Cheap warm-step rejection before the cold fit.  NB: NOT recorded
            # in the ledger -- it judges this round's PARENT orbit, which
            # differs per LOV variant.
            _t_trial = time.perf_counter()
            if _screen_trial(ctx, self.parent, cand, names):
                self._refused(r, "warm screen", t0=_t_trial)
                continue
            if trc.wants(L_DETAIL):
                trc(f"trial [{names}]: trial fit "
                    f"(seed {len(ctx.pub_obs80.splitlines())} lines "
                    + (f"+ {len(self.hypothesis_obs80.splitlines())} hypothesis "
                       if self.hypothesis_obs80 else "")
                    + f"+ {sum(len(cands[i]['rows']) for i in r)} candidate "
                      f"obs)...")
            tr = _fit(ctx, self.base_text + cand)
            _note_fit(ctx)      # work spent; reset by _note_progress on a take
            if not tr.ok:
                _hypothesis_record(ctx, hkey, True, where=self.round_label)
                if trc.wants(L_DETAIL):
                    trc(f"trial [{names}]: fit FAILED ({tr.error}) -- next trial")
                self._refused(r, "fit failed", t0=_t_trial)
                continue
            cand_full = cand
            tr, cand = _prune_trial(ctx, tr, self.base_text, cand, names)
            if cand != cand_full:
                # obs clipped: on a low-q object, was it the model that disagreed?
                rescued = self._yarkovsky(cand_full, names, "pruned", tr)
                if rescued is not None:
                    tr, cand = rescued
            rej = _trial_rejection(ctx, tr, self.prior_linkage_used,
                                  self.single_candidate, self.seed_lvl,
                                  self.seed_keep, self.hypothesis_groups,
                                  added_lines=cand,
                                  parent_seed_rms=self.parent_seed_rms)
            if rej is not None:
                rescued = self._yarkovsky(cand, names, rej[2], tr)
                if rescued is not None:
                    tr, cand = rescued
                    rej = None
            if rej is not None:
                trc_msg, log_msg, tag = rej
                # settled: besides the fit, the criterion reads only
                # search-constant ctx.cfg / ctx.pub
                _hypothesis_record(ctx, hkey, True, where=self.round_label,
                            tags=tuple(tracklet_tag(cands[i]) for i in r))
                if log_msg:
                    ctx.log(f"trial of {len(r)}: {log_msg} -- skip")
                if trc.wants(L_DETAIL):
                    trc(f"trial [{names}]: {trc_msg} -- next trial")
                self._refused(r, tag, t0=_t_trial, tr=tr)
                continue
            return self._take(r, tr, cand, names, hkey, _t_trial)
        self._close(trc)
        return None

    def _take(self, r, tr, cand, names, hkey, t0):
        """Trial ``r`` passed the take criterion: narrate it, record it, and
        hand back ``(trial, obs_out, meta)`` -- the hypothesis to recurse on."""
        ctx = self.ctx
        cands = self.cands
        trc = _ctx_tracer(ctx)
        if trc.enabled:
            _trial_queue_outcome(trc, self.outcomes, self.n_tried,
                                 len(self.queue))
            trc.field("TAKEN", f"{names}", L_HEAD)
            trc.cont(f"used {tr.used}/{tr.count}, rms {tr.rms:.3f}\" "
                     f"-- recursing on the improved orbit", L_HEAD)
        ctx.log(f"recursing on {len(r)} tracklet(s) (used {tr.used})")
        self._dl(r, "TAKEN", t0=t0, tr=tr)
        # Real progress restarts the no-progress clock -- but not a take the
        # queue reached by resuming after a barren one: that is the round
        # still looking, and counting it would leave backtracking bounded only
        # by the search ceiling.  Takes in the branch below it count as usual.
        if not self.resumed:
            _note_progress(ctx)
        meta = {
            "indices": list(r),
            # confirmed-pair signature: exactly the top-2 by rank in ONE trial
            "is_top_pair": len(r) == 2 and list(r) == self.ranked[:2],
            # with the singles REJECTED before this take: _exhaustion_verdict
            # fits the rivals these do not cover
            "tracklets": [cands[i] for i in r],
            "failed_tracklets": [cands[i] for i in self.singles_failed],
            "ml_ps": [cands[i].get("ml_p") for i in r],
            # the ledger key, so _iterate can settle this take's subtree
            "hyp_key": hkey,
            "tags": tuple(tracklet_tag(cands[i]) for i in r),
            "round": self.round_label,
            "obs_ids": [
                {obs_unique_id(l)
                 for l in cands[i]["astrometry"].splitlines() if l.strip()}
                for i in r],
        }
        # Remember each taken tracklet AS SWEPT, before pruning: the acceptance
        # gate's retention test needs the original denominator.  (Absent on
        # the tests' namespace, where the gate reads one tracklet per line.)
        swept = getattr(ctx, "swept_tracklets", None)
        if swept is not None:
            for ids, t in zip(meta["obs_ids"], meta["tracklets"]):
                entry = (tracklet_tag(t), frozenset(ids))
                for i in ids:
                    swept[i] = entry
                lines = getattr(ctx, "swept_lines", None)
                if lines is not None:
                    lines[entry[1]] = t["astrometry"]
        self.n_takes += 1
        # A take that passed clause (d) while expelling linked tracklets
        # SUPERSEDES them: hand down the corrected hypothesis.
        hypothesis_down = self.hypothesis_obs80
        expelled = (_expelled_tracklets(tr, self.hypothesis_groups)
                    if self.hypothesis_groups else [])
        if expelled:
            gone = set().union(*(i for _t, i, _l, _n in expelled))
            hypothesis_down = drop_obs_ids(self.hypothesis_obs80, gone)
            meta["superseded"] = [t for t, _i, _l, _n in expelled]
            n_in, n_out = _supersedes(tr, expelled, cand)
            if trc.enabled:
                why = (f"brings {n_in} night(s) that reject {n_out} a "
                       f"shallower level linked"
                       if n_in > n_out else
                       f"a one-for-one swap ({n_in} night for {n_out}) "
                       f"that fits the object's own obs no worse than the "
                       f"linkage it replaces")
                trc.field("SUPERSEDED", ", ".join(meta["superseded"])
                          + f" -- this take {why}, so that take was the "
                          f"mistake; dropped from the hypothesis", L_HEAD)
            ctx.log(f"superseded {', '.join(meta['superseded'])} "
                    f"({n_in} nights in, {n_out} out)")
        return tr, hypothesis_down + cand, meta

    def mark_failed(self, meta, survived):
        """The take described by ``meta`` proved BARREN (its subtree returned
        nothing reviewable); the queue resumes at the next untried trial.
        ``survived`` is the take's obs ids minus hypothesis and seed; returns
        the tracklets that actually RODE, for the caller's trace.

        It bans NOTHING: the unit not to repeat is the HYPOTHESIS, which the
        ledger already holds, while {seed + A + B} must stay askable -- A
        alone may be barren because A needs B.

        A barren single counts as exhaustion evidence.  A barren JOINT take's
        clipped members are re-queued next: no later trial necessarily
        re-forms them (top-N batches dedup to one trial on a small pool)."""
        self.outcomes["barren take"] += 1
        self.resumed = True
        r = meta["indices"]
        rode = [t for t, tids in zip(meta["tracklets"], meta["obs_ids"])
                if tids & survived]
        rode_ids = {t["_ids"] for t in rode}
        if len(r) == 1:
            self.singles_failed.append(r[0])
        elif len(rode) < len(r):
            # the members the fit clipped never got their own chance
            r2 = [i for i in r if self.cands[i]["_ids"] not in rode_ids]
            if r2 and tuple(r2) not in self.tried_keys:
                self.queue.insert(self.qi, r2)
        return rode

    def _close(self, trc):
        if self._closed:
            return
        self._closed = True
        if trc.enabled and self.queue:
            _trial_queue_outcome(trc, self.outcomes, self.n_tried, len(self.queue))
            trc.field("outcome",
                      "no link -- no trial improved the fit cleanly"
                      if not self.n_takes
                      else f"no link -- {self.n_takes} take(s) all came back "
                           f"barren and no later trial improved the fit "
                           f"cleanly",
                      L_HEAD)


def _recurse_trial_queue(ctx, likely, rank_key, prior_linkage_used, hypothesis_obs80="",
                    parent=None, **kw):
    """The first take of a fresh :class:`TrialQueue`, in the
    ``(trial, obs_out, meta)`` / ``(None, "", None)`` contract -- the
    non-backtracking queue, as the offline tests drive it."""
    take = TrialQueue(ctx, likely, rank_key, prior_linkage_used, hypothesis_obs80,
                   parent=parent, **kw).next_take()
    return take if take is not None else (None, "", None)


def display_obs80(ctx, obs_out):
    """The astrometry the CALLER should see: every seed line, plus the ITF
    lines the linkage kept.

    An ITF search must never make one of the object's own observations
    disappear.  The web UI rebuilds its observation list by re-fitting this
    string, so a line clipped by :func:`_clip_poor_fitters` and missing here
    could never be judged or restored.  With the full seed, the caller's fit
    shows it rejected instead.  The linkage document and acceptance gates
    keep the clipped-down text.
    """
    return ctx.pub_obs80 + obs_out


# ---------------------------------------------------------------------------
# Yarkovsky: the one force model a search switches on mid-flight
# ---------------------------------------------------------------------------
def _nongrav_gate(ctx, *fits):
    """The shared gates on every Yarkovsky retry: enabled, not already on,
    inside the attempt budget (``cfg.nongrav_rescue_max``), and q under
    ``cfg.nongrav_rescue_q``, where Yarkovsky is measurable at all.  Returns
    that q, or None when a gate refuses.

    ``fits`` are orbits whose q may inform the call, plus the seed's.  The
    lowest wins: a short arc barely determines q (2021 JK2)."""
    cfg = ctx.cfg
    if not cfg.nongrav_rescue_q or getattr(ctx, "nongrav", False):
        return None
    if ctx.telemetry.get("nongrav_tries", 0) >= cfg.nongrav_rescue_max:
        return None
    qs = [getattr(f, "q", None) for f in fits] + [getattr(ctx.pub, "q", None)]
    qs = [q for q in qs if q is not None]
    if not qs or min(qs) >= cfg.nongrav_rescue_q:
        return None
    return min(qs)


def _nongrav_fit(ctx, text):
    """Spend one Yarkovsky attempt: fit ``text`` with the nongrav model."""
    ctx.telemetry["nongrav_tries"] = ctx.telemetry.get("nongrav_tries", 0) + 1
    return fit_astrometry(text, ctx.obscodes, registry=_ctx_registry(ctx),
                          nongrav=True)


def _nongrav_latch(ctx, tr, what):
    """A Yarkovsky retry worked: every later fit in this search keeps the model."""
    ctx.nongrav = True
    ctx.nongrav_ref = getattr(tr, "fit", None)
    ctx.telemetry["nongrav_rescued"] = True
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.field("nongrav", f"Yarkovsky {what} (used {tr.used}/{tr.count}, rms "
                  f"{tr.rms:.3f}\") -- every later fit in this search keeps "
                  f"the model", L_HEAD)


def _yarkovsky_retry(ctx, text, why, judge, what, *fits, level=L_HEAD):
    """Before cutting observations the gravity model disagrees with, ask
    whether it is the MODEL that is wrong: over a decade-plus gap a small
    low-perihelion object's Yarkovsky drift exceeds the astrometry's error,
    and every symptom (a clipped ob, a failed tracklet, a refused trial) looks
    like a bad tracklet.  Needed for 2021 JK2; without it, half its tracklets
    were thrown away.

    Gated by :func:`_nongrav_gate`.  Re-fits ``text`` with Yarkovsky and hands
    the trial to ``judge``, which applies the SAME test the gravity fit failed
    and returns its answer, or None.  On an answer the model latches on for the
    rest of the search (:func:`_nongrav_latch`, wording ``what``) and the
    answer is returned; otherwise None and the caller cuts."""
    q = _nongrav_gate(ctx, *fits)
    if q is None:
        return None
    trc = _ctx_tracer(ctx)
    # Yarkovsky is a drift ACROSS apparitions: on one apparition it is an extra
    # free parameter that absorbs whatever misfits, and once latched its
    # coefficient is held fixed in every later fit (_fit, ``nongrav_ref``).
    # 2007 DB61's 45-tracklet search latched it on a 2026-only take (the model
    # still clipped 2 of 42 obs) -- 3/5 apparitions.
    if opp_profile(text).n_oppositions < 2:
        if trc.enabled:
            trc(f"{why}, but the arc is one apparition -- Yarkovsky cannot be "
                f"measured on it, so no retry", level)
        return None
    if trc.enabled:
        trc(f"{why}, q {q:.2f} < {ctx.cfg.nongrav_rescue_q} -- re-fitting WITH "
            f"Yarkovsky first", level)
    tr = _nongrav_fit(ctx, text)
    _note_fit(ctx)
    out = judge(tr) if tr.ok else None
    if out is None:
        if trc.enabled:
            trc("Yarkovsky does not hold it either -- staying gravity-only", level)
        return None
    _nongrav_latch(ctx, tr, what)
    return out


def _clip_poor_fitters(ctx, trial, combined, obs_out):
    """Clip obs that don't fit the linkage orbit (legacy thresholds: 0.5" for
    a well-determined orbit, U < 3.5, else 0.74") and re-fit.  Returns the
    possibly-updated ``(trial, combined, obs_out, error_reason)`` --
    ``error_reason`` is None unless the post-drop fit failed."""
    thresh = 0.5 if trial.U < 3.5 else 0.74
    k = ctx.cfg.clip_sigma_mult
    # The ITF side clips as TRACKLETS: what falls below cfg.min_tracklet_obs
    # goes whole, and `combined` loses the same obs.
    obs_out_dropped = drop_offending(trial.fit, obs_out, thresh, sigma_mult=k,
                                     min_tracklet_obs=ctx.cfg.min_tracklet_obs)
    gone = obs_ids(obs_out) - obs_ids(obs_out_dropped) - ctx.pub_ids
    # The seed is NEVER clipped: with the seed gone the seed-preservation
    # clause passes vacuously (_accept_quality_ok).  Needed for 2025 NV385;
    # without it, a trial fitted to a different body reached the acceptance
    # gates.  A misfitting seed obs stays, rejected by the fit, and enters the
    # Delta-chi2 at its full residual.
    kept_ids = {obs_unique_id(l) for l
                in drop_offending(trial.fit, combined, thresh,
                                  sigma_mult=k).splitlines() if l.strip()}
    combined_dropped = "".join(
        l + "\n" for l in split_obs80_long_lines(combined).splitlines()
        if l.strip() and (oid := obs_unique_id(l)) not in gone
        and (oid in ctx.pub_ids or oid in kept_ids))
    if (trial.used > ctx.pub.used
            and abs(len(combined) - len(combined_dropped)) > 40
            and combined_dropped.strip()):
        def _holds(tr):
            still = drop_offending(tr.fit, obs_out, thresh, sigma_mult=k,
                                   min_tracklet_obs=ctx.cfg.min_tracklet_obs)
            return tr if not (obs_ids(obs_out) - obs_ids(still)) else None
        ng = _yarkovsky_retry(ctx, combined,
                              f"about to clip {len(gone)} linked obs at > "
                              f"{thresh}\" on a low-q object", _holds,
                              "HELD the linkage together, nothing clipped",
                              trial) if gone else None
        if ng is not None:
            return ng, combined, obs_out, None
        ctx.log(f"dropping poor fitters (> {thresh}\")")
        combined, obs_out = combined_dropped, obs_out_dropped
        trial = _fit(ctx, combined)
        if not trial.ok:
            return (trial, combined, obs_out,
                    f"post-drop fit failed ({trial.error})")
    return trial, combined, obs_out, None


# ---------------------------------------------------------------------------
# Judging a linkage
#
# Every terminal candidate -- and every alternative to it: the linkage minus
# its failing tracklets, the seed's own opposition extended, a final-sweep
# augmentation -- is judged by the SAME three steps:
#
#   1. _assess          is it CONSISTENT?  It gains astrometry, every linked
#                       tracklet fits its own uncertainties, the seed is still
#                       the object (SeedCheck); plus the opposition profile.
#   2. _extension_check the cautions a reviewer must see on a consistent one.
#   3. _judge           what is it WORTH?  A submittable profile with nothing
#                       withdrawing it is LINKED; a consistent linkage that
#                       cannot be submitted as it stands is a 0.6 POSSIBLE;
#                       anything else is refused with the clause that refused.
# ---------------------------------------------------------------------------
@dataclass
class Assessment:
    """One linkage -- the seed plus ``obs_out`` -- under one fit, measured
    once: everything the verdict reads about its consistency and profile."""
    trial: object
    combined: str
    obs_out: str
    profile: OppProfile
    spans: int
    profile_ok: bool          # the opposition-profile gate, confirmation folded in
    confirmed: bool           # a refused 1.75 the confirmation upgraded to 3
    gains: bool               # strictly more used AND total obs than the seed
    quality_ok: bool          # every clause of _accept_quality_ok
    notes: list               # that function's per-clause trace lines
    bad_tracklets: list       # (tag, ids) failing a PER-TRACKLET clause
    other_fail: bool          # a clause that is not per-tracklet failed
    same_opp: tuple | None    # the seed's own opposition extended (1.75 only)
    fails: list = field(default_factory=list)   # structured failed clauses
    n_groups: int = 0         # linked tracklets, as swept
    seed: object = None       # the SeedCheck
    vouched: dict = field(default_factory=dict)   # tracklets its repair vouched for
    dropped: frozenset = frozenset()  # the seed night its repair dropped

    @property
    def consistent(self):
        return self.gains and self.quality_ok

    def accepts_profile(self, ctx):
        return _opp_profile_clause(ctx.cfg, self.trial, self.profile_ok,
                                   ctx.initial_opp_profile)

    def acceptable(self, ctx):
        """Consistent, on a profile that may be accepted, and not dropping a
        seed span it has not paid for: what a linkage that will be SUBMITTED
        must be (``SeedCheck.unpaid``)."""
        return (self.consistent and self.accepts_profile(ctx)
                and (self.seed is None or self.seed.unpaid is None))


def _assess(ctx, trial, combined, obs_out, prior_linkage_used, trial_meta=None,
            depth=-1, vouched=None, dropped=frozenset()):
    """Measure one linkage (see :class:`Assessment`).  ``vouched`` /
    ``dropped`` -- what a repair of this linkage earned (:func:`_with_vouched`,
    :func:`_drop_seed_night`); recorded on the Assessment, never kept anywhere
    else, so a sibling alternative cannot inherit them.

    A refused three-thin (1.75) profile is offered to the confirmation
    (:func:`_confirm_top_pair`), measured against the seed's own opposition
    extended; a profile the gate already passes needs no confirmation and is
    never held to its no-warnings condition."""
    profile, spans, ok = _opp_profile_gate(ctx, trial, combined)
    thin175 = profile.grade is OppProfileGrade.THREE_THIN
    same_opp = _seed_opp_extension(ctx, obs_out) if thin175 else None
    confirmed = False
    if (thin175 and not ok
            and _confirm_top_pair(ctx, trial, combined, trial_meta, depth,
                                  same_opp=same_opp)):
        profile, ok, confirmed = profile.confirmed(), True, True
    detail = {}
    vouched = dict(vouched or {})
    dropped = frozenset(dropped)
    quality_ok, notes = _accept_quality_ok(ctx, trial, obs_out,
                                           prior_linkage_used, detail=detail,
                                           vouched=vouched, dropped=dropped)
    return Assessment(
        trial=trial, combined=combined, obs_out=obs_out, profile=profile,
        spans=spans, profile_ok=ok, confirmed=confirmed,
        gains=trial.used > ctx.pub.used and trial.count > ctx.pub.count,
        quality_ok=quality_ok, notes=notes,
        bad_tracklets=detail.get("bad_tracklets") or [],
        other_fail=bool(detail.get("other_fail")), same_opp=same_opp,
        fails=detail.get("fails") or [], n_groups=detail.get("n_groups", 0),
        seed=detail.get("seed"), vouched=vouched, dropped=dropped)


def _trace_assessment(ctx, a, head="vetting"):
    trc = _ctx_tracer(ctx)
    if not trc.enabled:
        return
    trial = a.trial
    trc.field(head, f"used {trial.used} (seed {ctx.pub.used}), obs "
              f"{trial.count} (seed {ctx.pub.count})")
    for note in a.notes:
        trc.cont(note)
    trc.cont(f"opps {ctx.initial_opp_profile} -> {a.profile}, spans {a.spans}, "
             f"opposition profile  {'ok' if a.profile_ok else 'FAIL'}"
             + (" (confirmed 1.75 -> 3)" if a.confirmed else "")
             + (" (waived: NEO -- a proposed link for the MPC recovery page)"
                if not a.profile_ok and _neo_orbit_waiver(ctx.cfg, trial)
                else ""))
    trc.cont(f"q {trial.q:.2f}au, e {trial.e:.3f}, "
             f"arc {trial.arclength_days:.1f}d "
             f"(seed {ctx.pub.arclength_days:.1f}d), overall rms "
             f"{trial.rms:.3f}\" (reported, not gated)")


def _refusal_reasons(ctx, a):
    """The clauses that refused ``a``, as phrases a reader can act on, in the
    order :func:`_judge` asks them.  The failing clause notes are reused
    verbatim, so reason and trace agree; never re-derived from quantities
    acceptance does not consult (a refusal for a degraded seed must not read
    as an rms failure, 2013 SM26)."""
    trial = a.trial
    out = []
    if not a.gains:
        out.append(f"adds nothing to the seed fit (used {trial.used} of "
                   f"{trial.count}, seed {ctx.pub.used} of {ctx.pub.count})")
    if not a.accepts_profile(ctx):
        out.append("too few nights/oppositions to submit")
    if not a.quality_ok:
        chosen = [f["note"] for f in a.fails if f.get("note")]
        out.extend(chosen[:2])
        if len(chosen) > 2:
            out.append(f"and {len(chosen) - 2} more failing clause(s)")
        if not chosen:
            out.append("a linkage-quality clause failed")
    return out


def _refused(ctx, result, a, reasons=None):
    """Fill ``result`` as refused, naming the clauses."""
    why = reasons if reasons is not None else _refusal_reasons(ctx, a)
    cal_nights = count_nights(a.combined)
    nights_txt = (f"{a.spans}" if cal_nights == a.spans
                  else f"{a.spans} ({cal_nights} nights)")
    profile_txt = (f"opps {ctx.initial_opp_profile} -> {a.profile}, spans "
                   f"{nights_txt}, rms {a.trial.rms:.3f}\"")
    result.verdict = SearchVerdict.NONE
    result.reason = (f"{'; '.join(why)} ({profile_txt})" if why
                     else f"abandoned ({profile_txt})")
    if a.seed is not None:
        result.vetting = a.seed.vetting()
    ctx.log(result.reason)
    return result


@dataclass
class ExtensionCheck:
    """The cautions a reviewer must see on a consistent linkage
    (:func:`_extension_check`)."""
    vetting: dict
    cautions: list = field(default_factory=list)   # doubts: gate the submit
                                        # click-through, withdraw a confirmation
    held: list = field(default_factory=list)       # cautions that may never
                                        # take a submission away
    diagnostics: list = field(default_factory=list)
    seed_caution: str | None = None
    seed_degraded: bool = False         # materially worse: demotes to 0.6
    magt: dict | None = None
    pickups: list = field(default_factory=list)


def _extension_check(ctx, a):
    """The softer findings on a CONSISTENT linkage, for the reviewer: a
    paid-for seed clip, a seed degradation (which may demote, see
    :class:`SeedCheck`), a systematically offset night, a photometric outlier
    tracklet, a partial pickup of a designated object, residual trends."""
    cfg = ctx.cfg
    trial = a.trial
    trc = _ctx_tracer(ctx)
    seed = a.seed
    # `pre` is the OVERALL per-axis rms, for the overall_* fields and the
    # coherent-offset noise floor -- not the seed's subset_rms currency
    pre = ctx.pub.rms
    x = ExtensionCheck(vetting={
        "overall_rms_pre": round(pre, 4),
        "overall_rms_post": round(trial.rms, 4),
        **seed.vetting(),
        "opps_pre": ctx.initial_opp_profile.legacy_code,
        "opps_post": a.profile.legacy_code,
        "nights_post": a.spans,
    })
    # A dropped seed span is a finding in its own right: it leads -- paid for
    # (a submittable linkage), or not (only a 0.6 gets this far unpaid).
    ledger = seed.ledger
    if seed.unpaid:
        x.cautions.append(f"linking it drops part of the object's own arc: "
                          f"{seed.unpaid}. Check that night's astrometry -- "
                          f"this lead cannot be submitted as it stands")
    elif ledger["lost_nights"]:
        note = _clip_note(ledger, seed.n_linked, a.profile)
        x.cautions.append(note)
        ctx.log(note)
    elif ledger["shed_obs"]:
        # Ordinary rejection: logged, deliberately NOT a warning -- routine
        # click-throughs are how real cautions stop being read.
        ctx.log(f"{ledger['shed_obs']} seed obs clipped across surviving nights "
                f"{[e['night'] for e in ledger['partial_nights']]} -- ordinary "
                f"rejection, every night of the original arc still represented")
    if seed.caution is not None:
        x.cautions.append(seed.caution)
        x.seed_caution, x.seed_degraded = seed.caution, seed.degraded
        if not seed.degraded and trc.enabled:
            trc.head(f"seed degradation NOT a demotion: {seed.caution}")
    from pyroorb import ident_check
    coh = ident_check.coherent_offset_misfit(trial.fit, noise_floor=pre)
    if coh:
        # say WHOSE night: the offender can be the object's own (2011 QH108)
        own_nights = _own_nights_as_ident(ctx, trial.fit)
        whose = ("a night of the object's OWN published arc"
                 if any(n in coh for n in own_nights) else "a linked night")
        x.cautions.append(f"{whose} is systematically offset: {coh}")
    # Residual ORDER rather than size (itf/residual_trend.py): a DIAGNOSTIC
    # only -- it flags about as many correct tracklets as wrong ones.
    linked = _linked_ids(ctx, a.combined)
    trendw = measure_trends(trial.fit, linked, ctx.pub_ids)
    if trendw:
        x.diagnostics.append(f"{TREND_NOTE} a linked tracklet: {trendw}")
        ctx.log("residual trend detail: " + measure_trends(
            trial.fit, linked, ctx.pub_ids, detailed=True))
    # Photometry of whole TRACKLETS: a caution, never a veto -- reported
    # magnitudes are too unreliable to refuse a good astrometric match
    # (2015 BO519).
    magt = mag_outlier_tracklet(trial.fit, cfg.warn_mag_resid, cfg.warn_mag_p)
    if magt:
        sig = ("single observation" if magt["p"] is None
               else f"{magt['n']} obs, p={magt['p']:.2g}")
        # say WHOSE tracklet: on the object's own, "possible different body"
        # is false (2025 NX363)
        magt["own"] = _mag_tracklet_is_own(ctx, trial, magt)
        whose = ("the object's OWN published arc" if magt["own"]
                 else "a linked tracklet")
        tail = "" if magt["own"] else " (possible different body)"
        when = ("" if magt.get("jd") is None
                else f", {_night_label(magt['jd'])}")
        x.cautions.append(
            f"a tracklet of {whose} ({magt['station']}{when}, {sig}) averages "
            f"{magt['mean']:+.1f} mag from the object's predicted brightness"
            f"{tail}")
    x.magt = magt
    # Partial pickup of a designated object whose other tracklets misfit
    # (itf/pickup.py).  A designation the check could not MEASURE is held: an
    # unreachable database is absence of evidence and must not take a
    # linkage away, but its warning still gates every submit path.
    x.pickups = pickup.review(a.obs_out, getattr(ctx, "source", None),
                              fit=trial.fit, obscodes=ctx.obscodes, cfg=cfg,
                              own_desigs=getattr(ctx, "own_desigs", ()),
                              log=ctx.log)
    if x.pickups:
        x.vetting["partial_pickups"] = x.pickups
        for rec in x.pickups:
            if rec["compromised"]:
                x.cautions.append(rec["warning"])
            elif rec.get("unchecked"):
                x.held.append(rec["warning"])
            else:
                ctx.log(rec["warning"])
        unchecked = [r["desig"] for r in x.pickups if r.get("unchecked")]
        if unchecked:
            x.vetting["pickup_unchecked"] = unchecked
    # A two-axis residual walk is a reviewer CAUTION that may ask for a look,
    # never take a submission away.
    trend2 = two_axis_trend_warning(trial.fit, linked)
    if trend2:
        x.held.append(trend2)
        x.vetting["ramp_two_axis"] = True
        if trc.enabled:
            trc.warn(trend2)
    # informational number (and batch CSV column): one bad ob is not a warning
    dmag, _omag = worst_mag_residual(trial.fit)
    x.vetting["worst_mag_resid"] = round(dmag, 2)
    x.vetting["mag_tracklet"] = magt
    return x


def _judge(ctx, a, tracklets_all, prior_linkage_used, trial_meta=None,
           depth=-1, sweep="all", lead=()):
    """THE verdict on one linkage (see the section comment above).

    ``sweep`` -- the post-acceptance final sweep (:func:`_final_sweep`):
    ``"all"``, ``"archive"`` (the deepest recursion already swept this exact
    orbit over the ordinary corridor) or ``None``.  ``lead`` -- warnings that
    go first (the reduced linkage's account of what it declined)."""
    result = ItfSearchResult(telemetry=ctx.telemetry)
    result.warnings.extend(lead)
    # A thin profile the gate refused (which implies a one-opposition seed:
    # _opp_profile_clause) that still GAINS astrometry is a find the search
    # distrusts rather than nothing: it stays a 0.6 whatever refuses it.
    distrust = a.gains and a.profile.is_thin and not a.accepts_profile(ctx)
    if not a.consistent:
        if distrust:
            return _distrusted(ctx, result, a, _plain_refusal(a),
                               clauses=_refusal_reasons(ctx, a))
        return _refused(ctx, result, a)
    if a.accepts_profile(ctx) and not a.acceptable(ctx):
        return _refused(ctx, result, a, [f"ITF extension is not paid for: "
                                         f"{a.seed.unpaid}"])
    if a.accepts_profile(ctx) and sweep:
        a = _final_sweep(ctx, a, prior_linkage_used,
                         skip_modern=(sweep == "archive"))
    if a.accepts_profile(ctx) and a.seed is not None and a.seed.bad_night:
        a = _drop_seed_night(ctx, a, prior_linkage_used) or a
    x = _extension_check(ctx, a)
    if a.accepts_profile(ctx):
        return _accepted(ctx, result, a, x, tracklets_all)
    if a.profile.grade is OppProfileGrade.THREE_THIN:
        return _thin_possible(ctx, result, a, x)
    if a.profile.grade is OppProfileGrade.TWO_THIN:
        return _lone_night(ctx, result, a, x, trial_meta, depth, tracklets_all)
    return _refused(ctx, result, a)


def _keep(ctx, result, a, x):
    """``result`` carries ``a``'s fit, astrometry and ``x``'s findings."""
    result.fit = a.trial.fit
    result.combined_obs80 = display_obs80(ctx, a.obs_out)
    result.itf_obs80 = a.obs_out
    result.warnings.extend(x.cautions)
    result.diagnostics.extend(x.diagnostics)


def _new_opposition_unpredicted(ctx, a):
    """Why a single-opposition seed's NEW opposition is not vouched for, or None.

    A seed seen in one opposition leaves its orbit loose along the line of
    variations, so the first tracklet taken in another opposition largely chooses
    a point on that line; a second one there may agree only with the first -- two
    detections of one other body. 2022 DF9: a 3-night seed (U 11.4) took two 2006
    SDSS pairs (each moving under 0.5") at 0.48", every per-tracklet clause passed,
    and both sit ~121,000" off DF9's real orbit.

    Two questions, and the linkage is unvouched only when BOTH say no:

    * the identification penalty: the seed's chi2 rise under the joint orbit
      (:class:`SeedPreservation`) plus the new opposition's own chi2 there (kept
      observations, station-floored sigma, as clause 2 scores them), against chi2_6
      at ``cfg.new_opposition_penalty_conf`` -- what joining costs, the new
      opposition's stand-alone chi2 taken as zero;
    * a warm leave-out (:func:`leaveout.warm_predictive`): some new-opposition
      tracklet is predicted by the rest at p < ``cfg.new_opposition_leaveout_p``.

    Measured on 520 production new oppositions the MPC has resolved: the penalty
    alone refuses nine correct linkages, the leave-out alone two; together, none."""
    from scipy.stats import chi2 as _chi2
    from pyroorb.obs80 import mpc80_to_radec
    from .leaveout import warm_predictive
    cfg = ctx.cfg
    if ctx.initial_opp_profile.n_oppositions != 1 or ctx.pub.fit is None:
        return None
    trial = a.trial
    lo, hi = ctx.pub.earliest_jd, ctx.pub.latest_jd
    jd_of = {}
    for line in a.obs_out.splitlines():
        if line.strip():
            try:
                jd_of[obs_unique_id(line)] = mpc80_to_radec(line)[0]
            except Exception:                       # noqa: BLE001 -- an unplaceable line
                continue
    gap = mpcrules.MPC_OPPOSITION_GAP_DAYS
    new = []
    for tag, ids in _linked_tracklets_as_swept(ctx, a.obs_out):
        jds = [jd_of[i] for i in ids if i in jd_of]
        if jds and (max(jds) < lo - gap or min(jds) > hi + gap):
            new.append((tag, ids))
    if len(new) < 2:
        return None             # one tracklet there is a lone night: already a possible
    included, _ = subset_included(trial.fit, set().union(*(ids for _t, ids in new)))
    penalty = SeedPreservation(ctx, trial, a.dropped).delta
    for _tag, ids in new:
        kept = {i for i in ids if i in included}
        if kept:
            penalty += subset_chi2(trial.fit, kept, station_floor=True, included_only=True)[0]
    cap = float(_chi2.ppf(cfg.new_opposition_penalty_conf, _STATE_DOF))
    if penalty <= cap:
        return None
    worst = None
    for tag, ids in new:
        t = warm_predictive(trial.fit, set(ids))
        if "p" in t and (worst is None or t["p"] < worst[1]["p"]):
            worst = (tag, t)
    if worst is None or worst[1]["p"] >= cfg.new_opposition_leaveout_p:
        return None
    tag, t = worst
    return (f"the new opposition is not vouched for by the seed: joining costs chi2 "
            f"{penalty:.1f} vs {cap:.1f} ({_STATE_DOF} dof), and {tag}, left out, is "
            f"predicted at p {t['p']:.4f} ({t['miss']:.1f}\" off)")


def _demote_to_possible(ctx, result, trial, obs_out, vetting, flag, why, head, log):
    """Make the find a reviewable 0.6: keep its fit and astrometry, flag
    ``vetting[flag]``, put ``head`` first among the warnings, log ``log``."""
    result.verdict = SearchVerdict.POSSIBLE
    result.fit = trial.fit
    result.combined_obs80 = display_obs80(ctx, obs_out)
    result.itf_obs80 = obs_out
    vetting["possible_unconfirmed"] = True
    vetting[flag] = True
    result.reason = f"possible linkage, kept for review: {why}"
    result.warnings.insert(0, head)
    vetting["warnings"] = list(result.warnings)
    ctx.log(log)
    return result


def _accepted(ctx, result, a, x, tracklets_all):
    """A consistent linkage on a submittable profile: LINKED, unless a rule
    makes it a reviewable 0.6 instead.

    Three rules demote a linkage that cleared every gate.  It keeps its fit,
    astrometry and warnings, is reported on every surface, and is never
    transmitted -- a human with the warnings in front of them decides:

    1. THE SEED FIT GOT MATERIALLY WORSE (``x.seed_degraded``), the
       measurement :func:`extension_degrades` refuses on, one threshold inside.
    2. A CONFIRMED 1.75 CARRYING ANY CAUTION.  The confirmation says "these
       lone single-night oppositions are no coincidence" on the profile alone;
       any caution is evidence against that.  Sole exception: a seed
       degradation the seed's own opposition, extended, already carries
       (:func:`_seed_opp_holds`).
    3. A NEW OPPOSITION A SINGLE-OPPOSITION SEED CANNOT VOUCH FOR
       (:func:`_new_opposition_unpredicted`): joining costs more than the orbit's
       own uncertainty allows AND one of its tracklets was not predicted by the
       rest.

    Then the document rules (:func:`_document_arc`)."""
    cfg = ctx.cfg
    trial = a.trial
    vetting = x.vetting
    result.warnings.extend(x.cautions)
    result.diagnostics.extend(x.diagnostics)
    result.vetting = vetting
    vetting["diagnostics"] = list(result.diagnostics)
    withdrawing = list(x.cautions)
    if (a.confirmed and x.seed_caution is not None and not x.seed_degraded
            and _seed_opp_holds(ctx, trial, a.obs_out)):
        withdrawing.remove(x.seed_caution)
    if x.seed_degraded or (a.confirmed and withdrawing):
        vetting["demoted"] = True
        if x.seed_degraded:
            flag = "seed_degraded"
            why = (f"the object's own observations fit materially worse under "
                   f"the linked orbit ({vetting['seed_rms_pre']:.3f}\" -> "
                   f"{vetting['seed_rms_post']:.3f}\")")    # per-axis
            head = ("NOT SUBMITTABLE: the object's own astrometry fits worse "
                    "under this linkage. That is what a contaminant looks "
                    "like, short of the outright veto -- so the linkage is "
                    "reported for review, never submitted as it stands. "
                    "Check the added tracklet(s) by hand.")
        else:
            flag = "confirm_withdrawn"
            why = ("this linkage was treated as three oppositions even "
                   "though two of them are single nights, and the warning(s) "
                   "withdraw that")
            head = ("NOT SUBMITTABLE: this linkage rests on two apparitions "
                    "that each have only a single night. It was accepted only "
                    "because the seed/pair confirmation judged it "
                    "non-coincidental -- and the warning(s) below withdraw "
                    "that judgement. Review them, and confirm with a further "
                    "night before submitting.")
        _demote_to_possible(ctx, result, trial, a.obs_out, vetting, flag, why,
                            head, f"linkage DEMOTED to possible (status 0.6, "
                                  f"not submittable): {why}")
        result.warnings.extend(x.held)
        vetting["warnings"] = list(result.warnings)
        return result
    unvouched = _new_opposition_unpredicted(ctx, a)
    if unvouched:
        vetting["demoted"] = True
        _ctx_tracer(ctx).cont(f"new opposition: {unvouched}  DEMOTE")
        head = ("NOT SUBMITTABLE: the object was seen in one opposition only, and "
                "the tracklets this linkage adds in another opposition are held by "
                "the joint fit rather than predicted by the object's own orbit -- "
                "the signature of a second body's detections. Reported for review, "
                "never submitted as it stands; confirm with another night.")
        _demote_to_possible(ctx, result, trial, a.obs_out, vetting,
                            "new_opposition_unvouched", unvouched, head,
                            f"linkage DEMOTED to possible (status 0.6, not "
                            f"submittable): {unvouched}")
        result.warnings.extend(x.held)
        vetting["warnings"] = list(result.warnings)
        return result
    # added only now, past the withdrawal, so they warn without demoting
    result.warnings.extend(x.held)
    _leaveout_cautions(ctx, result, vetting, a.obs_out, a.combined, x.magt,
                       a.confirmed)
    vetting["warnings"] = list(result.warnings)
    vetting["diagnostics"] = list(result.diagnostics)
    ctx.log(f"linkage accepted: used {trial.used}/{trial.count}, "
            f"rms {trial.rms:.3f}\", opps {a.profile}, spans {a.spans}"
            + (f" [WARN: {'; '.join(result.warnings)}]" if result.warnings else ""))
    # the published object fell out of the fit: an unrelated ITF-ITF link is
    # only transmitted when it is extremely clean
    natural_desigs, _, _ = collect_ids(a.combined)
    transmit = not (ctx.primary_desigs and not natural_desigs
                    and trial.rms > cfg.pure_itf_max_rms)
    if not transmit:
        result.reason = (f"pure ITF-ITF link with rms > "
                         f"{cfg.pure_itf_max_rms:g}\"")
    doc_combined, doc_out, refused = _document_arc(
        ctx, result, trial, a.combined, a.obs_out, vetting, x.pickups)
    if refused is not None:
        return refused
    return _build_and_submit(ctx, result, trial, a.obs_out, tracklets_all,
                             doc_combined, doc_out, transmit)


def _thin_possible(ctx, result, a, x=None):
    """A linkage on a refused three-thin (1.75) profile: two of its
    oppositions rest on one night each, which a short arc can thread by
    coincidence.  Kept as a reviewable 0.6, never submitted by the search.

    ``promotable_on_step``: the web UI may offer it for submission after a
    re-solve WITHOUT the observation set growing (``engine_api._pending_verdict``)
    only when its opposition profile ALONE refused it -- it is consistent
    (``x``, its extension check, is given) and carries no seed degradation
    that would demote an accepted linkage.  Every other possible keeps the
    growth rule."""
    promotable = (x is not None and not x.seed_degraded
                  and (a.seed is None or a.seed.unpaid is None))
    result.verdict = SearchVerdict.POSSIBLE
    result.reason = f"{a.profile} opps -- too fragile to submit, abandoning"
    result.warnings.append(
        "Possible linkage, not confirmed. Two of its three oppositions rest "
        "on a single night each, and a short arc can thread single-night "
        "tracklets by coincidence. What would confirm it: an arc extension of "
        "the existing opposition or at least one more tracklet from an "
        "opposition not yet in the arc, from archival images or a new "
        "observation. Use your scientific judgement before submitting! "
        + ("A full step or multi step from here will enable linkage "
           "download/submission." if promotable else
           # never promise a button the growth rule holds back
           "Something other than the opposition profile of the arc also "
           "refused this linkage, so re-solving it will not enable submission "
           "on its own -- the observation set has to grow first."))
    if x is not None:
        _keep(ctx, result, a, x)
        result.warnings.extend(x.held)
    else:
        result.fit = a.trial.fit
        result.combined_obs80 = display_obs80(ctx, a.obs_out)
        result.itf_obs80 = a.obs_out
    result.vetting = {**(x.vetting if x is not None else
                         {"overall_rms_post": round(a.trial.rms, 4)}),
                      "possible_unconfirmed": True,
                      "promotable_on_step": promotable,
                      "warnings": list(result.warnings)}
    ctx.log(f"possible-but-unconfirmed linkage kept for review "
            f"(opps {a.profile}, used {a.trial.used}/{a.trial.count}, "
            f"rms {a.trial.rms:.3f}\")")
    return result



def _distrusted(ctx, result, a, why, clauses=()):
    """A thin-profile find off a one-opposition seed that something refused
    (``why``): it DID link, so it is reported as distrusted, never as
    "nothing found".  A 1.75 stays a reviewable 0.6 that a re-solve cannot
    promote (:func:`_thin_possible`); a lone night is a bare 0.6 saying, in
    English, why it is not shown."""
    if a.profile.grade is OppProfileGrade.THREE_THIN:
        out = _thin_possible(ctx, result, a)
        out.vetting["refused_because"] = "; ".join(clauses) or why
        return out
    ctx.show_refusal = why
    result.verdict = SearchVerdict.POSSIBLE
    result.vetting = {"show_suppressed": True,
                      "overall_rms_post": round(a.trial.rms, 4),
                      "opps_pre": ctx.initial_opp_profile.legacy_code,
                      "opps_post": a.profile.legacy_code}
    result.reason = f"nothing shown: {_sentence(why)}"
    ctx.log(result.reason)
    return result


def _plain_refusal(a):
    """The clause that refused ``a``, as a sentence for a reader -- read after
    "nothing shown: ", so it does not punctuate itself."""
    fails = a.fails or [{}]
    seed = next((f for f in fails
                 if f.get("kind") in ("seed", "seed_missing", "veto")),
                None)
    misfit = [f for f in fails if f.get("kind") == "chi2"]
    loose = [f for f in fails if f.get("kind") == "retention"]
    if seed is not None and seed["kind"] == "veto":
        return (f"linking it degrades the existing fit: the object's own "
                f"observations go {seed['shown']}, the size of shift a wrong "
                f"tracklet forces on an orbit -- rejected as a likely contaminant")
    if seed is not None and seed["kind"] == "seed_missing":
        return (f"linking it would throw away {seed['n']} of the object's own "
                f"{seed['of']} observations -- a linkage that discards the "
                f"object's astrometry is not a linkage of that object")
    if misfit and len(misfit) < a.n_groups:
        return (f"the candidate night is made of several tracklets and one of "
                f"them ({misfit[0]['tag']}) does not fit its own uncertainties "
                f"-- a partly-wrong night is not an identification")
    if misfit:
        red = max(f["red"] for f in misfit)
        return (f"the candidate night lands near the orbit but not on it: its "
                f"residuals are {red:.1f}x its own quoted uncertainties, more "
                f"than measurement noise explains")
    if loose:
        f = loose[0]
        return (f"the candidate night does not hold together under the orbit: "
                f"{f['tag']} keeps only {f['kept']} of its {f['n']} "
                f"observations in the fit")
    if seed is not None:
        return ("adding the candidate night pulls the object's own "
                "observations off the orbit by more than their own scatter "
                "allows -- the usual signature of a wrong tracklet")
    if any(f.get("kind") == "night" for f in fails):
        return ("the linked night's residuals are coherently displaced, the "
                "signature of a wrong tracklet")
    return "it does not fit the object's orbit cleanly"


def _neo_orbit_waiver(cfg, trial):
    """May ``trial`` be accepted on an opposition profile the gate refused,
    because it is a NEO?  (What the waiver is FOR: :func:`_opp_profile_gate`.)

    q under ``neo_waiver_q`` AND a BOUND orbit (``e < neo_waiver_max_e``): a
    range-degenerate TNO arc can converge on a hyperbola whose perihelion is
    an artefact.  Needed for 2002 PS152; without it, a TNO's lone new night
    was accepted as a NEO link.  A NaN ``e`` fails the comparison, so an
    orbit we cannot check never skips the gate.
    """
    return anchor.is_neo(cfg, trial.q, trial.e)


def _opp_profile_clause(cfg, trial, opp_profile_ok, initial_opp_profile):
    """May a linkage be accepted on this opposition profile?  When the gate
    passes it, when the SEED already spans more than one opposition, or under
    the NEO waiver (:func:`_neo_orbit_waiver`).  The web's re-judge
    (:func:`opp_profile_acceptable`) and the search share this ONE reading."""
    return bool(opp_profile_ok or initial_opp_profile.grade is not OppProfileGrade.ONE
                or _neo_orbit_waiver(cfg, trial))


def opp_profile_acceptable(cfg, combined, trial, *, pub_arclength_days, pub_ids,
                     initial_opp_profile):
    """``(ok, why)`` -- THE night/opposition acceptance question, ctx-free.

    The search asks it via :meth:`Assessment.accepts_profile`; the web UI asks
    it again on the LIVE observation set after a step grows a possible
    (``engine_api._pending_verdict``), with seed values stashed at search
    time -- so a later investigation passes the gate the search applied.
    ``why`` names the opposition profile when it refuses.
    """
    opps, nights, opp_profile_ok = opp_profile_gate(cfg, combined, trial,
                                       pub_arclength_days=pub_arclength_days,
                                       pub_ids=pub_ids)
    if _opp_profile_clause(cfg, trial, opp_profile_ok, initial_opp_profile):
        return True, ""
    return False, (f"opposition profile not submittable: "
                   f"{opps:g} opposition(s), {nights} night(s)")


def _opp_profile_gate(ctx, trial, combined):
    """The ported opposition-profile acceptance test: returns
    ``(trial_opp_profile, curr_nights, opp_profile_ok)`` for the linkage's combined arc.

    ``opp_profile_ok`` refuses the 1.5 profile (one lone night in a new
    opposition), but a NEO is waived (:func:`_neo_orbit_waiver`): its 1.5
    link is a PROPOSAL for the archival community to check, right for an
    object worth attention.  A non-NEO 1.5 goes to the 0.6 lone-night path
    (:func:`_lone_night`).  Junk 1.5s are stopped by the per-tracklet quality
    gate (:func:`_accept_quality_ok`), not here.

    The waiver ACCEPTS but does not SUBMIT: the MPC night rule
    (:func:`itf.mpcrules.submittable_nights`, asked in :func:`_document_arc`)
    still turns a lone new night into a 0.6."""
    return opp_profile_gate(ctx.cfg, combined, trial,
                      pub_arclength_days=ctx.pub.arclength_days,
                      pub_ids=ctx.pub_ids)


def opp_profile_gate(cfg, combined, trial, *, pub_arclength_days, pub_ids):
    """The ctx-free body of :func:`_opp_profile_gate` (whose docstring is the
    reference): ``(trial_opp_profile, curr_nights, opp_profile_ok)`` for
    ``combined`` fitted as ``trial``, given the seed's arc length and observation ids."""
    trial_opp_profile = opp_profile(combined)
    # SPANS (cfg.span_days): the ported unit all clauses but one were
    # calibrated on -- two calendar nights 22 h apart are ONE span.
    curr_spans = count_spans(combined, cfg.span_days)
    # The two-opposition clause counts observer NIGHTS (half a day apart), not
    # spans, which merge the consecutive nights a second apparition often
    # arrives as.  Needed for 2013 SZ67; without it, a correct two-opposition
    # linkage was refused.
    cal_nights = count_nights(combined)
    arclength = trial.arclength_days
    pub_arclength = pub_arclength_days
    pub_in_output = any(obs_unique_id(l) in pub_ids
                        for l in combined.splitlines() if l.strip())
    grade = trial_opp_profile.grade
    opp_profile_ok = (grade is not OppProfileGrade.TWO_THIN and (
        curr_spans >= 8
        or grade is OppProfileGrade.MULTI
        or (trial.q < 1.6 and curr_spans >= 6)
        or (cal_nights >= 5 and grade is OppProfileGrade.TWO)
        or (((curr_spans >= 3 and arclength < 13) or curr_spans >= 4)
            and grade is OppProfileGrade.ONE)
        or (grade is OppProfileGrade.ONE and pub_in_output
            and (arclength < 22 or arclength < 2 * pub_arclength))
    ))
    return trial_opp_profile, curr_spans, opp_profile_ok


def _reduced_submittable(ctx, trial, reduced_combined, reduced_out):
    """``(ok, why_not)`` -- is what remains after a banned designation is
    dropped still a document the MPC will take?

    The MPC's criteria decide this (:func:`itf.mpcrules.check`), not
    :func:`_opp_profile_gate`: our belief in the identification was already
    settled on the FULL arc, and the ban removes a NAME, not evidence (the
    tracklet stays in the fit).  Needed for 2024 TF21; without it, a reduced
    linkage the MPC would accept was refused as a thin 1.75.

    The orbit-quality clause gets the FULL fit (no refit; the orbit barely
    moves), so its element sigmas are the linkage's, not the reduced arc's.
    """
    from . import mpcrules
    if not (reduced_out or "").strip():
        return False, "nothing linked is left once it is dropped"
    ok, why = mpcrules.check(reduced_combined, fit=trial.fit,
                             rms_arcsec=trial.rms)
    if ok:
        return True, ""
    return False, ("what is left would not be accepted as an identification: "
                   + "; ".join(why))


def _seed_opp_extension(ctx, obs_out):
    """Seed + only those linked ITF tracklets that fall in the SEED'S OWN
    opposition, fitted alone -- the object's discovery apparition EXTENDED.

    This is the part of a 1.75-opposition linkage that makes no
    cross-opposition claim, so it carries none of the combinatoric risk.
    :func:`_confirm_top_pair` judges the thin oppositions against its
    strength; :func:`_reduced_linkage` submits it alone when the 1.75 is
    refused.

    Returns ``(trial, combined, obs_same)``, or None when nothing splits off
    (no linked tracklet in the seed's opposition, or all of them are), when
    the subset is not a single opposition, or when its fit fails.
    """
    lines = [l for l in obs_out.splitlines() if l.strip()]
    seed_days = [obs_night_ordinal(l) for l in ctx.pub_obs80.splitlines()
                 if l.strip()]
    if not lines or not seed_days:
        return None
    lo, hi = min(seed_days), max(seed_days)
    same = [l for l in lines if lo - 275 <= obs_night_ordinal(l) <= hi + 275]
    if not same or len(same) == len(lines):
        return None
    obs_same = "".join(l + "\n" for l in same)
    combined = ctx.pub_obs80 + obs_same
    # the 275 d window is only a fast filter; opp_profile is the authority
    if opp_profile(combined).grade is not OppProfileGrade.ONE:
        return None
    trial = _fit(ctx, combined)
    return (trial, combined, obs_same) if trial.ok else None


def _seed_opp_holds(ctx, trial, obs_out):
    """Does the seed's own opposition, extended, fit under the full linkage
    within the same seed-degradation threshold :class:`SeedCheck` applies?

    The seed-degradation caution compares against the bare seed, so
    same-opposition tracklets alone can push a short, over-fit seed past the
    threshold and the caution then says nothing about the cross-opposition
    ones.  Needed for 2015 KK542; without it, a same-opposition tracklet alone
    withdrew the confirmation of its thin oppositions.

    False when there is no same-opposition subset to measure against."""
    same = _seed_opp_extension(ctx, obs_out)
    if same is None:
        return False
    tr_same, comb_same, _obs_same = same
    ids = obs_ids(comb_same)
    pre, n_pre = subset_rms(tr_same.fit, ids)
    post, n_post = subset_rms(trial.fit, ids)
    if not n_pre or not n_post:
        return False
    cfg = ctx.cfg
    lvl = max(pre * cfg.warn_rms_factor, pre + cfg.veto_rms_floor_arcsec)
    held = post < lvl
    if _ctx_tracer(ctx).enabled:
        _ctx_tracer(ctx).head(
            f"seed degradation vs the seed's own opposition extended "
            f"({n_pre} obs): {per_axis(pre):.3f}\" -> {per_axis(post):.3f}\" "
            f"(warn at {per_axis(lvl):.3f}\") -- "
            + ("the cross-opposition tracklets add none; the caution does not "
               "withdraw the confirmation" if held else
               "the cross-opposition tracklets degrade it too"))
    return held


def _pair_confirms(cfg, trial, combined, id_sets, ml_ps):
    """The confirmed-pair evidence: are both tracklets strongly ML-rated, and
    does every observation they were swept with sit in the final linkage
    inside ``confirm_max_resid_arcsec``?

    Returns ``(confirmed, worst_resid)``.  A missing ``ml_p`` (the scan path
    scores nothing) refuses; scan-path searches rely on the seed-strength
    paths.
    """
    if not all(p is not None and p >= cfg.confirm_min_ml_p for p in ml_ps):
        return False, None
    have = obs_ids(combined)
    resid = residuals_by_obs_id(trial.fit)
    ids = set().union(*id_sets)
    worst = max((resid.get(i, float("inf")) for i in ids), default=float("inf"))
    ok = all(i in have and resid.get(i) is not None
             and resid[i] <= cfg.confirm_max_resid_arcsec for i in ids)
    return ok, worst


def _linked_ids(ctx, combined):
    """The ITF observation ids in ``combined`` -- everything that is not the
    object's own seed astrometry."""
    return obs_ids(combined) - getattr(ctx, "pub_ids", set())


def _mag_tracklet_is_own(ctx, trial, magt):
    """Does the photometric outlier :func:`mag_outlier_tracklet` flagged belong
    to the object's OWN published arc rather than to a linked ITF tracklet?

    Re-derives THE SAME (station, night) group with ``_mag_tracklets`` on the
    same rows, matched by station and first epoch, and asks whether ANY of its
    observations is a seed line.  Do not re-group by a date label: that is not
    the detector's night rule and can split a night.  A mixed group reads as
    "own", so the caution never wrongly accuses the ITF."""
    if magt.get("jd") is None:
        return False
    from .fitter import _mag_tracklets
    try:
        rows = trial.fit.magnitude_residuals()
    except Exception:
        return False
    pub_ids = getattr(ctx, "pub_ids", set())
    own_keys = {(o.get("obscode"), o["jd_utc"]) for o in trial.fit._obs
                if o.get("obs80_line")
                and obs_unique_id(o["obs80_line"]) in pub_ids}
    for station, group in _mag_tracklets(rows):
        if station != magt.get("station") or group[0]["jd_utc"] != magt["jd"]:
            continue
        return any((r.get("obscode"), r["jd_utc"]) in own_keys for r in group)
    return False


def _confirm_top_pair(ctx, trial, combined, trial_meta, depth, same_opp=None):
    """The three-thin upgrade (:func:`_confirm_top_pair_impl`), recording each
    attempt's evidence and verdict when :mod:`itf.gatelog` is on."""
    confirmed = _confirm_top_pair_impl(ctx, trial, combined, trial_meta, depth,
                                       same_opp)
    if gatelog.enabled():
        gatelog.record("confirm_175", **gatestats.confirm_row(
            ctx, trial, combined, trial_meta, depth, same_opp, confirmed))
    return confirmed


def _confirm_top_pair_impl(ctx, trial, combined, trial_meta, depth,
                           same_opp=None):
    """Confirmed upgrade of the thin 1.75 profile.

    Two single-night extension oppositions read as "1.75 opps" -- possibly a
    coincidence when the recursion picked one tracklet, bent the orbit, then
    found another in a crowded sky.  The profile is CONFIRMED (the caller
    treats it as 3 oppositions) when any of these holds:

    1. **Seed strength**: the initial arc is longer than
       ``confirm_seed_arc_days`` OR its U is at most ``confirm_seed_U_max``;
       its corridor is then tight enough that independently matched
       single-night oppositions are no coincidence.
    2. **Same-opposition strength**: path 1's thresholds applied to the seed
       plus the tracklets linked inside its OWN opposition (``same_opp``, from
       :func:`_seed_opp_extension`), since ``ctx.pub`` understates a short
       discovery arc's opposition.  Not circular: no thin-opposition obs are
       in that orbit.  Needed for 2006 QR56; without it, a correct
       3-opposition linkage was refused as thin.
    3. **The seed's own top pair**: the linkage is exactly the depth-0 sweep's
       top two (``ctx.depth0_top2``), both strongly rated and every swept obs
       still tight in the final orbit (none clipped).  The SEED's ranking named
       both before either was linked, however the recursion assembled them.
       Anything more or less than those two is a different claim, left to the
       other paths and 0.6 review.

    ``trial_meta`` and ``depth`` are recorded by the study log only."""
    cfg = ctx.cfg
    trc = _ctx_tracer(ctx)
    if (ctx.pub.arclength_days > cfg.confirm_seed_arc_days
            or ctx.pub.U <= cfg.confirm_seed_U_max):
        if trc.enabled:
            trc.story(f"confirmed by seed strength: initial arc "
                f"{ctx.pub.arclength_days:.1f}d (need > "
                f"{cfg.confirm_seed_arc_days:g}d) / U {ctx.pub.U:.2f} (need "
                f"<= {cfg.confirm_seed_U_max:g}) -> 1.75 opps -> 3")
        ctx.log("confirmed by seed strength: the well-constrained initial "
                "arc makes the 1.75-opposition profile submittable -- "
                "treating as 3 oppositions")
        return True
    if same_opp is not None:
        so = same_opp[0]
        if (so.arclength_days > cfg.confirm_seed_arc_days
                or so.U <= cfg.confirm_seed_U_max):
            if trc.enabled:
                trc.story(f"confirmed by same-opposition strength: seed + its "
                    f"own-opposition tracklets fit {so.used}/{so.count} at "
                    f"{so.rms:.3f}\", arc {so.arclength_days:.1f}d (need > "
                    f"{cfg.confirm_seed_arc_days:g}d) / U {so.U:.2f} (need <= "
                    f"{cfg.confirm_seed_U_max:g}) -> 1.75 opps -> 3")
            ctx.log(f"confirmed by same-opposition strength: the object's own "
                    f"opposition, extended to {so.arclength_days:.1f}d "
                    f"(U {so.U:.2f}), constrains the single-night oppositions "
                    f"well enough to submit -- treating as 3 oppositions")
            return True
    top2 = getattr(ctx, "depth0_top2", None)
    if not (top2 and len(top2) == 2):
        return False
    id_sets = [t["ids"] for t in top2]
    ml_ps = [t["ml_p"] for t in top2]
    if _linked_ids(ctx, combined) != id_sets[0] | id_sets[1]:
        return False
    confirmed, worst = _pair_confirms(cfg, trial, combined, id_sets, ml_ps)
    if trc.enabled:
        trc.story(f"top-pair check: the linkage IS the depth-0 sweep's "
            f"top two ({' + '.join(t['tag'] for t in top2)}, P(match) "
            f"{', '.join('n/a' if p is None else f'{p:.3f}' for p in ml_ps)}"
            f"), worst linked |O-C| "
            + (f"{worst:.3f}\"" if worst is not None else "n/a")
            + f" (cap {cfg.confirm_max_resid_arcsec}\") -> "
            + ("CONFIRMED: 1.75 opps -> 3" if confirmed else "not confirmed"))
    if confirmed:
        ctx.log("confirmed pair: the linkage is exactly the two "
                "tracklets the seed orbit alone ranked best, both "
                "tight -- treating 1.75 opps as 3")
    return confirmed

def extension_degrades(cfg, pre_rms, seed_post_rms, seed_arc_days=None,
                       seed_post_sigma=None):
    """Is a linked ITF tracklet a likely contaminant?  True when it makes the
    object's OWN pre-search observations fit dramatically worse than they did
    before the link.

    BOTH arguments must be in :func:`fitter.subset_rms` currency -- callers get
    ``pre_rms`` from :func:`_seed_pre_rms`, never from ``pub.rms`` (a per-axis
    rms, smaller by sqrt(2), which would silently scale every factor below).

    The seed obs ARE the object, so whether they still fit is the faithful
    test of "did the orbit get pulled off the object?".  It is only HALF the
    test: an orbit that bends far enough to REJECT the seed obs it no longer
    fits scores an improvement here; the other half is
    :func:`_seed_clip_ledger`.  Do NOT also require the OVERALL rms to worsen:
    that lets the intruder's own fit vote on whether it is an intruder, and a
    tight candidate dragging the object's obs off is the clearest contaminant
    signature (2026 FU24).

    A RELATIVE test against the seed's own achieved rms, deliberately not a
    reported-sigma one: a loosely self-reported contaminant would score SAFER
    in sigma than a genuine precovery with tight errors.  An overfit short
    seed settling onto its true orbit widens modestly; a contaminant the orbit
    bends to swallow blows it up.  ``veto_rms_factor`` (3.0) sits between the
    two; the additive floor keeps a near-zero pre-rms from making the
    threshold trivially tight.  The null is NOT 1.0: the pre-link rms is
    in-sample, so correct links widen it, often past 1.5 -- hence
    ``warn_rms_factor`` 2.0.

    A degenerate seed widens most, so an arc of at most
    ``veto_short_arc_days`` is judged by ``veto_rms_factor_short_arc``.
    Needed for 2020 KX6; without it, a real self-recovery was banned.
    ``seed_arc_days`` None keeps the strict factor.  Neither factor lets FU24
    through."""
    factor = cfg.veto_rms_factor
    if (seed_arc_days is not None
            and seed_arc_days <= cfg.veto_short_arc_days):
        factor = max(factor, cfg.veto_rms_factor_short_arc)
    lvl = max(pre_rms * factor, pre_rms + cfg.veto_rms_floor_arcsec)
    if seed_post_rms < lvl:
        return False
    # The factor tripped -- but a short seed can fit itself tightly by luck, so
    # there is a statistical escape: if the object's own obs still fit their
    # uncertainties, the link did not pull the orbit off it.  Needed for 2015
    # FO479; without it, a correct linkage was refused for a lucky seed fit.
    # ``seed_post_sigma`` is normalised by the STATION floor, never the
    # self-reported sigma, or an under-reporting station lets a contaminant
    # through (2026 FU24).
    if (seed_post_sigma is not None
            and cfg.veto_within_sigma > 0
            and seed_post_sigma <= cfg.veto_within_sigma):
        return False
    return True


def _linked_tracklet_count(obs_out):
    """How many distinct ITF tracklets the linkage brought in, counted per
    (trksub, station, night) as the submission document does
    (:func:`linkage.collect_ids`)."""
    _desigs, triples, _neocp = collect_ids(obs_out)
    return len(triples)


def _seed_clip_ledger(ctx, trial):
    """What did this linkage cost the SEED?  Returns a dict describing every
    published observation the accepted orbit no longer includes, grouped into
    the seed's own observing nights.

    A link that quietly discards the object's own astrometry is the failure
    :func:`extension_degrades` cannot see.  But refusing every clip would leave
    an object with one bad published night permanently unrecoverable, and a
    real recovery is often how a bad night is PROVEN bad.  So the clip is
    PRICED (:func:`_clip_evidence_ok`) and REPORTED, not forbidden.

    Keys: ``shed_obs`` / ``kept_obs``, ``n_nights``, ``lost_nights`` (nothing
    left: date, stations, obs count, worst |O-C| under the final orbit),
    ``partial_nights`` (lost some) and ``kept_nights``.  The unit is the SPAN
    (``opp_profile.spans_of``, the gate's 36-hour unit); the keys say "night"
    because the batch CSV and JSONL records do."""
    seed_lines = [l for l in ctx.pub_obs80.splitlines() if l.strip()]
    included, resid = subset_included(trial.fit, ctx.pub_ids)
    lost, partial = [], []
    n_clipped = n_kept = 0
    # the gate's spans, so "span 4" here is span 4 of its count
    nights = spans_of(seed_lines)
    for i, night in enumerate(nights):
        gone = [l for l in night if obs_unique_id(l) not in included]
        n_clipped += len(gone)
        n_kept += len(night) - len(gone)
        if not gone:
            continue
        worst = max((resid.get(obs_unique_id(l), 0.0) for l in gone), default=0.0)
        entry = {
            "night": i + 1,                       # 1-based: "the 4th night"
            "date": night[0][15:25].strip(),
            "stations": sorted({l[77:80] for l in gone}),
            "n_obs": len(night),
            "n_shed": len(gone),
            # worst clipped |O-C| under the FINAL orbit: the evidence it is bad
            # (an ob absent from the fit entirely reads 0.0)
            "worst_resid_arcsec": round(worst, 3),
        }
        (lost if len(gone) == len(night) else partial).append(entry)
    return {
        "n_nights": len(nights),
        "kept_nights": len(nights) - len(lost),
        "lost_nights": lost,
        "partial_nights": partial,
        "shed_obs": n_clipped,
        "kept_obs": n_kept,
    }


def _clip_evidence_ok(cfg, ledger, n_linked_tracklets):
    """May this linkage be accepted given what it clipped from the seed?  Returns
    ``(ok, reason)``.

    1. **A whole night may go, but it must be BOUGHT**: "that published night
       is bad" is earned only by at least ``clip_span_price_tracklets`` (3)
       newly linked tracklets per night dropped, agreeing on an orbit the
       dropped night contradicts.
    2. **The seed must still BE a seed**: at most ``clip_spans_max`` (1) span
       may go and at least ``clip_keep_spans_min`` (2) must remain; below that
       the identification rests on the ITF tracklets alone, a much weaker
       claim.
    3. **Scattered obs are ordinary**: obs lost from nights that SURVIVE are
       routine rejection, left to the overall retention gate.

    Nothing clipped -> always ok."""
    lost = ledger["lost_nights"]
    if not lost:
        return True, ""
    n_lost = len(lost)
    if n_lost > cfg.clip_spans_max:
        return False, (f"the link drops {n_lost} whole spans of the object's "
                       f"own arc (at most {cfg.clip_spans_max} may go)")
    if ledger["kept_nights"] < cfg.clip_keep_spans_min:
        return False, (f"the link leaves only {ledger['kept_nights']} span(s) "
                       f"of the object's own arc (need at least "
                       f"{cfg.clip_keep_spans_min}, or the identification "
                       f"rests on the ITF tracklets alone)")
    price = cfg.clip_span_price_tracklets * n_lost
    if n_linked_tracklets < price:
        return False, (f"the link drops span {lost[0]['night']} "
                       f"({lost[0]['date']}) of the object's own arc but brings "
                       f"only {n_linked_tracklets} new tracklet(s) -- {price} "
                       f"are needed to call a published span bad")
    return True, ""


def _clip_note(ledger, n_linked_tracklets, trial_opp_profile):
    """The human sentence for an accepted linkage that dropped a seed span:
    what was dropped, and what pays for it."""
    n = ledger["lost_nights"][0]
    which = ", ".join(f"span {e['night']} ({e['date']}, "
                      f"{'/'.join(e['stations'])}, {e['n_shed']} obs)"
                      for e in ledger["lost_nights"])
    return (f"To find this object we had to DROP {which} of its published arc: "
            f"under the linked orbit that span misfits by {n['worst_resid_arcsec']}\", "
            f"and the remaining {ledger['kept_nights']}-span seed plus "
            f"{n_linked_tracklets} newly linked tracklet(s) across {trial_opp_profile} "
            f"oppositions all agree without it. Treat the dropped span as bad "
            f"astrometry -- it is not part of the fit.")


def _night_label(jd_utc):
    """``YYYY-MM-DD``: the UT date of an observation, as its OBS80 columns
    16-25 and ``ident_check.coherent_offset_misfit`` give it.  A NIGHT is
    labelled by its FIRST observation's date.

    Do not use ``round(jd - 0.5)``: a noon-UT boundary splits a Pacific night
    under two dates, one matching no OBS80 line, and the whose-night string
    comparison then silently misses it.
    """
    from datetime import datetime, timedelta
    return str((datetime(1858, 11, 17)
                + timedelta(days=math.floor(jd_utc - 2400000.5))).date())


def _own_nights_as_ident(ctx, fit):
    """Labels of the included nights holding at least one of the object's OWN
    observations, grouped and dated EXACTLY as
    ``ident_check.coherent_offset_misfit`` does (gap ``NIGHT_GAP_DAYS``, named
    by the first observation), so the labels match the detector's sentence.
    Per-observation dates would disagree for a night straddling midnight UT."""
    from pyroorb import ident_check
    m = list(fit.is_included())
    idx = sorted((i for i in range(len(fit._obs)) if m[i]),
                 key=lambda i: fit._obs[i]["jd_utc"])
    times = [fit._obs[i]["jd_utc"] for i in idx]
    own = set()
    for ni in ident_check._split(times, ident_check.NIGHT_GAP_DAYS):
        for k in ni:
            line = fit._obs[idx[k]].get("obs80_line")
            if line and obs_unique_id(line) in ctx.pub_ids:
                own.add(_night_label(times[ni[0]]))
                break
    return own


def _seed_degradation_one_night(ctx, trial, warn_lvl):
    """The description half of :func:`_seed_offending_night`, or None."""
    night = _seed_offending_night(ctx, trial, warn_lvl)
    return night[0] if night else None


def _seed_offending_night(ctx, trial, warn_lvl):
    """Is the seed's degradation the fault of ONE night of its own astrometry?

    Returns ``(description, obs_ids)`` -- ``obs_ids`` every seed observation
    of that night, the ones the fit already rejected included -- when a single
    seed night carries a coherent offset AND the rest of the seed fits under
    ``warn_lvl``: one bad night is fighting the better orbit.  None
    otherwise.  Needed for 2011 QH108 and 2022 WG18.
    """
    fit = trial.fit
    try:
        rows = fit.residuals()
        lines = [o.get("obs80_line") for o in fit._obs]
        inc = list(fit.is_included())
        jds = [o["jd_utc"] for o in fit._obs]
    except Exception:
        return None
    own = sorted(((jd, (r["d_ra_arcsec"], r["d_dec_arcsec"]))
                  for line, jd, r, ok in zip(lines, jds, rows, inc)
                  if ok and line and obs_unique_id(line) in ctx.pub_ids),
                 key=lambda t: t[0])
    # Nights by orbit_common.night_index (anchored on each night's first ob),
    # not a noon-UT boundary, which would split a Pacific night and could blame
    # half a night or lift a one-night seed over the ``len(by) < 2`` bar.
    by, first = {}, {}
    for n, (jd, d) in zip(night_index([jd for jd, _ in own]), own):
        by.setdefault(n, []).append(d)
        first.setdefault(n, jd)
    if len(by) < 2:
        return None                      # one night IS the seed; nothing to blame
    worst, worst_off = None, 0.0
    for night, v in by.items():
        arr = np.asarray(v, float)
        off = math.hypot(arr[:, 0].mean(), arr[:, 1].mean())
        if off > worst_off:
            worst, worst_off = night, off
    rest = [x for n, v in by.items() if n != worst for x in v]
    if not rest:
        return None
    rest = np.asarray(rest, float)
    # per-observation, to match warn_lvl; shown per-axis below
    rest_rms = math.sqrt(float((rest ** 2).sum(axis=1).mean()))
    if rest_rms >= warn_lvl:
        return None                      # the rest are degraded too: not one night
    arr = np.asarray(by[worst], float)
    mean_ra, mean_dec = arr[:, 0].mean(), arr[:, 1].mean()
    scatter = math.sqrt(float(((arr - arr.mean(axis=0)) ** 2).sum(axis=1).mean()))
    if worst_off < 2.0 * max(scatter, 0.05):
        return None                      # scattered, not coherently offset
    day = _night_label(first[worst])
    t_lo = min(jd for n, (jd, _d) in zip(night_index([jd for jd, _ in own]), own)
               if n == worst) - 0.3
    t_hi = max(jd for n, (jd, _d) in zip(night_index([jd for jd, _ in own]), own)
               if n == worst) + 0.3
    ids = frozenset(obs_unique_id(line) for line, jd in zip(lines, jds)
                    if line and obs_unique_id(line) in ctx.pub_ids
                    and t_lo <= jd <= t_hi)
    return ((f"it is ONE night of the object's own arc that misfits: {day} "
             f"({len(arr)} obs) sits {worst_off:.2f}\" off coherently "
             f"(dRA {mean_ra:+.2f}\", dDec {mean_dec:+.2f}\"), and without it "
             f"the rest of the seed fits at {per_axis(rest_rms):.3f}\" -- "
             f"bad astrometry in the seed, not a contaminant in the ITF"), ids)


def _leaveout_cautions(ctx, result, vetting, obs_out, combined, magt, promoted_175):
    """Move the in-scope cautions the leave-out orbit vouches for from
    ``result.warnings`` to ``result.diagnostics`` (:mod:`itf.leaveout`).

    Does nothing on a linkage the three-thin (1.75 -> 3) confirmation made
    submittable: that withdrawal is never softened."""
    cfg = ctx.cfg
    if promoted_175 or not getattr(cfg, "leaveout_confirm", False):
        return
    if not result.warnings and not result.diagnostics:
        return
    from . import leaveout
    result.warnings, result.diagnostics, record = leaveout.confirm_cautions(
        result.warnings, result.diagnostics, obs_out, combined, ctx.obscodes,
        registry=_ctx_registry(ctx), mag_tracklet=magt,
        max_sec=cfg.trial_fit_max_sec or None, log=ctx.log)
    if record:
        vetting["leaveout"] = [{k: v for k, v in r.items() if k != "text"} for r in record]
    vetting["diagnostics"] = list(result.diagnostics)


def _document_arc(ctx, result, trial, combined, obs_out, vetting, pickups):
    """The arc the submission document may carry, after the three rules that
    can make an accepted linkage unsubmittable as it stands (a designation that
    is itself a mislinkage, a comet or interstellar id, the MPC night rule).
    Returns ``(doc_combined, doc_out, None)``, or ``(None, None, demoted)``
    with ``result`` demoted to a reviewable possible."""
    # A designation the partial-pickup check condemned is never claimed: the
    # document drops it, while the fit and the UI keep it (it is real
    # astrometry of this object).  If what is left is not a linkage in its own
    # right, the find is reported for review instead.
    doc_combined, doc_out = combined, obs_out
    banned = pickup.banned_designations(pickups)
    if banned:
        doc_combined = pickup.drop_designations(combined, banned)
        doc_out = pickup.drop_designations(obs_out, banned)
        ok_reduced, why_reduced = _reduced_submittable(ctx, trial,
                                                       doc_combined, doc_out)
        vetting["linkage_dropped_designations"] = sorted(banned)
        vetting["warnings"] = list(result.warnings)
        if not ok_reduced:
            return None, None, _demote_to_possible(
                ctx, result, trial, obs_out, vetting, "not_submittable_partial_pickup",
                f"it rests on {', '.join(sorted(banned))}, which cannot be "
                f"claimed as a whole designation, and {why_reduced}",
                "NOT SUBMITTABLE: this identification cannot be made without "
                "claiming a designation that is itself a mislinkage (see the "
                "caution below), and " + why_reduced + ". The linkage is "
                "reported for review -- take the whole thing to the MPC as a "
                "Jira ticket if it stands up.",
                f"linkage NOT submittable (status 0.6): "
                f"{', '.join(sorted(banned))} may not be claimed and {why_reduced}")
        ctx.log(f"submission document drops {', '.join(sorted(banned))} "
                f"(partial pickup of a designated object); the fit keeps it")
    # The MPC's identification pipeline takes minor-planet designations only
    # (see ``comet_refusal``).  Checked here as well as in build_linkage so the
    # refusal says what it is rather than a generic "no linkage".
    comet_why = comet_refusal(doc_combined, ctx.primary_desigs)
    if comet_why:
        return None, None, _demote_to_possible(ctx, result, trial, obs_out, vetting,
                                   "not_submittable_comet", comet_why,
                                   "NOT SUBMITTABLE: " + comet_why,
                                   f"linkage NOT submittable (status 0.6): {comet_why}")
    # ...and the MPC's own night rule (``mpcrules.submittable_nights``),
    # whichever gate or waiver accepted it, judged on the arc the document
    # carries.  Needed for K26E25U; without it, two tracklets hours apart were
    # offered as two nights.
    nights_ok, nights_why = mpcrules.submittable_nights(doc_combined)
    if not nights_ok:
        return None, None, _demote_to_possible(ctx, result, trial, obs_out, vetting,
                                   "not_submittable_mpc_nights", nights_why,
                                   "NOT SUBMITTABLE: " + nights_why
                                   + ". Find another night in that apparition "
                                   "to make it submittable.",
                                   f"linkage NOT submittable (status 0.6): {nights_why}")
    # ...and our own floor: one apparition well observed in its own right
    # (``anchor.anchor_problem``: 3+ tracklets; a non-NEO also 3+ nights).  An
    # arc of thin apparitions pins no single orbit (2013 EZ43, a mislinkage of
    # 1+2+1 nights), and the MPC rule above passes one at 5+ apparitions or on
    # a single one.
    anchor_why = anchor.anchor_problem(
        doc_combined, neo=anchor.is_neo(ctx.cfg, getattr(trial, "q", None),
                                        getattr(trial, "e", None)))
    if anchor_why:
        return None, None, _demote_to_possible(ctx, result, trial, obs_out, vetting,
                                   "not_submittable_no_anchor", anchor_why,
                                   "NOT SUBMITTABLE: " + anchor_why
                                   + ". Find more of one apparition to make it "
                                   "submittable.",
                                   f"linkage NOT submittable (status 0.6): {anchor_why}")

    return doc_combined, doc_out, None


def _build_and_submit(ctx, result, trial, obs_out, tracklets_all,
                      doc_combined, doc_out, transmit):
    """Build the identification document for the accepted arc, fill
    ``result`` as LINKED, and hand the document to :func:`submit_linkage`
    (dry-run unless enabled twice) when ``transmit``.  No document, no
    linkage: a refusal leaves ``result`` unlinked with the reason."""
    cfg = ctx.cfg
    # Name each tracklet by the database's trksub, not columns 6-12 of its line
    # (see linkage.db_trksubs).  The registry covers tracklets linked at a
    # shallower level, which are not among this level's rows.
    out_ids = obs_ids(obs_out)
    _reg = _ctx_registry(ctx)
    trksubs = _reg.trksubs(out_ids) if _reg is not None else {}
    trksubs.update((k, v) for k, v in db_trksubs(
        r for t in tracklets_all for r in t["rows"]).items() if k in out_ids)
    respelt = sorted({(line_identifier(l), trksubs[obs_unique_id(l)])
                      for l in obs_out.splitlines()
                      if l.strip() and obs_unique_id(l) in trksubs
                      and line_identifier(l) != trksubs[obs_unique_id(l)]})
    if respelt:
        ctx.log("trksub from the database, not the 80-column line: "
                + ", ".join(f"{a} -> {b}" for a, b in respelt))
    doc, err = build_linkage(doc_combined, cfg,
                             primary_desigs=ctx.primary_desigs,
                             itf_obs_out=doc_out,
                             co_credit=ctx.co_credit,
                             fit=trial.fit,
                             trksubs=trksubs)
    if err is not None:
        # E.g. fewer than two distinct ids: reporting that as LINKED would show
        # the seed's own tracklets back to the user as an ITF hit.
        result.reason = f"no linkage ({err})"
        return result

    result.verdict = SearchVerdict.LINKED
    result.fit = trial.fit
    result.linkage = doc
    result.combined_obs80 = display_obs80(ctx, obs_out)
    result.itf_obs80 = obs_out
    result.itf_trksubs = trksubs
    ctx.telemetry["obs_count"] = trial.used - ctx.pub.used
    try:
        ctx.telemetry["U_diff"] = ctx.pub.U - trial.U
    except Exception:
        ctx.telemetry["U_diff"] = 0
    if doc is not None and transmit:
        # (trksub, stn, night) -> trkid, so the SQL ledger gets joinable keys
        trkid_map = {}
        for t in tracklets_all:
            for r in t["rows"]:
                if r.get("trkid"):
                    trkid_map[(r["trksub"], r["stn"],
                               r["obs80"][15:25])] = r["trkid"]
        held = getattr(ctx, "held_transmits", None)
        if held is not None:
            held.append((result, doc, trkid_map))
        else:
            _transmit(ctx, result, doc, trkid_map)
    return result


def _transmit(ctx, result, doc, trkid_map):
    """Hand ``doc`` to :func:`submit_linkage` (dry-run unless enabled twice)."""
    result.submitted, note, _mpc_msg = submit_linkage(doc, ctx.cfg,
                                                   do_submit=ctx.do_submit,
                                                   log=ctx.log,
                                                   trkid_map=trkid_map)
    result.reason = result.reason or note


# gravity-only orbital state dimension -- the dof of the seed's Delta-chi2
# confidence region in the one-night-show seed-preservation test
_STATE_DOF = 6


def seed_preserve_cap(conf, frac, pre_chi2):
    """How much the seed's own chi2 may rise when a linkage pulls the orbit.

    Two terms, because "the seed is not degraded" is two questions:

    * ``chi2.ppf(conf, _STATE_DOF)`` -- the seed's LINEAR confidence region:
      a correct link displaces the orbit within the seed's own 6-D parameter
      uncertainty.  Right for a small seed.

    * ``frac * pre_chi2`` -- a FRACTIONAL allowance on the seed's own chi2.
      Delta is a SUM over every seed observation, so it grows with their
      number and with any systematic the seed fit absorbed; chi2_6 grows with
      neither.  And a single-apparition seed's linear covariance
      under-predicts its multi-year extrapolation error.  Needed for 1997
      QK1; without it, a correct four-apparition linkage was refused merely
      because the seed had many observations.

    A linkage passes if it clears the LOOSER of the two, which makes the test
    invariant to seed size.  ``frac`` 0.25 is ~12% growth in the seed's rms;
    a wrong link is orders of magnitude past it.
    """
    from scipy.stats import chi2 as _chi2
    return float(_chi2.ppf(conf, _STATE_DOF)) + frac * max(pre_chi2, 0.0)


def _scored_field(tracklets_all):
    """The ML-scored candidates, best P(match) first."""
    return sorted((t for t in (tracklets_all or ()) if t.get("ml_p") is not None),
                  key=lambda t: -t["ml_p"])


def _rival_fit(ctx, rv, seed_lvl, seed_keep, name=""):
    """Fit rival ``rv`` against the BARE SEED -- an alternative to the winner,
    not an addition to it -- and judge it by the trial queue's own take
    criterion, under the same force model as every other fit in the search.
    Returns ``(trial, rejection)``; ``rejection`` is None when the rival links
    on its own."""
    cand = rv["astrometry"]
    tr = _fit(ctx, ctx.pub_obs80 + cand)
    if not tr.ok:
        return tr, (f"fit FAILED ({tr.error})", None, "fit failed")
    tr, _cand = _prune_trial(ctx, tr, ctx.pub_obs80, cand, name)
    return tr, _trial_rejection(ctx, tr, ctx.pub.used, False, seed_lvl, seed_keep)


def _exhaustion_verdict(ctx, trial_meta, tracklets_all, depth):
    """MEASURE whether the taken tracklet is the only one that fits.

    Walk the ML-ranked field (likely + maybe together) for the first candidate
    whose P(match) falls below ``show_exhaust_frac`` x the WINNER's P -- the
    cliff -- looking no further than ``show_exhaust_max_rank``.  Every rival
    above the cliff must be ruled out: already rejected by the trial queue, or
    fitted here by :func:`_rival_fit`.  A field that ends inside the rank
    bound needs no cliff: every member is a rival.

    Returns ``(sole_standout, info)``.  NOT a sole standout when the scan hits
    the rank bound with no cliff (comparable candidates may lie past it), when
    a rival links, or when the budget runs out mid-test: an unfinished
    exhaustion is not evidence of exclusivity."""
    cfg = ctx.cfg
    trc = _ctx_tracer(ctx)
    taken = trial_meta.get("tracklets") or []
    # Everything the trial TOOK is part of the winner, never a rival: a second
    # tracklet on the same night would otherwise be a "rival that also links"
    # (2024 QF7).
    taken_ids = {id(t) for t in taken}
    winner = (taken or [None])[0]
    p_winner = (winner or {}).get("ml_p")
    field = _scored_field(tracklets_all)
    if winner is None or p_winner is None or not field:
        return False, {"exhausted": False, "why": "no ML scores to rank"}

    bar = p_winner * cfg.show_exhaust_frac
    bound = min(len(field), cfg.show_exhaust_max_rank)
    cliff = next((j for j in range(bound) if field[j]["ml_p"] < bar), None)
    if cliff is None and bound < len(field):
        # Everything up to the rank bound is above the bar: exclusivity is not
        # claimable.
        if trc.enabled:
            trc(f"[depth {depth}] exhaustion: no candidate falls below "
                f"{cfg.show_exhaust_frac:.2f} x the winner's P {p_winner:.3f} "
                f"(bar {bar:.3f}) within rank {bound} -- the field is too dense "
                f"to claim exclusivity, multiplicity banner")
        return False, {"exhausted": False, "why": "many similarly-ranked candidate tracklets", "scanned": bound}
    if cliff is None:
        # The field ended inside the rank bound: every member is a rival.
        cliff = len(field)

    rivals = [t for t in field[:cliff] if id(t) not in taken_ids]
    # A sharp cliff can leave NOTHING above it, and exclusivity on zero fits
    # rests on the same ML scores that picked the winner -- the thing this test
    # exists to be independent of.  So reach past the cliff for at least
    # cfg.show_exhaust_min_rivals candidates.  Needed for 2024 PQ7; without it,
    # the unfitted candidate just past the cliff was the real tracklet.
    if len(rivals) < cfg.show_exhaust_min_rivals:
        n_taken_in_field = sum(1 for t in field if id(t) in taken_ids)
        want = min(cfg.show_exhaust_min_rivals, cfg.show_exhaust_max_rank,
                   len(field) - n_taken_in_field)
        seen_rivals = {id(t) for t in rivals}
        for t in field:
            if len(rivals) >= want:
                break
            if id(t) not in taken_ids and id(t) not in seen_rivals:
                rivals.append(t)
                seen_rivals.add(id(t))
    already = {id(t) for t in (trial_meta.get("failed_tracklets") or [])}
    n_pre = sum(1 for t in rivals if id(t) in already)
    if trc.enabled:
        edge = (f"cliff at rank {cliff + 1} "
                f"(P {field[cliff]['ml_p']:.3f} < {bar:.3f})"
                if cliff < len(field) else
                f"no cliff needed -- the whole scored field is {len(field)} "
                f"candidate(s), all inside rank {cfg.show_exhaust_max_rank}")
        trc.story(f"[depth {depth}] exhaustion: {edge}; {len(rivals)} rival(s) to "
            f"rule out, {n_pre} already done by the trial queue -- fitting the "
            f"remaining {len(rivals) - n_pre} against the bare seed")
    seed_lvl, seed_keep = _seed_guard_levels(ctx)
    for rv in rivals:
        if id(rv) in already:
            continue
        if _budget_spent(ctx):
            if trc.enabled:
                trc(f"[depth {depth}] exhaustion: [budget spent: "
                    f"{ctx.budget.why()}] cannot finish ruling out the rivals "
                    f"-- multiplicity banner")
            return False, {"exhausted": False, "why": "budget"}
        name = tracklet_tag(rv) if trc.enabled else ""
        tr, rej = _rival_fit(ctx, rv, seed_lvl, seed_keep, name)
        if rej is None:
            if trc.enabled:
                trc.warn(f"exhaustion: rival [{name}] ALSO links "
                    f"(used {tr.used}/{tr.count}, rms {tr.rms:.3f}\") -- the "
                    f"winner is not the only fit, multiplicity banner")
            return False, {"exhausted": True, "why": "a rival also links",
                           "rival": tracklet_tag(rv)}
        if trc.enabled:
            trc(f"[depth {depth}] exhaustion: rival [{name}] ruled out "
                f"({rej[0]})")
    return True, {"exhausted": True, "n_rivals": len(rivals),
                  "n_refitted": len(rivals) - n_pre,
                  "cliff_rank": cliff + 1 if cliff < len(field) else None}


def _sentence(text):
    """``text`` with a full stop, unless it already ends in one.

    The gate's refusal clauses are phrases meant to be read after "nothing
    shown: ", so none of them punctuates itself."""
    text = (text or "").rstrip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _refuse(ctx, human):
    """Decline to show a single-night extension, and SAY WHY in English.

    Every refusal hides the night, but "so do six other tracklets" and "it
    does not fit its own error bars" are opposite findings to the reader.
    The sentence recorded here is what :func:`_lone_night` prints."""
    ctx.show_refusal = human
    return None


def count_linking_rivals(ctx, trial_meta, tracklets_all, n_top):
    """How many of the top ``n_top`` rival tracklets ALSO link, on their own?

    When :func:`_exhaustion_verdict` finds the field too dense for a cliff, a
    show rests on P alone with NOTHING fitted -- where most of the one-night
    show's mistakes are.  This measures it, as a COUNT: "one of two, or one of
    ten?" is the reader's question, and no ML-score density proxy answers it,
    because P is a ranker, not a calibrated probability.

    A rival the trial queue already rejected counts as a non-linker for free;
    every other one costs a trial fit.  No early exit: the count IS the answer.
    """
    taken_ids = {id(t) for t in (trial_meta.get("tracklets") or [])}
    already = {id(t) for t in (trial_meta.get("failed_tracklets") or [])}
    rivals = [t for t in _scored_field(tracklets_all)
              if id(t) not in taken_ids][:n_top]
    seed_lvl, seed_keep = _seed_guard_levels(ctx)
    out, linkers, tested, fitted = [], 0, 0, 0
    t_all = time.time()
    t_fit = 0.0
    for rv in rivals:
        if _budget_spent(ctx):
            break                       # a partial count still bounds linkers
        if id(rv) in already:
            out.append({"tag": tracklet_tag(rv), "ml_p": rv.get("ml_p"),
                        "links": False, "source": "ladder"})
            tested += 1
            continue
        if not rv.get("astrometry"):
            continue
        t0 = time.time()
        tr, rej = _rival_fit(ctx, rv, seed_lvl, seed_keep)
        fitted += 1
        tested += 1
        links = rej is None
        linkers += bool(links)
        secs = time.time() - t0
        t_fit += secs
        out.append({"tag": tracklet_tag(rv), "ml_p": rv.get("ml_p"),
                    "links": bool(links), "source": "fitted",
                    "why": None if links else str(rej[0])[:80],
                    "rms": round(float(tr.rms), 4) if tr.ok else None,
                    "used": tr.used if tr.ok else None,
                    "count": tr.count if tr.ok else None,
                    "seconds": round(secs, 3)})
    return {"k": n_top, "n_tested": tested, "n_fitted": fitted,
            "n_linking": linkers, "rivals": out,
            "seconds": round(time.time() - t_all, 3),
            "fit_seconds": round(t_fit, 3),
            "sec_per_fit": (round(t_fit / fitted, 3) if fitted else None)}


def _note_show_floor(ctx, tracklets_all, depth, windows):
    """Say UP FRONT when the one-night show is already foreclosed on P.

    A 1-opposition seed's most likely find is a lone new night, reportable
    only through :func:`_show_lone_night`, which refuses when the field's best
    ``ml_p`` is under ``show_min_ml_p_abs``.  When the WHOLE field is under it,
    that is settled before any trial fit (2006 UP17 otherwise spent many rounds
    reaching it).  Measured once, on the linear depth-0 field, which is swept
    before the expensive variant fan.

    A warning and NOTHING else: a multi-night apparition does not need the
    show, so the search still runs in full.
    """
    cfg = ctx.cfg
    tel = getattr(ctx, "telemetry", None)
    if (depth != 0 or windows is not None or tel is None
            or getattr(getattr(ctx, "initial_opp_profile", None), "grade",
                       None) is not OppProfileGrade.ONE
            or "show_floor" in tel):
        return
    ps = [t["ml_p"] for t in (tracklets_all or []) if t.get("ml_p") is not None]
    if not ps:
        return
    best = max(ps)
    tel["show_floor"] = {"best_ml_p": round(best, 4),
                         "floor": cfg.show_min_ml_p_abs,
                         "foreclosed": best < cfg.show_min_ml_p_abs}
    if best >= cfg.show_min_ml_p_abs:
        return
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.warn(f"one-night show foreclosed on P: the best candidate in this "
                 f"field scores {best:.3f}, under the {cfg.show_min_ml_p_abs:g} "
                 f"floor this report requires -- so a SINGLE new-night tracklet "
                 f"cannot be reported by this search however well it fits, and "
                 f"only a multi-night apparition can. ({len(ps)} candidate(s) "
                 f"scored.)")


def _rank_candidates(ctx, tracklets_all, depth):
    """Sort ``tracklets_all`` best-first in place and return
    ``(rank_key, likely subset, lov_refine_tail)`` -- the last being the id sets of
    the best LOV-evidenced candidates, for the trial queue to try AFTER everything
    the ranker proposed (see _build_trials).

    ml/certainty ranking need the index path's per-tracklet features; on the
    scan path fall back to the legacy dist+qual score."""
    cfg = ctx.cfg
    ml_ranked = (cfg.ranking_mode == "ml" and cfg.candidate_path == "index"
                 and any("ml_p" in t for t in tracklets_all))
    certainty = (cfg.ranking_mode in ("certainty", "ml")
                 and cfg.candidate_path == "index" and not ml_ranked)
    if ml_ranked:
        rank_key = lambda d: 1.0 - d.get("ml_p", 0.0)   # 1 - P(match), best first
    elif certainty:
        rank_key = lambda d: d.get("match_chi2", float("inf"))
    else:
        if cfg.ranking_mode == "ml":
            warning = ("WARNING: ML ranking requested but unavailable on the "
                       f"{cfg.candidate_path} candidate path; using legacy "
                       "distance + rate-residual ranking")
            ctx.log(warning)
            _ctx_tracer(ctx).warn(warning)
        rank_key = lambda d: d["score"]
    tracklets_all.sort(key=rank_key)
    # LOV-refinement last-resort trials (itf/lov_refine.py): a BOUNDED pick on
    # the measured geometry, never a re-sort.  Three log terms, so none can
    # run away:
    #
    #   log10(cut / residual)   the cross-track miss from the LOV curve at its
    #                           own best k, which no k can absorb -- separates
    #                           the truth far better than corridor offset
    #   log10(1 + nights)       how many OTHER nights want the same k
    #                           (per-tracklet clustering, no threshold)
    #   (log10 P + 6) / 6       the model, on the same scale
    #
    # The top cfg.lov_refine_promote go to the BACK of the trial queue.  Do not
    # re-rank the whole list by this score: 2015 BE559 recovers only under the
    # ML ordering.
    lov_refine_tail = []
    n_promote = int(getattr(cfg, "lov_refine_promote", 0) or 0)
    if getattr(ctx, "lov_refine", None) and getattr(cfg, "lov_refine_rank", False) \
            and n_promote > 0:
        w_sup = float(cfg.lov_refine_support_weight)
        w_ml = float(cfg.lov_refine_ml_weight)
        cut = max(float(cfg.lov_refine_resid_max), 1.0)

        def _evidence(d):
            hit = ctx.lov_refine.get(d["_ids"])
            if hit is None:
                return None
            _k, resid, nights = hit
            ml = (math.log10(max(d.get("ml_p", 0.0), 1e-6)) + 6.0) / 6.0
            return (math.log10(max(cut / max(resid, 1.0), 1.0))
                    + w_sup * math.log10(1.0 + max(nights, 0))
                    + w_ml * ml)

        scored = [(e, i) for i, d in enumerate(tracklets_all)
                  if (e := _evidence(d)) is not None]
        scored.sort(key=lambda t: -t[0])
        lov_refine_tail = [tracklets_all[i]["_ids"] for _e, i in scored[:n_promote]]
        trc_t = _ctx_tracer(ctx)
        if lov_refine_tail and trc_t.enabled:
            trc_t.cont("LOV-refinement last-resort trials: " + ", ".join(
                f"{tracklet_tag(d)} ({ctx.lov_refine[d['_ids']][1]:.0f}\" off the "
                f"LOV, {ctx.lov_refine[d['_ids']][2]} night(s) agree)"
                for d in tracklets_all
                if d["_ids"] in set(lov_refine_tail)))

    likely = [t for t in tracklets_all if t["traj_category"] == "likely"]
    # ML override of the geometric gate: a best-ranked candidate that is only
    # "maybe" (motion-mismatch 1.0-2.5") still earns a trial fit, whether or not
    # the orbit is well-determined.  Only that ONE is promoted; the trial queue
    # and acceptance gates still vet it.
    trc = _ctx_tracer(ctx)
    promoted = None
    if (ml_ranked and cfg.try_top_ranked_maybe and tracklets_all
            and tracklets_all[0]["traj_category"] == "maybe"):
        promoted = tracklets_all[0]
        likely = [promoted] + likely
        if trc.enabled:
            trc.story(f"top ML candidate {tracklet_tag(promoted)} is "
                f"'maybe' (P(match) {promoted.get('ml_p', 0.0):.3f}) -- promoted "
                f"for a trial fit; ML confidence overrides the maybe gate")
    if ctx.verbose:
        ctx.log(f"candidates: {len(tracklets_all)} tracklets, "
                f"{len(likely)} likely"
                + (" (incl. 1 ML-promoted maybe)" if promoted is not None else ""))
    if trc.enabled and tracklets_all:
        mode = ("ml P(match)" if ml_ranked
                else "certainty chi2" if certainty else "legacy distance + rate residual")
        # STORY: the best three ("was anything promising?"); DETAIL: top 12.
        n_show = 12 if trc.wants(L_DETAIL) else 3
        trc.cont(f"ranked by {mode}:")
        for i, cand in enumerate(tracklets_all[:n_show]):
            if "ml_p" in cand:
                score = f"P(match) {cand['ml_p']:.3f}"
            elif math.isfinite(cand.get("match_chi2", math.inf)):
                score = f"chi2 {cand['match_chi2']:.1f}"
            else:
                score = f"score {cand['score']:.0f}"
            trc.cont(f"#{i + 1} {tracklet_tag(cand)} [{cand['traj_category']}]"
                     f"  {score}  motion-mismatch {cand['qual']:.1f}\", "
                     f"offset {cand['dist']:.0f}\"")
        if len(tracklets_all) > n_show:
            trc.cont(f"... and {len(tracklets_all) - n_show} more")
    return rank_key, likely, lov_refine_tail


def _top2_record(likely, rank_key):
    """The two best-ranked candidates as ``{"ids", "ml_p", "tag"}`` records --
    what the three-thin confirmed-pair upgrade compares the final linkage
    against (:func:`_confirm_top_pair` path 3).  None with fewer than two
    candidates (the thin profile needs two new oppositions anyway).

    ``ids`` is the tracklet AS SWEPT, before any pruning (path 2's currency),
    so "the linkage is exactly these two" also means nothing was clipped."""
    if len(likely) < 2:
        return None
    best = sorted(likely, key=rank_key)[:2]
    return [{"ids": {obs_unique_id(l)
                     for l in t["astrometry"].splitlines() if l.strip()},
             "ml_p": t.get("ml_p"),
             "tag": tracklet_tag(t)}
            for t in best]


def _sweep_tiers(pervar, size):
    """``pervar`` (central-out) cut into consecutive tiers of ``size``
    variants; one tier holding everything when ``size`` is 0 or covers it."""
    if not size or size >= len(pervar):
        return [list(pervar)]
    return [list(pervar[i:i + size]) for i in range(0, len(pervar), size)]


class _CachedSource:
    """Serves the variant scan's candidate rows from a SINGLE union sweep so
    each variant's retrieval is in-memory, not a fresh DB round-trip.

    Otherwise each of up to ``variant_max_scan`` variants re-sweeps the whole
    ITF era against the database (most of an hour on a negative search).
    Windows NOT in the cache -- the deeper recursion's corridor on the
    tightened orbit -- fall through to the real ``base`` source, so recursion
    still reaches the live ITF.

    Keyed by object identity (a Window is an unhashable dataclass); each entry
    holds its window, so no id can be recycled while the cache lives.  Sound
    only because the SAME Window objects go to the union sweep and to every
    consumer -- see :meth:`lookup`.
    """

    def __init__(self, base):
        self.base = base
        self.cache = {}                    # id(win) -> (win, rows, chi2, feats)
        self.last_sweep_skipped = 0

    def remember(self, window, rows, chi2, feats):
        self.cache[id(window)] = (window, rows, chi2, feats)

    def lookup(self, window):
        """What the UNION sweep holds for ``window`` -- ``(win, rows, chi2,
        feats)`` -- or None.

        THE accessor for a consumer that wants the union sweep's rows rather
        than to sweep (:func:`lov_profile.build_lov_profiles`).  Do not reach
        into ``self.cache`` and spell the key elsewhere: a re-keying would then
        make every LOV profile lookup miss in silence (no sharp candidates, no
        priority variants or cluster trials) with the offline suite green.
        ``test_lov_profile_reads_the_union_sweep_through_the_accessor`` guards it.
        """
        return self.cache.get(id(window))

    def sweep_candidates(self, windows, uncert_mult, extrawidth, want_ranking,
                         row_budget=None, prefilter=None):
        miss = []
        for w in windows:
            hit = self.cache.get(id(w))
            if hit is not None:
                yield (w, *hit[1:])
            else:
                miss.append(w)
        if miss:
            yield from self.base.sweep_candidates(miss, uncert_mult, extrawidth,
                                                  want_ranking,
                                                  row_budget=row_budget,
                                                  prefilter=prefilter)


def _variant_scan(ctx, eph_trial, prior_linkage_used, extrawidth, depth,
                  hypothesis_obs80, fan=None, prior_done=False):
    """Scan LOV variant orbits for an UNDER-DETERMINED seed (itf/lov_variants.py).

    Samples orbits along the Line of Variations central-out, each with a thin
    fixed-width corridor (tracks from one shared ASSIST integration), and runs
    the ordinary round on ONE variant at a time.  Early-exits on the first
    accepted linkage: one variant reaching one apparition suffices, since the
    refit collapses U and deeper recursion takes the normal corridor.

    Returns the linked/possible/no-match :class:`ItfSearchResult`, or ``None``
    when the variant machinery can't run (degenerate LOV, nongrav-augmented
    state) so the caller falls back to the linear corridor.

    ``fan`` scans those ``(states, ks)`` alone, with no refinement and no retry: the
    population variant's pre-pass, run here before the fan unless ``prior_done``.
    """
    cfg = ctx.cfg
    from .lov_variants import (variant_states, variant_windows,
                           refine_variant_fan, prior_variant)
    from .lov_profile import build_lov_profiles
    # Budget already spent?  Do not pay for the union sweep only to stop at
    # variant #1.
    if _budget_spent(ctx):
        out = ItfSearchResult(telemetry=ctx.telemetry)
        out.reason = (f"the search budget ({ctx.budget.why()}) was spent "
                      f"before the LOV variant scan could start")
        if _ctx_tracer(ctx).enabled:
            _ctx_tracer(ctx)(f"[depth {depth}] variant corridor: budget already spent "
                      f"({ctx.budget.why()}) -- skipping the scan entirely "
                      f"rather than paying for its union sweep")
        return out
    if (fan is None and not prior_done and cfg.variant_prior_min_U
            and eph_trial.U >= cfg.variant_prior_min_U):
        # The pre-pass only adds a variant, so a failure in it (the population table, the
        # pinned re-correction) leaves the fan to run as it would have, never ends the scan.
        try:
            prior = prior_variant(eph_trial.fit, ctx.obscodes)
        except Exception as e:
            ctx.log(f"population variant failed ({type(e).__name__}: {e}); "
                    f"scanning the fan without it")
            prior = (None, None)
        if prior[0] is not None:
            if _ctx_tracer(ctx).enabled:
                _ctx_tracer(ctx).head(f"[depth {depth}] variant corridor: the population's "
                                      f"orbit first, LOV offset {prior[1][0]:+.4f} sigma")
            first = _variant_scan(ctx, eph_trial, prior_linkage_used, extrawidth, depth,
                                  hypothesis_obs80, fan=prior)
            if first is not None and first.verdict is SearchVerdict.LINKED:
                return first
            # A possible stands: the fan below will not repeat a fit this search has
            # made, so it could not find it again (2006 RY118).
            rest = _variant_scan(ctx, eph_trial, prior_linkage_used, extrawidth, depth,
                                 hypothesis_obs80, prior_done=True)
            if first is not None and (rest is None
                                      or _verdict_rank(first) > _verdict_rank(rest)):
                return first
            return rest
    _t = time.perf_counter()
    if fan is not None:
        states, lov_sigmas = fan
        tracks, n_extra = None, 0
    else:
        states, lov_sigmas = variant_states(eph_trial.fit, cfg)
        if states is None:
            return None
        # Densify the fan where its corridors leave a hole (see
        # lov_variants.refine_variant_fan).  Its tracks are handed on, saving
        # variant_windows a second integration.
        states, lov_sigmas, tracks, n_extra = refine_variant_fan(eph_trial.fit,
                                                         ctx.obscodes, cfg,
                                                         states, lov_sigmas)
    try:
        pervar = variant_windows(eph_trial.fit, ctx.obscodes, cfg,
                                 ctx.max_uncert_deg, states, lov_sigmas, tracks=tracks)
    except Exception as e:
        ctx.log(f"variant corridor failed ({type(e).__name__}: {e}); "
                f"falling back to the linear corridor")
        return None
    if not pervar:
        return None
    t_gen = time.perf_counter() - _t
    # LOV offset k -> that variant's state.  Read with .get: the float key only
    # happens to come back bit-identical, and a miss must cost the SCREEN (cold
    # fit), not the SCAN.
    lov_sigma_index = {float(kk): i for i, kk in enumerate(lov_sigmas)}
    # The refinement's extras raise the cap: the fan is central-out, so the
    # plain cap would drop exactly the outer tail they were added to cover.
    n_scan = min(len(pervar), cfg.variant_max_scan + n_extra)
    pervar = pervar[:n_scan]
    trc = _ctx_tracer(ctx)

    # ONE union sweep, served from memory (see _CachedSource), epoch-sorted so
    # windows overlapping across variants chunk together in the DB query.  The
    # variant COUNT (variant_n_steps / variant_max_scan) sizes it, and it
    # dominates the scan's wall time.
    def say(msg):
        """One progress line to both channels (verbose log + detail trace)."""
        if ctx.verbose:
            ctx.log(f"variant corridor: {msg}")
        if trc.enabled:
            trc.head(f"[depth {depth}] variant corridor: {msg}")

    want_rank = (cfg.ranking_mode == "certainty"
                 and cfg.candidate_path == "index")
    cached = _CachedSource(ctx.source)
    # Its OWN row budget: cfg.sweep_row_budget is sized for ONE corridor, and
    # with every variant window the same width its "widest first" order is
    # degenerate, so it would drop a CONTIGUOUS block of epochs -- silent
    # recall loss on the severe seeds the scan exists for (2015 BE559),
    # invisible offline because the budget lives in DbSource.
    union_budget = cfg.variant_sweep_row_budget
    # The LOV offset features (_lov_feature_pass) are fed DURING the sweep, a
    # bounded batch at a time: measured after the cache is full they can run
    # out of memory on a severe seed (2006 JE41).
    def _sweep(tier_windows, credit=0.0):
        """Sweep ``tier_windows`` into ``cached``; returns the seconds it took."""
        tier_windows = sorted(tier_windows, key=lambda w: w.unix_start)
        _t = time.perf_counter()
        for window, rows, chi2, feats in ctx.source.sweep_candidates(
                tier_windows, ctx.uncert_mult, extrawidth, want_rank,
                row_budget=union_budget,
                prefilter=False if _legacy_collect(ctx) else None):
            cached.remember(window, rows, chi2, feats)
        # Fold this sweep's truncation in now, before the per-variant collects
        # reuse the same DbSource: otherwise each cached variant re-adds the
        # union's tally to the "N windows dropped" report.
        _fold_sweep_truncation(ctx)
        dt = time.perf_counter() - _t
        # Every tier's sweep is one-time SETUP, not grind: credited back to
        # both budget clocks (SearchBudget.setup_done) so the variants still
        # get scanned.
        _note_setup(ctx, dt + credit)
        return dt

    scan_ctx = replace(ctx, source=cached, is_variant=True)
    if getattr(ctx, "dlog", None) is not None:
        ctx.dlog.begin_scan(n_scan, eph_trial.U)
    # LOV offset geometry as ranker FEATURES -- measured passively, tier by
    # tier, into ``scan_ctx.lov_feat``; distinct from the last-resort retry
    # below, which ACTS on it.
    lovk = _lov_feature_pass(ctx, eph_trial, pervar)
    # Probability-order TIERS (cfg.variant_sweep_tier; 0 = one union sweep):
    # each is swept, profiled and scanned before the next is swept, so an
    # early link (2005 UD562, variant #2) does not pay for the whole union.
    tiers = _sweep_tiers(pervar, cfg.variant_sweep_tier)
    all_windows, swept, scanned = [], [], set()
    # LOV profiles (itf/lov_profile.py): cross-variant evidence aggregated
    # before any variant is fitted, credited to setup like the sweep.  An
    # optimisation, so a failure degrades to a no-op rather than aborting.
    def _profile(swept):
        lov_profiles = None
        if cfg.lov_profile:
            _t = time.perf_counter()
            try:
                lov_profiles = build_lov_profiles(scan_ctx, swept, cached, lov_sigmas,
                                       eph_trial=eph_trial, extrawidth=extrawidth)
            except Exception as e:
                ctx.log(f"LOV-profile pass failed ({type(e).__name__}: {e}); "
                        f"scanning without it")
                if isinstance(e, MemoryError):
                    # Lost recall (no priority-variant or cluster trials), said
                    # on the capacity channel: `warnings` is for doubts about
                    # the LINKAGE.  setdefault keeps an up-front verdict.
                    ctx.telemetry.setdefault("capacity", {
                        "verdict": _CAPACITY_SHALLOW,
                        "reason": (f"the server ran out of memory part-way through "
                                   f"({e}), so the candidate ranking ran without "
                                   f"its LOV-profile pass -- the search still "
                                   f"completed, but less thoroughly than usual")})
            else:
                _note_setup(ctx, time.perf_counter() - _t)
                ksum = lov_profiles.summary()
                ctx.telemetry["kprofile"] = ksum
                say(f"LOV profiles ({time.perf_counter() - _t:.1f}s): "
                    f"{ksum['n_tracklets']} candidates profiled, "
                    f"{ksum['n_sharp']} sharp; priority variants (homes of the "
                    f"top sharp candidates by P): "
                    + (", ".join(f"LOV offset {v['k']:+.2f} ({v['n_centered']} centered)"
                                 for v in ksum["variants"][:5]) or "none")
                    + ("; visited first" if cfg.lov_profile_order
                       else "; visit order unchanged (report only)"))
                if trc.enabled and ksum["sharp"]:
                    trc.cont("best profiles: " + ", ".join(
                        f"{s['tag']} LOV offset {s['kstar']:+.2f}, rate resid {s['qual']:.2f}\""
                        + (f" P {s['ml_p']:.3f}" if s.get("ml_p") is not None
                           else "")
                        + f" ({s['served']}v)" for s in ksum["sharp"][:5]))
                if ksum["clusters"]:
                    say("cluster proposals (>=2 sharp members, distinct nights): "
                        + "; ".join(f"LOV offset {c['k']:+.2f}: {', '.join(c['tags'])}"
                                    for c in ksum["clusters"][:3]))
        return lov_profiles
    # lov_round (LOV-centered + cluster trials) only when a consumer flag asks;
    # profiles alone are report-only
    lov_sources_on = (cfg.lov_profile_center_singles > 0
                 or cfg.lov_profile_cluster_trials > 0)

    def _rounds(pervar, scan_ctx, lov_profiles, tag=""):
        """One pass over the variant fan.  Returns
        ``(linked_result | None, best_possible, hit_budget, n_scanned)``."""
        best_possible = None
        hit_budget = False
        scan_i = -1
        _t = time.perf_counter()
        for scan_i, (k, vwins) in enumerate(pervar):
            # A seed with nothing to find would otherwise scan every variant.
            if _budget_spent(scan_ctx):
                hit_budget = True
                say(f"budget spent ({scan_ctx.budget.why()}) at variant "
                    f"#{scan_i + 1}/{n_scan}{tag} -- stopping")
                break
            # Screen against THIS variant's own orbit (see _VariantParent), with
            # its own margin: a variant state is not converged on the seed obs.
            vi = lov_sigma_index.get(k)
            vparent = (_VariantParent(eph_trial.fit, states[vi])
                       if (cfg.screen_variant_trials
                           and eph_trial.fit is not None and vi is not None)
                       else None)
            lov_round = (lov_profiles.round_for(k)
                      if (lov_profiles is not None and lov_sources_on) else None)
            # the serving variant's k, a ranker feature (training._v4_frame)
            scan_ctx.variant_lov_sigma = float(k)
            if getattr(ctx, "dlog", None) is not None:
                ctx.dlog.variant_lov_sigma = float(k)
            result = _iterate(scan_ctx, eph_trial, prior_linkage_used, extrawidth,
                           depth, hypothesis_obs80=hypothesis_obs80, windows=vwins,
                           screen_parent=vparent, lov_round=lov_round,
                           round_label=f"depth {depth}, LOV variant "
                                       f"{scan_i + 1}/{n_scan}{tag} "
                                       f"(LOV offset {k:+.2f} sigma)")
            if result.verdict is SearchVerdict.LINKED:
                say(f"LINKED at variant #{scan_i + 1}/{n_scan}{tag} "
                    f"(LOV offset {k:+.2f} sigma) after "
                    f"{time.perf_counter() - _t:.1f}s of per-variant scan")
                return result, best_possible, hit_budget, scan_i + 1
            if result.verdict is SearchVerdict.POSSIBLE and best_possible is None:
                best_possible = result
        else:
            say(f"scanned all {len(pervar)} variants{tag}, no link "
                f"({time.perf_counter() - _t:.1f}s of per-variant scan)")
        return None, best_possible, hit_budget, scan_i + 1

    best_possible, hit_budget, n_done, lov_profiles = None, False, 0, None
    for ti, tier in enumerate(tiers):
        tier_windows = [w for _k, ws in tier for w in ws]
        t_sweep = _sweep(tier_windows, credit=t_gen if ti == 0 else 0.0)
        all_windows += tier_windows
        swept += tier
        if lovk is not None:
            scan_ctx.lov_feat = _lov_feature_offer(ctx, lovk, cached,
                                                   tier_windows)
        ttag = f" (tier {ti + 1}/{len(tiers)})" if len(tiers) > 1 else ""
        if ti == 0:
            say(f"under-determined seed (U {eph_trial.U:.2f} >= "
                f"{cfg.variant_corridor_U}) -> {n_scan} LOV variants "
                f"central-out, thin {cfg.variant_width_deg}deg corridors, "
                + (f"{len(all_windows)} windows in one union sweep "
                   if len(tiers) == 1 else
                   f"swept in {len(tiers)} probability-order tiers, tier 1: "
                   f"{len(tier)} variants / {len(tier_windows)} windows ")
                + f"(gen {t_gen:.1f}s, sweep {t_sweep:.1f}s), early-exit on "
                  f"a link")
        else:
            say(f"tier {ti + 1}/{len(tiers)}: {len(tier)} more variants, "
                f"{len(tier_windows)} windows (sweep {t_sweep:.1f}s)"
                + (" -- swept for the LOV-refinement retry only, the scan's "
                   "budget is spent" if hit_budget else ""))
        # Profiles see everything swept so far: a priority variant is visited
        # once its tier is swept, never before.
        lov_profiles = _profile(swept)
        if hit_budget:
            # keep sweeping: the LOV-refinement retry below must see the whole
            # fan's cache (2008 UM18 links only there)
            continue
        order = (lov_profiles.reorder(swept) if (lov_profiles is not None
                                         and cfg.lov_profile_order) else swept)
        todo = [e for e in order if float(e[0]) not in scanned]
        if not todo:
            continue
        linked, bp, hb, n = _rounds(todo, scan_ctx, lov_profiles, tag=ttag)
        scanned.update(float(k) for k, _w in todo[:n])
        n_done += n
        if best_possible is None:
            best_possible = bp
        if linked is not None:
            return linked
        hit_budget = hit_budget or hb
    if lov_profiles is not None and cfg.lov_profile_order:
        pervar = lov_profiles.reorder(pervar)

    # NOTHING LINKED.  Only now is the LOV-refinement evidence
    # (itf/lov_refine.py) acted on, and this ordering IS the safety property.
    # The scan early-exits on the first accepted linkage, so any extra trial
    # given to a round that was going to fail can make it a WRONG success that
    # pre-empts a later variant's correct link.  Every mid-scan use (promotion,
    # reordering, even trials at the back of a queue) linked foreign tracklets
    # on 2021 SA70.  Deferred, an object that links never reaches this code.
    # The rows are still cached, so this costs no second sweep.
    if cfg.lov_refine and fan is None and (best_possible is not None or hit_budget
                      or n_done >= n_scan):
        _tk = time.perf_counter()
        try:
            from .lov_refine import StreamingLovRefine, install as _lov_refine_install
            lov_stream = StreamingLovRefine(eph_trial.fit, cfg, pervar, ctx.obscodes,
                                 log=ctx.log)
            if lov_stream.prepare(all_windows):
                for window, rows, _chi2, _feats in cached.cache.values():
                    if rows:
                        lov_stream.offer(window, rows, ctx.pub_ids)
                _prio, _lov_rounds, _lov_report = lov_stream.rounds()
                lov_stream.release()
                ctx.telemetry["kfine"] = _lov_report
                say(f"nothing linked -- LOV refinement ({time.perf_counter() - _tk:.1f}s): "
                    f"{_lov_report['n_seen']} candidates measured over the cached "
                    f"sweep, {_lov_report['n_kept']} within "
                    f"{cfg.lov_refine_resid_max:.0f}\" of the LOV curve; "
                    + (", ".join(f"LOV offset {c['k']:+.3f}: {c['n_nights']} nights, "
                                 f"median {c['resid']:.0f}\""
                                 for c in _lov_report["clusters"][:3])
                       or "no multi-night LOV-offset cluster")
                    + " -- re-scanning on that evidence")
                if trc.enabled:
                    for c in _lov_report["clusters"][:3]:
                        trc.cont(f"k-cluster {c['k']:+.4f}: {c['n_nights']} "
                                 f"nights / {c['n_members']} members, median "
                                 f"{c['resid']:.0f}\" over {c['span_days']}d, "
                                 f"{', '.join(c['stations'])}")
                retry_ctx = scan_ctx
                if cfg.lov_refine_rank:
                    retry_ctx = replace(scan_ctx, lov_refine=lov_stream.lookup())
                retry_pervar = pervar
                if _lov_rounds and cfg.lov_refine_order:
                    lov_profiles = _lov_refine_install(lov_profiles, _prio, _lov_rounds)
                    retry_pervar = lov_profiles.reorder(pervar)
                # pass 1 ends by SPENDING the budget, so the retry gets a
                # bounded slice of its own -- never a raised ceiling.
                if retry_ctx.budget is not None:
                    retry_ctx.budget.grant_retry(cfg.lov_refine_retry_budget_frac)
                linked2, best2, _hb2, _n2 = _rounds(retry_pervar, retry_ctx,
                                                    lov_profiles, tag=" (k-retry)")
                if linked2 is not None:
                    return linked2
                if best_possible is None:
                    best_possible = best2
        except Exception as e:
            # The retry is a whole nested search, so this can catch a real bug:
            # keep the first pass's answer, but log the traceback.
            ctx.log(f"LOV-refinement retry failed ({type(e).__name__}: {e}); "
                    f"reporting the first pass\n{traceback.format_exc()}")
            if trc.enabled:
                trc.warn(f"LOV-refinement retry raised {type(e).__name__}: {e} "
                         f"-- the first pass's verdict stands")

    if best_possible is not None:
        return best_possible
    out = ItfSearchResult(telemetry=ctx.telemetry)
    out.reason = (
        f"no LOV variant corridor found a linkable tracklet within the search "
        f"budget (scanned {n_done}/{n_scan})"
        if hit_budget else
        f"no LOV variant corridor found a linkable tracklet (scanned {n_scan})")
    return out


def _lov_feature_pass(ctx, eph_trial, pervar):
    """A prepared :class:`lov_refine.StreamingLovRefine` for FEATURES, or None.

    On when a decision log rides, ``cfg.lov_features`` asks, or the ranker
    was trained on the geometry.  Keeps every candidate under the residual cut
    (``cfg.lov_feature_keep``), not just the retry's best few, so a feature
    does not depend on how crowded the fan was.  Prepared over the WHOLE fan up
    front.  Any failure degrades to no features."""
    cfg = ctx.cfg
    if (getattr(ctx, "dlog", None) is None
            and not getattr(cfg, "lov_features", False)
            and not _ranker_wants_lov(cfg)):
        return None
    _t = time.perf_counter()
    try:
        from .lov_refine import StreamingLovRefine
        fcfg = replace(cfg, lov_refine_keep=int(cfg.lov_feature_keep),
                       lov_refine_halfwidth=float(cfg.lov_feature_halfwidth))
        lovk = StreamingLovRefine(eph_trial.fit, fcfg, pervar, ctx.obscodes,
                          log=ctx.log)
        if not lovk.prepare([w for _k, ws in pervar for w in ws]):
            return None
    except Exception as e:                                   # noqa: BLE001
        ctx.log(f"LOV feature pass unavailable ({type(e).__name__}: {e})")
        return None
    dt = time.perf_counter() - _t
    if ctx.dlog is not None:
        ctx.dlog.note(ev="lov_prepare", sec=round(dt, 3))
    _note_setup(ctx, dt)
    return lovk


def _ranker_wants_lov(cfg):
    """True when a configured ranker was trained on the continuous LOV offset
    columns (itf/trainv4.py) -- serving it without them would score every
    variant candidate as 'beyond the residual cut'."""
    if cfg.ranking_mode != "ml":
        return False
    try:
        from .training import load_ranker
        from .training import default_model_path
        return any("lov_resid" in load_ranker(p)[1].get("features", ())
                   for p in (cfg.ml_model_path or default_model_path(),
                             cfg.ml_second_model_path)
                   if p)
    except Exception:                                        # noqa: BLE001
        return False


def _lov_feature_offer(ctx, lovk, cached, tier_windows):
    """Measure one tier's swept candidates; the updated feature lookup."""
    _t = time.perf_counter()
    try:
        for w in tier_windows:
            hit = cached.lookup(w)
            if hit is not None and hit[1]:
                lovk.offer(hit[0], hit[1], ctx.pub_ids)
        out = lovk.lookup()
    except Exception as e:                                   # noqa: BLE001
        ctx.log(f"LOV feature measure failed ({type(e).__name__}: {e})")
        return None
    dt = time.perf_counter() - _t
    if ctx.dlog is not None:
        ctx.dlog.note(ev="lov_offer", sec=round(dt, 3), n_seen=lovk.n_seen,
                      n_kept=len(out))
    _note_setup(ctx, dt)
    return out


def _carried_pool(ctx, carried, tracklets_all, rank_key):
    """The untaken-candidate pool one recursion level hands to the next.

    Union of what this level inherited and swept, de-duplicated by tracklet,
    ranked best-first and truncated to ``cfg.carried_max`` (a wide corridor can
    retrieve tens of thousands).  Taken tracklets are NOT filtered here: they
    drop out one level down as ``own_ids`` duplicates, the sweep's own test.
    """
    cap = int(getattr(ctx.cfg, "carried_max", 0) or 0)
    if cap <= 0:
        return None
    pool = {}
    # This level's own scored copy first: an inherited copy has been stripped
    # of its score (see the return below), so if it won the de-duplication a
    # tracklet this level re-swept and scored well would rank as the worst
    # candidate and fall to the cap.  Measured: 2018 TF3 lost its 2021
    # tracklet -- and with it three oppositions -- to exactly that, once a
    # wider corridor had already served the tracklet (unscored, unlikely) one
    # level up.
    for tr in list(tracklets_all or ()) + list(carried or ()):
        ids = tr.get("_ids")
        if ids is None or tr.get("traj_category") == "unlikely":
            continue
        pool.setdefault(ids, tr)
    if not pool:
        return None
    out = sorted(pool.values(), key=rank_key)[:cap]
    # keep only what survives an orbit change; the rest is re-derived on arrival
    return [{"_ids": t["_ids"], "rows": t["rows"], "_feat": t.get("_feat")}
            for t in out]


def _iterate(ctx, eph_trial, prior_linkage_used, extrawidth, depth,
             hypothesis_obs80="", windows=None, screen_parent=None, round_label=None,
             carried=None, lov_round=None):
    """One ROUND (see the module docstring): sweep ``eph_trial``'s corridor,
    rank, take subsets via the trial queue, recurse on each take, and vet the
    terminal linkage with :func:`_vet_terminal`.  Returns the best
    :class:`ItfSearchResult` (linked, possible 0.6, or rejected with a reason).

    ``hypothesis_obs80`` -- the ITF obs every shallower level already took:
    they ride in every trial fit here and are filtered out of the sweep.
    ``windows`` -- a variant's corridor (None: ``eph_trial``'s own);
    ``screen_parent`` is then that variant's screen parent.  ``carried`` --
    the untaken pool from the level above (:func:`_carried_pool`).
    ``round_label`` -- the trace's name for this pass; only the caller knows
    whether it is a recursion level or the Nth LOV variant."""
    cfg = ctx.cfg
    result = ItfSearchResult(telemetry=ctx.telemetry)
    if depth > _MAX_DEPTH:
        result.reason = "recursion depth limit"
        return result
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.round_begin(round_label or f"depth {depth}")
        trc.field_changed("orbit", f"used {eph_trial.used}/{eph_trial.count}, "
                          f"rms {eph_trial.rms:.3f}\", U {eph_trial.U:.2f} "
                          f"(best prior linkage used {prior_linkage_used})")

    own_ids = ctx.pub_ids
    if hypothesis_obs80:
        own_ids = own_ids | {obs_unique_id(l)
                             for l in hypothesis_obs80.splitlines() if l.strip()}
    rid = (ctx.dlog.begin_round(round_label or f"depth {depth}", depth,
                                windows is not None and ctx.is_variant, eph_trial)
           if getattr(ctx, "dlog", None) is not None else None)
    tracklets_all = _collect_tracklets(ctx, eph_trial, extrawidth,
                                       relaxed=_relaxed(cfg, eph_trial),
                                       depth=depth, own_ids=own_ids, windows=windows,
                                       carried=carried)
    rank_key, likely, lov_refine_tail = _rank_candidates(ctx, tracklets_all, depth)
    if rid is not None:
        ctx.dlog.round_size(rid, len(tracklets_all), len(likely))
    # The LOV-refinement tail exists for candidates the ranker does NOT rate
    # likely, so an empty likely set is no reason to skip it.
    if not likely and not lov_refine_tail:
        trc.field("outcome", "no likely candidates -- nothing to try here",
                  L_HEAD)
        result.reason = "no likely matches"
        result.no_candidates = True
        return result

    _note_show_floor(ctx, tracklets_all, depth, windows)

    # The field as the SEED orbit ranks it, before any trial bends the orbit
    # (_confirm_top_pair path 3 is entitled to this ranking).
    if depth == 0:
        ctx.depth0_top2 = _top2_record(likely, rank_key)

    # Warm-screen parent: inside a variant scan, that variant's own state
    # (see :class:`_VariantParent`), never the nominal orbit.
    screen_from = eph_trial if windows is None else screen_parent
    # The maybe pool, for the trial sources entitled to fish below the likely
    # cliff: LOV-centered/cluster trials (a profile measured ACROSS variants
    # must not be hidden by one window's likely/maybe flip -- 2015 BE559) and,
    # under cfg.second_singles_maybe, the second-opinion singles.
    extra_pool = None
    if lov_round is not None or (cfg.second_singles_maybe
                              and cfg.recurse_second_singles > 0
                              and cfg.ml_second_model_path):
        in_likely = {id(t) for t in likely}
        extra_pool = [t for t in tracklets_all
                      if t["traj_category"] == "maybe"
                      and id(t) not in in_likely]
    # The tail's own picks join the trial pool whatever their grade: the queue
    # can only trial what the pool holds, and an 'unlikely' pick would
    # otherwise be named as a last-resort trial and silently never fitted.
    if lov_refine_tail:
        pooled = {id(t) for t in likely} | {id(t) for t in extra_pool or ()}
        tail_ids = set(lov_refine_tail)
        extra_pool = (extra_pool or []) + [
            t for t in tracklets_all
            if t["_ids"] in tail_ids and id(t) not in pooled]
    carried_down = _carried_pool(ctx, carried, tracklets_all, rank_key)
    # Backtracking (cfg.backtrack_per_round): a take whose whole subtree comes
    # back with NOTHING (not a linkage, not a reviewable 0.6) does not
    # foreclose the round; the queue resumes at its next untried trial.
    # DEPTH 0 ONLY: each resume buys a subtree, and multiplying subtrees at
    # every depth exhausts the budget; at depth 0 a barren branch off a weak
    # seed has a tiny corridor.
    backtracks_left = (max(int(getattr(cfg, "backtrack_per_round", 0) or 0), 0)
                       if depth == 0 else 0)
    best_result = None
    trial_queue = TrialQueue(ctx, likely, rank_key, prior_linkage_used, hypothesis_obs80,
                     parent=screen_from, lov_round=lov_round, extra_pool=extra_pool,
                     linkage_fit=getattr(eph_trial, "fit", None),
                     interleave=backtracks_left > 0,
                     round_label=round_label or f"depth {depth}",
                     lov_refine_tail=lov_refine_tail)
    # ...and at ANY depth in a flat field on a linear corridor: the ranker
    # cannot choose there, and a lone new tracklet is a hypothesis one fit
    # cannot settle (a short arc fits several distractors about as well as the
    # truth -- picking the best-fitting pass instead was tried and favoured
    # 2-obs tracklets, which fit "better" by carrying fewer constraints).  What
    # tells them apart is whether the branch below gathers corroboration.  As at
    # depth 0, the cost lands only on a take whose branch came back barren.
    # 2015 HU248: its 2019 truth is the third passing single, the two above it
    # lead nowhere.  A resumed take does not restart the no-progress clock
    # (TrialQueue.resumed), so these resumes are bounded by time, not only by
    # the search ceiling.
    if depth > 0 and trial_queue.flat:
        backtracks_left = max(int(getattr(cfg, "backtrack_per_round", 0) or 0), 0)
    while (take := trial_queue.next_take()) is not None:
        trial, obs_out, trial_meta = take
        # Did the subtree search the WHOLE corridor?  Dropped windows (row
        # budget, cancelled query, memory floor) make its answer not final
        # (see _hypothesis_key); read as a before/after of the same counter.
        dropped_pre = _windows_dropped(ctx)
        child = _iterate(ctx, trial, trial.used, 0, depth + 1,
                         hypothesis_obs80=obs_out, carried=carried_down)
        if child.verdict is SearchVerdict.LINKED:
            return child
        if trc.enabled:
            trc.head(f"<<< depth {depth + 1} came back without a deeper "
                     f"linkage ({child.reason or 'nothing further'}) -- depth "
                     f"{depth}'s trial is the terminal candidate; vetting it")
        # A linkage this vet accepts is transmitted only once the terminal fan
        # (below) has decided whether it stands: hold its document while the
        # fan could still run.  Fresh per take, and cleared before the fan, so
        # the fan's own vets transmit as usual.
        ctx.held_transmits = [] if _terminal_fan_possible(cfg, ctx, trial) else None
        try:
            latest = _vet_terminal(ctx, trial, obs_out, trial_meta, child,
                                   tracklets_all, prior_linkage_used, depth,
                                   hypothesis_obs80)
        finally:
            held, ctx.held_transmits = ctx.held_transmits, None
        # Asked and answered: no later round re-fits this hypothesis unless
        # its subtree searched only part of the ITF.  Also feeds
        # _recurrent_notes.
        _hypothesis_record(ctx, trial_meta.get("hyp_key"),
                    _windows_dropped(ctx) == dropped_pre,
                    where=trial_meta.get("round"), tags=trial_meta.get("tags"),
                    refusal=getattr(ctx, "show_refusal", None),
                    used=trial.used, count=trial.count, rms=trial.rms)
        # A terminal orbit still under-determined gets the LOV fan the seed
        # would have had (see ItfConfig.terminal_fan): its linear corridor can
        # be far over max_uncert_deg at another apparition while the truth
        # sits on its Line of Variations.  Once per search, so only from a
        # take the vet calls reviewable -- see _terminal_fan_due.  A link from
        # the fan supersedes this terminal; anything less leaves it standing.
        fanned = _terminal_fan(ctx, trial, obs_out, latest, held, depth)
        if fanned is not None:
            return fanned
        # keep the strongest verdict across laps
        if best_result is None or _verdict_rank(latest) > _verdict_rank(best_result):
            best_result = latest
        # Only a REVIEWABLE possible ends the round: a bare 0.6 IS the barren
        # branch backtracking exists for.
        if latest.reviewable or backtracks_left <= 0 or _budget_spent(ctx):
            return best_result
        # Barren take (see TrialQueue.mark_failed: the ledger, not a ban,
        # stops this hypothesis).
        backtracks_left -= 1
        survived = obs_ids(obs_out) - obs_ids(hypothesis_obs80) - ctx.pub_ids
        rode = trial_queue.mark_failed(trial_meta, survived)
        if rid is not None:
            ctx.dlog.barren(rid, trksubs_of(trial_meta.get("tracklets") or ()))
        if trc.enabled:
            trc.head(f"BACKTRACK: the branch below "
                     f"[{' + '.join(tracklet_tag(t) for t in rode) or 'the taken trial'}]"
                     f" came back empty ({latest.reason or 'nothing'}) -- "
                     f"resuming this round's trial queue "
                     f"({backtracks_left} resume(s) left). This exact set is "
                     f"not asked again; its members stay free to ride in a "
                     f"LARGER one, which may be what they needed")
    if best_result is not None:
        return best_result       # every take was barren; the last verdict stands
    result.reason = ("only found tracklets that sig. worsen fit")
    return result


def _lone_night(ctx, result, a, x, trial_meta, depth, tracklets_all):
    """A consistent linkage whose new opposition holds ONE night (the 1.5
    profile).  Surfaced as a 0.6 possible/likely when trustworthy enough
    (:func:`_show_lone_night`), never submittable; otherwise a bare 0.6 that
    says why nothing is shown -- "we found something and distrust it", not
    "we found nothing"."""
    shown = _show_lone_night(ctx, result, a, x, trial_meta, depth, tracklets_all)
    # Study instrumentation (itf/gatelog.py), off by default: the whole clause
    # set re-measured, to ask what a different threshold would have done.
    if gatelog.enabled():
        gatelog.record("one_night", **gatestats.one_night_row(
            ctx, a.trial, a.obs_out, trial_meta, a.profile, depth, tracklets_all,
            shown=shown is not None,
            reason=(shown.reason if shown is not None
                    else getattr(ctx, "show_refusal", None)),
            run_exhaustion=gatelog.want_exhaustion()))
    if shown is not None:
        return shown
    result.verdict = SearchVerdict.POSSIBLE
    result.vetting = {"show_suppressed": True,
                      "overall_rms_post": round(a.trial.rms, 4),
                      "opps_pre": ctx.initial_opp_profile.legacy_code,
                      "opps_post": a.profile.legacy_code}
    refusal = getattr(ctx, "show_refusal", None)
    result.reason = (f"nothing shown: {_sentence(refusal)}" if refusal
                     else f"{a.profile} opps without progress, abandoning")
    ctx.log(result.reason)
    return result


def _show_lone_night(ctx, result, a, x, trial_meta, depth, tracklets_all):
    """Is a lone new night trustworthy enough to put in front of a reviewer?
    Returns the filled ``result`` on a show, else None with the refusal
    recorded (:func:`_refuse`).

    Its consistency is already established -- the night fits its own
    uncertainties and the seed is not degraded (:func:`_assess`,
    :func:`_extension_check`).  A lone night ALSO needs (config ``show_*``):

      1. PROFILE -- depth 0, ONE new night lifting the 1-opposition seed, on a
         seed not badly under-determined.  One NIGHT, not one TRACKLET:
         surveys split a night into several trksubs (2024 QF7).
      2. A FIELD worth reading -- its best P(match) clears
         ``show_min_ml_p_abs``.
      3. IT IS THE ONLY ONE THAT FITS -- measured by
         :func:`_exhaustion_verdict`.  Failing that, a field whose best P
         clears ``show_multi_min_ml_p`` is still shown, with a banner that the
         night may be one of several.

    PATH-AGNOSTIC: an LOV-variant hit qualifies exactly as a linear one does;
    the exhaustion clause, not the corridor, is the anti-coincidence guard."""
    cfg = ctx.cfg
    trc = _ctx_tracer(ctx)
    trial, obs_out = a.trial, a.obs_out
    # --- clause 1: the profile ---------------------------------------------
    # The trial need NOT be the best-ranked prefix of the queue: clause 3
    # measures exclusivity rather than assuming it from rank.
    n_new_nights = count_spans(obs_out) if count_obs(obs_out) else 0
    taken = (trial_meta or {}).get("tracklets") or []
    if not (depth == 0 and n_new_nights == 1 and taken
            and ctx.pub.U < cfg.single_night_show_max_U):
        if trc.enabled and depth == 0:
            trc.warn(f"one-night show: profile/seed clause fails (seed U "
                     f"{ctx.pub.U:.2f} vs cap {cfg.single_night_show_max_U:g}, "
                     f"{len(taken)} tracklet(s) over {n_new_nights} new "
                     f"night(s) in the trial) -- not shown")
        # Name the sub-clause that failed: only the first is about the object
        # rather than the search.
        if ctx.pub.U >= cfg.single_night_show_max_U:
            return _refuse(ctx, f"the object's own orbit is too weakly "
                f"determined (U {ctx.pub.U:.1f}, past this report's limit of "
                f"{cfg.single_night_show_max_U:g}) for a single new night to "
                f"mean anything -- the corridor it was found in is wide enough "
                f"that unrelated tracklets fall inside it routinely")
        if n_new_nights != 1:
            return _refuse(ctx, f"the candidate arrived as "
                f"{n_new_nights} new nights at once rather than one clean "
                f"new night, which is not a case this report can vet")
        if not taken:
            return _refuse(ctx, "the extra night is not traceable to a "
                "particular tracklet, so there is nothing specific to put in "
                "front of you")
        return _refuse(ctx, "the candidate turned up part-way down a chain of "
            "earlier links rather than as one clean new night on the "
            "object's own arc -- evidence too indirect to show")
    if ctx.pub.fit is None or trial.fit is None:
        return _refuse(ctx, "the fits needed to test the match against its "
            "own error bars were not available")
    # --- clause 2: a field worth reading -----------------------------------
    ml_ps = [t["ml_p"] for t in (tracklets_all or []) if t.get("ml_p") is not None]
    if not ml_ps:
        if trc.enabled:
            trc.warn("one-night show: no ML scores in the field -- a "
                     "confidence claim needs them, not shown")
        return _refuse(ctx, "no match probabilities were computed for this "
            "field, so there is no way to say whether the match is a "
            "standout or one of many")
    best_ml = max(ml_ps)
    # The floor reads the field's best, but the sentence must quote the score
    # of the candidate it refuses; the two routinely differ (2006 UP17).
    won = [p for p in ((trial_meta or {}).get("ml_ps") or []) if p is not None]
    winner_ml = max(won) if won else None
    ctx.show_ml = {"best_in_field": round(best_ml, 4),
                   "this_candidate": (round(winner_ml, 4)
                                      if winner_ml is not None else None),
                   "floor": cfg.show_min_ml_p_abs}
    _p = f"{winner_ml:.3f}" if winner_ml is not None else "not scored"
    if best_ml < cfg.show_min_ml_p_abs:
        if trc.enabled:
            trc.warn(f"one-night show: best P in field {best_ml:.3f} < "
                     f"{cfg.show_min_ml_p_abs:g} absolute floor (this candidate "
                     f"scores {_p}) -- not shown")
        return _refuse(ctx, f"nothing in this field is strong enough to be "
            f"worth your time: the candidate that linked scores {_p} as a "
            f"match, and the best anything here scores {best_ml:.2f}, under "
            f"the {cfg.show_min_ml_p_abs:g} floor this report requires")
    # What the reviewer is told about the fit: the new night against its own
    # noise (rejected obs at full residual), and what the seed did.
    new_ids = obs_ids(obs_out)
    chi2_new, n_new = subset_chi2(trial.fit, new_ids, station_floor=True,
                                  included_only=False)
    red_chi2_new = chi2_new / (2 * n_new) if n_new else float("inf")
    seed = SeedPreservation(ctx, trial)
    seed_pre_rms = _seed_pre_rms(ctx)
    seed_post_rms, n_seed_rms = subset_rms(trial.fit, ctx.pub_ids)
    seed_sigma = (math.sqrt(seed.post / (2.0 * seed.n_post)) if seed.n_post
                  else None)
    seed_note = (f"the object's own astrometry moves "
                 f"{per_axis(seed_pre_rms):.3f}\" -> {per_axis(seed_post_rms):.3f}\" "
                 f"({seed_post_rms / seed_pre_rms:.1f}x"
                 + (f", {seed_sigma:.2f} sigma)" if seed_sigma is not None else ")")
                 if n_seed_rms and seed_pre_rms > 0 else
                 "the object's own astrometry could not be re-measured under "
                 "the link")
    # --- clause 3: it is the only one that fits (costs trial fits) ----------
    sole_standout, exhaust = _exhaustion_verdict(ctx, trial_meta, tracklets_all,
                                                 depth)
    if not sole_standout and best_ml < cfg.show_multi_min_ml_p:
        if trc.enabled:
            trc.warn(f"one-night show: not the only fit "
                     f"({exhaust.get('why')}) and best P {best_ml:.3f} < "
                     f"{cfg.show_multi_min_ml_p:g} -- not shown")
        return _refuse(ctx, f"the tracklet that linked is almost certainly "
            f"one of many that would fit this orbit just as well "
            f"({exhaust.get('why')}); at a match probability of {best_ml:.2f} "
            f"it is nowhere near the standout this report needs before naming "
            f"a night, so showing it would most likely be showing you a "
            f"coincidence")
    # ...and HOW MANY fit, when clause 3 could not say (count_linking_rivals;
    # off unless a surface sets show_n_rivals: it costs k trial fits)
    rival_count = None
    if not sole_standout and cfg.show_n_rivals:
        rival_count = count_linking_rivals(ctx, trial_meta, tracklets_all,
                                           cfg.show_n_rivals)
        exhaust = {**exhaust, "rival_count": rival_count}
        n_link = rival_count["n_linking"]
        if trc.enabled:
            trc.story(f"one-night show: {n_link} of {rival_count['n_tested']} "
                      f"best-ranked rival(s) also link on their own "
                      f"({rival_count['seconds']:.1f}s, {rival_count['n_fitted']} "
                      f"fit(s)) -- cap {cfg.show_rival_count_max_linkers}")
        if n_link >= cfg.show_rival_count_max_linkers:
            ctx.log(f"one-night show suppressed: {n_link} rival tracklets fit "
                    f"this orbit as well as the one taken")
            # HOW MANY fit, that it is a FLOOR, and WHAT WOULD SETTLE IT
            return _refuse(ctx, f"at least {n_link} of the "
                f"{rival_count['n_tested']} best-ranked candidates fit this "
                f"orbit as well as the one that linked -- and those "
                f"{rival_count['n_tested']} are only the top of a field of "
                f"{len(ml_ps)}, so more of them would very likely fit too. "
                f"When that many unrelated tracklets match a LONE night, "
                f"matching is not evidence for any one of them, and picking "
                f"the best-scoring one to show you would be picking at random. "
                f"It would take a second night of this object -- from archival "
                f"images or a new observation -- before naming one of them "
                f"meant anything, so none is being named")

    # --- all clauses pass: surface it, never submittable --------------------
    result.verdict = SearchVerdict.POSSIBLE
    made_of = (f"{len(taken)} tracklets on one night" if len(taken) > 1
               else "a single tracklet")
    fits_note = (f"the new night sits within {red_chi2_new:.1f}x its own noise "
                 f"and {seed_note}")
    archival_ask = ("a lone night cannot be submitted on its own. Search "
                    "archival images for a second night in that opposition to "
                    "confirm the linkage and make it submittable")
    if sole_standout:
        tier = ("likely" if (red_chi2_new <= cfg.show_likely_new_redchi2
                             and best_ml >= cfg.show_likely_min_ml_p) else "possible")
        n_rivals = exhaust.get("n_rivals", 0)
        ruled = (f"every one of the {n_rivals} comparably-ranked rival tracklets "
                 f"was fitted and failed to link" if n_rivals else
                 f"no other candidate ranked within "
                 f"{cfg.show_exhaust_frac:.2f}x of it")
        result.reason = (f"single-night extension -- a {tier} lone best-fit night "
                         f"({made_of}), not submittable without a second night "
                         f"outside the initial opposition")
        result.warnings.append(
            f"Single best-fit night found in a new opposition ({tier}, {made_of}): "
            f"it is the sole candidate that fits (P(match) {best_ml:.2f}; {ruled}; "
            f"{fits_note}), but {archival_ask}.")
    else:
        tier = "possible"
        result.reason = (f"single-night extension -- a strong candidate (P(match) "
                         f"{best_ml:.2f}) but one of several compatible tracklets, "
                         f"not submittable without a second night outside the "
                         f"initial opposition")
        result.warnings.append(
            f"Candidate night found in a new opposition ({made_of}; P(match) "
            f"{best_ml:.2f}): it fits the orbit, but {archival_ask}.")
        counted = ("" if not rival_count else
                   f" Of the {rival_count['n_tested']} best-ranked rivals "
                   f"fitted against your arc, "
                   + (f"{rival_count['n_linking']} also link"
                      if rival_count["n_linking"] else "none link")
                   + ".")
        result.warnings.append(
            "This is not necessarily the only match. It may be one of several "
            "potentially-compatible tracklets -- treat it as a lead to verify, "
            "not a confirmed identification." + counted)
    _keep(ctx, result, a, x)
    result.warnings.extend(x.held)
    result.vetting = {**x.vetting,
                      "possible_unconfirmed": True,
                      # BOTH scores: the floor reads the field's best, the
                      # reviewer looks at THIS candidate
                      "best_ml_p": round(best_ml, 4),
                      "this_candidate_ml_p": (round(winner_ml, 4)
                                              if winner_ml is not None else None),
                      "n_candidates": len(ml_ps),
                      "new_tracklet_redchi2": round(red_chi2_new, 3),
                      "n_tracklets": len(taken),
                      "seed_dchi2": round(seed.delta, 3),
                      "seed_sigma_post": (None if seed_sigma is None
                                          else round(seed_sigma, 3)),
                      "seed_degraded": x.seed_caution is not None,
                      "exhaustion": exhaust,
                      "tier": tier,
                      "single_best_fit": sole_standout,
                      "warnings": list(result.warnings)}
    if not sole_standout:
        result.vetting["multiple_compatible"] = True
    if trc.enabled:
        kind = ("sole standout, exhaustion proved it"
                if sole_standout else f"not exhaustible ({exhaust.get('why')})")
        trc.head(f"one-night show: SHOWN ({tier}, {kind}) -- best "
                 f"P {best_ml:.3f} over {len(ml_ps)} scored, new night red "
                 f"chi2 {red_chi2_new:.2f}, seed Delta-chi2 {seed.delta:.1f}, "
                 f"{seed_note}" + (" -- WARNED" if x.seed_caution else ""))
    ctx.log(f"possible single-night linkage kept for review ({tier}, "
            f"{'sole fit' if sole_standout else 'one of several'}, 1.5 opps, "
            f"used {trial.used}/{trial.count}, new-night red-chi2 "
            f"{red_chi2_new:.2f}, seed dchi2 {seed.delta:.1f}); suggest an "
            f"archival-image search for a second night")
    ctx.log(result.reason)
    return result


# ---------------------------------------------------------------------------
# Alternatives to a terminal linkage, and the order the recursion unwinds in
# ---------------------------------------------------------------------------
def _final_sweep(ctx, a, prior_linkage_used, skip_modern=False):
    """One bounded post-acceptance sweep with the accepted (tightest) orbit.

    A tracklet can become "likely" only under the accepted orbit, and the
    recursion never re-sweeps after acceptance.  Under a DETERMINED orbit
    (:func:`_relaxed`) a tracklet the corridor gate refused is also offered
    when most of its observations sit on the orbit's TOPOCENTRIC prediction
    (:func:`_topo_median_offset` <= ``likely_arcsec``): the corridor windows
    are geocentric quadratics, so parallax and one bad endpoint can push a
    tracklet of the object past the gate, while the median against the
    station's own ephemeris is untouched by either.  The augmented linkage is judged on every
    clause, and additions that fail their own are repaired like any linkage
    (:func:`_repaired`: Yarkovsky, vouching, dropping the offenders) -- a
    mop-up, never a side door.  Returns the augmented :class:`Assessment`, or
    ``a`` unchanged.

    ``skip_modern`` -- the deepest recursion already swept this exact orbit
    over the ordinary corridor, so only the ARCHIVE era is collected.  The
    archive always gets this pass: the deep slice only for a strong seed
    (whose ordinary sweeps carried the rest), else all of it (2020 OG4)."""
    cfg = ctx.cfg
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.round_begin("final sweep")
        trc.field("why", "one bounded re-sweep with the accepted (tightest) "
                  "orbit for tracklets that only now look likely"
                  + ("; the modern corridor is skipped (the deepest recursion "
                     "already swept this exact orbit and found no likely "
                     "tracklet)" if skip_modern else ""), L_HEAD)
    trial = a.trial
    determined = _relaxed(cfg, trial)
    tracklets = [] if skip_modern else _collect_tracklets(
        ctx, trial, 0.0, relaxed=determined, depth=-1, keep_unlikely=determined)
    deep_only = _archive_from_the_start(ctx) and not skip_modern
    tracklets = tracklets + _archive_tracklets(ctx, trial, deep_only=deep_only,
                                               keep_unlikely=determined)
    have = obs_ids(a.combined)
    add, n_new = "", 0
    swept = getattr(ctx, "swept_tracklets", None)
    for t in tracklets:
        ids = _tracklet_ids(t)
        if ids & have:                       # already linked (or published)
            continue
        if not (t["traj_category"] == "likely"
                or (determined and _topo_median_offset(ctx, trial.fit, t["rows"])
                    <= cfg.likely_arcsec)):
            continue
        add += t["astrometry"]
        have |= ids
        n_new += 1
        if swept is not None:              # judged per tracklet AS SWEPT
            for i in ids:
                swept[i] = (tracklet_tag(t), ids)
            lines = getattr(ctx, "swept_lines", None)
            if lines is not None:
                lines[ids] = t["astrometry"]
    if not add:
        return a
    new_trial = _fit(ctx, a.combined + add)
    if not (new_trial.ok and new_trial.used > trial.used):
        return a
    b = _assess(ctx, new_trial, a.combined + add, a.obs_out + add,
                prior_linkage_used, vouched=a.vouched, dropped=a.dropped)
    if not b.consistent:
        b = _repaired(ctx, b, prior_linkage_used, -1) or b
    if b.acceptable(ctx) and b.trial.used > trial.used:
        kept = _linked_tracklet_count(b.obs_out) - _linked_tracklet_count(a.obs_out)
        ctx.log(f"final sweep: +{kept} of {n_new} offered tracklet(s), "
                f"used {trial.used} -> {b.trial.used}")
        if trc.enabled:
            trc.cont(f"+{kept} of {n_new} offered tracklet(s) kept: used "
                     f"{trial.used} -> {b.trial.used}, opps {a.profile} -> "
                     f"{b.profile}")
        return b
    ctx.log(f"final-sweep additions refused ({'; '.join(_refusal_reasons(ctx, b))})"
            f" -- reverted")
    return a


def _drop_seed_night(ctx, a, prior_linkage_used):
    """``a`` refitted without the ONE seed night that carries its seed
    degradation (``a.seed.bad_night``), re-assessed -- or None.

    That night is bad astrometry fighting the better orbit, and the remedy is
    to leave it out, not to excuse the degradation it causes.  Leaving out the
    object's own observations is never free: the clip ledger prices the
    dropped span in newly linked tracklets (:func:`_clip_evidence_ok`), so the
    repair is kept only when it is paid for and leaves the seed clean."""
    ids = a.seed.bad_night
    trial = _fit(ctx, drop_obs_ids(a.combined, ids))
    b = None
    if trial.ok:
        b = _assess(ctx, trial, drop_obs_ids(a.combined, ids), a.obs_out,
                    prior_linkage_used, vouched=a.vouched,
                    dropped=a.dropped | ids)
    trc = _ctx_tracer(ctx)
    if b is None or not b.acceptable(ctx) or b.seed.degraded:
        if trc.enabled:
            trc.cont(f"dropping the offending seed night ({n_observations(ids)} obs) does "
                     f"not make an acceptable linkage"
                     + (f": {'; '.join(_refusal_reasons(ctx, b)) or b.seed.unpaid}"
                        if b is not None else "") + " -- kept as it was")
        return None
    if trc.enabled:
        trc.cont(f"dropped the offending seed night ({n_observations(ids)} obs), paid for "
                 f"by {b.seed.n_linked} linked tracklet(s): used {b.trial.used}/"
                 f"{b.trial.count}, rms {b.trial.rms:.3f}\"")
    ctx.log(f"dropped one offending seed night ({n_observations(ids)} obs) -- paid for by "
            f"{b.seed.n_linked} linked tracklet(s)")
    return b


def _topo_median_offset(ctx, fit, rows):
    """Median on-sky offset (arcsec) of a tracklet's observations from
    ``fit``'s TOPOCENTRIC prediction for their station -- two ephemeris points,
    the first and last epoch, interpolated linearly across the tracklet.
    ``inf`` when it cannot be computed."""
    from pyroorb.ephemerides import compute_ephemeris
    try:
        stn = rows[0]["stn"]
        t0, t1 = rows[0]["epoch_sec"], rows[-1]["epoch_sec"]
        ends = [compute_ephemeris(fit, ctx.obscodes, t / 86400.0 + 2440587.5,
                                  t / 86400.0 + 2440587.5, 1.0, site=stn,
                                  full=False)[0] for t in (t0, t1)]
    except Exception:                                        # noqa: BLE001
        return float("inf")
    span = (t1 - t0) or 1.0
    offs = []
    for r in rows:
        f = (r["epoch_sec"] - t0) / span
        ra = ends[0]["ra_deg"] + f * _wrap_delta(ends[1]["ra_deg"], ends[0]["ra_deg"])
        dec = ends[0]["dec_deg"] + f * (ends[1]["dec_deg"] - ends[0]["dec_deg"])
        offs.append(float(np.hypot(_wrap_delta(r["ra"], ra)
                                   * math.cos(math.radians(r["dec"])),
                                   r["dec"] - dec)) * 3600.0)
    return float(np.median(offs))


# _accept_quality_ok's failure kinds: the per-tracklet clauses, and the seed
# check's (SeedCheck.fail) -- the ones a bad tracklet can cause (_repaired).
_PER_TRACKLET_FAILS = frozenset({"retention", "chi2"})
_SEED_FAILS = frozenset({"seed", "seed_missing", "veto"})


def _repair_blocked(a):
    """True unless dropping ``a``'s bad tracklets is a repair worth trying:
    some tracklet failed, and every other failure is one the seed check
    raised (a failure with no recorded kind blocks, conservatively)."""
    if a.quality_ok or not a.bad_tracklets:
        return True
    if not a.other_fail:
        return False
    return not a.fails or any(f.get("kind") not in _PER_TRACKLET_FAILS | _SEED_FAILS
                              for f in a.fails)


def _repaired(ctx, a, prior_linkage_used, depth):
    """One bad tracklet must cost ONE tracklet, not the whole linkage.

    Otherwise one failure fails the lot, and the recursion falls back to a
    state from BEFORE the offender, discarding every good tracklet taken with
    or after it.  Needed for 2016 EZ157; without it, one bad 3-obs tracklet
    cost four oppositions of correct links.

    Only when the failures are per-tracklet -- or the seed check, which a bad
    tracklet can itself cause: the seed's degradation is judged afresh on the
    re-vetted linkage, so a seed failure the offender did NOT cause still
    refuses.  (2008 GU20: a lone 2000 tracklet clipped 0/2 still pulled the
    seed to Delta-chi2 37.6 vs 30.6, and refusing the drop over it discarded
    three clean 2024 tracklets with it; dropped, the seed passes.)  Progress
    and night-coherence failures still block it.  First ask whether
    Yarkovsky holds them (:func:`_yarkovsky_retry`), then whether the orbit
    vouches for them (:func:`_with_vouched`), else drop exactly those
    tracklets and refit.  ONE round by design: clipping until something
    passes is how a search talks itself into a linkage.  Returns the
    acceptable :class:`Assessment`, or None."""
    bad = a.bad_tracklets
    if _repair_blocked(a):
        return None

    def _acceptable(tr, combined, obs_out, vouched=None):
        b = _assess(ctx, tr, combined, obs_out, prior_linkage_used,
                    vouched=vouched, dropped=a.dropped)
        _trace_assessment(ctx, b, head="re-vetting")
        return b if b.acceptable(ctx) else None

    rescued = _yarkovsky_retry(
        ctx, a.combined, f"[depth {depth}] a linked tracklet failed its own clause",
        lambda tr: _acceptable(tr, a.combined, a.obs_out, a.vouched),
        "RESCUED the linkage", a.trial)
    if rescued is not None:
        return rescued
    vouched = _with_vouched(ctx, a, prior_linkage_used, depth)
    if vouched is not None:
        return vouched
    drop = set().union(*(ids for _tag, ids in bad))
    keep_out = drop_obs_ids(a.obs_out, drop)
    if not keep_out.strip():
        return None                      # nothing linked would remain
    keep_comb = drop_obs_ids(a.combined, drop)
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.head(f"[depth {depth}] dropping {len(bad)} refused tracklet(s) "
                 f"({', '.join(tag for tag, _ids in bad)}) and re-vetting the "
                 f"remaining {_linked_tracklet_count(keep_out)}")
    tr2 = _fit(ctx, keep_comb)
    return _acceptable(tr2, keep_comb, keep_out) if tr2.ok else None


def _vouch(ctx, a, tag, ids):
    """Does the orbit VOUCH for linked tracklet ``ids``, which failed its own
    retention or chi2 clause?  Those clauses guard against a mislinkage the
    fit absorbed: an intruder bends the orbit toward itself, and leaving it out
    moves the orbit away.  A tracklet the orbit does not move toward at all
    poses no such question -- it identifies the object, and its larger
    residuals are this object's noisy astrometry.

    Refit without the whole tracklet as swept (lines the clip cut restored)
    and measure (:func:`itf.leaveout.influence`).  Vouched when

    * the leave-out prediction at its epochs is determined (1-sigma <=
      ``vouch_max_sigma_arcsec``);
    * adding the tracklet moves that prediction by at most
      ``vouch_shift_frac`` of the tracklet's miss, or the prediction's own
      sigma if larger -- the fit does not bend toward it;
    * at least ``vouch_ok_frac`` of its observations, and at least two (a
      rate), sit within ``vouch_ok_sigma`` of the leave-out prediction
      against their own uncertainty and the prediction's.  The others are
      then blunders inside a true tracklet -- the clip leaves them out of the
      fit as such -- not a misidentification.

    Returns ``(note, restored_lines)`` or None."""
    from . import leaveout
    cfg = ctx.cfg
    swept = (getattr(ctx, "swept_lines", None) or {}).get(ids)
    unit = [l for l in (swept or a.obs_out).splitlines()
            if l.strip() and obs_unique_id(l) in ids]
    if not unit:
        return None
    have = obs_ids(a.combined)
    restored = "".join(l + "\n" for l in unit if obs_unique_id(l) not in have)
    full = a.trial if not restored else _fit(ctx, a.combined + restored)
    loo = _fit(ctx, drop_obs_ids(a.combined + restored, ids)) if full.ok else full
    if not (full.ok and loo.ok):
        return None
    m = leaveout.influence(unit, full.fit, loo.fit, ctx.obscodes)
    if "error" in m or m["smaj"] is None:
        return None
    # each observation against its own (station-floored) uncertainty and the
    # prediction's; one the fit rejected takes its tracklet-mates' sigma
    smap = sigma_map_for(full.fit, ids, station_floor=True)
    known = list(smap.values())
    fallback = ((float(np.median([s[0] for s in known])),
                 float(np.median([s[1] for s in known]))) if known else (1.0, 1.0))
    sig = []
    for line, de, dn in m["per_obs"]:
        s_ra, s_dec = smap.get(obs_unique_id(line), fallback)
        sig.append(math.hypot(de / math.hypot(s_ra, m["smaj"]),
                              dn / math.hypot(s_dec, m["smaj"])))
    n_ok = sum(1 for z in sig if z <= cfg.vouch_ok_sigma)
    bar = max(cfg.vouch_shift_frac * m["miss"], m["smaj"])
    ok = (m["smaj"] <= cfg.vouch_max_sigma_arcsec and m["shift"] <= bar
          and n_ok >= 2 and n_ok >= cfg.vouch_ok_frac * len(sig))
    trc = _ctx_tracer(ctx)
    if trc.enabled:
        trc.cont(f"leave-out of {tag}: the orbit without it predicts it to "
                 f"{m['smaj']:.2f}\" (1-sigma); its obs sit "
                 + ", ".join(f"{z:.1f}" for z in sig)
                 + f" sigma off ({n_ok}/{len(sig)} within "
                 f"{cfg.vouch_ok_sigma:g}); adding it moves the prediction "
                 f"{m['shift']:.3f}\" (bar {bar:.3f}\") -- "
                 + ("VOUCHED: it does not pull the orbit, the rest are blunders"
                    if ok else "not vouched"))
    if not ok:
        return None
    return (f"vouched by leave-out ({n_ok}/{len(sig)} obs within "
            f"{cfg.vouch_ok_sigma:g} sigma; adding it moves the prediction "
            f"{m['shift']:.3f}\", 1-sigma {m['smaj']:.2f}\")", restored)


def _with_vouched(ctx, a, prior_linkage_used, depth=-1):
    """``a`` with every tracklet that failed ONLY its own clause vouched for
    (:func:`_vouch`) and restored whole, re-assessed -- or None when one of
    them is not vouched, another clause failed, or the result is still not
    acceptable.  The seed clauses still apply in full."""
    if a.quality_ok or a.other_fail or not a.bad_tracklets:
        return None
    vouched, restored = dict(a.vouched), ""
    for tag, ids in a.bad_tracklets:
        if ids in vouched:
            continue
        got = _vouch(ctx, a, tag, ids)
        if got is None:
            return None
        vouched[ids], more = got
        restored += more
    trial = a.trial if not restored else _fit(ctx, a.combined + restored)
    if not trial.ok:
        return None
    b = _assess(ctx, trial, a.combined + restored, a.obs_out + restored,
                prior_linkage_used, vouched=vouched, dropped=a.dropped)
    _trace_assessment(ctx, b, head="re-vetting")
    return b if b.acceptable(ctx) else None


def _reduced_linkage(ctx, a, tracklets_all, prior_linkage_used, depth):
    """A 1.75 that is not LINKED: submit the part of it that ISN'T fragile --
    the seed's own opposition extended (``a.same_opp``), which makes no
    cross-opposition claim.  The MPC can act on it today, and the
    cross-opposition tracklets are easier to judge once the longer arc has
    tightened the corridor.

    Taken here, from the tracklets in hand, and SAYS what it declined:
    otherwise the costly LOV variant fallback re-derives the same extension
    and silently replaces the refused finding and its warnings.  Needed for
    2006 QR56.  Judged like any linkage, its modern final sweep skipped (it
    would only re-find the declined tracklets).  Returns the LINKED result, or
    None."""
    tr, comb, obs_same = a.same_opp
    b = _assess(ctx, tr, comb, obs_same, prior_linkage_used)
    trc = _ctx_tracer(ctx)
    if not (b.consistent and b.profile_ok):
        if trc.enabled and b.gains and b.profile_ok:
            trc.head(f"[depth {depth}] the reduced linkage does not clear the "
                     f"per-tracklet quality gate either:")
            for note in b.notes:
                trc.cont(note)
        return None
    refused = a.trial
    kept = _linked_tracklet_count(obs_same)
    in_subset = obs_ids(obs_same)
    declined_lines = "".join(
        l + "\n" for l in a.obs_out.splitlines()
        if l.strip() and obs_unique_id(l) not in in_subset)
    _d, declined, _n = collect_ids(declined_lines)
    larger = opp_profile(ctx.pub_obs80 + a.obs_out)
    note = (
        f"a LARGER linkage was assembled and refused: adding "
        f"{len(declined)} further tracklet(s) "
        f"({', '.join(f'{t[0].strip()} {t[1]} {t[2]}' for t in declined)}) "
        f"reached {larger} oppositions at "
        f"{refused.rms:.3f}\" ({refused.used}/{refused.count} obs), too "
        f"fragile to submit (single-night oppositions on a short arc can be "
        f"combinatoric coincidences). What IS submitted here is the "
        f"non-fragile part -- the object's own opposition extended with "
        f"{kept} tracklet(s). The refused tracklets are worth a look by hand, "
        f"and are easier to judge against this longer arc.")
    if trc.enabled:
        trc.head(f"[depth {depth}] 1.75 not linked -> trying the SUBMITTABLE "
                 f"part: seed + {kept} same-opposition tracklet(s), "
                 f"{tr.used}/{tr.count} at {tr.rms:.3f}\", U {tr.U:.2f}, arc "
                 f"{tr.arclength_days:.1f}d ({len(declined)} cross-opposition "
                 f"tracklet(s) declined and reported)")
    out = _judge(ctx, b, tracklets_all, prior_linkage_used, depth=depth,
                 sweep="archive", lead=(note,))
    if out.verdict is not SearchVerdict.LINKED:
        return None
    ctx.log(note)
    out.vetting["refused_larger_linkage"] = {
        "opps": larger.legacy_code,
        "tracklets_declined": len(declined),
        "declined": [list(t) for t in declined],
        "rms": round(refused.rms, 4),
        "used": refused.used,
        "count": refused.count,
    }
    return out


def _strength(result):
    """How much a verdict is worth when two answers to one take compete: the
    verdict, then whether it can be reviewed, then how much ITF astrometry it
    carries (a deeper level's possible carries more than its parent's)."""
    return (result.verdict.rank, result.reviewable,
            count_obs(result.itf_obs80 or ""))


def _vet_terminal(ctx, trial, obs_out, trial_meta, child, tracklets_all,
                  prior_linkage_used, depth, hypothesis_obs80):
    """Vet ONE terminal candidate -- a take whose deeper recursion found
    nothing further -- into an accepted, possible (0.6) or rejected
    :class:`ItfSearchResult`.

    The take is cleaned of poor-fitting obs, then judged (:func:`_judge`)
    together with its alternatives, in order, stopping at the first LINKED:

      1. the linkage itself;
      2. the linkage minus the tracklets that failed their own clause
         (:func:`_repaired`);
      3. the seed's own opposition extended, when it is a 1.75
         (:func:`_reduced_linkage`);
      4. ``child``, the deeper recursion's reviewable 0.6.

    Without a LINKED answer the strongest verdict wins (:func:`_strength`)."""
    trc = _ctx_tracer(ctx)
    ctx.show_refusal = None                 # this take's answer, not the last one's
    combined = ctx.pub_obs80 + obs_out
    preclip = trial
    trial, combined, obs_out, err = _clip_poor_fitters(ctx, trial, combined,
                                                       obs_out)
    if err is not None:
        return ItfSearchResult(reason=err, telemetry=ctx.telemetry)
    # The child already swept THIS EXACT orbit and found no likely tracklet
    # (_clip_poor_fitters returns the same trial when it drops nothing): the
    # modern final sweep can only re-find the same nothing.
    sweep = "archive" if (trial is preclip and child.no_candidates) else "all"
    # the state a resumed search would start from: the seed plus what the
    # shallower depths already took, not this depth's additions
    ctx.prior_ids = set(ctx.pub_ids) | obs_ids(hypothesis_obs80)
    a = _assess(ctx, trial, combined, obs_out, prior_linkage_used, trial_meta,
                depth)
    _trace_assessment(ctx, a)
    best = _judge(ctx, a, tracklets_all, prior_linkage_used, trial_meta, depth,
                  sweep=sweep)

    def repaired():
        b = _repaired(ctx, a, prior_linkage_used, depth)
        return (None if b is None else
                _judge(ctx, b, tracklets_all, prior_linkage_used, depth=depth))

    def reduced():
        return (None if a.same_opp is None else
                _reduced_linkage(ctx, a, tracklets_all, prior_linkage_used, depth))

    for alt in (repaired, reduced, lambda: child if child.reviewable else None):
        if best.verdict is SearchVerdict.LINKED:
            break
        r = alt()
        if r is not None and _strength(r) > _strength(best):
            if r is child and trc.enabled:
                trc.head(f"depth {depth + 1}'s possible-but-unconfirmed linkage "
                         f"supersedes depth {depth}'s verdict")
            best = r
    # A LINKED verdict stops the alternatives above before the child is
    # weighed, so a deeper 0.6 that linked strictly MORE (further
    # oppositions this level's linkage lacks) would be lost: keep it on the
    # linkage (ItfConfig.prefer picks the lead at the end of the search).
    if _fuller_possible(best, child):
        best = _pair(best, child)
        if trc.enabled:
            trc.head(f"depth {depth + 1}'s possible links more of the ITF than depth "
                     f"{depth}'s linkage -- both kept")
    return best


def _trace_preamble(ctx, tracer, primaries):
    """The trace's opening block: the seed, the sweep's memory headroom, and
    any reason known up front that the search cannot report its likeliest find."""
    cfg, pub, telemetry = ctx.cfg, ctx.pub, ctx.telemetry
    tracer.round_begin(f"ITF search: {', '.join(primaries) or 'seed obs'}")
    tracer.field("seed", f"{pub.used}/{pub.count} obs, rms {pub.rms:.3f}\", "
                 f"U {pub.U:.2f}, {ctx.initial_opp_profile} opp(s), arc "
                 f"{pub.arclength_days:.1f}d", L_HEAD)
    # The memory floor silently decides mid-search whether the sweep is
    # truncated; state it up front so an operator reading a truncated sweep
    # (or a dead worker) sees whether the guard was armed and the room left.
    if cfg.sweep_min_free_mb:
        free = _candidates._mem_available_mb()
        # Quote the ENFORCED floor (a small container scales it below the
        # configured value), or an operator hunts for a fault that never happened.
        floor_mb = _sweep_floor_mb(cfg.sweep_min_free_mb)
        if floor_mb != cfg.sweep_min_free_mb:
            tracer.field("memory floor", (
                f"{floor_mb} MB enforced, scaled down from the configured "
                f"{cfg.sweep_min_free_mb} MB: this container is too small "
                f"to reserve that much and still search"), L_HEAD)
        if free is None:
            tracer.field("memory", "not measurable here -- the "
                         f"{floor_mb} MB sweep floor cannot "
                         "fire, and a sweep too big for this box will be "
                         "answered by the OOM killer", L_HEAD)
        elif free < floor_mb:
            tracer.warn(f"available memory {free} MB is ALREADY under the "
                        f"{floor_mb} MB sweep floor: this "
                        f"search will truncate its sweep immediately")
        else:
            # The cgroup breakdown rides along because MemAvailable counts about
            # half of ALL file pages as available, the cgroup side only
            # inactive_file: a large `active` term means the floor may fire on
            # memory the kernel would have handed back.
            cg = _mem_breakdown()
            extra = (f" [cgroup {cg['usage']}/{cg['limit']} MB used, "
                     f"file pages {cg['inactive_file']} inactive + "
                     f"{cg['active_file']} active]") if cg else ""
            tracer.field("memory", f"{free} MB available, "
                         f"{floor_mb} MB sweep floor{extra}",
                         L_STORY)
        # The user's verdict -- will this box cut the search short? -- recorded
        # by the caller so every surface can warn before the minutes are spent.
        if telemetry.get("capacity"):
            tracer.warn(f"memory-constrained: "
                        f"{telemetry['capacity']['reason']}")
    # A lone new night off a 1-opp seed is silenced above this cap (see
    # _show_lone_night clause 1); say so up front rather than at the rejection.
    if (ctx.initial_opp_profile.grade is OppProfileGrade.ONE
            and pub.U >= cfg.single_night_show_max_U):
        tracer.warn(f"seed U {pub.U:.2f} >= one-night show cap "
                    f"{cfg.single_night_show_max_U:g}: a SINGLE new-night "
                    f"tracklet cannot be reported by this search, however "
                    f"well it fits -- only a multi-night apparition can")


def _incomplete_sweep_report(cfg, telemetry, result):
    """Record on ``result`` that the corridor was NOT swept completely, naming
    the cause, and return the note (None when the sweep was complete).

    Reported on EVERY verdict, a find included: the unswept windows could hold
    more tracklets of the object, so the result is a floor, not a ceiling."""
    skipped = telemetry.get("windows_skipped", 0)
    cancelled = telemetry.get("windows_cancelled", 0)
    lowmem = telemetry.get("windows_lowmem", 0)
    if not skipped:
        return None
    # Name the CAUSE: raising a row budget does nothing for a server cancel
    # and makes a memory shortfall worse.
    if lowmem >= skipped:
        fix = (f"available memory fell under the "
               f"{_sweep_floor_mb(cfg.sweep_min_free_mb)} MB sweep floor "
               f"(ItfConfig.sweep_min_free_mb) -- this box does not have "
               f"room for this corridor, so give it more memory or narrow "
               f"the corridor (a tighter --mag-limit, a shorter epoch "
               f"range). Do NOT raise a row budget: retaining more rows is "
               f"what ran the box out")
    elif lowmem:
        fix = (f"{lowmem} to the {_sweep_floor_mb(cfg.sweep_min_free_mb)} MB "
               f"sweep memory floor (ItfConfig.sweep_min_free_mb) and "
               f"{skipped - lowmem} to a row budget or a cancelled query "
               f"-- the memory share is not fixed by raising a budget")
    elif cancelled >= skipped:
        fix = (f"the server cancelled the query after "
               f"{cfg.sweep_statement_timeout_sec}s "
               f"(ItfConfig.sweep_statement_timeout_sec) -- this corridor "
               f"is too expensive for the server as asked, so narrow it "
               f"(a tighter --mag-limit, a shorter epoch range) rather "
               f"than raising a budget")
    elif cancelled:
        fix = (f"{skipped - cancelled} to a sweep row budget "
               f"(ItfConfig.sweep_row_budget = {cfg.sweep_row_budget}, "
               f"variant_sweep_row_budget = {cfg.variant_sweep_row_budget}) "
               f"and {cancelled} to a query the server cancelled after "
               f"{cfg.sweep_statement_timeout_sec}s -- raising the budget "
               f"addresses only the first")
    else:
        fix = (f"a sweep row budget "
               f"(ItfConfig.sweep_row_budget = {cfg.sweep_row_budget}, "
               f"variant_sweep_row_budget = {cfg.variant_sweep_row_budget}) "
               f"-- raise it for a full sweep")
    # On a find, the point is that MORE may lie in the part never queried.
    tail = ("This is not a clean negative."
            if result.verdict is not SearchVerdict.LINKED else
            "The linkage found is still valid -- but the unswept windows "
            "were never checked, so this find may be incomplete.")
    note = (f"The corridor was NOT swept completely: {skipped} window(s) "
            f"were dropped, {fix}. {tail}")
    result.incomplete_sweep = {
        "windows": skipped, "lowmem": lowmem, "cancelled": cancelled,
        "budget": max(0, skipped - lowmem - cancelled),
        "memory_floor_mb": _sweep_floor_mb(cfg.sweep_min_free_mb),
        "message": note,
        # did it change the ANSWER, or only its completeness?
        "verdict_affected": result.verdict is not SearchVerdict.LINKED,
    }
    result.warnings.append(note)
    return note


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
class ItfDbUnreachable(RuntimeError):
    """The live SBN database could not be reached / authenticated."""


def itf_search_live(designations=None, *, seed_obs80=None, primary_desigs=None,
                    seed_obs=None, seed_records=None, cfg=None,
                    max_uncert_deg=None,
                    uncert_mult=None, do_submit=False, verbose=True, log=print,
                    trace=None, telemetry=None):
    """One ITF search against the LIVE SBN database, owning the connection
    lifecycle (``connect_pub`` -> ``DbSource`` with reconnect -> close); the
    entry point shared by the CLIs and the web endpoint.  Raises
    :class:`ItfDbUnreachable` when the database cannot be reached; otherwise
    it is :func:`itf_search`, which stays source-agnostic for the harness and
    the offline tests."""
    from .candidates import DbSource
    from .dbio import connect_pub
    cfg = cfg or ItfConfig()
    try:
        conn = connect_pub(cfg)
    except Exception as ex:
        raise ItfDbUnreachable(str(ex)) from ex
    try:
        source = DbSource(conn, cfg, reconnect=lambda: connect_pub(cfg))
        return itf_search(source, designations, seed_obs80=seed_obs80,
                          primary_desigs=primary_desigs, seed_obs=seed_obs,
                          seed_records=seed_records,
                          cfg=cfg,
                          max_uncert_deg=max_uncert_deg,
                          uncert_mult=uncert_mult, do_submit=do_submit,
                          verbose=verbose, log=log, trace=trace,
                          telemetry=telemetry)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def itf_search(source, designations=None, *, seed_obs80=None,
               primary_desigs=None, seed_obs=None, seed_records=None,
               cfg=None, obscodes=None,
               max_uncert_deg=None, uncert_mult=None, extrawidth=0.0,
               do_submit=False, co_credit=None, telemetry=None,
               verbose=True, log=print, collector=None, trace=None,
               dlog=None):
    """Search the ITF for tracklets belonging to a known object.

    ``source`` -- a candidate source (:class:`pyroorb.itf.candidates.DbSource`,
    or an :class:`~pyroorb.itf.candidates.OverlaySource`).
    ``designations`` -- object designation (or list) to fetch from the MPC and
    seed with; alternatively pass ``seed_obs80`` (raw OBS80 text) directly,
    with ``primary_desigs`` naming the family primaries for the submission
    document -- and, when the caller HAS the parsed observations behind that
    text (the web UI does), ``seed_obs``, so their ADES fields survive the
    80-column rendering (see itf/obsreg.py).  ``do_submit`` transmits a found
    linkage only when ``cfg.submit`` is ALSO true; otherwise every acceptance
    is a dry-run.
    ``trace`` -- optional detail-feed sink (callable(str) or
    :class:`itf.trace.Tracer`): every decision the search makes is narrated to
    it, elapsed-stamped (the CLI live trace / the web UI "see trace" stream).

    Returns an :class:`ItfSearchResult`.
    """
    cfg = cfg or ItfConfig()
    # NB: a replace(cfg, ...) copy made here never reaches DbSource, which
    # holds the caller's cfg; per-call retrieval knobs travel as
    # sweep_candidates parameters (row_budget, prefilter).
    tracer = as_tracer(trace)
    if tracer.enabled:
        # narrate the DB layer too (chunked index queries / scan queries)
        for s_ in source_chain(source):
            s_.trace = tracer
    telemetry = telemetry if telemetry is not None else {}
    telemetry.setdefault("obs_count", 0)
    telemetry.setdefault("U_diff", 0)
    if obscodes is None:
        from pyroorb.obs_sources import load_obscodes
        obscodes = load_obscodes()
    if max_uncert_deg is None:
        max_uncert_deg = cfg.max_uncert_deg
    if uncert_mult is None:
        uncert_mult = cfg.uncert_mult

    # ADES data for a line-based search (itf/obsreg.py): the seed's obs, then
    # candidate rows as they are collected; every fit is enriched through it.
    registry = AdesRegistry()
    if seed_obs80 is not None:
        pub_obs80 = split_obs80_long_lines(seed_obs80)
        if cfg.jitter:
            pub_obs80 = "".join(maybe_jitter(l, cfg) + "\n"
                                for l in pub_obs80.splitlines())
        primaries = list(primary_desigs or [])
        registry.add_obs(seed_obs or ())
        # An `obs_sbn` row IS an ADES record: taken as-is, not through a parse
        # that would drop the reported sigmas and the full band.
        registry.add_records(seed_records or ())
    else:
        if designations is None:
            raise ValueError("pass designations or seed_obs80")
        pub_obs80, primaries = _pub_obs80_from_fetch(designations, obscodes,
                                                     cfg, log=log,
                                                     registry=registry)
        if primary_desigs is not None:
            primaries = list(primary_desigs)

    if not pub_obs80.strip():
        return ItfSearchResult(reason="no seed astrometry", telemetry=telemetry)

    log(f"seed: {len(pub_obs80.splitlines())} obs lines, "
        f"primaries {primaries or '(none)'}")
    pub = fit_astrometry(pub_obs80, obscodes, registry=registry)
    if not pub.ok:
        return ItfSearchResult(reason=f"seed fit failed ({pub.error})",
                               telemetry=telemetry)
    log(f"seed fit: used {pub.used}/{pub.count}, rms {pub.rms:.3f}\", "
        f"U {pub.U:.2f}")
    seed_field = None
    if cfg.seed_perturber_screen:
        seed_field, pub = seed_perturbers(pub)
        if seed_field:
            log(f"seed perturbers: {pub.fit.force_model} -- every trial fit runs in it "
                f"(used {pub.used}/{pub.count}, rms {pub.rms:.3f}\")")

    # Memory cost of the SEARCH (started after the seed fit, so not the
    # fetch), for calibrating ItfConfig.sweep_min_free_mb.
    rss = RssSampler().start()

    ctx = SearchContext(
        source=source, cfg=cfg, obscodes=obscodes,
        pub_obs80=pub_obs80,
        pub_ids=obs_ids(pub_obs80),
        own_desigs=own_designations(pub_obs80, primaries),
        registry=registry,
        pub=pub,
        perturbers=seed_field,
        initial_opp_profile=opp_profile(pub_obs80),
        primary_desigs=primaries,
        max_uncert_deg=max_uncert_deg,
        uncert_mult=uncert_mult,
        do_submit=do_submit,
        co_credit=co_credit,
        telemetry=telemetry,
        log=log,
        verbose=verbose,
        collector=collector,
        dlog=dlog,
        trace=tracer,
        budget=(SearchBudget(cfg.no_progress_sec,
                             cfg.no_progress_sec * cfg.search_ceiling_mult,
                             cfg.search_setup_credit_max_sec,
                             cfg.min_fruitless_fits)
                if cfg.no_progress_sec > 0 else None),
    )
    # The capacity verdict is user-facing: recorded even without a trace.
    if cfg.sweep_min_free_mb:
        cap = _search_capacity(cfg.sweep_min_free_mb)
        if cap and cap["verdict"] != _CAPACITY_OK:
            telemetry["capacity"] = cap
    if tracer.enabled:
        _trace_preamble(ctx, tracer, primaries)
    result = _iterate(ctx, pub, pub.used, extrawidth, 0,
                      round_label="depth 0, linear corridor")

    # Magnitude FALLBACK, OFF by default (see ItfConfig.mag_limit_fallback):
    # an object predicted too faint at every window links nothing, so an
    # unlinked pass is re-swept out to mag_limit_deep unless the budget is spent.
    if (result.verdict is not SearchVerdict.LINKED and cfg.mag_limit_fallback
            and cfg.mag_limit_deep > cfg.mag_limit
            and not _budget_spent(ctx)):
        if tracer.enabled:
            tracer.head(f"nothing linked at mag_limit {cfg.mag_limit:g} "
                        f"({result.reason or 'nothing'}) -> re-sweeping the "
                        f"corridor out to {cfg.mag_limit_deep:g}")
        label = f"depth 0, linear corridor (mag {cfg.mag_limit_deep:g})"
        # stay thorough: the deeper reach is the corridor for later fallbacks
        cfg = ctx.cfg = replace(cfg, mag_limit=cfg.mag_limit_deep)
        deeper = _iterate(ctx, pub, pub.used, extrawidth, 0, round_label=label)
        if _verdict_rank(deeper) > _verdict_rank(result):
            result = deeper

    # Variant-orbit FALLBACK: the linear corridor follows one nominal
    # ephemeris, so an under-determined seed's tracklets can lie outside it.
    # The gate is "no linkage", NOT "nothing retrieved": a wide corridor
    # routinely returns distractors while the real tracklet sits outside it
    # (2019 AE65 lost its linkage that way).  Cost is bounded by the U gate,
    # variant_max_scan and the search budget.
    if (result.verdict is not SearchVerdict.LINKED
            and cfg.variant_corridor_U > 0 and pub.U >= cfg.variant_corridor_U):
        # Announce the phase change on both channels: the fan is usually most
        # of a weak seed's run.  "about", not "up to": refine_variant_fan may
        # ADD samples past variant_max_scan; _variant_scan prints the count.
        n_plan = min(cfg.variant_n_steps, cfg.variant_max_scan)
        say = (f"linear corridor exhausted at {time.monotonic() - tracer.t0:.0f}s "
               f"({result.reason or 'nothing linked'}); seed U {pub.U:.2f} "
               f">= {cfg.variant_corridor_U:g} -> scanning a fan of about "
               f"{n_plan} LOV variant orbits. This is normally most of the "
               f"remaining time")
        if tracer.enabled:
            tracer.head(say)
        if verbose:
            log(f"ITF search: {say}")
        scanned = _variant_scan(ctx, pub, pub.used, extrawidth, 0, "")
        if scanned is not None and _verdict_rank(scanned) > _verdict_rank(result):
            # ...unless the fallback's linkage holds strictly less than the
            # linear pass's 0.6: then both are kept (ItfConfig.prefer)
            held = (result.alternative if result.verdict is SearchVerdict.LINKED
                    else result)
            if _fuller_possible(scanned, held):
                result = _pair(scanned, held)
                if tracer.enabled:
                    tracer.head("the variant fan's linkage holds less than the "
                                "linear pass's possible -- both kept")
            else:
                result = scanned
    # a linkage and a fuller possible: ItfConfig.prefer picks which leads
    result = _choose(cfg, result)
    if result.alternative is not None and tracer.enabled:
        tracer.head(f"two answers kept (prefer={cfg.prefer}): the "
                    f"{'possible' if result.verdict is SearchVerdict.POSSIBLE else 'linkage'}"
                    f" leads, the other rides along")

    # An incomplete sweep is neither a clean negative nor a complete find.
    note = _incomplete_sweep_report(cfg, telemetry, result)
    if note:
        log(note)
        if tracer.enabled:
            tracer.warn(note)
    # A LINKED-but-unshown candidate must not vanish (see _recurrent_notes).
    for note in _recurrent_notes(ctx, result):
        result.diagnostics.append(note)
        log(note)
    if tracer.enabled:
        # closing block: the verdict without reconstructing it from the rounds
        tracer.blank()
        verdict = {SearchVerdict.LINKED: "LINKED",
                   SearchVerdict.POSSIBLE:
                       "POSSIBLE (reviewable, never submittable)"}
        tracer.head(f"=== verdict: "
                    f"{verdict.get(result.verdict, 'no linkage')}")
        # legacy code; itf_wording glosses it for the web
        tracer.field("status", f"{result.verdict.legacy_status} -- "
                     f"{result.reason or 'linkage found'}", L_HEAD)
        for w in result.warnings:
            tracer.warn(w)
        for d in result.diagnostics:
            tracer.field("NOTE", d, L_HEAD)
    # "Nothing found" must be distinguishable from "nothing looked at" (a
    # corridor emptied by the magnitude gate; see _note_mag_blackout).
    bl = telemetry.get("mag_blackout") if isinstance(telemetry, dict) else None
    if bl and result.verdict is not SearchVerdict.LINKED:
        result.warnings.insert(0, (
            f"NO part of the sky was searched: this object is predicted fainter "
            f"than the corridor's magnitude limit ({bl['limit']:g}) at every "
            f"epoch (brightest {bl['min_mag']:.2f}), so every window was "
            f"skipped and 'no match' here means 'not looked for'. "
            f"{bl['windows_if_deeper']} windows open up with a deeper limit -- "
            f"re-run with a fainter --mag-limit (or --thorough) to search it."))
    # hand out the ADES data so a caller re-fitting combined_obs80 does not fit
    # the 80-column shadow of observations we fitted in full
    result.registry = registry
    result.seed_U = pub.U
    if result.alternative is not None:
        # the other answer is refitted by its caller the same way
        result.alternative.registry = registry
        result.alternative.seed_U = pub.U
    # Capacity and memory figures ride on TELEMETRY, never on `warnings`:
    # warnings gate the submit click-through and are for doubts about the
    # LINKAGE; a server-capacity message there teaches users to click through.
    mem = rss.stop()
    if mem:
        telemetry["memory"] = mem
        if tracer.enabled:
            cg = mem.get("cgroup")
            # the ENFORCED floor, as in _trace_preamble
            floor_mb = _sweep_floor_mb(cfg.sweep_min_free_mb)
            tracer.field("memory used", (
                f"peak RSS {mem['peak_rss_mb']} MB "
                f"(+{mem.get('rss_growth_mb', 0)} MB over the seed fit), "
                f"available fell to {mem.get('min_available_mb', '?')} MB "
                f"against a {floor_mb} MB floor"
                + (f" [cgroup active_file {cg['active_file']} MB, charged as "
                   f"spent here but not by MemAvailable]" if cg else "")),
                L_STORY)
    return result
