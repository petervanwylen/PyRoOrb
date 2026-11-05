r"""Continuous LOV coordinate: k as a MEASUREMENT, not a bin label.

The variant scan retrieves with a fan of ~31 orbits along the Line of
Variations because each variant costs a database sweep.  ``lov_profile.py`` then
reads every candidate's LOV offset k* off *which variants served it* -- so a
candidate served by exactly one variant gets that variant's k EXACTLY, and
"these tracklets agree on k" degenerates into "these tracklets were served by
the same corridor", which on a severe seed is every one of ten thousand
candidates.  The doc's own limit ("k* resolution is fan-spacing-limited") is
that degeneracy.

But scoring needs no database.  Re-correcting the LOV on a fine grid and
propagating the whole grid in ONE shared ASSIST integration costs seconds, and
the k a tracklet actually implies is ~50x sharper than the fan spacing.
Measured on 2006 JE41 (16 obs, 14-day arc, U 8.80), whose five known ITF
tracklets span 2016-2024 and are all served by the single variant k = -0.971:

    tracklet   date         k* (fine)   resid @ k*
    P10uAhe    2016-04-05     -1.0100        43"
    C2Y3KX2    2020-07-26     -1.0125        28"
    P113xQh    2020-08-10     -1.0125        25"
    P21ZSGS    2024-10-01     -1.0050       121"
    P221x4l    2024-11-02     -1.0050       126"

Five nights over 8.6 years agreeing on k to 0.0075 sigma, where moving k by
0.3 sigma moves the object 5-14 DEGREES.

Two products, both measured against the 12,761 real candidates that variant's
corridor served:

- **the residual at a solved k** -- distance to the LOV CURVE rather than to
  one variant's centreline.  The corridor asks "inside a 2.5 deg tube?"; this
  asks "how far off the line of admissible orbits?", which is the cross-track
  miss no choice of k can absorb.  Ranked by distance to the served centreline
  (what the search had) the five truths sit at 85, 99, 132, 744, 832 of 12761;
  ranked by the residual at k* they sit at **9, 3, 2, 33, 38**;

- **multi-night k* clusters** -- candidates on distinct nights whose k* agree
  to a tolerance the fan could never resolve.  On that field the top cluster by
  night count is 8 nights at k = -1.010 and holds all five truths; the next is
  5 nights of junk.

Both are model-free: nothing here consults the ML ranker.  That is the point --
the BE559 record is explicit that no global ML nomination reaches the truth
(its scored P is 0.0042 against junk at 0.9+), so the priority variants are
chosen by CLUSTER STRENGTH, which is a shape junk mass cannot out-vote.

**Memory is a CHOICE here, not a property of the corridor.**  Candidates are
measured as they come off the union sweep and what survives is a fixed
``cfg.lov_refine_keep`` records of ``(ids, k*, residual, night)`` -- no
observations, no feature rows, no per-candidate profile object.  The ordering
is what makes it work: the LOV grid is re-corrected and propagated BEFORE the
sweep, when memory is still plentiful, and the streaming pass then allocates
only a small batch at a time.  That matters because the pass this was first
written to refine (``build_lov_profiles``) runs AFTER the sweep and dies of the
sweep memory floor on exactly the severe seeds it exists for -- on 2006 JE41 it
died at its very first check, 512 of 20,978 windows, three separate corridor
settings running.  So this does not depend on it: with profiles it refines
them, without them it supplies the trial sources itself.

The consumers are unchanged either way -- the variant visit order (``reorder``)
and the front-of-queue trial sources (``round_for`` -> the cluster joint trials and
centered singles in ``search._build_trials``) keep working as validated.
"""

from __future__ import annotations

import math
import time

import numpy as np
from astropy.time import Time

#: Julian date of the Unix epoch.
JD_UNIX = 2440587.5

#: Nodes per shared integration in :meth:`LovRefineProfile.prepare` (16-64 are
#: all ~6.5 s on 2005 UD562's 1071 nodes; 1 is 42 s, one sim 55 s).
LOV_PROP_CHUNK = 32


def _jd(epoch_sec):
    return epoch_sec / 86400.0 + JD_UNIX


def node_grid(variant_lov_sigmas, cfg):
    """The k values to re-correct: a fine grid around EVERY variant, deduped.

    A candidate is only ever measured against its own serving variant's
    neighbourhood -- ``cfg.lov_refine_halfwidth`` covers the reach of a
    ``variant_width_deg`` corridor comfortably (a 2.5 deg corridor spans
    ~0.1-0.3 sigma of k at ITF-era epochs) -- so the union of those
    neighbourhoods is all that is ever needed.  Near the centre of an
    equal-probability fan the neighbourhoods overlap heavily and the dedup does
    most of the work; the sparse tails are where the node count goes.

    Bounded by ``cfg.lov_refine_max_nodes``: this is setup spend, and a LOV
    re-correction is ~11 ms.  Over the budget the STEP is coarsened rather than
    the span truncated -- losing the tail variants entirely would silently
    un-refine exactly the far-out orbits the fan was widened to reach.
    """
    step = float(cfg.lov_refine_step)
    half = float(cfg.lov_refine_halfwidth)
    n_side = max(int(round(half / step)), 1)
    while True:
        out = set()
        for k in variant_lov_sigmas:
            for j in range(-n_side, n_side + 1):
                out.add(round(float(k) + j * step, 6))
        if len(out) <= cfg.lov_refine_max_nodes or step > 1.0:
            return sorted(out), step
        step *= 2.0
        n_side = max(int(round(half / step)), 1)


class LovRefineProfile:
    """A dense re-corrected LOV, propagated once, ready to measure candidates.

    ``prepare()`` does the expensive work: one re-correction per node
    (``arc_extend.fit_lov_pinned`` through ``fit.lov_refit``) and one shared
    integration of every node state over the time grid.  ``measure()`` then
    costs numpy only -- states at an arbitrary epoch come from cubic Hermite
    interpolation of that grid, which is exact to well under an arcsecond
    because a heliocentric state over four days is a very smooth thing
    (position AND velocity are interpolated, so the cubic is the object's
    actual trajectory, not a chord through it).

    The propagated block is held as **float32**: at ~3 au that carries ~2e-7 au
    of position, which is ~0.02" of sky, against residuals that decide things
    at 25-500".  It halves the one allocation that scales with the fan.
    """

    def __init__(self, fit, cfg, variant_lov_sigmas, log=print):
        self.fit = fit
        self.cfg = cfg
        self.variant_lov_sigmas = [float(k) for k in variant_lov_sigmas]
        self.log = log
        self.lov_sigmas = None                  # (m,) node offsets
        self.step = None
        self._G = None                  # (n, m, 6) heliocentric states
        self._g0 = None                 # epoch of grid cell 0
        self._slot = {}                 # cell index -> row of _G
        self._cols = {}                 # variant k -> node columns near it
        self.t_refit = self.t_prop = 0.0
        self.n_failed = 0

    # -- build ---------------------------------------------------------------
    def set_origin(self, jd_lo):
        """Fix grid cell 0 one step below ``jd_lo``.  Before any cell lookup."""
        self._g0 = float(jd_lo) - float(self.cfg.lov_refine_grid_days)

    def cell_of(self, jd):
        return int(math.floor((jd - self._g0)
                              / float(self.cfg.lov_refine_grid_days)))

    def cells_for_windows(self, windows):
        """Every grid cell (plus its Hermite partner) a window can land in.

        Derived from the WINDOWS, not from the candidates: the windows are
        known before the sweep and the candidates are not, which is what lets
        the whole propagation happen up front -- while there is still memory
        for it -- and the streaming measurement allocate nothing per candidate.
        """
        out = set()
        for w in windows:
            a = self.cell_of(_jd(w.unix_start))
            b = self.cell_of(_jd(w.unix_end))
            out.update(range(a, b + 2))
        return out

    def prepare(self, cells):
        """Re-correct the node grid and propagate it over ``cells``.

        Returns False (and stays unusable) when the LOV is unavailable -- a
        nongrav-augmented state, a degenerate covariance, no live evaluator --
        which is the same condition the variant fan itself declines on.
        """
        refit = getattr(self.fit, "lov_refit", None)
        if refit is None or self._g0 is None or not cells:
            return False
        u, sigma, refit_fn = refit()
        if u is None:
            return False
        lov_sigmas, step = node_grid(self.variant_lov_sigmas, self.cfg)
        state0 = np.asarray(self.fit._state, float)
        t0 = time.perf_counter()
        states, kept = [], []
        for k in lov_sigmas:
            try:
                v = np.asarray(refit_fn(state0 + (k * sigma) * u, u, sigma, k),
                               float)
            except Exception:
                self.n_failed += 1
                continue
            if v.size >= 6 and np.all(np.isfinite(v[:6])):
                states.append(v[:6])
                kept.append(k)
            else:
                self.n_failed += 1
        if len(kept) < 3:
            return False
        self.t_refit = time.perf_counter() - t0
        self.lov_sigmas = np.asarray(kept, float)
        self.step = step
        half = float(self.cfg.lov_refine_halfwidth)
        for k in self.variant_lov_sigmas:
            c = np.where(np.abs(self.lov_sigmas - k) <= half + 1e-9)[0]
            if len(c) >= 3:
                self._cols[k] = c
        if not self._cols:
            return False

        gd = float(self.cfg.lov_refine_grid_days)
        need = np.asarray(sorted(cells), dtype=np.int64)
        self._slot = {int(c): i for i, c in enumerate(need)}
        grid = self._g0 + need * gd
        t0 = time.perf_counter()
        from pyroorb.force_assist import helio_states_multi
        tt = Time(grid, format="jd", scale="utc").tdb.jd
        # In CHUNKS of neighbouring nodes, not one simulation.  IAS15's step is
        # shared by every particle in a sim, so with ~1000 planet-crossing LOV
        # nodes SOME node is always mid-encounter and the whole grid crawls at
        # its step: 2005 UD562, 1071 nodes x 2733 epochs, 55 s in one sim vs
        # 6.4 s in chunks of 32.  Neighbours in k share their encounter
        # history, so a chunk's step is still cheap.  Same answer: nodes
        # 0..891 agree to < 1e-8 au; only the far tail differs, where the
        # orbits are chaotic (ejected to 1e3-1e9 au) and a node integrated
        # ALONE disagrees with both versions by hundreds of au.
        states = np.asarray(states)
        c = int(LOV_PROP_CHUNK)
        self._G = np.concatenate(
            [np.asarray(helio_states_multi(self.fit.epoch_tdb,
                                           list(states[i:i + c]), tt),
                        dtype=np.float32)
             for i in range(0, len(states), c)], axis=1)
        self.t_prop = time.perf_counter() - t0
        return True

    @property
    def ready(self):
        return self._G is not None

    def serves(self, lov_sigma):
        return float(lov_sigma) in self._cols

    # -- interpolate ---------------------------------------------------------
    def _states_at(self, jd, cols):
        """(len(jd), len(cols), 6) states by cubic Hermite on the time grid."""
        G, h = self._G, float(self.cfg.lov_refine_grid_days)
        cell = np.floor((jd - self._g0) / h).astype(np.int64)
        a = np.fromiter((self._slot[int(c)] for c in cell), np.int64, len(cell))
        b = np.fromiter((self._slot[int(c) + 1] for c in cell), np.int64,
                        len(cell))
        s = ((jd - (self._g0 + cell * h)) / h)[:, None, None]
        Ga = G[np.ix_(a, cols)].astype(np.float64)
        Gb = G[np.ix_(b, cols)].astype(np.float64)
        p0, v0 = Ga[:, :, :3], Ga[:, :, 3:] * h
        p1, v1 = Gb[:, :, :3], Gb[:, :, 3:] * h
        s2 = s * s
        s3 = s2 * s
        pos = ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * v0
               + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * v1)
        vel = ((6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * v0
               + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * v1) / h
        return pos, vel

    # -- measure -------------------------------------------------------------
    def measure(self, items, obscodes):
        """``[(best_lov_sigma, resid_arcsec), ...]`` for candidate endpoint pairs.

        Each item is ``(k_variant, (jd, stn, ra, dec) first, (...) last)``.
        BOTH ends are used, so the measure is position AND motion: a tracklet
        whose position can be reached at some k but whose sky RATE cannot pays
        for it in the residual, which is what stops k from absorbing every
        mismatch it is handed.

        The minimum is found on the node grid and then refined ANALYTICALLY:
        over one step the offset is linear in k to excellent approximation, so
        the local gradient (central difference) gives the least-squares
        minimiser in closed form.  Grid argmin alone would be useless here --
        one 0.005-sigma step is 300-800" at ITF-era epochs, an order of
        magnitude above the residuals that discriminate.
        """
        from pyroorb.obs_sources import observer_positions_batch
        from pyroorb.forward_model import _radec_from_helio

        if not items:
            return []
        jds = np.asarray([e[0] for it in items for e in (it[1], it[2])], float)
        stns = [e[1] for it in items for e in (it[1], it[2])]
        ra_o = np.asarray([e[2] for it in items for e in (it[1], it[2])], float)
        dec_o = np.asarray([e[3] for it in items for e in (it[1], it[2])],
                           float)
        Robs = np.asarray(observer_positions_batch(jds, stns, obscodes))

        out = [(float("nan"), float("inf"))] * len(items)
        by_lov_sigma = {}
        for idx, it in enumerate(items):
            by_lov_sigma.setdefault(float(it[0]), []).append(idx)
        for k, idxs in by_lov_sigma.items():
            cols = self._cols.get(k)
            if cols is None:
                continue
            rows = np.asarray([q for i in idxs for q in (2 * i, 2 * i + 1)])
            pos, vel = self._states_at(jds[rows], cols)
            # ONE geometry call for the whole (endpoint x node) block.  Called
            # per endpoint it was 184 us a candidate -- 62 s over a 339k-
            # candidate fan -- because each call did the light-time solve on a
            # stack of only ~60 nodes and paid numpy's per-call overhead 678k
            # times.  _radec_from_helio's (m,3) path vectorises over whatever
            # it is handed, so the node axis is folded into it and the observer
            # position is broadcast to match.
            n_r, n_c = len(rows), len(cols)
            Rb = np.repeat(Robs[rows], n_c, axis=0)
            ra, dec = _radec_from_helio(pos.reshape(-1, 3), vel.reshape(-1, 3),
                                        Rb)
            ra = ra.reshape(n_r, n_c)
            dec = dec.reshape(n_r, n_c)
            dra = (((ra - ra_o[rows][:, None] + 180.0) % 360.0) - 180.0)
            off = np.empty((n_r, n_c, 2))
            off[:, :, 0] = (dra * np.cos(np.radians(dec_o[rows]))[:, None]
                            * 3600.0)
            off[:, :, 1] = (dec - dec_o[rows][:, None]) * 3600.0
            oA, oB = off[0::2], off[1::2]                # (nc, m, 2)
            tot = (oA ** 2).sum(2) + (oB ** 2).sum(2)    # (nc, m)
            j0 = np.clip(np.argmin(tot, axis=1), 1, len(cols) - 2)
            r = np.arange(len(idxs))
            den = 2.0 * self.step
            gA = (oA[r, j0 + 1] - oA[r, j0 - 1]) / den
            gB = (oB[r, j0 + 1] - oB[r, j0 - 1]) / den
            fA, fB = oA[r, j0], oB[r, j0]
            gg = (gA ** 2).sum(1) + (gB ** 2).sum(1)
            num = (fA * gA).sum(1) + (fB * gB).sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                d_lov_sigma = np.where(gg > 0, -num / gg, 0.0)
            # the linear model is only trusted inside one step
            d_lov_sigma = np.clip(d_lov_sigma, -self.step, self.step)
            rA = fA + d_lov_sigma[:, None] * gA
            rB = fB + d_lov_sigma[:, None] * gB
            resid = np.sqrt((rA ** 2).sum(1) + (rB ** 2).sum(1))
            kk = self.lov_sigmas[cols][j0] + d_lov_sigma
            for q, i in enumerate(idxs):
                out[i] = (float(kk[q]), float(resid[q]))
        return out


def _windows(items, d_lov_sigma, min_nights):
    """Fixed-width k windows holding at least ``min_nights`` distinct nights.

    ``items`` is ``[(best_lov_sigma, night, payload), ...]``.  Windows, not
    single-linkage chaining: on a dense junk field single linkage walks the
    whole axis one member at a time and reports one enormous "cluster", which
    is precisely the failure a tolerance is meant to prevent.  Each start
    element opens a window of width ``2*dk``; overlapping proposals are deduped
    by their centre afterwards.
    """
    items = sorted(items, key=lambda t: t[0])
    out = []
    n = len(items)
    for i in range(n):
        j = i
        while j + 1 < n and items[j + 1][0] - items[i][0] <= 2.0 * d_lov_sigma:
            j += 1
        grp = items[i:j + 1]
        nights = {g[1] for g in grp}
        if len(nights) >= min_nights:
            out.append((len(nights), grp))
    return out


def cluster_is_evidence(grp, cfg):
    """Is this k-agreement IMPROBABLE, or merely local?

    Agreement on k is evidence because widely separated epochs have no reason
    to agree -- that is the whole premise.  Tracklets from ONE station in ONE
    observing season have every reason to: they are neighbours on the sky, so
    they are neighbours on the line.  Counting nights treats "four nights in
    one season at one telescope" as the same evidence as "eight nights across
    twelve years and four telescopes", and they are not remotely the same.

    Measured on 2021 SA70, where this cost a recovery outright (0 of 5 planted
    tracklets, and TWO foreign tracklets linked, against a clean 5/5 with the
    pass off).  Every cluster it found sat at k = 2.82-2.93 with near-sequential
    archive ids, essentially all from station 705 -- one survey's dense field
    lying along the LOV by geometry.  And they were TIGHT: the best was 4
    nights at a median 30.4", better than the 2006 JE41 cluster that is
    correct.  So no bar on nights or on residual separates them; only the
    BASELINE does.  2006 JE41's winning cluster spans 2012-2024 across F51,
    G96, F52 and 703.

    Two requirements, both cheap and both structural rather than tuned:
    the members must span ``lov_refine_cluster_min_span_days`` (more than one
    opposition, so the agreement survives the object having moved), and they
    must come from at least ``lov_refine_cluster_min_stations`` distinct
    observatories (so one survey's cadence cannot supply the whole vote).
    """
    nights = sorted({g[2].night for g in grp})
    span = (nights[-1] - nights[0]) if nights else 0
    stations = {g[2].stn for g in grp}
    return (span >= int(cfg.lov_refine_cluster_min_span_days)
            and len(stations) >= int(cfg.lov_refine_cluster_min_stations))


class _Record:
    """What survives the sweep for one candidate: an identity to come back to,
    where it sits on the LOV, how far off it stays there, and which night it
    is.  Deliberately NOT the observations -- five slots, no dict."""
    __slots__ = ("ids", "best_lov_sigma", "resid", "night", "tag", "stn")

    def __init__(self, ids, best_lov_sigma, resid, night, tag, stn):
        self.ids = ids
        self.best_lov_sigma = best_lov_sigma
        self.resid = resid
        self.night = night
        self.tag = tag
        self.stn = stn


class StreamingLovRefine:
    """Measure the union sweep's candidates as they arrive; keep the best few.

    Retention is ``cfg.lov_refine_keep`` records by residual, so a severe seed's 30k
    candidates cost what a quiet one's 300 do.  The residual is the right thing
    to rank by: it is what actually separates the truth (top 38 of 12,761 on
    2006 JE41), it is model-free, and unlike an ML pre-gate it cannot discard a
    candidate the ranker happens to be blind to -- which the BE559 record says
    is the normal case for a real recovery, not the exception (P 0.0042 against
    junk at 0.9+).
    """

    def __init__(self, fit, cfg, pervar, obscodes, log=print):
        self.cfg = cfg
        self.obscodes = obscodes
        self.log = log
        self.fine = LovRefineProfile(fit, cfg, [float(k) for k, _w in pervar], log=log)
        self._lov_sigma_of_window = {}
        for k, windows in pervar:
            for w in windows:
                self._lov_sigma_of_window[id(w)] = float(k)
        self._nominal_lov_sigma = (min((float(k) for k, _w in pervar), key=abs)
                           if pervar else None)
        self._pending = []              # buffered, until a batch is worth doing
        self._kept = []                 # [_Record], bounded
        self._nominal_hashes = set()    # candidates the NOMINAL corridor served
        self.n_seen = 0
        self.t_measure = 0.0
        self.t_offer = 0.0          # grouping + buffering, inside the sweep
        self.n_windows = 0

    # -- build ---------------------------------------------------------------
    def prepare(self, windows):
        if not windows:
            return False
        self.fine.set_origin(min(_jd(w.unix_start) for w in windows))
        return self.fine.prepare(self.fine.cells_for_windows(windows))

    @property
    def ready(self):
        return self.fine.ready

    # -- consume the sweep ---------------------------------------------------
    def offer(self, window, rows, own_ids):
        """Buffer every tracklet in one swept window; measure when full.

        Cheap on purpose: no trial orbit, no feature row, no ML call -- the
        run-length grouping, an identity, and four numbers per endpoint.
        """
        from .search import _group_rows
        k = self._lov_sigma_of_window.get(id(window))
        if k is None or not self.fine.serves(k):
            return
        _t_off = time.perf_counter()
        self.n_windows += 1
        nominal = self._nominal_lov_sigma is not None and k == self._nominal_lov_sigma
        for t in _group_rows(rows):
            if len(t) < 2:
                continue
            a, b = t[0], t[-1]
            if a.get("ra") is None or a.get("stn") is None:
                continue
            self.n_seen += 1
            # NOTHING derived is computed here.  Identity (obs_unique_id over
            # every row, a frozenset), the display tag and the own-seed test
            # all wait for flush(), where they are paid only for the ~0.2% of
            # candidates that clear the residual cut.  Doing them per candidate
            # cost 2021 SA70 ~150 s of the ~196 s the whole pass added -- work
            # thrown away for 209,936 of 210,458 candidates.  The rows are
            # already held by the sweep's own cache, so keeping the reference
            # costs nothing.
            self._pending.append((
                k, t, int(a["epoch_sec"] // 86400.0), nominal,
                (_jd(a["epoch_sec"]), a["stn"], a["ra"], a["dec"]),
                (_jd(b["epoch_sec"]), b["stn"], b["ra"], b["dec"])))
        self.t_offer += time.perf_counter() - _t_off
        if len(self._pending) >= int(self.cfg.lov_refine_batch):
            self.flush(own_ids)

    def flush(self, own_ids=()):
        if not self._pending:
            return
        if not self.ready:
            self._pending = []
            return
        from .obs80_fields import obs_unique_id
        from .trace import row_label
        items = [(p[0], p[4], p[5]) for p in self._pending]
        t0 = time.perf_counter()
        try:
            meas = self.fine.measure(items, self.obscodes)
        except Exception as e:                                # noqa: BLE001
            self.log(f"LOV-refinement measure failed "
                     f"({type(e).__name__}: {e}); dropping this batch")
            self._pending = []
            return
        self.t_measure += time.perf_counter() - t0
        cut = float(self.cfg.lov_refine_resid_max)
        for p, (kk, rr) in zip(self._pending, meas):
            if not (math.isfinite(kk) and rr <= cut):
                continue
            t = p[1]
            ids = frozenset(obs_unique_id(r["obs80"]) for r in t)
            if own_ids and ids & own_ids:
                continue
            if p[3]:
                # the linear pass already swept the nominal corridor and
                # failed, so a candidate inside it carries no LOV information
                # (lov_profile._lov_sigma_informative, by identity rather than by profile)
                self._nominal_hashes.add(hash(ids))
            a = t[0]
            self._kept.append(_Record(
                ids, kk, rr, p[2],
                f'{row_label(a)}@{a.get("stn", "?")}', p[4][1]))
        self._pending = []
        keep = max(int(self.cfg.lov_refine_keep), 1)
        if len(self._kept) > 2 * keep:
            self._kept.sort(key=lambda r: r.resid)
            del self._kept[keep:]

    def release(self):
        """Drop the propagated block once the rounds are out of it.  The
        records are what the scan needs from here on, and the grid is the only
        allocation big enough to be worth handing back before 30 variant rounds
        run on top of it."""
        self.fine._G = None
        self.fine._slot = {}

    # -- produce -------------------------------------------------------------
    def clusters(self):
        """The multi-night k clusters, strongest (most nights) first."""
        self.flush()
        cfg = self.cfg
        pool = [(r.best_lov_sigma, r.night, r) for r in self._kept
                if hash(r.ids) not in self._nominal_hashes]
        groups = _windows(pool, float(cfg.lov_refine_cluster_dsigma),
                          int(cfg.lov_refine_cluster_min_nights))
        # Nights are VOTES and each vote is worth what it is tight: score a
        # cluster ``n_nights * log10(cut / median residual)``.
        #
        # Ranking by night count first is wrong, and measured wrong on 2006
        # JE41's live fan (338,669 candidates, 3,095 under the 500" cut): the
        # truth came back as 8 nights at a median 27" off the LOV, and was
        # OUTRANKED by 10 junk nights at 163" and tied by 8 at 138".  With
        # three of 39 variants reachable inside the budget, that ordering sent
        # the scan to the junk and it never arrived (budget spent at variant
        # #2, no link, where the same object links without any of this).
        # There are hundreds of thousands of candidates in a fan this size, so
        # a ten-night chance agreement is cheap; a ten-night agreement at 27"
        # is not.  The log keeps one very tight night from outvoting a real
        # multi-night pile.
        cut = max(float(cfg.lov_refine_resid_max), 1.0)

        def _score(grp):
            med = max(float(np.median([g[2].resid for g in grp])), 1.0)
            return len({g[1] for g in grp}) * math.log10(max(cut / med, 1.001))

        groups.sort(key=lambda t: (-_score(t[1]),
                                   float(np.median([g[2].resid
                                                    for g in t[1]]))))
        out, seen = [], []
        for nn, grp in groups:
            kc = float(np.median([g[0] for g in grp]))
            if any(abs(kc - s) <= cfg.lov_refine_cluster_dsigma for s in seen):
                continue
            if not cluster_is_evidence(grp, cfg):
                continue
            seen.append(kc)
            out.append((nn, kc, [g[2] for g in grp]))
            if len(out) >= int(cfg.lov_refine_priority_rounds):
                break
        return out

    def lookup(self):
        """``{ids: (k*, residual, nights_agreeing)}`` for the candidate ranking.

        ``nights_agreeing`` is the clustering evidence carried PER TRACKLET:
        how many OTHER distinct nights put a kept candidate within
        ``lov_refine_cluster_dsigma`` of this one's k*.  Per-tracklet rather than as a
        group because the ranking then never has to choose between clusters --
        the step that went wrong live, where a 10-night junk pile at 163" beat
        the 8-night truth at 27" and the scan spent its budget on the wrong
        variant.  A tracklet with no company scores 0 here and is ranked on its
        own residual and the model, which is the right answer for it.
        """
        self.flush()
        recs = sorted(self._kept, key=lambda r: r.best_lov_sigma)
        d_lov_sigma = float(self.cfg.lov_refine_cluster_dsigma)
        out = {}
        lo = 0
        for i, r in enumerate(recs):
            while recs[lo].best_lov_sigma < r.best_lov_sigma - d_lov_sigma:
                lo += 1
            j = i
            nights = set()
            while j < len(recs) and recs[j].best_lov_sigma <= r.best_lov_sigma + d_lov_sigma:
                nights.add(recs[j].night)
                j += 1
            for q in range(lo, i):
                nights.add(recs[q].night)
            nights.discard(r.night)
            prev = out.get(r.ids)
            if prev is None or r.resid < prev[1]:
                out[r.ids] = (r.best_lov_sigma, r.resid, len(nights))
        return out

    def rounds(self):
        """``(priority_ks, {k: _LovRound}, report)`` for the trial sources."""
        from .lov_profile import _LovRound
        cfg = self.cfg
        chosen = self.clusters()
        variant_lov_sigmas = list(self.fine.variant_lov_sigmas)
        max_members = max(int(cfg.lov_refine_cluster_max_members), 2)
        rounds, prio = {}, []
        for _nn, kc, members in chosen:
            if not variant_lov_sigmas:
                break
            k = min(variant_lov_sigmas, key=lambda v: abs(v - kc))
            rnd = rounds.get(k)
            if rnd is None:
                rnd = rounds[k] = _LovRound(k=k)
                prio.append(k)
            # one member per night, best residual first: a joint trial faces the
            # warm screen as a whole, and a same-night pair adds no independent
            # evidence while doubling the chance one bad member sinks it
            best_by_night = {}
            for r in sorted(members, key=lambda r: r.resid):
                best_by_night.setdefault(r.night, r)
            pick = sorted(best_by_night.values(),
                          key=lambda r: r.resid)[:max_members]
            if len(pick) >= 2:
                rnd.cluster_groups.append({r.ids for r in pick})
            for r in pick:
                if r.ids not in rnd.center_ids:
                    rnd.center_ids.append(r.ids)
        # the rest of each priority variant's low-residual candidates get the
        # centered-singles trial source, best residual first
        for k, rnd in rounds.items():
            near = sorted((r for r in self._kept
                           if abs(r.best_lov_sigma - k) <= cfg.lov_refine_halfwidth
                           and hash(r.ids) not in self._nominal_hashes),
                          key=lambda r: r.resid)
            for r in near:
                if r.ids not in rnd.center_ids:
                    rnd.center_ids.append(r.ids)
        resids = [r.resid for r in self._kept]
        report = {
            "n_nodes": (int(len(self.fine.lov_sigmas))
                        if self.fine.lov_sigmas is not None else 0),
            "step": round(self.fine.step, 6) if self.fine.step else None,
            "n_seen": self.n_seen,
            "n_kept": len(self._kept),
            "t_refit": round(self.fine.t_refit, 2),
            "t_prop": round(self.fine.t_prop, 2),
            "t_measure": round(self.t_measure, 2),
            "t_offer": round(self.t_offer, 2),
            "n_windows": self.n_windows,
            "resid_best": round(min(resids), 1) if resids else None,
            "clusters": [{"k": round(kc, 4), "n_nights": nn,
                          "n_members": len(ms),
                          "resid": round(float(np.median([m.resid
                                                          for m in ms])), 1),
                          "span_days": (max(m.night for m in ms)
                                        - min(m.night for m in ms)),
                          "stations": sorted({m.stn for m in ms}),
                          "tags": sorted(m.tag for m in ms)[:6]}
                         for nn, kc, ms in chosen],
        }
        return prio, rounds, report


class LovRefineRounds:
    """A stand-in for the two :class:`~pyroorb.itf.lov_profile.LovProfileSet` methods the
    scan consumes, for when its pass could not run at all -- which the sweep
    memory floor makes the normal case on exactly the severe seeds this is for.
    Same contract, cluster evidence only."""

    def __init__(self, prio, rounds):
        self._priority_lov_sigmas = list(prio)
        self._rounds = dict(rounds)

    def reorder(self, pervar):
        pos = {k: i for i, k in enumerate(self._priority_lov_sigmas)}
        return sorted(pervar,
                      key=lambda e: (pos.get(float(e[0]), len(pos)),
                                     abs(e[0])))

    def round_for(self, lov_sigma):
        return self._rounds.get(float(lov_sigma))

    def summary(self, top=8):
        return {"n_tracklets": 0, "n_sharp": 0, "sharp": [], "clusters": [],
                "variants": [{"k": round(k, 3),
                              "n_centered": len(self._rounds[k].center_ids)}
                             for k in self._priority_lov_sigmas[:top]]}


def install(lov_profiles, prio, rounds):
    """Put the cluster-nominated rounds in FRONT of whatever the LOV profile pass
    nominated, keeping its own behind them rather than replacing them.

    They are different kinds of evidence -- several nights agreeing on one k,
    versus one candidate a model likes -- the ML nomination was validated as it
    stands, and a fan with no cluster at all must degrade to exactly today's
    behaviour.  With no profile set at all (its pass died) the clusters stand
    alone.
    """
    if not rounds:
        return lov_profiles
    if lov_profiles is None:
        return LovRefineRounds(prio, rounds)
    merged, order = dict(rounds), list(prio)
    for k, rnd in lov_profiles._rounds.items():
        if k not in merged:
            merged[k] = rnd
            order.append(k)
    lov_profiles._rounds = merged
    lov_profiles._priority_lov_sigmas = order
    return lov_profiles
