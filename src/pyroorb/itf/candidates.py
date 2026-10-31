"""Per-window candidate retrieval for the ITF search.

Two database paths, switchable via ``ItfConfig.candidate_path`` (both return
the same row shape so the search core is path-agnostic):

* ``"index"`` (default) -- the fast path: prefilter the precomputed
  ``itf.itf_tracklets`` table by healpix disc + epoch + dec band (all
  index-friendly), apply the exact quadratic-corridor test in Python on the
  small candidate set, then refetch the survivors' obs by ``trkid``.  Also
  produces the per-tracklet **certainty match-chi2** (position + motion,
  normalized by the tracklet's own fitted uncertainty combined with the
  ephemeris sigma) used for try-best-first ordering.
* ``"scan"`` -- the legacy path: box + two quadratic corridors straight
  against ``obs_sbn`` (+ ``obs_noorbit``) per window.  Kept for A/B of
  candidate completeness against the index.

:class:`OverlaySource` wraps either path (or nothing, for offline runs) and
injects synthetic tracklets subject to the SAME predicates -- the mechanism
the recovery test harness uses to plant an object's own tracklets in the ITF.

Candidate obs rows are dicts: ``obs80, ra, dec, epoch_sec, trksub, stn,
trkid`` (trkid None on the scan path, where obs_sbn isn't re-consulted),
plus the per-obs ADES payload (``obsid, rmsra, rmsdec, rmsmag, rmstime,
logsnr, mode, astcat`` -- see ``IndexSchema.ades_select``), which the search
banks in an :class:`itf.obsreg.AdesRegistry` and puts back on the observations
it fits.  A row from ``obs_sbn`` IS an ADES record: the table's columns are the
ADES field names.  Rows from a synthetic/overlay source carry none of it, and
everything downstream degrades cleanly to the 80-column data.
"""

import bisect
import math

import numpy as np

from pyroorb import deadline
from pyroorb.healpix import query_discs_nest

from .index import IndexConfig
from .memory import available_mb, sweep_floor_mb
from .trace import L_STORY, L_FIREHOSE


def _mem_available_mb():
    """Memory available to this process in MB, or None where it cannot be
    measured (non-Linux, no cgroup limit: the sweep memory floor simply never
    fires).  The CONTAINER's headroom counts as much as the host's free memory
    -- see :mod:`itf.memory`, and note that reading only /proc/meminfo left this
    floor inert on every hosted deploy."""
    return available_mb()


# ---------------------------------------------------------------------------
# Shared window predicates (used identically by the DB paths and the overlay)
# ---------------------------------------------------------------------------
#: Absolute slack added to every corridor, in DEGREES (20").  A multiplier
#: alone scales with the covariance, so the corridor is narrowest exactly when
#: the search is most confident -- and a 3-sigma along-track tail then sits
#: outside a 2.5-sigma corridor by a few arcsec.  2009 MG10 (q 30.9 au) is the
#: measured case: its seed corridor served all five planted tracklets, and the
#: IMPROVED orbit -- after three correct links tightened the covariance to
#: +-17.9" -- excluded the remaining two, which sat 52.9" out in RA against a
#: 47.6" tolerance.  Five arcseconds.
#:
#: Additive rather than a floor because it is graceful: a +-5" corridor becomes
#: +-32" instead of jumping to whatever the floor is, and a degrees-wide one is
#: untouched in proportion.  Cheap by construction, too -- the corridors this
#: widens are the narrow ones, where 20" of extra sky holds almost no rows.
CORRIDOR_PAD_DEG = 20.0 / 3600.0


def window_tolerances(window, uncert_mult, extrawidth, pad=CORRIDOR_PAD_DEG):
    """Corridor half-widths in degrees (RA-coordinate, Dec) for a window:
    scaled 1-sigma uncertainty + a fixed absolute pad + explicit widening +
    speed allowance (the 0.02*speed term absorbs trailing/timing error on fast
    movers).  See :data:`CORRIDOR_PAD_DEG` for why the pad is there."""
    cosd = max(window.uncert_ra / window.uncert, 1.0) if window.uncert else 1.0
    ra_tol = (window.uncert_ra * uncert_mult + pad * cosd + extrawidth
              + 0.02 * window.max_speed)
    dec_tol = window.uncert * uncert_mult + pad + extrawidth + 0.02 * window.max_speed
    return ra_tol, dec_tol


BOX_PAD_DEG = 0.04          # the legacy box's fixed extra slack

# Windows per HEALPix disc walk (window_healpix_discs).  The walk holds every
# disc's working arrays at once; on 2000 windows of a weak seed's width mix,
# peak memory is 655 MB in one call, 141 MB at 256, 69 MB at 64, 53 MB at 16,
# at the same speed down to 64 (0.7 s) -- identical pixel sets throughout.
DISC_BATCH = 64


def tracklet_in_window(feat, window, ra_tol, dec_tol):
    """Index-path predicate: does a tracklet feature row fall in the window's
    corridor?  Epoch/dec-band/corridor exactly as the fast-path SQL + Python
    refine; the tracklet's own half-extent widens the tolerance so a short arc
    straddling the corridor edge is still admitted.  Wrap-safe in RA.

    ``feat`` needs: epoch_mid/start/end_sec, center_ra, center_dec,
    rate_deg_day.  Returns the annotated dict (with ``_pred_ra/_pred_dec/_dt``
    for the chi2) or None.
    """
    es = feat["epoch_mid_sec"]
    if not (window.unix_start <= es <= window.unix_end):
        return None
    if not (window.min_dec - dec_tol - BOX_PAD_DEG
            <= feat["center_dec"]
            <= window.max_dec + dec_tol + BOX_PAD_DEG):
        return None
    pred_ra, pred_dec = window.predict(es)
    half_ext = ((feat["rate_deg_day"] or 0.0)
                * abs((feat["epoch_end_sec"] or es)
                      - (feat["epoch_start_sec"] or es)) / 2.0 / 86400.0)
    dra = ((feat["center_ra"] - pred_ra + 180.0) % 360.0) - 180.0
    if abs(dra) > ra_tol + half_ext:
        return None
    if abs(feat["center_dec"] - pred_dec) > dec_tol + half_ext:
        return None
    out = dict(feat)
    out["_pred_ra"] = pred_ra
    out["_pred_dec"] = pred_dec
    out["_dt"] = es - window.unix_cur
    return out


def motion_mismatch(feat, window):
    """``(mismatch_arcsec, parallax_allowance_arcsec)`` between a tracklet's OWN
    recorded motion and the corridor's predicted motion over the tracklet's own
    arc -- the index-only twin of the quantity the search's trajectory gate
    computes from observations (``search._collect_tracklets``'s ``rate_resid_arcsec`` and its
    ``traj_unc`` allowance).

    The index stores the weighted-LSQ rate (``dra_cosdec_rate``/``ddec_rate``)
    fitted to exactly the observations the gate would use, so this needs no obs.
    Returns ``None`` when the row cannot support the comparison (no fitted rate,
    no epoch span, a single-epoch tracklet)."""
    es0, es1 = feat.get("epoch_start_sec"), feat.get("epoch_end_sec")
    v_e, v_n = feat.get("dra_cosdec_rate"), feat.get("ddec_rate")
    if es0 is None or es1 is None or v_e is None or v_n is None:
        return None
    arc_sec = abs(es1 - es0)
    if arc_sec <= 0:
        return None
    arc_days = arc_sec / 86400.0
    cosd = math.cos(math.radians(feat["center_dec"]))
    p0_ra, p0_dec = window.predict(es0)
    p1_ra, p1_dec = window.predict(es1)
    pred_e = (((p1_ra - p0_ra + 180.0) % 360.0) - 180.0) * cosd
    pred_n = p1_dec - p0_dec
    mismatch = math.hypot(pred_e - v_e * arc_days,
                          pred_n - v_n * arc_days) * 3600.0
    # the gate's own allowance for a geocentric-speed object's parallax drift
    traj_unc = 0.00026 / 86400.0 * arc_sec / window.dist_au * 57.3 * 3600.0
    return mismatch, traj_unc


def motion_bound_ok(feat, window, cfg):
    """Could this tracklet POSSIBLY pass the search's trajectory gate?  A
    conservative pre-filter computed from index columns alone, so the obs of a
    tracklet that provably cannot be a candidate are never fetched.

    The search's real gate (``search._collect_tracklets``) compares the
    tracklet's OBSERVED displacement between its first and last observation with
    the corridor's PREDICTED displacement over the same interval, and calls
    anything past ``cfg.maybe_arcsec`` (+ a parallax-drift allowance)
    "unlikely" -- dropped, never fitted.  That comparison is
    :func:`motion_mismatch` (one implementation, two callers -- this and
    :func:`cap_rank_key`); here it is turned into a conservative BOUND.

    The index already stores the tracklet's
    own fitted motion (``dra_cosdec_rate``/``ddec_rate``, from the weighted-LSQ
    fit over exactly those observations) and the rms of that fit, so the same
    quantity can be bounded before any obs are read: the observed endpoints
    differ from the fitted line by the fit residuals, and
    ``prefilter_motion_slack_rms x fit_rms + prefilter_motion_slack_arcsec``
    covers that difference with a deliberately paranoid margin.

    Returns True to KEEP (fetch its obs) -- including whenever the bound cannot
    be computed at all (no fitted rate, no fit rms, a single-epoch tracklet), so
    a tracklet is only ever dropped on positive evidence.

    Measured on 2015 BE559's severe corridor (43312 corridor survivors): rules
    out 77% at a 10x-rms slack (85% at 2x, 65% even at 25x).

    Parked for months because at the time it saved no measurable time -- the
    per-window cap was already truncating the obs fetch it would have shortened.
    That stopped being true once the cap ranked by the right key and the
    variant sweep stopped being truncated: BE559's union sweep now
    fetches 308283 obs rows for 87799 corridor survivors, ~60 s of its ~130 s.

    NB this is NOT candidate-identical -- ``search._group_rows`` groups obs by
    trksub + station + 12 h rather than by tracklet id, so skipping one
    tracklet's obs re-groups its neighbours.  The acceptance test is therefore
    RECALL (the recovery suite, live and offline), not set identity.

    Applied in :meth:`DbSource._survivors`, which every sweep path shares and
    which feeds the obs fetch directly -- and, being ahead of the per-window cap,
    it also lets that cap bite on noise instead of on candidates."""

    rms = feat.get("fit_rms_arcsec")
    if rms is None:
        return True                         # unbounded -> keep
    mm = motion_mismatch(feat, window)
    if mm is None:
        return True     # no rate / single epoch -> unbounded (the gate's own 9999)
    rate_resid_arcsec, traj_unc = mm
    slack = cfg.prefilter_motion_slack_rms * rms + cfg.prefilter_motion_slack_arcsec
    return rate_resid_arcsec - slack <= cfg.maybe_arcsec + traj_unc


def obs_in_window(row, window, ra_tol, dec_tol):
    """Scan-path predicate: the legacy per-obs box + time-range + quadratic
    corridor test (the SQL WHERE clause, in Python -- used by the overlay so
    injected obs face exactly what the database would apply)."""
    es = row["epoch_sec"]
    if not (window.unix_start <= es <= window.unix_end):
        return False
    if not (window.min_ra - ra_tol - BOX_PAD_DEG <= row["ra"]
            <= window.max_ra + ra_tol + BOX_PAD_DEG):
        return False
    if not (window.min_dec - dec_tol - BOX_PAD_DEG <= row["dec"]
            <= window.max_dec + dec_tol + BOX_PAD_DEG):
        return False
    pred_ra, pred_dec = window.predict(es)
    if not (pred_ra - ra_tol <= row["ra"] <= pred_ra + ra_tol):
        return False
    if not (pred_dec - dec_tol <= row["dec"] <= pred_dec + dec_tol):
        return False
    return True


def match_chi2(cand, window):
    """Certainty match-chi2 of an (annotated) candidate tracklet vs the
    window prediction: an on-sky position term at epoch_mid plus a motion
    term against the corridor derivative, each normalized by the tracklet's
    own fitted sigma combined with the ephemeris sigma.  Lower = more certain;
    wrong-motion tracklets are heavily penalized (the discriminator the legacy
    distance + rate-residual score lacked).  Returns ``(total, pos_chi2, motion_chi2)``."""
    cdec = cand["center_dec"]
    cosd = np.cos(np.radians(cdec))
    eph_as = max(window.uncert, 1e-6) * 3600.0                # ephemeris 1sigma, arcsec

    d_e = ((((cand["center_ra"] - cand["_pred_ra"] + 180.0) % 360.0) - 180.0)
           * cosd * 3600.0)
    d_n = (cdec - cand["_pred_dec"]) * 3600.0
    sig_e = np.hypot(cand["sig_pos_e_arcsec"] or 0.0, eph_as)
    sig_n = np.hypot(cand["sig_pos_n_arcsec"] or 0.0, eph_as)
    pos_chi2 = (d_e / sig_e) ** 2 + (d_n / sig_n) ** 2

    dt = cand["_dt"]
    pred_ve = (2 * window.ra_quad[0] * dt + window.ra_quad[1]) * cosd * 86400.0
    pred_vn = (2 * window.dec_quad[0] * dt + window.dec_quad[1]) * 86400.0
    # ephemeris motion sigma ~ position sigma spread over the tracklet arc,
    # floored so a 2-obs tracklet's exact-but-fragile velocity isn't over-trusted
    arc = max(abs((cand["epoch_end_sec"] or 0) - (cand["epoch_start_sec"] or 0)), 1.0)
    eph_v = eph_as / 3600.0 / (arc / 86400.0)
    sig_ve = np.hypot(cand["sig_vel_e_degday"] or 0.0, eph_v)
    sig_vn = np.hypot(cand["sig_vel_n_degday"] or 0.0, eph_v)
    d_ve = (cand["dra_cosdec_rate"] or 0.0) - pred_ve
    d_vn = (cand["ddec_rate"] or 0.0) - pred_vn
    motion_chi2 = (d_ve / sig_ve) ** 2 + (d_vn / sig_vn) ** 2

    return float(pos_chi2 + motion_chi2), float(pos_chi2), float(motion_chi2)


def cap_rank_key(cand, window):
    """Sort key for ``cfg.max_candidates_per_window``: which candidates a window
    keeps when it has more than the cap.  Takes an ANNOTATED candidate (the
    output of :func:`tracklet_in_window`), as the sweep hands it.

    The key must agree with the gate the survivors face next -- the MOTION test
    (``search._collect_tracklets`` labels a tracklet "unlikely" and discards it
    on motion mismatch, never on distance from the centreline).  So rank by
    :func:`motion_mismatch`, less mismatch first, and put rows the index cannot
    evaluate (no fitted rate, single epoch) FIRST, since nothing rules them out.
    ``match_chi2`` breaks ties.

    It used to be ``match_chi2`` alone, which is wrong precisely where the cap
    bites.  Both of that score's terms are normalised by the WINDOW's sigma, so
    on the degrees-wide corridor of an under-determined seed (2015 BE559: +-4000")
    the motion term collapses to ~0 and the order degenerates to "closest to the
    centreline" -- while such a seed's true tracklets are deliberately FAR
    off-centreline (the whole reason for ``uncert_mult`` 2.5 and the LOV variant
    scan).  Measured over the 24 BE559 windows that exceed the cap: of 13788
    gate-passing candidates the old key retained 5641 (41%), this one 10989
    (80%), at the same cap and cost."""
    mm = motion_mismatch(cand, window)
    if mm is None:
        return (-1.0, match_chi2(cand, window)[0])     # unrankable -> keep first
    mismatch, traj_unc = mm
    return (max(mismatch - traj_unc, 0.0), match_chi2(cand, window)[0])


def _box_discs(min_ra, max_ra, min_dec, max_dec, pad_deg):
    """Unit-vector centres (n, 3) + angular radii (deg) of discs covering
    padded boxes -- 3D, so robust to RA wrap and the poles."""
    min_ra, max_ra, min_dec, max_dec, pad_deg = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (min_ra, max_ra, min_dec, max_dec,
                                               pad_deg)))
    ras = np.radians(np.stack([min_ra - pad_deg, max_ra + pad_deg,
                               min_ra - pad_deg, max_ra + pad_deg], axis=-1))
    decs = np.radians(np.stack([min_dec - pad_deg, min_dec - pad_deg,
                                max_dec + pad_deg, max_dec + pad_deg], axis=-1))
    decs = np.clip(decs, -np.pi / 2, np.pi / 2)
    cosd = np.cos(decs)
    corners = np.stack([cosd * np.cos(ras), cosd * np.sin(ras), np.sin(decs)],
                       axis=-1)                                 # (n, 4, 3)
    center = corners.mean(axis=-2)
    n = np.sqrt((center * center).sum(axis=-1))
    degenerate = n == 0                         # -> whole sky
    center = np.where(degenerate[..., None], [0.0, 0.0, 1.0],
                      center / np.where(degenerate, 1.0, n)[..., None])
    cosang = np.clip((corners * center[..., None, :]).sum(axis=-1), -1.0, 1.0)
    radius = np.where(degenerate, 180.0, np.degrees(np.arccos(cosang.min(axis=-1))))
    return center, radius


def _box_disc(min_ra, max_ra, min_dec, max_dec, pad_deg):
    """One box's covering disc: unit-vector centre + radius (deg)."""
    center, radius = _box_discs(min_ra, max_ra, min_dec, max_dec, pad_deg)
    return center, float(radius)


def window_healpix_discs(windows, tols, index_cfg, as_sets=True):
    """Covering healpix pixel sets for windows' padded boxes, one per
    ``(ra_tol, dec_tol)`` in ``tols`` -- computed together, which is what keeps
    a whole sweep's worth of disc lookups cheap.  ``as_sets=False`` returns
    compact int32 arrays instead, for a caller that holds many at once."""
    if not windows:
        return []
    if not index_cfg.healpix_nested:
        raise ValueError("the tracklet index is NESTED-ordered")
    pad = np.array([max(rt, dt) for rt, dt in tols]) + BOX_PAD_DEG
    center, radius = _box_discs([w.min_ra % 360.0 for w in windows],
                                [w.max_ra % 360.0 for w in windows],
                                [w.min_dec for w in windows],
                                [w.max_dec for w in windows], pad)
    radius = np.radians(np.atleast_1d(radius))
    center = np.asarray(center).reshape(-1, 3)
    out = []
    # In batches (DISC_BATCH): the HEALPix walk holds every disc's working
    # arrays at once, and near the max_uncert_deg cap a disc is half the sky
    # at nside 64, so one call over a whole sweep peaks at gigabytes.
    for lo in range(0, len(radius), DISC_BATCH):
        pix = query_discs_nest(index_cfg.healpix_nside, center[lo:lo + DISC_BATCH],
                               radius[lo:lo + DISC_BATCH])
        out.extend(np.asarray(p, dtype=np.int32) if not as_sets else set(p.tolist())
                   for p in pix)
    return out


def window_healpix_disc(window, ra_tol, dec_tol, index_cfg):
    """Covering healpix pixels for a window's padded box."""
    return window_healpix_discs([window], [(ra_tol, dec_tol)], index_cfg)[0]


def _row_sort_key(r):
    # the legacy ORDER BY DATE(obstime), trksub, stn, obstime
    return (int(r["epoch_sec"] // 86400), r["trksub"] or "", r["stn"] or "",
            r["epoch_sec"])


def _assemble_window(window, cands, obs_by_trkid, want_ranking):
    """One window's ``(obs rows, match-chi2 by tracklet, features by tracklet)``
    from its corridor survivors and the fetched obs -- the index path's yield
    shape, built the same way whether the feature rows came from a fresh query
    or a cached box."""
    rows, chi2, feats_by = [], {}, {}
    for c in cands:
        rows.extend(obs_by_trkid.get(c["tracklet_id"], []))
        feats_by[c["tracklet_id"]] = c
        if want_ranking:
            chi2[c["tracklet_id"]] = match_chi2(c, window)[0]
    rows.sort(key=_row_sort_key)
    return rows, chi2, feats_by


# The index feature columns every candidate query returns (aliased `t` so the
# set-join can qualify them) -- the corridor test, match-chi2, ML features and
# magnitude screen all read off this row shape.
_FEATURE_COLS = ", ".join("t." + c for c in (
    "tracklet_id", "center_ra", "center_dec", "epoch_mid_sec",
    "epoch_start_sec", "epoch_end_sec", "rate_deg_day", "pa_deg",
    "dra_cosdec_rate", "ddec_rate", "sig_pos_e_arcsec", "sig_pos_n_arcsec",
    "sig_vel_e_degday", "sig_vel_n_degday", "rate_sigma", "pa_sigma",
    "fit_rms_arcsec", "fit_chi2_dof", "n_obs", "linear_ok", "mag_mean",
    "mag_rms", "band", "n_mag"))


def source_chain(source):
    """``source`` and any source it wraps (``.base``) -- the overlay/cached
    sources the search layers over a real :class:`DbSource`."""
    out = []
    while source is not None:
        out.append(source)
        source = getattr(source, "base", None)
    return out


# ---------------------------------------------------------------------------
# Database-backed source
# ---------------------------------------------------------------------------
class DbSource:
    """Candidate source over the live SBN database (read-only ``pub`` role).

    THE CONTRACT EVERY SWEEP PATH KEEPS -- **SQL narrows, Python decides.**
    Whatever a path ships to the server (per-window box, healpix range, motion
    bound) is rounded OUTWARD into a strict SUPERSET of the window's own test,
    so the server can never drop a row the exact test would keep; `_survivors`
    then re-applies that exact Python predicate to whatever comes back.  This
    is what makes the three paths (chunk / setjoin / rangejoin) interchangeable
    and their A/Bs candidate-identical -- they differ in COST, never in result.
    Downstream methods just tag themselves with the short form rather than
    restating it."""

    # windows per chunked index query: consecutive windows overlap heavily on
    # sky, so one query serves many -- the sweep drops from one ~50 ms round
    # trip per window to a couple per chunk. SWEEP_PIX_BUDGET bounds a chunk's
    # union healpix disc (nside 64) so wide-corridor sweeps stay in memory.
    SWEEP_CHUNK = 64
    SWEEP_PIX_BUDGET = 4096
    SWEEP_SPAN_SEC = 90 * 86400        # cap a chunk's epoch span (see the sweep)

    def __init__(self, conn, cfg, index_cfg: IndexConfig = None, reconnect=None):
        self.conn = conn
        self.cfg = cfg
        self.index_cfg = index_cfg or IndexConfig()
        # zero-arg factory for a fresh connection; a dropped connection
        # (proxy-reaped tunnel, server restart) then costs one retry instead
        # of the whole search
        self.reconnect = reconnect
        # optional itf.trace.Tracer, injected by itf_search(trace=...) --
        # narrates the actual database round trips
        self.trace = None
        # windows dropped from the LAST sweep, for whatever reason (the search's
        # sweep summary reports it), and how many of those the SERVER dropped by
        # cancelling a query.  The split matters at the surface: the remedy for
        # a row budget is to raise it, and there is no such remedy for a query
        # the server would not finish -- see search._sweep_summary.
        self.last_sweep_skipped = 0
        self.last_sweep_cancelled = 0
        # ...and how many the MEMORY FLOOR dropped, which is a third cause with
        # a third remedy.  Without this the sweep summary attributes a
        # low-memory truncation to the row budget and tells the reader to RAISE
        # it -- advice that makes an out-of-memory search worse.  It could not
        # fire on a hosted deploy until the floor was made container-aware
        # (itf/memory.py), which is exactly where it now fires most.
        self.last_sweep_lowmem = 0
        self._sweep_cancels = 0        # cancellations in the LAST sweep
        # set by _obs_by_trkid when it drops a refetch batch for MEMORY rather
        # than for a server cancel; consumed by _cancelled_batch
        self._lowmem_batch = False

    def _retry(self, fn):
        """Run ``fn()``; on a dead connection, reconnect once and retry.

        A statement TIMEOUT is explicitly not that.  ``psycopg2.errors``'
        ``QueryCanceled`` subclasses ``OperationalError``, so without this
        clause the server-side ceiling armed in ``dbio._connect`` would be
        caught here, the connection dropped and reconnected, and THE SAME
        pathological query re-run -- paying the ceiling twice and then raising
        anyway.  A query the server gave up on will not go faster the second
        time: hand it straight to the caller, which knows how to drop those
        windows and say so (see :meth:`sweep_candidates`).
        """
        import psycopg2
        from psycopg2 import errors as _pgerr
        try:
            return fn()
        except _pgerr.QueryCanceled:
            raise
        except psycopg2.OperationalError:
            if self.reconnect is None:
                raise
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = self.reconnect()
            return fn()

    # Sentinel: the server's statement_timeout killed this query.  Not None and
    # not an empty result -- "we do not know what this query would have
    # returned" is a different answer from "it returned nothing", and only the
    # first one has to be reported as an incomplete sweep.
    CANCELLED = object()

    def _guarded(self, fn):
        """:meth:`_retry` ``fn``, or :attr:`CANCELLED` when the server gave up
        on it (``statement_timeout``, armed on the search role in
        ``dbio._connect``).

        EVERY sweep query funnels through here, because the ceiling is armed on
        the ROLE: it applies to the set-join and range-join paths (which
        ``_route_sweep`` picks for most corridors), to the obs refetch and to
        the legacy scan -- not just to the chunked path where the degradation
        was first written.  Without this, a cancellation on those paths would
        propagate out of the sweep generator and abort the whole search, which
        is not an improvement on the hang it replaced."""
        from psycopg2 import errors as _pgerr
        try:
            return self._retry(fn)
        except _pgerr.QueryCanceled:
            # The transaction is aborted; nothing else can run on this
            # connection until it is rolled back.
            try:
                self.conn.rollback()
            except Exception:
                pass
            return self.CANCELLED

    def _cancelled_batch(self, n_windows, n_remaining, trc):
        """Book one cancelled query against the sweep.  True when the caller
        should stop sweeping altogether.

        The running count is this object's, not the caller's: every sweep path
        kept its own ``cancels`` local and incremented it at each call site,
        which is four copies of a counter whose only reader is here.
        ``sweep_candidates`` resets it, and it is the only entry point -- the
        set-join and range-join paths are reached through it.

        The windows that query would have answered for are dropped into
        ``last_sweep_skipped``, which the search turns into its "corridor was
        NOT swept completely ... not a clean negative" warning -- a hang
        reported nothing at all, this reports exactly what was missed.

        And a per-statement ceiling does not bound a SWEEP: NV385's variant
        sweep is 315 chunks, so 315 x the ceiling is the real worst case.  Past
        ``cfg.sweep_max_cancels`` the corridor is not merely slow, it is one
        this server will not serve, and grinding the rest only spends the
        ceiling again on each."""
        if self._lowmem_batch:
            # Not the server's doing: _obs_by_trkid dropped this batch because
            # available memory fell under the sweep floor.  Count it as such,
            # do NOT spend one of cfg.sweep_max_cancels on it, and do not end
            # the sweep from here -- the loop-top valve does that on the next
            # iteration with the honest low-memory message and the remaining
            # window count.
            self._lowmem_batch = False
            self.last_sweep_skipped += n_windows
            self.last_sweep_lowmem += n_windows
            return False
        self.last_sweep_skipped += n_windows
        self.last_sweep_cancelled += n_windows
        cancels = self._sweep_cancels = self._sweep_cancels + 1
        if trc is not None and trc.enabled:
            trc.warn(f"query cancelled by the server after "
                     f"{self.cfg.sweep_statement_timeout_sec}s "
                     f"({n_windows} window(s) dropped, {cancels} so far this "
                     f"sweep)")
        if self.cfg.sweep_max_cancels and cancels >= self.cfg.sweep_max_cancels:
            self.last_sweep_skipped += n_remaining
            self.last_sweep_cancelled += n_remaining
            if trc is not None and trc.enabled:
                trc.warn(f"{cancels} cancelled query(s) -- abandoning the rest "
                         f"of this sweep ({n_remaining} more window(s) "
                         f"skipped)")
            return True
        return False

    def effective_floor_mb(self):
        """The floor actually enforced: ``cfg.sweep_min_free_mb``, scaled DOWN
        where a container is too small to pay it (:func:`itf.memory.sweep_floor_mb`).
        Every message about the floor must quote this, not the configured value.

        Deliberately NOT memoized on the source.  ``self.cfg`` is a frozen
        dataclass that the search REPLACES mid-run (the magnitude fallback
        rebuilds it), so a cached floor can outlive the config it came from --
        and the failure is silent, a valve enforcing a number nobody asked for.
        The expensive part, walking the cgroup hierarchy, is cached once per
        process inside :mod:`itf.memory`, so this is arithmetic.
        """
        return sweep_floor_mb(getattr(self.cfg, "sweep_min_free_mb", 0))

    def _low_memory_mb(self):
        """Available MB when it is under the effective floor, else None.
        The row budgets bound rows TESTED (politeness to the server); this
        floor bounds rows RETAINED (survival of the process) -- a variant
        union sweep caches every window it is served, and past this floor the
        next fetch is what the OOM killer answers."""
        floor = self.effective_floor_mb()
        if not floor:
            return None
        free = _mem_available_mb()
        return free if free is not None and free < floor else None

    def _sweep_stopped(self, n_remaining, trc, budget=0, rows_tested=0):
        """The sweep loops' shared stop valve: True when the row budget is
        spent or available memory is under the floor, with the remaining
        windows counted in ``last_sweep_skipped`` and the trace told why --
        a negative verdict after an incomplete sweep is not a clean negative.
        Both valves report through the same TOTAL, but the memory one is also
        counted in ``last_sweep_lowmem``: the surface tells the reader what to
        do about it, and "raise the row budget" is the wrong answer to an
        out-of-memory sweep."""
        # A stopped search (the web UI's Stop) unwinds here, between database
        # batches, rather than querying windows nobody is waiting for.
        deadline.check_cancelled()
        if budget and rows_tested >= budget:
            why = (f"sweep row budget exhausted ({rows_tested} feature rows "
                   f"tested)")
        else:
            free = self._low_memory_mb()
            if free is None:
                return False
            why = (f"available memory {free} MB is under the "
                   f"{self.effective_floor_mb()} MB sweep floor")
            self.last_sweep_lowmem += n_remaining
        self.last_sweep_skipped += n_remaining
        if trc is not None and trc.enabled:
            trc.warn(f"{why}: skipping the remaining {n_remaining} windows "
                     f"IN SWEEP ORDER")
        return True

    def _obs_by_trkid(self, trkids):
        """``trkid -> [obs row]`` for the corridor's surviving tracklets, in
        server-sized batches.  :attr:`CANCELLED` if the server gave up on any
        of them -- the refetch is as cancellable as the prefilter, and a batch
        missing its observations cannot be assembled into candidates.  Also
        CANCELLED when memory runs under the sweep floor mid-refetch: a
        100k-survivor chunk accumulates its obs rows right here (this is the
        exact line 2006 SJ252 died on), and dropping the chunk is the same
        recovery as a server cancel -- the loop-top valve then ends the sweep
        with the honest low-memory message."""
        out = {}
        for j in range(0, len(trkids), 8000):
            deadline.check_cancelled()
            free = self._low_memory_mb()
            if free is not None:
                self._lowmem_batch = True     # cause, for _cancelled_batch
                trc = self.trace
                if trc is not None and trc.enabled:
                    trc.warn(f"available memory {free} MB is under the sweep "
                             f"floor mid-refetch: dropping this chunk")
                return self.CANCELLED
            rows = self._obs_for_trkids(trkids[j:j + 8000])
            if rows is self.CANCELLED:
                return self.CANCELLED
            for row in rows:
                out.setdefault(row["trkid"], []).append(row)
        return out

    def sweep_candidates(self, windows, uncert_mult, extrawidth, want_ranking,
                         row_budget=None, prefilter=None):
        """Yield ``(win, rows, chi2_by_trkid, feats_by_trkid)`` for a whole
        corridor sweep.  ``row_budget`` overrides ``cfg.sweep_row_budget`` for
        this sweep -- a UNION of K corridors (the LOV variant scan) must be
        allowed K corridors' worth of work, or the valve calibrated for one
        corridor silently truncates it.  ``prefilter`` overrides
        ``cfg.prefilter_motion`` the same way (None -> the config value): the
        search passes False when an ML feature collector rides along, so the
        collector sees the pre-filter's hard negatives (:meth:`_survivors`).
        ``feats_by_trkid`` maps each candidate tracklet to its
        window-annotated index feature row (empty on the scan path) -- the
        search's scoring and the ML feature collector both read it.

        On the index path, windows are processed in chunks: one prefilter
        query over the union of the chunk's healpix discs + epoch span, the
        exact per-window corridor test in Python, and one obs refetch for the
        chunk's surviving trkids.  The scan path falls back to per-window
        queries (its predicate lives in SQL).
        """
        trc = self.trace
        budget = (self.cfg.sweep_row_budget if row_budget is None
                  else row_budget)
        rows_tested = 0
        self.last_sweep_skipped = self.last_sweep_cancelled = 0
        self.last_sweep_lowmem = 0
        self._sweep_cancels = 0
        self._lowmem_batch = False
        if self.cfg.candidate_path != "index":
            for k, window in enumerate(windows):
                if self._sweep_stopped(len(windows) - k, trc, budget, rows_tested):
                    return
                ra_tol, dec_tol = window_tolerances(window, uncert_mult,
                                                    extrawidth)
                rows = self._scan_path(window, ra_tol, dec_tol)
                if rows is self.CANCELLED:
                    if self._cancelled_batch(1, len(windows) - k - 1, trc):
                        return
                    continue
                rows_tested += len(rows)
                if trc is not None and trc.enabled:
                    trc.count("db_queries")
                    trc.count("db_rows", len(rows))
                if trc is not None and trc.wants(L_FIREHOSE) and rows:
                    trc(f"    db scan query {k + 1}/{len(windows)}: "
                        f"{len(rows)} obs rows", L_FIREHOSE)
                # no per-tracklet features on the scan path, so no chi2 ranking
                yield window, rows, {}, {}
            return

        self._prime_discs(windows, uncert_mult, extrawidth)
        path, why = self._route_sweep(windows, uncert_mult, extrawidth, trc)
        if trc is not None and trc.enabled:
            trc.cont(f"{path} path over {len(windows)} window(s) -- {why}")
        if path == "setjoin":
            yield from self._sweep_setjoin(windows, uncert_mult, extrawidth,
                                           want_ranking, trc, budget, prefilter)
            return
        if path == "rangejoin":
            yield from self._sweep_rangejoin(windows, uncert_mult, extrawidth,
                                             want_ranking, trc, budget,
                                             prefilter)
            return

        import psycopg2.extras
        s = self.index_cfg.schema
        i = 0
        while i < len(windows):
            if self._sweep_stopped(len(windows) - i, trc, budget, rows_tested):
                return
            # pixel-budget chunking: wide windows (sloppy seeds) can cover a
            # large sky area -- bound each prefilter query's return by the
            # union disc size instead of a fixed window count, so a monster
            # corridor degrades to smaller chunks instead of exhausting memory
            chunk, tols, pix_all = [], [], set()
            span_lo = span_hi = None
            while i < len(windows) and len(chunk) < self.SWEEP_CHUNK:
                window = windows[i]
                # sigma-band sweep order can place epoch-distant windows side
                # by side at band boundaries; an epoch-incoherent chunk would
                # widen the BETWEEN span and explode the query, so break the
                # chunk on a large epoch jump
                if chunk and abs(window.unix_start
                                 - chunk[-1].unix_start) > 200 * 86400:
                    break
                # ...AND cap the chunk's total epoch SPAN.  The prefilter is
                # `healpix = ANY(union) AND epoch BETWEEN chunk_lo..hi`, so a
                # chunk that spans years decouples each pixel from its epoch and
                # fetches every ecliptic tracklet in the whole space-time box
                # (a narrow ±10" corridor pulling 68k rows to test, ~0 kept). A
                # tight span keeps the union small (the object moves a short arc)
                # and the epoch window narrow -- same per-window corridor test,
                # far fewer wasted fetches.
                #
                # Measured PROSPECTIVELY and TWO-SIDED, against the same bounds
                # the query will use (t_lo = min start, t_hi = max end below).
                # `windows` is NOT in epoch order -- the jump guard above exists
                # precisely because band order interleaves distant epochs -- so
                # the older one-sided `window.unix_start - chunk[0].unix_start`
                # went NEGATIVE for any window earlier than the chunk's first
                # and could never trip, leaving backwards drift bounded only by
                # the 200-day pairwise guard: a chunk could span several hundred
                # days against a documented 90-day cap, which is exactly the
                # over-fetch this cap exists to prevent.
                lo = window.unix_start if span_lo is None else min(span_lo,
                                                                window.unix_start)
                hi = window.unix_end if span_hi is None else max(span_hi,
                                                              window.unix_end)
                if chunk and (hi - lo) > self.SWEEP_SPAN_SEC:
                    break
                ra_tol, dec_tol = window_tolerances(window, uncert_mult,
                                                    extrawidth)
                pix = self._window_disc(window, ra_tol, dec_tol)
                if chunk and len(pix_all | pix) > self.SWEEP_PIX_BUDGET:
                    break
                pix_all |= pix
                chunk.append(window)
                tols.append((ra_tol, dec_tol))
                span_lo, span_hi = lo, hi
                i += 1
            if not pix_all:
                for window in chunk:
                    yield window, [], {}, {}
                continue
            t_lo = min(w.unix_start for w in chunk)
            t_hi = max(w.unix_end for w in chunk)
            dec_lo = min(w.min_dec - t[1] - BOX_PAD_DEG
                         for w, t in zip(chunk, tols))
            dec_hi = max(w.max_dec + t[1] + BOX_PAD_DEG
                         for w, t in zip(chunk, tols))
            # The box is the UNION of this chunk's windows, so most of what it
            # returns cannot survive ANY of them -- on a LOV variant union
            # (thousands of thin corridors over the same era) that is nearly
            # all of it.  An EXISTS over the chunk's windows drops those rows
            # server-side while keeping the single-box shape: no join, so no
            # row is multiplied per matching window, and EXISTS short-circuits
            # on the first window that accepts.  SQL narrows, Python decides.
            wrows = [self._window_sql_row(window, ra_tol, dec_tol)
                     for window, (ra_tol, dec_tol) in zip(chunk, tols)]
            corridor = self._sql_corridor()
            motion = self._sql_motion_bound(prefilter)
            where_w = corridor + (f" AND {motion}" if motion else "")
            args = (self._window_arrays(self._W_CORRIDOR_COLS, wrows)
                    + [list(pix_all), t_lo, t_hi, dec_lo, dec_hi])
            if motion:
                args += [self.cfg.prefilter_motion_slack_rms,
                         self.cfg.prefilter_motion_slack_arcsec,
                         self.cfg.maybe_arcsec]

            def _fetch_chunk():
                with self.conn.cursor(
                        cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"""
                        WITH {self._window_cte(self._W_CORRIDOR_COLS)}
                        SELECT {_FEATURE_COLS}
                        FROM {s.target_qualified} t
                        WHERE healpix = ANY(%s)
                          AND epoch_mid_sec BETWEEN %s AND %s
                          AND center_dec BETWEEN %s AND %s
                          AND EXISTS (SELECT 1 FROM w
                                      WHERE t.epoch_mid_sec BETWEEN w.t_lo AND w.t_hi
                                        AND t.center_dec BETWEEN w.d_lo AND w.d_hi
                                        AND {where_w})
                    """, args)
                    return cur.fetchall()
            # The server gave up on this chunk (statement_timeout, armed in
            # dbio._connect).  Nothing bounded a query before that, and one
            # chunk out of NV385's 315 blocked a whole search for 45+ min --
            # py-spy showed the process parked in cur.execute with flat CPU,
            # and the SearchBudget cannot help because it is only consulted
            # BETWEEN trials and variants, never inside a sweep.  So drop the
            # chunk and keep sweeping (see _cancelled_batch for what that
            # costs and when it gives up).
            feats = self._guarded(_fetch_chunk)
            if feats is self.CANCELLED:
                if self._cancelled_batch(len(chunk), len(windows) - i, trc):
                    return
                continue
            rows_tested += len(feats)
            if trc is not None and trc.enabled:
                trc.count("db_queries")
                trc.count("db_feature_rows", len(feats))
            if trc is not None and trc.wants(L_FIREHOSE):
                trc(f"    db index query: {len(chunk)} windows in one shot "
                    f"({len(pix_all)} healpix pixels, epoch "
                    f"{t_lo:.0f}..{t_hi:.0f}, dec {dec_lo:.1f}..{dec_hi:.1f}) "
                    f"-> {len(feats)} tracklet feature rows to test")

            per_window = []
            need = set()
            for window, (ra_tol, dec_tol) in zip(chunk, tols):
                cands = self._survivors(window, feats, ra_tol, dec_tol, prefilter)
                need.update(c["tracklet_id"] for c in cands)
                per_window.append(cands)
            need_list = list(need)
            obs_by_trkid = self._obs_by_trkid(need_list)
            if obs_by_trkid is self.CANCELLED:
                if self._cancelled_batch(len(chunk), len(windows) - i, trc):
                    return
                continue
            if trc is not None and trc.enabled:
                trc.count("db_rows", sum(len(v) for v in obs_by_trkid.values()))
            if trc is not None and trc.wants(L_FIREHOSE) and need_list:
                trc(f"    db obs refetch: {len(need_list)} corridor-surviving "
                    f"tracklet id(s) -> "
                    f"{sum(len(v) for v in obs_by_trkid.values())} obs rows",
                    L_FIREHOSE)
            for window, cands in zip(chunk, per_window):
                yield (window, *_assemble_window(window, cands, obs_by_trkid,
                                              want_ranking))

    # ---- set-based sweep (one query per batch of windows) ------------------
    # (window, healpix) rows per query.  The payload is the only real cost left
    # once (healpix, epoch_mid_sec) is indexed -- server-side execution of a
    # whole 907-window corridor is ~60 ms -- so batch generously but bound it
    # so a monster corridor cannot build one giant array.
    SETJOIN_TUPLE_BATCH = 20000
    # ...and the guard that decides the set-join is the WRONG shape.  Its whole
    # advantage is latency on NARROW windows (a ~10" corridor covers ~10 nside-64
    # pixels, so per-window exactness is nearly free).  A degrees-wide window --
    # an under-determined seed's linear corridor, or a thin-but-2.5deg LOV variant
    # corridor -- covers HUNDREDS of pixels (2015 BE559's variant windows: 554
    # each, 10.8M (window, pixel) boxes for one union sweep), and there the union
    # box the chunker sends is both far smaller on the wire (pixels deduplicated
    # across the chunk's overlapping windows) and a good approximation of what
    # every window in the chunk wants.  Above this mean pixels-per-window the
    # sweep hands the whole corridor to the chunked path.
    SETJOIN_MAX_PIX_PER_WINDOW = 64
    SETJOIN_PROBE_WINDOWS = 8          # windows sampled to estimate the mean

    # ---- the window relation `w` the server-side paths join/EXISTS against ---
    #
    # All three DB paths ship a batch of windows to the server as an unnested
    # array relation, and each one has to keep THREE orderings in agreement: the
    # CTE's column-name list, the `unnest(%s::type[], ...)` cast list, and the
    # per-row value tuple.  Hand-aligning them (three parallel literals, one of
    # them 17 wide) is a silent-wrong-answer bug waiting to happen -- a shifted
    # value would not raise, it would just quietly filter on the wrong number
    # and drop candidates.  So the ordering is declared ONCE per relation here,
    # as (column, SQL element type), and `_window_cte` / `_window_arrays` derive
    # the SQL and the payload from it while `_window_sql_row` supplies values
    # BY NAME.  Adding a column is one entry in one tuple.
    #
    # Bounds are rounded OUTWARD by a whole second / 1e-4 deg -- coarse enough
    # to keep the wire form short (the arrays are sent as text), fine enough to
    # cost nothing downstream.
    _W_CORRIDOR_COLS = (
        ("t_lo", "float8"), ("t_hi", "float8"),         # epoch box
        ("d_lo", "float8"), ("d_hi", "float8"),         # dec box
        ("ra_a", "float8"), ("ra_b", "float8"), ("ra_c", "float8"),   # RA quad
        ("d_a", "float8"), ("d_b", "float8"), ("d_c", "float8"),      # dec quad
        ("tc", "float8"), ("dist_au", "float8"),        # quad epoch, range
        ("ra_tol", "float8"), ("dec_tol", "float8"),    # corridor half-widths
    )
    # ...plus the window index and one healpix RANGE per row (_sweep_rangejoin)
    _W_RANGEJOIN_COLS = ((("wi", "int"), ("lo", "bigint"), ("hi", "bigint"))
                         + _W_CORRIDOR_COLS)
    # ...or the window index and one healpix PIXEL per row, with the box only --
    # the set-join path applies no server-side corridor test (_sweep_setjoin)
    _W_SETJOIN_COLS = (("wi", "int"), ("pix", "bigint"),
                       ("t_lo", "bigint"), ("t_hi", "bigint"),
                       ("d_lo", "float8"), ("d_hi", "float8"))

    @staticmethod
    def _window_cte(cols, alias="w"):
        """The ``<alias>(...) AS (SELECT * FROM unnest(...))`` CTE body for a
        window relation -- names and casts from the SAME spec that fills the
        arrays, so the two cannot drift apart."""
        names = ", ".join(c for c, _ in cols)
        casts = ", ".join(f"%s::{t}[]" for _, t in cols)
        return f"{alias}({names}) AS (SELECT * FROM unnest({casts}))"

    @staticmethod
    def _window_arrays(cols, rows):
        """``rows`` (dicts from :meth:`_window_sql_row`) as the column-major
        arrays :meth:`_window_cte`'s placeholders expect, in spec order."""
        return [[r[c] for r in rows] for c, _ in cols]

    def _window_sql_row(self, window, ra_tol, dec_tol):
        """Every named quantity a window relation can carry, for ONE window.
        A given relation takes the subset its spec names; nothing here depends
        on the order they are declared in."""
        return {
            "t_lo": math.floor(window.unix_start),
            "t_hi": math.ceil(window.unix_end),
            "d_lo": math.floor((window.min_dec - dec_tol - BOX_PAD_DEG) * 1e4) / 1e4,
            "d_hi": math.ceil((window.max_dec + dec_tol + BOX_PAD_DEG) * 1e4) / 1e4,
            "ra_a": window.ra_quad[0], "ra_b": window.ra_quad[1],
            "ra_c": window.ra_quad[2],
            "d_a": window.dec_quad[0], "d_b": window.dec_quad[1],
            "d_c": window.dec_quad[2],
            "tc": window.unix_cur, "dist_au": window.dist_au,
            "ra_tol": ra_tol, "dec_tol": dec_tol,
        }

    def _prime_discs(self, windows, uncert_mult, extrawidth):
        """Compute every window's healpix disc for this sweep in one pass;
        :meth:`_window_disc` serves them.

        Held as int32 arrays, not sets: a window near the max_uncert_deg cap
        covers ~22000 pixels at nside 64 (about half the sky), ~2.8 MB as a
        Python set against ~88 kB as an array, and a sweep holds every window's
        disc at once.  Measured live: holding sets, with one disc walk over the
        whole sweep, took 2011 DV's peak RSS from 346 MB to 1492 MB (2013 DM1
        390 -> 1353 MB); arrays plus batched walks (DISC_BATCH) bring them to
        367 / 404 MB with identical results."""
        tols = [window_tolerances(w, uncert_mult, extrawidth) for w in windows]
        discs = window_healpix_discs(windows, tols, self.index_cfg, as_sets=False)
        self._discs = {id(w): (w, t, d) for w, t, d in zip(windows, tols, discs)}

    def _window_disc(self, window, ra_tol, dec_tol):
        """The window's healpix disc as a set: from this sweep's precomputed
        array when it is the same window at the same tolerances, else computed
        now.  The set lives only as long as its caller needs it."""
        hit = getattr(self, "_discs", {}).get(id(window))
        if hit is not None and hit[0] is window and hit[1] == (ra_tol, dec_tol):
            return set(hit[2].tolist())
        return window_healpix_disc(window, ra_tol, dec_tol, self.index_cfg)

    def _window_query_tuples(self, window, wi, uncert_mult, extrawidth):
        """One :attr:`_W_SETJOIN_COLS` row per healpix pixel ONE window covers
        -- its exact prefilter box -- plus the window's ``(ra_tol, dec_tol)``."""
        ra_tol, dec_tol = window_tolerances(window, uncert_mult, extrawidth)
        base = self._window_sql_row(window, ra_tol, dec_tol)
        pix = self._window_disc(window, ra_tol, dec_tol)
        return ([{**base, "wi": wi, "pix": p} for p in pix],
                (ra_tol, dec_tol))

    def _route_sweep(self, windows, uncert_mult, extrawidth, trc):
        """Which server-side sweep shape this corridor gets, and WHY (a short
        phrase for the trace).  The ONE place a path is chosen.

        ``cfg.sweep_mode`` is ``"auto"`` -- this router -- or the NAME OF A
        PATH, which forces it unconditionally for an A/B.  A forced path is
        exactly that: no silent fallback, or the A/B measures the router again
        instead of the path (forcing setjoin on a wide corridor really does
        send the exploded per-pixel tuples, and is meant to).

        Routing is MEASURED, not assumed -- no shape wins everywhere:
                              BE559 (wide, 272 win)   RX190 (narrow, 907 win)
          chunk                     31.2 s                    3.6 s
          setjoin                   33.0 s                    0.6 s
          rangejoin                 14.6 s                    8.1 s
        ...with candidates IDENTICAL in every cell (0 per-window differences).

        * NARROW corridor -> setjoin.  Few pixels per window, so its
          ``healpix = w.pix`` EQUALITY join is ideal and its tuple batches are
          large; rangejoin trades that for a BETWEEN and loses 13x.
        * WIDE corridor -> rangejoin.  setjoin cannot be used at all (the
          per-pixel tuples explode), and the old fallback -- the coarse
          union-box chunker -- over-fetches and re-tests every row against
          every window in Python.
        * WIDE and MANY windows -> chunk.  A per-window join returns each row
          ONCE PER MATCHING WINDOW, so heavily OVERLAPPING windows multiply the
          result set, while the union box fetches each row once.  The variant
          scan is exactly that case: 31 LOV variants over the same epochs =
          19459 thin, mutually overlapping windows.  Measured on 2015 BE559,
            linear corridor   272 windows, wide, distinct   : 32 s -> 14.6 s WIN
            variant union   19459 windows, thin, overlapping: 109 s -> 285 s LOSS
          so rangejoin is a TARGETED win.  The window COUNT is a proxy for the
          overlap that actually drives this (see RANGEJOIN_MAX_TOTAL_WINDOWS).
        """
        mode = self.cfg.sweep_mode
        if mode in ("setjoin", "rangejoin", "chunk"):
            return mode, "forced by cfg.sweep_mode (A/B)"
        if self._setjoin_fits(windows, uncert_mult, extrawidth, trc):
            return "setjoin", "narrow corridor: few healpix pixels per window"
        if len(windows) <= self.RANGEJOIN_MAX_TOTAL_WINDOWS:
            return "rangejoin", "wide corridor, windows few enough to be distinct"
        return "chunk", (f"wide corridor of {len(windows)} windows "
                         f"(> {self.RANGEJOIN_MAX_TOTAL_WINDOWS}): they overlap "
                         f"enough that a per-window join would multiply rows")

    def _setjoin_fits(self, windows, uncert_mult, extrawidth, trc):
        """Is this corridor narrow enough for the set-join to be the cheaper
        shape?  Estimated from the healpix footprint of a few evenly-spaced
        windows (see SETJOIN_MAX_PIX_PER_WINDOW).  A PREDICATE only -- where a
        corridor that fails this goes is :meth:`_route_sweep`'s decision (and
        for most wide corridors it is rangejoin, not chunk)."""
        if not windows:
            return True
        step = max(1, len(windows) // self.SETJOIN_PROBE_WINDOWS)
        probe = windows[::step][:self.SETJOIN_PROBE_WINDOWS]
        n_pix = 0
        for window in probe:
            ra_tol, dec_tol = window_tolerances(window, uncert_mult, extrawidth)
            n_pix += len(self._window_disc(window, ra_tol, dec_tol))
        mean_pix = n_pix / len(probe)
        ok = mean_pix <= self.SETJOIN_MAX_PIX_PER_WINDOW
        if trc is not None and trc.wants(L_STORY) and not ok:
            trc.cont(f"~{mean_pix:.0f} healpix pixels per window "
                     f"(> {self.SETJOIN_MAX_PIX_PER_WINDOW}) -- wide corridor, "
                     f"not the set-join shape")
        return ok

    def _sweep_setjoin(self, windows, uncert_mult, extrawidth, want_ranking,
                       trc, budget, prefilter=None):
        """Sweep a corridor with ONE set-based query per batch of windows.

        Every window's own (healpix, epoch, dec) box goes to the server as a
        joined ``unnest(...)`` relation, so Postgres applies the EXACT per-window
        prefilter (an index range scan per pixel on ``itf_trk_healpix_epoch``)
        and returns only rows some window actually asked for.  The chunked
        alternative (:meth:`sweep_candidates`'s other branch) instead unions many
        windows into one coarse box and pays a round trip per chunk.

        Measured on 2024 RX190 (907 windows), BEFORE the chunked path gained its
        own server-side EXISTS: chunked 5.4 s / 4461 rows -> set-join 0.5 s /
        321 rows, same candidates after the corridor test.

        SQL narrows, Python decides."""
        import psycopg2.extras
        s = self.index_cfg.schema
        rows_tested = 0
        i = 0
        n_query = 0
        while i < len(windows):
            if self._sweep_stopped(len(windows) - i, trc, budget, rows_tested):
                return
            batch, tols, tup = [], [], []
            while i < len(windows) and len(tup) < self.SETJOIN_TUPLE_BATCH:
                rows_, tol = self._window_query_tuples(windows[i], len(batch),
                                                       uncert_mult, extrawidth)
                tup.extend(rows_)
                batch.append(windows[i])
                tols.append(tol)
                i += 1
            if not tup:
                for window in batch:
                    yield window, [], {}, {}
                continue
            args = self._window_arrays(self._W_SETJOIN_COLS, tup)

            def _fetch():
                with self.conn.cursor(
                        cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"""
                        WITH {self._window_cte(self._W_SETJOIN_COLS)}
                        SELECT w.wi AS _wi, {_FEATURE_COLS}
                        FROM w JOIN {s.target_qualified} t
                          ON t.healpix = w.pix
                         AND t.epoch_mid_sec BETWEEN w.t_lo AND w.t_hi
                         AND t.center_dec BETWEEN w.d_lo AND w.d_hi
                    """, args)
                    return cur.fetchall()
            matched = self._guarded(_fetch)
            if matched is self.CANCELLED:
                if self._cancelled_batch(len(batch), len(windows) - i, trc):
                    return
                continue
            n_query += 1
            rows_tested += len(matched)
            per_window_feats = {}
            for r in matched:
                per_window_feats.setdefault(r["_wi"], []).append(r)
            if trc is not None and trc.enabled:
                trc.count("db_queries")
                trc.count("db_feature_rows", len(matched))
            if trc is not None and trc.wants(L_FIREHOSE):
                trc(f"    db set-join query {n_query}: {len(batch)} windows, "
                    f"{len(tup)} (window, healpix) boxes -> {len(matched)} "
                    f"tracklet feature rows in-corridor-box", L_FIREHOSE)

            per_window = []
            need = set()
            for wi, (window, (ra_tol, dec_tol)) in enumerate(zip(batch, tols)):
                cands = self._survivors(window, per_window_feats.get(wi, ()),
                                        ra_tol, dec_tol, prefilter)
                need.update(c["tracklet_id"] for c in cands)
                per_window.append(cands)
            need_list = list(need)
            obs_by_trkid = self._obs_by_trkid(need_list)
            if obs_by_trkid is self.CANCELLED:
                if self._cancelled_batch(len(batch), len(windows) - i, trc):
                    return
                continue
            if trc is not None and trc.enabled:
                trc.count("db_rows", sum(len(v) for v in obs_by_trkid.values()))
            if trc is not None and trc.wants(L_FIREHOSE) and need_list:
                trc(f"    db obs refetch: {len(need_list)} corridor-surviving "
                    f"tracklet id(s) -> "
                    f"{sum(len(v) for v in obs_by_trkid.values())} obs rows",
                    L_FIREHOSE)
            for window, cands in zip(batch, per_window):
                yield (window, *_assemble_window(window, cands, obs_by_trkid,
                                              want_ranking))

    # ---- range-join sweep: few queries, compact payload ------------------
    RANGEJOIN_MAX_WINDOWS = 48       # windows per query (measured sweet spot)
    RANGEJOIN_MAX_TOTAL_WINDOWS = 2000   # ...and the corridor-shape guard: past
                                         # this many windows they overlap enough
                                         # that the union-box chunker wins.  A
                                         # PROXY for overlap, which is what
                                         # actually drives it (see _route_sweep)
    RANGEJOIN_ROW_BATCH = 8000       # ...and a cap on (window, pixel-range) rows

    @staticmethod
    def _pixel_ranges(pix):
        """A window's healpix pixels as contiguous ``[lo, hi]`` runs.

        NESTED healpix ids are spatially local, so a disc collapses hard: a
        degrees-wide 2015 BE559 window is 17096 pixels but only 164 runs.  That
        matters because the pixel set is the QUERY PAYLOAD -- sent as a text
        array and planned against -- and a 17k-element ``= ANY(...)`` costs
        ~52 ms of PLANNING alone, against ~26 ms of actual execution."""
        p = sorted(pix)
        if not p:
            return []
        out, lo, prev = [], p[0], p[0]
        for x in p[1:]:
            if x == prev + 1:
                prev = x
            else:
                out.append((lo, prev))
                lo = prev = x
        out.append((lo, prev))
        return out

    # The motion bound (candidates.motion_bound_ok) as SQL over a joined window
    # relation `w`.  Kept a CONSERVATIVE SUPERSET of the Python predicate -- the
    # extra 1e-6" absorbs float differences (measured max relative discrepancy
    # 4e-11 over 630 real rows) -- because `_survivors` still applies the exact
    # Python test to whatever comes back.  Same contract as the box bounds:
    # SQL narrows, Python decides.
    _SQL_DRA = ("(w.ra_a*power(t.epoch_end_sec-w.tc,2)+w.ra_b*(t.epoch_end_sec-w.tc)+w.ra_c)"
                "-(w.ra_a*power(t.epoch_start_sec-w.tc,2)+w.ra_b*(t.epoch_start_sec-w.tc)+w.ra_c)")
    _SQL_DDEC = ("(w.d_a*power(t.epoch_end_sec-w.tc,2)+w.d_b*(t.epoch_end_sec-w.tc)+w.d_c)"
                 "-(w.d_a*power(t.epoch_start_sec-w.tc,2)+w.d_b*(t.epoch_start_sec-w.tc)+w.d_c)")

    # Predicted corridor position AT THE TRACKLET'S OWN MID-EPOCH -- the same
    # quadratic `Window.predict` evaluates, and the centre of the corridor test
    # below.
    _SQL_PRED_RA = ("(w.ra_a*power(t.epoch_mid_sec-w.tc,2)"
                    "+w.ra_b*(t.epoch_mid_sec-w.tc)+w.ra_c)")
    _SQL_PRED_DEC = ("(w.d_a*power(t.epoch_mid_sec-w.tc,2)"
                     "+w.d_b*(t.epoch_mid_sec-w.tc)+w.d_c)")
    # A tracklet's own half-extent widens the tolerance, so a short arc
    # straddling the corridor edge is still admitted (tracklet_in_window).
    _SQL_HALF_EXT = ("(coalesce(t.rate_deg_day,0.0)"
                     "*abs(coalesce(t.epoch_end_sec,t.epoch_mid_sec)"
                     "-coalesce(t.epoch_start_sec,t.epoch_mid_sec))/2.0/86400.0)")
    # Outward pad, same role as the 1e-6" on the motion bound and the 1e-4 deg
    # rounding on the dec box: absorb float8-vs-numpy disagreement at the
    # threshold so SQL can only ever be LOOSER than the Python predicate.
    CORRIDOR_PAD_DEG = 1e-6

    def _sql_corridor(self):
        """SQL for the per-window CORRIDOR test -- the thin quadratic tube --
        as opposed to the coarse dec BAND the join already applies.

        This is the selective one.  The band spans the whole window's dec
        extent, but the corridor is a tube of half-width ``ra_tol``/``dec_tol``
        about the predicted track, so on a long thin corridor the band admits
        orders of magnitude more tracklets than can survive: measured on 2022
        GL24, a single batch returned ~22k feature rows of which a few hundred
        survived.  Applying it server-side is what stops those rows becoming
        Python objects at all -- and the payload, per this class's own notes,
        is the only real cost left once (healpix, epoch_mid_sec) is indexed.

        Rounded outward by ``CORRIDOR_PAD_DEG``: SQL narrows, Python decides.
        """
        dra = f"(t.center_ra - {self._SQL_PRED_RA})"
        return f"""(
          abs(({dra}) - 360.0*floor((({dra})+180.0)/360.0))
              <= w.ra_tol + {self._SQL_HALF_EXT} + {self.CORRIDOR_PAD_DEG}
          AND abs(t.center_dec - {self._SQL_PRED_DEC})
              <= w.dec_tol + {self._SQL_HALF_EXT} + {self.CORRIDOR_PAD_DEG})"""

    def _sql_motion_bound(self, prefilter):
        """SQL for the motion pre-filter, or None when it is switched off.

        ``prefilter`` is the per-sweep override (None -> the config value),
        resolved the same way :meth:`_survivors` resolves it -- an ML feature
        collector sweeps with ``prefilter=False``, and the server-side bound
        must stand down with it or the collector's negative class is drawn
        from an already-pruned population."""
        if prefilter is None:
            prefilter = self.cfg.prefilter_motion
        if not prefilter:
            return None
        dra, ddec = self._SQL_DRA, self._SQL_DDEC
        arc = "abs(t.epoch_end_sec-t.epoch_start_sec)"
        return f"""(
          t.fit_rms_arcsec IS NULL OR t.epoch_start_sec IS NULL
          OR t.epoch_end_sec IS NULL OR t.dra_cosdec_rate IS NULL
          OR t.ddec_rate IS NULL OR {arc} <= 0
          OR 3600.0*sqrt(
               power((({dra}) - 360.0*floor((({dra})+180.0)/360.0))
                     *cos(radians(t.center_dec))
                     - t.dra_cosdec_rate*{arc}/86400.0, 2)
             + power(({ddec}) - t.ddec_rate*{arc}/86400.0, 2))
             - (%s*t.fit_rms_arcsec + %s) - 1e-6
             <= %s + 0.00026/86400.0*{arc}/w.dist_au*57.3*3600.0)"""

    def _sweep_rangejoin(self, windows, uncert_mult, extrawidth, want_ranking,
                         trc, budget, prefilter=None):
        """Sweep a corridor with ONE query per BATCH of windows, each window
        contributing compact healpix ranges, its own epoch/dec box and (when
        enabled) the motion bound -- all applied server-side, with rows assigned
        to windows by the join.

        Why this shape.  Measured on BE559's 272-window corridor, the per-query
        overhead -- not the rows, not the pixels, not the index -- was the cost:
        a 25 ms round-trip floor (proxy tunnel), ~52 ms of PLANNING driven by a
        17k-element pixel array, against 2-26 ms of actual index execution.  The
        index (`itf_trk_healpix_epoch`, btree on (healpix, epoch_mid_sec)
        INCLUDE center_dec) was never the problem.  So: fewer queries, smaller
        payloads.

            one query per window   272 queries  39.4 s
            batched, 48 windows      6 queries  16.0 s   <- 2.46x, same result
            batched, 96 windows      3 queries  16.4 s
            batched, all 272         1 query    20.3 s   (execution-bound)

        18165 survivors identical in every configuration.  Past ~48 windows the
        server-side join grows faster than the round trips it saves, hence
        RANGEJOIN_MAX_WINDOWS.

        SQL narrows, Python decides."""
        import psycopg2.extras
        s = self.index_cfg.schema
        motion = self._sql_motion_bound(prefilter)
        corridor = self._sql_corridor()
        rows_tested = 0
        i = 0
        n_query = 0
        while i < len(windows):
            if self._sweep_stopped(len(windows) - i, trc, budget, rows_tested):
                return
            batch, tols, wrows = [], [], []
            while (i < len(windows) and len(batch) < self.RANGEJOIN_MAX_WINDOWS
                   and len(wrows) < self.RANGEJOIN_ROW_BATCH):
                window = windows[i]
                ra_tol, dec_tol = window_tolerances(window, uncert_mult, extrawidth)
                rr = self._pixel_ranges(self._window_disc(window, ra_tol, dec_tol))
                base = self._window_sql_row(window, ra_tol, dec_tol)
                wi = len(batch)
                wrows.extend({**base, "wi": wi, "lo": lo, "hi": hi}
                             for lo, hi in rr)
                batch.append(window)
                tols.append((ra_tol, dec_tol))
                i += 1
            if not wrows:
                for window in batch:
                    yield window, [], {}, {}
                continue
            sql = f"""
                WITH {self._window_cte(self._W_RANGEJOIN_COLS)}
                -- no DISTINCT needed: a window's pixel ranges are disjoint
                -- by construction, so (wi, tracklet) cannot repeat
                SELECT w.wi AS _wi, {_FEATURE_COLS}
                FROM {s.target_qualified} t JOIN w
                  ON t.healpix BETWEEN w.lo AND w.hi
                 AND t.epoch_mid_sec BETWEEN w.t_lo AND w.t_hi
                 AND t.center_dec BETWEEN w.d_lo AND w.d_hi
            """ + f" WHERE {corridor}" + (f" AND {motion}" if motion else "")
            args = self._window_arrays(self._W_RANGEJOIN_COLS, wrows)
            if motion:
                args += [self.cfg.prefilter_motion_slack_rms,
                         self.cfg.prefilter_motion_slack_arcsec,
                         self.cfg.maybe_arcsec]

            def _fetch():
                with self.conn.cursor(
                        cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, args)
                    return cur.fetchall()
            matched = self._guarded(_fetch)
            if matched is self.CANCELLED:
                if self._cancelled_batch(len(batch), len(windows) - i, trc):
                    return
                continue
            n_query += 1
            rows_tested += len(matched)
            per_window_feats = {}
            for r in matched:
                per_window_feats.setdefault(r["_wi"], []).append(r)
            if trc is not None and trc.enabled:
                trc.count("db_queries")
                trc.count("db_feature_rows", len(matched))
            if trc is not None and trc.wants(L_FIREHOSE):
                trc(f"    db range-join query {n_query}: {len(batch)} windows, "
                    f"{len(wrows)} pixel ranges -> {len(matched)} tracklet "
                    f"feature rows"
                    + " (corridor"
                    + (" + motion bound" if motion else "")
                    + " applied server-side)")

            per_window, need = [], set()
            for wi, (window, (ra_tol, dec_tol)) in enumerate(zip(batch, tols)):
                cands = self._survivors(window, per_window_feats.get(wi, ()),
                                        ra_tol, dec_tol, prefilter)
                need.update(c["tracklet_id"] for c in cands)
                per_window.append(cands)
            need_list = list(need)
            obs_by_trkid = self._obs_by_trkid(need_list)
            if obs_by_trkid is self.CANCELLED:
                if self._cancelled_batch(len(batch), len(windows) - i, trc):
                    return
                continue
            if trc is not None and trc.enabled:
                trc.count("db_rows", sum(len(v) for v in obs_by_trkid.values()))
            if trc is not None and trc.wants(L_FIREHOSE) and need_list:
                trc(f"    db obs refetch: {len(need_list)} corridor-surviving "
                    f"tracklet id(s) -> "
                    f"{sum(len(v) for v in obs_by_trkid.values())} obs rows",
                    L_FIREHOSE)
            for window, cands in zip(batch, per_window):
                yield (window, *_assemble_window(window, cands, obs_by_trkid,
                                              want_ranking))

    def _survivors(self, window, feats, ra_tol, dec_tol, prefilter=None):
        """The feature rows of ``feats`` whose tracklet falls inside ``window``'s
        corridor, capped at ``cfg.max_candidates_per_window`` -- keeping the
        candidates whose OWN motion best matches the corridor (see
        :func:`cap_rank_key`).

        ``prefilter`` overrides ``cfg.prefilter_motion`` for this call (None ->
        the config value); like ``row_budget`` it travels per-sweep, because the
        search entry point cannot edit this object's config -- the source keeps
        its OWN reference, so a ``dataclasses.replace`` there never arrives
        here.  Every survivor carries the bound's verdict in
        ``_motion_bound_ok`` whether or not it was enforced, so an ML feature
        collector (which sweeps UNfiltered -- the pruned tracklets are its
        negative class) still records which rows the filter would have
        dropped.

        Survivors also carry the cap's own bookkeeping -- ``_n_in_corridor``
        (the pre-cap corridor population) and ``_cap_rank`` (this candidate's
        0-based position under :func:`cap_rank_key`) -- so the training
        collector can tell a candidate that was comfortably inside the cap from
        one that only just made it, and so a planted truth can be scored
        against the same ordering it never actually faced (see
        :meth:`OverlaySource._merge_fakes`)."""
        if prefilter is None:
            prefilter = self.cfg.prefilter_motion
        cands = []
        for r in feats:
            hit = tracklet_in_window(r, window, ra_tol, dec_tol)
            if hit is None:
                continue
            ok = motion_bound_ok(hit, window, self.cfg)
            if prefilter and not ok:
                continue
            hit["_motion_bound_ok"] = ok
            cands.append(hit)
        cap = self.cfg.max_candidates_per_window
        n_in = len(cands)
        # Rank ONLY when the cap actually bites.  The sort is O(n log n) on a
        # list the caller is about to iterate anyway, but ranking
        # unconditionally would put it on every window of every production
        # sweep, so it stays behind the one condition that needs it.  A window
        # the cap never touched hands the collector `_cap_rank = -1` rather
        # than a position: nothing competed, which is a different fact from
        # "ranked first" and the ranker should not read it as one.
        if cap and n_in > cap:
            cands.sort(key=lambda c: cap_rank_key(c, window))
            for i, c in enumerate(cands):
                c["_cap_rank"] = i
            cands = cands[:cap]
        for c in cands:
            c["_n_in_corridor"] = n_in
            c.setdefault("_cap_rank", -1)   # -1 = cap never bit this window
            c["_would_be_capped"] = False   # it is HERE, so it survived
        return cands

    def _obs_for_trkids(self, trkids):
        import psycopg2.extras
        s = self.index_cfg.schema
        def _fetch_obs():
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT {s.obs80_col} AS obs80, {s.ra_col}::float AS ra,
                           {s.dec_col}::float AS dec,
                           EXTRACT(EPOCH FROM {s.obstime_col})::float AS epoch_sec,
                           {s.trksub_col} AS trksub, {s.stn_col} AS stn,
                           {s.provid_col} AS provid,
                           {s.tracklet_id_col} AS trkid,
                           {s.ades_select()}
                    FROM {s.obs_table}
                    WHERE {s.tracklet_id_col} = ANY(%s)
                    ORDER BY DATE({s.obstime_col}), {s.trksub_col}, {s.stn_col},
                             {s.obstime_col}
                """, (list(trkids),))
                return [dict(r) for r in cur.fetchall()]
        return self._guarded(_fetch_obs)

    def designation_obs(self, provids):
        """Every archived observation of each DESIGNATION in ``provids``
        (unpacked ``provid`` spelling), for the partial-pickup check.

        Not a corridor query and deliberately not one: the question is "what
        ELSE is filed under this designation", which the corridor by
        construction cannot answer -- the tracklets it never retrieved are
        exactly the ones :mod:`itf.pickup` needs to test.  ``obs_sbn`` has a
        provid index, so this is a handful of index scans (14 buffers, 0.3 ms
        for a 10-observation designation) and never the seq scan the sweep
        paths are so careful to avoid.

        A cancelled query answers ``[]`` -- the check is advisory, and a
        caller that cannot ask simply does not warn.
        """
        import psycopg2.extras
        s = self.index_cfg.schema
        provids = [p for p in (provids or ()) if p]
        if not provids:
            return []

        def _fetch():
            with self.conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT {s.obs80_col} AS obs80,
                           {s.trksub_col} AS trksub, {s.stn_col} AS stn,
                           {s.provid_col} AS provid,
                           {s.tracklet_id_col} AS trkid
                    FROM {s.obs_table}
                    WHERE {s.provid_col} = ANY(%s)
                    ORDER BY {s.provid_col}, {s.obstime_col}
                """, (provids,))
                return [dict(r) for r in cur.fetchall()]
        rows = self._guarded(_fetch)
        return [] if rows is self.CANCELLED else rows

    # ---- legacy per-window scan over obs_sbn (+ obs_noorbit) --------------
    def _scan_path(self, window, ra_tol, dec_tol):
        """The proven notebook query: spatial box + time range + two quadratic
        corridors, over true ITF in ``obs_sbn`` plus asteroid-style
        ``obs_noorbit`` (the retired hand-maintained pseudo-ITF -- kept here
        verbatim for A/B fidelity with the old pipeline)."""
        import psycopg2.extras
        s = self.index_cfg.schema
        qa, qb, qc = window.ra_quad
        da, db_, dc = window.dec_quad
        corridor = f"""
              AND ra BETWEEN
                  %(qa)s*POWER(EXTRACT(EPOCH FROM obstime)-%(t0)s, 2)
                  + %(qb)s*(EXTRACT(EPOCH FROM obstime)-%(t0)s) + %(qc)s - %(ra_tol)s
              AND %(qa)s*POWER(EXTRACT(EPOCH FROM obstime)-%(t0)s, 2)
                  + %(qb)s*(EXTRACT(EPOCH FROM obstime)-%(t0)s) + %(qc)s + %(ra_tol)s
              AND dec BETWEEN
                  %(da)s*POWER(EXTRACT(EPOCH FROM obstime)-%(t0)s, 2)
                  + %(db)s*(EXTRACT(EPOCH FROM obstime)-%(t0)s) + %(dc)s - %(dec_tol)s
              AND %(da)s*POWER(EXTRACT(EPOCH FROM obstime)-%(t0)s, 2)
                  + %(db)s*(EXTRACT(EPOCH FROM obstime)-%(t0)s) + %(dc)s + %(dec_tol)s
        """
        base = f"""
            SELECT {s.obs80_col} AS obs80, {s.ra_col}::float AS ra,
                   {s.dec_col}::float AS dec,
                   EXTRACT(EPOCH FROM {s.obstime_col})::float AS epoch_sec,
                   {s.trksub_col} AS trksub, {s.stn_col} AS stn,
                   {s.provid_col} AS provid,
                   NULL::text AS trkid,
                   {s.ades_select()}
            FROM {{table}}
            WHERE ra >= %(ra_lo)s AND ra <= %(ra_hi)s
              AND dec >= %(dec_lo)s AND dec <= %(dec_hi)s
              AND obstime >= to_timestamp(%(t_lo)s)
              AND obstime <= to_timestamp(%(t_hi)s)
              {corridor}
        """
        params = {
            "qa": qa, "qb": qb, "qc": qc, "da": da, "db": db_, "dc": dc,
            "t0": window.unix_cur,
            "ra_lo": window.min_ra - ra_tol - BOX_PAD_DEG,
            "ra_hi": window.max_ra + ra_tol + BOX_PAD_DEG,
            "dec_lo": window.min_dec - dec_tol - BOX_PAD_DEG,
            "dec_hi": window.max_dec + dec_tol + BOX_PAD_DEG,
            "t_lo": window.unix_start, "t_hi": window.unix_end,
            "ra_tol": ra_tol, "dec_tol": dec_tol,
        }
        sql = (
            "(" + base.format(table=s.obs_table)
            + f" AND {s.itf_obs()}"
            + f" ORDER BY DATE({s.obstime_col}), trksub, stn, {s.obstime_col})"
            + " UNION ALL "
            + "(" + base.format(table="obs_noorbit")
            + f" AND {s.asteroid_obs()}"
            + f" ORDER BY DATE({s.obstime_col}), trksub, stn, {s.obstime_col})"
        )
        def _fetch_scan():
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
        return self._guarded(_fetch_scan)


# ---------------------------------------------------------------------------
# Overlay source (recovery harness / offline runs)
# ---------------------------------------------------------------------------
class OverlaySource:
    """Wraps a base source (or None for offline) and injects synthetic
    tracklets, admitting them ONLY through the same predicates the database
    paths apply -- so a planted tracklet reaches the search exactly iff a real
    ITF tracklet with the same geometry would have.

    ``fakes``: list of dicts ``{"feat": feature-dict (from
    :func:`pyroorb.itf.index.compute_tracklet_features`), "obs": [candidate obs
    row, ...]}``.
    """

    def __init__(self, base, fakes, cfg, index_cfg: IndexConfig = None):
        self.base = base
        self.fakes = fakes
        self.cfg = cfg
        self.index_cfg = index_cfg or IndexConfig()

    def sweep_candidates(self, windows, uncert_mult, extrawidth, want_ranking,
                         row_budget=None, prefilter=None):
        """Whole-sweep iteration: ride the base source's (possibly chunked)
        sweep and merge the gated fakes per window."""
        if self.base is not None:
            base_iter = self.base.sweep_candidates(windows, uncert_mult,
                                                   extrawidth, want_ranking,
                                                   row_budget=row_budget,
                                                   prefilter=prefilter)
        else:
            base_iter = ((w, [], {}, {}) for w in windows)
        for window, rows, chi2, feats in base_iter:
            yield (window, *self._merge_fakes(window, list(rows), dict(chi2),
                                           uncert_mult, extrawidth,
                                           want_ranking, feats=dict(feats)))

    def _merge_fakes(self, window, rows, chi2, uncert_mult, extrawidth,
                     want_ranking, feats=None):
        feats = feats if feats is not None else {}
        ra_tol, dec_tol = window_tolerances(window, uncert_mult, extrawidth)
        if self.cfg.candidate_path == "index":
            pix = None
            # The real population this window returned, ranked the way the cap
            # ranks -- computed ONCE per window, then bisected per plant.  A
            # plant is injected past _survivors and so never faces the cap; this
            # is how we find out what the cap WOULD have done to it, which is
            # the only way the harness can see a recall risk that only bites
            # real tracklets.  See training.META_COLS.
            real_keys = None
            cap = self.cfg.max_candidates_per_window
            n_in_corridor = next((c.get("_n_in_corridor") for c in feats.values()
                                  if c.get("_n_in_corridor") is not None), 0)
            planted = []
            for fk in self.fakes:
                feat = fk["feat"]
                # replicate the SQL prefilter: healpix disc + epoch + dec band
                if pix is None:
                    pix = window_healpix_disc(window, ra_tol, dec_tol, self.index_cfg)
                if feat["healpix"] not in pix:
                    continue
                hit = tracklet_in_window(feat, window, ra_tol, dec_tol)
                if hit is None:
                    continue
                # a planted truth never faces the motion pre-filter (it is
                # injected past the DB), but its would-be verdict is recorded
                # anyway: a collected POSITIVE with a False verdict is a true
                # tracklet the pre-filter would have pruned -- free recall
                # telemetry in the training data
                hit["_motion_bound_ok"] = motion_bound_ok(hit, window, self.cfg)
                # ...and, on the same principle, the cap's would-be verdict.
                # A plant that ranks below every real SURVIVOR of a window that
                # hit the cap is one the cap would have discarded: the survivors
                # are the top `cap` of the pre-cap population, so ranking worse
                # than the worst of them puts the plant outside the top `cap`.
                if real_keys is None:
                    real_keys = sorted(
                        cap_rank_key(c, window) for c in feats.values()
                        if c.get("_would_be_capped") is False)
                # The plant's position among the real candidates IS what its
                # rank would be if it had come from the database, so the cap
                # would discard it exactly when that position falls outside
                # `cap`.  (Note this is the right test even for a window merely
                # AT the cap: adding one more candidate is what pushes the last
                # one out.)
                rank = bisect.bisect_left(real_keys, cap_rank_key(hit, window))
                hit["_cap_rank"] = rank
                hit["_would_be_capped"] = bool(cap and rank >= cap)
                rows.extend(fk["obs"])
                feats[feat["tracklet_id"]] = hit
                planted.append(hit)
                if want_ranking:
                    chi2[feat["tracklet_id"]] = match_chi2(hit, window)[0]
            # The corridor POPULATION counts every candidate a real sweep would
            # have counted -- the plants included, since a real tracklet of the
            # same geometry would have been.  It used to hand a plant the
            # window's REAL count (0 when the window held no real row) and leave
            # the real rows' count without the plants: a real candidate can
            # never read 0, so "n_in_corridor == 0" marked a plant (median 0
            # for planted truths vs 470 for real rows, measured 2026-09-26) and
            # every ranker trained on harness rows could learn the label from it.
            if planted:
                n_all = n_in_corridor + len(planted)
                for c in feats.values():
                    c["_n_in_corridor"] = n_all
        else:
            for fk in self.fakes:
                passing = [r for r in fk["obs"]
                           if obs_in_window(r, window, ra_tol, dec_tol)]
                rows.extend(passing)

        rows.sort(key=_row_sort_key)
        return rows, chi2, feats
