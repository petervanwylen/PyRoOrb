"""The precomputed per-tracklet index (``itf.itf_tracklets``) and its upkeep.

One row per tracklet (on-sky motion + uncertainty, mean magnitude, geometry)
for the population the ITF search cares about -- true **ITF** (``obs_sbn``
``provid IS NULL AND status='I'``, and *pure*: every obs of the trkid ITF) plus
**pseudo-ITF** (asteroid-style designation families with <= ``max_tracklets``
distinct tracklets OR absent from ``mpc_orbits``).  The search then runs one
indexed healpix query per ephemeris window instead of scanning the ~538M-row
``obs_sbn``.

PyRoOrb owns this table's maintenance.  Three jobs, all via
``dbio.connect_index`` (the ``itf`` role), all idempotent and
watermark/checkpoint-resumable so a scheduler can just call them:

* :func:`refresh_incremental` -- every ~5-10 min.  Dirty trkids via a dual
  watermark (``updated_at`` OR ``id``; the signals cover each other), admits
  new pure-ITF tracklets, refreshes members, deletes rows that turned
  mixed/empty/gone.
* :func:`reconcile_membership` -- daily.  Owns the pseudo-ITF set: admits
  newly-qualifying families, evicts those that gained an orbit + >3 tracklets
  or vanished.
* :func:`sweep_validity` -- weekly.  Chunked full-table backstop for exits with
  no incremental signal (hard deletes; rare stale-``updated_at`` rows).

Every deletion is audited to ``itf.itf_tracklets_evictions``.

Ported from the proven ``ift outline/itflib/tracklet_index.py`` (validated
against the live SBN DB; schema facts confirmed 2026-07); logic unchanged.
Server etiquette: batch writes (``execute_values``), throttle bulk reads,
never seq-scan ``obs_sbn``.
"""

import time
from dataclasses import dataclass, field
from datetime import timedelta
from statistics import mode

import numpy as np

from pyroorb import healpix


def _weighted_linfit(t, y, w):
    """Weighted least-squares fit ``y = a + b*t``.

    Returns ``(a, b, cov)`` where ``cov`` is the 2x2 parameter covariance.
    With weights ``w = 1/sigma**2`` the normal matrix is the Fisher
    information, so its inverse propagates the astrometric uncertainties
    straight into position/velocity error.
    """
    Sw = w.sum(); Swt = (w * t).sum(); Swtt = (w * t * t).sum()
    AtWA = np.array([[Sw, Swt], [Swt, Swtt]])
    AtWy = np.array([(w * y).sum(), (w * t * y).sum()])
    cov = np.linalg.inv(AtWA)
    beta = cov @ AtWy
    return float(beta[0]), float(beta[1]), cov


def healpix_of(ra_deg, dec_deg, nside, nested=True):
    """Healpix pixel for a sky position."""
    return healpix.ang2pix(nside, float(ra_deg), float(dec_deg), nest=nested)


# ---------------------------------------------------------------------------
# Database name bindings (confirmed against the live SBN DB, 2026-07).
# ---------------------------------------------------------------------------
@dataclass
class IndexSchema:
    # obs source table + columns. NB: obs_noorbit is NOT used (hand-maintained
    # subset of obs_sbn); membership is computed here instead.
    obs_table: str = "obs_sbn"
    tracklet_id_col: str = "trkid"     # TEXT per-tracklet id, indexed on obs
    obs_id_col: str = "id"             # monotonic serial PK -> insert signal
    updated_at_col: str = "updated_at" # change signal; bumps on real
                                       # identifications (verified) -- paired
                                       # with the id belt for rare stale rows
    ra_col: str = "ra"
    dec_col: str = "dec"
    obstime_col: str = "obstime"
    provid_col: str = "provid"         # UNPACKED provisional designation
    stn_col: str = "stn"
    trksub_col: str = "trksub"
    obs80_col: str = "obs80"
    status_col: str = "status"
    # ADES rms/mag/band are INLINE columns (no separate ADES table).
    # rmsra/rmsdec are ON-SKY arcsec -- rmsra already includes cos(dec).
    rms_ra_col: str = "rmsra"
    rms_dec_col: str = "rmsdec"
    mag_col: str = "mag"
    band_col: str = "band"
    # the rest of the per-obs ADES payload, carried into the FIT (itf/obsreg.py)
    # rather than into the index: an 80-column rendering cannot hold any of it.
    ades_obsid_col: str = "obsid"      # ADES obsID (NB: obs_id_col is the PK)
    rms_mag_col: str = "rmsmag"
    rms_time_col: str = "rmstime"
    logsnr_col: str = "logsnr"
    mode_col: str = "mode"
    astcat_col: str = "astcat"
    # identification family rollup: obs.provid == unpacked_secondary (UNIQUE).
    ident_table: str = "current_identifications"
    ident_primary_col: str = "unpacked_primary_provisional_designation"
    ident_secondary_col: str = "unpacked_secondary_provisional_designation"
    # orbit presence = "real object" signal (primary_objects.no_orbit is DEAD).
    orbits_table: str = "mpc_orbits"
    orbits_primary_col: str = "unpacked_primary_provisional_designation"
    orbits_nobs_col: str = "nobs_total"
    # target + bookkeeping (schema owned by the itf role; its search_path is
    # "itf, public" so unqualified writes land there -- other roles must read
    # via ``target_qualified``).
    target_schema: str = "itf"
    target_table: str = "itf_tracklets"
    sync_table: str = "itf_tracklets_sync"
    audit_table: str = "itf_tracklets_evictions"

    def ades_select(self, alias: str = "") -> str:
        """The per-obs ADES payload as SELECT items, aliased to the ADES FIELD
        names -- which is what :func:`itf.obsreg._extras_from_record` (and
        :func:`ades.records_to_obs`) read.  One definition, so a candidate row
        carries the same ADES fields whichever query fetched it."""
        p = f"{alias}." if alias else ""
        return ", ".join(
            f"{p}{col} AS {name}" for col, name in (
                (self.ades_obsid_col, "obsid"),
                (self.rms_ra_col, "rmsra"),
                (self.rms_dec_col, "rmsdec"),
                (self.rms_mag_col, "rmsmag"),
                (self.rms_time_col, "rmstime"),
                (self.logsnr_col, "logsnr"),
                (self.mode_col, "mode"),
                (self.astcat_col, "astcat"),
            ))

    @property
    def target_qualified(self) -> str:
        return f"{self.target_schema}.{self.target_table}"

    @property
    def audit_qualified(self) -> str:
        return f"{self.target_schema}.{self.audit_table}"

    # THE definitions of the two membership predicates at the obs level.
    # Every query that means "this obs is ITF-state" / "asteroid-style"
    # must use these, so a policy or schema change lands everywhere at once.
    def itf_obs(self, alias: str = "") -> str:
        """SQL: the obs row is in the ITF state (unidentified, status 'I')."""
        p = f"{alias}." if alias else ""
        return f"({p}{self.provid_col} IS NULL AND {p}{self.status_col} = 'I')"

    def asteroid_obs(self, alias: str = "") -> str:
        """SQL: the obs80 record is asteroid-style (col 5 blank -- not a
        comet/satellite letter)."""
        p = f"{alias}." if alias else ""
        return f"substring({p}{self.obs80_col}, 5, 1) = ' '"


@dataclass
class IndexConfig:
    schema: IndexSchema = field(default_factory=IndexSchema)
    # pseudo-ITF membership: family has <= this many distinct tracklets OR no
    # mpc_orbits row ("fewer than 4 tracklets" == <= 3).
    max_tracklets: int = 3
    # membership prefilter: only families with no orbit or an orbit using <=
    # this many obs are tracklet-counted (a well-observed orbit cannot be a
    # <=max_tracklets object) -- keeps the count join off the obs_sbn seq scan.
    orbit_nobs_prefilter: int = 30
    watermark_overlap: timedelta = timedelta(minutes=15)
    feature_version: int = 2           # v2: weighted fit + per-axis sigmas
    # a >=3-obs tracklet is "linear_ok" below this chi2/dof
    linear_chi2_max: float = 3.0
    # fallback per-obs astrometric sigma (arcsec, on-sky) when ADES rms absent
    # (true-ITF obs almost never carry rmsra)
    default_sigma_arcsec: float = 1.0
    # healpix bucket for the spatial prefilter (no q3c/pg_sphere on the SBN
    # server). nside=64 ~ 55' pixels; the exact corridor test runs after.
    healpix_nside: int = 64
    healpix_nested: bool = True


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
def ensure_schema(conn, index_cfg: IndexConfig = None):
    """Create the target table, indexes, watermark + audit tables if absent.
    Idempotent."""
    index_cfg = index_cfg or IndexConfig()
    s = index_cfg.schema
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {s.target_table} (
        tracklet_id        text PRIMARY KEY,
        healpix            bigint,
        trksub             text,
        stn                text,
        provid             text,
        source             text,
        object_total_tracklets smallint,
        n_obs              int,
        epoch_start_sec    double precision,
        epoch_mid_sec      double precision,
        epoch_end_sec      double precision,
        center_ra          double precision,
        center_dec         double precision,
        ra_min double precision, ra_max double precision,
        dec_min double precision, dec_max double precision,
        rate_deg_day       double precision,
        pa_deg             double precision,
        dra_cosdec_rate    double precision,
        ddec_rate          double precision,
        curvature          double precision,
        rate_sigma         double precision,
        pa_sigma           double precision,
        sig_pos_e_arcsec   double precision,   -- on-sky pos sigma @epoch_mid, East
        sig_pos_n_arcsec   double precision,   -- on-sky pos sigma @epoch_mid, North
        sig_vel_e_degday   double precision,   -- velocity sigma, East (on-sky)
        sig_vel_n_degday   double precision,   -- velocity sigma, North
        fit_rms_arcsec     double precision,
        fit_chi2_dof       double precision,
        linear_ok          boolean,
        mag_mean           real,
        mag_rms            real,
        band               text,
        n_mag              int,
        feature_version    smallint,
        computed_at        timestamptz default now()
    );
    CREATE INDEX IF NOT EXISTS itf_trk_healpix
        ON {s.target_table} (healpix);
    -- The corridor sweep's real access pattern: one narrow EPOCH range inside a
    -- given healpix pixel, per window.  With healpix alone the planner reads
    -- every epoch ever recorded in the pixel and filters afterwards (measured:
    -- 573k rows/worker and 2.4M buffer hits to return 321 -- 668 ms); with the
    -- composite it is an index range scan per pixel (27k buffers, 60 ms).
    -- center_dec is INCLUDEd so the dec-band test is answered from the index.
    CREATE INDEX IF NOT EXISTS itf_trk_healpix_epoch
        ON {s.target_table} (healpix, epoch_mid_sec) INCLUDE (center_dec);
    CREATE INDEX IF NOT EXISTS itf_trk_epoch
        ON {s.target_table} USING brin (epoch_mid_sec);
    CREATE INDEX IF NOT EXISTS itf_trk_rate
        ON {s.target_table} (rate_deg_day);
    CREATE INDEX IF NOT EXISTS itf_trk_itf
        ON {s.target_table} (center_dec) WHERE provid IS NULL;

    CREATE TABLE IF NOT EXISTS {s.sync_table} (
        id             int PRIMARY KEY DEFAULT 1,
        last_watermark timestamptz,        -- updated_at belt (best-effort)
        last_id        bigint DEFAULT 0,   -- obs_sbn.id insert watermark
        last_run       timestamptz,
        backfill_started   timestamptz,    -- snapshot instant D
        backfill_through   timestamp,      -- obstime frontier
        backfill_done      boolean DEFAULT false,
        CHECK (id = 1)
    );
    ALTER TABLE {s.sync_table} ADD COLUMN IF NOT EXISTS backfill_started timestamptz;
    ALTER TABLE {s.sync_table} ADD COLUMN IF NOT EXISTS backfill_through timestamp;
    ALTER TABLE {s.sync_table} ADD COLUMN IF NOT EXISTS backfill_done boolean DEFAULT false;
    ALTER TABLE {s.sync_table} ADD COLUMN IF NOT EXISTS last_id bigint DEFAULT 0;
    INSERT INTO {s.sync_table}(id, last_watermark)
        VALUES (1, 'epoch') ON CONFLICT DO NOTHING;

    CREATE TABLE IF NOT EXISTS {s.audit_table} (
        tracklet_id text,
        reason      text,
        source      text,          -- 'itf' | 'pseudo_itf' at exit time
        evicted_at  timestamptz DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS itf_evict_at ON {s.audit_table} (evicted_at);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


# ---------------------------------------------------------------------------
# Feature computation (pure numpy -- curvature, chi2 and great-circle
# geometry are why this is not a SQL materialized view).
# ---------------------------------------------------------------------------
def compute_tracklet_features(rows, index_cfg: IndexConfig):
    """One feature dict from a tracklet's observation rows.

    ``rows``: list of dicts with keys ``es`` (epoch seconds), ``ra``, ``dec``
    (deg), ``rms_ra``, ``rms_dec`` (arcsec on-sky, may be None), ``mag``,
    ``band``, ``stn``, ``trksub``, ``provid``, ``tracklet_id``.  Returns None
    for degenerate (single-epoch) tracklets.
    """
    es = np.array([r["es"] for r in rows], float)
    ra = np.array([r["ra"] for r in rows], float)
    dec = np.array([r["dec"] for r in rows], float)
    n = len(rows)
    if n < 2 or es.max() == es.min():
        return None

    # great-circle center via mean unit vector (wrap-/pole-safe)
    rar, decr = np.radians(ra), np.radians(dec)
    vx = np.cos(decr) * np.cos(rar)
    vy = np.cos(decr) * np.sin(rar)
    vz = np.sin(decr)
    mv = np.array([vx.mean(), vy.mean(), vz.mean()])
    mv /= np.linalg.norm(mv)
    center_dec = np.degrees(np.arcsin(mv[2]))
    center_ra = np.degrees(np.arctan2(mv[1], mv[0])) % 360
    cosd = np.cos(np.radians(center_dec))

    # tangent-plane coords about the center (deg), East/North
    dra = ((ra - center_ra + 180) % 360 - 180) * cosd
    ddec = dec - center_dec
    t = es - es.mean()

    day = 86400.0
    # rms_ra is ALREADY on-sky (cos-dec included) so it pairs directly with the
    # East coordinate -- cos(dec) enters exactly once, in dra, never the sigma.
    d0 = index_cfg.default_sigma_arcsec
    sig_e_as = np.array([r["rms_ra"] if r["rms_ra"] else d0 for r in rows], float)
    sig_n_as = np.array([r["rms_dec"] if r["rms_dec"] else d0 for r in rows], float)
    we = 1.0 / (sig_e_as / 3600.0) ** 2
    wn = 1.0 / (sig_n_as / 3600.0) ** 2

    aE, bE, covE = _weighted_linfit(t, dra, we)            # deg, deg/sec
    aN, bN, covN = _weighted_linfit(t, ddec, wn)
    dra_cosdec_rate = bE * day
    ddec_rate = bN * day
    rate_deg_day = float(np.hypot(dra_cosdec_rate, ddec_rate))
    pa_deg = float(np.degrees(np.arctan2(dra_cosdec_rate, ddec_rate)) % 360)

    # At the weighted mean epoch the intercept/slope covariance ~0, so position
    # and velocity sigmas propagate independently downstream. Kept per-axis
    # (E/N) so the search can build a proper chi2.
    sig_pos_e_arcsec = float(np.sqrt(covE[0, 0]) * 3600.0)
    sig_pos_n_arcsec = float(np.sqrt(covN[0, 0]) * 3600.0)
    sig_vel_e_degday = float(np.sqrt(covE[1, 1]) * day)
    sig_vel_n_degday = float(np.sqrt(covN[1, 1]) * day)
    rate_sigma = float(np.hypot(sig_vel_e_degday, sig_vel_n_degday))
    pa_sigma = (float(np.degrees(rate_sigma / rate_deg_day))
                if rate_deg_day > 0 else None)

    fit_rms_arcsec = None
    fit_chi2_dof = None
    curvature = None
    linear_ok = None
    if n >= 3:
        resid_e = dra - (aE + bE * t)
        resid_n = ddec - (aN + bN * t)
        # PER-OBSERVATION (divides by n, not 2n) -- an exception to the
        # per-axis reporting standard (see itf.fitter's docstring) and one that
        # must stay: this is a STORED column of itf.itf_tracklets with millions
        # of rows already written, a live ML feature (training.trk_fit_rms) and
        # a term in the SQL motion prefilter's pushdown.  Re-basing it would
        # need the whole index re-derived and every ranker retrained, for a
        # number no person reads.  Convert with fitter.per_axis if one ever does.
        fit_rms_arcsec = float(np.sqrt(np.mean(resid_e**2 + resid_n**2)) * 3600)
        dof = 2 * n - 4
        if dof > 0:
            chi2 = float(np.sum(we * resid_e**2 + wn * resid_n**2))
            fit_chi2_dof = chi2 / dof
            linear_ok = bool(fit_chi2_dof < index_cfg.linear_chi2_max)
        # quadratic accel magnitude (deg/day^2); normalize t so the Vandermonde
        # columns are well-conditioned.
        baseline = es.max() - es.min()
        tn = t / baseline
        ae = np.polyfit(tn, dra, 2)[0] * 2 * (day / baseline) ** 2
        an = np.polyfit(tn, ddec, 2)[0] * 2 * (day / baseline) ** 2
        curvature = float(np.hypot(ae, an))

    mags = [r["mag"] for r in rows if r["mag"] is not None]
    bands = [r["band"] for r in rows if r.get("band")]
    return {
        "tracklet_id": rows[0]["tracklet_id"],
        "trksub": rows[0]["trksub"], "stn": rows[0]["stn"],
        "provid": rows[0]["provid"],
        # provid NULL -> true ITF; set -> pseudo-ITF
        "source": "pseudo_itf" if rows[0]["provid"] else "itf",
        "healpix": healpix_of(center_ra, center_dec,
                              index_cfg.healpix_nside, index_cfg.healpix_nested),
        "n_obs": n,
        "epoch_start_sec": float(es.min()),
        "epoch_mid_sec": float(es.mean()),
        "epoch_end_sec": float(es.max()),
        "center_ra": float(center_ra), "center_dec": float(center_dec),
        "ra_min": float(ra.min()), "ra_max": float(ra.max()),
        "dec_min": float(dec.min()), "dec_max": float(dec.max()),
        "rate_deg_day": rate_deg_day, "pa_deg": pa_deg,
        "dra_cosdec_rate": float(dra_cosdec_rate), "ddec_rate": float(ddec_rate),
        "curvature": curvature,
        "rate_sigma": rate_sigma, "pa_sigma": pa_sigma,
        "sig_pos_e_arcsec": sig_pos_e_arcsec, "sig_pos_n_arcsec": sig_pos_n_arcsec,
        "sig_vel_e_degday": sig_vel_e_degday, "sig_vel_n_degday": sig_vel_n_degday,
        "fit_rms_arcsec": fit_rms_arcsec, "fit_chi2_dof": fit_chi2_dof,
        "linear_ok": linear_ok,
        "mag_mean": float(np.mean(mags)) if mags else None,
        "mag_rms": float(np.std(mags)) if len(mags) > 1 else None,
        "band": mode(bands) if bands else None,
        "n_mag": len(mags),
    }


# ---------------------------------------------------------------------------
# Incremental (~5-10 min) refresh: tracklet-level only.
# ---------------------------------------------------------------------------
def refresh_incremental(conn, index_cfg: IndexConfig = None):
    """Frequent refresh: keeps geometry fresh and admits new *true ITF*
    tracklets immediately.  pseudo-ITF membership is owned by
    :func:`reconcile_membership`."""
    index_cfg = index_cfg or IndexConfig()
    s = index_cfg.schema
    with conn.cursor() as cur:
        cur.execute(f"SELECT last_watermark, COALESCE(last_id, 0) "
                    f"FROM {s.sync_table} WHERE id=1")
        wm, last_id = cur.fetchone()
        scan_from = wm - index_cfg.watermark_overlap
        # dirty trkids via BOTH signals (mutually covering), restricted to ITF
        # candidates or existing members
        cur.execute(f"""
            SELECT DISTINCT o.{s.tracklet_id_col} FROM {s.obs_table} o
                WHERE (o.{s.updated_at_col} > %s OR o.{s.obs_id_col} > %s)
                  AND ( {s.itf_obs('o')}
                        OR EXISTS (SELECT 1 FROM {s.target_table} t
                                   WHERE t.tracklet_id = o.{s.tracklet_id_col}) )
        """, (scan_from, last_id))
        dirty = [r[0] for r in cur.fetchall()]
    if not dirty:
        _bump_watermark(conn, index_cfg)
        return 0

    with conn.cursor() as cur:
        cur.execute(f"SELECT tracklet_id, source FROM {s.target_qualified} "
                    f"WHERE tracklet_id = ANY(%s)", (dirty,))
        in_target = dict(cur.fetchall())         # tracklet_id -> current source

    # Classify (purity rule) and act by current source:
    #  * pure_itf                         -> admit / refresh as ITF (also re-admits a
    #                                        row whose identification was reverted);
    #  * pseudo member still 'designated' -> refresh its geometry;
    #  * ANY other in-target row -- an ITF row that was identified (I->p/P), left
    #    status 'I', went mixed, dropped <2 obs, or lost its obs -> BACK IT OUT NOW
    #    (audited). reconcile re-admits it as pseudo_itf next run IF the identified
    #    object actually qualifies (<=3 tracklets / no orbit); that's the safety net.
    cls = _classify_trkids(conn, s, dirty)
    to_compute, to_delete = [], []
    for t in dirty:
        c, src = cls[t], in_target.get(t)
        if c == 'pure_itf':
            to_compute.append(t)
        elif src == 'pseudo_itf' and c == 'designated':
            to_compute.append(t)
        elif src is not None:
            to_delete.append(t)

    n = 0
    if to_compute:
        rows_by = _load_tracklet_obs(conn, index_cfg, to_compute)
        feats = [f for f in (compute_tracklet_features(v, index_cfg)
                             for v in rows_by.values()) if f]
        _upsert(conn, index_cfg, feats)
        n = len(feats)
    if to_delete:
        with conn.cursor() as cur:
            _delete_with_audit(cur, s, "t.tracklet_id = ANY(%s)",
                               "refresh: no longer valid ITF (identified / status != I / mixed / gone)",
                               params=(to_delete,))
    _bump_watermark(conn, index_cfg)
    conn.commit()
    return n


def _classify_trkids(conn, s, trkids):
    """Classify each trkid by its obs composition (single trkid-index scan):

      'pure_itf'   -- >=2 obs, ALL (provid NULL AND status='I')
      'designated' -- >=2 obs, NONE ITF-state       -> pseudo-ITF candidate
      'mixed'      -- both kinds                    -> EXCLUDED entirely
      'degenerate' -- <2 obs
      'gone'       -- no obs at all
    """
    out = {t: 'gone' for t in trkids}
    if not trkids:
        return out
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT {s.tracklet_id_col},
                   count(*) FILTER (WHERE {s.itf_obs()}) AS n_itf,
                   count(*) FILTER (WHERE NOT {s.itf_obs()}) AS n_other,
                   count(*) AS n_tot
            FROM {s.obs_table} WHERE {s.tracklet_id_col} = ANY(%s)
            GROUP BY {s.tracklet_id_col}
        """, (list(trkids),))
        for trk, n_itf, n_other, n_tot in cur.fetchall():
            if n_tot < 2:
                out[trk] = 'degenerate'
            elif n_other == 0:
                out[trk] = 'pure_itf'
            elif n_itf == 0:
                out[trk] = 'designated'
            else:
                out[trk] = 'mixed'
    return out


def _load_tracklet_obs(conn, index_cfg, tracklet_ids):
    """All obs for the given trkids from obs_sbn, grouped by trkid (rms/mag/
    band are inline columns; no join)."""
    s = index_cfg.schema
    out = {}
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT o.{s.tracklet_id_col},
                   EXTRACT(EPOCH FROM o.{s.obstime_col})::float,
                   o.{s.ra_col}::float, o.{s.dec_col}::float,
                   o.{s.rms_ra_col}::float, o.{s.rms_dec_col}::float,
                   o.{s.mag_col}::float, o.{s.band_col},
                   o.{s.trksub_col}, o.{s.stn_col}, o.{s.provid_col}
            FROM {s.obs_table} o
            WHERE o.{s.tracklet_id_col} = ANY(%s)
            ORDER BY o.{s.tracklet_id_col}, o.{s.obstime_col}
        """, (list(tracklet_ids),))
        for tid, es, ra, dec, rra, rdec, mag, band, trksub, stn, provid in cur.fetchall():
            out.setdefault(tid, []).append({
                "tracklet_id": tid, "es": es, "ra": ra, "dec": dec,
                "rms_ra": rra, "rms_dec": rdec, "mag": mag, "band": band,
                "trksub": trksub, "stn": stn, "provid": provid,
            })
    return out


UPSERT_COLS = ["tracklet_id", "healpix", "trksub", "stn", "provid", "source",
               "n_obs", "epoch_start_sec", "epoch_mid_sec", "epoch_end_sec",
               "center_ra", "center_dec", "ra_min", "ra_max", "dec_min", "dec_max",
               "rate_deg_day", "pa_deg", "dra_cosdec_rate", "ddec_rate", "curvature",
               "rate_sigma", "pa_sigma",
               "sig_pos_e_arcsec", "sig_pos_n_arcsec",
               "sig_vel_e_degday", "sig_vel_n_degday",
               "fit_rms_arcsec", "fit_chi2_dof", "linear_ok",
               "mag_mean", "mag_rms", "band", "n_mag"]


def _upsert(conn, index_cfg, feats):
    if not feats:
        return
    import psycopg2.extras
    s = index_cfg.schema
    cols = UPSERT_COLS
    updates = ",".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "tracklet_id")
    sql = (f"INSERT INTO {s.target_qualified} ({','.join(cols)},feature_version) "
           f"VALUES %s "
           f"ON CONFLICT (tracklet_id) DO UPDATE SET {updates}, "
           f"feature_version=EXCLUDED.feature_version, computed_at=now()")
    template = "(" + ",".join(["%s"] * (len(cols) + 1)) + ")"
    values = [[f[c] for c in cols] + [index_cfg.feature_version] for f in feats]
    # one round-trip per page, not per row -- the remote RTT dominates
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, values, template=template,
                                       page_size=500)


def _delete_with_audit(cur, s, where_sql, reason, params=()):
    """Delete target rows matching ``where_sql``, logging each exit to the
    audit table in one atomic statement.  Returns the count.
    NB: the target's id column is ``tracklet_id`` (the obs column is ``trkid``)."""
    cur.execute(f"""
        WITH del AS (
            DELETE FROM {s.target_qualified} t
            WHERE {where_sql}
            RETURNING t.tracklet_id, t.source)
        INSERT INTO {s.audit_qualified} (tracklet_id, source, reason)
        SELECT tracklet_id, source, %s FROM del
    """, tuple(params) + (reason,))
    return cur.rowcount


def _bump_watermark(conn, index_cfg):
    s = index_cfg.schema
    with conn.cursor() as cur:
        # advance both watermarks to the current max (both index-backed);
        # the signals back each other up across runs
        cur.execute(f"""
            UPDATE {s.sync_table} SET last_run = now(),
                last_watermark = GREATEST(
                    COALESCE((SELECT max({s.updated_at_col}) FROM {s.obs_table}), last_watermark),
                    last_watermark),
                last_id = GREATEST(
                    COALESCE((SELECT max({s.obs_id_col}) FROM {s.obs_table}), last_id),
                    COALESCE(last_id, 0))
            WHERE id = 1
        """)
    conn.commit()


# ---------------------------------------------------------------------------
# Cold-start backfill: chunked + throttled, by obstime window.
# ---------------------------------------------------------------------------
def backfill_window(conn, index_cfg=None, start=None, end=None,
                    batch=2000, sleep_s=0.0, log=print):
    """Backfill *true ITF* tracklets whose obs fall in ``[start, end)``.

    Gentle, resumable, idempotent.  ITF tracklets are single-night, so an
    obstime window (indexed range scan) captures complete tracklets; obs are
    (re)loaded per trkid so a boundary night is loaded in full.  Does not touch
    the watermark -- :func:`full_sync_itf` owns the checkpointing.
    """
    index_cfg = index_cfg or IndexConfig()
    s = index_cfg.schema
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT {s.tracklet_id_col} FROM {s.obs_table}
            WHERE {s.obstime_col} >= %s AND {s.obstime_col} < %s
              AND {s.itf_obs()}
            GROUP BY {s.tracklet_id_col}
            HAVING count(*) >= 2
        """, (start, end))
        trkids = [r[0] for r in cur.fetchall()]
    if trkids:
        cls = _classify_trkids(conn, s, trkids)
        trkids = [t for t in trkids if cls[t] == 'pure_itf']
    log(f"window {start}..{end}: {len(trkids)} eligible (pure) ITF tracklets")

    total = 0
    for i in range(0, len(trkids), batch):
        chunk = trkids[i:i + batch]
        rows_by = _load_tracklet_obs(conn, index_cfg, chunk)
        feats = [f for f in (compute_tracklet_features(v, index_cfg)
                             for v in rows_by.values()) if f]
        _upsert(conn, index_cfg, feats)
        conn.commit()
        total += len(feats)
        log(f"  {total} upserted (chunk {i // batch + 1})")
        if sleep_s:
            time.sleep(sleep_s)
    return total


def full_sync_itf(conn, index_cfg=None, start=None, until=None, step_days=30,
                  batch=2000, sleep_s=0.3, log=print):
    """Complete, resumable cold-start of the true-ITF population.

    Sweeps obstime windows with a two-part checkpoint (``backfill_started`` =
    snapshot instant D; ``backfill_through`` = obstime frontier), committing
    after each window so an interruption resumes where it left off.  On
    completion sets ``backfill_done`` and ``last_watermark = D``, handing off
    cleanly to :func:`refresh_incremental` (which catches everything changed
    during the sweep).  Throttles in proportion to work done.
    """
    from datetime import datetime, timedelta as td

    index_cfg = index_cfg or IndexConfig()
    s = index_cfg.schema
    with conn.cursor() as cur:
        cur.execute(f"""SELECT backfill_started, backfill_through, backfill_done,
                               now(), now()::timestamp
                        FROM {s.sync_table} WHERE id = 1""")
        started, through, done, now_tz, now_naive = cur.fetchone()
        if done:
            log("full ITF sync already complete; nothing to do")
            return 0
        if started is None:
            started = now_tz
            cur.execute(f"UPDATE {s.sync_table} SET backfill_started=%s WHERE id=1",
                        (started,))
            conn.commit()
        if through is None:
            if start is not None:
                through = datetime.fromisoformat(start)
            else:
                cur.execute(f"""SELECT min({s.obstime_col}) FROM {s.obs_table}
                                WHERE {s.itf_obs()}""")
                through = cur.fetchone()[0]

    end = now_naive
    if until is not None:
        end = min(end, datetime.fromisoformat(until))
    reached_present = end >= now_naive
    log(f"full ITF sync: D={started}, sweeping obstime {through}..{end} "
        f"(step {step_days}d){'' if reached_present else ' [partial]'}")
    total = 0
    w = through
    while w < end:
        w_end = min(w + td(days=step_days), end)
        n = backfill_window(conn, index_cfg, w, w_end, batch=batch, sleep_s=0.0,
                            log=lambda _m: None)
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {s.sync_table} SET backfill_through=%s, last_run=now() "
                        f"WHERE id=1", (w_end,))
        conn.commit()
        total += n
        if n:
            log(f"  {w.date()}..{w_end.date()}: +{n}  (total {total})")
            if sleep_s:
                time.sleep(min(sleep_s * n / batch, sleep_s * 3))
        w = w_end

    if not reached_present:
        log(f"partial sync paused at {end} (not marked done): +{total} this run")
        return total
    with conn.cursor() as cur:
        cur.execute(f"""UPDATE {s.sync_table}
                        SET backfill_done=true, last_watermark=%s, last_run=now()
                        WHERE id=1""", (started,))
    conn.commit()
    log(f"full ITF sync COMPLETE: {total} tracklets; incremental watermark = D = {started}")
    return total


# ---------------------------------------------------------------------------
# Membership reconcile: owns the pseudo-ITF set + orphan cleanup.
# ---------------------------------------------------------------------------
def _pseudo_itf_provids_sql(s, index_cfg):
    """SQL selecting ``(provid, prim, has_orbit)`` for every designation in a
    *candidate* pseudo-ITF family.

    Family = all secondaries of a primary (a designated object's obs may be
    filed under several provisional designations).  Only candidate primaries
    (no orbit, or an orbit with <= ``orbit_nobs_prefilter`` obs) are returned
    for counting, so the count join hits ``obs_sbn`` via the provid index for
    a small set instead of a seq scan.  Comets are excluded downstream by the
    asteroid-style obs80 filter (column 5 blank).
    """
    return f"""
        WITH cand AS (
            SELECT ci.{s.ident_primary_col} AS prim,
                   bool_or(m.{s.orbits_primary_col} IS NOT NULL) AS has_orbit
            FROM {s.ident_table} ci
            LEFT JOIN {s.orbits_table} m
                   ON m.{s.orbits_primary_col} = ci.{s.ident_primary_col}
            WHERE m.{s.orbits_primary_col} IS NULL
               OR m.{s.orbits_nobs_col} <= {index_cfg.orbit_nobs_prefilter}
            GROUP BY ci.{s.ident_primary_col})
        SELECT ci.{s.ident_secondary_col} AS provid, cand.prim, cand.has_orbit
        FROM cand JOIN {s.ident_table} ci
               ON ci.{s.ident_primary_col} = cand.prim
    """


def reconcile_membership(conn, index_cfg: IndexConfig = None, batch=3000,
                         sleep_s=0.2, log=print):
    """Owns pseudo-ITF membership: admit newly-qualifying tracklets, evict
    members that no longer qualify (gained orbit + >3 tracklets, or obs gone).

    pseudo-ITF = asteroid-style designation family with <= ``max_tracklets``
    distinct tracklets OR absent from ``mpc_orbits``, evaluated at the PRIMARY.
    Server-safe: prefiltered candidates, Python-batched counts over the provid
    index, throttled.  Idempotent.
    """
    import psycopg2.extras
    from collections import defaultdict

    index_cfg = index_cfg or IndexConfig()
    s = index_cfg.schema

    # 1) candidate families: provid -> prim, prim -> has_orbit. No obs_sbn.
    with conn.cursor() as cur:
        cur.execute(_pseudo_itf_provids_sql(s, index_cfg))
        prov_prim = {}
        prim_orbit = {}
        for provid, prim, has_orbit in cur.fetchall():
            prov_prim[provid] = prim
            prim_orbit[prim] = has_orbit
    provids = list(prov_prim)
    log(f"reconcile: {len(provids)} candidate provids, {len(prim_orbit)} families")

    # 2) family tracklet counts via per-provid SQL aggregates (each trkid has
    #    exactly one provid, so the family count is the sum over secondaries).
    fam_count = defaultdict(int)
    for bi, i in enumerate(range(0, len(provids), batch), 1):
        chunk = provids[i:i + batch]
        with conn.cursor() as cur:
            cur.execute(f"""SELECT {s.provid_col}, count(DISTINCT {s.tracklet_id_col})
                            FROM {s.obs_table}
                            WHERE {s.provid_col} = ANY(%s)
                              AND {s.asteroid_obs()}
                            GROUP BY {s.provid_col}""", (chunk,))
            for provid, n in cur.fetchall():
                fam_count[prov_prim[provid]] += n
        if bi % 25 == 0:
            log(f"reconcile: counted {min(i + batch, len(provids))}/{len(provids)} provids")
        if sleep_s:
            time.sleep(sleep_s)
    log(f"reconcile: {len(fam_count)} families with asteroid-style obs")

    # 3) qualifying families -> their trkids (>=2 obs so re-runs stay idempotent)
    qualifying_primaries = {prim for prim, cnt in fam_count.items()
                            if (not prim_orbit.get(prim)) or cnt <= index_cfg.max_tracklets}
    qualifying_provids = [p for p, prim in prov_prim.items() if prim in qualifying_primaries]
    log(f"reconcile: {len(qualifying_primaries)} qualifying families, "
        f"{len(qualifying_provids)} provids")
    qualifying_trkids = set()
    for i in range(0, len(qualifying_provids), batch):
        chunk = qualifying_provids[i:i + batch]
        with conn.cursor() as cur:
            cur.execute(f"""SELECT {s.tracklet_id_col} FROM {s.obs_table}
                            WHERE {s.provid_col} = ANY(%s)
                              AND {s.asteroid_obs()}
                            GROUP BY {s.tracklet_id_col} HAVING count(*) >= 2""", (chunk,))
            qualifying_trkids.update(r[0] for r in cur.fetchall())
        if sleep_s:
            time.sleep(sleep_s)
    log(f"reconcile: {len(qualifying_trkids)} qualifying pseudo-ITF tracklets")

    # 4) admit absent trkids; flip former-ITF rows that now qualify as pseudo
    qualifying_trkid_list = list(qualifying_trkids)
    with conn.cursor() as cur:
        cur.execute(f"SELECT tracklet_id, provid FROM {s.target_qualified} "
                    f"WHERE tracklet_id = ANY(%s)", (qualifying_trkid_list,))
        existing = dict(cur.fetchall())
    to_admit = [t for t in qualifying_trkid_list
                if t not in existing or existing[t] is None]
    # PURITY: only admit genuinely-designated tracklets. This skips mixed
    # cross-state trkids AND prevents wrongly converting a still-pure-ITF row to
    # pseudo (a designation qualifying does NOT make a *pure ITF* trkid pseudo).
    if to_admit:
        acls = _classify_trkids(conn, s, to_admit)
        to_admit = [t for t in to_admit if acls[t] == 'designated']
    log(f"reconcile: admitting {len(to_admit)} pseudo-ITF tracklets")
    admitted = 0
    for i in range(0, len(to_admit), 2000):
        rows_by = _load_tracklet_obs(conn, index_cfg, to_admit[i:i + 2000])
        feats = [f for f in (compute_tracklet_features(v, index_cfg)
                             for v in rows_by.values()) if f]
        _upsert(conn, index_cfg, feats)
        conn.commit()
        admitted += len(feats)
        if sleep_s:
            time.sleep(sleep_s)

    # 5) evict pseudo rows no longer qualifying (audited). ITF-population
    #    orphans are handled by sweep_validity (a full anti-join is too heavy).
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _qual (tracklet_id text PRIMARY KEY) "
                    "ON COMMIT DROP")
        psycopg2.extras.execute_values(
            cur, "INSERT INTO _qual VALUES %s ON CONFLICT DO NOTHING",
            [(t,) for t in qualifying_trkid_list], page_size=1000)
        evicted = _delete_with_audit(
            cur, s,
            "t.provid IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM _qual q WHERE q.tracklet_id = t.tracklet_id)",
            "reconcile: pseudo-ITF no longer qualifies (orbit+>3 trk, or obs gone)")
    conn.commit()
    log(f"reconcile: admitted {admitted}, evicted {evicted} pseudo")
    return admitted


def sweep_validity(conn, index_cfg: IndexConfig = None, batch=5000, sleep_s=0.2,
                   log=print):
    """Weekly chunked backstop for exits with no incremental signal: ITF rows
    no longer pure (attributed / status left 'I' / mixed) and rows whose trkid
    vanished from obs_sbn (hard delete).  Audited; never one big anti-join."""
    index_cfg = index_cfg or IndexConfig()
    s = index_cfg.schema
    with conn.cursor() as cur:
        cur.execute(f"SELECT tracklet_id, provid FROM {s.target_qualified} "
                    f"ORDER BY tracklet_id")
        rows = cur.fetchall()
    itf_removed = orphan_removed = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids = [r[0] for r in chunk]
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT {s.tracklet_id_col} FROM {s.obs_table} "
                        f"WHERE {s.tracklet_id_col} = ANY(%s)", (ids,))
            present = {r[0] for r in cur.fetchall()}
            orphans = [t for t in ids if t not in present]
            if orphans:
                orphan_removed += _delete_with_audit(
                    cur, s, "t.tracklet_id = ANY(%s)",
                    "sweep: trkid gone from obs_sbn (hard delete)", params=(orphans,))
            # Purity for BOTH populations (classify present trkids once):
            #  * itf rows (provid NULL) no longer pure_itf -> attributed/status/mixed;
            #  * pseudo rows (provid set) that are mixed / <2 obs -> impure.
            # (A pseudo row that is cleanly 'designated' is valid here -- whether it
            #  still QUALIFIES is reconcile's job, not the sweep's.)
            present_rows = [(t, pv) for (t, pv) in chunk if t in present]
            if present_rows:
                cls = _classify_trkids(conn, s, [t for t, _ in present_rows])
                itf_bad = [t for (t, pv) in present_rows
                           if pv is None and cls[t] != 'pure_itf']
                mixed_bad = [t for (t, pv) in present_rows
                             if pv is not None and cls[t] in ('mixed', 'degenerate')]
                if itf_bad:
                    itf_removed += _delete_with_audit(
                        cur, s, "t.tracklet_id = ANY(%s)",
                        "sweep: ITF no longer pure (attributed / status / mixed)",
                        params=(itf_bad,))
                if mixed_bad:
                    itf_removed += _delete_with_audit(
                        cur, s, "t.tracklet_id = ANY(%s)",
                        "sweep: pseudo not pure (mixed / <2 obs)", params=(mixed_bad,))
        conn.commit()
        if sleep_s:
            time.sleep(sleep_s)
    log(f"sweep_validity: removed {orphan_removed} orphans, {itf_removed} impure")
    return orphan_removed + itf_removed
