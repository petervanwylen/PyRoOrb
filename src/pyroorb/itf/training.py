"""ML training data + model for candidate-tracklet ranking.

The synthetic recovery harness is a labeled-data factory: run the search on
objects the MPC has ALREADY linked (seeded with 3-4 nights of one
opposition, the rest of the object planted as fake ITF), and every candidate
tracklet served into a corridor window becomes one training row --
**positive** when it is truly the object's (the planted, MPC-linked truth),
**negative** when it is a real ITF distractor that was served but does not
belong.  Features are frozen at DECISION TIME (the trial orbit of that
recursion level), never the final orbit -- the model must see what the
search saw.

Feature groups (wide on purpose -- selection is cheap later, regeneration is
not): corridor geometry (dist/qual + the corridor WIDTH so "close in a wide
corridor" ranks below "close in a narrow one"), uncertainty-normalized
chi2 terms, motion (rate + position-angle mismatch), tracklet quality
(n_obs, WLS rms/chi2/sigmas), photometry (mag residual vs the ephemeris V,
band), observing circumstances (obscode categorical, delta, elongation,
predicted V), and trial-orbit context (U, rms, used/count, oppositions,
arc span, the time gap being bridged, recursion depth).

    python -m pyroorb.itf.collect --objects-per-split 40 --out itf_train.csv   # live
    python -m pyroorb.itf train-model itf_train.csv               # xgboost
"""

import csv
import math
import os

import numpy as np

# columns that identify a row / carry the label -- never model inputs.
# ``split``/``split_key``/``stratum`` pin the permanent train-vs-eval wall and
# the sampling stratum onto every row (see :func:`split_of`), so a CSV can be
# partitioned without re-deriving anything and a leak is visible by inspection.
KEY_COLS = ["designation", "trkid", "trksub", "label", "label_source",
            "window_t", "split", "split_key", "stratum"]
CAT_COLS = ["stn", "band", "traj_category"]

FEATURE_COLS = [
    # corridor geometry + width
    "dist_arcsec", "qual_arcsec", "traj_unc_arcsec",
    "width_ra_arcsec", "width_dec_arcsec", "win_uncert_arcsec",
    "dist_over_width", "qual_over_width",
    # uncertainty-normalized match terms
    "pos_chi2", "motion_chi2", "match_chi2",
    # motion
    "pred_rate_degday", "trk_rate_degday", "rate_ratio", "pa_diff_deg",
    # tracklet quality
    "trk_n_obs", "trk_arc_min", "trk_fit_rms", "trk_chi2_dof",
    "trk_linear_ok", "trk_rate_sigma", "trk_sig_pos_arcsec",
    "trk_sig_vel_degday",
    # photometry
    "mag_resid", "mag_mean", "mag_rms", "n_mag", "win_vmag",
    # observing circumstances
    "delta_au", "elong_deg", "win_speed_asmin",
    # trial-orbit context (decision-time state)
    "trial_U", "trial_rms", "trial_used", "trial_count", "trial_opps",
    "trial_arc_days", "gap_days", "years_from_arc", "depth",
    # the two HAND-WRITTEN prioritizers this model replaced, as derived
    # features.  Both are closed-form combinations of columns already present,
    # so they add no information in the Shannon sense -- but an axis-aligned
    # tree cannot synthesize `a + b + c` or `max(a - b, 0)` from the parts
    # without spending depth on it, and handing over a ranking function that
    # already works is cheaper than making the model rediscover it.
    #   legacy_score    = search._collect_tracklets' "score" (ranking_mode
    #                     "legacy"): dist + qual + window sigma, all arcsec.
    #   cap_key_primary = the primary term of candidates.cap_rank_key, the
    #                     motion-mismatch margin the per-window cap ranks on.
    "legacy_score", "cap_key_primary",
    # which corridor served this row: 0 = the ordinary trial-orbit corridor,
    # 1 = an LOV variant corridor (fixed cfg.variant_width_deg half-width, only
    # at trial_U >= cfg.variant_corridor_U).  The width features mean something
    # different in each regime, so the model gets told which one it is looking
    # at rather than having to infer it from the widths.
    "is_variant",
    # How crowded the corridor this candidate came from actually was, pre-cap.
    # Promoted from telemetry to a FEATURE because base rate is not one number:
    # measured over the v3 collection the 'likely' stratum runs at a 32%
    # positive rate overall but ~0.02% inside 2022 GL24's 17,082-candidate
    # corridor -- a spread of ~1000x.  A model with no view of corridor density
    # is forced into a single global calibration and is wildly over-confident
    # exactly where the search is hardest.  Known at serve time (candidates.
    # _survivors computes it before any scoring), so it is fair game.
    "n_in_corridor",
] + CAT_COLS

# Recorded with every collected row but NEVER model inputs.
#
# ``motion_bound_ok`` -- the obs-fetch motion pre-filter's verdict
# (candidates.motion_bound_ok) on this candidate.  Collection sweeps with the
# filter BYPASSED so the negative class covers the full corridor population;
# this column lets a later retrain choose either population -- the full one, or
# the filtered one the search actually scores -- without a new live collection
# run.  (As a model input it would be useless anyway: at serve time the filter
# makes it constant-True.)
#
# ``n_in_corridor`` / ``cap_rank`` / ``would_be_capped`` -- what
# cfg.max_candidates_per_window did, or WOULD have done, to this candidate.
# The cap is applied in candidates._survivors on the real DB population, but a
# harness-planted truth is injected past it (OverlaySource._merge_fakes), so a
# collected POSITIVE can sit in a window where a real tracklet of the same
# geometry would have been capped out.  Recording the rank rather than
# enforcing it keeps the search's behavior untouched while making that recall
# risk countable -- the same trick ``motion_bound_ok`` plays for the prefilter.
#
# ``truncated`` -- the object's search hit the collection deadline, so this row
# comes from a search that never finished: its ``planted``/``distractor``
# labels stand (they do not depend on the outcome) but ``linked_foreign``
# could not be determined, and deeper recursion levels are missing.
META_COLS = ["motion_bound_ok", "cap_rank", "would_be_capped", "truncated"]

ALL_COLS = KEY_COLS + FEATURE_COLS + META_COLS

# The DECISION context a serve-faithful collection (itf/decisions.py) adds to
# every row: which round and variant scan the candidate competed in, the
# deployed pair's own scores for it (computed in-process at serve time, so the
# baseline is what the search actually saw), and the continuous LOV offset geometry.
# Rows collected this way carry HONEST-scale features (no _pre_floor_scale):
# ``floor_factor`` records the fit's s2 floor so the old scale stays derivable.
#
#   round_id / scan_id      -- decisions.DecisionLog ids (scan -1 = none)
#   in_variant_corridor     -- the corridor IS a LOV variant's thin tube.  NOT
#                              ``is_variant``, which stays set on the ordinary
#                              covariance corridors of a variant scan's deeper
#                              recursion levels
#   variant_k               -- that variant's LOV offset (sigma), else NaN
#   ml_p / ml_p2            -- the deployed primary / second-opinion P
#   lov_kstar / lov_resid   -- the solved LOV offset and the residual off the
#                              LOV curve there (arcsec; NaN = beyond
#                              cfg.lov_refine_resid_max, or not a variant round)
#   lov_nights              -- other nights whose k* agrees within
#                              cfg.lov_refine_cluster_dsigma
#   lov_dk                  -- lov_kstar - variant_k
#   n_round                 -- candidates the round scored
DECISION_COLS = ["round_id", "scan_id", "in_variant_corridor", "variant_k",
                 "ml_p", "ml_p2", "lov_kstar", "lov_resid", "lov_nights",
                 "lov_dk", "floor_factor", "trk_ids"]
V4_ALL_COLS = ALL_COLS + DECISION_COLS

# The v2 feature set, frozen.  v3 may ADD features but must never drop or
# redefine one of these, or the deployed v2 model cannot be scored on v3 eval
# rows and the head-to-head comparison becomes impossible.
V2_FEATURE_COLS = [
    "dist_arcsec", "qual_arcsec", "traj_unc_arcsec",
    "width_ra_arcsec", "width_dec_arcsec", "win_uncert_arcsec",
    "dist_over_width", "qual_over_width",
    "pos_chi2", "motion_chi2", "match_chi2",
    "pred_rate_degday", "trk_rate_degday", "rate_ratio", "pa_diff_deg",
    "trk_n_obs", "trk_arc_min", "trk_fit_rms", "trk_chi2_dof",
    "trk_linear_ok", "trk_rate_sigma", "trk_sig_pos_arcsec",
    "trk_sig_vel_degday",
    "mag_resid", "mag_mean", "mag_rms", "n_mag", "win_vmag",
    "delta_au", "elong_deg", "win_speed_asmin",
    "trial_U", "trial_rms", "trial_used", "trial_count", "trial_opps",
    "trial_arc_days", "gap_days", "years_from_arc", "depth",
    "stn", "band",
]
assert not set(V2_FEATURE_COLS) - set(FEATURE_COLS), \
    "v3 dropped a v2 feature -- the v2/v3 comparison needs all of them"


# ---------------------------------------------------------------------------
# The permanent train / eval wall
# ---------------------------------------------------------------------------
EVAL_KEYS = (8, 9)          # split keys reserved for evaluation, forever


def split_key(designation):
    """A permanent 0-9 key for one object, derived from its designation.

    Deliberately a HASH rather than the designation's own trailing digit.  For
    MBAs and TNOs the two are equivalent (their provisional cycle counts are
    large -- median 96 and 156 -- so the last digit has wrapped many times and
    is uniform noise).  For NEOs it is not: NEO cycle counts are small (median
    5), because a fast mover is followed up and designated early in its
    half-month rather than accumulating in a queue.  Measured over 6000 sampled
    NEOs, splitting on the trailing digit gives an eval side with median cycle
    count 18 against the training side's 5 -- i.e. it sorts NEOs by how late in
    a crowded half-month they were designated, a proxy for follow-up urgency
    and discovery survey.  Oversampling fixes the resulting count shortfall
    (12.6% vs 20%) but not the skew, because the skew is in the rule.  A hash
    is uniform across every population (measured 20.3 / 19.8 / 19.7%), is
    defined for the ~15% of NEO designations that end in a letter rather than a
    digit, and carries no astronomical meaning at all."""
    import hashlib
    d = str(designation).strip()
    return int(hashlib.sha1(d.encode()).hexdigest(), 16) % 10


def split_of(designation):
    """``"eval"`` or ``"train"`` for one object -- permanent, object-level."""
    return "eval" if split_key(designation) in EVAL_KEYS else "train"


def _pa_diff(a, b):
    """Absolute position-angle difference in degrees, in [0, 180]."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def feature_uncert_mult(ctx):
    """The corridor multiple the RANKER's width features are built with.

    The width features (width_ra/dec, dist/qual over width) are defined at the
    multiple the training collection swept with (``cfg.ml_feature_uncert_mult``),
    not at whatever retrieval multiple the search uses today -- otherwise a
    change to ``cfg.uncert_mult`` (a recall/cost lever on the CORRIDOR) would
    silently move every model input.  Any per-search relaxation of the
    multiple (``ctx.uncert_mult`` differing from the config's) carries over in
    proportion, so with the two constants equal this is exactly
    ``ctx.uncert_mult``."""
    cfg = ctx.cfg
    base = float(cfg.uncert_mult) or 1.0
    want = float(getattr(cfg, "ml_feature_uncert_mult", 0.0) or base)
    return float(ctx.uncert_mult) * want / base


def _pre_floor_scale(window, eph_trial):
    """``(window, trial_U)`` on the covariance scale the rankers were TRAINED on.

    Every deployed ranker (v2, v3e) learned from rows whose covariance-derived
    features -- the window's sky sigma and everything built on it (corridor
    widths, dist/qual over width, the three match chi2s, the legacy score) and
    the trial orbit's U -- came from a covariance scaled by the fit's own
    chi2/dof with no floor.  Since the s2 floor (``FittedOrbit._param_inverse``)
    that scale is gone from the ENGINE, rightly, but a model scored on the new
    scale is being asked about a population it never saw: live on 2021 SA70
    the true tracklet at the same variant, same corridor, dropped from P 0.09
    (#67) to 0.009 (#134+) and the search never reached it.

    So the FEATURE definition keeps the pre-floor scale, exactly: the sky sigma
    part of the window's half-width (everything but the topocentric parallax
    allowance :func:`itf.ephem.corridor_windows` adds) is divided by the fit's
    own floor factor ``f = 1/sqrt(s2_raw)``, and U drops by ``ln(f)/1.49`` (U
    is ``ln(runoff)/1.49 + 1`` with runoff proportional to sigma; checked to
    0.01 over 24 paired seeds).  This is the feature space -- collection and
    serving both pass through it -- so old training rows, new ones and live
    scoring agree without a retrain.  The CORRIDOR (retrieval, gates) keeps
    the honest window; only the model's view of it is rescaled.  A fit with
    s2 >= 1 (f = 1) is untouched, and a variant window (parallax only) keeps
    its width and moves only in U."""
    import dataclasses
    from .obs80_fields import EARTH_RAD_AU

    U = eph_trial.U
    try:
        f = float(eph_trial.fit.cov_floor_factor())
    except Exception:              # a test double, or a fit without a Jacobian
        f = 1.0
    if not (f > 1.0) or not math.isfinite(f):
        return window, U
    U_old = U - math.log(f) / 1.49 if U is not None else U
    try:
        par = EARTH_RAD_AU / window.dist_au * 57.3 if window.dist_au else 0.0
        par = min(max(par, 0.0), window.uncert)
        u_old = par + (window.uncert - par) / f
        if u_old == window.uncert or not window.uncert:
            return window, U_old
        window = dataclasses.replace(window, uncert=u_old,
                                  uncert_ra=window.uncert_ra * u_old / window.uncert)
    except Exception:
        pass
    return window, U_old


def candidate_feature_row(tr, eph_trial, depth, uncert_mult, extrawidth,
                          is_variant=0, pre_floor=True):
    """One training/scoring row for a scored candidate tracklet ``tr`` (from
    ``_collect_tracklets``), in the context of the trial orbit that generated
    the corridor.  Labels are attached later by the harness.

    ``pre_floor`` -- build the covariance-derived features on the scale the
    deployed v2/v3e rankers were trained on (:func:`_pre_floor_scale`).  False
    gives the honest scale the ENGINE uses, which is what a model trained on
    decision-log rows (DECISION_COLS) sees."""
    from .candidates import window_tolerances, match_chi2

    if pre_floor:
        window, trial_U = _pre_floor_scale(tr["_win"], eph_trial)
    else:
        window, trial_U = tr["_win"], eph_trial.U
    feat = tr.get("_feat") or {}
    rows = tr["rows"]
    ra_tol, dec_tol = window_tolerances(window, uncert_mult, extrawidth)
    width_ra = ra_tol * 3600.0
    width_dec = dec_tol * 3600.0
    nan = float("nan")

    pos_chi2 = motion_chi2 = total_chi2 = nan
    pred_rate = trk_rate = rate_ratio = pa_diff = nan
    if feat.get("_pred_ra") is not None:
        total_chi2, pos_chi2, motion_chi2 = match_chi2(feat, window)
        dt = feat["_dt"]
        cosd = math.cos(math.radians(feat["center_dec"]))
        ve = (2 * window.ra_quad[0] * dt + window.ra_quad[1]) * cosd * 86400.0
        vn = (2 * window.dec_quad[0] * dt + window.dec_quad[1]) * 86400.0
        pred_rate = float(np.hypot(ve, vn))
        pred_pa = float(np.degrees(np.arctan2(ve, vn)) % 360.0)
        trk_rate = feat.get("rate_deg_day") or nan
        if trk_rate == trk_rate:
            rate_ratio = trk_rate / pred_rate if pred_rate > 0 else nan
        if feat.get("pa_deg") is not None:
            pa_diff = _pa_diff(feat["pa_deg"], pred_pa)

    mag_mean = feat.get("mag_mean")
    win_vmag = window.mag
    mag_resid = (mag_mean - win_vmag
                 if mag_mean is not None and win_vmag is not None else nan)
    es_mid = feat.get("epoch_mid_sec",
                      0.5 * (rows[0]["epoch_sec"] + rows[-1]["epoch_sec"]))
    jd_mid = es_mid / 86400.0 + 2440587.5
    lo, hi = eph_trial.earliest_jd, eph_trial.latest_jd
    gap_days = max(0.0, lo - jd_mid, jd_mid - hi)
    years_from_arc = (jd_mid - 0.5 * (lo + hi)) / 365.25

    try:
        trial_opps = eph_trial.fit.n_oppositions()
    except Exception:
        trial_opps = nan

    def _f(key):
        v = feat.get(key)
        return nan if v is None else float(v)

    sig_pos = float(np.hypot(_f("sig_pos_e_arcsec"), _f("sig_pos_n_arcsec")))
    sig_vel = float(np.hypot(_f("sig_vel_e_degday"), _f("sig_vel_n_degday")))
    linear_ok = feat.get("linear_ok")

    return {
        "designation": "",                       # filled by the harness
        "trkid": (feat.get("tracklet_id") or rows[0].get("trkid")
                  or rows[0]["trksub"]),
        "trksub": rows[0]["trksub"],
        "label": -1,                             # filled by the harness
        "label_source": "",
        "window_t": window.unix_cur,
        "split": "", "split_key": -1, "stratum": "",   # filled by the harness
        "dist_arcsec": tr["dist"],
        "qual_arcsec": tr["qual"],
        "traj_unc_arcsec": tr["traj_uncert"],
        "width_ra_arcsec": width_ra,
        "width_dec_arcsec": width_dec,
        "win_uncert_arcsec": window.uncert * 3600.0,
        "dist_over_width": tr["dist"] / width_dec if width_dec > 0 else nan,
        "qual_over_width": tr["qual"] / width_dec if width_dec > 0 else nan,
        "pos_chi2": pos_chi2,
        "motion_chi2": motion_chi2,
        "match_chi2": total_chi2,
        "pred_rate_degday": pred_rate,
        "trk_rate_degday": trk_rate,
        "rate_ratio": rate_ratio,
        "pa_diff_deg": pa_diff,
        "trk_n_obs": len(rows),
        "trk_arc_min": (rows[-1]["epoch_sec"] - rows[0]["epoch_sec"]) / 60.0,
        "trk_fit_rms": _f("fit_rms_arcsec"),
        "trk_chi2_dof": _f("fit_chi2_dof"),
        "trk_linear_ok": (nan if linear_ok is None else float(bool(linear_ok))),
        "trk_rate_sigma": _f("rate_sigma"),
        "trk_sig_pos_arcsec": sig_pos,
        "trk_sig_vel_degday": sig_vel,
        "mag_resid": mag_resid,
        "mag_mean": nan if mag_mean is None else float(mag_mean),
        "mag_rms": _f("mag_rms"),
        "n_mag": _f("n_mag"),
        "win_vmag": nan if win_vmag is None else float(win_vmag),
        "delta_au": window.dist_au,
        "elong_deg": nan if window.elong is None else float(window.elong),
        "win_speed_asmin": window.max_speed,
        "trial_U": trial_U,
        "trial_rms": eph_trial.rms,
        "trial_used": eph_trial.used,
        "trial_count": eph_trial.count,
        "trial_opps": trial_opps,
        "trial_arc_days": eph_trial.arclength_days,
        "gap_days": gap_days,
        "years_from_arc": years_from_arc,
        "depth": depth,
        # the two hand-written prioritizers, recomputed exactly as their
        # owners define them (search._collect_tracklets' "score";
        # candidates.cap_rank_key's primary term)
        "legacy_score": tr["dist"] + tr["qual"] + window.uncert * 3600.0,
        "cap_key_primary": max(tr["qual"] - tr["traj_uncert"], 0.0),
        "is_variant": float(is_variant),
        "traj_category": tr.get("traj_category") or "",
        "stn": rows[0]["stn"],
        "band": feat.get("band") or "",
        "motion_bound_ok": (nan if feat.get("_motion_bound_ok") is None
                            else float(feat["_motion_bound_ok"])),
        "n_in_corridor": _f("_n_in_corridor"),
        "cap_rank": _f("_cap_rank"),
        "would_be_capped": (nan if feat.get("_would_be_capped") is None
                            else float(feat["_would_be_capped"])),
        "truncated": 0,                          # set by the harness on timeout
    }


def decision_context(ctx, tr, eph_trial, in_variant):
    """The DECISION_COLS for one candidate, read off the search context."""
    nan = float("nan")
    dlog = ctx.dlog
    vk = dlog.variant_lov_sigma if (in_variant and dlog.variant_lov_sigma is not None) else nan
    hit = (ctx.lov_feat or {}).get(tr.get("_ids")) if in_variant else None
    try:
        f = float(eph_trial.fit.cov_floor_factor())
    except Exception:
        f = nan
    lov_sigmas, rr, nn = hit if hit is not None else (nan, nan, nan)
    ids = tr.get("_ids")
    return {
        "round_id": dlog.cur_round, "scan_id": dlog.cur_scan,
        "in_variant_corridor": float(bool(in_variant)), "variant_k": vk,
        "ml_p": tr.get("ml_p", nan), "ml_p2": tr.get("ml_p2", nan),
        "lov_kstar": lov_sigmas, "lov_resid": rr,
        "lov_nights": nn if hit is not None else nan,
        "lov_dk": (lov_sigmas - vk) if hit is not None and vk == vk else nan,
        "floor_factor": f,
        # the candidate's obs identities, so a label can be attached by
        # OBSERVATION rather than by trksub (two tracklets can share one)
        "trk_ids": "|".join(sorted(ids)) if ids else "",
    }


def stratified_subsample(rows, n_maybe=10, n_unlikely=10, seed=0):
    """Thin one object's rows for storage: keep everything that matters.

    Keeps EVERY positive and EVERY 'likely' candidate, then samples only
    n_maybe / n_unlikely of the rest.  The rationale is what the search
    actually consumes: only 'likely' candidates (plus a single ML-promoted
    'maybe') ever reach a trial, so the likely stratum is the population the
    model's ranking decides over and must be complete.  'maybe' and 'unlikely'
    are kept in token numbers so the model still learns to push them down,
    without their bulk swamping the set -- measured over the v3 collection they
    are 89% of all rows and 6% of all positives.

    Categories come from the row's own ``traj_category``, which is the search's
    verdict copied verbatim (search._collect_tracklets), not a re-derivation
    from qual/traj_unc thresholds here -- a second copy of that rule would
    drift from the engine the first time cfg.likely_arcsec changed.
    """
    import random as _random
    rng = _random.Random(seed)
    keep, maybes, unlikelies = [], [], []
    for r in rows:
        if r.get("label") == 1 or r.get("traj_category") == "likely":
            keep.append(r)
        elif r.get("traj_category") == "maybe":
            maybes.append(r)
        else:
            unlikelies.append(r)
    for pool, n in ((maybes, n_maybe), (unlikelies, n_unlikely)):
        keep.extend(pool if len(pool) <= n else rng.sample(pool, n))
    return keep


def hard_negative_subsample(rows, top_n=10, n_easy=25, model_path=None, seed=0):
    """Positives + the TOP-``top_n`` negatives per window under the deployed
    ranker + ``n_easy`` random others per object.

    The alternative to :func:`stratified_subsample`, for collections whose job is
    teaching good-vs-GREAT: the search only ever tries a window's top few
    candidates, so the discrimination that decides a live recovery lives
    entirely inside that band -- and a store dominated by mediocre 'likely'
    rows spends the training loss pushing down junk instead (measured:
    2015 BE559's planted truth ranked #74 of 518 by a model trained on the
    bulk store, #1 by v2).  Ranking the negatives with the DEPLOYED model at
    collection time keeps exactly the ones a serving search would actually
    try first.  The random easy sample keeps the junk-rejection boundary
    taught; positives are always kept."""
    import random as _random

    import pandas as pd

    from .traineval import score_with

    if not rows:
        return rows
    rng = _random.Random(seed)
    df = pd.DataFrame(rows)
    p = score_with(df, model_path or default_model_path())
    keep_idx = set(df.index[df["label"] == 1])
    neg = df[df["label"] != 1].assign(_p=p[(df["label"] != 1).values])
    for _w, g in neg.groupby("window_t", sort=False):
        keep_idx.update(g.nlargest(top_n, "_p").index)
    rest = [i for i in neg.index if i not in keep_idx]
    keep_idx.update(rng.sample(rest, min(n_easy, len(rest))))
    return [rows[i] for i in sorted(keep_idx)]


def write_rows(path, rows, batch=500, fields=None):
    """Append rows to a CSV (header written on first creation).

    Flush + fsync every ``batch`` rows, not once at the end: this box's
    filesystem tears large buffered writes (measured on the 2026-08-05 v3b
    collection -- objects that appended >5k rows in one flush lost 9.6% of
    them in dropped spans plus a handful of merged records, while every
    small-append object survived byte-perfect).  Small synced batches keep
    each physical write under the tearing threshold."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ALL_COLS,
                           extrasaction="ignore")
        if new:
            w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(r)
            if i % batch == 0:
                fh.flush()
                os.fsync(fh.fileno())
        fh.flush()
        os.fsync(fh.fileno())      # a killed run must not leave torn blocks


# ---------------------------------------------------------------------------
# Model training (xgboost native API -- no sklearn dependency)
# ---------------------------------------------------------------------------
def fill_categorical(s):
    """``s`` as a category column with missing values as "".

    read_dataset now returns these columns ALREADY categorical (that is most
    of the memory saving), and ``fillna("")`` on a Categorical whose categories
    lack "" raises rather than widening -- so add the category first.  Going
    via astype(str) would work too and would materialise a Python string per
    row, which is the cost the categories exist to avoid."""
    import pandas as pd
    if isinstance(s.dtype, pd.CategoricalDtype):
        if "" not in s.cat.categories:
            s = s.cat.add_categories([""])
        return s.fillna("")
    return s.fillna("").astype("category")


def _object_weights(df, log=print):
    """Row weights that give every OBJECT equal total say in the loss.

    Positives are wildly concentrated: measured on the first 19 objects of the
    v3 run, the top three held 95% of all positives and ONE object held 88%
    (2000 PL8, 3267 of 3702).  That is intrinsic -- a numbered NEO with many
    apparitions and deep recursion emits orders of magnitude more rows than a
    2-apparition main-belt object -- so the row count badly overstates the
    effective sample size, and an unweighted fit is largely a fit to whichever
    handful of objects happened to be prolific.

    Each object's rows are scaled to sum to the same total, which leaves the
    pos/neg ratio WITHIN an object untouched (scale_pos_weight still handles
    class imbalance) while stopping one object from owning the loss.  The
    grouped ranking metrics in :mod:`itf.traineval` are per-group means and were
    never vulnerable to this; the training loss is.
    """
    import numpy as np
    n_by = df.groupby("designation", observed=True)["label"].transform("size")
    w = (len(df) / df["designation"].nunique()) / n_by
    top = (df.assign(_w=w).groupby("designation", observed=True)["_w"]
           .sum().sort_values())
    log(f"object weights: {df['designation'].nunique()} objects, "
        f"row counts {int(n_by.min())}-{int(n_by.max())}, "
        f"total weight per object {top.iloc[0]:.0f}")
    return np.asarray(w, dtype="float32")


def read_dataset(csv_path, neg_per_window=None, seed=0, log=print,
                 split=None, usecols=None):
    """Load a traindata CSV without running the box out of memory.

    Two levers, in order of how much they cost you:

    ``float32`` for every numeric column -- free.  pandas defaults to float64,
    which doubles a 3M-row dataset for precision no feature here has.

    ``neg_per_window`` -- keep every positive and at most this many negatives
    per (designation, window), chosen at RANDOM.  Wide corridors are why this
    exists: one pilot object produced 62,702 rows for 7 positives, and holding
    that population in memory alongside an xgboost DMatrix does not fit in 4 GB.
    Random-within-window is a SAMPLE of what the search saw, not a filter on it
    -- it thins each window's negatives uniformly, so within-window ranking (the
    metric that matters) is preserved in expectation and the feature
    distribution is unchanged.  That is the crucial difference from dropping
    negatives by score, which would silently teach the model that the easy
    negatives it will meet at serve time do not exist.  The kept fraction is
    reported so the base rate can be reasoned about.
    """
    import numpy as np
    import pandas as pd

    # String columns are what actually blows the budget: 2M rows x 9 object
    # columns costs more than every float put together, because each cell is a
    # separate Python str.  The low-cardinality ones (a few hundred stations,
    # ~25 bands, a few hundred designations) collapse to almost nothing as
    # categories; trkid/trksub cannot -- they are unique per row -- so callers
    # that do not need them should not ask for them (``usecols``).
    # Every non-numeric column must be listed here or the reader tries to
    # parse it as float32 and dies mid-chunk.  Derived from CAT_COLS so a new
    # categorical FEATURE cannot be added without the reader learning about it.
    cat_str = tuple({"designation", "label_source", "split", "stratum",
                     *CAT_COLS})
    free_str = ("trkid", "trksub")
    head = pd.read_csv(csv_path, nrows=1)
    # `split` must survive column selection or the filter below silently
    # matches nothing and returns the WHOLE dataset -- a wrong answer that
    # looks like a right one (measured: asking for 208 train objects and
    # getting all 277).
    # Column selection must pull in whatever the requested OPERATIONS read,
    # or they fail confusingly (thinning KeyErrors on window_t) or silently
    # (a split filter over a frame with no split column matches nothing and
    # hands back the whole dataset -- measured: asking for 208 train objects
    # and getting all 277).
    need = None
    if usecols is not None:
        need = set(usecols) | {"label"}
        if split:
            need.add("split")
        if neg_per_window:
            need |= {"designation", "window_t"}
    cols = list(head.columns) if need is None else [
        c for c in head.columns if c in need]
    missing = (need - set(cols)) if need else set()
    if missing:
        raise ValueError(f"columns required for this read are absent from "
                         f"{csv_path}: {sorted(missing)}")
    dtypes = {}
    for c in cols:
        dtypes[c] = ("category" if c in cat_str
                     else "str" if c in free_str else "float32")
    dtypes["label"] = "int8"

    def _thin(chunk, rng):
        chunk = chunk[chunk["label"].isin([0, 1])]
        if split and "split" in chunk.columns:
            chunk = chunk[chunk["split"] == split]
        if not neg_per_window or chunk.empty:
            return chunk
        chunk = chunk.assign(_r=rng.random(len(chunk)))
        neg = chunk[chunk["label"] == 0]
        # observed=True is load-bearing, not tidiness: `designation` is a
        # CATEGORICAL now, and pandas groupby over categoricals defaults to
        # materialising every category x key combination whether or not it
        # occurs.  With hundreds of designations against hundreds of thousands
        # of window_t values that is a cartesian product, and it OOMs the box
        # (exit 137, no traceback) on a dataset the thinning is supposed to
        # make small.
        keep = (neg.groupby(["designation", "window_t"], sort=False,
                            observed=True)["_r"]
                .rank(method="first") <= neg_per_window)
        return pd.concat([chunk[chunk["label"] == 1],
                          neg[keep]]).drop(columns="_r")

    # ALWAYS chunk, even with nothing to thin or filter.  There was a fast path
    # here that read the file whole when neither was asked for -- which is
    # precisely how the audit loads a dataset, so the audit was OOM-killed on
    # the finished 1.5 GB CSV: exit 137, no output, no traceback.  A shortcut
    # whose only distinguishing feature is using more memory is not a shortcut.
    #
    # Thin WHILE reading.  Measured at 1.5x the CSV size in memory, so a 1.5 GB
    # dataset peaks at ~2.2 GB -- more than this box has spare once xgboost
    # builds its DMatrix.  Reading in chunks bounds the peak at one chunk plus
    # the (much smaller) thinned result.
    #
    # A window's rows are contiguous in the file, so a chunk boundary can split
    # at most one window per chunk and let it keep up to neg_per_window extra
    # negatives.  That is a rounding error against a 200k-row chunk, and it
    # cannot bias WHICH negatives survive -- still uniform within each part.
    # Skip TORN records.  This filesystem tears large appends, so a raw
    # collection CSV can hold merged/split records (24 tears on the v3b run).
    # on_bad_lines only catches rows with EXTRA fields; a tear that drops a
    # span NaN-pads short rows and can glue two values into one field
    # ('22555.012023 RZ77') that lands in a float column mid-chunk.  So scan
    # once with the stdlib csv reader -- which tolerates any field size and
    # counts fields exactly -- and hand pandas the precise data-row indices to
    # skip: wrong field count, or a non-label where the label belongs.
    # (Full numeric validation happens implicitly: any surviving glued field
    # would still raise in the typed parse, which is a wanted loud failure --
    # measured, none survive the field-count + label check.)
    bad_rows = set()
    lim = csv.field_size_limit()
    csv.field_size_limit(1 << 27)
    try:
        with open(csv_path, newline="") as fh:
            _r = csv.reader(fh)
            hdr = next(_r)
            _lab = hdr.index("label")
            for _i, _row in enumerate(_r):
                # NUL bytes are a zero-filled dropped extent: the row may
                # have a clean field count and label and still be garbage
                if (len(_row) != len(hdr)
                        or _row[_lab] not in ("0", "1", "-1")
                        or any("\x00" in f for f in _row)):
                    bad_rows.add(_i + 1)          # +1: pandas counts the header
    finally:
        csv.field_size_limit(lim)
    if bad_rows:
        log(f"skipping {len(bad_rows)} torn records in {csv_path}")

    rng = np.random.default_rng(seed)
    parts, n0 = [], 0
    for chunk in pd.read_csv(csv_path, dtype=dtypes, usecols=cols,
                             chunksize=200_000,
                             skiprows=(lambda i: i in bad_rows)
                                      if bad_rows else None):
        n0 += len(chunk)
        kept = _thin(chunk, rng)
        if len(kept):
            parts.append(kept)
    df = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=cols))
    # categories differ per chunk, so concat widens them back to object --
    # put them back, or the saving is lost exactly when the data is largest
    for c in cat_str:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if neg_per_window:
        log(f"negatives thinned to <={neg_per_window}/window: "
            f"{n0} -> {len(df)} rows ({len(df) / max(1, n0) * 100:.1f}% kept)")
    return df



def train_model(csv_path, model_out=None, test_frac=0.2, seed=0,
                num_rounds=400, log=print, balance=True, features=None,
                split="train", neg_per_window=None,
                object_weight=False, pos_weight=None):
    """Train the tracklet-match classifier and report holdout metrics.

    The train/valid split is BY OBJECT (designation), never by row -- rows of
    one object are correlated, and a row-level split would leak.  Categorical
    columns (obscode, band) use xgboost's native categorical support.
    Returns (booster, metrics dict).

    ``split`` restricts training to one side of the permanent eval wall
    (:func:`split_of`); the default keeps the eval objects entirely unseen.

    ``balance`` sets ``scale_pos_weight`` to the negative/positive ratio.  It
    is the right choice for RANKING and the wrong one for CALIBRATION: with
    ~40x more negatives than positives it inflates every predicted probability
    by roughly that factor.  That did not matter when ``ml_p`` was only a sort
    key, but the search now compares it against four absolute thresholds
    (cfg.confirm_min_ml_p, show_min_ml_p_abs, show_likely_min_ml_p,
    show_multi_min_ml_p), so an unbalanced model and a balanced one are not
    interchangeable behind those gates.  Train both and compare them at the
    deployed thresholds (:mod:`itf.traineval`) rather than assuming.
    """
    import xgboost as xgb

    df = read_dataset(csv_path, neg_per_window=neg_per_window, seed=seed,
                      log=log, split=split)
    log(f"split '{split}': {df['designation'].nunique()} objects "
        f"(the rest are held out behind the permanent wall)")
    for c in CAT_COLS:
        df[c] = fill_categorical(df[c])

    rng = np.random.default_rng(seed)
    objs = np.array(sorted(df["designation"].unique()))
    rng.shuffle(objs)
    n_test = max(1, int(len(objs) * test_frac))
    test_objs = set(objs[:n_test])
    te = df[df["designation"].isin(test_objs)]
    tr = df[~df["designation"].isin(test_objs)]
    log(f"dataset: {len(df)} rows / {len(objs)} objects "
        f"({int(df['label'].sum())} pos); "
        f"train {len(tr)} rows / {len(objs) - n_test} obj, "
        f"valid {len(te)} rows / {n_test} obj")

    feats = features or FEATURE_COLS
    wtr = _object_weights(tr, log) if object_weight else None
    dtr = xgb.DMatrix(tr[feats], label=tr["label"], weight=wtr,
                      enable_categorical=True)
    dte = xgb.DMatrix(te[feats], label=te["label"], enable_categorical=True)
    pos = max(1, int(tr["label"].sum()))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 5,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        # scale_pos_weight below 1 DOWN-weights positives, which is how a
        # training set collected at one base rate is made to behave like the
        # rate it will be served at.  Collection cannot control that rate: it
        # is 27.7% in the stratified store, 3.6% raw, and ~0.02% inside a
        # 17,000-candidate corridor.  pos_weight makes it a dial rather than
        # an accident.
        "scale_pos_weight": (pos_weight if pos_weight is not None
                             else ((len(tr) - pos) / pos) if balance else 1.0),
        "seed": seed,
    }
    evals_result = {}
    bst = xgb.train(params, dtr, num_boost_round=num_rounds,
                    evals=[(dtr, "train"), (dte, "valid")],
                    early_stopping_rounds=30, evals_result=evals_result,
                    verbose_eval=False)
    valid_auc = float(evals_result["valid"]["auc"][bst.best_iteration])
    p = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
    y = te["label"].values
    tp = int(((p >= 0.5) & (y == 1)).sum()); fp = int(((p >= 0.5) & (y == 0)).sum())
    fn = int(((p < 0.5) & (y == 1)).sum()); tn = int(((p < 0.5) & (y == 0)).sum())
    metrics = {
        "valid_auc": valid_auc,
        "best_iteration": int(bst.best_iteration),
        "confusion@0.5": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "recall@0.5": tp / max(1, tp + fn),
        "precision@0.5": tp / max(1, tp + fp),
    }
    log(f"valid AUC {valid_auc:.4f} (best iter {bst.best_iteration}); "
        f"@0.5: recall {metrics['recall@0.5']:.3f}, "
        f"precision {metrics['precision@0.5']:.3f} "
        f"(tp {tp} fp {fp} fn {fn} tn {tn})")
    gain = bst.get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:15]
    log("top features by gain:")
    for name, g in top:
        log(f"  {name:>22} {g:10.1f}")
    if model_out:
        bst.save_model(model_out)
        # xgboost native categoricals are positional codes -- persist the
        # category levels so prediction-time encoding matches training.
        # Provenance rides along so "which data produced the deployed model?"
        # is answerable from the artifact alone.
        import hashlib
        import json as _json
        from datetime import datetime, timezone
        # STREAM the checksum.  fh.read() pulled the whole file into memory to
        # hash it, which on a 1.5 GB dataset killed the process (exit 137)
        # AFTER training had finished and reported its metrics -- the most
        # expensive possible place to run out of memory.
        _h = hashlib.sha256()
        with open(csv_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                _h.update(block)
        dataset_sha = _h.hexdigest()
        meta = {"features": feats,
                "categories": {c: [str(v) for v in df[c].cat.categories]
                               for c in CAT_COLS},
                "provenance": {
                    "dataset": os.path.basename(str(csv_path)),
                    "dataset_sha256": dataset_sha,
                    "rows": len(df), "objects": len(objs),
                    "trained": datetime.now(timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "xgboost": xgb.__version__,
                    "seed": seed,
                    "balance": bool(balance),
                    "pos_weight": pos_weight,
                    "object_weight": bool(object_weight),
                    "neg_per_window": neg_per_window,
                    "train_split": split,
                    "metrics": metrics,
                }}
        with open(str(model_out) + ".meta.json", "w") as fh:
            _json.dump(meta, fh)
        log(f"model saved to {model_out} (+ .meta.json)")
    return bst, metrics


# ---------------------------------------------------------------------------
# Deployed ranking: score live candidates with the trained model
# ---------------------------------------------------------------------------
_RANKER_CACHE = {}


def default_model_path():
    """The DEPLOYED ranker: v4, ONE model (2026-09-27).  It retired the
    v3e-primary + v2-second-opinion pair.

    Trained on serve-faithful decision rows whose truths are REAL ITF
    tracklets the MPC later identified, with honest-scale
    features and the continuous LOV offset (LOV-curve) geometry, over two DAgger-style
    collections.  Live A/B vs the pair: 27/27
    fixtures each (v4 +4% time, all of it 2005 UD562 at ~2x); 40 weak-seed fan
    cases 67 vs 61 of 84 truths at -41% time; 35 natural/weak cases 66 = 66
    at -54%; genuinely wrong links 4 vs 1 (3 reviewable, one submittable:
    2022 DF9, a gap in the acceptance gates, not in the ranking).  At every
    deployed ml_p gate it has both higher recall and far higher precision than
    v3e, so the thresholds are unchanged.  The retired pair is not shipped; where
    its files are present it stays loadable for A/Bs (default_second_model_path)."""
    return os.path.join(os.path.dirname(__file__), "training_data",
                        "itf_match_v4.json")


def default_second_model_path():
    """The v2 artifact -- the second opinion of the RETIRED v3e+v2 pair (no
    longer a default: ``ItfConfig.ml_second_model_path`` is None since v4).  Ranker DIVERSITY
    is the validated recall mechanism on weak seeds (the BE559 truth scores
    0.0003 under v3e and 0.0042 under max(v3e, v2) -- no single ranking
    reaches it; the woven v2 trial source + backtracking does)."""
    return os.path.join(os.path.dirname(__file__), "training_data",
                        "itf_match_v2.json")


def load_ranker(path=None):
    """(booster, meta) for the tracklet-match model, cached per path."""
    import json as _json
    import xgboost as xgb
    path = path or default_model_path()
    if path not in _RANKER_CACHE:
        bst = xgb.Booster()
        bst.load_model(path)
        with open(path + ".meta.json") as fh:
            meta = _json.load(fh)
        _RANKER_CACHE[path] = (bst, meta)
    return _RANKER_CACHE[path]


def _predict_frame(df_raw, model_path):
    """One booster's predictions over a raw candidate-feature frame, scored
    through the MODEL's own feature list, not FEATURE_COLS: a v2 booster
    scored inside a v3 tree must see exactly the 41 columns it was trained
    on, and extra columns here are simply unused.  Unseen categorical levels
    become missing (xgboost handles them as such)."""
    import pandas as pd
    import xgboost as xgb
    bst, meta = load_ranker(model_path)
    df = df_raw.copy()
    for c, levels in meta["categories"].items():
        df[c] = pd.Categorical(df[c].fillna("").astype(str), categories=levels)
    d = xgb.DMatrix(df[meta["features"]], enable_categorical=True)
    return bst.predict(d)


def _candidate_frame(tracklets, eph_trial, depth, uncert_mult, extrawidth,
                     is_variant=0):
    import pandas as pd
    return pd.DataFrame(
        [candidate_feature_row(tr, eph_trial, depth, uncert_mult, extrawidth,
                               is_variant=is_variant)
         for tr in tracklets])


def score_candidates(tracklets, eph_trial, depth, uncert_mult, extrawidth,
                     model_path=None, is_variant=0):
    """P(true match) for each scored candidate tracklet dict (from
    ``_collect_tracklets``), via ONE trained model.  ``depth`` is a model
    feature — it must be the real recursion depth, exactly as the training
    rows carried it.  For the deployed ranker PAIR use :func:`score_pair`."""
    return _predict_frame(
        _candidate_frame(tracklets, eph_trial, depth, uncert_mult, extrawidth,
                         is_variant=is_variant), model_path)


def _is_v4(model_path):
    """True for a v4-family artifact.  None means the DEPLOYED default, which
    is v4 -- resolving it here is what makes an unset cfg.ml_model_path get
    the v4 frame and not the pre-floor one."""
    return bool(load_ranker(model_path or default_model_path())[1].get("v4"))


def _v4_frame(tracklets, eph_trial, depth, uncert_mult, extrawidth,
              is_variant, v4ctx):
    """The v4 feature frame (itf/trainv4.py): honest-scale rows plus the
    decision context.  ``v4ctx`` = ``(in_variant, variant_lov_sigmas, lov_feat)`` --
    whether the serving corridor is a LOV variant's, each candidate's serving
    variant k (one float, or a per-candidate list), and the continuous LOV offset
    lookup keyed by candidate id set."""
    import pandas as pd

    from .trainv4 import derive
    in_variant, vks, lov = v4ctx if v4ctx is not None else (False, None,
                                                            None)
    nan = float("nan")
    rows = []
    for i, tr in enumerate(tracklets):
        r = candidate_feature_row(tr, eph_trial, depth, uncert_mult,
                                  extrawidth, is_variant=is_variant,
                                  pre_floor=False)
        vk = (vks[i] if isinstance(vks, (list, tuple)) else vks)
        hit = (lov or {}).get(tr.get("_ids")) if in_variant else None
        lov_sigmas, rr, nn = hit if hit is not None else (nan, nan, nan)
        r.update({"in_variant_corridor": float(bool(in_variant)),
                  "variant_k": (vk if (in_variant and vk is not None)
                                else nan),
                  "lov_kstar": lov_sigmas, "lov_resid": rr, "lov_nights": nn,
                  "lov_dk": (lov_sigmas - vk) if (hit is not None and in_variant
                                          and vk is not None) else nan})
        rows.append(r)
    return derive(pd.DataFrame(rows))


def score_pair(tracklets, eph_trial, depth, uncert_mult, extrawidth, cfg,
               is_variant=0, v4ctx=None):
    """``(primary_probs, second_probs_or_None)`` under the deployed ranker
    pair -- ``cfg.ml_model_path`` + ``cfg.ml_second_model_path``.  THE place
    pair semantics live: every consumer of both opinions (the sweep's
    ml_p/ml_p2, the LOV profile pass) scores through here, so the expensive
    per-candidate feature rows are built ONCE and each booster predicts
    through its own meta -- and a measurement that wants a single model says
    so in config (``ml_second_model_path=None``, see tests/itf_model_ab.py)
    rather than by calling half of a pair."""
    frames = {}

    def _frame(path):
        # a v4 model is served its own frame (honest scale + decision
        # context); v2/v3 boosters share the pre-floor one.  Each is built at
        # most once per call.
        key = "v4" if _is_v4(path) else "v3"
        if key not in frames:
            frames[key] = (_v4_frame(tracklets, eph_trial, depth,
                                     uncert_mult, extrawidth, is_variant,
                                     v4ctx) if key == "v4" else
                           _candidate_frame(tracklets, eph_trial, depth,
                                            uncert_mult, extrawidth,
                                            is_variant=is_variant))
        return frames[key]
    probs = _predict_frame(_frame(cfg.ml_model_path), cfg.ml_model_path)
    probs2 = (_predict_frame(_frame(cfg.ml_second_model_path),
                             cfg.ml_second_model_path)
              if cfg.ml_second_model_path else None)
    return probs, probs2
