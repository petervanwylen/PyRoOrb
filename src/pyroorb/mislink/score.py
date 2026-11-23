"""Mislinkage score of a 3-5 night single-opposition object.

A gradient-boosted classifier (XGBoost, ``model.json``) separates a pure
linkage (every tracklet one body) from a chimera (tracklets of two or more
bodies linked into one designation).  It reads the object's published
``obs_sbn`` rows, groups them into tracklets (``tracklets.py``), computes 26
features (``features.py``: gap structure, photometry, one joint fit, a
leave-out of the minority side of the largest gap, and a night-by-night chain
from the densest pair of nights) and returns

* ``score`` -- the model's chimera score, 0-1.  ``THRESHOLD`` (0.93) is the
  operating point: at it no control arc in the external test set was flagged
  while about half of the known mislinkages were.  ``TIERS`` grades the whole
  range (very likely / likely / possible / small chance / no sign).  It is a ranking,
  not a calibrated probability: the training mix is not the real-world chimera rate;
* ``explanation`` -- exact TreeSHAP contributions in log-odds, one per
  feature, which add up with ``base`` to the score's log-odds.

Domain: trained on main-belt to Hilda arcs (heliocentric 1.6-5 au).  Objects
with q < 1.3 au or a > 5.5 au are scored but marked ``in_domain: False``;
their scores are not meaningful.  Arcs of 3-5 nights only (a night =
``orbit_common.night_index``), at most 200 d long, with at least 3 nights
holding a multi-detection tracklet.  Tracklets follow the 120-min rule of
training; an arc that rule leaves with fewer than 3 such nights (a ~2-h survey
cadence) is scored on station-night tracklets instead (``tracklet_rule``).
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.json"
CANDIDATES_PATH = HERE / "candidates.json"

#: Score at or above which an arc is a likely mislinkage.
THRESHOLD = 0.93

#: How a score reads, highest first: (lower bound, level, verdict). The 0.93 operating
#: point flags no control arc of the external test set and about half the known
#: mislinkages; 0.27 is the 99th percentile of those controls (30 of 43 known
#: mislinkages score above it), so a score past it is already unusual for a real arc.
#: Of the 2020-2026 single-opposition census, 98.8% score below 0.27 and 0.9% fall
#: between 0.27 and 0.93. Below 0.27 a score only ranks how ordinary an arc looks, so
#: "no sign" is kept for the near-zero scores and the rest of that range reads as remote.
TIERS = (
    (0.97, "very-likely", "very likely mislinkage"),
    (THRESHOLD, "likely", "likely mislinkage"),
    (0.70, "possible", "possible mislinkage"),
    (0.27, "small", "small chance of mislinkage"),
    (0.02, "remote", "very remote chance of mislinkage"),
    (0.0, "none", "no sign of a mislinkage"),
)


def verdict(score):
    """(level, verdict) of a score under ``TIERS``."""
    for lo, level, text in TIERS:
        if score >= lo:
            return level, text
    return TIERS[-1][1], TIERS[-1][2]

NIGHTS = (3, 5)

#: Longer than this is no longer one opposition.
MAX_ARC_DAYS = 200.0

#: Fewer published detections than this cannot carry a leave-out refit.
MIN_OBS = 6

#: The model's columns, in the booster's order.
FEATURES = (
    "chain_max_D", "chain_max_drms", "chain_max_rate_miss", "chain_max_rate_n",
    "fit_a", "fit_e", "fit_max_dH", "fit_max_ramp", "fit_n_rej", "fit_q",
    "fit_rej_frac", "fit_rms", "fit_rms_n", "fit_sig_i", "gap_n_nights",
    "iso_dH", "iso_gap", "iso_p", "lo_D", "lo_miss", "meta_v_max", "meta_v_min",
    "mot_max_trk_chi2", "mot_rate_mean", "phot_trk_maxdev", "phot_trk_sd",
)

#: feature -> (short label, unit, what it measures).  Shown beside every explanation.
DESCRIPTIONS = {
    "chain_max_D": ("Chain: worst position miss", "sigma",
                    "Growing the arc night by night from its best-observed pair of nights, the "
                    "largest miss of an incoming tracklet against the predicted position, in "
                    "units of the prediction ellipse (Mahalanobis distance)."),
    "chain_max_drms": ("Chain: worst rms jump", "",
                       "The largest rise in normalised rms when one night joins the growing arc."),
    "chain_max_rate_miss": ("Chain: worst motion miss", "\"/min",
                            "The largest difference between a tracklet's measured sky motion and "
                            "the motion the arc so far predicts for it."),
    "chain_max_rate_n": ("Chain: worst motion miss", "sigma",
                         "The same motion miss in units of its uncertainty (tracklet rate sigma "
                         "and prediction growth combined). A chance tracklet rarely moves right."),
    "fit_a": ("Semimajor axis", "au", "Of the joint fit to every night."),
    "fit_e": ("Eccentricity", "", "Of the joint fit. Forcing two bodies onto one orbit tends to "
                                  "need an eccentric one."),
    "fit_max_dH": ("Worst tracklet |dH|", "mag",
                   "The largest difference between one tracklet's absolute magnitude and the "
                   "object's: a stranger is often a different brightness."),
    "fit_max_ramp": ("Worst residual ramp", "sigma",
                     "The strongest steady drift of residuals within one tracklet (leave-one-out "
                     "swing): a tracklet that fits only on average."),
    "fit_n_rej": ("Rejected observations", "", "Observations the joint fit rejects."),
    "fit_q": ("Perihelion distance", "au", "Of the joint fit."),
    "fit_rej_frac": ("Rejected fraction", "", "Share of observations the joint fit rejects."),
    "fit_rms": ("rms", "\"", "Per-axis rms of the joint fit."),
    "fit_rms_n": ("Normalised rms", "", "rms of the joint fit's residuals over their sigmas."),
    "fit_sig_i": ("sigma(inclination)", "deg", "Formal uncertainty of the inclination."),
    "gap_n_nights": ("Nights", "", "Number of observed nights."),
    "iso_dH": ("Isolated tracklet |dH|", "mag",
               "Photometric discrepancy of the most isolated tracklet (the one farthest from "
               "its nearest neighbour in time)."),
    "iso_gap": ("Isolated tracklet gap", "d",
                "Days from the most isolated tracklet to its nearest neighbour: how far that "
                "tracklet had to be reached."),
    "iso_p": ("Isolated tracklet p(mag)", "",
              "Probability of the isolated tracklet's magnitude under the object's H; small = "
              "unexpected."),
    "lo_D": ("Leave-out miss", "sigma",
             "Leave out the smaller side of the largest gap, refit, and predict it: the miss in "
             "units of the prediction ellipse. Near 1 or below means the prediction was so "
             "loose that almost anything there would have fitted."),
    "lo_miss": ("Leave-out miss", "\"", "The same leave-out miss in arcseconds."),
    "meta_v_max": ("Faintest V", "mag", "Faintest band-corrected magnitude reported."),
    "meta_v_min": ("Brightest V", "mag", "Brightest band-corrected magnitude reported."),
    "mot_max_trk_chi2": ("Worst tracklet linearity", "chi2/dof",
                         "Worst straight-line motion fit within one tracklet."),
    "mot_rate_mean": ("Mean sky rate", "deg/d", "Average on-sky rate of the tracklets."),
    "phot_trk_maxdev": ("Tracklet mag spread (max)", "mag",
                        "Largest deviation of one tracklet's mean V from the median tracklet."),
    "phot_trk_sd": ("Tracklet mag spread (sd)", "mag",
                    "Standard deviation of the tracklets' mean V."),
}

_lock = threading.Lock()
_booster = None
_cache = {}
#: A cached score is reused this long; published observations change slowly.
CACHE_SEC = 6 * 3600.0


def _model():
    global _booster
    with _lock:
        if _booster is None:
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(str(MODEL_PATH))
            if b.num_features() != len(FEATURES):
                raise RuntimeError(f"{MODEL_PATH.name} has {b.num_features()} features, "
                                   f"expected {len(FEATURES)}")
            _booster = b
        return _booster


def _num(v):
    if v is None or isinstance(v, bool):
        return None if v is None else float(v)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def vector(features):
    """Feature dict -> the model's row (NaN for a missing feature)."""
    return np.array([[np.nan if _num(features.get(k)) is None else _num(features[k])
                      for k in FEATURES]], dtype=float)


def explain(features):
    """Score + exact TreeSHAP explanation of one feature dict.

    ``base + sum(contributions)`` is the score's log-odds.
    """
    import xgboost as xgb
    X = vector(features)
    b = _model()
    d = xgb.DMatrix(X)
    p = float(b.predict(d)[0])
    c = b.predict(d, pred_contribs=True)[0]
    contrib = [{"feature": k, "value": _num(features.get(k)), "shap": round(float(c[i]), 4)}
               for i, k in enumerate(FEATURES)]
    contrib.sort(key=lambda e: -abs(e["shap"]))
    return {"score": round(p, 4), "base": round(float(c[-1]), 4), "contributions": contrib}


def in_domain(features):
    a, q = _num(features.get("fit_a")), _num(features.get("fit_q"))
    return bool(a is not None and a <= 5.5 and (q is None or q >= 1.3))


def score_records(rows, desig, obscodes):
    """Score one object's obs_sbn rows (dicts keyed by ``tracklets.COLUMNS``).

    Returns ``{"status": "ok", score, likely, in_domain, explanation, ...}``, or a
    status naming why the object is outside what the model scores.
    """
    from pyroorb.orbit_common import night_index
    from . import features as feat
    from .tracklets import build_tracklets, clean_record, epoch_sec
    out = {"designation": desig, "threshold": THRESHOLD}
    recs = []
    for r in rows:
        if (r.get("deprecated") or "").strip() or r.get("sys") or r.get("ra") is None:
            continue
        r = clean_record(r)
        r["obj"], r["nopp"], r["core"] = desig, 1, True
        r["_es"] = epoch_sec(r["obstime"])
        recs.append(r)
    out["n_obs"] = len(recs)
    if len(recs) < MIN_OBS:
        return {**out, "status": "too-few-obs"}
    n_nights = max(night_index([r["_es"] / 86400 for r in recs])) + 1
    out["nights"] = n_nights
    if not NIGHTS[0] <= n_nights <= NIGHTS[1]:
        return {**out, "status": "out-of-scope-nights"}
    arc = (max(r["_es"] for r in recs) - min(r["_es"] for r in recs)) / 86400
    out["arc_days"] = round(arc, 3)
    if arc > MAX_ARC_DAYS:
        return {**out, "status": "multi-opposition"}
    def tracklet_nights(trk):
        return max(night_index([t["epoch_mid_sec"] / 86400 for t in trk])) + 1 if trk else 0

    trk = build_tracklets(recs)
    out["tracklet_rule"] = "120min"
    if tracklet_nights(trk) < 3:
        # A survey cadence of ~2 h between detections (DECam W84) leaves lone
        # detections under the 120-min rule the model was trained on; such an arc
        # is scored on one tracklet per station-night rather than refused.  Arcs
        # the 120-min rule already scores never take this path, so their inputs
        # stay those of training.
        for r in recs:
            r.pop("tracklet_id", None)
        trk = build_tracklets(recs, station_night=True)
        out["tracklet_rule"] = "station-night"
    if tracklet_nights(trk) < 3:
        # nights made of single detections have no rate: the model saw only
        # tracklets of two or more detections
        return {**out, "status": "single-detection-nights"}
    by = {t["tracklet_id"]: [recs[k] for k in t["obs_idx"]] for t in trk}
    f = feat.all_features(trk, by, feat.PoolIndex([]), obscodes)
    if f.get("fit_ok") != 1:
        return {**out, "status": "fit-failed", "error": f.get("fit_err")}
    ex = explain(f)
    level, text = verdict(ex["score"])
    return {**out, "status": "ok", "score": ex["score"], "likely": ex["score"] >= THRESHOLD,
            "level": level, "verdict": text,
            "in_domain": in_domain(f), "explanation": ex,
            "n_tracklets": len(trk), "n_stations": len({t["stn"] for t in trk}),
            "gaps_days": [_num(f.get(f"gap_g{k}")) for k in (1, 2, 3)]}


def records_from_obs(obs, desig):
    """Engine observation dicts (a fit's ``_obs``) -> obs_sbn-shaped records for
    :func:`score_records`, so pasted astrometry is scored exactly as given.

    Every ground-based optical observation is kept, included in the fit or not:
    like the published arc, it is the astrometry as reported that is judged.
    Space-based and roving observations carry their position in fields this
    path does not rebuild, so they are left out (``n_skipped``).

    The position is the one REPORTED: a fit's observations are already star-catalogue
    debiased (``ra_raw``/``dec_raw`` keep the reported one), and scoring debiases its
    records itself, exactly as it does a published row -- a corrected position passed on
    would be corrected twice.
    """
    import datetime as dt
    j2000 = dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc)
    out, skipped = [], 0
    for i, o in enumerate(obs):
        if o.get("sat_pos_au") is not None or o.get("site_itrs_au") is not None \
                or o.get("radar") or o.get("ra_deg") is None or o.get("jd_utc") is None:
            skipped += 1
            continue
        t = j2000 + dt.timedelta(days=float(o["jd_utc"]) - 2451545.0)
        out.append({
            "obsid": f"page{i}", "trkid": None, "trksub": o.get("trksub"),
            "permid": None, "provid": desig, "stn": o.get("obscode"),
            "obstime": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "ra": float(o.get("ra_raw", o["ra_deg"])),
            "dec": float(o.get("dec_raw", o["dec_deg"])),
            "rmsra": o.get("ades_rms_ra"), "rmsdec": o.get("ades_rms_dec"), "rmscorr": None,
            "mag": o.get("mag"), "rmsmag": o.get("mag_sigma"),
            "band": (o.get("band") or " ").strip() or None,
            "astcat": o.get("astcat"), "mode": o.get("mode"), "deprecated": None, "sys": None,
            "notes": o.get("notes"), "prog": o.get("prog"), "ref": o.get("ref"),
            "prectime": o.get("prec_time"), "precra": o.get("prec_ra"),
            "precdec": o.get("prec_dec"), "rmstime": o.get("sigma_time_sec"),
        })
    return out, skipped


def score_observations(obs, desig, obscodes):
    """Score a fit's own observations (pasted / added astrometry), uncached."""
    t0 = time.time()
    rows, skipped = records_from_obs(obs, desig)
    res = score_records(rows, desig, obscodes)
    res.update(source="page", n_skipped=skipped, secs=round(time.time() - t0, 2),
               scored_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return res


def fetch_rows(conn, desig):
    """Published ('P') and accepted-pending ('p') obs_sbn rows of a provisional designation."""
    from .tracklets import COLUMNS
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT {', '.join(COLUMNS)} FROM obs_sbn "
                    f"WHERE provid = %s AND status IN ('P', 'p')", [desig])
        return [dict(zip(COLUMNS, r)) for r in cur.fetchall()]
    finally:
        cur.close()


def score_designation(desig, *, obscodes=None, conn=None):
    """Score a provisional designation from the SBN replica; cached ``CACHE_SEC``."""
    desig = " ".join(str(desig or "").split())
    if not desig:
        return {"status": "error", "error": "no designation"}
    hit = _cache.get(desig)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    if obscodes is None:
        from pyroorb.obs_sources import load_obscodes
        obscodes = load_obscodes()
    t0 = time.time()
    if conn is None:
        from pyroorb import sbn_obs
        if not sbn_obs.available():
            return {"designation": desig, "status": "unavailable", "source": "published",
                    "error": "no SBN replica configured"}
        from pyroorb.itf.config import ItfConfig
        from pyroorb.itf.dbio import connect_pub
        c = connect_pub(ItfConfig())
        try:
            rows = fetch_rows(c, desig)
        finally:
            c.close()
    else:
        rows = fetch_rows(conn, desig)
    res = score_records(rows, desig, obscodes)
    res["source"] = "published"
    res["secs"] = round(time.time() - t0, 2)
    res["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _cache[desig] = (time.time(), res)
    if len(_cache) > 2000:
        for k, _ in sorted(_cache.items(), key=lambda kv: kv[1][0])[:500]:
            _cache.pop(k, None)
    return res


def candidates():
    """The committed watch list (``candidates.json``) with the feature glossary."""
    data = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    data["descriptions"] = {k: {"label": v[0], "unit": v[1], "text": v[2]}
                            for k, v in DESCRIPTIONS.items()}
    data["threshold"] = THRESHOLD
    data["tiers"] = [{"min": lo, "level": lv, "verdict": t} for lo, lv, t in TIERS]
    return data
