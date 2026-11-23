"""obs_sbn rows -> the tracklets the mislinkage model reads.

The model was trained on tracklets built exactly this way, so an object is
scored only through here: one station, consecutive detections at most 120 min
apart (the ITF flat file's tracklet rule), each summarised by
``itf.index.compute_tracklet_features`` (sky position, rate, linear-motion
fit quality).
"""
from __future__ import annotations

import datetime as dt

#: Same tracklet rule as the ITF flat file: one station, gaps <= 120 min.
TRACKLET_GAP_DAYS = 120.0 / 1440.0

#: The obs_sbn columns every record carries (what ``ades.records_to_obs`` reads).
COLUMNS = ("obsid", "trkid", "trksub", "permid", "provid", "stn", "obstime",
           "ra", "dec", "rmsra", "rmsdec", "rmscorr", "mag", "rmsmag", "band",
           "astcat", "mode", "deprecated", "sys", "ctr", "pos1", "pos2", "pos3",
           "prectime", "precra", "precdec", "rmstime", "notes", "prog", "ref")


def clean_record(rec):
    """numeric -> float, timestamps -> ISO text (what ades.records_to_obs reads)."""
    from decimal import Decimal
    r = {}
    for k, v in rec.items():
        if isinstance(v, Decimal):
            v = float(v)
        elif isinstance(v, dt.datetime):
            v = v.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        r[k] = v
    return r


def epoch_sec(obstime):
    """ISO ``...Z`` text (UTC) -> Unix seconds.  Never ``datetime.timestamp()`` on a
    naive value: that reads it in the machine's local zone."""
    return dt.datetime.fromisoformat(obstime[:-1]).replace(tzinfo=dt.timezone.utc).timestamp()


def build_tracklets(recs, *, station_night=False):
    """Group records (each with ``obj``, ``stn``, ``_es``) into tracklets.

    Returns tracklet dicts (``tracklet_id``, ``epoch_mid_sec``, ``center_ra`` /
    ``center_dec``, rates and their sigmas, ``n_obs``, ``stn``, ``obs_idx`` into
    ``recs``); a run ``compute_tracklet_features`` cannot summarise is dropped.

    ``station_night``: one tracklet per station and night (``orbit_common.night_index``)
    instead of the 120-min rule.  A survey revisiting a field every ~2 h (DECam W84
    in 2014: four detections 2h12m-2h30m apart each night) leaves only lone
    detections under the 120-min rule, so its nights would be dropped although each
    holds a rate.  The model was trained on 120-min tracklets, so the caller uses
    this only for an arc the 120-min rule cannot score at all.
    """
    from pyroorb.itf.index import IndexConfig, compute_tracklet_features
    from pyroorb.orbit_common import night_index
    cfg = IndexConfig()
    if station_night:
        order = sorted(range(len(recs)), key=lambda i: recs[i]["_es"])
        night = dict(zip(order, night_index([recs[i]["_es"] / 86400 for i in order])))
    groups = {}
    for i, r in enumerate(recs):
        groups.setdefault((r["obj"], r["stn"]), []).append(i)
    tracklets = []
    for (obj, stn), idx in groups.items():
        idx.sort(key=lambda i: recs[i]["_es"])
        run = [idx[0]]
        for i in idx[1:] + [None]:
            same = i is not None and (
                night[i] == night[run[-1]] if station_night
                else recs[i]["_es"] - recs[run[-1]]["_es"] <= TRACKLET_GAP_DAYS * 86400)
            if same:
                run.append(i)
                continue
            tid = f"{obj}|{stn}|{int(recs[run[0]]['_es'])}"
            rows = [{"es": recs[k]["_es"], "ra": recs[k]["ra"], "dec": recs[k]["dec"],
                     "rms_ra": recs[k].get("rmsra"), "rms_dec": recs[k].get("rmsdec"),
                     "mag": recs[k].get("mag"), "band": recs[k].get("band"),
                     "stn": stn, "trksub": recs[k].get("trksub"), "provid": obj,
                     "tracklet_id": tid} for k in run]
            f = compute_tracklet_features(rows, cfg)
            if f is not None:
                f["obj"] = obj
                f["nopp"] = recs[run[0]]["nopp"]
                f["obs_idx"] = list(run)
                tracklets.append(f)
                for k in run:
                    recs[k]["tracklet_id"] = tid
            run = [i] if i is not None else []
    return tracklets
