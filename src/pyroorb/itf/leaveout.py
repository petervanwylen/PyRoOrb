"""Leave-out confirmation: a data-quality caution the find's own orbit vouches for
becomes a diagnostic.

ITF search warnings exist to flag a possibly WRONG identification, not noisy data
inside a correct one.  A photometric tracklet outlier, a coherently offset night or
a residual trend on a linked tracklet is one or the other.  The test that tells them
apart is the one the linked orbit cannot fake: refit WITHOUT the flagged tracklet and
predict it.  A contaminant the linked fit has bent to swallow sits well inside any
arcsec bar under the LINKED orbit, and 5-40" off under the leave-out orbit.

A tracklet is CONFIRMED when, under the leave-out orbit,

* the prediction is tight: 1-sigma semi-major axis <= ``MAX_SIGMA_ARCSEC`` (a loose
  one means the tracklet is load-bearing, so the check is not independent of it);
* its mean sits <= ``MAX_MISS_ARCSEC`` from the prediction, and <= ``MAX_D`` in
  Mahalanobis units against the prediction's covariance + the station-night offset
  (``nightmodel_table``) + ``SIGMA_OBS``/sqrt(n);
* its residual walks <= ``MAX_WALK_ARCSEC`` end to end -- it moves at the right RATE.
  Load-bearing: 2020 WT32's wrong tracklet (2020 WN29's) sat 0.4" from a 0.25"
  prediction and walked 2.8" in 22 minutes.  Three or more observations: a fitted
  line; TWO: the difference between them (it is still a rate); ONE: no rate can be
  measured at all, so a single-observation tracklet is never confirmed
  (deliberately: position alone does not vouch for it).

On the labelled populations (25,639 batch finds labelled by the MPC, 314 rebuilt,
65 live harness runs) this demotes 92 of 151 correct warnings and 0 of 10 wrong ones.

What it is applied to (``search._accepted``):

* the IN-SCOPE cautions only -- photometric tracklet outlier, coherent offset, the
  two-axis residual trend -- plus the residual-trend NOTE that the review layer
  (``itf.batch.review_warnings``) counts as a warning;
* a caution naming a LINKED tracklet: that tracklet must be confirmed (every linked
  tracklet it could name, when its text gives only a date);
* a caution about the object's OWN published arc names no linked data: every
  apparition's linked additions, left out TOGETHER, must be confirmed;
* NEVER on a linkage the three-thin (1.75 -> 3) confirmation made submittable: that
  withdrawal rule stays exactly as it is.

A confirmed caution is moved to ``diagnostics`` under ``LEAVEOUT_NOTE``, never the
trend prefix, so the CSV / megabatch review split no longer counts it.
"""

import math
import re
from .obs80_fields import is_companion, obs_unique_id

#: the confirmation bars (measured on the labelled populations above)
MAX_SIGMA_ARCSEC = 2.0
MAX_MISS_ARCSEC = 1.5
MAX_D = 3.0
MAX_WALK_ARCSEC = 1.5
#: per-observation astrometric noise assumed for D and the walk's standard error
SIGMA_OBS = 0.4

#: prefix of a demoted caution in ``diagnostics`` (deliberately NOT the trend prefix)
LEAVEOUT_NOTE = "confirmed by leave-out:"

#: a warning tracklet is the ITF tracklet observed this many days around its date
_DATE_SLACK = (-0.6, 1.6)
#: the gap that separates apparitions when an own-arc caution is confirmed per apparition
APPARITION_GAP_DAYS = 275.0

_DATE_RX = re.compile(r"(\d{4})-(\d\d)-(\d\d)")
_STN_RXS = (re.compile(r"\(([0-9A-Z]\d\d)[,\s]"), re.compile(r"\d{4}-\d\d-\d\d ([0-9A-Z]\d\d) "))


def in_scope(text):
    """``"photometry" | "offset" | "trend"`` for a caution this rule may demote, else None."""
    if not isinstance(text, str):
        return None
    low = text.lower()
    if "predicted brightness" in low:
        return "photometry"
    if "systematically offset" in low:
        return "offset"
    from .residual_trend import is_trend_note
    if is_trend_note(text):
        return "trend"
    return None


def is_own_arc(text):
    low = text.lower()
    return "own published arc" in low or "object's own" in low


def _jd_of_date(y, m, d):
    a = (14 - m) // 12
    y2, m2 = y + 4800 - a, m + 12 * a - 3
    return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045 - 0.5


def target(text, mag_tracklet=None):
    """``(station or None, jd or None)`` the caution names.  Photometry takes the
    search's own ``vetting["mag_tracklet"]`` (an older wording carries no date)."""
    stn = jd = None
    m = _DATE_RX.search(text)
    if m:
        jd = _jd_of_date(*map(int, m.groups()))
    for rx in _STN_RXS:
        s = rx.search(text)
        if s:
            stn = s.group(1)
            break
    if in_scope(text) == "photometry" and mag_tracklet:
        stn = mag_tracklet.get("station") or stn
        jd = mag_tracklet.get("jd") if mag_tracklet.get("jd") is not None else jd
    return stn, jd


def _lines(text):
    return [l for l in (text or "").splitlines() if l.strip()]


def _jd(line):
    """The UTC JD of an observation line OR its `s`/`v` companion: both carry the
    date in cols 16-32, but only the first carries a position."""
    from astropy.time import Time
    dfrac = float(line[23:32])
    day = int(dfrac)
    return Time(f"{int(line[15:19]):04d}-{int(line[20:22]):02d}-{day:02d}T00:00:00",
                scale="utc").jd + (dfrac - day)


def _observations(lines):
    """``lines`` without their `s`/`v` companion lines, in time order: what can be
    measured against a prediction (the companion only says where the observer was)."""
    return sorted((l for l in lines if not is_companion(obs_unique_id(l))), key=_jd)


def _unplaceable(lines, obscodes):
    """The first station among ``lines`` the ephemeris cannot put the observer at
    (a satellite or a roving site: col 15 `S`/`V`, or a listed station without
    parallax constants), or None.  Its prediction would be geocentric against
    an orbiting observer's astrometry: a parallax, not a miss."""
    for line in lines:
        stn = line[77:80].strip() or "500"
        row = (obscodes or {}).get(stn)
        if line[14] in "SV" or (stn != "500" and row
                                and abs(row[1]) < 1e-9 and abs(row[2]) < 1e-9):
            return stn
    return None


def itf_tracklets(itf_obs80):
    """``{(trksub, station): [lines]}`` of the linked ITF astrometry."""
    out = {}
    for l in _lines(itf_obs80):
        out.setdefault((l[5:12].strip(), l[77:80]), []).append(l)
    return out


def named_tracklets(text, itf_obs80, mag_tracklet=None):
    """The linked tracklets a caution names: ``[(key, lines)]`` -- empty when it names
    none of them (an own-arc caution, or a night only the seed observed)."""
    stn, jd = target(text, mag_tracklet)
    if jd is None:
        return []
    out = []
    for key, ls in itf_tracklets(itf_obs80).items():
        if stn is not None and key[1] != stn:
            continue
        if any(_DATE_SLACK[0] <= _jd(l) - jd <= _DATE_SLACK[1] for l in ls):
            out.append((key, ls))
    return out


def _tau(stn):
    try:
        from pyroorb.nightmodel_table import POOLED, STATIONS
    except ImportError:                                  # pragma: no cover
        return 0.15
    p = STATIONS.get(stn, POOLED)
    return math.hypot(p[2], p[3]) / math.sqrt(2)


def _slope(ts, ys):
    k = len(ts)
    tb = sum(ts) / k
    sxx = sum((t - tb) ** 2 for t in ts)
    if k < 3 or sxx <= 0:
        return 0.0
    return sum((t - tb) * y for t, y in zip(ts, ys)) / sxx


def tracklet_stats(stn, pts, cov):
    """``{miss, D, walk, n}`` for one tracklet's ``[(jd, dE, dN)]`` O-C (arcsec) against a
    prediction with 2x2 sky covariance ``cov`` (arcsec^2, East/North)."""
    k = len(pts)
    me = sum(p[1] for p in pts) / k
    mn = sum(p[2] for p in pts) / k
    nv = _tau(stn) ** 2 + SIGMA_OBS ** 2 / k
    a, b, d = cov[0][0] + nv, cov[0][1], cov[1][1] + nv
    det = a * d - b * b
    D = math.sqrt(max(0.0, (d * me * me - 2 * b * me * mn + a * mn * mn) / det)) if det > 0 else float("inf")
    ts = [p[0] for p in pts]
    span = max(ts) - min(ts)
    if k == 2:
        # two points ARE a rate: their difference is the drift end to end
        (t0, e0, n0), (t1, e1, n1) = sorted(pts)
        walk = math.hypot(e1 - e0, n1 - n0)
    else:
        walk = math.hypot(_slope(ts, [p[1] for p in pts]), _slope(ts, [p[2] for p in pts])) * span
    return {"miss": math.hypot(me, mn), "D": D, "walk": walk, "n": k}


def _sky_cov(row):
    a, b, pa = row["unc_smaj_arcsec"], row["unc_smin_arcsec"], math.radians(row["unc_pa_deg"])
    u, v = (math.sin(pa), math.cos(pa)), (-math.cos(pa), math.sin(pa))
    cross = a * a * u[0] * u[1] + b * b * v[0] * v[1]
    return [[a * a * u[0] ** 2 + b * b * v[0] ** 2, cross],
            [cross, a * a * u[1] ** 2 + b * b * v[1] ** 2]]


def measure(unit_lines, combined_obs80, obscodes, registry=None, max_sec=30.0):
    """Leave ``unit_lines`` out of ``combined_obs80``, refit, and measure them.

    Returns ``{"confirmed", "smaj", "miss", "D", "walk", "tracklets"}`` (the unit takes
    its worst tracklet) or ``{"confirmed": False, "error": ...}`` -- never raises.
    """
    from pyroorb.ephemerides import compute_ephemeris
    from pyroorb.obs80 import mpc80_to_radec
    from .fitter import fit_astrometry
    drop = set(unit_lines)
    keep = [l for l in _lines(combined_obs80) if l not in drop]
    if not keep or not unit_lines:
        return {"confirmed": False, "error": "nothing to leave out"}
    try:
        tf = fit_astrometry("\n".join(keep) + "\n", obscodes, registry=registry, max_sec=max_sec)
        if not tf.ok:
            return {"confirmed": False, "error": f"leave-out fit failed: {tf.error}"}
        groups, mid = {}, None
        order = _observations(unit_lines)
        bad = _unplaceable(order, obscodes)
        if bad is not None:
            return {"confirmed": False, "error": f"station {bad} cannot be placed"}
        for i, line in enumerate(order):
            jd, ra, dec, stn = mpc80_to_radec(line)
            row = compute_ephemeris(tf.fit, obscodes, jd, jd, 1.0, site=stn or "500", full=False)[0]
            cosd = math.cos(math.radians(dec))
            dE = ((ra - row["ra_deg"] + 180.0) % 360.0 - 180.0) * cosd * 3600.0
            dN = (dec - row["dec_deg"]) * 3600.0
            g = groups.setdefault(stn, [])
            if g and jd - g[-1][-1][0] >= 0.5:
                g.append([])
            if not g:
                g.append([])
            g[-1].append((jd, dE, dN))
            if i == len(order) // 2:
                mid = row
        smaj = mid["unc_smaj_arcsec"]
        cov = _sky_cov(mid)
        ts = [dict(tracklet_stats(stn, pts, cov), station=stn)
              for stn, runs in groups.items() for pts in runs]
    except Exception as ex:                              # noqa: BLE001
        return {"confirmed": False, "error": f"{type(ex).__name__}: {ex}"}
    out = {"smaj": smaj, "miss": max(t["miss"] for t in ts), "D": max(t["D"] for t in ts),
           "walk": max(t["walk"] for t in ts), "tracklets": ts}
    out["n_min"] = min(t["n"] for t in ts)
    out["confirmed"] = bool(smaj <= MAX_SIGMA_ARCSEC and out["miss"] <= MAX_MISS_ARCSEC
                            and out["D"] <= MAX_D and out["walk"] <= MAX_WALK_ARCSEC
                            and out["n_min"] >= 2)
    if out["n_min"] < 2:
        out["why"] = "a single observation: its rate cannot be checked"
    return out


def influence(unit_lines, full_fit, loo_fit, obscodes):
    """How hard ``unit_lines`` PULL the orbit: their positions as predicted by the
    fit WITH them (``full_fit``) and by the fit WITHOUT them (``loo_fit``).

    Returns ``{"smaj", "miss", "shift", "n", "per_obs"}`` in arcsec -- ``smaj`` the leave-out
    prediction's 1-sigma semi-major axis mid-unit, ``miss`` the unit's mean offset
    from the leave-out prediction, ``shift`` how far adding the unit moves the
    prediction (the mean of the per-observation vectors), ``per_obs`` each
    observation's ``(line, dE, dN)`` offset from the leave-out prediction -- or
    ``{"error": ...}``;
    never raises.  A mislinked tracklet the orbit bends to swallow has ``shift``
    close to ``miss``; one the orbit does not move toward at all has ``shift`` near
    zero, whatever its ``miss``."""
    from pyroorb.ephemerides import compute_ephemeris
    from pyroorb.obs80 import mpc80_to_radec
    order = _observations(unit_lines)
    if not order:
        return {"error": "nothing to measure"}
    bad = _unplaceable(order, obscodes)
    if bad is not None:
        return {"error": f"station {bad} cannot be placed"}
    try:
        de = dn = se = sn = 0.0
        smaj = None
        per_obs = []
        for i, line in enumerate(order):
            jd, ra, dec, stn = mpc80_to_radec(line)
            site = stn or "500"
            lo = compute_ephemeris(loo_fit, obscodes, jd, jd, 1.0, site=site, full=False)[0]
            fu = compute_ephemeris(full_fit, obscodes, jd, jd, 1.0, site=site, full=False)[0]
            cosd = math.cos(math.radians(dec))
            oe = ((ra - lo["ra_deg"] + 180.0) % 360.0 - 180.0) * cosd * 3600.0
            on = (dec - lo["dec_deg"]) * 3600.0
            per_obs.append((line, oe, on))
            de += oe
            dn += on
            se += ((fu["ra_deg"] - lo["ra_deg"] + 180.0) % 360.0 - 180.0) * cosd * 3600.0
            sn += (fu["dec_deg"] - lo["dec_deg"]) * 3600.0
            if i == len(order) // 2:
                smaj = lo["unc_smaj_arcsec"]
    except Exception as ex:                              # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"}
    k = len(order)
    return {"smaj": smaj, "miss": math.hypot(de / k, dn / k),
            "shift": math.hypot(se / k, sn / k), "n": k, "per_obs": per_obs}


def apparitions(itf_obs80, combined_obs80, gap_days=APPARITION_GAP_DAYS):
    """The linked ITF lines grouped by the apparition of the combined arc they fall in."""
    lines = sorted(_lines(combined_obs80), key=_jd)
    app_of, k, prev = {}, 0, None
    for l in lines:
        jd = _jd(l)
        if prev is not None and jd - prev > gap_days:
            k += 1
        app_of[l] = k
        prev = jd
    out = {}
    for l in _lines(itf_obs80):
        out.setdefault(app_of.get(l, -1), []).append(l)
    return list(out.values())


#: what separates a demoted caution's own text from the measurement appended to it
_NOTE_TAIL = " -- but the orbit fitted WITHOUT that data predicts it "


def original(text):
    """The caution a ``LEAVEOUT_NOTE`` diagnostic was made from, or None."""
    if not isinstance(text, str) or not text.startswith(LEAVEOUT_NOTE):
        return None
    body = text[len(LEAVEOUT_NOTE):].lstrip()
    i = body.rfind(_NOTE_TAIL)
    return body[:i] if i >= 0 else None


def note(text, m):
    """The diagnostic a confirmed caution becomes."""
    return (f"{LEAVEOUT_NOTE} {text}{_NOTE_TAIL}"
            f"{m['miss']:.2f}\" away (1-sigma {m['smaj']:.2f}\", D {m['D']:.1f}) and it moves "
            f"at the predicted rate (drift {m['walk']:.2f}\"), so it is noisy data of this "
            f"object, not a different body")


def confirm_cautions(warnings, diagnostics, itf_obs80, combined_obs80, obscodes,
                     registry=None, mag_tracklet=None, max_sec=30.0, log=None):
    """Split the in-scope cautions the leave-out orbit vouches for out of ``warnings``
    (and the trend notes out of ``diagnostics``).

    Returns ``(warnings, diagnostics, record)``: the warnings left, the diagnostics
    with every confirmed caution appended as a ``LEAVEOUT_NOTE`` (a confirmed trend
    note REPLACES its original), and ``record`` -- one entry per in-scope caution
    examined: ``{text, scope, own, confirmed, unit, ...measure}``.
    """
    memo = {}

    def meas(lines):
        key = tuple(sorted(lines))
        if key not in memo:
            memo[key] = measure(lines, combined_obs80, obscodes, registry, max_sec)
        return memo[key]

    own_ok = None

    def own_confirmed():
        nonlocal own_ok
        if own_ok is None:
            apps = apparitions(itf_obs80, combined_obs80)
            own_ok = bool(apps) and all(meas(ls)["confirmed"] for ls in apps)
        return own_ok

    def judge(text):
        scope = in_scope(text)
        if scope is None:
            return None
        rec = {"text": text, "scope": scope, "own": is_own_arc(text)}
        if rec["own"]:
            rec["unit"] = "every linked apparition"
            rec["confirmed"] = own_confirmed()
            ms = [meas(ls) for ls in apparitions(itf_obs80, combined_obs80)]
            if ms and all("miss" in m for m in ms):
                rec.update({k: max(m[k] for m in ms) for k in ("smaj", "miss", "D", "walk")})
            return rec
        named = named_tracklets(text, itf_obs80, mag_tracklet)
        if not named:
            rec.update(confirmed=False, unit=None, why="names no linked tracklet")
            return rec
        ms = [meas(ls) for _k, ls in named]
        rec["unit"] = ", ".join(f"{k[0]}@{k[1]}" for k, _ls in named)
        rec["confirmed"] = all(m["confirmed"] for m in ms)
        if all("miss" in m for m in ms):
            rec.update({k: max(m[k] for m in ms) for k in ("smaj", "miss", "D", "walk")})
            why = [m["why"] for m in ms if m.get("why")]
            if why:
                rec["why"] = why[0]
        else:
            rec["why"] = "; ".join(m["error"] for m in ms if "error" in m)
        return rec

    kept, diags, record = [], list(diagnostics or []), []
    for w in warnings or []:
        rec = judge(w)
        if rec is None:
            kept.append(w)
            continue
        record.append(rec)
        if rec["confirmed"] and "miss" in rec:
            diags.append(note(w, rec))
        else:
            kept.append(w)
    for i, d in enumerate(list(diags)):
        if in_scope(d) != "trend" or d in (warnings or []):
            continue
        rec = judge(d)
        if rec is None:
            continue
        record.append(rec)
        if rec["confirmed"] and "miss" in rec:
            diags[i] = note(d, rec)
    if log:
        for r in record:
            log(f"leave-out {'CONFIRMED' if r['confirmed'] else 'not confirmed'}: "
                f"{r.get('unit')} " + (f"miss {r['miss']:.2f}\" D {r['D']:.1f} walk "
                                       f"{r['walk']:.2f}\" sigma {r['smaj']:.2f}\""
                                       if "miss" in r else r.get("why", "")))
    return kept, diags, record
