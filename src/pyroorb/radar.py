"""Radar (delay / Doppler) observations — ingest + measurement model + post-fit refine.

find_orb ingests radar range/Doppler astrometry and folds each
measurement into the least-squares fit as a sigma-normalized residual row
(orb_func.cpp get_residual_data). This is the PyRoOrb port:

- MEASUREMENT MODEL — a port of find_orb `compute_radar_info` (mpc_obs.cpp:5138).
  The epoch is the UTC RECEIVE time. The round-trip delay is the down-leg (receiver
  at receive time -> object at bounce time, light-time iterated) plus the up-leg
  (transmitter at transmit time -> object at bounce time, iterated — the transmitter
  can be a different station, e.g. Goldstone-transmit / GBT-receive). The Doppler
  shift is the two-leg special-relativistic factor
  sqrt((1-v1)(1-v2)/((1+v1)(1+v2))) - 1 times the transmitter frequency, with v1/v2
  the object<->receiver / object<->transmitter radial rates in units of c. Like
  find_orb we model neither the Shapiro (GR) delay nor the troposphere (~1 us for
  near-Earth geometry; JPL models both), and the bounce-point flag (center-of-mass
  'C' vs peak-power surface 'P') is carried but not corrected for.

- PIPELINE PLACEMENT — radar obs do NOT ride the optical pipeline (IOD / extension /
  outlier rejection are sky-plane machinery; find_orb likewise sorts radar out of its
  Herget/IOD stages, orb_func.cpp sort_unused_obs_to_end). They join at the END as
  extra weighted rows in a final differential correction (`refine_radar`, mirroring
  the refine_asteroids post-fit pattern): same LM, optical rows from the fit's
  analytic-Jacobian evaluator + radar rows from a 7-particle states sweep
  (force_assist.helio_states_multi), so the combined covariance is radar-tightened.

- SOURCE — JPL's Small-Body Radar Astrometry API (the canonical Ostro/Benner/Naidu
  dataset; the MPC obs API serves NO radar in either OBS80 or ADES, verified 2026-07),
  fetched once as the full ~4700-row dump and cached like the other API caches. Plus
  the OBS80 two-line 'R'/'r' pair parser for file-based ingest (--batch and pasted
  astrometry), ported from find_orb mpc_obs.cpp extract_radar_value.
"""
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from astropy.time import Time

from pyroorb.forward_model import C_AU_DAY, MU_SUN
from pyroorb.obs_sources import A_EARTH_AU
from pyroorb.obs80 import geodetic_to_itrs_au

RADAR_API = "https://ssd-api.jpl.nasa.gov/sb_radar.api"

# Human-readable station names for the report (JPL numeric ids + MPC radar codes).
STATION_NAMES = {
    "-1": "Arecibo", "-2": "Haystack", "-9": "GBT", "-13": "DSS-13", "-14": "DSS-14",
    "-25": "DSS-25", "-34": "DSS-34", "-35": "DSS-35", "-36": "DSS-36", "-38": "Evpatoria",
    "-43": "DSS-43", "-47": "ATCA", "-63": "DSS-63", "-73": "EISCAT-Tromso",
    "251": "Arecibo", "252": "DSS-13", "253": "DSS-14", "254": "Haystack",
    "255": "Evpatoria", "256": "GBT", "257": "DSS-25",
}


def station_name(code):
    return STATION_NAMES.get(str(code), str(code))
RADAR_CACHE_TTL_S = 7 * 86400        # radar astrometry changes rarely; refresh weekly
SEC_PER_DAY = 86400.0
TG_SHAPIRO = 2 * 4.9254909e-6        # (1+gamma) GM_sun/c^3, seconds (gamma=1)

# Post-fit refine knobs.
RADAR_REJECT_Z = 10.0    # a radar row this many sigma off after the joint fit is excluded
                         # once and the fit re-run (old Arecibo delays are occasionally off)
RADAR_MAX_OPTICAL_DEGRADE = 1.5  # the radar-refined orbit may not fit the OPTICAL data worse
#                          than this multiple of the optical-only fit's normalized rms. The
#                          existing RADAR_ABANDON_Z guard below is one-sided -- it asks only
#                          whether the joint fit can represent the RADAR -- so a solution that
#                          satisfies neither side slips through: 2003 HM lands at radar median
#                          |z| 5.68 (inside the 25.0 bar) while the optical goes 0.96 -> 9.34
#                          sigma, a 6.6x degradation, and that compromise is what gets delivered.
#                          Adding 9 tight rows to 260 optical ones may legitimately cost a
#                          little, so this is a factor rather than "no worse"; 1.5x is far above
#                          what a genuine refinement costs (2005 YU55, the validation object,
#                          moves the optical rms by well under a percent) and far below the
#                          irreconcilable case. When it trips, the OPTICAL orbit is kept and the
#                          disagreement is reported -- an orbit fitting neither dataset is not a
#                          compromise worth shipping.
RADAR_ABANDON_Z = 25.0   # median |z| beyond this after refit = the model cannot represent
                         # these measurements (mis-attribution / bad station) -> keep the
                         # optical orbit and flag the report instead of wrecking the fit
RADAR_MAX_NFEV = 60


# ---------------------------------------------------------------------------
# Ingest — JPL Small-Body Radar Astrometry API
# ---------------------------------------------------------------------------
def _cache_fresh(path, ttl_s=RADAR_CACHE_TTL_S):
    try:
        return path is not None and path.exists() and time.time() - path.stat().st_mtime < ttl_s
    except OSError:
        return False


def _fetch_radar_dump(cache_dir=None):
    """The FULL JPL radar-astrometry table + station coordinates, as one cached JSON.
    ~4700 rows total across ~1100 objects, so one dump beats a per-object query: after
    the first fetch every lookup — including the 99% of objects with no radar — is a
    local dict hit. Returns the decoded API payload, or None offline with no cache."""
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / "radar_astrometry.json"
    if _cache_fresh(cache):
        try:
            return json.loads(cache.read_text())
        except (json.JSONDecodeError, ValueError):
            cache.unlink()
    try:
        raw = json.loads(urllib.request.urlopen(
            RADAR_API + "?coords=1", timeout=60).read())
    except Exception:
        # offline: fall back to a stale cache if one exists, else no radar
        if cache is not None and cache.exists():
            try:
                return json.loads(cache.read_text())
            except (json.JSONDecodeError, ValueError):
                return None
        return None
    if cache is not None:
        from pyroorb import datadirs
        datadirs.write_cache(cache, json.dumps(raw))
    return raw


def _row_to_obs(fields, row, stations):
    """One JPL API row -> our radar-obs dict (see module docstring for the model the
    fields feed). Delay rows carry `value`/`sigma` in microseconds -> stored in SECONDS;
    Doppler rows are Hz. The epoch is the UTC receive time."""
    d = dict(zip(fields, row))
    st_rx = stations.get(str(d["rcvr"])); st_tx = stations.get(str(d["xmit"]))
    if st_rx is None or st_tx is None:
        return None                       # unknown station: no coordinates, no model
    jd = Time(d["epoch"].replace(" ", "T"), format="isot", scale="utc").jd
    kind = "delay" if d["units"] == "us" else "doppler"
    scale = 1e-6 if kind == "delay" else 1.0
    return {"kind": kind, "jd_utc": float(jd),
            "value": float(d["value"]) * scale, "sigma": float(d["sigma"]) * scale,
            "freq_hz": float(d["freq"]) * 1e6,
            "rx": str(d["rcvr"]), "tx": str(d["xmit"]),
            "rx_itrs_au": st_rx, "tx_itrs_au": st_tx,
            "bp": d.get("bp", "C"), "desig": str(d["des"])}


def fetch_radar(desigs, cache_dir=None):
    """Radar obs dicts for any of `desigs` (a designation or a list), from the cached
    JPL dump. Designations are matched exactly as JPL prints them (numbered asteroids
    as the bare number, provisionals as 'YYYY XX'), plus a parens-stripped variant.
    Returns [] when the object has no radar (the common case) or offline w/o cache."""
    dump = _fetch_radar_dump(cache_dir)
    if not dump or not dump.get("data"):
        return []
    stations = {}
    for k, c in (dump.get("coords") or {}).items():
        try:
            stations[str(k)] = geodetic_to_itrs_au(
                float(c["longitude"]), float(c["latitude"]), float(c["altitude"]) * 1000.0)
        except (KeyError, ValueError):
            continue
    want = set()
    for d in ([desigs] if isinstance(desigs, str) else list(desigs)):
        s = str(d).strip()
        want |= {s, s.strip("()"), s.lstrip("(").split(")")[0]}
    fields = dump["fields"]
    out = []
    for row in dump["data"]:
        if str(row[0]) in want:
            o = _row_to_obs(fields, row, stations)
            if o is not None:
                out.append(o)
    out.sort(key=lambda o: (o["jd_utc"], o["kind"]))
    return out


def attach_radar(meta, cache_dir=None):
    """Attach meta['radar'] for the designations in `meta` (obs_sources.fetch_observations
    calls this after every fetch). Never raises; radar stays [] on any failure."""
    try:
        desigs = list(meta.get("designations") or [])
        if meta.get("designation"):
            desigs.append(meta["designation"])
        meta["radar"] = fetch_radar(desigs, cache_dir=cache_dir) if desigs else []
    except Exception:
        meta["radar"] = []
    return meta


# ---------------------------------------------------------------------------
# Ingest — OBS80 two-line 'R'/'r' pairs (file-based astrometry)
# ---------------------------------------------------------------------------
def _extract_radar_value(field):
    """Port of find_orb mpc_obs.cpp extract_radar_value: a 15-byte field with an
    implicit decimal point before the last 4 digits; byte 0 may be a sign or a
    unit-uncertainty tag ('S'/'C'); trailing spaces in the fraction read as zeros.
    Returns the float value, or 0.0 for a blank field."""
    buff = list(f"{field:<15.15}")
    for i in range(11, 15):
        if buff[i] == " ":
            buff[i] = "0"
    s = "".join(buff)
    body = s if s[0].isdigit() else s[1:]
    try:
        val = float(body) * 1e-4
    except ValueError:
        return 0.0
    return -val if s[0] == "-" else val


def _station_itrs_from_code(code, obscodes):
    """Earth-fixed ITRS position (au) of an MPC station from its parallax constants."""
    if code not in obscodes:
        return None
    lon, rcos, rsin = obscodes[code][0], obscodes[code][1], obscodes[code][2]
    lam = math.radians(lon)
    return [A_EARTH_AU * rcos * math.cos(lam), A_EARTH_AU * rcos * math.sin(lam),
            A_EARTH_AU * rsin]


def _mpc80_radar_jd(line):
    """UTC JD from an OBS80 radar line's date columns (16-32, the standard layout —
    radar epochs are integer-second UTC receive times). The RA/Dec columns hold the
    radar values, so the optical mpc80_to_radec parser cannot be reused."""
    y, mo = int(line[15:19]), int(line[20:22])
    dfrac = float(line[23:32])
    day = int(dfrac)
    return Time(f"{y:04d}-{mo:02d}-{day:02d}T00:00:00", scale="utc").jd + (dfrac - day)


def parse_obs80_radar_pair(line1, line2, obscodes):
    """OBS80 radar record ('R' line + its 'r' companion) -> list of radar-obs dicts
    (one per measurement actually present: delay and/or Doppler). Layout per
    https://www.minorplanetcenter.net/iau/info/RadarObs.html, ported from find_orb
    compute_radar_info's column reads: cols 33-47 delay (us, line1) / its sigma
    (line2); cols 48-62 Doppler (Hz) / sigma; cols 63-68 of BOTH lines the split
    transmitter frequency (MHz, implicit decimal after 5 digits); cols 69-71 the
    TRANSMITTING station, cols 78-80 the RECEIVING station. Times are UTC receive."""
    try:
        jd = _mpc80_radar_jd(line1)
    except ValueError:
        return []
    rx_code = line1[77:80].strip()
    tx_code = line1[68:71].strip() or rx_code
    rx = _station_itrs_from_code(rx_code, obscodes)
    tx = _station_itrs_from_code(tx_code, obscodes)
    if rx is None or tx is None:
        return []
    freq_txt = f"{line1[62:67]}.{line1[67]}{line2[62:68]}".replace(" ", "0")
    try:
        freq_hz = float(freq_txt) * 1e6
    except ValueError:
        return []
    delay = _extract_radar_value(line1[32:47]) * 1e-6      # us -> s
    delay_sig = _extract_radar_value(line2[32:47]) * 1e-6
    dopp = _extract_radar_value(line1[47:62])
    dopp_sig = _extract_radar_value(line2[47:62])
    from pyroorb.obs_sources import obs80_line_id
    desig = obs80_line_id(line1)
    out = []
    base = {"jd_utc": float(jd), "freq_hz": freq_hz, "rx": rx_code, "tx": tx_code,
            "rx_itrs_au": rx, "tx_itrs_au": tx,
            "bp": "P" if line2[32:33] == "S" else "C", "desig": desig}
    if delay and delay_sig:
        out.append({**base, "kind": "delay", "value": delay, "sigma": delay_sig})
    if dopp and dopp_sig:
        out.append({**base, "kind": "doppler", "value": dopp, "sigma": dopp_sig})
    return out


def ades_records_to_radar(records, obscodes):
    """ADES radar records -> radar-obs dicts. Per the ADES 2017 spec: `delay` is in
    SECONDS with `rmsDelay` in MICROSECONDS; `doppler`/`rmsDoppler` in Hz; `frq` in
    MHz; `trx`/`rcv` are the transmitting/receiving MPC station codes; `com`=1 means
    the value is center-of-mass referenced. (The MPC obs API serves no radar records
    — this covers user-supplied ADES files.)"""
    out = []
    for r in records:
        trx = (r.get("trx") or "").strip()
        rcv = (r.get("rcv") or (r.get("stn") or "")).strip()
        ot = r.get("obstime")
        if not (ot and (r.get("delay") or r.get("doppler")) and (trx or rcv)):
            continue
        trx, rcv = trx or rcv, rcv or trx
        tx = _station_itrs_from_code(trx, obscodes)
        rx = _station_itrs_from_code(rcv, obscodes)
        if tx is None or rx is None:
            continue
        try:
            jd = float(Time(str(ot).replace("Z", ""), format="isot", scale="utc").jd)
            freq_hz = float(r.get("frq")) * 1e6
        except (TypeError, ValueError):
            continue
        desig = (r.get("permid") or r.get("provid") or r.get("trksub") or "").strip()
        com = str(r.get("com") or "").strip()
        base = {"jd_utc": jd, "freq_hz": freq_hz, "rx": rcv, "tx": trx,
                "rx_itrs_au": rx, "tx_itrs_au": tx,
                "bp": "C" if com in ("1", "true", "True") else "P", "desig": desig}
        try:
            if r.get("delay"):
                out.append({**base, "kind": "delay", "value": float(r["delay"]),
                            "sigma": float(r.get("rmsdelay")) * 1e-6})
            if r.get("doppler"):
                out.append({**base, "kind": "doppler", "value": float(r["doppler"]),
                            "sigma": float(r.get("rmsdoppler"))})
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------
def _site_gcrs(jds_utc, itrs_list):
    """GEOCENTRIC GCRS site offsets (n,3 au) for Earth-fixed ITRS positions at UTC epochs
    (the ITRS->GCRS rotation half of obs_sources.observer_positions_batch — Earth
    orientation only, no ephemeris)."""
    from astropy import units as u
    from astropy.coordinates import ITRS, GCRS, CartesianRepresentation
    T = Time(np.asarray(jds_utc, float), format="jd", scale="utc")
    itrs = ITRS(CartesianRepresentation((np.asarray(itrs_list, float) * u.au).T), obstime=T)
    return itrs.transform_to(GCRS(obstime=T)).cartesian.xyz.to(u.au).value.T


def _station_posvel(jds_utc, itrs_list, obscodes=None, dt_s=10.0):
    """Heliocentric equatorial position (n,3 au) AND velocity (n,3 au/day) of Earth-fixed
    stations at UTC epochs. The EARTH's heliocentric state comes from the SAME DE440
    ephemeris the integration uses (force_assist) — astropy's builtin analytic ephemeris
    (what the optical path's observer_positions_batch uses) is 2-3 km off DE440, which
    is invisible optically (~0.03" at 0.05 au) but is 10-20 us of round-trip delay, the
    dominant error of a first cut of this model (verified against Horizons light-time,
    2026-07). The site's GCRS offset is astropy's ITRS->GCRS rotation (Earth orientation
    only), central-differenced over +-dt_s for the velocity (error ~(omega*dt)^2/6,
    ~1e-8 relative — far below radar Doppler sigmas)."""
    import pyroorb.force_assist as _fa
    jds = np.asarray(jds_utc, float)
    n = len(jds)
    dt = dt_s / SEC_PER_DAY
    all_jds = np.concatenate([jds, jds - dt, jds + dt])
    S = _site_gcrs(all_jds, list(itrs_list) * 3)
    t_tdb = Time(jds, format="jd", scale="utc").tdb.jd
    ephem = _fa.get_ephem()
    epos = np.empty((n, 3)); evel = np.empty((n, 3))
    for i, t in enumerate(t_tdb):
        e = ephem.get_particle("Earth", float(t) - ephem.jd_ref)
        sunp, sunv = _fa._sun_bary(ephem, float(t))
        epos[i] = [e.x - sunp[0], e.y - sunp[1], e.z - sunp[2]]
        evel[i] = [e.vx - sunv[0], e.vy - sunv[1], e.vz - sunv[2]]
    pos = epos + S[:n]
    vel = evel + (S[2 * n:] - S[n:2 * n]) / (2 * dt)
    return pos, vel


def _radial_rate(dp, dv):
    """d|Δr|/dt (find_orb relative_velocity): rate of change of the separation, au/day."""
    dist = np.sqrt((dp * dp).sum(axis=-1))
    return (dp * dv).sum(axis=-1) / dist


def radar_context(radar, obscodes):
    """Epoch-free precompute for a radar obs list: receive-time TDB epochs (the states-
    sweep times), receiver helio pos/vel at receive time, and per-row weights/values.
    The TRANSMITTER's state depends on the model delay, so it is resolved per
    evaluation in radar_model (one batched astropy call per LM evaluation)."""
    jd_utc = np.array([r["jd_utc"] for r in radar])
    t_tdb = Time(jd_utc, format="jd", scale="utc").tdb.jd
    rx_pos, rx_vel = _station_posvel(jd_utc, [r["rx_itrs_au"] for r in radar], obscodes)
    return {"radar": radar, "obscodes": obscodes,
            "jd_utc": jd_utc, "t_tdb": np.asarray(t_tdb, float),
            "rx_pos": rx_pos, "rx_vel": rx_vel,
            "tx_itrs": [r["tx_itrs_au"] for r in radar],
            "is_delay": np.array([r["kind"] == "delay" for r in radar]),
            "value": np.array([r["value"] for r in radar]),
            "sigma": np.array([r["sigma"] for r in radar]),
            "freq_hz": np.array([r["freq_hz"] for r in radar])}


def radar_model(states, rctx):
    """Computed (delay_s, doppler_hz), each (n, npart), for object states (n, npart, 6)
    heliocentric-equatorial au/au-day AT THE RECEIVE TDB EPOCHS rctx['t_tdb'].

    Port of find_orb compute_radar_info. The object's bounce state is Taylor-stepped
    back from the receive-epoch state (linear velocity + two-body solar acceleration;
    the jerk-term error over a <=500 s light time is ~mm, far under the ~15 m of a
    0.1 us delay sigma) rather than re-integrated, exactly the trick the optical
    residual's light-time back-step uses. The down-leg is a 4-pass fixed-point
    iteration; the up-leg linearizes the transmitter about one astropy lookup at the
    down-leg-estimated transmit epoch (curvature over the remaining ~25 ms is ~1e-5 m).
    The transmit epoch uses the NOMINAL particle (0) for all particles: a state
    perturbation moves the transmit epoch by ~us, which moves the station by ~mm."""
    r = states[..., :3]                              # (n, npart, 3)
    v = states[..., 3:]
    rx_p = rctx["rx_pos"][:, None, :]                # (n, 1, 3)
    rx_v = rctx["rx_vel"][:, None, :]
    rnorm3 = np.sqrt((r * r).sum(axis=-1, keepdims=True)) ** 3
    acc = -MU_SUN * r / rnorm3                       # two-body Taylor term (au/day^2)

    # --- down-leg: bounce epoch t_b = t_recv - tau_dn ------------------------------
    tau = np.sqrt(((r - rx_p) ** 2).sum(axis=-1)) / C_AU_DAY      # (n, npart) days
    for _ in range(4):
        t = tau[..., None]
        r_b = r - v * t + 0.5 * acc * t * t
        tau = np.sqrt(((r_b - rx_p) ** 2).sum(axis=-1)) / C_AU_DAY
    t = tau[..., None]
    r_b = r - v * t + 0.5 * acc * t * t              # object at bounce
    v_b = v - acc * t

    # --- up-leg: transmitter at t_x = t_b - tau_up ---------------------------------
    # one astropy lookup at the nominal-particle estimate t_recv - 2*tau_dn, then a
    # linearized fixed-point for the exact epoch
    tau_dn0 = tau[:, 0]                              # nominal particle, days
    jd_x0 = rctx["jd_utc"] - 2.0 * tau_dn0
    tx_p0, tx_v0 = _station_posvel(jd_x0, rctx["tx_itrs"], rctx["obscodes"])
    tau_up = tau.copy()
    for _ in range(3):
        # transmit epoch offset from the lookup epoch, days (per particle)
        dt_x = (2.0 * tau_dn0[:, None] - tau - tau_up)
        tx_p = tx_p0[:, None, :] + tx_v0[:, None, :] * dt_x[..., None]
        tau_up = np.sqrt(((r_b - tx_p) ** 2).sum(axis=-1)) / C_AU_DAY
    dt_x = (2.0 * tau_dn0[:, None] - tau - tau_up)
    tx_p = tx_p0[:, None, :] + tx_v0[:, None, :] * dt_x[..., None]
    tx_v = tx_v0[:, None, :]

    delay_s = (tau + tau_up) * SEC_PER_DAY
    # Shapiro (GR) delay per leg — one modeled term BEYOND find_orb (which is purely
    # geometric): (1+gamma)(GM_sun/c^3) ln((r1+r2+R)/(r1+r2-R)). JPL models it, and the
    # 2005 YU55 oracle run shows it plainly in the tightest delays (sigma 0.5 us rows sit
    # at +0.7..+1.3 us without it, within ~0.3 us with it). The Doppler-rate counterpart
    # is far below the Hz sigmas and is not modeled.
    r_b_n = np.sqrt((r_b * r_b).sum(axis=-1))
    rx_n = np.sqrt((rx_p * rx_p).sum(axis=-1))
    tx_n = np.sqrt((tx_p * tx_p).sum(axis=-1))
    d_dn = np.sqrt(((r_b - rx_p) ** 2).sum(axis=-1))
    d_up = np.sqrt(((r_b - tx_p) ** 2).sum(axis=-1))
    delay_s = delay_s + TG_SHAPIRO * (
        np.log((r_b_n + rx_n + d_dn) / (r_b_n + rx_n - d_dn))
        + np.log((r_b_n + tx_n + d_up) / (r_b_n + tx_n - d_up)))

    # --- Doppler: two-leg special-relativistic factor (find_orb) -------------------
    v1 = _radial_rate(r_b - rx_p, v_b - rx_v) / C_AU_DAY
    v2 = _radial_rate(r_b - tx_p, v_b - tx_v) / C_AU_DAY
    factor = np.sqrt((1.0 - v1) * (1.0 - v2) / ((1.0 + v1) * (1.0 + v2)))
    doppler_hz = (factor - 1.0) * rctx["freq_hz"][:, None]
    return delay_s, doppler_hz


def radar_residuals(states, rctx):
    """Sigma-normalized radar residual (obs - computed)/sigma, (n, npart): each row is
    its obs's own measurement kind (delay rows in delay sigmas, Doppler in Doppler's),
    exactly find_orb get_residual_data's radar branch."""
    delay_s, doppler_hz = radar_model(states, rctx)
    comp = np.where(rctx["is_delay"][:, None], delay_s, doppler_hz)
    return (rctx["value"][:, None] - comp) / rctx["sigma"][:, None]


# ---------------------------------------------------------------------------
# Post-fit refine — fold the radar rows into a final differential correction
# ---------------------------------------------------------------------------
def _optical_rms_norm(fit, state, mask):
    """Normalized rms of the OPTICAL residuals alone for `state`, on the fit's own included set --
    the same currency fit.rms_normalized() reports, so before/after compare like with like."""
    import numpy as _np
    try:
        ev = fit._ctx["ev"]
        return float(ev.rms_norm(_np.asarray(state, float)[:6], _np.asarray(mask, bool)))
    except Exception:
        return float("nan")


def refine_radar(fit, radar, obscodes, verbose=False):
    """Re-solve a converged optical orbit WITH its radar delay/Doppler measurements as
    extra weighted LM rows (find_orb folds them into every full_improvement; we join
    at the end, the refine_asteroids pattern — the optical pipeline's IOD/rejection
    machinery is sky-plane-only by construction). Optical rows keep the fit's
    analytic Jacobian; radar rows get forward differences from ONE 7-particle
    helio_states_multi sweep per evaluation. Returns a NEW FittedOrbit whose
    result/covariance carry the combined rows (so sigmas tighten), with a per-row
    report on fit.radar; or `fit` unchanged (report attached, applied=False) when
    radar is absent, the backend can't do a states sweep, or the joint fit cannot
    represent the radar data (median |z| > RADAR_ABANDON_Z — e.g. a mis-attribution)."""
    from types import SimpleNamespace
    import pyroorb.fit_object as _fo

    if not radar:
        return fit
    ctx = getattr(fit, "_ctx", None)
    if ctx is None:
        return fit
    backend = ctx["backend"]
    if not hasattr(backend, "helio_states_multi"):     # rebound fallback: no states sweep
        return fit
    ev = ctx["ev"]
    ng = getattr(ev, "nongrav", None)
    if ng is not None:
        # A non-gravitational orbit takes the JOINT state+A+radar solve: the
        # optical block below (ev.evaluate) is gravity-only by design, so refining the
        # 6-state alone against an active force model would smear the radar rows -- which
        # is why this used to be report-only, at the measured price of a 6.25x error in
        # 2003 HM's delivered A2 (radar was discarded exactly where it mattered most).
        return _refine_radar_nongrav(fit, radar, obscodes, verbose=verbose)
    rctx = radar_context(radar, obscodes)
    state0 = np.asarray(fit._state, float)
    mask = np.asarray(fit._mask, bool)
    n_r = len(radar)

    h = 1.49e-8 * np.maximum(1.0, np.abs(state0))

    def _sweep(state):
        pset = [np.asarray(state, float)]
        for k in range(6):
            p = np.asarray(state, float).copy()
            p[k] += h[k]
            pset.append(p)
        states = backend.helio_states_multi(ctx["epoch_tdb"], pset, rctx["t_tdb"],
                                            nongrav=ng)
        return radar_residuals(states, rctx)           # (n_r, 7)

    included = np.ones(n_r, bool)

    def _solve(seed):
        cache = {}

        def _compute(state):
            key = state.tobytes()
            hit = cache.get(key)
            if hit is None:
                cache.clear()
                opt = ev.evaluate(state, mask)
                rz = _sweep(state)
                hit = cache[key] = (opt, rz)
            return hit

        def fun(state):
            opt, rz = _compute(state)
            rr = np.where(np.isfinite(rz[:, 0]), rz[:, 0], 1e6)
            return np.concatenate([opt.fun, rr[included]])

        def jac(state):
            opt, rz = _compute(state)
            Jr = (rz[:, 1:] - rz[:, 0:1]) / h[None, :]     # forward diff: d[(o-c)/sig]/d(state)
            Jr = np.where(np.isfinite(Jr), Jr, 0.0)
            return np.vstack([opt.jac, Jr[included]])

        from scipy.optimize import least_squares
        lsq = least_squares(fun, seed, jac=jac, method="lm", x_scale="jac",
                            xtol=1e-10, ftol=1e-10, max_nfev=RADAR_MAX_NFEV)
        return lsq.x, lsq

    z0 = _sweep(state0)[:, 0]
    state, lsq = _solve(state0)
    z = _sweep(state)[:, 0]
    # one conservative rejection round: a grossly-off row (old coarse data, bad pairing)
    # is excluded and the fit re-run once — everything else stays at full weight
    bad = np.abs(z) > RADAR_REJECT_Z
    if bad.any() and (~bad).sum() > 0:
        included = ~bad
        state, lsq = _solve(state0)
        z = _sweep(state)[:, 0]

    z_used = z[included]
    report_rows = [{"kind": r["kind"], "jd_utc": r["jd_utc"], "desig": r["desig"],
                    "rx": r["rx"], "tx": r["tx"], "bp": r["bp"],
                    "value": r["value"], "sigma": r["sigma"],
                    "resid": float((z[i] * r["sigma"])),
                    "z": float(z[i]), "included": bool(included[i])}
                   for i, r in enumerate(radar)]
    if not np.isfinite(z_used).all() or np.median(np.abs(z_used)) > RADAR_ABANDON_Z:
        fit.radar = {"applied": False, "n": n_r, "n_used": 0, "rows": report_rows,
                     "note": "radar irreconcilable with the optical orbit; kept optical fit"}
        if verbose:
            print(f"  radar refine: median |z| {np.median(np.abs(z_used)):.1f} > "
                  f"{RADAR_ABANDON_Z}; kept the optical orbit")
        return fit
    # OPTICAL-SIDE guard, the symmetric partner of RADAR_ABANDON_Z above. That test asks whether
    # the joint fit can represent the RADAR; this one asks whether it still represents the SKY,
    # and a solution has to do both to be worth delivering. Without it a fit that satisfies
    # neither is shipped: 2003 HM's radar sits at median |z| 5.68 -- comfortably inside the 25.0
    # bar -- while its optical residuals go from 0.735" (0.96 sigma, matching the MPC's published
    # elements) to 4.836" (9.34 sigma). Both datasets fit badly, which is the signature of a
    # model that cannot span them, not of a compromise worth making.
    opt_before = float(fit.rms_normalized())
    opt_after = _optical_rms_norm(fit, state, mask)
    if (opt_before > 0 and np.isfinite(opt_after)
            and opt_after > RADAR_MAX_OPTICAL_DEGRADE * opt_before):
        fit.radar = {"applied": False, "n": n_r, "n_used": 0, "rows": report_rows,
                     "optical_rms_norm_before": opt_before, "optical_rms_norm_after": opt_after,
                     "note": (f"radar refit degraded the optical fit {opt_before:.2f} -> "
                              f"{opt_after:.2f} sigma; kept the optical orbit")}
        if verbose:
            print(f"  radar refine: optical rms {opt_before:.2f} -> {opt_after:.2f} sigma "
                  f"(> {RADAR_MAX_OPTICAL_DEGRADE}x); kept the optical orbit", flush=True)
        return fit
    # deliver: rebuild the orbit at the radar-corrected state with the COMBINED
    # rows as the covariance source
    ctx2 = dict(ctx)
    ctx2["meta"] = {**(ctx.get("meta") or {}), "radar_used": int(included.sum())}
    result = SimpleNamespace(x=state, fun=lsq.fun, jac=lsq.jac, success=True,
                             status=1, nfev=lsq.nfev,
                             cost=0.5 * float(np.dot(lsq.fun, lsq.fun)))
    fit2 = _fo._build_orbit(ctx2, state, mask, result, fit.history)
    fit2.radar = {"applied": True, "n": n_r, "n_used": int(included.sum()),
                  "rows": report_rows,
                  "pre_median_z": float(np.median(np.abs(z0[np.isfinite(z0)])))
                  if np.isfinite(z0).any() else None,
                  "post_median_z": float(np.median(np.abs(z_used)))}
    if getattr(fit, "nongrav_model", None):
        fit2.nongrav_model = fit.nongrav_model
        fit2.nongrav_params = fit.nongrav_params
        fit2.nongrav_sigmas = fit.nongrav_sigmas
    if verbose:
        print(f"  radar refine: {int(included.sum())}/{n_r} rows joined the fit "
              f"(median |z| {np.median(np.abs(z0[np.isfinite(z0)])):.2f} -> "
              f"{np.median(np.abs(z_used)):.2f}); optical rms "
              f"{fit.rms_residual():.3f}\" -> {fit2.rms_residual():.3f}\"")
    return fit2


def _refine_radar_nongrav(fit, radar, obscodes, verbose=False):
    """JOINT state + A + radar differential correction for a delivered NON-GRAVITATIONAL
    orbit (the "7-D" solve, generally 6+k).

    The parameter vector is refine_nongrav's own: [6-state, A's in the model's scale
    units], seeded from the delivered fit's `_state`. The optical rows are the SAME
    weighted (RA,Dec) residuals/Jacobian refine_nongrav fits (`_make_ng_residual`, so the
    angle-wrap, weighting and A-scaling cannot drift from the optical-only solve), and the
    radar rows ride ONE (7+k)-particle `helio_states_multi` sweep per LM evaluation: the
    nominal, 6 state-perturbed copies, and k A-PERTURBED copies -- each particle carrying
    its own A1/A2/A3 (the per-particle-params support added for exactly this) -- so
    d(delay,Doppler)/dA costs the same single integration the state columns do.

    Why the A columns matter: an 11-year radar baseline with sub-millisecond delays is
    precisely what pins a secular along-track drift, so on a radar object the OPTICAL-only
    A2 is a different (and much weaker) estimator than JPL's radar-anchored one -- 2003 HM
    delivered -1.87e-14 against JPL's -1.168e-13 +/- 1.9e-15 on the same arc. Solving the
    state WITH A frozen (what the gravity-path solve would do here) reproduces the failure
    this function's guards were originally calibrated on: neither dataset fits.

    Same conservative guards as the gravity path, same order: one RADAR_REJECT_Z rejection
    round, RADAR_ABANDON_Z on the surviving rows, and the optical-degradation gate --
    where "optical rms" is the joint solution's own weighted optical residuals, the same
    per-axis normalized currency fit.rms_normalized() reports. Delivery goes through
    fit_object._deliver_nongrav_orbit (the one deliverer refine_nongrav uses), with
    `nongrav_profile` OFF: the on-display profile-likelihood sigma re-fits WITHOUT the
    radar rows, so for a radar-anchored coefficient it would contradict the covariance
    this fit ships -- the linear sigma from the combined rows is the honest one here.
    """
    from types import SimpleNamespace
    import pyroorb.fit_object as _fo
    import pyroorb.nongrav as ngm

    ctx = fit._ctx
    backend = ctx["backend"]
    ev = ctx["ev"]
    model = getattr(fit, "nongrav_model", None)
    if not model or model == "none":
        # ev.nongrav set with no model on the orbit -- nothing coherent to solve jointly
        fit.radar = {"applied": False, "n": len(radar), "n_used": 0, "rows": [],
                     "note": "nongrav force active but no model on the orbit; radar report-only"}
        return fit
    spec = ngm.model(model)
    free_idx = ngm.free_indices(model)
    k = len(free_idx)
    scale, gr = spec["scale"], spec["gr"]
    t0 = ctx["epoch_tdb"]
    mask = np.asarray(fit._mask, bool)
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]
    Rs = [ev.Robs[i] for i in idx]
    ra, dec, cosd = ev.ra[idx], ev.dec[idx], ev.cosd[idx]
    sra, sdec = ev.fit_sig_ra[idx], ev.fit_sig_dec[idx]
    m = len(idx)
    hA_phys = np.full(k, 1e-3) * scale
    obs_resid, obs_jac, _ = _fo._make_ng_residual(
        backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr, hA_phys,
        a_of_p=lambda p: ngm.full_A(model, p[6:]),
        free_cols=free_idx, col_scale=np.full(k, scale), m=m)

    rctx = radar_context(radar, obscodes)
    n_r = len(radar)
    p0 = np.asarray(fit._state, float)
    if p0.shape[0] != 6 + k:                     # arity guard (the 6+k-vs-6 hazard class)
        p0 = _fo._lift_nongrav_state(fit, model)
    h6 = 1.49e-8 * np.maximum(1.0, np.abs(p0[:6]))
    hA_s = np.full(k, 1e-3)                      # forward step in the SCALED A the solver carries

    def _sweep(p):
        """Sigma-normalized radar residuals (n_r, 7+k): nominal + 6 state-perturbed +
        k A-perturbed particles, one integration."""
        A_phys = np.asarray(ngm.full_A(model, p[6:]), float)
        pset = [np.asarray(p[:6], float)]
        for j in range(6):
            q = pset[0].copy()
            q[j] += h6[j]
            pset.append(q)
        As = np.tile(A_phys, (7 + k, 1))
        for j, ax in enumerate(free_idx):
            pset.append(pset[0].copy())
            As[7 + j, ax] += hA_s[j] * scale     # scaled step, physical perturbation
        states = backend.helio_states_multi(t0, pset, rctx["t_tdb"], nongrav=(As, gr))
        return radar_residuals(states, rctx)

    included = np.ones(n_r, bool)
    steps = np.concatenate([h6, hA_s])

    def _solve(seed):
        cache = {}

        def _compute(p):
            key = p.tobytes()
            hit = cache.get(key)
            if hit is None:
                cache.clear()
                hit = cache[key] = (obs_resid(p), obs_jac(p), _sweep(p))
            return hit

        def fun(p):
            ro, _, rz = _compute(p)
            rr = np.where(np.isfinite(rz[:, 0]), rz[:, 0], 1e6)
            return np.concatenate([ro, rr[included]])

        def jac(p):
            _, Jo, rz = _compute(p)
            Jr = (rz[:, 1:] - rz[:, 0:1]) / steps[None, :]
            Jr = np.where(np.isfinite(Jr), Jr, 0.0)
            return np.vstack([Jo, Jr[included]])

        from scipy.optimize import least_squares
        lsq = least_squares(fun, seed, jac=jac, method="lm", x_scale="jac",
                            xtol=1e-10, ftol=1e-10, max_nfev=RADAR_MAX_NFEV)
        return lsq.x, lsq

    z0 = _sweep(p0)[:, 0]
    p, lsq = _solve(p0)
    z = _sweep(p)[:, 0]
    bad = np.abs(z) > RADAR_REJECT_Z
    if bad.any() and (~bad).sum() > 0:
        included = ~bad
        p, lsq = _solve(p0)
        z = _sweep(p)[:, 0]

    z_used = z[included]
    report_rows = [{"kind": r["kind"], "jd_utc": r["jd_utc"], "desig": r["desig"],
                    "rx": r["rx"], "tx": r["tx"], "bp": r["bp"],
                    "value": r["value"], "sigma": r["sigma"],
                    "resid": float(z[i] * r["sigma"]),
                    "z": float(z[i]), "included": bool(included[i])}
                   for i, r in enumerate(radar)]
    if not np.isfinite(z_used).all() or np.median(np.abs(z_used)) > RADAR_ABANDON_Z:
        fit.radar = {"applied": False, "n": n_r, "n_used": 0, "rows": report_rows,
                     "note": "radar irreconcilable with the joint nongrav orbit; kept optical fit"}
        if verbose:
            print(f"  radar refine (nongrav joint): median |z| "
                  f"{np.median(np.abs(z_used)):.1f} > {RADAR_ABANDON_Z}; kept the optical orbit")
        return fit
    # OPTICAL-side gate, same 1.5x currency as the gravity path: the joint solution's own
    # weighted optical residuals, per-axis normalized -- against the delivered fit's
    # rms_normalized(), which is the same statistic on the optical-only solution.
    opt_before = float(fit.rms_normalized())
    ro = obs_resid(p)
    opt_after = float(np.sqrt(np.mean(ro * ro))) if ro.size else float("nan")
    if (opt_before > 0 and np.isfinite(opt_after)
            and opt_after > RADAR_MAX_OPTICAL_DEGRADE * opt_before):
        fit.radar = {"applied": False, "n": n_r, "n_used": 0, "rows": report_rows,
                     "optical_rms_norm_before": opt_before, "optical_rms_norm_after": opt_after,
                     "note": (f"joint radar refit degraded the optical fit {opt_before:.2f} -> "
                              f"{opt_after:.2f} sigma; kept the optical orbit")}
        if verbose:
            print(f"  radar refine (nongrav joint): optical rms {opt_before:.2f} -> "
                  f"{opt_after:.2f} sigma (> {RADAR_MAX_OPTICAL_DEGRADE}x); kept the optical orbit",
                  flush=True)
        return fit

    params_before = dict(getattr(fit, "nongrav_params", None) or {})
    ctx2 = dict(ctx)
    ctx2["meta"] = {**(ctx.get("meta") or {}), "radar_used": int(included.sum())}
    result = SimpleNamespace(x=p, fun=lsq.fun, jac=lsq.jac, success=True,
                             status=1, nfev=lsq.nfev,
                             cost=0.5 * float(np.dot(lsq.fun, lsq.fun)))
    ng_diag = dict(getattr(fit, "nongrav_diag", None) or {})
    ng_diag["radar_joint"] = int(included.sum())
    fit2 = _fo._deliver_nongrav_orbit(ctx2, fit, model, result, mask,
                                      ng_diag=ng_diag, profile_sigmas=False)
    # the discovery record describes how the model was CHOSEN -- the joint refit changes the
    # value, not that history, and losing it here silently stripped `nongrav_discovery` (and
    # with it the #54 asteroids/recheck records) from every radar-refined nongrav orbit
    fit2.nongrav_discovery = getattr(fit, "nongrav_discovery", None)
    fit2.radar = {"applied": True, "n": n_r, "n_used": int(included.sum()),
                  "rows": report_rows, "joint_nongrav": model,
                  "nongrav_before": params_before,
                  "nongrav_after": dict(fit2.nongrav_params or {}),
                  "pre_median_z": float(np.median(np.abs(z0[np.isfinite(z0)])))
                  if np.isfinite(z0).any() else None,
                  "post_median_z": float(np.median(np.abs(z_used)))}
    if verbose:
        moved = ", ".join(
            f"{nm} {params_before.get(nm, 0.0):.3e} -> {v:.3e}"
            for nm, v in (fit2.nongrav_params or {}).items())
        print(f"  radar refine (nongrav joint, {model}): {int(included.sum())}/{n_r} rows "
              f"joined (median |z| {np.median(np.abs(z0[np.isfinite(z0)])):.2f} -> "
              f"{np.median(np.abs(z_used)):.2f}); {moved}; optical rms "
              f"{fit.rms_residual():.3f}\" -> {fit2.rms_residual():.3f}\"")
    return fit2
