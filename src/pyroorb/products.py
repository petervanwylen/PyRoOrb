"""What a fitted orbit predicts: its state at other times, osculating elements of any state,
close approaches to the planets and the Moon, and radar delay/Doppler -- each with a LINEAR
uncertainty from the fit's covariance (state, and any non-gravitational coefficients).

The public entry points are re-exported by ``pyroorb/__init__.py`` (``propagate``,
``osculating_elements``, ``close_approaches``, ``radar_residuals``, ``radar_prediction``) and
documented in docs/api.rst; this module is their implementation.

Two rules hold for everything here:

* **The fit's own force model.** Each product integrates in the field the orbit was fitted in:
  the fit's own backend (`obs_sources._force_backend(fit)`), which carries its asteroid
  perturbers on or off (forcemodel.PinnedForce) and a self-perturber's exclusion, and its
  non-gravitational acceleration (`ephemerides._nongrav_pair`). The
  process-wide perturber switch at CALL time does not change a product, so a fit made inside
  ``pyroorb.asteroid_perturbers()`` propagates with the asteroids after the block has closed.
* **Linear uncertainty from one integration.** The nominal state rides with copies displaced
  by +-1e-2 sigma along each fitted parameter (central differences): the 6 state axes, and for a
  non-gravitational fit each free A coefficient too (a copy carrying its own A), so the
  derivatives at every output time cost the same single integration as the nominal (the trick
  radar.py and the analytic sky Jacobian use). They are projected through the fit's full
  parameter covariance, so an uncertain acceleration -- and its correlation with the state --
  widens the prediction the longer it acts, as the covariance ensemble
  (``sample_orbits(method="covariance")``) does. Linear propagation is the right answer for a well-determined
  orbit and an under-statement for a poorly determined one, where the clone ensemble is the
  honest tool.

The radar model is radar.py's port of find_orb's compute_radar_info, called with synthetic
records for a prediction.
"""
import math

import numpy as np

AU_KM = 149597870.7
DAY_S = 86400.0
C_KM_S = 299792.458

# Coarse scan step (days) per body when the caller gives none: a planetocentric distance is
# convex through an encounter, so any sampling finds a local minimum of the sampled series at
# the grid point nearest the true one; the step only has to keep that grid point inside
# `max_distance` -- 0.5 d moves an object at 30 km/s by 0.009 au, well under Earth's 0.05 au.
_DEFAULT_STEP = {"earth": 0.5, "moon": 0.5, "mercury": 1.0, "venus": 1.0, "mars": 1.0}
_PLANETS = ("mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune",
            "pluto")
# The 16 massive asteroids of JPL's sb441-n16 kernel (the perturbers asteroid_perturbers adds);
# their positions need that kernel installed.
ASTEROIDS = ("ceres", "pallas", "juno", "vesta", "iris", "hygiea", "eunomia", "psyche",
             "euphrosyne", "europa", "cybele", "sylvia", "thisbe", "camilla", "davida",
             "interamnia")
_BODIES = _PLANETS + ASTEROIDS
BODY_DV_DT = 0.01        # days: the central-difference step for a body whose kernel has no velocity
# The relative displacement of the state copies, in units of each axis's 1-sigma. Floor: a
# fit with a vanishing sigma (dof 0) still gets a finite, round-off-safe step.
_STEP_SIGMAS = 1e-2
_STEP_FLOOR = 1e-12


def tdb(times):
    """TDB Julian dates from times given as the public API takes them: a number or numbers
    (UTC Julian dates), ISO string(s) (UTC), or an astropy ``Time`` of any scale."""
    from astropy.time import Time
    if isinstance(times, Time):
        return np.atleast_1d(np.asarray(times.tdb.jd, float))
    arr = np.atleast_1d(np.asarray(times, dtype=object))
    if all(isinstance(t, (int, float, np.integer, np.floating)) for t in arr):
        return np.atleast_1d(Time(arr.astype(float), format="jd", scale="utc").tdb.jd)
    return np.atleast_1d(Time(list(arr), scale="utc").tdb.jd)


def _utc_jd(jd_tdb):
    from astropy.time import Time
    return np.asarray(Time(np.asarray(jd_tdb, float), format="jd", scale="tdb").utc.jd, float)


def _utc_iso(jd_tdb):
    from astropy.time import Time
    return Time(float(jd_tdb), format="jd", scale="tdb").utc.iso[:19]


def fit_asteroids(fit):
    """Did this fit integrate with the asteroid perturbers? Read off the fit's own backend
    (forcemodel.PinnedForce, set when the fit was made or its force model swapped), never off
    the process-wide switch or the display label; `fit.asteroids` for a fit without one."""
    from pyroorb.forcemodel import pinned_asteroids
    pinned = pinned_asteroids((getattr(fit, "_ctx", None) or {}).get("backend"))
    return bool(getattr(fit, "asteroids", False)) if pinned is None else pinned


def _backend(fit):
    from pyroorb.obs_sources import _force_backend
    backend, _ = _force_backend(fit)
    if not hasattr(backend, "helio_states_multi"):
        raise RuntimeError("propagation needs ASSIST and the DE440 ephemeris "
                           "(run pyroorb-fetch-data)")
    return backend


def _nongrav(fit, nongrav):
    if not nongrav:
        return None
    from pyroorb.ephemerides import _nongrav_pair
    return _nongrav_pair(fit)


def _sweep(fit, times_tdb, nongrav, stm):
    """ONE integration of the nominal orbit, plus -- with `stm` -- a +h / -h copy along
    every parameter the linear uncertainty differentiates (ephemerides.linear_parameters):
    the 6 state components, then, for a non-gravitational fit propagated with its force,
    each free A coefficient (the copy keeps the nominal state and carries its own A, the
    per-particle acceleration helio_states_multi takes).

    Returns ``(raw, h, cov)``: `raw` the ``(n, 1 + 2p, 6)`` states in the caller's time
    order (column 0 the nominal, columns 1+2k / 2+2k the +h_k / -h_k copies), `h` the ``(p,)``
    steps and `cov` the ``(p, p)`` covariance over the same parameters (None without `stm`)."""
    from pyroorb.ephemerides import linear_parameters
    t = np.asarray(times_tdb, float).ravel()
    order = np.argsort(t)
    s0 = np.asarray(fit.state(), float)[:6]
    pair, free, cov = linear_parameters(fit, nongrav) if stm else (_nongrav(fit, nongrav), [], None)
    parts, ng, h = [s0], pair, None
    if stm:
        p = 6 + len(free)
        sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        h = _STEP_SIGMAS * sig
        h[:6] = np.maximum(h[:6], _STEP_FLOOR * np.maximum(1.0, np.abs(s0)))
        if pair is not None:
            A0 = np.asarray(pair[0], float)
            a = np.abs(A0[free])
            # an A axis with no usable sigma (zero degrees of freedom) still gets a finite step
            h[6:] = np.where(h[6:] > 0, h[6:], np.where(a > 0, 1e-3 * a, 1e-20))
            As = [A0]
        for k in range(p):
            for sgn in (1.0, -1.0):
                s, A = s0.copy(), (None if pair is None else A0.copy())
                if k < 6:
                    s[k] += sgn * h[k]
                else:
                    A[free[k - 6]] += sgn * h[k]
                parts.append(s)
                if pair is not None:
                    As.append(A)
        if pair is not None:
            ng = (np.array(As), pair[1])                      # (1 + 2p, 3): per-particle A
    out = _backend(fit).helio_states_multi(float(fit.epoch_tdb), parts, t[order], nongrav=ng,
                                           asteroids=fit_asteroids(fit))
    out = np.asarray(out, float)
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return out[back], h, cov


def _derivatives(values, h):
    """Central differences of a swept quantity: `values` ``(n, 1 + 2p, ...)`` laid out as
    _sweep's columns -> ``(n, ..., p)``, d value / d parameter."""
    v = np.asarray(values, float)
    d = [(v[:, 1 + 2 * k] - v[:, 2 + 2 * k]) / (2.0 * h[k]) for k in range(len(h))]
    return np.stack(d, axis=-1)


def states(fit, times_tdb, *, nongrav=True, stm=False):
    """Heliocentric-equatorial states of the fit at `times_tdb` (any order).

    Returns ``(n, 6)``; with ``stm`` also ``(n, 6, p)`` d state(t) / d parameter (the
    state-transition matrix, extended by a column per free A coefficient of a
    non-gravitational fit) and the ``(p, p)`` covariance of those parameters, by central
    differences inside the same integration."""
    raw, h, cov = _sweep(fit, times_tdb, nongrav, stm)
    if not stm:
        return raw[:, 0, :]
    return raw[:, 0, :], _derivatives(raw, h), cov


def propagate(fit, times, *, covariance=False, nongrav=True):
    """See pyroorb.propagate."""
    t = tdb(times)
    if not covariance:
        return states(fit, t, nongrav=nongrav)
    s, phi, c0 = states(fit, t, nongrav=nongrav, stm=True)
    cov = np.einsum("nij,jk,nlk->nil", phi, c0, phi)
    return s, 0.5 * (cov + np.transpose(cov, (0, 2, 1)))


def osculating_elements(state, epoch, *, center="sun"):
    """See pyroorb.osculating_elements."""
    from pyroorb.orbital_elements import ELEM_NAMES, element_vector
    t = float(tdb(epoch)[0])
    s = np.asarray(state, float).ravel()[:6]
    if str(center).lower() == "sun":
        return dict(zip(ELEM_NAMES, [float(x) for x in element_vector(s, t)]))
    from pyroorb import planetocentric as pc
    d = pc.planetocentric_elements(s, t, center)
    return {k: d[k] for k in ELEM_NAMES}


def body_states(body, times_tdb):
    """Heliocentric-equatorial states (n, 6) of a DE440 body (the planets are their system
    barycentres beyond Mars; the Moon and Earth are the bodies themselves), or of one of the 16
    asteroid perturbers from JPL's sb441-n16 kernel."""
    import pyroorb.force_assist as fa
    key = str(body).strip().lower()
    if key in ASTEROIDS and fa.asteroid_kernel_path() is None:
        raise FileNotFoundError(f"{body!r} needs the asteroid perturber kernel; "
                                "run pyroorb-fetch-data --asteroids")
    eph = fa.get_ephem(key in ASTEROIDS)
    body = key

    def pos(t):
        p = eph.get_particle(body, float(t) - eph.jd_ref)
        sp, _ = fa._sun_bary(eph, float(t))
        return np.array([p.x - sp[0], p.y - sp[1], p.z - sp[2]])

    out = np.empty((len(times_tdb), 6))
    for i, t in enumerate(np.asarray(times_tdb, float)):
        p = eph.get_particle(body, float(t) - eph.jd_ref)
        sp, sv = fa._sun_bary(eph, float(t))
        out[i] = [p.x - sp[0], p.y - sp[1], p.z - sp[2], p.vx - sv[0], p.vy - sv[1], p.vz - sv[2]]
        if not np.all(np.isfinite(out[i, 3:])):
            # The asteroid kernel gives positions only (its velocities read NaN): a central
            # difference over +-0.01 d is good to ~1e-9 of an asteroid's ~0.01 au/day speed.
            out[i, 3:] = (pos(t + BODY_DV_DT) - pos(t - BODY_DV_DT)) / (2.0 * BODY_DV_DT)
    return out


def _resolve_body(body, allow_sun=False):
    key = str(body).strip().lower()
    if key not in _BODIES and not (allow_sun and key == "sun"):
        raise ValueError(f"unknown body {body!r}; choose from {', '.join(_BODIES)}")
    return key


def public_body_states(body, times):
    """See pyroorb.body_states."""
    key = _resolve_body(body, allow_sun=True)
    t = tdb(times)
    if key == "sun":
        return np.zeros((len(t), 6))
    return body_states(key, t)


def close_approaches(fit, start, stop, *, body="earth", max_distance=0.05, step=None,
                     nongrav=True, uncertainty=True):
    """See pyroorb.close_approaches."""
    key = _resolve_body(body)
    t0, t1 = float(tdb(start)[0]), float(tdb(stop)[0])
    if not t1 > t0:
        raise ValueError("stop must be after start")
    step = float(step or _DEFAULT_STEP.get(key, 1.0 if key in ASTEROIDS else 2.0))
    grid = np.arange(t0, t1 + 0.5 * step, step)
    rel = states(fit, grid, nongrav=nongrav)[:, :3] - body_states(key, grid)[:, :3]
    d = np.linalg.norm(rel, axis=1)
    cands = [i for i in range(len(grid))
             if d[i] < max_distance and (i == 0 or d[i] <= d[i - 1])
             and (i == len(grid) - 1 or d[i] <= d[i + 1])]
    out = []
    for i in cands:
        lo, hi = max(grid[i] - step, t0), min(grid[i] + step, t1)
        fine = np.arange(lo, hi + 1e-9, 1.0 / 1440.0)            # one-minute samples
        if uncertainty:
            s, phi, c0 = states(fit, fine, nongrav=nongrav, stm=True)
        else:
            s, phi = states(fit, fine, nongrav=nongrav), None
        b = body_states(key, fine)
        r, v = s[:, :3] - b[:, :3], s[:, 3:] - b[:, 3:]
        j = int(np.argmin(np.linalg.norm(r, axis=1)))
        rj, vj = r[j], v[j]
        vv = float(vj @ vj)
        dt = -float(rj @ vj) / vv if vv > 0 else 0.0         # linear relative motion
        dt = max(-1.0 / 1440.0, min(1.0 / 1440.0, dt))
        rc = rj + vj * dt
        dist = float(np.linalg.norm(rc))
        edge = (fine[j] + dt <= t0 + 1e-9) or (fine[j] + dt >= t1 - 1e-9)
        row = {"body": key, "jd_tdb": float(fine[j] + dt),
               "jd_utc": float(_utc_jd([fine[j] + dt])[0]), "date_iso": _utc_iso(fine[j] + dt),
               "distance_au": dist, "v_rel_km_s": math.sqrt(vv) * AU_KM / DAY_S,
               "at_window_edge": bool(edge),
               "sigma_distance_au": None, "sigma_time_days": None}
        if phi is not None:
            P = phi[j]                                           # (6, p) at the nearest sample
            dr, dv = P[:3, :], P[3:, :]                          # d r / d p, d v / d p
            g_d = (rc / dist) @ dr                               # envelope: dd/dt = 0 at minimum
            g_t = -((vj @ dr) + (rj @ dv)) / vv
            row["sigma_distance_au"] = float(math.sqrt(max(0.0, g_d @ c0 @ g_d)))
            row["sigma_time_days"] = float(math.sqrt(max(0.0, g_t @ c0 @ g_t)))
        out.append(row)
    # two coarse minima can refine onto one encounter (a flat bottom); keep the closer
    merged = []
    for row in sorted(out, key=lambda r: r["jd_tdb"]):
        if merged and abs(row["jd_tdb"] - merged[-1]["jd_tdb"]) < step:
            if row["distance_au"] < merged[-1]["distance_au"]:
                merged[-1] = row
            continue
        merged.append(row)
    return merged


# JPL's radar-station ids (what the radar records carry in "tx"/"rx") -> the MPC code of the
# same antenna (radar.STATION_NAMES pairs them), so a prediction can name a station either way.
_JPL_TO_MPC = {"-1": "251", "-2": "254", "-9": "256", "-13": "252", "-14": "253",
               "-25": "257", "-38": "255"}


def _radar_context(rows, obscodes):
    from pyroorb import radar as _radar
    return _radar.radar_context(rows, obscodes)


def _radar_model(fit, rctx, *, nongrav=True, stm=False):
    from pyroorb import radar as _radar
    t = rctx["t_tdb"]
    raw, h, cov = _sweep(fit, t, nongrav, stm)
    dly, dop = _radar.radar_model(raw, rctx)                     # (n, 1 + 2p) each
    if not stm:
        return dly[:, 0], dop[:, 0], None, None, None
    return dly[:, 0], dop[:, 0], _derivatives(dly, h), _derivatives(dop, h), cov


def radar_residuals(fit, radar, obscodes, *, nongrav=True):
    """See pyroorb.radar_residuals."""
    rows = list(radar)
    if not rows:
        return []
    rctx = _radar_context(rows, obscodes)
    dly, dop, _, _, _ = _radar_model(fit, rctx, nongrav=nongrav)
    out = []
    for r, cd, cf in zip(rows, dly, dop):
        is_delay = r["kind"] == "delay"
        comp = float(cd if is_delay else cf)
        oc = float(r["value"]) - comp
        row = {"jd_utc": float(r["jd_utc"]), "kind": r["kind"], "tx": r.get("tx"),
               "rx": r.get("rx"), "observed": float(r["value"]), "computed": comp,
               "o_c": oc, "sigma": float(r["sigma"]), "z": oc / float(r["sigma"])}
        if is_delay:
            row["o_c_km"] = oc * C_KM_S / 2.0                    # round trip -> one-way range
        else:
            row["o_c_mm_s"] = oc * C_KM_S * 1e6 / (2.0 * float(r["freq_hz"]))
        out.append(row)
    return out


def radar_prediction(fit, times, station, obscodes, *, receiver=None, freq_hz=8560e6,
                     nongrav=True, uncertainty=True):
    """See pyroorb.radar_prediction."""
    from astropy.time import Time
    from pyroorb import radar as _radar
    t_tdb = tdb(times)
    jd_utc = np.asarray(Time(t_tdb, format="jd", scale="tdb").utc.jd, float)
    station = _JPL_TO_MPC.get(str(station), str(station))
    rx = _JPL_TO_MPC.get(str(receiver), str(receiver)) if receiver else station
    tx_itrs = _radar._station_itrs_from_code(str(station), obscodes)
    rx_itrs = _radar._station_itrs_from_code(rx, obscodes)
    if tx_itrs is None or rx_itrs is None:
        raise ValueError(f"no Earth-fixed position for station {station!r} / {rx!r}")
    rows = [{"jd_utc": float(j), "kind": "delay", "value": 0.0, "sigma": 1.0,
             "freq_hz": float(freq_hz), "tx": str(station), "rx": rx,
             "tx_itrs_au": tx_itrs, "rx_itrs_au": rx_itrs} for j in jd_utc]
    rctx = _radar_context(rows, obscodes)
    dly, dop, g_dly, g_dop, c0 = _radar_model(fit, rctx, nongrav=nongrav, stm=uncertainty)
    out = []
    for k, j in enumerate(jd_utc):
        row = {"jd_utc": float(j), "date_iso": Time(j, format="jd", scale="utc").iso[:19],
               "delay_s": float(dly[k]), "doppler_hz": float(dop[k]),
               "range_au": float(dly[k]) * C_KM_S / 2.0 / AU_KM,
               "sigma_delay_s": None, "sigma_doppler_hz": None}
        if g_dly is not None:
            row["sigma_delay_s"] = float(math.sqrt(max(0.0, g_dly[k] @ c0 @ g_dly[k])))
            row["sigma_doppler_hz"] = float(math.sqrt(max(0.0, g_dop[k] @ c0 @ g_dop[k])))
        out.append(row)
    return out
