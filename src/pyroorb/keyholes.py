"""Keyholes / resonant returns -- PHASE 3 of impact monitoring (LOCATOR + PROBABILITY).

A keyhole is a tiny region on the b-plane of a close ("deflecting") encounter such that an
asteroid passing through it is deflected onto a RESONANT RETURN -- a second, possibly
impacting, encounter an integer number of years later (the textbook case is Apophis: the
2029 encounter has a small keyhole leading to a 2036 return). The mechanism (Valsecchi,
Milani, Gronchi & Chesley 2003, A&A 408, 1179; logic openly ported, no source copied --
see PROVENANCE.md):

  * the deflecting encounter rotates the geocentric velocity, changing the heliocentric
    semimajor axis a' (hence the orbital period);
  * a return to the encounter recurs when k asteroid revolutions == h Earth years, i.e. the
    post-encounter a' hits the resonant value a'_resonant = (h/k)^(2/3);
  * the asteroid actually HITS on the return only for a narrow sub-interval of the b-plane
    (the keyhole), because the second-encounter miss distance is hugely amplified by the
    first encounter -- a metre at the keyhole maps to thousands of km a few years later.

This module both LOCATES keyholes (which resonance, what return year, where on the LOV /
b-plane, how wide) and estimates the impact PROBABILITY through each (the Gaussian mass of
the LOV interval that threads it). It is the layer the 1-D Phase-2 method in ``impact.py``
explicitly flags it cannot do: an impact hiding *behind* a deep prior encounter.

Scope (deliberately bounded -- "locator + probability, not full robustness"): we handle ONE
deflecting encounter feeding resonant returns. We do NOT chase cascades (return-of-a-return),
multiple deflectors, or the LOV folding that forces a full Monte Carlo (Sentry-II's regime).
Like the rest of impact monitoring this is an opt-in leaf product on a converged orbit -- it
never touches the fit, so there is zero regression-suite risk. For an operational, century-
scale keyhole search defer to JPL Sentry / ESA NEODyS.
"""
import math

import numpy as np

from pyroorb import impact
from pyroorb.bplane import MU_EARTH, R_EARTH_KM

MU_SUN = 0.01720209895 ** 2          # au^3/day^2 (Gaussian k^2)
YEAR_DAYS = 365.25
AU_KM = impact.AU_KM


# ===========================================================================
# Analytic b-plane geometry of a resonant return (Valsecchi 2003, Sect. 3)
# ===========================================================================

def resonant_circle(u_inf_km_s, theta_rad, h, k, v_planet_km_s, a_planet_km=AU_KM,
                    mu=MU_EARTH):
    """The b-plane circle of points leading to the ``h/k`` resonant return (h Earth years ==
    k asteroid revolutions, so the post-encounter semimajor axis is a'_resonant = (h/k)^(2/3) in
    units of the planet's). Returns ``(zeta_center_km, radius_km)`` -- the circle is centred
    on the zeta (timing) axis at xi = 0 -- or ``None`` if the resonance is unreachable at this
    geometry. ``theta_rad`` is the angle between the incoming asymptote U and the planet's
    heliocentric velocity; ``u_inf`` and ``v_planet`` set the Opik units.

    This is the analytic LOCATOR primitive; the numerical engine below cross-checks against it.
    """
    U = u_inf_km_s / v_planet_km_s                                # Opik units (planet speed = 1)
    a_resonant = (float(h) / float(k)) ** (2.0 / 3.0)             # post-encounter a' / a_planet
    cos_tp = (1.0 - U * U - 1.0 / a_resonant) / (2.0 * U)         # required post-encounter cos(theta')
    if not -1.0 <= cos_tp <= 1.0:
        return None                                              # resonance not reachable at this U
    cos_t = math.cos(theta_rad)
    denom = cos_tp - cos_t
    if abs(denom) < 1e-12:
        return None
    c = (mu / (v_planet_km_s ** 2)) / a_planet_km                # focusing length (Opik units ~ m_earth)
    sin_t = math.sin(theta_rad)
    sin_tp = math.sqrt(max(0.0, 1.0 - cos_tp * cos_tp))
    zeta_c = c * sin_t / denom
    radius = abs(c * sin_tp / denom)
    return zeta_c * a_planet_km, radius * a_planet_km            # -> km


# ===========================================================================
# Numerical engine: post-encounter a'(LOV), keyhole location, return probability
# ===========================================================================

def _helio_a(epoch_tdb, state, t_tdb, backend, base_sim):
    """Heliocentric osculating semimajor axis (au) of a clone at TDB time ``t_tdb``, from the
    integrated position central-differenced for velocity. Frame-invariant, so the backend's
    equatorial output is fine."""
    dt = 5e-3
    ts = np.asarray([t_tdb - dt, t_tdb, t_tdb + dt], float)
    P = np.asarray(backend.helio_positions_batch(base_sim, epoch_tdb, np.asarray(state, float), ts))
    r = P[1]
    v = (P[2] - P[0]) / (2.0 * dt)                                # au/day
    rmag = math.sqrt(float(r @ r))
    v2 = float(v @ v)
    energy = 2.0 / rmag - v2 / MU_SUN
    if abs(energy) < 1e-12:
        return math.inf
    return 1.0 / energy


def _utc_to_tdb(jd_utc):
    from astropy.time import Time
    return float(Time(jd_utc, format="jd", scale="utc").tdb.jd)


def _deflecting_encounter(fit, t_start, window_days, backend, base_sim, u, sigma, refit_fn,
                          n_sigma, max_perigee_km):
    """Pick the deepest Earth close approach in the window as the deflecting encounter (the one
    that resets a'). Returns the close-approach record or None if nothing comes deep enough to
    open keyholes."""
    cas = impact.find_close_approaches(fit, t_start, window_days, backend, base_sim,
                                       u, sigma, refit_fn, n_sigma=n_sigma)
    if not cas:
        return None
    deepest = min(cas, key=lambda c: c["env_min_km"])
    if deepest["env_min_km"] > max_perigee_km:
        return None
    return deepest


def _encounter_geometry(fit, t1_utc, backend, base_sim, u, sigma, refit_fn, n_sigma, n_probe):
    """b-plane crossing of the clone that comes DEEPEST at the deflecting encounter (~t1), used
    for the Opik geometry (U-infinity, theta). The nominal orbit can miss the encounter for an
    uncertain object, so we scan the LOV and keep the closest approacher. Returns the
    bplane.bplane_crossing dict (+ t_ca) or None."""
    epoch = fit.epoch_tdb
    state0 = np.asarray(fit._state, float)
    best = None
    for k in np.linspace(-n_sigma, n_sigma, n_probe):
        v = state0 if abs(k) < 1e-9 else np.asarray(refit_fn(state0 + k * sigma * u, u, sigma, k), float)
        if not np.all(np.isfinite(v)):
            continue
        bp = impact._close_approach_bplane(epoch, v, backend, base_sim, t1_utc - 20.0, 40.0)
        if bp is not None and (best is None or bp["b"] < best["b"]):
            best = bp
    return best


def _aprime_trace(fit, t1_utc, backend, base_sim, u, sigma, refit_fn, n_sigma, n_probe,
                  post_days):
    """Post-encounter a'(k) across the LOV: re-correct each of ``n_probe`` clones over
    +-``n_sigma``, propagate ``post_days`` past the deflecting encounter and read its
    heliocentric semimajor axis. Returns sorted arrays (ks, aps) over the clones that resolved."""
    epoch = fit.epoch_tdb
    state0 = np.asarray(fit._state, float)
    t_eval_tdb = _utc_to_tdb(t1_utc + post_days)
    ks, aps = [], []
    for k in np.linspace(-n_sigma, n_sigma, n_probe):
        v = state0 if abs(k) < 1e-9 else np.asarray(refit_fn(state0 + k * sigma * u, u, sigma, k), float)
        if not np.all(np.isfinite(v)):
            continue
        a = _helio_a(epoch, v, t_eval_tdb, backend, base_sim)
        if np.isfinite(a) and 0.1 < a < 100.0:
            ks.append(float(k))
            aps.append(float(a))
    return np.asarray(ks), np.asarray(aps)


def _resonances_in_span(a_lo, a_hi, h_max, return_window_years):
    """Coprime (h, k) whose resonant a'_resonant = (h/k)^(2/3) lies in [a_lo, a_hi] and whose return
    (h Earth years) fits the search horizon. Yields (h, k, a_resonant, period_years)."""
    out = []
    amin, amax = (a_lo, a_hi) if a_lo <= a_hi else (a_hi, a_lo)
    for h in range(1, h_max + 1):
        if h > return_window_years:
            break
        for k in range(1, h_max + 1):
            if math.gcd(h, k) != 1:
                continue
            a_resonant = (h / k) ** (2.0 / 3.0)
            if amin <= a_resonant <= amax:
                out.append((h, k, a_resonant, a_resonant ** 1.5))
    # nearest-to-1 first: the strongest, widest keyholes sit at low-order resonances
    out.sort(key=lambda r: abs(r[2] - 1.0))
    return out


def _crossing_ks(ks, aps, a_resonant):
    """LOV coordinates k where the monotone-ish a'(k) trace crosses a_resonant (linear interp at
    sign changes of a'(k) - a_resonant)."""
    g = aps - a_resonant
    out = []
    for i in range(len(g) - 1):
        if g[i] == 0.0:
            out.append(float(ks[i]))
        elif g[i] * g[i + 1] < 0.0:
            t = g[i] / (g[i] - g[i + 1])
            out.append(float(ks[i] + t * (ks[i + 1] - ks[i])))
    return out


def _assess_keyhole(fit, k_center, t1_utc, t2_center_utc, backend, base_sim, u, sigma, refit_fn,
                    n_sigma, search_half_days=150.0, half_width_sigma=0.5, n_scan=21):
    """Impact probability through one keyhole: scan a tight LOV neighbourhood of ``k_center``,
    propagate each clone to the resonant return near ``t2_center_utc``, and integrate the
    Gaussian over the sub-interval whose return margin m(k) = b - b_capture is negative.
    Returns a dict with the keyhole geometry + IP (or an upper bound when the impacting band is
    too narrow to bracket). The keyhole WIDTH is the zeta extent of that impacting interval on
    the DEFLECTING encounter's b-plane (``t1_utc``) -- the actual (sub-km, typically) keyhole,
    not the Earth-capture-scale window at the return."""
    epoch = fit.epoch_tdb
    state0 = np.asarray(fit._state, float)
    t2_lo = t2_center_utc - search_half_days
    window_days = 2.0 * search_half_days
    cache = {}

    def bp2(k):
        key = round(k, 7)
        if key in cache:
            return cache[key]
        v = np.asarray(refit_fn(state0 + (k * sigma) * u, u, sigma, k), float)
        bp = None
        if np.all(np.isfinite(v)):
            bp = impact._close_approach_bplane(epoch, v, backend, base_sim, t2_lo, window_days)
        cache[key] = bp
        return bp

    def margin(k):
        bp = bp2(k)
        return None if bp is None else (bp["b"] - bp["b_planet"])

    lo = max(-n_sigma, k_center - half_width_sigma)
    hi = min(n_sigma, k_center + half_width_sigma)
    ks = np.linspace(lo, hi, n_scan)
    pts = [(float(k), margin(k)) for k in ks]
    pts = [(k, m) for k, m in pts if m is not None]
    if len(pts) < 3:
        return {"resolved": None, "error": "return not reached by enough clones"}

    kk = [p[0] for p in pts]
    mm = [p[1] for p in pts]
    i = int(np.argmin(mm))
    dk = float(ks[1] - ks[0])
    # descend into a possibly sub-cell dip around the grid minimum
    k_min, m_min = impact._golden_min(margin, kk[max(0, i - 1)], kk[min(len(kk) - 1, i + 1)])
    bp_min = bp2(k_min)
    geom = {"center_sigma": float(k_min),
            "return_min_km": float(bp_min["b"]) if bp_min else None,
            "capture_radius_km": float(bp_min["b_planet"]) if bp_min else None,
            "return_date": impact._iso(bp_min["t_ca"]) if (bp_min and bp_min.get("t_ca")) else None}

    if m_min is None or m_min >= 0.0:
        # no clone in the neighbourhood hits: report the closest the return comes
        floor = max(impact._phi(k_min + dk / 2) - impact._phi(k_min - dk / 2), 1e-9)
        return {**geom, "resolved": False, "ip_upper": float(floor),
                "note": (f"closest return b={bp_min['b']:,.0f} km vs capture "
                         f"{bp_min['b_planet']:,.0f} km" if bp_min else "no impacting clone")}

    def boundary(direction):
        k_prev, k = k_min, k_min
        for _ in range(4 * n_scan):
            k += direction * (dk / 2.0)
            if abs(k) >= n_sigma or abs(k - k_min) > 4 * half_width_sigma:
                return k
            m = margin(k)
            if m is None:
                k_prev = k
                continue
            if m > 0.0:
                return impact._brentq(lambda kk_: (margin(kk_) if margin(kk_) is not None else 1.0),
                                      min(k_prev, k), max(k_prev, k))
            k_prev = k
        return k

    klo, khi = boundary(-1), boundary(+1)
    ip = float(min(1.0, impact._phi(khi) - impact._phi(klo)))
    v_imp = math.sqrt(bp_min["u_inf"] ** 2 + 2 * MU_EARTH / R_EARTH_KM) if bp_min else None

    # keyhole width on the DEFLECTING b-plane: map the impacting k-interval back to the first
    # encounter's zeta (the genuine, typically sub-km, keyhole -- not the Earth-capture window
    # at the return).
    def zeta1(k):
        v = np.asarray(refit_fn(state0 + (k * sigma) * u, u, sigma, k), float)
        if not np.all(np.isfinite(v)):
            return None
        bp = impact._close_approach_bplane(epoch, v, backend, base_sim, t1_utc - 20.0, 40.0)
        return bp.get("zeta") if bp else None
    z_lo, z_hi = zeta1(klo), zeta1(khi)
    width_km = (float(abs(z_hi - z_lo)) if (z_lo is not None and z_hi is not None) else None)
    return {**geom, "resolved": True, "ip": ip,
            "sigma_interval": [float(klo), float(khi)],
            "width_sigma": float(abs(khi - klo)),
            "keyhole_width_km": width_km,
            "impact_velocity_km_s": v_imp}


def find_keyholes(fit, years=12.0, from_jd=None, backend=None, n_sigma=5.0, n_probe=25,
                  h_max=12, max_deflector_perigee_km=2.0e6, post_days=180.0):
    """Locate keyholes on the deepest Earth encounter in the forward window and estimate the
    impact probability of each resonant return.

    Returns ``{ok, ...}``: ``deflecting_encounter`` (date, perigee, U-infinity, theta), the
    list of ``keyholes`` (each with resonance ``h/k``, return year, LOV centre sigma, width,
    analytic b-plane circle, and ``ip``), the summed ``ip_total``, and a caveat pointing to the
    operational systems. ``ok: False`` when the LOV is unavailable or no encounter is deep
    enough to open a keyhole.
    """
    from astropy.time import Time
    u, sigma, refit_fn = fit.lov_refit()
    method = "Valsecchi 2003 resonant-return keyholes (LOV b-plane); single-deflector, no cascades"
    common = {"ok": True, "designation": fit.designation, "method": method,
              "force_model": fit.force_model, "n_sigma": float(n_sigma)}
    if u is None:
        return {"ok": False, "reason": "LOV unavailable (degenerate, no live evaluator, or "
                "nongrav-augmented state) -- keyholes not assessable"}

    # Reuse Phase-2's confidence gate: a short AND highly-uncertain arc has a weighting-sensitive
    # LOV direction, so keyhole locations would be meaningless.
    try:
        jds = [o["jd_utc"] for o in fit._incl_obs()]
        arc_days = (max(jds) - min(jds)) if len(jds) > 1 else 0.0
        U = fit.uncertainty_parameter()
    except Exception:
        arc_days, U = None, None
    if arc_days is not None and U is not None and arc_days < 3.0 and U >= 6.48:
        return {**common, "assessed": False, "low_confidence": True,
                "confidence_note": ("Orbit too poorly constrained for keyhole analysis "
                                    f"({arc_days:.2f}-day arc, U={U:.0f}/9); consult JPL Sentry / ESA NEODyS.")}

    backend, base_sim = impact._backend_and_sim(fit, backend)
    epoch_utc = Time(fit.epoch_tdb, format="jd", scale="tdb").utc.jd
    t_start = from_jd if from_jd is not None else Time.now().utc.jd
    t_start = max(float(t_start), epoch_utc - 0.5)
    window_days = years * YEAR_DAYS

    enc = _deflecting_encounter(fit, t_start, window_days, backend, base_sim, u, sigma, refit_fn,
                                n_sigma, max_deflector_perigee_km)
    if enc is None:
        return {**common, "assessed": True, "keyholes": [], "ip_total": 0.0,
                "note": ("No Earth encounter deep enough to open a keyhole in the searched "
                         f"{years:.0f}-yr window.")}
    t1 = enc["t_ca"]

    # geometry of the deflecting encounter (from the deepest clone) for the analytic circle
    bp1 = _encounter_geometry(fit, t1, backend, base_sim, u, sigma, refit_fn, n_sigma, max(9, n_probe // 2))
    u_inf = float(bp1["u_inf"]) if bp1 else None
    theta = v_planet = None
    if bp1 is not None:
        from pyroorb.obs_sources import observer_positions_batch
        ddt = 1e-3
        ep = np.asarray(observer_positions_batch([bp1["t_ca"] + ddt], ["500"], {}))[0]
        em = np.asarray(observer_positions_batch([bp1["t_ca"] - ddt], ["500"], {}))[0]
        v_earth = (ep - em) / (2 * ddt) * AU_KM / impact.DAY_S
        v_planet = float(np.linalg.norm(v_earth))
        vp_hat = v_earth / v_planet
        theta = float(math.acos(max(-1.0, min(1.0, float(np.asarray(bp1["u_hat"]) @ vp_hat)))))

    ks, aps = _aprime_trace(fit, t1, backend, base_sim, u, sigma, refit_fn, n_sigma, n_probe, post_days)
    deflector = {"date": impact._iso(t1), "perigee_km": float(enc["env_min_km"]),
                 "u_infinity_km_s": u_inf, "theta_deg": (math.degrees(theta) if theta else None),
                 "post_encounter_a_span_au": ([float(aps.min()), float(aps.max())] if len(aps) else None)}
    if len(aps) < 5:
        return {**common, "assessed": True, "deflecting_encounter": deflector, "keyholes": [],
                "ip_total": 0.0, "note": "Too few clones resolved the deflecting encounter to trace a'(LOV)."}

    resonances = _resonances_in_span(aps.min(), aps.max(), h_max, years)
    keyholes = []
    for (h, k, a_resonant, period_yr) in resonances:
        for k_center in _crossing_ks(ks, aps, a_resonant):
            t2_center = t1 + h * YEAR_DAYS
            result = _assess_keyhole(fit, k_center, t1, t2_center, backend, base_sim, u, sigma, refit_fn, n_sigma)
            circ = (resonant_circle(u_inf, theta, h, k, v_planet) if (u_inf and theta) else None)
            entry = {"resonance": f"{h}/{k}", "a_prime_au": float(a_resonant),
                     "return_year_offset": int(h), "lov_center_sigma": float(k_center),
                     "resonant_circle": ({"zeta_center_km": circ[0], "radius_km": circ[1]} if circ else None),
                     **result}
            keyholes.append(entry)

    # keep the meaningful ones first (resolved impacts by IP, then unresolved)
    keyholes.sort(key=lambda e: (not e.get("resolved"), -(e.get("ip") or 0.0)))
    ips = [e["ip"] for e in keyholes if e.get("resolved")]
    return {**common, "assessed": True, "deflecting_encounter": deflector,
            "keyholes": keyholes, "n_keyholes": len(keyholes),
            "ip_total": (float(min(1.0, sum(ips))) if ips else 0.0),
            "caveat": ("EXPERIMENTAL: single-deflector resonant-return keyholes only; not a "
                       "cascade-complete search. For operational keyhole/impact monitoring use "
                       "JPL Sentry / ESA NEODyS.")}
