"""Impact geometry: where and when a fitted orbit reaches the ground.

PHASE 1 of the impact-monitoring feature (impact LOCATION / corridor for known or
near-certain impactors, IP ~ 1). This is the part with a *measured* ground truth
-- recovered meteorites and bolide cameras -- so it is built and validated first,
against the ESA "past impactors" set (2008 TC3, 2018 LA, 2023 CX1, 2024 BX1, ...).

Method (ported in logic from the OrbFit/NEODyS impact-corridor algorithm of
Dimare, Del Vigna, Bracali Ciocci & Bernardi 2020, "Use of the Semilinear Method
to predict the Impact Corridor on Ground", arXiv:2007.05407; see
PROVENANCE.md). Their impact map F^h = Sigma^h o Phi o Psi is:

    Psi    -- find the time t* at which the geodetic altitude reaches h, by a
              root find on  zeta(p(t)) - h = 0  during the close approach;
    Phi    -- propagate the orbit to t* (we reuse the engine's force model);
    Sigma^h-- project the J2000-equatorial geocentric state at t* to geodetic
              (lat, lon) on the WGS-84 ellipsoid -- this is where Earth's
              rotation enters (inertial GCRS -> Earth-fixed ITRS at t*).

For a certain impact (IP = 1) the "virtual impactor representative" is just the
nominal orbit, so the nominal impact point is F^h(x0). The 1-sigma footprint is a
later refinement (the semilinear uncertainty ellipse); this module computes the
nominal corridor first -- the thing the oracles pin down to ~km.

Honest-failure contract: if the orbit has no Earth close approach in the search
window, or perigee stays above the atmosphere, this returns an explicit "no
impact" result -- never a fabricated crossing.
"""
import math
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

from pyroorb import bplane
from pyroorb import eop                          # the one Earth-orientation table
eop.pin()

AU_KM = 149_597_870.7
DAY_S = 86400.0
R_EARTH_KM = bplane.R_EARTH_KM   # WGS-84 equatorial radius (single source: bplane.py)
ATM_ENTRY_KM = 100.0             # nominal atmospheric-interface altitude (h_max)
CLOSE_APPROACH_AU = 0.2          # geocentric distance gate for impact monitoring
LD_KM = 384_400.0                # one lunar distance
# A nominal Earth pass deeper than this BETWEEN the epoch and a target encounter folds the
# Line of Variations enough that our 1-D LOV scan is out of its validated domain (Milani et al.
# 2005; Del Vigna et al. 2020). Empirically separates the IN-DOMAIN encounters (2013 TP4,
# 2008 TC3: 0 intervening) from the out-of-domain ones (2000 SG344: 6, 2012 HG2: 1, with a
# 0.3-LD pass). This is only the DOMAIN gate; a further U>=5 gate below then refuses the numeric
# IP for an under-constrained orbit even when in-domain (2013 TP4, U=8, was 1.3x off Sentry --
# outside +-20%). See _intervening_approaches and the 500-object Sentry validation below.
INTERVENING_APPROACH_AU = 0.05   # ~19.4 LD


# ===========================================================================
# Shared propagation & geometry helpers (used by all three phases below)
# ===========================================================================

def _coverage_decline(err, jd):
    """If ``err`` is the ASSIST 'time outside ephemeris coverage' error, return a graceful
    decline note (the encounter is centuries out, past DE440's span -- e.g. numbered objects
    with a year-2880 Sentry virtual impactor); else None so the caller re-raises. Refusing
    cleanly beats an unhandled crash for an encounter our propagator simply cannot reach."""
    if "coverage" not in str(err).lower():
        return None
    return (f"Impact probability not estimated: the encounter near {_iso(jd)} is beyond the span "
            f"of the DE440 ephemeris our propagator uses, so we cannot integrate to it. For a "
            f"long-range assessment consult JPL Sentry / ESA NEODyS.")


def _backend_and_sim(fit, backend=None):
    from pyroorb.obs_sources import _force_backend
    if backend is None:
        backend, _ = _force_backend(fit)      # keeps a self-perturber's pinned force model
    base_sim = backend.build_base_sim(fit.epoch_tdb)
    return backend, base_sim


def _geo_pos(epoch_tdb, state, jds_utc, backend, base_sim):
    """Geocentric ICRS-equatorial positions (n,3 au) of a body with heliocentric ``state``
    at ``epoch_tdb`` over the given UTC JDs: asteroid heliocentric position minus Earth's
    heliocentric position (observer code 500 = geocentre). The state is explicit so LOV /
    Monte-Carlo clones propagate through the same path as the nominal orbit."""
    from pyroorb.obs_sources import observer_positions_batch
    jds = np.atleast_1d(np.asarray(jds_utc, dtype=float))
    tt = Time(jds, format="jd", scale="utc").tdb.jd
    helio = backend.helio_positions_batch(base_sim, epoch_tdb,
                                          np.asarray(state, float), tt)          # (n,3) au, ast-Sun
    earth = np.asarray(observer_positions_batch(jds, ["500"] * len(jds), {}))    # (n,3) au, Earth-Sun
    return np.asarray(helio) - earth


def geocentric_positions(fit, jds_utc, backend, base_sim):
    """Geocentric ICRS-equatorial positions (n,3 au) of the fitted (nominal) body."""
    return _geo_pos(fit.epoch_tdb, fit._state, jds_utc, backend, base_sim)


def geodetic_subpoint(r_geo_au, jds_utc):
    """Map geocentric ICRS-equatorial position vectors (n,3 au) to the WGS-84 geodetic
    sub-point at each UTC epoch: returns (lat_deg, lon_deg, alt_km) arrays.

    The inertial geocentric vector is treated as a GCRS Cartesian position (GCRS shares
    ICRS axes; the frame bias is ~17 mas => sub-metre on the ground) and rotated to the
    Earth-fixed ITRS frame at the observation time -- this applies precession, nutation
    and Earth rotation -- then read off as geodetic lat/lon/height on the WGS-84 ellipsoid
    (astropy's EarthLocation default)."""
    r = np.atleast_2d(np.asarray(r_geo_au, dtype=float))
    jds = np.atleast_1d(np.asarray(jds_utc, dtype=float))
    t = Time(jds, format="jd", scale="utc")
    cart = CartesianRepresentation(r[:, 0] * u.au, r[:, 1] * u.au, r[:, 2] * u.au)
    itrs = GCRS(cart, obstime=t).transform_to(ITRS(obstime=t))
    el = itrs.earth_location
    return el.lat.deg, el.lon.deg, el.height.to(u.km).value


def _altitude(fit, jd_utc, backend, base_sim):
    """Geodetic altitude (km) of the body's sub-point at a single UTC JD."""
    r = geocentric_positions(fit, [jd_utc], backend, base_sim)
    _, _, alt = geodetic_subpoint(r, [jd_utc])
    return float(alt[0])


def _refine_close_approach(fit, jd_lo, jd_hi, backend, base_sim):
    """Golden-section minimisation of geocentric distance on [jd_lo, jd_hi] to locate the
    close-approach (perigee) time to ~second precision. Returns (jd_ca, dist_km)."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = jd_lo, jd_hi

    def dist(j):
        r = geocentric_positions(fit, [j], backend, base_sim)[0]
        return math.sqrt(float(r @ r)) * AU_KM

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = dist(c), dist(d)
    for _ in range(60):
        if (b - a) < 1e-7:               # ~9 ms
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = dist(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = dist(d)
    j = 0.5 * (a + b)
    return j, dist(j)


def _brentq(f, a, b, tol=1e-9, maxit=80):
    """Minimal Brent root finder for f on [a,b] with f(a),f(b) opposite signs.
    (Self-contained so impact.py has no scipy dependency.)"""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("root not bracketed")
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc, d = a, fa, b
    mflag = True
    for _ in range(maxit):
        if fb == 0 or abs(b - a) < tol:
            return b
        if fa != fc and fb != fc:        # inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:                            # secant
            s = b - fb * (b - a) / (fb - fa)
        cond = ((s < (3 * a + b) / 4 or s > b) if b > a else (s > (3 * a + b) / 4 or s < b))
        if (cond or (mflag and abs(s - b) >= abs(b - c) / 2)
                or (not mflag and abs(s - b) >= abs(c - d) / 2)
                or (mflag and abs(b - c) < tol) or (not mflag and abs(c - d) < tol)):
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d, c, fc = c, b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    return b


# ===========================================================================
# PHASE 1 -- impact LOCATION (the nominal ground corridor for an IP ~ 1 impactor)
# ===========================================================================

def find_impact(fit, window_days=30.0, t_start_utc=None, step_hours=1.0,
                altitudes_km=(100.0, 0.0), backend=None):
    """Locate the impact (if any) of a fitted orbit by propagating it forward and finding
    where its sub-point crosses each target altitude.

    Search: a coarse grid (step_hours) over [t_start, t_start+window_days] locates the
    deepest close approach; t_start defaults to just before the fit epoch. The close
    approach is refined to perigee, and -- if perigee altitude < 0 (the body reaches the
    ground) -- the descending crossing of each altitude in `altitudes_km` (default
    100 km atmospheric interface and 0 km ground) is solved with a Brent root find.

    Returns a dict. `impact=True` carries `crossings` (one per altitude, with UTC time,
    geodetic lat/lon, geocentric entry speed and slope) and the close-approach summary.
    `impact=False` carries the close-approach miss distance -- explicit, never fabricated.
    """
    backend, base_sim = _backend_and_sim(fit, backend)
    epoch_utc = Time(fit.epoch_tdb, format="jd", scale="tdb").utc.jd
    if t_start_utc is None:
        t_start_utc = epoch_utc - 0.5

    grid = np.arange(t_start_utc, t_start_utc + window_days, step_hours / 24.0)
    rgeo = geocentric_positions(fit, grid, backend, base_sim)
    dist_km = np.linalg.norm(rgeo, axis=1) * AU_KM
    imin = int(np.argmin(dist_km))

    # No close approach at all in the window -> nothing to assess.
    if dist_km[imin] > CLOSE_APPROACH_AU * AU_KM:
        return {"impact": False, "reason": "no close approach in window",
                "min_geocentric_km": float(dist_km[imin]),
                "ca_time_utc": float(grid[imin])}

    lo = grid[max(0, imin - 1)]
    hi = grid[min(len(grid) - 1, imin + 1)]
    jd_ca, ca_dist_km = _refine_close_approach(fit, lo, hi, backend, base_sim)
    alt_ca = _altitude(fit, jd_ca, backend, base_sim)

    ca = {"ca_time_utc": float(jd_ca),
          "ca_time_iso": Time(jd_ca, format="jd", scale="utc").iso[:23],
          "ca_geocentric_km": float(ca_dist_km),
          "ca_min_altitude_km": float(alt_ca),
          "earth_radius_km": R_EARTH_KM}

    if alt_ca >= 0.0:                    # perigee above the surface -> flyby, not an impact
        return {"impact": False, "reason": "perigee above ground (flyby)", **ca}

    # Impact: solve the descending crossing of each target altitude between a pre-approach
    # point (well above the atmosphere) and perigee (below ground).
    j_above = grid[max(0, imin - 1)]
    if _altitude(fit, j_above, backend, base_sim) <= max(altitudes_km):
        # coarse step landed too close; back off until clearly above the highest target
        j_above = jd_ca - step_hours / 24.0
        while _altitude(fit, j_above, backend, base_sim) <= max(altitudes_km):
            j_above -= step_hours / 24.0

    crossings = []
    for h in altitudes_km:
        def f(j):
            return _altitude(fit, j, backend, base_sim) - h
        try:
            jt = _brentq(f, j_above, jd_ca)
        except ValueError:
            continue
        r = geocentric_positions(fit, [jt], backend, base_sim)[0]
        lat, lon, alt = geodetic_subpoint([r], [jt])
        speed, slope = _entry_kinematics(fit, jt, backend, base_sim, r)
        crossings.append({
            "altitude_km": float(h),
            "time_utc": float(jt),
            "time_iso": Time(jt, format="jd", scale="utc").iso[:23],
            "lat_deg": float(lat[0]),
            "lon_deg": float(lon[0]),
            "geo_speed_km_s": speed,
            "entry_slope_deg": slope,
        })
    return {"impact": True, "crossings": crossings, **ca}


def _entry_kinematics(fit, jd_utc, backend, base_sim, r_geo_au):
    """Geocentric entry speed (km/s) and slope below the local horizontal (deg) at the
    crossing, by central-differencing the geocentric position (dt ~ 0.5 s)."""
    dt = 5e-6                                          # day ~ 0.43 s
    rp = geocentric_positions(fit, [jd_utc + dt], backend, base_sim)[0]
    rm = geocentric_positions(fit, [jd_utc - dt], backend, base_sim)[0]
    v = (rp - rm) / (2 * dt) * AU_KM / 86400.0         # km/s
    speed = float(math.sqrt(v @ v))
    r_hat = np.asarray(r_geo_au) / math.sqrt(np.asarray(r_geo_au) @ np.asarray(r_geo_au))
    slope = float(math.degrees(math.asin(min(1.0, abs(v @ r_hat) / speed))))
    return speed, slope


# ===========================================================================
# PHASE 1b -- uncertainty footprint (the semilinear 1-sigma impact ellipse)
# ===========================================================================
# For a near-certain impactor (IP ~ 1) the nominal corridor (find_impact) is only half the
# answer; the operational systems also deliver the 1-sigma *ground footprint* -- the region
# the impact point occupies given the orbit's covariance. The exact image of the 6-D
# confidence ellipsoid under the impact map F^h is what Dimare/Del Vigna 2020 call the
# impact region; computing it by Monte-Carlo would need ~r^(N-1) propagations, so OrbFit uses
# the SEMILINEAR method (Milani & Valsecchi 1999): propagate only the 1-D BOUNDARY curve of
# the confidence ellipse that maps onto the boundary of the ground region, not the whole
# interior. We port that logic here.
#
# Construction (working in WHITENED state coordinates z, where x = x0 + C^{1/2} z so the
# 1-sigma ellipsoid is the unit sphere |z| = 1):
#   1. J_z = d(ground E,N)/dz, a 2x6 finite-difference Jacobian of the impact map at nominal.
#   2. SVD J_z = U S V^T. The LINEAR image of the unit sphere is the ellipse with semi-axes
#      sigma*S along the ground directions U; the unit-sphere points that map to that ellipse's
#      boundary are z(theta) = sigma*(V[0]cos t + V[1]sin t) (|z| = sigma, min-norm preimages).
#   3. SEMILINEAR refinement: propagate x0 + C^{1/2} z(theta) through the FULL nonlinear impact
#      map for each theta; the resulting closed ground curve is the 1-sigma footprint boundary
#      (it captures the curvature the linear ellipse misses -- a shifted centre, banana shape).
# Everything stays a leaf operation on the converged FittedOrbit => zero regression-suite risk.

def _to_local_en(lat, lon, lat0, lon0):
    """Ground (lat,lon) deg -> local East/North offset (km) on a plane tangent at (lat0,lon0)."""
    dlat = math.radians(lat - lat0)
    dlon = (math.radians(lon - lon0) + math.pi) % (2 * math.pi) - math.pi   # unwrap
    e = R_EARTH_KM * math.cos(math.radians(lat0)) * dlon
    n = R_EARTH_KM * dlat
    return e, n


def _from_local_en(e, n, lat0, lon0):
    """Inverse of _to_local_en: local East/North (km) -> (lat,lon) deg."""
    lat = lat0 + math.degrees(n / R_EARTH_KM)
    lon = lon0 + math.degrees(e / (R_EARTH_KM * math.cos(math.radians(lat0))))
    return lat, ((lon + 180.0) % 360.0) - 180.0


def _axis_azimuth_deg(vec):
    """Azimuth (deg, clockwise from North, folded to [0,180)) of an ellipse axis (E,N)."""
    return math.degrees(math.atan2(float(vec[0]), float(vec[1]))) % 180.0


def _ground_point(epoch_tdb, state, jd_center, h_km, backend, base_sim, half_window_days):
    """(jt, lat, lon) of the DESCENDING crossing of altitude ``h_km`` for a body with
    heliocentric ``state``, searched in a window around ``jd_center`` (the nominal crossing).
    Brackets the crossing on a coarse grid (widening the window up to 4x if the perturbed
    close approach drifted out of it), then Brent-refines. None if no crossing is found."""
    def alt_minus_h(j):
        r = _geo_pos(epoch_tdb, state, [j], backend, base_sim)
        _, _, a = geodetic_subpoint(r, [j])
        return float(a[0]) - h_km

    for hw in (half_window_days, 2 * half_window_days, 4 * half_window_days):
        grid = np.linspace(jd_center - hw, jd_center + hw, 41)
        r = _geo_pos(epoch_tdb, state, grid, backend, base_sim)
        _, _, alt = geodetic_subpoint(r, grid)
        f = alt - h_km
        idx = next((i for i in range(len(grid) - 1) if f[i] > 0 >= f[i + 1]), None)
        if idx is not None:
            jt = _brentq(alt_minus_h, grid[idx], grid[idx + 1], tol=1e-8)
            r = _geo_pos(epoch_tdb, state, [jt], backend, base_sim)
            lat, lon, _ = geodetic_subpoint(r, [jt])
            return jt, float(lat[0]), float(lon[0])
    return None


def impact_footprint(fit, sigma=1.0, n_boundary=24, h_km=100.0, window_days=30.0,
                     t_start_utc=None, half_window_days=0.05, fd_step=0.1, backend=None):
    """Semilinear 1-sigma impact FOOTPRINT (Phase 1b) for a fitted orbit whose
    NOMINAL orbit reaches the ground (a near-certain impactor; ``find_impact`` must report an
    impact). Propagates the boundary of the ``sigma``-confidence ellipsoid through the full
    nonlinear impact map and reports the resulting ground ellipse + boundary polygon at
    altitude ``h_km`` (default the 100 km atmospheric-entry interface, the Dimare 2020 / 2008
    TC3 oracle reference surface).

    Returns ``{ok: True, ...}`` with the nominal point, the ``linear`` ellipse (covariance
    image: semi-major/minor km + axis azimuth) and the ``semilinear`` ellipse (centroid offset,
    semi-axes, azimuth, and the ground ``boundary`` [[lat,lon],...] polygon). ``ok: False`` when
    the nominal orbit does not impact, or the impact map could not be differentiated (degenerate
    covariance / the perturbed crossings could not be located). Cost scales with ``n_boundary``
    propagations; opt-in leaf on the converged fit -- the default pipeline is untouched."""
    if np.asarray(fit._state).size != 6:
        # Nongrav-augmented state (6 orbital + A-terms): the state covariance is the 6x6
        # orbital block, so mixing it with the augmented state is ill-defined here -- decline
        # (same limit as the LOV paths). The certain-impactor footprint targets short-arc
        # gravity-only fits anyway.
        return {"ok": False, "designation": fit.designation,
                "reason": "nongrav-augmented state -- semilinear footprint not assessable"}
    backend, base_sim = _backend_and_sim(fit, backend)
    epoch = fit.epoch_tdb

    nom = find_impact(fit, window_days=window_days, t_start_utc=t_start_utc,
                      altitudes_km=(h_km,), backend=backend)
    if not nom.get("impact"):
        return {"ok": False, "designation": fit.designation,
                "reason": f"nominal orbit does not impact ({nom.get('reason', 'no impact')})",
                "nominal": nom}
    c0 = next((c for c in nom["crossings"] if c["altitude_km"] == h_km), None)
    if c0 is None:
        return {"ok": False, "designation": fit.designation,
                "reason": f"no {h_km:.0f} km crossing on the nominal orbit"}
    jd0, lat0, lon0 = c0["time_utc"], c0["lat_deg"], c0["lon_deg"]

    state0 = np.asarray(fit._state, float)
    C = np.asarray(fit.covariance_matrix(frame="state"), float)
    w, V = np.linalg.eigh(0.5 * (C + C.T))                 # symmetric PSD square root
    L = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T          # C^{1/2}, 6x6

    def ground_en(z):
        gp = _ground_point(epoch, state0 + L @ z, jd0, h_km, backend, base_sim, half_window_days)
        if gp is None:
            return None
        return np.array(_to_local_en(gp[1], gp[2], lat0, lon0))

    # 2x6 whitened Jacobian by central differences (each column ~ 0.1-sigma either side).
    Jz = np.zeros((2, 6))
    for k in range(6):
        e = np.zeros(6); e[k] = fd_step
        gp, gm = ground_en(e), ground_en(-e)
        if gp is None or gm is None:
            return {"ok": False, "designation": fit.designation,
                    "reason": "could not differentiate the impact map (covariance too broad "
                              "for the semilinear footprint -- use the LOV impact probability)"}
        Jz[:, k] = (gp - gm) / (2 * fd_step)

    U, S, Vt = np.linalg.svd(Jz, full_matrices=False)       # Jz = U @ diag(S) @ Vt, U is 2x2
    linear = {"semi_major_km": float(sigma * S[0]), "semi_minor_km": float(sigma * S[1]),
              "azimuth_deg": _axis_azimuth_deg(U[:, 0])}

    # Semilinear boundary: full nonlinear image of the confidence-ellipse boundary.
    en, ll = [], []
    for j in range(n_boundary):
        th = 2.0 * math.pi * j / n_boundary
        p = ground_en(sigma * (Vt[0] * math.cos(th) + Vt[1] * math.sin(th)))
        if p is None:
            continue
        en.append(p)
        ll.append(list(_from_local_en(p[0], p[1], lat0, lon0)))
    if len(en) < max(8, n_boundary // 2):
        return {"ok": False, "designation": fit.designation,
                "reason": "too few footprint-boundary crossings resolved"}

    P = np.asarray(en)
    ctr = P.mean(axis=0)
    # Points sampled uniformly in angle on an ellipse have covariance 0.5*diag(a^2,b^2) in the
    # principal frame, so the semi-axes are sqrt(2*eigenvalue) and the axis is the eigenvector.
    ew, ev = np.linalg.eigh(np.cov((P - ctr).T))
    ew = np.clip(ew, 0.0, None)
    clat, clon = _from_local_en(ctr[0], ctr[1], lat0, lon0)
    semilinear = {"semi_major_km": float(math.sqrt(2.0 * ew[1])),
                  "semi_minor_km": float(math.sqrt(2.0 * ew[0])),
                  "azimuth_deg": _axis_azimuth_deg(ev[:, 1]),
                  "center_lat_deg": clat, "center_lon_deg": clon,
                  "center_offset_km": float(math.hypot(ctr[0], ctr[1])),
                  "boundary": ll}
    return {"ok": True, "designation": fit.designation, "force_model": fit.force_model,
            "sigma": float(sigma), "altitude_km": float(h_km),
            "nominal": {"lat_deg": lat0, "lon_deg": lon0,
                        "time_iso": Time(jd0, format="jd", scale="utc").iso[:23]},
            "linear": linear, "semilinear": semilinear,
            "method": "semilinear 1-sigma impact ellipse (Milani-Valsecchi / Dimare-Del Vigna)"}


# ===========================================================================
# PHASE 2 -- impact PROBABILITY along the Line of Variations
# ===========================================================================
# Method (find_orb / NEODyS LOV, the pre-Sentry-II approach):
# step statistical clones along the LOV -- the curved valley of best-fitting orbits --
# find the LOV-coordinate interval(s) whose orbits reach the ground (perigee altitude < 0,
# with gravitational focusing already in the integrated trajectory), and integrate the
# Gaussian probability density over those intervals. IP = sum of Phi(k_hi) - Phi(k_lo).
#
# Honest floor: with an N-point scan over +-n_sigma the smallest resolvable probability is
# the Gaussian mass of one scan cell near the closest approach; below that we report a BOUND
# ("< floor, not resolved"), never a spuriously precise tiny number. This LOV method is the
# right 1-D approximation when the LOV dominates the uncertainty; it can MISS impacts behind
# a deep prior close encounter that scrambles the LOV ordering (the case Sentry-II's Monte
# Carlo was built for) -- flagged in the result, not silently wrong.

def _phi(k):
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(k / math.sqrt(2.0)))


def _golden_min(f, a, b, iters=16):
    """Golden-section minimisation of a scalar f on [a, b]; returns (x_min, f_min). Used to
    locate the smooth closest-approach point on the LOV precisely -- a narrow sub-zero
    (small-IP) dip can sit entirely between coarse scan samples."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    fc = fc if fc is not None else math.inf
    fd = fd if fd is not None else math.inf
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c); fc = fc if fc is not None else math.inf
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d); fd = fd if fd is not None else math.inf
    x = 0.5 * (a + b)
    fx = f(x)
    return x, (fx if fx is not None else min(fc, fd))


def _close_approach_bplane(epoch_tdb, state, backend, base_sim, t_start_utc, window_days,
                           coarse_step_days=1.0, fine_half_days=0.6, fine_step_min=1.0):
    """b-plane crossing of a clone's deepest close approach in the window, from a SINGLE
    forward propagation (coarse daily scan to find the approach day, then a fine scan to get
    an in-encounter (r, v) for the b-plane). Because the b-plane quantities (b, xi, zeta) come
    from the CONSERVED angular momentum and energy, an in-encounter sample is exact -- no need
    to sit exactly at perigee, which is what makes the multi-year propagation affordable.
    Returns the bplane.bplane_crossing dict (+ t_ca, grid perigee) or None if no close
    approach / the planetocentric two-body approximation does not apply."""
    from pyroorb.obs_sources import observer_positions_batch
    n_coarse = max(8, int(window_days / coarse_step_days))
    cg = np.linspace(t_start_utc, t_start_utc + window_days, n_coarse)
    dc = np.linalg.norm(_geo_pos(epoch_tdb, state, cg, backend, base_sim), axis=1) * AU_KM
    i = int(np.argmin(dc))
    if dc[i] > CLOSE_APPROACH_AU * AU_KM:
        return None
    t0 = cg[i]
    step = fine_step_min / 1440.0
    fg = np.arange(t0 - fine_half_days, t0 + fine_half_days, step)
    rf = _geo_pos(epoch_tdb, state, fg, backend, base_sim)
    df = np.linalg.norm(rf, axis=1) * AU_KM
    j = int(np.clip(np.argmin(df), 1, len(fg) - 2))
    t_ca = float(fg[j])
    r_ca = rf[j] * AU_KM                                           # km
    v_ca = (rf[j + 1] - rf[j - 1]) / (2 * step) * AU_KM / DAY_S    # km/s (central diff)
    ddt = 1e-3
    ep = np.asarray(observer_positions_batch([t_ca + ddt], ["500"], {}))[0]
    em = np.asarray(observer_positions_batch([t_ca - ddt], ["500"], {}))[0]
    v_earth = (ep - em) / (2 * ddt) * AU_KM / DAY_S               # Earth heliocentric km/s
    bp = bplane.bplane_crossing(r_ca, v_ca, v_earth)
    if bp is not None:
        bp["t_ca"] = t_ca
        bp["grid_min_km"] = float(df[j])
    return bp


def _iso(jd):
    return Time(jd, format="jd", scale="utc").iso[:19]


def _refine_min_dist(epoch_tdb, state, t_lo, t_hi, backend, base_sim):
    """Golden-section minimisation of geocentric distance (km) on [t_lo, t_hi]; (t_ca, dmin)."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = t_lo, t_hi

    def d(j):
        r = _geo_pos(epoch_tdb, state, [j], backend, base_sim)[0]
        return math.sqrt(float(r @ r)) * AU_KM

    c, dd = b - gr * (b - a), a + gr * (b - a)
    fc, fd = d(c), d(dd)
    for _ in range(40):
        if (b - a) < 1e-6:
            break
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - gr * (b - a); fc = d(c)
        else:
            a, c, fc = c, dd, fd
            dd = a + gr * (b - a); fd = d(dd)
    t = 0.5 * (a + b)
    return float(t), d(t)


def find_close_approaches(fit, t_start_utc, window_days, backend, base_sim,
                          u, sigma, refit_fn, n_sigma=5.0, n_probe=21,
                          threshold_au=CLOSE_APPROACH_AU, coarse_step_days=1.0, merge_days=30.0):
    """Close-approach EVENTS the LOV uncertainty region makes in [t_start, t_start+window].
    For each of ``n_probe`` clones spanning +-``n_sigma`` we find that clone's interior local
    minima of geocentric distance below ``threshold_au``; all such approaches across all clones
    are pooled and clustered (within ``merge_days``) into encounters.

    PER-CLONE, not an envelope-over-time: a narrow return reached only by a clone BETWEEN
    coarse probes is otherwise silently missed. (2009 VA's 2026 return sits at ~4.2 sigma in our
    solution -- a ~0.7-sigma-wide sub-0.2 au band -- which 1.67-sigma-spaced probes step over;
    0.5-sigma spacing catches it.) An uncertain arc's nominal can miss by far (151 LD for
    2013 TP4) while a variant grazes, so the nominal alone is never enough. Returns
    [{t_ca, t_iso, env_min_km}] in time order (``env_min_km`` = closest any probe came)."""
    n = max(8, int(window_days / coarse_step_days))
    grid = np.linspace(t_start_utc, t_start_utc + window_days, n)
    state0 = np.asarray(fit._state, float)
    thr = threshold_au * AU_KM
    hits = []                                                    # (t_ca, dist_km) across all clones
    for k in np.linspace(-n_sigma, n_sigma, n_probe):
        v = state0 if abs(k) < 1e-9 else np.asarray(refit_fn(state0 + k * sigma * u, u, sigma, k), float)
        if not np.all(np.isfinite(v)):
            continue
        d = np.linalg.norm(_geo_pos(fit.epoch_tdb, v, grid, backend, base_sim), axis=1) * AU_KM
        for i in range(1, n - 1):                                # interior local minima below threshold
            if d[i] < thr and d[i] <= d[i - 1] and d[i] <= d[i + 1]:
                t_ca, dmin = _refine_min_dist(fit.epoch_tdb, v, grid[i - 1], grid[i + 1],
                                              backend, base_sim)
                hits.append((t_ca, dmin))
    out = []
    for t, dmin in sorted(hits):
        if out and (t - out[-1]["t_ca"]) < merge_days:            # same encounter -> keep the closest probe
            if dmin < out[-1]["env_min_km"]:
                out[-1] = {"t_ca": t, "t_iso": _iso(t), "env_min_km": float(dmin)}
            continue
        out.append({"t_ca": float(t), "t_iso": _iso(t), "env_min_km": float(dmin)})
    return out


def _intervening_approaches(fit, t_lo_utc, t_hi_utc, backend, base_sim,
                            threshold_au=INTERVENING_APPROACH_AU, step_days=2.0):
    """Nominal-orbit Earth close approaches deeper than ``threshold_au`` strictly inside
    (``t_lo_utc``, ``t_hi_utc``) (UTC JD). A deep pass before a target encounter folds the Line
    of Variations, putting that later encounter beyond the validated regime of our 1-D LOV method
    (Milani 2005; Del Vigna 2020) -- the discriminator is intrinsic (no Sentry comparison) and the
    error scales with the count (2013 TP4: 0 -> 1.3x Sentry; 2000 SG344: 6 -> 30x low; 2012 HG2:
    1+a 0.3-LD pass -> 400x low). Propagates only the NOMINAL state, so it is one cheap sweep.
    Returns [{t_iso, dist_km}] in time order (closest-pass minima)."""
    span = t_hi_utc - t_lo_utc
    if span <= step_days:
        return []
    n = max(8, int(span / step_days))
    grid = np.linspace(t_lo_utc, t_hi_utc, n)
    v = np.asarray(fit._state, float)
    d = np.linalg.norm(_geo_pos(fit.epoch_tdb, v, grid, backend, base_sim), axis=1) * AU_KM
    thr = threshold_au * AU_KM
    out = []
    for i in range(1, n - 1):                                   # interior local minima below threshold
        if d[i] < thr and d[i] <= d[i - 1] and d[i] <= d[i + 1]:
            t_ca, dmin = _refine_min_dist(fit.epoch_tdb, v, grid[i - 1], grid[i + 1],
                                          backend, base_sim)
            out.append({"t_iso": _iso(t_ca), "dist_km": float(dmin)})
    return out


def _out_of_domain_note(interv):
    """Honest decline message for an encounter put out of our 1-D method's domain by prior deep
    Earth approaches. ``interv`` is a non-empty _intervening_approaches list."""
    deepest = min(interv, key=lambda a: a["dist_km"])
    return (f"Impact probability not estimated: {len(interv)} closer Earth approach(es) precede "
            f"this encounter (deepest {deepest['dist_km'] / LD_KM:.1f} lunar distances on "
            f"{deepest['t_iso'][:10]}). Each prior pass folds the line-of-variations, so this "
            f"later encounter is beyond the validated regime of our 1-D LOV method. For this "
            f"object, consult JPL Sentry / ESA NEODyS.")


def _assess_encounter(u, sigma, refit_fn, state0, epoch, backend, base_sim,
                      t_start, scan_window, n_sigma, n_scan):
    """LOV b-plane impact probability for ONE encounter window. Each re-corrected clone is
    propagated to its close approach and mapped to a b-plane crossing; the margin
    m(k) = b(k) - b_planet(k) is < 0 exactly where the clone hits (focusing included). We
    locate the closest approach on the LOV precisely (a narrow small-IP dip can sit between
    coarse samples) and integrate the 1-D Gaussian over the impacting sigma-interval. Returns
    a result dict WITHOUT the top-level wrapper: ``resolved`` True (with ``ip``), False (with
    ``ip_upper`` bound), or None (with ``error`` -- clones did not reach this approach)."""
    cache = {}

    def bp_at_k(k):
        v = np.asarray(refit_fn(state0 + (k * sigma) * u, u, sigma, k), float)
        if not np.all(np.isfinite(v)):
            return None
        bp = _close_approach_bplane(epoch, v, backend, base_sim, t_start, scan_window)
        if bp is not None:
            cache[round(k, 6)] = bp
        return bp

    def margin_at_k(k):
        bp = bp_at_k(k)
        return None if bp is None else (bp["b"] - bp["b_planet"])

    ks = np.linspace(-n_sigma, n_sigma, n_scan)
    pts = [(float(k), margin_at_k(k)) for k in ks]
    pts = [(k, m) for k, m in pts if m is not None]
    if len(pts) < 3:
        return {"resolved": None, "error": "too few LOV clones reach this close approach"}

    kk = [p[0] for p in pts]
    mm = [p[1] for p in pts]
    i_ca = int(np.argmin(mm))
    dk = float(ks[1] - ks[0])
    floor = max(_phi(kk[i_ca] + dk / 2) - _phi(kk[i_ca] - dk / 2), 1e-7)
    k_min, m_min = _golden_min(margin_at_k, kk[max(0, i_ca - 1)], kk[min(len(kk) - 1, i_ca + 1)])
    bp_min = cache.get(round(k_min, 6)) or bp_at_k(k_min)

    geom = {"closest_approach_sigma": float(k_min),
            "min_distance_km": float(bp_min["b"]) if bp_min else None,
            "capture_radius_km": float(bp_min["b_planet"]) if bp_min else None,
            "ca_time_iso": _iso(bp_min["t_ca"]) if (bp_min and bp_min.get("t_ca")) else None}

    if m_min >= 0.0:
        return {**geom, "resolved": False, "ip_upper": float(floor),
                "note": (f"closest b={bp_min['b']:,.0f} km vs capture {bp_min['b_planet']:,.0f} km "
                         f"at LOV {k_min:+.2f}sigma" if bp_min else "no impacting clone")}

    def boundary(direction):
        if (mm[0] if direction < 0 else mm[-1]) <= 0.0:
            return direction * n_sigma
        k_prev, k = k_min, k_min
        for _ in range(2 * n_scan):
            k += direction * dk
            if abs(k) >= n_sigma:
                return direction * n_sigma
            m = margin_at_k(k)
            if m is None:
                continue
            if m > 0.0:
                # A clone between the bracket ends can fail to reach the encounter (margin None,
                # e.g. a far/uncertain future approach); treat that as non-impacting (positive) so
                # the root find never multiplies a None -- otherwise _brentq crashes (2000 SG344).
                safe = lambda kk: (lambda v: v if v is not None else 1.0)(margin_at_k(kk))
                return _brentq(safe, min(k_prev, k), max(k_prev, k))
            k_prev = k
        return direction * n_sigma

    klo, khi = boundary(-1), boundary(+1)
    v_imp = math.sqrt(bp_min["u_inf"] ** 2 + 2 * bplane.MU_EARTH / bplane.R_EARTH_KM) if bp_min else None
    return {**geom, "resolved": True, "ip": float(min(1.0, _phi(khi) - _phi(klo))),
            "sigma_intervals": [[float(klo), float(khi)]],
            "impact_velocity_km_s": v_imp, "floor": float(floor)}


_METHOD = "LOV b-plane (find_orb/NEODyS); 1-D, may miss post-deep-encounter impacts"


def impact_probability(fit, n_sigma=5.0, n_scan=21, window_days=365.0, backend=None,
                       encounter_jd=None, search_half_days=120.0, from_jd=None):
    """Phase-2 impact PROBABILITY via the Line of Variations on the b-plane (find_orb/NEODyS
    method; Valsecchi 2003 target-plane geometry).

    Two modes:
      * ``encounter_jd`` (UTC JD) -- assess ONE specific approach within +-``search_half_days``
        (use for a known Sentry virtual-impactor date / to skip an unrelated discovery flyby);
      * otherwise SCAN: from ``from_jd`` (default: today, never before the data epoch) over
        ``window_days``, enumerate EVERY close approach of the nominal orbit, and assess each.
        The result names each approach's date so a "no impact" is never mistaken for a complete
        search -- it explicitly covers only the searched window. For a full century-scale search
        defer to JPL Sentry / ESA NEODyS.

    Returns ``{ok: True, ...}``; scan mode carries ``encounters`` (per-approach results) and the
    summed ``ip``; encounter mode carries the single result inline. ``ok: False`` only for a
    degenerate / nongrav-augmented LOV.
    """
    u, sigma, refit_fn = fit.lov_refit()
    if u is None:
        return {"ok": False, "reason": "LOV unavailable (degenerate, no live evaluator, "
                "or nongrav-augmented state -- impact probability not assessable)"}
    common = {"ok": True, "designation": fit.designation, "force_model": fit.force_model,
              "epoch_iso": fit.epoch_iso()[:19], "n_sigma": float(n_sigma),
              "n_scan": int(n_scan), "method": _METHOD}

    # DECLINE when our independent LOV orbit cannot be trusted to match a professional solution.
    # An under-constrained arc has a weighting-sensitive LOV *direction*, so OUR orbit differs
    # from JPL's and the resulting IP is unreliable. Calibrated against a random 500-object JPL
    # Sentry sample: our IP reproduced Sentry within +-20% for ZERO risk-list
    # objects and, when it emitted at all, was wrong by 1.3x to >100x (e.g. 2024 FS5 118x HIGH ->
    # a false alarm; 2023 VD3 200x LOW -> false reassurance). Sentry-list orbits are
    # under-constrained BY CONSTRUCTION -- a well-observed object is cleared off the list -- so
    # this is the regime our 1-D LOV method fundamentally cannot serve. The MPC uncertainty
    # parameter U cleanly separates the one case we still reproduce (2008 TC3: 689 obs, U=4,
    # genuine IP~1) from EVERY failure (all U>=5). So refuse any numeric IP for U>=5 (continuous
    # U>=4.5) or an undetermined U, and defer to the operational systems (Milani 2005; Del Vigna
    # 2020). Only a genuinely well-pinned orbit (U<5) is allowed to report a number.
    try:
        jds = [o["jd_utc"] for o in fit._incl_obs()]
        arc_days = (max(jds) - min(jds)) if len(jds) > 1 else 0.0
        n_obs = len(jds)
        U = fit.uncertainty_parameter()
    except Exception:
        arc_days, n_obs, U = None, fit.n_observations(), None
    # U-SHIFT (2026-09-21).  FittedOrbit._param_inverse used to scale the
    # covariance by the achieved chi2/dof with no floor, so any fit landing
    # inside its assigned sigmas had its uncertainty shrunk -- against JPL we
    # were tighter on 23 of 25 objects.  Flooring that at 1 made every sigma
    # honest and therefore RAISED U: a paired refit of 148 objects measures
    # +0.45 median (+0.41 in the populated U 0-2 band, +1.35 above U 7.5).
    #
    # This bar REFUSES a product on high U, so leaving it at 4.5 would decline
    # impact probability for objects whose determination never changed -- the
    # 2008 TC3 regression in test_impact.py is exactly that.  Shifted by the
    # shift measured AT THIS GATE'S OWN U (+0.53 at U 4.5), not by the
    # population median, because the shift grows with U.
    #
    # NB the Sentry validation behind the original 4.5 was run on the OLD U
    # scale; this shift preserves the POPULATION it selected, but the number
    # deserves re-deriving against Sentry on the new scale.
    if U is None or U >= 5.03:      # was 4.5 -- see the U-SHIFT note below
        ud = f"{U:.0f}/9" if U is not None else "undetermined"
        ad = f"{arc_days:.2f}-day arc, {n_obs} obs, " if arc_days is not None else ""
        return {**common, "assessed": False, "low_confidence": True,
                "confidence_note": (
                    f"Impact probability not estimated: this orbit is too poorly constrained "
                    f"({ad}MPC uncertainty U={ud}) for our Line-of-Variations method to match a "
                    f"professional solution. Validation against 500 JPL Sentry objects showed our "
                    f"IP is unreliable (off by ~1.3x to >100x) whenever U>=5, because the "
                    f"uncertainty direction is weighting-sensitive and our independent orbit "
                    f"differs from JPL's. For this object, consult JPL Sentry / ESA NEODyS.")}

    backend, base_sim = _backend_and_sim(fit, backend)
    state0 = np.asarray(fit._state, float)
    epoch = fit.epoch_tdb
    epoch_utc = Time(epoch, format="jd", scale="tdb").utc.jd

    if encounter_jd is not None:
        # DOMAIN GATE: a deep Earth pass between the epoch and this encounter scrambles the LOV,
        # putting the encounter beyond our 1-D method (Milani 2005 / Del Vigna 2020). The
        # intervening-approach count is intrinsic and tracks the Sentry error monotonically.
        try:
            interv = _intervening_approaches(fit, epoch_utc, float(encounter_jd) - search_half_days,
                                             backend, base_sim)
            if interv:
                return {**common, "mode": "encounter", "assessed": False, "out_of_domain": True,
                        "n_intervening": len(interv), "intervening_approaches": interv[:5],
                        "confidence_note": _out_of_domain_note(interv)}
            result = _assess_encounter(u, sigma, refit_fn, state0, epoch, backend, base_sim,
                                    float(encounter_jd) - search_half_days, 2.0 * search_half_days,
                                    n_sigma, n_scan)
        except RuntimeError as e:
            note = _coverage_decline(e, encounter_jd)
            if note is None:
                raise
            return {**common, "mode": "encounter", "assessed": False, "beyond_ephemeris": True,
                    "confidence_note": note}
        if result.get("resolved") is None:
            return {**common, "mode": "encounter", "resolved": False,
                    "note": f"could not assess the {_iso(encounter_jd)} encounter: {result['error']}"}
        return {**common, "mode": "encounter", **result}

    # SCAN mode: assess every close approach in the forward window.
    t_start = from_jd if from_jd is not None else Time.now().utc.jd
    t_start = max(float(t_start), epoch_utc - 0.5)
    searched = f"{_iso(t_start)} to {_iso(t_start + window_days)} UTC"
    try:
        cas = find_close_approaches(fit, t_start, window_days, backend, base_sim,
                                    u, sigma, refit_fn, n_sigma=n_sigma)
    except RuntimeError as e:
        note = _coverage_decline(e, t_start + window_days)
        if note is None:
            raise
        return {**common, "mode": "scan", "assessed": False, "beyond_ephemeris": True,
                "resolved": False, "encounters": [], "searched_window": searched,
                "confidence_note": note}
    scope = (f"Searched a {window_days:.0f}-day window only ({searched}) -- NOT a complete "
             f"impact search; JPL Sentry / ESA NEODyS scan ~100 years.")

    if not cas:
        return {**common, "mode": "scan", "resolved": False, "encounters": [],
                "searched_window": searched, "n_approaches": 0,
                "note": f"No close approach to Earth in the searched interval. {scope}"}

    LD = LD_KM
    encounters = []
    for ca in cas:
        # DOMAIN GATE: if deep Earth passes precede this encounter, the LOV is folded and our
        # 1-D scan is out of domain -- decline this encounter rather than report a number we
        # don't believe (Milani 2005 / Del Vigna 2020).
        interv = _intervening_approaches(fit, epoch_utc, ca["t_ca"] - 20.0, backend, base_sim)
        if interv:
            encounters.append({"approach_date": ca["t_iso"], "envelope_min_km": ca["env_min_km"],
                               "assessed": False, "out_of_domain": True, "resolved": False,
                               "n_intervening": len(interv), "intervening_approaches": interv[:5],
                               "note": _out_of_domain_note(interv)})
            continue
        r = _assess_encounter(u, sigma, refit_fn, state0, epoch, backend, base_sim,
                              ca["t_ca"] - 20.0, 40.0, n_sigma, n_scan)
        # If the scan was too sparse to integrate (a narrow return reached by only a few
        # clones), fall back to the detection geometry: when the whole region's closest pass
        # is far above Earth it is plainly no impact; only a grazing-but-unresolved return is
        # a genuine "cannot resolve".
        if r.get("resolved") is None:
            far = ca["env_min_km"] > 20 * R_EARTH_KM       # ~128,000 km: no clone can hit
            r = {"resolved": False, "ip_upper": None,
                 "note": (f"closest the uncertainty region comes is {ca['env_min_km']/LD:.0f} LD"
                          + (" — far above Earth, no impact in this solution" if far
                             else " — too narrow a return for the scan to resolve a precise IP"))}
        encounters.append({"approach_date": ca["t_iso"], "envelope_min_km": ca["env_min_km"], **r})
    ips = [e["ip"] for e in encounters if e.get("resolved")]
    return {**common, "mode": "scan", "encounters": encounters,
            "searched_window": searched, "n_approaches": len(cas),
            "resolved": bool(ips), "ip": (float(min(1.0, sum(ips))) if ips else None),
            "note": f"Assessed {len(cas)} close approach(es). {scope}"}
