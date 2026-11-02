"""Target-plane (b-plane) geometry of a planetary close approach.

The foundation for robust impact probability and for keyholes / resonant returns
(Valsecchi, Milani, Gronchi & Chesley 2003, A&A 408, 1179; logic openly ported,
no source copied). The b-plane is the plane through the
planet's centre orthogonal to the incoming asymptotic relative velocity U; an
asteroid's crossing point (xi, zeta) on it encodes the encounter:

  - xi   -- the local minimum-orbit-intersection distance (MOID) direction;
  - zeta -- the timing direction (how early/late the asteroid arrives);
  - b = sqrt(xi^2 + zeta^2) is the impact parameter, and the asteroid hits the
    planet when b < b_planet, the gravitationally-focused capture radius.

We work directly from the PLANETOCENTRIC state vector (r, v) at close approach --
which the engine's propagator already yields -- instead of the Opik element
formulas, so the geometry is exact and frame-consistent with the integrator.

Units are physical (km, km/s); GM in km^3/s^2. Everything here is pure geometry on
a supplied state, so this module imports nothing from the engine.
"""
import math
import numpy as np

MU_EARTH = 398600.4418            # GM_earth, km^3/s^2 (DE440)
R_EARTH_KM = 6378.137             # WGS-84 equatorial radius


def _u(v):
    n = math.sqrt(float(np.dot(v, v)))
    return (v / n), n


def bplane_crossing(r_geo_km, v_geo_km_s, v_planet_helio_km_s,
                    mu=MU_EARTH, r_planet=R_EARTH_KM):
    """b-plane crossing of a hyperbolic close approach, from the planetocentric position
    ``r_geo_km`` and velocity ``v_geo_km_s`` at (near) closest approach and the planet's
    heliocentric velocity ``v_planet_helio_km_s`` (which fixes the timing axis).

    Returns a dict: ``u_inf`` (km/s, unperturbed relative speed), ``b`` (km, impact
    parameter), ``b_planet`` (km, focused capture radius), ``xi``/``zeta`` (km, b-plane
    coordinates), ``impacts`` (b < b_planet), ``min_distance`` (km, geometric perigee of
    the hyperbola), and the frame unit vectors. Returns ``None`` if the object is bound to
    the planet (v^2 < 2mu/r -- not a hyperbolic flyby, outside this approximation)."""
    r_vec = np.asarray(r_geo_km, float)
    v_vec = np.asarray(v_geo_km_s, float)
    r = math.sqrt(float(r_vec @ r_vec))
    v2 = float(v_vec @ v_vec)

    u_inf2 = v2 - 2.0 * mu / r                       # vis-viva relative speed at infinity
    if u_inf2 <= 0.0:
        return None                                  # temporarily captured / bound
    u_inf = math.sqrt(u_inf2)

    h_vec = np.cross(r_vec, v_vec)                   # specific angular momentum (conserved)
    h = math.sqrt(float(h_vec @ h_vec))
    b = h / u_inf                                    # impact parameter b = L / U_inf

    # focused capture radius b_planet = R*sqrt(1 + v_esc^2 / U_inf^2)
    b_planet = r_planet * math.sqrt(1.0 + (2.0 * mu / r_planet) / u_inf2)
    # geometric perigee of the hyperbola (independent check; q < R == impact)
    e = math.sqrt(1.0 + (u_inf2 * h * h) / (mu * mu))     # hyperbolic eccentricity
    q = (h * h / mu) / (1.0 + e)                          # perigee distance

    # incoming asymptote direction U_hat (Valsecchi's U): from the eccentricity/ang-mom frame
    hhat = h_vec / h
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r       # eccentricity vector (-> perigee)
    ehat = e_vec / math.sqrt(float(e_vec @ e_vec))
    qhat = np.cross(hhat, ehat)                           # in-plane, 90 deg ahead of perigee
    u_hat = (ehat + math.sqrt(max(e * e - 1.0, 0.0)) * qhat) / e   # incoming asymptote unit
    if float(u_hat @ v_vec) < 0.0:                        # keep it along the direction of travel
        u_hat = -u_hat

    b_vec = np.cross(u_hat, h_vec) / u_inf                # impact-parameter vector (in b-plane)

    # b-plane frame: xi along (U x V_planet) = MOID direction; zeta = U x xi = timing axis
    vp = np.asarray(v_planet_helio_km_s, float)
    xi_hat, xn = _u(np.cross(u_hat, vp))
    zeta_hat = np.cross(u_hat, xi_hat)
    xi = float(b_vec @ xi_hat)
    zeta = float(b_vec @ zeta_hat)

    return {
        "u_inf": u_inf, "b": b, "b_planet": b_planet,
        "xi": xi, "zeta": zeta, "impacts": bool(b < b_planet),
        "min_distance": q, "eccentricity": e,
        "u_hat": u_hat, "xi_hat": xi_hat, "zeta_hat": zeta_hat,
    }


# Resonant-return keyhole geometry (the b-plane circle for an h/k return) is Phase 3 and lives
# in keyholes.py (resonant_circle + the numerical find_keyholes engine), kept separate from this
# Phase-2 geometry. keyholes imports MU_EARTH / R_EARTH_KM from here.
