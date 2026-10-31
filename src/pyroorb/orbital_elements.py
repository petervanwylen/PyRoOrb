"""Classical orbital elements (ecliptic J2000) from a heliocentric-equatorial state.

Pure two-body math, factored out of fit_object so the fitting engine reads as fitting
logic, not element bookkeeping. `element_vector` is the single source of the displayed
elements (a/e/q/Q/i/node/argperi/M/Tp/n/P); `FittedOrbit`, the constraint machinery, the
egress writers and the web report all read it. `n_oppositions` is find_orb's
opposition-count rule; `earth_ecliptic_elements` supplies Earth's osculating elements for
the MOID calc.

Leaf module: depends only on numpy/astropy and forward_model's MU_SUN -- never on
fit_object -- so it can be imported anywhere without a cycle. fit_object re-exports these
names for backward compatibility.
"""
import math
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u

from pyroorb.forward_model import MU_SUN

# Earth's mean obliquity (J2000): the equatorial<->ecliptic rotation angle.
OBLIQ = math.radians(23.43928111)


def _eq_to_ecl(v):
    c, s = math.cos(OBLIQ), math.sin(OBLIQ)
    return np.array([v[0], c*v[1] + s*v[2], -s*v[1] + c*v[2]])


# element vector order + which entries are angles (deg) for safe differencing
ELEM_NAMES = ["a", "e", "q", "Q", "i", "asc_node", "arg_per", "M", "Tp", "n", "P"]
_ANGLE_IDX = {4, 5, 6, 7}


def element_vector(state_eq, epoch_tdb, mu=MU_SUN):
    """Classical elements from a state in the CENTRAL body's frame.

    Default ``mu=MU_SUN`` gives the usual heliocentric elements (state is
    heliocentric-equatorial). For a planetocentric orbit, pass the state already
    relative to the planet and ``mu=GM_planet``; the two-body math is identical.
    """
    r = _eq_to_ecl(state_eq[:3]); v = _eq_to_ecl(state_eq[3:])
    rn = float(np.linalg.norm(r)); vn2 = float(np.dot(v, v))
    a = 1.0 / (2.0/rn - vn2/mu)
    h = np.cross(r, v); hn = float(np.linalg.norm(h))
    nvec = np.cross([0, 0, 1.0], h); nn = float(np.linalg.norm(nvec))
    evec = np.cross(v, h)/mu - r/rn; e = float(np.linalg.norm(evec))
    q = a*(1 - e); Q = a*(1 + e)
    i = math.degrees(math.acos(max(-1, min(1, h[2]/hn))))
    Om = math.degrees(math.atan2(nvec[1], nvec[0])) % 360 if nn > 1e-12 else 0.0
    if nn > 1e-12 and e > 1e-12:
        w = math.degrees(math.acos(max(-1, min(1, float(np.dot(nvec, evec))/(nn*e)))))
        if evec[2] < 0:
            w = 360 - w
    else:
        w = 0.0
    nu = math.acos(max(-1, min(1, float(np.dot(evec, r))/(e*rn)))) if e > 1e-12 else 0.0
    if float(np.dot(r, v)) < 0:
        nu = 2*math.pi - nu
    if e < 1.0 and a > 0:                  # elliptic: standard E, M
        E = 2*math.atan2(math.sqrt(1-e)*math.sin(nu/2), math.sqrt(1+e)*math.cos(nu/2))
        M = (E - e*math.sin(E))
        n_mean = math.sqrt(mu/a**3)                                  # rad/day
        P = 2*math.pi/n_mean/365.25                                  # years
        Tp = epoch_tdb - M/n_mean                                    # JD (perihelion)
        M_deg = math.degrees(M) % 360
    elif e > 1.0 and a < 0:
        # Hyperbolic (1I, 2I, 3I): M = e sinh F - F is NOT an angle -- signed, negative before
        # perihelion, unbounded -- so it is never wrapped. Tp and the mean motion are as well
        # defined as for an ellipse; only the period and the aphelion do not exist.
        nu_s = nu - 2*math.pi if nu > math.pi else nu
        F = 2*math.atanh(math.sqrt((e-1)/(e+1))*math.tan(nu_s/2))
        M = e*math.sinh(F) - F
        n_mean = math.sqrt(mu/(-a)**3)
        P = Q = float("nan")
        Tp = epoch_tdb - M/n_mean
        M_deg = math.degrees(M)
    else:                                  # parabolic to rounding / non-convergent: don't crash
        M_deg = n_mean = P = Tp = float("nan")
    return np.array([a, e, q, Q, i, Om, w, M_deg, Tp, math.degrees(n_mean), P])


def _ecl_to_eq(v):
    """Inverse of `_eq_to_ecl`: rotate an ecliptic vector back to the equatorial frame."""
    c, s = math.cos(OBLIQ), math.sin(OBLIQ)
    return np.array([v[0], c*v[1] - s*v[2], s*v[1] + c*v[2]])


def state_eq_from_elements(a, e, i_deg, node_deg, argper_deg, M_deg, mu=MU_SUN):
    """Heliocentric-EQUATORIAL cartesian state from classical ECLIPTIC elements -- the exact
    inverse of `element_vector` (same obliquity, same angle conventions). Used by the linkage
    search to seed a trial orbit at a chosen (semi-major axis, mean anomaly) while holding the
    other elements fixed: the period/phase pair is the under-determined direction across a
    multi-revolution gap, so gridding it in element space is the natural search axis (a state-
    space perturbation cannot hold e/i/node/argper fixed while sweeping only a and M)."""
    i, Om, w, M = map(math.radians, (i_deg, node_deg, argper_deg, M_deg))
    E = M                                           # solve Kepler M = E - e sinE (Newton)
    for _ in range(80):
        dE = (E - e*math.sin(E) - M) / (1 - e*math.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break
    nu = 2*math.atan2(math.sqrt(1+e)*math.sin(E/2), math.sqrt(1-e)*math.cos(E/2))
    p = a*(1 - e*e); r = p/(1 + e*math.cos(nu))
    r_pf = np.array([r*math.cos(nu), r*math.sin(nu), 0.0])                    # perifocal position
    v_pf = math.sqrt(mu/p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])  # perifocal velocity
    cO, sO, ci, si, cw, sw = (math.cos(Om), math.sin(Om), math.cos(i),
                              math.sin(i), math.cos(w), math.sin(w))
    R = np.array([[cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],   # perifocal -> ecliptic
                  [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
                  [sw*si,             cw*si,             ci]])
    return np.concatenate([_ecl_to_eq(R @ r_pf), _ecl_to_eq(R @ v_pf)])


def earth_ecliptic_elements(epoch_tdb):
    """Earth's osculating heliocentric ecliptic elements at `epoch_tdb` (for MOID)."""
    t = Time(epoch_tdb, format="jd", scale="tdb")
    p, v = get_body_barycentric_posvel("earth", t)
    ps, vs = get_body_barycentric_posvel("sun", t)
    pos = (p - ps).xyz.to(u.au).value
    vel = (v - vs).xyz.to(u.au/u.day).value
    ev = element_vector(np.concatenate([pos, vel]), epoch_tdb)
    return dict(zip(ELEM_NAMES, ev.tolist()))


# find_orb's opposition rule: a new opposition opens when an observation falls
# more than this many days after the FIRST observation of the current one (a
# span from the group's start, not a gap between neighbours).  It is the count
# every fit reports.  The ITF, the MPC rules and the linker split oppositions
# with other values and other algorithms.
FINDORB_OPPOSITION_SPAN_DAYS = 238.0


def n_oppositions(jds, span=FINDORB_OPPOSITION_SPAN_DAYS):
    """find_orb's rule: new opposition when the span from the opposition's first
    observation exceeds :data:`FINDORB_OPPOSITION_SPAN_DAYS`."""
    js = sorted(jds); count, anchor = 1, js[0]
    for t in js:
        if t - anchor > span:
            count += 1; anchor = t
    return count
