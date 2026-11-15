"""Two-body positions of a whole catalogue at once: one vectorised conic solve for every orbit.

Every orbit is held in PERIHELION form -- q, e, i, node, argument of perihelion, perihelion time
T -- because that one form covers ellipses, parabolas and hyperbolas alike: a catalogue of
asteroids and comets is propagated in a single call with no branch per orbit type. The state at
time t comes from the universal-variable Kepler equation written from perihelion (where the
radial velocity is zero):

    sqrt(mu) (t - T) = q chi + e chi^3 S(alpha chi^2),      alpha = (1 - e) / q = 1 / a

solved for chi by safeguarded Newton (each step clamped to a bracket that always holds the root,
so a near-parabolic comet or a hyperbola decades from perihelion converges as an asteroid does).
A Newton step of a monotonic function needs no derivative beyond r itself: dF/dchi = r.

This is the quick, approximate layer of the sky check (Sun-only gravity, no planets): it decides
which objects can be in a field, never where exactly they are. Numerical integration does that.
"""
import math

import numpy as np

from pyroorb.forward_model import MU_SUN
from pyroorb.orbital_elements import OBLIQ

_SQRT_MU = math.sqrt(MU_SUN)
_SERIES_Z = 0.1          # |z| below this: Stumpff functions from their series (no cancellation)
_Z_FLOOR = -4.9e5        # z = -(700)^2: cosh/sinh of sqrt(-z) stay finite
_MAX_ITER = 60
_TOL_CHI = 1e-13         # relative change in chi at convergence


def stumpff(z):
    """Stumpff functions C(z), S(z) for an array ``z`` (series near zero, closed forms elsewhere)."""
    z = np.asarray(z, float)
    C = np.empty_like(z)
    S = np.empty_like(z)
    small = np.abs(z) < _SERIES_Z
    pos = (z >= _SERIES_Z)
    neg = (z <= -_SERIES_Z)
    if small.any():
        zs = z[small]
        C[small] = (1/2 - zs/24 + zs**2/720 - zs**3/40320 + zs**4/3628800
                    - zs**5/479001600)
        S[small] = (1/6 - zs/120 + zs**2/5040 - zs**3/362880 + zs**4/39916800
                    - zs**5/6227020800)
    if pos.any():
        s = np.sqrt(z[pos])
        C[pos] = (1 - np.cos(s)) / z[pos]
        S[pos] = (s - np.sin(s)) / s**3
    if neg.any():
        s = np.sqrt(-np.maximum(z[neg], _Z_FLOOR))
        C[neg] = (np.cosh(s) - 1) / s**2
        S[neg] = (np.sinh(s) - s) / s**3
    return C, S


def orientation(incl_deg, node_deg, peri_deg):
    """Unit vectors P (to perihelion) and Q (90 deg ahead in the orbit plane), (n, 3) each, in
    the J2000 EQUATORIAL frame, for ecliptic-J2000 angles (the MPC's element frame)."""
    i, O, w = (np.radians(np.asarray(x, float)) for x in (incl_deg, node_deg, peri_deg))
    ci, si, cO, sO, cw, sw = np.cos(i), np.sin(i), np.cos(O), np.sin(O), np.cos(w), np.sin(w)
    P = np.stack([cO*cw - sO*sw*ci, sO*cw + cO*sw*ci, sw*si], axis=-1)
    Q = np.stack([-cO*sw - sO*cw*ci, -sO*sw + cO*cw*ci, cw*si], axis=-1)
    ce, se = math.cos(OBLIQ), math.sin(OBLIQ)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return P @ rot.T, Q @ rot.T


def angles(P, Q):
    """Ecliptic-J2000 (i, node, argument of perihelion) in degrees from equatorial P, Q: the
    inverse of :func:`orientation`."""
    ce, se = math.cos(OBLIQ), math.sin(OBLIQ)
    to_ecl = np.array([[1.0, 0.0, 0.0], [0.0, ce, se], [0.0, -se, ce]])
    P = np.atleast_2d(P) @ to_ecl.T
    Q = np.atleast_2d(Q) @ to_ecl.T
    h = np.cross(P, Q)
    incl = np.degrees(np.arccos(np.clip(h[:, 2], -1.0, 1.0)))
    node = np.degrees(np.arctan2(h[:, 0], -h[:, 1])) % 360.0
    n = np.stack([np.cos(np.radians(node)), np.sin(np.radians(node)), np.zeros(len(P))], 1)
    peri = np.degrees(np.arctan2((np.cross(n, P) * h).sum(1), (n * P).sum(1))) % 360.0
    return incl, node, peri


def perihelion_time(a, e, mean_anomaly_deg, epoch):
    """Perihelion time T of an ELLIPTIC orbit given (a, e, M at epoch): the passage nearest the
    epoch (M wrapped to [-180, 180) deg), so a state near the epoch needs the least propagation."""
    n = _SQRT_MU / np.asarray(a, float) ** 1.5                     # rad / day
    M = np.radians((np.asarray(mean_anomaly_deg, float) + 180.0) % 360.0 - 180.0)
    return np.asarray(epoch, float) - M / n


def _solve_chi(q, e, dt):
    """chi with sqrt(mu) dt = q chi + e chi^3 S(alpha chi^2), for arrays q, e, dt (dt any sign)."""
    alpha = (1.0 - e) / q
    sgn = np.where(dt < 0, -1.0, 1.0)
    tau = _SQRT_MU * np.abs(dt)                                    # solve for |dt|, then sign
    lo = np.zeros_like(tau)
    # Upper bracket. S(z) > 0 everywhere, so F(chi) >= q chi - tau: chi <= tau / q always.
    # Ellipse: |dt| <= half a period after the caller's reduction, so the eccentric anomaly is
    # <= pi and chi = sqrt(a) E <= pi sqrt(a). Parabola / hyperbola: S(z) >= 1/6 for z <= 0, so
    # F(chi) >= q chi + e chi^3 / 6 - tau, whose root (Cardano) is tighter for a long |dt|;
    # rounding can put that root a hair low, so it is kept only where F there is >= 0.
    hi = tau / q
    ell = alpha > 0
    hi = np.where(ell, np.minimum(hi, math.pi / np.sqrt(np.where(ell, alpha, 1.0))), hi)
    if (~ell).any():
        es = np.where(ell, 1.0, e)
        p, r = 6.0 * q / es, 6.0 * tau / es
        disc = np.sqrt((r / 2) ** 2 + (p / 3) ** 3)
        cubic = (np.cbrt(r / 2 + disc) + np.cbrt(r / 2 - disc)) * (1.0 + 1e-12)
        C, S = stumpff(alpha * cubic * cubic)
        holds = (~ell) & (q * cubic + e * cubic**3 * S - tau >= 0) & (cubic < hi)
        hi = np.where(holds, cubic, hi)
    # Ellipse start: chi = sqrt(a) E with Danby's E0 = M + 0.85 e (M in [0, pi] here), which
    # takes Newton to 1e-13 in a few steps at any eccentricity below 1.
    sqa = 1.0 / np.sqrt(np.where(ell, alpha, 1.0))
    M = tau * np.where(ell, alpha, 0.0) / sqa                     # n |dt| = tau / a^1.5
    chi = np.where(ell, sqa * (M + 0.85 * e), 0.5 * hi)
    chi = np.clip(chi, lo, hi)
    for _ in range(_MAX_ITER):
        z = alpha * chi * chi
        C, S = stumpff(z)
        F = q * chi + e * chi**3 * S - tau
        r_ = q + e * chi * chi * C                                 # dF/dchi = r > 0
        lo = np.where(F < 0, chi, lo)
        hi = np.where(F > 0, chi, hi)
        step = F / r_
        new = chi - step
        # strict: a converged step lands ON the bracket end it just set, and is fine there
        bad = ~np.isfinite(new) | (new < lo) | (new > hi)
        new = np.where(bad, 0.5 * (lo + hi), new)
        done = np.abs(new - chi) <= _TOL_CHI * np.maximum(np.abs(new), 1e-12)
        chi = new
        if done.all():
            break
    return sgn * chi, alpha


def state_at(q, e, T, P, Q, t):
    """Heliocentric equatorial position and velocity (n, 3) each, au and au/day, at time ``t``
    (scalar or (n,) array, same time scale as ``T``) of the orbits (q, e, T, P, Q)."""
    q = np.asarray(q, float)
    e = np.asarray(e, float)
    dt = np.asarray(t, float) - np.asarray(T, float)
    dt = np.broadcast_to(dt, q.shape).astype(float)
    ell = e < 1.0
    if ell.any():                                                  # whole revolutions drop out
        a = q[ell] / (1.0 - e[ell])
        period = 2.0 * math.pi * a ** 1.5 / _SQRT_MU
        dt[ell] = dt[ell] - period * np.round(dt[ell] / period)
    chi, alpha = _solve_chi(q, e, dt)
    z = alpha * chi * chi
    C, S = stumpff(z)
    r = q + e * chi * chi * C
    f = 1.0 - chi * chi * C / q
    g = dt - chi**3 * S / _SQRT_MU
    fdot = _SQRT_MU / (r * q) * (z * S - 1.0) * chi
    gdot = 1.0 - chi * chi * C / r
    vp = np.sqrt(MU_SUN * (1.0 + e) / q)                           # perihelion speed
    pos = (f * q)[:, None] * P + (g * vp)[:, None] * Q
    vel = (fdot * q)[:, None] * P + (gdot * vp)[:, None] * Q
    return pos, vel


def elliptic_elements(pos, vel, t):
    """Perihelion-form elements of ELLIPTIC heliocentric equatorial states at time ``t``:
    ``(q, e, T, P, Q)`` with P, Q equatorial unit vectors, the inverse of :func:`state_at`.
    ``T`` is the perihelion passage nearest ``t``. Rows that are not ellipses (e >= 1) come
    back as NaN; a circular orbit (e < 1e-10) takes perihelion at its current position."""
    pos = np.atleast_2d(np.asarray(pos, float))
    vel = np.atleast_2d(np.asarray(vel, float))
    r = np.linalg.norm(pos, axis=1)
    h = np.cross(pos, vel)
    hn = np.linalg.norm(h, axis=1)
    ev = np.cross(vel, h) / MU_SUN - pos / r[:, None]
    e = np.linalg.norm(ev, axis=1)
    circ = e < 1e-10
    P = np.where(circ[:, None], pos / r[:, None], ev / np.where(circ, 1.0, e)[:, None])
    Q = np.cross(h / hn[:, None], P)
    q = hn * hn / (MU_SUN * (1.0 + e))
    nu = np.arctan2((Q * pos).sum(1) / r, (P * pos).sum(1) / r)
    with np.errstate(invalid="ignore", divide="ignore"):
        E = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(nu / 2.0))
        a = q / (1.0 - e)
        M = E - e * np.sin(E)
        T = np.asarray(t, float) - M / (_SQRT_MU / a ** 1.5)
    bad = ~(e < 1.0)
    q, e, T = (np.where(bad, np.nan, x) for x in (q, e, T))
    P = np.where(bad[:, None], np.nan, P)
    Q = np.where(bad[:, None], np.nan, Q)
    return q, e, T, P, Q
