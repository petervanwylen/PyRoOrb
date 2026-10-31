"""Hyperbolic elements (e > 1): mean anomaly, mean motion and perihelion time are DEFINED for an
unbound orbit -- only the period and aphelion are not. `element_vector` used to return NaN for all
of them, which is what raised the web UI's "did not converge to a physical orbit" on every genuine
interstellar object.

The check is independent of the formula: start a body AT perihelion, integrate the two-body problem
numerically for dt days (plain RK4, no Kepler solve), and require element_vector to recover the
perihelion time it started from.
"""
import math

import numpy as np

from pyroorb.orbital_elements import element_vector, _ecl_to_eq

MU = 0.01720209895 ** 2
T_PERI = 2460000.5


def _accel(s):
    r3 = (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) ** 1.5
    return np.array([s[3], s[4], s[5], -MU * s[0] / r3, -MU * s[1] / r3, -MU * s[2] / r3])


def _propagate(s, t, h=0.05):
    n = max(1, int(abs(t) / h)); h = t / n
    for _ in range(n):
        k1 = _accel(s); k2 = _accel(s + 0.5 * h * k1)
        k3 = _accel(s + 0.5 * h * k2); k4 = _accel(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def _state(q, e, dt):
    """Equatorial state dt days after perihelion of an orbit lying in the ecliptic."""
    s0 = np.array([q, 0.0, 0.0, 0.0, math.sqrt(MU / (q * (1 + e))) * (1 + e), 0.0])
    s = _propagate(s0, dt)
    return np.concatenate([_ecl_to_eq(s[:3]), _ecl_to_eq(s[3:])])


CASES = [("1I-like", 0.2553, 1.2011), ("2I-like", 2.0066, 3.3565),
         ("3I-like", 1.3564, 6.1394), ("near-parabolic", 2.0, 1.0005)]


def test_hyperbolic_perihelion_time_matches_integration():
    for name, q, e in CASES:
        for dt in (-40.0, 3.0, 40.0):
            a, e_r, q_r, Q, _i, _node, _peri, M, Tp, n, P = element_vector(
                _state(q, e, dt), T_PERI + dt)
            assert a < 0 and abs(e_r - e) < 1e-6 and abs(q_r - q) < 1e-6, name
            assert abs(Tp - T_PERI) < 1e-5, (name, dt, Tp)
            assert math.isfinite(M) and math.copysign(1.0, M) == math.copysign(1.0, dt), (name, dt)
            assert n > 0 and math.isnan(P) and math.isnan(Q), name


def test_hyperbolic_mean_anomaly_is_not_wrapped():
    """A hyperbolic M is not an angle: 3I-like 40 d before perihelion is about -290 deg, and
    wrapping it to +70 would put the body AFTER perihelion."""
    M = element_vector(_state(1.3564, 6.1394, -40.0), T_PERI - 40.0)[7]
    assert M < -180.0


def test_ellipse_unchanged():
    q, e = 1.5, 0.3
    a0 = q / (1 - e)
    for dt in (-100.0, 50.0):
        a, e_r, _q, Q, _i, _n, _w, M, Tp, _nm, P = element_vector(_state(q, e, dt), T_PERI + dt)
        assert abs(a - a0) < 1e-6 and abs(e_r - e) < 1e-6 and abs(Q - a0 * (1 + e)) < 1e-6
        assert 0.0 <= M < 360.0 and abs(P - a0 ** 1.5) < 1e-3
        period_days = P * 365.25
        off = (Tp - T_PERI) % period_days
        assert min(off, period_days - off) < 1e-4
