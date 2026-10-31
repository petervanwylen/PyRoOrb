"""engine_api._orbit_path for an unbound orbit.

The 3D view used to get no path at all for e > 1 (and the page the "did not converge" warning).
The hyperbolic branch draws the branch out to a few perihelion distances and places the body by
the hyperbolic Kepler equation. The payload says `closed: False` so the client's ecliptic split
does not walk the wrap edge (which drew a chord from one end of the hyperbola to the other), and
carries the element dict's own mean motion `n` (sqrt(mu/(-a)^3), deg/day) so the client's date
slider -- which now solves the hyperbolic equation on e > 1 -- moves the body along the branch.
Pure function, no fit.
"""
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

from pyroorb.web import engine_api  # noqa: E402


_GAUSS_K = 0.01720209895                          # rad/day


def _elem(a, e, M, P=float("nan"), Q=float("nan")):
    n = math.degrees(_GAUSS_K / math.sqrt(abs(a) ** 3))   # what element_vector carries as `n`
    return {"a": a, "e": e, "q": a * (1.0 - e), "Q": Q, "i": 0.0, "asc_node": 0.0,
            "arg_per": 0.0, "M": M, "P": P, "n": n, "epoch": 2460000.5}


def test_unbound_path_is_drawn_and_body_sits_at_perihelion_when_M_is_zero():
    a, e = -0.2638, 6.1394                       # 3I-like
    q = a * (1.0 - e)
    out = engine_api._orbit_path(_elem(a, e, 0.0), n=90)
    assert out["path"] is not None and len(out["path"]) == 91
    x, y, z = out["position"]
    assert math.sqrt(x * x + y * y + z * z) == pytest.approx(q, rel=1e-9)
    xs, ys, zs = out["path"][0]
    assert math.sqrt(xs * xs + ys * ys + zs * zs) == pytest.approx(max(6.0, 4.0 * q), rel=1e-6)
    assert out["Q"] is None
    # OPEN curve: the two ends are far apart, and the client is told not to join them
    xe, ye, ze = out["path"][-1]
    assert math.dist((xs, ys, zs), (xe, ye, ze)) > 6.0
    assert out["closed"] is False
    # the mean motion rides along so the client's slider can advance M on the hyperbola
    assert out["n_deg_per_day"] == pytest.approx(math.degrees(_GAUSS_K / math.sqrt(0.2638 ** 3)))


def test_unbound_without_a_mean_motion_leaves_the_dot_static():
    el = _elem(-1.2723, 1.2011, 5.0)
    el["n"] = float("nan")
    out = engine_api._orbit_path(el, n=30)
    assert out["path"] is not None and out["n_deg_per_day"] is None and out["closed"] is False


def test_unbound_body_after_perihelion_is_on_the_outbound_leg():
    a, e = -1.2723, 1.2011                       # 1I-like, M = +20 deg: after perihelion
    out = engine_api._orbit_path(_elem(a, e, 20.0), n=90)
    x, y, _z = out["position"]
    assert y > 0 and math.hypot(x, y) > a * (1.0 - e)


def test_ellipse_path_unchanged():
    out = engine_api._orbit_path(_elem(2.5, 0.1, 30.0, P=2.5 ** 1.5, Q=2.75), n=90)
    assert out["path"] is not None and out["n_deg_per_day"] is not None
    assert out["path"][0] == pytest.approx(out["path"][-1])      # closed curve
    assert out["closed"] is True
