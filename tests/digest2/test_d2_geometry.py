import math

from pyroorb.digest2.geometry import ec_rotate, gc_fit, gc_pos, gc_rms, se2000


def test_se2000_returns_unit_length_direction_times_distance():
    (x, y, z), soe, coe = se2000(59938.4)
    r = math.sqrt(x * x + y * y + z * z)
    # Earth-Sun distance is close to 1 AU year-round
    assert 0.98 < r < 1.02
    assert abs(soe ** 2 + coe ** 2 - 1.0) < 1e-12


def test_ec_rotate_is_orthogonal_and_preserves_length():
    v = (0.3, -0.7, 0.5)
    soe, coe = math.sin(0.4), math.cos(0.4)
    r = ec_rotate(v, soe, coe)
    len_before = sum(c * c for c in v)
    len_after = sum(c * c for c in r)
    assert abs(len_before - len_after) < 1e-15
    # x is untouched by an ecliptic rotation about the x axis
    assert r[0] == v[0]


def test_gc_fit_two_points_linear_motion():
    mjd = [59000.0, 59000.1]
    sphr = [(1.0, 0.2), (1.001, 0.2005)]
    fit = gc_fit(mjd, sphr)
    ra, dec = gc_pos(fit, 59000.05)
    # midpoint should be close to the midpoint of the two positions
    assert abs(ra - 1.0005) < 1e-6
    assert abs(dec - 0.20025) < 1e-6
    assert gc_rms(fit) < 1e-6  # a 2-point fit has zero residual by construction


def test_gc_fit_three_collinear_points_low_rms():
    mjd = [59000.0, 59000.05, 59000.1]
    sphr = [(1.0, 0.2), (1.0005, 0.20025), (1.001, 0.2005)]
    fit = gc_fit(mjd, sphr)
    assert gc_rms(fit) < 0.01  # small residual; points are linear in (ra,dec), not exactly geodesic
