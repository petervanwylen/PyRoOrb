"""The bound COMPARATOR behind the unbound rules: where bound orbits are looked for, and
what the arc's evidence is counted in. Pure geometry / arithmetic -- no ephemeris, no fit."""
import math

import numpy as np
import pytest

import pyroorb.hyperbolic as hyp

K = 0.01720209895


def _circ(a, lon_deg, t):
    """Heliocentric position on a circular ecliptic-plane orbit of radius `a`, `t` days on."""
    lam = math.radians(lon_deg) + K / a ** 1.5 * t
    return np.array([a * math.cos(lam), a * math.sin(lam), 0.0])


def _two_sightlines(a, lon_deg, dt, earth_lon=0.0):
    o1, o2 = _circ(1.0, earth_lon, 0.0), _circ(1.0, earth_lon, dt)
    p1, p2 = _circ(a, lon_deg, 0.0), _circ(a, lon_deg, dt)
    u1, u2 = (p1 - o1) / np.linalg.norm(p1 - o1), (p2 - o2) / np.linalg.norm(p2 - o2)
    return o1, u1, o2, u2, float(np.linalg.norm(p1 - o1))


@pytest.mark.parametrize("a, lon, dt", [(2.6, 20.0, 2.0), (1.3, 35.0, 1.0), (36.4, 10.0, 1.17),
                                         (45.0, -15.0, 29.0)])
def test_the_true_range_lies_inside_the_admissible_region(a, lon, dt):
    """Whatever else the seeds do, a real bound object's own range must be one the scan calls
    bound, and a seed must land near it -- 2010 LP68 (a 36 au TNO over 1.17 d) is bound only
    between ~30 and ~48 au, and a fixed grid stepping 25 -> 50 found nothing there."""
    o1, u1, o2, u2, rho_true = _two_sightlines(a, lon, dt)
    seeds = hyp.bound_seeds(0.0, o1, u1, dt, o2, u2)
    assert seeds
    rho = np.array([s[2] for s in seeds])
    assert np.min(np.abs(np.log(rho / rho_true))) < math.log(1.8)      # seeds <= 3x apart
    for t_mid, state, _rho in seeds:
        assert t_mid == pytest.approx(dt / 2)
        r, v2 = np.linalg.norm(state[:3]), float(state[3:] @ state[3:])
        assert 0.5 * v2 - hyp.GM_SUN_AU / r < 0          # every seed IS a bound orbit


def test_the_seed_at_the_true_range_is_close_to_the_true_orbit():
    o1, u1, o2, u2, rho_true = _two_sightlines(2.6, 20.0, 2.0)
    mid, v, energy = hyp._slowest_through(np.array([rho_true]), 0.0, o1, u1, 2.0, o2, u2)
    truth = _circ(2.6, 20.0, 1.0)
    assert np.linalg.norm(mid[0] - truth) < 5e-3      # the nearest-point chord, not the true one
    assert np.linalg.norm(v[0]) == pytest.approx(K / math.sqrt(2.6), rel=0.02)


def test_no_seeds_without_a_baseline_or_an_admissible_range():
    o1, u1, o2, u2, _ = _two_sightlines(2.6, 20.0, 2.0)
    assert hyp.bound_seeds(5.0, o1, u1, 5.0, o2, u2) == []
    # two sightlines 40 deg apart a day later: nothing bound moves that fast at any range scanned
    far = np.array([math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0])
    assert hyp.bound_seeds(0.0, o1, u1, 1.0, o2, far, scan=(0.5, 300.0)) == []


def test_the_comparator_says_why_it_has_no_candidates():
    """`_admissible_bound_candidates` came back empty for five different reasons and
    the record held only the seed COUNT -- "scanned to 300 au and the geometry admits no bound
    orbit" and "never scanned" were the same 0. The status is the third return value. Only the
    exits BEFORE any integration are driven here; `ok` is asserted on real fits in
    test_unbound_door."""
    from types import SimpleNamespace
    import pyroorb.fit_object as fo

    def _ev(t, o, u):
        ra = [math.degrees(math.atan2(v[1], v[0])) for v in u]
        dec = [math.degrees(math.asin(v[2])) for v in u]
        return SimpleNamespace(times_tdb=np.array(t, float), Robs=np.array(o, float),
                               ra=np.array(ra), dec=np.array(dec), t0=float(t[0]))

    o1, u1, o2, u2, _ = _two_sightlines(2.6, 20.0, 2.0)
    ev = _ev([0.0, 2.0], [o1, o2], [u1, u2])
    assert fo._admissible_bound_candidates(ev, np.array([True, False]), 0.0)[2] == "too-few-obs"
    long = _ev([0.0, hyp.BOUND_SEED_MAX_ARC_DAYS + 1.0], [o1, o2], [u1, u2])
    assert fo._admissible_bound_candidates(long, np.array([True, True]), 0.0)[2] == "arc-too-long"
    # 40 deg of sky in a quarter of an hour: no bound orbit at any range in the scan
    far = np.array([math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0])
    fast = _ev([0.0, 0.01], [o1, o2], [u1, far])
    out, n, status = fo._admissible_bound_candidates(fast, np.array([True, True]), 0.0)
    assert (out, n, status) == ([], 0, "no-admissible-range")


def test_count_station_nights_counts_station_nights():
    t = [0.00, 0.01, 0.02, 0.30, 0.31, 1.00, 1.01]
    assert hyp.count_station_nights(t, ["F51"] * 7) == 2          # 0.30 is the same station-night
    assert hyp.count_station_nights(t, ["F51", "G96", "F51", "703", "703", "F51", "G96"]) == 5
    assert hyp.count_station_nights([], []) == 0


def test_the_per_observation_statistic_is_kept_only_as_a_diagnostic():
    """2010 LP68's first two nights, measured: 17 obs, chi2 1.967 free (e = 458052) against 8.79 for
    the best bound orbit -- which is its OWN orbit's range. Counted per observation that is 97 and a
    9.9 sigma refusal. Now the rules decide on the station-night model's Delta chi2
    (3.57 there, 1.9 sigma: `test_hyperbolic_refused`, `test_nightmodel`), and this number rides
    on the record as `dchi2_raw` for reading old records against."""
    assert hyp.bound_dchi2(1.967, 8.7912, 34) == pytest.approx(97.1, abs=0.5)
    assert hyp.bound_dchi2(1.0, 50.0, 6) is None                            # 6 - 6 = 0 dof: no number
    assert not hasattr(hyp, "effective_resid") and not hasattr(hyp, "MEASUREMENTS_PER_STATION_NIGHT")
