"""The revolution-count bound comparator (`boundbridge.py`): the multi-revolution Lambert
enumeration that seeds bound orbits across a multi-year gap, and the statuses
`fit_object._bridged_bound_candidates` reports when it does not run. Pure two-body arithmetic --
no ephemeris, no fit; the real astrometry is in test_unbound_bridge."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import pyroorb.boundbridge as bb
from pyroorb.twobody import kepler_propagate

MU = bb.MU_SUN


def _orbit_state(a, e, inc_deg, M_deg=0.0):
    """Heliocentric state of an ellipse (perihelion on +x, node on the x axis) at mean anomaly M."""
    E = math.radians(M_deg)
    for _ in range(50):
        E -= (E - e * math.sin(E) - math.radians(M_deg)) / (1 - e * math.cos(E))
    r = a * (1 - e * math.cos(E))
    x, y = a * (math.cos(E) - e), a * math.sqrt(1 - e * e) * math.sin(E)
    n = math.sqrt(MU / a ** 3)
    vx, vy = -a * n * math.sin(E) * a / r, a * n * math.sqrt(1 - e * e) * math.cos(E) * a / r
    ci, si = math.cos(math.radians(inc_deg)), math.sin(math.radians(inc_deg))
    return np.array([x, y * ci, y * si, vx, vy * ci, vy * si])


@pytest.mark.parametrize("a, e, inc, dt", [(2.7, 0.15, 5.0, 3000.0), (1.1, 0.4, 12.0, 4000.0),
                                            (5.2, 0.05, 2.0, 1500.0), (40.0, 0.2, 8.0, 800.0)])
def test_every_transfer_reaches_the_far_point_and_the_true_one_is_among_them(a, e, inc, dt):
    """Lagrange's equation, both branches, both ways round, every revolution count: each solution
    must actually arrive (checked by an independent universal-variable propagation), and the
    object's own velocity -- with its own revolution count -- must be one of them."""
    s0 = _orbit_state(a, e, inc, 30.0)
    s1 = kepler_propagate(s0, dt)
    sol = bb.lambert_multirev(s0[None, :3], s1[None, :3], dt)
    assert sol is not None and sol["v1"].shape[0] >= 1
    for v1 in sol["v1"]:
        arrive = kepler_propagate(np.r_[s0[:3], v1], dt)
        assert np.linalg.norm(arrive[:3] - s1[:3]) < 1e-7
    k = int(np.argmin(np.linalg.norm(sol["v1"] - s0[3:], axis=1)))
    assert np.linalg.norm(sol["v1"][k] - s0[3:]) < 1e-9
    n_true = int(dt / (2 * math.pi * math.sqrt(a ** 3 / MU)))
    assert int(sol["N"][k]) == n_true
    assert sol["a"][k] == pytest.approx(a, rel=1e-6)


def test_a_multi_year_gap_offers_many_revolution_counts():
    """What the walk from the hyperbola could not do: the same two points joined with every N the
    geometry allows, each a distinct bound orbit (107474's 39-year gap takes N = 0..~15)."""
    s0 = _orbit_state(2.7, 0.15, 5.0, 30.0)
    s1 = kepler_propagate(s0, 14000.0)
    sol = bb.lambert_multirev(s0[None, :3], s1[None, :3], 14000.0)
    assert set(range(0, 9)) <= set(sol["N"].tolist())
    assert np.all(sol["a"] > 0)


def test_the_vectorised_propagator_matches_the_universal_variable_one():
    s0 = _orbit_state(3.1, 0.62, 20.0, 200.0)
    dts = np.array([[-900.0, -1.0, 0.0, 17.0, 5000.0]])
    pos, vel = bb.kepler_states(s0[None, :3], s0[None, 3:], dts)
    for j, dt in enumerate(dts[0]):
        ref = kepler_propagate(s0, float(dt))
        assert np.linalg.norm(pos[0, j] - ref[:3]) < 1e-10
        assert np.linalg.norm(vel[0, j] - ref[3:]) < 1e-12
    hyp = np.r_[s0[:3], 3.0 * s0[3:]]                 # unbound: not this propagator's business
    assert np.all(np.isnan(bb.kepler_states(hyp[None, :3], hyp[None, 3:], dts)[0]))


def test_apparitions_split_on_the_gap_in_time_order():
    t = np.array([5000.0, 0.0, 0.3, 1.0, 5000.02, 9000.0])
    groups = bb.oppositions(t, np.arange(6), 100.0)
    assert [g.tolist() for g in groups] == [[1, 2, 3], [0, 4], [5]]
    assert bb.oppositions(t, np.array([], int), 100.0) == []


def _earth(t):
    lam = math.sqrt(MU) * t
    return np.array([math.cos(lam), math.sin(lam), 0.0])


def _synthetic_ev(state0, times, sig=0.3):
    """Topocentric (= geocentric, Earth on a circle) astrometry of `state0` at `times`, light-timed."""
    ra, dec, Robs = [], [], []
    for t in times:
        R = _earth(t)
        rel = kepler_propagate(state0, t)[:3] - R
        for _ in range(3):
            rel = kepler_propagate(state0, t - np.linalg.norm(rel) / bb.C_AU_DAY)[:3] - R
        ra.append(math.degrees(math.atan2(rel[1], rel[0])) % 360.0)
        dec.append(math.degrees(math.asin(rel[2] / np.linalg.norm(rel))))
        Robs.append(R)
    ra, dec = np.array(ra), np.array(dec)
    n = len(times)
    return SimpleNamespace(times_tdb=np.array(times, float), Robs=Robs, ra=ra, dec=dec,
                           cosd=np.cos(np.radians(dec)), fit_sig_ra=np.full(n, sig),
                           fit_sig_dec=np.full(n, sig), n=n)


def test_the_seeds_include_the_true_revolution_count_near_the_true_orbit():
    """107474's shape of problem, synthetic: two nights of an outer-main-belt asteroid, then one
    short night 8 years (~2 revolutions) later. The seeds span revolution counts and one of them is
    the object's own count with its own semi-major axis -- the starting point the capped fit needs."""
    a_true = 2.7
    s0 = _orbit_state(a_true, 0.15, 5.0, 30.0)
    times = [0.0, 0.03, 1.0, 1.03, 3000.0, 3000.02]
    ev = _synthetic_ev(s0, times)
    groups = bb.oppositions(ev.times_tdb, np.arange(ev.n), 100.0)
    t_ref, states, info = bb.bridge_seeds(ev, groups)
    assert t_ref in (0.03, 1.0)                        # the richer apparition anchors the search
    assert len(info) >= 2 and len({i["N"] for i in info}) >= 2
    for st in states:                                  # every seed is a bound orbit
        assert 0.5 * float(st[3:] @ st[3:]) - MU / np.linalg.norm(st[:3]) < 0
    n_true = int(3000.0 / (2 * math.pi * math.sqrt(a_true ** 3 / MU)))
    own = [i for i in info if i["N"] == n_true]
    assert own and min(abs(i["a"] - a_true) / a_true for i in own) < 0.1
    truth_at_ref = kepler_propagate(s0, t_ref)
    best_own = min(own, key=lambda i: i["chi2_twobody"])
    k = info.index(best_own)
    assert np.linalg.norm(states[k][:3] - truth_at_ref[:3]) < 0.3 * np.linalg.norm(truth_at_ref[:3])


def test_the_bridge_says_why_it_did_not_run():
    """The rule for the new comparator: a search that was not attempted must not read like
    one that found nothing. One apparition has no gap to bridge; an apparition longer than the
    comparator's short-arc bar pins its own range, which is the walk's regime."""
    import pyroorb.fit_object as fo
    import pyroorb.hyperbolic as hyp
    one = SimpleNamespace(times_tdb=np.array([0.0, 1.0, 20.0]))
    assert fo._bridged_bound_candidates(one, np.ones(3, bool), 0.0) == ([], 0, "single-apparition")
    long = SimpleNamespace(times_tdb=np.array([0.0, hyp.BOUND_SEED_MAX_ARC_DAYS + 5.0, 3000.0]))
    assert fo._bridged_bound_candidates(long, np.ones(3, bool), 0.0) == ([], 0, "long-apparition")
    # only INCLUDED observations make an apparition: switch the far one off and there is one
    assert fo._bridged_bound_candidates(long, np.array([True, True, False]), 0.0)[2] == "single-apparition"
