"""popprior.py: the digest2 population model as a prior on a short arc's distance.

Fast unit tests pin the two closed forms the density rests on and the posterior's bookkeeping;
the slow ones pin what it is FOR, on a real main-belt tracklet, and its refusals."""
import math
import numpy as np
import pytest

from pyroorb import popprior
from pyroorb.forward_model import MU_SUN


def test_the_delaunay_jacobian_is_the_closed_form():
    """|d(L, G, Hz) / d(q, e, i)| = L G^2 e sin i / (2 q (1 - e^2)), the factor that turns a
    (q, e, i) density into a phase-space one -- checked by finite differences."""
    def lgh(x):
        q, e, i = x
        L = math.sqrt(MU_SUN * q / (1 - e))
        G = L * math.sqrt(1 - e * e)
        return np.array([L, G, G * math.cos(i)])
    for x in [(1.2, 0.3, 0.4), (2.5, 0.05, 0.1), (0.6, 0.8, 1.2)]:
        J = np.zeros((3, 3))
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-6
            J[:, k] = (lgh(np.add(x, d)) - lgh(np.subtract(x, d))) / 2e-6
        q, e, i = x
        L, G, _ = lgh(x)
        assert abs(np.linalg.det(J)) == pytest.approx(L * G * G * e * math.sin(i) / (2 * q * (1 - e * e)),
                                                     rel=1e-6)


def test_a_bin_volume_matches_a_numerical_integral():
    from scipy import integrate
    f = lambda i, e, q: (math.sqrt(MU_SUN * q / (1 - e)) * MU_SUN * q * (1 + e) * e * math.sin(i)
                         / (2 * q * (1 - e * e)))
    num = integrate.tplquad(f, 2.0, 2.2, 0.1, 0.2, math.radians(2), math.radians(5))[0]
    F = lambda e: 2 / math.sqrt(1 - e) + 2 * math.sqrt(1 - e)
    cf = (MU_SUN ** 1.5 / 2 * (2 / 3) * (2.2 ** 1.5 - 2.0 ** 1.5) * (F(0.2) - F(0.1))
          * (math.cos(math.radians(2)) - math.cos(math.radians(5))))
    assert cf == pytest.approx(num, rel=1e-6)


def _circular(a, inc_deg=5.0):
    """Heliocentric EQUATORIAL state of a circular orbit of radius a, inclined to the ECLIPTIC."""
    from pyroorb.orbital_elements import state_eq_from_elements
    return np.asarray(state_eq_from_elements(a, 0.01, inc_deg, 30.0, 40.0, 50.0), float)


def test_the_main_belt_is_dense_and_an_unbound_orbit_is_empty():
    R = np.array([1.0, 0.0, 0.0])
    mb = popprior.density(_circular(2.7), R)[0]
    far = popprior.density(_circular(60.0), R)[0]
    st = _circular(2.7)
    st[3:] *= 1.6                                   # past escape
    assert mb > 0 and popprior.density(st, R)[0] == 0.0
    assert mb > 100 * far, (mb, far)


def test_h_from_the_magnitude_prefers_a_size_that_exists():
    """At V = 21 a main-belt orbit at 1.7 au from the observer means H ~ 17: common. The same
    state seen with V = 8 would mean H ~ 4: next to nothing in the model."""
    R = np.array([1.0, 0.0, 0.0])
    st = _circular(2.7)
    assert popprior.density(st, R, vmag=21.0)[0] > 100 * popprior.density(st, R, vmag=8.0)[0]


def test_the_posterior_median_sits_inside_the_profile_and_ranks_the_pcts():
    R0 = np.array([1.0, 0.0, 0.0])
    rows = []
    for rho in np.geomspace(0.02, 3.0, 25):
        u = np.array([1.0, 0.2, 0.05]) / np.linalg.norm([1.0, 0.2, 0.05])
        r = R0 + rho * u
        v = np.cross([0, 0, 1.0], r)
        v *= math.sqrt(MU_SUN / np.linalg.norm(r)) / np.linalg.norm(v)
        rows.append({"rho": float(rho), "chi2": 10.0, "state": np.concatenate([r, v]).tolist(),
                     "bound": True})
    post = popprior.posterior({"rows": rows, "R0": R0.tolist(), "t0_tdb": 2461000.5}, vmag=20.5)
    assert post is not None
    pc = post["rho_pct"]
    assert 0.02 <= pc["5"] <= pc["50"] <= pc["95"] <= 3.0
    assert post["rho_med"] == pytest.approx(pc["50"])
    assert sum(post["weight"]) > 0


def test_no_profile_no_posterior():
    assert popprior.posterior({"rows": []}) is None
    assert popprior.posterior(None) is None


@pytest.mark.slow
def test_the_most_probable_orbit_of_a_main_belt_tracklet_is_in_the_main_belt():
    """RANGEAMBIG_3NIGHT_MBA's first night: the range profile cannot fix the distance, and the
    delivered fit is wherever the walk stopped. The population's most probable orbit is a
    low-e main-belt orbit that fits the night as well as the delivered one does."""
    import os
    from _fixtures import load_fixture, FIX
    import pyroorb.fit_object as fo; from pyroorb import obs_sources
    fx = load_fixture(os.path.join(FIX, "RANGEAMBIG_3NIGHT_MBA.json"))
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    night = [o for o in obs if o["jd_utc"] <= obs[0]["jd_utc"] + 0.5]
    fit = fo.fit_orbit(night, obs_sources.load_obscodes(), desig="RANGEAMBIG", meta=fx["meta"])
    mp = popprior.most_probable_fit(fit)
    assert mp is not None
    el = mp.elements()
    assert 2.0 < el["a"] < 3.6 and el["e"] < 0.3, el
    assert mp.rms_residual() < 1.5 * fit.rms_residual() + 0.05
    pc = mp.most_probable["rho_pct"]
    assert pc["5"] <= mp.most_probable["rho_med"] <= pc["95"]


@pytest.mark.slow
def test_the_most_probable_orbit_fits_its_own_tracklet():
    """PAIR_2025OA537's richest tracklet: the first version showed a state interpolated between
    two profile nodes with its radial velocity swapped, fitting at 7.3 sigma where the delivered
    orbit fits at 0.84 (31 of 169 fixture tracklets past 2 sigma). The sky position and rates are
    now re-fitted with the chosen distance and radial velocity held -- and the distance shown is
    the posterior median itself, not the linear interpolation's."""
    import os
    from _fixtures import load_fixture, FIX
    import pyroorb.fit_object as fo; from pyroorb import obs_sources
    import sys
    from _fixtures import best_tracklet
    fx = load_fixture(os.path.join(FIX, "PAIR_2025OA537.json"))
    fit = fo.fit_orbit(best_tracklet(fx["obs"]), obs_sources.load_obscodes(), desig="x", meta=fx["meta"])
    mp = popprior.most_probable_fit(fit)
    assert mp is not None
    assert mp.rms_normalized() <= max(1.5 * fit.rms_normalized(), fit.rms_normalized() + 0.3)
    assert mp.geocentric_range() == pytest.approx(mp.most_probable["rho_med"], rel=1e-3)


def _first_nights(label, n):
    """The fit to a fixture's first `n` nights (a night ends at a gap over 0.5 d)."""
    import os
    import _fixtures as FX
    import pyroorb.fit_object as fo; from pyroorb import obs_sources
    fx = FX.load_fixture(os.path.join(FX.FIX, label + ".json"))
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    nights = 1 + sum(b["jd_utc"] - a["jd_utc"] > 0.5 for a, b in zip(obs, obs[1:]))
    assert nights > n
    last, seen = 0, 1
    for i in range(1, len(obs)):
        if obs[i]["jd_utc"] - obs[i - 1]["jd_utc"] > 0.5:
            seen += 1
            if seen > n:
                break
        last = i
    return fo.fit_orbit(obs[:last + 1], obs_sources.load_obscodes(),
                        desig=fx["meta"]["designation"], meta=fx["meta"])


@pytest.mark.slow
def test_a_near_parabolic_two_night_fit_is_replaced_by_the_population_orbit():
    """Two nights of a main-belt object (a 2.58) deliver a ~ 9700: the prior reads the LOV."""
    fit = _first_nights("PAIR_2014HH101", 2)
    assert fit.elements()["e"] >= popprior.LOV_E_FLOOR
    mp = popprior.most_probable_fit(fit)
    assert mp is not None
    el = mp.elements()
    assert 2.0 < el["a"] < 3.2 and el["e"] < 0.5, el
    assert mp.rms_residual() <= popprior.LOV_RMS_MAX * fit.rms_residual()
    info = mp.most_probable
    assert abs(info["k"]) <= popprior.LOV_K_MAX
    assert 0 < info["rho_pct"]["5"] <= info["rho_pct"]["95"]
    assert fit.elements()["e"] >= popprior.LOV_E_FLOOR, "the delivered fit must stay untouched"


@pytest.mark.slow
def test_an_ordinary_multi_night_orbit_is_left_alone():
    """Under e = 0.9 the delivered orbit is usually fair, and the prior's gain does not pay."""
    fit = _first_nights("107474", 2)
    assert fit.elements()["e"] < popprior.LOV_E_FLOOR
    assert popprior.most_probable_fit(fit) is None


@pytest.mark.slow
def test_a_re_correction_that_leaves_the_bound_orbits_is_refused(monkeypatch):
    """The pinned fit can slip off the strip the linear LOV crossed: the fit then stands."""
    fit = _first_nights("PAIR_2014HH101", 2)
    u, sigma, _refit = fit.lov_refit()
    monkeypatch.setattr(type(fit), "lov_refit",
                        lambda self: (u, sigma, lambda *a: np.asarray(fit._state, float)[:6]))
    assert popprior.most_probable_lov_fit(fit) is None


@pytest.mark.slow
def test_a_replacement_that_misfits_is_refused(monkeypatch):
    fit = _first_nights("PAIR_2014HH101", 2)
    monkeypatch.setattr(popprior, "LOV_RMS_MAX", 1e-6)
    assert popprior.most_probable_lov_fit(fit) is None
