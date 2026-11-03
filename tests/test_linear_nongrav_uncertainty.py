"""A non-gravitational fit's linear uncertainty carries its A coefficients.

The linear products (propagate's covariance, close-approach and radar sigmas, the ephemeris
ellipse) used to differentiate the six state components only and project through the state
covariance with A marginalised -- so the acceleration was propagated as if exactly known, and
its uncertainty never grew with time. Measured on 2003 HM (Yarkovsky A2 at 384 sigma): 30 years
past the epoch the state-only position sigma was 34.8 km against the covariance ensemble's
48.7 km; with the A columns it is 50.7 km. Gravity fits are unchanged.
"""
import os
import types

import numpy as np
import pytest

from pyroorb import products
import pyroorb.ephemerides as eph


class _Backend:
    """Records the particles and per-particle A of the one sweep; each particle's 'state'
    at t is its initial state plus t times a linear function of its A (so the derivatives
    are known exactly)."""
    def __init__(self):
        self.calls = []

    def helio_states_multi(self, t0, states, times, nongrav=None, asteroids=None):
        self.calls.append((np.array(states), nongrav))
        S = np.array(states)
        A = (np.zeros((len(S), 3)) if nongrav is None
             else np.broadcast_to(np.asarray(nongrav[0], float), (len(S), 3)))
        out = np.empty((len(times), len(S), 6))
        for i, t in enumerate(times):
            out[i] = S + (t - t0) * 1e12 * np.column_stack([A, A])
        return out


def _fit(nongrav_model=None, A2=None):
    """A stand-in fit; with `A2` a Yarkovsky one whose A2 is correlated with y."""
    cov = np.diag([1e-12] * 3 + [1e-16] * 3)
    if A2 is not None:
        cov = np.diag([1e-12] * 3 + [1e-16] * 3 + [1e-30])
        cov[1, 6] = cov[6, 1] = 0.5 * np.sqrt(1e-12 * 1e-30)
    f = types.SimpleNamespace(epoch_tdb=2460000.5, nongrav_model=nongrav_model,
                              nongrav_params={"A2": A2} if A2 is not None else {},
                              asteroids=False)
    f.state = lambda: np.array([1.0, 0.5, 0.1, 0.0, 0.017, 0.0])
    f.covariance_matrix = lambda frame="state": cov[:6, :6]
    names = ["x", "y", "z", "dx", "dy", "dz"] + (["A2"] if A2 is not None else [])
    f.parameter_covariance = lambda: (cov, names)
    return f


@pytest.fixture
def backend(monkeypatch):
    b = _Backend()
    monkeypatch.setattr(products, "_backend", lambda fit: b)
    return b


def test_a_gravity_fit_sweeps_the_six_state_axes_as_before(backend):
    fit = _fit()
    s, phi, cov = products.states(fit, [2460010.5], stm=True)
    parts, ng = backend.calls[-1]
    assert len(parts) == 13 and ng is None
    assert phi.shape == (1, 6, 6) and cov.shape == (6, 6)
    assert np.allclose(phi[0], np.eye(6))
    assert np.array_equal(cov, fit.covariance_matrix("state"))


def test_a_nongrav_fit_adds_a_column_per_free_coefficient(backend):
    fit = _fit("yarkovsky", A2=-1e-13)
    s, phi, cov = products.states(fit, [2460000.5, 2460100.5], stm=True)
    parts, ng = backend.calls[-1]
    assert len(parts) == 15                               # nominal + 2 x (6 state + 1 A)
    A = np.asarray(ng[0])
    assert A.shape == (15, 3)
    assert np.all(parts[13] == parts[0]) and np.all(parts[14] == parts[0])   # A copies: nominal state
    assert A[13, 1] > A[0, 1] > A[14, 1] and np.all(A[:13, 1] == A[0, 1])
    assert phi.shape == (2, 6, 7) and cov.shape == (7, 7)
    # d state / d A2 is exactly (t - t0) * 1e12 in this backend's model, on the A2 axes
    assert np.allclose(phi[1, [1, 4], 6], 100 * 1e12, rtol=1e-6)
    assert np.allclose(phi[0, :, 6], 0.0, atol=1e-6)


def test_the_propagated_covariance_grows_with_the_force_uncertainty(backend, monkeypatch):
    fit = _fit("yarkovsky", A2=-1e-13)
    monkeypatch.setattr(products, "tdb", lambda t: np.atleast_1d(np.asarray(t, float)))
    _, C = products.propagate(fit, [2460100.5], covariance=True)
    _, phi, cov = products.states(fit, [2460100.5], stm=True)
    assert np.allclose(C[0], phi[0] @ cov @ phi[0].T)
    _, C_off = products.propagate(fit, [2460100.5], covariance=True, nongrav=False)
    assert C[0][1, 1] > C_off[0][1, 1]                    # the A2 term, transverse
    parts, ng = backend.calls[-1]
    assert len(parts) == 13 and ng is None                # nongrav=False: no force, no A axes


def test_linear_parameters_name_the_free_axes_in_fit_order():
    fit = _fit("yarkovsky", A2=-1e-13)
    pair, free, cov = eph.linear_parameters(fit)
    assert free == [1] and cov.shape == (7, 7) and pair[0][1] == -1e-13
    pair0, free0, cov0 = eph.linear_parameters(fit, nongrav=False)
    assert pair0 is None and free0 == [] and cov0.shape == (6, 6)


def test_the_ephemeris_A_step_is_capped_by_its_sigma():
    cov = np.diag([1.0] * 6 + [1e-32])                    # sigma_A2 = 1e-16
    h = eph._jacobian_A_steps([0.0, -1e-13, 0.0], [1], cov)
    assert h[0] == pytest.approx(1e-18)                   # 0.01 sigma, below 1e-3 |A| = 1e-16
    loose = np.diag([1.0] * 6 + [1e-20])
    assert eph._jacobian_A_steps([0.0, -1e-13, 0.0], [1], loose)[0] == pytest.approx(1e-16)
    no_sigma = np.diag([1.0] * 6 + [0.0])
    assert eph._jacobian_A_steps([0.0, -1e-13, 0.0], [1], no_sigma)[0] == pytest.approx(1e-16)


# --- measured: the linear sigma now agrees with the covariance ensemble --------------------
@pytest.mark.slow
def test_2003_HM_linear_sigma_matches_the_ensemble_decades_out():
    import pyroorb
    import pyroorb.fit_object as fo
    import pyroorb.force_assist as fa
    from pyroorb import obs_sources
    from astropy.time import Time
    from _fixtures import load_fixture, iter_fixtures
    if not os.path.exists(fa.PLANETS_PATH):
        pytest.skip("DE440 not present")
    path = next(p for p in iter_fixtures() if os.path.basename(p).startswith("2003_HM"))
    fx = load_fixture(path)
    oc = obs_sources.load_obscodes()
    fit = fo.fit_orbit(fx["obs"], oc, desig=fx["meta"]["designation"], meta=fx["meta"],
                       force=fx.get("force", False))
    assert fit.nongrav_model == "yarkovsky"
    t = Time(fit.epoch_tdb + 365.25 * 30, format="jd", scale="tdb")
    _, C = pyroorb.propagate(fit, t, covariance=True)
    ens = pyroorb.sample_orbits(fit, n=1500, method="covariance", rng_seed=1)
    X = ens.propagate(t)[0, :, :3]
    lin = np.sqrt(np.trace(C[0][:3, :3]))
    mc = np.sqrt(np.trace(np.cov(X.T)))
    assert lin == pytest.approx(mc, rel=0.10), (lin, mc)
    # the state-only answer it replaces is well short of the ensemble there
    raw, h, _ = products._sweep(fit, t.tdb.jd, True, True)
    phi6 = products._derivatives(raw, h)[0, :3, :6]
    old = np.sqrt(np.trace(phi6 @ fit.covariance_matrix("state") @ phi6.T))
    assert old < 0.85 * mc, (old, mc)
    # the ephemeris ellipse carries the same term: its major axis matches the ensemble's
    jd = float(t.utc.jd)
    row = eph.compute_ephemeris(fit, oc, jd, jd, 1.0, site="500", full=False)[0]
    sky = ens.sky_positions(jd, "500")
    x = ((sky["ra_deg"][0] - row["ra_deg"] + 180) % 360 - 180) * \
        np.cos(np.radians(row["dec_deg"])) * 3600
    y = (sky["dec_deg"][0] - row["dec_deg"]) * 3600
    smaj_mc = np.sqrt(np.linalg.eigvalsh(np.cov(np.vstack([x, y])))[1])
    assert row["unc_smaj_arcsec"] == pytest.approx(smaj_mc, rel=0.10)
