"""Tests for the massive-asteroid perturber polish (fit_object.refine_asteroids) and the
per-call asteroid flag in force_assist. The observations are frozen fixtures (2023 QJ3 from
the regression set, 2009 HV119 in obs_fixtures/). The 16-perturber kernel (sb441-n16.bsp,
~616 MB) is NOT in CI, so the kernel-dependent checks SKIP cleanly without it; the always-on
checks cover the gate logic and the byte-identical gravity default.
"""
import os
os.environ.pop("PYROORB_ASTEROIDS_PATH", None)        # exercise the polish path, not whole-fit env
import numpy as np
import pytest
import json
import pyroorb.fit_object as fo
from pyroorb import obs_sources
import pyroorb.force_assist as fa
from _fixtures import fixture_obs

# refits real objects
pytestmark = pytest.mark.slow

_OBS_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs_fixtures")


def _have_kernel():
    return fa.asteroid_kernel_path() is not None


def _frozen(desig):
    """The frozen (obs, meta) of `desig`: a regression fixture, else obs_fixtures/."""
    stem = desig.replace(" ", "_")
    path = os.path.join(_OBS_FIXTURES, f"{stem}.json")
    if os.path.isfile(path):
        with open(path) as fh:
            d = json.load(fh)
        return d["obs"], d["meta"]
    return fixture_obs(stem)


def _fit(desig):
    """Fit `desig` from its frozen observations (never the network)."""
    obs, meta = _frozen(desig)
    oc = obs_sources.load_obscodes()
    return fo.fit_orbit(obs, oc, desig=meta.get("designation") or desig, meta=meta)


def test_per_call_flag_gravity_is_default():
    """force_assist: predict with asteroids=False is byte-identical to the env-unset
    default (so the gravity pipeline / regression suite are untouched), and the epsilon
    switches with the flag."""
    t0 = 2455000.5
    st = np.array([2.2, 0.5, 0.1, -0.005, 0.009, 0.003])
    tt = np.array([2455050.5, 2455400.5])
    Rs = [np.array([0.1, 0.95, 0.41]), np.array([0.2, 0.9, 0.39])]
    g = fa.predict_radec_batch(None, t0, st, tt, Rs, asteroids=False)
    d = fa.predict_radec_batch(None, t0, st, tt, Rs)          # default None, env unset
    assert np.array_equal(g, d), "gravity flag must match the env-unset default exactly"
    assert fa._epsilon(False) == fa.IAS15_EPSILON
    assert fa._epsilon(True) == fa.IAS15_EPSILON_ASTEROIDS


def test_single_opposition_skipped():
    """A single-opposition object is gated out -- refine_asteroids returns it unchanged
    (no kernel needed: the gate is checked before the kernel)."""
    fit = _fit("2023 QJ3")
    if fit.n_oppositions() >= fo.ASTEROID_MIN_OPP:
        pytest.skip("2023 QJ3 is multi-opposition in this cache -- gate not exercised")
    out = fo.refine_asteroids(fit)
    assert out is fit, "single-opposition object should be skipped (returned unchanged)"
    assert out.asteroids is False


def test_polish_helps_hv119():
    """2009 HV119 leans on the Vesta perturbation: the polish should redeem obs and/or
    lower the RMS, and flag the orbit as asteroid-active. Skips without the kernel/obs."""
    if not _have_kernel():
        pytest.skip("no sb441-n16 kernel")
    g = _fit("2009 HV119")
    a = fo.refine_asteroids(g)
    assert a is not g, "HV119 should be polished, not skipped"
    assert a.asteroids is True
    redeemed = a.n_observations() - g.n_observations()
    improved = a.rms_residual() < g.rms_residual()
    assert redeemed > 0 or improved, (
        f"polish did not help: obs {g.n_observations()}->{a.n_observations()}, "
        f"rms {g.rms_residual():.3f}->{a.rms_residual():.3f}")


def test_force_bypasses_gate():
    """force=True applies the perturbers even to a gated-out object (CLI --asteroids)."""
    if not _have_kernel():
        pytest.skip("no sb441-n16 kernel")
    fit = _fit("2023 QJ3")
    out = fo.refine_asteroids(fit, force=True)
    assert out.asteroids is True, "force=True should apply asteroids regardless of the gate"
