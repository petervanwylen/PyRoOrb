"""A fit with zero degrees of freedom must report itself as maximally uncertain.

Three observations are six measurements against six elements: the fit is exactly
determined, its residuals are identically zero and so is the residual variance s2.
Scaling the covariance by that s2 collapsed it by ~1e9 and reported U 0.00 -- the
best possible certainty for the worst possible arc, read by the CLI, the web UI, the
ITF corridor (which then never armed its under-determined-seed fan) and the impact
products. The covariance floor (``FittedOrbit._param_inverse``: s2 never below 1)
makes the matrix the assigned-sigma ``inv(J^T J)``, which for a square Jacobian is
exactly the linear propagation of the observation errors -- defined, and wide.

Measured on three fixtures, first observation of each of three nights:
2008 TE4 U 10.4, 2023 QJ3 U 10.7, 1998 VD32 U 9.0, every one with a floor factor
of 1e9-1e10 (the size of the collapse the floor undoes).
"""
import os

import numpy as np
import pytest

import _fixtures as FX
import pyroorb.fit_object as fo
from pyroorb.web import engine_api as E

pytestmark = pytest.mark.slow          # real engine fits


def _fit(obs, meta, idx):
    return fo.fit_orbit([obs[i] for i in idx], E._obscodes(), desig=meta["designation"],
                        meta=meta, verbose=False, auto_nongrav=False)


def _three_nights(name):
    fx = FX.load_fixture(os.path.join(FX.FIX, name + ".json"))
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    night = np.floor(np.array([o["jd_utc"] for o in obs]) - 0.5)
    per = [np.flatnonzero(night == n) for n in np.unique(night)[:3]]
    return obs, fx["meta"], per


def test_three_observations_read_as_maximally_uncertain():
    obs, meta, per = _three_nights("2008_TE4")
    fit = _fit(obs, meta, [int(p[0]) for p in per])
    J = np.asarray(fit._res.jac)
    assert J.shape[0] - J.shape[1] == 0, "the case must be exactly determined"
    # the raw residual variance did collapse: the floor is what is holding U up
    assert fit.cov_floor_factor() > 1e6
    U = fit.uncertainty_parameter()
    assert U is not None and U >= 9.0, f"zero-dof fit reported U {U}"
    assert np.all(np.isfinite(np.sqrt(np.diag(fit._state_cov))))


def test_more_observations_on_the_same_nights_never_read_less_certain():
    """The same three nights with two obs each (dof 6) may tighten U, never loosen it:
    a zero-dof fit must not look better than the over-determined fit of its own arc."""
    obs, meta, per = _three_nights("2008_TE4")
    u3 = _fit(obs, meta, [int(p[0]) for p in per]).uncertainty_parameter()
    u6 = _fit(obs, meta, sorted(int(i) for p in per for i in p[:2])).uncertainty_parameter()
    assert u6 <= u3 + 1e-6, f"6 obs U {u6} vs 3 obs U {u3}"
