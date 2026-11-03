"""The ephemeris uncertainty's Jacobian step: capped at 0.01 sigma per axis, so the covariance is
carried to the sky in its linear regime (a fixed 1.49e-8 step went non-linear through Apophis's
2029 pass and widened its ellipse 70x); orbits whose sigma already exceeds the fitter's step keep
exactly the old step. Offline: frozen fixtures."""
import os
import sys
import warnings

import numpy as np
import pytest

from pyroorb import ephemerides
from pyroorb.ephemerides import JAC_STEP_SIGMAS, _jacobian_steps

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

STATE = np.array([1.2, -0.4, 0.1, 0.004, 0.013, -0.002])
FITTER = 1.49e-8 * np.maximum(1.0, np.abs(STATE))


def test_step_is_capped_at_a_hundredth_of_sigma():
    sig = np.array([1e-9, 1e-9, 1e-9, 4e-11, 4e-11, 4e-11])          # a well-observed NEA
    h = _jacobian_steps(STATE, np.diag(sig ** 2))
    assert np.allclose(h, JAC_STEP_SIGMAS * sig)
    assert np.all(h < FITTER)


def test_a_loose_orbit_keeps_the_fitters_step_exactly():
    h = _jacobian_steps(STATE, np.diag(np.full(6, 1e-2) ** 2))         # a tracklet / ITF corridor
    assert np.array_equal(h, FITTER)


def test_degenerate_and_missing_sigmas_stay_finite():
    h = _jacobian_steps(STATE, np.zeros((6, 6)))                        # dof 0: no scale
    assert np.all(h > 0) and np.all(np.isfinite(h))
    bad = np.diag([np.nan, 1.0, 1.0, 1.0, 1.0, 1.0])
    h = _jacobian_steps(STATE, bad)
    assert h[0] == FITTER[0] and np.all(np.isfinite(h))


@pytest.mark.slow
def test_a_linear_orbit_reads_the_same_ellipse_with_either_step(monkeypatch):
    """2023 QJ3 (well determined, no encounter): the capped step and the fitter's step give the
    same ellipse to 1e-3 -- the cap changes nothing where the old step was already linear."""
    import pyroorb
    from _fixtures import fixture_obs
    records, meta = fixture_obs("2023_QJ3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pyroorb.fit_orbit(pyroorb.Observations(records, meta), designation="2023 QJ3")
    new = pyroorb.ephemeris(fit, "2026-01-01", "2027-01-01", step=60, site="568", full=False)
    monkeypatch.setattr(ephemerides, "_jacobian_steps",
                        lambda state, cov: 1.49e-8 * np.maximum(1.0, np.abs(np.asarray(state)[:6])))
    old = pyroorb.ephemeris(fit, "2026-01-01", "2027-01-01", step=60, site="568", full=False)
    for a, b in zip(new, old):
        for col in ("unc_smaj_arcsec", "unc_smin_arcsec", "unc_along_arcsec", "unc_cross_arcsec"):
            assert a[col] == pytest.approx(b[col], rel=1e-3), col
