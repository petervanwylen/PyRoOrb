"""Offline tests for constrained fits (fit_object.refit_constrained), find_orb's
limited_orbit / evaluate_limited_orbit: hold one or more orbital elements fixed during
the differential correction by appending a penalty residual (element - target)/sigma_pen
to the weighted LM. The penalty sigma is set RELATIVE to the element's free sigma so the
constraint is always effectively hard (a fixed absolute weight fails -- it is far weaker
than the astrometry's pull on a well-determined element).

The feature is OPT-IN: the default pipeline never calls it, so the whole regression suite
stays byte-identical. These tests exercise it directly on a frozen fixture object.
"""
import pytest
import sys, os
import pyroorb.fit_object as fo

from pyroorb import obs_sources
from _fixtures import fixture_obs

# constrained refits of a frozen fixture
pytestmark = pytest.mark.slow
_FIT = None


def _fit():
    """Fit a well-determined MBA once, reuse across tests.

    From the FROZEN fixture, not a fetch: the assertions below pin free values
    (e ~ 0.402) that only hold for one particular arc, so fresh astrometry could
    move them anyway -- and see _fixtures.fixture_obs for why a "cached" fetch is
    not offline.
    """
    global _FIT
    if _FIT is None:
        obscodes = obs_sources.load_obscodes()
        obs, meta = fixture_obs("2023_QJ3")
        _FIT = fo.fit_orbit(obs, obscodes, desig=meta["designation"], meta=meta)
    return _FIT


def test_single_element_binds():
    """Constraining e to a target far from the free value pins it there exactly, and
    collapses its reported sigma well below the free sigma."""
    fit = _fit()
    e_free, sig_free = fit.eccentricity()
    assert abs(e_free - 0.402) < 0.01, f"unexpected free e {e_free}"
    c = fo.refit_constrained(fit, {"e": 0.25})
    e_c, sig_c = c.eccentricity()
    assert abs(e_c - 0.25) < 1e-4, f"e not held: {e_c}"
    assert sig_c < sig_free, f"constrained sigma {sig_c} not below free {sig_free}"
    assert c.constraints == {"e": 0.25}


def test_inclination_and_q_bind():
    """i (a frame-dependent, ecliptic element) and q both pin to their targets."""
    fit = _fit()
    ci = fo.refit_constrained(fit, {"i": 5.0})
    assert abs(ci.inclination()[0] - 5.0) < 1e-3, ci.inclination()
    cq = fo.refit_constrained(fit, {"q": 2.0})
    assert abs(cq.perihelion_distance()[0] - 2.0) < 1e-4, cq.perihelion_distance()


def test_multi_constraint_binds_both():
    """Two simultaneous constraints both hold."""
    fit = _fit()
    c = fo.refit_constrained(fit, {"e": 0.3, "i": 20.0})
    assert abs(c.eccentricity()[0] - 0.3) < 1e-4
    assert abs(c.inclination()[0] - 20.0) < 1e-3


def test_parabolic_preset():
    """e=1 (the parabolic preset) drives eccentricity to unity."""
    fit = _fit()
    c = fo.refit_constrained(fit, {"e": 1.0})
    assert abs(c.eccentricity()[0] - 1.0) < 1e-4, c.eccentricity()


def test_empty_constraints_returns_same_fit():
    """An empty / unknown-only constraint set is a no-op (returns the input fit)."""
    fit = _fit()
    assert fo.refit_constrained(fit, {}) is fit
    assert fo.refit_constrained(fit, {"bogus": 1.0}) is fit


def test_constraint_does_not_mutate_input():
    """refit_constrained returns a NEW orbit; the gravity fit it was given is untouched."""
    fit = _fit()
    e_before = fit.eccentricity()[0]
    fo.refit_constrained(fit, {"e": 0.1})
    assert fit.eccentricity()[0] == e_before, "input fit was mutated"
    assert fit.constraints == {}
