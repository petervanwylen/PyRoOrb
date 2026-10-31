"""The interactive determinacy floor is 3 observations, not 6 -- and dof is reported, not enforced.

An optical observation supplies TWO residual rows (RA, Dec), so a 6-element state is
determined at ``2n >= 6``: three observations. That is the engine's own pipeline-wide floor
(``arc_extend.FIT_MIN_OBS``) and find_orb's (``full_improvement`` refuses below 3 included
obs, ``orb_func.cpp``). The interactive endpoints used to hard-code 6, refusing re-fits on
the user's own inclusion mask that the CLI performs. What lives between 3 and 6 obs is a
statistics problem -- dof is 0 at n=3 -- which the payload REPORTS for the banner.

    python -m pytest tests/web/test_min_obs.py
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from pyroorb import arc_extend
import pyroorb.web.engine_api as E
import pyroorb.fit_object as fo


def test_floor_is_the_engine_constant_not_six():
    assert arc_extend.FIT_MIN_OBS == 3
    for n in (3, 4, 5, 6, 40):
        assert E._underdetermined(n) is None, f"{n} obs must be fittable"


def test_two_obs_is_genuinely_underdetermined():
    for n in (0, 1, 2):
        err = E._underdetermined(n)
        assert err is not None and err["ok"] is False
        assert "at least 3" in err["error"]


def test_nongrav_counts_residual_rows_not_observations():
    """6+k parameters need 6+k ROWS, i.e. ceil((6+k)/2) observations -- the old check
    compared an observation count to a parameter count and demanded twice what it needed."""
    assert E._underdetermined(4, 7) is None       # 8 rows >= 7 params (old code refused)
    assert E._underdetermined(3, 7) is not None   # 6 rows < 7 params
    assert E._underdetermined(5, 9) is None       # 10 rows >= 9 params
    assert E._underdetermined(4, 9) is not None   # 8 rows < 9 params
    # never below the engine floor, however few parameters are claimed
    assert E._underdetermined(2, 2) is not None


class _FakeFit:
    """Just the surface degrees_of_freedom() reads."""
    def __init__(self, n_incl, npar=6, constraints=None):
        self._mask = np.ones(n_incl, dtype=bool)
        self._state = np.zeros(npar)
        self.constraints = dict(constraints or {})
    n_resids = fo.FittedOrbit.n_resids
    n_free_params = fo.FittedOrbit.n_free_params
    degrees_of_freedom = fo.FittedOrbit.degrees_of_freedom


def test_dof_arithmetic_matches_find_orb():
    """find_orb's dof is n_included*2 - n_params (orb_func.cpp)."""
    assert _FakeFit(3).degrees_of_freedom() == 0        # exactly determined
    assert _FakeFit(4).degrees_of_freedom() == 2
    assert _FakeFit(10).degrees_of_freedom() == 14
    assert _FakeFit(4, npar=7).degrees_of_freedom() == 1     # a nongrav A1 costs one row


def test_held_fixed_elements_cost_no_degree_of_freedom():
    """refit_constrained pins elements with penalty rows rather than a shorter parameter
    vector, but a pinned element is not fitted -- it must not be charged as free."""
    assert _FakeFit(3, constraints={"a": 2.5}).degrees_of_freedom() == 1
    assert _FakeFit(3, constraints={"a": 2.5, "e": 0.1}).degrees_of_freedom() == 2
