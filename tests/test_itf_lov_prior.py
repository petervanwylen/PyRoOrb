"""The LOV coordinate as an honest Gaussian, and the rankers' pre-floor view.

Since the covariance s2 floor (``FittedOrbit._param_inverse``) the LOV offset k
is measured in honest sigmas -- over all 26 ITF recovery fixtures the truth sits
at a median |k| of 0.67 (N(0,1): 0.674), max 2.72.  Three consumers had been
written against the old, over-confident scale; these pin what each now does.
"""
import math

from pyroorb.itf.config import ItfConfig
from pyroorb.itf.ephem import Window
from pyroorb.itf.lov_profile import LovProfileSet
from pyroorb.itf.training import _pre_floor_scale
from pyroorb.itf.lov_variants import _lov_prob_scale


def _ktr(name, rate_resid_arcsec, night):
    return {"_ids": frozenset({name}), "qual": rate_resid_arcsec, "dist": 100.0,
            "traj_category": "likely",
            "rows": [{"trksub": name, "stn": "F51",
                      "epoch_sec": night * 86400.0 + 3600.0}]}


def test_fan_holes_are_priced_in_honest_lov_sigma():
    """A variant sits at k LOV sigmas, so a hole between two variants holds
    Phi(k_b) - Phi(k_a) of the truth's probability -- factor 1.  The legacy
    0.426 is kept behind its A/B switch and nowhere else."""
    assert _lov_prob_scale(ItfConfig()) == 1.0
    legacy = _lov_prob_scale(ItfConfig(variant_prob_legacy=True))
    assert 0.40 < legacy < 0.45


def _two_homes(prior_sigma):
    """A far-tail home whose junk the ranker LIKES (P 0.9 at k -4.72, SA70's
    live shape) against a core home with a modest candidate (P 0.3 at k -0.19)."""
    cfg = ItfConfig(lov_profile=True, lov_profile_prior_sigma=prior_sigma)
    lov_sigmas = [0.0, -0.19, -4.72, -5.0]
    lov_profiles = LovProfileSet(cfg, lov_sigmas)
    lov_profiles.add(-4.72, _ktr("TAILJUNK", 0.5, night=10))
    lov_profiles.add(-0.19, _ktr("CORE", 0.8, night=20))
    lov_profiles._finalize_stats()
    lov_profiles.profiles[frozenset({"TAILJUNK"})].ml_p = 0.9
    lov_profiles.profiles[frozenset({"CORE"})].ml_p = 0.3
    lov_profiles._build_rounds()
    return [k for k, _ in lov_profiles.reorder([(k, []) for k in lov_sigmas])]


def test_priority_nominations_carry_the_lov_prior():
    # the prior: the core home goes first, the tail can still be visited
    order = _two_homes(1.0)
    assert order[0] == -0.19 and -4.72 in order
    # ...and 0 restores the pure-evidence order the live SA70 run showed
    assert _two_homes(0.0)[0] == -4.72


class _Fit:
    def __init__(self, f):
        self._f = f

    def cov_floor_factor(self):
        return self._f


class _Trial:
    def __init__(self, U, f):
        self.U = U
        self.fit = _Fit(f)


def _window(uncert_deg, dist_au=1.0):
    return Window(unix_cur=0.0, unix_start=0.0, unix_end=1.0,
                  ra_quad=(0.0, 0.0, 10.0), dec_quad=(0.0, 0.0, 5.0),
                  min_ra=10.0, max_ra=10.0, min_dec=5.0, max_dec=5.0,
                  uncert=uncert_deg, uncert_ra=uncert_deg * 1.2,
                  max_speed=0.5, dist_au=dist_au)


def test_rankers_see_the_pre_floor_scale():
    """The sky-sigma part of the window shrinks by f, the parallax allowance
    does not, the RA/Dec aspect is kept, and U drops by ln(f)/1.49 -- the law
    that reproduces a real unfloored seed's U to 0.003 (SA70 8.970 -> 8.457,
    measured unfloored 8.46; HM543 at f 10.07: 9.397 -> 7.847, 7.85)."""
    from pyroorb.itf.obs80_fields import EARTH_RAD_AU
    par = EARTH_RAD_AU / 1.0 * 57.3
    window, U = _pre_floor_scale(_window(par + 0.5), _Trial(8.97, 2.148))
    assert math.isclose(window.uncert, par + 0.5 / 2.148, rel_tol=1e-12)
    assert math.isclose(window.uncert_ra / window.uncert, 1.2, rel_tol=1e-12)
    assert math.isclose(U, 8.97 - math.log(2.148) / 1.49, rel_tol=1e-12)


def test_pre_floor_scale_leaves_what_the_floor_did_not_touch():
    # a fit that already missed its sigmas (s2 >= 1) is untouched
    w = _window(0.3)
    assert _pre_floor_scale(w, _Trial(5.0, 1.0)) == (w, 5.0)
    # a variant window is parallax-only: width kept, only U moves
    from pyroorb.itf.obs80_fields import EARTH_RAD_AU
    wv = _window(EARTH_RAD_AU / 2.0 * 57.3, dist_au=2.0)
    window, U = _pre_floor_scale(wv, _Trial(9.0, 3.0))
    assert window.uncert == wv.uncert and U < 9.0
    # a test double with no Jacobian degrades to the served scale
    class _Bare:
        U = 4.0
        fit = object()
    assert _pre_floor_scale(w, _Bare()) == (w, 4.0)
