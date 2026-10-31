"""hyperbolic.py: the link bar, the planetary-kick reach, the inbound reminder -- and the pieces of
the engine an unbound orbit used to fall out of: the MOID of a hyperbola, the Monte Carlo clone
summary, the covariance determinacy verdict, and the MPCORB writers. Offline and fit-free.

Planet elements are FIXED dicts (circular, ecliptic) so the tests do not depend on an ephemeris.
"""
import math

import numpy as np
import pytest

from pyroorb import constraint
from pyroorb import egress
import pyroorb.hyperbolic as hyp
from pyroorb import mc
from pyroorb.hazard import moid
from pyroorb.orbital_elements import ELEM_NAMES, _ecl_to_eq, element_vector

MU = 0.01720209895 ** 2
EPOCH = 2460000.5


def _circle(r):
    return {"a": r, "e": 0.0, "i": 0.0, "asc_node": 0.0, "arg_per": 0.0}


PLANETS = {name: _circle(a_p) for name, _gm, _r, a_p in hyp.GIANTS}


def _hyperbola(q, e, i=0.0, node=0.0, peri=0.0):
    return {"a": q / (1.0 - e), "e": e, "q": q, "i": i, "asc_node": node, "arg_per": peri}


def _peri_state(q, e):
    """Equatorial state AT perihelion of an orbit lying in the ecliptic."""
    r = np.array([q, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU * (1.0 + e) / q), 0.0])
    return np.concatenate([_ecl_to_eq(r), _ecl_to_eq(v)])


# --- MOID of a hyperbola --------------------------------------------------------------------
def test_moid_hyperbola_outside_a_coplanar_circle_is_perihelion_gap():
    """Coplanar, perihelion 1.5 au, circle at 1 au: the closest approach of the two curves is the
    perihelion point, 0.5 au. The ellipse formula at e > 1 drew a straight segment instead."""
    assert moid(_hyperbola(1.5, 2.0), _circle(1.0)) == pytest.approx(0.5, abs=1e-6)


def test_moid_hyperbola_crossing_a_coplanar_circle_is_zero():
    assert moid(_hyperbola(0.5, 2.0), _circle(1.0)) < 1e-4


def test_moid_ellipse_unchanged():
    assert moid({"a": 2.0, "e": 0.25, "i": 0.0, "asc_node": 0.0, "arg_per": 0.0},
                _circle(1.0)) == pytest.approx(0.5, abs=1e-6)


# --- link bar ---------------------------------------------------------------------------------
def test_count_nights_groups_one_night_and_splits_on_gaps():
    assert hyp.count_nights([]) == 0
    assert hyp.count_nights([0.0, 0.02, 0.05, 1.0, 1.01, 3.0]) == 3


def test_count_nights_is_the_canonical_night_in_the_untested_band():
    """This module counted nights by a 0.3 d consecutive gap, the only
    such value in the tree, and no test probed 0.3-0.5 d.  Two exposures 8 h
    apart from one station are ONE night here, as everywhere else; the arc bar
    it feeds cannot be made easier by a long night."""
    from pyroorb import orbit_common
    assert hyp.count_nights([0.0, 0.35]) == 1                      # was 2
    assert hyp.count_nights([0.0, 0.35]) == orbit_common.count_nights([0.0, 0.35])
    assert hyp.unbound_arc_ok([0.0, 0.35, 1.0, 1.35, 2.0, 2.35, 3.0, 3.35])[1] == 4
    assert not hyp.unbound_arc_ok([0.0, 0.35, 1.0, 1.35, 2.0, 2.35])[0]   # 3 nights, not 6
    # an object followed round the clock earns two nights a day (the window rule's point)
    assert hyp.count_nights([0.0, 0.25, 0.5, 0.75, 1.0]) == 3


def test_unbound_arc_ok_four_nights_short_five_nights_long():
    assert hyp.unbound_arc_ok([0.0, 1.0, 2.0, 20.0])[0] is True
    assert hyp.unbound_arc_ok([0.0, 1.0, 2.0, 40.0])[0] is False
    assert hyp.unbound_arc_ok([0.0, 1.0, 2.0, 3.0, 40.0])[0] is True
    assert hyp.unbound_arc_ok([0.0, 1.0, 2.0])[0] is False


# --- kick reach --------------------------------------------------------------------------------
def test_encounter_kick_calibration():
    """The numbers the module docstring quotes."""
    _name, gm, radius, a_p = hyp.GIANTS[0]
    for miss, expect in ((0.05, 0.117), (0.5, 0.0174), (1.0, 0.00896), (5.0, 0.00183)):
        assert hyp.encounter_kick(gm, radius, a_p, miss) == pytest.approx(expect, rel=0.02)
    assert hyp.encounter_kick(gm, radius, a_p, 0.228) > 0.017     # Bowell's pass explains Bowell


def test_kick_reach_floor_when_no_planet_is_close():
    far = _hyperbola(0.3, 1.2, i=90.0)          # polar, perihelion inside Mercury
    reach, _nearest = hyp.kick_reach(far, EPOCH, PLANETS)
    assert hyp.PERTURBATION_FLOOR <= reach < 0.8   # nowhere near 1I's |1/a|


# --- inbound reminder -------------------------------------------------------------------------
def test_reminder_fires_for_a_bowell_like_jupiter_ejection():
    """q 3.36 au, e 1.057, nearly in the ecliptic: crosses Jupiter's orbit, |1/a| 0.017."""
    bowell = _hyperbola(3.3639, 1.057, i=1.66, node=114.56, peri=135.08)
    got = hyp.inbound_reminder(bowell, 0.0005, EPOCH, PLANETS)
    assert got is not None and got["planet"] == "Jupiter"
    assert "may have arrived bound" in got["text"]
    assert "nan" not in got["text"].lower() and math.isfinite(got["inv_a"])


@pytest.mark.parametrize("q,e,i", [(0.2553, 1.2011, 122.74),   # 1I
                                   (2.0066, 3.3565, 44.05),    # 2I
                                   (1.3564, 6.1394, 175.11)])  # 3I
def test_reminder_silent_for_interstellars(q, e, i):
    assert hyp.inbound_reminder(_hyperbola(q, e, i=i), 1e-4, EPOCH, PLANETS) is None


def test_reminder_silent_when_compatible_with_parabolic():
    """e - 1 within 3 sigma: not clearly unbound, so rule 1 owns it, not the reminder."""
    assert hyp.inbound_reminder(_hyperbola(3.3639, 1.0003, i=1.66), 0.001, EPOCH, PLANETS) is None


def test_reminder_silent_without_a_sigma_or_on_a_bound_orbit():
    bowell = _hyperbola(3.3639, 1.057, i=1.66)
    assert hyp.inbound_reminder(bowell, None, EPOCH, PLANETS) is None
    assert hyp.inbound_reminder(bowell, float("nan"), EPOCH, PLANETS) is None
    bound = {"a": 50.0, "e": 0.9, "i": 1.0, "asc_node": 0.0, "arg_per": 0.0}
    assert hyp.inbound_reminder(bound, 1e-4, EPOCH, PLANETS) is None


# --- Monte Carlo clone summary ------------------------------------------------------------------
TIGHT = np.diag([1e-12] * 3 + [1e-16] * 3)


def test_mc_characterises_an_unbound_orbit():
    """3I-like at perihelion with a tight cloud: every clone is unbound. Its clones used to convert
    to NaN mean anomaly / Tp and be dropped wholesale, so an interstellar orbit had no sigmas."""
    sig, U, rows, bf = mc.monte_carlo_elements(_peri_state(1.3564, 6.1394), TIGHT, EPOCH,
                                               element_vector, ELEM_NAMES, n=200)
    assert sig is not None and bf == 0.0 and U is None
    for nm in ("a", "e", "q", "i", "M", "Tp"):
        assert math.isfinite(sig[nm][0]) and math.isfinite(sig[nm][1]), nm
    assert math.isnan(sig["Q"][1]) and math.isnan(sig["P"][1])
    rows2 = mc.element_rows_from_cov(_peri_state(1.3564, 6.1394), TIGHT, EPOCH, element_vector,
                                     n=50)
    assert rows2.shape[0] == 50 and np.all(rows2[:, 1] > 1.0)


def test_mc_unchanged_for_an_ellipse():
    sig, _U, _rows, bf = mc.monte_carlo_elements(_peri_state(1.5, 0.3), TIGHT, EPOCH,
                                                 element_vector, ELEM_NAMES, n=200)
    assert bf == 1.0 and all(math.isfinite(sig[nm][1]) for nm in ELEM_NAMES)


# --- covariance determinacy verdict ---------------------------------------------------------
class _CovFit:
    def __init__(self, state, cov):
        self._s, self._c = np.asarray(state, float), np.asarray(cov, float)

    def state(self):
        return self._s

    def covariance_matrix(self, kind):
        return self._c

    def elements(self):
        return dict(zip(ELEM_NAMES, element_vector(self._s, EPOCH).tolist()))

    def bound_fraction(self):
        return mc.bound_fraction_from_cov(self._s, self._c)


def test_constraint_does_not_banner_a_determined_unbound_orbit():
    """An unbound nominal used to be banner-ed -- orbit suppressed, exports refused -- for being
    unbound alone. A tight cloud, all of it unbound, is a determined orbit."""
    assert constraint._cov_assess(_CovFit(_peri_state(1.3564, 6.1394), TIGHT)) is None


def test_constraint_banners_an_unbound_nominal_whose_cloud_straddles_e1(monkeypatch):
    monkeypatch.setattr(constraint, "_banner", lambda fit, method, **kw: {"method": method, **kw})
    cloud = np.diag([1e-12] * 3 + [1e-8] * 3)        # speed sigma >> the margin over escape
    got = constraint._cov_assess(_CovFit(_peri_state(2.0, 1.0005), cloud))
    assert got is not None and 0.1 < got["bf"] < 0.9


# --- MPCORB writers ----------------------------------------------------------------------------
class _UnboundStub:
    designation = packed_designation = "TEST"

    def elements(self):
        return {"a": -1.27, "e": 1.2011, "q": 0.2553, "Q": float("nan"), "i": 122.7,
                "asc_node": 24.6, "arg_per": 241.8, "M": -40.0, "Tp": EPOCH, "n": 0.7,
                "P": float("nan")}


def test_mpcorb_refuses_an_unbound_orbit_by_name():
    with pytest.raises(ValueError, match="no MPCORB form"):
        egress.to_mpcorb(_UnboundStub(), written_jd=EPOCH)


def test_mpcorb_batch_skips_an_unbound_orbit():
    assert egress.mpcorb_batch([_UnboundStub()], written_jd=EPOCH) == egress.MPCORB_HEADER


# --- the pipeline gates an unbound orbit used to fall out of --------------------------------
def test_unphysical_penalty_allow_unbound_drops_only_the_unbound_term():
    from pyroorb.orbit_common import unphysical_penalty
    assert unphysical_penalty(-1.27, 1.2, 0.255) == 1e6
    assert unphysical_penalty(-1.27, 1.2, 0.255, allow_unbound=True) == 0.0
    assert unphysical_penalty(-1.0, 2.0, 0.01, allow_unbound=True) == 1e6      # sun-plunging still


@pytest.mark.parametrize("desig,is_comet", [("3I", True), ("10P", True), ("73P-B", True),
                                            ("2I/Borisov", True), ("C/2025 N1", True),
                                            ("I/2017 U1", True), ("2023 QJ3", False),
                                            ("433", False), ("1862", False)])
def test_numbered_comet_designations_trigger_the_comet_path(desig, is_comet):
    from pyroorb import fit_object

    class _F:
        designation = desig
        designations = []

    assert fit_object._is_comet_designation(_F()) is is_comet
