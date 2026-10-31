"""find_orb-compatible `covar.json` egress (egress.covariance_document / to_covar_json).

find_orb writes this file at the end of every successful `full_improvement()`
(orb_func.cpp): three keys -- `covar`, `state_vect`, `epoch` -- describing the solved
parameter vector in the heliocentric J2000 ECLIPTIC frame, au and au/day.

Two kinds of check here. The LAYOUT tests pin the byte shape against a real find_orb
run (the skeleton below was produced by `fo` on a fixture's astrometry and is what
makes a diff against find_orb meaningful). The SEMANTIC tests pin the two things a
consumer would silently misread: that `state_vect` is the ECLIPTIC state (not our
native equatorial one) and that the covariance's nongrav rows are in the units
`nongrav_params` reports, not the LM's internal scaled ones.
"""
import json
import math
import os
import re

import numpy as np
import pytest

from pyroorb import egress


# The exact shape find_orb's own covar.json has, numbers replaced by N. Produced by
# running the compiled `fo` on a fixture's OBS80 and skeletonising the result; a
# rendering that stops matching this has stopped being diffable against find_orb.
FINDORB_SKELETON = """{ "covar": [
   [N, N, N, N, N, N  ],
   [N, N, N, N, N, N  ],
   [N, N, N, N, N, N  ],
   [N, N, N, N, N, N  ],
   [N, N, N, N, N, N  ],
   [N, N, N, N, N, N  ]
], "state_vect": [
    N,
    N,
    N,
    N,
    N,
    N ],
  "epoch": N
}"""


def _skeleton(text):
    return re.sub(r"-?\d+\.?\d*(e[-+]\d+)?", "N", text)


class _FakeFit:
    """The minimum surface `covariance_document` reads: a parameter covariance with
    its names, the equatorial state, the epoch and the nongrav model name."""

    def __init__(self, cov, names, state, epoch=2459000.5, nongrav_model=None):
        self._cov, self._names = np.asarray(cov, float), list(names)
        self._state, self._epoch = np.asarray(state, float), float(epoch)
        self.nongrav_model = nongrav_model

    def parameter_covariance(self):
        return self._cov, self._names

    def state(self):
        return self._state.copy()

    def epoch_jd(self):
        return self._epoch


def _gravity_fit():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(6, 6))
    cov = A @ A.T * 1e-8
    return _FakeFit(cov, ["x", "y", "z", "dx", "dy", "dz"],
                    [1.5, -0.4, 0.1, 0.002, 0.013, 0.004])


# --- layout ---------------------------------------------------------------------
def test_layout_matches_a_real_find_orb_file():
    text = egress.to_covar_json(_gravity_fit())
    assert _skeleton(text) == FINDORB_SKELETON


def test_layout_is_valid_json_with_findorb_key_order():
    doc = json.loads(egress.to_covar_json(_gravity_fit()))
    assert list(doc) == ["covar", "state_vect", "epoch"]
    assert len(doc["covar"]) == 6 and all(len(r) == 6 for r in doc["covar"])
    assert len(doc["state_vect"]) == 6


def test_state_vect_stays_six_long_on_a_nongrav_fit():
    # find_orb prints `for (i = 0; i < 6; i++)` whatever n_params is: the matrix grows,
    # the state vector does not.
    cov = np.eye(7) * 1e-10
    fit = _FakeFit(cov, ["x", "y", "z", "dx", "dy", "dz", "A2"],
                   [1.5, -0.4, 0.1, 0.002, 0.013, 0.004], nongrav_model="yarkovsky")
    doc = json.loads(egress.to_covar_json(fit))
    assert len(doc["covar"]) == 7
    assert len(doc["state_vect"]) == 6


# --- semantics ------------------------------------------------------------------
def test_state_vect_is_the_ecliptic_state():
    fit = _gravity_fit()
    doc = egress.covariance_document(fit)
    M = egress._eq_to_ecl_6x6()
    assert np.allclose(doc["state_vect"], M @ fit.state())
    # and it is genuinely NOT the equatorial one we hold natively
    assert not np.allclose(doc["state_vect"], fit.state())


def test_covariance_is_rotated_and_symmetric():
    fit = _gravity_fit()
    C = np.array(egress.covariance_document(fit)["covar"])
    M = egress._eq_to_ecl_6x6()
    cov, _ = fit.parameter_covariance()
    assert np.allclose(C, M @ cov @ M.T)
    assert np.allclose(C, C.T, rtol=0, atol=0)     # exactly, not nearly


def test_nongrav_rows_are_not_rotated():
    # A1/A2/A3 are radial/transverse/normal scalars in a co-moving frame -- the obliquity
    # rotation touches the six state rows and nothing else.
    cov = np.eye(7)
    cov[6, 6] = 4.0
    fit = _FakeFit(cov, ["x", "y", "z", "dx", "dy", "dz", "A2"],
                   [1.5, -0.4, 0.1, 0.002, 0.013, 0.004], nongrav_model="yarkovsky")
    C = np.array(egress.covariance_document(fit)["covar"])
    assert C[6, 6] == pytest.approx(4.0)


def test_one_parameter_srp_is_converted_to_findorbs_gm_fraction():
    # find_orb's FORCE_MODEL_SRP slot-6 parameter is a dimensionless fraction of the
    # solar GM (runge.cpp subtracts it from the solar term), not an acceleration.
    cov = np.eye(7)
    fit = _FakeFit(cov, ["x", "y", "z", "dx", "dy", "dz", "A1"],
                   [1.5, -0.4, 0.1, 0.002, 0.013, 0.004], nongrav_model="srp")
    C = np.array(egress.covariance_document(fit)["covar"])
    assert C[6, 6] == pytest.approx(1.0 / egress._FO_SOLAR_GM ** 2)
    # the two-parameter SRP model keeps au/day^2 on both sides -- no conversion
    fit2 = _FakeFit(np.eye(8), ["x", "y", "z", "dx", "dy", "dz", "A1", "A2"],
                    [1.5, -0.4, 0.1, 0.002, 0.013, 0.004], nongrav_model="srp2")
    C2 = np.array(egress.covariance_document(fit2)["covar"])
    assert C2[6, 6] == pytest.approx(1.0) and C2[7, 7] == pytest.approx(1.0)


# --- offline end-to-end on a cached fixture -------------------------------------
@pytest.mark.slow
def test_end_to_end_on_a_fixture():
    """A real fit's covar.json: the state round-trips back to the reported elements
    (which proves the rotation direction) and the matrix is the fit's own covariance."""
    from _fixtures import load_fixture, iter_fixtures
    import pyroorb.fit_object as fo
    from pyroorb import obs_sources
    from pyroorb.orbital_elements import element_vector, ELEM_NAMES
    path = next(p for p in iter_fixtures() if os.path.basename(p).startswith("2001_KQ76"))
    fx = load_fixture(path)
    fit = fo.fit_orbit(fx["obs"], obs_sources.load_obscodes(),
                       desig=fx["meta"]["designation"], meta=fx["meta"],
                       verbose=False, force=fx.get("force", False))
    doc = json.loads(egress.to_covar_json(fit))
    assert _skeleton(egress.to_covar_json(fit)) == FINDORB_SKELETON
    assert doc["epoch"] == pytest.approx(fit.epoch_jd(), abs=1e-5)   # written as "%f"
    M = egress._eq_to_ecl_6x6()
    back = np.linalg.solve(M, np.array(doc["state_vect"]))
    el = dict(zip(ELEM_NAMES, element_vector(back, fit.epoch_jd())))
    for name in ("a", "e", "i", "asc_node", "arg_per", "M"):
        assert el[name] == pytest.approx(fit.elements()[name], rel=1e-9, abs=1e-9)
    # the state block IS the fit's state covariance, rotated
    C = np.array(doc["covar"])
    assert np.allclose(C[:6, :6], M @ fit.covariance_matrix("state") @ M.T)


@pytest.mark.slow
def test_nongrav_diagonal_matches_the_reported_sigma():
    """The A-coefficient variance must be in au/day^2, i.e. sqrt of its diagonal is the
    sigma the fit reports -- the LM carries it in `nongrav.MODELS[...]["scale"]` units,
    and forgetting that lift misreports the variance by scale^2 (1e-28 for yarkovsky)."""
    from _fixtures import load_fixture, iter_fixtures
    import pyroorb.fit_object as fo
    from pyroorb import obs_sources
    path = next(p for p in iter_fixtures() if os.path.basename(p).startswith("2003_HM"))
    fx = load_fixture(path)
    fit = fo.fit_orbit(fx["obs"], obs_sources.load_obscodes(),
                       desig=fx["meta"]["designation"], meta=fx["meta"],
                       verbose=False, force=fx.get("force", False))
    assert fit.nongrav_model == "yarkovsky"
    cov, names = fit.parameter_covariance()
    assert names == ["x", "y", "z", "dx", "dy", "dz", "A2"]
    assert math.sqrt(cov[6, 6]) == pytest.approx(fit.nongrav_sigmas["A2"], rel=1e-9)
    doc = json.loads(egress.to_covar_json(fit))
    assert len(doc["covar"]) == 7 and len(doc["state_vect"]) == 6
