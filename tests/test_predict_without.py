"""FittedOrbit.predict_without: a warm leave-out scored as a predictive chi2.

On the frozen 2022 DF9 negative fixture (a 3-night seed plus two 2006 SDSS pairs that are
NOT the object): the September pair, left out, is NOT predicted by the seed plus the August
pair; the August pair, left out, is. And the test never changes the fit it is asked of.
"""
import json
import os

import numpy as np
import pytest

from pyroorb.obs_sources import load_obscodes
from pyroorb.itf.fitter import fit_astrometry
from pyroorb.itf.leaveout import warm_predictive
from pyroorb.itf.obs80_fields import obs_unique_id

pytestmark = pytest.mark.slow

_FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "itf_negative", "2022_DF9.json")


@pytest.fixture(scope="module")
def df9():
    from pyroorb.itf.obsreg import AdesRegistry
    from _itf_fixtures import load_obs
    with open(_FX) as fh:
        fx = json.load(fh)
    oc = load_obscodes()
    reg = AdesRegistry()
    reg.add_obs(load_obs(fx, oc)[0])        # weighted as the search weights them
    tf = fit_astrometry(fx["obs80"], oc, registry=reg, max_sec=300)
    assert tf.ok and tf.used == tf.count, tf.error
    lines = [l for l in fx["obs80"].splitlines() if l.strip()]
    pairs = {t: {obs_unique_id(l) for l in lines if l[5:12].strip() == t}
             for t in fx["foreign_trksubs"]}
    return tf.fit, pairs


def test_the_pair_only_the_joint_fit_holds_is_not_predicted(df9):
    fit, pairs = df9
    sep = warm_predictive(fit, pairs["0f63ff"])
    aug = warm_predictive(fit, pairs["0e3b9d"])
    assert sep["dof"] == 4 and aug["dof"] == 4
    assert sep["p"] < 0.01, sep
    assert aug["p"] > 0.05, aug


def test_it_never_changes_the_fit(df9):
    fit, pairs = df9
    state, mask = np.array(fit._state, copy=True), np.array(fit._mask, copy=True)
    warm_predictive(fit, pairs["0f63ff"])
    assert np.array_equal(fit._state, state) and np.array_equal(fit._mask, mask)


def test_an_empty_selection_is_an_error_not_a_number(df9):
    fit, _pairs = df9
    assert "error" in fit.predict_without(lambda o: False)
