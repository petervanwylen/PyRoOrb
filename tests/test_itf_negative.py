"""ITF NEGATIVE fixtures: frozen cases where the candidate tracklets are NOT the object, and the
search must not deliver them as a submittable linkage.

Each fixture in ``itf_negative/`` freezes a seed and foreign tracklets as verbatim OBS80 (plus
the ADES fields the search weights by). The harness seeds from the object's own apparition and
plants the foreign tracklets as the only candidates, offline. The recovery suite asks "is the
truth found"; this one asks "is a falsehood refused" -- a wrong identification sent to the MPC
is public and costly, a refused correct one costs a review.

    pytest tests/test_itf_negative.py
"""
import glob
import json
import os

import pytest

from pyroorb.obs_sources import load_obscodes
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.harness import run_recovery

from _itf_fixtures import load_obs

pytestmark = pytest.mark.slow

_NEG = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "itf_negative", "*.json")))


@pytest.mark.parametrize("path", _NEG, ids=[os.path.basename(p)[:-5] for p in _NEG])
def test_foreign_tracklets_are_not_submitted(path):
    with open(path, encoding="utf-8") as fh:
        fx = json.load(fh)
    oc = load_obscodes()
    obs, meta = load_obs(fx, oc)
    rep = run_recovery(fx["designation"], cfg=ItfConfig(), base_source=None, obscodes=oc,
                       obs_meta=(obs, meta), log=lambda *a: None)
    assert rep.n_fake_tracklets >= 1, f"no foreign tracklet was planted: {rep.reason}"
    assert rep.status < 1.0, (
        f"{fx['designation']}: the search delivered status {rep.status} (submittable) on "
        f"{rep.recovered_tracklets} foreign tracklet(s), rms {rep.rms:.3f}\" -- {fx['note']}")
