"""The multi-revolution bridge must take the BEST gluing candidate, not the first.

2008 VC46's first two apparitions -- 11 obs over two 2008 nights and one 4-obs F51
tracklet in 2012 -- sit 172 degrees apart on the 2008-only orbit, so the period sweep
offers candidate orbits for the revolution count. Taking the first that glued put the
tracklet at 1.45" (about 7 sigma at F51's 0.2"), rejection then dropped 3 of the 4 F51
obs, and the delivered orbit sat 242 sigma from the full-arc semi-major axis with a
two-degree miss at the next apparition. Another candidate in the same list fits all 15
obs at chi2 10.3. The observations are taken from the PAIR_2016UK176 fixture, whose two
earliest apparitions are all 2008 VC46.
"""
import os

import numpy as np
import pytest

import _fixtures as FX
import pyroorb.fit_object as fo
from pyroorb.web import engine_api as E

pytestmark = pytest.mark.slow          # real engine fits

FULL_ARC_A = 2.5856                    # the full-arc orbit's semi-major axis, au


def _vc46_2008_2012():
    fx = FX.load_fixture(os.path.join(FX.FIX, "PAIR_2016UK176.json"))
    obs = sorted((o for o in fx["obs"] if o["designation"] == "2008 VC46"),
                 key=lambda o: o["jd_utc"])
    return [o for o in obs if o["jd_utc"] < 2456300.0]      # 2008 and the 2012 F51 tracklet


def test_vc46_two_apparitions_keep_the_whole_2012_tracklet():
    obs = _vc46_2008_2012()
    assert len(obs) == 15 and sum(o["obscode"] == "F51" for o in obs) == 4
    f = fo.fit_orbit(obs, E._obscodes(), desig="2008 VC46",
                     meta={"designation": "2008 VC46"}, verbose=False)
    kept = np.asarray(f._mask, bool)
    f51 = [k for o, k in zip(f._obs, kept) if o["obscode"] == "F51"]
    assert all(f51), f"the 2012 F51 tracklet lost {f51.count(False)} of its 4 obs"
    assert abs(f._val("a") - FULL_ARC_A) < 0.002, f"a = {f._val('a'):.5f} au"
