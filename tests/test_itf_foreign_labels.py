"""The recovery harness labels each foreign tracklet of a linkage against the object's own
full-arc orbit (``harness.label_foreign``), so a live scan counts its false links instead of
listing them for someone to check.

2014 EJ64: one of its own tracklets, relabelled as an ITF tracklet, is genuine; the G96
tracklet VS8CE87 (2014 Sep 23), which a 2-night seed of EJ64 once linked, is another body.
"""
import json
import os

import pytest

from pyroorb.fit_object import fit_orbit
from pyroorb.obs_sources import load_obscodes
from pyroorb.itf.harness import _fake_line, label_foreign

from _itf_fixtures import load_fixture, load_obs

pytestmark = pytest.mark.slow

HERE = os.path.dirname(os.path.abspath(__file__))


def test_foreign_tracklets_are_labelled_against_the_full_arc():
    oc = load_obscodes()
    obs, meta = load_obs(load_fixture(os.path.join(HERE, "itf_fixtures", "2014_EJ64.json")), oc)
    full = fit_orbit(obs, oc, desig="2014 EJ64", meta=meta)
    own = [_fake_line(o["obs80_line"], "OWNTRK1") for o in obs
           if o["obs80_line"][15:25] == "2020 09 17"]
    assert len(own) >= 3, "the 2020 Sep 17 F51 tracklet is the own control"
    with open(os.path.join(HERE, "itf_negative", "2014_EJ64_2night.json"), encoding="utf-8") as fh:
        stranger = [l for l in json.load(fh)["obs80"].splitlines() if l[5:12].strip() == "VS8CE87"]
    assert len(stranger) == 4
    labels = label_foreign(full, "\n".join(own + stranger), ["OWNTRK1", "VS8CE87"], oc)
    assert labels["OWNTRK1"]["label"] == "genuine", labels
    assert labels["VS8CE87"]["label"] == "false", labels
    assert labels["VS8CE87"]["miss_arcsec"] > 1e5
