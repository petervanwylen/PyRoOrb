"""Bound-first across a multi-year gap. Real astrometry on both sides:

* three nights of a known-BOUND object spread over two apparitions 8-39 years apart. The free fit
  is a wrong-revolution hyperbola joining them; the only comparator used to be a penalised walk
  from it, which stopped in a wrong basin (24.6" for 107474, 15 000" for 2010 TA192), so the free
  hyperbola was delivered, labelled `unbound-unsearched`. The revolution-count search finds bound
  orbits that fit these observations BETTER than the hyperbola did;
* 2I/Borisov's first nights, one 117-day apparition: nothing to bridge, the label and the
  unbound orbit stay.
"""
import pytest

import pyroorb.fit_object as fo
from pyroorb import obs_sources
from _fixtures import fixture_obs

pytestmark = pytest.mark.slow


def _nights(obs):
    obs = sorted(obs, key=lambda o: o["jd_utc"])
    out = [[obs[0]]]
    for a, b in zip(obs, obs[1:]):
        if b["jd_utc"] - a["jd_utc"] > 0.5:
            out.append([])
        out[-1].append(b)
    return out


def _fit(name, pick):
    obs, meta = fixture_obs(name)
    ns = _nights(obs)
    sub = [o for k in pick for o in ns[k]]
    return fo.fit_orbit(sub, obs_sources.load_obscodes(), desig=meta["designation"], meta=meta,
                        auto_nongrav=False)


@pytest.mark.parametrize("name, pick, was_e, walk_rms", [
    ("107474", (0, 1, 2), 1.44, 24.6),            # two 1961 plates + one 1999 night, 39 yr
    ("2010_TA192", (-3, -2, -1), 5.9e12, 15000.0),  # 2020 + 2022 (the 2013 night is rejected), N = 0
    ("PAIR_2026DV62", (0, 1, 2), 38.4, 2.35),     # 2016 W84 + 2024 F52/G96, 8.6 yr
])
def test_a_long_gap_arc_of_a_bound_object_is_delivered_bound(name, pick, was_e, walk_rms):
    fit = _fit(name, pick)
    uc = fit.unbound_check
    assert uc["e_free"] > min(1.3, was_e)                    # the free fit is still the hyperbola
    assert uc["rms_attempt"] > 0.9 * walk_rms                # ...and the walk still cannot leave it
    assert uc["kept"] == "bound" and fit.elements()["e"] < 1.0
    assert uc["comparator"] == "bridge" and uc["admissible"] == "bridged"
    assert uc["bridge"] == "bridged" and uc["bridge_seeds"] > 0 and uc["bridge_revs"] >= 0
    # the bound orbit fits no worse than the hyperbola did: the data never refused it
    assert uc["dchi2_block"] < 0 and not uc["damaged"]
    assert fit.rms_residual() <= uc["rms_free"] + 0.01


def test_a_single_far_observation_is_bridged_too():
    """PAIR 1999 KK17's last three nights: one 1999 observation and two 2016 nights. The free fit sat
    at 10" with e 1.006 and no bound fit at all (`unbound-no-bound-fit`); a bound orbit threads the
    five observations exactly. Five observations over 17 years at one far position are likely a
    mislinkage, but that is not this gate's question: bound-first delivers the bound orbit."""
    fit = _fit("PAIR_1999KK17", (-3, -2, -1))
    uc = fit.unbound_check
    assert uc["rms_free"] > 5.0
    assert uc["kept"] == "bound" and uc["comparator"] == "bridge"
    assert fit.rms_residual() < 0.1


def test_one_apparition_of_borisov_is_not_bridged():
    """2I's first four nights (94 days, 5 observations): a single apparition, so the bridge is not
    attempted, says so, and the unbound verdict and its unearned label are what they were."""
    fit = _fit("2I", (0, 1, 2, 3))
    uc = fit.unbound_check
    assert uc["bridge"] == "single-apparition" and uc["bridge_seeds"] == 0
    assert uc["admissible"] == "arc-too-long" and uc["kept"] == "unbound-unsearched"
    assert fit.elements()["e"] > 3.0
