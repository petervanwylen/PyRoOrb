"""No exit from bound-first skips the rules, and the bound orbit they weigh is looked for where
bound orbits are. Real astrometry on both sides of the question:

* two nights of an ORDINARY asteroid, whose free fit runs off to e = 8.8 ... 458052 along the flat
  range valley. The capped walk started from that optimum lands at 8-38" and used to be read as
  "no bound orbit exists" -- one exit delivered the unbound fit with NEITHER rule applied,
  and the other scored the comparison at hundreds of sigma. Each object's own orbit fits the same
  nights at 0.08-0.31";
* two nights of 1I/'Oumuamua, which must stay unbound through the same code.
"""
import pytest

import pyroorb.fit_object as fo
from pyroorb import hyperbolic
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


@pytest.mark.parametrize("name, pick, true_e, was_e", [
    ("107474", (-2, -1), 0.15, 8.8),            # main belt, 6 obs: left by the unweighed exit
    ("2010_LP68", (0, 1), 0.05, 458052.6),      # a = 36 au, 17 obs: bound only between ~30 and ~48 au
    ("2014_FO69", (-2, -1), 0.30, 1905.9),      # 4 obs: "bound refused at 462 sigma" on two dof
])
def test_two_nights_of_a_bound_object_are_not_delivered_unbound(name, pick, true_e, was_e):
    fit = _fit(name, pick)
    uc = fit.unbound_check
    assert uc["e_free"] > 2.0                    # the free fit still runs away: that is the arc
    assert uc["kept"] == "bound" and fit.elements()["e"] < 1.0
    assert uc["comparator"] == "admissible"      # ...and the walk was not the one that found it
    assert uc["admissible"] == "ok" and uc["bound_seeds"] > 0     # the record says the scan ran
    assert not uc["arc_ok"] and uc["station_nights"] == 2
    # the bound orbit is a FIT, not the 8-38" the walk called the best available
    assert fit.rms_residual() < 0.30
    assert uc["rms_attempt"] > 8.0


def test_two_nights_of_oumuamua_stay_unbound():
    """2017-10-18 and -19: 18 obs from six stations. The admissible-region fits run and the best
    bound orbit is still refused at the tracklet level -- the night bar's escape, earned."""
    fit = _fit("1I", (2, 3))
    uc = fit.unbound_check
    assert uc["kept"] == "unbound" and fit.elements()["e"] == pytest.approx(1.20, abs=0.05)
    assert uc["station_nights"] == 6 and not uc["arc_ok"]
    assert uc["damaged"] and uc["bound_refused"]
    assert uc["rms_bound"] > 1.8 * uc["rms_free"]


def test_the_full_oumuamua_arc_is_weighed_not_waved_through():
    """The capped walk cannot get under e = 1 here (it stops at e 1.14 and 39"). That used to END
    the function -- unbound delivered, no sigma recorded. The attempt is a LOWER BOUND on what any
    bound orbit costs, and the rules are applied to it."""
    obs, meta = fixture_obs("1I")
    fit = fo.fit_orbit(obs, obs_sources.load_obscodes(), desig="1I", meta=meta, auto_nongrav=False)
    uc = fit.unbound_check
    assert uc["kept"] == "unbound" and uc["arc_ok"]
    # The bound orbit is refused overwhelmingly: the raw statistic runs to 1e4+, which as a
    # significance saturates the reportable ceiling (hyperbolic.EVIDENCE_SIGMA_CAP) -- the
    # record says so rather than printing a number double precision cannot support.
    assert uc["damaged"] and uc["sigma_is_floor"]
    assert uc["sigma"] == pytest.approx(hyperbolic.EVIDENCE_SIGMA_CAP)
    assert uc["dchi2_raw"] > 1e4
    assert uc["e_attempt"] > 1.0 and uc["rms_attempt"] > 20.0
