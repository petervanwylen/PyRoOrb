"""iod.SEED_RIVAL_FACTOR: the range-ambiguity re-rank must reach a right basin that scores
well behind the best raw seed.

P/2026 U1 (68 obs over 9 days, frozen in the P2026_U1 fixture): the best raw Vaisala seed
(4.96") is a near-Sun Aten 1.9 au from Earth; the right basin (a~2.9, 3.3 au away) scores
12.58" raw -- 2.5x behind, because the Vaisala distance grid steps 1.4x. Corrected, the right
basin fits 0.36" and the wrong one 1.03". The wrong distance leaves a hemisphere-split parallax
residual (northern stations +Dec, the southern W68 -3.5" Dec), which plain rejection "fixes"
by throwing out the only southern station.
"""
import numpy as np
import pytest

from _fixtures import fixture_obs
from pyroorb import iod, obs_sources
from pyroorb.fit_object import _observer_geometry, fit_orbit, los_unit


def _u1():
    obs, meta = fixture_obs("P2026_U1")
    return sorted(obs, key=lambda o: o["jd_utc"]), meta


def test_rerank_reaches_the_far_basin():
    obs, _ = _u1()
    oc = obs_sources.load_obscodes()
    times = np.array([o["jd_utc"] for o in obs])
    rh = [los_unit(o["ra_deg"], o["dec_deg"]) for o in obs]
    R = _observer_geometry(obs, oc)["Robs"]
    st, _method, _score, _sub, _t0 = iod.initial_orbit(obs, times, rh, R)
    a, e, q = iod.quick_aeq(st)
    assert q > 1.0, f"IOD kept the near-Sun basin: a={a:.3f} e={e:.3f} q={q:.3f}"


@pytest.mark.slow
def test_fit_keeps_the_southern_station():
    obs, meta = _u1()
    fit = fit_orbit(obs, obs_sources.load_obscodes(), desig=meta["designation"], meta=meta)
    a, e = fit.semimajor_axis()[0], fit.eccentricity()[0]
    assert a * (1 - e) > 1.0
    kept_w68 = sum(1 for r in fit.residuals() if r["obscode"] == "W68" and r["included"])
    assert kept_w68 >= 7, f"only {kept_w68}/8 W68 obs kept"
    assert fit.rms_residual() < 0.45
