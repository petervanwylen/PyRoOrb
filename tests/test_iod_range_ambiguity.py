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


# GS26063 (40 NEOCP obs over 2.2 d, frozen in its fixture): a retrograde near-parabolic orbit
# (q 0.673 au, i 124 deg). Its re-rank rivals correct to e 1.01-1.08 at 1.2-2.5", far better
# than the 17.5" Atira seed -- judged unphysical for e >= 1 they lost, and the Atira fit threw
# out the discovery night at 100 sigma.

def test_rerank_keeps_a_near_parabolic_basin():
    obs, _ = fixture_obs("GS26063")
    obs = sorted(obs, key=lambda o: o["jd_utc"])
    oc = obs_sources.load_obscodes()
    times = np.array([o["jd_utc"] for o in obs])
    rh = [los_unit(o["ra_deg"], o["dec_deg"]) for o in obs]
    R = _observer_geometry(obs, oc)["Robs"]
    st, _method, score, _sub, _t0 = iod.initial_orbit(obs, times, rh, R)
    a, e, q = iod.quick_aeq(st)
    assert 0.6 < q < 0.75 and e > 0.9, f"IOD kept the Atira basin: a={a:.3f} e={e:.3f} q={q:.3f}"
    assert a > 0 and e < 1.0, "the seed must be handed on bound"
    assert score < 5.0


def test_inside_parabola_keeps_position_and_direction():
    mu = iod.MU_SUN
    r = np.array([0.7, 0.2, 0.1])
    vesc = np.sqrt(2 * mu / np.linalg.norm(r))
    v = np.array([0.1, 1.0, 0.3]); v *= 1.05 * vesc / np.linalg.norm(v)   # hyperbolic
    out = iod._inside_parabola(np.concatenate([r, v]))
    a, e, _ = iod.quick_aeq(out)
    assert a > 0 and e < 1.0
    assert np.allclose(out[:3], r)
    assert np.allclose(out[3:] / np.linalg.norm(out[3:]), v / np.linalg.norm(v))
    bound = np.concatenate([r, v * 0.8])
    assert iod._inside_parabola(bound) is bound


@pytest.mark.slow
def test_gs26063_delivers_the_near_parabolic_orbit():
    """End to end on the frozen GS26063 obs: the delivered ORBIT, not just its rms. The
    regression golden alone would excuse a wrong basin that fits fewer obs at a lower rms.
    find_orb: q 0.678, i 124.13, node 211.23, peri 105.08 deg, 33/40 @ 0.54". The spurious
    Atira (q 0.43, i 3.9 deg, 27/40 @ 3.9", N86 discovery night rejected) must never return."""
    obs, meta = fixture_obs("GS26063")
    fit = fit_orbit(obs, obs_sources.load_obscodes(), desig=meta["designation"], meta=meta)
    q, i = fit.perihelion_distance()[0], fit.inclination()[0]
    node, peri = fit.ascending_node()[0], fit.argument_of_perihelion()[0]
    assert 0.65 < q < 0.70, f"q = {q:.3f}"
    assert 121.0 < i < 127.0, f"i = {i:.2f}"
    assert 208.0 < node < 215.0, f"node = {node:.2f}"
    assert 102.0 < peri < 108.0, f"peri = {peri:.2f}"
    assert fit.eccentricity()[0] > 0.95
    assert fit.n_observations() >= 33, f"only {fit.n_observations()}/40 kept"
    assert fit.rms_residual() < 0.8
    kept_n86 = sum(1 for r in fit.residuals() if r["obscode"] == "N86" and r["included"])
    assert kept_n86 == 3, f"discovery night: {kept_n86}/3 N86 obs kept"
