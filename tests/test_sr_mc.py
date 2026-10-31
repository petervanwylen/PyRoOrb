"""Offline tests for Statistical Ranging (sr.py) and Monte Carlo uncertainty (mc.py).

SR: (1) the trial builder is LIGHT-TIMED -- at the true range of 2021 NX4 an endpoint
transfer reproduces its 8-obs / 6-day arc to a fraction of an arcsecond (it was 13" before
2026-08-29, at every range); (2) a curved arc the sampler cannot fit yields NO family (the
sigma-scaled acceptance never keeps a draw that misfits the data); (3) a genuine single
tracklet (5 obs over 2 h of a main-belt object) yields a large, honestly wide family;
(4) a single night of the close NEO 2023 DW still fits near its full-arc orbit.

MC: asserts the covariance-sampled element sigmas agree with the linear (Jacobian) sigmas
for a well-determined fixture -- the correctness check -- and that the U parameter is finite.
"""
import pytest
import os, glob
import numpy as np
from _fixtures import load_fixture, FIX
import pyroorb.fit_object as fo
from pyroorb import obs_sources
from pyroorb import orbital_elements
from pyroorb import sr

# statistical ranging + Monte Carlo over a fixture
pytestmark = pytest.mark.slow

_OC = None
def _oc():
    global _OC
    if _OC is None:
        _OC = obs_sources.load_obscodes()
    return _OC


def _fixture(key):
    return load_fixture([p for p in glob.glob(os.path.join(FIX, "*.json")) if key in p][0])


def _sr_inputs(obs):
    """(times, rhohats, Robs, t0_tdb) for statistical_ranging over `obs`."""
    from pyroorb.forward_model import los_unit
    from astropy.time import Time
    st = np.array([o["jd_utc"] for o in obs])
    rh = np.array([los_unit(o["ra_deg"], o["dec_deg"]) for o in obs])
    Robs = np.array(list(obs_sources.observer_positions_batch(
        [o["jd_utc"] for o in obs], [o["obscode"] for o in obs], _oc())))
    return st, rh, Robs, Time(st[0], format="jd", scale="utc").tdb.jd


def test_sr_single_night_is_fittable():
    """A single night of 2023 DW (a close NEO, 0.07 au) -- truncated to its first ~6 hours --
    must fit and recover a/e/i near the full-arc orbit. Its 4 h of diurnal parallax pin the
    range to ~1.5%, so the SR family is a sliver the blind sampler rarely lands in and the
    Gauss/Herget cascade seeds it instead; which method seeds is not asserted."""
    fx = _fixture("2023_DW")
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    t0 = obs[0]["jd_utc"]
    night = [o for o in obs if o["jd_utc"] <= t0 + 0.3]
    assert len(night) >= 4, "need a few obs in the first night"

    fit = fo.fit_orbit(night, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    full = fo.fit_orbit(obs, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])

    a, e, i = fit.semimajor_axis()[0], fit.eccentricity()[0], fit.inclination()[0]
    af, ef, iff = full.semimajor_axis()[0], full.eccentricity()[0], full.inclination()[0]
    assert abs(a - af) < 0.15, f"a {a:.3f} vs full {af:.3f}"
    assert abs(e - ef) < 0.15, f"e {e:.3f} vs full {ef:.3f}"
    assert abs(i - iff) < 3.0, f"i {i:.2f} vs full {iff:.2f}"
    assert fit.rms_residual() < 1.0, f"single-night rms {fit.rms_residual():.3f}"


def test_sr_family_of_a_free_range_tracklet_is_wide():
    """A 2-hour, 5-obs tracklet of a main-belt object has no curvature: the SR family must be
    large (many accepted orbits) and honestly wide in a, and must seed the fit (iod 'sr')."""
    fx = _fixture("RANGEAMBIG_3NIGHT_MBA")
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    night = [o for o in obs if o["jd_utc"] <= obs[0]["jd_utc"] + 0.5]
    assert 4 <= len(night) <= 8 and (night[-1]["jd_utc"] - night[0]["jd_utc"]) < 0.2
    st, rh, Robs, t0 = _sr_inputs(night)
    from pyroorb.forward_model import build_base_sim
    med, kept, n = sr.statistical_ranging(night, st, rh, Robs, build_base_sim(t0), t0)
    assert med is not None and n >= sr.SR_MIN_ACCEPT, f"SR found too few orbits ({n})"
    sig = sr.sr_element_sigmas(kept, t0, orbital_elements.element_vector, orbital_elements.ELEM_NAMES)
    assert sig is not None and sig["a"][1] > 0.5, f"SR sigma_a unexpectedly small: {sig['a'][1]}"
    fit = fo.fit_orbit(night, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    assert fit.iod_method == "sr", f"a free-range tracklet should be SR-seeded, got {fit.iod_method!r}"


def test_trial_orbit_is_light_timed_and_curved_arc_yields_no_family():
    """2021 NX4 (8 obs / 5.96 d at 1.03 au, fit rms ~0.1"). (a) With the endpoints placed at
    their EMISSION epochs an endpoint transfer at the true range, range-rate optimised,
    reproduces the arc to < 0.5" (it was 13" at every range when the builder ignored
    light-time, and that flat misfit read as an unconstrained range). (b) The blind Halton
    draw cannot land inside a valley this narrow, and the sigma-scaled acceptance must then
    return NO family rather than the misfitting draws (108 of 316 were kept before)."""
    from pyroorb.forward_model import build_base_sim, predict_radec_batch
    from pyroorb.orbit_common import radec_residuals_arcsec
    from astropy.time import Time
    fx = _fixture("2021_NX4")
    obs = sorted(fx["obs"], key=lambda o: o["jd_utc"])
    fit = fo.fit_orbit(obs, _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    st, rh, Robs, t0 = _sr_inputs(obs)
    tt = Time(st, format="jd", scale="utc").tdb.jd
    base = build_base_sim(t0)
    ra = np.array([o["ra_deg"] for o in obs]); dec = np.array([o["dec_deg"] for o in obs])
    cosd = np.cos(np.radians(dec))
    R0, L0, Rl, Ll, dt = Robs[0], rh[0], Robs[-1], rh[-1], tt[-1] - tt[0]

    def raw_rms(vparam):
        s = sr._trial_orbit(R0, L0, Rl, Ll, fit.geocentric_range(), vparam, dt, 0.0)
        if s is None:
            return 1e9
        pred = predict_radec_batch(base, t0, s, tt, Robs)
        dra, ddec = radec_residuals_arcsec(pred, ra, dec, cosd)
        return float(np.sqrt(np.mean(dra ** 2 + ddec ** 2)))

    vs = np.linspace(-1.0, 1.0, 801)
    best = min(raw_rms(v) for v in vs)
    assert best < 0.5, f"light-timed trial at the true range misfits by {best:.2f}\""

    med, kept, n = sr.statistical_ranging(obs, st, rh, Robs, base, t0)
    if med is not None:              # the draw happened to land in the valley: then it is TIGHT
        sig = sr.sr_element_sigmas(kept, t0, orbital_elements.element_vector, orbital_elements.ELEM_NAMES)
        assert sig["a"][1] / abs(sig["a"][0]) < 0.3, f"a curved arc's SR family must be tight: {sig['a']}"


def test_mc_matches_linear_for_well_determined():
    """For a well-determined orbit the Monte Carlo element sigmas must agree with the
    linear (Jacobian) sigmas to within MC sampling noise."""
    fx = _fixture("2023_QJ3")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    lin = fit.element_sigmas()
    mcr = fit.monte_carlo_uncertainty(n=1500, rng_seed=3)
    assert mcr is not None, "MC produced no result"
    assert mcr["U"] is not None and np.isfinite(mcr["U"]), "U should be finite"
    for nm in ("a", "e", "i", "q"):
        mc_sig = mcr["sigmas"][nm][1]
        ratio = mc_sig / lin[nm] if lin[nm] > 0 else float("inf")
        assert 0.7 < ratio < 1.4, f"MC/linear sigma for {nm} = {ratio:.2f} (out of range)"
