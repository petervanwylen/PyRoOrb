"""Offline tests for radar ingest + measurement model (radar.py).

- The OBS80 'R'/'r' pair parser is checked against the REAL sample record from
  https://www.minorplanetcenter.net/iau/info/RadarObs.html (433 Eros, Arecibo 1975:
  delay 150.885360 s +- 15 us, Doppler -1.3 +- 2.0 Hz @ 430 MHz, surface bounce).
- The JPL radar-API row conversion runs against a canned dump written to a temp
  cache dir (no network).
- The measurement model (delay incl. Shapiro + two-leg SR Doppler) is regression-
  gated on 2005 YU55 rows whose object states (JPL Horizons, radar-fit orbit) and
  expected outputs were frozen 2026-07; at freeze time every one of the 19 real
  Arecibo/Goldstone measurements matched within 1.05 sigma. Station positions are
  recomputed live (astropy ITRS->GCRS + the ASSIST DE440 Earth), so this also
  guards the station chain — the tolerance (0.05 us / 0.05 Hz) is far under the
  smallest real sigma (0.5 us / 0.15 Hz) but roomy against IERS-table drift.
"""
import json, tempfile
import numpy as np
import pytest

from pyroorb import radar
from pyroorb.obs_sources import load_obscodes

OBSCODES = load_obscodes()

EROS_L1 = "00433         R1975 01 22.187500  150885360    -         13     430 251 AJ102251"
EROS_L2 = "00433         r1975 01 22.187500S        15              20         251 AJ102251"


# --- pure parsing -----------------------------------------------------------
def test_extract_radar_value():
    # implicit decimal before the last 4 digits; trailing spaces in the fraction = zeros
    assert radar._extract_radar_value("  150885360    ") == 150885360.0
    assert radar._extract_radar_value("-         13   ") == -1.3
    assert radar._extract_radar_value("S        15    ") == 15.0     # unit tag byte skipped
    assert radar._extract_radar_value("               ") == 0.0
    assert radar._extract_radar_value("+    7544376250") == 754437.625


def test_parse_obs80_radar_pair():
    rows = radar.parse_obs80_radar_pair(EROS_L1, EROS_L2, OBSCODES)
    assert len(rows) == 2, rows
    d, p = rows
    assert d["kind"] == "delay" and p["kind"] == "doppler"
    assert abs(d["value"] - 150.885360) < 1e-9 and abs(d["sigma"] - 15e-6) < 1e-12
    assert abs(p["value"] - (-1.3)) < 1e-9 and abs(p["sigma"] - 2.0) < 1e-9
    assert d["freq_hz"] == 430e6 and d["rx"] == "251" and d["tx"] == "251"
    assert d["bp"] == "P"                      # 'S' tag = surface (peak-power) bounce
    assert abs(d["jd_utc"] - 2442434.6875) < 1e-9
    # Arecibo's ITRS position resolves from the obscode parallax constants (~Earth radius)
    r = np.linalg.norm(d["rx_itrs_au"]) * 1.495978707e8
    assert 6350 < r < 6400, r


def test_ades_records_to_radar():
    rec = {"obstime": "2011-11-13T02:47:00Z", "delay": "32.606296070", "rmsdelay": "0.5",
           "doppler": "", "frq": "2380", "trx": "251", "rcv": "251", "com": "1",
           "permid": "308635"}
    rows = radar.ades_records_to_radar([rec], OBSCODES)
    assert len(rows) == 1 and rows[0]["kind"] == "delay"
    assert abs(rows[0]["value"] - 32.60629607) < 1e-12
    assert abs(rows[0]["sigma"] - 0.5e-6) < 1e-15          # rmsDelay is in MICROSECONDS
    assert rows[0]["bp"] == "C" and rows[0]["desig"] == "308635"


def test_fetch_radar_from_canned_dump():
    dump = {"count": "2",
            "fields": ["des", "epoch", "value", "sigma", "units", "freq", "rcvr", "xmit", "bp"],
            "data": [["308635", "2011-11-13 02:47:00", "32606296.07", "0.500", "us",
                      "2380", "-1", "-1", "C"],
                     ["433", "1975-01-22 04:30:00", "-1.3", "2.0", "Hz",
                      "430", "-99", "-99", "C"]],       # unknown station -> skipped
            "coords": {"-1": {"longitude": "293.2473068", "latitude": "18.3442199",
                              "altitude": "0.45334", "alt_units": "km",
                              "name": "Arecibo"}}}
    with tempfile.TemporaryDirectory() as td:
        (radar.Path(td) / "radar_astrometry.json").write_text(json.dumps(dump))
        rows = radar.fetch_radar(["308635"], cache_dir=td)
        assert len(rows) == 1 and rows[0]["kind"] == "delay"
        assert abs(rows[0]["value"] - 32.60629607) < 1e-9   # us -> s
        assert abs(rows[0]["sigma"] - 0.5e-6) < 1e-15
        assert rows[0]["freq_hz"] == 2380e6
        rows433 = radar.fetch_radar(["433"], cache_dir=td)
        assert rows433 == []                                # unknown station dropped


# --- measurement model (frozen 2005 YU55 rows; see module docstring) --------
_ARECIBO_ITRS = [1.597942227960685e-05, -3.71979335410032e-05, 1.3333919320604381e-05]

YU55_ROWS = [
    {"kind": "delay", "jd_utc": 2455306.4402777776, "value": 15.46106867, "sigma": 1e-06,
     "freq_hz": 2380e6, "rx": "-1", "tx": "-1", "rx_itrs_au": _ARECIBO_ITRS,
     "tx_itrs_au": _ARECIBO_ITRS, "bp": "C", "desig": "308635"},
    {"kind": "doppler", "jd_utc": 2455306.4402777776, "value": -43977.4883, "sigma": 0.15,
     "freq_hz": 2380e6, "rx": "-1", "tx": "-1", "rx_itrs_au": _ARECIBO_ITRS,
     "tx_itrs_au": _ARECIBO_ITRS, "bp": "C", "desig": "308635"},
    {"kind": "delay", "jd_utc": 2455884.6319444445, "value": 80.4709578, "sigma": 7.5e-07,
     "freq_hz": 2380e6, "rx": "-1", "tx": "-1", "rx_itrs_au": _ARECIBO_ITRS,
     "tx_itrs_au": _ARECIBO_ITRS, "bp": "C", "desig": "308635"},
]

# Heliocentric ICRF-equatorial states (au, au/day) at the receive TDB epochs, from
# JPL Horizons (radar-fit orbit) — frozen 2026-07 at full precision (an 8-digit
# repr loses ~1.5 km of position = ~5 us of delay, swamping the gate).
YU55_STATES = np.array([
    [[-0.8787726159484035, -0.44286294675337989, -0.1902107580823312,
      0.01458457472932107, -0.0099328528846775524, -0.0044784812899023993]],
    [[-0.87877261595088907, -0.44286294675168708, -0.19021075808156801,
      0.014584574729276829, -0.0099328528846992139, -0.0044784812899116289]],
    [[0.60974017002243219, 0.80193231768443485, 0.34995522957419578,
      -0.0086506484140882831, 0.013703755183769499, 0.0060528553593484167]],
])

# Expected model outputs at freeze (computed with the validated chain).
YU55_EXP_DELAY = [15.4610690632, None, 80.4709580522]
YU55_EXP_DOPPLER = [None, -43977.463949, None]


def test_radar_model_frozen_yu55():
    rctx = radar.radar_context(YU55_ROWS, OBSCODES)
    delay_s, dopp_hz = radar.radar_model(YU55_STATES, rctx)
    for i, exp in enumerate(YU55_EXP_DELAY):
        if exp is not None:
            assert abs(delay_s[i, 0] - exp) < 0.05e-6, (i, delay_s[i, 0], exp)
    for i, exp in enumerate(YU55_EXP_DOPPLER):
        if exp is not None:
            assert abs(dopp_hz[i, 0] - exp) < 0.05, (i, dopp_hz[i, 0], exp)
    # and the frozen orbit reproduces the REAL measurements within ~1 sigma
    z = radar.radar_residuals(YU55_STATES, rctx)[:, 0]
    assert np.all(np.abs(z) < 2.0), z


def test_residual_rows_match_kind():
    rctx = radar.radar_context(YU55_ROWS, OBSCODES)
    delay_s, dopp_hz = radar.radar_model(YU55_STATES, rctx)
    z = radar.radar_residuals(YU55_STATES, rctx)[:, 0]
    assert abs(z[0] - (YU55_ROWS[0]["value"] - delay_s[0, 0]) / 1e-6) < 1e-9
    assert abs(z[1] - (YU55_ROWS[1]["value"] - dopp_hz[1, 0]) / 0.15) < 1e-9


# ---------------------------------------------------------------------------
# Joint state + A + radar solve
# ---------------------------------------------------------------------------
# The 9 published JPL radar measurements of 2003 HM (sb_radar.api, frozen 2026-08-27),
# in radar.fetch_radar's row format -- station ITRS coordinates included so the test
# needs no dump, no cache and no network.
HM_RADAR = [
 {
  "kind": "doppler",
  "jd_utc": 2452755.4125,
  "value": -146454.0,
  "sigma": 4.0,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "doppler",
  "jd_utc": 2452755.4319444443,
  "value": -147572.1264,
  "sigma": 2.0,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "delay",
  "jd_utc": 2452755.4583333335,
  "value": 61.46923344,
  "sigma": 1e-06,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "doppler",
  "jd_utc": 2456767.388888889,
  "value": 87427.586,
  "sigma": 3.0,
  "freq_hz": 8560000000.0,
  "rx": "-14",
  "tx": "-14",
  "rx_itrs_au": [
   -1.573298742048472e-05,
   -3.102545141943669e-05,
   2.4579576588579436e-05
  ],
  "tx_itrs_au": [
   -1.573298742048472e-05,
   -3.102545141943669e-05,
   2.4579576588579436e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "doppler",
  "jd_utc": 2456772.4166666665,
  "value": -137240.0681,
  "sigma": 0.5,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "delay",
  "jd_utc": 2456772.4430555557,
  "value": 55.72161457,
  "sigma": 4e-06,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "delay",
  "jd_utc": 2456772.480555556,
  "value": 55.912192829999995,
  "sigma": 5e-07,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "doppler",
  "jd_utc": 2456772.480555556,
  "value": -141142.8101,
  "sigma": 0.15,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 },
 {
  "kind": "delay",
  "jd_utc": 2456773.456944444,
  "value": 61.16483007,
  "sigma": 5e-07,
  "freq_hz": 2380000000.0,
  "rx": "-1",
  "tx": "-1",
  "rx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "tx_itrs_au": [
   1.597942227960685e-05,
   -3.71979335410032e-05,
   1.3333919320604381e-05
  ],
  "bp": "C",
  "desig": "2003 HM"
 }
]


def _assist():
    from pyroorb import obs_sources
    backend, _ = obs_sources._force_backend()
    return backend if hasattr(backend, "helio_states_multi") else None


def test_helio_states_multi_per_particle_A():
    """The joint radar solve's d/dA columns ride A-PERTURBED particles: an (npart, 3) A
    gives each particle its own coefficients, a particle carrying the same A as a
    single-A call matches that call, and a wrong shape is refused loudly."""
    backend = _assist()
    if backend is None:
        pytest.skip("ASSIST backend unavailable")
    import pyroorb.nongrav as ngm
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0172, 0.0])   # ~circular 1 au test orbit
    t0, times = 2460000.5, np.array([2460050.5, 2460100.5])
    A = np.array([1e-8, 0.0, 0.0])
    one = backend.helio_states_multi(t0, [state], times, nongrav=(A, ngm.INVSQ_GR))
    per = backend.helio_states_multi(t0, [state, state], times,
                                     nongrav=(np.array([[0.0, 0.0, 0.0], A]), ngm.INVSQ_GR))
    assert np.allclose(per[:, 1, :], one[:, 0, :], atol=1e-12)
    assert not np.allclose(per[:, 0, :3], per[:, 1, :3], atol=1e-9)  # the A moved it
    try:
        backend.helio_states_multi(t0, [state], times,
                                   nongrav=(np.zeros((3, 3)), ngm.INVSQ_GR))
        assert False, "shape mismatch must raise"
    except ValueError:
        pass


@pytest.mark.slow
def test_joint_radar_nongrav_2003hm():
    """End-to-end, offline: 2003 HM's auto-discovered Yarkovsky orbit plus its 9
    frozen radar rows must deliver JPL's radar-anchored A2 -- the measured pre-fix failure
    was A2 = -1.87e-14 (6.25x small) with every radar row silently dropped, and the
    gravity-path joint solve on this object degraded the optical fit 6.6x (the case
    RADAR_MAX_OPTICAL_DEGRADE was calibrated on). The joint state+A solve must fit BOTH:
    JPL publishes A2 = -1.167683803864719e-13 +/- 1.909e-15 (we measured -1.172e-13,
    0.23 sigma; the bar here is deliberately loose so astrometry updates do not flake)."""
    pytest.importorskip("scipy")
    if _assist() is None:
        pytest.skip("ASSIST backend unavailable")
    import os
    from _fixtures import load_fixture, FIX
    import pyroorb.fit_object as fo
    fx = load_fixture(os.path.join(FIX, "2003_HM.json"))
    fit = fo.fit_orbit(fx["obs"], OBSCODES, desig=fx["meta"]["designation"], meta=fx["meta"])
    assert fit.nongrav_model == "yarkovsky", fit.nongrav_model
    a2_optical = fit.nongrav_params["A2"]
    f2 = radar.refine_radar(fit, HM_RADAR, OBSCODES)
    assert f2.radar["applied"], f2.radar.get("note")
    assert f2.radar["n_used"] == 9
    assert f2.radar["joint_nongrav"] == "yarkovsky"
    a2 = f2.nongrav_params["A2"]
    assert a2 == pytest.approx(-1.1677e-13, rel=0.25), \
        f"joint A2 {a2:.3e} not at JPL's radar-anchored value (optical-only was {a2_optical:.3e})"
    assert f2.radar["post_median_z"] < 2.0
    # both datasets fit: the sky no worse than the optical-degrade gate allows
    assert f2.rms_normalized() < radar.RADAR_MAX_OPTICAL_DEGRADE * fit.rms_normalized()

