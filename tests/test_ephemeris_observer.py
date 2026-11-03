"""ephemeris(observer=...): a geocentric position vector in place of the station offset.

A space telescope has no station offset; the fit places it from the vector on each observation
record (`sat_pos_au`), but an ephemeris was given only the code and put C57 (TESS) ~800" off.
With the vector the ephemeris reproduces exactly what the fit computed for that observation,
so observed minus ephemeris equals the fit's own residual. Offline on a frozen fixture.
"""
import numpy as np
import pytest

import pyroorb
from _fixtures import fixture_obs


@pytest.fixture(scope="module")
def wise_fit():
    records, meta = fixture_obs("162385")           # 336 obs, 22 of them WISE (C51) with vectors
    fit = pyroorb.fit_orbit(pyroorb.Observations(records, meta), designation="162385")
    sat = [o for o in records if o.get("sat_pos_au") and o["obscode"] == "C51"]
    assert sat, "fixture lost its satellite observations"
    return fit, sat


def _oc(o, row):
    dra = (o["ra_deg"] - row["ra_deg"]) * 3600 * np.cos(np.radians(o["dec_deg"]))
    return dra, (o["dec_deg"] - row["dec_deg"]) * 3600


@pytest.mark.slow
def test_observer_vector_reproduces_the_fit_residual(wise_fit):
    fit, sat = wise_fit
    resid = {(round(r["jd_utc"], 6), r["obscode"]): (r["d_ra_arcsec"], r["d_dec_arcsec"])
             for r in fit.residuals()}
    for o in sat[:3]:
        row = pyroorb.ephemeris(fit, o["jd_utc"], o["jd_utc"], site="C51", full=False,
                                observer=o["sat_pos_au"])[0]
        want = resid[(round(o["jd_utc"], 6), "C51")]
        got = _oc(o, row)
        assert abs(got[0] - want[0]) < 0.05 and abs(got[1] - want[1]) < 0.05, (got, want)


@pytest.mark.slow
def test_observer_vector_is_a_parallax_of_the_right_size(wise_fit):
    fit, sat = wise_fit
    jd = sat[0]["jd_utc"]
    geo = pyroorb.ephemeris(fit, jd, jd, full=False)[0]
    shifted = pyroorb.ephemeris(fit, jd, jd, full=False, observer=[0.001, 0.0, 0.0])[0]
    # the target direction and the offset's component across it
    ra, dec = np.radians(geo["ra_deg"]), np.radians(geo["dec_deg"])
    u = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    v = np.array([0.001, 0.0, 0.0])
    expected = np.degrees(np.linalg.norm(np.cross(v, u)) / geo["delta_au"]) * 3600
    sep = np.degrees(np.arccos(np.clip(
        u @ np.array([np.cos(np.radians(shifted["dec_deg"])) * np.cos(np.radians(shifted["ra_deg"])),
                      np.cos(np.radians(shifted["dec_deg"])) * np.sin(np.radians(shifted["ra_deg"])),
                      np.sin(np.radians(shifted["dec_deg"]))]), -1, 1))) * 3600
    assert abs(sep - expected) < 0.02 * expected + 0.05, (sep, expected)


@pytest.mark.slow
def test_observer_vector_shapes(wise_fit):
    fit, sat = wise_fit
    jd = sat[0]["jd_utc"]
    one = pyroorb.ephemeris(fit, jd, jd + 2, step=1.0, full=False, observer=[0.001, 0.0, 0.0])
    per = pyroorb.ephemeris(fit, jd, jd + 2, step=1.0, full=False,
                            observer=[[0.001, 0.0, 0.0]] * 3)
    assert [r["ra_deg"] for r in one] == [r["ra_deg"] for r in per]
    with pytest.raises(ValueError, match="observer"):
        pyroorb.ephemeris(fit, jd, jd + 2, step=1.0, full=False, observer=[[0.001, 0.0, 0.0]] * 2)
    # the full table is computed too, with the ground-site columns left empty
    full = pyroorb.ephemeris(fit, jd, jd, site="568", observer=[0.001, 0.0, 0.0])[0]
    assert full["alt_deg"] is None and "constellation" in full
