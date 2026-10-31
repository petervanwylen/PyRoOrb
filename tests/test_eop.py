"""The pinned Earth-orientation table (`eop.py`).

IERS-B alone ended 2026-05-02, so every later observation was rotated ITRS->GCRS with
UT1-UTC = 0 and the mean pole, and astropy warned twice per batch in the web log. The
pinned table is now B through its end (unchanged there) continued by the bundled
Bulletin A rapid + predicted values; past THAT the fallback is silent by design."""
import logging
import warnings

import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
from astropy.coordinates import ITRS, GCRS, CartesianRepresentation

from pyroorb import eop


@pytest.fixture(scope="module")
def table():
    t = eop.pin()
    assert t is not None
    return t


def _rotate(mjd):
    T = Time([mjd, mjd + 0.3], format="mjd")
    site = CartesianRepresentation(np.full((3, 2), 4e-5) * u.au)
    return ITRS(site, obstime=T).transform_to(GCRS(obstime=T)).cartesian.xyz.to_value(u.m)


def test_pin_is_the_installed_table_and_idempotent(table):
    assert iers.earth_orientation_table.get() is table
    assert eop.pin() is table


def test_identical_to_iers_b_over_its_span(table):
    """No pre-2026-05 observation moves: the old pin's values, bit for bit."""
    b = iers.IERS_B.open()
    t = Time(np.linspace(b["MJD"][0].value + 1, b["MJD"][-1].value - 1, 5000), format="mjd")
    assert np.array_equal(table.ut1_utc(t).value, b.ut1_utc(t).value)
    for x, y in zip(table.pm_xy(t), b.pm_xy(t)):
        assert np.array_equal(x.value, y.value)


def test_extends_past_iers_b_with_real_eop(table):
    b_end = iers.IERS_B.open()["MJD"][-1].value
    assert eop.table_end_mjd(table) > b_end + 180          # Bulletin A predictions ~1 yr out
    t = Time(b_end + 60, format="mjd")
    dut, status = table.ut1_utc(t, return_status=True)
    assert status in (iers.FROM_IERS_A, iers.FROM_IERS_A_PREDICTION)
    assert abs(dut.to_value(u.s)) > 1e-4                   # a real value, not the 0 fallback


def test_no_warning_inside_the_table_past_iers_b(table):
    b_end = iers.IERS_B.open()["MJD"][-1].value
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _rotate(b_end + 30)
    assert not [x for x in w if "IERS" in str(x.message)]


def test_the_fallback_past_the_table_is_silenced(table):
    far = eop.table_end_mjd(table) + 400
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _rotate(far)                                       # unsilenced: both warnings fire
        assert any("IERS" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        eop.silence_fallback_warnings()
        _rotate(far)
    assert not [x for x in w if "IERS" in str(x.message)]


def test_other_astropy_warnings_are_not_swallowed():
    from astropy.utils.exceptions import AstropyWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        eop.silence_fallback_warnings()
        warnings.warn("something else about IERS entirely", AstropyWarning)
    assert len(w) == 1


def test_stale_table_logs_once(table, caplog, monkeypatch):
    now = Time(eop.table_end_mjd(table) - 5, format="mjd")
    monkeypatch.setattr(Time, "now", classmethod(lambda cls: now))
    with caplog.at_level(logging.WARNING, logger="eop"):
        eop._stale_check(table)
    assert any("astropy-iers-data" in r.getMessage() for r in caplog.records)
    caplog.clear()
    monkeypatch.setattr(Time, "now", classmethod(lambda cls: Time(eop.table_end_mjd(table) - 200, format="mjd")))
    with caplog.at_level(logging.WARNING, logger="eop"):
        eop._stale_check(table)
    assert not caplog.records
