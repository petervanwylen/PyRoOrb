"""The supported Python API (``import pyroorb``): read or fetch observations, fit, check
determinacy, ephemeris, export. Offline: the inputs are tracked files and frozen fixtures."""
import os
import sys
import warnings

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyroorb  # noqa: E402
from _fixtures import fixture_obs  # noqa: E402

D2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digest2", "fixtures")
OBS80 = os.path.join(D2, "sample.obs")
PSV = os.path.join(D2, "sample.psv")


def test_read_observations_from_a_path_or_text_in_either_format():
    a = pyroorb.read_observations(OBS80)
    assert isinstance(a, pyroorb.Observations) and len(a) == 3
    assert a.meta["format"] == "OBS80"
    with open(OBS80) as fh:
        b = pyroorb.read_observations(fh.read())
    assert [o["jd_utc"] for o in b] == [o["jd_utc"] for o in a]
    p = pyroorb.read_observations(PSV)
    assert p.meta["format"] == "ADES" and len(p) == 3
    assert pyroorb.read_observations(OBS80, format="obs80").meta["format"] == "OBS80"


def test_times_accept_iso_jd_and_astropy():
    from astropy.time import Time
    jd = pyroorb._to_jd("2026-01-01T00:00:00")
    assert jd == pytest.approx(2461041.5)
    assert pyroorb._to_jd(jd) == jd
    assert pyroorb._to_jd(Time(jd, format="jd", scale="utc")) == pytest.approx(jd)


def test_an_unknown_export_format_is_refused():
    with pytest.raises(ValueError, match="mpcorb"):
        pyroorb.export(object(), "votable")


@pytest.mark.slow
def test_a_single_tracklet_is_flagged_not_refused():
    obs = pyroorb.read_observations(OBS80)
    with pytest.warns(pyroorb.UndeterminedOrbitWarning):
        fit = pyroorb.fit_orbit(obs)
    assert fit.is_determined is False
    info = fit.determinacy()
    assert info and info["headline"]
    assert {"a", "e", "i"} <= set(fit.elements())      # the nominal solution is returned


@pytest.mark.slow
def test_a_determined_orbit_fits_ephemerides_and_exports():
    records, meta = fixture_obs("2023_QJ3")
    obs = pyroorb.Observations(records, meta)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroorb.UndeterminedOrbitWarning)
        fit = pyroorb.fit_orbit(obs, designation="2023 QJ3")
    assert fit.is_determined and fit.determinacy() is None
    assert fit.n_observations() == 20 and fit.rms_residual() < 0.5
    rows = pyroorb.ephemeris(fit, "2026-11-01", "2026-11-03", step=1.0, site="568",
                             full=False)
    assert len(rows) == 3
    for fmt in ("mpcorb", "sof", "eightline", "covar"):
        text = pyroorb.export(fit, fmt)
        assert isinstance(text, str) and text.strip(), fmt
