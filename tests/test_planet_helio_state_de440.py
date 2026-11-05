"""planet_helio_state reads the planet from DE440, the ephemeris the fit integrates in.

It used to take astropy's built-in ephemeris, which puts Jupiter ~80,000 km from DE440, so the
public osculating_elements(center="jupiter") and the planetocentric sigmas measured a satellite
against a planet the fit had not used (notebook 09 fetched its own DE440 Jupiter to work round
it). The built-in ephemeris stays as the fallback when the DE440 file is absent.
"""
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

import pyroorb
from pyroorb import planetocentric as pc

AU_KM = 1.495978707e8


def test_planet_helio_state_is_the_de440_planet():
    import pyroorb.force_assist as fa
    if not Path(fa.PLANETS_PATH).exists():
        pytest.skip("DE440 planet ephemeris not installed")
    jd = 2452694.95                                   # Ersa's epoch in notebook 09
    ours = pc.planet_helio_state("jupiter", jd)
    public = pyroorb.body_states("jupiter", Time(jd, format="jd", scale="tdb"))[0]
    assert ours.shape == (6,)
    assert np.allclose(ours, public, rtol=0, atol=1e-12)
    # and it is no longer astropy's built-in ephemeris
    from astropy.coordinates import get_body_barycentric_posvel
    import astropy.units as u
    t = Time(jd, format="jd", scale="tdb")
    p, _ = get_body_barycentric_posvel("jupiter", t)
    s, _ = get_body_barycentric_posvel("sun", t)
    builtin = (p - s).xyz.to(u.au).value
    assert np.linalg.norm(ours[:3] - builtin) * AU_KM > 1e4


def test_planet_helio_state_falls_back_without_de440_and_downloads_nothing(tmp_path, monkeypatch):
    """The real path, not a mocked body_states: with no DE440 file the built-in ephemeris
    answers at once. Building the ASSIST ephemeris would start the ~102 MB download."""
    import urllib.request
    import pyroorb.force_assist as fa

    attempts = []

    def no_download(*a, **k):        # recorded, not raised: the fallback's except would eat it
        attempts.append(a)
        raise OSError("offline")
    monkeypatch.setattr(fa, "PLANETS_PATH", str(tmp_path / "absent" / "de440.bsp"))
    monkeypatch.setattr(fa, "_EPHEM", None)
    monkeypatch.setattr(fa, "ensure_planets_kernel", no_download)
    monkeypatch.setattr(urllib.request, "urlretrieve", no_download)
    state = pc.planet_helio_state("jupiter", 2452694.95)
    assert state.shape == (6,) and np.all(np.isfinite(state))
    assert 4.9 < np.linalg.norm(state[:3]) < 5.5          # Jupiter, from the built-in ephemeris
    assert attempts == [], "planet_helio_state tried to provision DE440"


def test_planet_helio_state_falls_back_on_an_unreadable_de440(tmp_path, monkeypatch):
    import pyroorb.force_assist as fa
    import pyroorb.products as products

    bad = tmp_path / "de440.bsp"
    bad.write_bytes(b"not a kernel")
    monkeypatch.setattr(fa, "PLANETS_PATH", str(bad))

    def unreadable(*a, **k):
        raise OSError("cannot read the kernel")
    monkeypatch.setattr(products, "body_states", unreadable)
    state = pc.planet_helio_state("jupiter", 2452694.95)
    assert 4.9 < np.linalg.norm(state[:3]) < 5.5
