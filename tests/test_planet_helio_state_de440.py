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


def test_planet_helio_state_falls_back_without_de440(monkeypatch):
    import pyroorb.products as products

    def no_file(*a, **k):
        raise FileNotFoundError("no DE440 here")
    monkeypatch.setattr(products, "body_states", no_file)
    state = pc.planet_helio_state("jupiter", 2452694.95)
    assert state.shape == (6,) and np.all(np.isfinite(state))
    assert 4.9 < np.linalg.norm(state[:3]) < 5.5          # Jupiter, from the built-in ephemeris
