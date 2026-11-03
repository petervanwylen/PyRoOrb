"""An orbit is predicted in the force model it was determined in: the ephemeris of an orbit
polished with the 16 asteroid perturbers uses them, a planets-only orbit does not, and the
process-wide switch at call time changes neither. (Read from the switch, a polished (197) Arete's
ephemeris sat 0.08-0.38" -- 7-25 sigma of its own ellipse -- off its own orbit.)
Offline: frozen fixtures."""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_the_ephemeris_follows_the_fit_not_the_switch():
    import pyroorb
    import pyroorb.force_assist as fa
    from pyroorb import fit_object as fo
    from _fixtures import fixture_obs
    if fa.asteroid_kernel_path() is None:
        pytest.skip("the asteroid perturber kernel is not installed")
    records, meta = fixture_obs("107474")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grav = pyroorb.fit_orbit(pyroorb.Observations(records, meta), designation="107474")
    polished = fo.refine_asteroids(grav)
    assert polished.asteroids and "+16ast" in polished.force_model
    for f in (grav, polished):
        off = pyroorb.ephemeris(f, "2031-12-01", "2031-12-01", site="500", full=False)[0]
        with pyroorb.asteroid_perturbers():
            on = pyroorb.ephemeris(f, "2031-12-01", "2031-12-01", site="500", full=False)[0]
        assert on["ra_deg"] == off["ra_deg"] and on["dec_deg"] == off["dec_deg"]
    a = pyroorb.ephemeris(grav, "2031-12-01", "2031-12-01", site="500", full=False)[0]
    b = pyroorb.ephemeris(polished, "2031-12-01", "2031-12-01", site="500", full=False)[0]
    assert (a["ra_deg"], a["dec_deg"]) != (b["ra_deg"], b["dec_deg"])   # the models differ
