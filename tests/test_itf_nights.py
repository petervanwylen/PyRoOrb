"""The ITF search's night boundary was NOON UT (``round(jd - 0.5)``),
the middle of the observing night for the Pacific stations the ITF is fullest
of (F51/F52, T05/T08/T09, 703/G96).  Three sites in ``itf/search.py`` read it:

* ``_seed_degradation_one_night`` grouped the seed's residuals by it, so one
  Hawaiian night split into two half-samples -- the test could blame half a
  night, miss a whole blameable one (each half "degrades" the other), or lift
  a genuine one-night seed over its own ``len(by) < 2`` bar;
* ``_night_label`` printed every observation from 12:00 UT onward under the
  NEXT day's date, so a night printed under two dates, one of which matched
  no OBS80 line -- and the string comparison that decides whether a
  coherently-offset night is the object's OWN silently missed it;
* ``_mag_tracklet_is_own`` re-grouped by that label instead of by the
  photometric detector's own (station, 1.6-day) rule.

Offline and synthetic: a fake fit with hand-set residuals, on real OBS80 lines
of a real noon-straddling F51 night (2021 JK2, 2011-04-24 11:20-12:27 UT).
"""
from datetime import datetime
from types import SimpleNamespace

import pytest

from pyroorb.itf.obs80_fields import obs_unique_id
from pyroorb.itf.search import (_mag_tracklet_is_own, _night_label,
                        _own_nights_as_ident, _seed_degradation_one_night)


def _jd(y, m, d, frac):
    """JD (UTC) of a calendar date plus a day fraction, as an OBS80 line
    carries it."""
    return (datetime(y, m, d) - datetime(1858, 11, 17)).days + 2400000.5 + frac


def _line(y, m, d, frac, stn="F51"):
    """A syntactically valid OBS80 line dated ``y-m-d + frac``; identity (date,
    position, station) is all the search reads off it here."""
    pos = f"{14 + int(frac * 100) % 10:02d} 40 {frac * 40:06.3f}+31 11 {frac * 50:05.2f}"
    return (f"     K21J02K  C{y:4d} {m:02d} {d + frac:09.6f}{pos}"
            f"         21.57G      {stn}")


class _Fit:
    """A FittedOrbit stand-in: ``_obs`` with lines and epochs, hand-set
    residuals, everything included, optional photometry rows."""

    def __init__(self, obs, resid, mags=None):
        self._obs = obs
        self._resid = resid
        self._mags = mags or []

    def residuals(self, included_only=False):
        return [{"d_ra_arcsec": a, "d_dec_arcsec": b} for a, b in self._resid]

    def is_included(self):
        return [True] * len(self._obs)

    def magnitude_residuals(self):
        return self._mags


def _obs(y, m, d, frac, stn="F51"):
    return {"obs80_line": _line(y, m, d, frac, stn), "jd_utc": _jd(y, m, d, frac),
            "obscode": stn}


# the real shape of a Pacific night: F51 at 11:17, 11:43, 12:07, 12:27 UT
STRADDLE = [0.470, 0.488, 0.505, 0.519]


def test_night_label_is_the_obs80_lines_own_date():
    """An observation prints under the date its own OBS80 line carries --
    before AND after 12:00 UT.  Under the noon rule the 12:07 and 12:27 UT
    observations printed as 2011-04-25, a date no line of the night carries."""
    for frac in STRADDLE:
        assert _night_label(_jd(2011, 4, 24, frac)) == "2011-04-24", frac
    # ...and the boundary is midnight UT, exactly as in the line
    assert _night_label(_jd(2011, 4, 24, 0.999)) == "2011-04-24"
    assert _night_label(_jd(2011, 4, 25, 0.001)) == "2011-04-25"


def test_a_straddling_night_is_one_night_to_the_degradation_test():
    """The seed is ONE Hawaiian night straddling noon UT.  Under the noon rule
    it read as two nights and the test went on to blame one half of it; one
    night IS the seed and there is nothing to blame."""
    obs = [_obs(2011, 4, 24, f) for f in STRADDLE]
    fit = _Fit(obs, [(0.0, 0.0)] * 2 + [(1.5, 0.0)] * 2)
    ctx = SimpleNamespace(pub_ids={obs_unique_id(o["obs80_line"]) for o in obs})
    trial = SimpleNamespace(fit=fit)
    assert _seed_degradation_one_night(ctx, trial, warn_lvl=0.5) is None


def test_a_whole_straddling_night_can_be_blamed_and_is_named_by_its_date():
    """Two seed nights: the straddling one sits +1.0" off coherently, the
    other fits.  Under the noon rule the offset night was two half-nights of
    +1.0" each, so whichever half was 'worst', the rest still degraded -- and
    the blameable night went unnamed.  Now it is named, whole, with the date
    its OBS80 lines carry."""
    night_a = [_obs(2011, 4, 24, f) for f in STRADDLE]
    night_b = [_obs(2011, 4, 27, f) for f in (0.40, 0.41, 0.42, 0.43)]
    obs = night_a + night_b
    fit = _Fit(obs, [(1.0, 0.0)] * 4 + [(0.0, 0.0)] * 4)
    ctx = SimpleNamespace(pub_ids={obs_unique_id(o["obs80_line"]) for o in obs})
    out = _seed_degradation_one_night(ctx, SimpleNamespace(fit=fit), warn_lvl=0.5)
    assert out is not None
    assert "2011-04-24 (4 obs)" in out, out
    assert "2011-04-25" not in out


def test_mag_tracklet_ownership_uses_the_detectors_own_grouping():
    """The photometric detector groups by (station, 1.6-day night) and names
    the group by its FIRST epoch.  The ownership check must find THAT group:
    re-grouping by a noon-UT date label put the post-noon (own) observations
    under another date than the group's first epoch and answered 'linked'."""
    linked = [_obs(2011, 4, 24, f) for f in STRADDLE[:2]]      # 11:17, 11:43 UT
    own = [_obs(2011, 4, 24, f) for f in STRADDLE[2:]]         # 12:07, 12:27 UT
    obs = linked + own
    mags = [{"jd_utc": o["jd_utc"], "obscode": "F51", "dmag": 0.9, "used": True}
            for o in obs]
    fit = _Fit(obs, [(0.0, 0.0)] * 4, mags)
    ctx = SimpleNamespace(pub_ids={obs_unique_id(o["obs80_line"]) for o in own})
    from pyroorb.itf.fitter import mag_outlier_tracklet
    magt = mag_outlier_tracklet(fit, floor_mag=0.5, p_max=0.05)
    assert magt is not None and magt["jd"] == linked[0]["jd_utc"]
    assert _mag_tracklet_is_own(ctx, SimpleNamespace(fit=fit), magt) is True
    # and a group with no own observation in it is still 'linked'
    ctx_none = SimpleNamespace(pub_ids=set())
    assert _mag_tracklet_is_own(ctx_none, SimpleNamespace(fit=fit), magt) is False


def test_own_nights_are_labelled_as_the_offset_detector_labels_them():
    """``ident_check.coherent_offset_misfit`` names a night by the UT date of
    its FIRST observation.  A European night straddles MIDNIGHT UT: the own
    observations after midnight carry the next day's date on their own lines,
    but the night the detector names is 2011-04-24 -- so 'whose night is it'
    has to be answered in the detector's grouping, not per observation."""
    from pyroorb import ident_check
    linked = [_obs(2011, 4, 24, f, "J04") for f in (0.90, 0.92)]   # 21:36, 22:05 UT
    own = [_obs(2011, 4, 25, f, "J04") for f in (0.04, 0.06)]      # 00:58, 01:26 UT
    obs = linked + own
    fit = _Fit(obs, [(0.0, 0.30)] * 4)
    fit._mask = [True] * 4
    import numpy as np
    fit._resid_all = np.array([[0.0, 0.30]] * 4)
    ctx = SimpleNamespace(pub_ids={obs_unique_id(o["obs80_line"]) for o in own})
    coh = ident_check.coherent_offset_misfit(fit, noise_floor=0.03)
    assert coh and "2011-04-24" in coh, coh
    nights = _own_nights_as_ident(ctx, fit)
    assert nights == {"2011-04-24"}, nights
    assert any(n in coh for n in nights)
