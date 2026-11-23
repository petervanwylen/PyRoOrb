"""ident_check.offset_nights_misfit: an apparition whose nights mostly sit coherently OFF the orbit
(a level offset per night, not a trend) reaches the fit banner (residual_warning); a lone off
night, a minority of off nights, small offsets and noisy nights do not."""
import numpy as np

from pyroorb import ident_check


class _Fit:
    """Stand-in carrying what the residual audits read: time-sorted _obs (jd_utc, obscode),
    _mask, and _resid_all (n x 2 arcsec: dRA*cos(dec), dDec)."""
    def __init__(self, nights, code="W84", jd0=2457509.7, step=1.0, n=5, sd=0.07, seed=1):
        rng = np.random.default_rng(seed)
        self._obs, rows = [], []
        for k, (ra, dec) in enumerate(nights):
            for j in range(n):
                self._obs.append({"jd_utc": jd0 + k * step + 0.01 * j, "obscode": code})
                rows.append((ra + rng.normal(0, sd), dec + rng.normal(0, sd)))
        # a second, clean apparition two years on (the other half of a two-apparition fit)
        for k in range(2):
            for j in range(3):
                self._obs.append({"jd_utc": jd0 + 800 + k * 7 + 0.01 * j, "obscode": "F52"})
                rows.append((rng.normal(0, sd), rng.normal(0, sd)))
        self._mask = [True] * len(self._obs)
        self._resid_all = np.array(rows)


def test_jn82_like_offset_nights_flag():
    # 2016 JN82's 2016 W84 nights: Dec +0.47/-0.21/-0.30", RA -0.04/-0.47/+0.28", each ~0.07" tight
    fit = _Fit([(-0.04, 0.47), (-0.47, -0.21), (0.28, -0.30)])
    assert ident_check.systematic_misfit(fit) is None      # below every older arm's floor
    desc = ident_check.offset_nights_misfit(fit)
    assert desc and "of 3 nights sit off the orbit" in desc
    assert ident_check.residual_warning(fit) == desc


def test_single_off_night_never_flags():
    fit = _Fit([(0.0, 0.6), (0.02, -0.03), (-0.01, 0.02)])
    assert ident_check.offset_nights_misfit(fit) is None


def test_minority_of_off_nights_does_not_flag():
    # two displaced nights among six: an older station's ordinary bad nights, not the apparition
    fit = _Fit([(0.5, 0), (0, 0), (0, 0), (-0.5, 0), (0, 0), (0, 0)])
    assert ident_check.offset_nights_misfit(fit) is None


def test_small_offsets_do_not_flag():
    fit = _Fit([(0.15, 0.15), (-0.15, -0.15), (0.15, -0.15)], sd=0.02, n=20)
    assert ident_check.offset_nights_misfit(fit) is None


def test_noisy_nights_do_not_flag():
    # 0.4" means but 1" scatter over 3 obs: the means are noise, not offsets
    fit = _Fit([(0.4, 0), (-0.4, 0), (0.4, 0)], sd=1.0, n=3, seed=3)
    assert ident_check.offset_nights_misfit(fit) is None
