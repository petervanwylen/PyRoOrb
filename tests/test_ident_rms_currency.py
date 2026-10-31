"""Offline tests for the ident-check "do the kept obs fit?" gate (ident_check.kept_obs_fit).

Pure -- no fitting -- so these run in the fast `pytest -m "not slow"` loop. The gate used to be a
single ABSOLUTE arcsec bar (MAX_KEPT_RMS), which measures astrometric QUALITY rather than identity:
243P + 2025 MT356 was filed as a flat 'no' at 0.647" even though 243P fitted ALONE lands at 0.649"
(so the identification made the fit marginally BETTER) and 2025 MT356's own 6 obs are the tightest
part of it at 0.036". The sigma-normalized clause is what admits that; RMS_CEILING is what keeps it
from admitting a genuinely wrong orbit whose ancient astrometry carries huge uncertainties.
"""
import numpy as np
from pyroorb import ident_check


class _Fit:
    """Minimal stand-in exposing what kept_obs_fit reads: the two rms accessors."""
    def __init__(self, rms, nrms):
        self._rms, self._nrms = rms, nrms

    def rms_residual(self):  return self._rms
    def rms_normalized(self): return self._nrms


def test_absolute_bar_still_decides_when_met():
    # Unchanged behaviour for the ordinary case: under the arcsec bar passes with no note, and the
    # normalized value is not even consulted (a modern arc's 0.3" fit is an ID however its sigmas
    # are scaled).
    ok, note = ident_check.kept_obs_fit(_Fit(0.30, 5.0))
    assert ok and note is None, f"sub-{ident_check.MAX_KEPT_RMS}\" must pass outright, got {note!r}"


def test_comet_case_admitted_by_normalized_clause():
    # 243P + 2025 MT356's actual numbers: over the arcsec bar, mid-population in sigma.
    ok, note = ident_check.kept_obs_fit(_Fit(0.647, 0.75))
    assert ok, "an arc fitting within its own stated uncertainties must not be a non-match"
    assert "sigma" in note and "0.65" in note, f"note should explain the relaxation, got {note!r}"


def test_over_ceiling_refused_however_good_the_sigma():
    # The load-bearing guard on the relaxation: old astrometry is assigned 1-3" uncertainties, so a
    # wrong orbit through two ancient arcs can fit at ~1 sigma. Past RMS_CEILING the arcsec bar
    # decides alone, exactly as before.
    ok, note = ident_check.kept_obs_fit(_Fit(ident_check.RMS_CEILING + 0.01, 0.4))
    assert not ok and "ceiling" in note, f"over the ceiling must be refused, got {ok} {note!r}"
    ok, _ = ident_check.kept_obs_fit(_Fit(2.5, 0.2))
    assert not ok, "a 2.5\" fit is a wrong orbit even at 0.2 sigma"


def test_over_bar_and_over_sigma_refused():
    # Over the arcsec bar AND worse than the astrometry's own claimed uncertainties -> a real misfit.
    ok, note = ident_check.kept_obs_fit(_Fit(0.9, ident_check.MAX_KEPT_NRMS + 0.1))
    assert not ok and "sigma" in note, f"a genuine misfit must be refused, got {ok} {note!r}"


def test_normalized_bar_clears_the_measured_population():
    # MAX_KEPT_NRMS is calibrated on the 98 PAIR fixtures, whose normalized rms maxes at 1.31 (median
    # 0.52) -- 87 of them clean identifications. Guard the margin so a future retune cannot quietly
    # drop the bar into the population it was chosen to clear.
    assert ident_check.MAX_KEPT_NRMS >= 1.5, \
        "the pair-fixture population reaches 1.31 sigma; a bar below ~1.5 would reject known-good links"
    # ...and the relaxation must stay bounded: it is not a licence for an arbitrarily bad fit.
    assert ident_check.RMS_CEILING > ident_check.MAX_KEPT_RMS, "the ceiling must sit above the bar"
    assert ident_check.RMS_CEILING <= 2.0, "an arcsec ceiling this loose stops guarding anything"


def test_per_designation_null_not_reintroduced():
    # The intuitive per-designation form ("is one side the reason for the excess?") was implemented,
    # measured on the 98 pair fixtures, and found to separate NOTHING (worst-ratio median 1.65x for
    # accepted vs 1.78x for rejected in arcsec; 1.78x vs 1.79x normalized; the population maximum
    # belongs to an accepted identification). Its absence is a deliberate finding, not an oversight,
    # so pin it: nothing here may start keying on per-designation residual ratios.
    assert not hasattr(ident_check, "RMS_RELATIVE_FACTOR"), \
        "the per-designation LOO ratio is a measured null -- see the constants block before re-adding"
