"""Positional reads across two different observation lists -- the codebase's one
recurring silent-corruption bug class, gathered in one file.

A mask, a residual row and an obs record are all POSITIONAL claims about one specific
observation list. Where two lists meet (a rescue's reduced arc, a client's mask, a
caller-supplied obs list) the shorter one wins silently: zip() stops, np.flatnonzero
yields only what it covers, numpy slices truncate. Nothing raises until much later, if
at all. Each test below pins one such seam.

Part 1 -- the single-night poison rescue must index cand's mask by IDENTITY, not position.

`cand = fit_orbit(reduced, ...)` is not guaranteed to come back fit on exactly `reduced`:
the drop-suspect-night recovery stage fits a shorter list and, when _represent_over_full_arc
cannot re-present that orbit over the full arc, hands the caller the REDUCED fit
(fit_object.py:381). cand.is_included() is then SHORTER than `reduced`, and the rescue's three
reduced->cand lookups (the dominance test, the bridge anchor mask, the excluded_full set) used
to index it by position in `reduced` -- an IndexError out of the rescue when the mask overruns,
or a silent read of the WRONG observation when the drop was interior. Offline and fit-free:
the arithmetic under test is pure bookkeeping.
"""
import math

import numpy as np
import pytest

import pyroorb.fit_object as fo


class _StubFit:
    def __init__(self, obs, mask, rms, opps=2):
        self._obs, self._mask, self._rms, self._opps = obs, np.array(mask, bool), rms, opps
        self._state, self.epoch_tdb = np.zeros(6), 2450000.0

    def is_included(self):        return self._mask.copy()
    def n_excluded(self):         return int((~self._mask).sum())
    def n_observations(self):     return int(self._mask.sum())
    def n_total_observations(self): return len(self._obs)
    def n_oppositions(self):      return self._opps
    def rms_residual(self):       return self._rms


def _arc():
    """4-obs single-night opposition, then a 16-obs multi-night one four hundred days later."""
    obs = [{"jd_utc": 2450000.0 + i * 0.01, "ra_deg": 0.0, "dec_deg": 0.0,
            "obscode": "500"} for i in range(4)]
    obs += [{"jd_utc": 2450400.0 + i * 0.2, "ra_deg": 0.0, "dec_deg": 0.0,
             "obscode": "500"} for i in range(16)]
    return obs


def _poisoned(obs):
    """A `best` that trips the rescue: elevated rms and clipping GOOD (multi-night) obs."""
    mask = [True] * len(obs)
    mask[10] = mask[11] = mask[12] = False
    return _StubFit(obs, mask, 2.4)


@pytest.fixture(autouse=True)
def _no_degenerate_gate(monkeypatch):
    monkeypatch.setattr(fo, "_ident_doomed_degenerate", lambda *a, **k: False)


def test_short_cand_does_not_overrun_the_mask(monkeypatch):
    obs = _arc()
    best = _poisoned(obs)
    # cand was itself reduced inside fit_orbit: it holds obs[4:18], two short of `reduced`.
    cand = _StubFit(obs[4:18], [True] * 14, 0.05)
    monkeypatch.setattr(fo, "fit_orbit", lambda *a, **k: cand)
    out = fo._rescue_single_night_poison(best, obs, {}, "x", None, False, False)
    # obs 18/19 are included in `best` and absent from cand -> the drop DID strand them,
    # so the rescue must decline (and above all must not raise).
    assert out is best


def test_interior_drop_is_pinned_off_not_read_off_the_wrong_obs(monkeypatch):
    obs = _arc()
    best = _poisoned(obs)
    # cand dropped an INTERIOR obs (obs[8]); every remaining position past it shifts by one, so a
    # positional read stays in range for most of the walk and answers off the wrong observation.
    # Here cand still links 15 of the 16 `reduced` obs, so the >=90% arm admits the rescue and the
    # bookkeeping below has to place obs[8] correctly: it is in NO fit, so it must be pinned off.
    kept = [o for i, o in enumerate(obs[4:]) if i != 4]
    cand = _StubFit(kept, [True] * len(kept), 0.05)
    monkeypatch.setattr(fo, "fit_orbit", lambda *a, **k: cand)
    monkeypatch.setattr(fo, "_bridge_rescue_tracklets", lambda *a, **k: None)
    seen = {}
    def _capture(state, epoch, obs_, obscodes, desig, meta, excluded_full, **kw):
        seen["excluded"] = set(excluded_full)
        return None
    monkeypatch.setattr(fo, "_full_fit_excluding", _capture)
    out = fo._rescue_single_night_poison(best, obs, {}, "x", None, False, False)
    assert out is cand
    assert seen["excluded"] == {0, 1, 2, 3, 8}      # the single-night opp + the obs cand dropped


def test_aligned_cand_still_dominates(monkeypatch):
    obs = _arc()
    best = _poisoned(obs)
    cand = _StubFit(obs[4:], [True] * 16, 0.05)      # same length as `reduced`, links everything
    monkeypatch.setattr(fo, "fit_orbit", lambda *a, **k: cand)
    monkeypatch.setattr(fo, "_bridge_rescue_tracklets", lambda *a, **k: None)
    monkeypatch.setattr(fo, "_full_fit_excluding", lambda *a, **k: None)
    out = fo._rescue_single_night_poison(best, obs, {}, "x", None, False, False)
    assert out is cand              # nothing stranded + rms cut >= 20% -> rescue accepted


def test_cand_positions_maps_by_identity():
    obs = _arc()
    cand = _StubFit([obs[0], obs[3]], [True, True], 0.1)
    assert fo._cand_positions(cand, [obs[0], obs[1], obs[3]]) == [0, None, 1]
    aligned = _StubFit(obs, [True] * len(obs), 0.1)
    assert fo._cand_positions(aligned, obs) == list(range(len(obs)))


# --- part 2: a caller-supplied mask must match the fit it is handed to -------------

def test_caller_mask_length_is_checked():
    """The interactive re-fit entry points (nongrav, constraints, epoch, asteroids,
    force-model swap) take an external mask and hand it to the evaluator, where
    np.flatnonzero accepts ANY length -- so a wrong-length mask used to FIT a truncated
    arc and die later in the products. _caller_mask refuses it up front."""
    obs = _arc()
    fit = _StubFit(obs, [True] * len(obs), 0.3)
    assert list(fo._caller_mask(fit, None)) == [True] * len(obs)
    assert list(fo._caller_mask(fit, [True] * len(obs))) == [True] * len(obs)
    for bad in ([True] * (len(obs) - 3), [True] * (len(obs) + 1)):
        with pytest.raises(ValueError, match="observations"):
            fo._caller_mask(fit, bad)


# --- the same bug class elsewhere: a positional read across two arcs ----------------

def test_residuals_fallback_stays_aligned_with_obs():
    """FittedOrbit.residuals() must emit one row PER OBS even without _resid_all.

    Every caller zips its rows against fit._obs (to attach the sky position, the obs80
    line, the photometry row). The no-_resid_all fallback used to return only the LM's
    INCLUDED rows, so on any fit with a rejection each row was read against the identity
    of a different observation. Excluded obs now carry NaN residuals and keep their slot.
    """
    obs = _arc()
    mask = np.array([True] * len(obs)); mask[2] = mask[7] = False
    f = np.zeros((int(mask.sum()), 2))                 # the LM residual vector: included only
    fit = fo.FittedOrbit.__new__(fo.FittedOrbit)
    fit._resid_all, fit._obs, fit._mask = None, obs, mask
    fit._res = type("R", (), {"fun": f.ravel()})()
    fit.designation = "x"
    rows = fit.residuals()
    assert len(rows) == len(obs)
    assert [r["included"] for r in rows] == list(mask)
    assert [r["jd_utc"] for r in rows] == [o["jd_utc"] for o in obs]
    assert math.isnan(rows[2]["d_ra_arcsec"]) and not math.isnan(rows[3]["d_ra_arcsec"])
    assert len(fit.residuals(included_only=True)) == int(mask.sum())


def test_night_structure_refuses_a_foreign_obs_list():
    """night_structure zips the caller's obs/residuals against the FIT's mask; a list
    that is not the fit's own arc would mix two arcs, silently, via zip()."""
    from pyroorb.itf import night_coherence
    obs = _arc()
    fit = _StubFit(obs, [True] * len(obs), 0.3)
    fit.residuals = lambda included_only=False: [
        {"jd_utc": o["jd_utc"], "obscode": "500", "d_ra_arcsec": 0.1, "d_dec_arcsec": 0.1,
         "sigma_ra": 0.5, "sigma_dec": 0.5, "included": True} for o in obs]
    assert night_coherence.night_structure(fit, obs=obs[:-4])["n_nights"] == 0
