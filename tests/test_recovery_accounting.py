"""The recovery stages must not DELETE observations from the delivered fit.

A fit's observation list is its provenance: every observation the caller handed in has
to be present in the result, either used or explicitly excluded (with an O-C and an 'x'),
because that is what the CLI's rejection block, the web obs table and --ident-check's
`except:` echo all read. A stage that re-fits on a REDUCED obs list therefore cannot
return that reduced fit as-is -- the dropped observations would vanish from
n_total_observations(), from excluded_observations(), and from the record entirely, and
the fit would report "45/45 obs" for an arc that was fetched with 51.

fit_object._recover_drop_night (force-mode recovery stage 2, also reached WITHOUT --force
whenever the fit is `_still_troubled`) used to do exactly that. It now routes its winner
through _represent_over_full_arc / _full_fit_excluding, the same re-presentation the
single-night poison rescue already used.

These tests run OFFLINE on a stored fixture arc -- no network.
"""
import pytest
import os
import numpy as np
import pyroorb.fit_object as fo
from pyroorb import obs_sources
import _fixtures

# full-arc represent/expand refits
pytestmark = pytest.mark.slow

FIXTURE = os.path.join(_fixtures.FIX, "2002_TD301.json")


def _arc():
    """(obs, meta, obscodes) for a small stored arc."""
    fx = _fixtures.load_fixture(FIXTURE)
    return fx["obs"], fx["meta"], obs_sources.load_obscodes()


def _drop_a_night(obs):
    """Indices of the LAST night in `obs` (a plausible 'suspect night' to drop)."""
    jds = [o["jd_utc"] for o in obs]
    last = jds[-1]
    return [i for i, j in enumerate(jds) if last - j < 0.5]


def test_represent_over_full_arc_keeps_every_input_obs():
    """The core invariant: a fit found on a reduced arc, re-presented, carries the FULL
    input count -- the dropped obs come back as EXCLUSIONS, not as absences."""
    obs, meta, obscodes = _arc()
    dropped = set(_drop_a_night(obs))
    assert 0 < len(dropped) < len(obs), f"fixture gave a degenerate night split: {len(dropped)}"
    reduced_full = [i for i in range(len(obs)) if i not in dropped]
    reduced = [obs[i] for i in reduced_full]

    cand = fo.fit_orbit(reduced, obscodes, desig=meta["designation"], meta=meta,
                        verbose=False, auto_nongrav=False)
    assert cand.n_total_observations() == len(reduced)      # the reduced fit knows only its own obs

    full = fo._represent_over_full_arc(cand, obs, reduced_full, obscodes,
                                       meta["designation"], meta)
    assert full is not None, "re-presentation declined on a clean reduced fit"
    # 1. every input observation is present
    assert full.n_total_observations() == len(obs), (full.n_total_observations(), len(obs))
    # 2. the orbit is the one that was accepted -- same obs used, same rms to rounding
    assert full.n_observations() == cand.n_observations()
    assert abs(full.rms_residual() - cand.rms_residual()) <= 0.02 * cand.rms_residual() + 1e-9
    # 3. the dropped night is REPORTED as excluded, with residuals, not silently missing
    excl_jds = {round(r["jd_utc"], 6) for r in full.excluded_observations()}
    for i in dropped:
        assert round(obs[i]["jd_utc"], 6) in excl_jds, f"dropped obs {i} is not reported as excluded"
    for r in full.excluded_observations():
        assert r["included"] is False
        assert np.isfinite(r["d_ra_arcsec"]) and np.isfinite(r["d_dec_arcsec"])


def test_represent_declines_rather_than_degrade_the_orbit():
    """The re-presentation is bookkeeping, never a trade: handed a state that does NOT
    belong to the arc, the full re-fit lands somewhere else and the helper returns None
    so the caller keeps the orbit it already accepted."""
    obs, meta, obscodes = _arc()
    dropped = set(_drop_a_night(obs))
    reduced_full = [i for i in range(len(obs)) if i not in dropped]
    reduced = [obs[i] for i in reduced_full]
    cand = fo.fit_orbit(reduced, obscodes, desig=meta["designation"], meta=meta,
                        verbose=False, auto_nongrav=False)

    class _Wrecked:                       # cand with a state that is nowhere near the arc
        def __init__(self, c):
            self._state = np.asarray(c._state, float) * 1.5
            self.epoch_tdb, self.iod_method = c.epoch_tdb, c.iod_method
            self.history = []
            self._n_obs, self._rms = c.n_observations(), c.rms_residual()
        def is_included(self): return np.ones(len(reduced), dtype=bool)
        def n_observations(self): return self._n_obs
        def rms_residual(self): return self._rms

    out = fo._represent_over_full_arc(_Wrecked(cand), obs, reduced_full, obscodes,
                                      meta["designation"], meta)
    assert out is None, "a wrecked seed must not be accepted as a re-presentation"


def test_expand_history_masks_reindexes_snapshots():
    """Teaching-mode snapshots are indexed by the obs list they were recorded on; the
    re-presented fit must replay against the FULL arc, so each `_mask` is re-indexed and
    the dropped obs read as off."""
    reduced_full = [0, 2, 3]              # obs 1 was the dropped night
    hist = [{"stage": "iod", "_mask": [True, False, True]},
            {"stage": "extend", "_mask": [True, True, True]},
            {"stage": "no-snapshot"}]
    out = fo._expand_history_masks(hist, reduced_full, 4)
    assert out[0]["_mask"] == [True, False, False, True]
    assert out[1]["_mask"] == [True, False, True, True]
    assert "_mask" not in out[2]
    assert hist[0]["_mask"] == [True, False, True], "input history was mutated"
