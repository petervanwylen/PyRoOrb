"""Unit tests for the two halves of the stale-exclusion defect (2002 QG17).

An observation was reported as REJECTED while its O-C against the delivered orbit was
0.4 sigma. Two independent causes, one test module each half:

  * `Evaluator._unstall` -- an LM solve that stops on the TOLERANCE rather than at a
    minimum hands back the seed verbatim (nfev=2, status=3, dx=0). The caller records a
    new stage, but the orbit never saw the observations that were just added, and the
    reject loop then scores every obs against an orbit blind to them. The guard re-solves
    once, tighter, and keeps the better residual sum.

  * `arc_extend._settle_final_mask` -- `_final_lm` is the last thing that MOVES the orbit,
    and every reject/redeem decision was taken before it. The settle pass re-runs the
    redemption amnesty against the DELIVERED orbit so "excluded" describes the orbit that
    actually ships.
"""
import pytest
from types import SimpleNamespace
import numpy as np

from pyroorb.arc_extend import (Evaluator, _settle_final_mask, FIT_STALL_TOL_FACTOR,
                        FINAL_SETTLE_ROUNDS, REDEEM_SIGMA)


# --- Evaluator._unstall: the tolerance-stall retry ---------------------------------
def _res(x, nfev, status, ssr):
    """A least_squares-shaped result whose np.dot(fun, fun) is exactly `ssr`."""
    fun = np.zeros(4); fun[0] = np.sqrt(ssr)
    return SimpleNamespace(x=np.asarray(x, float), nfev=nfev, status=status, fun=fun)


SEED = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
MOVED = np.array([1.00002, 0.0, 0.0, 0.0, 0.0, 0.0])


def _patched_least_squares(monkeypatch, outcome):
    """Capture the retry's kwargs and return `outcome` from it."""
    seen = {}
    def fake(resid, x0, **kw):
        seen.update(kw); seen["x0"] = np.asarray(x0, float).copy()
        return outcome
    monkeypatch.setattr("pyroorb.arc_extend.least_squares", fake)
    return seen


def test_stalled_solve_is_retried_tighter_and_the_better_result_wins(monkeypatch):
    stalled = _res(SEED, nfev=2, status=3, ssr=100.0)       # seed verbatim = never moved
    seen = _patched_least_squares(monkeypatch, _res(MOVED, nfev=46, status=3, ssr=9.0))
    out = Evaluator._unstall(lambda s: None, SEED, stalled, 1e-5, 400, jac=None)
    assert out.nfev == 46 and np.allclose(out.x, MOVED), "the retry's result must be delivered"
    assert seen["xtol"] == 1e-5 * FIT_STALL_TOL_FACTOR, "the retry must TIGHTEN the tolerance"
    assert seen["ftol"] == 1e-5 * FIT_STALL_TOL_FACTOR
    assert np.allclose(seen["x0"], SEED), "the retry re-solves from the ORIGINAL seed"


def test_retry_that_comes_back_worse_is_discarded(monkeypatch):
    """The guard can never make a fit worse than it already was."""
    stalled = _res(SEED, nfev=2, status=3, ssr=100.0)
    _patched_least_squares(monkeypatch, _res(MOVED, nfev=9, status=3, ssr=400.0))
    out = Evaluator._unstall(lambda s: None, SEED, stalled, 1e-5, 400, jac=None)
    assert out is stalled


@pytest.mark.parametrize("result,why", [
    (_res(MOVED, nfev=2, status=3, ssr=1.0), "it MOVED, so it did not stall"),
    (_res(SEED, nfev=46, status=3, ssr=1.0), "it iterated, so it did not stall"),
    (_res(SEED, nfev=2, status=0, ssr=1.0), "status 0 has its own retry in _final_lm"),
])
def test_non_stalled_solves_pay_nothing(monkeypatch, result, why):
    def fake(*a, **k):
        raise AssertionError(f"no retry expected: {why}")
    monkeypatch.setattr("pyroorb.arc_extend.least_squares", fake)
    assert Evaluator._unstall(lambda s: None, SEED, result, 1e-5, 400, jac=None) is result


def test_jac_is_forwarded_to_the_retry_when_the_caller_had_one(monkeypatch):
    """The analytic Jacobian is what makes an LM iteration cost one sweep instead of
    seven -- a retry that dropped it would finite-difference the whole solve."""
    stalled = _res(SEED, nfev=2, status=3, ssr=100.0)
    seen = _patched_least_squares(monkeypatch, _res(MOVED, nfev=46, status=3, ssr=9.0))
    jac = lambda s: None
    Evaluator._unstall(lambda s: None, SEED, stalled, 1e-5, 400, jac=jac)
    assert seen.get("jac") is jac


# --- _settle_final_mask: exclusions must describe the DELIVERED orbit --------------
class _SettleEv:
    """Evaluator stub in which the orbit IMPROVES on each delivery fit, so obs excluded
    against an early orbit become clean against the delivered one -- the 2002 QG17 shape.
    `state` is the residual (in sigmas) that every excluded obs currently shows.
    """
    def __init__(self, n, sigmas):
        self.n = n
        self.sigmas = list(sigmas)          # per-fit: the sigma-residual of excluded obs
        self.fits = 0

    def _cur(self, state):
        return self.sigmas[min(int(state[0]), len(self.sigmas) - 1)]

    def resid_sigmas(self, state):
        return np.full(self.n, self._cur(state))

    def resid_arcsec(self, state):
        # raw arcsec large enough that only the SIGMA gate can ever redeem
        return np.full(self.n, 5.0), np.zeros(self.n)

    def fit(self, seed, mask, max_nfev=None, thin=True, polish=False, stall_guard=False):
        self.fits += 1
        return np.array([float(seed[0]) + 1.0]), SimpleNamespace(status=1, fun=np.zeros(4))

    def rms_arcsec(self, state, mask):  return 0.4
    def rms_norm(self, state, mask):    return 0.5
    def resid_subset(self, state, mask):
        k = int(np.sum(mask))
        return np.zeros(k), np.zeros(k)
    sig_ra = sig_dec = property(lambda self: np.ones(self.n))


def _mask(n, excluded):
    m = np.ones(n, dtype=bool); m[list(excluded)] = False
    return m


def test_obs_clean_against_the_delivered_orbit_are_redeemed(monkeypatch):
    """The regression: excluded at 7 sigma against the pre-delivery orbit, 0.4 sigma
    against the one that ships -> it must come back, not be reported as rejected.

    The entry state IS the delivered orbit (`_settle_final_mask` runs after `_final_lm`),
    so the stub starts at the clean 0.4-sigma reading -- the 7 sigma that clipped them
    belonged to the pre-delivery orbit, upstream of this function."""
    monkeypatch.setattr("pyroorb.arc_extend._score", lambda ev, s, m: 0.5)
    ev = _SettleEv(10, sigmas=[0.4, 0.4, 0.4])
    mask = _mask(10, [3, 7])
    hist = []
    state, mask2, result = _settle_final_mask(ev, np.array([0.0]), mask, None, hist, False)
    assert mask2.all(), "both obs sit inside REDEEM_SIGMA of the delivered orbit"
    assert any(h["stage"] == "settle" for h in hist), "the re-delivery must be traced"
    assert result is not None, "a re-delivered orbit must carry its own covariance source"


def test_genuinely_discordant_obs_stay_excluded(monkeypatch):
    monkeypatch.setattr("pyroorb.arc_extend._score", lambda ev, s, m: 0.5)
    ev = _SettleEv(10, sigmas=[7.0, 7.0, 7.0])       # bad against every orbit
    mask = _mask(10, [3, 7])
    hist = []
    _, mask2, result = _settle_final_mask(ev, np.array([0.0]), mask, "orig", hist, False)
    assert list(np.flatnonzero(~mask2)) == [3, 7], "a real outlier is not amnestied"
    assert hist == [] and result == "orig", "no change -> no extra fit, no history noise"
    assert ev.fits == 0


def test_a_full_mask_short_circuits(monkeypatch):
    """The common case: nothing was ever excluded, so the settle must cost nothing."""
    monkeypatch.setattr("pyroorb.arc_extend._score", lambda ev, s, m: 0.5)
    ev = _SettleEv(10, sigmas=[0.4])
    _, mask2, result = _settle_final_mask(ev, np.array([0.0]), np.ones(10, bool), "orig", [], False)
    assert mask2.all() and result == "orig" and ev.fits == 0


def test_settle_is_monotone_and_bounded(monkeypatch):
    """Redemption-only: the mask can only grow, so the loop cannot oscillate, and it is
    capped besides."""
    monkeypatch.setattr("pyroorb.arc_extend._score", lambda ev, s, m: 0.5)
    # one obs redeemed per round, forever -- the cap is what stops it
    ev = _SettleEv(10, sigmas=[REDEEM_SIGMA - 0.1] * 20)
    mask = _mask(10, [1, 2, 3, 4, 5, 6, 7, 8])
    hist = []
    _, mask2, _ = _settle_final_mask(ev, np.array([0.0]), mask, None, hist, False)
    assert mask2.sum() >= mask.sum(), "the settle may never CLIP an observation"
    assert len([h for h in hist if h["stage"] == "settle"]) <= FINAL_SETTLE_ROUNDS


# --- converge before the reject stage: the solve that MOVED but stopped short ---------
@pytest.mark.slow
def test_linked_close_approach_night_is_judged_against_a_converged_orbit():
    """2015 VZ1: the multirev bridge glues the 2026 recovery at new=1.9" and its solve stops
    short WITHOUT stalling (it moved, so `_unstall` never arms). Scored against that orbit,
    the sigma-0.1" F51 night of 2026-10-06 sits at 11-25 sigma and round 0 clips all of it;
    the refit without it leaves F51 7" off. The same 35 obs fit at 0.49" once converged, and
    the MPC's orbfit uses all 35. The generic regression guard cannot see this one -- 31/35
    at 0.463" reads to it as a cleaner fit -- so the exclusion is asserted directly."""
    import os
    from _fixtures import FIX, load_fixture
    from test_regression import refit
    fx = load_fixture(os.path.join(FIX, "2015_VZ1.json"))
    fit, _ = refit(fx)
    lost = [r for r in fit.excluded_observations()
            if r["obscode"] == "F51" and r["jd_utc"] > 2461319.5]
    assert not lost, f"F51 2026-10-06 obs excluded: {lost}"
    assert fit.n_observations() == fit.n_total_observations() == 35
