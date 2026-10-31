"""Unit tests for the delivery-LM status-0 retry contract (arc_extend._final_lm).

The final clean fit is the covariance source, so a MINPACK status-0 result (stopped
at the max_nfev cap, not at a minimum) gets ONE warm restart from the reached state.
The contract these tests pin, with a scripted evaluator so every branch is
deterministic:

  * status > 0 first try            -> no second fit at all
  * restart better OR EQUAL in SSR  -> the restart is delivered (ties to the restart)
  * restart WORSE                   -> the status-0 original is delivered, so the fit
                                       ships FLAGGED non-converged even though a
                                       converged-but-worse restart existed
  * still status 0 after the retry  -> the better of the two, still flagged

"Flagged" is FittedOrbit's `converged = status > 0` reading of the delivered result.
"""
import pytest
from types import SimpleNamespace
import numpy as np
from pyroorb.arc_extend import _final_lm

# full LM delivery fit
pytestmark = pytest.mark.slow


class _ScriptedEv:
    """An Evaluator stub whose fit() replays a scripted list of (state, status, ssr)
    outcomes, recording each call's seed so the warm-restart seeding is checkable."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []                       # (seed, thin) per fit() call

    def fit(self, seed, mask, max_nfev=None, thin=True, polish=False):
        self.calls.append((np.asarray(seed, float).copy(), thin))
        state, status, ssr = self.script.pop(0)
        fun = np.zeros(4); fun[0] = np.sqrt(ssr)          # np.dot(fun, fun) == ssr
        state = np.asarray(state, float)
        return state, SimpleNamespace(x=state, status=status, fun=fun)


S0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # the seed handed to _final_lm
S1 = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # first fit's reached state
S2 = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # restart's reached state


def _delivered_converged(result):
    """FittedOrbit's reading of the delivered result (fit_object.FittedOrbit)."""
    return bool(getattr(result, "status", 1) > 0)


def test_converged_first_try_runs_no_retry():
    ev = _ScriptedEv([(S1, 1, 4.0)])
    state, result = _final_lm(ev, np.array(S0), None)
    assert len(ev.calls) == 1, "a converged delivery fit must not be re-run"
    assert list(state) == S1 and _delivered_converged(result)
    assert ev.calls[0][1] is False, "the delivery fit must be full-arc (thin=False)"


def test_status0_retry_is_warm_started_and_a_tie_goes_to_it():
    # equal SSR: the restart is delivered (<=), so its converged status is what ships
    ev = _ScriptedEv([(S1, 0, 4.0), (S2, 1, 4.0)])
    state, result = _final_lm(ev, np.array(S0), None)
    assert len(ev.calls) == 2
    assert list(ev.calls[1][0]) == S1, "the retry must warm-start from the REACHED state"
    assert ev.calls[1][1] is False, "the retry must also be full-arc (thin=False)"
    assert list(state) == S2 and _delivered_converged(result)


def test_status0_worse_restart_is_discarded_and_the_fit_ships_flagged():
    # the restart converges but lands on a HIGHER residual sum: keep the status-0
    # original -- the delivered fit is flagged non-converged even though a
    # converged alternative existed. This is the contract's one subtle branch.
    ev = _ScriptedEv([(S1, 0, 4.0), (S2, 1, 5.0)])
    state, result = _final_lm(ev, np.array(S0), None)
    assert list(state) == S1
    assert not _delivered_converged(result), "a worse restart must not clear the flag"


def test_status0_twice_delivers_the_better_state_still_flagged():
    ev = _ScriptedEv([(S1, 0, 4.0), (S2, 0, 3.0)])
    state, result = _final_lm(ev, np.array(S0), None)
    assert list(state) == S2, "of two capped results, the smaller SSR is delivered"
    assert not _delivered_converged(result)
