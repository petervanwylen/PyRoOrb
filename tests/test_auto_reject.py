"""Unit tests for the auto_reject evaporation guard's frozen baseline.

find_orb's auto_reject_obs measures every rejection round against the same fixed
first-to-last span of the arc under rejection; re-deriving the baseline from the
currently-included obs each round lets successive rounds each clip another
"legal" 50% of whatever survived the previous round (compounding to 25%, 12.5%,
... of the original arc). Multi-round callers freeze the span once with
_included_span and pass it through orig_span; these tests pin both behaviors.
"""
import pytest
import numpy as np
from pyroorb.arc_extend import auto_reject, _included_span

# full fits with the reject loop
pytestmark = pytest.mark.slow


class _StubEv:
    """The minimum of Evaluator that auto_reject touches when `rs` is supplied."""
    def __init__(self, times):
        self.times_tdb = np.asarray(times, float)
        self.n = len(self.times_tdb)


def _arc():
    # 29 obs over 100 days: lone obs at t=0 and t=49, a dense core at 74..99,
    # and the far endpoint at t=100. Rejecting the t=0 obs cuts the span to 51
    # days (legal, 51%); then rejecting the t=49 obs cuts it to 26 days -- legal
    # against a re-derived 51-day baseline (26 > 25.5) but an evaporation to 26%
    # of the arc that a frozen 100-day baseline must revert.
    times = np.concatenate([[0.0, 49.0], np.linspace(74.0, 99.0, 26), [100.0]])
    return _StubEv(times)


def _rs_rejecting(ev, *idx):
    # already-clipped obs stay in the bad set across rounds (a genuinely discordant
    # obs keeps its large residual, so the redeem side never pulls it back)
    rs = np.full(ev.n, 0.1)
    rs[list(idx)] = 10.0                # far outside any outer limit -> rejected
    return rs


def test_frozen_span_blocks_compound_evaporation():
    ev = _arc()
    mask = np.ones(ev.n, dtype=bool)
    span0 = _included_span(ev, mask)
    assert span0 == 100.0
    # round 1: clipped t=0 -> span 51, over half of the frozen 100 -> allowed
    assert auto_reject(ev, mask, None, 3.0, 3.3,
                       rs=_rs_rejecting(ev, 0), orig_span=span0) == 1
    assert not mask[0] and mask[1:].all()
    # round 2: clipping t=49 would leave 26 of the ORIGINAL 100 days -> reverted
    assert auto_reject(ev, mask, None, 3.0, 3.3,
                       rs=_rs_rejecting(ev, 0, 1), orig_span=span0) == -1
    assert mask[1] and not mask[0]      # round 2 rolled back, round 1 kept


def test_rebaselined_span_would_have_evaporated():
    # the same round 2 WITHOUT the frozen baseline passes (26 > 51*0.5): the
    # compounding this guard exists to stop -- pinned so a regression to
    # per-round re-baselining fails loudly here.
    ev = _arc()
    mask = np.ones(ev.n, dtype=bool)
    mask[0] = False                     # round 1 already applied
    assert auto_reject(ev, mask, None, 3.0, 3.3, rs=_rs_rejecting(ev, 0, 1)) == 1
    assert not mask[1]


def test_single_shot_default_unchanged():
    # a lone call with no orig_span keeps the entry-mask baseline: rejecting the
    # t=0 obs alone (span 100 -> 51) is still allowed.
    ev = _arc()
    mask = np.ones(ev.n, dtype=bool)
    assert auto_reject(ev, mask, None, 3.0, 3.3, rs=_rs_rejecting(ev, 0)) == 1
    assert not mask[0]
