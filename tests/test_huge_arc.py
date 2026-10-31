"""Offline tests for the huge-arc ceilings (arc_extend, fit_object).

No fitting happens here. The subsampling is pure index arithmetic over an
observation-time array, so it is exercised against a synthetic century arc
rather than a real object -- which is the point: the behaviour has to be
provable for arcs far larger than any fixture the suite can afford to carry.

The property that matters most is the NEGATIVE one: below the threshold these
paths must be byte-for-byte what they always were, because that is the entire
argument that the regression goldens still pin the delivered orbits.
"""
import os
import sys

import numpy as np
import pytest

import pyroorb.arc_extend as ae  # noqa: E402
import pyroorb.fit_object as fo  # noqa: E402


class _Times:
    """The only state the subsamplers touch is ``times_tdb``."""
    _spread_over_oppositions = ae.Evaluator._spread_over_oppositions
    _cap_huge_arc = ae.Evaluator._cap_huge_arc

    def __init__(self, times):
        self.times_tdb = np.asarray(times, float)


def _century_arc(n_app=60, per_app=320, gap=365.25):
    """`n_app` apparitions of `per_app` obs each, one apparition per year."""
    t = []
    for a in range(n_app):
        base = 2450000.0 + a * gap
        t.extend(base + np.linspace(0.0, 30.0, per_app))
    return np.asarray(t, float)


# --- the golden-safety property ---------------------------------------------

@pytest.mark.parametrize("n", [10, 400, 4356, ae.HUGE_ARC_OBS - 1, ae.HUGE_ARC_OBS])
def test_below_the_threshold_nothing_is_dropped(n):
    """AT and BELOW HUGE_ARC_OBS an un-thinned fit still sees every observation.

    4356 is the largest regression fixture (10052). If this ever fails, the cap has
    started reaching arcs the goldens pin and the suite is no longer measuring the
    same path it was baselined on.
    """
    ev = _Times(_century_arc(n_app=max(1, n // 200), per_app=max(1, n // max(1, n // 200)))[:n])
    idx = np.arange(len(ev.times_tdb))
    assert np.array_equal(ev._cap_huge_arc(idx), idx)


def test_above_the_threshold_the_set_is_capped():
    t = _century_arc()                       # 60 apparitions x 320 = 19200 obs
    ev = _Times(t)
    idx = np.arange(len(t))
    out = ev._cap_huge_arc(idx)
    assert len(idx) > ae.HUGE_ARC_OBS
    assert len(out) <= ae.HUGE_ARC_FIT_OBS + len(t) // 320   # +at most one per group
    assert len(out) < len(idx)


# --- what the subsample must PRESERVE ---------------------------------------

def test_every_apparition_survives_the_cap():
    """A flat stride would spend the budget on the densest apparition and drop the
    far-separated ones -- which are precisely the ones constraining the orbit."""
    t = _century_arc(n_app=60, per_app=320)
    ev = _Times(t)
    out = ev._cap_huge_arc(np.arange(len(t)))
    kept = t[out]
    # one apparition per year: every year present in the input is present in the output
    years_in = {int((x - 2450000.0) // 365.25) for x in t}
    years_out = {int((x - 2450000.0) // 365.25) for x in kept}
    assert years_in == years_out


def test_the_endpoints_of_the_arc_survive():
    """The extreme epochs carry the most leverage on the orbit; a subsample that
    loses them shortens the arc, which is a different fit, not a cheaper one."""
    t = _century_arc()
    ev = _Times(t)
    out = ev._cap_huge_arc(np.arange(len(t)))
    assert out[0] == 0
    assert out[-1] == len(t) - 1


def test_a_dense_apparition_cannot_crowd_out_a_sparse_one():
    """One apparition with 10000 obs and one with 50: both must be represented."""
    dense = 2450000.0 + np.linspace(0.0, 30.0, 10000)
    sparse = 2450000.0 + 4000.0 + np.linspace(0.0, 30.0, 50)
    ev = _Times(np.concatenate([dense, sparse]))
    out = ev._cap_huge_arc(np.arange(10050))
    assert (out >= 10000).sum() > 0, "the sparse apparition was dropped entirely"


def test_spread_is_a_no_op_under_its_cap():
    ev = _Times(_century_arc(n_app=2, per_app=10))
    idx = np.arange(20)
    assert np.array_equal(ev._spread_over_oppositions(idx, 100), idx)


# --- the settle loop stays clock-free ----------------------------------------

def test_the_settle_loop_is_bounded_in_rounds_not_in_time():
    """A wall-clock deadline was added here and REVERTED, and this records why.

    `_reject_settle` lives in the fit core, where elapsed time may be measured but
    never compared (test_no_wallclock_branch): a branch on the clock
    lets machine load pick the numerical path, which is what made 490581 fit to
    0.249" on an idle box and 625" under four concurrent workers. The rounds bound
    is the only bound this loop may have; a deadline that aborts and reports itself
    is legitimate, but belongs outside these modules.
    """
    assert not hasattr(fo, "SETTLE_MAX_SECS"), \
        "the settle deadline was reverted -- it branched the fit core on the clock"
    assert fo.MPC_SEED_REJECT_ROUNDS >= 1
