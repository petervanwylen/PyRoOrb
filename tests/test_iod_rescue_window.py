"""iod.rescue_window: the shrink walk behind the outcome-gated long-window rescue.

The seeds are STUBBED so the walk itself is under test, in milliseconds: select_subarc returns the
window [0, arclen] and a window's seed score depends only on its span. What matters is where the walk
stops and what it refuses -- the two guards measured on C/2013 A1 (a night floor, and a stop at the
first non-catastrophic seed instead of chasing ACCEPTABLE_SCORE down to one night).
"""
import numpy as np
import pytest

from pyroorb import iod


@pytest.fixture
def stubbed(monkeypatch):
    def install(times, score_of_span):
        times = np.asarray(times, float)
        calls = []

        def select_subarc(t, rhohats=None, max_arc_len=iod.ARC_CAP):
            end = int(np.searchsorted(times, times[0] + max_arc_len, side="right")) - 1
            return 0, max(0, end)

        def seed_for_subarc(gs, ge, obs, t, rhohats, Robs, base, t0):
            span = float(times[ge] - times[gs])
            calls.append(span)
            return object(), "stub", score_of_span(span)

        monkeypatch.setattr(iod, "select_subarc", select_subarc)
        monkeypatch.setattr(iod, "_seed_for_subarc", seed_for_subarc)
        monkeypatch.setattr(iod, "build_base_sim", lambda t0: None)
        monkeypatch.setattr(iod, "_utc_to_tdb", lambda jd: jd)
        return times, calls
    return install


def _score(span):
    """Long windows seed catastrophically, mid ones badly, short ones usably."""
    return 5000.0 if span >= 40 else (300.0 if span >= 20 else 50.0)


def test_walk_stops_at_the_first_non_catastrophic_seed(stubbed):
    times, calls = stubbed(np.arange(100.0), _score)          # one obs per night
    window = iod.rescue_window([None] * 100, times, None, None, (0, 99), 1.0e5)
    assert window == (0, 16)                                     # first window seeding <= 100"
    assert calls[-1] < 20 and min(calls) == calls[-1]         # ...and it did not walk further


def test_windows_under_three_nights_are_never_seeded(stubbed):
    # all obs in the first 2.2 days fall on two nights; a 1-night window scores best of all
    times = np.concatenate([[0.0, 0.1, 0.2, 2.0, 2.1, 2.2], np.arange(5.0, 100.0)])
    times, calls = stubbed(times, lambda span: 1.0 if span < 3 else 5000.0)
    window = iod.rescue_window([None] * len(times), times, None, None, (0, len(times) - 1), 1.0e5)
    assert all(span >= 3 for span in calls)                   # the trivial 1-2 night windows skipped
    assert window is not None and times[window[1]] - times[window[0]] >= 3


def test_nothing_returned_when_no_window_beats_the_failed_seed(stubbed):
    times, _ = stubbed(np.arange(100.0), lambda span: 9.0e5)
    assert iod.rescue_window([None] * 100, times, None, None, (0, 99), 1.0e5) is None
