"""The peak-RSS sampler (itf/memory_sampler.py) -- the instrument for a search's memory cost.

Nothing in the engine recorded what a search costs in memory, which is why
ItfConfig.sweep_min_free_mb is one object's anecdote.  These pin the sampler's
contract, not any particular number: it must catch a peak that a start/end pair
would miss, must report growth over the level the process was ALREADY holding
(the web server runs every search in one process), and must never raise.
"""
import time

from pyroorb.itf.memory_sampler import RssSampler, rss_mb


def test_rss_is_readable_or_cleanly_absent():
    got = rss_mb()
    assert got is None or 0 < got < 1024 * 1024


def test_sampler_catches_a_transient_peak():
    """The whole point of sampling: a peak that has been FREED by the time the
    search returns is still the peak the OOM killer would have seen."""
    if rss_mb() is None:
        return                                  # not Linux: the sampler is inert
    s = RssSampler(interval=0.02).start()
    blob = [bytearray(16 * 1024 * 1024) for _ in range(4)]
    for b in blob:                              # touch it: RSS, not address space
        b[::4096] = b"x" * len(b[::4096])
    time.sleep(0.2)
    del blob
    time.sleep(0.1)
    rep = s.stop()
    assert rep["peak_rss_mb"] >= rep["start_rss_mb"] + 32, rep
    assert rep["rss_growth_mb"] >= 32, rep
    assert rep["samples"] >= 2


def test_report_is_empty_when_nothing_can_be_measured(monkeypatch):
    """A kernel without /proc/self/status must leave the search untouched."""
    import pyroorb.itf.memory_sampler as R
    monkeypatch.setattr(R, "rss_mb", lambda: None)
    s = R.RssSampler(interval=0.01)
    s.start()
    assert s.stop() == {}


def test_stop_is_idempotent():
    s = RssSampler(interval=0.01).start()
    first = s.stop()
    assert s.stop() == first or not first


def test_growth_is_measured_over_the_process_baseline():
    """`peak_rss_mb` is a whole-process figure and in the web server that
    includes twelve resident sessions; `rss_growth_mb` is the part THIS search
    is answerable for, which is what an admission decision would need."""
    if rss_mb() is None:
        return
    s = RssSampler(interval=0.02).start()
    rep = s.stop()
    assert rep["rss_growth_mb"] == max(0, rep["peak_rss_mb"] - rep["start_rss_mb"])
