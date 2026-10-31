"""``/api/memory`` carries the propagator sweep watch: a sweep in flight on another
thread is reported with that thread's (safe) name, its age, and the trial state --
the record the wedged ITF searches of 2026-09-26 left no trace of.  Never imports
sweepwatch to answer: an unloaded module reads as ``None`` like any other table."""
import os
import sys
import threading

import numpy as np

from pyroorb.web import webmem  # noqa: E402


def test_the_report_shows_another_thread_s_sweep_in_flight():
    import pyroorb.sweepwatch as sw
    sw._reset_for_tests()
    started, release = threading.Event(), threading.Event()

    def worker():
        rec = sw.begin("rebound", 2460000.5, np.array([2.5, 0, 0, 0, 0.0109, 0]),
                       np.array([2460001.5, 2460031.5]), "iod")
        rec[sw.AT] = 1
        started.set()
        release.wait(5)
        sw.end(rec)

    t = threading.Thread(target=worker, name="itf-search-abcdef1234567890")
    t.start()
    assert started.wait(5)
    try:
        rep = webmem.report(with_misc=False)
        rows = rep["sweeps"]["inflight"]
        assert len(rows) == 1
        row = rows[0]
        assert row["backend"] == "rebound" and row["label"] == "iod"
        assert row["at"] == 1 and row["n"] == 2 and row["target_days"] == 31.0
        assert row["elements"]["a_au"] > 0
        assert row["thread"] == webmem._safe_thread_name(t.name)   # the page's own rule
        assert rep["sweeps"]["wedged"] == 0
    finally:
        release.set()
        t.join(5)
    assert webmem.report(with_misc=False)["sweeps"]["inflight"] == []


def test_the_report_survives_a_watch_that_is_not_loaded(monkeypatch):
    monkeypatch.setattr(webmem, "_mod", lambda name: None)
    assert webmem._sweeps() is None
