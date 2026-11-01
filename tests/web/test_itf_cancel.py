"""A user can stop a running ITF search from the web UI (POST /api/itf_cancel).

The search is a thread, and Python cannot kill a thread, so stopping it is
cooperative: the job's cancel token is set, the search unwinds at its next
agreement point (``deadline.check_cancelled`` / the propagator), and the job is
answered ``cancelled`` and releases its admission slot -- only once the worker
is actually gone, since until then it still holds the memory the slot counts.
"""
import os
import sys
import threading
import time

import pytest  # noqa: E402

from pyroorb import deadline  # noqa: E402
import pyroorb.web.engine_api as E  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_jobs(monkeypatch):
    saved = dict(E._ITF_JOBS)
    E._ITF_JOBS.clear()
    # admission reads live headroom and a slot count derived from the machine's RAM (one
    # slot on a small CI runner); these tests are about stopping, not capacity
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 4)
    monkeypatch.setattr(E, "_itf_capacity_block", lambda: None)
    # a session with an obs list, so itf_search_web reaches the searcher seam
    monkeypatch.setattr(E, "base_obs_for", lambda **kw: (
        [{"designation": "TEST 1", "jd_utc": 2460000.5}], None))
    monkeypatch.setattr(E, "_fit_for", lambda **kw: None)
    yield
    E._ITF_JOBS.clear()
    E._ITF_JOBS.update(saved)


def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _looping_searcher(started):
    """A search that runs until stopped, passing the agreement point the real
    search passes between every unit of work."""
    def run(seed_obs80, primaries, cfg):
        started.set()
        while True:
            deadline.check_cancelled()
            time.sleep(0.005)
    return run


def test_a_running_search_stops_and_frees_its_slot():
    started = threading.Event()
    r = E.itf_search_start("s1", {"trace_id": "trStop1"},
                           _searcher=_looping_searcher(started))
    assert r == {"ok": True, "job": "trStop1"}
    assert started.wait(5)
    assert E.itf_search_status("trStop1")["done"] is False
    with E._ITF_JOBS_LOCK:
        assert E._itf_running() == 1

    got = E.itf_search_cancel("trStop1")
    assert got["ok"] and got["cancelling"]

    assert _wait(lambda: E.itf_search_status("trStop1")["done"])
    res = E.itf_search_status("trStop1")["result"]
    assert res["ok"] is False and res["cancelled"] is True
    with E._ITF_JOBS_LOCK:
        assert E._itf_running() == 0, "a stopped search must give its slot back"


def test_the_trace_feed_says_why_it_ended():
    started = threading.Event()
    E.itf_search_start("s1", {"trace_id": "trStop2"},
                       _searcher=_looping_searcher(started))
    assert started.wait(5)
    E.itf_search_cancel("trStop2")
    assert _wait(lambda: E.itf_search_status("trStop2")["done"])
    feed = E.itf_trace_feed("trStop2")
    assert feed["done"]
    assert any("stopped at the user's request" in ln for ln in feed["lines"])


def test_a_stopping_search_says_so_while_it_unwinds():
    """Between the click and the unwind the slot is still held, and the poll
    says the search is stopping rather than searching."""
    gate, started = threading.Event(), threading.Event()

    def slow(seed_obs80, primaries, cfg):
        started.set()
        gate.wait(5)                     # not at an agreement point yet
        deadline.check_cancelled()
        return None
    E.itf_search_start("s1", {"trace_id": "trStop3"}, _searcher=slow)
    assert started.wait(5)
    E.itf_search_cancel("trStop3")
    st = E.itf_search_status("trStop3")
    assert st["done"] is False and st["cancelling"] is True
    with E._ITF_JOBS_LOCK:
        assert E._itf_running() == 1
    gate.set()
    assert _wait(lambda: E.itf_search_status("trStop3")["done"])
    assert E.itf_search_status("trStop3")["result"]["cancelled"] is True


def test_a_stop_does_not_reach_another_search():
    """The token is armed on the search's own thread: stopping one search must
    leave a neighbour running."""
    a, b = threading.Event(), threading.Event()
    E.itf_search_start("s1", {"trace_id": "trA"}, _searcher=_looping_searcher(a))
    E.itf_search_start("s2", {"trace_id": "trB"}, _searcher=_looping_searcher(b))
    assert a.wait(5) and b.wait(5)
    E.itf_search_cancel("trA")
    assert _wait(lambda: E.itf_search_status("trA")["done"])
    time.sleep(0.05)
    assert E.itf_search_status("trB")["done"] is False
    E.itf_search_cancel("trB")
    assert _wait(lambda: E.itf_search_status("trB")["done"])


def test_cancel_is_idempotent_and_harmless_on_unknown_or_finished_jobs():
    assert E.itf_search_cancel("nope")["expired"] is True
    assert E.itf_search_cancel(None)["expired"] is True

    def instant(seed_obs80, primaries, cfg):
        raise RuntimeError("finished badly, but finished")
    E.itf_search_start("s1", {"trace_id": "trDone"}, _searcher=instant)
    assert _wait(lambda: E.itf_search_status("trDone")["done"])
    assert E.itf_search_cancel("trDone") == {"ok": True, "done": True,
                                             "cancelling": False}

    started = threading.Event()
    E.itf_search_start("s1", {"trace_id": "trTwice"},
                       _searcher=_looping_searcher(started))
    assert started.wait(5)
    assert E.itf_search_cancel("trTwice")["cancelling"]
    assert E.itf_search_cancel("trTwice")["ok"]          # a double click
    assert _wait(lambda: E.itf_search_status("trTwice")["done"])


def test_a_background_slot_cannot_be_stopped_from_the_web():
    """The NEOCP monitor's follow-up holds a slot with no token: it is not a
    page's search to stop."""
    with E.itf_background_slot("neocp") as ok:
        assert ok
        bg = next(k for k in E._ITF_JOBS if k.startswith("bg:"))
        got = E.itf_search_cancel(bg)
        assert got["ok"] is False and "cannot be stopped" in got["error"]


def test_the_route_is_wired():
    from pyroorb.web import server
    handler = next(v for v in vars(server).values()
                   if isinstance(v, type) and "_JSON_POSTS" in vars(v))
    assert "/api/itf_cancel" in handler._JSON_POSTS
