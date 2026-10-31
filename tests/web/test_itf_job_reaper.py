"""A search slot is held by a RUNNING search, not by a dead entry.

``_ITF_MAX_RUNNING`` bounds concurrent ITF searches because six of them is
already ~1.1 GB in the median (measured).  The table it counts,
``_ITF_JOBS``, is deliberately never evicted while a job is unfinished -- losing
a live search's result is the failure that mechanism exists to prevent -- and
nothing else cleared one either.  So an entry stuck at ``done: False`` held its
slot for the life of the process: six of those and every later search is refused
"server busy" until someone restarts the server, which is a worse outage than
the concurrency the cap was added to prevent.

``_itf_reap`` closes it by asking the exact question -- is the worker thread
alive? -- rather than guessing with a timeout, because a legitimate variant-scan
search runs for many minutes and any timeout safe for it is too long to help.
"""
import os
import sys
import threading
import time

import pytest  # noqa: E402

import pyroorb.web.engine_api as E  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_jobs():
    saved = dict(E._ITF_JOBS)
    E._ITF_JOBS.clear()
    yield
    E._ITF_JOBS.clear()
    E._ITF_JOBS.update(saved)


def _dead_thread():
    """A Thread object that has run and finished -- what a stranded job's
    worker looks like from the table's side."""
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


def _entry(**kw):
    e = {"done": False, "started": time.time(), "ended": 0.0, "collected": 0.0,
         "payload": None, "thread": None, "rss_start_mb": None}
    e.update(kw)
    return e


def test_a_job_whose_worker_is_gone_releases_its_slot():
    E._ITF_JOBS["stranded"] = _entry(thread=_dead_thread())
    assert E._itf_running() == 0, \
        "a worker that will never answer is not a search in flight"


def test_a_live_search_keeps_its_slot():
    """The reaper must not free a slot a real search is using -- that is the
    concurrency the cap exists to refuse."""
    go = threading.Event()
    t = threading.Thread(target=go.wait, daemon=True)
    t.start()
    try:
        E._ITF_JOBS["live"] = _entry(thread=t)
        assert E._itf_running() == 1
    finally:
        go.set()
        t.join(timeout=2)


def test_a_job_with_no_thread_recorded_is_left_alone():
    """The blocking path runs on the request thread and frees its own slot in a
    ``finally``; it records no thread, and must not be reaped out from under
    itself while it is still working."""
    E._ITF_JOBS["sync"] = _entry()
    assert E._itf_running() == 1


def test_a_reaped_job_answers_the_client_instead_of_polling_forever():
    """Freeing the slot is half the fix.  The client polling /api/itf_result
    for that job would otherwise be told "still running" for ever, so the
    stranded entry is marked done WITH an error rather than deleted."""
    E._ITF_JOBS["stranded"] = _entry(thread=_dead_thread())
    out = E.itf_search_status("stranded")
    assert out["done"] is True
    assert out["result"]["ok"] is False
    assert "without returning a result" in out["result"]["error"]


def test_a_reaped_job_then_ages_out_like_any_other():
    """Marked done, it becomes ordinary evictable history -- it must not linger
    in the table as a permanent tombstone either."""
    E._ITF_JOBS["stranded"] = _entry(thread=_dead_thread())
    E._itf_jobs_evict(time.time() + E._ITF_JOB_TTL + 1)
    assert "stranded" not in E._ITF_JOBS


def test_a_worker_that_cannot_be_started_does_not_strand_its_slot(monkeypatch):
    """The exact case the guard is for: the box is out of memory, the thread
    cannot be created -- and the entry is already in the table counting against
    the cap.  It must answer and release, not sit there holding a slot for a
    search that never began."""
    def _boom(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _boom)
    # ...and this test is about the thread, not the box: on a squeezed runner
    # the capacity check would refuse admission first and we would never reach
    # the start (see test_itf_capacity's note on RAM-sensitive tests).
    monkeypatch.setattr(E, "_itf_capacity_block", lambda: None)
    out = E.itf_search_start("no-such-session", {"trace_id": "doomed"})
    assert out["ok"] is False and "could not start" in out["error"]
    assert E._itf_running() == 0
    # ...and the client that POSTed gets the failure, not a job it must poll
    status = E.itf_search_status("doomed")
    assert status["done"] is True and status["result"]["ok"] is False


def test_the_reaper_does_not_disturb_a_finished_job():
    """A collected payload is what a user's minutes bought; reaping runs on
    every admission and must not touch it."""
    E._ITF_JOBS["done"] = _entry(done=True, payload={"ok": True, "kept": 1},
                                 ended=time.time(), thread=_dead_thread())
    E._itf_reap()
    assert E._ITF_JOBS["done"]["payload"] == {"ok": True, "kept": 1}


def test_an_unstarted_thread_is_not_mistaken_for_a_dead_one(monkeypatch):
    """The ordering trap this file's fix had to avoid: ``Thread.is_alive()`` is
    False BEFORE ``start()`` exactly as it is after the thread dies.  Record the
    worker on the entry before starting it and a reap in that window strands a
    search which then runs perfectly well -- releasing its slot early and
    answering the client with an error it will later contradict.

    So the entry carries no thread until the worker is running, and this pins
    it: reaping while a search is in flight must leave it alone.
    """
    started = threading.Event()
    release = threading.Event()

    def _blocking(session, options=None, **kw):
        started.set()
        release.wait(10)
        return {"ok": True, "itf": {"found": False}}

    monkeypatch.setattr(E, "_itf_capacity_block", lambda: None)
    monkeypatch.setattr(E, "itf_search_web", _blocking)
    out = E.itf_search_start("no-such-session", {"trace_id": "inflight"})
    try:
        assert out["ok"] is True
        assert started.wait(5), "the worker never ran"
        # reap repeatedly while it works: the slot must stay held throughout
        for _ in range(5):
            assert E._itf_running() == 1
            time.sleep(0.01)
        assert E._ITF_JOBS["inflight"]["done"] is False
    finally:
        release.set()
    for _ in range(200):                 # let the worker finish and answer
        if E._ITF_JOBS["inflight"]["done"]:
            break
        time.sleep(0.01)
    assert E._ITF_JOBS["inflight"]["done"] is True
    assert E._itf_running() == 0


def test_a_reap_racing_the_start_cannot_strand_the_job(monkeypatch):
    """...and the window itself, deterministically.

    The above pins the invariant; this reproduces the exact interleaving that
    would break it -- a reap landing between "record the worker" and "start
    it".  Recording first, the reaper finds a thread that is not alive (because
    it has not begun) and files a perfectly good search as stranded.
    """
    real_start = threading.Thread.start

    def _reap_then_start(self):
        with E._ITF_JOBS_LOCK:           # the racing caller, exactly here
            E._itf_reap()
        real_start(self)

    release = threading.Event()

    def _blocking(session, options=None, **kw):
        release.wait(10)                 # so _run cannot overwrite the verdict
        return {"ok": True}

    monkeypatch.setattr(E, "_itf_capacity_block", lambda: None)
    monkeypatch.setattr(E, "itf_search_web", _blocking)
    monkeypatch.setattr(threading.Thread, "start", _reap_then_start)
    try:
        E.itf_search_start("no-such-session", {"trace_id": "raced"})
        # The worker is still inside the search here, so whatever the entry says
        # now is what the reap did to it -- not what the search later reported.
        entry = E._ITF_JOBS["raced"]
        assert entry["done"] is False, \
            "a reap during the start filed a running search as stranded"
        assert E._itf_running() == 1, "...and handed its slot away"
    finally:
        release.set()
