"""The /healthz payload must be answerable, always.

/healthz is the instrument every capacity decision is read from, and the ONE
route whose failure is invisible from the outside: Railway's healthcheck points
at `/`, which keeps serving static HTML while the app is in trouble, so a broken
/healthz is not merely a missing metric -- it is the monitoring going dark
without anything going red.

This exists because it went dark. A refactor deleted the `_QUEUE_MAX` module
global; `queue_max()` raised NameError; /healthz 502'd on the deployed server
while every other route answered normally. The offline suite was green, because
its own fixture had been assigning that global into existence.

So this calls the real thing, with no fixture preparing the module.
"""
import os
import sys

from pyroorb.web import fitqueue  # noqa: E402


def test_stats_answers_without_raising():
    """Exactly what the /healthz handler calls, on an untouched module."""
    st = fitqueue.stats()
    for key in ("workers", "pool_size", "running", "queued", "queue_max",
                "jobs_held", "median_fit_sec", "cpu_quota", "memory_limit_mb",
                "disk_cache"):
        assert key in st, f"/healthz lost its {key!r} field"


def test_the_bounds_it_reports_are_usable_numbers():
    st = fitqueue.stats()
    assert st["pool_size"] >= 1
    assert st["queue_max"] >= st["pool_size"], \
        "a queue shallower than the pool would refuse work the workers are free to take"


def test_queue_max_is_memoised_not_conjured():
    """The global must EXIST at module scope -- not be created by whoever asks
    first, and not by a test fixture (see the module docstring)."""
    assert hasattr(fitqueue, "_QUEUE_MAX")


def test_healthz_names_the_running_revision():
    """A deploy marker that has to be inferred gets inferred wrong.

    Without this, "did my fix ship?" was answered from side effects -- a reset
    disk cache, a dropped RSS -- which are equally the signature of a plain
    restart. The process already knows its revision (it is folded into every fit
    cache key so a deploy cannot serve the previous version's payload); this just
    stops it being a secret.
    """
    from pyroorb.web import server
    h = server.Handler._healthz(object.__new__(server.Handler))
    assert "revision" in h
    assert isinstance(h["revision"], str) and h["revision"]


def test_healthz_carries_the_sweep_watch():
    """A thread parked inside one integrate() call for an hour is the outage the
    fit deadline cannot reach (Python cannot interrupt a C call).  /healthz is
    where a monitor sees it: sweeps in flight, the oldest one's age, and how many
    are past the wedged bar -- three numbers, no state, no names."""
    from pyroorb.web import server
    h = server.Handler._healthz(object.__new__(server.Handler))
    assert "sweeps" in h
    assert set(h["sweeps"]) >= {"inflight", "oldest_s", "wedged"}
    assert h["sweeps"]["inflight"] == 0 and h["sweeps"]["wedged"] == 0
