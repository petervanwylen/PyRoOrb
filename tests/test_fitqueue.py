"""Offline tests for the web fit admission queue (src/pyroorb/web/fitqueue.py).

No fitting happens here: the queue's contract is about ADMISSION -- what gets
coalesced, what gets refused, what a caller is told while it waits -- and all of
that is exercised with trivial work functions.  The one property that needs a
real fit (a restored session reproducing its template) lives in
test_fitqueue_restore.py, which is marked slow.
"""
import os
import sys
import time
import threading

import pytest

from pyroorb.web import fitdisk  # noqa: E402
import pyroorb.web.fitqueue as fq  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_disk_cache(tmp_path, monkeypatch):
    """Point the disk tier at a per-test directory.

    The cache is process-global and lives in $TMPDIR, so without this a key
    written by an earlier test -- or an earlier RUN, since the whole point is
    that it outlives the process's in-memory table -- is served straight back
    and the work under test never executes.  Every one of these tests asserts
    something about work running, so they must each start from an empty cache.
    """
    monkeypatch.setenv("PYROORB_FIT_CACHE_DIR", str(tmp_path / "fitcache"))
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)
    # hermetic: the box's own free space must not decide these tests --
    # below fitdisk.MIN_FREE_MB `put` refuses, which is correct and tested on purpose
    # by test_a_nearly_full_disk_stops_writing, which pins its own low value
    monkeypatch.setattr(fitdisk, "_free_mb", lambda d: 1e9)
    yield
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)


@pytest.fixture(autouse=True)
def _fresh_capacity(monkeypatch):
    """The derived caps are memoised per process (container limits cannot change
    under a running process), so a test that fakes a different box must not
    leave its answer behind for the next one -- nor inherit one.

    ``monkeypatch.setattr``, NOT a bare assignment, and that is the whole point:
    setattr raises when the attribute does not exist, while an assignment would
    CREATE it.  An earlier version of this fixture assigned, and so quietly
    supplied a module global that a refactor had deleted -- the suite stayed green
    while ``queue_max()`` raised NameError on the deployed server and took
    /healthz down.  A fixture that manufactures the thing under test cannot fail.
    """
    fq.capacity._reset_for_tests()
    monkeypatch.setattr(fq, "_QUEUE_MAX", None)
    yield
    fq.capacity._reset_for_tests()


@pytest.fixture(autouse=True)
def _clean_queue():
    """Each test gets an empty table.  The worker threads are process-global and
    daemonised, so they are left running between tests -- only the state they
    read is reset."""
    with fq._LOCK:
        fq._JOBS.clear()
        fq._BY_KEY.clear()
        fq._WAITING.clear()
        fq._DURATIONS.clear()
    yield
    with fq._LOCK:
        fq._JOBS.clear()
        fq._BY_KEY.clear()
        fq._WAITING.clear()


def _done(key, value):
    r = fq.submit(key, lambda: {"ok": True, "v": value})
    assert fq.wait(r["job"], 30), "work never finished"
    return r


def test_key_is_content_not_spelling():
    """Same fit, differently typed, must share a key -- otherwise the herd this
    whole module exists for splits into one fit per spelling."""
    assert fq.fit_key(["2026 ab123"], False) == fq.fit_key([" 2026 AB123 "], False)
    assert fq.fit_key(["A", "B"], False) == fq.fit_key(["B", "A"], False)
    assert fq.fit_key(["A"], False) != fq.fit_key(["A"], True)
    assert fq.fit_key(["A"], False) != fq.fit_key(["A", "B"], False)


def test_the_key_is_stable_under_concurrent_first_use():
    """Two callers arriving together must derive the SAME key.

    fit_key folds in the engine revision, which used to be memoised lazily: two
    simultaneous first-callers both found the memo empty, both shelled out to
    git, and a `rev-parse` that timed out on a slow filesystem yielded "?" while
    the other yielded the sha.  Two revisions is two keys for one object, so the
    herd single-flight exists to collapse split into two fits and two cache
    entries instead -- found in local load testing, where twelve simultaneous
    requests for one designation produced two identical cache files.
    """
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        keys = set(pool.map(lambda _i: fq.fit_key(["2023 QJ3"], False), range(64)))
    assert len(keys) == 1, f"the same object derived {len(keys)} keys: {keys}"


def test_key_includes_engine_revision():
    """A deploy must not serve a payload computed by the previous code."""
    k1 = fq.fit_key(["A"], False)
    saved, fq._REV = fq._REV, "deadbeef"
    try:
        assert fq.fit_key(["A"], False) != k1
    finally:
        fq._REV = saved


def test_single_flight_coalesces_identical_work():
    """A hundred requests for one object must produce ONE run of the work."""
    started = threading.Event()
    release = threading.Event()
    runs = []

    def work():
        runs.append(1)
        started.set()
        release.wait(30)
        return {"ok": True}

    first = fq.submit("hot", work)
    assert started.wait(30)
    others = [fq.submit("hot", work) for _ in range(99)]
    assert all(o["job"] == first["job"] for o in others)
    release.set()
    assert fq.wait(first["job"], 30)
    assert len(runs) == 1, f"work ran {len(runs)} times, not once"


def test_concurrent_submit_on_a_cold_key_creates_exactly_one_job(monkeypatch):
    """Two callers racing on a key NEITHER has ever fitted must still coalesce
    into one job.

    submit() releases the lock between the in-memory check and the disk-tier
    read (so a slow disk read cannot hold up the worker pool), then
    re-acquires it to enqueue.  A caller that raced through that gap on a truly
    cold key -- the exact case a launch spike hits hardest, since nobody has
    fitted the object yet and every simultaneous request misses the disk tier
    too -- used to fall all the way through to job creation a second time,
    running the identical work twice. Widen the gap with a sleep in
    ``_from_disk`` so the race reproduces reliably rather than by luck of
    scheduling.
    """
    real_from_disk = fq._from_disk

    def _slow_from_disk(key, max_age=None):
        time.sleep(0.05)
        return real_from_disk(key, max_age)

    monkeypatch.setattr(fq, "_from_disk", _slow_from_disk)
    barrier = threading.Barrier(8)
    runs = []

    def work():
        runs.append(1)
        return {"ok": True}

    def go():
        barrier.wait(5)
        return fq.submit("cold-key", work)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: go(), range(8)))
    jobs = {r.get("job") for r in results if r.get("job")}
    assert len(jobs) == 1, f"one cold key produced {len(jobs)} distinct job(s)"
    (jid,) = jobs
    assert fq.wait(jid, 30)
    assert len(runs) == 1, f"work ran {len(runs)} times for one cold key, not once"


def test_finished_job_is_reused_within_the_ttl():
    """The done-job table IS the L1 response cache: a later identical request is
    served from it rather than re-queued."""
    r1 = _done("k", 1)
    r2 = fq.submit("k", lambda: {"ok": True, "v": 2})
    assert r2["job"] == r1["job"]
    assert fq.status(r2["job"])["result"]["v"] == 1


def test_finished_job_is_not_reused_once_stale():
    r1 = _done("k", 1)
    with fq._LOCK:                      # age it past the reuse window
        fq._JOBS[r1["job"]]["ended"] -= (fq.DONE_TTL + 1)
    r2 = fq.submit("k", lambda: {"ok": True, "v": 2})
    assert r2["job"] != r1["job"]


def _age_disk_entry(key, secs):
    """Move a disk-tier record's `stored` clock ``secs`` into the past."""
    import json
    path = fitdisk._path(fitdisk._resolve_dir(), key)
    with open(path) as fh:
        rec = json.load(fh)
    rec["stored"] -= secs
    with open(path, "w") as fh:
        json.dump(rec, fh)


def _on_disk(key):
    """Block until the write-through lands -- it runs AFTER waiters are released."""
    path = fitdisk._path(fitdisk._resolve_dir(), key)
    for _ in range(300):
        if os.path.exists(path):
            return
        time.sleep(0.01)
    raise AssertionError("disk write-through never landed")


def _forget_memory():
    with fq._LOCK:
        fq._JOBS.clear()
        fq._BY_KEY.clear()


def test_max_age_shortens_the_memory_reuse_window():
    """A NEOCP object's arc grows by the hour: "P22r4rx P/2015 X3" kept serving a
    fit made from 4 of P22r4rx's observations while "P22r4rx" alone showed 8.
    A caller's max_age caps how long a finished answer is reused."""
    r1 = _done("k", 1)
    with fq._LOCK:
        fq._JOBS[r1["job"]]["ended"] -= 200          # well inside DONE_TTL
    assert fq.submit("k", lambda: {"ok": True, "v": 2})["job"] == r1["job"]
    r3 = fq.submit("k", lambda: {"ok": True, "v": 3}, max_age=120)
    assert r3["job"] != r1["job"]


def test_max_age_shortens_the_disk_reuse_window():
    _done("k", 1)
    _on_disk("k")
    _forget_memory()
    _age_disk_entry("k", 200)                         # well inside TTL_SECS
    hit = fq.submit("k", lambda: {"ok": True, "v": 2})
    assert hit.get("cached") == "disk" and hit["result"]["v"] == 1
    miss = fq.submit("k", lambda: {"ok": True, "v": 3}, max_age=120)
    assert miss.get("cached") != "disk"
    assert fq.wait(miss["job"], 30)
    assert fq.status(miss["job"])["result"]["v"] == 3


def test_max_age_keeps_a_fresh_answer():
    r1 = _done("k", 1)
    assert fq.submit("k", lambda: {"ok": True, "v": 2}, max_age=120)["job"] == r1["job"]
    _on_disk("k")
    _forget_memory()
    hit = fq.submit("k", lambda: {"ok": True, "v": 2}, max_age=120)
    assert hit.get("cached") == "disk" and hit["result"]["v"] == 1


def test_a_fit_naming_any_temporary_id_lives_as_long_as_its_astrometry():
    from pyroorb import obs_sources
    from pyroorb.obs80 import NEOCP_CACHE_TTL_S
    short = NEOCP_CACHE_TTL_S
    assert obs_sources.fit_reuse_max_age(["P22r4rx", "P/2015 X3"]) == short
    assert obs_sources.fit_reuse_max_age(["P22r4rx"]) == short
    assert obs_sources.fit_reuse_max_age("C46MJF1") == short
    for published in (["P/2015 X3"], ["433"], ["10P"], ["2013 PY26", "1I"],
                      ["C/2023 A3"]):
        assert obs_sources.fit_reuse_max_age(published) is None, published


def test_a_failed_fetch_is_not_replayed_to_the_retry():
    """"The MPC did not answer ... try again in a moment" used to be served from
    the done-job table for DONE_TTL (15 min) to the very retry it asked for, long
    after a twenty-second outage had ended.  A failure is re-run, not replayed."""
    r1 = fq.submit("k", lambda: {"ok": False, "retryable": True,
                                 "error": "the MPC did not answer for 2013 LT7"})
    assert fq.wait(r1["job"], 30)
    r2 = fq.submit("k", lambda: {"ok": True, "v": 2})
    assert r2["job"] != r1["job"]
    assert fq.wait(r2["job"], 30)
    assert fq.status(r2["job"])["result"]["v"] == 2
    # and the success that replaced it IS cached
    r3 = fq.submit("k", lambda: {"ok": True, "v": 3})
    assert r3["job"] == r2["job"]


def test_a_timed_out_fit_is_still_replayed():
    """A timeout is a property of the object, and each retry would hold a worker
    for FIT_MAX_SEC -- that one the cache must keep answering."""
    r1 = fq.submit("k", lambda: {"ok": False, "timed_out": True, "error": "slow"})
    assert fq.wait(r1["job"], 30)
    r2 = fq.submit("k", lambda: {"ok": True, "v": 2})
    assert r2["job"] == r1["job"]


def test_queue_refuses_past_its_depth_and_says_so_plainly():
    """A bounded queue is the point: past the cap the honest answer is a refusal,
    not an accepted job the box will never reach."""
    release = threading.Event()
    keys = []
    try:
        for i in range(fq.queue_max() + fq.pool_size() + 5):
            r = fq.submit(f"k{i}", lambda: (release.wait(30), {"ok": True})[1])
            keys.append(r)
            if not r.get("ok"):
                break
        assert not keys[-1].get("ok"), "queue accepted work past its cap"
        assert keys[-1].get("busy") is True
        assert "full" in keys[-1]["error"]
    finally:
        release.set()


def test_snapshot_reports_position_and_an_eta():
    release = threading.Event()
    try:
        jobs = [fq.submit(f"q{i}", lambda: (release.wait(30), {"ok": True})[1])
                for i in range(fq.pool_size() + 3)]
        waiting = [j for j in jobs if j["state"] == "queued"]
        assert waiting, "nothing queued -- pool too large for this assertion"
        assert waiting[0]["ahead"] == waiting[0]["waiting"] - 1
        assert waiting[-1]["waiting"] > waiting[0]["waiting"]
        assert waiting[0]["eta_sec"] > 0
    finally:
        release.set()


def test_session_is_never_handed_out_and_a_restore_token_replaces_it():
    """Coalescing means one payload reaches many clients; the interactive handle
    inside it must not, or they would edit each other's fit."""
    tmpl = {"pretend": "session"}
    r = fq.submit("s", lambda: {"ok": True, "session": "s1-secret", "_retain": tmpl})
    assert fq.wait(r["job"], 30)
    for _ in range(3):
        result = fq.status(r["job"])["result"]
        assert "session" not in result
        assert result["restore"] == r["job"]
    assert fq.retained(r["job"]) is tmpl


def test_retain_is_stripped_from_the_cached_payload():
    """`_retain` carries live objects; it must never reach a client or the JSON
    encoder."""
    r = fq.submit("s", lambda: {"ok": True, "_retain": object()})
    assert fq.wait(r["job"], 30)
    assert "_retain" not in fq.status(r["job"])["result"]


def test_a_crashing_fit_still_answers_its_waiters():
    """The failure that must never happen is a client polling forever."""
    def boom():
        raise RuntimeError("integrator exploded")
    r = fq.submit("bad", boom)
    assert fq.wait(r["job"], 30)
    result = fq.status(r["job"])["result"]
    assert result["ok"] is False and "integrator exploded" in result["error"]


def test_unknown_job_expires_rather_than_spinning():
    out = fq.status("nope")
    assert out["ok"] is False and out["expired"] is True


def test_running_jobs_are_never_evicted():
    """Evicting a live job would not stop its thread -- it would only lose the
    result somebody is waiting for."""
    release = threading.Event()
    try:
        r = fq.submit("live", lambda: (release.wait(30), {"ok": True})[1])
        with fq._LOCK:
            for i in range(fq.JOBS_MAX + 10):     # flood the table with finished jobs
                fq._JOBS[f"z{i}"] = {"key": f"z{i}", "work": None, "state": "done",
                                     "created": 0, "started": 0, "ended": time.time(),
                                     "collected": time.time(), "payload": {"ok": True},
                                     "claims": 0, "retain": None}
            fq._evict(time.time())
            assert r["job"] in fq._JOBS
    finally:
        release.set()


def test_pool_and_queue_are_bounded_and_positive():
    assert fq.pool_size() >= 1
    assert fq.queue_max() >= 1
    st = fq.stats()
    assert st["pool_size"] == fq.pool_size()
    assert st["queue_max"] == fq.queue_max()


def test_pool_size_respects_a_memory_limit(monkeypatch):
    """A small container must not run a fit per core: the pool is the MINIMUM of
    what the CPUs and the memory limit allow.

    The arithmetic itself now lives in `capacity`, where it is decided together
    with the ITF cap (they spend the same box), so this patches the box THERE.
    The assertion stays here because the pool is still the queue's contract --
    this is the property callers depend on, wherever it is computed.
    """
    monkeypatch.delenv("PYROORB_FIT_WORKERS", raising=False)
    monkeypatch.setattr(fq.capacity, "cpu_quota", lambda: 8)
    monkeypatch.setattr(fq.capacity, "limit_mb", lambda: 1024)
    fq.capacity._reset_for_tests()                   # the answer is memoised
    # (1024 - 400) // 250 == 2, against 7 by CPU
    assert fq.pool_size() == 2
    monkeypatch.setattr(fq.capacity, "limit_mb", lambda: 512)
    fq.capacity._reset_for_tests()
    assert fq.pool_size() == 1          # never zero: one fit at a time still works
    monkeypatch.setattr(fq.capacity, "limit_mb", lambda: None)
    fq.capacity._reset_for_tests()
    assert fq.pool_size() == 7          # uncapped: CPU-bound, one core left over


def test_worker_override_wins(monkeypatch):
    monkeypatch.setenv("PYROORB_FIT_WORKERS", "3")
    assert fq.pool_size() == 3


def test_a_one_off_job_keeps_its_session():
    """run_bounded work (a pasted blob, a re-auto-solve) is keyed uniquely and can
    never be shared, so its payload must reach the caller exactly as the engine
    built it -- session included.  Stripping it would log the user out of the fit
    they just pasted."""
    out = fq.run_bounded(lambda: {"ok": True, "session": "s1-mine"}, 30, label="paste")
    assert out["ok"] and out["session"] == "s1-mine"
    assert "restore" not in out


def test_one_off_work_is_never_coalesced():
    runs = []
    for _ in range(3):
        fq.run_bounded(lambda: runs.append(1) or {"ok": True}, 30, label="paste")
    assert len(runs) == 3


def test_one_off_work_is_still_admitted():
    """The point of routing paste/reauto through the pool: they are refused when
    the queue is full, instead of running unbounded on the request thread."""
    import time
    release = threading.Event()
    blocked = lambda: (release.wait(30), {"ok": True})[1]  # noqa: E731
    try:
        # Admission counts WAITING jobs. Fillers can arrive faster than the workers take
        # them, so a first refusal may come while workers are still idle; once each
        # worker has picked a job up the queue drains by that many. Fill until a
        # refusal still holds after the workers have had time to start.
        n, deadline = 0, time.time() + 20
        while True:
            assert time.time() < deadline, "queue never stayed full"
            n += 1
            if fq.submit(f"filler{n}", blocked).get("ok"):
                continue
            time.sleep(0.3)
            n += 1
            if not fq.submit(f"filler{n}", blocked).get("ok"):
                break
        out = fq.run_bounded(lambda: {"ok": True}, 5, label="paste")
        assert out.get("ok") is False and out.get("busy") is True
    finally:
        release.set()
