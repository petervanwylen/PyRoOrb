"""The /memory attribution report (web/webmem.py).

What these guard, in order of how much they would hurt if they broke:

* the walk cannot RAISE. It runs over live server structures, including objects
  whose attributes execute code on access, so every failure mode has to degrade
  to a skipped subtree. A memory page that 500s during an incident is useless.
* the walk cannot MEASURE ITSELF. Its own `seen` set holds one id() per visited
  object; an early version reported that set (16 MB) and its ints (13 MB) as the
  largest things on the heap, burying every real table.
* nothing double-counts. The tables' combined figure and misc share one `seen`,
  so tables + misc is a partition of the reachable heap, not an overlapping sum.
* no credential leaks. Session handles authorise every interactive endpoint;
  only their readable prefix may appear.
"""
import json
import os
import sys
import threading
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

from pyroorb.web import engine_api  # noqa: E402
from pyroorb.web import fitcache  # noqa: E402
from pyroorb.web import webmem  # noqa: E402


# --- the walker ------------------------------------------------------------
def test_walk_counts_nested_content():
    payload = [bytearray(200000), {"k": "v" * 50000}]
    got = webmem.walk([payload])
    assert got["bytes"] > 240000, got
    assert got["objects"] >= 4


def test_walk_dedups_within_one_call():
    shared_leaf = bytearray(500000)
    one = webmem.walk([[shared_leaf]])["bytes"]
    twice = webmem.walk([[shared_leaf, shared_leaf]])["bytes"]
    assert twice - one < 1000, "an object held twice must be charged once"


def test_shared_seen_set_prevents_double_counting():
    """The rule the report's totals depend on: two tables holding the same fit
    must contribute it once, not twice."""
    obj = bytearray(400000)
    seen = set()
    first = webmem.walk([obj], seen=seen)["bytes"]
    second = webmem.walk([obj], seen=seen)["bytes"]
    assert first > 300000
    assert second == 0


def test_walk_never_measures_its_own_bookkeeping():
    """Regression: the `seen` set and its id() ints once dominated the report."""
    seen = set()
    webmem.walk([list(range(20000))], seen=seen)
    before = len(seen)
    after = webmem.walk([seen], seen=seen)
    assert after["bytes"] == 0, "the walker charged itself"
    # the only new ids are the walker's own scratch containers for THIS call
    assert len(seen) - before <= 2


def test_walk_survives_an_object_that_raises_on_attribute_access():
    """A lazy module proxy imports on touch -- this is the live `_gdbm` failure
    that took the whole report down with ModuleNotFoundError."""
    class Hostile:
        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named '_nope'")

    class HostileLen:
        def __len__(self):
            raise ValueError("no")

        def __sizeof__(self):
            raise RuntimeError("no")

    got = webmem.walk([Hostile(), HostileLen(), {"x": Hostile()}])
    assert got["objects"] >= 2


def test_walk_survives_a_container_mutating_underneath_it():
    class Mutating(dict):
        def items(self):
            raise RuntimeError("dictionary changed size during iteration")

    m = Mutating()
    m["a"] = bytearray(1000)
    assert webmem.walk([m])["objects"] >= 1


def test_walk_respects_the_node_budget():
    got = webmem.walk([list(range(50000))], budget=100)
    assert got["truncated"] is True
    assert got["objects"] <= 100


def test_walk_tally_and_big_objects():
    tally, big = {}, []
    webmem.walk([{"a": bytearray(900000)}], tally=tally, big=big, big_min=500000)
    assert "bytearray" in tally and tally["bytearray"][0] > 800000
    assert big and "bytearray" in big[0][1]


def test_describe_never_reveals_contents():
    """The page is unauthenticated by design: shapes and sizes only."""
    d = webmem._describe("super-secret-designation-string")
    assert "secret" not in d and d == "str len=31"


# --- the report ------------------------------------------------------------
def test_report_is_json_safe_and_complete():
    rep = webmem.report()
    json.dumps(rep)                       # must survive the HTTP layer
    assert rep["ok"] is True
    keys = {t["key"] for t in rep["tables"]}
    assert {"sessions", "fitcache", "itf_jobs", "traces", "threads"} <= keys
    for key in ("tables_combined_mb", "misc_mb", "accounted_mb", "rss_mb"):
        assert key in rep["totals"]


def test_tables_plus_misc_partition_the_heap():
    rep = webmem.report()
    t = rep["totals"]
    assert abs(t["accounted_mb"] - (t["tables_combined_mb"] + t["misc_mb"])) < 0.02
    # gross >= combined always: the difference IS the overlap, never negative
    assert t["tables_gross_mb"] >= t["tables_combined_mb"] - 0.01
    assert t["overlap_mb"] >= -0.01


def test_tables_only_mode_skips_the_expensive_pass():
    rep = webmem.report(with_misc=False)
    assert rep["misc"] is None
    assert rep["totals"]["misc_mb"] is None


def test_a_cached_fit_shows_up_and_its_size_is_plausible():
    """fitcache is sized from its real contents, not a per-observation guess."""
    class FakeFit:
        designation = "2001 KQ76"

        def __init__(self):
            self.payload = [bytearray(300000)]

        def n_total_observations(self):
            return 12

    key = fitcache.put(FakeFit(), key="webmem-test")
    try:
        rep = webmem.report(with_misc=False)
        tab = next(t for t in rep["tables"] if t["key"] == "fitcache")
        assert tab["loaded"] and tab["mb"] >= 0.28, tab
        row = next(r for r in tab["rows"] if r["id"] == "webmem-test")
        assert row["n_obs"] == 12 and row["label"] == "2001 KQ76"
    finally:
        fitcache._CACHE.pop(key, None)


def test_session_handles_are_not_leaked_in_full():
    """A handle is the whole authorisation for the interactive endpoints; the page
    may show its readable prefix and nothing more."""
    class FakeFit:
        designation = "2019 QM111"
        history = []
        asteroids = False
        epoch_tdb = 2460000.5
        _ctx = {}
        _state = None

        class _res:
            x = [0.0] * 6

    sid = None
    try:
        sid = engine_api._store_session(FakeFit())
        rep = webmem.report(with_misc=False)
        blob = json.dumps(rep)
        assert sid not in blob, "the full session handle reached the report"
        assert sid.split("-")[0] in blob, "the readable prefix should still show"
    finally:
        if sid:
            with engine_api._SES_LOCK:
                engine_api._SESSIONS.pop(sid, None)
                if sid in engine_api._SES_ORDER:
                    engine_api._SES_ORDER.remove(sid)


def test_every_resident_session_is_listed_even_at_the_table_cap():
    """The per-entry rows must cover the whole session table, not the largest N.

    The row cap used to be a literal 16, which was invisible while the session
    table held at most twelve -- every session was always listed.  Raising
    _SES_MAX to 64 (a crowd interacting, each on their own private session)
    silently pushed the smallest sessions off the page, so "is my fit still
    resident?" could not be answered for the very sessions most likely to be
    evicted next.  The cap now follows the session cap, and anything past it is
    counted in `rows_omitted` rather than dropped without a word.
    """
    class FakeFit:
        designation = "2019 QM111"
        history = []
        asteroids = False
        epoch_tdb = 2460000.5
        _ctx = {}
        _state = None

        class _res:
            x = [0.0] * 6

    made = []
    try:
        for _ in range(engine_api._SES_MAX):        # fill it to the cap
            made.append(engine_api._store_session(FakeFit()))
        sid = engine_api._store_session(FakeFit())
        made.append(sid)
        rep = webmem.report(with_misc=False)
        blob = json.dumps(rep)
        tab = next(t for t in rep["tables"] if t.get("key") == "sessions")
        assert tab["rows_omitted"] == 0, "sessions were dropped from the page"
        assert len(tab["rows"]) == len(engine_api._SESSIONS)
        assert sid not in blob, "the full session handle reached the report"
        assert sid.split("-")[0] in blob, "a resident session is missing from the page"
    finally:
        with engine_api._SES_LOCK:
            for s in made:
                engine_api._SESSIONS.pop(s, None)
                if s in engine_api._SES_ORDER:
                    engine_api._SES_ORDER.remove(s)


def test_a_working_thread_is_attributed_to_the_threads_table():
    """The category no table can hold: a running ITF search's working set lives in
    locals on a worker thread, and this is the route by which it is priced."""
    stop = threading.Event()
    started = threading.Event()

    def worker():
        big_local_working_set = [bytearray(2_000_000) for _ in range(3)]  # noqa: F841
        started.set()
        stop.wait(20)

    # a realistic worker name: itf-search-<job_id>, where the job id is the
    # /api/itf_result collection key -- redacted on the unauthenticated page
    job_id = "tr" + "deadbeefcafef00d1234"
    th = threading.Thread(target=worker, name="itf-search-" + job_id,
                          daemon=True)
    th.start()
    try:
        assert started.wait(5)
        time.sleep(0.05)
        rep = webmem.report(with_misc=False)
        tab = next(t for t in rep["tables"] if t["key"] == "threads")
        assert tab["loaded"], "worker threads should be visible"
        assert tab["mb"] >= 5.5, f"the worker's 6 MB working set was missed: {tab}"
        # the row is present, but the collectible job id is redacted to a prefix
        blob = json.dumps(tab)
        assert job_id not in blob, "the full job id (a live credential) leaked"
        assert any(r["id"].startswith("itf-search-") and r["id"].endswith("…")
                   for r in tab["rows"]), tab["rows"]
    finally:
        stop.set()
        th.join(timeout=5)


def test_thread_working_sets_survive_many_threads():
    """The threads TOTAL must not collapse as the thread count rises.

    `walk` identifies visited objects by id(), which is unique only among LIVE
    objects, and `_children(frame)` synthesizes a fresh `dict(f_locals)` that
    nothing else holds.  Freed the moment the walk pops it, its address was
    handed straight to the next frame's snapshot -- which then read as
    already-seen and lost its whole subtree.  One worker thread made that a
    coincidence; a fit-worker pool (or six concurrent ITF searches) made it
    reliable, and the table that exists to price a running search reported 0.02
    MB against a 6 MB working set while the per-thread rows still showed it.

    So: many idle threads plus one fat one, and the total must still contain the
    fat one.  The per-row figure is deliberately NOT the assertion -- rows are
    walked in their own calls and never showed the bug.

    ORDER-SENSITIVE, and honestly so: run alone it passes even with the fix
    reverted, because id reuse needs a heap that has already churned.  It fails
    without the fix once a fit-worker pool is resident -- `pytest
    tests/test_fitqueue.py tests/web/test_webmem.py`, which is the
    arrangement the suite produces and the state every live server is in.  A
    self-contained reproducer would have to pin CPython's allocator, which would
    test the allocator rather than this walk.
    """
    stop = threading.Event()
    ready = threading.Event()
    n_idle = 12

    def idle():
        stop.wait(20)

    def fat():
        held = [bytearray(2_000_000) for _ in range(3)]      # noqa: F841
        ready.set()
        stop.wait(20)

    threads = [threading.Thread(target=idle, name=f"idle-{i}", daemon=True)
               for i in range(n_idle)]
    threads.append(threading.Thread(target=fat, name="itf-search-fat", daemon=True))
    for t in threads:
        t.start()
    try:
        assert ready.wait(5)
        time.sleep(0.05)
        rep = webmem.report(with_misc=False)
        tab = next(t for t in rep["tables"] if t["key"] == "threads")
        assert tab["mb"] >= 5.5, (
            f"the fat thread's working set vanished from the TOTAL with "
            f"{n_idle} other threads resident: {tab['mb']} MB")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)


def test_worker_thread_job_id_is_redacted():
    """The worker-thread name is itf-search-<job_id>, and that job id is the
    /api/itf_result collection key whose payload carries a session handle. On an
    unauthenticated page it must be redacted to a short prefix, exactly as the
    _ITF_JOBS table redacts it."""
    long = "itf-search-tr0123456789abcdef"
    got = webmem._safe_thread_name(long)
    assert got == "itf-search-tr012345…", got
    assert "6789abcdef" not in got
    # a non-job thread name is shown as-is (length-capped)
    assert webmem._safe_thread_name("MainThread") == "MainThread"
    assert webmem._safe_thread_name("x" * 40) == "x" * 28


def test_report_never_imports_a_module_to_measure_it():
    """Importing ASSIST to size its ephemeris would allocate the thing we are
    measuring. An absent module is reported as not-loaded."""
    before = set(sys.modules)
    webmem.report(with_misc=False)
    new = set(sys.modules) - before
    assert not new, f"the report imported {new}"


# ---------------------------------------------------------------------------
# The report must not be a memory amplifier
# ---------------------------------------------------------------------------
def test_repeated_reports_do_not_inflate_misc():
    """Calling the deep pass repeatedly must not grow the number it reports.

    The walk follows thread frames, so its `seen` set and the frames holding it
    form a cycle; gc.get_objects() then hands the NEXT pass that uncollected
    debris as if it were live.  Unfixed, misc doubled per call on a completely
    idle process (2.2 -> 4.9 -> 8.1 -> 14.6 -> 27.7 -> 53.8 MB) and RSS went 22
    -> 145 MB, all of it recoverable by a single gc.collect().
    """
    seen = [webmem.report(with_misc=True)["totals"]["misc_mb"] for _ in range(4)]
    first, last = seen[0], seen[-1]
    assert first > 0
    # Some drift is honest (the test process really does allocate); doubling is not.
    assert last < first * 1.5, f"misc grew across repeated reports: {seen}"


def test_repeated_reports_do_not_grow_object_counts():
    """The same guard on the object count, which is what actually ran away."""
    counts = [webmem.report(with_misc=True)["misc"]["objects"] for _ in range(4)]
    assert counts[-1] < counts[0] * 1.5, f"misc object count ran away: {counts}"


# ---------------------------------------------------------------------------
# Native memory: the half of RSS no heap walk can reach
# ---------------------------------------------------------------------------
def test_native_block_is_present_and_cheap():
    """`native` rides along with the tables pass -- no heap walk, no password."""
    rep = webmem.report(with_misc=False)
    nat = rep.get("native")
    assert nat is not None
    assert "malloc" in nat and "maps" in nat
    json.dumps(rep)                                  # still JSON-safe


def test_native_thread_counts_distinguish_os_from_python():
    """A large OS-vs-Python thread gap is native pools (OpenMP, BLAS), each of
    which also takes its own allocator arena -- worth seeing, so report both."""
    nat = webmem.report(with_misc=False)["native"]
    th = nat.get("threads")
    if th is None:                                   # not Linux
        return
    assert th["os"] >= 1 and th["python_frames"] >= 1
    assert th["os"] >= th["python_frames"]


def test_smaps_counts_a_second_mapping_of_one_file():
    """The headline: a file mapped TWICE pays its resident pages twice, and that
    is invisible to every Python-side table (an assist.Ephem is a tiny wrapper
    over a C mmap of the 747 MB of kernels).  `duplicated` is what names it.

    Uses mmap on a real temp file so the check exercises the smaps parser, not a
    stub -- including the fact that the device field ("08:01") carries a colon
    and so must never be used to split the header line.
    """
    import mmap
    import tempfile

    maps = webmem._smaps_by_file()
    if maps is None:                                 # not Linux
        return
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(b"\0" * (4 << 20))                  # 4 MB, big enough to show
        fh.flush()
        m1 = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        m2 = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            m1[:]                                    # fault the pages in, both maps
            m2[:]
            rows = webmem._smaps_by_file()
            hit = [r for r in rows["duplicated"] if r["path"].endswith(".bin")]
            assert hit, f"a twice-mapped file was not flagged: {rows['duplicated']}"
            assert hit[0]["opens"] == 2
        finally:
            m1.close()
            m2.close()


def test_shared_library_segments_are_not_reported_as_duplicates():
    """A .so is mapped as several VMAs at different file offsets -- one open, not
    a dozen duplicates.  Without the offset-0 rule numpy alone reads as 12."""
    rows = webmem._smaps_by_file()
    if rows is None:
        return
    for r in rows["duplicated"]:
        assert r["opens"] > 1
    # every row carries both numbers, so a segmented library is legible
    for r in rows["files"]:
        assert r["vmas"] >= r["opens"] >= 0


# ---------------------------------------------------------------------------
# Allocator retention: the free-but-held memory a completed job leaves behind
# ---------------------------------------------------------------------------
def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def test_trim_allocator_reports_without_trimming_below_threshold():
    """The gate is what keeps this off the ordinary path: an idle process holds
    little, so nothing should be handed back."""
    out = webmem.trim_allocator(threshold_mb=1e9)     # never reachable
    if out is None:
        return                                        # not glibc
    assert out["trimmed"] is False
    assert out["free_held_mb"] >= 0
    assert out["threshold_mb"] == 1e9


def test_trim_allocator_is_a_noop_off_glibc(monkeypatch):
    """Every caller sits behind this, so it must degrade rather than raise."""
    monkeypatch.setattr(webmem, "_LIBC", None)
    monkeypatch.setattr(webmem, "_LIBC_TRIED", True)
    assert webmem.trim_allocator() is None
    assert webmem._malloc_info() is None
    rep = webmem.report(with_misc=False)              # the report still builds
    assert rep["native"]["malloc"] is None


@pytest.mark.slow
def test_trim_allocator_returns_pages_a_finished_job_left_behind():
    """The real thing: a big transient with a few scattered survivors leaves the
    arena resident, and only an explicit trim gives it back.

    This is the production failure -- 99.6 MB in use pinning 4760 MB held, a
    933 MB headroom, and an ITF gate refusing every search. Neither glibc env
    knob helps (MALLOC_TRIM_THRESHOLD_ governs the main arena's top only;
    MALLOC_ARENA_MAX=2 held the same and raised RSS), so the trim is the fix.
    """
    import gc
    import random
    import threading

    if webmem._mallinfo() is None:
        pytest.skip("not glibc")
    before = _rss_mb()
    if before is None:
        pytest.skip("no /proc")

    survivors = []
    lock = threading.Lock()

    def churn():
        blocks = []
        for i in range(400_000):
            # SMALL blocks: below the mmap threshold, so they live in the arena
            # and are what pins it. Large ones are mmap'd and return on free.
            blocks.append(bytearray(random.randrange(64, 700)))
            if i % 1500 == 0:
                with lock:
                    survivors.append(blocks[-1])
        blocks.clear()

    t = threading.Thread(target=churn)
    t.start()
    t.join()
    gc.collect()

    held = webmem.trim_allocator(threshold_mb=1e9)["free_held_mb"]
    if held < 32:
        pytest.skip(f"allocator returned it unprompted ({held:.0f} MB held)")
    peak = _rss_mb()
    out = webmem.trim_allocator(threshold_mb=8)
    assert out["trimmed"] is True
    after = _rss_mb()
    # The pages come back; the FREE LIST does not shrink, so fordblks is not the
    # check -- RSS is. (That asymmetry is documented on trim_allocator.)
    assert after < peak - 16, (
        f"trim recovered only {peak - after:.0f} MB of {held:.0f} MB held")
