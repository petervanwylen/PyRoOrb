"""The ephemeral on-disk fit cache (src/pyroorb/web/fitdisk.py) and its wiring into the
admission queue.

The cache is a CAPACITY tier that must vanish with the container, so the tests
that matter are about where it refuses to live, what it does when the disk is
full or the file is torn, and that a disk hit never occupies a worker slot.
"""
import json
import os
import sys
import time

import pytest

from pyroorb.web import fitdisk  # noqa: E402
import pyroorb.web.fitqueue as fq  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROORB_FIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)
    # hermetic: the box's own free space must not decide these tests --
    # below fitdisk.MIN_FREE_MB `put` refuses, which is correct and tested on purpose
    # by test_a_nearly_full_disk_stops_writing, which pins its own low value
    monkeypatch.setattr(fitdisk, "_free_mb", lambda d: 1e9)
    yield
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)


# --- where it may live ------------------------------------------------------

@pytest.mark.parametrize("path", ["/data", "/data/fits", "/data/../data/x",
                                  "/mnt/data/cache"])
def test_refuses_to_live_on_a_persistent_volume(path, monkeypatch):
    """The cache is designed to be discarded; a store designed to be discarded
    must never be pointed at storage designed to be kept.  /data is Railway's
    volume mount."""
    monkeypatch.setenv("PYROORB_FIT_CACHE_DIR", path)
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)
    assert fitdisk._resolve_dir() is None
    st = fitdisk.stats()
    assert st["enabled"] is False and "persistent" in st["reason"]
    assert fitdisk.put("k", {"ok": True}) is False
    assert fitdisk.get("k") is None


def test_a_normal_scratch_dir_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROORB_FIT_CACHE_DIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)
    assert fitdisk._resolve_dir() is not None
    assert fitdisk.stats()["enabled"] is True


def test_an_unusable_directory_disables_rather_than_raises(tmp_path, monkeypatch):
    """A cache that can take the server down is a liability, not an optimisation."""
    blocker = tmp_path / "notadir"
    blocker.write_text("i am a file")
    monkeypatch.setenv("PYROORB_FIT_CACHE_DIR", str(blocker))
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)
    assert fitdisk._resolve_dir() is None
    assert fitdisk.put("k", {"ok": True}) is False       # no exception


# --- round trip -------------------------------------------------------------

def test_round_trip_carries_the_payload_and_the_designation():
    assert fitdisk.put("k1", {"ok": True, "v": 7}, desig="2026 AB123")
    rec = fitdisk.get("k1")
    assert rec["payload"] == {"ok": True, "v": 7}
    assert rec["desig"] == "2026 AB123"


def test_a_missing_key_is_a_miss():
    assert fitdisk.get("nope") is None


def test_an_unserialisable_payload_is_not_cached():
    assert fitdisk.put("k", {"ok": True, "bad": object()}) is False
    assert fitdisk.get("k") is None


def test_a_torn_file_is_deleted_and_reported_as_a_miss():
    """This filesystem tears large writes; a half-written payload must not
    poison a key for the life of the container."""
    fitdisk.put("k", {"ok": True})
    p = fitdisk._path(fitdisk._resolve_dir(), "k")
    with open(p, "r+", encoding="utf-8") as fh:
        blob = fh.read()
        fh.seek(0)
        fh.write(blob[: len(blob) // 2])
        fh.truncate()
    assert fitdisk.get("k") is None
    assert not os.path.exists(p), "the torn file should have been dropped"


def test_a_record_without_a_payload_is_a_miss():
    d = fitdisk._resolve_dir()
    with open(fitdisk._path(d, "weird"), "w", encoding="utf-8") as fh:
        json.dump({"something": "else"}, fh)
    assert fitdisk.get("weird") is None


def test_a_stale_entry_is_dropped(monkeypatch):
    """The engine revision is already in the key, so the TTL is about the
    ASTROMETRY going stale, not the code.

    Ages the record's `stored` field, NOT its mtime: mtime is the LRU's clock
    and every read touches it, so it can never measure how old an answer is.
    This test used to age mtime, which is exactly the confusion that let a
    popular object serve its first fit forever.
    """
    fitdisk.put("k", {"ok": True})
    p = fitdisk._path(fitdisk._resolve_dir(), "k")
    rec = json.load(open(p, encoding="utf-8"))
    rec["stored"] = time.time() - (fitdisk.TTL_SECS + 60)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    assert fitdisk.get("k") is None
    assert not os.path.exists(p)


# --- budget -----------------------------------------------------------------

def test_the_budget_evicts_oldest_first(monkeypatch):
    monkeypatch.setattr(fitdisk, "BUDGET_BYTES", 4000)
    payload = {"ok": True, "pad": "x" * 500}
    for i in range(20):
        fitdisk.put(f"k{i}", payload)
        time.sleep(0.01)                    # distinct mtimes for a stable order
    st = fitdisk.stats()
    assert st["bytes"] <= fitdisk.BUDGET_BYTES, st
    assert fitdisk.get("k19") is not None, "the newest entry should survive"
    assert fitdisk.get("k0") is None, "the oldest should have been evicted"


def test_a_nearly_full_disk_stops_writing(monkeypatch):
    """A full disk breaks logging, the ephemerides and the process; none of that
    is worth a cache hit."""
    monkeypatch.setattr(fitdisk, "_free_mb", lambda d: 1.0)
    monkeypatch.setattr(fitdisk, "MIN_FREE_MB", 512)
    assert fitdisk.put("k", {"ok": True}) is False
    assert fitdisk.get("k") is None


def test_clear_empties_the_cache():
    for i in range(3):
        fitdisk.put(f"k{i}", {"ok": True})
    fitdisk.clear()
    assert fitdisk.stats()["entries"] == 0


# --- wiring into the queue --------------------------------------------------

@pytest.fixture
def _clean_queue():
    with fq._LOCK:
        fq._JOBS.clear(); fq._BY_KEY.clear(); fq._WAITING.clear()
    yield
    with fq._LOCK:
        fq._JOBS.clear(); fq._BY_KEY.clear(); fq._WAITING.clear()


def test_a_finished_fit_is_written_through(_clean_queue):
    """The write lands SHORTLY AFTER the job is marked done, not before it.

    Deliberate: the payload is published and the waiters released first, so a
    slow disk never delays the people who asked for the fit -- a failed cache
    write is only a future miss.  So this polls rather than asserting
    immediately, which is also the honest description of the guarantee.
    """
    r = fq.submit("wt", lambda: {"ok": True, "v": 1}, desig="2026 AB123")
    assert fq.wait(r["job"], 30)
    rec = None
    for _ in range(100):
        rec = fitdisk.get("wt")
        if rec is not None:
            break
        time.sleep(0.05)
    assert rec is not None and rec["payload"]["v"] == 1
    assert rec["desig"] == "2026 AB123"


def test_a_failed_fit_is_not_cached(_clean_queue):
    """An error may be transient (a fetch timeout); caching it would keep
    serving a failure the next attempt would not have hit."""
    r = fq.submit("bad", lambda: {"ok": False, "error": "boom"})
    assert fq.wait(r["job"], 30)
    assert fitdisk.get("bad") is None


def test_a_disk_hit_never_runs_the_work(_clean_queue):
    """The point on a launch spike: the hundred-and-first viewer costs a file
    read, not a worker slot."""
    fitdisk.put("hot", {"ok": True, "v": 42}, desig="2026 AB123")
    ran = []
    out = fq.submit("hot", lambda: ran.append(1) or {"ok": True}, desig="2026 AB123")
    assert not ran, "a cached key must not enqueue work"
    assert out["done"] and out["cached"] == "disk"
    assert out["result"]["v"] == 42


def test_a_disk_hit_offers_a_refit_and_no_session(_clean_queue):
    """No live template exists for a disk hit, so the payload must not pretend
    otherwise -- it names what to re-fit instead."""
    fitdisk.put("hot", {"ok": True, "session": "s1-leak", "restore": "old"},
                desig="2026 AB123")
    out = fq.submit("hot", lambda: {"ok": True}, desig="2026 AB123")
    result = out["result"]
    assert "session" not in result and "restore" not in result
    assert result["refit"] == "2026 AB123"


def test_a_live_job_beats_the_disk(_clean_queue):
    """A key that is IN FLIGHT must coalesce onto the running fit, never be
    answered from a file left over from an earlier run of the same key.

    Order matters here: the live job has to exist first, because a submit that
    finds nothing live is entitled to answer from disk (that is the whole
    feature).  What must not happen is a stale file overtaking a fit that is
    already running for this exact key.
    """
    import threading
    release = threading.Event()
    first = fq.submit("k", lambda: (release.wait(30), {"ok": True, "v": "fresh"})[1],
                      desig="d")
    assert first["job"], "the first submit should have started a real fit"
    fitdisk.put("k", {"ok": True, "v": "stale"}, desig="d")   # a leftover file
    try:
        second = fq.submit("k", lambda: {"ok": True, "v": "other"}, desig="d")
        assert second["job"] == first["job"], "should have joined the live fit"
        assert second.get("cached") != "disk", "a live fit was overtaken by a file"
    finally:
        release.set()
    assert fq.wait(first["job"], 30)
    assert fq.status(first["job"])["result"]["v"] == "fresh"


# --- the disk tier must not become a one-way door -------------------------

def test_a_refit_can_bypass_the_disk_so_a_cached_object_can_go_interactive(_clean_queue):
    """THE path that made a shared link read-only forever.

    A disk hit carries a payload but no template, so it offers `refit` and no
    `restore`.  The client then re-runs the fit to get a session -- and that
    re-run, going through submit() again, was served the SAME disk hit.  The fit
    never re-ran, no session ever appeared, and the user was told "that fit has
    expired, re-run it" by a re-run that could not possibly succeed.  Once an
    object aged out of the in-memory table, every later visitor was read-only
    for the life of the cache entry.
    """
    fitdisk.put("k", {"ok": True, "v": "cached"}, desig="2026 AB123")
    ran = []

    def work():
        ran.append(1)
        return {"ok": True, "v": "fresh", "_retain": {"template": True}}

    viewer = fq.submit("k", work, desig="2026 AB123")
    assert viewer["cached"] == "disk" and not ran
    assert "restore" not in viewer["result"]

    refit = fq.submit("k", work, desig="2026 AB123", bypass_cache=True)
    assert refit.get("job"), "the bypass must start a REAL fit, not answer from disk"
    assert fq.wait(refit["job"], 30)
    result = fq.status(refit["job"])["result"]
    assert ran, "the bypass did not run the work"
    assert result.get("restore"), "the re-fit produced no template to restore from"


def test_the_bypass_still_uses_a_live_job(_clean_queue):
    """Only the DISK read is skipped.  A job still in memory carries its
    template, so a restore succeeds and re-fitting would be waste."""
    r = fq.submit("k", lambda: {"ok": True, "_retain": {"t": 1}}, desig="d")
    assert fq.wait(r["job"], 30)
    again = fq.submit("k", lambda: {"ok": True}, desig="d", bypass_cache=True)
    assert again.get("cached") == "memory", again
    assert again["result"].get("restore")


def test_the_ttl_ages_from_when_the_record_was_stored_not_last_read():
    """Reading must not postpone expiry.

    The LRU touches mtime on every read, so a TTL measured from mtime is reset
    by the act of serving the entry: an object read at least once per window
    would serve its FIRST fit forever and never pick up new astrometry -- and
    the objects read that often are precisely the popular ones.
    """
    fitdisk.put("k", {"ok": True}, desig="d")
    p = fitdisk._path(fitdisk._resolve_dir(), "k")
    rec = json.load(open(p, encoding="utf-8"))
    rec["stored"] = time.time() - (fitdisk.TTL_SECS + 60)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    os.utime(p, None)                       # as a fresh read would leave it
    assert fitdisk.get("k") is None, "an mtime touch postponed the TTL"


def test_a_record_with_no_stored_time_is_expired_not_immortal():
    d = fitdisk._resolve_dir()
    with open(fitdisk._path(d, "old"), "w", encoding="utf-8") as fh:
        json.dump({"payload": {"ok": True}}, fh)      # no `stored`
    assert fitdisk.get("old") is None
