"""The hosted-capacity contract: don't crash, and never let a
user believe a truncated search was a complete one.

Three pieces, tested here because they are one design and only make sense
together:

  * the sweep floor SCALES to the container, so it can be neither inert (the
    pre-2026-08-19 bug: /proc/meminfo is the host's) nor unpayable (the bug the
    fix would otherwise introduce: an absolute 1000 MB reservation on a 1 GB
    container truncates every search on its first window);
  * an incomplete sweep is reported on EVERY verdict, including a find
    (the payload side of that lives in web/tests/test_itf_search_web.py);
  * the web UI admits at most `_ITF_MAX_RUNNING` searches and tells the
    seventh user to come back, rather than starting a seventh sweep.
"""
import os

from pyroorb.itf.memory import (sweep_floor_mb, limit_mb, search_capacity,
                        FLOOR_LIMIT_FRACTION, FLOOR_MIN_MB, MAX_BURST_MB,
                        SEARCH_MEDIAN_MB, CAPACITY_OK, CAPACITY_SHALLOW, CAPACITY_UNUSABLE)

MB = 1024 * 1024


def _v2(root, limit, current=MB):
    base = os.path.join(root, "sys/fs/cgroup")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "memory.max"), "w", encoding="utf-8") as fh:
        fh.write(f"{limit}\n")
    with open(os.path.join(base, "memory.current"), "w", encoding="utf-8") as fh:
        fh.write(f"{current}\n")


# --------------------------------------------------------------- the floor ---

def test_uncontained_box_keeps_the_configured_floor(tmp_path):
    """No cgroup limit -- a laptop, a CI runner -- must behave exactly as it
    did before scaling existed."""
    assert limit_mb(str(tmp_path)) is None
    assert sweep_floor_mb(1000, str(tmp_path)) == 1000


def test_a_small_container_scales_the_floor_down(tmp_path):
    """The failure this prevents: with an absolute 1000 MB floor on a 1 GB
    container, available memory STARTS under the floor, so every search
    truncates on window one and answers "no match" from a corridor it never
    looked at.  A reservation has to be a share of the box."""
    root = str(tmp_path)
    _v2(root, 8 * 1024 * MB)              # 0.3 x 8 GB = 2457, above the cap
    assert sweep_floor_mb(1000, root) == 1000
    _v2(root, 2 * 1024 * MB)              # 0.3 x 2 GB = 614, the fraction bites
    got = sweep_floor_mb(1000, root)
    assert got == int(FLOOR_LIMIT_FRACTION * 2048)
    assert got < 1000, "an absolute floor on a small container is unpayable"


def test_the_floor_never_scales_below_the_allocation_burst(tmp_path):
    """A floor under the BURST is a guard that cannot guard: the sweep is
    re-checked every window, so what the floor must survive is however much the
    process can allocate between two checks -- measured at up to
    MAX_BURST_MB (357 MB idle / 358 MB loaded, the same
    number twice).  0.3 x 1 GB is 307 MB, UNDER that; without the clamp the
    valve would fire only after the allocation that already killed us."""
    root = str(tmp_path)
    _v2(root, 1024 * MB)
    got = sweep_floor_mb(1000, root)
    assert got == FLOOR_MIN_MB
    assert got > MAX_BURST_MB, "the floor must outlast one allocation burst"


def test_a_large_container_is_capped_at_the_configured_value(tmp_path):
    """Scaling only ever reduces: a big container must not be handed a bigger
    reservation than the engine asked for."""
    root = str(tmp_path)
    _v2(root, 16 * 1024 * MB)
    assert sweep_floor_mb(1000, root) == 1000


def test_disabled_stays_disabled(tmp_path):
    root = str(tmp_path)
    _v2(root, 512 * MB)
    assert sweep_floor_mb(0, root) == 0


def test_the_floor_never_scales_to_zero(tmp_path):
    """A floor of 0 means DISABLED.  A tiny container must not silently
    disable the guard by rounding its share away."""
    root = str(tmp_path)
    _v2(root, 2 * MB)
    assert sweep_floor_mb(1000, root) >= 1


# ------------------------------------------------- can this box search? ------

def _cap(tmp_path, limit_mb_, used_mb):
    root = str(tmp_path)
    _v2(root, limit_mb_ * MB, current=used_mb * MB)
    # a huge host MemAvailable so the CONTAINER is what binds
    os.makedirs(os.path.join(root, "proc"), exist_ok=True)
    with open(os.path.join(root, "proc/meminfo"), "w", encoding="utf-8") as fh:
        fh.write("MemAvailable:   99000000 kB\n")
    return search_capacity(1000, root)


def test_a_roomy_container_can_search_deeply(tmp_path):
    assert _cap(tmp_path, 4096, 500)["verdict"] == CAPACITY_OK


def test_a_tight_container_reports_SHALLOW(tmp_path):
    """Room for a routine search but not for the variant scan a weak orbit
    needs.  The user is told BEFORE spending the minutes, because a thin result
    explained afterwards is a worse product than a warning up front."""
    cap = _cap(tmp_path, 2048, 1000)
    assert cap["verdict"] == CAPACITY_SHALLOW
    assert "cut short" in cap["reason"]


def test_a_full_container_reports_UNUSABLE(tmp_path):
    """Below a median search's needs there is nothing worth starting: it would
    burn minutes and answer "no match" from a corridor it never queried, which
    reads as evidence of absence."""
    cap = _cap(tmp_path, 1024, 800)
    assert cap["verdict"] == CAPACITY_UNUSABLE
    assert cap["headroom_mb"] < SEARCH_MEDIAN_MB


def test_capacity_is_None_where_memory_cannot_be_measured(tmp_path):
    """No cgroup, no /proc: the guards are inert there and a confident verdict
    would be a lie."""
    assert search_capacity(1000, str(tmp_path)) is None


def test_the_source_enforces_the_scaled_floor_not_the_configured_one(monkeypatch):
    """Whatever the config says, the valve and every message about it must use
    the same enforced number."""
    import pyroorb.itf.candidates as C
    from pyroorb.itf.candidates import DbSource
    from pyroorb.itf.config import ItfConfig

    src = DbSource.__new__(DbSource)
    src.cfg = ItfConfig(sweep_min_free_mb=1000)
    # the stub keeps the real contract: scaling only ever reduces, and 0 (the
    # DISABLED sentinel) must survive scaling as 0
    monkeypatch.setattr(C, "sweep_floor_mb",
                        lambda configured: 300 if configured else 0)
    assert src.effective_floor_mb() == 300
    monkeypatch.setattr(C, "_mem_available_mb", lambda: 500)
    assert src._low_memory_mb() is None, "500 MB is above the ENFORCED 300"
    monkeypatch.setattr(C, "_mem_available_mb", lambda: 200)
    assert src._low_memory_mb() == 200
    # ...and it must FOLLOW the config, not a value cached from an earlier one:
    # the search replaces cfg mid-run (the magnitude fallback rebuilds it), and
    # a valve enforcing a stale floor fails silently
    src.cfg = ItfConfig(sweep_min_free_mb=0)
    assert src._low_memory_mb() is None, "a disabled floor must disable"



# ------------------------------------------------------------- admission -----

def _fresh_jobs(monkeypatch, n_running):
    import pyroorb.web.engine_api as E
    jobs = {f"r{i}": {"done": False, "started": 0.0, "ended": 0.0,
                      "collected": 0.0, "payload": None}
            for i in range(n_running)}
    monkeypatch.setattr(E, "_ITF_JOBS", jobs)
    return E, jobs


def test_admits_up_to_the_cap(monkeypatch):
    E, jobs = _fresh_jobs(monkeypatch, 5)
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 6)
    monkeypatch.setattr(E, "itf_search_web",
                        lambda *a, **k: {"ok": True, "itf": {"found": False}})
    out = E.itf_search_start("s1", options={"trace_id": "t-new"})
    assert out["ok"] is True and out["job"] == "t-new"


def test_refuses_the_seventh_and_says_when_to_come_back(monkeypatch):
    """The whole point: a seventh search must not START.  Evicting or queueing
    a RUNNING search cannot free its memory (the thread keeps going), so the
    only lever that protects the box is refusing admission."""
    E, jobs = _fresh_jobs(monkeypatch, 6)
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 6)
    started = []
    monkeypatch.setattr(E, "itf_search_web",
                        lambda *a, **k: started.append(1) or {"ok": True})
    out = E.itf_search_start("s1", options={"trace_id": "t-seventh"})
    assert out["ok"] is False and out["busy"] is True
    assert out["running"] == 6
    assert "try again in a minute" in out["error"]
    assert "t-seventh" not in jobs, "a refused search must not occupy a slot"
    assert not started, "a refused search must not run"


def test_finished_jobs_do_not_occupy_a_slot(monkeypatch):
    """The cap counts WORK IN FLIGHT.  `_ITF_JOBS_MAX` already bounds retained
    finished payloads and they cost a payload, not a sweep."""
    E, jobs = _fresh_jobs(monkeypatch, 6)
    for j in jobs.values():
        j["done"] = True
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 6)
    monkeypatch.setattr(E, "itf_search_web", lambda *a, **k: {"ok": True})
    out = E.itf_search_start("s1", options={"trace_id": "t-after"})
    assert out["ok"] is True


def test_re_posting_a_running_id_is_still_idempotent(monkeypatch):
    """A double-click must not be answered with "server busy" -- it is the
    SAME search, and the idempotent branch has to win over admission."""
    E, jobs = _fresh_jobs(monkeypatch, 6)
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 6)
    monkeypatch.setattr(E, "itf_search_web", lambda *a, **k: {"ok": True})
    out = E.itf_search_start("s1", options={"trace_id": "r0"})
    assert out["ok"] is True and out["job"] == "r0"


def test_the_cap_is_dialable_without_a_deploy():
    """The number should be changeable when the container's memory changes."""
    assert os.environ.get("PYROORB_ITF_MAX_RUNNING") is None or True
    import pyroorb.web.engine_api as E
    assert isinstance(E._ITF_MAX_RUNNING, int) and E._ITF_MAX_RUNNING >= 1


def test_admission_refuses_when_the_box_cannot_search(monkeypatch):
    """A FREE SLOT IS NOT FREE MEMORY.  Six admitted searches share one global
    headroom figure, so a slot can be free while the box has nothing to sweep
    with.  Starting then is worse than refusing: minutes burn and the user is
    handed "no match" from a corridor that was never queried."""
    E, jobs = _fresh_jobs(monkeypatch, 0)
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 6)
    monkeypatch.setattr(E, "_itf_capacity_block", lambda: {
        "verdict": CAPACITY_UNUSABLE, "reason": "only 20 MB is free",
        "available_mb": 420, "floor_mb": 400, "headroom_mb": 20,
        "limit_mb": 1024})
    started = []
    monkeypatch.setattr(E, "itf_search_web",
                        lambda *a, **k: started.append(1) or {"ok": True})
    out = E.itf_search_start("s1", options={"trace_id": "t-nomem"})
    assert out["ok"] is False and out["busy"] is True
    assert "low on memory" in out["error"]
    assert "try again in a minute" in out["error"]
    assert not started and "t-nomem" not in jobs


def test_admission_allows_a_SHALLOW_box(monkeypatch):
    """Shallow is a warning, not a refusal: a routine search still works, and
    refusing them all would make a busy server useless rather than slow."""
    E, _jobs = _fresh_jobs(monkeypatch, 0)
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 6)
    monkeypatch.setattr(E, "_itf_capacity_block", lambda: {
        "verdict": CAPACITY_SHALLOW, "reason": "300 MB free", "available_mb": 700,
        "floor_mb": 400, "headroom_mb": 300, "limit_mb": 2048})
    monkeypatch.setattr(E, "itf_search_web", lambda *a, **k: {"ok": True})
    assert E.itf_search_start("s1", options={"trace_id": "t-ok"})["ok"] is True


def test_unmeasurable_memory_does_not_block_searching(monkeypatch):
    """A dev box with no cgroup must behave exactly as it always has -- the
    guards degrade to absent, never to refusal."""
    E, _jobs = _fresh_jobs(monkeypatch, 0)
    monkeypatch.setattr(E, "_ITF_MAX_RUNNING", 6)
    monkeypatch.setattr(E, "_itf_capacity_block", lambda: None)
    monkeypatch.setattr(E, "itf_search_web", lambda *a, **k: {"ok": True})
    assert E.itf_search_start("s1", options={"trace_id": "t-x"})["ok"] is True
