"""The propagator sweep telemetry: what a thread is integrating, for how long, and
the slowest sweeps lately -- the record a wedged ITF search left NO trace of.

Both propagators register every sweep (rebound here, ASSIST when its ephemeris
is on the box), a sweep that raises is still unregistered, nesting restores the
outer sweep, and the report derives elements from the recorded state without
importing anything.  The rebound hang-guard itself lives in test_rebound_min_dt.
"""
import threading
import time

import numpy as np
import pytest

import pyroorb.forward_model as fm
import pyroorb.sweepwatch as sw


@pytest.fixture(autouse=True)
def _clean():
    sw._reset_for_tests()
    yield
    sw._reset_for_tests()


def _circular_state(r=2.5):
    return np.array([r, 0.0, 0.0, 0.0, np.sqrt(fm.MU_SUN / r), 0.0])


def test_begin_mark_end_bookkeeping():
    rec = sw.begin("rebound", 2460000.5, _circular_state(), np.array([2460001.5, 2460010.5]))
    assert sw.health()["inflight"] == 1
    sw.mark(rec, 1)
    snap = sw.snapshot()
    row = snap["inflight"][0]
    assert row["backend"] == "rebound" and row["n"] == 2 and row["at"] == 1
    assert row["target_days"] == 10.0 and row["span_days"] == [1.0, 10.0]
    assert row["elements"]["a_au"] == pytest.approx(2.5, rel=1e-6)
    assert row["elements"]["e"] == pytest.approx(0.0, abs=1e-6)
    assert row["wedged"] is False
    sw.end(rec)
    assert sw.health()["inflight"] == 0 and sw.snapshot()["inflight"] == []


def test_nesting_restores_the_outer_sweep_and_a_raise_still_unregisters():
    outer = sw.begin("assist", 2460000.5, _circular_state(), np.array([2460001.5]))
    inner = sw.begin("rebound", 2460000.5, _circular_state(3.0), np.array([2460002.5]))
    assert sw.snapshot()["inflight"][0]["backend"] in ("assist", "rebound")
    assert sw.health()["inflight"] == 1            # one thread, one in-flight slot
    try:
        raise RuntimeError("the propagator raised mid-sweep")
    except RuntimeError:
        sw.end(inner)
    assert sw.snapshot()["inflight"][0]["backend"] == "assist"
    sw.end(outer)
    assert sw.health()["inflight"] == 0


def test_a_slow_sweep_is_filed_with_its_step_count(monkeypatch):
    monkeypatch.setattr(sw, "SLOW_SWEEP_SEC", 0.0)

    class _Sim:
        steps_done = 12345

    rec = sw.begin("rebound", 2460000.5, _circular_state(), np.array([2460001.5]), label="iod")
    sw.mark(rec, 0)
    sw.end(rec, sim=_Sim())
    slow = sw.snapshot()["slow"]
    assert len(slow) == 1
    assert slow[0]["steps_done"] == 12345 and slow[0]["label"] == "iod"
    assert slow[0]["state"] == pytest.approx(list(_circular_state()))
    assert sw.health()["slow_kept"] == 1


def test_a_fast_sweep_is_not_filed():
    rec = sw.begin("rebound", 2460000.5, _circular_state(), np.array([2460001.5]))
    sw.end(rec)
    assert sw.snapshot()["slow"] == []


def test_another_thread_s_sweep_is_visible_with_its_age_and_state():
    started = threading.Event()
    release = threading.Event()

    def worker():
        rec = sw.begin("rebound", 2460000.5, _circular_state(), np.array([2460001.5]))
        sw.mark(rec, 0)
        started.set()
        release.wait(5)
        sw.end(rec)

    t = threading.Thread(target=worker, name="itf-worker-test")
    t.start()
    assert started.wait(5)
    time.sleep(0.05)
    snap = sw.snapshot(names={t.ident: t.name})
    assert snap["inflight"][0]["thread"] == "itf-worker-test"
    assert snap["inflight"][0]["elapsed_s"] >= 0.0
    assert snap["inflight"][0]["state"] is not None
    assert sw.health()["oldest_s"] >= 0.0
    release.set()
    t.join(5)
    assert sw.health()["inflight"] == 0


def test_the_wedged_verdict_is_by_age(monkeypatch):
    monkeypatch.setattr(sw, "WEDGED_SEC", 0.0)
    rec = sw.begin("rebound", 2460000.5, _circular_state(), np.array([2460001.5]))
    assert sw.snapshot()["wedged"] == 1 and sw.health()["wedged"] == 1
    sw.end(rec)


def test_the_rebound_propagator_registers_its_sweeps(monkeypatch):
    monkeypatch.setattr(sw, "SLOW_SWEEP_SEC", 0.0)
    t0 = 2460000.5
    base = fm.build_base_sim(t0)
    times = t0 + np.array([1.0, 10.0, -5.0])
    out = fm.predict_radec_batch(base, t0, _circular_state(), times, [np.zeros(3)] * 3)
    assert np.isfinite(out).all()
    slow = sw.snapshot()["slow"]
    assert [r["backend"] for r in slow] == ["rebound", "rebound"]   # forward leg, backward leg
    assert all(r["steps_done"] is not None and r["steps_done"] > 0 for r in slow)
    assert sw.health()["inflight"] == 0


def test_the_assist_propagator_registers_its_sweeps(monkeypatch):
    fa = pytest.importorskip("force_assist")
    try:
        ephem = fa.get_ephem()
    except Exception as ex:                       # no DE440 on this box
        pytest.skip(f"ASSIST ephemeris unavailable: {ex}")
    monkeypatch.setattr(sw, "SLOW_SWEEP_SEC", 0.0)
    t0 = 2460000.5
    times = t0 + np.array([1.0, 10.0])
    out = fa.predict_radec_batch(fa.build_base_sim(t0), t0, _circular_state(), times,
                                 [np.zeros(3)] * 2)
    assert np.isfinite(out).all()
    slow = sw.snapshot()["slow"]
    assert slow and slow[0]["backend"] == "assist" and slow[0]["steps_done"] > 0
    assert sw.health()["inflight"] == 0
    del ephem


def test_a_broken_state_does_not_break_the_report():
    rec = sw.begin("rebound", 2460000.5, object(), None)
    row = sw.snapshot()["inflight"][0]
    assert row["state"] is None and row["elements"] is None and row["n"] is None
    sw.end(rec)


def test_a_numpy_progress_index_still_reports_as_json(monkeypatch):
    # The propagators mark with `i` drawn from a NumPy index group, so rec[AT]
    # is an np.int64; copied raw into `at`, it made /api/memory unencodable
    # (live 502s, 2026-09-27) once any slow sweep had been filed.
    import json
    monkeypatch.setattr(sw, "SLOW_SWEEP_SEC", 0.0)
    rec = sw.begin("assist", 2460000.5, _circular_state(), np.array([2460001.5, 2460002.5]))
    for i in np.arange(2):
        sw.mark(rec, i)
    json.dumps(sw.snapshot())                   # in flight
    sw.end(rec)
    snap = sw.snapshot()
    json.dumps(snap)                            # filed in the slow ring
    assert snap["slow"][0]["at"] == 1 and type(snap["slow"][0]["at"]) is int
