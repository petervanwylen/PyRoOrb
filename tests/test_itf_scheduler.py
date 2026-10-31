"""Offline unit tests for the ITF maintenance scheduler (src/pyroorb/itf/scheduler.py).

No network / no database: a fake connection answers the advisory-lock queries,
and the three index jobs + the state-table helpers are stubbed so we can assert
the *orchestration* -- single-flight, refresh-always, due-gated reconcile/sweep,
lock-deferral, and failure isolation.
"""
import os

from pyroorb.itf import scheduler as S
from pyroorb.itf import index as ti

# run_tick logs through `log` (default: print). Several tests here INDUCE job failures on
# purpose, and the scheduler's failure path logs the exception plus a full traceback -- so
# with the default the suite printed real-looking "FAILED ... SSL SYSCALL error" tracebacks
# from a PASSING run, which is indistinguishable from a broken test until you read the
# source. Collect the log instead; a test that cares about a message can assert on _LOG.
_LOG = []


# --- fake psycopg2 connection ------------------------------------------------
class _Cur:
    def __init__(self, lock_ok): self.lock_ok = lock_ok
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self._sql = sql
    def fetchone(self):
        # only the advisory-lock/unlock SELECTs reach a real fetch in these
        # tests (state-table helpers are stubbed out).
        return (self.lock_ok,)


class _Conn:
    def __init__(self, lock_ok=True):
        self.lock_ok = lock_ok
        self.closed = False
        self.rolled = 0
    def cursor(self): return _Cur(self.lock_ok)
    def commit(self): pass
    def rollback(self): self.rolled += 1
    def close(self): self.closed = True


class _Patch:
    """Swap module attributes for the duration of a test, then restore."""
    def __init__(self, **kw): self.kw = kw; self.old = {}
    def __enter__(self):
        for name, val in self.kw.items():
            mod = ti if name.startswith("ti.") else S
            key = name.split(".", 1)[-1]
            self.old[(mod, key)] = getattr(mod, key)
            setattr(mod, key, val)
        return self
    def __exit__(self, *a):
        for (mod, key), val in self.old.items():
            setattr(mod, key, val)
        return False


def _harness(conn, due):
    """Common stubs: fake connection, no-op state table, controllable dueness,
    recording job stubs. Returns (calls list, recorded list, patch kwargs)."""
    calls, recorded = [], []
    kw = dict(
        connect_index=lambda cfg: conn,
        _ensure_state=lambda conn, s: None,
        _due=lambda conn, s, job, interval: due.get(job, False),
        _record=lambda conn, s, job, status, err, dt: recorded.append((job, status)),
    )
    kw["ti.refresh_incremental"] = lambda c, i: calls.append("refresh")
    kw["ti.reconcile_membership"] = lambda c, i, log=None: calls.append("reconcile")
    kw["ti.sweep_validity"] = lambda c, i, log=None: calls.append("sweep")
    return calls, recorded, kw


# --- tests -------------------------------------------------------------------
def test_maintain_enabled_truthy():
    saved = os.environ.get("ITF_MAINTAIN_TURNED_ON")
    try:
        for v in ("True", "true", "1", "yes", "on", " TRUE "):
            os.environ["ITF_MAINTAIN_TURNED_ON"] = v
            assert S.maintain_enabled() is True, v
        for v in ("False", "false", "0", "no", "off", "", "maybe"):
            os.environ["ITF_MAINTAIN_TURNED_ON"] = v
            assert S.maintain_enabled() is False, v
    finally:
        if saved is None:
            os.environ.pop("ITF_MAINTAIN_TURNED_ON", None)
        else:
            os.environ["ITF_MAINTAIN_TURNED_ON"] = saved


def test_tick_refresh_only_when_nothing_due():
    conn = _Conn(lock_ok=True)
    calls, recorded, kw = _harness(conn, due={})       # neither reconcile nor sweep due
    with _Patch(**kw):
        S.run_tick(cfg=object(), index_cfg=ti.IndexConfig(), log=_LOG.append)
    assert calls == ["refresh"], calls
    assert recorded == [("refresh", "ok")], recorded
    assert conn.closed                                  # connection always released


def test_tick_runs_due_jobs_in_order():
    conn = _Conn(lock_ok=True)
    calls, recorded, kw = _harness(conn, due={"reconcile": True, "sweep": True})
    with _Patch(**kw):
        S.run_tick(cfg=object(), index_cfg=ti.IndexConfig(), log=_LOG.append)
    # refresh first (always), then reconcile, then sweep -- one at a time
    assert calls == ["refresh", "reconcile", "sweep"], calls
    assert [j for j, _ in recorded] == ["refresh", "reconcile", "sweep"]


def test_tick_defers_when_lock_held():
    conn = _Conn(lock_ok=False)                         # pg_try_advisory_lock -> false
    calls, recorded, kw = _harness(conn, due={"reconcile": True, "sweep": True})
    with _Patch(**kw):
        S.run_tick(cfg=object(), index_cfg=ti.IndexConfig(), log=_LOG.append)
    assert calls == [], calls                           # nothing runs under contention
    assert recorded == []
    assert conn.closed


def test_tick_isolates_job_failure():
    conn = _Conn(lock_ok=True)
    calls, recorded, kw = _harness(conn, due={"reconcile": True, "sweep": True})

    def boom(c, i, log=None):
        calls.append("reconcile")
        raise RuntimeError("db blew up")
    kw["ti.reconcile_membership"] = boom
    with _Patch(**kw):
        S.run_tick(cfg=object(), index_cfg=ti.IndexConfig(), log=_LOG.append)   # must NOT raise
    # refresh ok, reconcile failed (recorded error), sweep still got its turn
    assert calls == ["refresh", "reconcile", "sweep"], calls
    assert ("reconcile", "error") in recorded
    assert ("sweep", "ok") in recorded
    assert conn.rolled >= 1                              # aborted txn cleared


def test_record_resilient_falls_back_to_fresh_conn():
    # Regression: a job that DROPS the connection must still persist its run-state
    # (else the daily/weekly gate never records it -> the job retries every tick,
    # a storm against the shared server). _record on the dead conn raises; the
    # helper must open a fresh connection and record there without propagating.
    dead = _Conn()
    def _dead_cursor():
        raise RuntimeError("connection already closed")
    dead.cursor = _dead_cursor                      # _record(dead, ...) blows up
    fresh = _Conn()
    with _Patch(connect_index=lambda cfg: fresh):
        S._record_resilient(object(), dead, ti.IndexConfig().schema,
                            "reconcile", "error", "boom", 1.0, lambda *a: None)
    assert fresh.closed is True                     # fresh conn used + closed


def test_conn_alive_reflects_closed_flag():
    c = _Conn()
    assert S._conn_alive(c) is True
    c.close()
    assert S._conn_alive(c) is False


def test_tick_skips_rest_when_connection_dies_midjob():
    # If refresh kills the connection, reconcile/sweep _due must NOT run on the
    # dead conn (no 'connection already closed' every tick); the failure is still
    # recorded (via a fresh conn) so the gate holds.
    conn = _Conn(lock_ok=True)
    calls, recorded, kw = _harness(conn, due={"reconcile": True, "sweep": True})
    def refresh_kills_conn(c, i):
        calls.append("refresh")
        conn.closed = True                          # simulate the drop
        raise RuntimeError("SSL SYSCALL error: EOF detected")
    kw["ti.refresh_incremental"] = refresh_kills_conn
    with _Patch(**kw):
        S.run_tick(cfg=object(), index_cfg=ti.IndexConfig(), log=_LOG.append)   # must not raise
    assert calls == ["refresh"], calls              # reconcile/sweep NOT attempted
    assert ("refresh", "error") in recorded


def test_event_log_policy():
    # every failure logged; daily/weekly successes logged; per-minute refresh
    # SUCCESS deliberately not logged (would bury the signal).
    assert S._worth_logging("refresh", "ok") is False
    assert S._worth_logging("refresh", "error") is True
    assert S._worth_logging("reconcile", "ok") is True
    assert S._worth_logging("sweep", "ok") is True
    assert S._worth_logging("reconcile", "error") is True


def test_run_job_emits_events_per_policy():
    conn = _Conn(lock_ok=True)
    events = []
    with _Patch(_record=lambda *a: None,
                _log_event=lambda c, s, job, status, dur, detail: events.append((job, status))):
        sch = ti.IndexConfig().schema
        S._run_job(object(), conn, sch, "refresh", lambda: 5, lambda *a: None)
        S._run_job(object(), conn, sch, "reconcile", lambda: 12, lambda *a: None)
        def boom():
            raise RuntimeError("x")
        S._run_job(object(), conn, sch, "refresh", boom, lambda *a: None)
    assert ("refresh", "ok") not in events       # per-minute success NOT logged
    assert ("reconcile", "ok") in events         # daily success logged
    assert ("refresh", "error") in events        # any failure logged
