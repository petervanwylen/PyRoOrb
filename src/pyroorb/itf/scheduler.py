"""Always-on maintenance scheduler for the ``itf.itf_tracklets`` index.

Runs the three upkeep jobs (:mod:`pyroorb.itf.index`) on one clock, single-flight,
gated by the ``ITF_MAINTAIN_TURNED_ON`` environment variable (**off** by
default):

    refresh_incremental   every ~60 s
    reconcile_membership   daily   (>= 24 h since it last ran)
    sweep_validity         weekly  (>= 7 d  since it last ran)

**One job at a time.**  Everything runs sequentially on a single thread, so a
long reconcile or sweep naturally *defers* the next refresh until it finishes.
As a cross-process backstop -- two web replicas each starting a scheduler, or an
operator running ``python -m pyroorb.itf.maintain reconcile`` by hand -- every tick
first grabs a Postgres **session advisory lock** on the itf-index connection; a
tick that cannot get the lock does nothing and retries next period.  (A session
advisory lock survives ``COMMIT``/``ROLLBACK`` and is released only by
``pg_advisory_unlock`` or when the connection closes -- exactly what we want
across the internal commits the jobs make.)

Last-run instants live in ``itf.itf_maintain_runs`` so a redeploy/restart does
not re-fire the weekly sweep.  Intended to run as a daemon thread inside the web
service (``src/pyroorb/web/server.py``) -- set ``ITF_MAINTAIN_TURNED_ON=True`` on Railway
and redeploy -- but is equally runnable standalone: ``python -m pyroorb.itf.maintain
scheduler`` (loop) or ``python -m pyroorb.itf.maintain tick`` (one shot).
"""

import time
import traceback
from datetime import timedelta

from .config import ItfConfig
from .dbio import connect_index
from . import index as ti

# Arbitrary-but-stable 64-bit key naming the whole maintenance critical section
# on the shared server (any concurrent holder means a maintenance run is live).
_LOCK_KEY = 728_314_159

# The tiny state table (one row per job) recording when each last ran.  Lives in
# the itf schema alongside the index it maintains.  Drives the due-gate.
_STATE_TABLE = "itf_maintain_runs"

# Append-only event log for monitoring.  Policy (see _worth_logging): record
# every FAILURE (any job) and every daily/weekly SUCCESS (reconcile, sweep) -- but
# NOT the every-minute refresh's successes (pure noise).  A window into "is it
# actually working?" that the one-row state table (overwritten every 60s) can't
# give; queryable directly or via `python -m pyroorb.itf.maintain events`.
_EVENTS_TABLE = "itf_maintain_events"

DAILY = timedelta(hours=24)
WEEKLY = timedelta(days=7)


def maintain_enabled():
    """True iff ``ITF_MAINTAIN_TURNED_ON`` is set truthy (exported variable
    first, gitignored repo-root ``.env`` as fallback -- see
    :func:`itf.dbio.flag_enabled`)."""
    from .dbio import flag_enabled
    return flag_enabled("ITF_MAINTAIN_TURNED_ON")


# ---------------------------------------------------------------------------
# DB helpers (state table + advisory lock)
# ---------------------------------------------------------------------------
def state_qualified(s):
    return f"{s.target_schema}.{_STATE_TABLE}"


def events_qualified(s):
    return f"{s.target_schema}.{_EVENTS_TABLE}"


def _ensure_state(conn, s):
    """Create the per-job last-run table and the append-only event log if absent
    (called under the lock, so no concurrent-create race)."""
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {state_qualified(s)} (
                job         text PRIMARY KEY,
                last_run    timestamptz,
                last_status text,
                last_error  text,
                duration_s  double precision,
                updated_at  timestamptz DEFAULT now()
            )""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {events_qualified(s)} (
                id          bigserial PRIMARY KEY,
                ts          timestamptz DEFAULT now(),
                job         text NOT NULL,
                status      text NOT NULL,          -- 'ok' | 'error'
                duration_s  double precision,
                detail      text                    -- error text, or a success summary
            )""")
        cur.execute(f"""CREATE INDEX IF NOT EXISTS itf_maintain_events_ts
                        ON {events_qualified(s)} (ts DESC)""")
    conn.commit()


def _try_lock(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,))
        return bool(cur.fetchone()[0])


def _unlock(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    except Exception:
        pass                     # connection close releases it regardless


def _due(conn, s, job, interval):
    """True iff ``job`` never ran or last ran at least ``interval`` ago (compared
    on the DB clock, so the scheduler host's clock is irrelevant)."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT now(), (SELECT last_run FROM {state_qualified(s)} "
            f"WHERE job=%s)", (job,))
        now, last = cur.fetchone()
    return last is None or (now - last) >= interval


def _record(conn, s, job, status, error, duration_s):
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {state_qualified(s)}
                (job, last_run, last_status, last_error, duration_s, updated_at)
            VALUES (%s, now(), %s, %s, %s, now())
            ON CONFLICT (job) DO UPDATE SET
                last_run    = EXCLUDED.last_run,
                last_status = EXCLUDED.last_status,
                last_error  = EXCLUDED.last_error,
                duration_s  = EXCLUDED.duration_s,
                updated_at  = now()""",
            (job, status, error, duration_s))
    conn.commit()


def _conn_alive(conn):
    """False once psycopg2 has marked the connection closed (it sets ``closed``
    nonzero after a drop).  A job that killed the connection must not have the
    rest of the tick run on the corpse."""
    return getattr(conn, "closed", 1) == 0


def _persist(cfg, conn, log, what, write):
    """Run ``write(connection)`` on the live connection; on ANY failure -- e.g.
    the job that just ran dropped the connection -- retry once on a FRESH
    connection.  Never raises.

    Persisting run-state and events is exactly what gives the due-gate and the
    monitoring log their memory, so the write must survive the very drops we most
    want to record (if a failed reconcile couldn't record, it would look un-run
    and retry every tick -- a storm against the shared server)."""
    try:
        write(conn)
        return
    except Exception as e:
        # keep the first failure's diagnostic -- if the fresh-connection retry
        # succeeds, this line is the only trace of what actually went wrong
        log(f"itf-maintain: persist {what} failed on the live connection "
            f"({e!r}); retrying on a fresh one")
    try:
        c2 = connect_index(cfg)
        try:
            write(c2)
        finally:
            c2.close()
    except Exception as e:
        log(f"itf-maintain: WARNING could not persist {what}: {e!r}")


def _record_resilient(cfg, conn, s, job, status, error, duration_s, log):
    """Upsert the one-row-per-job run-state (drives the due-gate + failure
    back-off), resilient to a dropped connection -- see :func:`_persist`."""
    _persist(cfg, conn, log, f"{job} run-state",
             lambda c: _record(c, s, job, status, error, duration_s))


def _log_event(conn, s, job, status, duration_s, detail):
    detail = None if detail is None else str(detail)[:2000]
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {events_qualified(s)} (job, status, duration_s, detail) "
            f"VALUES (%s, %s, %s, %s)", (job, status, duration_s, detail))
    conn.commit()


def _worth_logging(job, status):
    """Event-log policy: record every failure, and every daily/weekly SUCCESS --
    but NOT the every-minute refresh's successes (they'd bury the signal)."""
    return status != "ok" or job in ("reconcile", "sweep")


def _persist_event(cfg, conn, s, job, status, duration_s, detail, log):
    """Append an event to the monitoring log when policy says it's worth it,
    resilient to a dropped connection (a failure is the case we MOST want logged
    and also the case where the connection may be dead)."""
    if _worth_logging(job, status):
        _persist(cfg, conn, log, f"{job} event",
                 lambda c: _log_event(c, s, job, status, duration_s, detail))


def _run_job(cfg, conn, s, job, fn, log):
    """Run one maintenance job, timing it and recording the outcome.  A failure
    is logged and recorded but never propagates -- the scheduler must not die and
    the other jobs must still get their turn.

    On failure we still stamp ``last_run`` (back off a full interval rather than
    hammer the shared server every tick); the jobs are idempotent, so a skipped
    cycle is caught by the next one.  The stamp is written through
    :func:`_record_resilient`, which falls back to a fresh connection when the
    job itself dropped this one -- otherwise the back-off would not persist and
    the job would retry every tick.  Separately, an event is appended to the
    monitoring log per :func:`_worth_logging` (every failure + daily/weekly
    successes; never the every-minute refresh's successes)."""
    t0 = time.monotonic()
    try:
        result = fn()
        dt = time.monotonic() - t0
        _record_resilient(cfg, conn, s, job, "ok", None, dt, log)
        _persist_event(cfg, conn, s, job, "ok", dt,
                       None if result is None else f"result={result}", log)
        log(f"itf-maintain: {job} ok ({dt:.1f}s)")
    except Exception as e:
        dt = time.monotonic() - t0
        try:
            conn.rollback()      # clear the aborted txn if the conn is still live
        except Exception:
            pass
        _record_resilient(cfg, conn, s, job, "error", repr(e)[:1000], dt, log)
        _persist_event(cfg, conn, s, job, "error", dt, repr(e)[:2000], log)
        log(f"itf-maintain: {job} FAILED ({dt:.1f}s): {e!r}")
        log(traceback.format_exc())


# ---------------------------------------------------------------------------
# One tick (safe to call standalone) and the daemon loop
# ---------------------------------------------------------------------------
def run_tick(cfg=None, index_cfg=None, log=print):
    """One maintenance cycle: refresh always; reconcile if a day has passed;
    sweep if a week has.  Everything runs under a single advisory lock on one
    connection, so nothing overlaps; a tick that cannot take the lock defers."""
    cfg = cfg or ItfConfig()
    index_cfg = index_cfg or ti.IndexConfig()
    s = index_cfg.schema
    conn = connect_index(cfg)
    try:
        if not _try_lock(conn):
            log("itf-maintain: another run holds the lock; deferring this tick")
            return
        try:
            _ensure_state(conn, s)
            _run_job(cfg, conn, s, "refresh",
                     lambda: ti.refresh_incremental(conn, index_cfg), log)
            # A job that dropped the connection hands the rest of the tick to a
            # fresh connection next tick -- never run _due/_unlock on the corpse
            # (that would raise 'connection already closed' every tick).
            if _conn_alive(conn) and _due(conn, s, "reconcile", DAILY):
                _run_job(cfg, conn, s, "reconcile",
                         lambda: ti.reconcile_membership(conn, index_cfg, log=log), log)
            if _conn_alive(conn) and _due(conn, s, "sweep", WEEKLY):
                _run_job(cfg, conn, s, "sweep",
                         lambda: ti.sweep_validity(conn, index_cfg, log=log), log)
        finally:
            if _conn_alive(conn):
                _unlock(conn)
    finally:
        conn.close()


def run_scheduler(cfg=None, index_cfg=None, period_s=60, log=print):
    """Forever: every ``period_s`` seconds run a tick when maintenance is enabled.

    Never returns and never raises -- designed to be the target of a daemon
    thread (web service) or the body of ``python -m pyroorb.itf.maintain scheduler``.
    The env gate is re-checked each period so the process idles cheaply while
    ``ITF_MAINTAIN_TURNED_ON`` is off."""
    cfg = cfg or ItfConfig()
    index_cfg = index_cfg or ti.IndexConfig()
    log(f"itf-maintain: scheduler starting (refresh~{period_s}s, reconcile "
        f"daily, sweep weekly; enabled={maintain_enabled()})")
    disabled_logged = False
    while True:
        t0 = time.monotonic()
        if maintain_enabled():
            disabled_logged = False
            try:
                run_tick(cfg, index_cfg, log=log)
            except Exception as e:                     # e.g. DB unreachable
                log(f"itf-maintain: tick error (continuing): {e!r}")
                log(traceback.format_exc())
        elif not disabled_logged:
            log("itf-maintain: disabled (ITF_MAINTAIN_TURNED_ON not set); idling")
            disabled_logged = True
        elapsed = time.monotonic() - t0
        time.sleep(max(1.0, period_s - elapsed))
