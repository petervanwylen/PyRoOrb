"""Maintenance CLI for the ``itf.itf_tracklets`` index.

PyRoOrb owns the index's freshness.  All jobs are idempotent and checkpointed
in the database, so a scheduler only ever needs to invoke them:

    python -m pyroorb.itf.maintain refresh        # every 5-10 min
    python -m pyroorb.itf.maintain reconcile      # daily
    python -m pyroorb.itf.maintain sweep          # weekly
    python -m pyroorb.itf.maintain backfill       # one-time cold start (resumable)
    python -m pyroorb.itf.maintain status         # sync row + row counts

Rather than wire a scheduler yourself, run all three on one clock (single-flight,
gated by ``ITF_MAINTAIN_TURNED_ON``; see :mod:`pyroorb.itf.scheduler`):

    python -m pyroorb.itf.maintain scheduler      # forever loop (refresh 60s / daily / weekly)
    python -m pyroorb.itf.maintain tick           # one cycle then exit (e.g. an external cron)

On Railway this scheduler runs as a daemon thread inside the web service -- just
set ``ITF_MAINTAIN_TURNED_ON=True`` and redeploy.  To check it's working:

    python -m pyroorb.itf.maintain events         # current state + recent event log

The scheduler writes an ``itf.itf_maintain_events`` row on every FAILURE and every
daily/weekly SUCCESS (not the every-minute refresh) -- a durable window into
whether maintenance is healthy, queryable from anywhere with the ``itf`` creds.

(credentials in the environment or the repo-root
``.env``; see :mod:`pyroorb.itf.dbio`).
"""

import argparse
import sys

from .config import ItfConfig
from .dbio import connect_index
from . import index as ti


def _status(conn, index_cfg):
    s = index_cfg.schema
    with conn.cursor() as cur:
        cur.execute(f"SELECT source, count(*) FROM {s.target_qualified} "
                    f"GROUP BY 1 ORDER BY 1")
        for src, n in cur.fetchall():
            print(f"{src:>12}: {n}")
        cur.execute(f"SELECT last_watermark, last_id, last_run, backfill_done "
                    f"FROM {s.sync_table} WHERE id=1")
        wm, last_id, last_run, done = cur.fetchone()
        print(f"watermark={wm}  last_id={last_id}  last_run={last_run}  "
              f"backfill_done={done}")
        cur.execute(f"SELECT count(*), max(evicted_at) FROM {s.audit_qualified}")
        n_ev, last_ev = cur.fetchone()
        print(f"evictions logged: {n_ev} (last {last_ev})")


def _events(conn, index_cfg, limit):
    """The scheduler's monitoring window: current per-job state plus the recent
    event log (every failure + every daily/weekly success; the every-minute
    refresh's successes are intentionally not logged)."""
    from .scheduler import events_qualified, state_qualified
    s = index_cfg.schema
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT job, last_run, last_status, "
                        f"round(duration_s::numeric, 1) "
                        f"FROM {state_qualified(s)} ORDER BY job")
            state = cur.fetchall()
    except Exception:
        conn.rollback()
        print("(no maintenance state yet -- the scheduler hasn't run)")
        return
    print("current state (drives the daily/weekly gate):")
    for job, last_run, st, dur in state:
        print(f"  {job:>9}: last_run={last_run}  status={st}  dur={dur}s")
    print(f"last {limit} events (newest first; failures + daily/weekly successes):")
    with conn.cursor() as cur:
        cur.execute(f"SELECT ts, job, status, round(duration_s::numeric, 1), detail "
                    f"FROM {events_qualified(s)} ORDER BY ts DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
    if not rows:
        print("  (none logged yet)")
    for ts, job, status, dur, detail in rows:
        d = f"  {detail}" if detail else ""
        print(f"  {ts:%Y-%m-%d %H:%M:%S%z} {job:>9} {status:>5} {dur}s{d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="job", required=True)
    sub.add_parser("ensure-schema", help="create tables/indexes if absent")
    sub.add_parser("refresh", help="incremental refresh (5-10 min cadence)")
    p = sub.add_parser("reconcile", help="pseudo-ITF membership (daily)")
    p.add_argument("--batch", type=int, default=3000)
    p.add_argument("--sleep", type=float, default=0.2)
    p = sub.add_parser("sweep", help="validity sweep backstop (weekly)")
    p.add_argument("--batch", type=int, default=5000)
    p.add_argument("--sleep", type=float, default=0.2)
    p = sub.add_parser("backfill", help="resumable cold-start ITF sync")
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--until", default=None,
                   help="ISO date to pause the sweep at (bounded phase)")
    sub.add_parser("status", help="sync row + row counts")
    p = sub.add_parser("scheduler", help="run all jobs on one clock, forever "
                       "(gated by ITF_MAINTAIN_TURNED_ON)")
    p.add_argument("--period", type=float, default=60.0,
                   help="refresh cadence in seconds (default 60)")
    sub.add_parser("tick", help="run one maintenance cycle then exit")
    p = sub.add_parser("events", help="monitoring: current per-job state + recent "
                       "event log (failures + daily/weekly successes)")
    p.add_argument("--limit", type=int, default=50)
    args = ap.parse_args(argv)

    cfg = ItfConfig()
    index_cfg = ti.IndexConfig()

    # The scheduler owns its own connection per tick -- don't open one here.
    if args.job in ("scheduler", "tick"):
        from .scheduler import run_scheduler, run_tick
        if args.job == "scheduler":
            run_scheduler(cfg, index_cfg, period_s=args.period)
        else:
            run_tick(cfg, index_cfg)
        return 0

    conn = connect_index(cfg)
    try:
        if args.job == "ensure-schema":
            ti.ensure_schema(conn, index_cfg)
            print("schema ensured")
        elif args.job == "refresh":
            n = ti.refresh_incremental(conn, index_cfg)
            print(f"refreshed {n} tracklets")
        elif args.job == "reconcile":
            ti.reconcile_membership(conn, index_cfg, batch=args.batch,
                                    sleep_s=args.sleep)
        elif args.job == "sweep":
            ti.sweep_validity(conn, index_cfg, batch=args.batch, sleep_s=args.sleep)
        elif args.job == "backfill":
            ti.full_sync_itf(conn, index_cfg, step_days=args.step_days,
                             batch=args.batch, sleep_s=args.sleep,
                             until=args.until)
        elif args.job == "status":
            _status(conn, index_cfg)
        elif args.job == "events":
            _events(conn, index_cfg, args.limit)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    from pyroorb.console import utf8_stdio
    utf8_stdio()
    sys.exit(main())
