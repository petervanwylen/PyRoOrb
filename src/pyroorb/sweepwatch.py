"""What every propagator sweep is doing RIGHT NOW, and the slowest ones lately.

Why this exists: two live ITF searches were found parked inside one rebound
``Simulation.integrate`` call for 17 and 41 hours (pyroorb.net/memory,
2026-09-26), holding two of the four search slots until a restart.  The IAS15
step floor (``forward_model.IAS15_MIN_DT``) is the mechanism we BELIEVE explains
it -- a trial state bound within a hair of a planet's point mass drives the
adaptive step toward zero inside a single C call, where no between-call budget
or deadline can reach it -- and the floor was measured to turn such a call from
"never returns" into milliseconds.  But the wedge was diagnosed from a thread's
innermost frame name and nothing more: not which orbit, not how far along, not
even which backend's call.  If it happens again under the floor, that is all
there would be again.

So this records, per thread, the sweep in flight -- backend, epoch, the trial
STATE, the target epochs and how far through them the sweep has got -- and keeps
the slowest completed sweeps with their step counts.  ``/api/memory`` and
``/healthz`` report both.  A thread parked for an hour then reads as "rebound,
stuck at obs 7 of 300 for 3612 s, integrating THIS state to THAT epoch", which
is a case that can be reproduced offline from the report alone.

Cost is the point, because IOD scores thousands of three-observation trial
orbits per fit through this seam: ``begin`` is two clock reads, a small list
and a dict store; ``end`` is a pop, one clock read and a comparison; the
per-observation progress mark is a single list-item store.  Nothing here
computes anything from the state -- elements, distances and elapsed times are
derived at REPORT time, by whoever asks, from what was recorded.  Stdlib only:
this must import from the propagators without dragging anything in.
"""
import collections
import math
import os
import threading
import time

#: A completed sweep slower than this (seconds) is kept in the ring below.
SLOW_SWEEP_SEC = float(os.environ.get("PYROORB_SLOW_SWEEP_SEC", "5"))
#: How many slow sweeps the ring keeps (newest wins).
SLOW_SWEEP_KEEP = int(os.environ.get("PYROORB_SLOW_SWEEP_KEEP", "24"))
#: An in-flight sweep older than this (seconds) is reported as WEDGED by
#: ``/healthz`` -- past any legitimate sweep, well short of the hours a stuck
#: one has run.  Nothing acts on it; it is what a monitor can alarm on.
WEDGED_SEC = float(os.environ.get("PYROORB_SWEEP_WEDGED_SEC", "600"))

_K2 = 0.01720209895 ** 2          # GM_sun, au^3 / day^2 -- for report-time elements only

# thread ident -> the in-flight record (a list; see begin()).  Each thread only
# ever touches its own key, so the dict needs no lock: CPython's dict ops are
# atomic under the GIL and a reader that races a writer sees a whole record or
# none.
_INFLIGHT = {}
_SLOW = collections.deque(maxlen=SLOW_SWEEP_KEEP)
_SLOW_LOCK = threading.Lock()

# record layout (a list, mutated in place: ``rec[AT] = i`` is the per-observation
# progress mark -- a bare list store, because the propagators do it thousands of
# times a second and a function call there would be the one measurable cost)
_WALL, _PERF, _BACKEND, _T0, _STATE, _TIMES, _AT, _PREV, _LABEL = range(9)
AT = _AT


def begin(backend, t0_tdb, state0, times_tdb, label=None):
    """Register the sweep this thread is about to run.  Returns the record to hand
    to ``mark`` and ``end``.  Keeps references, copies nothing: `state0` and
    `times_tdb` are read only if a report is asked for while the sweep runs."""
    ident = threading.get_ident()
    rec = [time.time(), time.perf_counter(), backend, t0_tdb, state0, times_tdb,
           -1, _INFLIGHT.get(ident), label]
    _INFLIGHT[ident] = rec
    return rec


def mark(rec, i):
    """The sweep is about to integrate to its i-th target epoch."""
    rec[_AT] = i


def end(rec, sim=None):
    """The sweep returned (or raised).  Restores whatever sweep this thread had
    in flight underneath, and files the sweep in the slow ring when it earned it."""
    ident = threading.get_ident()
    prev = rec[_PREV]
    if prev is not None:
        _INFLIGHT[ident] = prev
    else:
        _INFLIGHT.pop(ident, None)
    elapsed = time.perf_counter() - rec[_PERF]
    if elapsed >= SLOW_SWEEP_SEC:
        _file_slow(rec, elapsed, sim)


def _file_slow(rec, elapsed, sim):
    steps = None
    if sim is not None:
        try:
            steps = int(sim.steps_done)          # rebound and ASSIST sims both carry it
        except Exception:
            steps = None
    row = _describe(rec, time.time())
    row.update({"elapsed_s": round(elapsed, 2), "steps_done": steps,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
    with _SLOW_LOCK:
        _SLOW.append(row)


# --- reporting ------------------------------------------------------------

def _elements(state):
    """Two-body heliocentric a, e, q from a 6-state (au, au/day), or None.  For the
    report only: says at a glance whether a trial orbit plunges through the Sun
    (q ~ 0) or is nonsense (e >> 1), without importing anything."""
    try:
        x, y, z, vx, vy, vz = (float(v) for v in state[:6])
        r = math.sqrt(x * x + y * y + z * z)
        v2 = vx * vx + vy * vy + vz * vz
        if r <= 0.0 or not math.isfinite(r) or not math.isfinite(v2):
            return None
        hx, hy, hz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
        h2 = hx * hx + hy * hy + hz * hz
        inv_a = 2.0 / r - v2 / _K2
        e2 = 1.0 - h2 * inv_a / _K2
        e = math.sqrt(e2) if e2 > 0 else 0.0
        a = 1.0 / inv_a if inv_a != 0.0 else float("inf")
        q = h2 / _K2 / (1.0 + e)
        return {"r_au": round(r, 6), "a_au": round(a, 6) if math.isfinite(a) else None,
                "e": round(e, 6), "q_au": round(q, 6)}
    except Exception:
        return None


def _describe(rec, now):
    times = rec[_TIMES]
    state = rec[_STATE]
    at = rec[_AT]
    try:
        n = len(times)
        t0 = float(rec[_T0])
        span = (float(min(times)) - t0, float(max(times)) - t0) if n else (0.0, 0.0)
        target = float(times[at]) - t0 if 0 <= at < n else None
    except Exception:
        n, t0, span, target = None, None, None, None
    try:
        state_out = [float(v) for v in state[:6]]
    except Exception:
        state_out = None
    return {"backend": rec[_BACKEND], "label": rec[_LABEL],
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(rec[_WALL])),
            "elapsed_s": round(now - rec[_WALL], 1),
            # int(): the propagators mark with an index drawn from a NumPy group
            # array, and an np.int64 here made every later report unencodable
            "t0_tdb": t0, "n": n, "at": int(at) if at >= 0 else None,
            "target_days": None if target is None else round(target, 4),
            "span_days": None if span is None else [round(span[0], 3), round(span[1], 3)],
            "state": state_out, "elements": _elements(state) if state_out else None}


def snapshot(names=None):
    """JSON-safe: every sweep in flight (oldest first) and the slow ring (newest
    first).  `names` maps thread ident -> display name; the caller decides what
    a name may show (see webmem._safe_thread_name)."""
    now = time.time()
    rows = []
    for ident, rec in list(_INFLIGHT.items()):
        row = _describe(rec, now)
        row["thread"] = (names or {}).get(ident, ident)
        row["wedged"] = bool(row["elapsed_s"] >= WEDGED_SEC)
        rows.append(row)
    rows.sort(key=lambda r: -r["elapsed_s"])
    with _SLOW_LOCK:
        slow = list(reversed(_SLOW))
    return {"inflight": rows, "wedged": sum(1 for r in rows if r["wedged"]),
            "wedged_sec": WEDGED_SEC, "slow": slow, "slow_sec": SLOW_SWEEP_SEC}


def health():
    """The three numbers a monitor wants, with no state or names attached."""
    now = time.time()
    ages = [now - rec[_WALL] for rec in list(_INFLIGHT.values())]
    return {"inflight": len(ages),
            "oldest_s": round(max(ages), 1) if ages else 0.0,
            "wedged": sum(1 for a in ages if a >= WEDGED_SEC),
            "slow_kept": len(_SLOW)}


def _reset_for_tests():
    _INFLIGHT.clear()
    with _SLOW_LOCK:
        _SLOW.clear()
