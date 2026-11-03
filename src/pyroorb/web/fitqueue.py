"""Admission control for web fits: a bounded worker pool, single-flight
coalescing, and a queue the user can see their place in.

WHY THIS EXISTS
---------------
``server.py`` is a ``ThreadingHTTPServer`` -- one unbounded OS thread per
connection -- and ``/api/fit`` used to run the whole fit inside the request.
Two consequences, both fatal on a public launch day:

  * A hundred people curious about the same object at the same time is a
    hundred concurrent fits, each holding a FittedOrbit plus a rebound/ASSIST
    base sim, all contending for one or two vCPUs.  The container reaches its
    memory limit and is OOM-killed long before it merely gets slow, and
    ``railway.json``'s ``restartPolicyMaxRetries: 3`` means the third such
    death takes the site down until a human intervenes.
  * A hosted reverse proxy will not hold one HTTP request open for a slow fit.
    This is the same wall the ITF search hit ("upstream error" as a response
    body), which is why that route already backgrounds and polls.

So fits go through here.  The pool bounds concurrency, which converts the
failure mode from *fatal* to *slow* -- the single most valuable property this
module has.  Queueing is also strictly better than fair-sharing under a spike:
with N workers the first requester gets an answer in one fit-time, whereas a
hundred fair-shared threads deliver everybody's answer in a hundred fit-times
and OOM before any of them arrive.

SINGLE-FLIGHT IS THE OTHER HALF
-------------------------------
A cache only helps once the first fit has finished.  The herd that matters
arrives *before* that -- a hundred simultaneous requests for an object nobody
has fitted yet all miss, and a queue alone would faithfully run a hundred
identical fits, just politely and one at a time.  So work is keyed by CONTENT
(:func:`fit_key`) and a request whose key is already in flight attaches to the
running job instead of enqueuing another.  A finished job stays keyed for
``DONE_TTL`` and is served straight from memory (a success or a timeout only --
any other failure is re-run, see :func:`_reusable`), which makes this the L1
response cache as well; the ephemeral-disk tier is a later layer under the same
key.

SESSIONS ARE NOT SHARED
-----------------------
Coalescing creates a hazard that did not exist before: ``fit_obs_list`` ends by
calling ``engine_api._store_session``, so one shared fit carries ONE session
handle.  Handing that handle to a hundred clients would let any one of them
mutate everybody else's open fit -- their inclusion mask, their force model,
their epoch -- because ``_apply_interactive`` writes through to the session
dict and steps a shared Evaluator over a shared rebound base sim.

So the session goes to the FIRST claimant only (normally the person whose
request triggered the fit).  Everyone else receives the identical payload with
``session`` replaced by a ``restore`` token, and rehydrates a PRIVATE session
only if and when they actually interact -- which most people never do.  That
is cheap by construction: ``fit_object._build_evaluator`` rebuilds the context
from the observations in about a second, because finding the state is the
expensive part of a fit and reconstructing the machinery around it is not.
The restore seed is deliberately plain data (observations, mask, state, epoch,
selectors) so the same record can later be written to ephemeral disk without
pickling a FittedOrbit -- which would not pickle anyway, since ``fit._ctx``
holds a live rebound simulation.
"""
import os
import time
import uuid
import hashlib
import threading
import collections

from pyroorb import deadline
from pyroorb.web import fitdisk
from pyroorb.web import capacity

# --- sizing -----------------------------------------------------------------
# Both of this server's concurrency caps -- the fit pool here and the ITF search
# cap in engine_api -- are derived in `capacity`, TOGETHER, from the container's
# own cgroup limits.  They belong in one place because they spend the same two
# budgets: a core this pool takes is one a search cannot have, and the memory a
# full pool can transiently hold is memory a sweep must not assume is free.
# Sizing them in separate modules is exactly how the ITF cap came to be a
# hand-set 3, justified in its own comment by a container size that was never
# this container's.
#
# Re-exported under their historical names so this module's callers, its tests
# and /healthz keep the surface they already had.
FIT_PEAK_MB = capacity.FIT_PEAK_MB
BASELINE_MB = capacity.FIT_BASELINE_MB

DONE_TTL = float(os.environ.get("PYROORB_FIT_DONE_TTL") or 900)   # keyed reuse window
JOB_TTL = float(os.environ.get("PYROORB_FIT_JOB_TTL") or 1800)    # collectable window
JOBS_MAX = int(os.environ.get("PYROORB_FIT_JOBS_MAX") or 256)

# How long one fit may RUN on a worker (seconds; 0 disables).  Distinct from
# PYROORB_SYNC_FIT_DEADLINE, which is how long a caller WAITS before being handed
# a job handle -- that one never stopped the work, which is how a single request
# came to hold a worker for twelve hours.  Three minutes is ~80x the median fit
# (2.1 s) and clear of the slowest healthy ones (10P re-fits in ~51 s from its
# published-orbit seed), so it bounds the pathological case without cutting any
# object the web is expected to serve.  The CLI is deliberately unbounded.
FIT_MAX_SEC = float(os.environ.get("PYROORB_FIT_MAX_SEC") or 180)

# Fallback fit duration (seconds) for the queue ETA before any fit has been
# timed on this box.  Replaced by a running median as soon as there are samples.
DEFAULT_FIT_SECS = 20.0

# Memoised queue bound.  The container's limits cannot change under a running
# process, and this is read on every submit; `queue_max` fills it on first use.
_QUEUE_MAX = None


def pool_size():
    """How many fits may run at once on THIS box.

    Derived in :mod:`capacity` alongside the ITF cap, from the container's own
    CPU quota and memory limit -- see that module for why the two caps are
    decided together rather than each from its own reading of the box.
    """
    return capacity.fit_pool_size()


def queue_max():
    """How many fits may WAIT before new ones are refused.

    A queue with no bound is just an unbounded thread pool with extra latency:
    it accepts work the box will never reach, holds the memory of every pending
    request, and answers everyone late instead of answering some people now.
    Past this depth the honest answer is "we are swamped, come back in a
    minute", which is a better experience than a spinner that resolves after
    the user has closed the tab.
    """
    global _QUEUE_MAX
    override = os.environ.get("PYROORB_FIT_QUEUE_MAX")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if _QUEUE_MAX is None:
        _QUEUE_MAX = pool_size() * 8
    return _QUEUE_MAX


# --- state ------------------------------------------------------------------
_LOCK = threading.Condition()
_JOBS = {}                       # job id -> entry
_BY_KEY = {}                     # content key -> job id (live AND recently done)
_WAITING = collections.deque()   # job ids not yet picked up by a worker
_WORKERS = []
_DURATIONS = collections.deque(maxlen=32)   # recent fit wall-times, for the ETA


def fit_key(desigs, force, extra=None):
    """The coalescing key: same key == same work, so one fit serves everybody.

    Keyed on CONTENT, not on the designation as typed -- ``2026 AB123`` and
    ``2026 ab123`` and a reordered linkage list are the same fit and must
    share it.  The engine revision is folded in so a deploy cannot serve a
    payload computed by the previous version of the code, and the station
    table's content token so a fit that DROPPED a then-unknown station's
    observations is not served once the periodic refresh has learnt it.
    """
    norm = ",".join(sorted(str(d).strip().upper() for d in desigs if str(d).strip()))
    seed = f"{norm}|{bool(force)}|{_engine_rev()}|{_obscodes_rev()}|{extra or ''}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _compute_rev():
    """The running code's git revision, or '?' outside a checkout."""
    rev = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or ""
    if rev:
        return rev
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, encoding="utf-8", timeout=5).stdout.strip() or "?"
    except Exception:
        return "?"


# Resolved ONCE, at import, on one thread -- never lazily on first use.
#
# It was lazy, and that is a cache-splitting race: two requests for the same
# object arriving together both find the memo empty, both shell out, and if one
# `git rev-parse` times out or fails on a slow filesystem it yields "?" while
# the other yields the sha.  Two revisions means two KEYS for one object, which
# means the herd this module exists to collapse splits into two fits and two
# cache entries.  Caught in local load testing: twelve simultaneous requests for
# 2023 QJ3 produced two distinct cache files with identical contents.
#
# Resolving at import costs one subprocess per process start (none at all on
# Railway, where RAILWAY_GIT_COMMIT_SHA is set) and cannot race.
_REV = _compute_rev()


def _engine_rev():
    """The running code's git revision -- fixed for the life of the process."""
    return _REV


def _obscodes_rev():
    """The station table's content token, or '?' when it cannot be had (no cache and
    no network: the fit itself will say so)."""
    try:
        from pyroorb import obs_sources
        return obs_sources.obscodes_token()
    except Exception:
        return "?"


def _median_secs():
    """Typical fit wall-time for the ETA.  Falls back to the default until a
    few fits have actually been timed on this box -- one sample is not a
    distribution, and a single trivially-cached fit would otherwise promise
    everyone behind it an instant answer.  Floored so an ETA is never zero."""
    if len(_DURATIONS) < 3:
        return DEFAULT_FIT_SECS
    vals = sorted(_DURATIONS)
    return max(1.0, vals[len(vals) // 2])


def _evict(now):
    """Drop jobs nobody can still want.  Caller holds the lock.

    A RUNNING or QUEUED job is never evicted -- losing a result somebody is
    waiting for is the one failure this table exists to prevent -- so the cap
    can be exceeded transiently by live work, exactly as the ITF job table
    allows.
    """
    for jid, j in list(_JOBS.items()):
        if j["state"] != "done":
            continue
        age = now - j["ended"]
        if age > JOB_TTL or (age > DONE_TTL and j["collected"]):
            _JOBS.pop(jid, None)
            if _BY_KEY.get(j["key"]) == jid:
                _BY_KEY.pop(j["key"], None)
    while len(_JOBS) > JOBS_MAX:
        done = sorted((not j["collected"], j["ended"], k)
                      for k, j in _JOBS.items() if j["state"] == "done")
        if not done:
            break
        jid = done[0][2]
        j = _JOBS.pop(jid, None)
        if j is not None and _BY_KEY.get(j["key"]) == jid:
            _BY_KEY.pop(j["key"], None)


def _reusable(j, now, max_age=None):
    """May a later identical request be answered by job ``j``?  Caller holds the lock.

    Live work always (that is the single-flight).  A FINISHED job only within
    DONE_TTL, and only if its answer is one worth repeating: a success, or a fit
    that ran out of time -- that one is a property of the object, and replaying
    it is what stops a crowd retrying it from each holding a worker for
    FIT_MAX_SEC.  Any other failure is NOT replayed.  "The MPC did not answer"
    says "try again in a moment" in so many words, and the table used to serve
    that very sentence to the retry for the next fifteen minutes although the
    outage had lasted twenty seconds; a fetch can also come back empty for a
    NEOCP object minutes before its astrometry appears.  Failures are cheap to
    re-run, so re-running them is the safe direction -- the disk tier already
    stores only successes for the same reason.

    ``max_age`` shortens DONE_TTL for a caller whose astrometry is still
    arriving (see :func:`submit`); live work is joined regardless, since it
    began moments ago and will see whatever the MPC has now.
    """
    if j["state"] != "done":
        return True
    ttl = DONE_TTL if max_age is None else min(DONE_TTL, max_age)
    if (now - j["ended"]) > ttl:
        return False
    p = j.get("payload")
    return isinstance(p, dict) and bool(p.get("ok") or p.get("timed_out"))


def _position(job_id):
    """1-based place in the queue, or 0 once a worker has picked the job up.
    Caller holds the lock."""
    try:
        return list(_WAITING).index(job_id) + 1
    except ValueError:
        return 0


def _snapshot(j, jid):
    """The client-facing view of a job.  Caller holds the lock."""
    now = time.time()
    pos = _position(jid)
    out = {"ok": True, "job": jid, "state": j["state"],
           "done": j["state"] == "done",
           "waiting": pos, "ahead": max(0, pos - 1),
           "queue_depth": len(_WAITING), "workers": len(_WORKERS) or pool_size(),
           "shared": j["claims"] > 0}
    if j["state"] == "queued":
        # Time until the answer, not time until the job starts: the fit itself
        # still has to run after the queue drains.
        workers = max(1, len(_WORKERS) or pool_size())
        out["eta_sec"] = round(((pos - 1) / workers + 1.0) * _median_secs(), 1)
    elif j["state"] == "running":
        out["elapsed"] = round(now - j["started"], 1)
    else:
        out["elapsed"] = round(j["ended"] - j["started"], 1)
    return out


def _ensure_workers():
    """Start the pool on first use.  Caller holds the lock.

    Lazily, so importing this module (a test, the CLI, a tool that only wants
    :func:`pool_size`) never spawns threads.
    """
    if _WORKERS:
        return
    for i in range(pool_size()):
        th = threading.Thread(target=_worker, name=f"fit-worker-{i}", daemon=True)
        try:
            th.start()
        except Exception:
            # An out-of-memory box cannot create threads, and that is precisely
            # when this route matters.  One worker is enough to make progress;
            # refusing to start any would be worse than a small pool.
            break
        _WORKERS.append(th)


def _worker():
    """One pool thread, for the life of the process.

    The body is defensive to the point of paranoia because a worker that dies
    is invisible: the pool silently shrinks, throughput drops, and eventually
    every fit queues behind nothing at all.  There is no reaper here (unlike the
    ITF job table, whose slots are held by per-search threads) -- the workers ARE
    the resource, so the cheaper guarantee is that one can never exit.
    """
    while True:
        try:
            _work_once()
        except Exception:                # never let a worker thread die
            time.sleep(0.05)


def _work_once():
    """Take one job off the queue, run it, publish the answer."""
    with _LOCK:
        while not _WAITING:
            _LOCK.wait()
        jid = _WAITING.popleft()
        j = _JOBS.get(jid)
        if j is None or j["state"] != "queued":
            return          # coalesced away or evicted before a worker got to it
        j["state"] = "running"
        j["started"] = time.time()
        work = j["work"]
        key, desig = j["key"], j.get("desig")
    try:
        # THE bound on a web fit.  `wait()` below bounds how long a CALLER waits;
        # this bounds how long the work runs, which is a different thing and the
        # one that was missing -- a "4 vesta" request held one of seven workers
        # for twelve hours and only a restart got it back.  Set here because every
        # web path that starts engine work comes through this pool, and on the
        # worker thread because a ContextVar binds this fit and no neighbour's.
        # An ITF search runs on its own thread and so never inherits it.
        with deadline.deadline(FIT_MAX_SEC, label="this fit"):
            payload = work()
    except deadline.DeadlineExceeded as ex:
        # Not a crash: the object is too slow for an interactive surface, which is
        # worth saying in those words rather than as a bare timeout.
        payload = {"ok": False, "timed_out": True, "error": str(ex) + (
            " -- it is too slow to fit interactively. The command line has no such "
            "bound if you need this object.")}
    except Exception as ex:      # a crashed fit must still ANSWER its waiters
        payload = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
    if not isinstance(payload, dict):
        # Every engine path returns a dict; anything else is a programming
        # error upstream.  Reporting it as a failed job keeps the waiters
        # answered and this thread alive, which a bare AttributeError in the
        # bookkeeping below would not.
        payload = {"ok": False,
                   "error": f"fit returned {type(payload).__name__}, not a result"}
    with _LOCK:
        # A fit produces two things: the JSON the browser reads, and the
        # live objects a later interaction needs (the pristine template
        # session).  The latter cannot travel in the payload -- it holds a
        # rebound simulation -- so `work` hands it over under the reserved
        # `_retain` key and it is moved onto the job entry here, where it
        # lives exactly as long as the cached payload does.
        if "_retain" in payload:
            payload = dict(payload)
            j["retain"] = payload.pop("_retain")
        j["payload"] = payload
        j["ended"] = time.time()
        j["state"] = "done"
        j["work"] = None         # release the closure (it holds the obs list)
        if payload.get("ok"):
            _DURATIONS.append(j["ended"] - j["started"])
        _LOCK.notify_all()
    # Write-through to the ephemeral disk tier, OUTSIDE the lock: the payload is
    # already published and the waiters already released, so a slow disk cannot
    # delay the people who asked for this fit, and a failed write is only a
    # future miss.  Only SUCCESSFUL fits are stored -- an error may be transient
    # (a fetch that timed out), and a cached failure would keep serving it.
    if payload.get("ok"):
        fitdisk.put(key, payload, desig=desig)
    # Hand back whatever the allocator kept, if it kept a lot.  A single fit is
    # small (~1.74 KB/obs), so this is a no-op on the ordinary path -- the gate is
    # 0.4 us and only fires past webmem.TRIM_THRESHOLD_MB.  It earns its place on
    # the fit that ISN'T ordinary: a mega-arc, or a grind that ran for hours, whose
    # freed arena would otherwise stay resident for the life of the process and
    # push the ITF admission gate into refusing every search.
    try:
        from pyroorb.web import webmem
        webmem.trim_allocator()
    except Exception:
        pass


def submit(key, work, desig=None, bypass_cache=False, max_age=None):
    """Enqueue ``work`` under ``key``, or attach to the identical work already
    in flight.  Returns a snapshot dict; ``{"ok": False, "busy": True}`` when
    the queue is full.

    ``work`` is a zero-argument callable returning the payload dict.  It runs on
    a pool thread, so it must not touch request-local state.  ``desig`` is
    recorded so a disk-cached payload can tell a client what to re-fit if it
    later wants to interact (see :func:`_from_disk`).

    ``bypass_cache`` skips the DISK tier for this one call, and exists because
    without it a disk-cached object could never become interactive again.  The
    disk tier stores a payload but no template, so it answers with `refit` and
    no `restore`; the client then re-runs the fit to obtain a session -- and
    that re-run, going through this same function, was served the identical
    disk hit.  The fit never re-ran, no session ever appeared, and the viewer
    was told "that fit has expired, re-run it" by a re-run that could not
    succeed.  Once an object aged out of the in-memory table (DONE_TTL, or
    JOBS_MAX eviction) EVERY later visitor was read-only for the life of the
    cache entry -- which on a shared link is the common path, not the corner.

    The live in-memory table is still consulted: a done job there carries its
    template, so a restore succeeds and no re-fit is needed.  Only the disk
    read is skipped.

    ``max_age`` (seconds) caps how old a FINISHED answer may be, in memory and
    on disk alike, for work whose input is still changing.  A NEOCP object's
    arc grows by the hour, and the 15-minute memory window and 6-hour disk
    window kept serving the pair "P22r4rx P/2015 X3" with 4 of P22r4rx's
    observations while "P22r4rx" alone -- a different key, fitted later --
    showed all 8.  The caller passes the astrometry cache's own lifetime, so a
    fit can never outlive the observations it was computed from.
    """
    now = time.time()
    with _LOCK:
        _evict(now)
        _ensure_workers()
        jid = _BY_KEY.get(key)
        j = _JOBS.get(jid) if jid else None
        finished = None
        if j is not None:
            # Live or recently finished, either way this caller wants exactly
            # what it produced.  `claims` counts everyone after the first, who
            # are the ones that must NOT be handed the shared session.
            if _reusable(j, now, max_age):
                j["claims"] += 1
                if j["state"] != "done":
                    snap = _snapshot(j, jid)
                    snap["coalesced"] = True     # joined a fit already running
                    return snap
                # Already finished: answer with the payload NOW rather than a
                # handle the client has to poll.  This is the hot path -- one
                # object, many viewers -- and making the ninety-ninth of them
                # wait a poll interval for an answer already in memory would
                # give the cache away for nothing.
                finished = jid
    if finished is not None:
        out = status(finished)
        out["cached"] = "memory"                 # answered from the live table
        return out
    # Nothing live under this key -- but the answer may still be on disk, where
    # it is far cheaper than the fit and outlives the in-memory table.  Checked
    # BEFORE admission so a cached object never occupies a worker slot or a
    # queue place, which is the whole point on a launch spike: a hundred people
    # on one object cost one fit, and the hundred-and-first costs a file read.
    hit = None if bypass_cache else _from_disk(key, max_age)
    if hit is not None:
        return hit
    with _LOCK:
        # RE-CHECK under the lock: the disk read above ran WITHOUT it, so another
        # thread's submit() for this same key could have registered a job in the
        # gap. Without this check both callers fall through and each creates its
        # own job -- the exact herd this module exists to collapse, and the one
        # this function's whole design is meant to prevent for a cold key, which
        # is precisely the case a launch spike hits hardest (nobody has fitted
        # the object yet, so every simultaneous request misses the disk tier
        # too). Caught with a concurrent-submit test that fires N threads at one
        # cold key and asserts exactly one job is created.
        jid = _BY_KEY.get(key)
        j = _JOBS.get(jid) if jid else None
        if j is not None and _reusable(j, now, max_age):
            j["claims"] += 1
            snap = _snapshot(j, jid)
            if j["state"] != "done":
                snap["coalesced"] = True
            else:
                snap["cached"] = "memory"
            return snap
        if len(_WAITING) >= queue_max():
            return {"ok": False, "busy": True, "queue_depth": len(_WAITING),
                    "workers": len(_WORKERS),
                    "error": (f"{len(_WAITING)} fits are already waiting and the "
                              f"server is at capacity. Please try again in a "
                              f"minute -- your object is not the problem, the "
                              f"server is just full.")}
        jid = f"fit{uuid.uuid4().hex[:12]}"
        _JOBS[jid] = j = {"key": key, "work": work, "desig": desig,
                          "state": "queued",
                          "created": now, "started": 0.0, "ended": 0.0,
                          "collected": 0.0, "payload": None, "claims": 0,
                          "retain": None}
        _BY_KEY[key] = jid
        _WAITING.append(jid)
        _LOCK.notify()
        return _snapshot(j, jid)


def _from_disk(key, max_age=None):
    """Serve ``key`` from the ephemeral disk cache, or None on a miss.

    The payload comes back WITHOUT a `restore` token: the live template session
    that a token redeems lives only in memory, and this key has no live job.  So
    the answer carries `refit` (the designation) instead, and a client that
    later wants to interact re-runs the fit for itself -- the trade the disk
    tier is built on, since almost nobody interacts.
    """
    rec = fitdisk.get(key, max_age=max_age)
    if not rec:
        return None
    payload = rec.get("payload")
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    payload = dict(payload)
    payload.pop("session", None)
    payload.pop("restore", None)
    if rec.get("desig"):
        payload["refit"] = rec["desig"]
    return {"ok": True, "job": None, "state": "done", "done": True,
            "cached": "disk", "waiting": 0, "ahead": 0,
            "queue_depth": len(_WAITING), "workers": len(_WORKERS) or pool_size(),
            "shared": True, "elapsed": 0.0, "result": payload}


def status(job_id):
    """Poll a job.  While it runs, the snapshot carries the queue position and
    an ETA; once done it carries ``result``.

    On a COALESCED job nobody is handed a session -- not even the first caller.
    The payload carries a ``restore`` token instead, and a private session is
    built from the job's pristine template only when a client actually
    interacts (see the module docstring).  Uniformity is the point: if the
    first caller took the template and then edited its mask, every later
    restore would clone the edit.  A one-off job (:func:`run_bounded`) has no
    template and cannot be shared, so its payload passes through untouched.

    An unknown id answers ``expired`` so a client stops polling rather than
    spinning forever.
    """
    with _LOCK:
        j = _JOBS.get(str(job_id or ""))
        if j is None:
            return {"ok": False, "expired": True,
                    "error": "unknown or expired fit job"}
        jid = str(job_id)
        if j["state"] != "done":
            return _snapshot(j, jid)
        out = _snapshot(j, jid)
        payload = j["payload"]
        j["collected"] = j["collected"] or time.time()
        has_template = j["retain"] is not None
    if isinstance(payload, dict) and payload.get("ok") and has_template:
        # A job with a template is a COALESCED one: its payload may reach many
        # clients, so the interactive handle inside it is replaced by a restore
        # token they each redeem privately.  A one-off job (run_bounded) is keyed
        # uniquely and can never be shared, so its payload -- session and all --
        # passes through exactly as the engine built it.
        payload = dict(payload)
        payload.pop("session", None)
        payload["restore"] = jid
    out["result"] = payload
    return out


def run_bounded(work, timeout, label="fit"):
    """Run one-off fit work through the pool and wait for it, with NO coalescing.

    For the fit endpoints that are not a plain designation lookup -- a pasted
    observation blob, a re-auto-solve of an open session.  Each is a genuinely
    unique piece of work, so there is nothing to share, but each is also a full
    fit: leaving them to run on the request thread would let a crowd start
    arbitrarily many of them and make the pool decorative, since the pool only
    protects the box if EVERY path that can start a fit goes through it.

    Returns the payload, or a refusal when the queue is full, or -- past
    ``timeout`` -- a job handle the caller can hand back for polling.
    """
    key = f"oneoff:{label}:{uuid.uuid4().hex}"
    started = submit(key, work)
    if not started.get("ok"):
        return started
    if started.get("done") and started.get("result"):
        return started["result"]
    job = started["job"]
    if wait(job, timeout):
        out = status(job)
        return out.get("result") or out
    started["pending"] = True
    return started


def wait(job_id, timeout):
    """Block until a job finishes or ``timeout`` seconds elapse; return True if
    it finished.  Used by the compatibility ``GET /api/fit`` path, which answers
    inline when the fit is quick and hands back a job handle when it is not --
    so a legacy caller still gets admission control without a request that a
    hosted proxy would kill."""
    deadline = time.time() + max(0.0, float(timeout))
    with _LOCK:
        while True:
            j = _JOBS.get(str(job_id or ""))
            if j is None or j["state"] == "done":
                return j is not None
            left = deadline - time.time()
            if left <= 0:
                return False
            _LOCK.wait(left)


def retained(job_id):
    """The live objects a finished job kept back (its pristine template
    session), or None once the job has aged out.  Read by the restore path,
    which turns it into a private session."""
    with _LOCK:
        j = _JOBS.get(str(job_id or ""))
        return j["retain"] if j is not None else None


def stats():
    """Live admission numbers, for /healthz and the memory page.

    The cgroup reads are done OUTSIDE the queue lock: /healthz is the route
    someone hits when the box is already struggling, and holding the lock the
    workers need across a handful of syscalls would make the health check
    itself a small source of the contention it is there to report.
    """
    cpu, lim = capacity.cpu_quota(), capacity.limit_mb()
    pool, qmax = pool_size(), queue_max()
    with _LOCK:
        running = sum(1 for j in _JOBS.values() if j["state"] == "running")
        return {"workers": len(_WORKERS) or pool, "pool_size": pool,
                "running": running, "queued": len(_WAITING),
                "queue_max": qmax, "jobs_held": len(_JOBS),
                "median_fit_sec": round(_median_secs(), 1),
                "cpu_quota": cpu, "memory_limit_mb": lim,
                "disk_cache": fitdisk.stats()}
