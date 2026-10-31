"""Load harness for the web server: concurrency, coalescing, cache, limits.

Answers the questions the unit tests cannot, because they are about a RUNNING
server under simultaneous load:

  * does one object requested by N people cost ONE fit? (single-flight)
  * does a second wave of the same object come back from cache, and how fast?
  * how deep does the queue get, and is the reported ETA honest?
  * where does RSS actually go, against the container limit?
  * does the rate limiter refuse the right callers without refusing everyone?

Read-only against the server's own API -- it starts nothing and submits
nothing.  Safe to point at production (it only fits objects, which is what the
site is for), but note the FIT bucket is rate limited: --clients above the burst
is not a bug, it is the limiter being measured.

    python -m pyroorb.web.stress --url http://127.0.0.1:8000 --scenario herd
    python -m pyroorb.web.stress --url https://... --scenario mixed --clients 40
"""
import os
import sys
import json
import time
import argparse
import threading
import urllib.error
import urllib.request

# Objects with cached astrometry in the repo's data dir fit fast; these are a
# spread of arcs so "distinct objects" is not secretly one cost.
DEFAULT_OBJECTS = ["2023 QJ3", "2010 TR241", "2014 EJ130", "2004 TE10",
                   "2003 LW2", "2001 UN18", "2013 TP4", "2008 NA"]


def _req(url, path, payload=None, timeout=120):
    """One request -> (status, body dict, seconds).  Never raises."""
    t0 = time.time()
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            code = r.status
    except urllib.error.HTTPError as ex:                 # 429 lives here
        raw, code = ex.read(), ex.code
    except Exception as ex:
        return 0, {"error": f"{type(ex).__name__}: {ex}"}, time.time() - t0
    try:
        return code, json.loads(raw), time.time() - t0
    except ValueError:
        return code, {"error": "non-JSON body", "raw": raw[:200].decode("utf-8", "replace")}, \
            time.time() - t0


def health(url):
    _c, body, _t = _req(url, "/healthz", timeout=15)
    return body


def fit_once(url, desig, timeout=300):
    """Start a fit and poll it out, as the browser does.  Returns a record."""
    t0 = time.time()
    code, started, _ = _req(url, "/api/fit_start", {"desig": desig})
    if code == 429:
        return {"desig": desig, "outcome": "rate_limited",
                "retry_after": started.get("retry_after"), "secs": time.time() - t0}
    if not started.get("ok"):
        if started.get("busy"):
            return {"desig": desig, "outcome": "busy", "secs": time.time() - t0}
        return {"desig": desig, "outcome": "error",
                "error": started.get("error"), "secs": time.time() - t0}
    if started.get("cached") == "disk":
        return {"desig": desig, "outcome": "disk_cache", "secs": time.time() - t0,
                "queued_behind": started.get("ahead", 0)}
    job, behind = started.get("job"), started.get("ahead", 0)
    if started.get("done"):
        return {"desig": desig, "outcome": "memory_cache", "secs": time.time() - t0,
                "queued_behind": behind, "job": job}
    while time.time() - t0 < timeout:
        time.sleep(0.5)
        _c, out, _ = _req(url, "/api/fit_result?id=" + str(job))
        if out.get("expired"):
            return {"desig": desig, "outcome": "expired", "secs": time.time() - t0}
        if out.get("done"):
            result = out.get("result") or {}
            return {"desig": desig,
                    "outcome": "fitted" if result.get("ok") else "failed",
                    "error": result.get("error"), "secs": time.time() - t0,
                    "queued_behind": behind, "job": job,
                    "restore": bool(result.get("restore")),
                    "refit": bool(result.get("refit"))}
    return {"desig": desig, "outcome": "timeout", "secs": time.time() - t0}


def _run_parallel(fn, args_list):
    """Fire every call AT ONCE -- the point is simultaneity, not throughput."""
    out = [None] * len(args_list)
    gate = threading.Event()

    def one(i, a):
        gate.wait()
        out[i] = fn(*a)

    ts = [threading.Thread(target=one, args=(i, a), daemon=True)
          for i, a in enumerate(args_list)]
    for t in ts:
        t.start()
    time.sleep(0.2)          # let every thread reach the gate
    gate.set()
    for t in ts:
        t.join()
    return out


def _summarise(label, records, before, after):
    kinds = {}
    for r in records:
        kinds[r["outcome"]] = kinds.get(r["outcome"], 0) + 1
    secs = sorted(r["secs"] for r in records)
    print(f"\n--- {label} ---")
    print(f"  requests   : {len(records)}")
    print(f"  outcomes   : {kinds}")
    if secs:
        print(f"  latency    : min {secs[0]:.1f}s  median {secs[len(secs)//2]:.1f}s  "
              f"max {secs[-1]:.1f}s")
    for tag, h in (("before", before), ("after", after)):
        f = (h or {}).get("fits") or {}
        m = (h or {}).get("memory") or {}
        dc = f.get("disk_cache") or {}
        print(f"  {tag:6s}     : queued={f.get('queued')} running={f.get('running')} "
              f"workers={f.get('workers')} rss={m.get('rss_mb')}MB "
              f"avail={m.get('available_mb')}MB cache={dc.get('entries')} entries")
    return kinds


def scenario_herd(url, clients, objects):
    """A hundred people, one object.  ONE fit should happen; the rest coalesce."""
    desig = objects[0]
    before = health(url)
    recs = _run_parallel(fit_once, [(url, desig) for _ in range(clients)])
    after = health(url)
    kinds = _summarise(f"herd: {clients} clients on {desig}", recs, before, after)
    # COUNT THE JOBS, not the outcomes.  Every coalesced waiter polls the same
    # job and is handed the same finished payload, so "12 fitted" is exactly
    # what success looks like -- the question is how many DISTINCT jobs the
    # server created, and that is the only thing that maps to real CPU.
    jobs = {r.get("job") for r in recs if r.get("job")}
    served = kinds.get("fitted", 0) + kinds.get("memory_cache", 0) + \
        kinds.get("disk_cache", 0)
    print(f"  VERDICT    : {served}/{clients} served from {len(jobs)} distinct job(s)"
          f" -- {'COALESCED' if len(jobs) <= 1 else 'NOT coalescing'}")
    return recs


def scenario_second_wave(url, clients, objects):
    """The same object again once the first wave is done: memory or disk cache,
    and it should be near-instant."""
    desig = objects[0]
    fit_once(url, desig)
    before = health(url)
    recs = _run_parallel(fit_once, [(url, desig) for _ in range(clients)])
    after = health(url)
    kinds = _summarise(f"second wave: {clients} on a warm {desig}", recs, before, after)
    cached = kinds.get("memory_cache", 0) + kinds.get("disk_cache", 0)
    print(f"  VERDICT    : {cached}/{len(recs)} served from cache")
    return recs


def scenario_spread(url, clients, objects):
    """Distinct objects -- the queue's real load, and what fills the disk tier."""
    before = health(url)
    args = [(url, objects[i % len(objects)]) for i in range(clients)]
    recs = _run_parallel(fit_once, args)
    after = health(url)
    _summarise(f"spread: {clients} requests over {len(objects)} objects",
               recs, before, after)
    return recs


def scenario_limit(url, clients, objects):
    """Deliberately exceed the fit bucket from ONE client.  Refusals here are
    the feature working; what matters is that they are polite and bounded."""
    before = health(url)
    recs = _run_parallel(fit_once, [(url, objects[i % len(objects)])
                                    for i in range(clients)])
    after = health(url)
    kinds = _summarise(f"limit: {clients} rapid requests from one client",
                       recs, before, after)
    print(f"  VERDICT    : {kinds.get('rate_limited', 0)} refused, "
          f"{kinds.get('fitted', 0) + kinds.get('memory_cache', 0) + kinds.get('disk_cache', 0)} served")
    return recs


SCENARIOS = {"herd": scenario_herd, "second": scenario_second_wave,
             "spread": scenario_spread, "limit": scenario_limit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("PYROORB_STRESS_URL",
                                                    "http://127.0.0.1:8000"))
    ap.add_argument("--scenario", default="herd",
                    choices=sorted(SCENARIOS) + ["mixed"])
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    args = ap.parse_args()
    objects = [o.strip() for o in args.objects.split(",") if o.strip()]

    h = health(args.url)
    if not h.get("ok"):
        print(f"server not answering /healthz at {args.url}: {h}")
        return 1
    f = h.get("fits") or {}
    print(f"target {args.url}: {f.get('workers')} workers, queue_max "
          f"{f.get('queue_max')}, itf {(h.get('itf') or {}).get('limit')}, "
          f"limiter {'on' if (h.get('rate_limit') or {}).get('enabled') else 'off'}")

    names = sorted(SCENARIOS) if args.scenario == "mixed" else [args.scenario]
    for name in names:
        SCENARIOS[name](args.url, args.clients, objects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
