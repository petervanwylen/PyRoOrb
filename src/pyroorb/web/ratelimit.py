"""Per-client rate limiting for the expensive endpoints.

WHY, GIVEN THERE IS ALREADY A QUEUE
-----------------------------------
The fit queue bounds what the box will DO at once, so no caller can push the
container over any more.  What it cannot do is decide WHOSE work gets done: the
queue is fair by arrival, so one script in a loop fills it and every real
visitor is told the server is busy.  That is a denial of service achieved
entirely through the front door, with the queue behaving exactly as designed.

So the queue protects the process and this protects the queue.  They are
different jobs and neither substitutes for the other.

WHAT IS AND IS NOT LIMITED
--------------------------
Only work that COSTS something is limited: starting a fit, pasting astrometry,
re-solving, drawing a random object, launching an ITF search.  The polling
endpoints (`/api/fit_result`, `/api/itf_result`, `/api/itf_trace`) are
deliberately exempt and must stay that way -- they exist because the work is
slow, a client is *told* to poll them, and limiting them would break the very
flow the queue introduced.  Static assets are exempt for the same reason: they
are cheap, and a browser fetches a dozen per page load.

IDENTITY
--------
Behind a hosted proxy the peer address is the proxy, so the client is taken
from the first hop of ``X-Forwarded-For`` when present.  That header is
trivially forged by anyone talking to the origin directly, so this is a
courtesy limit against runaway scripts and accidental hammering -- NOT a
security control.  Real abuse protection belongs at the edge (Cloudflare), which
can see the true peer; this is what keeps the site usable until it gets there,
and it is honest to say so rather than imply the origin can authenticate a
client it cannot see.
"""
import os
import time
import threading

# Buckets: name -> (capacity, refill per second).  Capacity is the burst a
# person can spend at once; the rate is what they sustain.  Sized for a human
# who is exploring, not for a client that is scripting: a visitor may fit half a
# dozen objects in a minute while reading, and will never do it for an hour.
_BUCKETS = {
    # Starting fits -- the queue's own currency.
    "fit": (int(os.environ.get("PYROORB_RL_FIT_BURST") or 12),
            float(os.environ.get("PYROORB_RL_FIT_RATE") or 0.2)),      # 12/min
    # Pasting astrometry or re-solving: a full fit from an arbitrary body.
    "paste": (int(os.environ.get("PYROORB_RL_PASTE_BURST") or 6),
              float(os.environ.get("PYROORB_RL_PASTE_RATE") or 0.1)),  # 6/min
    # ITF searches: minutes of CPU and hundreds of MB each.
    "itf": (int(os.environ.get("PYROORB_RL_ITF_BURST") or 3),
            float(os.environ.get("PYROORB_RL_ITF_RATE") or 0.02)),     # ~1/min
    # The random button: one click, one arbitrary object, always a cache miss.
    "random": (int(os.environ.get("PYROORB_RL_RANDOM_BURST") or 10),
               float(os.environ.get("PYROORB_RL_RANDOM_RATE") or 0.1)),
}

# How many distinct clients to track.  A bucket is three floats, so this is
# small -- but it must be BOUNDED, or the limiter becomes the memory leak it was
# added to prevent (one entry per source address, and a spike brings many).
_MAX_CLIENTS = int(os.environ.get("PYROORB_RL_MAX_CLIENTS") or 4096)

# A client with a full bucket and no recent activity is indistinguishable from
# one that never visited, so it can be forgotten.
_IDLE_SECS = 900.0

_DISABLED = (os.environ.get("PYROORB_RL_DISABLED") or "").lower() in ("1", "true", "yes")

_LOCK = threading.Lock()
_STATE = {}          # (client, bucket) -> [tokens, last_seen]


def client_of(handler):
    """Who to charge.  See IDENTITY in the module docstring -- forgeable, and
    deliberately used anyway."""
    xff = handler.headers.get("X-Forwarded-For")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first[:64]
    try:
        return str(handler.client_address[0])[:64]
    except Exception:
        return "?"


def _evict(now):
    """Forget idle clients; if still over the cap, forget the least recently
    seen.  Caller holds the lock."""
    for k, v in list(_STATE.items()):
        if now - v[1] > _IDLE_SECS:
            _STATE.pop(k, None)
    if len(_STATE) <= _MAX_CLIENTS:
        return
    for k, _v in sorted(_STATE.items(), key=lambda kv: kv[1][1])[
            : len(_STATE) - _MAX_CLIENTS]:
        _STATE.pop(k, None)


def check(client, bucket, cost=1.0):
    """Spend ``cost`` from ``client``'s ``bucket``.

    Returns ``(True, 0.0)`` when allowed, else ``(False, retry_after_secs)``.
    An unknown bucket name is allowed rather than refused: a typo in a route
    table must not lock users out of an endpoint.
    """
    if _DISABLED:
        return True, 0.0
    spec = _BUCKETS.get(bucket)
    if spec is None:
        return True, 0.0
    cap, rate = spec
    now = time.time()
    key = (client, bucket)
    with _LOCK:
        slot = _STATE.get(key)
        if slot is None:
            slot = _STATE[key] = [float(cap), now]
        tokens, last = slot
        tokens = min(cap, tokens + (now - last) * rate)
        if tokens >= cost:
            slot[0] = tokens - cost
            slot[1] = now
            out = (True, 0.0)
        else:
            slot[0] = tokens
            slot[1] = now
            # How long until the shortfall is refilled.  Reported so the client
            # can wait the right amount instead of retrying blindly.
            out = (False, round((cost - tokens) / rate, 1) if rate > 0 else 60.0)
        # Evict AFTER inserting, never before: trimming to the cap and then
        # adding one leaves the table permanently one over, which is the whole
        # bound this exists to hold.  The entry just touched has the newest
        # timestamp, so it is the last thing eviction would take.
        _evict(now)
        return out


def refund(client, bucket, cost=1.0):
    """Give back an allowance that bought no work.

    The limiter charges before the server knows what a request will cost, but
    what it is rationing is WORK -- and a request answered from cache, or one
    that merely joined a fit already running, causes none.  Charging those is
    backwards precisely where it matters most: the hot object everything else
    in this design exists to serve cheaply is the one a crowd hits hardest, and
    without a refund the hundredth viewer of a cached orbit is refused while the
    server sits idle.  Measured locally: 8 of 10 requests for a warm object were
    turned away with the fit table already holding the answer.
    """
    if _DISABLED or bucket not in _BUCKETS:
        return
    cap = _BUCKETS[bucket][0]
    with _LOCK:
        slot = _STATE.get((client, bucket))
        if slot is not None:
            slot[0] = min(float(cap), slot[0] + cost)


def refusal(bucket, retry_after):
    """The body for a refused request -- plain about what happened and when to
    come back, because a rate limit the user cannot interpret reads as the site
    being broken."""
    human = {"fit": "fits", "paste": "pasted fits", "itf": "ITF searches",
             "random": "random objects"}.get(bucket, "requests")
    return {"ok": False, "rate_limited": True, "retry_after": retry_after,
            "error": (f"You've started a lot of {human} in a short time. "
                      f"Please wait about {int(retry_after) + 1} seconds and try "
                      f"again — this keeps the server responsive for everyone.")}


def stats():
    with _LOCK:
        return {"enabled": not _DISABLED, "clients": len(_STATE),
                "max_clients": _MAX_CLIENTS,
                "buckets": {k: {"burst": v[0], "per_sec": v[1]}
                            for k, v in _BUCKETS.items()}}


def reset():
    """Tests only."""
    with _LOCK:
        _STATE.clear()
