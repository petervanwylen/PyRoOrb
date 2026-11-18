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
Behind a hosted proxy the socket peer is the proxy, so the client is read from what
the proxy adds (:func:`client_of`, ``PYROORB_CLIENT_IP``):

* ``X-Real-IP`` first: the hosting edge sets it and replaces any copy the client sent;
* else the LAST ``X-Forwarded-For`` hop -- the one the nearest proxy appended.  Every
  earlier hop is whatever the client wrote, so reading the first hop
  let one caller be a new client on every request and walk past every limit and
  password lockout keyed on it;
* else the socket peer.

``PYROORB_CLIENT_IP=peer`` uses the socket peer alone, for a server reached with no
proxy in front, where every header is the client's own.  A CDN in front of the host
turns the edge's view of the client into the CDN's address; the limits then key on
that, and the identity must come from the CDN's own header instead.
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
    # Submitting a linkage: a person sending what they found, a few an hour at most.
    "submit": (int(os.environ.get("PYROORB_RL_SUBMIT_PER_HOUR") or 30),
               int(os.environ.get("PYROORB_RL_SUBMIT_PER_HOUR") or 30) / 3600.0),
}

# How many distinct clients to track.  A bucket is three floats, so this is
# small -- but it must be BOUNDED, or the limiter becomes the memory leak it was
# added to prevent (one entry per source address, and a spike brings many).
_MAX_CLIENTS = int(os.environ.get("PYROORB_RL_MAX_CLIENTS") or 4096)

# A client with a full bucket and no recent activity is indistinguishable from
# one that never visited, so it can be forgotten -- but only once its bucket HAS
# refilled. A slow bucket (one an hour) is not full after this long, and
# forgetting it early hands the client a fresh burst: ten submissions every
# quarter hour against a stated ten an hour.
_IDLE_SECS = 900.0

_DISABLED = (os.environ.get("PYROORB_RL_DISABLED") or "").lower() in ("1", "true", "yes")

_LOCK = threading.Lock()
_STATE = {}          # (client, bucket) -> [tokens, last_seen]

#: The client every caller shares: ``check(EVERYONE, bucket)`` bounds an endpoint for all
#: clients together, whatever identity each claims.  Its slots are never evicted to make room
#: for others -- the table is filled by forgeable identities, and evicting the shared slot
#: would hand everyone a fresh burst after a few thousand forged requests to any endpoint.
EVERYONE = "*"


def _header(handler, name):
    try:
        return (handler.headers.get(name) or "").strip()
    except Exception:
        return ""


def client_of(handler):
    """Who to charge: see IDENTITY in the module docstring.  Never :data:`EVERYONE`
    (a header naming it is charged as its own client)."""
    who = None
    if (os.environ.get("PYROORB_CLIENT_IP") or "auto").strip().lower() != "peer":
        who = _header(handler, "X-Real-IP")
        if not who:
            hops = [h.strip() for h in _header(handler, "X-Forwarded-For").split(",")]
            hops = [h for h in hops if h]
            who = hops[-1] if hops else None
    if who:
        who = who[:64]
    else:
        try:
            who = str(handler.client_address[0])[:64]
        except Exception:
            who = "?"
    return "?" + who if who == EVERYONE else who


def _refilled(bucket, slot, now):
    """Whether ``slot`` (tokens, last_seen) of ``bucket`` is full again by ``now``."""
    spec = _BUCKETS.get(bucket)
    if spec is None:
        return True
    cap, rate = spec
    return slot[0] + (now - slot[1]) * rate >= cap


def _evict(now):
    """Forget idle clients whose bucket has refilled; if still over the cap,
    forget the least recently seen.  Caller holds the lock."""
    for k, v in list(_STATE.items()):
        if now - v[1] > _IDLE_SECS and _refilled(k[1], v, now):
            _STATE.pop(k, None)
    if len(_STATE) <= _MAX_CLIENTS:
        return
    clients = [kv for kv in _STATE.items() if kv[0][0] != EVERYONE]   # shared slots stay
    for k, _v in sorted(clients, key=lambda kv: kv[1][1])[
            : len(_STATE) - _MAX_CLIENTS]:
        _STATE.pop(k, None)


def register(bucket, burst, rate):
    """Add a bucket for an endpoint the server learns about at start-up (a web add-on):
    ``burst`` requests at once, ``rate`` per second sustained. A bucket that already
    exists keeps its size, so registering twice (a reloaded add-on) changes nothing."""
    with _LOCK:
        _BUCKETS.setdefault(str(bucket), (int(burst), float(rate)))


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
             "random": "random objects", "submit": "linkage submissions",
             "idqueue_docs": "identification submissions",
             "idqueue_links": "identification links"}.get(bucket, "requests")
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
