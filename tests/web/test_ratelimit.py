"""Per-client rate limiting (src/pyroorb/web/ratelimit.py) and the request body cap.

The queue bounds what the box will DO; this bounds whose work gets done.  One
script in a loop can fill a fair queue and lock every real visitor out while the
queue behaves exactly as designed, so the two guards are not substitutes.
"""
import os
import sys

import pytest

_WEB = os.path.dirname(__import__("pyroorb.web", fromlist=["_"]).__file__)
from pyroorb.web import ratelimit  # noqa: E402
from pyroorb.web import server     # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    ratelimit.reset()
    yield
    ratelimit.reset()


class _FakeHandler:
    """Just enough of BaseHTTPRequestHandler for client_of."""
    def __init__(self, xff=None, peer="10.0.0.7"):
        self.headers = {"X-Forwarded-For": xff} if xff else {}
        self.client_address = (peer, 12345)


def test_a_burst_is_allowed_then_refused(monkeypatch):
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (3, 0.0))
    for i in range(3):
        ok, _ = ratelimit.check("a", "fit")
        assert ok, f"call {i} should be inside the burst"
    ok, retry = ratelimit.check("a", "fit")
    assert not ok and retry > 0


def test_clients_are_independent(monkeypatch):
    """One noisy caller must not spend anybody else's allowance -- that would
    make the limiter itself the outage."""
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (2, 0.0))
    assert ratelimit.check("noisy", "fit")[0]
    assert ratelimit.check("noisy", "fit")[0]
    assert not ratelimit.check("noisy", "fit")[0]
    assert ratelimit.check("quiet", "fit")[0], "a different client must be unaffected"


def test_buckets_are_independent(monkeypatch):
    """Exhausting ITF searches must not stop the person looking at orbits."""
    monkeypatch.setitem(ratelimit._BUCKETS, "itf", (1, 0.0))
    assert ratelimit.check("a", "itf")[0]
    assert not ratelimit.check("a", "itf")[0]
    assert ratelimit.check("a", "fit")[0]


def test_tokens_refill_over_time(monkeypatch):
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (1, 100.0))   # refills fast
    assert ratelimit.check("a", "fit")[0]
    ok = False
    import time
    for _ in range(50):
        time.sleep(0.01)
        ok, _ = ratelimit.check("a", "fit")
        if ok:
            break
    assert ok, "the bucket never refilled"


def test_an_unknown_bucket_is_allowed():
    """A typo in a route table must not lock users out of an endpoint."""
    assert ratelimit.check("a", "no-such-bucket")[0]


def test_disabled_allows_everything(monkeypatch):
    monkeypatch.setattr(ratelimit, "_DISABLED", True)
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (0, 0.0))
    assert ratelimit.check("a", "fit")[0]


class _ProxiedHandler:
    def __init__(self, headers, peer="10.0.0.7"):
        self.headers = headers
        self.client_address = (peer, 12345)


def test_client_identity_is_what_the_proxy_added_never_what_the_client_wrote(monkeypatch):
    """Behind a proxy the peer is the proxy, so every visitor would share one bucket.
    But only what the proxy ADDS identifies the client: X-Real-IP (the edge replaces
    any copy the client sent), else the LAST X-Forwarded-For hop.  The first hop is
    the client's own text -- read, it made each request a new client and walked past
    every limit and password lockout keyed on it."""
    monkeypatch.delenv("PYROORB_CLIENT_IP", raising=False)
    assert ratelimit.client_of(_FakeHandler(xff="203.0.113.9, 10.1.1.1")) == "10.1.1.1"
    assert ratelimit.client_of(_FakeHandler(xff="198.51.100.4")) == "198.51.100.4"
    assert ratelimit.client_of(_FakeHandler(xff=" , ")) == "10.0.0.7"
    assert ratelimit.client_of(_FakeHandler()) == "10.0.0.7"
    forged = [ratelimit.client_of(_ProxiedHandler(
        {"X-Forwarded-For": f"1.2.3.{k}, 203.0.113.9"})) for k in range(5)]
    assert set(forged) == {"203.0.113.9"}
    both = _ProxiedHandler({"X-Real-IP": "192.0.2.5", "X-Forwarded-For": "1.1.1.1, 9.9.9.9"})
    assert ratelimit.client_of(both) == "192.0.2.5"


def test_client_ip_peer_mode_ignores_every_header(monkeypatch):
    """A server reached with no proxy in front: every header is the client's own."""
    monkeypatch.setenv("PYROORB_CLIENT_IP", "peer")
    h = _ProxiedHandler({"X-Real-IP": "192.0.2.5", "X-Forwarded-For": "1.1.1.1"})
    assert ratelimit.client_of(h) == "10.0.0.7"


def test_the_client_table_is_bounded(monkeypatch):
    """A limiter that grows one entry per source address is the leak it exists
    to prevent."""
    monkeypatch.setattr(ratelimit, "_MAX_CLIENTS", 50)
    for i in range(500):
        ratelimit.check(f"c{i}", "fit")
    assert len(ratelimit._STATE) <= 50


def test_a_flood_of_forged_clients_never_evicts_the_shared_slot(monkeypatch):
    """The all-clients bucket bounds an endpoint whatever identity each request claims. Kept in
    the same table as the forgeable per-client slots, a few thousand forged requests to ANY
    endpoint pushed it out as the least recently seen, and it came back full."""
    monkeypatch.setattr(ratelimit, "_DISABLED", False)
    monkeypatch.setattr(ratelimit, "_MAX_CLIENTS", 50)
    monkeypatch.setitem(ratelimit._BUCKETS, "_all", (2, 0.0))
    assert ratelimit.check(ratelimit.EVERYONE, "_all")[0]
    assert ratelimit.check(ratelimit.EVERYONE, "_all")[0]
    for i in range(500):
        ratelimit.check(f"forged{i}", "fit")
    assert len(ratelimit._STATE) <= 51
    assert not ratelimit.check(ratelimit.EVERYONE, "_all")[0]


def test_no_client_is_ever_identified_as_everyone():
    who = ratelimit.client_of(_FakeHandler(xff=ratelimit.EVERYONE))
    assert who != ratelimit.EVERYONE and ratelimit.client_of(_FakeHandler(peer="*")) != "*"


def test_an_idle_client_is_forgotten_only_once_its_bucket_has_refilled(monkeypatch):
    """Forgetting an emptied slow bucket after the idle time handed back a full burst: ten an
    hour became ten every quarter hour. A refilled one is still forgotten (the table's bound)."""
    monkeypatch.setattr(ratelimit, "_DISABLED", False)
    monkeypatch.setitem(ratelimit._BUCKETS, "_slow", (10, 10 / 3600.0))
    clock = [1_000_000.0]
    monkeypatch.setattr(ratelimit.time, "time", lambda: clock[0])
    assert all(ratelimit.check("c", "_slow")[0] for _ in range(10))
    clock[0] += ratelimit._IDLE_SECS + 1
    ratelimit.check("other", "fit")                     # any call runs the eviction
    got = sum(ratelimit.check("c", "_slow")[0] for _ in range(10))
    assert got == 2                                      # 901 s at 10/h refills 2.5, not 10
    ratelimit.check("d", "fit")
    clock[0] += ratelimit._IDLE_SECS + 1                 # "fit" refills 12 in a minute
    ratelimit.check("e", "fit")
    assert ("d", "fit") not in ratelimit._STATE and ("c", "_slow") in ratelimit._STATE


def test_refusal_says_when_to_come_back():
    body = ratelimit.refusal("fit", 12.0)
    assert body["ok"] is False and body["rate_limited"] is True
    assert body["retry_after"] == 12.0
    assert "seconds" in body["error"]


# --- refunds ----------------------------------------------------------------

def test_a_refund_returns_the_allowance(monkeypatch):
    """What the limiter rations is WORK.  A request answered from cache, or one
    that joined a fit already running, causes none -- and charging those refuses
    the crowd on the hot object precisely when the server is idle.  Measured on
    a live server before this existed: 8 of 10 requests for a WARM object were
    turned away with the answer already in memory."""
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (2, 0.0))
    assert ratelimit.check("a", "fit")[0]
    assert ratelimit.check("a", "fit")[0]
    assert not ratelimit.check("a", "fit")[0]
    ratelimit.refund("a", "fit")
    assert ratelimit.check("a", "fit")[0], "the refunded token was not returned"


def test_a_refund_never_exceeds_the_burst(monkeypatch):
    """Refunds must not become free allowance, or a cache-heavy client would
    accumulate an unbounded budget to spend on real fits later."""
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (2, 0.0))
    ratelimit.check("a", "fit")
    for _ in range(20):
        ratelimit.refund("a", "fit")
    assert ratelimit.check("a", "fit")[0]
    assert ratelimit.check("a", "fit")[0]
    assert not ratelimit.check("a", "fit")[0], "refunds inflated the bucket"


def test_a_refund_for_an_unknown_client_is_harmless():
    ratelimit.refund("never-seen", "fit")      # must not raise or create state
    assert ("never-seen", "fit") not in ratelimit._STATE


def test_free_requests_are_refunded_by_the_handler():
    """The server refunds when the queue says the request cost nothing."""
    seen = []
    monkeypatch_target = server.Handler._refund_if_free

    class _H:
        headers = {}
        client_address = ("10.0.0.1", 1)
        _RATE_BUCKETS = server.Handler._RATE_BUCKETS
        _refund_if_free = monkeypatch_target

    h = _H()
    import pyroorb.web.ratelimit as rl
    real = rl.refund
    rl.refund = lambda c, b, cost=1.0: seen.append((c, b))
    try:
        h._refund_if_free("/api/fit_start", {"ok": True, "cached": "disk"})
        h._refund_if_free("/api/fit_start", {"ok": True, "coalesced": True})
        h._refund_if_free("/api/fit_start", {"ok": True})          # real work
        h._refund_if_free("/api/itf_defaults", {"cached": "disk"})  # not limited
    finally:
        rl.refund = real
    assert seen == [("10.0.0.1", "fit"), ("10.0.0.1", "fit")], seen


def test_a_busy_refusal_is_refunded():
    """A queue-full refusal ran no fit.  Charging it means a saturated server
    also rations the people it just turned away -- told to slow down for work
    that never happened -- and the busy and rate-limited counts blur together
    exactly when you need to tell them apart."""
    seen = []

    class _H:
        headers = {}
        client_address = ("10.0.0.2", 1)
        _RATE_BUCKETS = server.Handler._RATE_BUCKETS
        _refund_if_free = server.Handler._refund_if_free

    import pyroorb.web.ratelimit as rl
    real = rl.refund
    rl.refund = lambda c, b, cost=1.0: seen.append(b)
    try:
        _H()._refund_if_free("/api/fit_start", {"ok": False, "busy": True})
    finally:
        rl.refund = real
    assert seen == ["fit"], seen


# --- routing ----------------------------------------------------------------

def test_the_polling_routes_are_never_limited():
    """A client polls these BECAUSE the work is slow; limiting them would break
    the flow the queue introduced."""
    buckets = server.Handler._RATE_BUCKETS
    for path in ("/api/fit_result", "/api/itf_result", "/api/itf_trace",
                 "/api/session", "/api/itf_defaults"):
        assert path not in buckets, f"{path} must not be rate limited"


def test_every_costly_route_is_limited():
    buckets = server.Handler._RATE_BUCKETS
    for path in ("/api/fit", "/api/fit_start", "/api/fit_text", "/api/add_obs",
                 "/api/reauto", "/api/itf_search", "/api/random_object"):
        assert path in buckets, f"{path} starts real work and must be limited"


def test_the_listen_backlog_is_raised_from_the_stdlib_default():
    """socketserver defaults request_queue_size to FIVE.

    Everything this server does to answer gracefully under load -- the queue,
    the cache, the limiter -- sits downstream of an accept() that a five-deep
    backlog never performs, so a burst is refused by the kernel and the visitor
    sees a connection reset instead.  Measured locally: 60 simultaneous requests
    produced a ConnectionResetError at the default and none at 128.
    """
    import socketserver
    assert socketserver.TCPServer.request_queue_size == 5, \
        "the stdlib default changed; re-check the reasoning above"
    src = open(os.path.join(_WEB, "server.py"), encoding="utf-8").read()
    assert "request_queue_size" in src, "the backlog is left at the stdlib default"


def test_the_body_cap_is_sane():
    """Unbounded Content-Length meant one POST could ask the process to
    allocate anything it liked -- before any queue could see it."""
    assert 1024 * 1024 <= server._MAX_BODY_BYTES <= 64 * 1024 * 1024


def test_a_registered_bucket_limits_and_keeps_its_first_size(monkeypatch):
    monkeypatch.setattr(ratelimit, "_DISABLED", False)
    monkeypatch.setitem(ratelimit._BUCKETS, "_addon_test", (2, 0.0))
    ratelimit.register("_addon_test", 50, 1.0)          # already there: unchanged
    assert ratelimit._BUCKETS["_addon_test"] == (2, 0.0)
    monkeypatch.delitem(ratelimit._BUCKETS, "_addon_test")
    ratelimit.register("_addon_test", 2, 0.0)
    try:
        assert ratelimit.check("c", "_addon_test")[0]
        assert ratelimit.check("c", "_addon_test")[0]
        assert not ratelimit.check("c", "_addon_test")[0]
    finally:
        ratelimit._BUCKETS.pop("_addon_test", None)
