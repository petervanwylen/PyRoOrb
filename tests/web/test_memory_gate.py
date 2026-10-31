"""``/api/memory``'s whole-heap pass is gated, and every report is throttled.

The tables pass is attribution over structures the server already holds.  The
MISC pass is different in kind: it walks every object the process can reach --
``gc.get_objects()`` plus each thread's frames -- and holds the GIL while it
does.  That makes an unauthenticated route a free way for anyone who can reach
the port to take CPU away from the ITF searches this page exists to keep alive,
and the origin guard is no help: it deliberately admits requests carrying no
``Origin`` at all, which is every non-browser client.

What has to be true at once, and is why this is table-driven:

* the deep pass opens for the configured password, in a header (a query
  parameter would land in access logs and proxy traces along the way);
* it opens for a LOOPBACK caller when no password is configured -- a developer
  on their own machine sees exactly what they saw before, with no setup;
* it does NOT open for a loopback-looking peer whose request came through a
  proxy: on a hosted deploy the proxy may well connect over loopback, and that
  is not somebody sitting at the console;
* a locked request still gets the tables -- a 403 would blank a page that can
  perfectly well render everything except one bucket;
* a wrong password says so, and does not silently behave like no password.
"""
import json
import os
import sys

import pytest  # noqa: E402

from pyroorb.web import server  # noqa: E402


class _Headers(dict):
    """HTTPMessage-alike: header lookup is case-insensitive."""

    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


PW = "hunter2-and-then-some"

CASES = [
    # (id, password_configured, headers, peer, allowed, reason)
    ("password-opens-it", PW, {"X-Memory-Password": PW}, "10.1.2.3",
     True, "ok"),
    ("wrong-password-says-so", PW, {"X-Memory-Password": "nope"}, "10.1.2.3",
     False, "password-wrong"),
    ("no-password-offered-when-one-is-set", PW, {}, "10.1.2.3",
     False, "password-required"),
    # ...even from the console: a configured password is the way in, full stop,
    # so the rule does not change depending on where you sit
    ("configured-password-applies-to-loopback-too", PW, {}, "127.0.0.1",
     False, "password-required"),
    ("loopback-when-none-configured", "", {}, "127.0.0.1", True, "ok"),
    ("loopback-v6-when-none-configured", "", {}, "::1", True, "ok"),
    ("remote-when-none-configured", "", {}, "203.0.113.9",
     False, "remote-no-password"),
    # a proxy that happens to connect over loopback is not the console
    ("proxied-loopback-is-not-local", "", {"X-Forwarded-For": "203.0.113.9"},
     "127.0.0.1", False, "remote-no-password"),
    ("password-offered-where-none-is-set", "", {"X-Memory-Password": PW},
     "127.0.0.1", False, "password-not-configured"),
    ("unparseable-peer", "", {}, "", False, "remote-no-password"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_misc_gate(case, monkeypatch):
    _id, configured, headers, peer, allowed, reason = case
    monkeypatch.setenv(server._MEMORY_PW_ENV, configured)
    got_allowed, got_reason = server._misc_allowed(_Headers(headers), peer)
    assert (got_allowed, got_reason) == (allowed, reason), _id


def test_every_refusal_reason_has_a_message():
    """A locked page must say WHICH key it is missing -- 'locked' alone leaves
    the reader with nothing to do about it."""
    reasons = {c[5] for c in CASES} - {"ok"}
    for r in reasons:
        assert server._MISC_LOCK_MESSAGE.get(r), r


# --------------------------------------------------------------------------
# The route itself: locked answers 200 with the tables, and is throttled.
# --------------------------------------------------------------------------
class _FakeHandler(server.Handler):
    """Handler with the socket machinery replaced -- ``_memory`` is pure
    request-in/payload-out, so it can be exercised without a live server."""

    def __init__(self, headers, peer):
        self.headers = _Headers(headers)
        self.client_address = (peer, 12345)
        self.sent = []

    def _send(self, code, body, ctype="application/json"):
        self.sent.append((code, body))


@pytest.fixture(autouse=True)
def _clear_cache():
    server._MEMORY_CACHE.update({"t": 0.0, "misc": False, "payload": None})
    yield
    server._MEMORY_CACHE.update({"t": 0.0, "misc": False, "payload": None})


def _report(with_misc):
    """A stand-in for the real walk: this test is about the gate, not the walk."""
    return {"ok": True, "when": "now", "tables": [],
            "misc": {"mb": 1.0} if with_misc else None}


def test_locked_request_still_gets_the_tables(monkeypatch):
    monkeypatch.setenv(server._MEMORY_PW_ENV, PW)
    monkeypatch.setattr(server.webmem, "report",
                        lambda with_misc=True: _report(with_misc))
    h = _FakeHandler({}, "203.0.113.9")
    h._memory({})
    code, body = h.sent[-1]
    assert code == 200, "a locked deep pass is not an error -- the page renders"
    assert body["ok"] is True and body["misc"] is None
    assert body["misc_locked"]["reason"] == "password-required"
    assert body["misc_locked"]["password_required"] is True


def test_unlocked_request_gets_the_whole_heap(monkeypatch):
    monkeypatch.setenv(server._MEMORY_PW_ENV, PW)
    monkeypatch.setattr(server.webmem, "report",
                        lambda with_misc=True: _report(with_misc))
    h = _FakeHandler({"X-Memory-Password": PW}, "203.0.113.9")
    h._memory({})
    _code, body = h.sent[-1]
    assert body["misc"] == {"mb": 1.0}
    assert "misc_locked" not in body


def test_no_password_configured_offers_no_password_box(monkeypatch):
    """Where no password exists, the UI must not invite the user to guess at a
    lock with no key: `password_required` stays False and the message says why."""
    monkeypatch.setenv(server._MEMORY_PW_ENV, "")
    monkeypatch.setattr(server.webmem, "report",
                        lambda with_misc=True: _report(with_misc))
    h = _FakeHandler({}, "203.0.113.9")
    h._memory({})
    _code, body = h.sent[-1]
    assert body["misc_locked"]["password_required"] is False
    assert server._MEMORY_PW_ENV in body["misc_locked"]["message"]


def test_unauthenticated_reports_are_throttled(monkeypatch):
    """The cost bound: N anonymous requests inside the interval pay for ONE
    walk.  The page's own auto-refresh is 10 s, so nobody watching it notices;
    a loop hitting the route stops being a way to spend the server's CPU."""
    monkeypatch.setenv(server._MEMORY_PW_ENV, "")
    calls = []

    def _counted(with_misc=True):
        calls.append(with_misc)
        return _report(with_misc)

    monkeypatch.setattr(server.webmem, "report", _counted)
    for _ in range(5):
        h = _FakeHandler({}, "203.0.113.9")
        h._memory({})
    assert len(calls) == 1, calls
    assert h.sent[-1][1]["cached_age_s"] is not None, \
        "a re-served report must say it is cached, not pass as fresh"


def test_the_operator_always_gets_a_fresh_walk(monkeypatch):
    """Throttling is a bound on anonymous cost, not on the person who typed the
    password -- they asked for the measurement, and a cached one is not it."""
    monkeypatch.setenv(server._MEMORY_PW_ENV, PW)
    calls = []

    def _counted(with_misc=True):
        calls.append(with_misc)
        return _report(with_misc)

    monkeypatch.setattr(server.webmem, "report", _counted)
    for _ in range(3):
        h = _FakeHandler({"X-Memory-Password": PW}, "203.0.113.9")
        h._memory({})
    assert len(calls) == 3, calls


def test_tables_only_and_deep_are_cached_separately(monkeypatch):
    """A tables-only reader must not be handed a cached deep report, nor the
    other way about -- they are different questions, not different freshness."""
    monkeypatch.setenv(server._MEMORY_PW_ENV, "")
    calls = []

    def _counted(with_misc=True):
        calls.append(with_misc)
        return _report(with_misc)

    monkeypatch.setattr(server.webmem, "report", _counted)
    _FakeHandler({}, "127.0.0.1")._memory({"tables": ["1"]})
    h = _FakeHandler({}, "127.0.0.1")
    h._memory({})
    assert calls == [False, True], calls
    assert h.sent[-1][1]["misc"] == {"mb": 1.0}


class _WireHandler(server.Handler):
    """The REAL ``_send`` over a byte buffer, so the encoding is exercised."""

    def __init__(self):
        import io
        self.wfile = io.BytesIO()
        self.request_version = "HTTP/1.1"
        self.requestline = "GET /api/memory HTTP/1.1"
        self.client_address = ("127.0.0.1", 0)
        self.command = "GET"

    def log_message(self, *a):
        pass

    def reply(self):
        raw = self.wfile.getvalue().decode("utf-8")
        head, _, body = raw.partition("\r\n\r\n")
        return int(head.split()[1]), json.loads(body)


def test_numpy_numbers_in_a_reply_are_encoded_not_fatal():
    # Found live 2026-09-27: one np.int64 (a sweep's progress index) in the
    # memory report made json.dumps raise TypeError inside _send, the handler
    # thread died with nothing written, and the hosted proxy answered with its
    # own 502 "Application failed to respond" on every later report.
    import numpy as np
    h = _WireHandler()
    h._send(200, {"ok": True, "at": np.int64(7), "x": np.float64(1.5),
                  "v": np.arange(3), "bad": np.float64("nan")})
    code, d = h.reply()
    assert code == 200
    assert d == {"ok": True, "at": 7, "x": 1.5, "v": [0, 1, 2], "bad": None}


def test_an_unencodable_reply_is_still_an_answer():
    h = _WireHandler()
    h._send(200, {"ok": True, "thing": object()})
    code, d = h.reply()
    assert code == 500 and d["ok"] is False and "unencodable" in d["error"]
