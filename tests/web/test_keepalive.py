"""HTTP/1.1 keep-alive correctness: an early refusal in do_POST must not leave
an undrained request body on the wire to corrupt whatever the client sends
next on the same connection.

This needs a REAL socket and a REAL running server -- the bug lives in the
interaction between BaseHTTPRequestHandler's connection-reuse machinery and an
early ``return`` before the body is read, which no in-process call into
``do_POST`` can exercise.

Found on the live deploy: a burst of concurrent POSTs dropped roughly 1 in 8 to
1 in 12, reported by clients as "RemoteDisconnected: Remote end closed
connection without response".  Reproduced exactly with a raw socket: request 2
was refused (429) before its JSON body was read; request 3's response line came
back as ``Bad request syntax ('{"desig": "2003 LW2"}POST /api/session
HTTP/1.1')`` -- request 2's undrained body, verbatim, prepended to request 3's
request line.  Through a proxy that turns a mangled response into a dropped
connection instead of surfacing the 400, that IS the live-deploy symptom.
"""
import json
import os
import re
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from pyroorb.web import server  # noqa: E402


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    """A real server.Handler on an ephemeral loopback port."""
    monkeypatch.setenv("PYROORB_FIT_CACHE_DIR", str(tmp_path / "cache"))
    from pyroorb.web import fitdisk
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        yield httpd.server_address
    finally:
        httpd.shutdown()
        th.join(timeout=5)


def _read_response(sock, timeout=8):
    """(status_line, body) off a raw socket, honouring Content-Length.  Body
    is None (not empty) when the peer closed before sending a full response --
    the caller needs to tell "closed cleanly" apart from "sent nothing"."""
    sock.settimeout(timeout)
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            return (buf.split(b"\r\n", 1)[0] if buf else None), None
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    m = re.search(rb"Content-Length: (\d+)", head, re.I)
    need = int(m.group(1)) if m else 0
    while len(rest) < need:
        chunk = sock.recv(4096)
        if not chunk:
            break
        rest += chunk
    return head.split(b"\r\n", 1)[0], rest[:need]


def _send_post(sock, path, body, origin=None):
    b = json.dumps(body).encode()
    headers = (f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
              f"Content-Type: application/json\r\nContent-Length: {len(b)}\r\n"
              f"Connection: keep-alive\r\n")
    if origin:
        headers += f"Origin: {origin}\r\n"
    sock.sendall((headers + "\r\n").encode() + b)


def test_a_refused_post_never_corrupts_the_next_request_on_the_connection(
        running_server):
    host, port = running_server
    s = socket.create_connection((host, port), timeout=10)
    try:
        # req1: a cross-origin POST -- refused by _cross_origin_refused BEFORE
        # the body is read.  Deterministic and needs no engine/network work,
        # unlike routing this through the rate limiter would.
        _send_post(s, "/api/session", {"job": "leftover-body-marker"},
                  origin="https://evil.example")
        status1, _body1 = _read_response(s)
        assert status1 is not None and b"403" in status1

        # req2: a normal request on the SAME connection.  Pre-fix, req1's
        # undrained body ("...leftover-body-marker...") corrupts this
        # request's parse.  Post-fix, the server closed the connection after
        # req1, so this must fail CLEANLY -- never a "Bad request syntax"
        # response built from someone else's leftover bytes.
        try:
            _send_post(s, "/api/session", {"job": "innocent-request"})
            status2, body2 = _read_response(s, timeout=3)
        except ConnectionError:     # reset, broken pipe, or Windows' abort
            status2, body2 = None, None  # a clean reset is a PASS

        if status2 is not None:
            assert b"400" not in status2, (
                f"corrupted parse survived the fix: {status2!r} {body2!r}")
            assert b"leftover-body-marker" not in (body2 or b""), (
                "request 1's body leaked into request 2's response")
    finally:
        s.close()


def test_successful_requests_still_reuse_the_connection(running_server):
    """The fix must not defeat keep-alive on the HAPPY path -- only the
    early-refusal branches force a close."""
    host, port = running_server
    s = socket.create_connection((host, port), timeout=10)
    try:
        _send_post(s, "/api/session", {"job": "a"})
        status1, _ = _read_response(s)
        assert status1 is not None and b"200" in status1

        _send_post(s, "/api/session", {"job": "b"})
        status2, body2 = _read_response(s, timeout=4)
        assert status2 is not None and b"200" in status2, (
            "a second request on the same connection failed -- keep-alive "
            "broke on the path that must keep working")
        assert json.loads(body2).get("expired") is True
    finally:
        s.close()


def test_a_rate_limited_post_also_closes_rather_than_corrupts(
        running_server, monkeypatch):
    """The same hazard exists on the rate-limit branch, not just cross-origin
    -- exercised directly since it needs no real fit."""
    from pyroorb.web import ratelimit
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (0, 0.0))  # always refused
    host, port = running_server
    s = socket.create_connection((host, port), timeout=10)
    try:
        _send_post(s, "/api/fit_start", {"desig": "leftover-fit-marker"})
        status1, _ = _read_response(s)
        assert status1 is not None and b"429" in status1

        try:
            _send_post(s, "/api/session", {"job": "x"})
            status2, body2 = _read_response(s, timeout=3)
        except ConnectionError:     # reset, broken pipe, or Windows' abort
            status2, body2 = None, None

        if status2 is not None:
            assert b"400" not in status2
            assert b"leftover-fit-marker" not in (body2 or b"")
    finally:
        s.close()
