"""What a hostile request can no longer do: read past the body cap, pass for a new client
on every request, guess passwords without end, read the server's database errors, or
put markup into the page through a label it chose."""
import io
import re
from pathlib import Path

import pytest

from pyroorb.web import addons, errors, pwguard, ratelimit, relay, server

STATIC = Path(server.__file__).resolve().parent / "static"


# ------------------------------------------------------------------ Content-Length
@pytest.mark.parametrize("raw, want", [
    (None, 0), ("", 0), ("  ", 0), ("0", 0), ("12", 12), (" 7 ", 7),
    ("-1", None), ("+5", None), ("1e3", None), ("abc", None), ("5 5", None),
    ("٥", None),                     # an Arabic-Indic five: int() takes it
])
def test_declared_length_is_a_plain_non_negative_decimal(raw, want):
    assert addons.declared_length({} if raw is None else {"Content-Length": raw}) == want


class _BodyHandler(server.Handler):
    """``_read_body`` without a socket: the body is whatever the client keeps sending."""

    def __init__(self, length, body=b"x" * 1000):
        self.headers = {"Content-Length": length}
        self.rfile = io.BytesIO(body)
        self.close_connection = False
        self.sent = []

    def _send(self, code, body, ctype="application/json", extra_headers=()):
        self.sent.append((code, body))


def test_a_negative_length_is_refused_not_read_to_the_end():
    """``int("-1")`` passed the ``n > cap`` test and ``rfile.read(-1)`` read until the client
    hung up: the cap did not apply at all."""
    h = _BodyHandler("-1")
    assert h._read_body() is None
    assert h.sent and h.sent[0][0] == 400 and h.close_connection
    assert h.rfile.tell() == 0                      # not one byte read


def test_an_honest_length_reads_exactly_that_many_bytes():
    h = _BodyHandler("5")
    assert h._read_body() == b"xxxxx" and not h.sent


def test_an_oversized_length_is_still_a_413(monkeypatch):
    monkeypatch.setattr(server, "_MAX_BODY_BYTES", 10)
    h = _BodyHandler("11")
    assert h._read_body() is None and h.sent[0][0] == 413


def test_two_different_lengths_are_refused():
    from email.message import Message
    m = Message()
    m["Content-Length"] = "5"
    m["Content-Length"] = "50"
    assert addons.declared_length(m) is None
    m = Message()
    m["Content-Length"] = "5"
    m["Content-Length"] = "5"
    assert addons.declared_length(m) == 5


def _exchange(raw):
    """Send ``raw`` to a live Handler on a loopback socket; every byte it answers."""
    import socket
    import threading
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(srv.server_address, timeout=10) as s:
            s.sendall(raw)
            out = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                out += chunk
        return out
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_chunked_body_is_refused_not_run_as_a_second_request():
    """A chunked POST was read as length 0: its body stayed on the keep-alive connection
    and was parsed as the next request (here a hidden GET /healthz)."""
    hidden = b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n"
    body = b"%x\r\n" % len(hidden) + hidden + b"\r\n0\r\n\r\n"
    out = _exchange(b"POST /api/itf_cancel HTTP/1.1\r\nHost: x\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n" + body)
    assert out.startswith(b"HTTP/1.1 411") and out.count(b"HTTP/1.1 ") == 1


def test_conflicting_lengths_close_the_connection():
    out = _exchange(b"POST /api/itf_cancel HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
                    b"Content-Length: 40\r\n\r\n{}GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
    assert out.startswith(b"HTTP/1.1 400") and out.count(b"HTTP/1.1 ") == 1


# ------------------------------------------------------------------ client identity
class _Req:
    def __init__(self, headers, peer="10.0.0.7"):
        self.headers = headers
        self.client_address = (peer, 1)


def test_a_forged_first_hop_never_makes_a_new_client(monkeypatch):
    """Every request below claims a new first hop; the proxy's own last hop is the same,
    so the limiter sees ONE client and refuses it once its burst is spent."""
    monkeypatch.delenv("PYROORB_CLIENT_IP", raising=False)
    monkeypatch.setattr(ratelimit, "_DISABLED", False)
    monkeypatch.setitem(ratelimit._BUCKETS, "fit", (3, 0.0))
    allowed = [ratelimit.check(ratelimit.client_of(
        _Req({"X-Forwarded-For": f"1.2.3.{k}, 203.0.113.9"})), "fit")[0] for k in range(6)]
    assert allowed == [True, True, True, False, False, False]


# ------------------------------------------------------------------ wrong passwords
class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_a_caller_is_locked_out_after_its_wrong_passwords_then_let_back_in():
    clock = _Clock()
    g = pwguard.FailureGuard(per_client=3, everyone=100, window_sec=600, clock=clock)
    for _ in range(3):
        assert g.refused("a") is None
        g.failed("a")
    assert g.refused("a") and g.refused("b") is None      # only "a" is locked
    clock.t += 601
    assert g.refused("a") is None                         # its failures left the window


def test_many_addresses_together_are_held_to_the_server_wide_count():
    """A caller with many addresses gets ``per_client`` guesses at each; the backstop
    bounds them all together."""
    g = pwguard.FailureGuard(per_client=5, everyone=10, window_sec=600, clock=_Clock())
    for k in range(10):
        assert g.refused(f"10.0.0.{k}") is None
        g.failed(f"10.0.0.{k}")
    assert "server" in g.refused("192.0.2.1")


def test_a_right_password_clears_only_that_caller():
    g = pwguard.FailureGuard(per_client=2, everyone=100, clock=_Clock())
    g.failed("a")
    g.failed("a")
    g.succeeded("a")
    assert g.refused("a") is None


def test_the_failure_table_is_bounded():
    g = pwguard.FailureGuard(per_client=5, everyone=10 ** 6, max_clients=50, clock=_Clock())
    for k in range(500):
        g.failed(f"c{k}")
    assert len(g._fails) <= 50


# ------------------------------------------------------------------ error text
class _DbError(Exception):
    pass


_DbError.__module__ = "psycopg2.errors"


class _Wrapped(RuntimeError):
    pass


def test_an_engine_error_reaches_the_caller_as_it_is():
    assert errors.describe(ValueError("no observations found for 2099 ZZ")) == \
        "ValueError: no observations found for 2099 ZZ"


@pytest.mark.parametrize("ex", [
    _DbError('relation "itf.secret" does not exist\nLINE 1: SELECT * FROM itf.secret'),
    ConnectionError("HTTPSConnectionPool(host='10.1.2.3', port=5432)"),
    FileNotFoundError(2, "No such file", "/srv/data/private.json"),
    RuntimeError('connection to server at "db.internal" (10.1.2.3), port 5432 failed: '
                 'FATAL: password authentication failed for user "itf_rw"'),
])
def test_a_server_error_reaches_the_caller_as_its_class_only(ex, capsys):
    out = errors.describe(ex)
    assert out.startswith(type(ex).__name__ + ":") and "internal error" in out
    for secret in ("itf.secret", "10.1.2.3", "/srv/data", "db.internal", "itf_rw"):
        assert secret not in out
    assert capsys.readouterr().err                         # ...and the server log has it


def test_an_error_raised_from_a_database_error_is_redacted_too():
    try:
        try:
            raise _DbError("could not connect to db.internal")
        except _DbError as inner:
            raise _Wrapped("database unreachable: could not connect to db.internal") from inner
    except _Wrapped as ex:
        assert "db.internal" not in errors.describe(ex)


def test_a_long_message_is_capped():
    assert len(errors.describe(ValueError("x" * 5000))) < errors.MAX_MESSAGE + 30


# ------------------------------------------------------------------ relay identity
def test_a_relay_names_its_visitor_only_with_the_secret(monkeypatch):
    monkeypatch.delenv("IDQUEUE_RELAY_KEY", raising=False)
    h = relay.headers("198.51.100.4")
    assert relay.relayed_client(h) == "198.51.100.4"
    forged = dict(h, **{relay.RELAY_HEADER: "guess"})
    assert relay.relayed_client(forged) is None
    assert relay.relayed_client({relay.CLIENT_HEADER: "198.51.100.4"}) is None
    assert relay.headers(None) == {}


def test_a_configured_relay_key_is_shared_between_servers(monkeypatch):
    monkeypatch.setenv("IDQUEUE_RELAY_KEY", "k-123")
    assert relay.headers("v")[relay.RELAY_HEADER] == "k-123"
    assert relay.relayed_client({relay.RELAY_HEADER: "k-123", relay.CLIENT_HEADER: "v"}) == "v"


# ------------------------------------------------------------------ markup from labels
def test_the_paste_progress_line_escapes_its_label():
    """Callers build the busy text from labels other people chose (a queued link's key,
    any string at all); written raw into innerHTML it ran their markup on the page that
    holds the reviewer's password."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    body = js[js.index("async function runPaste("):]
    body = body[: body.index("\n}\n")]
    for m in re.finditer(r"innerHTML\s*=\s*`([^`]*)`", body):
        assert "${busy}" not in m.group(1), m.group(0)
    assert "escHtml(busy)" in body
