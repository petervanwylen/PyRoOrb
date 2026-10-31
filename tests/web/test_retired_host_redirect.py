"""A retired hostname permanently redirects to the current one; no other Host
is redirected."""
import http.client
import os
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

from pyroorb.web import server  # noqa: E402

_OLD = "web-production-fee0b.up.railway.app"


@pytest.fixture
def addr(tmp_path, monkeypatch):
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


def _request(addr, method, path, host, body=None):
    conn = http.client.HTTPConnection(*addr, timeout=10)
    headers = {"Host": host}
    if body is not None:
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp.status, resp.getheader("Location")


@pytest.mark.parametrize("host", [_OLD, _OLD.upper(), _OLD + ":443", _OLD + "."])
def test_get_on_retired_host_is_301_to_same_path(addr, host):
    status, loc = _request(addr, "GET", "/fit?desig=2023%20QJ3&x=1", host)
    assert status == 301
    assert loc == "https://www.pyroorb.net/fit?desig=2023%20QJ3&x=1"


def test_post_on_retired_host_is_308(addr):
    status, loc = _request(addr, "POST", "/api/session", _OLD, body=b'{"desig": "2003 LW2"}')
    assert status == 308
    assert loc == "https://www.pyroorb.net/api/session"


@pytest.mark.parametrize("host", [
    "www.pyroorb.net", "pyroorb.net", "127.0.0.1", "localhost:8000",
    "other.up.railway.app", "x." + _OLD, _OLD + ".evil.example", "",
])
def test_other_hosts_are_not_redirected(addr, host):
    status, loc = _request(addr, "GET", "/healthz", host)
    assert status not in (301, 308)
    assert loc is None


def test_redirect_target_helper():
    assert server._retired_redirect({"Host": _OLD}, "/") == "https://www.pyroorb.net/"
    assert server._retired_redirect({"Host": "www.pyroorb.net"}, "/") is None
    assert server._retired_redirect({}, "/") is None
