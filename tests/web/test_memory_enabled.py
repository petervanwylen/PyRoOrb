"""The /memory page and /api/memory exist for an operator, not for every visitor.

A hosted server answers them only when its operator configured a memory password
(``PYROORB_MEMORY_PASSWORD``); without one a remote request gets a plain 404, the same as
a route that does not exist. The local console always gets them, so a developer running
the server on their own machine needs no setup. A request that came through a proxy is
never the local console, even when the proxy connects over loopback.
"""
import http.client
import os
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

from pyroorb.web import server  # noqa: E402

PW = "hunter2-and-then-some"


class _Headers(dict):
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


CASES = [
    # (id, password configured, headers, peer, enabled)
    ("console-without-password", "", {}, "127.0.0.1", True),
    ("console-v6-without-password", "", {}, "::1", True),
    ("remote-without-password", "", {}, "203.0.113.9", False),
    ("proxied-loopback-without-password", "", {"X-Forwarded-For": "203.0.113.9"},
     "127.0.0.1", False),
    ("remote-with-password-configured", PW, {}, "203.0.113.9", True),
    ("proxied-with-password-configured", PW, {"X-Forwarded-For": "203.0.113.9"},
     "127.0.0.1", True),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_memory_enabled(case, monkeypatch):
    _id, configured, headers, peer, enabled = case
    monkeypatch.setenv(server._MEMORY_PW_ENV, configured)
    assert server._memory_enabled(_Headers(headers), peer) is enabled, _id


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


def _status(addr, path, headers=()):
    conn = http.client.HTTPConnection(*addr, timeout=30)
    conn.request("GET", path, headers=dict(headers))
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp.status


@pytest.mark.parametrize("path", ["/memory", "/api/memory?tables=1"])
def test_a_proxied_request_needs_a_configured_password(addr, monkeypatch, path):
    proxied = {"X-Forwarded-For": "203.0.113.9"}
    monkeypatch.setenv(server._MEMORY_PW_ENV, "")
    assert _status(addr, path, proxied) == 404
    assert _status(addr, path) == 200                  # the local console
    monkeypatch.setenv(server._MEMORY_PW_ENV, PW)
    assert _status(addr, path, proxied) == 200
