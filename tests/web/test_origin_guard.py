"""The API refuses cross-origin requests.

A session handle is the whole credential for every ``/api/`` route, and the
server sends no CORS headers -- which stops another origin READING a response,
but not from SENDING the request, so a hostile page could still drive
state-changing POSTs (CSRF).  ``server._origin_ok`` closes that.

The rules it has to satisfy at once, which is why this is table-driven:

* the UI's own fetches must pass, on localhost, on a bare IP, and behind a
  TLS-terminating proxy (where ``Origin:`` is https but ``Host:`` is a bare
  name -- so only host:port may be compared, never the scheme);
* non-browser clients (curl, scripts, the platform healthcheck) send no
  ``Origin`` at all and must keep working;
* a person clicking a link to an export URL from another site must still get
  the document -- but a cross-site form POST, which is also a "navigation",
  must not;
* anything a browser labels as another origin is refused.
"""
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


UI_HOST = "localhost:8000"

CASES = [
    # (id, headers, method, allowed)
    ("ui-fetch-same-origin",
     {"Origin": "http://localhost:8000", "Host": UI_HOST,
      "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}, "POST", True),
    ("ui-fetch-behind-tls-proxy",
     {"Origin": "https://pyroorb.example.app", "Host": "pyroorb.example.app",
      "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}, "POST", True),
    ("ui-fetch-on-a-vm-ip",
     {"Origin": "http://192.168.64.3:8000", "Host": "192.168.64.3:8000"}, "POST", True),
    ("curl-no-origin-get", {"Host": UI_HOST}, "GET", True),
    ("curl-no-origin-post", {"Host": UI_HOST}, "POST", True),
    ("link-click-from-another-site-is-a-get-navigation",
     {"Host": UI_HOST, "Sec-Fetch-Site": "cross-site",
      "Sec-Fetch-Mode": "navigate"}, "GET", True),
    ("typed-url-or-bookmark",
     {"Host": UI_HOST, "Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "navigate"}, "GET", True),

    ("cross-origin-fetch", {"Origin": "https://evil.example", "Host": UI_HOST,
                            "Sec-Fetch-Site": "cross-site",
                            "Sec-Fetch-Mode": "cors"}, "POST", False),
    ("cross-origin-form-post-no-sec-fetch-headers",
     {"Origin": "https://evil.example", "Host": UI_HOST}, "POST", False),
    ("cross-origin-form-post-is-a-navigation-too",
     {"Origin": "https://evil.example", "Host": UI_HOST,
      "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}, "POST", False),
    ("cross-origin-get-xhr", {"Origin": "https://evil.example", "Host": UI_HOST,
                              "Sec-Fetch-Site": "cross-site",
                              "Sec-Fetch-Mode": "cors"}, "GET", False),
    ("sibling-subdomain-is-same-site-but-not-same-origin",
     {"Origin": "https://other.example.app", "Host": "pyroorb.example.app",
      "Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors"}, "POST", False),
    ("port-mismatch-is-a-different-origin",
     {"Origin": "http://localhost:9999", "Host": UI_HOST}, "POST", False),
]


@pytest.mark.parametrize("headers,method,allowed",
                         [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_origin_verdicts(headers, method, allowed):
    assert server._origin_ok(_Headers(headers), method) is allowed


def test_allowed_origins_env_opts_one_in(monkeypatch):
    """A deliberately separate front-end origin can be opted in, and only it.

    The headers carry ``Sec-Fetch-Site: cross-site`` -- what a real browser
    ACTUALLY sends on this call -- because the allowlist used to be consulted
    only AFTER the Sec-Fetch-Site refusal, so it admitted no real browser at
    all; a test that omitted the header passed while the feature was dead."""
    monkeypatch.setenv("PYROORB_ALLOWED_ORIGINS", "front.example.com, other.test:8080")
    hdr = lambda o: _Headers({"Origin": o, "Host": "api.example.com",  # noqa: E731
                              "Sec-Fetch-Site": "cross-site",
                              "Sec-Fetch-Mode": "cors"})
    assert server._origin_ok(hdr("https://front.example.com"), "POST") is True
    assert server._origin_ok(hdr("http://other.test:8080"), "POST") is True
    assert server._origin_ok(hdr("https://evil.example"), "POST") is False


def test_guard_is_wired_into_both_dispatchers():
    """An unenforced check is not a check -- both entry points must call it."""
    import inspect
    for fn in (server.Handler.do_GET, server.Handler.do_POST):
        src = inspect.getsource(fn)
        assert "_cross_origin_refused" in src, f"{fn.__name__} does not guard"


def test_guard_covers_api_and_leaves_static_open():
    """Static assets and the page itself stay open (public files, no side
    effects); everything under /api/ is guarded."""
    calls = []

    class _H:
        headers = _Headers({"Origin": "https://evil.example", "Host": UI_HOST})
        _send = lambda self, code, body: calls.append(code)  # noqa: E731
        _cross_origin_refused = server.Handler._cross_origin_refused

    h = _H()
    assert h._cross_origin_refused("/", "GET") is False
    assert h._cross_origin_refused("/static/app.js", "GET") is False
    assert calls == []
    assert h._cross_origin_refused("/api/fit", "GET") is True
    assert h._cross_origin_refused("/api/itf_submit", "POST") is True
    assert calls == [403, 403]
