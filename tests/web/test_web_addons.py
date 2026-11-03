"""Web add-ons (web/addons.py): the server serves whatever an add-on registers -- routes,
path prefixes, pages, static files, Tools-menu links, scripts and styles -- without knowing
any add-on by name, and a broken add-on never stops it from serving."""
import http.client
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

from pyroorb.web import addons  # noqa: E402
from pyroorb.web import server  # noqa: E402


def _fake(tmp_path, visible=True):
    static = tmp_path / "static"
    static.mkdir()
    (static / "fake.html").write_text("<p>fake page</p>", encoding="utf-8")
    (static / "fake.js").write_text("window.FAKE = 1;", encoding="utf-8")
    (static / "fake.css").write_text(".fake{}", encoding="utf-8")
    calls = []

    def prefixed(ctx, handler, method, path, qs):
        calls.append((method, path))
        if method == "POST":
            handler.close_connection = True
        return handler._send(418, {"prefix": path, "method": method})

    return addons.WebAddon(
        name="fake",
        static_dir=static,
        pages={"/fake": "fake.html"},
        get={"/api/fake": lambda ctx, h, qs: (200, {"q": (qs.get("x") or [""])[0],
                                                     "ctx": ctx is addons.CONTEXT})},
        post={"/api/fake_post": lambda ctx, h, d: (200, {"echo": d.get("v")})},
        prefixes=[("/fakeq", prefixed)],
        menu=[addons.MenuLink(id="fakeLink", href="/fake", label="Fake <tool>",
                              title='say "hi"', visible=lambda: visible)],
        scripts=["fake.js"], styles=["fake.css"]), calls


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROORB_FIT_CACHE_DIR", str(tmp_path / "cache"))
    from pyroorb.web import fitdisk
    monkeypatch.setattr(fitdisk, "_DIR", None)
    monkeypatch.setattr(fitdisk, "_DISABLED_REASON", None)
    addon, calls = _fake(tmp_path)
    addons.set_for_tests([addon])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        yield httpd.server_address, calls
    finally:
        httpd.shutdown()
        th.join(timeout=5)
        addons.set_for_tests(None)


def _req(addr, method, path, body=None):
    conn = http.client.HTTPConnection(*addr, timeout=30)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers=headers)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, data


def test_routes_pages_and_static_files(served):
    addr, calls = served
    code, body = _req(addr, "GET", "/api/fake?x=7")
    assert code == 200 and json.loads(body) == {"q": "7", "ctx": True}
    code, body = _req(addr, "POST", "/api/fake_post", {"v": 3})
    assert code == 200 and json.loads(body) == {"echo": 3}
    assert _req(addr, "GET", "/fake") == (200, b"<p>fake page</p>")
    assert _req(addr, "GET", "/static/fake.js") == (200, b"window.FAKE = 1;")
    assert _req(addr, "GET", "/static/style.css")[0] == 200       # the server's own still win
    assert _req(addr, "GET", "/static/../fake.html")[0] == 404
    assert _req(addr, "GET", "/static/nope.js")[0] == 404


def test_a_prefix_owns_every_path_under_it(served):
    addr, calls = served
    assert _req(addr, "GET", "/fakeq")[0] == 418
    assert _req(addr, "GET", "/fakeq/a/b")[0] == 418
    assert _req(addr, "POST", "/fakeq/x", {})[0] == 418
    assert _req(addr, "GET", "/fakeqz")[0] == 404                  # not a path under it
    assert calls == [("GET", "/fakeq"), ("GET", "/fakeq/a/b"), ("POST", "/fakeq/x")]


def test_the_main_page_carries_the_add_on_links_scripts_and_styles(served):
    addr, _ = served
    code, html = _req(addr, "GET", "/")
    html = html.decode()
    assert code == 200
    assert '<a id="fakeLink" href="/fake" target="_blank" rel="noopener" ' \
           'title="say &quot;hi&quot;">Fake &lt;tool&gt;</a>' in html
    assert '<script src="/static/fake.js"></script>' in html
    assert '<link rel="stylesheet" href="/static/fake.css">' in html
    assert "<!-- web add-ons" not in html
    assert html.index("fakeLink") > html.index('id="toolsMenu"')


def test_a_hidden_link_and_no_add_ons_leave_no_trace(tmp_path):
    addon, _ = _fake(tmp_path, visible=False)
    addons.set_for_tests([addon])
    try:
        assert "fakeLink" not in addons.render_index("<!-- web add-ons: menu -->")
        addons.set_for_tests([])
        page = addons.render_index("a<!-- web add-ons: menu -->b<!-- web add-ons: scripts -->c")
        assert page == "abc"
    finally:
        addons.set_for_tests(None)


def test_discovery_reads_the_registry_file_and_skips_a_broken_entry(tmp_path, monkeypatch):
    mod = tmp_path / "fake_addon_mod.py"
    mod.write_text("from pyroorb.web.addons import WebAddon\nADDON = WebAddon(name='from-file')\n"
                   "NOT_AN_ADDON = 3\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = tmp_path / "web_addons.txt"
    reg.write_text("# comment\nfake_addon_mod:ADDON\nfake_addon_mod:NOT_AN_ADDON\n"
                   "no_such_module:ADDON\nfake_addon_mod:ADDON  # duplicate\n", encoding="utf-8")
    monkeypatch.setenv(addons.FILE_ENV, str(reg))
    log = []
    found = addons.discover(log=log.append)
    assert [a.name for a in found if a.name == "from-file"] == ["from-file"]
    assert any("NOT_AN_ADDON" in m for m in log) and any("no_such_module" in m for m in log)


def test_a_start_hook_that_fails_does_not_stop_the_others():
    started = []

    def bad(ctx):
        raise RuntimeError("boom")

    addons.set_for_tests([addons.WebAddon(name="a", start=bad),
                          addons.WebAddon(name="b", start=lambda ctx: started.append(ctx))])
    try:
        log = []
        addons.start_all("CTX", log=log.append)
        assert started == ["CTX"] and "boom" in log[0]
    finally:
        addons.set_for_tests(None)
