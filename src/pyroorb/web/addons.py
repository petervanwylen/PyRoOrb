"""Web add-ons: features the server learns about at start-up instead of importing by name.

An add-on is a :class:`WebAddon` object. The server knows nothing about any particular one;
it asks the registry for the installed add-ons and lets each contribute

* routes -- exact GET/POST paths (``get`` / ``post``) and whole path prefixes
  (``prefixes``) that it handles itself;
* pages and assets -- files in its own ``static_dir``, served at ``/static/<name>`` after
  the server's own static files, plus page paths (``pages``) mapped to those files;
* the main page -- Tools-menu links (``menu``), and scripts / stylesheets (``scripts`` /
  ``styles``) loaded by ``index.html``;
* background work -- a ``start(ctx)`` callable run once when the server starts.

Discovery, in this order (duplicates by name are ignored):

1. Python entry points in the group ``pyroorb.web_addons`` -- the route for an add-on that
   is installed as its own package. Each entry point names a WebAddon object.
2. A registry file, ``web_addons.txt`` at the checkout root (or the path in
   ``PYROORB_WEB_ADDONS_FILE``): one ``module:attribute`` per line, ``#`` comments allowed --
   the route for add-ons that live in the same checkout as the server.

A broken add-on is reported and skipped; it never stops the server from serving.

Route handlers receive the server context (:class:`Context`) and the request handler, whose
``_send``, ``_read_body``, ``_static_file``, ``headers`` and ``client_address`` they may use.
"""
import importlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

GROUP = "pyroorb.web_addons"
FILE_ENV = "PYROORB_WEB_ADDONS_FILE"
_CHECKOUT = Path(__file__).resolve().parents[3]


@dataclass
class MenuLink:
    """A Tools-menu link. ``visible`` (no arguments) decides per page load whether it shows."""
    id: str
    href: str
    label: str
    title: str = ""
    new_tab: bool = True
    visible: Optional[Callable[[], bool]] = None


@dataclass
class WebAddon:
    name: str
    static_dir: Optional[Path] = None
    pages: dict = field(default_factory=dict)       # "/path" -> file in static_dir
    get: dict = field(default_factory=dict)         # "/path" -> fn(ctx, handler, qs) -> (code, body)
    post: dict = field(default_factory=dict)        # "/path" -> fn(ctx, handler, data) -> (code, body)
    prefixes: list = field(default_factory=list)    # [(prefix, fn(ctx, handler, method, path, qs))]
    menu: list = field(default_factory=list)        # [MenuLink]
    scripts: list = field(default_factory=list)     # files in static_dir, loaded by index.html
    styles: list = field(default_factory=list)      # files in static_dir, linked by index.html
    start: Optional[Callable] = None                # fn(ctx), once at server start


@dataclass
class Context:
    """What the server lends its add-ons. ``pooled(label, fn)`` runs ``fn`` through the bounded
    worker pool; ``itf_slot`` is the shared background ITF-search slot; ``obscodes()`` and
    ``fit_for(session, fit_key)`` read the server's station table and a displayed fit."""
    pooled: Callable
    itf_slot: Callable
    obscodes: Callable
    fit_for: Callable


_LOCK = threading.Lock()
_LOADED = None          # list[WebAddon] once loaded
CONTEXT: Optional[Context] = None


def _from_spec(spec):
    mod, _, attr = spec.strip().partition(":")
    obj = importlib.import_module(mod)
    for part in (attr or "ADDON").split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, WebAddon):
        raise TypeError(f"{spec} is not a WebAddon")
    return obj


def _registry_file():
    want = os.environ.get(FILE_ENV, "").strip()
    return Path(want) if want else _CHECKOUT / "web_addons.txt"


def discover(log=print):
    """Every add-on this server can find (entry points, then the registry file)."""
    found, names = [], set()

    def add(obj):
        if obj.name not in names:
            names.add(obj.name)
            found.append(obj)

    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group=GROUP):
            try:
                add(_from_spec(ep.value))
            except Exception as ex:
                log(f"web add-on {ep.name}: not loaded ({type(ex).__name__}: {ex})")
    except Exception as ex:                        # pragma: no cover - defensive
        log(f"web add-ons: entry points unreadable ({ex})")
    path = _registry_file()
    if path.is_file():
        for line in path.read_text().splitlines():
            spec = line.split("#", 1)[0].strip()
            if not spec:
                continue
            try:
                add(_from_spec(spec))
            except Exception as ex:
                log(f"web add-on {spec}: not loaded ({type(ex).__name__}: {ex})")
    return found


def installed():
    """The loaded add-ons (discovered once per process)."""
    global _LOADED
    if _LOADED is None:
        with _LOCK:
            if _LOADED is None:
                _LOADED = discover()
    return _LOADED


def set_for_tests(addons):
    """Replace the loaded add-ons (tests only); None re-discovers on next use."""
    global _LOADED
    _LOADED = addons


def start_all(ctx, log=print):
    for a in installed():
        if a.start is None:
            continue
        try:
            a.start(ctx)
        except Exception as ex:                    # pragma: no cover - defensive
            log(f"web add-on {a.name}: start failed ({type(ex).__name__}: {ex})")


def find_static(rel):
    """An add-on's static file for ``/static/<rel>``, or None."""
    for a in installed():
        if a.static_dir is None:
            continue
        root = os.path.realpath(str(a.static_dir))
        path = os.path.realpath(os.path.join(root, rel))
        if path.startswith(root + os.sep) and os.path.isfile(path):
            return path
    return None


def page_file(path):
    """The static file an add-on serves at page ``path``, or None."""
    for a in installed():
        name = a.pages.get(path)
        if name and a.static_dir is not None:
            return os.path.join(str(a.static_dir), name)
    return None


def route(method, path):
    """The add-on handler for an exact ``path``, or None."""
    for a in installed():
        fn = (a.get if method == "GET" else a.post).get(path)
        if fn is not None:
            return fn
    return None


def prefix_route(path):
    for a in installed():
        for prefix, fn in a.prefixes:
            if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                return fn
    return None


def _esc(s):
    return (str(s).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def menu_html():
    out = []
    for a in installed():
        for m in a.menu:
            try:
                if m.visible is not None and not m.visible():
                    continue
            except Exception:
                continue
            tab = ' target="_blank" rel="noopener"' if m.new_tab else ""
            out.append(f'<a id="{_esc(m.id)}" href="{_esc(m.href)}"{tab} '
                       f'title="{_esc(m.title)}">{_esc(m.label)}</a>')
    return "\n      ".join(out)


def head_html():
    return "\n".join(f'<link rel="stylesheet" href="/static/{_esc(s)}">'
                     for a in installed() for s in a.styles)


def scripts_html():
    return "\n".join(f'<script src="/static/{_esc(s)}"></script>'
                     for a in installed() for s in a.scripts)


MENU_MARK = "<!-- web add-ons: menu -->"
HEAD_MARK = "<!-- web add-ons: styles -->"
SCRIPTS_MARK = "<!-- web add-ons: scripts -->"


def render_index(html):
    """``index.html`` with every add-on's menu links, stylesheets and scripts in place."""
    return (html.replace(MENU_MARK, menu_html())
                .replace(HEAD_MARK, head_html())
                .replace(SCRIPTS_MARK, scripts_html()))
