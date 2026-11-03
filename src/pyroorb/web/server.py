"""Zero-dependency HTTP server for the PyRoOrb web UI (Phase 0 read-only viewer).

Uses only the Python stdlib so it runs offline with no pip install and no Node
build step -- the most robust way to get the viewer in front of a browser today.
The JSON it serves is produced by ``engine_api`` (framework-agnostic), so the
documented React + FastAPI stack (WEB_UI_PLAN §1, §8) can later replace this
server and the static front end without touching the engine bridge.

    pyroorb-web            # -> http://127.0.0.1:8000
    pyroorb-web --port 9000

Endpoints:
    GET  /                      static index.html (the workspace)
    GET  /static/<file>         js/css assets
    GET  /api/fit?desig=...     fetch + fit + serialise one object/linkage
    POST /api/fit_text          fit a pasted OBS80 / ADES blob  (body = raw text)
    POST /api/add_obs           merge a pasted blob into the open fit + re-fit
                                (body = raw text; ?session=/?fit_key= identify the open fit)
    POST /api/fullstep          differential correction on the edited obs set
                                {session, mask, single} -- single:true = ONE step (Full step),
                                else re-solve to convergence (Multi-step)
    POST /api/reauto            re-run the full automatic pipeline on the open fit's obs
                                {session, fit_key} -- as if freshly pasted (Re-auto-solve)
    POST /api/itf_search        search the live ITF for tracklets of the open fit's object
                                {session, fit_key, mask, options{max_uncert_arcsec,
                                uncert_mult, rank_mode, candidate_source, jitter, trace_id}} -- returns a
                                fresh full fit payload + .itf on a find, {ok,
                                itf.found:false} otherwise; NEVER submits to the MPC
                                (CLI-only decision).  With {async:true} it returns
                                {ok, job} at once and the search runs on a worker
                                thread -- a search outlives a hosted proxy's request
                                limit, so the request must not wait for it
    GET  /api/itf_result        collect a backgrounded search (?id=<job>) ->
                                {ok, done:false, elapsed} while it runs, then
                                {ok, done:true, result:<the payload above>}
    POST /api/itf_cancel        stop a backgrounded search {job} -> {ok, cancelling}; the
                                worker unwinds at its next check and the next
                                /api/itf_result answers {ok:false, cancelled:true}
    GET  /api/itf_defaults      the ITF options modal's defaults, from ItfConfig
    GET  /api/random_families   the random-object dropdown's families + opposition cap
    GET  /api/random_object     one random designation from the cached MPCORB index
                                (?family=any|neo|mba|…) -> {designation, oppositions,
                                a, e, incl, H} -- the top bar's "Random asteroid"
    GET  /api/itf_trace         poll the live trace of a running ITF search
                                (?id=<trace_id from itf_search options>&from=<cursor>
                                 [&level=1..4, re-levels the running search]) ->
                                {lines, next, done} -- the "see trace" feed
    POST /api/itf_submit        send the linkage on this session {session, name, email,
                                route}: route "queue" (default) POSTs it to the
                                identification queue (IDQUEUE_URL), "mpc" straight to the
                                MPC (only with DIRECT_SUBMIT_ALLOWED) -- the ONLY web submit
                                path; the server rebuilds the document; one-shot per find
    POST /api/itf_download      the submit menu's "Download linkage file":
                                build the identification JSON for the session's pending
                                linkage {session, name, email} and return it as
                                {filename, document} for a client-side file download
    GET  /api/export/mpec       find_orb-style pseudo-MPEC (HTML) for ?desig= or ?key=
    GET  /api/export/ephem      fixed-width ephemeris table for ?desig= or ?key=
                                (?faint=MAG / ?unobs=1 apply find_orb's display cuts)
    GET  /api/export/state      cartesian state (x,y,z,dx,dy,dz) + 6x6 covariance as
                                JSON, equatorial AND ecliptic J2000, for ?session=/?key=/?desig=
    GET  /api/export/covar      find_orb's covar.json (covar / state_vect / epoch, ecliptic
                                J2000) for ?session=/?key=/?desig=
    GET  /api/ephem_site        whether ?site= is a ground station with a horizon
    GET  /api/impact            Phase-1 impact location (where/when it hits) for ?session=/?key=/?desig=
    GET  /api/environment       read-only host inventory (data files / ephemerides / env) for Settings
    GET  /memory                the memory-attribution page (static). It and /api/memory
                                answer only the local console unless $PYROORB_MEMORY_PASSWORD
                                is configured; otherwise a remote request gets a 404
    GET  /api/memory            where the process's memory is: the long-lived tables
                                (sessions, fitcache, ITF jobs/traces, shared tables),
                                running threads' working sets, an uncategorised "misc"
                                bucket for the rest of the heap, and the process /
                                container totals (?tables=1 = skip the misc pass).
                                The misc pass walks the whole heap under the GIL, so
                                it needs $PYROORB_MEMORY_PASSWORD in X-Memory-Password
                                (or a loopback caller when no password is set); the
                                tables always answer, and every report is rate-limited
    GET  /api/nongrav_models    the non-gravitational models the UI can request
    POST /api/nongrav           re-solve a session's orbit with a nongrav model {session, model}
    GET  /api/constraint_elements  the orbital elements the UI can hold fixed
    POST /api/constrain         re-solve with elements held fixed {session, constraints}
    POST /api/nominal           show an orbit anyway on an undetermined arc {session}
    POST /api/d2_tracklets      digest2 scores per tracklet, for q < 1.8 {session | fit_key}
    POST /api/asteroids         toggle the 16 asteroid perturbers + re-fit {session, on}

Web add-ons (web/addons.py) add their own routes, pages, Tools-menu links and scripts; the
server serves whatever the installed add-ons register and knows none of them by name.
"""
import os
import sys
import json
import hmac
import ipaddress
import threading
import time
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    # Run as a program (`python -m pyroorb.web.server`, the dev server's child): load the
    # `.env` settings file before the engine imports read their settings.
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.console import utf8_stdio
    utf8_stdio()
from pyroorb.web import engine_api  # noqa: E402
from pyroorb.web import loader      # noqa: E402  (paste-an-obs-blob loader)
from pyroorb.web import exporters   # noqa: E402  (pseudo-MPEC + ephemeris text)
from pyroorb.web import fitcache    # noqa: E402  (reuse / re-derive a fit for export)
from pyroorb.web import webmem      # noqa: E402  (the /memory attribution page)
from pyroorb.web import ratelimit   # noqa: E402  (per-client limits on the costly routes)
from pyroorb.web import addons      # noqa: E402  (web add-ons: routes/pages/menu links registered at start)

_STATIC = os.path.join(_HERE, "static")
_CTYPE = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
          ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
          ".png": "image/png", ".ico": "image/x-icon", ".jpg": "image/jpeg",
          ".webmanifest": "application/manifest+json"}


# How long GET /api/fit waits inline before handing back a job handle.  Well
# under any hosted proxy's request limit (Railway's is 60s), and long enough
# that the median object -- which fits in seconds -- still answers in one round
# trip.
_SYNC_FIT_DEADLINE = float(os.environ.get("PYROORB_SYNC_FIT_DEADLINE") or 25)


def _pooled(label, work):
    """Run one piece of engine work through the fit pool and wait for it.

    Every INTERACTIVE endpoint goes through here, not just the ones that fit a
    brand-new object.  A Full step / Multi-step, a nongrav re-solve, a
    constrained re-solve, an asteroid-perturber toggle and an epoch re-anchor
    all re-enter the LM on a live Evaluator -- work of the same ORDER as the
    fit that produced the session, and `nongrav_sigmas` is up to 16 independent
    refits.  Run on the request thread (which is what they all did) they are
    outside BOTH bounds the pool exists to impose: the concurrency cap, so a
    crowd holding down "Multi-step" starts arbitrarily many simultaneous
    solves on a `ThreadingHTTPServer` that never refuses a connection, and
    `deadline.FIT_MAX_SEC`, which is armed in `fitqueue._work_once` and
    nowhere else -- so a pathological re-solve pinned a request thread with no
    backstop, the same failure the three-minute bound was introduced for.

    `run_bounded` does not coalesce: each of these is unique work against one
    private session, and two steps of the SAME session must not share a result.
    Past the inline deadline the caller gets the `pending` job stub every other
    one-off fit route already returns, which the client polls out through
    `FitJob`.
    """
    return engine_api.fitqueue.run_bounded(work, _SYNC_FIT_DEADLINE, label=label)

# Largest request body accepted, in bytes.  The paste endpoints used to read
# Content-Length bytes with no ceiling, so a single request could ask the
# process to allocate as much as it liked -- an out-of-memory kill from one
# POST, which no amount of downstream queueing can prevent because the
# allocation happens before any of it runs.  8 MB is far past the largest real
# astrometry paste (a 4000-observation arc in OBS80 is ~330 KB).
_MAX_BODY_BYTES = int(os.environ.get("PYROORB_MAX_BODY_BYTES") or 8 * 1024 * 1024)


def _nan_safe(o):
    """Recursively replace NaN/Infinity floats with None so the result is valid JSON."""
    import math
    if getattr(o, "shape", None) == () and hasattr(o, "item"):
        o = o.item()                    # a NumPy scalar, NaN included
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _nan_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_safe(v) for v in o]
    return o


def _json_fallback(o):
    """``json.dumps(default=...)``: a NumPy scalar or array becomes its Python
    value; anything else still raises.  An engine number that is an np.int64 is
    the ordinary case, and it must not cost the caller their answer."""
    item = getattr(o, "item", None)
    if item is not None and getattr(o, "shape", None) == ():
        return item()
    tolist = getattr(o, "tolist", None)
    if tolist is not None:
        return tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# --- retired hostnames ------------------------------------------------------
# A hostname the site used to answer on, mapped to the one it moved to.  A
# request whose Host is EXACTLY a retired name gets a permanent redirect to the
# same path and query on the new host; every other Host (the current domain,
# 127.0.0.1, a LAN address, the platform's healthcheck) is served as before.
# GET gets 301; POST gets 308, so a client re-sends the same method and body
# instead of silently downgrading it to a GET.
_RETIRED_HOSTS = {
    "web-production-fee0b.up.railway.app": "www.pyroorb.net",
}


def _retired_redirect(headers, raw_path):
    """The redirect target for a request to a retired hostname, else None."""
    host = (headers.get("Host") or "").strip().lower()
    if host.count(":") == 1:                 # host:port (not a bare IPv6 literal)
        host = host.split(":", 1)[0]
    new = _RETIRED_HOSTS.get(host.rstrip("."))
    if not new:
        return None
    return "https://" + new + (raw_path if raw_path.startswith("/") else "/" + raw_path)


# --- same-origin enforcement ------------------------------------------------
# A session handle is the only credential the API has, so a page on ANOTHER
# origin must not be able to drive it.  Serving no CORS headers (we don't)
# already stops a cross-origin script from READING a response -- but it does
# not stop the REQUEST, so a hostile page could still fire state-changing POSTs
# and have them execute (classic CSRF).  Hence: when the browser tells us where
# a request came from, it has to be us.
#
# The allowed origin is derived from the request's OWN Host header rather than
# from configuration, because this server is meant to run anywhere with no
# setup -- 127.0.0.1, a sandbox VM's IP, a LAN address, a hosted domain.  Only
# host:port is compared, never the scheme: behind a TLS-terminating proxy the
# browser sends `Origin: https://name` while `Host:` is the bare name, so
# demanding a scheme match would reject every request in a normal deployment.
#
# $PYROORB_ALLOWED_ORIGINS (comma-separated host[:port]) opts in extra origins
# for the deliberate case of the UI being served from somewhere else.
def _extra_origins():
    raw = os.environ.get("PYROORB_ALLOWED_ORIGINS", "")
    return {o.strip().lower() for o in raw.split(",") if o.strip()}


def _origin_ok(headers, method):
    """True iff this request did not come from a different origin.

    A MISSING Origin is allowed on purpose: non-browser clients (curl, scripts,
    the platform healthcheck) never send one, and browsers omit it on ordinary
    same-origin navigations.  What matters is that a browser ALWAYS sends it on
    a cross-origin fetch/XHR and on a cross-origin form POST, which is exactly
    the traffic this is here to refuse.

    A top-level GET navigation is exempt: that is a person clicking a link to
    us, it renders as a new document on our own origin, and it can neither read
    another origin's data nor change anything -- so shareable links (the
    pseudo-MPEC and the export URLs are meant to be opened by hand) keep
    working.  The exemption is GET-only; a cross-site form POST is also a
    navigation and must stay blocked.
    """
    if method == "GET" and (headers.get("Sec-Fetch-Mode") or "").lower() == "navigate":
        return True
    origin = (headers.get("Origin") or "").strip()
    netloc = urlparse(origin).netloc.lower()
    # An opted-in origin is admitted BEFORE the Sec-Fetch-Site refusal below:
    # a browser stamps `Sec-Fetch-Site: cross-site` on exactly the deliberate
    # cross-origin call PYROORB_ALLOWED_ORIGINS exists to permit, so consulting
    # the allowlist after that check made the setting dead for every real
    # browser (it only ever reached clients that send no Sec-Fetch headers).
    if netloc and netloc in _extra_origins():
        return True
    # Modern browsers label the relationship directly; trust it when it says no.
    if (headers.get("Sec-Fetch-Site") or "").strip().lower() in ("same-site", "cross-site"):
        return False
    if not origin:
        return True
    host = (headers.get("Host") or "").strip().lower()
    return bool(host) and netloc == host


# --- /api/memory: the whole-heap pass costs, so it is gated and throttled ----
# The tables pass is attribution over structures we already hold; the MISC pass
# walks everything the process can reach (gc.get_objects() plus every thread's
# frames) and holds the GIL while it does.  It is ~0.5 s idle and a few seconds
# on a loaded server -- fine as a diagnostic somebody opens, and a free way for
# anyone who can reach the port to take CPU away from the searches this page
# exists to keep alive.  The origin guard does not help: it deliberately admits
# requests with no ``Origin`` at all, which is every non-browser client.
#
# So the deep pass needs a password, and an ANONYMOUS report is re-served from
# a short-lived cache rather than walked again.  Together those bound what a
# caller who cannot authenticate can make this route spend: never the deep
# walk, and at most one tables walk per interval however fast they ask.  It is
# not authentication for the SERVER and does not pretend to be -- the report is
# shapes and sizes, never contents -- it is a cost bound.
_MEMORY_PW_ENV = "PYROORB_MEMORY_PASSWORD"
_MEMORY_PW_HEADER = "X-Memory-Password"

# Reports are cached and re-served for this long to an unauthenticated caller.
# The page's own auto-refresh is 10 s, so a person watching it never notices;
# a loop hitting the route pays for one report per interval instead of one per
# request.  A caller with the password always gets a fresh walk -- they asked
# for the measurement, and they are the operator.
_MEMORY_MIN_INTERVAL = 5.0
_MEMORY_CACHE = {"t": 0.0, "misc": False, "payload": None}
_MEMORY_CACHE_LOCK = threading.Lock()
# Single-flight: the cache alone bounds how often a report is COMPUTED only if
# concurrent misses wait for the one in flight instead of each starting their
# own.  Without this, N simultaneous first requests are N walks under the GIL
# -- the stampede the interval exists to prevent, arriving all at once.
_MEMORY_WALK_LOCK = threading.Lock()


def _memory_password():
    """The configured password, or "" when none is set."""
    return (os.environ.get(_MEMORY_PW_ENV) or "").strip()


def _is_loopback(addr):
    try:
        return ipaddress.ip_address((addr or "").strip()).is_loopback
    except ValueError:
        return False


def _is_console(headers, peer_addr):
    """A request from this machine's own console: a loopback peer that did not come
    through a proxy (on a hosted deploy the proxy may well connect over loopback)."""
    return not headers.get("X-Forwarded-For") and _is_loopback(peer_addr)


def _memory_enabled(headers, peer_addr):
    """Does this server answer /memory and /api/memory for THIS request at all?

    They are an operator's diagnostic, so a hosted server offers them only when its
    operator configured a memory password; without one a remote request gets the same
    404 as a route that does not exist. The local console always gets them, so a
    developer running the server on their own machine needs no setup."""
    return bool(_memory_password()) or _is_console(headers, peer_addr)


def _misc_allowed(headers, peer_addr):
    """May THIS request have the whole-heap pass?  ``(allowed, reason)``.

    Two ways in, and the fallback matters as much as the password:

    * the password in ``X-Memory-Password``, compared in constant time.  A
      HEADER and not a query parameter, because a query string lands in every
      access log and proxy trace along the way;
    * failing that, a LOOPBACK peer when no password is configured at all --
      so a developer running this on their own machine sees exactly what they
      saw before, with no setup.  A request that arrived through a proxy
      (``X-Forwarded-For``) is never treated as local however it reaches us:
      on a hosted deploy the proxy may well connect over loopback, and that
      must not be mistaken for somebody sitting at the console.

    Anything else gets the tables, and is told which of the two it needs.
    """
    pw = _memory_password()
    supplied = (headers.get(_MEMORY_PW_HEADER) or "").strip()
    if pw:
        if not supplied:
            return False, "password-required"
        if hmac.compare_digest(supplied, pw):
            return True, "ok"
        return False, "password-wrong"
    if supplied:
        # A password offered where none is configured is a misconfiguration
        # worth naming: the operator thinks the route is protected and it is
        # not.  Do not silently accept it as if it had matched something.
        return False, "password-not-configured"
    if headers.get("X-Forwarded-For"):
        return False, "remote-no-password"
    return (True, "ok") if _is_loopback(peer_addr) else (False, "remote-no-password")


_MISC_LOCK_MESSAGE = {
    "password-required": (
        "The whole-heap pass needs the memory password. Enter it above."),
    "password-wrong": (
        "That password does not match. The whole-heap pass stays locked."),
    "password-not-configured": (
        "No memory password is set on this server, so the whole-heap pass "
        "cannot be unlocked remotely. Set " + _MEMORY_PW_ENV + " and restart."),
    "remote-no-password": (
        "The whole-heap pass is limited to the local console unless a "
        "password is set. Set " + _MEMORY_PW_ENV + " on the server to unlock "
        "it from here."),
}


def _memory_report(want_misc, authed):
    """A memory report, re-serving a recent one rather than walking again.

    The cache key is only "did it include misc": a tables-only reader must not
    be handed a cached deep report's misc block, and a deep reader must not be
    handed a tables-only one.  An authenticated caller always walks fresh.
    """
    def _fresh_enough():
        """The cached payload if it still serves this caller, else None."""
        now = time.time()
        with _MEMORY_CACHE_LOCK:
            c = _MEMORY_CACHE
            if (c["payload"] is not None and c["misc"] == want_misc
                    and now - c["t"] < _MEMORY_MIN_INTERVAL):
                out = dict(c["payload"])
                out["cached_age_s"] = round(now - c["t"], 1)
                return out
        return None

    if not authed:
        hit = _fresh_enough()
        if hit is not None:
            return hit
    # Only one walk at a time, and re-check the cache after acquiring: a
    # caller that queued behind another's walk wants that walk's answer, not
    # a second one.  An authenticated caller still gets a FRESH report -- they
    # asked for the measurement -- but waits its turn rather than racing.
    with _MEMORY_WALK_LOCK:
        if not authed:
            hit = _fresh_enough()
            if hit is not None:
                return hit
        payload = webmem.report(with_misc=want_misc)
        with _MEMORY_CACHE_LOCK:
            _MEMORY_CACHE.update({"t": time.time(), "misc": want_misc,
                                  "payload": payload})
        return payload


def _ephem_filters(qs):
    """find_orb's two ephemeris display cuts as read off the query string, shared by the
    JSON pane and the text export so both hide exactly the same epochs:
    ``?faint=21.5`` (V-mag limit; absent/blank/non-numeric = no limit) and
    ``?unobs=1`` (suppress epochs the observatory can't shoot)."""
    raw = (qs.get("faint") or [""])[0]
    try:
        faint = float(raw) if str(raw).strip() != "" else None
    except ValueError:
        faint = None
    unobs = (qs.get("unobs") or ["0"])[0]
    return {"faint_limit": faint,
            "suppress_unobservable": str(unobs).lower() in ("1", "true", "yes", "on")}


def _orbit_export_refusal(fit):
    """A refusal message when this fit has no orbit to export, else None.

    By design, an orbit the astrometry does not determine is not published
    in any form that outlives the page it was shown on. The browser already hides the
    buttons (`orbit_suppressed` on the payload), but an export URL is a plain GET anyone can
    open by hand, and an MPCORB line travels much further than the banner that qualified it
    -- so the refusal lives on the server too. `constraint.assess` is the same one verdict
    the elements panel obeys; there is no second opinion here.

    The ephemeris export is deliberately NOT covered: it is a sky prediction, not an orbit,
    and it is what a short-arc object is actually wanted for."""
    if getattr(fit, "nominal_optin", None):
        # The user asked for an orbit anyway, on THIS orbit (engine_api.nominal_optin), and every
        # export then works. The flag lives on the one
        # FittedOrbit, so a re-solve -- or a fresh fit by key/desig -- is refused again.
        return None
    try:
        from pyroorb import constraint
        verdict = constraint.assess(fit)
    except Exception:
        return None                       # never let the guard itself break an export
    if not verdict:
        return None
    return ("export refused: this arc does not determine an orbit "
            f"({verdict.get('headline', 'essentially unconstrained')}). No element set, "
            "MPCORB line, SOF record or state vector is published for it -- the page shows "
            "the range each element could take instead. The ephemeris export still works: a "
            "sky prediction carrying its own uncertainty is the honest product of a short arc.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json", extra_headers=()):
        if isinstance(body, (dict, list)):
            # NaN/Infinity are valid Python floats but INVALID JSON -- the browser's
            # JSON.parse rejects a literal NaN. Fast path dumps with allow_nan=False
            # (raises on any NaN/Inf); only a degenerate payload pays the deep clean.
            # A body that cannot be encoded must still get an ANSWER: raising here
            # kills the handler thread with nothing sent, the socket closes, and a
            # hosted proxy reports its own "Application failed to respond" 502 --
            # which is how one np.int64 in /api/memory read as a dead server.
            try:
                try:
                    body = json.dumps(body, allow_nan=False,
                                      default=_json_fallback).encode("utf-8")
                except ValueError:
                    body = json.dumps(_nan_safe(body), allow_nan=False,
                                      default=_json_fallback).encode("utf-8")
            except (TypeError, ValueError) as ex:
                code = 500
                body = json.dumps({"ok": False, "error": f"unencodable reply: {ex}"}
                                  ).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:      # e.g. Retry-After on a 429
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _static(self, rel):
        """A file from the server's own static folder, else from an add-on's."""
        path = os.path.normpath(os.path.join(_STATIC, rel))
        if not path.startswith(_STATIC + os.sep) or not os.path.isfile(path):
            path = addons.find_static(rel)
            if path is None:
                return self._send(404, {"error": "not found"})
        return self._static_file(path)

    def _static_file(self, path):
        """Serve one file the caller has already resolved (add-ons use this for pages)."""
        if not path or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as fp:
            self._send(200, fp.read(), _CTYPE.get(ext, "application/octet-stream"))

    def _index(self):
        """index.html with the installed add-ons' menu links, styles and scripts in place."""
        with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as fp:
            html = addons.render_index(fp.read())
        self._send(200, html.encode("utf-8"), _CTYPE[".html"])

    def _memory(self, qs):
        """GET /api/memory -- the attribution report.

        The tables always answer; the whole-heap MISC pass is gated
        (:func:`_misc_allowed`) and every report is rate-limited
        (:func:`_memory_report`).  A locked request is NOT an error and not a
        403: the page renders the tables perfectly well without misc, so it
        gets them plus a ``misc_locked`` block saying which key it is missing.
        A 403 would have the page show nothing at all, which is a worse answer
        to "the server is busy" than a partial one.
        """
        # ?tables=1 skips the whole-heap "misc" pass (milliseconds instead of
        # ~0.5 s) for anyone polling this while a search runs.
        tables_only = (qs.get("tables") or ["0"])[0] in ("1", "true", "yes")
        peer = (self.client_address or ("",))[0]
        allowed, reason = _misc_allowed(self.headers, peer)
        want_misc = (not tables_only) and allowed
        try:
            out = _memory_report(want_misc, authed=allowed)
        except Exception as ex:
            return self._send(200, {"ok": False,
                                    "error": f"{type(ex).__name__}: {ex}"})
        if not tables_only and not allowed:
            out = dict(out)
            out["misc_locked"] = {
                "reason": reason,
                "password_required": reason in ("password-required",
                                                "password-wrong"),
                "message": _MISC_LOCK_MESSAGE.get(
                    reason, "The whole-heap pass is locked."),
            }
        return self._send(200, out)

    def _cross_origin_refused(self, path, method):
        """Guard every /api/ route. `/` and /static/ are deliberately NOT guarded:
        they are public files with no side effects and no secrets, and leaving them
        open keeps plain linking/embedding of the page working."""
        if not path.startswith("/api/") or _origin_ok(self.headers, method):
            return False
        self._send(403, {"ok": False, "error": "cross-origin request refused"})
        return True

    # HTTP/1.1, not the stdlib's HTTP/1.0 default.
    #
    # An HTTP/1.0 origin closes the connection after every response.  A hosted
    # proxy keeps connections to the origin ALIVE and reuses them, so the two
    # race: the proxy sends a request on a socket the server is closing and the
    # client gets "RemoteDisconnected: Remote end closed connection without
    # response".  Measured on the live deploy at 1 request in 20 under a
    # 20-client burst -- silent, intermittent, and it looks like a server fault
    # to the visitor because it is one.
    #
    # Safe here because `_send` is the ONLY method that writes a response and it
    # always sets an accurate Content-Length; under HTTP/1.1 a reply without one
    # would leave the client waiting forever instead of merely disconnecting.
    protocol_version = "HTTP/1.1"

    # ...and with keep-alive a handler thread stays bound to its connection
    # waiting for the next request, so idle clients would pin threads that
    # ThreadingHTTPServer never reclaims.  This bounds that wait: on expiry
    # handle_one_request closes the connection and frees the thread.
    timeout = float(os.environ.get("PYROORB_HTTP_IDLE_TIMEOUT") or 30)

    # Route -> rate-limit bucket.  Only work that COSTS something is listed;
    # the polling routes (/api/fit_result, /api/itf_result, /api/itf_trace) are
    # deliberately absent, because the client polls them BECAUSE the work is
    # slow and limiting them would break the flow the queue introduced.
    _RATE_BUCKETS = {
        "/api/fit": "fit",
        "/api/fit_start": "fit",
        "/api/fit_text": "paste",
        "/api/add_obs": "paste",
        "/api/reauto": "paste",
        "/api/itf_search": "itf",
        "/api/random_object": "random",
        # Cheap for US and a request to the MPC for THEM: limited so the site cannot be
        # turned into an unmetered proxy onto their orbit service. Shares the `random`
        # bucket, the other one-click-one-object lookup.
        "/api/published_orbit": "random",
    }

    def _rate_refused(self, path):
        """True (and the refusal already sent) when this caller has spent its
        allowance for ``path``."""
        bucket = self._RATE_BUCKETS.get(path)
        if bucket is None:
            return False
        ok, retry = ratelimit.check(ratelimit.client_of(self), bucket)
        if ok:
            return False
        body = ratelimit.refusal(bucket, retry)
        self._send(429, body, extra_headers=[("Retry-After", str(int(retry) + 1))])
        return True

    def _refund_if_free(self, path, out):
        """Return the allowance for a request that caused no work.

        The limiter has to charge before anyone knows what a request will cost.
        What it rations is WORK, though, and a payload answered from cache -- or
        one that merely joined a fit already running -- causes none.  Without
        this, the hot object the whole design exists to serve cheaply is the one
        a crowd is refused on, while the server sits idle.
        """
        bucket = self._RATE_BUCKETS.get(path)
        if bucket is None or not isinstance(out, dict):
            return
        # `busy` belongs here too: a queue-full refusal ran no fit, and
        # charging it means a saturated server also rations the people it just
        # turned away -- they are told to slow down for work that never
        # happened, and the busy and rate-limited counts blur together exactly
        # when you most need to tell them apart.
        if out.get("cached") or out.get("coalesced") or out.get("busy"):
            ratelimit.refund(ratelimit.client_of(self), bucket)

    def _read_body(self):
        """The request body, capped.  Returns None (and has already answered)
        when the client declared more than we will accept.

        The 413 case forces the connection closed (see the guard note above
        do_POST): reading and discarding up to _MAX_BODY_BYTES just to keep the
        socket reusable would let an oversized upload cost real time for
        nothing, and skipping the read is exactly the undrained-body hazard
        this whole guard exists for.
        """
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > _MAX_BODY_BYTES:
            self.close_connection = True
            self._send(413, {"ok": False, "error":
                             f"that request body is {n} bytes; the limit is "
                             f"{_MAX_BODY_BYTES}. Paste a smaller batch of "
                             f"observations."})
            return None
        return self.rfile.read(n) if n else b""

    def _redirect_retired(self, code):
        """Answer a request to a retired hostname with a redirect; True if sent."""
        target = _retired_redirect(self.headers, self.path)
        if target is None:
            return False
        if int(self.headers.get("Content-Length", 0) or 0):
            self.close_connection = True     # the body is never read; see do_POST
        self._send(code, {"moved": target}, extra_headers=(("Location", target),))
        return True

    def do_GET(self):
        if self._redirect_retired(301):
            return
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if self._cross_origin_refused(path, "GET"):
            return
        if self._rate_refused(path):
            return
        prefixed = addons.prefix_route(path)
        if prefixed is not None:
            return prefixed(addons.CONTEXT, self, "GET", path, qs)
        if path == "/" or path == "/index.html":
            return self._index()
        if path in ("/memory", "/api/memory") and not _memory_enabled(
                self.headers, (self.client_address or ("",))[0]):
            return self._send(404, {"error": "not found"})
        if path == "/memory":
            return self._static("memory.html")
        page = addons.page_file(path)
        if page is not None:
            return self._static_file(page)
        # Browsers ask for these at the ROOT whatever the document links to,
        # so serve the icons there as well as under /static/icons/.
        if path == "/favicon.ico":
            return self._static("icons/favicon.ico")
        if path in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"):
            return self._static("icons/pyroorb-180.png")
        if path == "/api/memory":
            return self._memory(qs)
        if path == "/api/idqueue_links":
            # where the identification queue's pages live (IDQUEUE_URL's host): the Tools menu
            from pyroorb.web.engine_api import queue_pages
            return self._send(200, queue_pages())
        addon_get = addons.route("GET", path)
        if addon_get is not None:
            code, body = addon_get(addons.CONTEXT, self, qs)
            return self._send(code, body)
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/api/nongrav_models":
            return self._send(200, {"models": engine_api.NONGRAV_MODELS})
        if path == "/api/ephem_site":         # does ?site= have a horizon? (modal gating)
            return self._send(200, {"ok": True,
                                    "site_has_horizon": engine_api.site_has_horizon(
                                        (qs.get("site") or ["500"])[0] or "500")})
        if path == "/api/constraint_elements":
            return self._send(200, {"elements": engine_api.CONSTRAINABLE_ELEMENTS})
        if path == "/api/fit":
            # COMPATIBILITY PATH (bookmarks, scripts, an older cached client).
            # It no longer fits on the request thread: it submits to the same
            # queue as everyone else and waits a bounded time for the answer, so
            # a quick object still comes back inline while a slow one hands back
            # a job handle to poll instead of dying at the proxy.  Admission is
            # what matters here -- a route that skipped the pool would make the
            # pool decorative.
            desig = (qs.get("desig") or [""])[0]
            force = (qs.get("force") or ["0"])[0] in ("1", "true", "yes")
            if not desig:
                return self._send(400, {"ok": False, "error": "missing ?desig="})
            try:
                started = engine_api.run_fit_queued(desig, force=force)
                self._refund_if_free(path, started)
                if not started.get("ok"):
                    return self._send(200, started)
                # Already answered -- from the live table or from disk.  Return
                # the payload FLAT, as this route always has.  A disk hit
                # carries job:None, so falling through would wait(None), fail
                # instantly and hand a bookmark or script a pending stub
                # pointing at "/api/fit_result?id=None" while the real answer
                # sat one key away in the same dict.
                if started.get("done") and started.get("result"):
                    return self._send(200, started["result"])
                job = started["job"]
                if engine_api.fitqueue.wait(job, _SYNC_FIT_DEADLINE):
                    out = engine_api.fit_job_status(job)
                    return self._send(200, out.get("result") or out)
                started["pending"] = True
                started["hint"] = ("still fitting -- poll /api/fit_result?id="
                                   + str(job))
                return self._send(200, started)
            except Exception as ex:  # never 500 with an empty body -> show the message
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/fit_result":
            try:
                return self._send(200, engine_api.fit_job_status(
                    (qs.get("id") or [""])[0]))
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        if path == "/healthz":
            return self._send(200, self._healthz())
        if path == "/api/export/mpec":
            return self._export_mpec(qs)
        if path == "/api/export/ephem":
            return self._export_ephem(qs)
        if path == "/api/export/state":
            return self._export_state(qs)
        if path == "/api/export/covar":
            return self._export_covar(qs)
        if path in ("/api/export/elements", "/api/export/mpcorb", "/api/export/sof"):
            return self._export_egress(path.rsplit("/", 1)[1], qs)
        if path == "/api/ephem":
            return self._ephem_json(qs)
        if path == "/api/impact":
            return self._impact_json(qs)
        if path == "/api/itf_trace":
            try:
                return self._send(200, engine_api.itf_trace_feed(
                    (qs.get("id") or [""])[0], (qs.get("from") or ["0"])[0],
                    (qs.get("level") or [None])[0]))
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/itf_result":
            try:
                return self._send(200, engine_api.itf_search_status(
                    (qs.get("id") or [""])[0]))
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/itf_defaults":
            try:
                return self._send(200, engine_api.itf_defaults())
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/published_orbit":
            # The MPC's own orbit, in one cheap request -- shown while OUR fit of the
            # same object is still running (a 18k-observation arc takes minutes). The
            # payload is stamped `source: "MPC"`; it is never our result.
            try:
                return self._send(200, engine_api.published_orbit(
                    (qs.get("desig") or [""])[0]))
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/random_families":
            return self._send(200, engine_api.random_families())
        if path == "/api/random_object":
            try:
                return self._send(200, engine_api.random_object(
                    (qs.get("family") or ["any"])[0]))
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/environment":
            try:
                return self._send(200, engine_api.environment_status())
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        return self._send(404, {"error": "not found"})

    def _healthz(self):
        """Liveness + LOAD, in one cheap read.

        The Railway healthcheck points at "/", which serves static HTML and so
        keeps answering happily while every fit times out -- good for avoiding a
        restart loop, useless for knowing anything is wrong.  This is the route
        that can actually say so: queue depth, worker occupancy and resident
        size against the container's own limit.  No fit, no heap walk, no lock
        held longer than a dict read, so it stays answerable precisely when the
        box is busy.
        """
        # WHICH CODE is answering.  The process already knows -- the revision is
        # folded into every fit cache key so a deploy cannot serve a payload the
        # previous version computed -- but nothing exposed it, so "is my fix live
        # yet?" had to be inferred from side effects (a reset disk cache, an RSS
        # drop), which are also what a plain restart looks like.  A deploy marker
        # that has to be guessed at is one that gets guessed wrong during exactly
        # the incident where it matters.
        out = {"ok": True, "revision": engine_api.fitqueue._engine_rev(),
               "fits": engine_api.fit_queue_stats(),
               "rate_limit": ratelimit.stats()}
        try:
            out["itf"] = engine_api.itf_slots_now()
        except Exception:
            pass
        try:
            from pyroorb.itf import memory as _mem
            out["memory"] = {"rss_mb": engine_api._rss_mb(),
                             "limit_mb": _mem.limit_mb(),
                             "available_mb": _mem.available_mb()}
        except Exception:
            pass
        # Sweeps in flight and how old the oldest is -- a thread parked inside one
        # integrate() call for an hour is the outage the deadline cannot reach, and
        # this is where a monitor sees it (sweepwatch.WEDGED_SEC).  Cheap: a dict scan.
        try:
            from pyroorb import sweepwatch
            out["sweeps"] = sweepwatch.health()
        except Exception:
            pass
        q = out["fits"]
        out["saturated"] = bool(q.get("queued", 0) >= q.get("queue_max", 1))
        return out

    # --- export helpers ----------------------------------------------------
    def _resolve_fit(self, qs):
        """A FittedOrbit for export. Prefers the live ?session= fit -- the orbit the
        user is actually looking at, reflecting their current included/excluded mask,
        nongrav model, epoch and any Full/Multi steps -- so an export never silently
        re-runs a fresh autofit that discards those edits. Falls back to ?key= (a
        cached/pasted fit) or ?desig= (fetched + fitted cache-first). Raises with a
        readable message."""
        sid = (qs.get("session") or [""])[0]
        if sid:
            fit = engine_api.session_fit(sid)
            if fit is not None:
                return fit
            # session expired (evicted after _SES_MAX): fall through to key/desig so
            # the export still succeeds rather than dead-ending on a stale handle.
        key = (qs.get("key") or [""])[0]
        if key:
            fit = fitcache.get(key)
            if fit is None:
                raise ValueError("export handle expired -- re-fit the object")
            return fit
        desig = (qs.get("desig") or [""])[0]
        if not desig:
            raise ValueError("missing ?desig= or ?key=")
        force = (qs.get("force") or ["0"])[0] in ("1", "true", "yes")
        return fitcache.get_by_desig(desig, engine_api._obscodes(), force=force)

    def _export_mpec(self, qs):
        try:
            fit = self._resolve_fit(qs)
            refusal = _orbit_export_refusal(fit)
            if refusal:
                return self._send(200, refusal, "text/plain; charset=utf-8")
            html = exporters.pseudo_mpec(fit)
        except Exception as ex:
            return self._send(200, f"export failed: {type(ex).__name__}: {ex}",
                              "text/plain; charset=utf-8")
        return self._send(200, html, "text/html; charset=utf-8")

    def _export_egress(self, kind, qs):
        """Plain-text orbit egress for the Export menu: the find_orb 8-line element
        block ('elements'), the one-line MPCORB record ('mpcorb'), or the SOF
        record ('sof'). All read the engine's egress.py."""
        try:
            from pyroorb import egress
            fit = self._resolve_fit(qs)
            refusal = _orbit_export_refusal(fit)
            if refusal:
                return self._send(200, refusal, "text/plain; charset=utf-8")
            if kind == "elements":
                text = egress.to_eightline(fit)
            elif kind == "mpcorb":
                text = egress.MPCORB_HEADER + "\n" + egress.to_mpcorb(fit)
            else:                                  # sof
                text = egress.to_sof(fit)
        except Exception as ex:
            return self._send(200, f"export failed: {type(ex).__name__}: {ex}",
                              "text/plain; charset=utf-8")
        return self._send(200, text, "text/plain; charset=utf-8")

    def _export_state(self, qs):
        """Cartesian state vector + 6x6 covariance as JSON (egress.state_vector_document).
        Pretty-printed rather than compact because this lands in the Export copy/paste
        box; NaN-cleaned by hand for the same reason _send does it -- a degenerate fit
        must still hand back parseable JSON."""
        try:
            from pyroorb import egress
            fit = self._resolve_fit(qs)
            refusal = _orbit_export_refusal(fit)
            if refusal:
                return self._send(200, {"error": refusal})
            doc = _nan_safe(egress.state_vector_document(fit))
            text = json.dumps(doc, indent=2)
        except Exception as ex:
            return self._send(200, {"error": f"export failed: {type(ex).__name__}: {ex}"})
        return self._send(200, text, "application/json")

    def _export_covar(self, qs):
        """find_orb's `covar.json` for the displayed orbit (egress.to_covar_json).

        Served as text/plain, not application/json, precisely because it is written in
        find_orb's own byte layout -- the point of the product is that it can be diffed
        against a real find_orb run, and a JSON re-serialisation on the way out would
        destroy that. It is an orbit, so it obeys the same suppression refusal as the
        other element products."""
        try:
            from pyroorb import egress
            fit = self._resolve_fit(qs)
            refusal = _orbit_export_refusal(fit)
            if refusal:
                return self._send(200, refusal, "text/plain; charset=utf-8")
            text = egress.to_covar_json(fit)
        except Exception as ex:
            return self._send(200, f"export failed: {type(ex).__name__}: {ex}",
                              "text/plain; charset=utf-8")
        return self._send(200, text, "text/plain; charset=utf-8")

    def _export_ephem(self, qs):
        try:
            fit = self._resolve_fit(qs)
            obscodes = engine_api._obscodes()
            jd0 = float((qs.get("start") or [fit.epoch_tdb])[0])
            step = float((qs.get("step") or ["1.0"])[0])
            n = int(float((qs.get("n") or ["30"])[0]))
            jd1 = float((qs.get("stop") or [jd0 + step * max(n - 1, 0)])[0])
            site = (qs.get("site") or ["500"])[0] or "500"
            text = exporters.ephemeris_text(fit, obscodes, jd0, jd1, step, site=site,
                                            **_ephem_filters(qs))
        except Exception as ex:
            return self._send(200, f"export failed: {type(ex).__name__}: {ex}",
                              "text/plain; charset=utf-8")
        return self._send(200, text, "text/plain; charset=utf-8")

    def _ephem_json(self, qs):
        """Structured (JSON) ephemeris for the in-pane View Ephemeris table. Prefers the
        live ?session= fit (current toggles/model/epoch); falls back to ?key=/?desig=."""
        try:
            fit = self._resolve_fit(qs)
            jd0 = float((qs.get("start") or [fit.epoch_tdb])[0])
            step = float((qs.get("step") or ["1.0"])[0])
            n = int(float((qs.get("n") or ["30"])[0]))
            site = (qs.get("site") or ["500"])[0] or "500"
            data = engine_api.ephemeris_rows(fit, jd0, step, n, site=site,
                                             **_ephem_filters(qs))
        except Exception as ex:
            return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        return self._send(200, data)

    def _impact_json(self, qs):
        """Phase-1 impact LOCATION for the in-pane Impact view. Prefers the live
        ?session= fit (current toggles/model/epoch); falls back to ?key=/?desig=."""
        try:
            fit = self._resolve_fit(qs)
            window = float((qs.get("window") or ["30"])[0])
            data = engine_api.impact_assessment(fit, window_days=window)
        except Exception as ex:
            return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        return self._send(200, data)

    # JSON-body POST endpoints: path -> handler(body dict).  Every entry reads
    # one JSON object body and answers HTTP 200 with the handler's dict, or
    # {ok: False, error} on an exception (the client branches on `ok`, an
    # established convention).  /api/fit_text and /api/add_obs stay special:
    # their bodies are raw astrometry text, options ride the query string.
    _JSON_POSTS = {
        # Queued fitting.  /api/fit_start submits and answers at once with a job
        # handle + queue position; the client polls GET /api/fit_result.  A fit
        # is never run on the request thread any more: under a crowd that is
        # both an OOM (a hundred concurrent fits) and a proxy timeout (a slow
        # one), which is the same pair of problems /api/itf_search solved.
        "/api/fit_start": lambda d: engine_api.run_fit_queued(
            d.get("desig") or "", force=bool(d.get("force")),
            fresh=bool(d.get("fresh"))),
        # Claim a PRIVATE interactive session on a shared (single-flighted) fit.
        # The client asks for one the first time the user actually interacts, so
        # a hundred people looking at one object cost one session, not a hundred.
        "/api/session": lambda d: engine_api.session_from_job(d.get("job")),
        # An interactive step is a full differential correction on a live
        # session -- admitted and time-bounded exactly like a fit (_pooled).
        "/api/fullstep": lambda d: _pooled("fullstep", lambda: engine_api.full_step(
            d.get("session"), d.get("mask") or [], single=bool(d.get("single")))),
        # `async: true` (what the shipped client sends) starts the search on a
        # worker thread and answers with a job handle at once -- an ITF search
        # outlives a hosted proxy's request limit, and a proxy that gives up
        # replies with a non-JSON body the client cannot read (Railway's
        # "upstream error").  Polled out via GET /api/itf_result.  An older
        # cached client that omits the flag still gets the blocking answer --
        # through itf_search_sync, which is admission-gated exactly like the
        # async path (the raw itf_search_web bypassed the concurrency cap and
        # the unusable-box refusal).
        "/api/itf_search": lambda d: (
            engine_api.itf_search_start if d.get("async")
            else engine_api.itf_search_sync)(
            d.get("session"), d.get("options") or {},
            mask_list=d.get("mask"), fit_key=d.get("fit_key")),
        # The user's Stop: sets the job's cancel token and answers at once (not
        # rate-limited -- it FREES capacity, and a user hammering it costs a dict
        # lookup).  The job id is the handle the client picked when it started
        # the search, so only the page that started a search can stop it.
        "/api/itf_cancel": lambda d: engine_api.itf_search_cancel(d.get("job")),
        # pooled: a Submit may re-judge leave-out cautions on the live observation
        # set (_leaveout_rejudge), up to one refit each -- fit-class work
        "/api/itf_submit": lambda d: _pooled("itf_submit", lambda: engine_api.itf_submit_web(
            d.get("session"), d.get("acked_warnings"), name=d.get("name"),
            email=d.get("email"), route=d.get("route") or "queue")),
        "/api/itf_download": lambda d: engine_api.itf_download_web(
            d.get("session"), d.get("name"), d.get("email")),
        # Re-auto-solve re-runs the whole automatic pipeline: a full fit, and a
        # button any visitor can hold down. Admitted like every other fit.
        "/api/reauto": lambda d: engine_api.fitqueue.run_bounded(
            lambda: engine_api.reauto_solve(
                session=d.get("session"), fit_key=d.get("fit_key")),
            _SYNC_FIT_DEADLINE, label="reauto"),
        "/api/nongrav": lambda d: _pooled("nongrav", lambda: engine_api.run_nongrav(
            d.get("session"), d.get("model"), d.get("mask"))),
        # The jackknife is up to JACKKNIFE_MAX_REFITS independent refits: the
        # most expensive thing behind any single button on the page.
        "/api/nongrav_sigmas": lambda d: _pooled(
            "nongrav_sigmas", lambda: engine_api.nongrav_sigmas(d.get("session"))),
        "/api/constrain": lambda d: _pooled("constrain", lambda: engine_api.run_constrained(
            d.get("session"), d.get("constraints") or {}, d.get("mask"))),
        "/api/asteroids": lambda d: _pooled("asteroids", lambda: engine_api.set_asteroids(
            d.get("session"), d.get("on"), d.get("mask"))),
        "/api/set_epoch": lambda d: _pooled("set_epoch", lambda: engine_api.set_epoch(
            d.get("session"), d.get("epoch"), d.get("mask"))),
        # "Show me an orbit anyway" on an undetermined arc (engine_api.nominal_optin). Pooled:
        # on a tracklet it reads the range profile, which a crowd must not run unbounded.
        # digest2 tracklet by tracklet for the observations table (engine_api.d2_tracklets).
        "/api/d2_tracklets": lambda d: _pooled("d2_tracklets", lambda: engine_api.d2_tracklets(
            d.get("session"), d.get("fit_key"))),
        "/api/nominal": lambda d: _pooled("nominal", lambda: engine_api.nominal_optin(
            d.get("session"), d.get("on", True) is not False)),
    }

    def do_POST(self):
        if self._redirect_retired(308):
            return
        u = urlparse(self.path)
        # UNDRAINED-BODY GUARD.  Every early return below this point, up to and
        # including _read_body() itself, can fire BEFORE the request body has
        # been read off the socket.  Under HTTP/1.1 keep-alive that is not
        # cosmetic: the unread bytes sit in the stream, and the client's NEXT
        # request on the same reused connection gets those bytes prepended to
        # its own request line.  Reproduced exactly: a 429 on request 2 left its
        # JSON body undrained, and request 3 on the same socket came back
        # `Bad request syntax ('{"desig": "2003 LW2"}POST /api/session
        # HTTP/1.1')` -- the leftover body literally concatenated onto the next
        # request line.  Through a proxy that silently drops the mangled
        # response instead of surfacing this 400, it reads as
        # "RemoteDisconnected: Remote end closed connection without response" --
        # measured on the live deploy at roughly 1 in 8 to 1 in 12 POSTs under
        # concurrent load, including requests with NOTHING to do with rate
        # limiting or CORS, because the corruption follows the CONNECTION, not
        # the request that caused it.
        #
        # Every early exit here forces the connection closed instead of trying
        # to drain a body it was never going to read anyway (untrusted length,
        # possibly absent, possibly enormous) -- BaseHTTPRequestHandler then
        # tears the socket down after this response rather than reusing it, so
        # nothing is left for the next request on it to trip over.
        if self._cross_origin_refused(u.path, "POST"):
            self.close_connection = True
            return
        if self._rate_refused(u.path):
            self.close_connection = True
            return
        prefixed = addons.prefix_route(u.path)
        if prefixed is not None:
            return prefixed(addons.CONTEXT, self, "POST", u.path, parse_qs(u.query))
        if u.path in ("/api/fit_text", "/api/add_obs"):
            qs = parse_qs(u.query)
            raw = self._read_body()
            if raw is None:
                return
            text = raw.decode("utf-8", "replace")
            force = (qs.get("force") or ["0"])[0] in ("1", "true", "yes")
            # Both of these run a FULL fit, so they go through the same worker
            # pool as /api/fit -- one-off (nothing to coalesce: a pasted blob is
            # unique), but admitted, because the pool only protects the box if
            # every path that can start a fit goes through it.
            try:
                if u.path == "/api/fit_text":
                    return self._send(200, engine_api.fitqueue.run_bounded(
                        lambda: loader.fit_pasted(
                            text, engine_api._obscodes(), force=force),
                        _SYNC_FIT_DEADLINE, label="paste"))
                return self._send(200, engine_api.fitqueue.run_bounded(
                    lambda: loader.add_to_fit(
                        text, engine_api._obscodes(),
                        session=(qs.get("session") or [""])[0] or None,
                        fit_key=(qs.get("fit_key") or [""])[0] or None,
                        force=force),
                    _SYNC_FIT_DEADLINE, label="add_obs"))
            except Exception as ex:
                return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        addon_post = addons.route("POST", u.path)
        handler = self._JSON_POSTS.get(u.path)
        if handler is None and addon_post is None:
            self.close_connection = True     # see the guard note above do_POST
            return self._send(404, {"error": "not found"})
        raw = self._read_body()
        if raw is None:
            return
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"ok": False, "error": "bad JSON body"})
        try:
            if handler is None:
                code, out = addon_post(addons.CONTEXT, self, data if isinstance(data, dict) else {})
                return self._send(code, out)
            out = handler(data)
            self._refund_if_free(u.path, out)
            return self._send(200, out)
        except Exception as ex:
            return self._send(200, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})


def _obscodes_refresh_hours():
    try:
        return float(os.environ.get("PYROORB_OBSCODES_REFRESH_HOURS", "12"))
    except ValueError:
        return 12.0


def _obscodes_refresh_loop(hours, first_delay=60.0):
    """Re-check the MPC station table every `hours` (first check shortly after boot, so
    a box whose cached copy is months old is fixed at once). One failed check is
    logged and retried at the next tick; the cached table stays in use meanwhile."""
    import time
    from pyroorb import obs_sources
    time.sleep(first_delay)
    while True:
        try:
            r = obs_sources.refresh_obscodes()
            if r["error"]:
                print(f"obscodes: refresh failed, keeping the cached table ({r['error']})")
            elif r["changed"]:
                print(f"obscodes: table updated from {r['source']} -- {r['n']} stations, "
                      f"added {', '.join(r['added']) or 'none'}, "
                      f"removed {', '.join(r['removed']) or 'none'}")
        except Exception as e:                # pragma: no cover - defensive
            print(f"obscodes: refresh error ({e!r})")
        time.sleep(hours * 3600.0)


def _maybe_start_obscodes_refresh():
    """Start the periodic station-table refresh (PYROORB_OBSCODES_REFRESH_HOURS, default
    12; 0 turns it off). Guarded: a refresh problem never breaks web boot."""
    hours = _obscodes_refresh_hours()
    if hours <= 0:
        print("obscodes: periodic refresh disabled (PYROORB_OBSCODES_REFRESH_HOURS=0)")
        return
    import threading
    threading.Thread(target=_obscodes_refresh_loop, args=(hours,),
                     name="obscodes-refresh", daemon=True).start()
    print(f"obscodes: refresh thread started (MPC, find_orb fallback; every {hours:g} h)")


def _maybe_start_itf_maintenance():
    """Start the ITF-index maintenance scheduler in a daemon thread when
    ``ITF_MAINTAIN_TURNED_ON`` is set (off by default).

    The jobs are database-I/O-bound, so the thread never contends for the CPU the
    web server needs; the import and start are fully guarded so a maintenance-side
    problem can never break web boot or the Railway healthcheck."""
    try:
        from pyroorb.itf.scheduler import maintain_enabled, run_scheduler
    except Exception as e:                    # pragma: no cover - defensive
        print(f"itf-maintain: scheduler unavailable ({e!r}); web serving only")
        return
    if not maintain_enabled():
        print("itf-maintain: disabled (set ITF_MAINTAIN_TURNED_ON=True to enable)")
        return
    import threading
    threading.Thread(target=run_scheduler, name="itf-maintain",
                     daemon=True).start()
    print("itf-maintain: scheduler thread started "
          "(refresh ~60s, reconcile daily, sweep weekly)")


# What the server lends its web add-ons (web/addons.py): set at import so a test that drives
# Handler directly sees the same context as a running server.
# Each callable looks its target up when called, so a patched engine_api is seen too.
addons.CONTEXT = addons.Context(
    pooled=lambda label, work: _pooled(label, work),
    itf_slot=lambda *a, **k: engine_api.itf_background_slot(*a, **k),
    obscodes=lambda: engine_api._obscodes(),
    fit_for=lambda session=None, fit_key=None: engine_api._fit_for(session, fit_key))


def main():
    ap = argparse.ArgumentParser()
    # $PORT is the platform-assigned port on PaaS hosts (Railway/Render/Heroku);
    # honour it by default so the same command works locally and when deployed.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    # Default to all interfaces: dev runs inside a Linux microVM/sandbox and the
    # browser is on the host, so it must reach the VM's IP -- 127.0.0.1 would be
    # VM-loopback-only and invisible from the host. Pass --host 127.0.0.1 to scope
    # it back to loopback when running directly on the same machine as the browser.
    # $HOST overrides too (set HOST=127.0.0.1 to scope to loopback without a flag).
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()
    _maybe_start_itf_maintenance()
    _maybe_start_obscodes_refresh()
    addons.start_all(addons.CONTEXT)
    # THE LISTEN BACKLOG, not a detail.  socketserver.TCPServer defaults
    # request_queue_size to FIVE, so a burst of simultaneous connections is
    # refused by the kernel before a single line of our code runs -- the queue,
    # the cache and the limiter all sit downstream of an accept() that never
    # happened, and the visitor sees a connection reset rather than any of the
    # graceful answers this server works hard to give.  Measured locally: 60
    # simultaneous requests produced a ConnectionResetError at the default, and
    # none at 128.  A hundred people opening one page at once is exactly the
    # event the rest of this design is for.
    ThreadingHTTPServer.request_queue_size = int(
        os.environ.get("PYROORB_LISTEN_BACKLOG") or 128)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PyRoOrb web UI  ->  http://{args.host}:{args.port}")
    print("  (Phase 0 read-only viewer; Ctrl-C to stop)")
    if args.host == "0.0.0.0":
        print("  from the host browser, try http://localhost:%d  (or the VM's IP:%d)"
              % (args.port, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
