"""SBN database connections for the ITF search.

Two roles on the shared SBN PostgreSQL server (see ``ItfConfig`` for host/db):

* ``pub``  -- read-only public data; used by the search itself.
* ``itf``  -- read-only on ``public`` but OWNS the ``itf`` schema (the tracklet
  index lives there); used by all index-maintenance jobs.

Passwords are never stored in the repo: they come from the environment
(``SBN_PUB_PASSWORD`` / ``ITF_DB_PASSWORD``) or a gitignored ``.env`` at the
repo root.  The server is shared and load-sensitive -- callers must batch
writes (`execute_values`) and never seq-scan ``obs_sbn`` (~538M rows).

Sandboxed environments often have NO direct TCP route -- all egress rides an
HTTP proxy (``https_proxy`` env).  Postgres clients don't speak HTTP CONNECT,
so when a direct connection is refused and a proxy is configured, this module
transparently starts a tiny local forwarder (one thread per connection) that
tunnels ``127.0.0.1:<port> -> proxy CONNECT -> server:5432`` and points the
driver at it.  The proxy must allow CONNECT to the DB host (e.g. Docker
sandboxes: ``docker sandbox network proxy <name> --bypass-cidr <db-cidr>``;
takes effect immediately).
"""

import os
import socket
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_dotenv():
    """Populate ``os.environ`` from a gitignored ``.env`` (repo root, then cwd
    walking up).  Only sets keys not already present, so a real exported
    variable always wins.  Values may be quoted."""
    candidates = [_REPO_ROOT / ".env"]
    here = Path.cwd().resolve()
    for _ in range(6):
        candidates.append(here / ".env")
        if here.parent == here:
            break
        here = here.parent
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        return


_TRUTHY = {"1", "true", "t", "yes", "y", "on"}


def flag_enabled(name):
    """True iff the env var ``name`` is set truthy ("1/true/t/yes/y/on",
    case-insensitive).  A real (exported / Railway) variable always wins; the
    gitignored repo-root ``.env`` is consulted only as a fallback
    (:func:`load_dotenv` uses setdefault).  THE way to read a boolean feature
    flag in this package -- one parser, one .env convention."""
    try:
        load_dotenv()
    except Exception:
        pass
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def env_default(name, fallback):
    """The env var ``name`` (with the same ``.env`` fallback rules as
    :func:`flag_enabled`), or ``fallback`` when unset/empty."""
    try:
        load_dotenv()
    except Exception:
        pass
    return os.environ.get(name, "").strip() or fallback


def _require_host(host, role):
    """The configured DB host, or a clear error.  There is no built-in default:
    psycopg2 given no host silently tries a local Unix socket, which reads as a
    database problem rather than a missing setting."""
    if not host:
        raise RuntimeError(
            f"DB host not set: export SBN_DB_HOST (role {role!r}), set it on the "
            f"config, or add it to .env at the repo root")
    return host


def _resolve_password(explicit, env_name, role):
    if explicit:
        return explicit
    pw = os.environ.get(env_name)
    if not pw:
        load_dotenv()
        pw = os.environ.get(env_name)
    if not pw:
        raise RuntimeError(
            f"DB password not set: export {env_name} (role {role!r}), "
            f"set it on the config, or add it to .env at the repo root")
    return pw


# ---------------------------------------------------------------------------
# HTTP-CONNECT tunnel for proxy-only sandboxes
# ---------------------------------------------------------------------------
_TUNNELS = {}                    # (host, port) -> local port
_TUNNEL_LOCK = threading.Lock()


def _proxy_addr():
    url = (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
           or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))
    if not url:
        return None
    hostport = url.split("://", 1)[-1].rstrip("/").split("@")[-1]
    host, _, port = hostport.partition(":")
    return host, int(port or 3128)


def _open_via_proxy(target_host, target_port, proxy, timeout):
    s = socket.create_connection(proxy, timeout=timeout)
    # keep the tunnel alive across long CPU-bound gaps between queries --
    # idle CONNECT tunnels are a favorite thing for proxies to reap
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, val in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10),
                     ("TCP_KEEPCNT", 3)):
        if hasattr(socket, opt):
            s.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
    s.sendall(f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
              f"Host: {target_host}:{target_port}\r\n\r\n".encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
    status = resp.split(b"\r\n", 1)[0].decode(errors="replace")
    if " 200" not in status:
        s.close()
        raise OSError(f"proxy CONNECT to {target_host}:{target_port} refused: "
                      f"{status or '(no response)'} -- the proxy must allow "
                      f"this host (docker sandbox: --bypass-cidr)")
    # BLOCKING from here on.  `timeout` is a CONNECT-handshake budget, and
    # `socket.create_connection` leaves it on the socket for every later
    # operation -- so `_pump`'s recv() raised socket.timeout after `timeout`
    # seconds of QUIET, its `finally` shut both ends down, and the forwarder
    # tore down its own tunnel every time the caller paused.  With
    # db_connect_timeout at 15 s that made any gap longer than 15 seconds fatal:
    # measured, a session died at exactly 15.0 s idle while the raw tunnel with
    # the timeout cleared sat open indefinitely.  An ITF search queries in
    # bursts and got away with it; anything that THINKS between queries (the
    # ramp-gate calibration fits orbits for minutes) reconnected on essentially
    # every query -- 102 reconnects in 7 objects, and 38 of 70 objects lost
    # outright before the caller retried with backoff.  It reads as "the server
    # keeps dropping us"; it was us.
    s.settimeout(None)
    return s


def _pump(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for sk in (a, b):
            try:
                sk.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _ensure_tunnel(target_host, target_port, timeout):
    """Local forwarder port for target (one per process, lazily started)."""
    proxy = _proxy_addr()
    if proxy is None:
        raise OSError("no https_proxy configured and direct TCP refused")
    key = (target_host, target_port)
    with _TUNNEL_LOCK:
        if key in _TUNNELS:
            return _TUNNELS[key]
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(16)
        local_port = lsock.getsockname()[1]

        def serve():
            while True:
                try:
                    client, _addr = lsock.accept()
                except OSError:
                    return
                try:
                    upstream = _open_via_proxy(target_host, target_port,
                                               proxy, timeout)
                except OSError:
                    client.close()
                    continue
                threading.Thread(target=_pump, args=(client, upstream),
                                 daemon=True).start()
                threading.Thread(target=_pump, args=(upstream, client),
                                 daemon=True).start()

        threading.Thread(target=serve, daemon=True).start()
        _TUNNELS[key] = local_port
        return local_port


def _connect(host, port, user, password, dbname, timeout,
             statement_timeout_sec=0):
    """psycopg2 connection, transparently tunneled through the HTTP proxy
    when the sandbox has no direct route.

    ``statement_timeout_sec`` (0 = none) arms the SERVER-side statement
    timeout.  Server-side is the point: a cancelled query stops costing the
    SHARED server, where a client socket timeout would abandon us but leave it
    running.  ``connect_timeout`` bounds only the handshake -- nothing bounded
    a query, which is how one corridor sweep hung a whole search (see
    `DbSource.sweep_candidates`)."""
    import psycopg2
    opts = {}
    if statement_timeout_sec:
        opts["options"] = f"-c statement_timeout={int(statement_timeout_sec * 1000)}"
    try:
        return psycopg2.connect(database=dbname, host=host, port=port,
                                user=user, password=password,
                                connect_timeout=timeout, **opts)
    except psycopg2.OperationalError as e:
        if "refused" not in str(e).lower():
            raise
    local_port = _ensure_tunnel(host, port, timeout)
    return psycopg2.connect(database=dbname, host="127.0.0.1",
                            port=local_port, user=user, password=password,
                            connect_timeout=timeout, **opts)


def connect_pub(cfg):
    """Read-only search connection (the ``pub`` role)."""
    host = _require_host(cfg.db_host, cfg.db_user)
    password = _resolve_password(cfg.db_password, cfg.db_password_env, cfg.db_user)
    # The SEARCH role only: the index-maintenance jobs run on connect_index
    # and legitimately take many minutes (the weekly sweep), so they must not
    # inherit a query ceiling sized for a corridor chunk.
    return _connect(host, cfg.db_port, cfg.db_user, password,
                    cfg.db_name, cfg.db_connect_timeout,
                    statement_timeout_sec=cfg.sweep_statement_timeout_sec)


def connect_index(cfg):
    """Writable connection for the ``itf.itf_tracklets`` index (the ``itf``
    role; ``search_path = itf, public``)."""
    host = _require_host(cfg.db_index_host, cfg.db_index_user)
    password = _resolve_password(cfg.db_index_password, cfg.db_index_password_env,
                                 cfg.db_index_user)
    return _connect(host, cfg.db_port, cfg.db_index_user, password,
                    cfg.db_name, cfg.db_connect_timeout)
