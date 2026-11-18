"""What a failed request may tell its caller about the exception behind it.

An engine error names the problem in the caller's own terms ("no observations found",
a parse error at line 12) and goes back as it is.  A database driver or operating-system
error names the SERVER instead: a database driver's connect error carries the host, its
address, port and role ("connection to server at ... failed: FATAL: password authentication
failed for user ..."), a statement error carries an excerpt of the SQL, and an
``OSError`` carries file paths and peer addresses.  Those are logged in full on the server
and reach the caller only as their class name.
"""
import re
import sys
import traceback

#: modules whose exceptions describe the server's own database
_DB_MODULES = ("psycopg2", "psycopg", "sqlite3", "pg8000", "asyncpg")

#: message text that names a database endpoint or credential, whatever raised it
_SERVER_TEXT = re.compile(
    r"connection to server|server at \"|password authentication|FATAL:|"
    r"postgres(?:ql)?://|\bport \d{2,5}\b|\buser \"", re.IGNORECASE)

#: longest message passed through
MAX_MESSAGE = 300


def _chain(ex, depth=8):
    """``ex`` and the exceptions it was explicitly raised from (``raise ... from``)."""
    seen = []
    while ex is not None and len(seen) < depth and ex not in seen:
        seen.append(ex)
        ex = ex.__cause__
    return seen


def server_side(ex):
    """True when ``ex`` (or what it was raised from) describes the server, not the request."""
    for e in _chain(ex):
        mod = (type(e).__module__ or "").split(".")[0]
        if mod in _DB_MODULES or isinstance(e, OSError) or _SERVER_TEXT.search(str(e)):
            return True
    return False


def describe(ex, limit=MAX_MESSAGE):
    """``"Type: message"`` for a caller: the message capped at ``limit`` characters, or,
    for an error that describes the server, the class name alone (the whole traceback
    goes to the server log)."""
    name = type(ex).__name__
    if server_side(ex):
        log(ex)
        return f"{name}: internal error (details are in the server log)"
    msg = str(ex)
    if len(msg) > limit:
        msg = msg[:limit] + "…"
    return f"{name}: {msg}"


def log(ex):
    """Write ``ex`` with its traceback to the server's stderr."""
    try:
        print("".join(traceback.format_exception(type(ex), ex, ex.__traceback__)).rstrip(),
              file=sys.stderr, flush=True)
    except Exception:                                   # logging never fails a request
        pass
