"""UTF-8 standard streams for every PyRoOrb command.

The commands print sigma, degree, arc-second and arrow signs. On Linux and macOS standard
output is UTF-8; on Windows it is UTF-8 only on an interactive console, and a pipe or a file
(a batch child read by its parent, the web server's log, ``pyroorb X > out.txt``) gets the
ANSI code page, cp1252, which cannot encode sigma: the first such print raised
UnicodeEncodeError after the fit had finished. Each command therefore switches its streams
to UTF-8 at start-up, beside ``envfile.load_env_file``; the parents that read a child's
output decode it as UTF-8 to match. Importing ``pyroorb`` as a library changes neither stream.
"""
import sys


def utf8_stdio():
    """Re-open stdout and stderr as UTF-8 where they are not already (a no-op on Linux and
    macOS, and on a Windows console). A character that still cannot be written is escaped,
    never raised: a stream that refuses one sign must not lose the result printed around it."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass


def for_log(value, limit=80):
    """``value`` as a log line may carry it: unchanged when it is short printable ASCII,
    else escaped (``ascii``) -- so a request cannot write control characters (a forged
    line break, terminal escapes) or invisible text into the log -- and cut to ``limit``
    characters either way."""
    s = str(value)
    if not (s.isascii() and s.isprintable()):
        s = ascii(s)[1:-1]
    return s if len(s) <= limit else s[: limit - 3] + "..."
