#!/usr/bin/env python3
"""Auto-reloading dev runner for the PyRoOrb web server.

Runs ``server.py`` as a child process and restarts it whenever any ``.py`` file
in the package changes -- so backend edits (engine_api, server, loader, the
engine modules, …) go live without a manual restart. Static assets
(html/css/js) never needed a restart anyway; just refresh the browser.

    .venv/bin/python -m pyroorb.web.devserver              # -> http://0.0.0.0:8000
    .venv/bin/python -m pyroorb.web.devserver --port 9000

Ctrl-C stops it. If an edit has a syntax error the child will exit; fix the file
and the next save restarts it automatically. `the download cache` (API caches written
during fits), ``__pycache__`` and ``tests`` are ignored so normal fitting does
NOT trigger a reload.
"""
import os
import sys
import time
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)            # src/pyroorb
SERVER = os.path.join(_HERE, "server.py")
_IGNORE_DIRS = {"__pycache__", "data", "tests", ".git"}


def _watched():
    """Mtimes of every .py in the package (minus caches/tests)."""
    snap = {}
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    snap[p] = os.stat(p).st_mtime
                except OSError:
                    pass
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default="8000")
    a = ap.parse_args()
    cmd = [sys.executable, SERVER, "--host", a.host, "--port", str(a.port)]

    def spawn():
        print(f"[devserver] start: {' '.join(cmd)}", flush=True)
        return subprocess.Popen(cmd)

    proc = spawn()
    prev = _watched()
    warned_dead = False
    try:
        while True:
            time.sleep(1.0)
            cur = _watched()
            changed = sorted(f for f in cur if cur[f] != prev.get(f))
            changed += sorted(f for f in prev if f not in cur)   # deletions
            if changed:
                names = ", ".join(os.path.relpath(c, _PKG) for c in changed[:4])
                more = "" if len(changed) <= 4 else f" (+{len(changed) - 4})"
                print(f"[devserver] changed: {names}{more} -> restarting", flush=True)
                prev = cur
                warned_dead = False
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                proc = spawn()
            elif proc.poll() is not None and not warned_dead:
                print("[devserver] server exited (syntax error?); waiting for a .py save",
                      flush=True)
                warned_dead = True
    except KeyboardInterrupt:
        print("\n[devserver] stopping", flush=True)
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
