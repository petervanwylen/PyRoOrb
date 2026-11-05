"""Refuse to run against an ASSIST that was built for a different REBOUND.

ASSIST is a C library compiled against REBOUND's headers and loaded against REBOUND's shared
library, so the two must be the same release. When they are not (an older REBOUND already in
the environment while ``pip`` builds ASSIST in isolation against the newest 4.x; a later
``pip install -U rebound``; a cached ASSIST wheel reused by ``uv`` in another environment) the
struct layouts differ. Measured 2026-11-03 with ASSIST built against REBOUND 4.6.0 and run
against 4.4.11: the first integration returned the UNCHANGED state with an IAS15
non-convergence warning, then the process aborted (``corrupted size vs. prev_size``, exit 134).
No traceback, a dead Jupyter kernel -- and ASSIST records no REBOUND version to compare, so
the versions cannot be checked, only the behaviour.

So the behaviour is checked: a short ASSIST integration runs in a child process (a crash there
costs nothing), once per (REBOUND, ASSIST library) pair, and the verdict is stamped in the
cache directory (about 0.3 s, once). Only the two signatures of the mismatch raise -- the child
dying on a signal (on Windows: exiting with an NTSTATUS error code), or an integration that
left the state unchanged. Any other failure of the
check (no kernel, no subprocess) is not evidence of a mismatch and lets the fit go on.
``PYROORB_SKIP_ABI_CHECK=1`` turns the check off.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SKIP_ENV = "PYROORB_SKIP_ABI_CHECK"
_UNCHANGED = 3                      # the child's exit code for "integrated, but nothing moved"
_verified = False
# Windows reports a native crash as the process's positive exit code, the NTSTATUS of the
# exception: 0xC0000005 access violation, 0xC0000374 heap corruption, 0xC0000409 fail-fast.
# Every NTSTATUS error has the top two bits set; a POSIX exit code never exceeds 255.
_NTSTATUS_ERROR = 0xC0000000


def _crashed(returncode):
    """True when the canary died natively: a POSIX signal (negative), bash's 128+SIGABRT, or
    a Windows exception code."""
    return returncode < 0 or returncode == 134 or returncode >= _NTSTATUS_ERROR

_CANARY = r"""
import sys, rebound, assist
sim = rebound.Simulation()
eph = assist.Ephem(sys.argv[1])
sim.t = 2460000.5 - eph.jd_ref
sim.add(x=1.5, y=0.1, z=0.0, vx=0.0, vy=0.01, vz=0.002)
ex = assist.Extras(sim, eph)
x0 = sim.particles[0].x
sim.integrate(sim.t + 100.0)
sys.exit(0 if abs(sim.particles[0].x - x0) > 1e-3 else 3)
"""


class AssistMismatchError(RuntimeError):
    """ASSIST was built against a different REBOUND than the one installed."""


def _fix_message(reason):
    import rebound
    return (f"ASSIST cannot run against the installed REBOUND {rebound.__version__} ({reason}). "
            "ASSIST is compiled against REBOUND's headers and the two must be the same "
            "release; this happens when REBOUND was installed or upgraded after ASSIST was "
            "built. Rebuild ASSIST against the installed REBOUND:\n"
            "    pip install setuptools wheel\n"
            "    pip install --no-build-isolation --force-reinstall --no-deps "
            "--no-cache-dir assist\n"
            "(with uv: `uv cache clean assist`, then `uv pip install setuptools wheel` and "
            "`uv pip install --reinstall --no-build-isolation --no-deps assist`), or re-run "
            "install.sh. Nothing was fitted: continuing would crash this process.")


def _key():
    import assist
    import rebound
    parts = [rebound.__version__, getattr(assist, "__version__", "?")]
    for mod in (assist, rebound):
        # the compiled library, not the Python wrapper, is what must match
        path = getattr(mod, "__libpath__", None) or getattr(mod, "__file__", "")
        try:
            st = os.stat(path)
            parts.append(f"{path}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(str(path))
    return "|".join(parts)


def _stamp_path():
    from pyroorb import datadirs
    return Path(datadirs.cache_dir()) / "abi_check.json"


def verify(planets_kernel):
    """Raise :class:`AssistMismatchError` if ASSIST and REBOUND do not match; otherwise return.

    ``planets_kernel`` is the DE440 file the canary integrates with. Cheap after the first
    call in a process (a flag) and after the first call per installed pair (a stamp file)."""
    global _verified
    if _verified or os.environ.get(SKIP_ENV, "").strip() not in ("", "0"):
        return
    try:
        key = _key()
    except Exception:
        return
    stamp = None
    try:
        stamp = _stamp_path()
        if json.loads(stamp.read_text(encoding="utf-8")).get("ok") == key:
            _verified = True
            return
    except Exception:
        pass
    try:
        run = subprocess.run([sys.executable, "-I", "-c", _CANARY, str(planets_kernel)],
                             capture_output=True, encoding="utf-8", errors="replace",
                             timeout=120)
    except Exception:
        return                      # cannot run the check here: not evidence of a mismatch
    if _crashed(run.returncode) or run.returncode == _UNCHANGED \
            or "corrupted" in run.stderr or "free(): invalid" in run.stderr:
        reason = ("the test integration left the state unchanged" if run.returncode == _UNCHANGED
                  else "the test integration crashed")
        raise AssistMismatchError(_fix_message(reason))
    if run.returncode == 0:
        _verified = True
        try:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(json.dumps({"ok": key}), encoding="utf-8")
        except Exception:
            pass
