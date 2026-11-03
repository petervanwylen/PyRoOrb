"""Text is UTF-8 on every platform.

Python's default text encoding is the locale's: UTF-8 on Linux and macOS, the ANSI code page
(cp1252) on Windows. A text-mode ``open`` without ``encoding=`` therefore reads a cached MPC
answer, an ADES file or the web UI's index.html correctly on one OS and raises
UnicodeDecodeError on another, and a print of sigma to a pipe raises UnicodeEncodeError.
These checks run on every OS, so a Linux run catches what would break on Windows."""
import ast
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "pyroorb"


def _text_io_without_encoding(path):
    """(line, call) for each text-mode builtin ``open``, ``read_text()``, ``write_text(s)``
    or text-mode subprocess call in ``path`` that names no encoding."""
    tree = ast.parse(path.read_bytes())
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        kws = {k.arg for k in n.keywords}
        if "encoding" in kws or None in kws:
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id == "open":
            mode = n.args[1] if len(n.args) > 1 else next(
                (k.value for k in n.keywords if k.arg == "mode"), None)
            if isinstance(mode, ast.Constant) and "b" in mode.value:
                continue
            out.append((n.lineno, "open"))
        elif isinstance(f, ast.Attribute) and f.attr == "read_text" and not n.args:
            out.append((n.lineno, "read_text"))
        elif isinstance(f, ast.Attribute) and f.attr == "write_text" and len(n.args) == 1:
            out.append((n.lineno, "write_text"))
        elif (isinstance(f, ast.Attribute) and f.attr in ("run", "check_output", "Popen")
              and ({"text", "universal_newlines"} & kws)):
            out.append((n.lineno, f"subprocess.{f.attr}"))
    return out


def test_engine_text_io_names_its_encoding():
    bad = [f"{p.relative_to(REPO).as_posix()}:{line}: {what}"
           for p in sorted(SRC.rglob("*.py"))
           for line, what in _text_io_without_encoding(p)]
    assert not bad, "text I/O on the locale encoding (cp1252 on Windows):\n" + "\n".join(bad)


def test_a_command_prints_sigma_to_a_non_utf8_pipe():
    """PYTHONIOENCODING=cp1252 gives the child the stdout a Windows pipe has; the
    command's start-up switch must still let sigma through, as UTF-8."""
    code = ("from pyroorb.console import utf8_stdio; utf8_stdio(); "
            "print('RMS 0.4\\u03c3'); import sys; print('\\u2192', file=sys.stderr)")
    env = dict(os.environ, PYTHONIOENCODING="cp1252",
               PYTHONPATH=os.pathsep.join([str(SRC.parent), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert r.stdout.decode("utf-8").strip() == "RMS 0.4σ"
    assert r.stderr.decode("utf-8").strip() == "→"
