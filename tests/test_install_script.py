"""install.sh, the recommended install: it must parse, answer --help, and refuse an unknown
option before touching anything. The full end-to-end run (uv, compiling ASSIST, the web UI
answering) is CI's install-script job on fresh Linux and macOS runners."""
import pathlib
import shutil
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "install.sh"
# Not on Windows: install.ps1 is the installer there (tests/test_install_ps1.py), and the "bash"
# a Windows runner finds first is WSL's launcher, which with no distribution installed runs nothing.
pytestmark = pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32",
                                reason="needs bash, on macOS or Linux")


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_help_describes_it_and_exits_cleanly():
    out = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "macOS and Linux" in out.stdout and "--no-data" in out.stdout


def test_an_unknown_option_is_refused_first():
    out = subprocess.run(["bash", str(SCRIPT), "--frobnicate"], capture_output=True, text=True)
    assert out.returncode != 0 and "unknown option" in out.stderr


def test_the_default_install_has_no_extras():
    """The ITF extra (xgboost, and Homebrew's libomp on macOS) needs an SBN database almost
    nobody has, so it is opt-in: --extras itf."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert '\nEXTRAS=""\n' in text
    assert "brew install libomp" in text and ',$EXTRAS," == *",itf,"*' in text


def test_it_pins_rebound_4_for_the_assist_build():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--build-constraint" in text and "rebound>=4.4.11,<5" in text


def test_assist_is_built_against_the_rebound_it_runs_with():
    """ASSIST compiles against REBOUND's headers and runs against REBOUND's library: the
    script installs REBOUND first, then builds ASSIST (and resolves the rest) against exactly
    that version, and never reuses an ASSIST wheel cached from another REBOUND."""
    text = SCRIPT.read_text(encoding="utf-8")
    first = text.index("uv pip install --python .venv/bin/python 'rebound>=4.4.11,<5'")
    main = text.index('-e "$TARGET"')
    assert first < main
    assert 'echo "rebound==$REBOUND_VERSION" > "$CONSTRAINTS"' in text
    block = text[text.index("uv pip install --python .venv/bin/python --constraint"):main]
    assert '--constraint "$CONSTRAINTS"' in block and '--build-constraint "$CONSTRAINTS"' in block
    # uv reuses a cached ASSIST wheel whatever REBOUND it was built against (measured: 0.35 ms
    # "Prepared", no build, even with --refresh-package); the cache entry must go first
    assert text.index("uv cache clean assist") < text.index(
        "uv pip install --python .venv/bin/python --constraint")
