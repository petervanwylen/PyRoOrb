"""install.sh, the recommended install: it must parse, answer --help, and refuse an unknown
option before touching anything. The full end-to-end run (uv, compiling ASSIST, the web UI
answering) is CI's install-script job on fresh Linux and macOS runners."""
import pathlib
import shutil
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "install.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_help_describes_it_and_exits_cleanly():
    out = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "macOS and Linux" in out.stdout and "--no-data" in out.stdout


def test_an_unknown_option_is_refused_first():
    out = subprocess.run(["bash", str(SCRIPT), "--frobnicate"], capture_output=True, text=True)
    assert out.returncode != 0 and "unknown option" in out.stderr


def test_it_pins_rebound_4_for_the_assist_build():
    text = SCRIPT.read_text()
    assert "--build-constraint" in text and "rebound>=4.4.11,<5" in text
