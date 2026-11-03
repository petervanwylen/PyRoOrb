"""install.ps1, the Windows install: it must parse, and the ASSIST it installs must be the one CI
and the install guide name. The ASSIST release on PyPI does not build on Windows yet, so all
three pin the same commit of the fork with native Windows support; a fork bump that misses one
would test one ASSIST and ship another. The end-to-end run is CI's install-script job on a
fresh Windows runner."""
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "install.ps1"
_PIN = re.compile(r"assist @ git\+https://github\.com/[\w.-]+/[\w.-]+@[0-9a-f]{40}")


def _pins(path):
    return set(_PIN.findall(path.read_text(encoding="utf-8")))


def test_install_ps1_ci_and_the_guide_pin_the_same_assist():
    script = _pins(SCRIPT)
    assert len(script) == 1, f"install.ps1 should pin one ASSIST commit, found {script}"
    for other in (REPO / ".github" / "workflows" / "ci.yml", REPO / "docs" / "INSTALL.md"):
        assert _pins(other) == script, f"{other.name} pins {_pins(other)}, install.ps1 {script}"


def test_install_ps1_builds_assist_against_the_rebound_it_runs_with():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--build-constraint" in text and "rebound>=4.4.11,<5" in text


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")),
                    reason="needs PowerShell")
def test_install_ps1_parses():
    shell = shutil.which("pwsh") or shutil.which("powershell")
    check = ("$e = $null; [System.Management.Automation.Language.Parser]::ParseFile("
             f"'{SCRIPT}', [ref]$null, [ref]$e) | Out-Null; exit $e.Count")
    assert subprocess.run([shell, "-NoProfile", "-Command", check]).returncode == 0
