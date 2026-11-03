"""install.ps1, the Windows install: it must parse, and the ASSIST it installs must be the one CI
and the install guide name. The ASSIST release on PyPI does not build on Windows yet, so all
three pin the same commit of the fork with native Windows support; a fork bump that misses one
would test one ASSIST and ship another. The end-to-end run is CI's install-script job on a
fresh Windows runner."""
import os
import pathlib
import re
import shutil
import subprocess
import sys

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


def _scripts_table():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return sorted(re.findall(r'^([\w-]+) = "', block, re.M))


def test_both_installers_put_every_pyroorb_command_on_path():
    ps1 = re.search(r'\$Commands = @\(([^)]*)\)', SCRIPT.read_text(encoding="utf-8")).group(1)
    sh = re.search(r'^COMMANDS="([^"]*)"', (REPO / "install.sh").read_text(encoding="utf-8"), re.M).group(1)
    assert sorted(re.findall(r'"([\w-]+)"', ps1)) == _scripts_table()
    assert sorted(sh.split()) == _scripts_table()


# --- -Uninstall, in a sandbox: a fake install under a fake profile, so nothing real is touched.
# No "path-added" mark, so the user PATH in the registry is never written.
COMMANDS = ("pyroorb", "pyroorb-web", "pyroorb-itf", "pyroorb-fetch-data")
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="install.ps1 -Uninstall runs on Windows")


def _install(tmp_path, cloned=True):
    """A PyRoOrb folder as install.ps1 leaves it: a git checkout, a .venv, the commands' copies in
    ~\.local\bin (each naming its .venv's python.exe, as uv's launchers do) and the tables."""
    home = tmp_path / "home"
    root = home / "PyRoOrb"
    root.mkdir(parents=True)
    shutil.copy(SCRIPT, root / "install.ps1")
    (root / "pyproject.toml").write_text('[project]\nname = "pyroorb"\n', encoding="utf-8")
    (root / ".gitignore").write_text(".venv/\n.env\n", encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "core.autocrlf=false"]
    subprocess.run(git + ["init", "-q"], cwd=root, check=True)
    subprocess.run(git + ["add", "."], cwd=root, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "c"], cwd=root, check=True)
    scripts = root / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    links = home / ".local" / "bin"
    links.mkdir(parents=True)
    for c in COMMANDS:
        launcher = b"MZ launcher " + str(scripts / "python.exe").encode() + b" PK"
        (scripts / f"{c}.exe").write_bytes(launcher)
        (links / f"{c}.exe").write_bytes(launcher)
    (links / "uv.exe").write_bytes(b"MZ not ours")
    if cloned:
        (root / ".venv" / "pyroorb-install").write_text("cloned\n", encoding="ascii")
    local, roaming, temp = tmp_path / "Local", tmp_path / "Roaming", tmp_path / "Temp"
    cache, state = local / "pyroorb" / "Cache", roaming / "pyroorb"
    cache.mkdir(parents=True)
    (cache / "de440.bsp").write_text("tables", encoding="utf-8")
    state.mkdir(parents=True)
    temp.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYROORB_")}
    env.update(USERPROFILE=str(home), LOCALAPPDATA=str(local), APPDATA=str(roaming),
               TEMP=str(temp), TMP=str(temp))
    return root, links, cache, state, env


def _uninstall(root, env, *args, cwd=None):
    shell = shutil.which("powershell") or shutil.which("pwsh")
    return subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-File", str(root / "install.ps1"), "-Uninstall", *args],
                          env=env, cwd=cwd or root.parent.parent, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)


@windows_only
def test_uninstall_removes_what_a_downloaded_install_made(tmp_path):
    root, links, cache, state, env = _install(tmp_path)
    (links / "pyroorb-itf.exe").write_bytes(b"MZ C:\other\.venv\Scripts\python.exe")
    out = _uninstall(root, env, "-Yes")
    assert out.returncode == 0, out.stdout + out.stderr
    assert not root.exists() and not cache.parent.exists()
    assert sorted(p.name for p in links.iterdir()) == ["pyroorb-itf.exe", "uv.exe"]
    assert state.exists() and str(state) in out.stdout          # saved state: named, kept


@windows_only
def test_uninstall_keeps_a_clone_it_did_not_download(tmp_path):
    root, links, _cache, _state, env = _install(tmp_path, cloned=False)
    out = _uninstall(root, env, "-Yes")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "install.ps1").exists() and not (root / ".venv").exists()
    assert "this script did not download it" in out.stdout
    assert [p.name for p in links.iterdir()] == ["uv.exe"]


@windows_only
def test_uninstall_keeps_a_downloaded_folder_holding_files_of_yours(tmp_path):
    root, _links, _cache, _state, env = _install(tmp_path)
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")      # ignored by git, still yours
    out = _uninstall(root, env, "-Yes")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / ".env").exists() and not (root / ".venv").exists()
    assert ".env" in out.stdout


@windows_only
def test_uninstall_keep_data_keeps_the_tables(tmp_path):
    root, _links, cache, _state, env = _install(tmp_path)
    assert _uninstall(root, env, "-Yes", "-KeepData").returncode == 0
    assert (cache / "de440.bsp").exists() and not root.exists()


@windows_only
def test_uninstall_asks_first_and_removes_nothing_unanswered(tmp_path):
    root, links, cache, _state, env = _install(tmp_path)
    out = _uninstall(root, env)
    assert out.returncode == 0 and "Nothing removed." in out.stdout, out.stdout + out.stderr
    assert (root / ".venv").exists() and cache.exists() and len(list(links.iterdir())) == 5


@windows_only
def test_uninstall_started_inside_the_folder_still_removes_it(tmp_path):
    """The update route leaves people in the folder. Windows will not delete a folder a process
    is in, and Set-Location alone does not move the script's own process out of it."""
    root, _links, _cache, _state, env = _install(tmp_path)
    out = _uninstall(root, env, "-Yes", cwd=root)
    assert out.returncode == 0, out.stdout + out.stderr
    assert not root.exists()


@windows_only
def test_uninstall_run_in_your_own_session_leaves_your_location_alone(tmp_path):
    """Run as ./install.ps1, the script shares the user's PowerShell location: removing only
    .venv must not move the user out of their clone (the old code went to $HOME)."""
    root, _links, _cache, _state, env = _install(tmp_path, cloned=False)
    shell = shutil.which("powershell") or shutil.which("pwsh")
    cmd = f"Set-Location '{root}'; & '{root / 'install.ps1'}' -Uninstall -Yes | Out-Null; (Get-Location).Path"
    out = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                          "-Command", cmd], env=env, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL)
    assert out.stdout.strip().splitlines()[-1] == str(root), out.stdout + out.stderr
    assert not (root / ".venv").exists()
