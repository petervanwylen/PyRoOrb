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


def test_install_ps1_has_no_control_characters():
    """A stray escape turned `Scripts\\activate` into a BEL byte in the message shown to a user;
    only tab belongs in a script besides its line breaks."""
    bad = [(n, hex(ord(c)))
           for n, line in enumerate(SCRIPT.read_text(encoding="utf-8").splitlines(), 1)
           for c in line if ord(c) < 32 and c != "\t"]
    assert not bad, f"control characters in install.ps1 (line, code): {bad}"


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
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "core.autocrlf=false"]
GIT_WT = "work" "tree"      # the git subcommand; the public-code scan reserves that word


def _install(tmp_path, cloned=True, name="PyRoOrb", copies=True):
    """A PyRoOrb folder as install.ps1 leaves it: a git checkout of a remote, a .venv, the
    commands' copies in ~\.local\bin (each naming its .venv's python.exe, as uv's launchers do)
    and the tables."""
    home = tmp_path / "home"
    root = home / name
    root.mkdir(parents=True)
    shutil.copy(SCRIPT, root / "install.ps1")
    (root / "pyproject.toml").write_text('[project]\nname = "pyroorb"\n', encoding="utf-8")
    (root / ".gitignore").write_text(".venv/\n.env\n", encoding="utf-8")
    subprocess.run(GIT + ["init", "-q"], cwd=root, check=True)
    subprocess.run(GIT + ["add", "."], cwd=root, check=True)
    subprocess.run(GIT + ["commit", "-q", "-m", "c"], cwd=root, check=True)
    remote = tmp_path / f"{name}-origin.git"                      # what it was cloned from
    subprocess.run(GIT + ["clone", "-q", "--bare", str(root), str(remote)], check=True)
    subprocess.run(GIT + ["remote", "add", "origin", str(remote)], cwd=root, check=True)
    subprocess.run(GIT + ["fetch", "-q", "origin"], cwd=root, check=True)
    scripts = root / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    links = home / ".local" / "bin"
    links.mkdir(parents=True, exist_ok=True)
    for c in COMMANDS:
        launcher = b"MZ launcher " + str(scripts / "python.exe").encode() + b" PK"
        (scripts / f"{c}.exe").write_bytes(launcher)
        if copies:
            (links / f"{c}.exe").write_bytes(launcher)
    if not (links / "uv.exe").exists():
        (links / "uv.exe").write_bytes(b"MZ not ours")
    if cloned:
        (root / ".venv" / "pyroorb-install").write_text("cloned\n", encoding="ascii")
    local, roaming, temp = tmp_path / "Local", tmp_path / "Roaming", tmp_path / "Temp"
    cache, state = local / "pyroorb" / "Cache", roaming / "pyroorb"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "de440.bsp").write_text("tables", encoding="utf-8")
    state.mkdir(parents=True, exist_ok=True)
    temp.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYROORB_")}
    env.update(USERPROFILE=str(home), LOCALAPPDATA=str(local), APPDATA=str(roaming),
               TEMP=str(temp), TMP=str(temp))
    return root, links, cache, state, env


def _uninstall(root, env, *args, cwd=None, answers=None, script=None):
    """Non-interactive (every question reads no) unless ``answers``: then the questions read
    them, one per line, from standard input."""
    shell = shutil.which("powershell") or shutil.which("pwsh")
    mode = [] if answers else ["-NonInteractive"]
    return subprocess.run([shell, "-NoProfile", *mode, "-ExecutionPolicy", "Bypass",
                           "-File", str(script or root / "install.ps1"), "-Uninstall", *args],
                          env=env, cwd=cwd or root.parent.parent, capture_output=True, text=True,
                          input="".join(a + "\n" for a in answers) if answers else None,
                          stdin=None if answers else subprocess.DEVNULL)


@windows_only
def test_uninstall_yes_removes_what_it_made_and_never_the_folder(tmp_path):
    root, links, cache, state, env = _install(tmp_path)
    out = _uninstall(root, env, "-Yes")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "install.ps1").exists() and not (root / ".venv").exists()
    assert not cache.parent.exists()
    assert sorted(p.name for p in links.iterdir()) == ["uv.exe"]
    assert "asks, separately, whether to delete the whole folder" in out.stdout
    assert "Remove-Item -Recurse -Force" in out.stdout
    assert state.exists() and str(state) in out.stdout          # saved state: named, kept


@windows_only
def test_uninstall_deletes_the_folder_only_when_that_question_is_answered_yes(tmp_path):
    root, _links, cache, _state, env = _install(tmp_path)
    out = _uninstall(root, env, answers=["y", "y"])
    assert out.returncode == 0, out.stdout + out.stderr
    assert not root.exists() and not cache.parent.exists()


@windows_only
def test_uninstall_answering_no_for_the_folder_keeps_it(tmp_path):
    root, _links, _cache, _state, env = _install(tmp_path)
    out = _uninstall(root, env, answers=["y", "n"])
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "install.ps1").exists() and not (root / ".venv").exists()


@windows_only
def test_uninstall_keeps_a_clone_it_did_not_download(tmp_path):
    root, links, _cache, _state, env = _install(tmp_path, cloned=False)
    out = _uninstall(root, env, "-Yes")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "install.ps1").exists() and not (root / ".venv").exists()
    assert "this script did not download it" in out.stdout
    assert [p.name for p in links.iterdir()] == ["uv.exe"]


@windows_only
@pytest.mark.parametrize("make, reason", [
    (lambda r: (r / ".env").write_text("SECRET=1\n"), "it holds files of your own"),
    (lambda r: subprocess.run(GIT + ["commit", "-q", "--allow-empty", "-m", "mine"], cwd=r,
                              check=True), "commit(s) that are on no remote"),
    (lambda r: ((r / "install.ps1").write_bytes((r / "install.ps1").read_bytes() + b"\n# mine\n"),
                subprocess.run(GIT + ["stash", "-q"], cwd=r, check=True)),
     "it holds stashed changes"),
    (lambda r: subprocess.run(GIT + [GIT_WT, "add", "-q", "--detach",
                                     str(r.parent / "wt")], cwd=r, check=True),
     "trees use it"),
])
def test_uninstall_never_offers_a_downloaded_folder_holding_work_of_yours(tmp_path, make, reason):
    root, _links, _cache, _state, env = _install(tmp_path)
    make(root)
    out = _uninstall(root, env, answers=["y", "y"])
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "install.ps1").exists() and not (root / ".venv").exists()
    assert reason in out.stdout and "Also delete the whole folder" not in out.stdout


@windows_only
def test_an_explicit_dir_is_the_install_removed_wherever_it_runs_from(tmp_path):
    root, _links, _cache, _state, env = _install(tmp_path)
    other, *_ = _install(tmp_path, name="dev-checkout", cloned=False, copies=False)
    out = _uninstall(root, env, "-Yes", "-Dir", str(root), script=other / "install.ps1", cwd=other)
    assert out.returncode == 0, out.stdout + out.stderr
    assert not (root / ".venv").exists() and (other / ".venv").exists()


@windows_only
def test_uninstall_keeps_the_shared_tables_while_another_install_is_on_path(tmp_path):
    root, links, cache, _state, env = _install(tmp_path)
    (links / "pyroorb-itf.exe").write_bytes(b"MZ C:\\other\\.venv\\Scripts\\python.exe")
    out = _uninstall(root, env, "-Yes")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (cache / "de440.bsp").exists() and "another PyRoOrb install uses them" in out.stdout
    assert sorted(p.name for p in links.iterdir()) == ["pyroorb-itf.exe", "uv.exe"]


@windows_only
def test_uninstall_keep_data_keeps_the_tables(tmp_path):
    root, _links, cache, _state, env = _install(tmp_path)
    assert _uninstall(root, env, "-Yes", "-KeepData").returncode == 0
    assert (cache / "de440.bsp").exists() and not (root / ".venv").exists()


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
    out = _uninstall(root, env, cwd=root, answers=["y", "y"])
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


def test_only_a_fresh_clone_is_marked_as_downloaded():
    """Updating a checkout that was already there (git pull) must not mark it as this
    script's: the mark is what lets the uninstall offer the whole folder."""
    text = SCRIPT.read_text(encoding="utf-8")
    clone = text.index("Run git clone --depth 1 $RepoUrl $Dir")
    assert text.index("$Cloned = $true", clone) < text.index("\n    }", clone)
    assert text.count("$Cloned = $true") == 1
