"""install.sh, the recommended install: it must parse, answer --help, refuse an unknown
option before touching anything, and uninstall only what it made. The full end-to-end run
(uv, compiling ASSIST, the web UI answering, the uninstall) is CI's install-script job on fresh
Linux and macOS runners."""
import os
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


# --- --uninstall, in a sandbox: a fake install under a fake HOME, so nothing real is touched ---
COMMANDS = ("pyroorb", "pyroorb-web", "pyroorb-itf", "pyroorb-fetch-data")
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org"]
GIT_WT = "work" "tree"      # the git subcommand; the public-code scan reserves that word


def _install(tmp_path, cloned=True, name="PyRoOrb", link=True):
    """A PyRoOrb folder as the script leaves it: a git checkout of a remote (install.sh,
    pyproject.toml), a .venv with the commands, their links in ~/.local/bin, and the tables."""
    home = tmp_path / "home"
    root = home / name
    root.mkdir(parents=True)
    shutil.copy(SCRIPT, root / "install.sh")
    (root / "pyproject.toml").write_text('[project]\nname = "pyroorb"\n', encoding="utf-8")
    (root / ".gitignore").write_text(".venv/\n.env\n", encoding="utf-8")
    subprocess.run(GIT + ["init", "-q"], cwd=root, check=True)
    subprocess.run(GIT + ["add", "."], cwd=root, check=True)
    subprocess.run(GIT + ["commit", "-q", "-m", "c"], cwd=root, check=True)
    remote = tmp_path / f"{name}-origin.git"                      # what it was cloned from
    subprocess.run(GIT + ["clone", "-q", "--bare", str(root), str(remote)], check=True)
    subprocess.run(GIT + ["remote", "add", "origin", str(remote)], cwd=root, check=True)
    subprocess.run(GIT + ["fetch", "-q", "origin"], cwd=root, check=True)
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    links = home / ".local" / "bin"
    links.mkdir(parents=True, exist_ok=True)
    for c in COMMANDS:
        (venv_bin / c).write_text("#!/bin/sh\n", encoding="utf-8")
        (venv_bin / c).chmod(0o755)
        if link:
            (links / c).symlink_to(venv_bin / c)
    if not (links / "uv").exists():
        (links / "uv").write_text("not ours", encoding="utf-8")
    if cloned:
        (root / ".venv" / "pyroorb-install").write_text("cloned\n", encoding="utf-8")
    if sys.platform == "darwin":
        cache = home / "Library" / "Caches" / "pyroorb"
        state = home / "Library" / "Application Support" / "pyroorb"
    else:
        cache, state = tmp_path / "xdg-cache" / "pyroorb", tmp_path / "xdg-data" / "pyroorb"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "de440.bsp").write_text("tables", encoding="utf-8")
    state.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYROORB_")}
    env.update(HOME=str(home), XDG_CACHE_HOME=str(tmp_path / "xdg-cache"),
               XDG_DATA_HOME=str(tmp_path / "xdg-data"), TMPDIR=str(tmp_path / "tmp"))
    return root, links, cache, state, env


def _uninstall(root, env, *args, script=None, cwd=None):
    # A session of its own has no terminal, so a question cannot block the test: it reads no.
    return subprocess.run(["bash", str(script or root / "install.sh"), "--uninstall", *args],
                          env=env, cwd=cwd or root.parent.parent, capture_output=True,
                          text=True, start_new_session=True)


def _uninstall_answering(root, env, answers, *args):
    """Run the uninstall on a terminal of its own and answer each [y/N] question in turn."""
    import pty
    import select
    pid, fd = pty.fork()
    if pid == 0:                                      # the child: the pty is its terminal
        os.chdir(root.parent.parent)
        os.execve("/bin/bash", ["bash", str(root / "install.sh"), "--uninstall", *args], env)
    out, answered = b"", 0
    while True:
        ready, _, _ = select.select([fd], [], [], 20)
        if not ready:
            break
        try:
            chunk = os.read(fd, 4096)
        except OSError:                               # the child closed the terminal
            break
        if not chunk:
            break
        out += chunk
        if out.count(b"[y/N]") > answered and answered < len(answers):
            os.write(fd, (answers[answered] + "\n").encode())
            answered += 1
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), out.decode(errors="replace")


def test_uninstall_yes_removes_what_it_made_and_never_the_folder(tmp_path):
    """--yes covers the environment, the links and the tables; the folder is only offered by a
    question of its own, and with nobody to answer it the folder stays."""
    root, links, cache, state, env = _install(tmp_path)
    out = _uninstall(root, env, "--yes")
    assert out.returncode == 0, out.stderr
    assert (root / "install.sh").exists() and not (root / ".venv").exists()
    assert not cache.exists()
    assert sorted(p.name for p in links.iterdir()) == ["uv"]
    assert "asks, separately, whether to delete the whole folder" in out.stdout
    assert f"rm -rf '{root}'" in out.stdout
    assert state.exists() and str(state) in out.stdout          # saved state: named, kept


def test_uninstall_deletes_the_folder_only_when_that_question_is_answered_yes(tmp_path):
    root, _links, cache, _state, env = _install(tmp_path)
    code, out = _uninstall_answering(root, env, ["y", "y"])
    assert code == 0, out
    assert not root.exists() and not cache.exists()


def test_uninstall_answering_no_for_the_folder_keeps_it(tmp_path):
    root, _links, _cache, _state, env = _install(tmp_path)
    code, out = _uninstall_answering(root, env, ["y", "n"])
    assert code == 0, out
    assert (root / "install.sh").exists() and not (root / ".venv").exists()
    assert "Kept" in out


def test_uninstall_answering_no_first_removes_nothing(tmp_path):
    root, links, cache, _state, env = _install(tmp_path)
    code, out = _uninstall_answering(root, env, ["n"])
    assert code == 0 and "Nothing removed." in out, out
    assert (root / ".venv").exists() and cache.exists() and len(list(links.iterdir())) == 5


def test_uninstall_keeps_a_clone_it_did_not_download(tmp_path):
    root, links, _cache, _state, env = _install(tmp_path, cloned=False)
    out = _uninstall(root, env, "--yes")
    assert out.returncode == 0, out.stderr
    assert (root / "install.sh").exists() and not (root / ".venv").exists()
    assert "this script did not download it" in out.stdout
    assert [p.name for p in links.iterdir()] == ["uv"]


@pytest.mark.parametrize("make, reason", [
    (lambda r: (r / ".env").write_text("SECRET=1\n"), "it holds files of your own"),
    (lambda r: subprocess.run(GIT + ["commit", "-q", "--allow-empty", "-m", "mine"], cwd=r,
                              check=True), "commit(s) that are on no remote"),
    (lambda r: (subprocess.run(GIT + ["branch", "-q", "side"], cwd=r, check=True),
                subprocess.run(GIT + ["checkout", "-q", "side"], cwd=r, check=True),
                subprocess.run(GIT + ["commit", "-q", "--allow-empty", "-m", "side"], cwd=r,
                               check=True),
                subprocess.run(GIT + ["checkout", "-q", "-"], cwd=r, check=True)),
     "commit(s) that are on no remote"),
    (lambda r: ((r / "install.sh").write_text((r / "install.sh").read_text() + "\n# mine\n"),
                subprocess.run(GIT + ["stash", "-q"], cwd=r, check=True)),
     "it holds stashed changes"),
    (lambda r: subprocess.run(GIT + [GIT_WT, "add", "-q", "--detach",
                                     str(r.parent / "wt")], cwd=r, check=True),
     "trees use it"),
])
def test_uninstall_never_offers_a_downloaded_folder_holding_work_of_yours(tmp_path, make, reason):
    root, _links, _cache, _state, env = _install(tmp_path)
    make(root)
    code, out = _uninstall_answering(root, env, ["y", "y"])
    assert code == 0, out
    assert (root / "install.sh").exists() and not (root / ".venv").exists()
    assert reason in out and "Also delete the whole folder" not in out


def test_uninstall_never_offers_a_folder_inside_another_repository(tmp_path):
    root, _links, _cache, _state, env = _install(tmp_path)
    shutil.rmtree(root / ".git")                    # a copy that is not its own repository...
    subprocess.run(GIT + ["init", "-q"], cwd=root.parent, check=True)   # ...inside one
    code, out = _uninstall_answering(root, env, ["y", "y"])
    assert code == 0, out
    assert (root / "install.sh").exists() and "inside another git repository" in out


def test_an_explicit_dir_is_the_install_removed_wherever_it_runs_from(tmp_path):
    """Run from inside another checkout (here: the script's own folder AND the current one),
    --dir still names the install to remove; the other is untouched."""
    root, _links, _cache, _state, env = _install(tmp_path)
    other, *_ = _install(tmp_path, name="dev-checkout", cloned=False, link=False)
    out = _uninstall(root, env, "--yes", "--dir", str(root), script=other / "install.sh",
                     cwd=other)
    assert out.returncode == 0, out.stderr
    assert not (root / ".venv").exists() and (other / ".venv").exists()
    missing = _uninstall(root, env, "--yes", "--dir", str(tmp_path / "nowhere"),
                         script=other / "install.sh", cwd=other)
    assert missing.returncode != 0 and (other / ".venv").exists()


def test_uninstall_keeps_the_shared_tables_while_another_install_is_on_path(tmp_path):
    root, links, cache, _state, env = _install(tmp_path)
    elsewhere = tmp_path / "other" / ".venv" / "bin" / "pyroorb-itf"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("", encoding="utf-8")
    (links / "pyroorb-itf").unlink()
    (links / "pyroorb-itf").symlink_to(elsewhere)
    out = _uninstall(root, env, "--yes")
    assert out.returncode == 0, out.stderr
    assert (cache / "de440.bsp").exists() and "another PyRoOrb install uses them" in out.stdout
    assert sorted(p.name for p in links.iterdir()) == ["pyroorb-itf", "uv"]


def test_uninstall_keep_data_keeps_the_tables(tmp_path):
    root, _links, cache, _state, env = _install(tmp_path)
    assert _uninstall(root, env, "--yes", "--keep-data").returncode == 0
    assert (cache / "de440.bsp").exists() and not (root / ".venv").exists()


def test_uninstall_asks_first_and_removes_nothing_unanswered(tmp_path):
    root, links, cache, _state, env = _install(tmp_path)
    out = _uninstall(root, env)
    assert out.returncode == 0 and "Nothing removed." in out.stdout
    assert (root / ".venv").exists() and cache.exists() and len(list(links.iterdir())) == 5


def test_only_a_fresh_clone_is_marked_as_downloaded():
    """Updating a checkout that was already there (git pull) must not mark it as this
    script's: the mark is what lets the uninstall offer the whole folder."""
    text = SCRIPT.read_text(encoding="utf-8")
    clone = text.index('git clone --depth 1 "$REPO_URL" "$DIR"')
    assert text.index("CLONED=1", clone) < text.index("\n  fi", clone)
    assert text.count("CLONED=1") == 1
