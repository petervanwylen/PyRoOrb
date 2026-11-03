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


def _install(tmp_path, cloned=True):
    """A PyRoOrb folder as the script leaves it: a git checkout (install.sh, pyproject.toml), a
    .venv with the commands, their links in ~/.local/bin, and the downloaded tables."""
    home = tmp_path / "home"
    root = home / "PyRoOrb"
    root.mkdir(parents=True)
    shutil.copy(SCRIPT, root / "install.sh")
    (root / "pyproject.toml").write_text('[project]\nname = "pyroorb"\n', encoding="utf-8")
    (root / ".gitignore").write_text(".venv/\n.env\n", encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org"]
    subprocess.run(git + ["init", "-q"], cwd=root, check=True)
    subprocess.run(git + ["add", "."], cwd=root, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "c"], cwd=root, check=True)
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    links = home / ".local" / "bin"
    links.mkdir(parents=True)
    for c in COMMANDS:
        (venv_bin / c).write_text("#!/bin/sh\n", encoding="utf-8")
        (venv_bin / c).chmod(0o755)
        (links / c).symlink_to(venv_bin / c)
    (links / "uv").write_text("not ours", encoding="utf-8")
    if cloned:
        (root / ".venv" / "pyroorb-install").write_text("cloned\n", encoding="utf-8")
    if sys.platform == "darwin":
        cache = home / "Library" / "Caches" / "pyroorb"
        state = home / "Library" / "Application Support" / "pyroorb"
    else:
        cache, state = tmp_path / "xdg-cache" / "pyroorb", tmp_path / "xdg-data" / "pyroorb"
    cache.mkdir(parents=True)
    (cache / "de440.bsp").write_text("tables", encoding="utf-8")
    state.mkdir(parents=True)
    (tmp_path / "tmp").mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYROORB_")}
    env.update(HOME=str(home), XDG_CACHE_HOME=str(tmp_path / "xdg-cache"),
               XDG_DATA_HOME=str(tmp_path / "xdg-data"), TMPDIR=str(tmp_path / "tmp"))
    return root, links, cache, state, env


def _uninstall(root, env, *args):
    # A session of its own has no terminal, so a question cannot block the test.
    return subprocess.run(["bash", str(root / "install.sh"), "--uninstall", *args], env=env,
                          cwd=root.parent.parent, capture_output=True, text=True,
                          start_new_session=True)


def test_uninstall_removes_what_a_downloaded_install_made(tmp_path):
    root, links, cache, state, env = _install(tmp_path)
    other = tmp_path / "other" / ".venv" / "bin" / "pyroorb-itf"   # another install's link
    other.parent.mkdir(parents=True)
    other.write_text("", encoding="utf-8")
    (links / "pyroorb-itf").unlink()
    (links / "pyroorb-itf").symlink_to(other)
    out = _uninstall(root, env, "--yes")
    assert out.returncode == 0, out.stderr
    assert not root.exists() and not cache.exists()
    assert sorted(p.name for p in links.iterdir()) == ["pyroorb-itf", "uv"]
    assert state.exists() and str(state) in out.stdout          # saved state: named, kept


def test_uninstall_keeps_a_clone_it_did_not_download(tmp_path):
    root, links, _cache, _state, env = _install(tmp_path, cloned=False)
    out = _uninstall(root, env, "--yes")
    assert out.returncode == 0, out.stderr
    assert (root / "install.sh").exists() and not (root / ".venv").exists()
    assert "this script did not download it" in out.stdout
    assert [p.name for p in links.iterdir()] == ["uv"]


def test_uninstall_keeps_a_downloaded_folder_holding_files_of_yours(tmp_path):
    root, _links, _cache, _state, env = _install(tmp_path)
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")      # ignored by git, still yours
    out = _uninstall(root, env, "--yes")
    assert out.returncode == 0, out.stderr
    assert (root / ".env").exists() and not (root / ".venv").exists()
    assert ".env" in out.stdout


def test_uninstall_keep_data_keeps_the_tables(tmp_path):
    root, _links, cache, _state, env = _install(tmp_path)
    assert _uninstall(root, env, "--yes", "--keep-data").returncode == 0
    assert (cache / "de440.bsp").exists() and not root.exists()


def test_uninstall_asks_first_and_removes_nothing_unanswered(tmp_path):
    root, links, cache, _state, env = _install(tmp_path)
    out = _uninstall(root, env)
    assert out.returncode == 0 and "Nothing removed." in out.stdout
    assert (root / ".venv").exists() and cache.exists() and len(list(links.iterdir())) == 5
