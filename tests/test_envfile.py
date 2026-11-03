"""The `.env` settings file: one file for every setting, loaded by each command before the engine
imports read their settings (the ephemeris and debias folders are read once, at import)."""
import ast
import os
import pathlib
import subprocess
import sys

import pytest

from pyroorb import envfile

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pyroorb"


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """No checkout .env, no explicit file, a fresh load state."""
    monkeypatch.setattr(envfile, "_CHECKOUT", tmp_path / "no-checkout")
    monkeypatch.delenv(envfile.ENV_FILE_VAR, raising=False)
    monkeypatch.setattr(envfile, "_LOADED", {"path": None, "done": False})
    return tmp_path


def test_parse_skips_comments_and_unquotes():
    text = '# c\n\nA=1\nexport B="two words"\nC=\'x\'\nnot a line\n D = spaced \n'
    assert envfile.parse(text) == [("A", "1"), ("B", "two words"), ("C", "x"), ("D", "spaced")]


def test_a_real_variable_wins_over_the_file(isolated, monkeypatch):
    (isolated / ".env").write_text("PYROORB_T_ONE=file\nPYROORB_T_TWO=file\n", encoding="utf-8")
    monkeypatch.setenv("PYROORB_T_ONE", "real")
    monkeypatch.delenv("PYROORB_T_TWO", raising=False)
    assert envfile.load_env_file(cwd=isolated) == isolated / ".env"
    assert os.environ["PYROORB_T_ONE"] == "real"
    assert os.environ.pop("PYROORB_T_TWO") == "file"


def test_found_in_a_folder_above_the_one_the_command_runs_from(isolated, monkeypatch):
    (isolated / ".env").write_text("PYROORB_T_UP=yes\n", encoding="utf-8")
    deep = isolated / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.delenv("PYROORB_T_UP", raising=False)
    assert envfile.load_env_file(cwd=deep) == isolated / ".env"
    assert os.environ.pop("PYROORB_T_UP") == "yes"


def test_an_explicit_file_comes_first(isolated, monkeypatch):
    (isolated / ".env").write_text("PYROORB_T_WHICH=nearby\n", encoding="utf-8")
    other = isolated / "elsewhere.env"
    other.write_text("PYROORB_T_WHICH=explicit\n", encoding="utf-8")
    monkeypatch.setenv(envfile.ENV_FILE_VAR, str(other))
    monkeypatch.delenv("PYROORB_T_WHICH", raising=False)
    assert envfile.load_env_file(cwd=isolated) == other
    assert os.environ.pop("PYROORB_T_WHICH") == "explicit"


def test_a_command_reads_the_data_folder_from_the_file(tmp_path):
    """End to end: a folder set only in the .env reaches the engine's import-time setting."""
    ephem = tmp_path / "ephem"
    ephem.mkdir()
    (ephem / "linux_p1550p2650.440").write_bytes(b"")   # present -> nothing is downloaded
    env_file = tmp_path / "settings.env"
    env_file.write_text(f"PYROORB_EPHEM_DIR={ephem}\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYROORB_EPHEM_DIR", envfile.ENV_FILE_VAR)}
    env[envfile.ENV_FILE_VAR] = str(env_file)
    env["PYTHONPATH"] = str(SRC.parent) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-m", "pyroorb.fetch_data", "--no-debias"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert f"ephem dir: {ephem}" in out.stdout, out.stdout + out.stderr


def _first_engine_import(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("pyroorb")):
            return node.lineno
        if isinstance(node, ast.Import) and any(a.name.startswith("pyroorb") for a in node.names):
            return node.lineno
    return None


def _loads_env(node):
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_env_file"
               for n in ast.walk(node))


@pytest.mark.parametrize("rel", ["web/server.py", "fetch_data.py", "itf/__main__.py",
                                 "fit_object.py"])
def test_every_runnable_module_loads_the_file_before_the_engine(rel):
    tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
    guard = next((n for n in tree.body if isinstance(n, ast.If) and _loads_env(n)), None)
    assert guard is not None, f"{rel} never loads the .env settings file when run"
    first = _first_engine_import(tree)
    assert first is None or guard.lineno < first, f"{rel}: the .env is loaded after line {first}"


@pytest.mark.parametrize("name", ["_cli", "_web", "_itf", "_fetch_data"])
def test_every_console_command_loads_the_file_first(name):
    tree = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    calls = [i for i, n in enumerate(body) if _loads_env(n)]
    imports = [i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom))
               and "envfile" not in ast.unparse(n)]
    assert calls and calls[0] < imports[0], f"{name} imports the engine before loading .env"
