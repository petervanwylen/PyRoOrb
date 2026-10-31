"""The `pyroorb` command: help is help (never a designation sent to the MPC), and the child
processes the ident-check / batch drivers spawn run the package module, not a file path."""
import os
import subprocess
import sys

from pyroorb import cli


def test_help_and_no_arguments_print_the_usage(capsys, monkeypatch):
    for argv in ([], ["--help"], ["-h"], ["2023 QJ3", "--help"]):
        monkeypatch.setattr(sys, "argv", ["pyroorb", *argv])
        monkeypatch.setattr(cli, "fetch_observations",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
        cli.main()
        out = capsys.readouterr().out
        assert out.startswith("usage: pyroorb DESIG"), argv


def test_children_run_the_package_module_with_src_on_the_path():
    cmd = cli._child_cmd("A", "B", "--ident-check", "out.txt")
    assert cmd[:3] == [sys.executable, "-m", "pyroorb.fit_object"] and cmd[3:5] == ["A", "B"]
    env = cli._child_env()
    src = os.path.dirname(os.path.dirname(os.path.abspath(cli.__file__)))
    assert env["PYTHONPATH"].split(os.pathsep)[0] == src


def test_the_module_entry_prints_help_from_any_directory(tmp_path):
    r = subprocess.run(cli._child_cmd("--help"), cwd=tmp_path, capture_output=True, text=True,
                       env=cli._child_env(), timeout=300)
    assert r.returncode == 0 and r.stdout.startswith("usage: pyroorb"), r.stderr[-400:]
