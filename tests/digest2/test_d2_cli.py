"""CLI-level tests, including a couple of exact-output regressions
against the real C ``digest2`` binary (see the port's validation notes
for how the expected strings were captured -- a standalone build of
Smithsonian/mpc-public's digest2/digest2 sources against this same
model/obscodes data, run with ``-u 1`` for deterministic output order).
"""
from pathlib import Path

import pytest

from pyroorb.digest2.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parents[2] / "pydigest2" / "data"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PYDIGEST2_MODEL", str(DATA / "digest2.model.csv"))
    monkeypatch.setenv("PYDIGEST2_OBSCODES", str(DATA / "digest2.obscodes"))


def test_repeatable_config_sample_obs_matches_c_reference(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(FIXTURES)
    rc = main(["-c", "repeatable.config", "sample.obs"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["Desig.    RMS NEO MB1", "K16S99K  0.73   3  90"]


def test_multi_tracklet_streaming_order_matches_file_order(tmp_path, capsys, monkeypatch):
    # S1795's NEO score (row 8) is the same already-diagnosed +/-1 point
    # sub-ULP-cascade divergence covered by
    # test_integration.py::test_class_restricted_scoring_matches_c_reference
    # (this fixture is a real, very short/sparse arc); every other line
    # matches the C reference verbatim, including output *order* -- which
    # depends on this port's tracklet-grouping matching the reference
    # program's streaming (not whole-file) grouping exactly.
    monkeypatch.chdir(FIXTURES)
    rc = main(["-c", "repeatable.config", "-u", "1", "three-hr-tracklets.obs"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    expected = [
        "Desig.    RMS NEO MB1",
        "  23662  0.10  39  55",
        "  65558  0.33  24  69",
        "  99516  0.09  14  87",
        "  J6666  0.06  39  58",
        "  L7488  0.12  22  17",
        "  M9358  0.12  19  79",
        "  A0421  0.09  13  72",
        None,  # S1795 -- checked separately below
        "  i9130  0.20   7  92",
        None,  # K21V32W -- same story, checked separately below
    ]
    for got, exp in zip(out, expected):
        if exp is not None:
            assert got == exp
    assert out[8] in ("  S1795  0.16  12   2", "  S1795  0.16  13   2")
    assert out[10] in ("K21V32W  0.17   2  67", "K21V32W  0.17   2  68")


def test_numbered_designation_formatting(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(FIXTURES)
    rc = main(["-c", "repeatable.config", "sample_old.obs"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    # numbered objects show as "  NNNNN" (2 spaces + the 5-char packed number)
    assert out[1].startswith("NE00030")
    assert out[2].startswith("NE00199")
    assert out[3].startswith("NE00269")


def test_stdin_input(capsys, monkeypatch):
    import sys
    from io import StringIO
    monkeypatch.chdir(FIXTURES)
    text = (FIXTURES / "sample.obs").read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", StringIO(text))
    rc = main(["-c", "repeatable.config", "-"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["Desig.    RMS NEO MB1", "K16S99K  0.73   3  90"]


def test_no_args_prints_usage_and_errors(capsys):
    rc = main([])
    assert rc == 1


def test_version_flag(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pydigest2 version" in out


def test_bad_limit_spec_errors_cleanly(capsys, monkeypatch):
    monkeypatch.chdir(FIXTURES)
    rc = main(["-l", "garbage", "sample.obs"])
    assert rc == 1


def test_invalid_line_produces_skip_message(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(FIXTURES)
    bad_file = tmp_path / "bad.obs"
    bad_file.write_text("not a valid mpc80 line\nalso not valid\n", encoding="utf-8")
    rc = main(["-c", str(FIXTURES / "repeatable.config"), str(bad_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 lines skipped for ." in out
