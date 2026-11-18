"""``--ident-check`` classifies by the same verdict the web UI shows.

The batch command (``cli._ident_check_one``) used to keep its own copy of the identification
gates. The copy never gained the ITF submission floor nor the pre-existing-fault rule, so a
thin non-NEO link was written as a confirmed 'y' that the web refused, and a trend already in
one body's own record made a 'y-systematic' that the web called clean. Now the code, the
reason and the advisory suffixes all come from ``ident_check.linkage_verdict``; the fetch, the
fit and the verdict are stubbed here, so no network or engine fit runs.

    pytest tests/test_ident_check_cli_verdict.py
"""
import types

import pytest

from pyroorb import cli, ident_check

RECORD = "2001 AB1 = 2020 CD2\n"


def _fit():
    return types.SimpleNamespace(n_observations=lambda: 20, n_total_observations=lambda: 22,
                                 n_excluded=lambda: 2, rms_residual=lambda: 0.31)


@pytest.fixture
def check(monkeypatch, tmp_path):
    """Run ``_ident_check_one`` on a stubbed pair under the verdict ``v`` (and ``record``, what
    identify() renders): returns (code, stderr text, what was written to the file)."""
    monkeypatch.setattr(cli, "_fetch_ident_pair",
                        lambda desigs, oc: ([{"designation": "x"}], {"designation": "x"}))
    monkeypatch.setattr(cli, "fetch_failure_message", lambda meta: None)
    monkeypatch.setattr(cli, "fit_orbit", lambda *a, **k: _fit())
    out = tmp_path / "ids.txt"

    def run(v, capsys, record=RECORD):
        v = {"reason": "", "cautions": [], "notes": [], "flags": {}, **v}
        monkeypatch.setattr(ident_check, "linkage_verdict", lambda fit: v)
        monkeypatch.setattr(ident_check, "identify", lambda fit: record)
        code = cli._ident_check_one("2001 AB1", "2020 CD2", str(out), {})
        written = out.read_text(encoding="utf-8") if out.exists() else ""
        return code, capsys.readouterr().err, written
    return run


def test_a_link_short_of_the_itf_floor_is_examined_not_written(check, capsys):
    """The case the old copy got wrong: identify() renders a record, but the verdict holds the
    link to the ITF floor -- 'y-examine', nothing written."""
    code, err, written = check({"code": "y-examine",
                                "reason": "NOT a clean identification -- ... anchor"}, capsys)
    assert code == "y-examine" and written == ""
    assert "[ident-check-class] y-examine" in err and "EXAMINE" in err


def test_a_clean_match_is_written_with_the_verdicts_suffixes(check, capsys):
    v = {"code": "y", "flags": {"falsepos": "2020 CD2 collapses to a 3.1 h kept sub-arc",
                                "short": "9 obs", "mag": "?"},
         "cautions": ["Short link: 9 obs."], "notes": ["Set aside as already in the record."]}
    code, err, written = check(v, capsys)
    assert code == "y-falsepos?-short-mag?"
    assert written.startswith("# FLAG false-positive risk: 2020 CD2 collapses")
    assert RECORD in written
    assert "caution: Short link" in err and "note: Set aside" in err
    assert cli._is_written_match(code)


def test_a_systematic_link_is_not_written(check, capsys):
    code, _err, written = check({"code": "y-systematic", "flags": {"mag": "!"}}, capsys)
    assert code == "y-systematic-mag!" and written == ""
    assert not cli._is_written_match(code)


def test_a_no_carries_no_suffix(check, capsys):
    code, err, written = check({"code": "no", "reason": "retention 40%",
                                "flags": {"short": "5 obs", "mag": ""}}, capsys)
    assert code == "no" and written == ""
    assert "no clean match" in err and "retention 40%" in err


def test_a_match_with_no_record_to_write_is_unknown(check, capsys):
    code, err, written = check({"code": "y2"}, capsys, record=None)
    assert code == "unknown" and written == ""
    assert "not written [unknown]" in err
    assert not cli._is_written_match(code)


def test_the_verdict_names_its_advisory_flags():
    """linkage_verdict returns the flags the batch suffixes read; an early refusal has none."""
    v = ident_check.linkage_verdict(types.SimpleNamespace(n_requested=1))
    assert v["code"] == "no" and not v.get("flags")
