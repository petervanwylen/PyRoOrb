"""`fit_object.py DESIG --itf-search`: the hand-rolled option parser.

This surface has no argparse, so nothing refuses an option it does not know: an
unknown `--itf-*` flag used to be dropped silently and its VALUE read as a
designation to fit (`--itf-ranking certainty` fitted an object called
"certainty"), and a boolean one simply did nothing.  And `--itf-trace-level`
must mean what `python -m pyroorb.itf search --trace-level` means, 0 included.

No fit and no network: the search itself is replaced by a recorder.  Slow-marked
only for the cli import chain, like the other `cli.main` tests.
"""
import sys

import pytest

pytestmark = pytest.mark.slow


class _Searched(Exception):
    """Raised by the recorder so `main` stops where the search would start."""


def _run(monkeypatch, *argv, verbose_env=False):
    """Run `cli.main` on ``argv``; return the recorder's call, or None when the
    parser refused before reaching the search."""
    from pyroorb import cli
    import pyroorb.itf.search; from pyroorb import itf

    seen = {}

    def fake_search(desigs, **kw):
        seen.update(kw, desigs=list(desigs))
        raise _Searched

    monkeypatch.setattr(itf.search, "itf_search_live", fake_search)
    if verbose_env:
        monkeypatch.setenv("PYROORB_VERBOSE", "1")
    else:
        monkeypatch.delenv("PYROORB_VERBOSE", raising=False)
    monkeypatch.setattr(sys, "argv", ["fit_object.py", *argv])
    try:
        cli.main()
    except _Searched:
        return seen
    return None


def test_known_options_reach_the_search(monkeypatch):
    seen = _run(monkeypatch, "2017 CL37", "--itf-search", "--itf-rank-mode", "certainty",
                "--itf-candidate-source", "scan", "--itf-jitter")
    assert seen["desigs"] == ["2017 CL37"]
    cfg = seen["cfg"]
    assert (cfg.ranking_mode, cfg.candidate_path, cfg.jitter) == ("certainty", "scan", True)
    assert seen["trace"] is None


@pytest.mark.parametrize("extra", [
    ("--itf-ranking", "certainty"),      # a value-taking flag this surface does not know
    ("--itf-noise",),                    # ...and a boolean one
    ("--itf-rnk-mode", "ml"),            # a plain typo is the same case
])
def test_an_unknown_itf_option_is_refused_by_name(monkeypatch, capsys, extra):
    """Refused before anything is fetched or fitted -- the value must never be
    read as a designation, and the flag must never be dropped in silence."""
    assert _run(monkeypatch, "2017 CL37", "--itf-search", *extra) is None
    err = capsys.readouterr().err
    assert "unknown option" in err and extra[0] in err


def test_trace_level_zero_is_off_as_on_the_itf_cli(monkeypatch):
    """`python -m pyroorb.itf search --trace-level 0` is "no trace"; this surface used to
    clamp 0 up to the headline level and print one anyway."""
    assert _run(monkeypatch, "2017 CL37", "--itf-search",
                "--itf-trace-level", "0")["trace"] is None
    # an explicit 0 wins over --itf-trace and the environment, as --quiet does there
    assert _run(monkeypatch, "2017 CL37", "--itf-search", "--itf-trace",
                "--itf-trace-level", "0", verbose_env=True)["trace"] is None


def test_a_trace_level_is_passed_through(monkeypatch):
    from pyroorb.itf.trace import Tracer
    trace = _run(monkeypatch, "2017 CL37", "--itf-search", "--itf-trace-level", "3")["trace"]
    assert isinstance(trace, Tracer) and trace.level == 3
    # --itf-trace alone keeps the tracer's own default level
    assert _run(monkeypatch, "2017 CL37", "--itf-search", "--itf-trace")["trace"] is print


@pytest.mark.parametrize("bad", ["7", "-1", "loud"])
def test_a_trace_level_out_of_range_is_refused(monkeypatch, capsys, bad):
    assert _run(monkeypatch, "2017 CL37", "--itf-search", "--itf-trace-level", bad) is None
    assert "--itf-trace-level: expected 0-4" in capsys.readouterr().err
