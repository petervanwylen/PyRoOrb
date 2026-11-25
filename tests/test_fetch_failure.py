"""Offline tests for telling "the MPC did not answer" from "no such observations".

The distinction is the whole point. Before this, every fetch failure -- a timeout, a
throttle, a TLS error, a response shape we could not read -- was swallowed and reported
as "no observations found", which is a confident statement ABOUT THE OBJECT made on the
strength of a network error. Under load (a throttled container on a launch day) the site
would tell every visitor at once that their object does not exist.

Nothing here touches the network: the fetchers are replaced.
"""
import os
import sys

import pytest

import pyroorb.obs_sources as osrc  # noqa: E402


@pytest.fixture
def no_neocp(monkeypatch):
    """Keep the NEOCP branch out of the way; it is not what these tests are about."""
    from pyroorb import ades
    monkeypatch.setattr(ades, "is_neocp_desig", lambda d: False, raising=False)
    return ades


def _all_fetchers(monkeypatch, ades, behaviour):
    """Point every fetch route -- batch and per-designation -- at `behaviour`."""
    monkeypatch.setattr(ades, "fetch_combined_ades", behaviour, raising=False)
    monkeypatch.setattr(osrc, "fetch_combined", behaviour, raising=False)


def test_a_transport_failure_is_recorded_not_disguised(monkeypatch, no_neocp):
    """An exception means the MPC did not ANSWER -- a fact about the network."""
    def boom(*a, **k):
        raise TimeoutError("connection timed out")
    _all_fetchers(monkeypatch, no_neocp, boom)
    obs, meta = osrc._fetch_observations(["2003 LW2"], {})
    assert obs == []
    assert "2003 LW2" in (meta.get("fetch_errors") or {}), \
        "a fetch that FAILED must not be reported as an object with no observations"
    assert "TimeoutError" in meta["fetch_errors"]["2003 LW2"]


def test_the_public_record_names_the_source_not_the_server(monkeypatch, no_neocp):
    """A replica connect error names host, port and user: the web caller's record carries
    only which source failed and the exception's class (the full text is logged)."""
    def boom(*a, **k):
        raise ConnectionError('connection to server at "10.0.0.5", port 5432 failed: user "ro"')
    _all_fetchers(monkeypatch, no_neocp, boom)
    obs, meta = osrc._fetch_observations(["2003 LW2"], {})
    assert "10.0.0.5" in meta["fetch_errors"]["2003 LW2"]           # the owner's record
    pub = meta["fetch_errors_public"]["2003 LW2"]
    assert "ConnectionError" in pub and "10.0.0.5" not in pub and "5432" not in pub
    from pyroorb.web import engine_api
    assert engine_api._public_fetch_errors(meta) == {"2003 LW2": pub}
    assert engine_api._public_fetch_errors({"fetch_errors": {"x": "SBN: E: host"}}) == \
        {"x": "no answer"}


def test_a_genuine_empty_result_records_no_error(monkeypatch, no_neocp):
    """A designation the MPC simply has nothing for is NOT a fetch failure, and must
    not be dressed up as one -- the user should still be told it is unknown."""
    _all_fetchers(monkeypatch, no_neocp, lambda *a, **k: ([], {}))
    obs, meta = osrc._fetch_observations(["not a real object"], {})
    assert obs == []
    assert not (meta.get("fetch_errors") or {})


def test_a_partial_failure_is_recorded_even_though_rows_came_back(monkeypatch, no_neocp):
    """The dangerous case: a two-designation linkage where ONE fetch fails.

    Rows arrive, so the fit would run and be reported as a success -- on half the
    evidence the caller asked for. The failure has to survive onto the meta of a
    NON-EMPTY result, or nothing downstream can know the arc is short.

    Driven through the PER-DESIGNATION path, because that is where a partial outcome
    is visible at all: the batch fast path ahead of it is all-or-nothing and returns
    the moment it has rows (see the caveat test below).
    """
    good = [{"jd_utc": 2451000.0 + i, "packed": "K03L02W", "designation": "2003 LW2"}
            for i in range(3)]

    def only_lw2(desigs, obscodes, cache_dir=None, **k):
        ds = list(desigs) if not isinstance(desigs, str) else [desigs]
        if len(ds) > 1:                      # the BATCH attempt: fail, to fall through
            raise ConnectionResetError("connection reset by peer")
        if ds[0] == "2003 LW2":
            return list(good), {}
        raise ConnectionResetError("connection reset by peer")

    def always_boom(*a, **k):
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(no_neocp, "fetch_combined_ades", always_boom, raising=False)
    monkeypatch.setattr(osrc, "fetch_combined", only_lw2, raising=False)
    monkeypatch.setattr(osrc, "_unpack", lambda p: "2003 LW2", raising=False)

    obs, meta = osrc._fetch_observations(["2003 LW2", "2010 TR241"], {})
    assert obs, "the designation that DID fetch should still have produced rows"
    failed = meta.get("fetch_errors") or {}
    assert "2010 TR241" in failed, \
        "a linkage fitted on half its designations must say so"
    assert "2003 LW2" not in failed


def test_the_batch_fast_path_is_all_or_nothing_by_design(monkeypatch, no_neocp):
    """A KNOWN LIMIT, pinned so it is a decision and not a surprise.

    When the whole-batch fetch succeeds it returns immediately, without checking that
    every requested designation is represented -- so a batch that came back short is
    not recorded. That check is deliberately NOT made here: a numbered object expands
    to its historical aliases, so the designations present routinely differ from the
    ones requested, and "was everything I asked for returned?" would fire on every
    aliased object. Detecting a short BATCH needs alias resolution, not a set compare.
    """
    good = [{"jd_utc": 2451000.0 + i, "packed": "K03L02W", "designation": "2003 LW2"}
            for i in range(3)]
    monkeypatch.setattr(no_neocp, "fetch_combined_ades",
                        lambda *a, **k: (list(good), {"designations": ["2003 LW2"]}),
                        raising=False)
    _obs, meta = osrc._fetch_observations(["2003 LW2", "2010 TR241"], {})
    assert not (meta.get("fetch_errors") or {})   # documents the gap, does not bless it


def test_the_failure_detail_names_which_route_failed(monkeypatch, no_neocp):
    """ADES and OBS80 are separate services; which one failed is the first thing
    anyone debugging this needs, so it is kept rather than flattened."""
    def boom(*a, **k):
        raise OSError("tls handshake failed")
    _all_fetchers(monkeypatch, no_neocp, boom)
    _obs, meta = osrc._fetch_observations(["433"], {})
    detail = (meta.get("fetch_errors") or {}).get("433", "")
    assert "ADES" in detail and "OBS80" in detail


def test_fetch_failure_message_is_the_shared_refusal():
    """The one plain-text form of the refusal, shared by the CLI, --ident-check and the
    ITF seed fetch (the web routes phrase their own from the same record)."""
    assert osrc.fetch_failure_message(None) is None
    assert osrc.fetch_failure_message({}) is None
    assert osrc.fetch_failure_message({"fetch_errors": {}}) is None
    msg = osrc.fetch_failure_message(
        {"fetch_errors": {"2000 XY1": "ADES: TimeoutError: timed out; OBS80: URLError: x"}})
    assert "2000 XY1" in msg
    assert "TimeoutError" in msg            # the first route's failure, as the detail
    assert "did not answer" in msg          # the claim is about the network, not the object


def _failed_meta(errors):
    return {"designation": sorted(errors)[0], "designations": [], "combined": False,
            "other_designations": [], "fetch_errors": dict(errors), "format": "OBS80"}


@pytest.mark.slow
def test_the_cli_main_refuses_a_failed_fetch(monkeypatch, capsys):
    """The CLI must not be the one surface that quietly proceeds: a linkage where one
    designation's fetch failed still returns rows, and before this the CLI fitted them
    and reported success from half the requested evidence (the web routes already
    refuse). Slow-marked only for the cli import chain; no network, no fit."""
    from pyroorb import cli
    monkeypatch.setattr(cli, "load_obscodes", lambda: {})
    monkeypatch.setattr(cli, "fetch_observations",
                        lambda desigs, oc: ([], _failed_meta({"2000 XY1": "OBS80: URLError: x"})))
    monkeypatch.setattr(sys, "argv", ["fit_object.py", "2000 XY1"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 3
    assert "did not answer" in capsys.readouterr().err


@pytest.mark.slow
def test_ident_check_files_an_outage_as_no_fetch_not_no(monkeypatch, capsys):
    """A "no" here is a durable classification row: filing an outage as a non-match
    records a wrong verdict that looks like a right one. The distinct code counts as a
    skip to every reader and marks the row re-runnable."""
    from pyroorb import cli
    monkeypatch.setattr(cli, "_fetch_ident_pair",
                        lambda desigs, oc: ([], _failed_meta({"2025 AB1": "ADES: TimeoutError: t"})))
    code = cli._ident_check_one("2025 AB1", "2025 CD2", "/nonexistent/out.txt", {})
    assert code == "no-fetch"
    err = capsys.readouterr().err
    assert "fetch failed" in err and "[ident-check-class] no-fetch" in err


@pytest.mark.slow
def test_itf_seed_fetch_refuses_a_partial_arc(monkeypatch):
    """A corridor swept from a partially-fetched seed is aimed by half the arc -- the
    search would run to completion and answer "no match" with conviction. The seed
    fetch must fail the search with the transport truth instead."""
    from pyroorb.itf import search as itfsearch
    monkeypatch.setattr(osrc, "fetch_observations",
                        lambda desigs, oc: ([], _failed_meta({"2019 XY9": "ADES: URLError: u"})))
    with pytest.raises(RuntimeError) as e:
        itfsearch._pub_obs80_from_fetch(["2019 XY9", "2005 QQ1"], {}, None)
    assert "seed fetch" in str(e.value) and "did not answer" in str(e.value)


def test_an_empty_obs80_answer_is_empty_not_an_error(monkeypatch, tmp_path):
    """The API answers a name it files nothing under (a comet fragment) with an empty OBS80
    block. That is an answer: [] and a meta, never an IndexError the caller then reports as
    "the MPC did not answer"."""
    monkeypatch.setattr(osrc, "fetch_obs80", lambda d, cache=None: "\n")
    obs, meta = osrc.fetch_combined(["73P-C"], {}, cache_dir=tmp_path)
    assert obs == [] and meta["designations"] == [] and meta["designation"] == "73P-C"


def _route_spy(monkeypatch, sbn_rows):
    """Record which route each designation is fetched from; the replica answers `sbn_rows`."""
    from pyroorb import ades
    asked = []

    def api(tag):
        def f(desigs, *a, **k):
            asked.append((tag, tuple(desigs)))
            return [{"designation": d, "jd_utc": 2460000.0} for d in desigs if d == "73P"], {}
        return f
    monkeypatch.setattr(ades, "fetch_combined_ades", api("ADES"), raising=False)
    monkeypatch.setattr(osrc, "fetch_combined", api("OBS80"), raising=False)
    monkeypatch.setattr(ades, "fetch_combined_neocp", api("NEOCP"), raising=False)

    def sbn(d, obscodes, why):
        asked.append(("SBN", (d,)))
        return [dict(r, designation=d) for r in sbn_rows]
    monkeypatch.setattr(osrc, "_fetch_from_sbn", sbn)
    return asked


def test_a_fragment_is_fetched_from_the_replica_only(monkeypatch):
    """The published-obs API serves no fragment astrometry under the fragment's name or the
    parent's; the replica files it under the fragment's own name. A fragment never asks the
    API (or the NEOCP)."""
    asked = _route_spy(monkeypatch, [{"jd_utc": 2453800.0}])
    obs, meta = osrc._fetch_observations(["73P-C"], {})
    assert asked == [("SBN", ("73P-C",))]
    assert len(obs) == 1 and meta["source"] == "SBN" and meta["designation"] == "73P-C"


def test_a_request_with_a_fragment_skips_the_whole_batch_path(monkeypatch):
    """The batch API call would answer for 73P alone and quietly drop the fragment, so a
    request carrying one goes per designation: the parent from the API, the fragment from
    the replica."""
    asked = _route_spy(monkeypatch, [{"jd_utc": 2453800.0}])
    obs, meta = osrc._fetch_observations(["73P-C", "73P"], {})
    assert ("ADES", ("73P-C", "73P")) not in asked and ("OBS80", ("73P-C", "73P")) not in asked
    assert ("SBN", ("73P-C",)) in asked and ("ADES", ("73P",)) in asked
    assert sorted(meta["designations"]) == ["73P", "73P-C"]


def test_a_fragment_the_replica_has_nothing_for_says_so(monkeypatch):
    """No replica rows (none configured, or none for it): no transport failure is recorded
    and the refusal says where fragments come from."""
    asked = _route_spy(monkeypatch, [])
    obs, meta = osrc._fetch_observations(["73P-C"], {})
    assert obs == [] and asked == [("SBN", ("73P-C",))]
    assert not (meta.get("fetch_errors") or {})
    assert meta["fragments_unserved"] == ["73P-C"]
    msg = osrc.fetch_failure_message(meta)
    assert "73P-C" in msg and "SBN replica only" in msg and "(73P)" in msg
    assert "did not answer" not in msg


def test_comet_fragment_forms():
    assert osrc.is_comet_fragment("73P-C") and osrc.is_comet_fragment("73P-BV")
    assert osrc.is_comet_fragment("C/2019 Q4-A")
    for d in ("73P", "C/2019 Q4", "2013 PY26", "P12pwHF", "433"):
        assert not osrc.is_comet_fragment(d)
