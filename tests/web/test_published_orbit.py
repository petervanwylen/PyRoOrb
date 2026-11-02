"""Offline tests for GET /api/published_orbit (engine_api.published_orbit).

This endpoint hands a visitor the MINOR PLANET CENTER's orbit, not ours, in the seconds
before our own fit of the same object finishes -- 433 Eros takes ~644 s to fit and one
request to look up. Two things therefore have to hold on every path: the payload must
say whose orbit it is, and an unreachable MPC must not be reported as an object the MPC
has no orbit for.

The fetch is replaced; nothing here touches the network.
"""
import os
import sys

import pytest

from pyroorb.web import engine_api  # noqa: E402
from pyroorb import mpc_orbit  # noqa: E402


_DOC = {"source": "MPC", "designation": "(433)", "name": "Eros",
        "epoch_jd": 2461200.5,
        "elements": {"a": 1.4582437, "e": 0.2228779, "incl": 10.8285467,
                     "node": 304.2679687, "argperi": 178.9181420,
                     "mean_anomaly": 62.5114491},
        "H": 10.395, "G": 0.15, "n_oppositions": 59,
        "n_observations": 18660, "arc": "1893-2026", "nongrav": False}


def test_a_published_orbit_is_always_attributed(monkeypatch):
    """The one thing this must never do is let the MPC's orbit read as ours."""
    monkeypatch.setattr(mpc_orbit, "fetch_published_elements",
                        lambda d, cache=None, timeout=60: dict(_DOC))
    out = engine_api.published_orbit("433")
    assert out["ok"] is True
    assert out["source"] == "MPC"
    assert out["elements"]["a"] == pytest.approx(1.4582437)
    assert out["n_observations"] == 18660


def test_an_unreachable_mpc_is_not_reported_as_a_missing_orbit(monkeypatch):
    """A timeout is a fact about the network. Saying "the MPC has no orbit for this"
    instead would be a claim about the object that we have no evidence for."""
    def boom(d, cache=None, timeout=60):
        raise TimeoutError("connection timed out")
    monkeypatch.setattr(mpc_orbit, "fetch_published_elements", boom)
    out = engine_api.published_orbit("433")
    assert out["ok"] is False
    assert out.get("retryable") is True
    assert "did not answer" in out["error"]


def test_a_genuinely_absent_orbit_says_so(monkeypatch):
    """...and the converse: when the MPC answers and simply has nothing, the honest
    answer is that there is no published orbit -- not a retryable network error."""
    monkeypatch.setattr(mpc_orbit, "fetch_published_elements",
                        lambda d, cache=None, timeout=60: None)
    out = engine_api.published_orbit("2099 ZZ999")
    assert out["ok"] is False
    assert not out.get("retryable")
    assert "no published orbit" in out["error"]


def test_an_empty_designation_is_refused_without_a_fetch(monkeypatch):
    """Nothing should reach the MPC on an empty box."""
    def never(*a, **k):
        raise AssertionError("should not have fetched")
    monkeypatch.setattr(mpc_orbit, "fetch_published_elements", never)
    out = engine_api.published_orbit("   ")
    assert out["ok"] is False


def test_the_route_is_rate_limited():
    """Cheap for us, a request to the MPC for them: the site must not be usable as an
    unmetered proxy onto their orbit service."""
    from pyroorb.web import server
    assert server.Handler._RATE_BUCKETS.get("/api/published_orbit") == "random"


def test_a_local_failure_is_not_blamed_on_the_mpc(monkeypatch):
    """OUR bug is not THEIR outage.

    The first version of this endpoint caught every exception and reported it as
    "the MPC did not answer" -- and the first thing it caught in production was a
    FileNotFoundError writing our own cache directory, while the MPC had answered
    perfectly. Committing the very category error the message exists to prevent.
    """
    def local_boom(d, cache=None, timeout=60):
        raise FileNotFoundError("[Errno 2] No such file or directory: '/app/cache/x'")
    monkeypatch.setattr(mpc_orbit, "fetch_published_elements", local_boom)
    out = engine_api.published_orbit("433")
    assert out["ok"] is False
    assert "did not answer" not in out["error"], "blamed the MPC for a local error"
    assert not out.get("retryable")
    assert "FileNotFoundError" in out["detail"]


def test_a_cache_that_cannot_be_written_does_not_fail_the_fetch(tmp_path, monkeypatch):
    """A cache is an optimisation. Writing it must never turn a successful MPC
    lookup into an error -- which is what took the route down on a container whose
    data directory did not exist yet."""
    import json as _json
    payload = [{"mpc_orb": [{"KEP": {"coefficient_names": ["a"],
                                     "coefficient_values": [1.0]},
                             "epoch_data": {"epoch": 61200.0, "timeform": "MJD"}}]}]

    class _Resp:
        def read(self): return _json.dumps(payload).encode()
    monkeypatch.setattr(mpc_orbit.urllib.request, "urlopen",
                        lambda *a, **k: _Resp())
    deep = tmp_path / "does" / "not" / "exist" / "c.json"
    out = mpc_orbit._get_orb_raw("433", cache=str(deep))
    assert out == payload
    assert deep.exists(), "the parent directory should have been created"
