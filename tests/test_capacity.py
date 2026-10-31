"""Offline tests for the container capacity budget (src/pyroorb/web/capacity.py).

Nothing here touches a real cgroup: the module's contract is a pair of
ARITHMETIC decisions -- how many fits and how many ITF searches this box may
run at once -- and every input to them is faked, so the same assertions hold on
a laptop, in CI, and on the deploy.

The properties worth pinning are the ones that were WRONG before this module
existed: a cap that ignores the box (the literal 3, reasoned against a ~1 GB
container and deployed on a 7.6 GB one), and a cap that is generous exactly
where it knows least (an unreadable memory limit).
"""
import os
import sys

import pytest

from pyroorb.web import capacity  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Drop the memoised caps and any inherited override before every test."""
    for var in ("PYROORB_FIT_WORKERS", "PYROORB_ITF_MAX_RUNNING"):
        monkeypatch.delenv(var, raising=False)
    capacity._reset_for_tests()
    yield
    capacity._reset_for_tests()


def _box(monkeypatch, cores, limit_mb):
    """Present a container of `cores` CPUs and `limit_mb` memory."""
    monkeypatch.setattr(capacity, "cpu_quota", lambda: cores)
    monkeypatch.setattr(capacity, "limit_mb", lambda: limit_mb)
    capacity._reset_for_tests()


# --- the fit pool -----------------------------------------------------------

def test_pool_leaves_a_core_for_the_request_threads(monkeypatch):
    """A pool that saturates every core cannot answer the polls that collect
    its own results, which reads to a user as the site being down."""
    _box(monkeypatch, cores=8, limit_mb=7629)
    assert capacity.fit_pool_size() == 7


def test_pool_is_memory_bound_on_a_small_box(monkeypatch):
    """A 4-core 1 GB container is bound by memory, not by its cores -- sizing
    from the CPUs alone is how a small box gets a pool it cannot hold."""
    _box(monkeypatch, cores=4, limit_mb=1024)
    by_mem = int((1024 - capacity.FIT_BASELINE_MB) // capacity.FIT_PEAK_MB)
    assert capacity.fit_pool_size() == max(1, by_mem) < 3


def test_pool_never_reaches_zero(monkeypatch):
    """A box too small to be sized for still has to answer somebody."""
    _box(monkeypatch, cores=1, limit_mb=256)
    assert capacity.fit_pool_size() >= 1


# --- the ITF cap ------------------------------------------------------------

def test_itf_cap_tracks_the_box_it_is_given(monkeypatch):
    """The whole point: the same image on a bigger container admits more.

    This is the regression that motivated the module -- a literal 3 justified
    by "six searches is more than the container has", deployed on a container
    that had plenty.
    """
    _box(monkeypatch, cores=8, limit_mb=2048)
    small = capacity.itf_max_running()
    _box(monkeypatch, cores=8, limit_mb=7629)
    big = capacity.itf_max_running()
    assert big > small


def test_itf_cap_respects_the_memory_left_after_the_fit_pool(monkeypatch):
    """Searches may only spend what the fit pool and the sweep floor have not.

    Recomputed here from the module's own inputs rather than hardcoded, so the
    test pins the RELATIONSHIP and not a number that a re-measured constant
    would invalidate.
    """
    _box(monkeypatch, cores=8, limit_mb=7629)
    reserved = (capacity.FIT_BASELINE_MB
                + capacity.fit_pool_size() * capacity.FIT_PEAK_MB
                + capacity.sweep_floor_mb(capacity._sweep_floor_configured()))
    by_mem = int((7629 - reserved) // capacity.SEARCH_P90_MB)
    assert capacity.itf_max_running() <= by_mem


def test_itf_cap_is_railed_by_the_core_count(monkeypatch):
    """A large-memory box must not derive an unbounded number of sweeps.

    The rail stands in for the shared SBN database and tail latency, not for
    local CPU -- searches are I/O-bound and do not hold a core -- and it sits at
    the measured throughput knee, half the usable cores.
    """
    _box(monkeypatch, cores=4, limit_mb=65536)
    assert capacity.itf_max_running() == 2           # railed, not the 100+ memory allows


def test_itf_cap_on_the_reference_box_matches_the_measurement(monkeypatch):
    """The 8 vCPU / 7629 MB deploy this was calibrated on.

    Memory alone would allow 7 there; the measured curve says throughput has
    already flattened by then (5 concurrent = 0.0425 searches/s, 8 = 0.0447)
    while latency has not (117.7 s vs 179 s).  The delivered cap must come from
    the rail, not from the memory arithmetic -- if this ever reads 7 again the
    latency evidence has been dropped on the floor.
    """
    _box(monkeypatch, cores=8, limit_mb=7629)
    assert capacity.itf_max_running() == 4


def test_itf_cap_falls_back_to_the_legacy_literal_when_unsized(monkeypatch):
    """An unreadable memory limit is the one case with NO basis to be generous.

    Memory is the budget that bounds a search, `search_capacity` reads the same
    limits and is equally inert here, so falling back to the core count would
    hand an uncapped 16-core box sixteen concurrent sweeps with nothing behind
    them.  The honest answer is the hand-set number this module replaced.
    """
    _box(monkeypatch, cores=16, limit_mb=None)
    assert capacity.itf_max_running() == capacity.ITF_UNSIZED_DEFAULT


def test_itf_cap_never_reaches_zero(monkeypatch):
    """Clamped to one even where the arithmetic goes negative.

    Safe only because admission independently consults `search_capacity`,
    which declines honestly on a box with no room -- this clamp decides how
    many MAY run, not that one can.
    """
    _box(monkeypatch, cores=1, limit_mb=512)
    assert capacity.itf_max_running() >= 1


# --- overrides --------------------------------------------------------------

@pytest.mark.parametrize("var,fn", [
    ("PYROORB_FIT_WORKERS", "fit_pool_size"),
    ("PYROORB_ITF_MAX_RUNNING", "itf_max_running"),
])
def test_env_override_wins_over_the_derived_value(monkeypatch, var, fn):
    """An operator who sets the number explicitly gets the number they set."""
    _box(monkeypatch, cores=8, limit_mb=7629)
    monkeypatch.setenv(var, "2")
    capacity._reset_for_tests()
    assert getattr(capacity, fn)() == 2


@pytest.mark.parametrize("var,fn", [
    ("PYROORB_FIT_WORKERS", "fit_pool_size"),
    ("PYROORB_ITF_MAX_RUNNING", "itf_max_running"),
])
def test_unparseable_override_falls_through_to_the_derived_value(
        monkeypatch, var, fn):
    """A typo in a service variable must not take the server down; the derived
    value is a better answer than a crash on import."""
    _box(monkeypatch, cores=8, limit_mb=7629)
    monkeypatch.setenv(var, "not-a-number")
    capacity._reset_for_tests()
    assert getattr(capacity, fn)() >= 1


# --- reporting --------------------------------------------------------------

def test_describe_reports_both_caps_and_their_inputs(monkeypatch):
    """A deploy must be answerable about what it thinks it is, rather than
    having the answer inferred from its behaviour under load."""
    _box(monkeypatch, cores=8, limit_mb=7629)
    d = capacity.describe()
    assert d["fit_pool_size"] == capacity.fit_pool_size()
    assert d["itf_max_running"] == capacity.itf_max_running()
    assert d["memory_limit_mb"] == 7629
    assert d["usable_cores"] == 8
