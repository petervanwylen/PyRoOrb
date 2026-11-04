"""Shared pytest setup for the PyRoOrb suite.

The package is imported as ``pyroorb`` (``src/`` is on the path through pyproject.toml's
``pythonpath``, as is ``tests/`` for the shared helpers ``_fixtures`` / ``_itf_fixtures``).
The repo-root ``conftest.py`` makes the default run offline: no network, no developer
cache.

It also offers an ``obscodes`` fixture for NEW tests, so they need not repeat the
hand-rolled ``_OC`` global that half a dozen modules carry:

    def test_something(obscodes):        # the parsed MPC station table
        fit = fo.fit_orbit(obs, obscodes, ...)

Markers (declared in pyproject.toml, so a typo is an error not a silent no-op):

    slow      a full engine fit or a multi-object sweep -- deselect with -m "not slow"
    live      needs the network or the SBN database; DESELECTED by default,
              opt in with --live (see pytest_collection_modifyitems below)

Run the suite from the REPO ROOT:

    pytest                       # offline, everything but `live`
    pytest -n 4                  # in parallel (pytest-xdist)
    pytest -m "not slow"         # the fast unit tests only
    pytest --live                # include the DB/network-backed tests
    pytest --durations=10        # where the time goes
"""
import warnings

import pytest

# REBOUND/ASSIST and astropy are chatty about ephemeris fallbacks and leap seconds;
# the suite asserts on values, not on warning text.
warnings.filterwarnings("ignore")


def pytest_addoption(parser):
    parser.addoption("--benchmark", action="store_true", default=False,
                     help="also run the `benchmark`-flagged regression fixtures -- heavy "
                          "objects kept for SPEED measurement only (their fit quality is "
                          "deliberately not guarded). See tests/README.md.")
    parser.addoption("--speed", action="store_true", default=False,
                     help="force the regression suite's TIMING guard on even under xdist. "
                          "Off in parallel runs by default because worker contention "
                          "inflates CPU time past the tolerance -- see "
                          "test_regression._speed_guarded.")
    parser.addoption("--live", action="store_true", default=False,
                     help="also run tests marked `live` (network / SBN database). "
                          "Offline is deterministic but cannot see anything on the DB "
                          "side of the ITF search -- run this before trusting a "
                          "recall-affecting change.")


def pytest_collection_modifyitems(config, items):
    """`live` tests are opt-in.  They are slow (minutes), need credentials, and
    turn a green offline run into a hang when the network is unreachable -- so a
    default `pytest` invocation must never pick them up."""
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="needs --live (network / SBN database)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def obscodes():
    """The parsed MPC observatory-code table -- built ONCE per session (or per
    xdist worker) instead of once per test module."""
    from pyroorb import obs_sources
    return obs_sources.load_obscodes()


@pytest.fixture(autouse=True)
def _no_ambient_memory_floor(monkeypatch):
    """The sweep memory floor (candidates._mem_available_mb) reads the BOX's
    available RAM, so a co-running fit or a parallel xdist worker can make an
    offline sweep test skip windows it would normally serve.  Neutralize it
    for every test; the floor's own test monkeypatches the function again
    explicitly."""
    try:
        import pyroorb.itf.candidates as _C
    except ImportError:
        return
    monkeypatch.setattr(_C, "_mem_available_mb", lambda: None)


@pytest.fixture(autouse=True)
def _fetch_logger_restored():
    """A test that runs an application's main (cli, web server, ITF tools) calls
    obs_sources.route_fetch_notes_to_stdout(), which leaves a stdout handler on the
    `pyroorb.fetch` logger for the rest of the session; every later test that
    expects library-quiet fetch notes then fails, depending on test order.  Put
    the logger back as it was after each test."""
    import logging
    log = logging.getLogger("pyroorb.fetch")
    saved = (list(log.handlers), log.level, log.propagate)
    yield
    log.handlers[:] = saved[0]
    log.setLevel(saved[1])
    log.propagate = saved[2]


@pytest.fixture(autouse=True)
def _no_ambient_sbn_fallback(monkeypatch, request):
    """An offline test that makes the MPC fetchers fail must not then reach the
    real SBN replica through the observation-fetch fallback (sbn_obs) just
    because the box has SBN_DB_HOST configured.  Off for every test not marked
    `live`; the fallback's own tests turn it back on with fakes in place."""
    if "live" not in request.keywords:
        monkeypatch.setenv("PYROORB_SBN_FALLBACK", "0")
