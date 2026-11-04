"""Guards from the first outside install of 0.9.1 (a Linux user who pip-installed it):

* ASSIST built against another REBOUND than the one installed must be refused, not crash the
  process (``pyroorb.abi_check``);
* ``ephemeris`` must refuse input it used to answer with nonsense (an unknown station, a span
  that runs backwards);
* ``fetch_observations`` raises ``FetchError`` (still a ``RuntimeError``);
* the package namespace carries no stray ``os`` / ``warnings``.

Offline: the checks run before any engine call, and the canary's child process is faked."""
import subprocess
import types

import pytest

import pyroorb
from pyroorb import abi_check


# --- ephemeris input ------------------------------------------------------------------------

def test_ephemeris_refuses_an_unknown_station():
    with pytest.raises(ValueError, match="XYZ"):
        pyroorb.ephemeris(None, "2026-11-01", "2026-11-02", site="XYZ")


def test_ephemeris_refuses_a_backwards_span_and_a_bad_step():
    with pytest.raises(ValueError, match="before start"):
        pyroorb.ephemeris(None, "2026-11-03", "2026-11-01")
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="step"):
            pyroorb.ephemeris(None, "2026-11-01", "2026-11-03", step=bad)


def test_a_spacecraft_position_needs_no_station_code(monkeypatch):
    """With ``observer=`` the code only labels the rows, so an unlisted one (C57) is fine."""
    import pyroorb.ephemerides as eph
    seen = {}
    monkeypatch.setattr(eph, "compute_ephemeris", lambda *a, **k: seen.setdefault("site", k["site"]))
    pyroorb.ephemeris(None, "2026-11-01", "2026-11-02", site="C57", observer=[0.0, 0.0, 0.0])
    assert seen["site"] == "C57"


# --- fetch errors ---------------------------------------------------------------------------

def test_fetch_failure_is_a_fetcherror_and_still_a_runtimeerror(monkeypatch):
    from pyroorb import obs_sources
    monkeypatch.setattr(obs_sources, "fetch_observations",
                        lambda d, oc: ([], {"fetch_errors": {"2099 ZZ9": "ADES: HTTPError: 500"}}))
    with pytest.raises(pyroorb.FetchError, match="2099 ZZ9") as ei:
        pyroorb.fetch_observations("2099 ZZ9")
    assert isinstance(ei.value, RuntimeError)
    assert "FetchError" in pyroorb.__all__


def test_fetch_chatter_goes_to_logging_not_stdout(capsys, caplog):
    import logging
    from pyroorb import obs_sources
    with caplog.at_level(logging.INFO, logger="pyroorb.fetch"):
        obs_sources._log.warning("could not reach the MPC for %s: %s", "X", "why")
    assert "could not reach the MPC for X" in caplog.text
    assert capsys.readouterr().out == ""


# --- namespace ------------------------------------------------------------------------------

def test_the_package_namespace_has_no_stdlib_leaks():
    for name in ("os", "warnings"):
        assert not hasattr(pyroorb, name)
    assert "load_obscodes" not in pyroorb.__all__          # `obscodes()` is the documented one
    assert callable(pyroorb.load_obscodes)                  # but the old spelling still imports


# --- the ASSIST / REBOUND canary ------------------------------------------------------------

@pytest.fixture
def canary(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROORB_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(abi_check.SKIP_ENV, raising=False)
    monkeypatch.setattr(abi_check, "_verified", False)

    def child(returncode, stderr=""):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")
        monkeypatch.setattr(abi_check.subprocess, "run", fake_run)
        return calls
    return child


@pytest.mark.parametrize("rc, err", [(-6, ""), (134, ""), (abi_check._UNCHANGED, ""),
                                     (1, "free(): invalid size")])
def test_a_mismatched_assist_is_refused_with_the_rebuild_command(canary, rc, err):
    canary(rc, err)
    with pytest.raises(abi_check.AssistMismatchError) as ei:
        abi_check.verify("kernel")
    msg = str(ei.value)
    assert "--no-build-isolation" in msg and "setuptools" in msg and "REBOUND" in msg
    assert not abi_check._verified


def test_a_matching_assist_is_verified_once_then_stamped(canary):
    calls = canary(0)
    abi_check.verify("kernel")
    assert len(calls) == 1 and abi_check._verified
    abi_check._verified = False                  # a new process: the stamp answers, no child
    abi_check.verify("kernel")
    assert len(calls) == 1 and abi_check._verified


def test_an_unrelated_child_failure_is_not_a_mismatch(canary):
    """A missing kernel or other ordinary error says nothing about the ABI: let the fit go on
    (and fail with its own, better message) instead of blaming the install."""
    canary(1, "FileNotFoundError: planets kernel")
    abi_check.verify("kernel")
    assert not abi_check._verified


def test_the_check_can_be_switched_off(canary, monkeypatch):
    calls = canary(-6)
    monkeypatch.setenv(abi_check.SKIP_ENV, "1")
    abi_check.verify("kernel")
    assert calls == []


def test_the_canary_really_runs_and_passes_on_this_install(tmp_path, monkeypatch):
    """The real child process, on the environment under test (which must be consistent)."""
    from pyroorb import force_assist
    monkeypatch.setenv("PYROORB_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(abi_check.SKIP_ENV, raising=False)
    monkeypatch.setattr(abi_check, "_verified", False)
    try:
        kernel = force_assist.ensure_planets_kernel()
    except Exception:
        pytest.skip("no DE440 kernel in this environment")
    abi_check.verify(kernel)
    assert abi_check._verified


# --- historical (pre-1960) dates ------------------------------------------------------------

def _astropy_jd(iso):
    """The reference: astropy's UTC Julian date, with ERFA's dubious-year notice muted."""
    import warnings
    from astropy.time import Time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(Time(iso, format="isot", scale="utc").jd)


def test_pre_1960_julian_dates_equal_astropys_to_the_last_bit():
    import random
    from pyroorb.ades import _jd_utc
    from pyroorb.obs80 import utc_jd_before_1960
    rng = random.Random(1893)
    samples = [(1900, 2, 28), (1900, 3, 1), (1896, 2, 29), (1959, 12, 30), (1840, 1, 1)]
    samples += [(rng.randint(1840, 1958), rng.randint(1, 12), rng.randint(1, 28)) for _ in range(300)]
    for y, mo, d in samples:
        frac = rng.random()
        hh, mi, ss = int(frac * 24), int(frac * 1440) % 60, (frac * 86400) % 60
        iso = f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mi:02d}:{ss:06.3f}"
        assert _jd_utc(iso + "Z") == _astropy_jd(iso), iso
        assert utc_jd_before_1960(y, mo, d) == _astropy_jd(f"{y:04d}-{mo:02d}-{d:02d}T00:00:00")
    assert utc_jd_before_1960(1959, 12, 31) is None and utc_jd_before_1960(2014, 9, 28) is None


def test_old_plates_parse_without_an_erfa_warning_and_from_1960_the_path_is_unchanged():
    import warnings
    from pyroorb.ades import _jd_utc
    from pyroorb.obs80 import mpc80_to_radec
    with open("tests/digest2/fixtures/sample.obs", encoding="utf-8") as fh:
        line = next(l.rstrip("\n") for l in fh if len(l.rstrip("\n")) >= 80)
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any warning, ERFA's included, fails the test
        old = mpc80_to_radec(line[:15] + "1893 06 01.50000" + line[31:])
        _jd_utc("1893-06-01T12:00:00Z")
    assert _jd_utc("1893-06-01T12:00:00Z") == 2412616.0              # a known Julian date
    assert old[0] is not None
    # 1959-12-31 is the one day astropy stretches (ERFA's 1.4 s offset for 1960-01-01): it keeps
    # the original path, so nothing changes for it
    assert _jd_utc("1959-12-31T12:00:00Z") == _astropy_jd("1959-12-31T12:00:00")
    assert _jd_utc("2014-09-28T08:40:58.656Z") == _astropy_jd("2014-09-28T08:40:58.656")


# --- applications keep the fetch notes on stdout --------------------------------------------

@pytest.fixture
def fetch_logger_restored():
    import logging
    from pyroorb import obs_sources
    log = obs_sources._log
    saved = (list(log.handlers), log.level, log.propagate)
    yield log
    log.handlers[:], log.level, log.propagate = saved[0], saved[1], saved[2]


def test_an_application_gets_the_old_stdout_lines_a_library_user_does_not(capsys, fetch_logger_restored):
    from pyroorb import obs_sources
    log = obs_sources._log
    log.info("fetched 31 obs for P12nCge from the NEOCP")          # a notebook: quiet
    assert capsys.readouterr().out == ""
    obs_sources.route_fetch_notes_to_stdout()
    obs_sources.route_fetch_notes_to_stdout()                      # idempotent: one handler
    assert len([h for h in log.handlers if isinstance(h, obs_sources._StdoutNow)]) == 1
    log.info("fetched 31 obs for P12nCge from the NEOCP")
    log.info("ADES batch unavailable: HTTP Error 500; falling back to OBS80")
    log.warning("could not reach the MPC for 2099 ZZ9: ADES: HTTPError")
    assert capsys.readouterr().out.splitlines() == [
        "(fetched 31 obs for P12nCge from the NEOCP)",
        "(ADES batch unavailable: HTTP Error 500; falling back to OBS80)",
        "(could not reach the MPC for 2099 ZZ9: ADES: HTTPError)"]


def test_the_command_line_installs_the_handler(monkeypatch, capsys):
    from pyroorb import cli, obs_sources
    calls = []
    monkeypatch.setattr(obs_sources, "route_fetch_notes_to_stdout", lambda: calls.append(1))
    monkeypatch.setattr("sys.argv", ["pyroorb", "--help"])
    cli.main()
    assert calls == [1]
