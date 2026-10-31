"""OFFLINE regression test: re-fit every cached object and assert it still links
as well as it did when the fixture was built. No network -- reads the static
astrometry in tests/fixtures/ (built by build_fixtures.py).

    pytest tests/test_regression.py            # the guard; one test per fixture
    pytest tests/test_regression.py -n 6       # ... in parallel (pytest-xdist)
    pytest tests/test_regression.py --benchmark  # also run the heavy speed-only fixtures

Each fixture is its OWN test item (rather than one loop inside a single test), which
is what lets xdist spread ~180 independent fits across cores and what makes a failure
name the object instead of a wall of text.

Guards (per object), catching BOTH fitting and speed regressions:
  - link STATUS (ok/suspect) must match the golden;
  - used-obs count: a DROP beyond USED_TOL is flagged only if RMS did NOT also
    improve — fewer obs at equal-or-better RMS is a cleaner fit, not a regression
    (more obs is always fine);
  - raw RMS must not worsen by more than max(RMS_ABS_TOL=0.05", RMS_REL_TOL=25%);
  - CPU time must not worsen by more than max(30%, 0.5s) of the baseline -- EXCEPT
    identification PAIRs (the hard, trick-requiring cases), which get a lenient ~2x/+1s
    guard: we optimise the BULK and tolerate a pair slowing onto its careful path.
A genuine regression (stops linking, RMS worsens, or a slowdown) fails; incidental
churn does not. Refresh the baselines (update_goldens.py) after an intentional
improvement -- ESPECIALLY timing, since the speed guard is tight.

TIMING IS CPU TIME (`time.process_time`), not wall clock. Wall clock made the guard a
function of what else the box was doing -- it drifted ~9% run to run on an idle
machine, and it made the suite unparallelisable, because N fits sharing M cores each
"take longer" without anything having regressed. CPU time is the quantity the engine
actually controls. `secs` in every golden is therefore a CPU-second baseline; the two
agree closely for this single-threaded engine (measured CPU/wall = 1.00 per fit, and
the suite stayed green serially against wall-clock-era baselines), so the existing
`secs` were not bulk-rewritten at the switch -- they become CPU-native as objects are
re-baselined normally with update_goldens.py --write.
"""
import os
import time

import pytest

from _fixtures import link_status, load_fixture, iter_fixtures, nongrav_golden
import pyroorb.fit_object as fo
from pyroorb import obs_sources

USED_TOL = 2            # obs  -- a drop beyond this (without an rms win) is a regression
RMS_ABS_TOL = 0.05     # arcsec  -- flag if rms worse by more than ...
RMS_REL_TOL = 0.25     # 25%     -- ... max(RMS_ABS_TOL, RMS_REL_TOL * golden)
SPEED_REL_TOL = 0.30   # 30%     -- flag if slower by more than ...
SPEED_ABS_TOL = 0.5    # CPU seconds -- ... max(SPEED_REL_TOL * baseline, SPEED_ABS_TOL)
# Identification PAIRs are the deliberately HARD cases (mislinkage tricks / careful slow
# path). The goal is total-list time: speed up the BULK, and it's fine if a few pairs slow
# down a bit when a global win pushes them onto the careful path. So pairs get a lenient
# speed guard (still catches a pair that blows up, just not a modest slowdown).
SPEED_REL_TOL_PAIR = 1.0   # pairs may be up to ~2x ...
SPEED_ABS_TOL_PAIR = 1.0   # ... + 1 s slower before it's flagged

_OC = None


def _obscodes():
    global _OC
    if _OC is None:
        _OC = obs_sources.load_obscodes()
    return _OC


def refit(fx):
    """Re-fit one fixture's stored obs, returning (fit, cpu_seconds).

    The ONE place the suite decides what a fixture's timing baseline measures --
    shared with update_goldens.py so the guard and the re-baseline can never
    disagree about the currency."""
    t0 = time.process_time()
    # A production-path fixture (opts "mpc_seed": true) carries its published-orbit seed
    # STATE frozen in the JSON -- offline and drift-proof -- so the refit measures the
    # seeded path the object is actually fit on (see build_fixtures; 1862 Apollo).
    fit = fo.fit_orbit(fx["obs"], _obscodes(), desig=fx["meta"]["designation"],
                       meta=fx["meta"], verbose=False, force=fx.get("force", False),
                       mpc_seed=fx.get("mpc_seed_state"))
    return fit, time.process_time() - t0


def _nongrav_probs(g, fit):
    """Force-model regressions for one fixture: the delivered model, and each coefficient's sign.

    Returns [] when the golden carries no `nongrav` key -- that is a fixture baselined before this
    guard existed, not a passing check, and `update_goldens.py --nongrav` fills them in. A sign map
    is only compared for the axes BOTH sides carry, so the model check owns "different axes" and
    this does not report the same flip twice."""
    if "nongrav" not in g:
        return []
    probs = []
    got = getattr(fit, "nongrav_model", None)
    if got != g["nongrav"]:
        probs.append(f"nongrav {g['nongrav']}->{got}")
    want = g.get("nongrav_signs") or {}
    have = nongrav_golden(fit)["nongrav_signs"]
    for ax in sorted(set(want) & set(have)):
        if want[ax] != have[ax]:
            probs.append(f"{ax} sign {want[ax]:+d}->{have[ax]:+d}")
    return probs


def _check(path, bench_run=False, speed=True):
    """Return (label, ok, message) for one fixture. ok is True/False, or None to mean
    SKIPPED. Flags fitting AND speed regressions, treating fewer-obs/lower-RMS as an
    improvement. A `benchmark`-flagged fixture is a heavy object kept ONLY for speed
    measurement: it is skipped unless bench_run, and even then only its SPEED is guarded
    (we don't gate fit-quality on benchmark objects).

    `speed=False` drops the timing guard -- see `_speed_guarded`."""
    label = os.path.splitext(os.path.basename(path))[0]
    try:
        fx = load_fixture(path)
    except Exception as ex:
        return label, False, f"unreadable fixture: {ex}"
    label, g = fx["label"], fx["golden"]
    is_bench = fx.get("benchmark", False)
    # A benchmark fixture is normally SPEED-only. `"quality": true` opts one back into the fit
    # guards as well -- for an object that is both heavy AND fragile enough that its ORBIT is
    # worth pinning. Without it the two exemptions compose into a blind spot: no quality gate,
    # plus a speed gate that goes stale, and a real regression can sit unnoticed. 2008 NA did
    # exactly that -- 120/122 @ 0.5931" -> 113/122 @ 0.6864" for twelve days across d0c7a28,
    # invisible because it is a benchmark.
    gate_quality = (not is_bench) or bool(fx.get("quality", False))
    if is_bench and not bench_run:
        return label, None, "skipped (benchmark-only; run with --benchmark)"
    fit, secs = refit(fx)
    used, total, rms = fit.n_observations(), fit.n_total_observations(), fit.rms_residual()
    status = link_status(rms, used, total)
    probs = []
    if gate_quality:                      # fit-quality guards (benchmarks: only if "quality")
        if status != g["status"]:
            probs.append(f"status {g['status']}->{status}")
        # USED count: more obs is always fine; a DROP is a regression only if RMS did NOT
        # also improve. Fewer obs at equal-or-better RMS is a cleaner fit, not a fail.
        if (g["used"] - used) > USED_TOL and rms >= g["rms"]:
            probs.append(f"used {g['used']}->{used}")
        # RMS materially worse (improvements never flagged)
        if (rms - g["rms"]) > max(RMS_ABS_TOL, RMS_REL_TOL * g["rms"]):
            probs.append(f"rms {g['rms']}->{rms:.3f}\"")
        # FORCE MODEL. A selection flip is a hard fail with no tolerance, because there is no
        # such thing as a slightly different force model -- and because the numeric guards above
        # cannot see one: 2003 HM changing None -> yarkovsky moved rms 0.7355 -> 0.7466, inside
        # every tolerance here. There is no "improvement" direction to exempt either: gaining a
        # nongrav and losing one are both selection changes that must be looked at and
        # re-baselined deliberately. Absent key = a fixture not yet baselined with one (the goldens
        # predate this guard); update_goldens --nongrav backfills without touching anything else.
        probs += _nongrav_probs(g, fit)
    # SPEED regression vs the recorded CPU-time baseline. Bulk objects: >30% or >0.5 s.
    # Identification PAIRs (the hard cases): lenient -- ~2x or +1 s (we optimize the bulk
    # and tolerate a pair slowing onto its careful path).
    base = g.get("secs")
    is_pair = label.startswith("PAIR_")
    rel = SPEED_REL_TOL_PAIR if is_pair else SPEED_REL_TOL
    pad = SPEED_ABS_TOL_PAIR if is_pair else SPEED_ABS_TOL
    if speed and base is not None and (secs - base) > max(rel * base, pad):
        probs.append(f"slow {base:.1f}s->{secs:.1f}s cpu")
    msg = f"{used}/{total} @ {rms:.3f}\" {status} [{secs:.0f}s cpu]" + (" (bench)" if is_bench else "")
    return label, (not probs), (msg if not probs else msg + "  <<< " + "; ".join(probs))


def _speed_guarded(config):
    """Is the timing guard meaningful in THIS run?

    Only when the fits have the box to themselves. CPU time removes the wall-clock
    guard's sensitivity to unrelated load, but it is not free of contention: measured
    on a 10-core box, 4 xdist workers left CPU time flat while 6 inflated it ~40% (the
    fits compete for memory bandwidth and cache, not for cores), which is more than the
    30% tolerance and tripped objects that had not regressed at all.

    So under xdist the CORRECTNESS guards still run on every fixture -- that is the
    part worth parallelising -- and the timing guard steps aside rather than report
    contention as a regression. Run the suite serially (or with `--speed`) to measure
    speed, exactly as testing-framework.md has always advised.

    Never under continuous integration (the ``CI`` environment variable, which GitHub
    Actions and most CI services set), not even with ``--speed``: the baselines are CPU
    seconds measured on one development machine, so a slower or busier runner would
    fail a build in which nothing regressed."""
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false", "no"):
        return False
    if config.getoption("--speed"):
        return True
    return getattr(config, "workerinput", None) is None


@pytest.mark.slow
@pytest.mark.parametrize("path", iter_fixtures(),
                         ids=lambda p: os.path.splitext(os.path.basename(p))[0])
def test_fixture_still_links(path, request):
    """One fixture, re-fit offline and compared with its recorded golden."""
    label, ok, msg = _check(path, bench_run=request.config.getoption("--benchmark"),
                            speed=_speed_guarded(request.config))
    if ok is None:
        pytest.skip(msg)
    assert ok, f"{label}: {msg}"
