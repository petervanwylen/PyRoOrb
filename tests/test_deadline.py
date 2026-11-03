"""The wall-clock bound on one unit of engine work (`deadline.py`).

Two things have to hold at once: a wedged web fit must actually STOP, and the
CLI / batch / regression paths -- which set no deadline -- must be untouched,
including paying nothing per observation for the seam that makes it possible.
"""
import json
import os
import threading
import time

import numpy as np
import pytest

from pyroorb import deadline
from pyroorb import force_assist


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------
def test_no_deadline_by_default():
    assert deadline.active() is None
    assert deadline.remaining() is None
    deadline.check()                       # must not raise


@pytest.mark.parametrize("falsy", [0, 0.0, None, -1])
def test_a_falsy_budget_is_no_deadline(falsy):
    """Callers that may or may not want a bound must not special-case it."""
    with deadline.deadline(falsy):
        assert deadline.active() is None
        deadline.check()


def test_check_raises_once_past_and_carries_the_budget():
    with deadline.deadline(0.05, label="a fit"):
        deadline.check()                   # not yet
        time.sleep(0.08)
        with pytest.raises(deadline.DeadlineExceeded) as got:
            deadline.check()
    ex = got.value
    assert ex.label == "a fit"
    assert ex.budget_sec == pytest.approx(0.05)
    assert ex.elapsed_sec >= 0.05
    assert "0.05" in str(ex) and "a fit" in str(ex)


def test_the_deadline_is_released_on_the_way_out():
    with deadline.deadline(5):
        assert deadline.active() is not None
    assert deadline.active() is None


def test_it_is_released_even_when_the_work_raises():
    with pytest.raises(ValueError):
        with deadline.deadline(5):
            raise ValueError("boom")
    assert deadline.active() is None


def test_a_nested_budget_may_tighten_but_never_extend():
    """Otherwise a nested deadline would be a way out of the outer one."""
    with deadline.deadline(0.5, label="outer"):
        outer_at = deadline.active()[0]
        with deadline.deadline(10, label="inner-longer"):
            assert deadline.active()[0] == outer_at        # kept the stricter bound
            assert deadline.active()[2] == "outer"
        with deadline.deadline(0.01, label="inner-shorter"):
            assert deadline.active()[0] < outer_at         # tightening is allowed
            assert deadline.active()[2] == "inner-shorter"
        assert deadline.active()[0] == outer_at            # and restored after


def test_a_deadline_does_not_leak_to_another_thread():
    """Web fits share one process across pool threads: a worker that runs out of
    time must not take the clock away from its neighbours."""
    seen = {}

    def other():
        seen["active"] = deadline.active()

    with deadline.deadline(5):
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert seen["active"] is None


def test_remaining_counts_down_and_goes_negative():
    with deadline.deadline(0.05):
        assert 0 < deadline.remaining() <= 0.05 + 1e-6   # a coarse clock (Windows) may not tick
        time.sleep(0.08)
        assert deadline.remaining() < 0


# ---------------------------------------------------------------------------
# The propagator actually honours it
# ---------------------------------------------------------------------------
# Sized against a MEASURED warm sweep, not a guess: 60 yr / 4000 pts is 0.23 s and
# a 0.25 s budget raced it (the first draft of this file flaked exactly there once
# the ephemeris was warm). The margin below keeps that rule on ANY machine.
#
# It has to be measured at RUN time, not written down: a hand-set 0.3 s encoded
# "this sweep takes ~2.6 s", which is a property of the box the constant was
# written on. On a machine ~9x faster the same sweep is 0.29 s, the margin is 1.0x
# instead of 8x, and the test fails with DID NOT RAISE -- intermittently, because
# under `-n 4` the workers contend and the sweep slows back past the budget, so it
# passes in parallel and fails when run alone. A timing test that depends on how
# loaded the box is reports the box, not the code.
_CUT_YEARS, _CUT_N = 200.0, 20000
_CUT_MARGIN = 8               # the sweep must outlast its budget by this much
_CUT_BUDGET_CAP = 0.3         # ...but never wait longer than the original constant
_MEASURED = None


def _sweep(years=_CUT_YEARS, n=_CUT_N):
    """A propagation long enough to outlast the budget with room to spare."""
    t0 = 2451545.0
    times = np.linspace(t0, t0 + years * 365.25, n)
    state = np.array([1.5, 0.2, 0.05, -0.004, 0.011, 0.001])
    robs = [np.array([0.0, 0.0, 0.0])] * n
    return force_assist.predict_radec_batch(None, t0, state, times, robs)


def _cut_budget():
    """A budget this machine's full sweep is certain to outrun, measured once.

    Warms the ephemeris first (a cold sweep maps DE440 and would time the mapping,
    not the propagation), then times one unbounded sweep. Capped, so a slow or
    loaded box cannot make the bounded tests sit for longer than the hand-set
    value ever asked for -- there the sweep is slow and the cap is already well
    inside the margin.
    """
    global _MEASURED
    if _MEASURED is None:
        _sweep(years=5.0, n=400)                      # warm
        t0 = time.perf_counter()
        _sweep()
        _MEASURED = time.perf_counter() - t0
    return min(_CUT_BUDGET_CAP, _MEASURED / _CUT_MARGIN)


@pytest.mark.slow
def test_a_running_propagation_is_cut_by_the_deadline():
    """The whole point: work already running on a thread stops, because the
    propagator's inner loop agrees to stop. Python cannot kill it from outside."""
    budget = _cut_budget()                            # warms, and measures the sweep
    t0 = time.monotonic()
    with pytest.raises(deadline.DeadlineExceeded):
        with deadline.deadline(budget, label="this fit"):
            _sweep()
    cut_after = time.monotonic() - t0
    # Cut PROMPTLY: the check is per observation, so this must land near the budget
    # and nowhere near the _MEASURED seconds the sweep would have taken. Relative,
    # because "near" is only meaningful against this machine's own sweep.
    assert cut_after < _MEASURED / 2, (
        f"took {cut_after:.3f}s to honour a {budget:.3f}s budget "
        f"(the full sweep is {_MEASURED:.3f}s)")


@pytest.mark.slow
def test_the_same_propagation_completes_when_unbounded():
    """The control: the sweep the deadline cut is otherwise a normal one, so the
    test above is measuring the bound and not a broken propagation."""
    assert deadline.active() is None
    out = _sweep()
    assert out.shape == (_CUT_N, 2)
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_an_expired_deadline_does_not_poison_later_work():
    """A cut fit unwinds mid-integration; the next one must be unaffected. Nothing
    shared is half-written (the Sun memo is append-only, ephemerides read-only)."""
    budget = _cut_budget()                            # warms, and measures the sweep
    with pytest.raises(deadline.DeadlineExceeded):
        with deadline.deadline(budget):
            _sweep()
    out = _sweep()                                    # unbounded again
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Cost when nothing is bounded -- the CLI and the suite run this way
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_seam_costs_nothing_measurable_when_unbounded():
    """One ContextVar read per sweep, and nothing per observation. Guarded because
    a per-observation lookup here would tax every fit in the project."""
    _sweep(years=60.0, n=4000)                        # warm
    t0 = time.perf_counter()
    for _ in range(3):
        _sweep(years=60.0, n=4000)
    unbounded = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(3):
        with deadline.deadline(3600):                 # generous: never fires
            _sweep(years=60.0, n=4000)
    bounded = time.perf_counter() - t0
    # The bounded run does the extra per-observation compare; it must be noise.
    assert bounded < unbounded * 1.5 + 0.5, (
        f"deadline seam cost too much: {unbounded:.3f}s -> {bounded:.3f}s")


# ---------------------------------------------------------------------------
# Only the web fit pool sets one
# ---------------------------------------------------------------------------
def _fitqueue():
    import sys
    from pyroorb.web import fitqueue
    return fitqueue


def test_the_pool_ships_with_a_bound():
    assert _fitqueue().FIT_MAX_SEC == pytest.approx(180)


def test_the_fit_pool_bounds_its_work_and_says_so_plainly(monkeypatch):
    """The POOL sets the deadline -- the work does not opt in. Patched down to a
    tiny budget because the shipped one is three minutes."""
    fitqueue = _fitqueue()
    monkeypatch.setattr(fitqueue, "FIT_MAX_SEC", 0.05)

    def slow_work():
        time.sleep(0.2)
        deadline.check()          # what the propagator does on every observation
        return {"ok": True, "never": "reached"}

    out = fitqueue.run_bounded(slow_work, 5.0, label="test")
    assert out.get("ok") is False
    assert "never" not in out
    assert out.get("timed_out") is True
    assert "too slow to fit interactively" in out["error"]
    assert "command line" in out["error"]
    json.dumps(out)                                   # the client has to read it


def test_an_itf_search_thread_never_inherits_the_fit_deadline():
    """The search has its OWN budget (itf.search.SearchBudget: a no-progress clock
    plus an absolute ceiling). It runs on its own thread, so a fit's deadline can
    never reach it -- this pins that, since the two bounds must not compound."""
    seen = {}

    def search_thread():
        seen["active"] = deadline.active()

    with deadline.deadline(180, label="this fit"):
        t = threading.Thread(target=search_thread, name="itf-search-x")
        t.start()
        t.join()
    assert seen["active"] is None


def test_a_zero_budget_leaves_the_pool_unbounded(monkeypatch):
    """PYROORB_FIT_MAX_SEC=0 is the way back to the old behaviour, and the way a
    box that legitimately needs a long fit can have one."""
    fitqueue = _fitqueue()
    monkeypatch.setattr(fitqueue, "FIT_MAX_SEC", 0)

    def work():
        time.sleep(0.1)
        assert deadline.active() is None
        deadline.check()
        return {"ok": True, "finished": True}

    out = fitqueue.run_bounded(work, 5.0, label="test")
    assert out.get("ok") is True and out.get("finished") is True


def test_only_the_web_fit_pool_sets_a_deadline():
    """WEB UI ONLY, pinned structurally.

    The CLI, `--batch` (which has its own subprocess `--batch-timeout`) and the
    regression suite must stay unbounded: a legitimate command-line fit can take
    many minutes (10P was 919 s before the published-orbit seed), and a bound
    there would turn a slow answer into no answer. The engine modules only READ
    the ContextVar; setting one is the web pool's job alone.
    """
    import pathlib
    import re

    pkg = pathlib.Path(__file__).resolve().parent.parent / "src" / "pyroorb"
    setters = []
    for path in pkg.rglob("*.py"):
        rel = path.relative_to(pkg).as_posix()
        if rel.startswith("tests/") or rel == "deadline.py":
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):        # pragma: no cover
            continue
        if re.search(r"\bdeadline\.deadline\s*\(", src):
            setters.append(rel)
    # ...plus the ITF search's per-candidate TRIAL fit (itf/fitter.py), which
    # is not a user's fit but one audition inside a search that has its own
    # budget -- see ItfConfig.trial_fit_max_sec.
    assert sorted(setters) == ["itf/fitter.py", "web/fitqueue.py"], (
        f"a deadline is set outside the web fit pool: {setters}")


def test_the_timeout_is_not_an_exception_and_no_broad_handler_can_eat_it():
    """A deadline is a control-flow signal, not a failure to absorb.

    The engine is full of defensive `except Exception` handlers whose job is to
    let an optional stage fail without losing the fit -- nongrav discovery, the
    radar refine, `autodetect_center`, an IOD candidate's scoring. Every one of
    them is right about a numerical blow-up and wrong about a timeout: as an
    `Exception` the deadline was caught by whichever stage happened to be
    running, and the fit then carried on through every remaining stage with no
    time left, returning a normal-looking orbit silently missing whatever was
    interrupted. Deriving from `BaseException` (the `KeyboardInterrupt`
    reasoning) makes that structural rather than a re-raise remembered at each
    of dozens of sites.
    """
    assert not issubclass(deadline.DeadlineExceeded, Exception)
    assert issubclass(deadline.DeadlineExceeded, BaseException)

    caught = []
    try:
        try:
            raise deadline.DeadlineExceeded("this fit", 1.0, 2.0)
        except Exception as ex:                      # the engine's own idiom
            caught.append(ex)
    except deadline.DeadlineExceeded:
        pass
    assert caught == [], "a broad handler swallowed the deadline"


def test_the_engine_has_no_handler_that_could_swallow_a_deadline():
    """The guarantee above holds only while nothing catches BaseException.

    A bare `except:` or an `except BaseException` anywhere the propagator can
    be reached from would put the swallow back, and would do it silently.
    """
    import pathlib
    import re

    pkg = pathlib.Path(__file__).resolve().parent.parent / "src" / "pyroorb"
    offenders = []
    for path in pkg.rglob("*.py"):
        rel = path.relative_to(pkg).as_posix()
        if rel.startswith("tests/"):
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):        # pragma: no cover
            continue
        for n, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            # An actual clause at the start of a line -- not prose naming one,
            # which several docstrings (including deadline.py's own) do.
            if re.match(r"\s*except\s*(:|BaseException\b)", code):
                offenders.append(f"{rel}:{n}")
    assert offenders == [], (
        "these handlers can swallow a DeadlineExceeded: " + ", ".join(offenders))


def test_every_interactive_endpoint_runs_through_the_bounded_pool():
    """The pool bounds only what goes THROUGH it.

    Six interactive endpoints used to call `engine_api` straight from the
    request thread, outside both the concurrency cap and this module's bound --
    and a Multi-step is a full re-solve, `nongrav_sigmas` up to 16 refits. A
    route that skips admission makes the pool decorative, which is the argument
    `fitqueue.run_bounded`'s own docstring makes about the paste endpoints.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "src" / "pyroorb" / "web" / "server.py").read_text(encoding="utf-8")
    table = src.split("_JSON_POSTS = {", 1)[1].split("\n    }", 1)[0]
    # Each "/api/x": lambda d: <body> entry, with the body up to the next route.
    entries = re.split(r'\n\s*"(/api/[a-z_]+)":', "\n" + table)
    routes = dict(zip(entries[1::2], entries[2::2]))
    for path in ("/api/fullstep", "/api/nongrav", "/api/nongrav_sigmas",
                 "/api/constrain", "/api/asteroids", "/api/set_epoch",
                 "/api/reauto"):
        assert path in routes, f"{path} is no longer a JSON POST route"
        assert "_pooled(" in routes[path] or "run_bounded(" in routes[path], (
            f"{path} runs engine work on the request thread, outside the pool "
            f"and outside the {'%s' % 'FIT_MAX_SEC'} bound")


# ---------------------------------------------------------------------------
# The cancel token: the web UI's "Stop" on a running ITF search
# ---------------------------------------------------------------------------
def test_a_cancel_token_stops_work_at_the_next_check():
    ev = threading.Event()
    with deadline.cancellable(ev, label="ITF search"):
        deadline.check()                   # not set yet
        deadline.check_cancelled()
        assert deadline.remaining() is None, "a cancel token is not a clock"
        ev.set()
        with pytest.raises(deadline.Cancelled) as got:
            deadline.check_cancelled()
        with pytest.raises(deadline.Cancelled):
            deadline.check()
    assert "ITF search" in str(got.value)
    assert deadline.active() is None
    deadline.check()                       # released on the way out


def test_a_nested_trial_fit_deadline_neither_hides_nor_absorbs_a_cancel():
    """The ITF search sets a per-trial-fit deadline and catches DeadlineExceeded
    BY NAME to score one trial as failed.  A cancel raised inside that fit must
    survive both: the nested budget keeps the token, and the timeout's handler
    does not catch the cancel."""
    assert not issubclass(deadline.Cancelled, deadline.DeadlineExceeded)
    assert not issubclass(deadline.Cancelled, Exception)
    ev = threading.Event()
    with deadline.cancellable(ev):
        with pytest.raises(deadline.Cancelled):
            try:
                with deadline.deadline(30, label="ITF trial fit"):
                    ev.set()
                    st = deadline.active()
                    assert deadline.expired(st)
                    deadline.raise_for(st)         # what the propagator does
            except deadline.DeadlineExceeded:      # itf/fitter.py's handler
                pytest.fail("the trial-fit handler absorbed the cancel")


def test_a_cancel_token_keeps_an_outer_deadline():
    with deadline.deadline(0.01, label="outer"):
        with deadline.cancellable(threading.Event()):
            time.sleep(0.03)
            with pytest.raises(deadline.DeadlineExceeded):
                deadline.check()
            deadline.check_cancelled()     # only the token; the clock is not its job


def test_check_cancelled_ignores_an_expired_clock():
    """Database-sweep loops call check_cancelled: a fit's deadline is not theirs."""
    with deadline.deadline(0.01):
        time.sleep(0.03)
        deadline.check_cancelled()


@pytest.mark.slow
def test_a_running_propagation_is_stopped_by_a_cancel():
    """The same seam as the deadline: a cancel set from ANOTHER thread stops a
    propagation already running."""
    budget = _cut_budget()                            # warms, and measures the sweep
    ev = threading.Event()
    timer = threading.Timer(budget, ev.set)
    t0 = time.monotonic()
    timer.start()
    try:
        with pytest.raises(deadline.Cancelled):
            with deadline.cancellable(ev):
                _sweep()
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < _MEASURED / 2
