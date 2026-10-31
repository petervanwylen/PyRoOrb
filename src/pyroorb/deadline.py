"""A wall-clock bound on ONE unit of engine work, enforced inside the propagator.

Why this exists: a web fit had no bound at all.  ``fitqueue.wait`` bounds how long
the CALLER waits, not how long the work runs, and Python cannot kill a thread -- so
a request for "4 vesta" (two designations, which disqualifies the published-orbit
fast path, over a 204-year 8275-observation arc) held one of seven pool workers for
twelve hours, and only a process restart got it back.  The ITF search's own budget
says as much in its own docstring: its checks sit BETWEEN fits, so "a single
pathological fit needs per-fit bounding".  This is that bounding.  The search keeps
its own budget (``itf.search.SearchBudget``) and is not touched by this.

It is COOPERATIVE, because it has to be: the only way to stop work already running
on a thread is for the work to agree to stop.  The agreement point is the
propagator's inner loop (``force_assist._integrate_split`` for ASSIST,
``forward_model.predict_radec_batch`` for the rebound fallback), which every fit,
arc extension, rejection round, non-grav trial and interactive step passes through
many times a second.  Nothing else needs to know about any of this: raise from
there and the whole cascade unwinds to whoever set the deadline.

Unwinding mid-fit is safe because nothing SHARED is left half-written -- the
Sun-barycentre memo is append-only, the ephemerides are read-only once built, and
an Evaluator's memo slots die with the abandoned fit.

A ContextVar, not a global: web fits share one process across pool threads, and a
thread that runs out of time must not take the clock away from its neighbours.
(The same reasoning as ``force_assist.perturbers_off_for``.)  Threads do not
inherit one another's context, so a deadline set on a pool worker binds that
worker's fit and nothing else -- which is also why an ITF search, running on its
own thread, never sees one.

The same seam carries a CANCEL token (:func:`cancellable`): the web UI's "Stop"
on a running ITF search sets a ``threading.Event`` and the search unwinds at its
next propagator step with :class:`Cancelled` -- a different exception, so the
per-trial-fit deadline handlers inside the search cannot absorb it.  Loops outside
any fit (database sweeps, the search's between-fit budget check) call
:func:`check_cancelled`.

**Off by default, and free when off.**  The CLI, the batch tool and the regression
suite set no deadline; the propagator reads the ContextVar ONCE per sweep and then
branches on a local, so an unbounded run pays one context lookup per sweep and
nothing per observation.
"""
import contextlib
import contextvars
import time


class DeadlineExceeded(BaseException):
    """Raised inside the propagator when the current unit of work is out of time.

    Carries the budget and the label so the surface that set the deadline can say
    what was abandoned and after how long, rather than reporting a bare timeout.

    **Not an ``Exception``**, deliberately, and for the same reason
    ``KeyboardInterrupt`` is not: this is a control-flow signal that must reach
    the scope that set the deadline, and the engine is FULL of defensive
    ``except Exception`` handlers whose job is to let an optional stage fail
    without losing the fit.  Every one of them is right about a numerical
    blow-up and wrong about this -- as an ``Exception`` the timeout was caught
    by whichever optional stage happened to be running (nongrav discovery,
    radar refinement, ``autodetect_center``, an IOD candidate's scoring, an
    extension's unstall), logged only under ``verbose``, and then the fit
    CARRIED ON through every remaining stage with no time left.  A fit could
    therefore blow far past ``FIT_MAX_SEC`` and still return a normal-looking
    ``FittedOrbit`` silently missing whatever the timeout interrupted, which is
    the exact failure the bound was introduced to prevent.

    Fixing that per-site would mean an ``except DeadlineExceeded: raise`` ahead
    of every broad handler in the engine, forever, with a silent regression
    each time a new one is added and the pair forgotten.  Deriving from
    ``BaseException`` makes the property structural: no ``except Exception``
    can swallow it, so the only handlers that see it are the ones that name it
    (``fitqueue._work_once``, which turns it into the caller's answer).
    Audited when this changed: the engine has no bare ``except:`` and no
    ``except BaseException`` outside the tests.
    """

    def __init__(self, label, budget_sec, elapsed_sec):
        self.label = label
        self.budget_sec = budget_sec
        self.elapsed_sec = elapsed_sec
        super().__init__(
            f"{label} exceeded its {budget_sec:g} s budget (ran {elapsed_sec:.1f} s)")


class Cancelled(BaseException):
    """Raised inside the engine when the current unit of work was CANCELLED by
    whoever started it (the web UI's "Stop search" button), as opposed to having
    run out of time.

    A separate class from :class:`DeadlineExceeded`, not a subclass of it: the ITF
    search sets a per-trial-fit deadline and catches ``DeadlineExceeded`` BY NAME
    around each fit to score that one trial as failed and move on.  A cancel caught
    there would cost the search one trial and nothing more.  A cancel has to unwind
    the whole search, so the only handler that may name it is the one that armed it.
    ``BaseException`` for the reason given on :class:`DeadlineExceeded`.
    """

    def __init__(self, label):
        self.label = label
        super().__init__(f"{label} was cancelled")


# (expiry_monotonic, budget_sec, label, cancel_event), or None when unbounded.
# The expiry is +inf when only a cancel token is armed; cancel_event is None when
# only a wall-clock bound is.
_DEADLINE = contextvars.ContextVar("pyroorb_deadline", default=None)
_INF = float("inf")


@contextlib.contextmanager
def deadline(seconds, label="fit"):
    """Bound the work in this block to `seconds` of wall clock.

    A falsy or non-positive `seconds` is NO deadline, so a caller that may or may
    not want one does not have to special-case the unbounded path -- which is what
    keeps the CLI and the suite clear of this entirely.

    Nested deadlines take the EARLIER expiry: an inner block may tighten the bound
    but never extend past the one it runs inside, or a nested budget would be a way
    to escape the outer one.  A cancel token armed outside is carried through
    either way: a nested budget must not hide a cancel either.
    """
    if not seconds or seconds <= 0:
        yield
        return
    at = time.monotonic() + float(seconds)
    budget, lab, ev = float(seconds), label, None
    outer = _DEADLINE.get()
    if outer is not None:
        ev = outer[3]
        if outer[0] < at:
            at, budget, lab = outer[0], outer[1], outer[2]   # the outer bound is stricter
    token = _DEADLINE.set((at, budget, lab, ev))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


@contextlib.contextmanager
def cancellable(event, label="search"):
    """Let the work in this block be stopped from another thread by setting
    `event` (a ``threading.Event``).  The work stops at its next agreement point
    -- the propagator's inner loop, or an explicit :func:`check_cancelled` -- by
    raising :class:`Cancelled`, which only the caller that armed it should catch.

    `event` None is no token, like a falsy deadline.  Any wall-clock bound already
    in force is kept.
    """
    if event is None:
        yield
        return
    outer = _DEADLINE.get()
    at, budget, lab = (outer[0], outer[1], outer[2]) if outer is not None else (_INF, 0.0, label)
    token = _DEADLINE.set((at, budget, lab, (event, label)))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def active():
    """The current deadline tuple, or None.

    Read ONCE per sweep and then branched on as a local -- see the module docstring
    on why the inner loop must not do this per observation.
    """
    return _DEADLINE.get()


def expired(state):
    """True when `state` (from :func:`active`) is past its expiry or cancelled.
    Split out so the propagator's inner loop is a null check, one float compare
    and (only when a cancel token is armed) one flag read."""
    return state is not None and (time.monotonic() > state[0]
                                  or (state[3] is not None and state[3][0].is_set()))


def raise_for(state):
    """Raise :class:`Cancelled` or :class:`DeadlineExceeded` for an expired `state`.
    A cancel wins: it is the stronger instruction, and must not be mistaken for a
    timeout that a nested handler would absorb."""
    at, budget, label, cancel = state
    if cancel is not None and cancel[0].is_set():
        raise Cancelled(cancel[1])
    raise DeadlineExceeded(label, budget, budget + (time.monotonic() - at))


def check():
    """Raise if the current deadline has passed or the work was cancelled -- the
    whole thing in one call, for sites that are not hot enough to bother splitting
    it."""
    state = _DEADLINE.get()
    if expired(state):
        raise_for(state)


def check_cancelled():
    """Raise :class:`Cancelled` if the current work was cancelled; ignore the
    wall clock.  For loops that sit OUTSIDE any fit (a database sweep, a search's
    between-fit budget check): a deadline there is not theirs to enforce."""
    state = _DEADLINE.get()
    if state is not None and state[3] is not None and state[3][0].is_set():
        raise Cancelled(state[3][1])


def remaining():
    """Seconds left, or None when unbounded.  Negative once past."""
    state = _DEADLINE.get()
    if state is None or state[0] == _INF:
        return None
    return state[0] - time.monotonic()
