"""How much of THIS container goes to what.

The web server runs two independent pools of expensive work, and they spend the
same two budgets: the fit worker pool (:mod:`fitqueue`) and the live ITF
searches (:mod:`engine_api`).  Sizing them apart is how the deployed ITF cap
came to be the literal ``3`` whose own comment justifies it with "six
concurrent searches is ~3.4 GB ... which is more than the container has" --
true of the ~1 GB box that number was reasoned against, and false of the
7.6 GB one it actually runs on.  The reasoning was sound; the box was not what
it said.  A cap that has to be re-derived by hand every time the deploy moves
is a cap that is wrong after the next move, so both are derived here, from the
container's own cgroup limits, and derived TOGETHER because neither can spend a
core or a megabyte the other has already spent.

Two budgets, and a cap is the smaller of what each allows:

  * MEMORY -- the container's limit, minus the resident process, minus what a
    full fit pool can transiently hold, minus the ITF sweep's own safety floor
    (which is a reservation held back from every search, not spare room).
  * CPU -- the cgroup quota, never ``os.cpu_count()``: on a PaaS box that
    reports the HOST's cores, a number the container is not allowed to use, and
    sizing a pool from it builds a thread pool that only ever thrashes.

Measured on the live 8 vCPU / 7629 MB deploy (2026-09-03, this module's reason
for existing): fits are CPU-bound there and searches are memory-bound, so
neither budget alone gets both right -- memory alone would admit 7 searches
against 7 fit workers on 8 cores, and CPU alone would ignore that one search
can hold 570 MB for a quarter of an hour.

Every function degrades to the old hand-set behaviour when the limits cannot be
read (a laptop, a CI runner, a kernel without cgroups), because a box with no
declared limit is not a box we can size for.
"""

import os

from pyroorb.itf.memory import (            # noqa: E402
    limit_mb, sweep_floor_mb, SEARCH_P90_MB,
)

# --- what the workloads cost -------------------------------------------------
# FACTS ABOUT THE WORKLOAD, not knobs -- changing one does not change behaviour,
# it makes the derived caps wrong.  The search figures live in itf.memory (measured
# over 24 live ITF searches); the fit figures are here because nothing
# else owns them.

# Peak resident growth attributable to ONE in-flight fit, in MB.  A deliberate
# over-estimate: the steady-state cost of a fitted object is small (~1.74 KB per
# observation), but the transient during integration -- ASSIST/rebound buffers,
# the Jacobian batch, the LM working set -- is what decides whether two
# concurrent fits fit in the box, and it scales with the ARC, not the median
# object.  Sized to be wrong in the safe direction: too high costs throughput,
# too low costs the process.
FIT_PEAK_MB = float(os.environ.get("PYROORB_FIT_PEAK_MB") or 250)

# Memory the process needs that is not a fit: the interpreter, the DE440 kernel
# mappings, the 19.5 MB debias grid, obscodes, the resident sessions.
FIT_BASELINE_MB = float(os.environ.get("PYROORB_FIT_BASELINE_MB") or 400)

# The concurrency guard on searches, expressed in cores.  NOT a CPU share, and
# the distinction is measured rather than assumed: on the live 8 vCPU box three
# concurrent searches of the SAME object took 97.6 s each against 81.8 s alone
# (+19%), and running them alongside a saturated 7-worker fit pool left the
# pool's wall time unchanged (130.8 s vs 131.5 s).  A search is overwhelmingly
# I/O -- SBN database round trips -- so it does not hold a core, and sizing this
# cap as a fraction of the CPUs would starve a box that has the memory for more.
#
# What the core count stands in for here is the two things that DO degrade with
# more simultaneous searches, neither of them local CPU: the shared external
# SBN database, which is not ours to saturate, and the tail latency a user
# waits through with no queue position to explain it.
#
# HALF the usable cores, and the half is measured.  The full scaling curve on the
# live 8 vCPU box, same object so every search costs the same:
#
#     1 search    81.8 s      3 searches   97.6 s each   (+19%)
#     5 searches 117.7 s each (+44%)       8 searches   179 s each (+119%)
#
# Throughput peaks and then flattens: 5 concurrent deliver 0.0425 searches/s and 8
# deliver 0.0447 -- 5% more work for 52% more waiting, which is not a trade to make
# on behalf of someone staring at a spinner.  The knee is at 5, and the UI promises
# "typically 30-120 s", a promise 5 keeps (117.7 s) and 8 breaks outright.
#
# So the rail is deliberately at or just under the measured knee rather than above
# it.  MEMORY is still the hard safety bound (see :func:`itf_max_running`); this is
# the one that keeps the site feeling answered, and it also stops a large-memory box
# deriving 23 concurrent sweeps because it arithmetically could.
#
# Calibrated on ONE box.  Halving the cores is the shape, not a law -- a deploy that
# has measured its own knee should set PYROORB_ITF_MAX_RUNNING and skip the estimate.
def _itf_cores_rail(cores):
    return max(_MIN_CAP, cores // 2)

# Never derive a cap of zero: a box too small to be sized for still has to
# answer somebody, and refusing every search on a box that could run one
# sequentially is a worse answer than running it slowly.  `search_capacity`
# already declines honestly when there is genuinely no room.
_MIN_CAP = 1

# What to allow when the container's memory limit cannot be read at all -- a
# laptop, a CI runner, a kernel without cgroups, an uncapped VM.  Memory is the
# budget that bounds searches, so a box whose memory we cannot see is a box we
# cannot size: the honest fallback is the hand-set number this module replaced,
# NOT the core count.  Falling back to the cores would quietly turn an uncapped
# 16-core dev box into sixteen concurrent sweeps with no memory guard behind
# them -- `search_capacity` is inert there too, since it reads the same limits.
ITF_UNSIZED_DEFAULT = 3


def _sweep_floor_configured():
    """The ITF sweep's configured memory floor, from the one place that owns it.

    Read from ``ItfConfig`` rather than duplicated, so the reservation this
    module subtracts and the reservation the sweep actually enforces cannot
    drift apart.  ``itf.config`` imports only ``os`` and ``dataclasses``, so
    this costs nothing; it still degrades to the documented default rather than
    raising, because a capacity calculation must never be the thing that fails.
    """
    try:
        from pyroorb.itf.config import ItfConfig
        return int(ItfConfig().sweep_min_free_mb)
    except Exception:
        return 1000


def cpu_quota():
    """This process's CPU allowance in whole cores, or None when unlimited.

    Reads the cgroup quota rather than ``os.cpu_count()``, which on a container
    reports the HOST's cores -- on Railway that is a large number the container
    is not allowed to use.  ``itf.memory`` does the same job for memory (v2 and
    v1, every level this process belongs to); CPU has no equivalent there, so
    it is read here, once, for both callers.
    """
    try:                                     # cgroup v2: "<quota> <period>" or "max <period>"
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
    except (OSError, ValueError, IndexError):
        pass
    try:                                     # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as fh:
            quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as fh:
            period = int(fh.read().strip())
        if quota > 0 and period > 0:
            return max(1, int(quota / period))
    except (OSError, ValueError):
        pass
    return None


def usable_cores():
    """Cores this process may actually schedule on, cgroup quota first."""
    return cpu_quota() or os.cpu_count() or 2


_POOL_SIZE = None
_ITF_MAX = None


def fit_pool_size():
    """How many fits may run at once on THIS box.

    The minimum of what the CPUs can usefully carry and what the memory limit
    can hold.  Both matter and neither implies the other: a 4-core 1 GB
    container is memory-bound, an 8 GB single-core one is CPU-bound, and sizing
    from either alone gets one of them badly wrong.

    Leaves one core for the request threads, the ITF maintenance scheduler and
    the interpreter itself -- a pool that saturates every core makes the server
    unable to answer the very poll requests that collect its results, which
    reads to a user as the site being down.
    """
    global _POOL_SIZE
    override = os.environ.get("PYROORB_FIT_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if _POOL_SIZE is not None:
        return _POOL_SIZE           # container limits do not change under us, and
                                    # this is read on every submit and every poll
    by_cpu = max(1, usable_cores() - 1)
    lim = limit_mb()
    if not lim:
        _POOL_SIZE = by_cpu
    else:
        by_mem = int((lim - FIT_BASELINE_MB) // FIT_PEAK_MB)
        _POOL_SIZE = max(_MIN_CAP, min(by_cpu, by_mem))
    return _POOL_SIZE


def itf_max_running():
    """How many live ITF searches may run at once on THIS box.

    The smaller of two budgets, for the same reason the fit pool is:

      * MEMORY -- what is left of the container after the resident process, a
        FULL fit pool's transient, and the sweep's own safety floor, divided by
        what a search actually costs.  The p90 is the divisor, not the median:
        the cost is bimodal (a routine search against one that fans out to the
        LOV variant scan), and a cap that only holds for the median is a cap
        that fails exactly when several expensive objects arrive together --
        which is the case it exists for.
      * CONCURRENCY -- half the usable cores, as a rail rather than a budget,
        set at the MEASURED throughput knee (see the note above the constants:
        past it the box does 5% more work for 52% more waiting).  Searches are
        I/O-bound, so this is not a CPU share; it stands in for the shared SBN
        database and the tail latency a user waits through with no queue
        position to explain it.

    Subtracting a full fit pool is deliberately pessimistic: fits and searches
    do not always peak together, so this reserves room that is often idle.  The
    alternative is a cap that is only correct while the site is quiet, and the
    whole point of a cap is the moment it stops being quiet.
    """
    global _ITF_MAX
    override = os.environ.get("PYROORB_ITF_MAX_RUNNING")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if _ITF_MAX is not None:
        return _ITF_MAX
    by_cpu = _itf_cores_rail(usable_cores())
    lim = limit_mb()
    if not lim:
        _ITF_MAX = min(by_cpu, ITF_UNSIZED_DEFAULT)
    else:
        reserved = FIT_BASELINE_MB + fit_pool_size() * FIT_PEAK_MB + \
            sweep_floor_mb(_sweep_floor_configured())
        by_mem = int((lim - reserved) // SEARCH_P90_MB)
        _ITF_MAX = max(_MIN_CAP, min(by_cpu, by_mem))
    return _ITF_MAX


def describe():
    """Every derived cap and the inputs it came from -- for /healthz and
    /memory, so a deploy can be asked what it thinks it is rather than having
    the answer inferred from behaviour under load."""
    lim, cores = limit_mb(), cpu_quota()
    floor = sweep_floor_mb(_sweep_floor_configured()) if lim else None
    return {
        "cpu_quota": cores,
        "usable_cores": usable_cores(),
        "memory_limit_mb": lim,
        "fit_pool_size": fit_pool_size(),
        "itf_max_running": itf_max_running(),
        "sweep_floor_mb": floor,
        "fit_peak_mb": FIT_PEAK_MB,
        "search_p90_mb": SEARCH_P90_MB,
    }


def _reset_for_tests():
    """Drop the memoised caps so a test can re-derive them against a rewritten
    fake cgroup tree.  Never called in production -- the limits it caches
    cannot change for a running process."""
    global _POOL_SIZE, _ITF_MAX
    _POOL_SIZE = _ITF_MAX = None
