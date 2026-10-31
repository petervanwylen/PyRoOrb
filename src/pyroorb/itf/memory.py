"""How much memory is actually left -- correctly, inside a container.

The ITF sweep has a memory floor (``ItfConfig.sweep_min_free_mb``) whose whole
job is to stop serving rows before the OOM killer ends the search with no
verdict and no traceback.  That floor asks this module how much room is left.

The question has two answers and the floor needs the SMALLER of them:

  * the HOST's ``MemAvailable`` from ``/proc/meminfo`` -- free memory plus the
    page cache the kernel would reclaim under pressure; and
  * this process's own CGROUP headroom, which is what a container is actually
    allowed to use.

Reading only the first is why the floor was inert on a hosted deploy.  Inside a
container ``/proc/meminfo`` is the HOST's -- the kernel does not virtualise it
(absent lxcfs) -- so a 512 MB container running on a 32 GB machine reads tens of
gigabytes free, never trips a 1000 MB floor, and dies to the cgroup OOM killer
at 512 MB instead.  Locally, where the two answers coincide, the same guard
works perfectly, which is exactly why this failed only in production.

Cgroup usage is counted the way container tooling counts a "working set":
``current - inactive_file``.  A cgroup's ``memory.current`` includes the page
cache it has touched, and that cache is reclaimable rather than spent -- charging
it as used would have the floor fire on a container that has merely READ a lot
of files (the ephemeris and debias tables, say), truncating sweeps that had
plenty of room.  Subtracting the inactive file pages is the same reclaimability
heuristic ``MemAvailable`` itself applies on the host side.

Everything degrades to None, never to an exception: on a non-Linux box, on a
kernel without cgroups, or on a cgroup with no limit set, the floor simply does
not fire -- which is the behaviour that shipped before this module existed.
"""

import os

_V2_ROOT = "/sys/fs/cgroup"
_V1_ROOT = "/sys/fs/cgroup/memory"


def _read_int(path):
    """First whitespace-separated token of ``path`` as an int, or None -- for
    the unreadable file, the missing one, and cgroup v2's literal "max"."""
    try:
        with open(path) as fh:
            return int(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _stat_field(path, key):
    """``key``'s value from a cgroup ``memory.stat``-style file, or None."""
    try:
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == key:
                    return int(parts[1])
    except (OSError, ValueError):
        pass
    return None


def meminfo_available_mb(root="/"):
    """Host ``MemAvailable`` in MB, or None where /proc/meminfo is absent."""
    try:
        with open(os.path.join(root, "proc/meminfo")) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _own_cgroup_paths(root, controller):
    """Every cgroup directory this process belongs to, leaf first, from
    ``/proc/self/cgroup`` -- the leaf and each of its ancestors up to the mount
    root.  The limit is not always at the mount root: a runtime that puts the
    process in a sub-cgroup writes "max" there and the real cap one or more
    levels down, so checking only the root reads as unlimited."""
    rel = None
    try:
        with open(os.path.join(root, "proc/self/cgroup")) as fh:
            for line in fh:
                hier, _, rest = line.strip().partition(":")
                ctrls, _, path = rest.partition(":")
                if (controller == "" and hier == "0") or \
                        (controller and controller in ctrls.split(",")):
                    rel = path
                    break
    except OSError:
        pass
    out = []
    if rel and rel != "/":
        parts = [p for p in rel.strip("/").split("/") if p]
        while parts:
            out.append(os.path.join(*parts))
            parts.pop()
    out.append("")                      # the mount root itself, last
    return out


# Resolved once per process for the real root: which cgroup levels bound this
# process and what their limits are.  Neither can change for a running process
# (a level's limit is written by the runtime at start-up), while usage changes
# constantly -- and this is called from inside the sweep loop, once per window,
# where walking /proc/self/cgroup and stat-ing both hierarchies every time is
# ~30 opens to re-learn something already known.  Only root="/" is cached: the
# re-based root that tests use stays uncached, so a test can rewrite its fake
# /sys between calls and see the change.
_LEVELS_CACHE = None


def _cgroup_levels(root="/"):
    """``[(base_dir, limit_bytes, usage_file, inactive_key)]`` for every cgroup
    level that actually bounds this process, tightest-first order irrelevant --
    the caller mins over them.  Empty when there is no cgroup or no limit.

    Tries cgroup v2 (``memory.max`` / ``memory.current`` / ``memory.stat``'s
    ``inactive_file``) and falls back to v1's ``memory.limit_in_bytes`` /
    ``memory.usage_in_bytes`` / ``total_inactive_file``.  A v1 "limit" of
    effectively-infinity (the kernel writes a huge sentinel when unset) reads as
    no limit, the same as v2's "max".  Every level this process sits under is
    kept -- a nested cgroup is bounded by its own limit and by every
    ancestor's."""
    global _LEVELS_CACHE
    if root == "/" and _LEVELS_CACHE is not None:
        return _LEVELS_CACHE
    levels = []
    for mount, ctrl, lim, use, inactive in (
            (_V2_ROOT, "", "memory.max", "memory.current", "inactive_file"),
            (_V1_ROOT, "memory", "memory.limit_in_bytes",
             "memory.usage_in_bytes", "total_inactive_file")):
        mount = os.path.join(root, mount.lstrip("/"))
        for rel in _own_cgroup_paths(root, ctrl):
            base = os.path.join(mount, rel) if rel else mount
            limit = _read_int(os.path.join(base, lim))
            if limit is None or _read_int(os.path.join(base, use)) is None:
                continue
            # v1 with no limit set writes a sentinel near 2**63; anything at or
            # above the machine's plausible RAM is "unlimited" either way.
            if limit >= (1 << 62):
                continue
            levels.append((base, limit, use, inactive))
        if levels:
            break                       # v2 answered: do not also read v1
    if root == "/":
        _LEVELS_CACHE = levels
    return levels


def cgroup_available_mb(root="/"):
    """This process's cgroup headroom in MB (limit minus working set), or None
    when there is no cgroup, no limit, or the files are unreadable.  The
    TIGHTEST level wins.

    Note this is a STRICTER reckoning than the host's ``MemAvailable``, which
    also counts about half of the ACTIVE file pages as available; here only
    ``inactive_file`` is treated as reclaimable.  A process holding a large
    mmapped working set of file pages (this engine mmaps ~700 MB of ephemeris)
    therefore reads lower headroom than the same process would on the host
    side.  That is deliberately the conservative direction, but it does mean
    a floor calibrated against ``MemAvailable`` bites sooner here -- see
    :func:`breakdown`, which exists so that gap can be measured rather than
    guessed at."""
    best = None
    for base, limit, use, inactive in _cgroup_levels(root):
        usage = _read_int(os.path.join(base, use))
        if usage is None:
            continue
        cache = _stat_field(os.path.join(base, "memory.stat"), inactive) or 0
        # the working set is what cannot be reclaimed: usage minus the file
        # pages the kernel would drop before killing anything
        working = max(0, usage - cache)
        headroom = max(0, (limit - working) // (1024 * 1024))
        best = headroom if best is None else min(best, headroom)
    return best


def breakdown(root="/"):
    """The tightest cgroup level's numbers in MB, or None when there is no
    limit to report: ``{"limit", "usage", "inactive_file", "active_file",
    "headroom"}``.

    Diagnostic only -- nothing decides on it.  It exists because
    :func:`cgroup_available_mb` charges ``active_file`` as used while
    ``MemAvailable`` would not, and nobody has measured how large that term is
    on a real deploy that has just read the ephemeris and debias tables.  If it
    turns out to be a large share of the sweep floor, the floor is firing on
    memory the kernel would have handed back."""
    best = None
    for base, limit, use, inactive in _cgroup_levels(root):
        usage = _read_int(os.path.join(base, use))
        if usage is None:
            continue
        stat = os.path.join(base, "memory.stat")
        inact = _stat_field(stat, inactive) or 0
        # v1 spells the active counterpart "total_active_file"
        act = _stat_field(stat, inactive.replace("inactive", "active")) or 0
        mb = 1024 * 1024
        row = {"limit": limit // mb, "usage": usage // mb,
               "inactive_file": inact // mb, "active_file": act // mb,
               "headroom": max(0, (limit - max(0, usage - inact)) // mb)}
        if best is None or row["headroom"] < best["headroom"]:
            best = row
    return best


def available_mb(root="/"):
    """Memory available to THIS process in MB -- the smaller of the host's
    ``MemAvailable`` and the cgroup headroom -- or None when neither can be
    read (non-Linux, no cgroup limit: the caller's floor never fires).

    ``root`` re-bases every path for testing."""
    vals = [v for v in (meminfo_available_mb(root), cgroup_available_mb(root))
            if v is not None]
    return min(vals) if vals else None


def limit_mb(root="/"):
    """The tightest cgroup memory LIMIT on this process in MB, or None when it
    is unlimited (or unreadable).  Not headroom -- the ceiling itself."""
    best = None
    for _base, limit, _use, _inactive in _cgroup_levels(root):
        mb = limit // (1024 * 1024)
        best = mb if best is None else min(best, mb)
    return best


# ---------------------------------------------------------------------------
# What a search costs, measured: every number below comes from running the 24
# live ITF recovery fixtures with a sampler attached.
# They are here rather than in ItfConfig because they are FACTS ABOUT THE
# WORKLOAD, not knobs: changing one does not change behaviour, it makes the
# guards wrong.
# ---------------------------------------------------------------------------

# The process before a search starts: ephemeris + debias tables, already resident.
BASELINE_MB = 180
# Growth over that baseline, per search.  Bimodal -- 19 of 24 fixtures sit in a
# tight band (the median) and five tail objects run the LOV variant scan, whose
# union sweep caches every window it is served (the p90 and the max).
SEARCH_MEDIAN_MB = 190          # measured p50 189
SEARCH_P90_MB = 570             # measured p90 563
SEARCH_MAX_MB = 1060            # measured max, 2022 GL24
# The most this process can allocate BETWEEN two floor checks.  This -- not the
# total cost -- is what the floor must beat: a floor below the burst fires only
# after the allocation that already killed the process.  Measured identically
# (357 / 358 MB) on an idle and a loaded box, so it is a property of the search
# and not of the machine.
MAX_BURST_MB = 360

# The floor may never scale below this: just above MAX_BURST_MB, so the valve
# still has room to stop the sweep before the next allocation lands.  A pure
# fraction fails exactly where it is needed -- 0.3 x 1 GB is 307 MB, UNDER the
# burst.
FLOOR_MIN_MB = 400

# What fraction of a CONTAINER's whole limit may be reserved as the sweep
# floor.  The floor is a reservation held back from the search, so on a small
# container an absolute figure is not conservative, it is fatal: with the
# configured 1000 MB and a 1 GB container, `available` starts BELOW the floor
# and every search truncates on its first window -- returning "no match" from a
# corridor it never looked at.  Before the floor was made container-aware it
# could not fire in production at all; afterwards, unscaled, it can fire always.
# Both are the same bug -- an absolute number applied to a relative quantity.
#
# 0.3 leaves the search 70% of the box.  Measured over the recovery
# fixtures: the median search grows 189 MB and the p90 grows
# 563 MB, so a 2 GB container reserves 614 MB and still runs the p90; a 4 GB
# one reaches the 1000 MB cap and behaves exactly as it does today.
FLOOR_LIMIT_FRACTION = 0.3


def sweep_floor_mb(configured, root="/"):
    """The floor to actually enforce, given what the config asks for.

    ``min(configured, max(FLOOR_MIN_MB, FLOOR_LIMIT_FRACTION * limit))`` inside
    a container, ``configured`` unchanged outside one -- so every laptop and CI
    runner behaves exactly as before, and only a container small enough for the
    absolute number to be unpayable scales down.  ``0`` in, ``0`` out:
    disabling the floor stays disabled.

    Both halves are load-bearing.  Without the fraction, a 1 GB container
    starts BELOW a 1000 MB floor and truncates every search on its first
    window -- answering "no match" from a corridor it never queried.  Without
    the FLOOR_MIN_MB clamp, the fraction hands a small container a floor under
    the allocation burst, which is a guard that cannot guard.

    The clamp can exceed the container: 400 MB of a 512 MB box leaves nothing
    to search with.  That is the honest answer -- see :func:`search_capacity`,
    which says so rather than pretending -- and it is still the safe one, since
    the alternative is the OOM killer."""
    if not configured:
        return 0
    lim = limit_mb(root)
    if lim is None:
        return int(configured)
    scaled = max(FLOOR_MIN_MB, int(FLOOR_LIMIT_FRACTION * lim))
    return max(1, min(int(configured), scaled))


# Verdicts from :func:`search_capacity`.
CAPACITY_OK = "ok"                   # room for a deep (variant-scan) search
CAPACITY_SHALLOW = "shallow"         # room for a typical one; deep ones will truncate
CAPACITY_UNUSABLE = "unusable"       # not enough room to sweep anything worth having


def search_capacity(configured_floor, root="/"):
    """Can this box run an ITF search right now, and how deep?

    Returns ``None`` when memory cannot be measured at all (the guards are
    inert there and pretending otherwise would be worse than silence), else::

        {"verdict": CAPACITY_OK | CAPACITY_SHALLOW | CAPACITY_UNUSABLE,
         "available_mb", "floor_mb", "headroom_mb", "limit_mb", "reason"}

    ``headroom_mb`` is what a search may actually consume: available memory
    minus the floor it is not allowed to eat into.  Compared against the
    measured cost of a search, that answers a question nobody could ask before
    -- whether starting this search is worth the minutes it will take.  A user
    told "no match" by a sweep that never had room to run has been told
    something false in effect, and the remedy is to not start it."""
    avail = available_mb(root)
    if avail is None:
        return None
    floor = sweep_floor_mb(configured_floor, root)
    headroom = max(0, avail - floor)
    if headroom < SEARCH_MEDIAN_MB:
        verdict = CAPACITY_UNUSABLE
        reason = (f"only {headroom} MB is free above the {floor} MB safety "
                  f"floor, and a typical ITF search needs about "
                  f"{SEARCH_MEDIAN_MB} MB -- it would stop almost immediately "
                  f"and report nothing useful")
    elif headroom < SEARCH_P90_MB:
        verdict = CAPACITY_SHALLOW
        reason = (f"{headroom} MB is free above the {floor} MB safety floor: "
                  f"enough for a routine search, but a deep one (a weakly "
                  f"constrained orbit, which fans out to many trial "
                  f"corridors and can need {SEARCH_P90_MB} MB) will be cut "
                  f"short")
    else:
        verdict = CAPACITY_OK
        reason = f"{headroom} MB free above the {floor} MB safety floor"
    return {"verdict": verdict, "available_mb": avail, "floor_mb": floor,
            "headroom_mb": headroom, "limit_mb": limit_mb(root),
            "reason": reason}
