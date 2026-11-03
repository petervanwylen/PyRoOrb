"""Where this server's memory actually is -- the data behind ``GET /memory``.

Motivation: the hosted deploy OOMs on concurrent ITF searches, and
every number we had about it was a guess.  Nothing measured what a session costs,
what a finished search payload costs, or how much of the container's ceiling the
long-lived tables occupy before a search allocates anything on top.  This module
answers that from the live process, so effort goes to whatever is actually large
rather than to whatever is easiest to imagine being large.

Three layers, because they answer different questions:

* **Process / container.**  RSS, the peak RSS this process has ever reached
  (``VmHWM`` -- the number the OOM killer reacted to, long after the spike is
  gone), and the same headroom figure the ITF sweep floor consults
  (``itf.memory``).  This is the ground truth; everything else is attribution.
* **The tables.**  Every long-lived structure we could find, deep-walked, with
  per-entry detail -- "which session is the 40 MB" is the question a total cannot
  answer.  Plus the working set of every running thread, which is where an
  in-flight ITF search lives and which no table can show.
* **Misc, and what is left after it.**  Everything else the process can reach,
  grouped by type with its largest single objects named, so nothing hides in a
  remainder.  Tables + misc partition the reachable Python heap; RSS minus that
  is the C side -- the interpreter, NumPy/rebound/ASSIST buffers, allocator
  slack, the memory-mapped ephemerides -- reported as ``native`` because no cache
  on this page can free it.  (``gc.get_objects()`` summed is also reported, but
  only for scale: it sees tracked containers, so it misses the ``str``/``float``
  atoms that are half of an observation record.)

Deep-walked, not estimated.  A per-observation rule of thumb (fits run ~1.7 KB
per observation) is fine for a back-of-envelope and useless for the question this
page exists to answer -- "which of these is the main thing?" -- because the
estimate encodes the belief being tested.  The walk is affordable: ~1.4 M objects
per second, so the worst fit in the regression suite (4356 obs, ~100 k objects)
costs 0.07 s and a full twelve-session table well under a second.  A node budget
bounds the pathological case and says so in the output rather than hanging.

Two accounting rules keep the numbers honest:

* **Gross, plus one overlap line.**  Each table is walked independently (deduped
  within itself), then all tables are walked together with one shared ``seen``
  set.  ``sum(gross) - combined`` is what more than one table holds -- and it is
  real: an ITF search stores its fit in ``_SESSIONS`` *and* its payload in
  ``_ITF_JOBS``, and a pasted object sits in ``_SESSIONS`` and ``fitcache`` at
  once.  Per-table "exclusive" figures would need full reference analysis; a
  single overlap line is cheap and does not pretend.
* **Nothing is imported to measure it.**  A module absent from ``sys.modules``
  is reported as "not loaded", never imported -- importing ASSIST to size its
  ephemeris would allocate hundreds of MB to answer a question about memory.

Read-only and side-effect free.  It takes the same locks the tables' owners take,
copies the containers, and walks the copies, so it cannot wedge a running search;
a structure mutating mid-walk degrades to a skipped subtree, never an exception.
"""
import ctypes
import re as _re_smaps
import gc
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
# Ceiling on objects visited per walk.  Reached only by something pathological
# (a table holding a graph orders of magnitude past anything we ship); the report
# then says "truncated" instead of quietly under-reporting or taking minutes.
_NODE_BUDGET = 4_000_000

# Types that are never counted and never descended into: they belong to the
# interpreter, not to the table that happens to reference one.  Following a
# module or a class would drag in most of the process through __globals__ and
# attribute dicts, and would charge it to whichever table was walked first.
_SKIP = (types.ModuleType, type, types.BuiltinFunctionType,
         types.MethodDescriptorType, types.WrapperDescriptorType,
         types.GetSetDescriptorType, types.MemberDescriptorType)


# An object at or above this size is worth naming individually in the misc
# bucket: one 20 MB array nobody remembered allocating is exactly the finding
# this page is for, and it would otherwise hide inside a "ndarray: 45 MB" row.
_BIG_OBJECT_BYTES = 512 * 1024


def _describe(o):
    """A SHAPE-only description of one object -- never its contents.

    The tables pass answers any caller (by design: reachable without
    publicising it -- only the whole-heap pass is password-gated, and that for
    its COST, not its content), so it reports how big things are, never what
    they say.  A str is a length, an array is a shape and dtype, a container is
    a length.
    """
    t = type(o)
    name = "%s.%s" % (t.__module__, t.__name__) if t.__module__ != "builtins" else t.__name__
    try:
        if hasattr(o, "shape") and hasattr(o, "dtype"):
            return "%s %s %s" % (name, tuple(o.shape), o.dtype)
        if isinstance(o, (str, bytes, bytearray)):
            return "%s len=%d" % (name, len(o))
        if hasattr(o, "__len__"):
            return "%s len=%d" % (name, len(o))
    except Exception:
        pass
    return name


def _children(o):
    """What to traverse from ``o``.  Returns a list (possibly empty).

    Containers yield their contents; instances yield ``__dict__``/``__slots__``.
    Functions yield only their closure cells and defaults -- NOT ``__globals__``,
    which is the module namespace and would attribute the whole interpreter to
    the first table that happens to hold a callback.  The engine's evaluator
    closures capture real per-fit arrays, so the cells genuinely matter.

    Every failure here is a skipped subtree, never an exception.  Two real ones:
    a container mutating underneath us (a live search appending to its trace)
    raises RuntimeError, and merely READING an attribute can run code -- a lazy
    module proxy's ``__getattr__`` imports on touch, which is how a walk of this
    process died with ``ModuleNotFoundError: _gdbm``.  A memory report must not
    be able to fail because of what it looked at.
    """
    try:
        if isinstance(o, dict):
            out = []
            for k, v in list(o.items()):
                out.append(k)
                out.append(v)
            return out
        if isinstance(o, (list, tuple, set, frozenset)):
            return list(o)
        if isinstance(o, types.CellType):
            try:
                return [o.cell_contents]
            except ValueError:          # empty cell
                return []
        if isinstance(o, (types.FunctionType, types.LambdaType)):
            out = list(o.__closure__ or ())
            out.extend(o.__defaults__ or ())
            return out
        if isinstance(o, types.MethodType):
            return [o.__self__, o.__func__]
        if isinstance(o, types.FrameType):
            # A running ITF search's working set (sweep rows, candidate features,
            # trial fits) is held in locals on a worker thread and reachable no
            # other way: no table refers to it. Frames are GC-tracked, so walking
            # f_locals is what makes an in-flight search visible in `misc`.
            try:
                return [dict(o.f_locals)]
            except Exception:
                return []
        out = []
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            out.append(d)
        for slot in getattr(type(o), "__slots__", ()) or ():
            try:
                out.append(getattr(o, slot))
            except Exception:           # a descriptor that computes, or raises
                pass
        # A NumPy view's getsizeof reports only its header; the buffer belongs to
        # the base array, which must be reached or a view-heavy table reads as free.
        if type(o).__module__ == "numpy":
            base = getattr(o, "base", None)
            if base is not None:
                out.append(base)
        return out
    except Exception:
        return []


def walk(roots, seen=None, budget=_NODE_BUDGET, tally=None, big=None,
         big_min=_BIG_OBJECT_BYTES):
    """Deep size of everything reachable from ``roots``.

    ``seen`` is a set of ids already charged elsewhere -- pass a shared one to
    measure a group without double-counting, or omit it to size a table on its
    own.  Returns ``{"bytes", "objects", "truncated"}``.

    ``tally`` (a dict) accumulates ``type name -> [bytes, count]`` for every
    object visited, and ``big`` (a list) collects single objects at or above
    ``big_min``.  Both are what turn the leftover "misc" bucket from a number
    into something actionable -- they are off by default because the tables
    themselves only need the total.
    """
    if seen is None:
        seen = set()
    stack = list(roots)
    # SYNTHESIZED children must outlive the walk.  `seen` identifies objects by
    # id(), which is only unique among LIVE objects: _children(frame) builds a
    # fresh `dict(f_locals)` that nothing else references, so it is freed as soon
    # as the walk pops it -- and CPython hands the same address to the next
    # frame's snapshot, which then reads as already-seen and has its whole
    # subtree pruned.  With one worker thread that is a rare coincidence; with
    # ten (a full fit-worker pool, or six concurrent ITF searches) it is close to
    # certain, and it silently deleted exactly what this table exists to price --
    # a running search's 6 MB working set totalled 0.02 MB while the per-thread
    # rows, walked in separate calls, still showed 5.73.  Holding a reference
    # keeps every id distinct for the walk's lifetime.
    keep = []
    # The walker must not measure itself.  `seen` holds one id() per visited
    # object -- ints big enough to be real objects -- so a whole-heap pass that
    # reached it would report its own bookkeeping as the largest thing on the
    # heap (measured: a 16 MB set and 13 MB of ints, dwarfing every real table).
    for own in (seen, stack, tally, big, keep):
        if own is not None:
            seen.add(id(own))
    total = 0
    n = 0
    truncated = False
    while stack:
        o = stack.pop()
        if isinstance(o, _SKIP):
            continue
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        if n >= budget:
            truncated = True
            break
        n += 1
        try:
            size = sys.getsizeof(o)
        except Exception:               # objects with a hostile __sizeof__
            continue
        total += size
        if tally is not None:
            slot = tally.get(type(o).__name__)
            if slot is None:
                tally[type(o).__name__] = [size, 1]
            else:
                slot[0] += size
                slot[1] += 1
        if big is not None and size >= big_min:
            big.append((size, _describe(o)))
        try:
            kids = _children(o)
            if isinstance(o, types.FrameType):
                keep.extend(kids)       # see `keep` above: these are synthesized
            stack.extend(kids)
        except Exception:
            pass
    return {"bytes": total, "objects": n, "truncated": truncated}


def _mb(b):
    return round(b / (1024.0 * 1024.0), 2)


# --------------------------------------------------------------------------
# Process and container totals
# --------------------------------------------------------------------------
def _proc_status():
    """The /proc/self/status fields worth having, in MB.  {} off Linux."""
    want = {"VmRSS": "rss_mb", "VmHWM": "rss_peak_mb", "VmSize": "vsize_mb",
            "VmData": "data_mb", "Threads": "threads"}
    out = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in want:
                    parts = rest.split()
                    val = int(parts[0])
                    out[want[key]] = val if key == "Threads" else round(val / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return out


def _container():
    """What the ITF sweep floor sees: host MemAvailable, cgroup headroom, and the
    smaller of the two -- plus the floor it is compared against, so the page shows
    how close a search is to being truncated."""
    out = {}
    try:
        from pyroorb.itf import memory as itfmem
        out["host_available_mb"] = itfmem.meminfo_available_mb()
        out["cgroup_available_mb"] = itfmem.cgroup_available_mb()
        out["available_mb"] = itfmem.available_mb()
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"
    try:
        from pyroorb.itf.config import ItfConfig
        from pyroorb.itf.memory import sweep_floor_mb
        configured = ItfConfig().sweep_min_free_mb
        # The floor the sweep ACTUALLY enforces -- on a small container this is
        # scaled below the configured value, and the headroom card must colour
        # against the number the valve compares, not the one the config asks
        # for.  Both are shown so the scaling is visible when it happens.
        out["sweep_min_free_mb"] = sweep_floor_mb(configured)
        out["sweep_min_free_configured_mb"] = configured
    except Exception:
        pass
    return out


_SMAPS_HEADER = _re_smaps.compile(r"^[0-9a-fA-F]+-[0-9a-fA-F]+ ")


class _MallInfo2(ctypes.Structure):
    """glibc's ``struct mallinfo2`` (all size_t since 2.33 -- the older
    ``mallinfo`` used ints that wrap silently past 2 GB, which is precisely the
    size where the answer starts to matter)."""
    _fields_ = [(n, ctypes.c_size_t) for n in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
                 "fsmblks", "uordblks", "fordblks", "keepcost")]


_LIBC = None
_LIBC_TRIED = False


def _libc():
    """The glibc handle, loaded once.  None off glibc (musl, macOS), where every
    caller here degrades to a no-op rather than raising."""
    global _LIBC, _LIBC_TRIED
    if not _LIBC_TRIED:
        _LIBC_TRIED = True
        try:
            lib = ctypes.CDLL("libc.so.6", use_errno=True)
            lib.mallinfo2.restype = _MallInfo2
            lib.mallinfo2()                       # prove it before keeping it
            _LIBC = lib
        except Exception:
            _LIBC = None
    return _LIBC


def _mallinfo():
    """Raw ``struct mallinfo2``, or None.  0.4 us, so it is cheap enough to gate on."""
    lib = _libc()
    if lib is None:
        return None
    try:
        return lib.mallinfo2()
    except Exception:
        return None


# Free-but-held memory above which a completed unit of work pays for a trim.
# Below it the retention is ordinary high-water that the next request will reuse,
# and returning those pages only buys them back as page faults.
TRIM_THRESHOLD_MB = float(os.environ.get("PYROORB_TRIM_THRESHOLD_MB") or 256)


def trim_allocator(threshold_mb=None):
    """Hand back what the allocator is sitting on.  The ONE mutating entry point
    in this module; everything else here only reads.

    glibc frees a non-main arena's pages only when the top of its heap is free, so
    a handful of small survivors scattered through a big transient pin the whole
    thing: measured on the shape a search leaves behind, 211 MB of live data fell
    to 1 MB while RSS stayed at 542 MB with 212 MB "free". It is bounded by the
    CONCURRENT peak rather than the number of jobs -- successive searches reuse one
    arena and plateau (45 MB held), four at once keep four arenas and hold ~165 MB
    -- so a box sized for four concurrent searches retains roughly four searches'
    worth, for the life of the process.

    That is what took pyroorb.net to 6677 MB RSS with 99.6 MB actually in use and
    4760 MB held, which is a 933 MB headroom and an ITF admission gate that
    correctly refuses every search.

    Neither glibc env knob helps and both were measured: ``MALLOC_TRIM_THRESHOLD_``
    governs the main arena's top only (no effect on per-thread heaps), and
    ``MALLOC_ARENA_MAX=2`` left the same memory held while raising RSS. An explicit
    ``malloc_trim`` is the only thing that walks every arena and madvises the free
    pages back, and it costs ~1 ms.

    Returns the mallinfo dict from BEFORE the trim (with ``trimmed``/``skipped``),
    or None off glibc. NOTE: ``fordblks`` does not fall afterwards -- the free list
    is unchanged, the PAGES are returned -- so verify with RSS, never with this.
    """
    mi = _mallinfo()
    if mi is None:
        return None
    held_mb = mi.fordblks / (1024.0 * 1024.0)
    cut = TRIM_THRESHOLD_MB if threshold_mb is None else threshold_mb
    out = {"free_held_mb": round(held_mb, 2), "in_use_mb": _mb(mi.uordblks),
           "threshold_mb": cut, "trimmed": False}
    if held_mb < cut:
        return out
    try:
        _libc().malloc_trim(0)
        out["trimmed"] = True
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def _malloc_info():
    """What the C allocator is holding, or None where it cannot be asked.

    ``fordblks`` is the part that matters on a box that will not shrink: memory
    freed by the program but still owned by the allocator, so it counts in RSS
    and cannot be recovered by dropping any cache on this page.
    """
    mi = _mallinfo()
    if mi is None:
        return None
    return {"arena_mb": _mb(mi.arena), "mmap_mb": _mb(mi.hblkhd),
            "mmap_blocks": mi.hblks, "in_use_mb": _mb(mi.uordblks),
            "free_held_mb": _mb(mi.fordblks), "trim_top_mb": _mb(mi.keepcost)}


def _smaps_by_file(limit_lines=400000):
    """Resident memory per MAPPED FILE, and how many separate mappings each has.

    The reason this exists: a file mapped twice pays its resident pages TWICE in
    RSS, and nothing on the Python side of this page can see it -- an
    ``assist.Ephem`` is a tiny wrapper over a C mmap of the DE440 / sb441-n16
    kernels, so N of them is N x ~747 MB of real, resident, unattributable
    memory.  Counting mappings per path is what turns that from a mystery into a
    line item, so ``duplicated`` below is the headline: any file mapped more than
    once, largest first.
    """
    per = {}
    anon_kb = 0
    cur = None
    at_zero = False
    try:
        with open("/proc/self/smaps", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i > limit_lines:                 # a runaway map is not a reason to hang
                    break
                if _SMAPS_HEADER.match(line):
                    # "addr-addr perms offset dev inode  [path]" -- note the dev
                    # field ("08:01") carries a colon, so this must be matched on
                    # the address range, never by partitioning on ":".
                    parts = line.split(maxsplit=5)
                    path = parts[5].strip() if len(parts) > 5 else ""
                    cur = path if path.startswith("/") else None
                    # A mapping that starts at file offset 0 is one whole-file
                    # map -- one dlopen, or one mmap of a kernel.  Counting those
                    # separates a SECOND copy of a data file (the thing worth
                    # alarming about) from a shared library's several segments,
                    # which are one open at different offsets and would otherwise
                    # read as a dozen "duplicates".
                    at_zero = len(parts) > 2 and parts[2].strip("0") == ""
                elif line.startswith("Rss:"):
                    try:
                        kb = int(line.split()[1])
                    except (ValueError, IndexError):
                        continue
                    if cur:
                        slot = per.setdefault(cur, [0, 0, 0])
                        slot[0] += kb
                        slot[1] += 1
                        slot[2] += 1 if at_zero else 0
                    else:
                        anon_kb += kb
                    cur = None
    except OSError:
        return None
    rows = [{"path": path, "opens": opens, "vmas": n, "rss_mb": _mb(kb * 1024)}
            for path, (kb, n, opens) in per.items()]
    rows.sort(key=lambda r: -r["rss_mb"])
    dup = [r for r in rows if r["opens"] > 1 and r["rss_mb"] >= 1.0]
    return {"anon_mb": _mb(anon_kb * 1024),
            "file_mb": round(sum(r["rss_mb"] for r in rows), 2),
            "files": rows[:12], "duplicated": dup[:12]}


def _native():
    """The half of RSS no walk of the Python heap can reach: the C allocator's
    books and the file mappings, plus the thread counts that explain both.

    Cheap (no heap walk, a couple of /proc reads), so it rides along with the
    tables pass and needs no password -- the whole-heap walk is what is gated,
    and this is the part that was missing when 6.2 GB went unattributed.
    """
    out = {"malloc": _malloc_info(), "maps": _smaps_by_file()}
    try:
        # OS threads vs threads actually running Python: a large gap is native
        # thread pools (OpenMP under xgboost, the BLAS pool), each of which also
        # gets its own allocator arena.
        out["threads"] = {"os": len(os.listdir("/proc/self/task")),
                          "python_frames": len(sys._current_frames())}
    except Exception:
        pass
    return out


def python_heap(exclude=None):
    """Every object the GC tracks, summed -- a FLOOR on the Python heap, not the
    heap.  ``gc.get_objects`` returns tracked containers only, so plain ``str``,
    ``float`` and ``int`` are missing, and those are not a rounding error here:
    half of a fit's observation records is boxed floats and strings.  Cheap
    (~0.02 s for 140 k objects) and reported for scale; the accounted total to
    trust is ``tables_combined_mb + misc.mb``, which is a deep walk and does
    reach the atoms."""
    t0 = time.perf_counter()
    objs = gc.get_objects()
    skip = {id(objs)} | set(exclude or ())
    total = 0
    for o in objs:
        if id(o) in skip:
            continue
        try:
            total += sys.getsizeof(o)
        except Exception:
            pass
    n = len(objs)
    del objs
    return {"bytes": total, "objects": n, "secs": round(time.perf_counter() - t0, 3)}


# --------------------------------------------------------------------------
# The tables.  Each source yields (roots, rows) -- roots are walked, rows are
# the per-entry detail shown under the total.
# --------------------------------------------------------------------------
def _mod(name):
    """A module ONLY if it is already imported.  Never imports: loading ASSIST to
    size its ephemeris would allocate what we are trying to measure."""
    if name.startswith("pyroorb."):
        return sys.modules.get(name)
    return sys.modules.get(f"pyroorb.web.{name}") or sys.modules.get(f"pyroorb.{name}")


def _desig_of(fit):
    for attr in ("designation", "desig"):
        v = getattr(fit, attr, None)
        if v:
            return str(v)[:40]
    return "?"


def _nobs(fit):
    try:
        return int(fit.n_total_observations())
    except Exception:
        return None


def _src_sessions():
    """_SESSIONS: interactive fits.  Count-capped at _SES_MAX, no TTL, and evicted
    by creation order rather than use -- so these sit resident under everything."""
    eng = _mod("engine_api")
    if eng is None:
        return None
    with eng._SES_LOCK:
        items = list(eng._SESSIONS.items())
    rows = []
    for sid, s in items:
        fit = s.get("fit")
        rows.append({
            # the handle is a credential (it authorises every interactive endpoint
            # and, where enabled, submission), so only its readable prefix is shown
            "id": str(sid).split("-")[0],
            "label": _desig_of(fit) if fit is not None else "?",
            "n_obs": _nobs(fit),
            "bytes": walk([s])["bytes"],
            "note": ", ".join(filter(None, [
                "nongrav:" + str(s.get("model")) if s.get("model") else None,
                "constrained" if s.get("constraints") else None,
                "asteroids" if s.get("asteroids") else None,
                "ng_cache:%d" % len(s.get("ng_cache") or {}) if s.get("ng_cache") else None,
            ])) or None,
        })
    return {"roots": [dict(items)], "rows": rows,
            "count": len(items), "cap": eng._SES_MAX}


def _src_fitcache():
    """fitcache._CACHE: finished FittedOrbits kept for export / paste / re-auto."""
    fc = _mod("fitcache")
    if fc is None:
        return None
    items = list(fc._CACHE.items())
    rows = [{"id": str(k)[:16], "label": _desig_of(v), "n_obs": _nobs(v),
             "bytes": walk([v])["bytes"]} for k, v in items]
    return {"roots": [dict(items)], "rows": rows,
            "count": len(items), "cap": fc._MAX}


def _src_fit_jobs():
    """fitqueue._JOBS: the admission queue's job table.

    THE LARGEST TABLE ON A BUSY SERVER, and it was invisible here.  Every
    finished fit keeps two things: the serialised payload the browser reads,
    and -- under `retain` -- the pristine TEMPLATE SESSION that a `restore`
    token redeems, which holds an evaluator context and a rebound base sim.
    That is the same shape of object `sessions` prices, but it is no longer in
    `_SESSIONS`: the queue lifts it out precisely so cached fits cannot evict
    real users' open sessions.  So the sessions row reads zero on a server that
    is holding dozens of fits, and the memory the page exists to find was
    attributed to nothing at all.

    Bounded by `fitqueue.JOBS_MAX`, which is therefore the knob that decides
    this table's size -- reported as `cap` so the two can be read together.

    The stored `work` closure is skipped: on a QUEUED job it holds the pending
    call's arguments, and following it would charge this table for an
    observation list that the fit itself will own moments later.
    """
    fq = _mod("fitqueue")
    if fq is None:
        return None
    with fq._LOCK:
        items = [(jid, {k: v for k, v in j.items() if k != "work"})
                 for jid, j in fq._JOBS.items()]
    if not items:
        return {"roots": [], "rows": [], "count": 0, "cap": fq.JOBS_MAX}
    rows = []
    for jid, j in items:
        rows.append({
            "id": str(jid)[:10] + "…",          # a job id redeems a session
            "label": str(j.get("desig") or "?")[:40],
            "note": ("%s%s" % (j.get("state"),
                               ", template" if j.get("retain") is not None else "")),
            "bytes": walk([j])["bytes"],
        })
    return {"roots": [j for _jid, j in items], "rows": rows,
            "count": len(items), "cap": fq.JOBS_MAX}


def _src_itf_jobs():
    """_ITF_JOBS: backgrounded ITF searches -- the finished payload of each, and
    the ones still running (whose real cost is working memory on a thread, not
    anything in this table; see `searches` in the report)."""
    eng = _mod("engine_api")
    if eng is None:
        return None
    with eng._ITF_JOBS_LOCK:
        # WITHOUT the worker Thread the entry carries (engine_api._itf_reap
        # holds one to tell a live search from a stranded slot).  Following it
        # would walk the thread's target closure -- the session, the options,
        # and through the frame the search's whole working set -- and charge
        # all of it to this table, when that memory is exactly what the
        # `threads` source below reports on its own.  One walk, one owner.
        items = [(jid, {k: v for k, v in j.items() if k != "thread"})
                 for jid, j in eng._ITF_JOBS.items()]
    now = time.time()
    rows = []
    for jid, j in items:
        done = bool(j.get("done"))
        rows.append({
            "id": str(jid)[:8] + "…",   # a job id collects someone's search payload
            "label": "finished" if done else "RUNNING",
            "bytes": walk([j])["bytes"],
            "note": "%.0fs %s" % ((j.get("ended") or now) - (j.get("started") or now),
                                  "collected" if j.get("collected") else "uncollected"),
        })
    return {"roots": [dict(items)], "rows": rows,
            "count": len(items), "cap": eng._ITF_JOBS_MAX}


def _src_traces():
    """_TRACES: the "see trace" narration feeds, capped at 20 000 lines each."""
    eng = _mod("engine_api")
    if eng is None:
        return None
    with eng._TRACES_LOCK:
        items = list(eng._TRACES.items())
    rows = []
    for tid, ent in items:
        with ent["lock"]:
            lines = list(ent["lines"])
            done = ent["done"]
        rows.append({"id": str(tid)[:8] + "…",
                     "label": "%d lines" % len(lines),
                     "bytes": walk([lines])["bytes"],
                     "note": "finished" if done else "live"})
    return {"roots": [dict(items)], "rows": rows,
            "count": len(items), "cap": eng._TRACES_MAX}


def _src_debias():
    """debias._BIAS: the EFCC18 star-catalog table, one (49152, 26, 4) array."""
    db = _mod("debias")
    if db is None or getattr(db, "_BIAS", None) is None:
        return None
    arr = db._BIAS
    return {"roots": [arr], "rows": [],
            "count": 1, "cap": None,
            "detail": "%s %s" % (getattr(arr, "shape", "?"), getattr(arr, "dtype", ""))}


def _src_sun_memo():
    """force_assist._SUN_CACHE: Sun-barycentre positions memoised per epoch, capped
    at 300 000 entries -- a cap in ENTRIES, never audited in bytes."""
    fa = _mod("force_assist")
    if fa is None:
        return None
    cache = getattr(fa, "_SUN_CACHE", None)
    if not cache:
        return None
    snap = dict(cache)
    return {"roots": [snap], "rows": [],
            "count": len(snap), "cap": getattr(fa, "_SUN_CACHE_MAX", None)}


def _src_geo_memo():
    """obs_sources._GEOCENTRIC_MEMO: a handful of Earth-Sun grids."""
    os_mod = _mod("obs_sources")
    if os_mod is None:
        return None
    memo = getattr(os_mod, "_GEOCENTRIC_MEMO", None)
    if not memo:
        return None
    snap = dict(memo)
    return {"roots": [snap], "rows": [], "count": len(snap),
            "cap": getattr(os_mod, "_GEOCENTRIC_MEMO_MAX", None)}


def _src_obscodes():
    """The MPC observatory-code table -- one dict, shared by every fit."""
    eng = _mod("engine_api")
    if eng is None or getattr(eng, "_OBSCODES", None) is None:
        return None
    roots = [eng._OBSCODES]
    if getattr(eng, "_OBSNAMES", None) is not None:
        roots.append(eng._OBSNAMES)
    return {"roots": roots, "rows": [], "count": len(eng._OBSCODES), "cap": None}


def _src_rankers():
    """itf.training._RANKER_CACHE: loaded ML ranker boosters (the deployed
    ranker is ONE v4 model since 2026-09-27, so expect one; an A/B that re-arms
    a second-opinion trial source loads two)."""
    tr = _mod("pyroorb.itf.training")
    if tr is None:
        return None
    cache = getattr(tr, "_RANKER_CACHE", None)
    if not cache:
        return None
    snap = dict(cache)
    rows = [{"id": os.path.basename(str(k))[:28], "label": "booster",
             "bytes": walk([v])["bytes"]} for k, v in snap.items()]
    return {"roots": [snap], "rows": rows, "count": len(snap), "cap": None}


#: Worker threads name themselves after the handle they serve --
#: ``itf-search-<job_id>`` (engine_api.itf_search_start).  That job id IS the
#: /api/itf_result collection key, and its payload carries the fresh session
#: handle, which authorises every interactive endpoint (and, where enabled,
#: submission).  This page is unauthenticated by design, so a full thread name
#: would hand an anonymous reader a live search's credential.  Redact the id to
#: the same readable prefix the _ITF_JOBS table shows and no more.
import re as _re
_JOB_THREAD_RE = _re.compile(r"^(itf-search-)(.+)$")


def _safe_thread_name(name):
    """A thread name safe to show on the unauthenticated /memory page: the
    ``itf-search-<job_id>`` handle is collapsed to its first 8 id characters
    plus an ellipsis (matching :func:`_src_itf_jobs`), everything else is shown
    as-is and length-capped."""
    s = str(name)
    m = _JOB_THREAD_RE.match(s)
    if m:
        return m.group(1) + m.group(2)[:8] + "…"
    return s[:28]


def _src_threads():
    """Working memory of every OTHER thread -- above all a running ITF search.

    This is the category the tables cannot show and the one that kills the box:
    a search's sweep rows, candidate features and trial fits are locals on a
    worker thread, referenced by nothing the server keeps.  Frames are reachable
    (``sys._current_frames``), so the walk can price them.

    The calling thread is excluded -- it is the HTTP handler running this report,
    whose locals are the report itself.
    """
    import threading
    try:
        frames = sys._current_frames()
    except Exception:
        return None
    me = threading.get_ident()
    names = {t.ident: t.name for t in threading.enumerate()}
    roots, rows = [], []
    for ident, frame in list(frames.items()):
        if ident == me:
            continue
        chain = []
        f = frame
        while f is not None and len(chain) < 200:   # a runaway stack is not a reason to hang
            chain.append(f)
            f = f.f_back
        roots.extend(chain)
        rows.append({"id": _safe_thread_name(names.get(ident, ident)),
                     "label": "%d frames" % len(chain),
                     "bytes": walk(chain)["bytes"],
                     "note": (chain[0].f_code.co_name if chain else None)})
    if not roots:
        return None
    return {"roots": roots, "rows": rows, "count": len(rows), "cap": None}


def _src_net_reduced():
    """obs80._NET_REDUCED: the per-object reduced-astrometry cache."""
    o80 = _mod("obs80")
    if o80 is None:
        return None
    cache = getattr(o80, "_NET_REDUCED", None)
    if not cache:
        return None
    snap = dict(cache)
    return {"roots": [snap], "rows": [], "count": len(snap), "cap": None}


def misc_breakdown(shared, budget=_NODE_BUDGET, top=14, big_top=12):
    """Everything on the Python heap that no named table reaches -- the "what else
    is in here?" bucket, and the honest answer to whether the tables are even the
    main story.

    Seeded from ``gc.get_objects()`` with the tables' shared ``seen`` set already
    populated, so anything already attributed is skipped.  It is a deep walk, not
    a sum over tracked objects, so it reaches the untracked atoms (str/float/int)
    that ``python_heap`` cannot see -- and, through thread frames, the working set
    of any ITF search running right now.

    Reported two ways because they fail differently: BY TYPE says what kind of
    thing is accumulating, and the LARGEST INDIVIDUAL objects catch the single
    forgotten 20 MB array that a type total would bury.  Names and shapes only,
    never contents.
    """
    t0 = time.perf_counter()
    # COLLECT FIRST.  gc.get_objects() hands back cyclic garbage that has not been
    # collected yet, and this walk necessarily makes some: it follows thread frames,
    # so the frames of this call reference `shared`, and `shared` (through the walk)
    # reaches those frames.  Left alone, each pass therefore MEASURES the debris of
    # the pass before it and adds its own -- misc doubles per call and RSS with it
    # (an idle server went 22 -> 145 MB over six reports, and a run gc.collect()
    # gave it all back).  A report that inflates the number it exists to report is
    # worse than no report, so the garbage goes before the census is taken.
    gc.collect()
    tally, big = {}, []
    roots = gc.get_objects()
    # gc.get_objects() returns a list of ~everything, and this walk now follows
    # thread frames -- including this one -- so the roots list, the caller's
    # report dict and our own tallies are all reachable.  Exclude them by id.
    shared.add(id(roots))
    shared.add(id(sys._getframe()))
    w = walk(roots, seen=shared, budget=budget, tally=tally, big=big)
    del roots
    rows = sorted(((v[0], k, v[1]) for k, v in tally.items()), reverse=True)[:top]
    big.sort(reverse=True)
    seen_desc = set()
    biggest = []
    for size, desc in big:
        if len(biggest) >= big_top:
            break
        biggest.append({"mb": _mb(size), "what": desc})
        seen_desc.add(desc)
    out = {
        "mb": _mb(w["bytes"]), "objects": w["objects"], "truncated": w["truncated"],
        "by_type": [{"type": name, "mb": _mb(b), "count": c} for b, name, c in rows],
        "biggest": biggest,
        "secs": round(time.perf_counter() - t0, 3),
    }
    # Drop our own big structures before returning, so the cycle this pass made is
    # as small as possible for the collect at the top of the next one.  `shared` is
    # the caller's -- it clears that itself once the totals are computed.
    tally.clear()
    big.clear()
    del w
    return out


# key, title, note, getter -- order is the report's order.
_SOURCES = [
    ("sessions", "Interactive fit sessions (_SESSIONS)",
     "One per open fit: evaluator context, observations, gravity + nongrav fits. "
     "No TTL; evicted by creation order, not by use.", _src_sessions),
    ("fitcache", "Finished-fit cache (fitcache._CACHE)",
     "Whole FittedOrbits kept for export, paste-append and re-auto. True LRU.",
     _src_fitcache),
    ("fit_jobs", "Fit queue jobs (fitqueue._JOBS)",
     "One per queued/running/cached fit: the payload the browser reads, plus the "
     "template session a `restore` token redeems (evaluator context + base sim). "
     "Bounded by JOBS_MAX, not by _SES_MAX.", _src_fit_jobs),
    ("itf_jobs", "ITF search jobs (_ITF_JOBS)",
     "Backgrounded searches: the finished payload of each, held 1800 s for collection.",
     _src_itf_jobs),
    ("traces", "ITF trace feeds (_TRACES)",
     "Live search narration, 20 000 lines per feed maximum.", _src_traces),
    ("rankers", "ML rankers (itf.training._RANKER_CACHE)",
     "Loaded candidate-ranking boosters.", _src_rankers),
    ("debias", "EFCC18 debias table (debias._BIAS)",
     "Star-catalog bias grid, loaded once and shared by every fit.", _src_debias),
    ("obscodes", "Observatory codes (engine_api._OBSCODES)",
     "Station parallax constants + names, shared by every fit.", _src_obscodes),
    ("sun_memo", "Sun-barycentre memo (force_assist._SUN_CACHE)",
     "Per-epoch Sun positions. Capped at 300 000 ENTRIES -- never audited in bytes.",
     _src_sun_memo),
    ("geo_memo", "Earth-Sun grid memo (obs_sources._GEOCENTRIC_MEMO)",
     "A few (n,3) grids.", _src_geo_memo),
    ("net_reduced", "Reduced-astrometry cache (obs80._NET_REDUCED)",
     "Per-object reduced observation lines.", _src_net_reduced),
    ("threads", "Working memory of running threads",
     "Locals of every thread but this one -- where a running ITF search's sweep "
     "rows, candidate features and trial fits actually live. No table holds these.",
     _src_threads),
]


def _sweeps():
    """Propagator sweeps in flight on every thread, and the slowest completed lately
    (``sweepwatch``).  This is the record the two wedged ITF searches of 2026-09-26
    left NO trace of: which backend's integrate() a thread is parked in, integrating
    WHAT state to WHICH epoch, for how long.  The JSON carries the full trial state,
    so a wedge can be reproduced offline from the report alone.  Names go through
    ``_safe_thread_name`` like the thread rows; nothing is imported to answer."""
    sw = _mod("sweepwatch")
    if sw is None:
        return None
    import threading
    names = {t.ident: _safe_thread_name(t.name) for t in threading.enumerate()}
    try:
        return sw.snapshot(names)
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


def _searches():
    """Running ITF searches -- the thing this page exists for and the one thing no
    table holds.  A search's working set (sweep rows, candidate features, trial
    fits) lives in locals on a worker thread and is invisible to any walk of the
    tables, so it is reported as evidence rather than as a size: how many are
    running, for how long, and how far RSS has moved since each started.

    That delta is a HINT, not an attribution -- concurrent searches, a fit in
    another tab and the allocator all move RSS too.  It is still the number that
    matters, because two searches at once are what kill the box.
    """
    eng = _mod("engine_api")
    if eng is None:
        return {"running": 0, "rows": []}
    now = time.time()
    rss = _proc_status().get("rss_mb")
    rows = []
    with eng._ITF_JOBS_LOCK:
        items = list(eng._ITF_JOBS.items())
    for jid, j in items:
        if j.get("done"):
            continue
        started = j.get("started") or now
        r0 = j.get("rss_start_mb")
        rows.append({"id": str(jid)[:8] + "…",
                     "elapsed_s": round(now - started, 1),
                     "rss_start_mb": r0,
                     "rss_delta_mb": (round(rss - r0, 1)
                                      if (rss is not None and r0 is not None) else None)})
    return {"running": len(rows), "rows": rows}


# Per-entry rows shown for one table.  Sized to cover the largest table's own
# cap (_SESSIONS, PYROORB_SESSION_MAX, default 64) so a session is never missing
# from the page merely because sixteen bigger ones exist -- "is my fit still
# resident?" is as much the question here as "which one is the 40 MB".  Anything
# past this is counted in `rows_omitted` rather than dropped without a word.
_MAX_ROWS = max(int(os.environ.get("PYROORB_MEMORY_MAX_ROWS") or 0),
                int(os.environ.get("PYROORB_SESSION_MAX") or 0), 64)


def report(budget=_NODE_BUDGET, with_misc=True):
    """The whole picture, as JSON-safe dicts.  Read-only; safe to call at any time,
    including while a search runs.

    ``with_misc=False`` skips the whole-heap pass -- the tables alone cost
    milliseconds on an idle server, while misc walks everything the process can
    reach (~0.5 s).

    Cost scales with what is resident, and the per-entry rows mean a table is
    walked three times: once per entry for its detail line, once whole for its
    gross figure, once more in the combined pass.  Twelve four-thousand-obs
    sessions is the realistic worst case, ~2.5 s -- which the report states in
    ``cost_secs`` rather than hiding.  Per-entry rows are worth that: "which
    session is the 40 MB" is the question the total cannot answer."""
    t0 = time.perf_counter()
    proc = _proc_status()
    out = {
        "ok": True,
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
        "pid": os.getpid(),
        "process": proc,
        "container": _container(),
        "native": _native(),
        "searches": _searches(),
        "sweeps": _sweeps(),
        "tables": [],
    }

    gathered = []
    for key, title, note, getter in _SOURCES:
        try:
            got = getter()
        except Exception as ex:
            out["tables"].append({"key": key, "title": title, "note": note,
                                  "loaded": False,
                                  "error": f"{type(ex).__name__}: {ex}"})
            continue
        if got is None:
            out["tables"].append({"key": key, "title": title, "note": note,
                                  "loaded": False, "mb": 0.0})
            continue
        gathered.append((key, title, note, got))

    # Gross: each table sized on its own (deduped within itself).
    for key, title, note, got in gathered:
        w = walk(got["roots"], budget=budget)
        all_rows = sorted(got.get("rows") or [], key=lambda r: -(r.get("bytes") or 0))
        rows = all_rows[:_MAX_ROWS]
        for r in rows:
            r["mb"] = _mb(r.pop("bytes", 0) or 0)
        out["tables"].append({
            "key": key, "title": title, "note": note, "loaded": True,
            "mb": _mb(w["bytes"]), "objects": w["objects"],
            "truncated": w["truncated"],
            "count": got.get("count"), "cap": got.get("cap"),
            "detail": got.get("detail"), "rows": rows,
            # Never truncate SILENTLY: the page's whole claim is that it says
            # where the memory is, and "16 rows" read as "16 entries" when the
            # session table could hold sixty-four of them.
            "rows_omitted": max(0, len(all_rows) - len(rows)),
        })

    # Measured before the combined pass builds its `seen` set, so that set is not
    # part of the heap it reports.
    heap = python_heap()

    # Combined: one shared `seen`, so anything two tables hold is charged once.
    shared = set()
    combined = 0
    for _key, _t, _n, got in gathered:
        combined += walk(got["roots"], seen=shared, budget=budget)["bytes"]

    # Misc: the rest of the heap, walked with the SAME `shared` set, so the two
    # add up instead of overlapping.  Must run after the combined pass.
    misc = misc_breakdown(shared, budget=budget) if with_misc else None
    out["misc"] = misc
    # `shared` holds an id per object the whole heap reaches -- millions on a busy
    # box, and reachable from this frame, which the walk itself followed.  Dropping
    # it here is what keeps a report from outliving itself; see misc_breakdown.
    shared.clear()
    for _e in gathered:
        _e[3]["roots"] = []

    gross = sum(t.get("mb", 0.0) for t in out["tables"] if t.get("loaded"))
    rss_mb = proc.get("rss_mb")
    tables_mb = _mb(combined)
    misc_mb = misc["mb"] if misc else None
    accounted = tables_mb + (misc_mb or 0.0)
    out["totals"] = {
        "tables_gross_mb": round(gross, 2),
        "tables_combined_mb": tables_mb,
        "overlap_mb": round(gross - tables_mb, 2),
        "misc_mb": misc_mb,
        # tables + misc = every Python object reachable from the GC's roots.
        "accounted_mb": round(accounted, 2) if misc else None,
        "python_heap_tracked_mb": _mb(heap["bytes"]),
        "python_objects_tracked": heap["objects"],
        "rss_mb": rss_mb,
        # What no Python object explains: the interpreter and its C extensions
        # (NumPy/rebound/ASSIST buffers held behind opaque objects), the
        # memory-mapped DE440 ephemerides, and allocator slack -- memory that is
        # real, resident, and not freeable by shrinking any cache here.
        "native_mb": (round(rss_mb - accounted, 2)
                      if (rss_mb is not None and misc) else None),
    }
    out["cost_secs"] = round(time.perf_counter() - t0, 3)
    return out
