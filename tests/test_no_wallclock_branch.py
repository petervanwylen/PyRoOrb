"""The fit's numerical path must not depend on MACHINE LOAD.

`force_assist._integrate_split` once chose its IAS15 tolerance with a 4-second wall-clock
timer, so a Jacobian batch that finished quickly on an idle box and slowly on a loaded one
was integrated at a DIFFERENT epsilon -- and the LM then landed in a different basin. The
wrong answers reproduced digit-for-digit (490581: 1207/1208 @ 0.249" quiet, 173/1208 @ 625"
under four concurrent workers), which is the signature of a deterministic branch taken on a
timing threshold. The timer was deleted 2026-08-25; this test is the guard that keeps the
whole CLASS out of the fit core, since the defect hid for months with no detector at all.

The rule is narrow and mechanical: in the modules that compute an orbit, elapsed time may be
MEASURED (STAGE_TIMES, verbose traces, `--time` reporting) but may never be COMPARED -- a
comparison is a branch, and a branch on the clock is a result chosen by the box. Bounded-work
deadlines that ABORT and say so (itf's no_link_deadline, the ident-check subprocess timeout)
are a different, reported thing and live in modules outside this list.
"""
import ast
import pathlib

import pytest

# The modules that turn observations into an orbit. Not itf/ (search deadlines are bounded-work
# budgets that report themselves), not web/.
FIT_CORE = ["force_assist.py", "forward_model.py", "arc_extend.py", "iod.py", "fit_object.py",
            "nongrav.py", "orbit_common.py", "lov.py", "guards.py", "weights.py"]

_CLOCKS = {"perf_counter", "monotonic", "monotonic_ns", "perf_counter_ns", "time", "process_time"}


def _bare_clock_names(tree):
    """Names bound to a clock by a `from time import perf_counter [as tick]`.

    Without this the detector sees only the `time.perf_counter()` spelling, and the exact
    defect it exists to keep out re-enters as `from time import perf_counter` + a bare
    `perf_counter() - t0 > 4.0` -- same branch, same machine-load dependence, invisible."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "time":
            out |= {a.asname or a.name for a in n.names if a.name in _CLOCKS}
    return out


def _reads_clock(node, bare=()):
    """True if this subtree calls time.perf_counter() / time.time() / time.monotonic()...
    `bare` carries the from-imported clock names for the module being scanned."""
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute) and n.func.attr in _CLOCKS \
                and isinstance(n.func.value, ast.Name) \
                and n.func.value.id.lstrip("_") in ("time", "stime"):
            return True
        if isinstance(n.func, ast.Name) and n.func.id in bare:
            return True
    return False


def _clock_comparisons(src):
    """Every comparison in `src` that has a clock reading anywhere under it, as (line, dump)."""
    tree = ast.parse(src)
    bare = _bare_clock_names(tree)
    # Names bound to a clock reading (t0 = time.perf_counter()) -- a later `now - t0 > X` is
    # still a clock branch even though the comparison itself calls nothing.
    stamps = {t.id for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and _reads_clock(n.value, bare)
              for t in n.targets if isinstance(t, ast.Name)}
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Compare):
            continue
        names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        if _reads_clock(n, bare) or (names & stamps):
            out.append((n.lineno, ast.dump(n)[:120]))
    return out


@pytest.mark.parametrize("name", FIT_CORE)
def test_fit_core_never_branches_on_the_clock(name):
    p = pathlib.Path(__file__).resolve().parent.parent / "src" / "pyroorb" / name
    if not p.exists():                       # a module renamed away is not a failure of this rule
        pytest.skip(f"{name} not present")
    # encoding is explicit: the engine sources carry non-ASCII (arcsec sigmas, em
    # dashes), and read_text() defaults to the LOCALE encoding -- cp1252 on Windows,
    # where this guard died with UnicodeDecodeError on four of its ten modules
    # instead of checking them. A guard that cannot read the file cannot guard.
    hits = _clock_comparisons(p.read_text(encoding="utf-8"))
    assert not hits, (
        f"{name} compares a wall-clock reading: {hits}. A branch on elapsed time lets machine "
        f"load pick the numerical path. Measure time freely; never compare it.")


def test_the_guard_can_actually_see_one():
    """The detector, on the code it was written against (the deleted THRASH_SECONDS branch)."""
    src = ("import time\n"
           "def f(bail, cnt):\n"
           "    t_start = time.perf_counter()\n"
           "    if bail and (cnt & 63) == 0 and time.perf_counter() - t_start > 4.0:\n"
           "        raise RuntimeError\n")
    assert _clock_comparisons(src)
    # ...the same branch written with a from-import, which reads no attribute at all.
    assert _clock_comparisons("from time import perf_counter\n"
                              "def f(bail, t_start):\n"
                              "    if bail and perf_counter() - t_start > 4.0:\n"
                              "        raise RuntimeError\n")
    assert _clock_comparisons("from time import monotonic as tick\n"
                              "def f():\n"
                              "    t0 = tick()\n"
                              "    while tick() - t0 < 2.0:\n"
                              "        pass\n")
    # ...and does not fire on measurement-only bookkeeping, which the fit core is full of.
    assert not _clock_comparisons("import time\n"
                                  "def g(d, k):\n"
                                  "    t0 = time.perf_counter()\n"
                                  "    d[k] = d.get(k, 0.0) + (time.perf_counter() - t0)\n")
