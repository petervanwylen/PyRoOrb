"""The memory measurement the sweep floor stands on (itf/memory.py).

The bug these cover: /proc/meminfo inside a container is the HOST's, so a
512 MB container on a big machine read tens of GB free and never tripped a
1000 MB floor -- the guard was inert exactly where it was needed.  Every case
here builds a fake /sys and /proc under tmp_path and re-bases the reader onto
it, so they assert on real file layouts without needing a container.
"""

import os

from pyroorb.itf.memory import (available_mb, cgroup_available_mb,
                        meminfo_available_mb)

MB = 1024 * 1024


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def _meminfo(root, avail_kb):
    _write(os.path.join(root, "proc/meminfo"),
           f"MemTotal:       32000000 kB\nMemAvailable:   {avail_kb} kB\n")


def _cgroup_v2(root, limit=None, current=None, inactive=None):
    base = os.path.join(root, "sys/fs/cgroup")
    if limit is not None:
        _write(os.path.join(base, "memory.max"), f"{limit}\n")
    if current is not None:
        _write(os.path.join(base, "memory.current"), f"{current}\n")
    if inactive is not None:
        _write(os.path.join(base, "memory.stat"),
               f"anon 12345\ninactive_file {inactive}\nslab 999\n")


def _cgroup_v1(root, limit=None, usage=None, inactive=None):
    base = os.path.join(root, "sys/fs/cgroup/memory")
    if limit is not None:
        _write(os.path.join(base, "memory.limit_in_bytes"), f"{limit}\n")
    if usage is not None:
        _write(os.path.join(base, "memory.usage_in_bytes"), f"{usage}\n")
    if inactive is not None:
        _write(os.path.join(base, "memory.stat"),
               f"cache 5\ntotal_inactive_file {inactive}\n")


def test_cgroup_v2_headroom(tmp_path):
    root = str(tmp_path)
    _cgroup_v2(root, limit=512 * MB, current=200 * MB, inactive=0)
    assert cgroup_available_mb(root) == 312


def test_inactive_file_is_not_charged_as_used(tmp_path):
    """Page cache is reclaimable: charging it would fire the floor on a
    container that has merely READ a lot (ephemeris/debias tables)."""
    root = str(tmp_path)
    _cgroup_v2(root, limit=512 * MB, current=400 * MB, inactive=300 * MB)
    assert cgroup_available_mb(root) == 412        # working set is 100 MB


def test_v2_max_means_no_limit(tmp_path):
    root = str(tmp_path)
    _cgroup_v2(root, limit="max", current=200 * MB)
    assert cgroup_available_mb(root) is None


def test_v1_fallback(tmp_path):
    root = str(tmp_path)
    _cgroup_v1(root, limit=256 * MB, usage=100 * MB, inactive=10 * MB)
    assert cgroup_available_mb(root) == 166


def test_v1_unset_limit_sentinel_is_no_limit(tmp_path):
    """An unlimited v1 cgroup writes a near-2**63 sentinel, not a real cap."""
    root = str(tmp_path)
    _cgroup_v1(root, limit=9223372036854771712, usage=100 * MB)
    assert cgroup_available_mb(root) is None


def test_nothing_readable_is_none_not_an_exception(tmp_path):
    root = str(tmp_path)
    assert cgroup_available_mb(root) is None
    assert meminfo_available_mb(root) is None
    assert available_mb(root) is None            # floor simply never fires


def test_container_headroom_wins_over_host_free(tmp_path):
    """The production bug, end to end: 30 GB free on the host, 512 MB
    container.  The old reader answered 30000; the floor must see 312."""
    root = str(tmp_path)
    _meminfo(root, 30_000_000)
    _cgroup_v2(root, limit=512 * MB, current=200 * MB, inactive=0)
    assert meminfo_available_mb(root) == 29296
    assert available_mb(root) == 312


def test_host_pressure_still_wins_when_smaller(tmp_path):
    """A generous cgroup limit does not make host pressure disappear."""
    root = str(tmp_path)
    _meminfo(root, 300_000)                       # ~293 MB free on the host
    _cgroup_v2(root, limit=64 * 1024 * MB, current=MB, inactive=0)
    assert available_mb(root) == 292


def test_real_machine_answers_something_sane():
    """No fixtures: whatever this box is, the reader must not raise and must
    not answer an absurd figure."""
    got = available_mb()
    assert got is None or (0 <= got < 64 * 1024 * 1024)


def _proc_cgroup(root, text):
    _write(os.path.join(root, "proc/self/cgroup"), text)


def test_limit_in_a_sub_cgroup_is_found(tmp_path):
    """Some runtimes leave the mount root at "max" and cap the process one
    level down; reading only the root would call that unlimited."""
    root = str(tmp_path)
    _proc_cgroup(root, "0::/system.slice/app.service\n")
    _cgroup_v2(root, limit="max", current=MB)          # mount root: no cap
    leaf = os.path.join(root, "sys/fs/cgroup/system.slice/app.service")
    _write(os.path.join(leaf, "memory.max"), f"{512 * MB}\n")
    _write(os.path.join(leaf, "memory.current"), f"{200 * MB}\n")
    assert cgroup_available_mb(root) == 312


def test_tightest_ancestor_wins(tmp_path):
    """A nested cgroup is bounded by its own limit AND every ancestor's."""
    root = str(tmp_path)
    _proc_cgroup(root, "0::/parent/child\n")
    base = os.path.join(root, "sys/fs/cgroup")
    _write(os.path.join(base, "parent/memory.max"), f"{256 * MB}\n")
    _write(os.path.join(base, "parent/memory.current"), f"{100 * MB}\n")
    _write(os.path.join(base, "parent/child/memory.max"), f"{1024 * MB}\n")
    _write(os.path.join(base, "parent/child/memory.current"), f"{10 * MB}\n")
    assert cgroup_available_mb(root) == 156            # the parent's 256-100


def test_breakdown_reports_both_file_page_classes(tmp_path):
    """The diagnostic behind the headroom discrepancy: cgroup headroom charges
    active_file as used while the host's MemAvailable would count about half of
    it as free, so the two sides of the min() disagree by exactly this term --
    and nobody has measured it on a deploy that just mmapped 700 MB of
    ephemeris.  The breakdown exists so a production trace can settle it."""
    from pyroorb.itf.memory import breakdown
    root = str(tmp_path)
    _write(os.path.join(root, "sys/fs/cgroup/memory.max"), f"{512 * MB}\n")
    _write(os.path.join(root, "sys/fs/cgroup/memory.current"), f"{400 * MB}\n")
    _write(os.path.join(root, "sys/fs/cgroup/memory.stat"),
           f"anon {100 * MB}\ninactive_file {120 * MB}\n"
           f"active_file {180 * MB}\n")
    got = breakdown(root)
    assert got == {"limit": 512, "usage": 400, "inactive_file": 120,
                   "active_file": 180, "headroom": 232}
    # the decision function sees only the inactive half as reclaimable
    assert cgroup_available_mb(root) == 232


def test_breakdown_is_none_without_a_limit(tmp_path):
    from pyroorb.itf.memory import breakdown
    root = str(tmp_path)
    _cgroup_v2(root, limit="max", current=MB)
    assert breakdown(root) is None


def test_v1_breakdown_uses_v1_field_names(tmp_path):
    from pyroorb.itf.memory import breakdown
    root = str(tmp_path)
    base = os.path.join(root, "sys/fs/cgroup/memory")
    _write(os.path.join(base, "memory.limit_in_bytes"), f"{256 * MB}\n")
    _write(os.path.join(base, "memory.usage_in_bytes"), f"{200 * MB}\n")
    _write(os.path.join(base, "memory.stat"),
           f"total_inactive_file {50 * MB}\ntotal_active_file {30 * MB}\n")
    got = breakdown(root)
    assert got["active_file"] == 30 and got["inactive_file"] == 50


def test_a_rebased_root_is_never_cached(tmp_path):
    """The real root's cgroup levels are resolved once (they cannot change for
    a running process, and this is called once per sweep window).  A re-based
    root must NOT be, or a test could not change its own fake /sys."""
    root = str(tmp_path)
    _cgroup_v2(root, limit=512 * MB, current=200 * MB, inactive=0)
    assert cgroup_available_mb(root) == 312
    _write(os.path.join(root, "sys/fs/cgroup/memory.current"), f"{400 * MB}\n")
    assert cgroup_available_mb(root) == 112
