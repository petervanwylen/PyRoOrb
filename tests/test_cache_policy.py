"""The fetch cache is short-term burst protection for the MPC observations API: a published
designation's observations are reused for three hours, a NEOCP / temporary id's for two
minutes, expired entries are deleted, and the cache is held under its size cap without
ever touching reference data."""
import os
import time

import pytest

from pyroorb import datadirs
from pyroorb import obs80


def test_lifetimes_follow_the_designation():
    assert obs80.ASTROMETRY_CACHE_TTL_S == 3 * 3600
    assert obs80.astrometry_ttl("2013 PY26") == obs80.ASTROMETRY_CACHE_TTL_S
    assert obs80.astrometry_ttl("433") == obs80.ASTROMETRY_CACHE_TTL_S
    assert obs80.astrometry_ttl("C/2023 A3") == obs80.ASTROMETRY_CACHE_TTL_S
    assert obs80.astrometry_ttl("P22r4rx") == obs80.NEOCP_CACHE_TTL_S == 120


def test_a_two_hour_old_published_cache_is_still_fresh(tmp_path):
    f = tmp_path / "ades_2013_PY26.json"
    f.write_text("[]", encoding="utf-8")
    two_hours = time.time() - 2 * 3600
    os.utime(f, (two_hours, two_hours))
    assert obs80.cache_fresh(f, obs80.astrometry_ttl("2013 PY26"))
    assert not obs80.cache_fresh(f, obs80.astrometry_ttl("P22r4rx"))


def _file(root, name, size, age_s, now):
    p = root / name
    p.write_bytes(b"x" * size)
    os.utime(p, (now - age_s, now - age_s))
    return p


def test_trim_deletes_expired_then_oldest_and_keeps_reference_data(tmp_path):
    now = time.time()
    expired = _file(tmp_path, "ades_OLD.json", 10, 4 * 3600, now)
    a = _file(tmp_path, "obs80_A.json", 400_000, 3000, now)      # oldest live entry
    b = _file(tmp_path, "neocp_B.json", 400_000, 2000, now)
    c = _file(tmp_path, "mpcorb_C.json", 400_000, 1000, now)
    catalogue = _file(tmp_path, "MPCORB_index.npz", 500_000, 30 * 86400, now)
    table = _file(tmp_path, "ObsCodes.htm", 100_000, 30 * 86400, now)
    keep = _file(tmp_path, "itf_train_v3b.csv", 100_000, 90 * 86400, now)
    (tmp_path / "catalogue_dir").mkdir()
    deleted, total = datadirs.trim_cache(now=now, max_mb=1.5, root=tmp_path)
    assert not expired.exists() and not a.exists()
    assert b.exists() and c.exists()
    assert catalogue.exists() and table.exists() and keep.exists()
    assert deleted == 2 and total <= 1.5e6


def test_trim_under_the_cap_only_removes_expired(tmp_path):
    now = time.time()
    fresh = _file(tmp_path, "ades_X.json", 10, 60, now)
    old = _file(tmp_path, "obs80_Y.json", 10, 4 * 3600, now)
    assert datadirs.trim_cache(now=now, max_mb=2000, root=tmp_path)[0] == 1
    assert fresh.exists() and not old.exists()


def test_the_default_cap_is_two_gigabytes():
    assert datadirs.CACHE_MAX_MB == float(os.environ.get("PYROORB_CACHE_MAX_MB") or 2000)


def test_a_missing_cache_directory_is_not_an_error(tmp_path):
    assert datadirs.trim_cache(root=tmp_path / "absent") == (0, 0)


def test_trims_are_throttled(monkeypatch):
    calls = []
    monkeypatch.setattr(datadirs, "trim_cache", lambda: calls.append(1))
    monkeypatch.setitem(datadirs._LAST_TRIM, "t", None)
    datadirs.note_cache_write()
    datadirs.note_cache_write()
    assert calls == [1]
    monkeypatch.setitem(datadirs._LAST_TRIM, "t", time.monotonic() - 2 * datadirs._TRIM_EVERY_S)
    datadirs.note_cache_write()
    assert calls == [1, 1]


@pytest.mark.parametrize("name", ["monitor_training.csv", "MPCORB.DAT.gz", "radar_astrometry.json"])
def test_names_outside_the_astrometry_prefixes_are_never_trimmed(tmp_path, name):
    now = time.time()
    p = _file(tmp_path, name, 3_000_000, 365 * 86400, now)
    datadirs.trim_cache(now=now, max_mb=1, root=tmp_path)
    assert p.exists()
