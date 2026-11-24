"""datadirs.replace_file: the atomic rename every concurrent writer uses, retried while Windows
refuses a rename onto a target another writer or a reader is busy with."""
import os

import pytest

from pyroorb import datadirs


def _flaky_replace(monkeypatch, refusals):
    """os.replace that refuses ``refusals`` times (as Windows does while the target is busy)."""
    real = os.replace
    calls = []

    def replace(src, dst):
        calls.append(1)
        if len(calls) <= refusals:
            raise PermissionError(13, "Access is denied")
        return real(src, dst)
    monkeypatch.setattr(datadirs.os, "replace", replace)
    return calls


def test_it_replaces_the_target(tmp_path):
    src, dst = tmp_path / "new.part", tmp_path / "f"
    dst.write_text("old")
    src.write_text("new")
    datadirs.replace_file(src, dst)
    assert dst.read_text() == "new" and not src.exists()


def test_windows_waits_out_a_busy_target(tmp_path, monkeypatch):
    calls = _flaky_replace(monkeypatch, refusals=3)
    src, dst = tmp_path / "new.part", tmp_path / "f"
    src.write_text("new")
    datadirs.replace_file(src, dst, _platform="win32")
    assert dst.read_text() == "new" and len(calls) == 4


def test_windows_gives_up_after_its_budget(tmp_path, monkeypatch):
    _flaky_replace(monkeypatch, refusals=10 ** 6)
    src = tmp_path / "new.part"
    src.write_text("new")
    with pytest.raises(PermissionError):
        datadirs.replace_file(src, tmp_path / "f", retry_sec=0.05, _platform="win32")


def test_elsewhere_a_permission_error_is_real(tmp_path, monkeypatch):
    calls = _flaky_replace(monkeypatch, refusals=1)
    src = tmp_path / "new.part"
    src.write_text("new")
    with pytest.raises(PermissionError):
        datadirs.replace_file(src, tmp_path / "f", _platform="linux")
    assert len(calls) == 1


def test_concurrent_mpcorb_refreshes_all_succeed_on_windows(tmp_path, monkeypatch):
    """The CI failure: three threads refresh MPCORB at once, and Windows refuses a rename
    onto a target another rename is busy with. Each refresh must still succeed."""
    import gzip
    import threading
    import time
    from pyroorb import mpcorb
    real = os.replace
    busy = threading.Lock()

    def windows_replace(src, dst):
        if not busy.acquire(blocking=False):
            raise PermissionError(13, "Access is denied")
        try:
            time.sleep(0.05)                   # the rename takes a moment
            return real(src, dst)
        finally:
            busy.release()
    monkeypatch.setattr(datadirs.os, "replace", windows_replace)
    monkeypatch.setattr(datadirs.sys, "platform", "win32")
    dst = tmp_path / "MPCORB.DAT.gz"
    good = gzip.compress(b"x" * 1000)
    barrier = threading.Barrier(3)
    errors = []

    def download(url, tmp, verbose=False):
        tmp.write_bytes(good)
        barrier.wait(timeout=10)
        return tmp

    def run():
        try:
            mpcorb.download_gz_atomic("u", dst, download=download)
        except Exception as e:                 # noqa: BLE001
            errors.append(e)
    ths = [threading.Thread(target=run) for _ in range(3)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert errors == [] and gzip.decompress(dst.read_bytes()) == b"x" * 1000
    assert not list(tmp_path.glob("*.part"))
