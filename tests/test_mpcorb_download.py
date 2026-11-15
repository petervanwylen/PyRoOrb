"""mpcorb.download_gz_atomic: concurrent refreshes of one gzip file never lose it, and a
failed or corrupt download leaves the existing copy untouched."""
import gzip
import threading

import pytest

from pyroorb import mpcorb


def _fake_download(payload):
    def download(url, dst, verbose=False):
        dst.write_bytes(payload)
        return dst
    return download


def test_concurrent_writers_each_use_a_private_temp_file(tmp_path):
    dst = tmp_path / "MPCORB.DAT.gz"
    good = gzip.compress(b"x" * 1000)
    barrier = threading.Barrier(3)
    errors, temps = [], []

    def download(url, tmp, verbose=False):
        temps.append(tmp.name)
        tmp.write_bytes(good)
        barrier.wait(timeout=10)             # all three downloads in flight together
        return tmp

    def run():
        try:
            mpcorb.download_gz_atomic("u", dst, download=download)
        except Exception as e:               # noqa: BLE001
            errors.append(e)
    ths = [threading.Thread(target=run) for _ in range(3)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert errors == [] and len(set(temps)) == 3
    assert gzip.decompress(dst.read_bytes()) == b"x" * 1000
    assert sorted(p.name for p in tmp_path.iterdir()) == ["MPCORB.DAT.gz"]


def test_a_corrupt_download_keeps_the_old_copy_and_leaves_no_temp(tmp_path):
    dst = tmp_path / "MPCORB.DAT.gz"
    dst.write_bytes(gzip.compress(b"old"))
    with pytest.raises(Exception):
        mpcorb.download_gz_atomic("u", dst, download=_fake_download(b"not gzip at all"))
    assert gzip.decompress(dst.read_bytes()) == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["MPCORB.DAT.gz"]

    def fails(url, tmp, verbose=False):
        tmp.write_bytes(b"partial")
        raise RuntimeError("download failed at byte 4194304")
    with pytest.raises(RuntimeError):
        mpcorb.download_gz_atomic("u", dst, download=fails)
    assert gzip.decompress(dst.read_bytes()) == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["MPCORB.DAT.gz"]
