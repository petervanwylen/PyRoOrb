"""Nothing is written beside the source: every location comes from datadirs, which picks
the explicit variable, then a development checkout's existing data, then the platform's
per-user directory."""
from pathlib import Path

import pytest

from pyroorb import datadirs


def test_platform_directories():
    env = {}
    linux = datadirs.user_cache_dir("linux", env)
    assert linux.parts[-2:] == (".cache", "pyroorb")
    assert datadirs.user_cache_dir("linux", {"XDG_CACHE_HOME": "/x/c"}) == Path("/x/c/pyroorb")
    assert datadirs.user_data_dir("linux", {"XDG_DATA_HOME": "/x/d"}) == Path("/x/d/pyroorb")
    assert datadirs.user_cache_dir("darwin", env).parts[-3:] == ("Library", "Caches", "pyroorb")
    assert datadirs.user_data_dir("darwin", env).parts[-3:] == (
        "Library", "Application Support", "pyroorb")
    win = datadirs.user_cache_dir("win32", {"LOCALAPPDATA": "C:/Users/u/AppData/Local"})
    assert win == Path("C:/Users/u/AppData/Local/pyroorb/Cache")
    assert datadirs.user_data_dir("win32", {"APPDATA": "C:/R"}) == Path("C:/R/pyroorb")


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """An install-like layout: no historical directories, a private home."""
    monkeypatch.setattr(datadirs, "_CHECKOUT", tmp_path / "site-packages")
    monkeypatch.delenv(datadirs.CACHE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xc"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xd"))
    monkeypatch.setattr(datadirs.sys, "platform", "linux")
    return tmp_path


def test_an_install_uses_the_user_directories(fresh):
    assert datadirs.cache_dir() == fresh / "xc" / "pyroorb"
    assert datadirs.table_dir("ephem") == fresh / "xc" / "pyroorb" / "ephem"
    assert datadirs.persistent_default() == fresh / "xd" / "pyroorb" / "persistent"


def test_the_variable_wins(fresh, monkeypatch):
    monkeypatch.setenv(datadirs.CACHE_ENV, str(fresh / "mine"))
    assert datadirs.cache_dir() == fresh / "mine"
    assert datadirs.table_dir("debias") == fresh / "mine" / "debias"


def test_a_checkout_keeps_its_existing_tables_and_state(fresh):
    co = fresh / "site-packages"
    for d in (co / "ephem", co / ".local_persistent"):
        d.mkdir(parents=True)
    assert datadirs.cache_dir() == fresh / "xc" / "pyroorb"          # the cache never does
    assert datadirs.table_dir("ephem") == co / "ephem"
    assert datadirs.table_dir("debias") == fresh / "xc" / "pyroorb" / "debias"   # absent there
    assert datadirs.persistent_default() == co / ".local_persistent"


def test_nothing_is_created(fresh):
    datadirs.cache_dir(), datadirs.table_dir("ephem"), datadirs.persistent_default()
    assert not (fresh / "xc").exists() and not (fresh / "xd").exists()
