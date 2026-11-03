"""The SBN database host comes only from the environment.

There is no built-in server address: every operator runs its own SBN database
clone.  An unset SBN_DB_HOST must fail at connect with a message naming the
variable -- psycopg2 given no host would silently try a local Unix socket.
"""
import os
import pathlib
import re

import pytest

from pyroorb.itf import dbio
from pyroorb.itf.config import ItfConfig


@pytest.fixture
def env_only(monkeypatch):
    """Read settings from os.environ alone: no repo-root .env fallback."""
    monkeypatch.setattr(dbio, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(dbio, "env_default",
                        lambda name, fallback: os.environ.get(name, "").strip() or fallback)
    for var in ("SBN_DB_HOST", "SBN_PUB_PASSWORD", "ITF_DB_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _never_connect(*a, **k):
    raise AssertionError("must not reach psycopg2 without a host")


@pytest.mark.parametrize("connect", [dbio.connect_pub, dbio.connect_index])
def test_unset_host_refuses_before_connecting(env_only, connect):
    env_only.setattr(dbio, "_connect", _never_connect)
    cfg = ItfConfig()
    assert cfg.db_host is None and cfg.db_index_host is None
    with pytest.raises(RuntimeError, match="SBN_DB_HOST"):
        connect(cfg)


def test_host_from_environment_reaches_the_connection(env_only):
    env_only.setenv("SBN_DB_HOST", "db.example.org")
    env_only.setenv("SBN_PUB_PASSWORD", "x")
    env_only.setenv("ITF_DB_PASSWORD", "y")
    seen = []
    env_only.setattr(dbio, "_connect", lambda host, *a, **k: seen.append(host))
    cfg = ItfConfig()
    dbio.connect_pub(cfg)
    dbio.connect_index(cfg)
    assert seen == ["db.example.org", "db.example.org"]


def test_no_server_address_in_the_config_source():
    src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "pyroorb" / "itf" / "config.py").read_text(encoding="utf-8")
    assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", src)
