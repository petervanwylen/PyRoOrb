"""The station table: a copy ships with the package, a newer cached copy wins, and a
download is tried only when it can matter -- a station the table lacks (at most once per
process per interval), the server's periodic re-check, or an explicit forced refresh.
Downloads try the MPC first, find_orb's mirror as fallback, and are validated before they
replace the cache; a changed table reaches the in-process caches (obs_sources.
refresh_obscodes / obscodes_token, engine_api._obscodes, fitqueue.fit_key)."""
import io

import pytest

from pyroorb import obs_sources

MPC, FINDORB = obs_sources.OBSCODES_URLS
NEW_STATION = "Z99 359.978740.595468+0.800687New Station"


def _table(n, extra=()):
    rows = ["<pre>", "Code  Long.   cos      sin    Name"]
    for i in range(n):
        rows.append(f"{i:03d}   1.0000 0.62411 +0.77873 Station {i}")
    rows += list(extra)
    rows.append("</pre>")
    return ("\n".join(rows) + "\n").encode()


class _Net:
    """urlopen stand-in: url -> bytes, or an Exception to raise."""
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        page = self.pages.get(url, OSError("unreachable"))
        if isinstance(page, Exception):
            raise page
        return io.BytesIO(page)


@pytest.fixture
def table(tmp_path, monkeypatch):
    """(cache path, net installer). The bundled table lists stations 000-099."""
    path = tmp_path / "cache" / "ObsCodes.htm"
    bundled = tmp_path / "bundled" / "ObsCodes.htm"
    bundled.parent.mkdir()
    bundled.write_bytes(_table(100))
    monkeypatch.setattr(obs_sources, "OBSCODES_CACHE", path)
    monkeypatch.setattr(obs_sources, "BUNDLED_OBSCODES", bundled)
    monkeypatch.setattr(obs_sources, "_OBSCODES_RESOLVED", None)
    monkeypatch.setattr(obs_sources, "_OBSCODES_TOKEN", None)
    monkeypatch.setattr(obs_sources, "_OBSCODES_MISS_AT", None)
    monkeypatch.setattr(obs_sources, "OBSCODES_MIN_STATIONS", 50)
    monkeypatch.delenv("PYROORB_OBSCODES_REFRESH", raising=False)

    def net(pages):
        n = _Net(pages)
        monkeypatch.setattr(obs_sources.urllib.request, "urlopen", n)
        return n
    return path, net


def test_the_bundled_table_is_the_real_one():
    """What ships must parse as the MPC table (the release step re-bundles it)."""
    text = obs_sources.BUNDLED_OBSCODES.read_text(encoding="utf-8", errors="ignore")
    assert len(obs_sources._station_codes(text)) >= 1500
    assert {"500", "568", "F51", "G96", "703"} <= set(obs_sources._station_codes(text))


def test_first_use_reads_the_bundled_table_without_a_download(table):
    path, net = table
    n = net({MPC: _table(120)})
    assert obs_sources.ensure_obscodes_file() == obs_sources.BUNDLED_OBSCODES
    assert n.calls == [] and len(obs_sources.load_obscodes()) == 100


def test_a_cached_copy_wins_over_the_bundled_one(table):
    path, net = table
    path.parent.mkdir(parents=True)
    path.write_bytes(_table(100, [NEW_STATION]))
    n = net({})
    assert obs_sources.ensure_obscodes_file() == path
    assert "Z99" in obs_sources.load_obscodes() and n.calls == []


def test_a_forced_refresh_downloads_at_first_use(table, monkeypatch):
    path, net = table
    monkeypatch.setenv("PYROORB_OBSCODES_REFRESH", "1")
    n = net({MPC: _table(100, [NEW_STATION])})
    assert obs_sources.ensure_obscodes_file() == path
    assert n.calls == [MPC] and "Z99" in obs_sources.load_obscodes()


def test_a_failed_forced_refresh_falls_back_to_the_bundled_table(table, monkeypatch):
    path, net = table
    monkeypatch.setenv("PYROORB_OBSCODES_REFRESH", "1")
    net({})
    assert obs_sources.ensure_obscodes_file() == obs_sources.BUNDLED_OBSCODES


def test_find_orb_only_when_the_mpc_fails(table):
    path, net = table
    n = net({MPC: OSError("down"), FINDORB: _table(100, [NEW_STATION])})
    r = obs_sources.refresh_obscodes()
    assert n.calls == [MPC, FINDORB] and r["source"] == FINDORB
    assert "Z99" in obs_sources.load_obscodes()


def test_an_error_page_from_the_mpc_is_not_the_table(table):
    path, net = table
    net({MPC: b"<html>Service Unavailable</html>", FINDORB: _table(100, [NEW_STATION])})
    r = obs_sources.refresh_obscodes()
    assert r["source"] == FINDORB and "Z99" in obs_sources.load_obscodes()


def test_refresh_picks_up_a_new_station_and_moves_the_token(table):
    path, net = table
    net({})
    tok0 = obs_sources.obscodes_token()
    net({MPC: _table(100, [NEW_STATION])})
    r = obs_sources.refresh_obscodes()
    assert r["changed"] and r["added"] == ["Z99"] and r["source"] == MPC
    assert "Z99" in obs_sources.load_obscodes()
    assert obs_sources.obscodes_token() != tok0


def test_an_unchanged_table_is_not_rewritten(table):
    path, net = table
    net({MPC: _table(100, [NEW_STATION])})
    obs_sources.refresh_obscodes()
    tok0 = obs_sources.obscodes_token()
    mtime = path.stat().st_mtime_ns
    r = obs_sources.refresh_obscodes()
    assert not r["changed"] and r["error"] is None
    assert path.stat().st_mtime_ns == mtime and obs_sources.obscodes_token() == tok0


def test_a_failed_or_shrunken_download_keeps_the_cached_table(table):
    path, net = table
    net({MPC: _table(100, [NEW_STATION])})
    obs_sources.refresh_obscodes()
    before = path.read_bytes()
    net({})                                                   # both sources down
    r = obs_sources.refresh_obscodes()
    assert r["error"] and not r["changed"]
    net({MPC: _table(60), FINDORB: _table(70)})               # truncated / shrunken
    r = obs_sources.refresh_obscodes()
    assert r["error"] and not r["changed"]
    assert path.read_bytes() == before


def test_never_means_no_download_of_any_kind(table, monkeypatch):
    path, net = table
    monkeypatch.setenv("PYROORB_OBSCODES_REFRESH", "never")
    n = net({MPC: _table(100, [NEW_STATION])})
    r = obs_sources.refresh_obscodes()
    assert not r["changed"] and "never" in r["error"]
    assert obs_sources.refresh_on_miss({"Z99"}) == set()
    assert n.calls == [] and not path.exists()


def test_a_miss_refreshes_once_per_interval(table, monkeypatch):
    path, net = table
    n = net({MPC: _table(100)})                               # the MPC does not list it yet
    assert obs_sources.refresh_on_miss({"Z99"}) == set()
    assert n.calls == [MPC] and not path.exists()             # nothing new: cache untouched
    n = net({MPC: _table(100, [NEW_STATION])})
    assert obs_sources.refresh_on_miss({"Z99"}) == set()      # within the interval: no call
    assert n.calls == []
    monkeypatch.setattr(obs_sources, "_OBSCODES_MISS_AT", None)   # the interval has passed
    assert obs_sources.refresh_on_miss({"Z99", "Q00"}) == {"Z99"}
    assert "Z99" in obs_sources.load_obscodes()


def test_a_fetch_naming_a_new_station_resolves_it_after_one_refresh(table, monkeypatch):
    """The caller's table is updated in place and the observation is not reported missing."""
    path, net = table
    net({MPC: _table(100, [NEW_STATION])})
    calls = []

    def fake_fetch(desigs, obscodes, cache_dir=None, prefer_ades=True):
        calls.append(1)
        if "Z99" in obscodes:
            return [{"obscode": "Z99"}], {}
        obs_sources.record_unknown_obscode("Z99")
        return [], {}

    monkeypatch.setattr(obs_sources, "_fetch_observations", fake_fetch)
    monkeypatch.setattr(obs_sources, "dedup_observations", lambda obs: (obs, []))
    from pyroorb import radar
    monkeypatch.setattr(radar, "attach_radar", lambda meta, cache_dir=None: None)
    obs_sources.pop_unknown_obscodes()
    oc = obs_sources.load_obscodes()
    obs, _meta = obs_sources.fetch_observations(["2099 AA"], oc)
    assert obs == [{"obscode": "Z99"}] and len(calls) == 2 and "Z99" in oc
    assert obs_sources.pop_unknown_obscodes() == {}


def test_offline_a_missing_station_is_still_reported(table, monkeypatch):
    path, net = table
    net({})

    def fake_fetch(desigs, obscodes, cache_dir=None, prefer_ades=True):
        obs_sources.record_unknown_obscode("Z99")
        return [], {}

    monkeypatch.setattr(obs_sources, "_fetch_observations", fake_fetch)
    obs_sources.pop_unknown_obscodes()
    obs_sources.fetch_observations(["2099 AA"], obs_sources.load_obscodes())
    assert obs_sources.pop_unknown_obscodes() == {"Z99": 1}


def test_engine_cache_and_fit_key_follow_a_refresh(table):
    from pyroorb.web import engine_api, fitqueue
    path, net = table
    net({})
    engine_api._OBSCODES = engine_api._OBSNAMES = None
    try:
        assert "Z99" not in engine_api._obscodes()
        k0 = fitqueue.fit_key(["2023 QJ3"], False)
        net({MPC: _table(100, [NEW_STATION])})
        obs_sources.refresh_obscodes()
        assert "Z99" in engine_api._obscodes()
        assert engine_api._obsnames()["Z99"] == "New Station"
        assert fitqueue.fit_key(["2023 QJ3"], False) != k0
    finally:
        engine_api._OBSCODES = engine_api._OBSNAMES = None
