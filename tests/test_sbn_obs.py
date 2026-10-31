"""Offline tests for the SBN-replica observation fallback (sbn_obs).

A fake cursor stands in for the replica: it answers the identification and
obs_sbn queries from small in-memory tables shaped like the real ones.
"""
import datetime
from decimal import Decimal

import pytest

from pyroorb import sbn_obs

# current_identifications: (packed secondary, unpacked secondary, packed primary, unpacked primary)
IDENT = [
    ("K10T19H", "2010 TH19", "K10T19H", "2010 TH19"),     # primary listed as its own secondary
    ("K05A01A", "2005 AA1", "K10T19H", "2010 TH19"),
    ("K25N32T", "2025 NT32", "K25N32T", "2025 NT32"),
]
# numbered_identifications: (permid, unpacked primary, iau_name)
NUMBERED = [
    ("499582", "2010 TH19", "Testname"),
    ("2P", "P/1818 W1", None),
]


def _row(obsid, provid=None, permid=None, status="P", **kw):
    r = {c: None for c in sbn_obs.ADES_COLUMNS}
    r.update(obsid=obsid, provid=provid, permid=permid, stn="F51",
             ra=Decimal("163.409487"), dec=Decimal("7.471219"), mag=Decimal("22.0"),
             obstime=datetime.datetime(2022, 1, 26, 12, 43, 45, 696000),
             obstime_text="2022-01-26T12:43:45.696Z", status=status)
    r.update(kw)
    return r


OBS = [
    _row("a1", permid="499582"),                          # numbered: provid NULL
    _row("a2", provid="2010 TH19", permid="499582"),
    _row("a3", provid="2005 AA1"),
    _row("a4", permid="499582", status="I"),              # not served by the API
    _row("b1", provid="2025 NT32", ref="   ", obstime_text=None),
    _row("c1", permid="2P"),
]


class FakeCursor:
    def __init__(self):
        self._rows, self.description = [], None

    def execute(self, sql, args):
        s = " ".join(sql.split())
        a = args[0]
        if "from numbered_identifications where iau_name" in s:
            self._rows = [(p, u) for p, u, n in NUMBERED if n == a]
        elif "from numbered_identifications where permid" in s:
            self._rows = [(u,) for p, u, n in NUMBERED if p == a]
        elif "from numbered_identifications where unpacked_primary" in s:
            self._rows = [(p,) for p, u, n in NUMBERED if u == a]
        elif "where packed_secondary_provisional_designation" in s:
            self._rows = [(up,) for ps, us, pp, up in IDENT if ps == a]
        elif "where packed_primary_provisional_designation" in s:
            self._rows = [(us,) for ps, us, pp, up in IDENT if pp == a]
        elif "from obs_sbn" in s:
            provids, permid, statuses = args
            cols = [c for c in sbn_obs.ADES_COLUMNS if c != "obstime"] + ["obstime", "obstime_text"]
            self.description = [(c,) for c in cols]
            self._rows = [tuple(r[c] for c in cols) for r in OBS
                          if (r["provid"] in provids or (permid and r["permid"] == permid))
                          and r["status"] in statuses]
        else:
            raise AssertionError(f"unexpected query: {s}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


@pytest.mark.parametrize("desig", ["499582", "Testname", "2005 AA1", "2010 TH19",
                                   "K05A01A"])
def test_every_name_of_a_numbered_family_resolves_to_all_of_it(desig):
    primary, permid, provids = sbn_obs.resolve_family(FakeCursor(), desig)
    assert (primary, permid) == ("2010 TH19", "499582")
    assert provids == ["2005 AA1", "2010 TH19"]


def test_numbered_rows_filed_only_under_permid_are_included():
    ids = {r["obsid"] for r in sbn_obs.family_records(FakeCursor(), "2005 AA1")}
    assert ids == {"a1", "a2", "a3"}          # a1 has no provid; a4 is status I


def test_numbered_comet_is_a_number():
    primary, permid, _ = sbn_obs.resolve_family(FakeCursor(), "2P")
    assert (primary, permid) == ("P/1818 W1", "2P")
    assert [r["obsid"] for r in sbn_obs.family_records(FakeCursor(), "2P")] == ["c1"]


def test_unidentified_designation_is_its_own_family():
    assert sbn_obs.resolve_family(FakeCursor(), "2030 ZZ9") == ("2030 ZZ9", None, ["2030 ZZ9"])


def test_records_are_rendered_as_the_api_serves_them():
    (r,) = sbn_obs.family_records(FakeCursor(), "2025 NT32")
    assert r["ra"] == "163.409487" and r["mag"] == "22.0"     # decimal text
    assert r["ref"] is None                                    # blank -> null
    assert r["obstime"] == "2022-01-26T12:43:45.696Z"          # rebuilt without obstime_text
    assert r["Obstype"] == "optical"
    assert "status" not in r


def test_fetch_falls_back_to_the_replica_when_the_api_fails(monkeypatch, obscodes):
    from pyroorb import ades
    import pyroorb.obs_sources as osrc

    def boom(*a, **k):
        raise TimeoutError("connection timed out")
    monkeypatch.setattr(ades, "fetch_combined_ades", boom)
    monkeypatch.setattr(ades, "fetch_combined_neocp", boom)
    monkeypatch.setattr(osrc, "fetch_combined", boom)
    monkeypatch.setenv("PYROORB_SBN_FALLBACK", "1")
    monkeypatch.setenv("SBN_DB_HOST", "replica.example")
    monkeypatch.setattr(sbn_obs, "fetch_records",
                        lambda d: sbn_obs.family_records(FakeCursor(), d))
    obs, meta = osrc._fetch_observations(["2005 AA1"], obscodes)
    assert len(obs) == 3
    assert meta["source"] == "SBN"
    assert not meta["fetch_errors"]           # the evidence is complete


def test_no_replica_keeps_the_transport_failure(monkeypatch, obscodes):
    from pyroorb import ades
    import pyroorb.obs_sources as osrc

    def boom(*a, **k):
        raise TimeoutError("connection timed out")
    monkeypatch.setattr(ades, "fetch_combined_ades", boom)
    monkeypatch.setattr(ades, "fetch_combined_neocp", boom)
    monkeypatch.setattr(osrc, "fetch_combined", boom)
    monkeypatch.setenv("PYROORB_SBN_FALLBACK", "0")
    obs, meta = osrc._fetch_observations(["2005 AA1"], obscodes)
    assert obs == [] and "2005 AA1" in meta["fetch_errors"]
