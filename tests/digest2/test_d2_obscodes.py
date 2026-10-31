from pathlib import Path

from pyroorb.digest2.obscodes import Site, SiteTable, load_obscodes, parse_cod3

DATA = Path(__import__("pyroorb.digest2", fromlist=["_"]).__file__).parent / "data" / "digest2.obscodes"


def test_parse_cod3_digits_and_letters():
    assert parse_cod3("000") == 0
    assert parse_cod3("500") == 500
    assert parse_cod3("G96") == 1696  # 'G' -> 16, 16*100 + 96
    assert parse_cod3("A00") == 1000  # 'A' -> 10


def test_parse_cod3_rejects_bad_input():
    assert parse_cod3("ab0") == -1  # lowercase not accepted
    assert parse_cod3("1x0") == -1
    assert parse_cod3("1") == -1


def test_site_table_unknown_code_is_geocentric_default():
    st = SiteTable()
    site = st[9999]
    assert site == Site()  # longitude=0, rho*=0, obs_err=UNSET


def test_site_table_set_obserr_creates_entry_if_missing():
    st = SiteTable()
    st.set_obserr(500, 0.5)
    assert st[500].obs_err == 0.5


def test_load_real_obscodes_file():
    sites = load_obscodes(DATA)
    assert len(sites) > 1000
    geocenter = sites[parse_cod3("500")]
    assert geocenter.rho_cos_phi == 0.0
    assert geocenter.rho_sin_phi == 0.0
    greenwich = sites[parse_cod3("000")]
    assert greenwich.longitude == 0.0
    assert greenwich.rho_cos_phi > 0
