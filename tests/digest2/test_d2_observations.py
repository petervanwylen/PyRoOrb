import math

from pyroorb.digest2.observations import (
    date_to_mjd, parse_mpc80, update_magnitude,
)


def test_date_to_mjd_matches_known_value():
    # 2016-12-25.5 -> MJD 57747.5 (independently known reference point)
    assert abs(date_to_mjd(2016, 12, 25.5) - 57747.5) < 1e-9


def test_date_to_mjd_handles_january_correctly():
    # regression check for the C-truncation-toward-zero quirk in the
    # (month - 14) // 12 term
    mjd_jan1 = date_to_mjd(2020, 1, 1.0)
    mjd_dec31_prior_year = date_to_mjd(2019, 12, 31.0)
    assert mjd_jan1 == mjd_dec31_prior_year + 1


def test_update_magnitude_v_band_is_unchanged():
    assert update_magnitude("V", 20.0) == 20.0


def test_update_magnitude_applies_band_correction():
    assert update_magnitude("R", 20.0) == 20.4
    assert update_magnitude("g", 20.0) == 20.0 - 0.28


def test_update_magnitude_zero_mag_untouched():
    assert update_magnitude("R", 0.0) == 0.0


SAMPLE_LINE = (
    "     K16S99K 1C2022 12 25.38496508 32 36.283+17 10 35.94         21.98GV     G96"
)


def test_parse_mpc80_sample_line():
    result = parse_mpc80(SAMPLE_LINE)
    assert result is not None
    desig, obs = result
    assert desig == "     K16S99K"
    assert obs.obscode == "G96"
    assert abs(obs.mjd - 59938.384965) < 1e-9
    # RA: 08h 32m 36.283s -> radians
    expected_ra = math.radians((8 + 32 / 60 + 36.283 / 3600) * 15)
    assert abs(obs.ra - expected_ra) < 1e-9
    # Dec: +17 10 35.94
    expected_dec = math.radians(17 + 10 / 60 + 35.94 / 3600)
    assert abs(obs.dec - expected_dec) < 1e-9
    assert abs(obs.vmag - (21.98 + 0.24)) < 1e-9  # band 'G' -> +0.24 correction


def test_parse_mpc80_rejects_short_line():
    assert parse_mpc80("too short") is None


def test_parse_mpc80_rejects_bad_note2():
    # note2 field (index 14) must be C/S/B; corrupt it
    bad = SAMPLE_LINE[:14] + "Z" + SAMPLE_LINE[15:]
    assert parse_mpc80(bad) is None
