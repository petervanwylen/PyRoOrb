"""The MPC's ITF-to-ITF acceptance rules, as we read them.

These are somebody else's rules. The tests exist to pin OUR reading of them --
above all the counting, because a rule stated in "nights" and applied to
observations is not a stricter rule, it is a different one.
"""

import pytest

from pyroorb.itf import mpcrules


def _line(date, frac, ra="08 30 15.00", dec="+12 30 00.0", stn="F51"):
    """One OBS80 line on a given date; only the columns these rules read."""
    return (f"     K24A00A  C{date}{frac}{ra} {dec}          21.0 GV~1abc{stn}")


def _night(date, n=3):
    """A tracklet: n exposures ~17 minutes apart on one night."""
    return [_line(date, f"{0.20000 + 0.012 * i:.5f}"[1:]) for i in range(n)]


def _blob(nights):
    return "".join(l + "\n" for d in nights for l in d)


# ------------------------------------------------------------------ counting
def test_a_night_is_a_night_not_an_observation():
    """THE bug this module was born with: obs_night_ordinal is a fractional
    ordinal DAY, so three exposures 17 min apart are three distinct values of
    it.  Counting those as nights turned a 4-night linkage into a 13-night one
    and every rule below read the wrong number."""
    blob = _blob([_night("2024 01 15"), _night("2024 01 18"),
                  _night("2024 01 20"), _night("2024 01 22")])
    assert len(blob.splitlines()) == 12          # 12 observations...
    assert len(mpcrules.night_ordinals(blob)) == 4   # ...on 4 nights


def test_arc_is_measured_between_observations():
    blob = _blob([_night("2024 01 15"), _night("2024 01 22")])
    assert mpcrules.arc_days(blob) == pytest.approx(7.0, abs=0.1)


def test_apparitions_split_on_the_mpc_gap_not_ours():
    """The MPC says 237 days; itf.opp_profile.opp_profile says 275.  An arc between
    the two is TWO apparitions here and ONE there, and that is deliberate."""
    assert mpcrules.MPC_OPPOSITION_GAP_DAYS == 237
    near = _blob([_night("2024 01 15"), _night("2024 01 18"),
                  _night("2024 10 01")])            # 260 d gap
    assert mpcrules.oppositions(near) == [2, 1]     # MPC: two apparitions
    from pyroorb.itf.opp_profile import opp_profile, OppProfileGrade
    assert opp_profile(near).grade is OppProfileGrade.ONE                    # ours: still one


# --------------------------------------------------------------- night rules
def test_two_nights_is_refused_before_fitting():
    blob = _blob([_night("2024 01 15"), _night("2024 01 20")])
    ok, why = mpcrules.check(blob, rms_arcsec=0.1)
    assert not ok and any("distinct nights" in w for w in why)


def test_a_three_day_minimum_arc():
    blob = _blob([_night("2024 01 15"), _night("2024 01 16"),
                  _night("2024 01 17")])
    ok, why = mpcrules.check(blob, rms_arcsec=0.1)
    assert not ok and any("arc" in w for w in why)


def test_three_nights_may_not_sprawl():
    """3 nights over more than 15 days is a linkage nobody can check."""
    blob = _blob([_night("2024 01 15"), _night("2024 01 25"),
                  _night("2024 02 20")])
    ok, why = mpcrules.check(blob, rms_arcsec=0.1)
    assert not ok and any("3 nights spanning" in w for w in why)


# ---------------------------------------------------------- multi-apparition
def test_two_apparitions_need_three_nights_and_two():
    ok, why = mpcrules.mpc_opp_profile_ok(_blob(
        [_night("2024 01 15"), _night("2024 01 18"), _night("2024 01 21"),
         _night("2025 03 01"), _night("2025 03 04")]))
    assert ok, why
    # ...and one lone night in the second apparition is not enough
    ok2, _ = mpcrules.mpc_opp_profile_ok(_blob(
        [_night("2024 01 15"), _night("2024 01 18"), _night("2024 01 21"),
         _night("2025 03 01")]))
    assert not ok2


def test_five_apparitions_accept_two_pairs():
    """5+ apparitions: one with >=3 nights OR two with >=2 each."""
    nights = []
    for yr, days in ((2020, 2), (2021, 2), (2022, 1), (2023, 1), (2024, 1)):
        for k in range(days):
            nights.append(_night(f"{yr} 03 {10 + 3 * k:02d}"))
    ok, why = mpcrules.mpc_opp_profile_ok(_blob(nights))
    assert ok, why


# --------------------------------------------------------------- fit quality
def test_a_three_night_case_without_a_fit_is_refused_not_passed():
    """Untested is not the same as passed -- the 3-night orbit-quality test
    needs element uncertainties, so no fit means no."""
    blob = _blob([_night("2024 01 15"), _night("2024 01 18"),
                  _night("2024 01 20")])
    ok, why = mpcrules.check(blob)
    assert not ok and any("no fit" in w for w in why)


# ------------------------------------------------- the last word on submission
def _tracklet(date, start_frac, n=4):
    """A tracklet: n exposures ~2.5 minutes apart from ``start_frac`` of the day."""
    return [_line(date, f"{start_frac + 0.0017 * i:.5f}"[1:], stn="W84")
            for i in range(n)]


def test_tracklets_under_12_hours_apart_are_one_night():
    """K26E25U: a 4-night 2026 seed plus two W84 tracklets on
    2019-06-03, 01:09 and 04:16 UT, was offered as a submittable second night.
    Under 12 hours apart is ONE night to the MPC, so two apparitions of [1, 4]
    nights fail its 3+2 rule -- however well the tracklets fit."""
    seed = [_night("2026 03 14"), _night("2026 03 15"), _night("2026 03 17"),
            _night("2026 03 19")]
    same_night = _blob(seed + [_tracklet("2019 06 03", 0.048),
                               _tracklet("2019 06 03", 0.178)])
    assert mpcrules.oppositions(same_night) == [1, 4]
    ok, why = mpcrules.submittable_nights(same_night)
    assert not ok and "12 hours" in why

    # ...and the same two tracklets 13 hours apart ARE two nights
    apart = _blob(seed + [_tracklet("2019 06 02", 0.400),
                          _tracklet("2019 06 02", 0.400 + 13 / 24)])
    assert mpcrules.oppositions(apart) == [2, 4]
    assert mpcrules.submittable_nights(apart) == (True, "")


def test_the_12_hour_rule_reads_the_whole_time_field():
    """The night window is read off the full fractional day, not its first two
    decimals: truncated to 0.01 d (~14 min), a tracklet starting at .04990 and
    one wholly inside the next 12 h (.54000-.54510) read as .04 and .54, exactly
    12 h, and passed as two nights."""
    seed = [_night("2026 03 14"), _night("2026 03 15"), _night("2026 03 17"),
            _night("2026 03 19")]
    short = _blob(seed + [_tracklet("2019 06 03", 0.04990),
                          _tracklet("2019 06 03", 0.54000)])
    assert mpcrules.oppositions(short) == [1, 4]
    assert not mpcrules.submittable_nights(short)[0]
    # exactly 12 h apart is two nights (the window is half-open)
    twelve = _blob(seed + [_tracklet("2019 06 03", 0.04990),
                           _tracklet("2019 06 03", 0.54990)])
    assert mpcrules.oppositions(twelve) == [2, 4]


def test_one_apparition_is_left_to_the_other_gates():
    """An extension inside the seed's own apparition makes no cross-apparition
    claim, so the night guard has nothing to say about it."""
    blob = _blob([_night("2026 03 14"), _night("2026 03 15")])
    assert mpcrules.submittable_nights(blob) == (True, "")
