"""Unit tests for the Pluto ban.

Minor planet (134340) Pluto is one of the masses baked into the DE440 ephemeris
that drives our force model, so fitting its own astrometry would propagate it in a
field that already contains Pluto -- it would be perturbed by itself. We therefore
refuse it in every form a request can arrive in: the name 'Pluto', the number
134340 (incl. its packed forms 'D4340'/'00134340'), the discovery designation
'1930 BM' (packed 'J30B00M'), and the per-obs designation that comes back from a
fetch or a paste. These tests pin all of those forms, the look-alikes that must NOT
be banned, and the web payload the front end raises a banner from.
"""
import pytest
import pyroorb.fit_object as fo

from pyroorb import guards

# exercises the fit_orbit backstop
pytestmark = pytest.mark.slow
# Every form a Pluto request can arrive in -- name, number, packed number, name+number,
# parenthesised number, discovery provisional designation and its packed form.
BANNED = ["Pluto", "pluto", " PLUTO ", "134340", "00134340", "D4340",
          "134340 Pluto", "(134340)", "(134340) Pluto", "1930 BM", "J30B00M"]
# Look-alikes that must keep fitting: other objects, near-miss numbers, and a packed
# provisional that is NOT Pluto (guards against an over-broad number/regex match).
NOT_BANNED = ["2023 QJ3", "433", "1930 BX", "13434", "1343400", "1343401",
              "Eris", "136199", "2010 LP68", "K16UL0A", "", None]


def test_is_pluto_matches_every_form():
    for d in BANNED:
        assert guards.is_pluto(d), f"is_pluto should ban {d!r}"


def test_is_pluto_spares_lookalikes():
    for d in NOT_BANNED:
        assert not guards.is_pluto(d), f"is_pluto should NOT ban {d!r}"


def test_assert_not_pluto_catches_obs_self_identifying_by_number():
    """Astrometry that self-identifies as 134340 (by designation OR packed code) is
    banned even when the typed designation was something innocuous -- the 'comes back
    with Pluto's number some odd way' case."""
    obs = [{"designation": "2023 QJ3", "packed": "K23Q03J"},
           {"designation": "134340", "packed": "D4340"}]
    try:
        guards.assert_not_pluto(desig="2023 QJ3", obs=obs)
        assert False, "obs self-identifying as 134340 should raise"
    except guards.PlutoBanError as ex:
        assert "Pluto" in str(ex)


def test_assert_not_pluto_allows_clean_request():
    guards.assert_not_pluto(desig="2023 QJ3",
                        meta={"designation": "2023 QJ3", "designations": ["2023 QJ3"]},
                        obs=[{"designation": "2023 QJ3", "packed": "K23Q03J"}])


def test_fit_orbit_backstop_raises():
    """The hard backstop inside fit_orbit (the funnel every fit path goes through)
    refuses Pluto before doing any fit work."""
    try:
        fo.fit_orbit([], {}, desig="Pluto")
        assert False, "fit_orbit should refuse Pluto"
    except guards.PlutoBanError:
        pass


def test_web_run_fit_returns_banner_payload():
    from pyroorb.web import engine_api
    for d in ("Pluto", "134340", "D4340", "(134340) Pluto", "1930 BM"):
        r = engine_api.run_fit(d)
        assert r.get("ok") is False and r.get("pluto_banned") is True, f"{d!r}: {r}"
        assert r.get("banner") == guards.PLUTO_BAN_MSG


def test_web_fit_pasted_bans_pluto_obs80():
    """A pasted OBS80 blob whose packed number is Pluto's (D4340) is refused with the
    banner payload -- the 'someone pastes Pluto astrometry' case."""
    from pyroorb.web import engine_api; from pyroorb.web import loader
    line = ("D4340         C2023 08 28.360000 00 00 00.00 "
            "+00 00 00.0          20.0 GV~0000309").ljust(80)
    r = loader.fit_pasted(line, engine_api._obscodes())
    assert r.get("ok") is False and r.get("pluto_banned") is True, r
