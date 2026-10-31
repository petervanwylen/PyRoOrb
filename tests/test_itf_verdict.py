"""The two enums that replaced the ITF search's float codes.

`SearchVerdict` replaced ``ItfSearchResult.status`` (1 / 0.6 / 0) and
`OppProfile` / `OppProfileGrade` replaced ``count_opps``' 1.5 / 1.75. Neither
may be compared with a number (a missed site must fail loudly, not misjudge
every result), and both still WRITE the old float wherever a file on disk
holds it -- which is what the store-compatibility checks below
pin."""

import pytest

from pyroorb.itf.opp_profile import OppProfile, OppProfileGrade as G, opp_profile
from pyroorb.itf.search import ItfSearchResult
from pyroorb.itf.verdict import SearchVerdict as V


def _line(date):
    return (f"     PR00001  C{date} 10 00 00.00 +10 00 00.0"
            f"          20.0 G      F51")


def test_verdict_legacy_round_trip_and_rank():
    assert [v.legacy_status for v in (V.NONE, V.POSSIBLE, V.LINKED)] == [0.0, 0.6, 1.0]
    for v in V:
        assert V.from_legacy(v.legacy_status) is v
    assert V.NONE.rank < V.POSSIBLE.rank < V.LINKED.rank
    # the value IS the word stored batch results already carry as `verdict`
    assert [v.value for v in V] == ["none", "possible", "linked"]
    with pytest.raises(ValueError):
        V.from_legacy(0.5)


def test_a_verdict_is_not_a_number():
    with pytest.raises(TypeError):
        V.LINKED == 1
    with pytest.raises(TypeError):
        V.POSSIBLE == 0.6
    assert ItfSearchResult().verdict is V.NONE


@pytest.mark.parametrize("dates,grade,legacy", [
    (["2015 01 01.1"], G.ONE, 1),
    (["2015 01 01.1", "2015 01 03.1", "2016 01 01.1"], G.TWO_THIN, 1.5),
    (["2015 01 01.1", "2016 01 01.1"], G.TWO, 2),          # neither multi-night: 2, not 1.5
    (["2015 01 01.1", "2015 01 03.1", "2016 01 01.1", "2016 01 03.1"], G.TWO, 2),
    (["2015 01 01.1", "2015 01 03.1", "2016 01 01.1", "2017 01 01.1"], G.THREE_THIN, 1.75),
    (["2015 01 01.1", "2015 01 03.1", "2016 01 01.1", "2016 01 03.1",
      "2017 01 01.1"], G.MULTI, 3),
])
def test_opp_profile_grades_match_the_legacy_codes(dates, grade, legacy):
    p = opp_profile("\n".join(_line(d + "0000") for d in dates))
    assert p.grade is grade
    assert p.legacy_code == legacy and type(p.legacy_code) is type(legacy)
    # it FORMATS as the legacy code, so every message and stored reason reads
    # exactly as it did before the rename
    assert f"{p} opps" == f"{legacy} opps" and f"{p:g}" == f"{legacy:g}"


def test_an_opp_profile_is_not_a_number_and_confirms_to_three():
    p = OppProfile(3, 1)
    for op in (lambda: p == 1.75, lambda: p >= 3, lambda: p < 2):
        with pytest.raises(TypeError):
            op()
    c = p.confirmed()
    assert c.grade is G.MULTI and c.legacy_code == 3 and not c.is_thin
    assert p.is_thin and OppProfile(2, 1).is_thin and not OppProfile(2, 2).is_thin


