"""Two answers from one ITF search: a submittable linkage and a fuller 0.6.

When the search holds a LINKED linkage and a reviewable 0.6 that links strictly
more of the ITF (2014 GV99: one same-opposition tracklet LINKED, against a 0.6
adding two single-night oppositions), it keeps both. ``ItfConfig.prefer`` picks
which leads and the other rides on ``ItfSearchResult.alternative``. The
consumers that submit (automated senders and the identification queue)
take the linkage; the command lines take the fuller one unless told otherwise
or asked to submit.
"""
from pyroorb.itf import search as S
from pyroorb.itf.config import ItfConfig, PREFER_CHOICES, default_prefer
from pyroorb.itf.verdict import SearchVerdict as V


def _line(trk, date, ra="10 00 00.00"):
    return f"     {trk}  C{date} {ra} +10 00 00.0          20.0 G      F51"


A = _line("PR00001", "2014 04 01.50000")
B = _line("PR00002", "2022 03 10.50000", "11 00 00.00")
C = _line("PR00003", "2006 05 02.50000", "12 00 00.00")


def _res(verdict, *lines, fit=True):
    return S.ItfSearchResult(verdict=verdict, itf_obs80="\n".join(lines) + "\n",
                             fit=object() if fit else None)


def test_fuller_possible_needs_a_strict_superset():
    linked = _res(V.LINKED, A)
    assert S._fuller_possible(linked, _res(V.POSSIBLE, A, B, C))
    # the same ITF obs: nothing gained, the linkage stands alone
    assert not S._fuller_possible(linked, _res(V.POSSIBLE, A))
    # a different tracklet set is a rival, not a fuller version of the linkage
    assert not S._fuller_possible(linked, _res(V.POSSIBLE, B, C))
    # a bare 0.6 (no fit) cannot be reviewed, so it is never offered
    assert not S._fuller_possible(linked, _res(V.POSSIBLE, A, B, fit=False))
    # the roles are fixed: a linkage on each side, or a 0.6 as the "linked" one
    assert not S._fuller_possible(linked, _res(V.LINKED, A, B))
    assert not S._fuller_possible(_res(V.POSSIBLE, A), _res(V.POSSIBLE, A, B))
    assert not S._fuller_possible(None, _res(V.POSSIBLE, A, B))
    assert not S._fuller_possible(linked, None)


def _paired(prefer):
    linked, possible = _res(V.LINKED, A), _res(V.POSSIBLE, A, B, C)
    return linked, possible, S._choose(ItfConfig(prefer=prefer), S._pair(linked, possible))


def test_inside_the_search_the_linkage_always_leads():
    # the search's own decisions (backtracking, the variant fallback fan) must
    # see the verdict they saw before the pair was kept
    linked, possible = _res(V.LINKED, A), _res(V.POSSIBLE, A, B, C)
    possible.alternative = _res(V.LINKED, A)            # an older pair
    got = S._pair(linked, possible)
    assert got is linked and got.verdict is V.LINKED
    assert got.alternative is possible and possible.alternative is None


def test_choose_leads_with_the_preferred_and_keeps_the_other():
    linked, possible, got = _paired("complete")
    assert got is possible and got.alternative is linked and linked.alternative is None
    linked, possible, got = _paired("submittable")
    assert got is linked and got.alternative is possible and possible.alternative is None


def test_choose_drops_a_pair_that_no_longer_holds():
    # the linkage grew after the pair was formed: the 0.6 is no longer fuller
    linked, possible = _res(V.LINKED, A), _res(V.POSSIBLE, A, B)
    S._pair(linked, possible)
    linked.itf_obs80 += C + "\n"
    got = S._choose(ItfConfig(prefer="complete"), linked)
    assert got is linked and got.alternative is None
    lone = _res(V.POSSIBLE, A, B)
    assert S._choose(ItfConfig(), lone) is lone


def test_alternative_note_names_the_other_result():
    assert S.alternative_note(_res(V.LINKED, A)) == ""
    _, _, lead = _paired("complete")
    assert "submittable linkage" in S.alternative_note(lead)
    assert "1 ITF obs" in S.alternative_note(lead)
    _, _, lead = _paired("submittable")
    assert "3 ITF obs" in S.alternative_note(lead)
    assert "unconfirmed" in S.alternative_note(lead)


def test_consumers_that_submit_take_the_linkage():
    assert ItfConfig().prefer == "complete"
    assert set(PREFER_CHOICES) == {"complete", "submittable"}
    assert default_prefer(False) == "complete"
    assert default_prefer(True) == "submittable"
