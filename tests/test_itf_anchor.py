"""The anchor apparition (``itf.anchor``): an ITF linkage is submitted only when
one apparition has 3+ tracklets, and -- for a non-NEO -- 3+ nights."""

from types import SimpleNamespace

from pyroorb.itf import anchor, mpcrules
from pyroorb.itf.config import ItfConfig


def _f(frac):
    return f"{frac:.6f}"[1:]


def _tracklet(date, frac, stn, n=3):
    """n exposures ~10 minutes apart."""
    return [(f"     K13E43Z  C{date}{_f(frac + 0.007 * i)}"
             + "10 00 00.00 +10 00 00.0          21.0 G").ljust(77) + stn
            for i in range(n)]


def _blob(*tracklets):
    return "".join(l + "\n" for t in tracklets for l in t)


# The arc of 2013 EZ43 as published: a mislinkage of 1 + 2 + 1 nights over three
# apparitions (G96 2008-01-31; F51 2013-03-06 and G96 2013-03-13; W84 2014-04-24).
EZ43 = _blob(_tracklet("2008 01 31", 0.276, "G96", 4),
             _tracklet("2013 03 06", 0.414, "F51"),
             _tracklet("2013 03 13", 0.399, "G96", 4),
             _tracklet("2014 04 24", 0.409, "W84"))


def test_2013_EZ43_has_no_anchor_and_is_refused_neo_or_not():
    assert anchor.apparition_counts(anchor.points_of(EZ43)) == [(1, 1), (2, 2), (1, 1)]
    for neo in (False, True):
        why = anchor.anchor_problem(EZ43, neo=neo)
        assert why and "3+ tracklets" in why


def test_the_mpc_profile_rule_passes_what_the_anchor_refuses():
    """Six apparitions of two nights each satisfy the MPC's 5+ clause (two with
    >= 2 nights) and give no apparition three nights."""
    years = range(2010, 2016)
    blob = _blob(*[t for y in years for t in (_tracklet(f"{y} 03 10", 0.3, "F51"),
                                              _tracklet(f"{y} 03 14", 0.3, "G96"))])
    assert mpcrules.submittable_nights(blob) == (True, "")
    assert anchor.anchor_problem(blob, neo=False)
    # two tracklets per apparition is short of the tracklet floor for a NEO too
    assert "3+ tracklets" in anchor.anchor_problem(blob, neo=True)


def test_a_neo_anchors_on_three_tracklets_a_non_neo_needs_three_nights():
    """Three tracklets on two nights (two stations on the second night)."""
    blob = _blob(_tracklet("2020 10 01", 0.20, "F51"),
                 _tracklet("2020 10 02", 0.20, "F51"),
                 _tracklet("2020 10 02", 0.30, "G96"),
                 _tracklet("2023 05 12", 0.30, "F51"))
    assert anchor.apparition_counts(anchor.points_of(blob)) == [(3, 2), (1, 1)]
    assert anchor.anchor_problem(blob, neo=True) is None
    why = anchor.anchor_problem(blob, neo=False)
    assert why and "3+ nights" in why and "non-NEO" in why


def test_three_nights_in_one_apparition_anchor_a_non_neo():
    blob = _blob(_tracklet("2020 10 01", 0.20, "F51"),
                 _tracklet("2020 10 03", 0.20, "F51"),
                 _tracklet("2020 10 05", 0.20, "G96"),
                 _tracklet("2023 05 12", 0.30, "F51"))
    assert anchor.anchor_problem(blob, neo=False) is None


def test_one_stations_two_visits_in_a_night_are_one_tracklet():
    """Two visits by one station 3 h apart are one night and -- conservatively --
    one tracklet; a second station that night is a second tracklet."""
    one_stn = _blob(_tracklet("2020 10 01", 0.10, "F51"),
                    _tracklet("2020 10 01", 0.225, "F51"),
                    _tracklet("2020 10 05", 0.20, "F51"))
    assert anchor.apparition_counts(anchor.points_of(one_stn)) == [(2, 2)]
    two_stn = _blob(_tracklet("2020 10 01", 0.10, "F51"),
                    _tracklet("2020 10 01", 0.225, "G96"),
                    _tracklet("2020 10 05", 0.20, "F51"))
    assert anchor.apparition_counts(anchor.points_of(two_stn)) == [(3, 2)]


def test_a_single_apparition_must_anchor_itself():
    """The MPC's profile rule says nothing about one apparition; ours does."""
    blob = _blob(_tracklet("2020 10 01", 0.20, "F51"),
                 _tracklet("2020 10 04", 0.20, "F51"))
    assert mpcrules.submittable_nights(blob) == (True, "")
    assert anchor.anchor_problem(blob, neo=True)
    assert anchor.anchor_problem(blob, neo=False)


def test_the_second_line_of_a_satellite_record_is_not_another_observation():
    first = _tracklet("2020 10 01", 0.20, "C51", 1)[0]
    second = first[:14] + "s" + first[15:]
    assert len(anchor.points_of(first + "\n" + second + "\n")) == 1


def test_nights_agree_with_the_search_night_counter():
    """One definition of a night: the per-apparition night counts sum to
    ``opp_profile.count_nights`` on an arc of apparitions far apart."""
    from pyroorb.itf.opp_profile import count_nights
    for blob in (EZ43, _blob(_tracklet("2020 10 01", 0.04990, "W84"),
                             _tracklet("2020 10 01", 0.54990, "W84"),
                             _tracklet("2020 10 02", 0.30, "F51"))):
        assert (sum(n for _, n in anchor.apparition_counts(anchor.points_of(blob)))
                == count_nights(blob))


def test_neo_is_the_search_waiver_test_and_unreadable_elements_are_not_a_neo():
    cfg = ItfConfig()
    assert anchor.is_neo(cfg, 1.1, 0.5)
    assert not anchor.is_neo(cfg, 1.5, 0.5)
    assert not anchor.is_neo(cfg, 0.8, 1.2)              # a hyperbola's q is no evidence
    assert not anchor.is_neo(cfg, 0.8, float("nan"))
    assert not anchor.is_neo(cfg, None, None)
    from pyroorb.itf.search import _neo_orbit_waiver
    assert _neo_orbit_waiver(cfg, SimpleNamespace(q=1.1, e=0.5))
    assert not _neo_orbit_waiver(cfg, SimpleNamespace(q=1.1, e=float("nan")))


# ------------------------------------------------------------- in the search
def _ctx_trial(q, e):
    ctx = SimpleNamespace(cfg=ItfConfig(), primary_desigs=("2013 EZ43",),
                          pub_obs80="", log=lambda *_a: None)
    trial = SimpleNamespace(fit=None, q=q, e=e)
    result = SimpleNamespace(warnings=[], verdict=None, reason="")
    return ctx, trial, result


THIN6 = _blob(*[t for y in range(2010, 2016)
                for t in (_tracklet(f"{y} 03 10", 0.3, "F51"),
                          _tracklet(f"{y} 03 14", 0.3, "G96"))])


def test_the_search_files_an_unanchored_linkage_as_a_possible():
    """Six apparitions of two nights pass the MPC's rule; the anchor, asked
    last, keeps the find for review (0.6) and nothing is built or sent."""
    from pyroorb.itf.search import _document_arc
    from pyroorb.itf.verdict import SearchVerdict
    ctx, trial, result = _ctx_trial(1.1, 0.4)                 # even a NEO
    vetting = {}
    doc_c, doc_o, demoted = _document_arc(ctx, result, trial, THIN6, THIN6, vetting, [])
    assert doc_c is None and doc_o is None and demoted is result
    assert result.verdict is SearchVerdict.POSSIBLE
    assert vetting["not_submittable_no_anchor"] and vetting["possible_unconfirmed"]
    assert result.warnings[0].startswith("NOT SUBMITTABLE: no apparition has 3+ tracklets")


def test_the_search_lets_a_neo_anchor_on_tracklets_and_holds_a_non_neo_to_nights():
    """One apparition, which the MPC's profile rule leaves alone."""
    from pyroorb.itf.search import _document_arc
    blob = _blob(_tracklet("2020 10 01", 0.20, "F51"), _tracklet("2020 10 02", 0.20, "F51"),
                 _tracklet("2020 10 02", 0.30, "G96"), _tracklet("2020 10 09", 0.30, "F51"))
    ctx, trial, result = _ctx_trial(2.5, 0.1)                 # 3 nights: the control
    assert _document_arc(ctx, result, trial, blob, blob, {}, [])[2] is None
    # drop the third night: 3 tracklets on 2 nights
    thin = "".join(l + "\n" for l in blob.splitlines() if "2020 10 09" not in l)
    ctx, trial, result = _ctx_trial(1.1, 0.4)                 # a NEO
    assert _document_arc(ctx, result, trial, thin, thin, {}, [])[2] is None
    ctx, trial, result = _ctx_trial(2.5, 0.1)                 # a main-belt object
    vetting = {}
    assert _document_arc(ctx, result, trial, thin, thin, vetting, [])[2] is result
    assert vetting["not_submittable_no_anchor"]


def test_the_mpc_rule_already_refuses_2013_EZ43_and_the_anchor_would_too():
    assert not mpcrules.submittable_nights(EZ43)[0]
