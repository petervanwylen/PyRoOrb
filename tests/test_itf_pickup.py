"""Partial pickup of a designated (pseudo-ITF) object.

The defect these pin, from the live case that found it: 2024 TF21 linked ONE
tracklet of 2017 SS396, and the document then claimed the whole designation --
whose other two tracklets sit a degree off the linked orbit.  What must hold
is that the search notices, says so in a caution a person can act on, and
never NAMES that designation in a submission document.
"""

import pytest

from pyroorb.itf import pickup


def _line(ident, date, frac, ra="23 36 57.750", dec="+05 30 00.00", stn="F51"):
    """One OBS80 line, built by COLUMN (cols 6-12 the id, 16-32 the date,
    33-44 RA, 45-56 Dec, 78-80 the station)."""
    line = (f"{'':5s}{ident:<7s} " f" C" f"{date}{frac}"
            f"{ra:<12s}{dec:<12s}" f"{'':9s}{'21.0':>5s}V" f"{'':6s}{stn:>3s}")
    assert len(line) == 80, len(line)
    return line


def _tracklet(ident, date, stn="F51", n=3):
    return [_line(ident, date, f"{0.200000 + 0.012 * i:.6f}"[1:], stn=stn)
            for i in range(n)]


def _rows(trkid, lines, provid, trksub):
    return [{"obs80": l, "trkid": trkid, "provid": provid, "trksub": trksub}
            for l in lines]


class _Source:
    """A candidate source that can answer for a designation's whole archive."""

    def __init__(self, rows):
        self._rows = rows

    def designation_obs(self, provids):
        return [r for r in self._rows if r["provid"] in set(provids)]


# 2017 SS396's three tracklets, as the archive holds them.
_ARCHIVE = (_rows("t1", _tracklet("K17Sd6S", "2017 09 21"), "2017 SS396", "P10DqG7")
            + _rows("t2", _tracklet("K17Sd6S", "2017 09 24"), "2017 SS396", "P10DEzB")
            + _rows("t3", _tracklet("K17Sd6S", "2017 10 15", stn="G96"),
                    "2017 SS396", "YT821F3"))

# ...and what a search linked: the 10-15 G96 tracklet, and nothing else of it.
_LINKED = "".join(l + "\n" for l in _tracklet("K17Sd6S", "2017 10 15", stn="G96"))


@pytest.fixture()
def offsets(monkeypatch):
    """Stub the ephemeris test: ``{station: (offset_arcsec, sigma)}``."""
    table = {}

    def _fake(fit, obscodes, lines):
        return table.get(lines[0][77:80], (None, None))

    monkeypatch.setattr(pickup, "_offset_arcsec", _fake)
    return table


# ------------------------------------------------------------ what is a pickup
def test_only_designated_lines_are_pickups():
    blob = (_LINKED
            + "".join(l + "\n" for l in _tracklet("P10DqFA", "2017 09 21")))
    picked = pickup.designated_pickups(blob)
    assert list(picked) == ["K17Sd6S"]          # the trksub is not a pickup
    assert len(picked["K17Sd6S"]) == 3


def test_the_objects_own_family_is_not_a_pickup():
    """A designation the seed already owns is the object, not a discovery --
    both spellings, since the ban set carries packed and unpacked."""
    assert pickup.designated_pickups(_LINKED, own_desigs={"2017 SS396"}) == {}
    assert pickup.designated_pickups(_LINKED, own_desigs={"K17Sd6S"}) == {}


# ------------------------------------------------------------------ the verdict
def test_leftovers_off_the_orbit_condemn_the_designation(offsets):
    offsets["F51"] = (3779.0, 0.08)             # the two that do not belong
    recs = pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(),
                         obscodes={}, cfg=None)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["desig"] == "2017 SS396" and rec["n_archive"] == 3
    assert len(rec["linked"]) == 1 and len(rec["missing"]) == 2
    assert rec["compromised"] is True
    assert pickup.banned_designations(recs) == {"K17Sd6S"}
    assert "Jira" in rec["warning"] and "2017 SS396" in rec["warning"]
    assert "1.05 deg off" in rec["warning"]     # 3779" read as degrees


def test_leftovers_that_fit_are_recorded_but_not_a_caution(offsets):
    """The ordinary part-take: the corridor never swept the other nights.  The
    designation IS this object, claiming it is right, and a click-through here
    would be spent on nothing."""
    offsets["F51"] = (0.4, 0.08)
    recs = pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(),
                         obscodes={}, cfg=None)
    assert len(recs) == 1                       # still recorded in the vetting
    assert recs[0]["compromised"] is False
    assert pickup.banned_designations(recs) == set()


def test_an_untestable_leftover_is_not_a_licence_to_claim(offsets):
    """No ephemeris, no evidence -- and no evidence is not permission."""
    recs = pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(),
                         obscodes={}, cfg=None)   # offsets table left empty
    assert recs[0]["compromised"] is True
    assert "not testable" in recs[0]["warning"]


def test_an_untestable_leftover_is_not_reported_as_a_misfit(offsets):
    """...and the SENTENCE has to say which of the two it is.  A tracklet we
    could not measure was being announced as one that 'do NOT fit the linked
    orbit -- so only part of 2017 SS396 is this object', which accuses a
    designation of being a mislinkage on a measurement that never happened."""
    recs = pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(),
                         obscodes={}, cfg=None)   # nothing measurable
    warn = recs[0]["warning"]
    assert "COULD NOT CHECK" in warn and "could not be measured" in warn
    assert "do NOT fit" not in warn and "MISLINKED" not in warn
    # ...while a leftover that really missed still says so, and a mixed set
    # reports each on its own terms.
    offsets["F51"] = (3779.0, 0.08)
    warn = pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(),
                         obscodes={}, cfg=None)[0]["warning"]
    assert "MISLINKED" in warn and "do NOT fit" in warn


@pytest.mark.parametrize("stn, obscodes, ok", [
    ("F51", {"F51": (243.14, 0.8484, +0.5283)}, True),    # a real ground site
    ("500", {}, True),                                    # the geocentre
    ("C51", {"F51": (243.14, 0.8484, +0.5283)}, False),   # WISE: not a station
    ("247", {"247": (0.0, 0.0, 0.0)}, False),             # roving: no fixed site
])
def test_an_observer_we_cannot_place_is_untestable(stn, obscodes, ok):
    """The ephemeris does not REFUSE a satellite or roving code -- it falls back
    to a geocentre placeholder -- so comparing its astrometry with a geocentric
    prediction reads as a parallax offset of ~12" for a main-belt target and far
    more for a NEO, i.e. past pickup_off_arcsec on geometry alone.  Ask first."""
    assert pickup._placeable(stn, obscodes) is ok


def test_a_whole_take_is_not_a_partial_pickup(offsets):
    """Every tracklet of the designation linked: nothing to warn about."""
    whole = "".join(
        l + "\n" for t in (_tracklet("K17Sd6S", "2017 09 21"),
                           _tracklet("K17Sd6S", "2017 09 24"),
                           _tracklet("K17Sd6S", "2017 10 15", stn="G96"))
        for l in t)
    assert pickup.review(whole, _Source(_ARCHIVE), fit=object(),
                         obscodes={}, cfg=None) == []


def _same_night_tracklet(ident, date, frac0, stn="G96", n=3):
    """A tracklet whose observations start at day-fraction ``frac0`` -- so two of
    them on the same night+station can be built minutes apart."""
    return [_line(ident, date, f"{frac0 + 0.001 * i:.6f}"[1:], stn=stn)
            for i in range(n)]


def test_a_same_night_leftover_is_not_absorbed_by_the_linked_tracklet(offsets):
    """The bug the epoch key fixes: two tracklets of one designation taken at the
    SAME station on the SAME UT night, only one linked, the other off the orbit.

    Under the old calendar-day key the off-orbit tracklet inherited the linked
    one's day and was filed as linked -- no missing tracklet, no record, and the
    mislinked designation claimed whole.  The epoch key separates them, so the
    leftover is measured, fails, and condemns the designation."""
    linked_trk = _same_night_tracklet("K17Sd6S", "2017 10 15", 0.20)
    off_trk = _same_night_tracklet("K17Sd6S", "2017 10 15", 0.40)   # ~4.8 h later
    archive = (_rows("s1", linked_trk, "2017 SS396", "G96aaaa")
               + _rows("s2", off_trk, "2017 SS396", "G96bbbb"))
    linked = "".join(l + "\n" for l in linked_trk)
    offsets["G96"] = (3779.0, 0.08)              # what the leftover reads
    recs = pickup.review(linked, _Source(archive), fit=object(),
                         obscodes={}, cfg=None)
    assert len(recs) == 1, "the same-night leftover was silently absorbed"
    rec = recs[0]
    assert len(rec["linked"]) == 1 and len(rec["missing"]) == 1
    assert rec["missing"][0]["trkid"] == "s2"
    assert rec["compromised"] is True
    assert pickup.banned_designations(recs) == {"K17Sd6S"}


def test_a_re_rendered_linked_line_still_matches_its_own_tracklet(offsets):
    """The epoch key must keep doing what the day key was there for: match a
    linked line to its archive tracklet even when the web re-render changed the
    SKY POSITION (so obs_unique_id misses).  Same time, different RA/Dec -> still
    linked, and a whole take is still not a pickup."""
    archive_trk = _same_night_tracklet("K17Sd6S", "2017 10 15", 0.20)
    archive = _rows("s1", archive_trk, "2017 SS396", "G96aaaa")
    # the linked lines: identical epochs, DIFFERENT positions (a re-render)
    rerendered = [l[:32] + "01 02 03.456" + "+04 05 06.78" + l[56:] + "\n"
                  for l in archive_trk]
    for l in rerendered:
        assert len(l) == 81                       # 80 + newline
    recs = pickup.review("".join(rerendered), _Source(archive), fit=object(),
                         obscodes={}, cfg=None)
    assert recs == [], "a re-rendered whole take must not read as a pickup"


def test_a_source_that_cannot_answer_is_silent(offsets):
    # No live archive at all -- the offline harness, the suite, a fixture run. There is
    # nothing anomalous about it and nothing to say, so it stays silent. This is the ONE
    # can't-check case that does: the others below happen WITH a database answering.
    class _Blind:
        pass
    assert pickup.review(_LINKED, _Blind(), fit=object(), obscodes={},
                         cfg=None) == []


def test_a_live_archive_that_cannot_be_matched_warns_without_banning(offsets):
    """Three ways a LIVE database leaves a designated pickup unverified. Each used to
    return silently, which is indistinguishable from "checked, nothing to report" -- so
    the guard against claiming a mislinked designation could be inert, permanently and
    invisibly, which is exactly how 2024 TF21 claimed 2017 SS396 whole.

    It must WARN, and it must NOT ban: a failure to retrieve is an absence of evidence,
    not evidence against, and banning would take a name off a document (or, via the 1.75
    withdrawal, delete the find from the batch JSON) because a query failed."""
    class _Empty:                       # answers, but knows nothing
        def designation_obs(self, provids):
            return []

    class _Boom:                        # the query itself fails
        def designation_obs(self, provids):
            raise RuntimeError("connection reset")

    class _OtherSpelling:               # rows come back filed under another provid
        def designation_obs(self, provids):
            return [dict(r, provid="2017 SS396 (alt)") for r in _ARCHIVE]

    for src, needle in ((_Empty(), "returned no observations"),
                        (_Boom(), "archive query failed"),
                        (_OtherSpelling(), "could not be matched")):
        recs = pickup.review(_LINKED, src, fit=object(), obscodes={}, cfg=None)
        assert len(recs) == 1, (src, recs)
        rec = recs[0]
        assert rec["unchecked"] is True and rec["compromised"] is False, rec
        assert rec["desig"] == "2017 SS396", rec
        assert needle in rec["why"], rec["why"]
        # the sentence a person reads says what happened and what it does NOT mean
        assert "COULD NOT CHECK 2017 SS396" in rec["warning"], rec["warning"]
        assert "trouble pulling in" in rec["warning"], rec["warning"]
        # ...and it is never banned from the document
        assert pickup.banned_designations(recs) == set(), recs


def test_the_check_can_be_switched_off(offsets):
    from pyroorb.itf.config import ItfConfig
    offsets["F51"] = (3779.0, 0.08)
    cfg = ItfConfig(pickup_check=False)
    assert pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(),
                         obscodes={}, cfg=cfg) == []


def test_thresholds_are_config(offsets):
    from pyroorb.itf.config import ItfConfig
    offsets["F51"] = (30.0, 0.08)
    assert pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(), obscodes={},
                         cfg=ItfConfig())[0]["compromised"] is True
    loose = ItfConfig(pickup_off_arcsec=60.0)
    assert pickup.review(_LINKED, _Source(_ARCHIVE), fit=object(), obscodes={},
                         cfg=loose)[0]["compromised"] is False


# ------------------------------------------------------- what the document says
def test_the_banned_designation_leaves_the_document():
    """The whole point: the lines stay in the arc, the designation leaves the
    submission."""
    seed = "".join(l + "\n" for l in _tracklet("K24T21F", "2024 10 03"))
    itf = "".join(l + "\n" for l in _tracklet("N571759", "2016 03 29",
                                              stn="W84"))
    combined = seed + itf + _LINKED
    reduced = pickup.drop_designations(combined, {"K17Sd6S"})
    assert "K17Sd6S" not in reduced
    assert "K24T21F" in reduced and "N571759" in reduced
    assert len(reduced.splitlines()) == len(combined.splitlines()) - 3

    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.linkage import build_linkage
    doc, err = build_linkage(reduced, ItfConfig(), primary_desigs=["2024 TF21"],
                             itf_obs_out=pickup.drop_designations(
                                 itf + _LINKED, {"K17Sd6S"}))
    assert err is None
    link = doc["links"]["link_0"]
    assert link["designations"] == ["K24T21F"]
    assert [t[0] for t in link["trksubs"]] == ["N571759"]


# ------------------------------------ is what is left still a SUBMITTABLE document
#
# The MPC's criteria decide this, not our own opposition-profile gate: the belief question
# was already settled on the full arc, and the ban takes away a NAME rather
# than evidence.  See search._reduced_submittable.
def _ctx(initial_opp_profile=None, pub_arc=8.0, pub_ids=()):
    from types import SimpleNamespace
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.opp_profile import OppProfile
    initial_opp_profile = initial_opp_profile or OppProfile(1, 0)
    return SimpleNamespace(cfg=ItfConfig(), initial_opp_profile=initial_opp_profile,
                           pub_ids=set(pub_ids),
                           pub=SimpleNamespace(arclength_days=pub_arc))


def _trial(q=2.4, e=0.15, rms=0.34):
    from types import SimpleNamespace
    return SimpleNamespace(q=q, e=e, rms=rms, fit=None)


def _seed_2024(nights=("2024 10 03", "2024 10 07", "2024 10 11",
                       "2024 10 23", "2024 10 26")):
    return "".join(l + "\n" for d in nights
                   for l in _tracklet("K24T21F", d))


def test_the_2024_tf21_remainder_is_submittable():
    """THE case this rule was got wrong on first.  Drop 2017 SS396 and what is
    left -- five published 2024 nights, one 2016 night, one 2017 night -- reads
    1.75 oppositions under our own fragility convention, which refused it.  The
    MPC's rule for 3 apparitions is one apparition of >= 3 nights, and the 2024
    apparition has five: it goes, without the designation."""
    from pyroorb.itf import search
    itf = "".join(l + "\n" for t in (_tracklet("N571759", "2016 03 29",
                                                stn="W84"),
                                     _tracklet("P10DqFA", "2017 09 21"))
                  for l in t)
    ok, why = search._reduced_submittable(_ctx(), _trial(),
                                          _seed_2024() + itf, itf)
    assert ok and why == "", why


def test_a_reduced_arc_that_collapses_blocks_the_document():
    """Take the pickup away and nothing linked is left: there is no
    identification to send, whatever the fit looks like."""
    from pyroorb.itf import search
    ok, why = search._reduced_submittable(_ctx(), _trial(),
                                          _seed_2024(), "")
    assert not ok and "nothing linked" in why


def test_a_lone_new_night_beside_a_thin_arc_blocks_the_document():
    """The 1.5 profile: one published night and one linked night.
    Two apparitions need one of >= 3 nights and the other of >= 2, so this is
    refused -- by the MPC's rule, not by ours."""
    from pyroorb.itf import search
    seed = "".join(l + "\n" for l in _tracklet("K24T21F", "2024 10 03"))
    itf = "".join(l + "\n" for l in _tracklet("N571759", "2016 03 29",
                                               stn="W84"))
    ok, why = search._reduced_submittable(_ctx(), _trial(), seed + itf, itf)
    assert not ok and "apparitions" in why


def test_a_175_whose_best_apparition_is_two_nights_blocks_the_document():
    """The other half of the example, and what separates it from the
    case above: 3 apparitions, none of them reaching 3 nights."""
    from pyroorb.itf import search
    seed = _seed_2024(("2024 10 03", "2024 10 07"))
    itf = "".join(l + "\n" for t in (_tracklet("N571759", "2016 03 29",
                                                stn="W84"),
                                     _tracklet("P10DqFA", "2017 09 21"))
                  for l in t)
    ok, why = search._reduced_submittable(_ctx(), _trial(), seed + itf, itf)
    assert not ok and ">=3 nights" in why
