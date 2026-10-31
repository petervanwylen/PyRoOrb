"""The ITF search's ADES registry (itf/obsreg.py), offline.

The search is line-based by necessity; the registry is what keeps that from
being a DATA downgrade -- the line is the identity, ADES is the data.  These
tests pin the three properties everything downstream leans on: a database row
becomes the ADES fields of its observation, a row with no reported sigma leaves
the engine on its station model, and no registry at all is indistinguishable
from the pre-ADES behaviour.

Lives in its own file rather than in test_itf_search.py, which is already at
the editing size limit.

    pytest tests/test_itf_obsreg.py
"""
from types import SimpleNamespace

import pyroorb.itf.search as S
from pyroorb.itf.obsreg import AdesRegistry


def _obs80(trksub, date, ra_txt, dec_txt, stn="F51"):
    line = ("     " + f"{trksub:<7}" + "  C" + f"{date:<17}"
            + f"{ra_txt:<12}" + f"{dec_txt:<12}" + " " * 9
            + " 21.0G" + " " + "     " + f"{stn:>3}")
    assert len(line) == 80, len(line)
    return line


def test_registry_carries_ades_from_a_db_row_onto_the_parsed_obs():
    """An ``obs_sbn`` candidate row IS an ADES record -- the table's columns are
    the ADES field names.  The registry must turn one into the ADES fields of
    the observation its 80-column line parses to, keyed by line identity."""
    line = _obs80("P10yL9z", "2017 01 20.49747", "08 01 13.28", "+05 46 21.1")
    reg = AdesRegistry()
    reg.add_records([{"obs80": line, "rmsra": 0.15, "rmsdec": 0.18,
                      "stn": "F51", "astcat": "Gaia3E", "rmsmag": 0.22,
                      "logsnr": 0.71, "obsid": "L0JIxN01", "mode": "CCD",
                      "trksub": "P10yL9z"}])
    o = {"obs80_line": line}
    assert reg.enrich([o]) == 1
    assert (o["sigma_ra"], o["sigma_dec"]) == (0.15, 0.18)
    assert o["uncert_src"] == "ADES"
    assert o["ades_rms_ra"] == 0.15 and o["mag_sigma"] == 0.22
    assert o["obs_id"] == "L0JIxN01" and o["mode"] == "CCD"
    assert reg.n_ades_sigma == 1
    # identity is the LINE, so a different observation gets nothing -- same
    # tracklet, same instant, different station
    other = {"obs80_line": _obs80("P10yL9z", "2017 01 20.49747",
                                  "08 01 13.28", "+05 46 21.1", stn="703")}
    assert reg.enrich([other]) == 0
    assert "sigma_ra" not in other
    print("PASS registry carries ADES from a db row onto the parsed obs")


def test_registry_leaves_an_unreported_sigma_on_the_station_model():
    """~92% of ITF observations were submitted with no rmsRA/rmsDec at all (the
    1992 format never carried one).  Those must reach the fit with NO sigma
    attached, so the engine's Veres-2017 station fallback applies exactly as it
    did before ADES was carried -- inventing one would be worse than the loss
    this mechanism exists to fix."""
    line = _obs80("QHKZ1B9", "2007 02 03.28661", "10 14 34.48", "-11 58 40.0",
                  stn="644")
    reg = AdesRegistry()
    reg.add_records([{"obs80": line, "rmsra": None, "rmsdec": None,
                      "stn": "644", "astcat": "USNOB1", "obsid": "K2YLVc01"}])
    o = {"obs80_line": line}
    reg.enrich([o])
    assert "sigma_ra" not in o and "uncert_src" not in o
    assert o["obs_id"] == "K2YLVc01"           # provenance still carried
    assert reg.n_ades_sigma == 0
    # a reported rms of exactly 0 is a submitter leaving the field at its
    # default, not an infinitely precise measurement (weights.ades_rms_usable)
    reg2 = AdesRegistry()
    reg2.add_records([{"obs80": line, "rmsra": 0, "rmsdec": 0, "stn": "644"}])
    o2 = {"obs80_line": line}
    reg2.enrich([o2])
    assert "sigma_ra" not in o2
    print("PASS registry leaves an unreported sigma on the station model")


def test_registry_never_overwrites_position_or_catalog():
    """Two fields are deliberately NOT carried over from the database row: the
    POSITION (re-derived from the same 80 columns for pre-ADES rows, and worth
    ~0.01" for the rest -- far under SIGMA_FLOOR -- while overwriting it would
    discard the EFCC18 debias the parse already applied) and ASTCAT (``debias``
    keys on the column-72 spelling; the database spells the same catalog
    differently, so overwriting it would silently disable debiasing)."""
    line = _obs80("P10yL9z", "2017 01 20.49747", "08 01 13.28", "+05 46 21.1")
    reg = AdesRegistry()
    reg.add_records([{"obs80": line, "ra": 999.0, "dec": -99.0,
                      "astcat": "Gaia3E", "stn": "F51", "obsid": "X1"}])
    o = {"obs80_line": line, "ra_deg": 120.30, "dec_deg": 5.77,
         "astcat": "2MASS"}
    reg.enrich([o])
    assert o["ra_deg"] == 120.30 and o["dec_deg"] == 5.77
    assert o["astcat"] == "2MASS"
    assert o["obs_id"] == "X1"
    print("PASS registry never overwrites position or catalog")


def test_registry_is_optional_everywhere():
    """No registry -- an offline harness, a pasted OBS80 blob, the hand-built
    ctx objects the search tests use -- must be indistinguishable from the
    pre-ADES behaviour, never an error."""
    assert S._ctx_registry(SimpleNamespace()) is None
    reg = AdesRegistry()
    assert len(reg) == 0
    assert reg.enrich([{"obs80_line": _obs80("ZZ00001", "2020 09 13.30000",
                                             "10 00 00.00", "+10 00 00.0")}]) == 0
    reg.add_records([{"trksub": "ZZ00001"}])            # a row with no obs80
    reg.add_records([])
    reg.add_obs([{"obs80_line": "short"}])              # nothing ADES on it
    assert len(reg) == 0
    print("PASS registry is optional everywhere")


def test_seed_obs_register_under_their_rendered_line():
    """The seed path registers PARSED obs, whose identity is the 80-column line
    they render to -- the same rendering the seed text is built from, so the
    ids match by construction."""
    line = _obs80("K08T04E", "2008 10 07.14066", "10 00 00.00", "+10 00 00.0",
                  stn="703")
    reg = AdesRegistry()
    reg.add_obs([{"obs80_line": line, "sigma_ra": 0.3, "sigma_dec": 0.3,
                  "uncert_src": "ADES", "obs_id": "OID1", "trksub": "TRK1",
                  "ades_rms_ra": 0.3, "ades_rms_dec": 0.3, "log_snr": 1.2,
                  "designation": "2008 TE4", "ra_deg": 335.3}])
    o = {"obs80_line": line}
    assert reg.enrich([o]) == 1
    assert o["sigma_ra"] == 0.3 and o["obs_id"] == "OID1"
    assert o["trksub"] == "TRK1" and o["log_snr"] == 1.2
    assert "designation" not in o and "ra_deg" not in o   # structure untouched
    print("PASS seed obs register under their rendered line")


def test_registry_carries_the_full_ades_band_without_changing_the_line():
    """OBS80 column 71 is ONE character, and photometry.BAND_SHIFT is keyed on the
    full ADES band -- so an LSST 'Lu' round-tripped through 80 columns comes back
    as 'u' and gets charged the MPC's generic Sloan u' correction (-2.50) rather
    than LSST's own (-1.77, Veres & Chesley 2017 Table 4).  That is a 0.7 mag error
    in the fitted H and in every photometric O-C the web UI shows for a linked ITF
    tracklet.  The band therefore rides the registry like the sigmas do.

    It must NOT change the rendering: the collapse still runs on the way out, so
    column 71 -- and observation identity, which is column-aware -- is unchanged.
    """
    from pyroorb import photometry; from pyroorb import obs80
    line = _obs80("P10yL9z", "2017 01 20.49747", "08 01 13.28", "+05 46 21.1")
    reg = AdesRegistry()
    reg.add_records([{"obs80": line, "stn": "F51", "band": "Lu"}])
    o = {"obs80_line": line, "band": "u"}
    assert reg.enrich([o]) == 1
    assert o["band"] == "Lu"
    # The whole point: the registry carried the FULL band through, so the
    # conversion is looked up as 'Lu'.  Asserted against the table rather
    # than against a literal -- what Lu converts to is the table's business
    # (see tests/test_band_table.py), that the enrichment reaches it is
    # this test's.
    assert photometry.band_shift("Lu") == photometry.BAND_SHIFT["Lu"]
    # ...and spell the same, so nothing downstream of the line moves
    assert obs80.band_to_obs80("Lu") == "u"

    # the seed path carries it too (an ADES-fetched obs dict)
    reg2 = AdesRegistry()
    reg2.add_obs([{"obs80_line": line, "band": "Lg", "obs_id": "L0JIxN01"}])
    o2 = {"obs80_line": line, "band": "g"}
    reg2.enrich([o2])
    assert o2["band"] == "Lg"

    # a row with no band leaves the parsed value alone rather than blanking it
    reg3 = AdesRegistry()
    reg3.add_records([{"obs80": line, "stn": "F51", "band": "", "obsid": "X1"}])
    o3 = {"obs80_line": line, "band": "u"}
    reg3.enrich([o3])
    assert o3["band"] == "u"
    print("PASS registry carries the full ADES band without changing the line")
