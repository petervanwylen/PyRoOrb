"""Offline tests for the two find_orb ephemeris DISPLAY cuts (ephemerides.py):
a faint-magnitude limit and "suppress unobservables".

Both only hide rows -- no orbit, no fit and no network is involved, so these run on
fabricated ephemeris rows plus (for the observer columns) the cached obscode table.

    pytest tests/test_ephem_filters.py
"""
from pyroorb import ephemerides as ephemeris
def _row(**kw):
    """A minimal ephemeris row: observable from a ground station, moderately bright."""
    r = {"jd_utc": 2461000.5, "date_iso": "2026-01-01T00:00:00",
         "ra_deg": 10.0, "dec_deg": 5.0, "ra_hms": "00 40 00.00", "dec_dms": "+05 00 00.0",
         "delta_au": 1.5, "r_au": 2.4, "elong_deg": 150.0, "phase_deg": 10.0,
         "v_mag": 19.0, "motion_arcsec_min": 0.5, "motion_pa_deg": 90.0,
         "unc_smaj_arcsec": 1.0, "unc_smin_arcsec": 0.5, "unc_pa_deg": 45.0,
         "unc_along_arcsec": 0.9, "unc_cross_arcsec": 0.4,
         "alt_deg": 60.0, "sun_alt_deg": -30.0,
         "lunar_elong_deg": 90.0, "moon_sd_deg": 0.26}
    r.update(kw)
    return r


def test_unobservable_reason():
    assert ephemeris.unobservable_reason(_row()) is None
    assert ephemeris.unobservable_reason(_row(alt_deg=-5.0)) == "horizon"
    assert ephemeris.unobservable_reason(_row(alt_deg=0.0)) == "horizon"   # AT the horizon
    assert ephemeris.unobservable_reason(_row(sun_alt_deg=+2.0)) == "daylight"
    # find_orb's rule is the geometric horizon, not twilight: the Sun a degree down is night
    assert ephemeris.unobservable_reason(_row(sun_alt_deg=-1.0)) is None
    # inside the lunar disc -> occulted ('l'/'L')
    assert ephemeris.unobservable_reason(_row(lunar_elong_deg=0.1)) == "moon"
    assert ephemeris.unobservable_reason(_row(lunar_elong_deg=0.4)) is None
    # below the horizon wins over daylight (find_orb's 'B' overrides)
    assert ephemeris.unobservable_reason(_row(alt_deg=-5.0, sun_alt_deg=10.0)) == "horizon"
    # a site with no horizon (geocentre / space-based / full=False) is never suppressed
    assert ephemeris.unobservable_reason(_row(alt_deg=None, sun_alt_deg=None)) is None
    assert ephemeris.unobservable_reason({"v_mag": 19.0}) is None
    print("ok  unobservable_reason")


def test_faint_limit():
    rows = [_row(v_mag=18.0), _row(v_mag=22.5), _row(v_mag=None)]
    kept, hidden = ephemeris.filter_rows(rows, faint_limit=20.0)
    assert [r["v_mag"] for r in kept] == [18.0, None], kept   # unknown mag is not faint
    assert hidden == {"faint": 1}, hidden
    # no limit -> nothing hidden
    kept, hidden = ephemeris.filter_rows(rows)
    assert len(kept) == 3 and hidden == {}
    # exactly at the limit survives (find_orb drops mag > limit)
    kept, _ = ephemeris.filter_rows([_row(v_mag=20.0)], faint_limit=20.0)
    assert len(kept) == 1
    print("ok  faint limit")


def test_combined_and_counts():
    rows = [_row(),                                  # kept
            _row(v_mag=25.0),                        # faint
            _row(alt_deg=-1.0),                      # horizon
            _row(sun_alt_deg=15.0),                  # daylight
            _row(lunar_elong_deg=0.05),              # moon
            _row(v_mag=25.0, alt_deg=-1.0)]          # faint takes precedence (counted once)
    kept, hidden = ephemeris.filter_rows(rows, faint_limit=21.0, suppress_unobservable=True)
    assert len(kept) == 1, len(kept)
    assert hidden == {"faint": 2, "horizon": 1, "daylight": 1, "moon": 1}, hidden
    assert sum(hidden.values()) == 5
    note = ephemeris.hidden_note(hidden)
    assert "5 epoch(s) hidden" in note and "2 too faint" in note and "1 in daylight" in note
    assert ephemeris.hidden_note({}) == ""
    print("ok  combined cuts + counts")


def test_break_marks_on_the_surviving_rows():
    """Each RUN of dropped epochs must leave a mark on the row after it (and on the last
    kept row when the run ends the table), so a renderer can break the series there."""
    rows = [_row(v_mag=18.0),                            # kept
            _row(v_mag=25.0), _row(v_mag=25.0),          # run of 2 faint
            _row(v_mag=18.5),                            # kept -> hidden_before {faint:2}
            _row(alt_deg=-1.0),                          # run of 1 below the horizon
            _row(v_mag=19.5),                            # kept -> hidden_before {horizon:1}
            _row(v_mag=25.0)]                            # trailing run -> hidden_after
    kept, hidden = ephemeris.filter_rows(rows, faint_limit=21.0, suppress_unobservable=True)
    assert len(kept) == 3
    assert "hidden_before" not in kept[0], kept[0]
    assert kept[1]["hidden_before"] == {"faint": 2}
    # the last kept row carries BOTH: a run before it and the run that ends the table
    assert kept[2]["hidden_before"] == {"horizon": 1}
    assert kept[2]["hidden_after"] == {"faint": 1}
    assert sum(hidden.values()) == 4
    # marks must not be written back onto the caller's rows
    assert all("hidden_before" not in r and "hidden_after" not in r for r in rows)
    # no cuts -> no marks at all
    kept, _ = ephemeris.filter_rows([_row(), _row()])
    assert all("hidden_before" not in r for r in kept)
    print("ok  per-run break marks")


def test_break_lines_in_the_text_table():
    rows = [_row(v_mag=18.0), _row(v_mag=25.0), _row(v_mag=25.0), _row(v_mag=18.5)]
    kept, hidden = ephemeris.filter_rows(rows, faint_limit=21.0)
    lines = ephemeris.format_table(kept, site="F51", hidden=hidden).splitlines()
    brk = [l for l in lines if l.startswith("....")]
    assert len(brk) == 1, lines
    assert "2 hidden (2 too faint)" in brk[0], brk[0]
    # the break sits BETWEEN the two data rows, not at either end
    data = [i for i, l in enumerate(lines) if l.startswith("2026-")]
    assert data[0] < lines.index(brk[0]) < data[1], lines
    # a run that ends the table is marked after the last row
    kept, hidden = ephemeris.filter_rows([_row(v_mag=18.0), _row(v_mag=25.0)], faint_limit=21.0)
    lines = ephemeris.format_table(kept, site="F51", hidden=hidden).splitlines()
    assert lines[-1].startswith("....") and "1 hidden" in lines[-1], lines
    print("ok  dotted break lines in the text table")


def test_format_table_notes():
    kept, hidden = ephemeris.filter_rows([_row(), _row(v_mag=25.0)], faint_limit=21.0)
    text = ephemeris.format_table(kept, site="F51", hidden=hidden)
    assert "observatory F51" in text
    assert "1 epoch(s) hidden: 1 too faint" in text
    # title + note + header + rule + 1 row + the trailing break for the dropped epoch
    assert len(text.splitlines()) == 6, text
    # everything hidden -> find_orb's explicit "no ephemeris output" rather than a bare header
    kept, hidden = ephemeris.filter_rows([_row(v_mag=25.0)], faint_limit=21.0)
    text = ephemeris.format_table(kept, site="F51", hidden=hidden)
    assert text.startswith("No ephemeris output.") and "too faint" in text
    print("ok  format_table notes")


def test_observer_columns_carry_moon_semidiameter():
    """The lunar-occultation test needs the Moon's apparent size, so the observer pass
    must supply it (and it must be ~0.25-0.28 deg, the real range)."""
    from pyroorb.obs_sources import load_obscodes
    rows = [{"jd_utc": 2461000.5 + i, "ra_deg": 120.0, "dec_deg": 20.0} for i in range(3)]
    ephemeris._add_observer_columns(rows, "F51", load_obscodes())
    for r in rows:
        assert 0.24 < r["moon_sd_deg"] < 0.29, r["moon_sd_deg"]
        assert r["alt_deg"] is not None and r["sun_alt_deg"] is not None
    # ... and the geocentre has no altitude at all, so nothing can be suppressed there
    rows = [{"jd_utc": 2461000.5, "ra_deg": 120.0, "dec_deg": 20.0}]
    ephemeris._add_observer_columns(rows, "500", load_obscodes())
    assert rows[0]["alt_deg"] is None
    assert ephemeris.unobservable_reason(rows[0]) is None
    print("ok  observer columns (moon semi-diameter, geocentric has no horizon)")


def test_server_query_parsing():
    """The ?faint= / ?unobs= query params the pane and the export both send."""
    from pyroorb.web import server
    assert server._ephem_filters({}) == {"faint_limit": None, "suppress_unobservable": False}
    assert server._ephem_filters({"faint": ["21.5"], "unobs": ["1"]}) == {
        "faint_limit": 21.5, "suppress_unobservable": True}
    assert server._ephem_filters({"faint": [""]})["faint_limit"] is None
    assert server._ephem_filters({"faint": ["junk"]})["faint_limit"] is None
    assert server._ephem_filters({"unobs": ["0"]})["suppress_unobservable"] is False
    assert server._ephem_filters({"unobs": ["true"]})["suppress_unobservable"] is True
    print("ok  server query parsing")


def test_site_has_horizon():
    from pyroorb.web import engine_api
    assert engine_api.site_has_horizon("F51") is True     # Pan-STARRS: a real mountain
    assert engine_api.site_has_horizon("500") is False    # geocentre
    print("ok  site_has_horizon")
