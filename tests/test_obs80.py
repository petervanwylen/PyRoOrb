"""Unit tests for the designation helpers in obs80.py (cache naming + packing).

These guard two comet-driven fixes: a comet's '/' must not leak into a cache
filename (it was read as a path separator -> ENOENT on C/2025 R2), and comet
designations must pack instead of raising.
"""
from pyroorb import obs80


def test_cache_stem_strips_slash():
    """A comet's '/' (and any path separator) collapses to '_', so the stem is a
    single flat filename component -- never a nested path."""
    assert obs80.cache_stem("C/2025 R2") == "C_2025_R2"
    assert "/" not in obs80.cache_stem("C/2025 R2")
    assert obs80.cache_stem("P/2023 JN16") == "P_2023_JN16"
    # asteroid designations have only spaces -> existing cache names are unchanged
    assert obs80.cache_stem("2008 TE4") == "2008_TE4"
    # a slash-form comet and a space-form pseudo-typo map to the SAME stem, so the
    # offline picker round-trips a cached comet back to its file.
    assert obs80.cache_stem("C 2025 R2") == obs80.cache_stem("C/2025 R2")


def test_pack_designation_handles_comets():
    """Comet designations pack via the native packer (packing.pack/pack_comet);
    anything unpackable comes back unchanged rather than raising."""
    assert obs80.pack_designation("P/2023 JN16") == "K23J16N"     # asteroid-style core, bare
    assert obs80.pack_designation("C/2023 A3") == "CK23A030"      # comet-form, type-prefixed
    assert obs80.pack_designation("C/2025 R2") == "CK25R020"      # comet-form, type-prefixed
    assert obs80.pack_designation("73P") == "0073P"               # numbered periodic
    assert obs80.pack_designation("2008 TE4") == "K08T04E"        # asteroid provisional, native


def test_packing_runtime_has_no_sbpy():
    """The designation pack/unpack runtime path must not depend on sbpy (it was
    removed as a project dependency; sbpy is now only an optional cross-check inside
    packing._validate). Guards against a fallback `import sbpy` creeping back in."""
    from pyroorb import packing; import inspect
    from pyroorb import obs_sources
    for mod, fn in ((obs80, "pack_designation"), (obs_sources, "ensure_unpacked")):
        src = inspect.getsource(getattr(mod, fn))
        assert "sbpy" not in src, f"{mod.__name__}.{fn} still references sbpy"
    # packing.py mentions sbpy ONLY inside _validate (optional cross-check)
    others = [name for name, obj in vars(packing).items()
              if inspect.isfunction(obj) and name != "_validate"
              and "sbpy" in inspect.getsource(obj)]
    assert not others, f"sbpy referenced outside packing._validate: {others}"


def test_satellite_two_line_record():
    """A space-based obs (sat_pos_au) emits the MPC two-line satellite record: first line
    'S' in col 15 with RA/Dec, second line 's' with the geocentric km offset in the exact
    columns (X 35-45 decimal at 41, Y 47-57, Z 59-69), units flag '1' at col 33. Byte layout
    validated against find_orb's real C51 example, then round-tripped back to km."""
    KM = obs80.KM_PER_AU
    kmXYZ = [520.1210, -2628.1408, -6370.9902]           # find_orb SatelliteObs example (C51)
    rec = {"packed": "K10S16O", "jd_utc": 2455456.78554, "ra_deg": 10.0, "dec_deg": -5.0,
           "obscode": "C51", "sat_units": "km", "sat_pos_au": [c / KM for c in kmXYZ]}
    rendered = obs80.obs80_base_line(rec)
    l1, l2 = rendered.split("\n")
    assert l1[14] == "S" and l2[14] == "s"               # note2 col 15 (S primary / s companion)
    assert l1[0:12] == l2[0:12]                           # companion shares the packed desig
    assert l2[32] == "1"                                  # units flag col 33 == km
    assert l2[34] == "+" and l2[40] == "."               # X sign col 35, decimal col 41
    assert l2[45:80].strip().endswith("C51")             # obscode on the companion too
    assert obs80.mpc80_satellite_offset(l2) is not None
    back_km = [c * KM for c in obs80.mpc80_satellite_offset(l2)]
    for got, want in zip(back_km, kmXYZ):
        assert abs(got - want) < 1e-3, (got, want)


def test_satellite_au_units_for_distant_spacecraft():
    """An ADES ICRF_AU satellite (a distant L2 spacecraft can't be written in km without
    overflowing the field) emits units flag '2' and au values, round-tripping in au."""
    au = [0.00812345, -0.00051234, 0.00933210]           # ~L2-scale geocentric au
    rec = {"packed": "K10S16O", "jd_utc": 2455456.78554, "ra_deg": 10.0, "dec_deg": -5.0,
           "obscode": "258", "sat_units": "au", "sat_pos_au": au}
    l2 = obs80.obs80_base_line(rec).split("\n")[1]
    assert l2[32] == "2"                                  # units flag col 33 == au
    back = obs80.mpc80_satellite_offset(l2)               # flag '2' -> already au, no /KM_PER_AU
    for got, want in zip(back, au):
        assert abs(got - want) < 1e-7, (got, want)


def test_roving_two_line_record():
    """A roving observer (site_geodetic) emits the MPC two-line 247 record: 'V' primary,
    'v' companion carrying E-longitude (cols 35-44), latitude (46-55, N+/S-), altitude
    (57-61, integer m), round-tripped back to the geodetic triple."""
    rec = {"packed": "K06O10E", "jd_utc": 2453946.5, "ra_deg": 120.0, "dec_deg": 33.0,
           "obscode": "247", "site_geodetic": (243.1234, -33.5678, 1234.0)}
    l1, l2 = obs80.obs80_base_line(rec).split("\n")
    assert l1[14] == "V" and l2[14] == "v"
    assert l2[77:80] == "247"
    assert abs(float(l2[34:44]) - 243.1234) < 1e-4        # E-longitude cols 35-44
    assert abs(float(l2[45:55]) - -33.5678) < 1e-4        # latitude cols 46-55 (sign kept)
    assert int(l2[56:61]) == 1234                         # altitude cols 57-61 (m)
    assert obs80.mpc80_roving_itrs_au(l2) is not None     # parseable as a 'v' line


def test_ground_obs_stays_single_line():
    """A plain ground optical obs is unchanged -- one 80-col line, note2 'C', no companion."""
    rec = {"packed": "K20A00A", "jd_utc": 2458853.875, "ra_deg": 200.0, "dec_deg": 12.0,
           "obscode": "703", "mag": 20.1, "band": "V"}
    line = obs80.obs80_base_line(rec)
    assert "\n" not in line and len(line) == 80 and line[14] == "C"


def test_ades_satellite_roundtrips_through_records_to_obs():
    """End-to-end: an ADES ICRF_KM satellite record parsed by ades.records_to_obs (with the
    obscode exempt from the known-station requirement) carries sat_units + sat_pos_au, so the
    web astrometry line (obs80_base_line) is the full two-line 's' record."""
    from pyroorb import ades
    recs = [{"provid": "2010 SO16", "stn": "C51", "ra": "10.0", "dec": "-5.0",
             "obstime": "2010-09-17T06:51:11.000Z", "sys": "ICRF_KM", "ctr": "399",
             "pos1": "520.1210", "pos2": "-2628.1408", "pos3": "-6370.9902"}]
    obs = ades.records_to_obs(recs, {})                   # empty obscodes: C51 exempt via sat vector
    assert len(obs) == 1 and obs[0].get("sat_units") == "km"
    assert obs80.obs80_base_line(obs[0]).split("\n")[1][14] == "s"


def _obs80_with_catalog(cat, ra=200.0, dec=12.0):
    """A valid 80-col ground record whose col-72 catalog byte is `cat`."""
    rec = {"packed": "K20A00A", "jd_utc": 2451545.5, "ra_deg": ra, "dec_deg": dec,
           "obscode": "703", "mag": 20.1, "band": "V"}
    line = obs80.obs80_base_line(rec)
    return line[:71] + cat + line[72:]


def test_obs80_carries_its_star_catalog_and_is_debiased():
    """OBS80 col 72 is the star-catalog code (find_orb's astrometric_net_code), and an
    OBS80-ingested obs must be EFCC18-debiased exactly like its ADES twin -- it wasn't,
    so the same astrometry read in the two formats fitted two different orbits. A
    tabulated catalog (USNO-A2 'c') moves the position; Gaia-DR2 ('V') is the reference
    frame and must not move."""
    from pyroorb import debias; import pyroorb.obs_sources as osrc
    codes = {"703": (0.0, 0.0, 0.0)}
    biased = osrc.parse_obs(_obs80_with_catalog("c"), codes)[0]     # USNO-A2
    gaia = osrc.parse_obs(_obs80_with_catalog("V"), codes)[0]       # Gaia-DR2 (the reference)
    assert biased["astcat"] == "c" and gaia["astcat"] == "V"
    assert obs80.mpc80_catalog(_obs80_with_catalog(" ")) is None    # blank col 72 -> no catalog
    if not debias.available():
        return                                     # table not downloaded: parsing is all we can check
    assert biased["debiased"] is True, "a tabulated catalog must be debiased on the OBS80 path"
    assert biased["ra_raw"] == 200.0 and biased["dec_raw"] == 12.0  # the raw position is kept
    dra, ddec = debias.bias_for(200.0, 12.0, "c", 2451545.5)        # the correction is SUBTRACTED
    assert abs(biased["ra_deg"] - (200.0 - dra)) < 1e-12
    assert abs(biased["dec_deg"] - (12.0 - ddec)) < 1e-12
    assert gaia["debiased"] is False and gaia["ra_deg"] == 200.0    # reference frame: untouched


def test_cat_code_takes_both_ades_names_and_raw_mpc_codes():
    """debias.cat_code is the one place the two ingest paths meet: ADES carries an astCat
    NAME, OBS80 the raw MPC letter. Both resolve to the same bias-table column, and the
    Gaia reference frames / untabulated catalogs resolve to None (zero correction)."""
    from pyroorb import debias
    assert debias.cat_code("USNOA2") == "c" and debias.cat_code("c") == "c"
    assert debias.cat_code("UCAC4") == "q" and debias.cat_code("q") == "q"
    assert debias.cat_code("Gaia2") is None and debias.cat_code("V") is None   # DR2 = reference
    assert debias.cat_code("W") is None and debias.cat_code("X") is None       # DR3 / EDR3
    assert debias.cat_code("h") is None          # GSC-1.0: a real MPC code EFCC18 doesn't tabulate
    assert debias.cat_code(None) is None and debias.cat_code(" ") is None


def test_debias_obs_is_idempotent():
    """Two ingest paths now call debias_obs, so a second pass over an already-corrected
    list must be a no-op -- double-subtracting the bias would be worse than never
    applying it."""
    from pyroorb import debias; import pyroorb.obs_sources as osrc
    if not debias.available():
        return
    obs = osrc.parse_obs(_obs80_with_catalog("c"), {"703": (0.0, 0.0, 0.0)})   # already debiased
    once = (obs[0]["ra_deg"], obs[0]["dec_deg"])
    assert debias.debias_obs(obs) == 0, "a second pass must correct nothing"
    assert (obs[0]["ra_deg"], obs[0]["dec_deg"]) == once


def test_debias_failure_warns_instead_of_passing_silently():
    """An UNEXPECTED debias failure must not silently ship raw positions. A missing
    bias table is a clean no-op INSIDE debias_obs, so an exception reaching the
    ingest means the step itself broke (a corrupt table, a programming error) --
    the ingest keeps the obs usable (raw) but says so on stderr, instead of the
    old bare `except: pass` that made the regression invisible. ades.records_to_obs
    carries the identical guard; the OBS80 path stands in for both here."""
    import io, contextlib
    from pyroorb import debias; import pyroorb.obs_sources as osrc

    def _boom(obs):
        raise RuntimeError("synthetic bias-table corruption")

    orig = debias.debias_obs
    debias.debias_obs = _boom
    try:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            obs = osrc.parse_obs(_obs80_with_catalog("c"), {"703": (0.0, 0.0, 0.0)})
    finally:
        debias.debias_obs = orig
    assert len(obs) == 1 and obs[0]["ra_deg"] == 200.0      # parsing survives, position raw
    msg = err.getvalue()
    assert "[debias]" in msg and "RuntimeError" in msg, f"no stderr warning emitted: {msg!r}"


def test_geocentric_observer_fast_path_is_exact():
    """``observer_positions_batch`` short-circuits a pure-GEOCENTRE request (site
    500, no satellite/roving override): the site offset is the zero vector, so
    the ITRS->GCRS rotation it skips could only ever return zero.  The fast path
    must therefore be EXACT, not approximate -- and its per-grid memo must hand
    back the same numbers (the ITF corridor sweeps one grid many times)."""
    import numpy as np
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import (ITRS, GCRS, CartesianRepresentation,
                                     get_body_barycentric)
    import pyroorb.obs_sources as OS

    jds = np.arange(2455000.5, 2455200.5, 7.0)
    n = len(jds)
    # the general path, spelled out (what the function does for a real station)
    T = Time(jds, format="jd", scale="utc")
    itrs = ITRS(CartesianRepresentation((np.zeros((n, 3)) * u.au).T), obstime=T)
    site_eq = itrs.transform_to(GCRS(obstime=T)).cartesian.xyz.to(u.au).value.T
    assert np.abs(site_eq).max() == 0.0, "zero site must rotate to zero"
    # the Earth-Sun leg comes from _earth_sun_eq -- THE single source of truth for observer
    # geometry (DE440 when available, builtin otherwise) -- so this test keeps asserting
    # "fast path == general path" rather than re-pinning which ephemeris serves them both
    general = OS._earth_sun_eq(T) + site_eq

    OS._GEOCENTRIC_MEMO.clear()
    fast = OS.observer_positions_batch(jds, ["500"] * n, {"500": (0.0, 0.0, 0.0)})
    again = OS.observer_positions_batch(jds, ["500"] * n, {"500": (0.0, 0.0, 0.0)})
    assert np.abs(general - fast).max() == 0.0, "geocentre fast path must be exact"
    assert np.abs(fast - again).max() == 0.0, "memo must return identical values"
    # a real station must NOT take the fast path (its offset is nonzero)
    stn = OS.observer_positions_batch(jds, ["F51"] * n,
                                      {"F51": (203.74, 0.936, 0.351)})
    assert np.abs(stn - fast).max() > 0.0, "a ground station must keep the offset"


# --------------------------------------------------------------------------
# ADES -> OBS80 field mapping (band, star catalogue, programme code, reference,
# reported precision). The reference is the MPC's OWN 80-column line for the same
# observation: `the download cache/` caches both an object's ADES and its OBS80, and the
# expectations below are lifted verbatim from that pair for 2026 DB65.
# --------------------------------------------------------------------------

def test_band_collapses_the_photometric_system():
    """A two-character ADES passband names the SYSTEM as well as the filter; the
    80-column record has one byte for it, and which byte depends on the system."""
    assert obs80.band_to_obs80("Pw") == "w"        # Pan-STARRS w -- was printing 'P'
    assert obs80.band_to_obs80("Ac") == "c"        # ATLAS cyan
    assert obs80.band_to_obs80("Lr") == "r"        # Rubin r
    assert obs80.band_to_obs80("Rc") == "R"        # Cousins R: filter comes FIRST
    assert obs80.band_to_obs80("V") == "V"
    assert obs80.band_to_obs80("UNK") == " "       # "didn't say" is a blank, not 'U'
    assert obs80.band_to_obs80(None) == " "


def test_rendered_band_keeps_the_engines_own_correction():
    """The collapse must be the SAME one photometry.band_shift applies, because
    the ITF search fits the rendered line rather than the ADES record: whatever
    lands in col 71 is what the H fit will later key its band->V shift on.

    Rubin is the case that proved it.  2025 NX363's arc is all X05, i/z on three
    nights and g on a fourth; with 'Lg' truncated to 'L' the g night carried
    band_shift +0.2 instead of -0.35 and read ~0.55 mag faint against an H
    fitted from its i/z siblings -- enough for the ITF search to raise a
    "possible different body" photometric caution on the object's own discovery
    arc.  Equal shifts across the round trip is the invariant that forbids it."""
    from pyroorb import photometry

    def band_col(band):
        return obs80.format_obs80({
            "packed": "K25Na3X", "jd_utc": 2460868.62, "ra_deg": 243.4,
            "dec_deg": -29.9, "mag": 22.86, "band": band, "obscode": "X05",
        }).splitlines()[0][70]

    assert band_col("Lg") == "g"          # the NX363 night itself (was 'L')
    assert band_col("Li") == "i"
    assert band_col("Lz") == "z"
    assert band_col("i") == "i"           # OBS80-native bands pass through
    assert band_col(" ") == " "
    # For a band with no system-specific entry, the correction survives an
    # ADES -> OBS80 -> parse round trip: the rendered character is what
    # band_shift would have keyed on anyway.
    for ades in ("Pw", "Ac", "Rc", "V", "Sr"):
        assert (photometry.band_shift(band_col(ades))
                == photometry.band_shift(ades)), ades
    # ...and for LSST it deliberately does NOT, which is the point of keying the
    # table on the full ADES band: Veres & Chesley (2017) give LSST its own
    # transformation, so 'Lu' converts as -1.77 while still SPELLING 'u' -- the
    # MPC's own rendering, which this port was byte-validated against.
    assert band_col("Lu") == "u"
    # Stated as the MECHANISM, not as "these two numbers differ": a full code
    # the table names is converted as itself, whatever the table sets it to.
    # (If the LSST rows were ever dropped, the fallback below is the correct
    # behaviour and this must not fail for it.)
    for full in ("Lu", "Lg"):
        if full in photometry.BAND_SHIFT:
            assert photometry.band_shift(full) == photometry.BAND_SHIFT[full]
        else:
            assert (photometry.band_shift(full)
                    == photometry.BAND_SHIFT[full[1]])


def test_star_catalogue_code_col72():
    """ADES astCat -> the col-72 byte, including the spellings that need the fuzzy
    reduction ('Gaia3E' is Gaia-EDR3) or an outright alias ('Tyc2' is Tycho-2)."""
    assert obs80.net_name_to_byte_code("Gaia3E") == "X"
    assert obs80.net_name_to_byte_code("Gaia2") == "V"
    assert obs80.net_name_to_byte_code("Gaia DR3") == "W"
    assert obs80.net_name_to_byte_code("UCAC4") == "q"
    assert obs80.net_name_to_byte_code("Tyc2") == "g"
    assert obs80.net_name_to_byte_code("2MASS") == "L"
    assert obs80.net_name_to_byte_code("UNK") == " "     # stated as unknown -> blank
    assert obs80.net_name_to_byte_code(None) == " "
    assert obs80.net_name_to_byte_code("Norbert's Catalogue") == "?"


def test_programme_code_col14():
    """ADES `prog` is a two-character mutant-hex index into the programme codes."""
    assert obs80.prog_to_note1("00") == "0"
    assert obs80.prog_to_note1("0A") == "!"
    assert obs80.prog_to_note1("0F") == "&"
    # Past the tabulated 35 the MPC's own lines walk printable ASCII, where
    # find_orb's ades2mpc falls back to mutant hex and diverges.
    assert obs80.prog_to_note1("0Z") == ":"
    assert obs80.prog_to_note1("0f") == "@"
    assert obs80.prog_to_note1("0q") == "K"
    assert obs80.prog_to_note1("12") == "W"
    assert obs80.prog_to_note1(None) is None


def test_packed_publication_reference():
    assert obs80.pack_mpc_reference("MPEC 2021-D85") == "ED085"
    assert obs80.pack_mpc_reference("MPS     6197") == "a6197"
    assert obs80.pack_mpc_reference("MPC     4811") == "M4811"     # under 10000: 'M'
    assert obs80.pack_mpc_reference("MPC    10244") == "10244"
    assert obs80.pack_mpc_reference("AN  174") is None             # journal: unpacked
    assert obs80.pack_mpc_reference(None) is None


def test_reported_precision_is_not_invented():
    """A position is rendered at the precision its submitter reported (ADES
    precRA/precDec/precTime), not padded out to the modern default."""
    assert obs80.ra_hms(210.0 + 1e-5, 2).endswith(" ")
    assert obs80.ra_hms(210.0, 2) == "14 00 00.00 "
    assert obs80.dec_dms(-11.5, 1) == "-11 30 00.0 "
    assert obs80._date_to_obs80(2451545.0, 5) == "2000 01 01.50000 "
    assert obs80._date_to_obs80(2451545.0, 6) == "2000 01 01.500000"


def test_ades_record_renders_the_mpcs_own_80_column_line():
    """End-to-end: the discovery observation of 2026 DB65 as the MPC publishes it.

    Every byte the ADES record determines -- packed designation, discovery
    asterisk, programme code, mode, date, RA/Dec, magnitude, band, catalogue,
    station -- must land in its column."""
    rec = {"packed": "K26D65B", "discovery": True, "prog": "00", "mode": "CCD",
           "jd_utc": 2461098.10829300, "ra_deg": 229.05363750,
           "dec_deg": -22.20363056, "mag": 21.57, "band": "Pw",
           "astcat": "Gaia3E", "obscode": "F51"}
    line = obs80.format_obs80(rec)
    assert len(line) == 80
    assert line[70] == "w", "col 71 is the FILTER, not the photometric system"
    assert line[71] == "X", "col 72 is the star catalogue (Gaia-EDR3)"
    assert line[13] == "0", "col 14 is the programme code"
    assert line == ("     K26D65B*0C2026 02 26.608293"
                    "15 16 12.873-22 12 13.07         21.57wX     F51")


def test_comet_magnitude_column_says_nuclear_or_total():
    """For a comet, col 71 carries N/T (which magnitude this is, ADES nucMag)
    instead of the filter band."""
    rec = {"packed": "0139P", "jd_utc": 2451133.90661, "ra_deg": 30.71542,
           "dec_deg": 19.98222, "mag": 19.6, "band": "R", "nucmag": 1.0,
           "astcat": "UCAC3", "obscode": "566", "mode": "CCD"}
    assert obs80.format_obs80(rec)[70] == "N"
    rec["nucmag"] = 0.0
    assert obs80.format_obs80(rec)[70] == "T"
    del rec["nucmag"]
    assert obs80.format_obs80(rec)[70] == "R"


def test_numbered_object_keeps_both_designation_fields():
    """The permanent number (cols 1-5) and the provisional (cols 6-12) are separate
    fields; a numbered object observed under its provisional fills both."""
    line = obs80.format_obs80({"packed_perm": "B1520", "packed": "K01Y98M",
                               "jd_utc": 2452257.9, "ra_deg": 10.0, "dec_deg": 5.0,
                               "obscode": "704"})
    assert line[:12] == "B1520K01Y98M"
    # A survey/NEOCP temporary id is 8 chars: it backs up to END at col 12.
    line = obs80.format_obs80({"packed": "PK24R020", "jd_utc": 2460532.0,
                               "ra_deg": 10.0, "dec_deg": 5.0, "obscode": "F52"})
    assert line[:12] == "    PK24R020"


def main():
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}")
            except Exception as ex:
                print(f"  FAIL  {name}: {ex}"); fails.append(name)
    print(f"\n{len(fails)} failed" if fails else "\nall obs80 tests passed")
    return 1 if fails else 0


def test_observer_positions_ride_the_integration_ephemeris():
    """The observer's Earth must come from the SAME DE440 ephemeris the integration uses.
    astropy's builtin analytic Earth is 2-3 km off DE440 -- invisible at survey distances,
    but at a deep-Earth-encounter range (0.003-0.01 au) it reads as 0.5-1.5" of astrometry
    and manufactured 2004 FH's phantom Yarkovsky. Skips when ASSIST/DE440 is
    absent, where the fallback deliberately matches the rebound backend's own ephemeris."""
    import numpy as np
    from astropy.time import Time
    import pyroorb.obs_sources as OS
    import pyroorb.obs_sources as _os
    try:
        import pyroorb.force_assist as fa
        fa.get_ephem()
    except Exception:
        import pytest
        pytest.skip("ASSIST/DE440 unavailable; fallback path is the builtin Earth by design")
    jds = np.array([2453081.9, 2455000.5, 2460000.5])
    T = Time(jds, format="jd", scale="utc")
    de440 = OS._earth_sun_eq(T)
    saved = OS._DE440_OBSERVER
    try:
        OS._DE440_OBSERVER = False
        builtin = OS._earth_sun_eq(T)
    finally:
        OS._DE440_OBSERVER = saved
    d_km = np.linalg.norm(de440 - builtin, axis=1) * 1.495978707e8
    # the known offset scale: kilometres, not metres and not thousands of km
    assert (d_km > 0.05).all() and (d_km < 50.0).all(), d_km


def test_provisional_comet_line_is_not_named_by_its_type_letter():
    """'    CK26B030' is C/2026 B3: cols 1-4 blank, col 5 the comet TYPE letter, cols 6-12
    the packed core. Reading cols 1-5 as the permanent number named the object 'C'."""
    import pyroorb.obs_sources as OS
    lines = [
        "    CK26B030*0C2026 01 29.49642412 14 39.740+18 11 24.77         20.01iX#0LMSF52",
        "    CK26B030 1C2026 01 30.53262912 14 03.700+18 12 21.35         19.52GV#0LMSI52",
    ]
    for ln in lines:
        assert OS.obs80_line_id(ln) == "CK26B030"
    obs = OS.parse_obs("\n".join(lines), {"F52": None, "I52": None})
    assert [o["packed"] for o in obs] == ["CK26B030", "CK26B030"]
    assert OS.ensure_unpacked("CK26B030") == "C/2026 B3"
    # round trip with the packer the ADES->OBS80 renderer uses
    assert obs80.pack_designation("C/2026 B3") == "CK26B030"
    # the other comet shapes
    assert OS.ensure_unpacked("PK26B03a") == "P/2026 B3-A"
    assert OS.ensure_unpacked("0073P") == "73P"
    assert OS.ensure_unpacked("0073Pc") == "73P-C"
    # untouched: numbers, asteroid provisionals, trksubs, numbered-comet line ids
    assert OS.obs80_line_id("0073P        C2026") == "0073P"
    assert OS.obs80_line_id("00433        C2026") == "00433"
    assert OS.obs80_line_id("     K23R67C  C2026") == "K23R67C"
    assert OS.obs80_line_id("     P21l7qL  C2026") == "P21l7qL"
    assert OS.ensure_unpacked("00433") == "433"
    assert OS.ensure_unpacked("K16UL0A") == "2016 UA210"
