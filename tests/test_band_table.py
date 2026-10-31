"""The band -> V LOOKUP.  Not the table.

THE TABLE IS DATA, AND IT IS CURATED.  `photometry.BAND_SHIFT` holds
calibration values that change when someone decides to change them -- a new
measurement, a new published transformation, a considered read of the MPC's
service.  Nothing in this file asserts what any of those numbers IS, and nothing
in this file reaches the network to find out.  That is deliberate, and it is a
reversal: an earlier version of this module pinned every value as a literal and
carried a `live` test that re-fetched the MPC's magnitude-band API and FAILED on
any disagreement.  Both were wrong, for the same reason from opposite ends --

  * pinning the values means editing the table is editing the test suite, so a
    deliberate recalibration arrives as a wall of red that has to be argued
    down rather than a one-line data change;
  * the live check hands the decision to an external service, so the MPC
    revising a number breaks the build on their schedule.  A table sourced from
    a live service still changes only when WE choose to take the change.

What is left here is the part that really is code, and it has really had bugs:
`ades_band_filter` once ended in "best effort: take the last character", which
read 'UNK' (the MPC's "the submitter declined to say") as K and charged it a
+1.7 mag INFRARED correction on 3449 tracklets of our own ITF index.  So these
tests ask only about BEHAVIOUR, always relative to whatever the table happens to
hold: does a full ADES code win over its filter letter, does a two-character
code fall back to that letter, do the several spellings of "no filter" agree,
and does the 80-column rendering stay independent of all of it.  Every
assertion below is still true after any edit to BAND_SHIFT.

The values' provenance and the arguments behind them live where they belong, in
the comment block above BAND_SHIFT in `src/pyroorb/photometry.py`.
"""
from pyroorb import photometry


def test_lookup_is_on_the_full_ades_band():
    """Internal processing never collapses.  A band the table names is answered
    as written -- so populating 'Lg' later needs no code change anywhere.

    Patches in its own value rather than reading one, so this stays a statement
    about the LOOKUP and not about the table."""
    patched = dict(photometry.BAND_SHIFT)
    sentinel = -12.5                            # not a plausible real shift
    patched["Lg"] = sentinel
    orig = photometry.BAND_SHIFT
    try:
        photometry.BAND_SHIFT = patched
        assert photometry.band_shift("Lg") == sentinel   # NOT g's value
    finally:
        photometry.BAND_SHIFT = orig
    # a band with no entry of its own still falls back through its filter letter
    assert "Pg" not in photometry.BAND_SHIFT
    assert photometry.band_shift("Pg") == photometry.BAND_SHIFT["g"]


def test_two_char_bands_fall_back_to_their_filter():
    """A two-character code the table does not name resolves through its filter
    letter -- our inference, not the MPC's, which publishes v_conversion=null
    for all 33 of them.  Note 'Rc' puts the FILTER first (Johnson-Cousins) while
    'Pw' puts the SYSTEM first (survey): both must land on the right letter."""
    for code, letter in (("Pw", "w"), ("Pi", "i"), ("Sr", "r"), ("Ao", "o"),
                         ("Rc", "R"), ("Bj", "B")):
        if code in photometry.BAND_SHIFT:
            continue                    # the MPC published it; then it wins
        assert photometry.band_shift(code) == photometry.BAND_SHIFT[letter], code


def test_v_equivalent_bands_take_no_shift():
    """V, and the codes that mean V, are 0 BY DEFINITION rather than by
    tabulation -- a rule of the lookup, so it belongs here."""
    for band in ("V", "Vj", "v"):
        assert photometry.band_shift(band) == 0.0, band


def test_unk_is_a_blank_band_not_a_k_band():
    """'UNK' is the MPC's "the submitter declined to state a filter".  The old
    collapse ended in "take the last character", so it read as K and drew an
    INFRARED correction -- on 3449 tracklets in our own ITF index alone.  It
    means no filter, so it takes whatever the no-filter shift is, which is also
    exactly what column 71 renders it as (a blank)."""
    assert photometry.ades_band_to_mpc("UNK") == " "
    assert photometry.band_shift("UNK") == photometry.band_shift(" ")


def test_an_unparseable_code_names_no_filter():
    """The distinction the old rule collapsed: a code can name a filter, name
    none explicitly, or be unrecoverable -- and only the middle one is blank.
    'IR' and 'VR' both used to land on 'R' by accident of spelling."""
    assert photometry.ades_band_filter("Lg") == "g"
    assert photometry.ades_band_filter("UNK") == " "     # states "none"
    assert photometry.ades_band_filter("") == " "
    assert photometry.ades_band_filter("IR") is None     # unrecoverable
    assert photometry.ades_band_filter("VR") is None     # DECam wide, not R
    assert photometry.ades_band_filter("W2") is None


def test_the_no_filter_spellings_agree():
    """Every way of saying "the submitter stated no filter" must land on the
    same shift.  ' ', '', None and 'UNK' are one fact spelled four ways, and
    before `ades_band_filter` they were not: 'UNK' collapsed to K.

    WHICH shift they land on is BLANK_BAND_SHIFT, i.e. the table's value, so it
    is read from the table rather than asserted."""
    answers = {photometry.band_shift(b) for b in (" ", "", None, "UNK")}
    assert answers == {photometry.BAND_SHIFT[" "]}


def test_the_unrecoverable_codes_agree():
    """And every way of saying "this code names no filter we can map" must
    likewise agree -- 'IR', 'VR' and 'W2'-'W4' used to split, two of them
    landing on 'R' by accident of spelling."""
    answers = {photometry.band_shift(b)
               for b in ("IR", "VR", "W2", "W3", "W4")}
    assert len(answers) == 1, f"unrecoverable codes disagree: {answers}"


def test_no_filter_and_unrecoverable_are_separately_decided():
    """These two GROUPS need not agree with each other, and that is a live
    coupling worth naming rather than asserting either way.

    `band_shift` returns a hardcoded 0.0 for an unrecoverable code, while a
    blank band returns BLANK_BAND_SHIFT.  They coincide today only because that
    constant is 0.0 -- and the comment above it explicitly invites setting it to
    0.43 (find_orb's unfiltered-CCD reading) or -0.8 (the legacy photographic
    one).  Doing so would silently split the two routes, which may well be
    RIGHT: "unfiltered CCD" is a real statement about a real bandpass, whereas a
    WISE W2 magnitude has no business inheriting it.  So this test asserts only
    that the split is decidable from the table, not which way it falls."""
    blank = photometry.band_shift(" ")
    unrecoverable = photometry.band_shift("W2")
    assert unrecoverable == 0.0        # the lookup's documented default
    assert blank == photometry.BAND_SHIFT[" "]
    if blank != 0.0:
        assert unrecoverable != blank  # ...and then they differ, deliberately


def test_an_unrecoverable_code_is_never_charged_a_filter_correction():
    """The specific regression: a code naming no filter must not pick up some
    letter's correction out of its spelling.  Stated against the table so it
    holds whatever 'R' and 'K' are set to."""
    for band in ("IR", "VR", "W2", "W3", "W4"):
        for letter in ("R", "K"):
            if photometry.BAND_SHIFT.get(letter, 0.0) == 0.0:
                continue                # nothing to be wrongly charged
            assert photometry.band_shift(band) != photometry.BAND_SHIFT[letter], \
                f"{band} was charged {letter}'s correction"


def test_the_band_table_does_not_change_what_column_71_spells():
    """The 80-column byte must stay what the MPC itself writes -- the plain
    prefix strip -- because that rendering was byte-validated against their
    published lines.  The table says how we CONVERT a band, not how we spell
    it, so this holds no matter which codes the table names."""
    from pyroorb import obs80
    assert obs80.band_to_obs80("Lu") == "u"
    assert obs80.band_to_obs80("Lg") == "g"
    assert obs80.band_to_obs80("UNK") == " "
    assert photometry.ades_band_filter("Lu") == "u"


def test_a_tabulated_full_code_still_spells_as_its_filter():
    """The two halves must not drift into each other: giving a full ADES code
    its own row changes its CONVERSION and must leave its 80-column character
    alone.  Checked over whatever multi-character codes the table currently
    names, so it keeps holding as rows are added."""
    from pyroorb import obs80
    for code in photometry.BAND_SHIFT:
        if len(code.strip()) < 2:
            continue
        letter = photometry.ades_band_filter(code)
        if letter is None:
            continue
        assert obs80.band_to_obs80(code) == letter, code
