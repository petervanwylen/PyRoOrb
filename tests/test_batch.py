"""Tests for the `--batch` astrometry-file driver in fit_object.py: format
detection (OBS80 vs ADES PSV/XML), file parsing + designation grouping, and an
offline end-to-end run that fits several designations from one OBS80 blob and
emits one MPCORB / SOF row per designation.

The batch is identification-agnostic: obs are grouped by the designation each
carries and fit independently (no cross-designation linking).
"""
import pytest
from pyroorb import cli
from pyroorb import obs_sources
from pyroorb import egress
from pyroorb import obs80

# drives the batch CLI over real astrometry
pytestmark = pytest.mark.slow


# --- format detection (no fitting) -----------------------------------------
def test_detect_format():
    obs80_blob = ("     J93T00Z  C1993 10 15.23721001 47 12.581+14 49 42.10"
                  "         18.7 V      691\n")
    assert obs_sources._detect_obs_format(obs80_blob) == "obs80"
    psv = ("# version=2017\n# observatory\n! mpcCode 691\n"
           "permID |provID  |mode|stn |obsTime |ra |dec\n"
           "       |2001 KQ76|CCD |691 |2001-05-24T00:19:31.6Z|180.0|10.0\n")
    assert obs_sources._detect_obs_format(psv) == "ades"
    xml = '<?xml version="1.0"?>\n<ades version="2017"><obsBlock/></ades>'
    assert obs_sources._detect_obs_format(xml) == "ades"


# --- native ADES PSV parsing (no third-party dependency) -------------------
def test_psv_parses_natively():
    """ades.parse_ades_file ingests a '|'-delimited PSV blob with no adam_core
    (or any third-party ADES package) on the path -- the Railway-deploy case where
    pasting pipe-separated obs raised ModuleNotFoundError: No module named
    'adam_core'. Exercises metadata-line skipping, fixed-width padding, missing-rms
    Veres fallback, and multiple header blocks in one blob."""
    from pyroorb import ades
    obscodes = {c: (0.0, 0.0, 0.0) for c in ("691", "695", "W84")}
    psv = (
        "# version=2017\n"
        "# observatory\n! mpcCode 695\n"
        "permID |provID   |obsTime                 |ra          |dec         "
        "|rmsRA |rmsDec|mag |band|stn |mode|astCat\n"
        "       |2001 KQ76|2001-05-24T00:19:31.600Z|180.000000  |10.000000   "
        "|0.500 |0.500 |20.0|V   |695 |CCD |Gaia2\n"
        # second header block (different columns, no rms) -> Veres-station sigma
        "permID|provID|obsTime|ra|dec|stn|mode|astCat\n"
        "      |2001 KQ76|2001-05-25T00:00:00.000Z|180.5|10.5|W84|CCD|Gaia2\n"
    )
    obs = ades.parse_ades_file(psv, obscodes)
    assert len(obs) == 2, len(obs)
    assert {o["designation"] for o in obs} == {"2001 KQ76"}, obs
    a, b = sorted(obs, key=lambda o: o["jd_utc"])
    assert abs(a["ra_deg"] - 180.0) < 1e-9 and abs(a["dec_deg"] - 10.0) < 1e-9
    assert a["obscode"] == "695" and b["obscode"] == "W84"
    assert a["uncert_src"] == "ADES" and abs(a["sigma_ra"] - 0.5) < 1e-9   # rms honoured
    assert b["uncert_src"] == "Veres2017"                                  # missing rms -> fallback


def _ades_records(ra, dec, **extra):
    """One ADES_DF-shaped record at (ra, dec), as the MPC API delivers them (numbers,
    not strings)."""
    r = {"stn": "691", "provid": "2001 KQ76", "mode": "CCD",
         "obstime": "2001-05-24T00:19:31.6Z", "ra": ra, "dec": dec}
    r.update(extra)
    return [r]


def test_zero_ra_or_dec_is_a_valid_observation():
    """0 deg RA (the meridian the coordinate system starts at) and 0 deg Dec (the
    celestial equator) are real positions. A truthiness guard -- `if not (stn and ra and
    dec and ot)` -- silently DISCARDED every observation that landed on either, with no
    warning and no unknown-obscode trace: the arc just came back short."""
    from pyroorb import ades
    obscodes = obs_sources.load_obscodes()
    for ra, dec in ((0.0, -1.5), (12.0, 0.0), (0.0, 0.0), (12.0, -1.5)):
        obs = ades.records_to_obs(_ades_records(ra, dec), obscodes)
        assert len(obs) == 1, f"ra={ra} dec={dec} was dropped"
        assert obs[0]["ra_deg"] == ra and obs[0]["dec_deg"] == dec
    # ...while a genuinely ABSENT field still drops the record (None and "" alike)
    for ra, dec in ((None, -1.5), (12.0, None), ("", -1.5), (12.0, "")):
        assert ades.records_to_obs(_ades_records(ra, dec), obscodes) == []


def test_uncert_src_label_matches_the_sigma_actually_used():
    """`uncert_src` must name where the sigma CAME from. A reported rms of 0 is not a
    perfect measurement (weights.sigma_for rejects it and falls back to the station
    model), so the obs is honestly Veres2017 -- the label and the value now share one
    predicate (weights.ades_rms_usable) and cannot drift apart."""
    from pyroorb import ades
    from pyroorb.weights import sigma_for
    obscodes = obs_sources.load_obscodes()
    for rms, want in ((0.3, "ADES"), (0.0, "Veres2017"), (None, "Veres2017"),
                      (-1.0, "Veres2017")):
        o = ades.records_to_obs(_ades_records(12.0, -1.5, rmsra=rms, rmsdec=rms),
                                obscodes)[0]
        assert o["uncert_src"] == want, (rms, o["uncert_src"])
        sra, sdec = sigma_for("691", None, rms, rms)
        assert (o["sigma_ra"], o["sigma_dec"]) == (sra, sdec)      # label follows the value


def test_psv_parser_has_no_adam_core_import():
    """Guard the regression directly: ades.py must not import adam_core (the missing
    dependency that broke pasted PSV on Railway)."""
    from pyroorb import ades; import inspect
    assert "import adam_core" not in inspect.getsource(ades), "adam_core import is back"


# --- parsing + grouping (no fitting) ---------------------------------------
def _two_designation_obs80_blob(min_obs=8, max_obs=20, n=2):
    """Build an OBS80 blob from the obs of `n` small single-designation fixtures."""
    from _fixtures import load_fixture, iter_fixtures
    lines, labels = [], []
    for p in iter_fixtures():
        fx = load_fixture(p)
        if fx.get("benchmark"):
            continue
        obs = fx["obs"]
        if len({o["designation"] for o in obs}) != 1:
            continue                               # skip combined / PAIR fixtures
        if not (min_obs <= len(obs) <= max_obs):
            continue
        lines += [obs80.format_obs80(o) for o in obs]
        labels.append(obs[0]["designation"])       # the real (spaced) designation
        if len(labels) == n:
            break
    return "\n".join(lines) + "\n", labels


def test_read_and_group_obs80():
    blob, labels = _two_designation_obs80_blob()
    assert len(labels) == 2, labels
    obscodes = obs_sources.load_obscodes()
    obs, fmt = obs_sources.read_obs_file(blob, obscodes)
    assert fmt == "OBS80"
    groups = obs_sources._group_by_designation(obs)
    assert len(groups) == 2, list(groups)          # two distinct designations
    assert all(g for g in groups.values())         # each non-empty


# --- end-to-end fit of several designations from one file ------------------
def test_run_batch_emits_one_row_per_designation():
    blob, labels = _two_designation_obs80_blob()
    obscodes = obs_sources.load_obscodes()
    fits, failures, fmt = cli.run_batch(blob, obscodes)
    assert not failures, failures
    assert len(fits) == 2, [f.designation for f in fits]
    # MPCORB: header + one row per object
    mp = egress.mpcorb_batch(fits).split("\n")
    assert mp[0] == egress.MPCORB_HEADER and len(mp) == 3
    # SOF: self-describing header + one aligned record per object
    sof = egress.sof_batch(fits).split("\n")
    assert sof[0] == egress.SOF_TEMPLATE and len(sof) == 3
    assert all(len(l) == len(egress.SOF_TEMPLATE) for l in sof[1:])
    # the fitted designations are exactly the two we put in (identification-agnostic)
    assert sorted(f.designation for f in fits) == sorted(labels)


def test_batch_one_restricts_to_single_designation():
    blob, labels = _two_designation_obs80_blob()
    obscodes = obs_sources.load_obscodes()
    fits, failures, _ = cli.run_batch(blob, obscodes, only=labels[0])
    assert len(fits) == 1 and fits[0].designation == labels[0], [f.designation for f in fits]


def test_too_few_obs_says_so_instead_of_crashing():
    """One observation cannot determine 6 orbital parameters. It used to reach the IOD
    cascade's Lambert solver with a zero-length baseline and come back as
    'fit failed: ZeroDivisionError: float division by zero' -- a real user report shape
    (paste a single line into the web loader). fit_orbit is the one door every path goes
    through, so the honest message is raised there."""
    from pyroorb import fit_object; from pyroorb import twobody; import numpy as np
    obscodes = obs_sources.load_obscodes()
    blob, _labels = _two_designation_obs80_blob()
    obs = obs_sources.parse_obs(blob, obscodes)[:1]
    assert len(obs) == 1
    try:
        fit_object.fit_orbit(obs, obscodes, "single", {"designation": "single"})
        assert False, "a 1-obs fit should be refused"
    except ValueError as ex:
        assert "at least" in str(ex) and "observation" in str(ex), ex
    # ...and the underlying degeneracy no longer divides by zero: a zero-baseline Lambert
    # (identical endpoints, dt=0) bails to the next IOD method instead of raising.
    r = np.array([1.0, 0.0, 0.0])
    assert twobody.lambert(r, r.copy(), 0.0) is None
