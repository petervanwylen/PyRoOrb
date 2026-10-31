"""Unit + offline end-to-end tests for orbit egress (egress.py): the MPCORB
one-line record and the SOF (Structured Orbit Format) element line.

The pure helpers (calendar, mutant-hex, SOF date truncation) are checked against
hand-computed values; the end-to-end check re-fits one cached fixture OFFLINE and
asserts the serialized fields round-trip back to the fitted elements and that the
SOF data line stays column-aligned with its self-describing header.
"""
import pytest
import os
from pyroorb import egress

# the end-to-end checks refit a cached fixture
pytestmark = pytest.mark.slow

WJ = 2459600.5     # a fixed Twritten JD (2022-01-21.0) for reproducible output


# --- pure helpers -----------------------------------------------------------
def test_jd_to_ymd():
    y, m, d = egress._jd_to_ymd(2451545.0)         # 2000-01-01 12:00 TT
    assert (y, m) == (2000, 1) and abs(d - 1.5) < 1e-6, (y, m, d)
    y, m, d = egress._jd_to_ymd(2457670.8936491474)  # 2016-10-09.39
    assert (y, m, int(d)) == (2016, 10, 9), (y, m, d)


def test_sof_date_is_truncated_not_rounded():
    # YYYYMMDD. + 11 day-fraction digits; the SOF engine clips to field width.
    s = egress._sof_date(2451545.0)
    assert s.startswith("20000101.5") and len(s) == 20, s
    # a fraction that would round UP must still truncate DOWN when clipped
    s = egress._sof_date(2457670.8936491474)
    assert s[:11] == "20161009.39", s


def test_mutant_hex_and_packed_epoch():
    assert (egress._mutant_hex_char(20), egress._mutant_hex_char(10),
            egress._mutant_hex_char(31)) == ("K", "A", "V")
    # 2016-10-09: century K(20), year 16, hex month A(10), day 9
    assert egress._packed_epoch(2457670.8936491474) == "K16A9"
    # 2000-01-01: century K(20), year 00, month 1, day 1
    assert egress._packed_epoch(2451544.5) == "K0011"


# --- offline end-to-end on a cached fixture ---------------------------------
def _fit_fixture(label="1993_TZ"):
    from _fixtures import load_fixture, iter_fixtures
    import pyroorb.fit_object as fo
    from pyroorb import obs_sources
    path = next(p for p in iter_fixtures() if os.path.basename(p).startswith(label))
    fx = load_fixture(path)
    oc = obs_sources.load_obscodes()
    return fo.fit_orbit(fx["obs"], oc, desig=fx["meta"]["designation"],
                        meta=fx["meta"], verbose=False, force=fx.get("force", False))


def test_sof_columns_align_and_round_trip():
    fit = _fit_fixture()
    sof = egress.to_sof(fit, written_jd=WJ)
    header, data = sof.split("\n")
    # the data line must be the same width as the self-describing header
    assert len(data) == len(header), (len(data), len(header))
    el = fit.elements()
    # locate the q / e columns from the header field boundaries and parse them back
    # (a coarse but real round-trip: the serialized number reproduces the element)
    fields, pos = [], 0
    for seg in header.split("|"):
        fields.append((seg.strip().split()[0] if seg.strip() else "", pos, pos + len(seg)))
        pos += len(seg) + 1                          # +1 for the separator space
    by_name = {n: (a, b) for n, a, b in fields}
    a0, b0 = by_name["e"]
    assert abs(float(data[a0:b0]) - el["e"]) < 1e-6, (data[a0:b0], el["e"])
    a0, b0 = by_name["q"]
    assert abs(float(data[a0:b0]) - el["q"]) < 1e-6, (data[a0:b0], el["q"])
    # n_u (used) and n_o (total) integers
    a0, b0 = by_name["n_u"]
    assert int(data[a0:b0]) == fit.n_observations()


def test_mpcorb_fixed_columns():
    fit = _fit_fixture()
    line = egress.to_mpcorb(fit, written_jd=WJ)
    # packed designation (cols 1-7) and packed epoch (cols 21-25)
    assert line[:7] == (fit.packed_designation or "").strip()[:7], line[:7]
    assert line[20:25] == egress._packed_epoch(fit.epoch_jd()), line[20:25]
    # semimajor axis at cols 92-103 reproduces the fitted a
    assert abs(float(line[91:103]) - fit.elements()["a"]) < 1e-6, line[91:103]
    # perturber tag for our always-on 8 planets
    assert line[142:145] == "M-v" and line[146:148] == "38", line[142:148]
    # find_orb caps the record at 202 chars
    assert len(line) <= 202, len(line)


class _StubFit:
    """Minimal stand-in exposing only what to_mpcorb reads -- lets us force a huge /
    hyperbolic semimajor axis without fitting a pathological object in an offline test."""
    def __init__(self, a):
        self._a = a
        self.packed_designation = self.designation = "RL00jsN"
        self._obs = [{"jd_utc": 2460859.5}, {"jd_utc": 2460879.5}]
        self._mask = [True, True]
    def elements(self):
        return {"a": self._a, "e": 0.9999, "q": 22.97, "Q": abs(self._a) * 2,
                "i": 2.678, "asc_node": 201.13, "arg_per": 125.97, "M": 359.99,
                "Tp": 2460000.5, "n": 1e-8, "P": 1.1e8}
    def absolute_magnitude(self): return (7.14, None)
    def slope_parameter(self): return 0.15
    def epoch_jd(self): return 2460859.6756
    def n_observations(self): return 2
    def n_oppositions(self): return 1
    def rms_residual(self): return 0.091


def test_mpcorb_huge_a_truncates_not_crashes():
    # An essentially-unconstrained short arc can fit a=2e5 au (RL00jsN: a=229650), which
    # overflows the 11-col a-field of %11.7f. find_orb clips at the column boundary
    # (elem_out.cpp buff[103]='\0'); we must do the same, not crash on an assert (the old
    # behaviour killed --batch on such objects with AssertionError: 106).
    for a in (229650.123, 1.5e6, -5000.0):        # huge-bound, larger, hyperbolic-negative
        line = egress.to_mpcorb(_StubFit(a), written_jd=WJ)
        assert len(line) <= 202, (a, len(line))    # still capped at find_orb's record length
        assert line[91] == " ", (a, repr(line[91]))  # a-field starts at col 92 -> col 91 is its space
        # the reference block + perturber tag still land at their fixed columns (165+)
        assert line[142:145] == "M-v", (a, line[142:148])


def test_batch_assembly_headers_and_rows():
    # one fitted object reused twice: a batch is just one header + N data rows.
    fit = _fit_fixture()
    mp = egress.mpcorb_batch([fit, fit], written_jd=WJ)
    lines = mp.split("\n")
    assert lines[0] == egress.MPCORB_HEADER
    assert len(lines) == 3 and lines[1] == lines[2]    # header + 2 identical rows
    sof = egress.sof_batch([fit, fit], written_jd=WJ)
    slines = sof.split("\n")
    assert slines[0] == egress.SOF_TEMPLATE
    assert len(slines) == 3
    # every SOF record stays column-aligned with the self-describing header
    assert all(len(l) == len(egress.SOF_TEMPLATE) for l in slines[1:]), [len(l) for l in slines]


def test_eightline_format_and_values():
    """The find_orb 8-line element block: structure, line prefixes, and a value
    round-trip out of the formatted columns. (Byte-exactness vs find_orb's
    elements_in_mpc_format was verified against the compiled lunar/showelem.cpp
    during development; this guards the structure + numbers in CI.)"""
    fit = _fit_fixture()
    block = egress.to_eightline(fit)
    lines = block.split("\n")
    assert len(lines) == 8, lines
    el = fit.elements()
    assert lines[0] == fit.designation
    assert lines[1].startswith("   Perihelion ") and lines[1].endswith(")")
    assert lines[2].startswith("Epoch ") and "TT = JDT" in lines[2]
    assert lines[3].startswith("M") and "(2000.0)" in lines[3] \
        and lines[3].rstrip().endswith("Q")
    assert lines[4].startswith("n") and "Peri." in lines[4]
    assert lines[5].startswith("a") and "Node" in lines[5]
    assert lines[6].startswith("e") and "Incl." in lines[6]
    assert lines[7].startswith("P") and "q " in lines[7] and "Q " in lines[7]
    # a (line 6) and e (line 7) read back to the fitted elements
    assert abs(float(lines[5].split("Node")[0][1:]) - el["a"]) < 1e-6, lines[5]
    assert abs(float(lines[6].split("Incl.")[0][1:]) - el["e"]) < 1e-6, lines[6]
    # the M line carries the (normalized) mean anomaly
    assert abs(float(lines[3].split("(2000.0)")[0][1:]) - (el["M"] % 360.0)) < 1e-4, lines[3]


class _UnboundStubFit:
    """Stand-in for an UNBOUND (e >= 1) solution, so the egress writers can be exercised
    on one without fitting a pathological object offline. An under-constrained short arc
    reaches this regime routinely -- the eccentricity cap only applies to the fit, not to
    what a caller may hand the writers."""
    def __init__(self, a, e):
        self._a, self._e = a, e
        self.packed_designation = self.designation = "K25Q03J"
        self._obs = [{"jd_utc": 2460859.5}, {"jd_utc": 2460879.5}]
        self._mask = [True, True]
    def elements(self):
        # Tp/M/n/P are NaN for an unbound orbit (orbital_elements.element_vector), which
        # is exactly what the writers are handed in the real case.
        return {"a": self._a, "e": self._e, "q": abs(self._a) * abs(1.0 - self._e),
                "Q": self._a * (1.0 + self._e), "i": 12.3, "asc_node": 201.13,
                "arg_per": 125.97, "M": float("nan"), "Tp": float("nan"),
                "n": float("nan"), "P": float("nan")}
    def absolute_magnitude(self): return (18.4, None)
    def slope_parameter(self): return 0.15
    def epoch_jd(self): return 2460859.6756
    def n_observations(self): return 2
    def n_total_observations(self): return 2
    def n_oppositions(self): return 1
    def rms_residual(self): return 0.42


def test_eightline_hyperbolic_omits_period_line_and_shows_z():
    """find_orb (showelem.cpp) guards the final P/H/q/Q line with `!is_cometary &&
    ecc < 1.` and prints the RECIPROCAL semi-major axis 'z' in place of 'a'. We used to do
    neither: `a^1.5` on a negative a raised `ValueError: math domain error`, killing
    --out-format elem / the batch elem writer / /api/export/elements."""
    block = egress.to_eightline(_UnboundStubFit(a=-3.0, e=1.4))
    lines = block.split("\n")
    # Perihelion line (line 2) needs a finite Tp, so an unbound block is 6 lines here:
    # designation / Epoch / M / n+Peri / z+Node / e+Incl -- and NO period line.
    assert len(lines) == 6, lines
    assert lines[4].startswith("z") and "Node" in lines[4], lines[4]
    assert abs(float(lines[4].split("Node")[0][1:]) - (1.0 / -3.0)) < 1e-9, lines[4]
    assert lines[5].startswith("e") and "Incl." in lines[5], lines[5]
    assert not any(l.startswith("P") for l in lines), block   # no period/apoapsis line
    assert "Q " not in block and " q " not in block, block    # nor a nonsensical apoapsis


def test_eightline_exactly_parabolic_does_not_divide_by_zero():
    """e == 1.0 exactly: the old apoapsis q(1+e)/(1-e) was a ZeroDivisionError. The
    period line is omitted, and with |a| beyond find_orb's 100000 au sanity bound the
    'z' field blanks rather than printing a meaningless reciprocal."""
    block = egress.to_eightline(_UnboundStubFit(a=1.0e9, e=1.0))
    lines = block.split("\n")
    assert not any(l.startswith("P") for l in lines), block
    node_line = [l for l in lines if "Node" in l][0]
    assert not node_line.startswith("a") and not node_line.startswith("z"), node_line
    assert node_line.lstrip().startswith("Node"), node_line


def test_eightline_marginally_unbound_still_prints_z():
    """A short-arc solution that drifts a hair past e=1 keeps a usable 1/a, so 'z' is
    printed (the blanking rule is for |a| outside find_orb's -0.0011 .. -100000 window)."""
    block = egress.to_eightline(_UnboundStubFit(a=-2500.0, e=1.0004))
    zline = [l for l in block.split("\n") if "Node" in l][0]
    assert zline.startswith("z"), zline
    assert abs(float(zline.split("Node")[0][1:]) - (1.0 / -2500.0)) < 1e-12, zline


def test_mpcorb_and_sof_survive_an_unbound_orbit():
    """All THREE writers must handle e >= 1, not just the eight-line block. MPCORB has no
    unbound form and now REFUSES by name, as find_orb does (its mpc-format file is guarded by
    ecc < .999999) -- it used to emit a clipped line of nonsense. SOF used to crash: its Tp
    column ran _sof_date on the NaN perihelion time of an unbound orbit and raised out of
    math.floor. A quantity the orbit does not have is now blanked (find_orb's absent-value
    rule), and the record stays column-aligned with its template."""
    fit = _UnboundStubFit(a=-3.0, e=1.4)
    with pytest.raises(ValueError, match="no MPCORB form"):
        egress.to_mpcorb(fit, written_jd=WJ)
    row = egress.to_sof(fit, with_header=False, written_jd=WJ)
    assert len(row) == len(egress.SOF_TEMPLATE), (len(row), len(egress.SOF_TEMPLATE))
    # Tp is the 4th '|'-delimited column of the template; it must be blank, not "nan"
    tp_col = egress.SOF_TEMPLATE.split("|")[3]
    start = len(egress.SOF_TEMPLATE.split("|")[0]) + len(egress.SOF_TEMPLATE.split("|")[1]) \
        + len(egress.SOF_TEMPLATE.split("|")[2]) + 3
    assert row[start:start + len(tp_col)].strip() == "", repr(row[start:start + len(tp_col)])
    assert "nan" not in row.lower(), row
    # ...while the quantities an unbound orbit DOES have are still written
    assert "1.20000" in row or abs(float(row.split()[2]) - 1.2) < 1e-6, row


# --- cartesian state + covariance (JSON) ------------------------------------
def test_state_rotation_is_orthogonal_and_block_diagonal():
    """The equatorial->ecliptic 6x6 is blockdiag(R, R) with R orthogonal, so it
    preserves both |r| and |v| and never mixes position into velocity."""
    import numpy as np
    M = egress._eq_to_ecl_6x6()
    assert np.allclose(M @ M.T, np.eye(6), atol=1e-14)
    assert np.allclose(M[:3, 3:], 0.0) and np.allclose(M[3:, :3], 0.0)
    assert np.allclose(M[:3, :3], M[3:, 3:])


def test_state_block_correlation_diagonal_is_unity():
    import numpy as np
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    cov6 = np.eye(6)
    cov6[:2, :2] = cov
    blk = egress._state_block(np.arange(6.0), cov6)
    assert abs(blk["sigma"]["x"] - 2.0) < 1e-12 and abs(blk["sigma"]["y"] - 3.0) < 1e-12
    assert abs(blk["correlation"][0][1] - 1.0 / 6.0) < 1e-12
    assert all(abs(blk["correlation"][i][i] - 1.0) < 1e-12 for i in range(6))


def test_state_json_matches_the_fit():
    """End-to-end on a cached fixture: the exported state IS the fit's own state and
    covariance, the two frames are one orbit under the obliquity rotation, and the
    ecliptic state reproduces the reported elements."""
    import json
    import numpy as np
    fit = _fit_fixture()
    doc = json.loads(egress.to_state_json(fit))

    assert doc["object"] == fit.designation
    assert abs(doc["epoch"]["jd_tdb"] - fit.epoch_jd()) < 1e-9
    assert abs(doc["epoch"]["mjd_tdb"] - (fit.epoch_jd() - 2400000.5)) < 1e-9
    assert doc["covariance_order"] == ["x", "y", "z", "dx", "dy", "dz"]

    eq = doc["equatorial_J2000"]
    state = np.asarray(fit.state(), float)[:6]
    assert np.allclose([eq["state"][k] for k in doc["covariance_order"]], state, rtol=0, atol=1e-15)
    cov = np.asarray(fit.covariance_matrix("state"), float)[:6, :6]
    assert np.allclose(np.asarray(eq["covariance"]), cov, rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(eq["covariance"]), np.asarray(eq["covariance"]).T)
    assert np.allclose([eq["sigma"][k] for k in doc["covariance_order"]],
                       np.sqrt(np.diag(cov)), rtol=1e-12, atol=0)

    # the ecliptic block is the SAME orbit rotated: distance/speed invariant, and the
    # covariance carried through as M C M^T (so its trace-in-position is preserved too)
    ec = doc["ecliptic_J2000"]
    r_eq = np.array([eq["state"][k] for k in ("x", "y", "z")])
    r_ec = np.array([ec["state"][k] for k in ("x", "y", "z")])
    assert abs(np.linalg.norm(r_eq) - np.linalg.norm(r_ec)) < 1e-14
    M = egress._eq_to_ecl_6x6()
    assert np.allclose(np.asarray(ec["covariance"]), M @ cov @ M.T, rtol=1e-10, atol=1e-30)

    # ...and it is the frame the reported elements live in: rebuilding the state from
    # the fit's own elements lands on the exported ecliptic position.
    import pyroorb.orbital_elements as oe
    el = fit.elements()
    rebuilt = oe.state_eq_from_elements(el["a"], el["e"], el["i"], el["asc_node"],
                                        el["arg_per"], el["M"])
    assert np.allclose(oe._eq_to_ecl(np.asarray(rebuilt)[:3]), r_ec, rtol=1e-6, atol=1e-9)
