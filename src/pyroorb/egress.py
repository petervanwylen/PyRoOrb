"""Orbit *egress* — serialize a ``FittedOrbit`` into the interchange formats the
wider ecosystem reads: MPC's one-line **MPCORB** record and Project Pluto's
**SOF** (Structured Orbit Format). This is the write side (we already *ingest*
both; this closes the round-trip).

Both formats at least initially were ported logic from Bill Gray's find_orb (GPL-2.0; logic openly
ported, no source lines copied):

  * MPCORB  — ``elem_out.cpp:elements_in_mpcorb_format`` (the ~203-col fixed line
    with the mutant-hex packed epoch, mean-motion, arc-length blurb and rms).
  * SOF     — ``elem_ou2.cpp:put_elements_into_sof`` driven by the ``orbitdef.sof``
    column template: each ``|``-delimited header field names a quantity and its
    width; a ``.`` inside the field fixes the decimal places; dates are rendered
    to 11 day-fraction places and then *truncated* (not rounded) to the field
    width — exactly find_orb's snprintf-into-a-short-buffer trick.
  * 8-line  — ``showelem.cpp:elements_in_mpc_format`` (Project Pluto's "pretty"
    element block: designation / Perihelion / Epoch / M..PQ / n+Peri.+PQ /
    a+Node+PQ / e+Incl.+PQ / P+H+G+q+Q), originally a byte-faithful port at
    find_orb's default precision 5. The inline ``+/- sigma`` surgery find_orb layers on top
    (elem_out.cpp) is NOT reproduced — that depends on find_orb-internal sigma
    storage; the bare block was ported to match what find_orb prints without sigmas.

Everything heliocentric, J2000 ecliptic, which is what both formats assume and
what ``FittedOrbit`` already produces.
"""
import math


# ---------------------------------------------------------------------------
# Calendar / packing helpers (port of find_orb's date + mutant-hex utilities)
# ---------------------------------------------------------------------------
def _jd_to_ymd(jd):
    """Julian Date -> (year, month, day_with_fraction) on the Julian/Gregorian
    calendar (Julian before 1582-10-15, Gregorian after) — find_orb's
    CALENDAR_JULIAN_GREGORIAN convention. ``day`` carries the day fraction."""
    jd = jd + 0.5
    Z = math.floor(jd)
    F = jd - Z
    if Z < 2299161:                       # Julian calendar
        A = Z
    else:                                 # Gregorian
        alpha = math.floor((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - math.floor(alpha / 4)
    B = A + 1524
    C = math.floor((B - 122.1) / 365.25)
    D = math.floor(365.25 * C)
    E = math.floor((B - D) / 30.6001)
    day = B - D - math.floor(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715
    return int(year), int(month), day


def _sof_date(jd):
    """A SOF date string: ``YYYYMMDD.`` + an 11-digit day fraction. The SOF engine
    truncates this to each field's width (find_orb writes 11 places then lets the
    fixed-width field clip it), so always hand back the full-precision string."""
    year, month, day = _jd_to_ymd(jd)
    di = int(day)
    frac = day - di
    frac11 = ("%.11f" % frac)[2:13]       # 11 digits after the point, no "0."
    return "%04d%02d%02d.%s" % (year, month, di, frac11)


_MUTANT = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _mutant_hex_char(n):
    """MPC 'mutant hex': 0-9 -> '0'-'9', 10-35 -> 'A'-'Z', 36-61 -> 'a'-'z'."""
    return _MUTANT[n] if 0 <= n < len(_MUTANT) else "~"


def _packed_epoch(jd):
    """The 5-char MPC packed epoch (e.g. 2021-04-20 -> 'K214K'): century-letter,
    2-digit year-in-century, hex month, mutant-hex day."""
    year, month, day = _jd_to_ymd(jd)
    di = int(day)
    cen = _mutant_hex_char(year // 100) if 0 <= year < 6200 else "~"
    return "%c%02d%X%c" % (cen, abs(year) % 100, month, _mutant_hex_char(di))


# ---------------------------------------------------------------------------
# MPCORB one-line export
# ---------------------------------------------------------------------------
MPCORB_MAX_ECC = 0.999999    # find_orb writes its mpc-format file only below this eccentricity


def to_mpcorb(fit, written_jd=None, reference="PyRoOrb "):
    """Serialize ``fit`` to a single MPCORB-format line (find_orb's
    ``elements_in_mpcorb_format``). Heliocentric J2000-ecliptic mean elements,
    the mutant-hex packed epoch, mean motion, arc-length blurb and unweighted rms.

    ``written_jd`` is the orbit-computation date stamped into the reference block
    (defaults to *now*); pass a fixed JD for reproducible output. ``reference``
    is the 8-char free-text source tag (kept at 8 chars to preserve columns)."""
    el = fit.elements()
    # An MPCORB line has no way to carry an unbound orbit (a mean anomaly in [0, 360), a
    # mean motion and a positive semimajor axis are its columns), and find_orb writes none:
    # its mpc-format file is guarded by `helio_elem.ecc < .999999` (elem_out.cpp,
    # write_out_elements_to_file). Refuse by name rather than emit a line of nonsense.
    if not float(el["e"]) < MPCORB_MAX_ECC:
        raise ValueError(f"an unbound orbit (e = {float(el['e']):.6f}) has no MPCORB form; "
                         "use the element block or SOF")
    packed = (fit.packed_designation or "").strip()[:8]
    H, _ = fit.absolute_magnitude()
    G = fit.slope_parameter()

    # cols 1-19: packed desig, H, G  (H blank when we have no photometry)
    hg = ("%5.2f" % max(-9.9, min(99.9, H))) if H is not None else " " * 5
    buff = "%-8s%s %5.2f " % (packed, hg, G)
    assert len(buff) == 20, len(buff)
    # cols 21-25: packed epoch
    buff += _packed_epoch(fit.epoch_jd())
    assert len(buff) == 25, len(buff)
    # cols 26-79: M, arg_per, asc_node, incl, e
    buff += "%10.5f%11.5f%11.5f%11.5f%11.7f" % (
        el["M"] % 360.0, el["arg_per"] % 360.0, el["asc_node"] % 360.0,
        el["i"] % 360.0, el["e"])
    assert len(buff) == 79, len(buff)
    # cols 80-91 (n) and 92-103 (a) are fixed-width fields. find_orb hard-clips each at
    # its column boundary (buff[91]='\0', buff[103]='\0' in elem_out.cpp) rather than
    # widening the line: a huge or hyperbolic-negative `a` (an under-constrained short arc
    # can fit a=2e5 au) overflows %11.7f, so we truncate exactly as find_orb does -- the
    # low-order digits are chopped, keeping the line 103 cols. Was an assert, which crashed
    # --batch on such objects (e.g. RL00jsN, a=229650 -> 106-char field).
    buff = (buff + " %11.8f" % el["n"])[:91]    # cols 80-91: mean motion (deg/day)
    buff = (buff + " %11.7f" % el["a"])[:103]   # cols 92-103: semimajor axis (au)

    n_used = fit.n_observations()
    n_opp = fit.n_oppositions()
    import time as _time
    wj = written_jd if written_jd is not None else 2440587.5 + _time.time() / 86400.0
    wy, wm, wd = _jd_to_ymd(wj)
    ref8 = (reference + " " * 8)[:8]
    # cols 104-165: reference block (computation date, n_used, n_opp, source, flags)
    buff += ("    FO %02d%02d%02d %5d %3d ****-**** ****         %s   %04x"
             % (wy % 100, wm, int(wd), n_used, n_opp, ref8, 0))
    assert len(buff) == 165, len(buff)

    # The tail (full designation + last-obs date) lands at fixed columns; build the
    # rest as a mutable buffer so the arc-length / rms / perturber overwrites land
    # exactly where find_orb pokes them (buff+127, +137, +142, +146).
    full = fit.designation
    if len(full) > 28:
        buff += full
    else:
        buff += " %-28s" % full
    incl = [o["jd_utc"] for o, m in zip(fit._obs, fit._mask) if m]
    j0, j1 = incl[0], incl[-1]
    ly, lm, ld = _jd_to_ymd(j1)
    buff += "%04d%02d%02d" % (ly, lm, int(ld))

    b = list(buff)

    def _poke(col, text):                 # overwrite at a fixed 0-indexed column
        for k, ch in enumerate(text):
            if col + k < len(b):
                b[col + k] = ch

    arc = j1 - j0                         # arc-length blurb at col 127
    if arc < 99.0 / 86400.0:
        _poke(127, "%4.1f sec " % (arc * 86400.0))
    elif arc < 99.0 / 1440.0:
        _poke(127, "%4.1f min " % (arc * 1440.0))
    elif arc < 2.0:
        _poke(127, "%4.1f hrs " % (arc * 24.0))
    elif arc < 600.0:
        _poke(127, "%4d days" % (int(arc) + 1))
    else:
        _poke(127, "%4d-%4d" % (int(2000.0 + (j0 - 2451545.0) / 365.25),
                                int(2000.0 + (j1 - 2451545.0) / 365.25)))
    b[136] = " "

    rms = fit.rms_residual()              # unweighted rms at col 137
    if rms < 9.9:
        _poke(137, "%4.2f" % rms)
    elif rms < 99.9:
        _poke(137, "%4.1f" % rms)
    elif rms < 9999.0:
        _poke(137, "%4.0f" % rms)
    b[141] = " "
    # perturber columns: we always carry the full 8 planets -> "M-v" / "38"
    _poke(142, "M-v")
    _poke(146, "38")
    return "".join(b).rstrip()


MPCORB_HEADER = (
    "Des'n        H     G   Epoch     M        Peri.      Node       Incl.   "
    "    e            n           a        Reference #Obs #Opp    Arc    rms  "
    "Perts   Computer")


# ---------------------------------------------------------------------------
# SOF (Structured Orbit Format) export
# ---------------------------------------------------------------------------
# The default find_orb element template (orbitdef.sof, verbatim incl. the trailing
# '^' record-end marker). The header line IS the column spec: each '|'-delimited
# field's width is its character count and a '.' fixes its decimal places.
SOF_TEMPLATE = ("Name         |C |Te      .  |Tp              |q                |"
                "i  .      |Om .      |om .      |e         |rms |n_u  |n_o  |"
                "Tfirst  .     |Tlast   .     |Twritten.   |H .  |G . ^")


def _sof_field_value(spec, fit, el, written_jd):
    """Resolve one SOF template field (its leading specifier) to a renderable
    ('text'|'num'|'int'|'date'|'blank', payload). Mirrors put_elements_into_sof's
    dispatch on the first character(s) of each field."""
    incl = [o["jd_utc"] for o, m in zip(fit._obs, fit._mask) if m]
    if spec[0] == "N":                    # Name: packed desig, left-justified
        return ("text", (fit.packed_designation or "").strip(), False)
    if spec[:3] == "n_o":
        return ("int", fit.n_total_observations())
    if spec[:3] == "n_u":
        return ("int", fit.n_observations())
    if len(spec) > 1 and spec[1] == " ":  # single-char specifier
        c = spec[0]
        if c == "q":  return ("num", el["q"])
        if c == "a":  return ("num", el["a"])
        if c == "Q":  return ("num", el["Q"])
        if c == "e":  return ("num", el["e"])
        if c == "i":  return ("num", el["i"] % 360.0)
        if c == "H":
            H, _ = fit.absolute_magnitude()
            return ("num", H) if H is not None else ("blank", None)
        if c == "G":  return ("num", fit.slope_parameter())
        if c == "C":  return ("int", 0)   # central object: Sun -> 0 -> blank
    if spec[0] == "T":
        if spec[1] == "p":  return ("date", el["Tp"])
        if spec[1] == "e":  return ("date", fit.epoch_jd())
        if spec[1] == "f":  return ("date", incl[0])
        if spec[1] == "l":  return ("date", incl[-1])
        if spec[1] == "w":  return ("date", written_jd)
    if spec[:2] == "Om": return ("num", el["asc_node"] % 360.0)
    if spec[:2] == "om": return ("num", el["arg_per"] % 360.0)
    if spec[:2] == "MA":
        # An unbound orbit's mean anomaly is not an angle (orbital_elements.element_vector): wrapped
        # into [0, 360) it would be wrong, and find_orb's own element writer omits M at e >= 1.
        # Tp still carries the phase.
        if not float(el["e"]) < 1.0:
            return ("blank", None)
        return ("num", el["M"] % 360.0)
    if spec[0] == "r":   return ("num", fit.rms_residual())
    return ("blank", None)


def _renderable(v):
    """True when `v` is a real, finite, non-zero number worth writing into a SOF field.

    find_orb blanks a field whose value is 0 -- 0 is its "absent" sentinel, see
    put_elements_into_sof's `if (date_to_put)` / `else if (val_to_put)` dispatch. We
    extend the same rule to NaN/inf, which is what our element math yields for the
    quantities an UNBOUND orbit does not have (element_vector returns NaN for Tp, M, n
    and P once e >= 1). Rendering those did not merely look wrong, it CRASHED the
    writer: _sof_date -> _jd_to_ymd -> math.floor(NaN) raises ValueError, so `to_sof`
    failed outright on any hyperbolic solution."""
    try:
        return v is not None and math.isfinite(float(v)) and float(v) != 0.0
    except (TypeError, ValueError):
        return False


def _render_sof_line(template, fit, el, written_jd):
    """Render one SOF data line from the column template (port of
    put_elements_into_sof). Field width = template-segment length; decimals fixed
    by the '.' position; a value that is zero or non-finite renders blank (see
    _renderable)."""
    out = []
    i = 0
    n = len(template)
    while i < n and template[i] >= " ":
        j = i
        while j < n and template[j] >= " " and template[j] != "|":
            j += 1
        seg = template[i:j]               # the field spec, e.g. "Te      .  "
        width = len(seg)
        kind, *payload = _sof_field_value(seg, fit, el, written_jd)

        if kind == "text":
            text, right = payload
            cell = ("%*s" if right else "%-*s") % (width, text)
            cell = cell[:width]
        elif kind == "date" and _renderable(payload[0]):
            cell = _sof_date(payload[0])[:width]      # 11-place date, clipped
            cell = "%*s" % (width, cell)
            cell = cell[:width]
        elif kind == "num" and _renderable(payload[0]):
            dot = seg.find(".")
            dec = (width - dot - 1) if dot >= 0 else (width - 1)
            cell = "%*.*f" % (width, dec, payload[0])
            cell = cell[:width]
        elif kind == "int" and payload[0]:
            cell = ("%*d" % (width, payload[0]))[:width]
        else:                                          # blank / zero
            cell = " " * width
        out.append(cell)

        i = j
        if i < n and template[i] == "|":  # field separator -> a single space
            i += 1
            out.append(" ")
    return "".join(out)


def _now_jd():
    import time as _time
    return 2440587.5 + _time.time() / 86400.0


def to_sof(fit, with_header=True, written_jd=None, template=SOF_TEMPLATE):
    """Serialize ``fit`` to a SOF record. With ``with_header`` the self-describing
    column template precedes the data line (so the result is a complete, find_orb-
    readable .sof file). ``written_jd`` stamps the Twritten column (defaults to
    *now*; pass a fixed JD for reproducible output)."""
    el = fit.elements()
    if written_jd is None:
        written_jd = _now_jd()
    line = _render_sof_line(template, fit, el, written_jd)
    return (template + "\n" + line) if with_header else line


# ---------------------------------------------------------------------------
# Multi-object (batch) export
# ---------------------------------------------------------------------------
# MPCORB is inherently one-line-per-object, and SOF is a self-describing table —
# one column-template header followed by one data record per object (find_orb's
# own ``orbits.sof`` is exactly this). So a batch of fits serializes to a single
# header plus N rows in both formats. (The covariance JSON / 8-line element text,
# by contrast, are one-object-per-file.)
def mpcorb_batch(fits, written_jd=None, header=True):
    """Serialize a sequence of ``FittedOrbit`` to MPCORB — one row per object,
    sharing one Twritten/computation date (defaults to *now*). With ``header`` the
    MPCORB column header precedes the rows."""
    if written_jd is None:
        written_jd = _now_jd()
    rows = [to_mpcorb(f, written_jd=written_jd) for f in fits
            if float(f.elements()["e"]) < MPCORB_MAX_ECC]      # an unbound orbit has no row
    return ("\n".join([MPCORB_HEADER] + rows) if header else "\n".join(rows))


def sof_batch(fits, written_jd=None, template=SOF_TEMPLATE):
    """Serialize a sequence of ``FittedOrbit`` to a single SOF table: the
    self-describing column-template header followed by one data record per object
    (find_orb's ``orbits.sof`` layout), all sharing one Twritten date."""
    if written_jd is None:
        written_jd = _now_jd()
    rows = [to_sof(f, with_header=False, written_jd=written_jd, template=template)
            for f in fits]
    return "\n".join([template] + rows)


# ---------------------------------------------------------------------------
# 8-line element block  (port of showelem.cpp:elements_in_mpc_format)
# ---------------------------------------------------------------------------
# Project Pluto's "pretty" element format -- the block find_orb writes to
# elements.txt and shows on screen. This was originally a byte-faithful port of
# elements_in_mpc_format at find_orb's default precision 5, including the
# digit-blanking (lop_digits) and distance-formatting (show_formatted_dist)
# quirks that set the per-field precision. The orientation P/Q columns are the
# Gaussian perihelion/sideways unit vectors rotated to equatorial J2000.
_AU_IN_KM = 1.495978707e8
_SIN_OBLIQ_2000 = 0.397777155931913701597179975942380896684
_COS_OBLIQ_2000 = 0.917482062069181825744000384639406458043
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _lop_digits(chars, pos, precision):
    """Port of showelem.cpp:lop_digits -- skip `precision` digits from `pos`, then
    blank (space-fill) every consecutive digit after that. Operates in place on a
    list of single chars."""
    n = len(chars)
    while precision and pos < n and chars[pos].isdigit():
        pos += 1
        precision -= 1
    while pos < n and chars[pos].isdigit():
        chars[pos] = " "
        pos += 1


def _show_formatted_dist(dist_au, precision):
    """Port of showelem.cpp:show_formatted_dist -- a distance rendered au (or km
    for sub-million-km) with leading spaces, trimmed to `precision`+4 significant
    fractional digits. Returns the string (leading spaces preserved)."""
    if dist_au > 999999.0:
        return " <HUGE>"
    in_km = False
    if dist_au > 9999.0:
        s = "%23.15f" % dist_au
    elif dist_au > 999999.0 / _AU_IN_KM:
        s = "%23.18f" % dist_au
    else:
        s = "%23.16f" % (dist_au * _AU_IN_KM)
        in_km = True
    i = 0
    while s[i] == " ":
        i += 1
    n_digits = 0
    while s[i] != ".":              # count integer digits; i ends just past '.'
        n_digits += 1
        i += 1
    i += 1
    if n_digits < precision + 4:
        s = s[:i + precision + 4 - n_digits]
    else:
        s = s[:i - 1]
    return s + ("km" if in_km else "")


def _normal_vect(v):
    """Ecliptic-J2000 vector -> normalized equatorial-J2000 (showelem.cpp:normal_vect)."""
    n0 = v[0]
    n1 = v[1] * _COS_OBLIQ_2000 - v[2] * _SIN_OBLIQ_2000
    n2 = v[2] * _COS_OBLIQ_2000 + v[1] * _SIN_OBLIQ_2000
    r = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return [n0 / r, n1 / r, n2 / r] if r else [n0, n1, n2]


def _pq_vectors(incl_deg, node_deg, argper_deg):
    """Gaussian perihelion (P) and sideways (Q) unit vectors in equatorial J2000,
    from the ecliptic orbital angles -- the orientation columns of the block."""
    i, om, w = (math.radians(incl_deg), math.radians(node_deg), math.radians(argper_deg))
    ci, si, co, so, cw, sw = (math.cos(i), math.sin(i), math.cos(om), math.sin(om),
                              math.cos(w), math.sin(w))
    p_ecl = [co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si]
    q_ecl = [-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si]
    return _normal_vect(p_ecl), _normal_vect(q_ecl)


def _add_pq_data(seg, p, q, precision):
    """Port of showelem.cpp:add_pq_data -- append the P/Q columns to an angle field
    (`seg` starts at "Peri."/"Node "/"Incl.") and lop all three numbers to precision."""
    ndts = precision if precision > 13 else 13
    seg = seg + ("%+*.*f%+*.*f" % (ndts + 3, ndts, p, ndts + 3, ndts, q))
    chars = list(seg)
    _lop_digits(chars, 11, precision)
    _lop_digits(chars, 14 + ndts, precision)
    _lop_digits(chars, 17 + ndts * 2, precision)
    return "".join(chars)


def _hhmmss(jd):
    """HH:MM:SS clock (TT day fraction) for the perihelion-time annotation. Mirrors
    full_ctime at precision 0: a +0.05 s nudge then TRUNCATE to the whole second
    (find_orb's non-rounding default), not round-to-nearest."""
    frac = (jd + 0.5) - math.floor(jd + 0.5)
    sec = int(math.floor(frac * 86400.0 + 0.05))
    if sec == 86400:                       # full_ctime's i==units guard
        sec -= 1
    return "%2d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)


def to_eightline(fit, precision=5, perih_time=True):
    """find_orb's 8-line element block (showelem.cpp:elements_in_mpc_format), as a
    newline-joined string. Heliocentric J2000-ecliptic; default precision 5 matches
    find_orb's `fo`/elements.txt output. Lines: designation / Perihelion / Epoch /
    M+(2000.0)+P+Q / n+Peri.+P+Q / a+Node+P+Q / e+Incl.+P+Q / P+H+G+q+Q."""
    el = fit.elements()
    a, e, q = el["a"], el["e"], el["q"]
    incl, node, argper = el["i"], el["asc_node"], el["arg_per"]
    M = el["M"] % 360.0
    n_mot = el["n"]
    Tp, epoch = el["Tp"], fit.epoch_jd()
    H, _ = fit.absolute_magnitude()
    G = fit.slope_parameter()
    ndts = precision if precision > 10 else 10        # n_digits_to_show
    p_vec, q_vec = _pq_vectors(incl, node, argper)
    lines = []

    # 1. designation
    lines.append(fit.designation)

    # 2. Perihelion (heliocentric -> "Perihelion")
    if 2.0 < Tp < 3000000.0:
        py, pm, pdd = _jd_to_ymd(Tp)
        s = "   Perihelion %d %s %.*f TT" % (py, _MONTHS[pm - 1], precision, pdd)
        if perih_time:
            s += " = %s (JD %.*f)" % (_hhmmss(Tp), precision, Tp)
        lines.append(s)

    # 3. Epoch (with find_orb's trailing-zero trim on the JD and the matching day)
    ey, em, edd = _jd_to_ymd(epoch)
    s = "Epoch %4d %s %9.6f TT = JDT %.6f" % (ey, _MONTHS[em - 1], edd + 1e-7, epoch + 1e-7)
    i = 0
    while i < 5 and s[len(s) - i - 1] == "0":
        i += 1
    s = s[:len(s) - i]                                # lop trailing zeroes off the JD
    cut = s.index(" TT =") - i                        # ...and the same count off the day
    s = s[:cut] + s[cut + i:]
    lines.append(s)

    # 4. M  (2000.0)  P  Q
    chars = list("M%20.15f" % M)
    _lop_digits(chars, 9, precision)
    lines.append("".join(chars) + " (2000.0)            P               Q")

    # 5. n  +  Peri. + P/Q   (one line)
    chars = list("n%*.*f" % (ndts + 8, ndts + 3, n_mot))
    _lop_digits(chars, 9, precision)
    n_part = "".join(chars)
    peri_seg = _add_pq_data("Peri.%*.*f" % (ndts + 6, ndts, argper), p_vec[0], q_vec[0], precision)
    lines.append(n_part + peri_seg)

    # 6. a (or z = 1/a)  +  Node + P/Q
    # find_orb writes the semi-major axis only for a BOUND orbit; for e >= 1 (its
    # `is_cometary || ecc >= 1.` branch) `a` is negative or meaningless, so it prints the
    # RECIPROCAL semi-major axis on a 'z' line instead -- and blanks even that when 1/a
    # would be absurd (|a| below ~0.0011 au can't fit the field, beyond 100000 au is
    # numerically meaningless).
    if e >= 1.0:
        a_str = "" if (a > -0.0011 or a < -100000.0) else ("z%*.*f" % (ndts + 5, ndts, 1.0 / a))
    else:
        a_str = "a" + _show_formatted_dist(a, precision)
    a_part = (a_str + " " * (ndts + 9))[:ndts + 9]
    node_seg = _add_pq_data("Node %*.*f" % (ndts + 6, ndts, node), p_vec[1], q_vec[1], precision)
    lines.append(a_part + node_seg)

    # 7. e  +  Incl. + P/Q
    chars = list("e%*.*f" % (ndts + 8, ndts + 3, e))
    _lop_digits(chars, 8, precision)
    e_part = ("".join(chars) + " " * (ndts + 9))[:ndts + 9]
    incl_seg = _add_pq_data("Incl.%*.*f" % (ndts + 6, ndts, incl), p_vec[2], q_vec[2], precision)
    lines.append(e_part + incl_seg)

    # 8. P (period) [+ H/G] + q + Q -- ELLIPTIC ONLY.
    # find_orb guards this entire line with `!is_cometary && elem->ecc < 1.`: an orbit
    # with e >= 1 has no period and no apoapsis, so the line is OMITTED (the block is 7
    # lines, not 8) rather than filled with nonsense. Computing it unconditionally used
    # to raise -- sqrt(a) on a negative a for a hyperbolic solution, and a division by
    # zero at exactly e == 1 -- which crashed `--out-format elem`, the batch elem writer
    # and /api/export/elements on a reachable class of under-constrained short arcs.
    # `a > 0` is belt-and-braces on top of find_orb's test: the two agree for any
    # physical orbit, and a degenerate state can't reach a^1.5 through the pair.
    if e < 1.0 and a > 0:
        GAUSS_K = 0.01720209895
        SOLAR_GM = GAUSS_K * GAUSS_K
        t0 = a * math.sqrt(a * SOLAR_GM / SOLAR_GM)        # = a^1.5 (years; helio gm)
        t0_days = t0 * 365.25
        apo = q * (1.0 + e) / (1.0 - e)
        if t0 > 1e8 - 1.0:
            s = "P!!!!!!! "
        elif t0 > 9999.0:
            s = "P%7d " % int(t0)
        elif t0_days > 999.9:
            s = "P%10.5f       " % t0
        else:
            s = "P%8.3f/%7.3fd" % (t0, t0_days)
        if H is not None:
            s += " H%8.2f    G %6.2f" % (H, G)
        qd = _show_formatted_dist(q, precision)
        s += "   q " + qd.lstrip()
        s += "  Q " + _show_formatted_dist(apo, precision).lstrip()
        lines.append(s)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cartesian state vector + 6x6 covariance (JSON)
# ---------------------------------------------------------------------------
# The one product that carries the fit's *native* parameters rather than a
# derived element set: the six numbers the LM actually solves for, plus the
# covariance they were solved with. Every other egress format (MPCORB, SOF,
# 8-line) is classical elements, so anything that wants to re-integrate or
# re-propagate this orbit -- REBOUND/ASSIST, JPL Horizons vector input, an
# independent covariance propagation -- had to reconstruct the state from
# elements and could not get the covariance at all.
#
# Both frames are emitted because consumers disagree: FittedOrbit stores
# heliocentric EQUATORIAL J2000 (au, au/day) and integrates in it, while most
# element-facing tools think in ECLIPTIC. The ecliptic block is the same state
# and the same covariance under the single obliquity rotation R (applied as
# blockdiag(R, R) C blockdiag(R, R)^T), so the two blocks are one orbit, not two.
_STATE_KEYS = ["x", "y", "z", "dx", "dy", "dz"]


def _eq_to_ecl_6x6():
    """The 6x6 equatorial->ecliptic rotation: blockdiag(R, R) with R the rotation
    about the x axis by the J2000 mean obliquity. The angle is taken from
    orbital_elements.OBLIQ -- the one element_vector rotates with -- NOT the
    showelem P/Q constants above (they differ in the 5th decimal of a degree), so
    the ecliptic state here is exactly the frame the reported elements live in."""
    import numpy as np
    from pyroorb.orbital_elements import OBLIQ
    c, s = math.cos(OBLIQ), math.sin(OBLIQ)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    M = np.zeros((6, 6))
    M[:3, :3] = R
    M[3:, 3:] = R
    return M


def _state_block(state6, cov6):
    """One frame's worth of the payload: the six values, their 1-sigma diagonal,
    the covariance and the correlation matrix (rows/cols in _STATE_KEYS order)."""
    import numpy as np
    d = np.sqrt(np.clip(np.diag(cov6), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov6 / np.outer(d, d)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    return {
        "state": dict(zip(_STATE_KEYS, [float(v) for v in state6])),
        "sigma": dict(zip(_STATE_KEYS, [float(v) for v in d])),
        "covariance": [[float(v) for v in row] for row in cov6],
        "correlation": [[float(v) for v in row] for row in corr],
    }


def state_vector_document(fit):
    """The cartesian-state export as a plain dict (JSON-ready).

    Heliocentric position/velocity at the fit's epoch in au and au/day, with the
    6x6 state covariance, in BOTH the engine's native equatorial J2000 frame and
    the ecliptic J2000 frame. ``covariance``/``correlation`` are row-major with
    rows and columns ordered by ``covariance_order`` (x, y, z, dx, dy, dz).

    On a non-gravitational fit the covariance is the 6x6 block of the FULL
    inverse -- i.e. it already marginalises over the out-gassing parameters (see
    FittedOrbit._compute_state_cov) -- so it is the right covariance for the six
    even though the solved parameter vector is longer.
    """
    import numpy as np
    state_eq = np.asarray(fit.state(), dtype=float)[:6]
    cov_eq = np.asarray(fit.covariance_matrix("state"), dtype=float)[:6, :6]
    M = _eq_to_ecl_6x6()
    state_ecl = M @ state_eq
    cov_ecl = M @ cov_eq @ M.T

    epoch = fit.epoch_jd()
    ey, em, ed = _jd_to_ymd(epoch)
    doc = {
        "object": fit.designation,
        "designations": list(fit.designations),
        "packed": fit.packed_designation,
        "epoch": {
            "jd_tdb": float(epoch),
            "mjd_tdb": float(epoch - 2400000.5),
            "calendar": "%d %s %.6f TT" % (ey, _MONTHS[em - 1], ed),
        },
        "center": "Sun (heliocentric)",
        "units": {"position": "au", "velocity": "au/day"},
        "covariance_order": list(_STATE_KEYS),
        "equatorial_J2000": _state_block(state_eq, cov_eq),
        "ecliptic_J2000": _state_block(state_ecl, cov_ecl),
        "n_obs": int(fit.n_observations()),
        "rms_arcsec": float(fit.rms_residual()),
        "nongrav_model": fit.nongrav_model,
        "source": "PyRoOrb",
    }
    if fit.nongrav_model not in (None, "none"):
        doc["nongrav_note"] = ("covariance is the 6x6 state block of the full inverse, "
                              "marginalised over the non-gravitational parameters")
    return doc


def to_state_json(fit, indent=2):
    """`state_vector_document` rendered as a JSON string (the web UI's Export item
    and anything else that wants the text)."""
    import json
    return json.dumps(state_vector_document(fit), indent=indent)


# --- find_orb covar.json ---------------------------------------------------------
# find_orb writes a `covar.json` beside `covar.txt` at the end of every successful
# `full_improvement()` (orb_func.cpp, guarded by `!err_code && *covariance_filename`).
# It is a small file with exactly three keys, in this order:
#
#     { "covar": [[...], ...], "state_vect": [x, y, z, dx, dy, dz], "epoch": JD }
#
# The conventions, all read off orb_func.cpp / runge.cpp rather than guessed:
#   * `state_vect` is HELIOCENTRIC J2000 ECLIPTIC, au and au/day -- find_orb's internal
#     `orbit` vector (its `obs_posn` is heliocentric ecliptic, so `orbit` must be too).
#     Our engine's native state is EQUATORIAL, so it is rotated here.
#   * `covar` is `n_params` square, in the RAW parameter units (`_tweak_orbit` adds the
#     step straight onto the parameter), i.e. au^2 / au^2 per day / au^2 per day^2 for
#     the state block. It is the inverse normal matrix scaled by the residual variance
#     (Danby p.243 (7.5.21)); our `s^2 (J^T J)^-1` is the same quantity. It is NOT the
#     same NUMBER: each program scales by its own residual variance under its own
#     weights, and find_orb's `sigma_squared` sums `w^2 (resid^2 + 1)` -- a floor that
#     keeps a well-fitting orbit from claiming a vanishing uncertainty. We publish OUR
#     covariance, the one behind every sigma the engine reports elsewhere; adopting
#     find_orb's inflation here would make this file disagree with the element sigmas
#     shown beside it. Measured on the 2001 KQ76 fixture against a real `fo` run
#     re-anchored to the same epoch: the state agrees to 2.4e-5 relative, the sigmas
#     to ~20% (find_orb's larger, near-uniformly across all six).
#   * `state_vect` is ALWAYS six long, even when `covar` is 7-10 square on a nongrav
#     model -- find_orb prints `for (i = 0; i < 6; i++)` regardless. We keep that.
#   * the extra parameters occupy slots 6.. in the model's own order: A1, A2, A3, then
#     the A1A2A3DT time lag. That is find_orb's slot layout for every model it shares
#     with us (comet2/comet3 -> COMET_TWO/THREE_PARAM, srp2/srp3 -> SRP_TWO/THREE_PARAM,
#     yarkovsky -> YARKO_A2 with A2 in slot 6), and A1..A3 are in au/day^2 on both
#     sides (runge.cpp: "The resulting accelerations should be in AU/day^2, which
#     fortunately matches the units used in this program").
#   * the ONE unit exception is the single-parameter SRP model (FORCE_MODEL_SRP): there
#     find_orb's slot-6 parameter is a DIMENSIONLESS fraction of the solar GM -- runge.cpp
#     subtracts it from the solar term before the -SOLAR_GM/r^3 factor -- so our physical
#     A1 (au/day^2) is divided by SOLAR_GM, row and column, to land in find_orb's units.
#     Getting this wrong would misreport that variance by a factor of ~1e4 squared.
# `epoch` is the fit epoch as a bare JD. find_orb's is TT; ours is TDB, which differs by
# under 2 ms -- far below the "%f" six decimals (~0.09 s) the file is written with.
_FO_GAUSS_K = 0.01720209895
_FO_SOLAR_GM = _FO_GAUSS_K * _FO_GAUSS_K       # 2.959122082855911e-4 au^3/day^2 (constant.h)


def _covar_findorb_units(cov, names, model):
    """`cov` with the nongrav rows/columns converted into find_orb's parameter units.

    A no-op for every model but the one-parameter SRP, whose slot-6 parameter find_orb
    carries as a fraction of the solar GM rather than as an acceleration (see above)."""
    import numpy as np
    if model != "srp" or len(names) < 7:
        return cov
    f = np.ones(len(names))
    f[6] = 1.0 / _FO_SOLAR_GM
    return cov * np.outer(f, f)


def covariance_document(fit):
    """find_orb's `covar.json` as a plain dict: ``covar``, ``state_vect``, ``epoch``.

    The full solved-parameter covariance and the state vector, both rotated into the
    heliocentric J2000 ECLIPTIC frame find_orb works in, in find_orb's parameter units
    and slot order. See the module comment above for every convention this obeys."""
    import numpy as np
    cov, names = fit.parameter_covariance()
    cov = np.asarray(cov, float)
    n = cov.shape[0]
    # blockdiag(R, R, I_k): A1/A2/A3 are radial/transverse/normal scalars in a co-moving
    # frame, so the obliquity rotation touches the six state rows and nothing else.
    M = np.eye(n)
    M[:6, :6] = _eq_to_ecl_6x6()
    cov_ecl = M @ cov @ M.T
    cov_ecl = 0.5 * (cov_ecl + cov_ecl.T)          # the rotation is symmetric only to round-off
    cov_ecl = _covar_findorb_units(cov_ecl, names, fit.nongrav_model)
    state_ecl = _eq_to_ecl_6x6() @ np.asarray(fit.state(), float)[:6]
    return {"covar": [[float(v) for v in row] for row in cov_ecl],
            "state_vect": [float(v) for v in state_ecl],
            "epoch": float(fit.epoch_jd())}


def to_covar_json(fit):
    """`covariance_document` rendered in find_orb's own byte layout.

    Not `json.dumps`: find_orb hand-writes this file with a specific shape (a leading
    "{ ", "%.12g" matrix entries with a trailing space per row, "%.18g" state components
    one per line, "%f" epoch), and a diff against a real find_orb run is the cheapest way
    to check an export is compatible. The result is ordinary JSON either way."""
    doc = covariance_document(fit)
    rows = doc["covar"]
    n = len(rows)
    out = ['{ "covar": [\n']
    for i, row in enumerate(rows):
        cells = "".join("%.12g%c " % (v, "," if j < n - 1 else " ")
                        for j, v in enumerate(row))
        out.append("   [%s]%s\n" % (cells, "," if i < n - 1 else ""))
    out.append('], "state_vect": [\n')
    for i, v in enumerate(doc["state_vect"]):
        out.append("    %.18g%s\n" % (v, " ]," if i == 5 else ","))
    out.append('  "epoch": %f\n}' % doc["epoch"])
    return "".join(out)
