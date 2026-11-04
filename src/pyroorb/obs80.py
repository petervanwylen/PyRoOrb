"""MPC 80-column astrometry format ("OBS80") reader.

The historical fixed-width MPC observation format. Nothing here but column
bookkeeping — read it once and skip it thereafter.
"""
import re
import time
from pathlib import Path
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

KM_PER_AU = 1.495978707e8                 # IAU 2012 astronomical unit (km)

# Astrometry caches are short-term burst protection, not a permanent store. Their job is
# to make a crowd interested in the same object cost the MPC's observations API one
# request, not one per visitor; and the MPC keeps publishing new observations, so a cache
# that never expired would pin a fit to a stale arc. A PUBLISHED designation's arc grows
# over days, so its cached observations are reused for three hours. A NEOCP / temporary
# id's arc can double within the hour, so its cache lives two minutes. Every entry is
# deleted once expired, and the whole cache is held under datadirs.CACHE_MAX_MB.
ASTROMETRY_CACHE_TTL_S = 3 * 3600.0       # published designations: 3 hours
NEOCP_CACHE_TTL_S = 120.0                 # NEOCP / temporary ids: 2 minutes

# Tolerance for a cache mtime that reads slightly AHEAD of ``time.time()``. A few
# seconds of future skew is benign (timestamp rounding, a filesystem clock a hair
# ahead of the process clock) and should still count as fresh; anything beyond it
# means the two clocks genuinely disagree and the mtime can't bound the age.
CACHE_CLOCK_SKEW_GRACE_S = 2.0


def cache_fresh(cache, ttl_s=ASTROMETRY_CACHE_TTL_S):
    """True iff ``cache`` exists and was written within ``ttl_s`` seconds.

    Used to gate the on-disk astrometry caches so a fit re-fetches an object's
    observations from the MPC once the cached copy goes stale.

    Robust to clock skew between the process wall clock and the filesystem's
    mtime clock. In a Docker-on-Mac sandbox these can differ by *days*: the
    container's ``time.time()`` lags the host-backed filesystem, so a file just
    written lands with an mtime well in the FUTURE relative to ``time.time()``.
    A naive ``time.time() - mtime <= ttl_s`` then sees a large negative age and
    reports every such cache as eternally fresh -- the cache never expires and a
    fit stays pinned to a stale arc no matter how much new astrometry the MPC
    publishes. So a cache whose mtime sits meaningfully in the future is treated
    as UNtrustworthy (stale) and re-fetched, rather than trusted forever."""
    if not cache:
        return False
    p = Path(cache)
    if not p.exists():
        return False
    age = time.time() - p.stat().st_mtime
    if age < -CACHE_CLOCK_SKEW_GRACE_S:   # mtime is in the future -> clocks disagree
        return False
    return age <= ttl_s


def astrometry_ttl(desig):
    """How long cached observations of `desig` stay fresh, seconds (see above)."""
    from pyroorb.obs_sources import is_published_designation
    return ASTROMETRY_CACHE_TTL_S if is_published_designation(desig) else NEOCP_CACHE_TTL_S


def cache_stem(desig):
    """Filesystem-safe stem for a designation used in cache filenames.

    A comet designation like ``C/2025 R2`` carries a ``/`` that, left in a
    filename, is read as a path separator -- the fetch then tries to write
    ``obs80_C/2025_R2.json`` into a non-existent ``obs80_C/`` directory and dies
    with ENOENT. Collapse spaces AND any path-hostile character to underscores so
    every designation maps to one flat file. Asteroid designations contain only
    spaces, so their existing cache names are unchanged (no cache invalidation)."""
    s = str(desig).strip()
    for ch in " /\\:":
        s = s.replace(ch, "_")
    return s


# Comet / interstellar type prefixes (MPC): C=long-period, P=periodic, D=defunct,
# X=uncertain orbit, A=reclassified-asteroidal, I=interstellar.
_COMET_TYPES = "CPDXAI"


def pack_designation(desig):
    """Return the MPC packed designation for ``desig``, handling COMETS (which the
    minor-planet packer in ``packing.pack`` returns ``None`` for -- it only knows
    numbers and asteroid provisional designations).

    Three comet shapes are recognised, peeled, and packed the MPC way:
      * a numbered periodic comet -- ``1P``, ``73P`` (optional fragment
        ``73P-C``) -> zero-padded number + type letter, e.g. ``0073P``;
      * a true comet-form provisional -- ``C/2023 A3`` (a half-month letter
        immediately followed by a number) -> type letter + ``packing.pack_comet``
        core, e.g. ``CK23A030``;
      * an asteroid-style provisional later flagged cometary -- ``P/2023 JN16``
        (two letters) -> the bare 7-char asteroid packing ``K23J16N`` (via
        ``packing.pack``), which is exactly what the MPC stores as that obs
        record's trkSub; the comet type is carried separately, so it is NOT
        prefixed here.
    Anything still unpackable is returned unchanged rather than raising, so a fit
    never dies on a purely cosmetic identifier.
    """
    from pyroorb import packing
    s = str(desig).strip()
    native = packing.pack(s)             # plain number / asteroid provisional / survey
    if native is not None:
        return native
    # Numbered periodic comet, optional fragment letter(s): 1P, 73P, 73P-C, 1I.
    m = re.fullmatch(r"(\d+)([%s])(?:-([A-Za-z]+))?" % _COMET_TYPES, s)
    if m:
        num, typ, frag = m.groups()
        return "%04d%s" % (int(num), typ) + (frag.lower() if frag else "")
    # Comet-type prefix on a provisional designation: P/2023 JN16, C/2023 A3.
    m = re.fullmatch(r"([%s])/(.+)" % _COMET_TYPES, s)
    if m:
        typ, core = m.groups()
        core = core.split("-")[0].strip()             # drop a fragment suffix
        # A comet-FORM core (one half-month letter then a digit) packs via pack_comet
        # and KEEPS the type letter; an asteroid-FORM core (two letters) packs bare.
        toks = core.split()
        comet_form = (len(toks) == 2 and len(toks[1]) >= 2
                      and toks[1][0].isalpha() and toks[1][1].isdigit())
        packed = packing.pack_comet(core) if comet_form else packing.pack(core)
        if packed is None:
            return s
        return (typ + packed) if comet_form else packed
    # Natural-satellite provisional: S/2021 J 7 -> SK21J070. The MPC packs it exactly like
    # a comet-form provisional, 'S' in the type-letter column 5 and the planet letter in the
    # half-month slot (find_orb's xdesig.txt: 'SK16J010 J054S  Jupiter LIV = S/2016 J 1').
    # Left unpacked, the 10-char readable form spilled across cols 1-12 and read back as a
    # number 'S/2' in cols 1-5 beside a trksub '021 J 7'.
    m = re.fullmatch(r"S/(\d{4}) ([JSUN]) ?(\d+)", s, re.IGNORECASE)
    if m:
        packed = packing.pack_comet("%s %s%s" % (m.group(1), m.group(2).upper(), m.group(3)))
        if packed is not None:
            return "S" + packed
    return s


def _placed(fields):
    """Assemble an 80-char line from (1-indexed start column, text) pairs."""
    buf = [" "] * 80
    for start, text in fields:
        for k, ch in enumerate(text):
            col = start - 1 + k
            if 0 <= col < 80:
                buf[col] = ch
    return "".join(buf)


def ra_hms(ra_deg, decimals=3):
    """RA (deg) -> 'HH MM SS.ddd' (also the OBS80 RA field, cols 33-44, 12 chars).
    Canonical sexagesimal RA formatter, shared with ephemeris display.

    ``decimals`` is how many places of seconds-of-TIME to print (0-3), always padded
    out to the full 12-column field. The MPC records an observation at the precision
    its submitter reported (ADES ``precRA``), so a 1970s photographic position stays
    'HH MM SS.dd ' rather than gaining two invented digits."""
    h = ra_deg / 15.0
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = (h - hh - mm / 60.0) * 3600.0
    if ss >= 60.0 - 0.5 * 10 ** -decimals:      # rounding would print 60 -> carry
        ss = 0.0
        mm += 1
        if mm == 60:
            mm = 0
            hh = (hh + 1) % 24
    w = 3 + decimals if decimals else 2
    return f"{hh:02d} {mm:02d} {ss:0{w}.{decimals}f}".ljust(12)


def dec_dms(dec_deg, decimals=2):
    """Dec (deg) -> 'sDD MM SS.dd' (also the OBS80 Dec field, cols 45-56, 12 chars).
    Canonical sexagesimal Dec formatter, shared with ephemeris display. ``decimals``
    is the reported precision in seconds of arc (ADES ``precDec``); see ra_hms."""
    sgn = "-" if dec_deg < 0 else "+"
    a = abs(dec_deg)
    dd = int(a)
    mm = int((a - dd) * 60)
    ss = (a - dd - mm / 60.0) * 3600.0
    if ss >= 60.0 - 0.5 * 10 ** -decimals:
        ss = 0.0
        mm += 1
        if mm == 60:
            mm = 0
            dd += 1
    w = 3 + decimals if decimals else 2
    return f"{sgn}{dd:02d} {mm:02d} {ss:0{w}.{decimals}f}".ljust(12)


def _mag_to_obs80(mag):
    """Magnitude (cols 66-70, 5 chars) the MPC way: the decimal point sits at col 68 (the
    middle of the field) -- integer part right-justified in cols 66-67, the fraction
    left-justified in cols 69-70 and space-padded. One decimal ('20.6 ') unless the value
    carries two ('20.61'); band goes in col 71 separately."""
    m = float(mag)
    s = f"{m:.1f}" if abs(round(m, 2) - round(m, 1)) < 1e-9 else f"{m:.2f}"
    ipart, frac = s.split(".")
    return f"{ipart:>2}.{frac}".ljust(5)[:5]


def _date_to_obs80(jd_utc, decimals=6):
    """UTC JD -> 'YYYY MM DD.dddddd' (cols 16-32, 17 chars), the day carried to
    ``decimals`` places (ADES ``precTime``; 6 = microday, the modern default)."""
    iso = Time(float(jd_utc), format="jd", scale="utc").iso     # 'YYYY-MM-DD hh:mm:ss.sss'
    y, mo, d = int(iso[0:4]), int(iso[5:7]), int(iso[8:10])
    frac = (float(jd_utc) + 0.5) % 1.0                          # fractional day from local midnight
    w = 3 + decimals if decimals else 2
    return f"{y:04d} {mo:02d} {d + frac:0{w}.{decimals}f}".ljust(17)


# --------------------------------------------------------------------------
# ADES -> OBS80 field mapping
#
# ADES splits apart what the 80-column record packs into single bytes: the
# magnitude band (col 71), the star catalogue (col 72), the programme code /
# note (col 14), the observation mode (col 15) and the publication reference
# (cols 73-77). The mappings below follow the reference converter, Bill Gray's
# `lunar/ades2mpc.cpp` (process_ades_tag) and its catalogue table in
# `lunar/mpc_fmt.cpp` (net_codes / net_name_to_byte_code).
# --------------------------------------------------------------------------

# MPC star-catalogue codes for col 72 -- (code, ADES/NET catalogue name). The
# MPC's own list (CatalogueCodes.html; G. V. Williams 2012), as carried by
# lunar's net_codes. 'Gaia-3E' is a second spelling of Gaia-EDR3: the MPC obs
# API says "Gaia3E", which does not reduce to the same string as "Gaia-EDR3".
NET_CODES = [
    ("a", "USNO-A1"), ("b", "USNO-SA1"), ("c", "USNO-A2"), ("d", "USNO-SA2"),
    ("e", "UCAC-1"), ("f", "Tycho-1"), ("g", "Tycho-2"), ("h", "GSC-1.0"),
    ("i", "GSC-1.1"), ("j", "GSC-1.2"), ("k", "GSC-2.2"), ("l", "ACT"),
    ("m", "GSC-ACT"), ("n", "SDSS-DR8"), ("o", "USNO-B1"), ("p", "PPM"),
    ("q", "UCAC-4"), ("r", "UCAC-2"), ("s", "USNO-B2"), ("t", "PPMXL"),
    ("u", "UCAC-3"), ("v", "NOMAD"), ("w", "CMC-14"), ("x", "HIP-2"),
    ("y", "HIP-1"), ("z", "GSC-?"),
    ("A", "AC"), ("B", "SAO 1984"), ("C", "SAO"), ("D", "AGK 3"), ("E", "FK4"),
    ("F", "ACRS"), ("G", "Lick Gaspra Catalogue"), ("H", "Ida93 Catalogue"),
    ("I", "Perth 70"), ("J", "COSMOS/UKST Southern Sky Catalogue"), ("K", "Yale"),
    ("L", "2MASS"), ("M", "GSC-2.3"), ("N", "SDSS-DR7"), ("O", "SST-RC1"),
    ("P", "MPOSC3"), ("Q", "CMC-15"), ("R", "SST-RC4"), ("S", "URAT-1"),
    ("T", "URAT-2"), ("U", "Gaia-DR1"), ("V", "Gaia-DR2"), ("W", "Gaia-DR3"),
    ("X", "Gaia-EDR3"), ("X", "Gaia-3E"), ("Y", "UCAC-5"), ("Z", "ATLAS-2"),
    ("0", "IHW"), ("1", "PS1_DR1"), ("2", "PS1_DR2"), ("3", "Gaia_Int"),
    ("4", "GZ"), ("5", "UBSC"), ("6", "Gaia_2016"), ("7", "ZZCAT"),
    ("8", "APASS"), ("9", "UBSC"), ("!", "AKARI"), ("@", "AG"), ("#", "WISE"),
    ("$", "LSST2502"), ("^", "IRASPSC"),
]


def _reduce_net_name(s):
    """Canonicalise a catalogue name for comparison: people spell one catalogue a
    dozen ways ('Gaia DR1', 'Gaia-DR1', 'Gaiadr1', 'Gaia1'), so drop spaces, '-',
    a trailing '.0' and the 'DR' of a Gaia release, and upper-case the rest."""
    out, i = [], 0
    s = str(s)
    while i < len(s):
        if s[i] in "- ":
            i += 1
        elif s[i:i + 2] == ".0":
            i += 2
        elif s[i:i + 2] == "DR":
            i += 2
        else:
            out.append(s[i].upper())
            i += 1
    return "".join(out)


# ADES astCat vocabulary spellings that don't reduce to their NET name -- the ADES
# list abbreviates where the MPC's NET list spells out (ADESFieldValues.html).
ADES_CAT_ALIASES = {"Tyc1": "f", "Tyc2": "g", "GSC": "z", "COSMOS": "J",
                    "LickGaspra": "G", "Ida93": "H"}

_NET_REDUCED = {}
for _code, _name in NET_CODES:
    _NET_REDUCED[_reduce_net_name(_name)] = _code       # a repeated name takes the LAST code
for _name, _code in ADES_CAT_ALIASES.items():
    _NET_REDUCED.setdefault(_reduce_net_name(_name), _code)


def net_name_to_byte_code(name):
    """ADES ``astCat`` name -> the single-byte MPC catalogue code for col 72.
    A name we don't recognise becomes '?' (find_orb's marker for "a catalogue was
    stated, but not one we know"); an absent astCat -- or the explicit ADES 'UNK',
    which is a statement that the submitter didn't say -- leaves the column blank."""
    s = str(name or "").strip()
    if not s or s.upper() == "UNK":
        return " "
    return _NET_REDUCED.get(_reduce_net_name(s), "?")


def band_to_obs80(band):
    """ADES ``band`` -> the col-71 magnitude band byte.

    Some observers report a two-character passband naming the photometric system as
    well as the filter -- Pan-STARRS/Sloan 'Pw'/'Sr', ATLAS 'Ac', Gaia 'Gb', LSST
    'Lr', Johnson-Cousins 'Rc'/'Vj'. The 80-column record has room only for the
    filter, and which of the two characters that is depends on the system, so the
    collapse is the one photometry already does for the band->V correction."""
    b = str(band or "").strip()
    if not b or b.upper() == "UNK":
        return " "
    from pyroorb import photometry
    return photometry.ades_band_to_mpc(b)[:1]


# ADES `mode` -> the col-15 note2 byte (ADESFieldValues.html). 'UNK' is the
# submitter declining to say, which the 80-column record spells as a blank -- not
# as find_orb's '?', which it uses for its own round-trip.
MODE_TO_NOTE2 = {"CCD": "C", "CMO": "B", "VID": "n", "PHO": "P", "ENC": "e",
                 "PMT": "p", "MIC": "M", "MER": "T", "TDI": "C", "OCC": "E",
                 "UNK": " "}

# Programme codes for col 14, indexed by the ADES `prog` value (0-34).
_PROGRAM_CODES = "0123456789!\"#$%&'()*+,-./[\\]^_`{|}~"


def prog_to_note1(prog):
    """ADES ``prog`` (a two-character mutant-hex programme code, e.g. '00', '0f',
    '12') -> the col-14 note1 byte, or None when absent/unrecognised.

    Codes 0-34 are tabulated (find_orb's ades2mpc.cpp keeps the same table); from 35
    up the MPC's published lines simply walk printable ASCII offset by 23 -- 35 -> ':',
    41 -> '@', 52 -> 'K', 64 -> 'W' -- where find_orb instead falls back to mutant hex."""
    s = str(prog or "").strip()
    if len(s) != 2:
        return None
    from pyroorb import packing
    try:
        idx = packing._dec(s)
    except (KeyError, ValueError):
        return None
    if 0 <= idx < len(_PROGRAM_CODES):
        return _PROGRAM_CODES[idx]
    ch = idx + 23
    return chr(ch) if 33 <= ch <= 126 else None


def pack_mpc_reference(ref):
    """ADES ``ref`` -> the 5-char packed publication reference for cols 73-77
    ('MPEC 2021-D85' -> 'ED085', 'MPS  6197' -> 'a6197'), or None when the
    reference isn't one of the packable forms."""
    from pyroorb import packing
    s = str(ref or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"MPEC\s+\d{4}-([A-Za-z])(\d{1,3})", s)
    if m:
        half, num = m.groups()
        return "E" + half + num.rjust(3, "0")
    m = re.fullmatch(r"(MPS|MPC)\s+(\d+)", s)
    if m:
        kind, num = m.group(1), int(m.group(2))
        if kind == "MPS":
            if num < 260000:
                return chr(ord("a") + num // 10000) + "%04d" % (num % 10000)
            return "~" + packing._enc(num - 260000, 4)
        if num < 10000:
            return "M%04d" % num          # the MPC's own lines, not find_orb's '%05d'
        if num < 100000:
            return "%05d" % num
        if num < 110000:
            return "@%04d" % (num % 100000)
        return "#" + packing._enc(num - 110000, 4)
    if s[0] == "!" or s.startswith("JPL"):
        return s[:5].ljust(5)
    return None


def _prec_decimals(prec, default):
    """ADES precRA/precDec ('0.001'/'0.01'/'0.1'/'1') -> decimal places on the
    seconds field. An absent or unrecognised value keeps ``default``."""
    return {"0.001": 3, "0.01": 2, "0.1": 1, "1": 0}.get(str(prec or "").strip(), default)


def _prec_time_decimals(prec, default=6):
    """ADES precTime (microdays: 1, 10, 100 ...) -> decimal places on the day."""
    try:
        p = int(float(prec))
    except (TypeError, ValueError):
        return default
    return {1: 6, 10: 5, 100: 4, 1000: 3, 10000: 2, 100000: 1}.get(p, default)


def _designation_fields(rec):
    """(col, text) placements for the packed designation: a 5-char permanent number in
    cols 1-5, a 7-char provisional in cols 6-12, anything else left-justified in 1-12.

    The two fields are independent, so a numbered object observed under a provisional
    designation fills BOTH ('B1520K01Y98M') -- the permanent number comes from the
    ADES permID (``packed_perm``) when the record carries one."""
    out = []
    perm = str(rec.get("packed_perm") or "").strip()
    if perm and len(perm) <= 5:
        out.append((1, perm.rjust(5)))          # permanent number (cols 1-5)
    pk = str(rec.get("packed") or "").strip()
    if not pk or pk == perm:
        return out
    if len(pk) <= 5 and pk[0].isdigit():
        return out or [(1, pk.rjust(5))]        # permanent number
    if len(pk) == 7:
        return out + [(6, pk)]                   # provisional (asteroid form)
    # Anything else -- a comet packing, a survey/NEOCP temporary id -- goes in the
    # provisional field, left-justified from col 6 while it fits and otherwise
    # backed up so it ENDS at col 12 (find_orb's ades2mpc rule for a trkSub).
    pk = pk[:12]
    return out or [(6 if len(pk) < 8 else 13 - len(pk), pk)]


def _sat_offset_field(val, is_km):
    """One geocentric coordinate as an 11-char signed OBS80 satellite-offset field: a '+'/'-'
    sign then the magnitude right-justified in 10 columns, carrying as many decimals as fit.
    km values default to 4 d.p. (decimal lands at col 41 for X, e.g. '+  520.1210'); au values
    to 8 d.p. (decimal at col 37). For a large magnitude the decimal shifts right as digits are
    dropped -- exactly the MPC's "shift right by one column" rule, generalised so a distant L2
    spacecraft (au) or a far geocentric range still fits the field instead of overflowing."""
    sign = "-" if val < 0 else "+"
    mag = abs(float(val))
    for dec in range(4 if is_km else 8, -1, -1):
        s = f"{mag:.{dec}f}"
        if len(s) <= 10:
            break
    return sign + s.rjust(10)


def _sat_second_line(rec):
    """The 's' companion line for a SPACE-BASED obs from its geocentric equatorial vector
    (``sat_pos_au``), or None. Col 15 's', col 33 the units flag ('1' km / '2' au, taken from
    the ADES-native ``sat_units``); cols 35-45/47-57/59-69 the signed X/Y/Z, obscode 78-80 --
    the standard MPC two-line satellite record (SatelliteObs.html)."""
    xyz = rec.get("sat_pos_au")
    if not xyz:
        return None
    is_km = (rec.get("sat_units") or "km").lower() != "au"
    vals = [c * KM_PER_AU for c in xyz] if is_km else list(xyz)
    fields = list(_designation_fields(rec))
    fields.append((15, "s"))
    if rec.get("jd_utc") is not None:
        fields.append((16, _date_to_obs80(rec["jd_utc"])))    # date must match the first line
    fields.append((33, "1" if is_km else "2"))                # parallax units flag
    for k, v in enumerate(vals[:3]):
        fields.append((35 + k * 12, _sat_offset_field(v, is_km)))   # X 35-45, Y 47-57, Z 59-69
    code = str(rec.get("obscode") or "").strip()[:3]
    fields.append((78, code.rjust(3)))
    return _placed(fields)


def _rov_second_line(rec):
    """The 'v' companion line for a ROVING observer from its geodetic site (``site_geodetic`` =
    (E-longitude deg, latitude deg, altitude m)), or None. Col 15 'v', cols 35-44 E-longitude,
    46-55 latitude (N +ve / S -ve), 57-61 altitude (integer metres), obscode 78-80 (247 or an
    MPC mutant-hex rover variant) -- per the MPC roving-observer record (RovingObs.html)."""
    lla = rec.get("site_geodetic")
    if not lla:
        return None
    lon, lat, alt = float(lla[0]), float(lla[1]), float(lla[2])
    fields = list(_designation_fields(rec))
    fields.append((15, "v"))
    if rec.get("jd_utc") is not None:
        fields.append((16, _date_to_obs80(rec["jd_utc"])))    # date must match the first line
    fields.append((35, f"{lon:10.4f}"))                       # cols 35-44 E-longitude (deg)
    fields.append((46, f"{lat:10.4f}"))                       # cols 46-55 latitude (N+, S-)
    fields.append((57, f"{int(round(alt)):5d}"))              # cols 57-61 altitude (m, integer)
    code = str(rec.get("obscode") or "").strip()[:3]
    fields.append((78, code.rjust(3)))
    return _placed(fields)


def _second_line(rec):
    """Reconstruct the 's'/'v' companion line for a satellite / roving obs, or None. Prefers the
    verbatim original (``obs80_line2``, kept when the obs came from an OBS80 file) over a
    synthesized one so an OBS80 round-trip is byte-exact."""
    orig = rec.get("obs80_line2")
    if orig:
        return orig
    if rec.get("sat_pos_au"):
        return _sat_second_line(rec)
    if rec.get("site_geodetic"):
        return _rov_second_line(rec)
    return None


def format_obs80(rec):
    """Synthesize a best-effort MPC 80-column ("OBS80") line from a parsed obs record (the fields
    the engine keeps: packed designation, jd_utc, ra_deg, dec_deg, mag, band, obscode). Used to
    render/copy astrometry when the obs came from the ADES API (which carries no original 80-col
    line). Column 65 is left blank -- callers mark an excluded obs find_orb-style by setting it to
    'x'. A SPACE-BASED obs (``sat_pos_au``) or ROVING observer (``site_geodetic``) is emitted as
    the MPC two-line record: the first line carries 'S'/'V' in col 15 with the RA/Dec, and a
    second line ('s'/'v') follows with the geocentric offset / geodetic site -- returned joined by
    a newline. Ground optical stays a single line."""
    sat = bool(rec.get("sat_pos_au"))
    rov = bool(rec.get("site_geodetic")) and not sat
    fields = list(_designation_fields(rec))
    mode = MODE_TO_NOTE2.get(str(rec.get("mode") or "").strip().upper())
    note2 = "S" if sat else ("V" if rov else (rec.get("note2") or mode or "C")[:1])
    if rec.get("discovery"):
        fields.append((13, "*"))              # discovery asterisk (ADES disc="*")
    # col 14 note1: the programme code (`prog`) the survey observed under, else the
    # ADES `notes` letter. The MPC prefers the programme code when a record has both.
    note1 = prog_to_note1(rec.get("prog")) or str(rec.get("notes") or "").strip()[:1]
    if note1:
        fields.append((14, note1))
    fields.append((15, note2))                                  # note2: CCD by default; S/V two-line
    if rec.get("jd_utc") is not None:
        fields.append((16, _date_to_obs80(rec["jd_utc"],
                                          _prec_time_decimals(rec.get("prec_time")))))
    if rec.get("ra_deg") is not None:
        fields.append((33, ra_hms(float(rec["ra_deg"]), _prec_decimals(rec.get("prec_ra"), 3))))
    if rec.get("dec_deg") is not None:
        fields.append((45, dec_dms(float(rec["dec_deg"]), _prec_decimals(rec.get("prec_dec"), 2))))
    mag = rec.get("mag")
    if mag is not None:
        try:
            fields.append((66, _mag_to_obs80(mag)))          # cols 66-70, decimal at col 68
            # A COMET's magnitude column says which magnitude it is -- 'N' nuclear,
            # 'T' total (ADES nucMag, a flag) -- and the filter band gives way to it.
            # Everything else goes through band_to_obs80, i.e. the MPC's own
            # ADES->MPC80 collapse. Truncating to band[:1] instead is not merely
            # lossy but WRONG (see band_to_obs80): it cost 2025 NX363 a spurious
            # "possible different body" caution on its own discovery night, its
            # three Rubin 'Lg' observations having rendered as band 'L'.
            nuc = rec.get("nucmag")
            fields.append((71, ("N" if float(nuc) else "T") if nuc is not None
                           else band_to_obs80(rec.get("band"))))
        except (TypeError, ValueError):
            pass
    fields.append((72, net_name_to_byte_code(rec.get("astcat"))))   # star catalogue
    if rec.get("neocp"):
        fields.append((73, "NEOCP"))          # reference field (cols 73-77): NEOCP-sourced astrometry
    else:
        packed_ref = pack_mpc_reference(rec.get("ref"))
        if packed_ref:
            fields.append((73, packed_ref))   # publication reference (MPEC/MPS/MPC)
    code = str(rec.get("obscode") or "").strip()[:3]
    fields.append((78, code.rjust(3)))
    line1 = _placed(fields)
    line2 = _second_line(rec) if (sat or rov) else None
    return f"{line1}\n{line2}" if line2 else line1


def obs80_base_line(rec):
    """The canonical 80-col record for an obs, col-65 marker CLEARED so the caller owns the
    include/exclude 'x'. Prefers the original line (OBS80 source) over a synthesized one. A
    space-based / roving obs is TWO lines (the RA/Dec line + its 's'/'v' companion) joined by a
    newline; the col-65 marker only ever applies to the FIRST line."""
    line = rec.get("obs80_line")
    line2 = None
    if not line:
        parts = format_obs80(rec).split("\n", 1)               # ADES/synth: may be two lines
        line = parts[0]
        line2 = parts[1] if len(parts) > 1 else None
    else:
        line2 = _second_line(rec)                              # OBS80 companion (verbatim if kept)
    line = (line.rstrip("\n") + " " * 80)[:80]
    if len(line) > 64 and line[64] == "x":         # drop any pre-existing exclusion mark
        line = line[:64] + " " + line[65:]
    if line2:
        return line + "\n" + (line2.rstrip("\n") + " " * 80)[:80]
    return line


def geodetic_to_itrs_au(lon_deg, lat_deg, alt_m):
    """Geocentric Earth-FIXED ITRS position (au) from geodetic East-longitude/latitude (deg) and
    altitude (m), via astropy EarthLocation (WGS84). Shared by the two roving-observer paths
    (the OBS80 'v' companion line and the ADES sys=WGS84 record)."""
    el = EarthLocation.from_geodetic(lon=lon_deg * u.deg, lat=lat_deg * u.deg, height=alt_m * u.m)
    return [el.x.to_value(u.au), el.y.to_value(u.au), el.z.to_value(u.au)]


def mpc80_roving_itrs_au(line):
    """Geocentric Earth-FIXED (ITRS) observer position (au) from an OBS80 roving-observer 'v'
    companion line (col 15 == 'v', obscode 247): cols 35-44 East longitude (deg), 46-55 latitude
    (deg, N+), 57-61 altitude (m) -- column slices per Sonia Keys' mpcutilities.obs80.parseRov.
    Unlike a satellite's inertial geocentric vector, this is fixed to the rotating Earth, so the
    caller rotates it ITRS->GCRS at the observation time exactly like a fixed station. Returns
    [x, y, z] in au, or None if it isn't a parseable 'v' line."""
    if len(line) < 61 or line[14] != "v":
        return None
    try:
        lon, lat, alt = float(line[34:44]), float(line[45:55]), float(line[56:61])
    except ValueError:
        return None
    return geodetic_to_itrs_au(lon, lat, alt)


def mpc80_satellite_offset(line):
    """Geocentric observer position (equatorial ICRF au) from an OBS80 satellite 's' companion
    line -- the second line of a two-line space-based record (col 15 == 's'). Col 33 is the units
    flag (1 = km, 2 = au); cols 34-69 hold the signed geocentric X/Y/Z (sign then value, e.g.
    '+ 4303.6772'). Returns [x, y, z] in au, or None if it is not a parseable 's' line. The fit's
    observer_positions_batch forms the observer as earth_center + this offset (a spacecraft has no
    obscode ground position). e.g. C51/WISE for 2010 OA81."""
    if len(line) < 70 or line[14] != "s":
        return None
    nums = re.findall(r"[+-]\s*\d+\.?\d*", line[33:72])
    if len(nums) < 3:
        return None
    try:
        xyz = [float(n.replace(" ", "")) for n in nums[:3]]
    except ValueError:
        return None
    scale = 1.0 if line[32] == "2" else (1.0 / KM_PER_AU)   # '1' km -> au; '2' already au
    return [c * scale for c in xyz]


def mpc80_magnitude(line):
    """(magnitude, band) from an OBS80 line: cols 66-70 are the observed magnitude and
    col 71 the band letter. Returns (None, ' ') when the field is blank or unparseable
    (many obs carry astrometry only). The fitter ignores mag; photometry/H and the
    --ident-check cross-opposition magnitude FLAG use it."""
    try:
        mag = float(line[65:70])
    except (ValueError, IndexError):
        mag = None
    band = line[70] if len(line) > 70 and line[70].strip() else " "
    return mag, band


def mpc80_catalog(line):
    """The star-catalog code from an OBS80 line: col 72, a SINGLE MPC catalogue letter
    (find_orb reads the same byte as `astrometric_net_code`, mpc_obs.cpp:1771). Returns
    None when blank -- most pre-2000 astrometry names no catalog, and radar records carry
    none at all. This is what EFCC18 debiasing keys on (debias.cat_code accepts the raw
    code as well as an ADES astCat name), so an OBS80-sourced obs is debiased exactly
    like its ADES twin."""
    c = line[71] if len(line) > 71 else " "
    return c if c.strip() else None


def utc_jd_before_1960(year, month, day, day_fraction=0.0):
    """The UTC Julian date of a calendar date before 1959-12-31, as a plain calendar count; for
    any later date, ``None`` (the caller uses ``astropy.time.Time``).

    UTC does not exist before 1960 and there were no leap seconds, so the date's Julian date is
    arithmetic. Going through ``astropy.time.Time`` returns the same number, but ERFA flags every
    such conversion "dubious year" (433 Eros alone has thousands of 1890s plates), and the only
    ways to silence that warning are a process-wide filter, which also hides the warning's other
    and genuine cause (a date after ~2029, where leap seconds are unknowable), or
    ``catch_warnings``, which is not thread-safe in the threaded web server. Equal to the ``Time``
    route to the last bit on every earlier day (tests/test_install_and_api_guards.py). 1959-12-31
    itself is left to astropy: ERFA stretches that one day by the 1.4 s offset that begins the
    next, so calendar arithmetic would differ from it by 0.17 s.
    """
    if (int(year), int(month), int(day)) >= (1959, 12, 31):
        return None
    import erfa
    return 2400000.5 + erfa.cal2jd(int(year), int(month), int(day))[1] + day_fraction


def mpc80_to_radec(line):
    y, mo = int(line[15:19]), int(line[20:22])
    dfrac = float(line[23:32])
    day = int(dfrac); frac = dfrac - day
    jd = utc_jd_before_1960(y, mo, day, frac)
    if jd is None:
        jd = Time(f"{y:04d}-{mo:02d}-{day:02d}T00:00:00", scale="utc").jd + frac
    rh, rm, rs = float(line[32:34]), float(line[35:37]), float(line[38:44])
    ra = 15.0 * (rh + rm/60 + rs/3600)
    sgn = -1.0 if line[44] == "-" else 1.0
    dd, dmn, ds = float(line[45:47]), float(line[48:50]), float(line[51:56])
    dec = sgn * (dd + dmn/60 + ds/3600)
    return jd, ra, dec, line[77:80].strip()
