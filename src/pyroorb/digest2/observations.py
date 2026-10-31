"""Observation parsing: MPC 80-column, ADES XML, and ADES PSV formats.

The MPC 80-column parser is ported field-for-field from ``parseMpc80``/
``parseMpcSat``/``parseMpcRoving`` (d2mpc.c) and ``updateMagnitude``
(common.c), including the exact Modified Julian Date arithmetic
(``d2mpc.c`` lines 170-176) -- this is the input side of the scoring
engine, so producing the same (mjd, ra, dec) triples the C parser would
is what makes everything downstream comparable.

ADES support (XML and pipe-separated-value) has no C reference: the
original CLI's ``d2ades.c`` parses ADES XML via libxml2 but is not part
of the reusable scoring library (``d2lib.c``) that this project's
algorithm core is validated against -- it is independent, equivalent
parsing of the same standard fields.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ._cparse import must_strtod, must_strtoi
from .constants import BAND_CORRECTIONS, DEFAULT_BAND_CORRECTION

# km -> AU, matching the sf constant in d2mpc.c / d2ades.c
_KM_TO_AU = 1.0 / 149.59787e6

_MJD_FLOOKUP = (0, 306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275)


def date_to_mjd(year: int, month: int, day: float) -> float:
    """Calendar date -> Modified Julian Date (d2mpc.c: parseMpc80, verbatim).

    Uses truncating (toward zero) integer division for the ``z`` term,
    as C does; ``//`` alone would floor instead and disagree once
    ``month - 14`` is negative (which it always is here, since month
    <= 12).
    """
    z = year + int((month - 14) / 12)
    m = _MJD_FLOOKUP[month] + 365 * z + z // 4 - z // 100 + z // 400 - 678882
    return m + day


def update_magnitude(band: str, mag: float) -> float:
    """Normalize an observed magnitude to V-band (common.c: updateMagnitude)."""
    if mag > 0:
        mag += BAND_CORRECTIONS.get(band, DEFAULT_BAND_CORRECTION)
    return mag


@dataclass
class Observation:
    """A single astrometric detection, in the units the scoring engine
    wants: MJD, RA/Dec in *radians*, magnitude already normalized to V.
    """

    mjd: float
    ra: float             # radians
    dec: float             # radians
    vmag: float = 0.0      # 0 means "no magnitude reported"
    obscode: str = "500"
    spacebased: bool = False
    earth_observer: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # AU, equatorial
    rms_ra: float = 0.0    # radians (ADES-reported uncertainty; 0 = not reported)
    rms_dec: float = 0.0   # radians


def _roving_position(x: float, y: float, altitude: float) -> Tuple[float, float, float]:
    """Port of ``roving_position`` in d2math.c.

    Preserved verbatim, including its apparently swapped use of ``x``
    and ``y`` between the ellipsoid-radius formula and the final
    cos/sin assignment -- this is what the reference implementation
    does, so bit-for-bit parity means reproducing it as-is rather than
    "fixing" it.
    """
    a = 6378137.0
    b = 6356752.314245
    numerator = (a * a * math.cos(y)) ** 2 + (b * b * math.sin(y)) ** 2
    denominator = (a * math.cos(y)) ** 2 + (b * math.sin(y)) ** 2
    r = math.sqrt(numerator / denominator) + altitude
    return (r * math.cos(x) * math.cos(y), r * math.cos(x) * math.sin(y), r * math.sin(x))


def parse_mpc80(line: str) -> Optional[Tuple[str, Observation]]:
    """Parse one MPC 80-column observation line.

    Returns (12-character designation field, Observation), or None if
    the line isn't a parseable primary observation record.
    """
    if len(line) < 80:
        return None
    if line[14] not in ("C", "S", "B"):
        return None

    obscode = line[77:80]
    from .obscodes import parse_cod3
    site = parse_cod3(obscode)
    if site < 0:
        return None

    band = line[70]
    mag_str = line[65:70].strip()
    mag = float(mag_str) if mag_str else 0.0  # blank -> 0, no error (matches strtod use)

    decs = must_strtod(line[51:56])
    decm = must_strtoi(line[48:50])
    decd = must_strtoi(line[45:47])
    decg = line[44]
    ras = must_strtod(line[38:44])
    ram = must_strtoi(line[35:37])
    rah = must_strtoi(line[32:34])
    day = must_strtod(line[23:32])
    month = must_strtoi(line[20:22])
    year = must_strtoi(line[15:19])
    if None in (decs, decm, decd, ras, ram, rah, day, month, year):
        return None
    if not (1 <= month <= 12):
        return None

    desig = line[0:12]
    mjd = date_to_mjd(year, month, day)
    ra = ((rah * 60 + ram) * 60 + ras) * math.pi / (12 * 3600)
    dec = ((decd * 60 + decm) * 60 + decs) * math.pi / (180 * 3600)
    if decg == "-":
        dec = -dec

    obs = Observation(mjd=mjd, ra=ra, dec=dec, vmag=update_magnitude(band, mag), obscode=obscode)
    return desig, obs


def apply_mpc80_second_line(line: str, kind: str, obs: Observation) -> bool:
    """Apply a satellite ('s') or roving-observer ('v') second line to the
    preceding primary observation. Port of parseMpcSat/parseMpcRoving.
    """
    if len(line) < 80:
        return False
    if kind == "s":
        obscode = line[77:80]
        from .obscodes import parse_cod3
        if parse_cod3(obscode) != parse_cod3(obs.obscode):
            return False
        x = must_strtod(line[34:45])
        y = must_strtod(line[46:57])
        z = must_strtod(line[58:69])
        if None in (x, y, z):
            return False
        if line[32] == "1":
            x, y, z = x * _KM_TO_AU, y * _KM_TO_AU, z * _KM_TO_AU
        obs.earth_observer = (x, y, z)
        obs.spacebased = True
        return True
    if kind == "v":
        x = must_strtod(line[34:45])
        y = must_strtod(line[46:57])
        z = must_strtod(line[58:69])
        if None in (x, y, z):
            return False
        x, y, z = _roving_position(x, y, z)
        obs.earth_observer = (x * _KM_TO_AU, y * _KM_TO_AU, z * _KM_TO_AU)
        obs.spacebased = True
        return True
    return False


def parse_mpc80_file(path) -> Dict[str, List[Observation]]:
    """Parse a whole MPC 80-column file, grouping all observations by
    designation regardless of ordering in the file.

    This is the convenient, library-style grouping (used by
    :mod:`pydigest2.api`); the CLI (:mod:`pydigest2.cli`) instead
    groups *consecutive* same-designation lines, matching the original
    program's streaming behavior -- see :mod:`pydigest2.intake`.
    """
    tracklets: Dict[str, List[Observation]] = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if len(line) < 80:
                continue
            if line[14] in ("s", "v"):
                desig = line[0:12]
                obs_list = tracklets.get(desig)
                if obs_list:
                    apply_mpc80_second_line(line, line[14], obs_list[-1])
                continue
            parsed = parse_mpc80(line)
            if parsed is None:
                continue
            desig, obs = parsed
            tracklets.setdefault(desig, []).append(obs)
    return {k: sorted(v, key=lambda o: o.mjd) for k, v in tracklets.items()}


# --- ADES -------------------------------------------------------------

def _iso_to_mjd(iso_str: str) -> float:
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1]
    dt = datetime.fromisoformat(s)
    day_fraction = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                    + dt.microsecond / 3_600_000_000.0)
    return date_to_mjd(dt.year, dt.month, dt.day + day_fraction / 24.0)


def _observer_position(sys_: Optional[str], pos1: Optional[str], pos2: Optional[str],
                        pos3: Optional[str]) -> Tuple[bool, Tuple[float, float, float]]:
    if not (sys_ and pos1 and pos2 and pos3):
        return False, (0.0, 0.0, 0.0)
    try:
        x, y, z = float(pos1), float(pos2), float(pos3)
    except ValueError:
        return False, (0.0, 0.0, 0.0)
    is_satellite = "_KM" in sys_
    is_roving = "WGS84" in sys_
    if is_roving:
        x, y, z = _roving_position(x, y, z)
    if is_satellite or is_roving:
        x, y, z = x * _KM_TO_AU, y * _KM_TO_AU, z * _KM_TO_AU
    return True, (x, y, z)


def _text(el, tag, ns):
    child = el.find(ns + tag)
    return child.text.strip() if child is not None and child.text else None


def _parse_ades_optical(optical, ns: str) -> Optional[Observation]:
    obs_time = _text(optical, "obsTime", ns)
    ra_s = _text(optical, "ra", ns)
    dec_s = _text(optical, "dec", ns)
    stn = _text(optical, "stn", ns)
    if not (obs_time and ra_s and dec_s and stn):
        return None
    try:
        mjd = _iso_to_mjd(obs_time)
        ra = math.radians(float(ra_s))
        dec = math.radians(float(dec_s))
    except ValueError:
        return None

    mag_s = _text(optical, "mag", ns)
    band = _text(optical, "band", ns)
    mag = float(mag_s) if mag_s else 0.0
    # Matches d2ades.c processOptical: with no <band>, vmag stays 0 even
    # if a magnitude was reported (there's nothing to convert it with).
    vmag = update_magnitude(band[0], mag) if band else 0.0

    rms_ra_s = _text(optical, "rmsRA", ns)
    rms_dec_s = _text(optical, "rmsDec", ns)
    rms_ra = math.radians(float(rms_ra_s) / 3600.0) if rms_ra_s else 0.0
    rms_dec = math.radians(float(rms_dec_s) / 3600.0) if rms_dec_s else 0.0

    spacebased, earth_obs = _observer_position(
        _text(optical, "sys", ns), _text(optical, "pos1", ns),
        _text(optical, "pos2", ns), _text(optical, "pos3", ns),
    )

    return Observation(mjd=mjd, ra=ra, dec=dec, vmag=vmag, obscode=stn,
                        spacebased=spacebased, earth_observer=earth_obs,
                        rms_ra=rms_ra, rms_dec=rms_dec)


def parse_ades_xml(path) -> Dict[str, List[Observation]]:
    """Parse an ADES XML observation file (stdlib ``xml.etree``, no libxml2)."""
    tree = ET.parse(path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    tracklets: Dict[str, List[Observation]] = {}

    def handle(optical_el):
        obs = _parse_ades_optical(optical_el, ns)
        if obs is None:
            return
        desig = None
        for tag in ("trkSub", "provID", "permID"):
            el = optical_el.find(ns + tag)
            if el is not None and el.text and el.text.strip():
                desig = el.text.strip()
                break
        tracklets.setdefault(desig or "unknown", []).append(obs)

    found_nested = False
    for obs_block in root.iter(ns + "obsBlock"):
        for obs_data in obs_block.iter(ns + "obsData"):
            for optical in obs_data.iter(ns + "optical"):
                found_nested = True
                handle(optical)
    if not found_nested:
        for optical in root.iter(ns + "optical"):
            handle(optical)

    return {k: sorted(v, key=lambda o: o.mjd) for k, v in tracklets.items()}


def _as_float(val: Optional[str], default: float = 0.0) -> float:
    if val is None or val == "" or val == "None":
        return default
    return float(val)


def parse_ades_psv(path) -> Dict[str, List[Observation]]:
    """Parse an ADES pipe-separated-value observation file."""

    def split(line: str) -> List[str]:
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts

    tracklets: Dict[str, List[Observation]] = {}
    headers: Optional[List[str]] = None

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("!") and "|" not in line:
                continue
            if line.startswith("!"):
                line = line[1:]
            if headers is None:
                headers = split(line)
                continue
            fields = split(line)
            if len(fields) != len(headers):
                continue
            row = dict(zip(headers, fields))

            desig = "unknown"
            for key in ("trkSub", "provID", "permID"):
                v = row.get(key, "")
                if v and v != "None":
                    desig = v
                    break

            try:
                mjd = _iso_to_mjd(row["obsTime"].strip())
                ra = math.radians(float(row["ra"]))
                dec = math.radians(float(row["dec"]))
                obscode = row["stn"].strip()
                if not obscode:
                    continue
            except (KeyError, ValueError):
                continue

            mag = _as_float(row.get("mag"))
            band = row.get("band")
            # Same "no band -> no conversion, vmag stays 0" rule as the
            # ADES XML parser (there is no C reference for PSV; this
            # keeps both ADES paths consistent with each other and with
            # d2ades.c's handling of MPC80/XML magnitudes).
            vmag = update_magnitude(band[0], mag) if band else 0.0
            rms_ra = math.radians(_as_float(row.get("rmsRA")) / 3600.0)
            rms_dec = math.radians(_as_float(row.get("rmsDec")) / 3600.0)

            def field_(key):
                v = row.get(key, "").strip()
                return v if v and v != "None" else None

            spacebased, earth_obs = _observer_position(
                field_("sys"), field_("pos1"), field_("pos2"), field_("pos3"))

            obs = Observation(mjd=mjd, ra=ra, dec=dec, vmag=vmag, obscode=obscode,
                               spacebased=spacebased, earth_observer=earth_obs,
                               rms_ra=rms_ra, rms_dec=rms_dec)
            tracklets.setdefault(desig, []).append(obs)

    return {k: sorted(v, key=lambda o: o.mjd) for k, v in tracklets.items()}
