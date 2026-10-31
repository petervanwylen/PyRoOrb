"""MPC observatory code parsing.

Ported from ``parseCod3`` (common.c) and ``readOCD`` (d2mpc.c). Each
observatory contributes a topocentric parallax constant pair
(:math:`\\rho\\cos\\phi'`, :math:`\\rho\\sin\\phi'`) and a longitude, used to
locate the observer relative to the geocenter at observation time.

The obscodes file is the flat-text table published at
https://minorplanetcenter.net/iau/lists/ObsCodes.html (fixed-width
columns, one observatory per line, wrapped in a ``<pre>`` tag when saved
straight from that page -- exactly as digest2's own ``getOCD()``
fetches it via curl). A header line is always present and skipped
unconditionally, matching the C reader.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ._cparse import must_strtod
from .constants import OBSCODE_NAMESPACE_SIZE

# Scale factor: 1 Earth equatorial radius, in AU (d2mpc.c readOCD).
_EARTH_RADIUS_AU = 6.37814e6 / 149.59787e9

UNSET_OBSERR = -1.0  # sentinel: "no per-site obserr configured"


@dataclass
class Site:
    longitude: float = 0.0    # circles (i.e. fraction of a full turn), east+
    rho_cos_phi: float = 0.0  # AU
    rho_sin_phi: float = 0.0  # AU
    obs_err: float = UNSET_OBSERR  # radians; UNSET_OBSERR = "use global default"


class ObscodeError(ValueError):
    pass


def parse_cod3(code: str) -> int:
    """Convert a 3-character MPC obscode to its site-table index.

    Returns -1 if the code doesn't parse (matches C's ``parseCod3``:
    first character must be 0-9 or A-Z, the other two must be digits).
    """
    if len(code) < 3:
        return -1
    c0 = code[0]
    if c0.isdigit():
        hp = ord(c0) - ord("0")
    elif "A" <= c0 <= "Z":
        hp = ord(c0) - ord("A") + 10
    else:
        return -1
    c1, c2 = code[1], code[2]
    if c1.isdigit() and c2.isdigit():
        return hp * 100 + (ord(c1) - ord("0")) * 10 + (ord(c2) - ord("0"))
    return -1


def parse_obscodes_text(text: str) -> Dict[int, Site]:
    """Parse an obscodes flat file (as text) into {site_index: Site}."""
    lines = text.split("\n")
    sites: Dict[int, Site] = {}
    if not lines:
        return sites
    # first line is always a header (or the "<pre>" wrapper tag from a
    # browser-saved page) and is unconditionally discarded, as in readOCD.
    for line in lines[1:]:
        if len(line) < 3:
            continue
        idx = parse_cod3(line[0:3])
        if idx < 0 or idx >= OBSCODE_NAMESPACE_SIZE:
            continue
        padded = line.ljust(30)
        lon = must_strtod(padded[4:13])
        rcos = must_strtod(padded[13:21])
        rsin = must_strtod(padded[21:30])
        if lon is None or rcos is None or rsin is None:
            continue
        sites[idx] = Site(
            longitude=lon / 360.0,
            rho_cos_phi=rcos * _EARTH_RADIUS_AU,
            rho_sin_phi=rsin * _EARTH_RADIUS_AU,
            obs_err=UNSET_OBSERR,
        )
    return sites


def load_obscodes(path: os.PathLike | str) -> Dict[int, Site]:
    text = Path(path).read_text(errors="replace")
    sites = parse_obscodes_text(text)
    if not sites:
        raise ObscodeError(f"no observatory codes could be parsed from {path}")
    return sites


class SiteTable:
    """Sparse site table with digest2's implicit default: any obscode not
    present in the file behaves as if at the geocenter (longitude=0,
    rho*=0) with no per-site obserr override -- exactly what a
    zero-initialized C ``site`` struct would give.
    """

    def __init__(self, sites: Optional[Dict[int, Site]] = None) -> None:
        self._sites = dict(sites) if sites else {}

    def __getitem__(self, index: int) -> Site:
        return self._sites.get(index, Site())

    def __setitem__(self, index: int, site: Site) -> None:
        self._sites[index] = site

    def __contains__(self, index: int) -> bool:
        return index in self._sites

    def set_obserr(self, index: int, obserr_rad: float) -> None:
        site = self._sites.get(index)
        if site is None:
            site = Site()
            self._sites[index] = site
        site.obs_err = obserr_rad

    @classmethod
    def from_file(cls, path: os.PathLike | str) -> "SiteTable":
        return cls(load_obscodes(path))
