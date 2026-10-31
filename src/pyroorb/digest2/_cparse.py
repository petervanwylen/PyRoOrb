"""Small numeric-field parsing helpers shared by the fixed-column readers
(obscodes table, MPC 80-column observations).

These mirror ``mustStrtod``/``mustStrtoi`` in ``common.c``: strict
parsers for a field the caller has already sliced out of a fixed-width
record. The one C quirk worth preserving is that a leading sign may be
followed by whitespace before the digits (``"- 3471.6659"`` shows up in
real MPC satellite-position lines); Python's ``float()`` alone rejects
that, so the sign is peeled off by hand first.
"""
from __future__ import annotations

from typing import Optional


def must_strtod(field: str) -> Optional[float]:
    """Parse a numeric field; return None if it holds no number at all
    (the C equivalent of ``mustStrtod`` leaving ``errno`` set)."""
    s = field.strip()
    if not s:
        return None
    neg = s[0] in "+-"
    sign = -1.0 if s[0] == "-" else 1.0
    if neg:
        s = s[1:].lstrip()
    try:
        value = float(s)
    except ValueError:
        return None
    return sign * value if neg else value


def must_strtoi(field: str) -> Optional[int]:
    """Parse a non-negative integer field (mustStrtoi in common.c
    disallows negatives outright)."""
    s = field.strip()
    if not s or not s.isdigit():
        return None
    return int(s)
