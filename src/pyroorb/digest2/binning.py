"""Bin lookups for the (q, e, i, H) population-model histogram.

Direct port of ``qeiToBin``/``hToBin``/``qeihToBin`` in ``d2model.c``.
Each partition array holds the *exclusive upper edge* of its bin: bin
index ``k`` is the smallest ``k`` such that ``value < partition[k]``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .constants import EPART, EX, HPART, HX, IPART, IX, QPART, QX


def qei_to_bin(q: float, e: float, i: float) -> Optional[Tuple[int, int, int]]:
    """Return (iq, ie, ii) bin indices, or None if out of model range."""
    iq = 0
    while q >= QPART[iq]:
        iq += 1
        if iq == QX:
            return None
    ie = 0
    while e >= EPART[ie]:
        ie += 1
        if ie == EX:
            return None
    ii = 0
    while i >= IPART[ii]:
        ii += 1
        if ii == IX:
            return None
    return iq, ie, ii


def h_to_bin(h: float) -> int:
    """Return H-magnitude bin index. The last bin is a catch-all (open-ended)."""
    ih = 0
    while h >= HPART[ih] and ih < HX - 1:
        ih += 1
    return ih


def qeih_to_bin(q: float, e: float, i: float, h: float):
    """Return (iq, ie, ii, ih), or None if (q, e, i) is out of model range."""
    bin3 = qei_to_bin(q, e, i)
    if bin3 is None:
        return None
    return (*bin3, h_to_bin(h))
