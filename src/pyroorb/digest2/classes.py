"""Orbit classification tests.

Direct port of the fifteen ``isXxx`` predicates in ``d2model.c``. Each
takes perihelion distance ``q`` (AU), eccentricity ``e``, inclination
``i`` (degrees) and absolute magnitude ``h``, and returns whether a
trial orbit with those elements belongs to the named dynamical class.

Order and indices below match ``classAbbr``/``classHeading``/``isClass``
in the C source exactly -- other modules refer to classes by this index
(e.g. bin-tag arrays, CLI ``--limit`` class lookup, config-file class
name matching), so the order must not change.
"""
from __future__ import annotations

import math

__all__ = [
    "CLASS_ABBR", "CLASS_HEADING", "CLASS_TESTS", "N_CLASSES",
    "class_index",
]


def is_mpc_interesting(q: float, e: float, i: float, h: float) -> bool:
    # any of: q < 1.3, e >= .5, i >= 40, or Q > 10
    return q < 1.3 or e >= .5 or i >= 40. or q * (1. + e) / (1. - e) > 10.


def is_neo(q: float, e: float, i: float, h: float) -> bool:
    return q < 1.3


def is_h22_neo(q: float, e: float, i: float, h: float) -> bool:
    return q < 1.3 and h < 22.5


def is_h18_neo(q: float, e: float, i: float, h: float) -> bool:
    return q < 1.3 and h < 18.5


def is_mars_crosser(q: float, e: float, i: float, h: float) -> bool:
    return q < 1.67 and q >= 1.3 and q * (1 + e) / (1 - e) > 1.58


def is_hungaria(q: float, e: float, i: float, h: float) -> bool:
    if e > .18 or i < 16 or i > 34:
        return False
    a = q / (1 - e)
    return 1.78 < a < 2.


def is_phocaea(q: float, e: float, i: float, h: float) -> bool:
    if q < 1.5 or i < 20 or i > 27:
        return False
    a = q / (1 - e)
    return 2.2 < a < 2.45


def is_inner_mb(q: float, e: float, i: float, h: float) -> bool:
    if q < 1.67:
        return False
    a = q / (1 - e)
    return 2.1 < a < 2.5 and i < ((a - 2.1) / .4) * 10 + 7


def is_hansa(q: float, e: float, i: float, h: float) -> bool:
    if e > .25 or i < 20 or i > 23.5:
        return False
    a = q / (1 - e)
    return 2.55 < a < 2.72


def is_pallas(q: float, e: float, i: float, h: float) -> bool:
    if e > .35 or i < 24 or i > 37:
        return False
    a = q / (1 - e)
    return 2.5 < a < 2.8


def is_mid_mb(q: float, e: float, i: float, h: float) -> bool:
    if e > .45 or i > 20:
        return False
    a = q / (1 - e)
    return 2.5 < a < 2.8


def is_outer_mb(q: float, e: float, i: float, h: float) -> bool:
    if e > .4:
        return False
    a = q / (1 - e)
    return 2.8 < a < 3.25 and i < ((a - 2.8) / .45) * 16 + 20


def is_hilda(q: float, e: float, i: float, h: float) -> bool:
    if i > 18 or e > .4:
        return False
    a = q / (1 - e)
    return 3.9 < a < 4.02


def is_trojan(q: float, e: float, i: float, h: float) -> bool:
    if e > .22 or i > 38:
        return False
    a = q / (1 - e)
    return 5.05 < a < 5.35


def is_jfc(q: float, e: float, i: float, h: float) -> bool:
    if q < 1.3:
        return False
    t = 5.2 * (1 - e) / q + 2 * math.sqrt(q * (1 + e) / 5.2) * math.cos(math.radians(i))
    return 2 < t < 3


# 12 characters looks nice, 13 is ok; anything longer is truncated at
# CLI print time. Order fixed by the C source -- do not reorder.
CLASS_HEADING = (
    "MPC interest.",
    "NEO(q < 1.3)",
    "NEO(H <= 22)",
    "NEO(H <= 18)",
    "Mars Crosser",
    "Hungaria gr.",
    "Phocaea group",
    "Inner MB",
    "Pallas group",
    "Hansa group",
    "Middle MB",
    "Outer MB",
    "Hilda group",
    "Jupiter tr.",
    "Jupiter Comet",
)

CLASS_ABBR = (
    "Int", "NEO", "N22", "N18", "MC", "Hun", "Pho", "MB1",
    "Pal", "Han", "MB2", "MB3", "Hil", "JTr", "JFC",
)

CLASS_TESTS = (
    is_mpc_interesting, is_neo, is_h22_neo, is_h18_neo, is_mars_crosser,
    is_hungaria, is_phocaea, is_inner_mb, is_pallas, is_hansa, is_mid_mb,
    is_outer_mb, is_hilda, is_trojan, is_jfc,
)

N_CLASSES = len(CLASS_ABBR)
assert len(CLASS_HEADING) == N_CLASSES
assert len(CLASS_TESTS) == N_CLASSES

_NAME_TO_INDEX = {}
for _i, (_abbr, _heading) in enumerate(zip(CLASS_ABBR, CLASS_HEADING)):
    _NAME_TO_INDEX[_abbr] = _i
    _NAME_TO_INDEX[_heading] = _i


def class_index(name: str) -> int | None:
    """Resolve a class abbreviation or long-form heading to its index.

    Matches must be exact, as in the C config-file parser (``d2cli.c:
    readConfig``) -- no case-folding or fuzzy matching.
    """
    return _NAME_TO_INDEX.get(name)
