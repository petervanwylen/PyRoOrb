"""The web object-box tokeniser (engine_api.parse_designations): splits an input string into the
designations to fit as one combined/identified object. Provisional forms with internal spaces must
stay whole -- asteroid ("2010 VL65"), comet ("C/2023 A3"), and natural-satellite ("S/2003 J 2",
which has TWO internal spaces) -- while genuinely separate objects still split. Pure/offline."""
from pyroorb.web import engine_api


CASES = [
    # single designations that must stay whole
    ("S/2003 J 2", ["S/2003 J 2"]),          # natural satellite (two internal spaces) -- the fix
    ("2003 J 2", ["2003 J 2"]),              # bare year form  \d\d\d\d \w \d
    ("S/2003 J2", ["S/2003 J2"]),            # no space before the index
    ("2010 VL65", ["2010 VL65"]),
    ("C/2023 A3", ["C/2023 A3"]),
    ("1999 AN10", ["1999 AN10"]),
    # multiple objects that must still split
    ("2010 VL65 2021 UA12", ["2010 VL65", "2021 UA12"]),   # two asteroids -- NOT over-merged
    ("S/2003 J 2 S/2003 J 3", ["S/2003 J 2", "S/2003 J 3"]),
    ("S/2003 J 2 = S/2003 J 3", ["S/2003 J 2", "S/2003 J 3"]),
    ("2003 J 2015 BR525", ["2003 J", "2015 BR525"]),       # a 4-digit YEAR is never swallowed as an index
]


def test_parse_designations():
    for text, expected in CASES:
        assert engine_api.parse_designations(text) == expected, (text, engine_api.parse_designations(text))
