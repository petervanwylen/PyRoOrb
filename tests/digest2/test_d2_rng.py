from pyroorb.digest2.rng import Lcg

# Reference values from a standalone C harness running the exact tkRand/
# initGlobals algorithm from d2math.c, seeded at 3 (digest2's "repeatable"
# seed). See the port's validation notes for how these were generated.
_REFERENCE = [
    0.0015762136730836773,
    0.38537207475475027,
    0.6771517073363665,
    0.1138408107073485,
    0.6751052116415753,
]


def test_lcg_matches_c_reference_sequence():
    lcg = Lcg(3)
    for expected in _REFERENCE:
        assert lcg.next() == expected


def test_lcg_values_in_unit_interval():
    lcg = Lcg(3)
    for _ in range(1000):
        v = lcg.next()
        assert 0.0 <= v < 1.0
