"""Physical and model constants for digest2.

Values are ported bit-for-bit from the reference C implementation
(``d2math.c: initGlobals()`` and ``d2model.c``) so that arithmetic here
lines up with the original to the last bit of double precision.
"""
from __future__ import annotations

import math

# Gaussian gravitational constant and derivatives (d2math.c initGlobals)
K = 0.01720209895
INV_K = 1.0 / K
U = K * K
TWO_PI = math.pi * 2.0

# arc-second -> radian, also used as the great-circle-fit rotation tolerance
ARCSEC_RAD = math.pi / (180 * 3600)
RTOL = ARCSEC_RAD

# default per-observation astrometric error, radians (1 arcsec)
DEFAULT_OBSERR_RAD = 1.0 * ARCSEC_RAD

# LCG "random" number generator constants (d2math.c initGlobals, tkRand)
#   LCGA = pow(13, 13)                     (13**13, exact in double)
#   LCGM = 1; LCGM <<= 59; invLCGM = 1/LCGM; LCGM--
# i.e. invLCGM is 1/2**59 (computed *before* the final decrement), while
# the mask used in the AND is 2**59 - 1. Order matters for bit parity.
LCGA = 13 ** 13
_LCGM_POW = 1 << 59
INV_LCGM = 1.0 / _LCGM_POW
LCGM_MASK = _LCGM_POW - 1

# search-space limits (d2math.c)
MIN_DISTANCE = 0.05      # AU
MAX_DISTANCE = 100.0     # AU
MIN_DISTANCE_STEP = 0.2  # AU  -- "big enough to always recurse"
MIN_ANGLE_STEP = 0.1     # radian
AGE_LIMIT = 1

# model bin extents (d2model.h)
D2CLASSES = 15
QX = 29
EX = 8
IX = 11
HX = 18

# model bin partitions (d2model.c) -- bin i holds values < partition[i]
QPART = (.4, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
         1.67, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.5,
         4., 4.5, 5., 5.5, 10., 20., 30., 40., 100.)
EPART = (.1, .2, .3, .4, .5, .7, .9, 1.1)
IPART = (2, 5, 10, 15, 20, 25, 30, 40, 60, 90, 180)
HPART = (6, 8, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25.5)

assert len(QPART) == QX
assert len(EPART) == EX
assert len(IPART) == IX
assert len(HPART) == HX

# obscode namespace size (3 char base-36-ish code -> int index)
OBSCODE_NAMESPACE_SIZE = 3600

# magnitude band -> V-band correction (common.c updateMagnitude)
BAND_CORRECTIONS = {
    "V": 0.0, "B": -0.8, "U": -1.3, "g": -0.28, "r": 0.23, "R": 0.4,
    "C": 0.4, "W": 0.4, "i": 0.39, "z": 0.37, "I": 0.8, "J": 1.2,
    "w": -0.16, "y": 0.36, "L": 0.2, "H": 1.4, "K": 1.7, "Y": 0.7,
    "G": 0.24, "v": 0.0, "c": -0.05, "o": 0.33, "u": 2.5,
}
DEFAULT_BAND_CORRECTION = -0.8  # any band not in the table above

DEFAULT_V_MAG = 21.0  # used when a tracklet has no photometry at all
