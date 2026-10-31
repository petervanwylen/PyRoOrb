"""The linear congruential generator (LCG) used to jiggle search midpoints.

Ported bit-for-bit from ``tkRand``/``initGlobals`` in ``d2math.c``. The
reference author avoided libc ``rand()`` for its thread-safety overhead
and rolled a simple 59-bit LCG per NAG. Reproducing the *exact* sequence
matters here: in ``repeatable`` mode this is the only source of
randomness driving which orbit bins get explored, so an identical seed
and generator give (very close to) bit-identical scores.
"""
from __future__ import annotations

from .constants import INV_LCGM, LCGA, LCGM_MASK

# Seed digest2 uses for every tracklet when running in "repeatable" mode.
REPEATABLE_SEED = 3


class Lcg:
    """Per-tracklet LCG state. ``state`` must be odd (matches C's rand64)."""

    __slots__ = ("state",)

    def __init__(self, seed: int = REPEATABLE_SEED) -> None:
        self.state = seed

    def next(self) -> float:
        """Advance the generator and return a value in [0, 1)."""
        # C computes this as 64-bit-unsigned (rand64 * LCGA) & LCGM, where
        # LCGM is a 59-bit mask. Python ints are unbounded, but ANDing with
        # a 59-bit mask afterward keeps only the low 59 bits either way, so
        # no explicit 64-bit wraparound step is needed for bit parity.
        self.state = (self.state * LCGA) & LCGM_MASK
        return self.state * INV_LCGM
