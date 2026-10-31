"""The verdict of one ITF search, as a word rather than a number.

A search ends in one of three verdicts. Until 2026-10 they were the floats
``1`` / ``0.6`` / ``0`` (``ItfSearchResult.status``), compared literally at
dozens of sites; ``0.6`` meant nothing a reader could guess. They are now the
members of :class:`SearchVerdict`, and the float survives only where it is
already written to disk (stored batch results, the batch CSV's ``status``
column), produced by :attr:`SearchVerdict.legacy_status`.

A verdict is never a number: comparing one with a number raises ``TypeError``
instead of quietly answering ``False``, so a site that still compares against
``0.6`` fails loudly rather than misjudging every result.
"""

import enum

__all__ = ["SearchVerdict"]


class SearchVerdict(enum.Enum):
    """What one search concluded.

    =========  ======  ====================================================
    member     legacy  meaning
    =========  ======  ====================================================
    NONE       0       nothing linked (or a linkage the search refused)
    POSSIBLE   0.6     a linkage kept for review, never submittable
    LINKED     1       a submittable linkage
    =========  ======  ====================================================

    The value is the word stored batch results already carry as ``verdict``.
    """

    NONE = "none"
    POSSIBLE = "possible"
    LINKED = "linked"

    @property
    def rank(self):
        """Strength for sorting: NONE 0 < POSSIBLE 1 < LINKED 2."""
        return _RANK[self]

    @property
    def legacy_status(self):
        """The float this verdict was before 2026-10 (0.0 / 0.6 / 1.0) --
        written ONLY where a file on disk already holds it."""
        return _LEGACY[self]

    @classmethod
    def from_legacy(cls, status):
        """The verdict a stored float (0 / 0.6 / 1) stands for."""
        f = float(status)
        for member, legacy in _LEGACY.items():
            if f == legacy:
                return member
        raise ValueError(f"not a legacy search status: {status!r}")

    def __eq__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            raise TypeError("a SearchVerdict is not a number; compare with "
                            "SearchVerdict members (or use .legacy_status)")
        return super().__eq__(other)

    __hash__ = enum.Enum.__hash__


_RANK = {SearchVerdict.NONE: 0, SearchVerdict.POSSIBLE: 1,
         SearchVerdict.LINKED: 2}
_LEGACY = {SearchVerdict.NONE: 0.0, SearchVerdict.POSSIBLE: 0.6,
           SearchVerdict.LINKED: 1.0}
