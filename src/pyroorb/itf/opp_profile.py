"""Night, span and opposition counting on raw OBS80 text, and the opposition
PROFILE of an arc (:class:`OppProfile`, :class:`OppProfileGrade`).

Split out of the former ``itf/obsutil.py``. These are the ITF search's own
counting rules, kept exactly as the notebook pipeline wrote them, even where
they disagree with the engine's other rules.
"""

import enum
from dataclasses import dataclass
from datetime import datetime

import pyroorb.orbit_common as _oc


def obs_night_ordinal(line):
    """Date of an obs line as ordinal-day float (for night/opp grouping).

    The WHOLE fractional-day field (columns 26-32): two decimals truncated the
    time to ~14 minutes, so two tracklets 11.9 h apart read as 12 h and passed
    the MPC's 12-hour night rule as two nights."""
    return (datetime.strptime(line[15:25], "%Y %m %d").toordinal()
            + float(line[25:32]))




# TWO units, each named for what it is.
#
# A SPAN is the ported notebook's unit: a window of SPAN_DAYS opened by its first
# observation, so two consecutive calendar nights (22 h apart) are ONE span, and
# the next span opens at the first observation >= SPAN_DAYS later.  The DESITF
# acceptance profile (spans >= 8 / 6 / 4 / 3) was calibrated on this counter
# (``itflib/obs.py:count_nights_2``) and keeps it -- deliberately "long": in
# that gate a second night 22 h later is not the independent evidence the
# thresholds were set against.  It was 1.6 d under the name "night"; it is
# now 1.5 (a 36-hour span) under its own word -- the count is
# identical on every apparition of every recovery fixture at 1.5 and 1.6.
#
# A NIGHT is what an observer means: the same window rule at NIGHT_DAYS (12 h,
# ``orbit_common.NIGHT_WINDOW_DAYS`` -- the tree's one definition of a night).
# The two-opposition clause of the opposition-profile gate, the MPC rules, the night-
# coherence statistic and ITF-to-ITF linking count nights; everything
# calibrated on the notebook counts spans.  The night count is >= the span
# count on every arc, so a clause moved from spans to nights is LOOSENED and
# has to be re-justified.
SPAN_DAYS = 1.5
NIGHT_DAYS = _oc.NIGHT_WINDOW_DAYS


def spans_of(lines, span_days=SPAN_DAYS):
    """Group OBS80 ``lines`` into SPANS (see :data:`SPAN_DAYS`): sorted by date,
    the first observation >= ``span_days`` after the CURRENT span's start opens
    a new one.  Returns ``[[line, ...], ...]`` in date order -- span 0 first.
    THE one implementation, so "span 4 of the seed" means the same thing in the
    opposition-profile gate, the harness and the seed-clip ledger.  Pass ``NIGHT_DAYS`` (or
    call :func:`nights_of`) for the observer's night."""
    dated = sorted(((obs_night_ordinal(l), l) for l in lines if l.strip()),
                   key=lambda t: t[0])
    if not dated:
        return []
    spans = [[]]
    start = dated[0][0]
    for d, line in dated:
        if d - start >= span_days:
            start = d
            spans.append([])
        spans[-1].append(line)
    return spans


def count_spans(astrometry, span_days=SPAN_DAYS):
    """Distinct SPANS in an OBS80 blob (see :data:`SPAN_DAYS`; ``>= span_days``
    from the current span's start opens the next)."""
    return len(spans_of(astrometry.splitlines(), span_days))


def nights_of(lines):
    """Group OBS80 ``lines`` into observing NIGHTS -- the observer's unit, the
    12-hour window of :data:`NIGHT_DAYS`."""
    return spans_of(lines, NIGHT_DAYS)


def count_nights(astrometry):
    """Distinct observing NIGHTS in an OBS80 blob (see :func:`nights_of`)."""
    return len(nights_of(astrometry.splitlines()))




# The opposition counter's own rules, kept exactly as the notebook pipeline
# wrote them (the rename changes no count). They disagree with the engine's
# other opposition and night rules.
#: a new opposition starts more than this many days after the CURRENT
#: opposition's first night (span from the opposition's start, not a gap)
ITF_OPPOSITION_SPAN_DAYS = 275
#: an opposition has a second night once a night falls this many days or more
#: after its first (a fourth night-like scale beside 0.5 / 0.5 / 1.5 d)
MULTINIGHT_MIN_SEPARATION_DAYS = 0.7
#: a new opposition that starts within this many days of the previous night is
#: credited with a second night at once (unexplained in the notebook; #79)
NEW_OPPOSITION_CARRY_DAYS = 90


class OppProfileGrade(enum.Enum):
    """How trustworthy an arc's opposition profile is.

    Until 2026-10 this was smuggled into the opposition COUNT as the floats
    1.5 and 1.75 (``count_opps``); :attr:`OppProfile.legacy_code` still writes
    them where a file on disk holds them.

    ==========  ======  ==================================================
    member      legacy  meaning
    ==========  ======  ==================================================
    ONE         1       one opposition
    TWO_THIN    1.5     two oppositions, only one with two or more nights
    THREE_THIN  1.75    three oppositions, only one with two or more nights
    TWO         2       two oppositions, not thin
    MULTI       n >= 3  three or more, not thin (or a confirmed three-thin)
    ==========  ======  ==================================================
    """

    ONE = "one"
    TWO_THIN = "two_thin"
    THREE_THIN = "three_thin"
    TWO = "two"
    MULTI = "multi"


@dataclass(frozen=True)
class OppProfile:
    """The nights-by-oppositions structure of an arc (see :func:`opp_profile`).

    Not a number: comparing one with a number raises ``TypeError``. Ask the
    :attr:`grade` or :attr:`n_oppositions` instead. It FORMATS as the legacy
    code (``f"{p} opps"`` reads ``1.75 opps``), so every message, log line and
    stored ``reason`` string reads exactly as before the rename."""

    n_oppositions: int
    n_multinight_oppositions: int
    #: the three-thin confirmation (``search._confirm_top_pair``) judged this
    #: profile as good as three oppositions -- see :meth:`confirmed`
    confirmed_three: bool = False

    @property
    def grade(self):
        n, m = self.n_oppositions, self.n_multinight_oppositions
        if n == 1:
            return OppProfileGrade.ONE
        if n == 2:
            return OppProfileGrade.TWO_THIN if m == 1 else OppProfileGrade.TWO
        if n == 3 and m == 1 and not self.confirmed_three:
            return OppProfileGrade.THREE_THIN
        return OppProfileGrade.MULTI

    @property
    def is_thin(self):
        """TWO_THIN or THREE_THIN -- the legacy 1.5 / 1.75."""
        return self.grade in (OppProfileGrade.TWO_THIN,
                              OppProfileGrade.THREE_THIN)

    @property
    def legacy_code(self):
        """The pre-2026-10 float: 1.5 / 1.75 for a thin profile, else the
        opposition count (3 for a confirmed three-thin). Written ONLY where a
        file on disk already holds it."""
        g = self.grade
        if g is OppProfileGrade.TWO_THIN:
            return 1.5
        if g is OppProfileGrade.THREE_THIN:
            return 1.75
        return self.n_oppositions

    def confirmed(self):
        """This THREE_THIN profile after the three-thin confirmation passed:
        graded as three oppositions from then on."""
        return OppProfile(self.n_oppositions, self.n_multinight_oppositions,
                          confirmed_three=True)

    def __format__(self, spec):
        return format(self.legacy_code, spec or "")

    def __str__(self):
        return str(self.legacy_code)

    def __eq__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            raise TypeError("an OppProfile is not a number; ask its .grade "
                            "or .n_oppositions (or use .legacy_code)")
        if not isinstance(other, OppProfile):
            return NotImplemented
        return (self.n_oppositions, self.n_multinight_oppositions,
                self.confirmed_three) == (other.n_oppositions,
                                          other.n_multinight_oppositions,
                                          other.confirmed_three)

    def __hash__(self):
        return hash((self.n_oppositions, self.n_multinight_oppositions,
                     self.confirmed_three))

    def _no_order(self, other):
        raise TypeError("an OppProfile is not ordered; ask its .grade or "
                        ".n_oppositions")

    __lt__ = __le__ = __gt__ = __ge__ = _no_order


def opp_profile(astrometry):
    """The :class:`OppProfile` of an OBS80 blob.

    The notebook pipeline's counter, unchanged (see the constants above): a
    new opposition opens ``> ITF_OPPOSITION_SPAN_DAYS`` after the current
    one's first night, and an opposition counts as multi-night once a night
    falls ``>= MULTINIGHT_MIN_SEPARATION_DAYS`` after its first. Two
    oppositions with only one multi-night are TWO_THIN, three with only one
    are THREE_THIN -- arcs too weak to trust a multi-opposition linkage."""
    dates = sorted(obs_night_ordinal(s) for s in astrometry.splitlines() if s.strip())
    opp_start = dates[0]
    prev = dates[0]
    opps = 1
    current_opp_2_nights = False
    opps_w_2_nights = 0
    for d in dates:
        if d - opp_start > ITF_OPPOSITION_SPAN_DAYS:
            opps += 1
            current_opp_2_nights = False
            if d - prev < NEW_OPPOSITION_CARRY_DAYS:
                current_opp_2_nights = True
                opps_w_2_nights += 1
            opp_start = d
        elif (d - opp_start >= MULTINIGHT_MIN_SEPARATION_DAYS
              and not current_opp_2_nights):
            opps_w_2_nights += 1
            current_opp_2_nights = True
        prev = d
    return OppProfile(opps, opps_w_2_nights)
