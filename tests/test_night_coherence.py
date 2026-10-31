"""The per-night coherence statistic: what its numbers MEAN.

Every assertion here is about the SCALE.  The clause is read by two places that
each carry a threshold (``ItfConfig.max_night_systematic_sigma`` in the engine,
an ITF-to-ITF campaign), so a statistic whose "perfect" value
is not 1.0 makes both thresholds arbitrary -- which is exactly what happened:
the first version divided a 2-D radial residual by a 1-AXIS sigma, so a flawless
arc scored sqrt(2) = 1.41 against a veto at 1.0, and only ITF's inflated station
sigmas kept it from firing on everything.
"""
import math
import random

import pytest

from pyroorb.itf.night_coherence import night_structure


def _line(y, m, d, frac=0.5, obscode="G96"):
    """An 80-column line whose only load-bearing part is its date."""
    head = "     K20A00A  C"                     # cols 1-15
    body = f"{y:04d} {m:02d} {d:02d}{frac:.5f}"[:17]
    return (head + body).ljust(56) + " " * 21 + obscode


class FakeFit:
    """The three things :func:`night_structure` asks a ``FittedOrbit`` for."""

    def __init__(self, rows):
        # rows: (line, d_ra, d_dec, sigma_ra, sigma_dec)
        self._obs = [{"obs80_line": r[0]} for r in rows]
        self._rows = [{"d_ra_arcsec": r[1], "d_dec_arcsec": r[2],
                       "sigma_ra": r[3], "sigma_dec": r[4]} for r in rows]

    def residuals(self, included_only=False):
        return list(self._rows)

    def is_included(self):
        return [True] * len(self._rows)


def _perfect_arc(n_nights, per_night, sigma_ra=0.4, sigma_dec=0.4, seed=7,
                 start_day=1):
    """Residuals drawn at EXACTLY the stated per-axis sigmas, no per-night
    structure: the arc a correct orbit produces."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_nights):
        day = start_day + 2 * i                  # unambiguously separate nights
        y, m, d = 2020, 1 + day // 28, 1 + day % 28
        for j in range(per_night):
            rows.append((_line(y, m, d, 0.1 + 0.001 * j),
                         rng.gauss(0, sigma_ra), rng.gauss(0, sigma_dec),
                         sigma_ra, sigma_dec))
    return FakeFit(rows)


# --------------------------------------------------------------- the scale
def test_a_perfect_arc_scores_one_not_root_two():
    """THE regression.  Residuals at exactly their stated sigmas must read 1.0,
    because 1.0 is what both thresholds are phrased against.  The old statistic
    read 1.41 here -- a 2-D radial over a 1-axis sigma -- and passed only
    because ITF's ~1.0" station floor sits over ~0.5" of real scatter, one error
    cancelling the other."""
    vals = [night_structure(_perfect_arc(6, 4, seed=s))["rms_sigma"]
            for s in range(40)]
    mean = sum(vals) / len(vals)
    assert 0.92 <= mean <= 1.08, mean
    # ...and emphatically NOT the sqrt(2) the old scale produced
    assert mean < 1.2


def test_the_gate_is_mean_one_whatever_the_arc_looks_like():
    """``night_coherence`` divides out BOTH shape terms -- observations per
    night (standard error) and number of nights (expected maximum) -- so one
    threshold means the same thing on a 4-night ITF-to-ITF pair-arc and on a
    19-night five-opposition recovery.  Without the second correction the long
    arc is penalised for being long, which is how the clause first came to veto
    856215."""
    def mean_gate(n_nights, per_night):
        v = [night_structure(_perfect_arc(n_nights, per_night, seed=s))
             ["night_coherence"] for s in range(60)]
        return sum(v) / len(v)

    short, long_ = mean_gate(4, 2), mean_gate(19, 5)
    assert 0.8 <= short <= 1.25, short
    assert 0.8 <= long_ <= 1.25, long_
    # the LENGTH BIAS is what the correction exists to remove
    assert abs(long_ - short) < 0.25, (short, long_)


def test_the_second_axis_is_normalised_by_its_own_sigma():
    """``sigma_dec`` was never read: a Dec-only misfit was divided by the RA
    uncertainty, so an arc with honest but unequal axes scored wrong in both
    directions."""
    fit = _perfect_arc(6, 4, sigma_ra=0.2, sigma_dec=1.2, seed=3)
    st = night_structure(fit)
    assert 0.85 <= st["rms_sigma"] <= 1.15, st["rms_sigma"]


def test_a_coherently_displaced_night_still_shows_up():
    """The statistic must stay SENSITIVE: one night pushed 4 sigma off, the rest
    perfect, has to clear the veto."""
    rows = list(_perfect_arc(6, 4, seed=11)._rows)
    lines = [o["obs80_line"] for o in _perfect_arc(6, 4, seed=11)._obs]
    out = []
    for i, (line, r) in enumerate(zip(lines, rows)):
        push = 1.6 if i < 4 else 0.0             # night 0, coherently in RA
        out.append((line, r["d_ra_arcsec"] + push, r["d_dec_arcsec"],
                    r["sigma_ra"], r["sigma_dec"]))
    st = night_structure(FakeFit(out))
    assert st["night_coherence"] > 2.0, st


# --------------------------------------------------------------- the nights
def test_a_night_crossing_ut_midnight_is_ONE_night():
    """Nights are a window (itf.opp_profile.nights_of), not a UT calendar date.  Keying
    on the date string split every western-hemisphere night in two: each half
    with a smaller n_obs (bigger standard error) and the night COUNT inflated
    (bigger max-correction), moving both terms of the statistic for a purely
    bookkeeping reason."""
    rows = [(_line(2020, 1, 5, 0.95), 0.1, 0.1, 0.4, 0.4),
            (_line(2020, 1, 5, 0.98), 0.1, 0.1, 0.4, 0.4),
            (_line(2020, 1, 6, 0.02), 0.1, 0.1, 0.4, 0.4),
            (_line(2020, 1, 6, 0.05), 0.1, 0.1, 0.4, 0.4)]
    st = night_structure(FakeFit(rows))
    assert st["n_nights"] == 1, st
    assert st["n_obs"] == 4


def test_two_real_nights_stay_two():
    rows = [(_line(2020, 1, 5, 0.2), 0.1, 0.1, 0.4, 0.4),
            (_line(2020, 1, 5, 0.3), 0.1, 0.1, 0.4, 0.4),
            (_line(2020, 1, 6, 0.2), 0.1, 0.1, 0.4, 0.4),
            (_line(2020, 1, 6, 0.3), 0.1, 0.1, 0.4, 0.4)]
    st = night_structure(FakeFit(rows))
    assert st["n_nights"] == 2 and st["evidence_dof"] == -2


def test_nothing_measurable_is_reported_as_nothing():
    assert night_structure(FakeFit([]))["n_nights"] == 0
    assert night_structure(object())["n_nights"] == 0


def test_a_missing_sigma_falls_back_without_pretending():
    """A zero or absent sigma used to become 1.0 silently; it still falls back,
    but to a named constant and per axis."""
    from pyroorb.itf.night_coherence import DEFAULT_SIGMA_ARCSEC

    rows = [(_line(2020, 1, 5, 0.2), 1.0, 1.0, 0, None),
            (_line(2020, 1, 7, 0.2), -1.0, -1.0, None, None)]
    st = night_structure(FakeFit(rows))
    assert st["sigma_arcsec"] == pytest.approx(DEFAULT_SIGMA_ARCSEC)
    assert st["rms_sigma"] == pytest.approx(1.0)


def test_the_expected_maximum_is_the_harmonic_number():
    """Documented because the divisor is the whole reason a threshold can be
    shared between arcs of different length."""
    from pyroorb.itf.night_coherence import _expected_max

    assert _expected_max(1) == pytest.approx(1.0)
    assert _expected_max(4) == pytest.approx(math.sqrt(1 + 1 / 2 + 1 / 3 + 1 / 4))
