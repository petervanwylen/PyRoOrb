"""hyperbolic's two rules on the block-model statistic, and the per-observation
statistic they retired, kept as a diagnostic.

The rules are stated as significance levels: rule 1 refuses a bound orbit "at
3 sigma" (Delta chi2 >= 9 for the one restriction), rule 2's escape at dBIC >= 15. The
statistic they are applied to is the Delta chi2 between the free and the capped fit each
re-optimised under the station-night error model (nightmodel.py), which IS chi2(1) under that
model -- so the bars apply as written on any arc and nothing is converted for the page. The
numbers below are what real arcs measured through `fit_object._prefer_bound_ecc` on 2026-09-26.
"""
import math

import pytest

import pyroorb.hyperbolic as hyp

MARGIN = 15.0


def test_oumuamua_on_two_nights_refuses_the_bound_orbit_under_the_model():
    """1I, 2017-10-18/19: 18 obs, six station-nights. Free 0.47", bound 0.97". Block Delta chi2
    57.4 with the arc's own scatter setting the scale on 16.9 dof (k 2.2x its stations), so an
    F(1, 16.9) variate: the one-dof equivalent is 24.4 (4.9 sigma); n_eff 31.7 -> dBIC 20.9:
    damaged AND refused, so the night bar's escape opens on two nights -- 'the data
    refuse the bound orbit outright'."""
    damaged, evid = hyp.block_damaged(57.36, scale_dof=16.9)
    assert damaged and evid == pytest.approx(24.38, abs=0.05)
    refused, dbic = hyp.block_refused(57.36, 31.7, MARGIN, scale_dof=16.9)
    assert refused and dbic == pytest.approx(24.38 - math.log(31.7), abs=0.05)
    # with no per-fit scale the same statistic is the chi2(1) itself
    assert hyp.block_damaged(57.36) == (True, pytest.approx(57.36))


def test_lp68_on_two_nights_does_not_refuse_its_own_orbit():
    """2010 LP68's first two nights, 17 obs: block Delta chi2 3.48 on 30.9 scale dof (one-dof
    equivalent 3.25, 1.8 sigma) where the per-observation statistic read 97 -- its OWN 36 au
    orbit refused at 9.9 sigma."""
    damaged, evid = hyp.block_damaged(3.483, scale_dof=30.9)
    assert not damaged and evid == pytest.approx(3.25, abs=0.03)
    refused, dbic = hyp.block_refused(3.483, 32.6, MARGIN, scale_dof=30.9)
    assert not refused and dbic == pytest.approx(3.25 - math.log(32.6), abs=0.03)
    assert hyp.bound_dchi2(1.9671, 8.7912, 34) == pytest.approx(97.1, abs=0.5)   # the retired reading


def test_an_estimated_scale_makes_the_statistic_an_f_and_the_verdict_still_reads_as_sigma():
    """damaged iff sqrt(evidence) >= 3 on ANY scale dof; the F bar is above 9 and falls to it as the
    scale becomes known (Student's correction, with the dof the scale was estimated on)."""
    for dof in (4.9, 8.0, 16.9, 30.9, 200.0):
        bar = hyp._f_bar(hyp.BOUND_DAMAGE_DCHI2, dof)
        assert bar > hyp.BOUND_DAMAGE_DCHI2
        assert hyp._equivalent_chi2(bar, dof) == pytest.approx(hyp.BOUND_DAMAGE_DCHI2, rel=1e-9)
        for d in (1.0, bar * 0.999, bar * 1.001, 40.0, 1e6):
            damaged, evid = hyp.block_damaged(d, scale_dof=dof)
            assert damaged == (math.sqrt(evid) >= math.sqrt(hyp.BOUND_DAMAGE_DCHI2))
            refused, dbic = hyp.block_refused(d, 12.0, MARGIN, scale_dof=dof)
            assert refused == (dbic >= MARGIN)
    assert hyp._f_bar(9.0, 20.0) == pytest.approx(11.71, abs=0.05)
    assert hyp._f_bar(9.0, 5.0) == pytest.approx(30.3, abs=0.5)
    assert hyp._f_bar(9.0, 5.0) > hyp._f_bar(9.0, 20.0) > hyp._f_bar(9.0, 200.0) > hyp.BOUND_DAMAGE_DCHI2
    assert hyp._f_bar(9.0, 1e6) == pytest.approx(9.0, rel=1e-3)


def test_the_bar_is_the_owners_three_sigma_exactly():
    assert hyp.block_damaged(9.0)[0] and not hyp.block_damaged(8.999)[0]
    assert hyp.block_damaged(9.01) == (True, pytest.approx(9.01))
    # rule 2: dbic = dchi2 - ln(n_eff), refused iff it reaches the margin
    for n_eff in (6.9, 13.1, 31.7, 400.0):
        bar = MARGIN + math.log(n_eff)
        assert hyp.block_refused(bar + 1e-6, n_eff, MARGIN)[0]
        assert not hyp.block_refused(bar - 1e-6, n_eff, MARGIN)[0]


def test_no_verdict_without_a_statistic():
    assert hyp.block_damaged(None) == (False, None)
    assert hyp.block_damaged(float("nan")) == (False, None)
    assert hyp.block_refused(None, 10.0, MARGIN) == (False, None)
    assert hyp.block_refused(20.0, 0.0, MARGIN) == (False, None)
    assert hyp.bound_dchi2(1.0, 50.0, 6) is None          # 3 obs, 6 params: no dof
    assert hyp.bound_dchi2(0.0, 50.0, 40) is None         # a perfect fit: no scale
    assert hyp.bound_dchi2(1.0, float("nan"), 40) is None


def test_the_per_observation_statistic_is_what_it_always_was():
    """`bound_dchi2` decides nothing now but old records carry it (`dchi2_raw`), so its
    numbers stay pinned: 1I's first 3 nights (16 obs, 0.21" free vs 0.39" bound) at 62.3 + ln 32,
    the 2I paste at 20.6, C/2017 K2's four tracklets at 0.49."""
    assert hyp.bound_dchi2(5.42, 19.1264, 32) == pytest.approx(62.3 + math.log(32), abs=0.2)
    assert hyp.bound_dchi2(32.934, 38.149, 136) == pytest.approx(20.6, abs=0.1)
    assert hyp.bound_dchi2(4.1959, 4.2885, 28) == pytest.approx(0.486, abs=0.01)


def test_the_reported_significance_saturates_and_says_that_it_did():
    """Past EVIDENCE_SIGMA_CAP the number is about the Gaussian tail, not the astrometry: the
    ceiling is reported as a floor, flagged, with the statistic kept beside it -- 1I's full arc
    reads Delta chi2 3.5e6 and 'at least 10 sigma', never 1868 sigma."""
    shown = hyp.reported_evidence(3.5e6)
    assert math.sqrt(shown) == pytest.approx(hyp.EVIDENCE_SIGMA_CAP)
    assert hyp.saturated_evidence(shown)
    assert not hyp.saturated_evidence(hyp.reported_evidence(9.0))
    assert hyp.reported_evidence(-0.3) == 0.0                 # a bound orbit that fits BETTER: no negative sigma
    # saturation can never cross the bar the boolean uses: the ceiling is far above it
    assert hyp.MAX_EVIDENCE_CHI2 > hyp.BOUND_DAMAGE_DCHI2
    assert hyp.block_damaged(3.5e6)[0]
