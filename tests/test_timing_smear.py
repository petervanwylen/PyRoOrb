"""Offline tests for the timing-error smear (weights.apparent_sky_motion /
timing_inflated_sigmas), find_orb's errors.cpp model: a Gaussian timing error σ_t
smears the measured position along the direction of motion by (sky rate)·σ_t,
combined in quadrature with the astrometric σ.

The feature is OPT-IN (PYROORB_TIME_SIGMA_SEC; default off) because, applied globally, it
shifts the σ-normalized residuals the bridge/rejection thresholds are tuned against and
destabilises the hardest linkage objects (2023 VM6). These tests exercise the math
directly so the machinery stays covered regardless of the suite default.
"""
import sys, os
import numpy as np
import pyroorb.weights as w


def test_sky_motion_from_tracklet():
    """Two same-station obs 30 min apart, object drifting +1000"/day in RA only:
    the estimated rate recovers ~1000"/day in RA, ~0 in Dec."""
    rate = 1000.0  # arcsec/day in RA (already cross-track, i.e. ×cosδ)
    dec = 20.0
    dt = 0.5 / 24
    ra0 = 10.0
    ra1 = ra0 + (rate * dt / 3600.0) / np.cos(np.radians(dec))  # undo cosδ for sky RA
    jds = [2460000.0, 2460000.0 + dt]
    vra, vdec = w.apparent_sky_motion(jds, [ra0, ra1], [dec, dec], ["G96", "G96"])
    assert abs(vra[0] - rate) < 1.0, f"v_RA {vra[0]:.2f} != {rate}"
    assert abs(vdec[0]) < 1.0, f"v_Dec {vdec[0]:.2f} should be ~0"


def test_lone_obs_has_no_motion():
    """A single obs (no same-station neighbour in-window) gets zero rate -> no
    timing inflation, the conservative fallback."""
    vra, vdec = w.apparent_sky_motion([2460000.0], [10.0], [20.0], ["G96"])
    assert vra[0] == 0.0 and vdec[0] == 0.0


def test_neighbour_beyond_window_ignored():
    """Two obs a full day apart exceed _MOTION_WINDOW_DAYS -> treated as lone."""
    jds = [2460000.0, 2460001.0]
    vra, _ = w.apparent_sky_motion(jds, [10.0, 11.0], [20.0, 20.0], ["G96", "G96"])
    assert vra[0] == 0.0 and vra[1] == 0.0


def test_findorb_worked_example():
    """find_orb/errors.cpp:36-46: astrometry good to 0.7", timing to 6 s, object
    moving 0.4"/s -> 2.4" along-track smear, combined to sqrt(0.7^2+2.4^2)=2.5"."""
    rate_per_day = 0.4 * 86400.0          # 0.4"/s -> "/day
    sig_t_days = 6.0 / 86400.0            # 6 s
    sra = np.array([0.7]); sdec = np.array([0.7])
    eff_ra, _ = w.timing_inflated_sigmas(sra, sdec, [rate_per_day], [0.0], [sig_t_days])
    assert abs(eff_ra[0] - 2.5) < 0.05, f"inflated σ {eff_ra[0]:.3f} != 2.5"


def test_inflation_only_grows_sigma():
    """The smear can only increase σ (never shrink below the astrometric floor),
    and a zero rate or zero σ_t leaves σ untouched."""
    sra = np.array([0.5, 0.5]); sdec = np.array([0.5, 0.5])
    eff_ra, eff_dec = w.timing_inflated_sigmas(sra, sdec, [0.0, 5000.0], [0.0, 0.0],
                                               [1.0 / 86400.0, 0.0])
    assert eff_ra[0] == 0.5 and eff_dec[0] == 0.5      # zero rate -> unchanged
    assert eff_ra[1] == 0.5                            # zero σ_t  -> unchanged
    assert np.all(np.hypot(eff_ra, eff_dec) >= np.hypot(sra, sdec) - 1e-12)
