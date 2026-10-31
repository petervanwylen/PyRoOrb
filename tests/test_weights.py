"""Unit tests for per-obs uncertainty assignment (weights.sigma_for).

Guards the clamp on reported ADES rmsRA/rmsDec: a credible measurement passes
through, an over-tight value is floored, and an absurd value (a discovery
tracklet self-reporting 72") is CAPPED so it can't shield a gross error from
outlier rejection -- the C/2025 R2 fix.
"""
import pyroorb.weights as w


def test_ades_rms_passthrough_and_clamp():
    # a sane reported rms is used as-is
    assert w.sigma_for("XXX", None, 0.8, 1.2) == (0.8, 1.2)
    # over-tight rms is floored
    sra, sdec = w.sigma_for("XXX", None, 0.01, 0.01)
    assert sra == w.SIGMA_FLOOR and sdec == w.SIGMA_FLOOR
    # absurd rms (e.g. C/2025 R2's 72") is capped at the ceiling, NOT trusted
    sra, sdec = w.sigma_for("C49", None, 72.0, 72.0)
    assert sra == w.SIGMA_CEILING and sdec == w.SIGMA_CEILING
    assert w.SIGMA_FLOOR < w.SIGMA_CEILING


def test_station_fallback_when_no_rms():
    # no reported rms -> Veres-2017 station lookup (equal RA/Dec), within the clamp
    s_ra, s_dec = w.sigma_for("F51")          # Pan-STARRS 1, a listed station
    assert s_ra == s_dec
    assert w.SIGMA_FLOOR <= s_ra <= w.SIGMA_CEILING
    # an unlisted station gets the catch-all default
    s_ra, _ = w.sigma_for("ZZZ_unlisted")
    assert s_ra == min(max(w.DEFAULT_SIGMA, w.SIGMA_FLOOR), w.SIGMA_CEILING)
