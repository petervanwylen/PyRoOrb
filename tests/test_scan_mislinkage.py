"""The mislinkage scanner (pyroorb.mislink.scan_mislinkage): what it measures about a
linkage's retained tracklets, and how it ranks suspects. Private: the scanner is not part
of the release."""
def _sm():
    from pyroorb.mislink import scan_mislinkage
    return scan_mislinkage


def test_drop_ratio_measures_the_retained_tracklets_only():
    """The drop statistic must compare the SAME observations under two orbits.

    A whole-arc ratio puts the suspect tracklet's own residuals in the numerator,
    so a tracklet from a noisy observatory scores as a foreign one, and it
    compares two different observation sets.  This pins the corrected shape:
    _subset_rms_norm reads only the observations it was asked for, and matches
    them by identity rather than by position -- the two fits hold different
    observation lists, so index k of one is not index k of the other.
    """
    sm = _sm()

    class _F:
        def __init__(self, obs, result):
            self._obs, self._res = obs, result
        def residuals(self):
            return self._res
        def is_included(self):
            return [True] * len(self._obs)

    def o(jd, code, ra, dec):
        return {"jd_utc": jd, "obscode": code, "ra_deg": ra, "dec_deg": dec}

    def r(dra, ddec):
        return {"d_ra_arcsec": dra, "d_dec_arcsec": ddec, "sigma_ra": 1.0, "sigma_dec": 1.0}

    a, b, c = o(1.0, "F51", 10.0, 5.0), o(2.0, "F51", 11.0, 5.0), o(3.0, "703", 12.0, 5.0)
    # `a` and `b` are the retained pair; `c` is the suspect and is WILDLY off.
    fit = _F([a, b, c], [r(1.0, 0.0), r(1.0, 0.0), r(50.0, 0.0)])
    wanted = {sm._obs_key(a), sm._obs_key(b)}
    rms, n = sm._subset_rms_norm(fit, wanted)
    assert n == 2, f"must read exactly the two retained obs, got {n}"
    # per-axis over 2 obs x 2 axes: sqrt((1+0+1+0)/4) = 0.707 -- the 50" suspect
    # residual must not appear anywhere in it
    assert abs(rms - 0.7071) < 1e-3, f"suspect's own residual leaked in: {rms}"

    # the same observations in a DIFFERENT list order must give the same answer
    shuffled = _F([c, b, a], [r(50.0, 0.0), r(1.0, 0.0), r(1.0, 0.0)])
    rms2, n2 = sm._subset_rms_norm(shuffled, wanted)
    assert (n2, round(rms2, 6)) == (n, round(rms, 6)), \
        "matching is positional, not by observation identity"




def test_dropobs_needs_concentration_and_a_walking_tracklet():
    """Two conditions, and the second is the one that matters.

    Concentration: a rejection anywhere is common; ALL of an arc's rejections in
    ONE tracklet is the shape a foreign tracklet makes.

    A walk: a rejection is a way of DISGUISING a trend, so it is evidence only
    when the tracklet it came from is actually walking.  One point off the line
    its own neighbours define is a blunder -- 2016 DA47's G96 tracklet trends 0.73
    sigma and loses an observation carrying no magnitude and a blank band, and it
    is not a mislinkage.  2023 QA131's trends 5.87 and is.
    """
    sm = _sm()
    walk, flat = 5.87, 0.73

    # concentrated AND walking -> fires
    e = [{"n_all": 3, "n_drop": 1, "ramp": walk}, {"n_all": 4, "n_drop": 0, "ramp": 1.76},
         {"n_all": 4, "n_drop": 0, "ramp": 0.86}]
    sm._dropobs(e)
    assert [x["dropobs"] for x in e] == [True, False, False], e

    # concentrated but NOT walking -> suppressed, and the record says why
    e = [{"n_all": 4, "n_drop": 1, "ramp": flat}, {"n_all": 4, "n_drop": 0, "ramp": 0.74}]
    sm._dropobs(e)
    assert not any(x["dropobs"] for x in e), f"a blunder must not fire: {e}"
    assert "does not walk" in e[0]["dropobs_why"], e[0]

    # walking but rejections SCATTERED across two tracklets -> neither fires
    e = [{"n_all": 4, "n_drop": 1, "ramp": walk}, {"n_all": 4, "n_drop": 1, "ramp": walk},
         {"n_all": 3, "n_drop": 0, "ramp": 0.5}]
    sm._dropobs(e)
    assert not any(x["dropobs"] for x in e), f"scattered rejections must not fire: {e}"

    # a clean arc -> nothing, and no division by zero on an empty tracklet
    e = [{"n_all": 4, "n_drop": 0, "ramp": walk}, {"n_all": 0, "n_drop": 0, "ramp": 0}]
    sm._dropobs(e)
    assert not any(x["dropobs"] for x in e)
    assert e[1]["drop_frac"] == 0.0


def test_sparsity_is_a_second_route_into_the_top_tier():
    """A thin arc admits a flagged tracklet that is NOT isolated -- and only the
    scope changes, never the evidence: an unflagged tracklet stays rank 0."""
    sm = _sm()
    flagged = {"iso": 0.3, "flags": [("mag", 1, "MAG dH=1.10 p=0.02")]}
    assert sm._rank(flagged, sparse=False) == 1, "not isolated, not sparse -> advisory"
    assert sm._rank(flagged, sparse=True) == 3, "sparse arc -> suspect"
    isolated = {"iso": 9.0, "flags": [("mag", 1, "MAG dH=1.10 p=0.02")]}
    assert sm._rank(isolated, sparse=False) == 3, "isolated alone is enough"
    clean = {"iso": 0.3, "flags": []}
    assert sm._rank(clean, sparse=True) == 0, "sparsity must never create evidence"


def test_anomaly_is_additive_and_accumulates_near_misses():
    """The point of the score: two signals just UNDER their thresholds must beat
    one signal just under a threshold.  The gate structure scored both at zero.

    Modelled on 2021 CE57, which carries dH 0.79 against a flag of 1.0 and
    drop_ratio 2.29 -- partial evidence on two independent lines.
    """
    sm = _sm()
    both = {"dH": 0.79, "p": 0.044, "drop_ratio": 2.29}
    one = {"dH": 0.79, "p": 0.044}
    s_both, parts = sm._anomaly(both)
    s_one, _ = sm._anomaly(one)
    assert s_both > s_one, "two partial lines must outscore one"
    assert set(parts) == {"mag", "drop"}, parts
    assert abs(parts["mag"] - 0.79) < 0.01, parts       # 1.0 == the flag threshold
    # each contribution is zero at its own NULL, so the sum means something
    assert sm._anomaly({"drop_ratio": 1.0})[0] == 0.0
    assert sm._anomaly({"z": 0.0})[0] == 0.0
    # an unresolved magnitude gap is NOT weak evidence, it is none
    assert sm._anomaly({"dH": 2.5, "p": 0.40})[0] == 0.0, "the p gate is a prerequisite"
    # and no single line may dominate the rest combined
    assert sm._anomaly({"dH": 8.93, "p": 1e-9})[0] == sm.ANOM_CAP


def test_anomaly_never_counts_scope_as_evidence():
    """iso and sparsity decide whether a tracklet is one this scan can JUDGE.
    Summing them in would let a wide gap manufacture evidence out of nothing."""
    sm = _sm()
    assert sm._anomaly({"iso": 60.0})[0] == 0.0
    # ...and an unflagged tracklet in scope still ranks 0, however isolated
    assert sm._rank({"iso": 60.0, "flags": [], "anomaly": 0.0}, sparse=True) == 0
