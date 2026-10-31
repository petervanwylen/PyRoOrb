"""Offline tests for the non-gravitational model layer (nongrav.py + refine_nongrav).

Validates the find_orb-style "turn on A1/A2/..." refinement against cached fixtures,
with NO network:
  - the model registry / parameter packing is self-consistent;
  - 'none' is a true no-op (the gravity orbit is returned unchanged);
  - the Yarkovsky A2 fit on (101955) Bennu recovers the right SIGN at high
    significance and does not worsen the normalized rms (a real detection);
  - the comet A1/A2/A3 fit on P/2023 JN16 returns the requested parameters, the
    normal term A3 is consistent with zero, and the rms does not worsen.

Requires the ASSIST backend (the nongrav force lives there); skips gracefully if its
ephemeris is absent.
"""
import os
import types

import numpy as np
import pytest
from _fixtures import load_fixture, FIX
import pyroorb.fit_object as fo
from pyroorb import obs_sources
import pyroorb.nongrav as ng

# every test here refits a cached fixture through the full engine
pytestmark = pytest.mark.slow

_OC = None
def _oc():
    global _OC
    if _OC is None:
        _OC = obs_sources.load_obscodes()
    return _OC


_FX = {}          # key -> parsed fixture
_FX_NAMES = []    # the fixtures/ listing, read once


def _fixture(key):
    """Load a fixture by substring of its filename, MEMOISED.

    This used to glob all ~180 files in fixtures/ on every one of its ~20 call
    sites, and re-read the matching JSON each time. The fixtures are frozen for
    the length of a run, so once is enough."""
    if key not in _FX:
        if not _FX_NAMES:
            _FX_NAMES.extend(sorted(n for n in os.listdir(FIX) if n.endswith(".json")))
        _FX[key] = load_fixture(os.path.join(FIX,
                                             next(n for n in _FX_NAMES if key in n)))
    return _FX[key]


def _fit(key):
    fx = _fixture(key)
    return fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])


def _assist_available():
    backend, _ = obs_sources._force_backend()
    return hasattr(backend, "predict_radec_jac_ng")


def test_registry_and_packing():
    """The model registry maps to the right free A axes and full_A places the scaled
    free values onto the correct physical A1/A2/A3 slots (fixed axes zero)."""
    assert ng.free_indices("comet2") == [0, 1]            # A1, A2
    assert ng.free_indices("comet3") == [0, 1, 2]         # A1, A2, A3
    assert ng.free_indices("yarkovsky") == [1]            # A2 only
    A = ng.full_A("yarkovsky", [3.0])                     # 3.0 in units of scale=1e-14
    assert A[0] == 0.0 and A[2] == 0.0
    assert abs(A[1] - 3.0 * 1e-14) < 1e-28
    try:
        ng.model("nope"); assert False, "unknown model should raise"
    except ValueError:
        pass


def test_none_is_noop():
    """Refining with model 'none' returns the very same gravity fit object."""
    fit = _fit("P_2023_JN16")
    assert fo.refine_nongrav(fit, "none") is fit


def test_yarkovsky_bennu():
    """(101955) Bennu has a strong, negative Yarkovsky A2. The fit must recover the
    sign at high significance and not worsen the normalized rms."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fit = _fit("Bennu_101955")
    f2 = fo.refine_nongrav(fit, "yarkovsky")
    assert f2.nongrav_model == "yarkovsky", "nongrav model should be attached"
    A2 = f2.nongrav_params["A2"]; sA2 = f2.nongrav_sigmas["A2"]
    assert A2 < 0, f"Bennu A2 should be negative (drag), got {A2:.3e}"
    assert abs(A2) / sA2 > 3.0, f"A2 only {abs(A2)/sA2:.1f}sigma -- expected a clear detection"
    assert f2.rms_normalized() <= fit.rms_normalized() + 1e-9, "nongrav should not worsen rms"


def test_comet_models():
    """The comet models return their A parameters; the normal term A3 is consistent
    with zero on this weakly-active body and adding it does not move A1/A2 much."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fit = _fit("P_2023_JN16")
    f2 = fo.refine_nongrav(fit, "comet2")
    assert set(f2.nongrav_params) == {"A1", "A2"}
    assert f2.rms_normalized() <= fit.rms_normalized() + 1e-9
    f3 = fo.refine_nongrav(fit, "comet3")
    assert set(f3.nongrav_params) == {"A1", "A2", "A3"}
    # A3 (normal) should be statistically indistinguishable from zero here ...
    assert abs(f3.nongrav_params["A3"]) < 5.0 * f3.nongrav_sigmas["A3"], "A3 spuriously large"
    # ... and the in-plane terms should be stable whether or not A3 is solved.
    assert abs(f3.nongrav_params["A1"] - f2.nongrav_params["A1"]) < 3.0 * f2.nongrav_sigmas["A1"]


def test_discover_choice():
    """The model-selection rule (fit_object._ng_choice): a model is justified iff its weighted-
    chi2 reduction >= NONGRAV_CHI2_GAIN (10%) AND a significant nested F-test AND plausibility.
    A PHYSICAL PRIOR then governs the pick: Yarkovsky (A2) wins whenever justified (the secular
    trend is its signature here); the radial-SRP A1 / A1+A2 models are only a fallback when A2 is
    not justified. `chi2` is the weighted chi2; `chi2_gain` its fractional reduction vs a gravity
    baseline of chi2=100."""
    # F=100 (sqrt=10) clears the NONGRAV_MIN_Z significance cut for the F-test-significant cases.
    # a1z is the radial term's MARGINAL z in the joint (srp2) fit -- only relevant to the
    # yarkovsky->srp2 upgrade gate; defaults low so plain cases keep the parsimonious A2-only.
    M = lambda chi2, ft=True, phys=True, a1z=0.0, rms=0.50: {
        "chi2": chi2, "chi2_gain": (100.0 - chi2) / 100.0, "ftest_ok": ft, "phys_ok": phys,
        "F": 100.0 if ft else 1.0, "z": {"A1": a1z}, "rms": rms}
    # A2 is a real, significant, plausible win -> pick yarkovsky
    assert fo._ng_choice({"srp": M(98), "yarkovsky": M(33),
                          "srp2": M(33, ft=False)}) == "yarkovsky"
    # a real effect (8%) but NOT significant by the F-test -> nothing justified
    assert fo._ng_choice({"srp": M(100), "yarkovsky": M(92, ft=False),
                          "srp2": M(92, ft=False)}) is None
    # a big A2 win but physically implausible -> vetoed
    assert fo._ng_choice({"srp": M(100), "yarkovsky": M(40, phys=False),
                          "srp2": M(40, ft=False)}) is None
    # PHYSICAL PRIOR: A2 justified -> yarkovsky wins even when A1+A2 has a lower chi2, AS LONG AS the
    # radial A1 is not independently determined (a degenerate A1 must not steal the pick -- 2008 DB:
    # A1 marginal z~4.3 < NONGRAV_SRP2_A1_Z).
    assert fo._ng_choice({"srp": M(70), "yarkovsky": M(45),
                          "srp2": M(35, a1z=4.3)}) == "yarkovsky"
    # ...but UPGRADE to A1+A2 when A1 IS independently, strongly significant AND srp2 decisively
    # beats A2-only on chi2 -- a genuine SRP+Yarkovsky object (2009 BD: A1 z~11). Both terms reported.
    assert fo._ng_choice({"srp": M(60), "yarkovsky": M(45),
                          "srp2": M(33, a1z=11.0)}) == "srp2"
    # high A1 marginal z but srp2 does NOT decisively beat A2-only -> stay yarkovsky (no real gain)
    assert fo._ng_choice({"srp": M(60), "yarkovsky": M(45),
                          "srp2": M(44, a1z=11.0)}) == "yarkovsky"
    # A2 NOT justified but A1+A2 beats A1 decisively -> fall back to srp2
    assert fo._ng_choice({"srp": M(70), "yarkovsky": M(98),
                          "srp2": M(50)}) == "srp2"
    # A2 NOT justified, only A1 stands on its own -> srp
    assert fo._ng_choice({"srp": M(60), "yarkovsky": M(98),
                          "srp2": M(58, ft=False)}) == "srp"
    # significant by F-test but a tiny effect size below the 10% bar (large-N trap) -> rejected
    assert fo._ng_choice({"srp": M(99), "yarkovsky": M(98),
                          "srp2": M(97.5)}) is None

    # --- the srp3 upgrade (2009 HC: an out-of-plane A3 no 2-axis model can express).
    # Same shape as the srp2 upgrade: BIC/floors-justified AND decisively beats the pick AND the
    # NEW axis is independently determined (marginal z >= NONGRAV_SRP2_A1_Z).
    # rms defaults give srp3 a 20% raw win over the 0.50" picks -- the raw-rms visibility floor
    # applied to the UPGRADE comparison is part of the clause (see _srp3_upgrade).
    M3 = lambda chi2, a3z=0.0, a1z=0.0, ft=True, phys=True, rms=0.40: {
        "chi2": chi2, "chi2_gain": (100.0 - chi2) / 100.0, "ftest_ok": ft, "phys_ok": phys,
        "F": 100.0 if ft else 1.0, "z": {"A1": a1z, "A2": 8.0, "A3": a3z}, "rms": rms}
    # A2-only picked, srp3 decisively better with a strongly-determined A3 -> upgrade
    assert fo._ng_choice({"srp": M(70), "yarkovsky": M(45), "srp2": M(44, a1z=2.0),
                          "srp3": M3(20, a3z=35.0)}) == "srp3"
    # ...A3 not independently determined (a degenerate axis buying joint chi2) -> stay A2-only
    assert fo._ng_choice({"srp": M(70), "yarkovsky": M(45), "srp2": M(44, a1z=2.0),
                          "srp3": M3(20, a3z=4.0)}) == "yarkovsky"
    # ...determined A3 but no decisive chi2 win over the pick -> stay A2-only
    assert fo._ng_choice({"srp": M(70), "yarkovsky": M(45), "srp2": M(44, a1z=2.0),
                          "srp3": M3(43, a3z=35.0)}) == "yarkovsky"
    # the srp2 pick upgrades by the same rule (2-axis object with a real third axis)
    assert fo._ng_choice({"srp": M(60), "yarkovsky": M(45), "srp2": M(33, a1z=11.0),
                          "srp3": M3(25, a3z=20.0)}) == "srp3"
    # ...but a determined, decisive srp3 that does NOT visibly improve the pick's raw rms is
    # refused -- the 2007 EG shape: weighted chi2 and a formal z are buyable by systematics,
    # the raw-rms visibility floor is not (same constant, applied to the upgrade comparison)
    assert fo._ng_choice({"srp": M(70), "yarkovsky": M(45), "srp2": M(44, a1z=2.0),
                          "srp3": M3(20, a3z=35.0, rms=0.49)}) == "yarkovsky"
    # srp3 never stands alone: nothing else justified -> None even with a great srp3
    assert fo._ng_choice({"srp": M(98), "yarkovsky": M(97), "srp2": M(96, ft=False),
                          "srp3": M3(20, a3z=35.0)}) is None


def test_srp3_strain_gate():
    """_srp3_continue: the srp3 step is fitted only when the best 1-2 axis trial is still
    sigma-strained (normalized rms above NONGRAV_SRP3_STRAIN) -- zero discovery cost on the
    ~99% of objects a smaller model already explains."""
    def out(best_chi2, N=200, chi2_0=1e9):
        return {"models": {"yarkovsky": {"chi2": best_chi2, "fit": object()}},
                "chi2_0": chi2_0, "N": N}
    # every step before srp3 always continues
    assert fo._srp3_continue(out(1e9), "srp") is True
    assert fo._srp3_continue(out(1e9), "yarkovsky") is True
    # best model comfortable (rms_n 0.6) -> srp3 skipped
    assert fo._srp3_continue(out(0.6 ** 2 * 200), "srp2") is False
    # best model strained (rms_n 0.95, the 2009 HC shape) -> srp3 fitted
    assert fo._srp3_continue(out(0.95 ** 2 * 200), "srp2") is True
    # a well-fitting GRAVITY baseline also stands down the step
    assert fo._srp3_continue({"models": {}, "chi2_0": 0.5 ** 2 * 200, "N": 200}, "srp2") is False


def test_discover_yarkovsky_bn4():
    """End-to-end automatic discovery on 2003 BN4 -- a 3-opposition, a~1.27 au NEO whose
    gravity fit ejects a whole apparition. discover_nongrav must pick 'yarkovsky', redeem
    the stranded obs, and land the full arc under 0.55\"."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2003_BN4")
    # Start from a PURE gravity fit (discovery is on by default now) so the routine has the
    # systematic-trend gravity orbit to work from, exactly as it does inside fit_orbit.
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                       auto_nongrav=False)
    out, info = fo.discover_nongrav(fit)
    assert info["model"] == "yarkovsky", f"picked {info.get('model')} ({info.get('reason')})"
    assert out.rms_residual() < 0.55, f"rms {out.rms_residual():.3f}\""
    assert out.nongrav_params["A2"] < 0, "BN4 A2 should be negative"
    assert ng.a2_is_plausible(out.nongrav_params["A2"]), "A2 outside the physical Yarkovsky range"
    # the routine should have redeemed the apparition gravity had to drop (fuller arc)
    assert out.n_observations() > fit.n_observations(), "expected more obs included with the nongrav"


def test_rescue_gravity_only_no_nongrav_leak():
    """2010 VQ -- a 3-opp a=0.86 Aten with a Yarkovsky signal that also trips the single-night
    poison rescue. Guards a fixed crash: _rescue_single_night_poison used to re-fit with the
    nongrav discovery default-ON, so its internal structural re-fit returned a 7-element nongrav
    state. That state then (a) LEAKED out of a fit_orbit(auto_nongrav=False) caller and (b) was
    seeded into _full_fit_excluding's GRAVITY evaluator -- a (6,)/(7,) Jacobian shape crash. The
    rescue is now gravity-only; the single nongrav decision is made once at the end of fit_orbit."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2010_VQ")
    # (a) auto_nongrav=False must yield a clean 6-element gravity state -- no leaked nongrav.
    grav = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                        auto_nongrav=False)
    assert len(np.asarray(grav._state, float)) == 6, "gravity baseline leaked a nongrav state"
    assert not grav.nongrav_params, f"unexpected nongrav on gravity fit: {grav.nongrav_params}"
    # (b) discover_nongrav on that gravity baseline must NOT crash and should pick the Yarkovsky.
    out, info = fo.discover_nongrav(grav)
    assert info["model"] == "yarkovsky", f"picked {info.get('model')} ({info.get('reason')})"
    assert out.nongrav_params["A2"] < 0, "2010 VQ A2 should be negative"


def test_more_yarkovsky():
    """Broaden the auto-Yarkovsky population guard: a spread of a<1.6 NEOs for which JPL fits a
    transverse A2 and our discover_nongrav must independently pick 'yarkovsky' with an A2 of the
    right sign and within a few tens of percent of JPL's published value. Complements the
    single-object acceptance tests (BN4/CL50/OS5/DB) with several more objects across the range."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    JPL_A2 = {  # JPL SBDB published transverse non-grav (au/day^2)
        "2004_JN1": -3.80e-13, "2015_SD": -3.80e-13,
        "2007_EG":  -4.30e-13, "2020_QJ1": -3.70e-13,
    }
    # 2007_EG delivers the FULLER family since the strain-gated srp3 step: its
    # A2-only fit was sigma-strained (rms 0.887", golden status SUSPECT) with an out-of-plane
    # term no 2-axis model can express -- srp3 halves the raw rms to 0.417" (retest chi2
    # 351 -> 130, A3 marginal z ~20, 55% visible gain, every upgrade clause cleared by a
    # mile). JPL's published value is the A2-only projection; our A2 still matches its sign
    # and ratio, which is what this test guards, and the determined new axis is asserted.
    EXPECT = {"2007_EG": "srp3"}
    for key, a2_jpl in JPL_A2.items():
        fx = _fixture(key)
        grav = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                            auto_nongrav=False)
        out, info = fo.discover_nongrav(grav)
        want = EXPECT.get(key, "yarkovsky")
        assert info["model"] == want, f"{key}: picked {info.get('model')} ({info.get('reason')})"
        a2 = out.nongrav_params["A2"]
        assert (a2 < 0) == (a2_jpl < 0), f"{key}: A2 sign {a2:+.2e} vs JPL {a2_jpl:+.2e}"
        assert ng.a2_is_plausible(a2), f"{key}: A2 {a2:+.2e} outside the physical Yarkovsky range"
        assert 0.5 <= a2 / a2_jpl <= 2.0, f"{key}: A2 {a2:+.2e} vs JPL {a2_jpl:+.2e} (ratio off)"
        if want == "srp3":
            z3 = abs(out.nongrav_params["A3"]) / out.nongrav_sigmas["A3"]
            assert z3 >= fo.NONGRAV_SRP2_A1_Z, f"{key}: srp3's A3 not independently determined"


def test_discover_srp2_2009bd():
    """2009 BD -- a tiny (H=28) 2-opp NEO where JPL fits BOTH radial SRP A1 and transverse A2, and
    BOTH are present in the optical astrometry. discover_nongrav must report the A1+A2 'srp2' model
    rather than the parsimonious A2-only: the radial A1 is independently significant (marginal
    z~11, well over NONGRAV_SRP2_A1_Z) and srp2 cuts chi2 ~26%% below A2-only. This guards the
    yarkovsky->srp2 upgrade -- distinct from 2008 DB, whose degenerate A1 (z~4.3) must stay
    yarkovsky. Both coefficients should land near JPL (A1=+5.6e-11, A2=-1.1e-12)."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2009_BD")
    grav = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                        auto_nongrav=False)
    out, info = fo.discover_nongrav(grav)
    assert info["model"] == "srp2", f"picked {info.get('model')} ({info.get('reason')})"
    a1, a2 = out.nongrav_params["A1"], out.nongrav_params["A2"]
    assert a1 > 0 and a2 < 0, f"2009 BD signs off: A1={a1:+.2e} A2={a2:+.2e}"
    assert 0.5 <= a1 / 5.60e-11 <= 2.0, f"2009 BD A1 {a1:+.2e} vs JPL +5.6e-11 (ratio off)"
    assert 0.5 <= a2 / -1.10e-12 <= 2.0, f"2009 BD A2 {a2:+.2e} vs JPL -1.1e-12 (ratio off)"


def test_discover_yarkovsky_cl50():
    """Automatic discovery on 2011 CL50 -- a tiny (H=27.3, ~12 m) 2-opp Aten whose Yarkovsky
    is LARGE but physical (JPL A2=+4.4e-12). It exercises the SIZE-AWARE A2 ceiling: the value
    is ~4x the old flat 1e-12 bound, so discovery used to see the 45%% rms-cut trend yet veto it
    as implausible and revert to gravity. discover_nongrav must now pick 'yarkovsky' with a
    positive A2 matching JPL to a few percent."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2011_CL50")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                       auto_nongrav=False)
    out, info = fo.discover_nongrav(fit)
    assert info["model"] == "yarkovsky", f"picked {info.get('model')} ({info.get('reason')})"
    a2 = out.nongrav_params["A2"]
    assert a2 > 0, "CL50 A2 should be positive"
    assert ng.a2_is_plausible(a2, H=out.absolute_magnitude()[0]), "A2 outside size-aware range"
    assert not ng.a2_is_plausible(a2), "this is the regression: A2 exceeds the FLAT ceiling"
    assert abs(a2 / 4.38e-12 - 1.0) < 0.20, f"A2={a2:.2e} should match JPL +4.38e-12 within 20%"
    assert out.rms_residual() < 0.45, f"rms {out.rms_residual():.3f}\""
    assert out.n_observations() > fit.n_observations(), "expected more obs included with the nongrav"


def test_discover_yarkovsky_os5():
    """Automatic discovery on 2009 OS5 -- the genuine MODEL-SELECTION case. Its Yarkovsky A2 is
    significant (z~14, JPL A2=-3.3e-13) and plausible, but the raw-arcsec rms gain is only ~+9%
    (under the old flat 10% floor), so it used to be rejected. It cuts the WEIGHTED chi2 by ~49%,
    which the chi2-effect-size + nested F-test selection accepts. Guards the metric change: the
    yarkovsky trial must clear the chi2-gain bar while its raw-rms gain stays under the old 10%."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2009_OS5")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                       auto_nongrav=False)
    out, info = fo.discover_nongrav(fit)
    assert info["model"] == "yarkovsky", f"picked {info.get('model')} ({info.get('reason')})"
    a2 = out.nongrav_params["A2"]
    assert a2 < 0, "OS5 A2 should be negative"
    assert abs(a2 / -3.26e-13 - 1.0) < 0.20, f"A2={a2:.2e} should match JPL -3.26e-13 within 20%"
    # the metric is the point: the yarkovsky trial's chi2 gain clears 20% while its raw-rms gain
    # is under the old 10% floor (so the old raw-rms rule would have rejected it).
    yk = info["compare"]["yarkovsky"]
    assert yk["chi2_gain"] >= fo.NONGRAV_CHI2_GAIN, f"chi2_gain={yk['chi2_gain']:.2f}"
    assert yk["ftest_ok"], "nested F-test should be significant"


def test_discover_yarkovsky_db():
    """Automatic discovery on 2008 DB -- guards the SRP<->Yarkovsky physical prior. With both A1
    and A2 free, the degenerate A1 (z~4.5) lets A1+A2 edge out A2-only on raw chi2; but the gate-
    triggering secular trend is Yarkovsky's and JPL fits A2-only (-1.04e-12), so the A2 prior must
    keep the pick on a clean yarkovsky (not srp2)."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2008_DB")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                       auto_nongrav=False)
    out, info = fo.discover_nongrav(fit)
    assert info["model"] == "yarkovsky", f"picked {info.get('model')} ({info.get('reason')})"
    a2 = out.nongrav_params["A2"]
    assert a2 < 0 and abs(a2 / -1.04e-12 - 1.0) < 0.20, f"A2={a2:.2e} vs JPL -1.04e-12"
    assert "A1" not in out.nongrav_params, "should be a clean A2-only Yarkovsky, not A1+A2"


def test_comet_choice():
    """The comet model-model sequence selection (fit_object._ng_comet_choice): climb comet1 -> comet2 ->
    comet3, accepting each step only if its INCREMENTAL BIC over the previous step,
    dchi2 - dp*ln(N), reaches NONGRAV_BIC_MARGIN AND a nested F-test passes AND every coefficient
    is plausible (per-step BIC since 2026-09-14; the 15% gain / 5.5 z floors are retired). Returns
    the highest justified step -- the 'most conservative model justified by being better than the
    previous one'. `out` carries N and the gravity chi2_0. At N=1000 one term must buy ~21.9 chi2."""
    N = 1000
    out = lambda chi2_0=1000.0: {"N": N, "chi2_0": chi2_0}
    M = lambda chi2, phys=True: {"chi2": chi2, "phys_ok": phys}
    # comet1 justified vs gravity (dchi2 200), comet2's increment over comet1 too small (10) -> comet1
    assert fo._ng_comet_choice({"comet1": M(800), "comet2": M(790), "comet3": M(785)}, out()) == "comet1"
    # comet1 and comet2 both earn their term; comet3's A3 adds nothing -> comet2 (the C/2002 X5 shape)
    assert fo._ng_comet_choice({"comet1": M(800), "comet2": M(600), "comet3": M(595)}, out()) == "comet2"
    # every step keeps paying its way -> comet3
    assert fo._ng_comet_choice({"comet1": M(800), "comet2": M(600), "comet3": M(450)}, out()) == "comet3"
    # the first step itself isn't justified (dchi2 10 < 15 + ln 1000) -> nothing fires
    assert fo._ng_comet_choice({"comet1": M(990), "comet2": M(900), "comet3": M(880)}, out()) is None
    # a SMALL fractional gain on a big N is still a detection (the 2I shape: 8%, dBIC 616)
    assert fo._ng_comet_choice({"comet1": M(920), "comet2": M(915)}, out()) == "comet1"
    # a strong but physically implausible first step is vetoed -> None
    assert fo._ng_comet_choice({"comet1": M(500, phys=False)}, out()) is None
    # a step the lazy evaluator never fitted is simply absent -> the climb stops there
    assert fo._ng_comet_choice({"comet1": M(800)}, out()) == "comet1"


def test_discover_comet_x5():
    """End-to-end comet out-gassing discovery on C/2002 X5 (Kudo-Fujikawa) -- a near-Sun long-period
    comet a pure-gravity fit cannot reconcile (it ejects ~24%% of the obs). discover_comet_nongrav
    must reactivate everything, find a Marsden model with A1 and A2 justified, recover most of
    the dropped obs, and report plausible, significant A1 and A2. Under the per-step BIC model sequence
    (2026-09-14) the A3 term earns its place too: 'comet3' (it was 'comet2' under the old 15%
    incremental gain floor)."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("C_2002_X5")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                       auto_nongrav=False)                       # pure-gravity starting point
    out, info = fo.discover_comet_nongrav(fit)
    assert info["model"] == "comet3", f"picked {info.get('model')} ({info.get('reason')})"
    A1, A2 = out.nongrav_params["A1"], out.nongrav_params["A2"]
    assert ng.comet_a_is_plausible(A1) and ng.comet_a_is_plausible(A2), "A1/A2 outside the comet range"
    assert abs(A1) / out.nongrav_sigmas["A1"] > 3.0, "A1 should be a clear detection"
    assert abs(A2) / out.nongrav_sigmas["A2"] > 3.0, "A2 should be a clear detection"
    # the comet model should reconcile (redeem) obs the gravity fit had to eject
    assert out.n_observations() > fit.n_observations(), "expected more obs included with the comet model"


def test_comet_gate_skips_plain_asteroid():
    """The comet trigger must NOT fire on an ordinary main-belt / NEO orbit (T_J > 3.01, no comet
    designation): discover_comet_nongrav returns the fit untouched with a 'gate not met' reason, so
    the asteroid Yarkovsky path keeps its objects. Guards against the comet net over-reaching."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2003_BN4")                                    # a~1.27 au NEO, T_J ~ 5
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                       auto_nongrav=False)
    out, info = fo.discover_comet_nongrav(fit)
    assert info["applied"] is None and out is fit, f"comet path should not touch BN4 ({info})"
    assert "gate not met" in (info.get("reason") or ""), info.get("reason")


def test_nongrav_constraint_composes():
    """refine_nongrav(constraints=...) holds an element fixed WHILE solving the out-gassing, so
    the web UI's force-model / nongrav / constraint toggles are ORTHOGONAL (changing one keeps
    the others). The constraint must be ENFORCED (e pinned to target) and the nongrav
    coefficients still solved -- before this composed, a constrained fit dropped the nongrav."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2003_BN4")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                       auto_nongrav=False)
    target = round(fit.eccentricity()[0] + 0.05, 4)        # a target away from the free value
    f = fo.refine_nongrav(fit, "yarkovsky", constraints={"e": target}, revert_if_worse=False)
    assert "A2" in f.nongrav_params, "yarkovsky A2 should still be solved with a constraint active"
    assert abs(f.eccentricity()[0] - target) < 1e-3, f"e not pinned: {f.eccentricity()[0]} vs {target}"
    assert f.constraints == {"e": target}, f"constraints not recorded on the combined fit: {f.constraints}"


def test_discover_srp_2026na1_warns():
    """2026 NA1: an A1-only SRP fits (rms 1.08" -> ~0.56"; find_orb reports it, our profile-seed
    solver reaches it). Its A1 implies a ~0.006 g/cm^3 (sub-rock) density for this H~25/~35 m body,
    so auto-discovery must PICK the srp model AND attach an advisory non-physical-values WARNING --
    NOT reject it. We deliberately deliver the fit: a high area/mass object (a dark comet, a rocket
    body, or an A1 absorbing pre-Gaia systematics) is exactly what the warning is meant to surface.
    Guards the fit-and-warn (not veto) behaviour end-to-end."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2026_NA1")
    auto = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    # The SRP FAMILY, not one member: what this test guards is fit-and-warn rather than veto, and
    # which radial model wins is a separate (statistical) question. Since 2026-08-25 the A1+A2
    # model wins here -- the size-aware A2 ceiling stopped VETOING Yarkovsky, which let the
    # A1-vs-A1+A2 comparison be decided on chi2, and srp2 reaches 0.513" where srp-only reached
    # 0.551". Every property this test actually asserts (A1 > 0, sub-floor implied density, the
    # advisory warning) holds for both.
    assert auto.nongrav_model in ("srp", "srp2"), \
        f"expected a radial-SRP model to be fitted, got {auto.nongrav_model}"
    assert auto.nongrav_params["A1"] > 0
    assert auto.nongrav_warning, "a non-physical-values warning should be attached"
    assert "density" in auto.nongrav_warning.lower(), auto.nongrav_warning
    # the warning is advisory: the density diagnostic really is sub-floor for this H
    H, _ = auto.absolute_magnitude()
    assert ng.a1_implied_density(auto.nongrav_params["A1"], H) < ng.A1_MIN_DENSITY_KG_M3


def test_nongrav_physical_warnings():
    """nongrav.nongrav_physical_warnings is the ADVISORY diagnostic (never a veto): it returns a
    warning string when A1 implies a sub-rock density for the body's size, when A1 is negative (SRP
    pushes outward), or when A2 exceeds the size-aware Yarkovsky ceiling -- and None when every term
    is physical. A genuine SRP object (2009 BD-like, rho~0.7 g/cm^3) draws no warning."""
    # H~25, ~35 m: A1=1.5e-9 -> rho ~0.006 g/cm^3 -> warn
    w = ng.nongrav_physical_warnings({"A1": 1.5e-9}, H=25.0)
    assert w and "density" in w.lower()
    assert "dark comet" not in w.lower()     # by design the term is not used in any advisory
    # 2009 BD-like tiny high-AMR body: A1=5.6e-11 at H=28.1 -> rho ~0.7 g/cm^3 -> physical, no warn
    assert ng.nongrav_physical_warnings({"A1": 5.6e-11}, H=28.1) is None
    # negative A1 -> warn regardless of H
    assert "negative" in (ng.nongrav_physical_warnings({"A1": -1e-10}, H=25.0) or "").lower()
    # H unknown -> no density check possible, a modest positive A1 draws no warning
    assert ng.nongrav_physical_warnings({"A1": 1.5e-9}, H=None) is None


def test_nongrav_a3_asteroid_warning():
    """An out-of-plane A3 on the asteroid family ALWAYS draws the rarity note
    (by design). The note reports the size of A3 against A2/A1, says srp3 may be a mix of SRP
    and Yarkovsky, offers "some other force, or an observational data issue" as the reading,
    points at comet3, and claims NO A3/A2 ratio bound and no depletion timescale.
    A2-only and A1+A2 fits draw no A3 note. The kind is "a3_asteroid" so the delivery guard
    (which acts on the A1 kinds only) ignores it."""
    # 2009 HC's shape: A2 = +1.78e-12, A3 = -2.71e-11
    w, kinds = ng.nongrav_physical_warnings({"A1": 1.8e-11, "A2": 1.78e-12, "A3": -2.71e-11},
                                            H=24.65, with_kinds=True)
    assert "a3_asteroid" in kinds
    assert "finding real A3 values in asteroids is rare" in w
    assert "15x |A2|" in w and "mix of solar radiation pressure" in w
    assert "some other force, or an observational data issue" in w
    assert "A2-only and A1+A2" in w and "each apparition" in w
    # by design: no comet3 pointer (a 1/r^2-like g(r) inside 1.5 au is a null test),
    # no "dark comet" anywhere (the term means A1~0 with A3 dominant; say less), no
    # ratio ceiling, no systematic verdict, no depletion timescale
    assert "comet3" not in w and "dark comet" not in w.lower()
    assert "ceiling" not in w and "systematic" not in w
    # the depletion timescale: 2.71e-11 au/day^2 ~ 5.4e-10 m/s^2 -> v/|A| ~ 2e4 yr
    assert 1.5e4 < ng.outgassing_depletion_years(2.71e-11) < 2.5e4
    assert " yr" not in w and "deplete" not in w
    # no A3 -> no A3 note; the existing clauses are untouched
    assert ng.nongrav_physical_warnings({"A1": 5.6e-11, "A2": -3e-13}, H=28.1) is None
    _, k = ng.nongrav_physical_warnings({"A2": -3e-13}, H=None, with_kinds=True)
    assert "a3_asteroid" not in k


def test_radial_seed_all_descent():
    """Two independent routes descend the degenerate A1+A2 comet model on 2026 NA1 -- the confirmed
    stuck case where the plain damped LM stalls near the gravity corner at rms ~0.62". (1) The wide
    radial profile-seed (refine_nongrav radial_seed_all=True, what the CLI --nongrav passes) lands on
    the valley floor; (2) find_orb's full-step corrector reaches it from a COLD start (radial_seed_all
    =False) -- see test_full_step_corrector. BOTH must reach the real out-gassing minimum (rms ~0.48",
    well under the stuck 0.62"), and comet3 (adding A3) must not be worse than comet2."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2026_NA1")
    grav = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                        auto_nongrav=False)
    c2 = fo.refine_nongrav(grav, "comet2", revert_if_worse=False, radial_seed_all=True)
    assert c2.rms_residual() < 0.55, \
        f"radial_seed_all comet2 should descend to the out-gassing minimum, got {c2.rms_residual():.4f}\""
    # a converged SRP fit reaches its true minimum too
    s2 = fo.refine_nongrav(grav, "srp2", revert_if_worse=False, radial_seed_all=True)
    assert s2.rms_residual() < 0.50, f"srp2 should reach its minimum, got {s2.rms_residual():.4f}\""
    # adding A3 (comet3) has an extra free parameter: never worse than comet2
    c3 = fo.refine_nongrav(grav, "comet3", revert_if_worse=False, radial_seed_all=True)
    assert c3.rms_residual() <= c2.rms_residual() + 5e-3, "comet3 should not be worse than comet2"


def test_full_step_corrector():
    """find_orb's full-step corrector (the ported `full_improvement`) descends a degenerate comet
    model from a COLD start -- no radial seed -- so the CHEAP auto-discovery cold path can reach it
    (this is what makes comet auto-DETECTION possible, not just an explicit CLI fit). On 2026 NA1 the
    cold LM alone stalls at rms ~0.62"; with the corrector the cold fit reaches ~0.48". It is gated
    (only a stuck, radial-A1 cold fit runs it) and kept only when it beats the LM, so the
    well-conditioned models are untouched: srp2 (not degenerate) and yarkovsky (transverse, no radial
    A1) must land exactly where the LM alone does."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2026_NA1")
    grav = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                        auto_nongrav=False)
    # COLD (no seed): the corrector must carry comet2 well below the un-seeded LM's ~0.62" stall.
    c2_cold = fo.refine_nongrav(grav, "comet2", revert_if_worse=False, radial_seed_all=False)
    assert c2_cold.rms_residual() < 0.55, \
        f"the cold corrector should descend comet2 to the minimum, got {c2_cold.rms_residual():.4f}\""
    # well-conditioned models are left to the LM (corrector gated out / kept only if better)
    yk = fo.refine_nongrav(grav, "yarkovsky", revert_if_worse=False, radial_seed_all=False)
    assert yk.rms_residual() < grav.rms_residual() + 1e-6, "yarkovsky must not be worsened"


def test_profile_likelihood_sigma():
    """The RATIONAL nongrav uncertainty (refine_nongrav profile_sigmas=True): the profile-likelihood
    interval (fix a coefficient, re-fit the state + other coefficients, find where chi^2 rises by 1)
    replaces the linear covariance sigma, which reports a fake-tight symmetric error for a term that
    is really degenerate with the state. On 2026 NA1 the Yarkovsky A2 is essentially UNCONSTRAINED --
    the profile sigma must be MUCH larger than the linear sigma and larger than |A2| itself (i.e. the
    term is consistent with zero), where the linear covariance dishonestly called it a detection."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2026_NA1")
    grav = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"],
                        auto_nongrav=False)
    f = fo.refine_nongrav(grav, "yarkovsky", revert_if_worse=False)   # profile on by default
    lin_a2 = f.nongrav_sigmas["A2"]                                   # the fast LINEAR covariance sigma
    rat_a2 = f.rational_nongrav_sigmas()["A2"]                        # the RATIONAL profile-likelihood sigma
    a2 = abs(f.nongrav_params["A2"])
    assert rat_a2 > 3.0 * lin_a2, "profile sigma should be much larger than the fake-tight linear sigma"
    assert rat_a2 > a2, "an unconstrained A2 must have sigma > |A2| (consistent with 0)"
    # a fit NOT flagged for profiling (discovery's internal comparison trials) keeps the linear sigma
    lin_only = fo.refine_nongrav(grav, "yarkovsky", revert_if_worse=False, profile_sigmas=False)
    assert lin_only.rational_nongrav_sigmas()["A2"] == lin_only.nongrav_sigmas["A2"]


def test_coherent_block_nights():
    """The structural tell that arms the profile-seeded srp rescue when discovery's un-seeded trial is
    blind to a deep (state,A1) degeneracy (fit_object._coherent_block_nights): the GRAVITY fit gutted a
    large, multi-night block of a single apparition, on a FEW-apparition arc. It must fire on a
    2-apparition object whose old apparition is gutted (the 2026 NA1 case: 13/31 over 3 nights) and NOT
    on (a) a long, well-observed arc with one lone gutted apparition among many clean ones (163693
    Atira: a data problem, not a force), (b) a single gutted tracklet (one night -> likely bad
    astrometry), or (c) thin scattered rejections below the fraction floor."""
    class _Stub:
        def __init__(self, obs, mask):
            self._obs = obs; self._mask = np.array(mask, bool)
    def arc(app_nights):
        """app_nights: list of apparitions, each a list of (n_obs, n_rejected) per night. Apparitions
        are spaced >1 year apart, nights >1 day apart within an apparition."""
        obs, mask, t = [], [], 0.0
        for nights in app_nights:
            t += 500.0                                    # new apparition (> 238 d gap)
            for n, rej in nights:
                t += 3.0                                  # new night
                for i in range(n):
                    obs.append({"jd_utc": 2450000.0 + t + i * 1e-3})
                    mask.append(i >= rej)                 # first `rej` rejected
        return _Stub(obs, mask)
    # 2026 NA1-shaped: old apparition gutted over 3 nights (2/8, 10/11, 1/3=12 rej) + clean recent one
    na1 = arc([[(8, 2), (11, 10), (3, 3), (6, 0), (3, 0)], [(20, 0), (27, 2)]])
    assert fo._coherent_block_nights(na1) >= 2, "gutted 2-apparition block must arm"
    # Atira-shaped: many clean apparitions + one gutted -> a data problem, must NOT arm (n_opp gate)
    atira = arc([[(20, 0)]] * 14 + [[(30, 12), (30, 12), (30, 12)]])
    assert fo._coherent_block_nights(atira) == 0, "lone gutted apparition on a long arc must NOT arm"
    # single gutted tracklet (one night) -> likely bad astrometry, must NOT arm (nights gate)
    one = arc([[(10, 0), (11, 10)], [(20, 0)]])
    assert fo._coherent_block_nights(one) == 0, "single-night block must NOT arm"
    # thin scattered rejections below the fraction floor -> must NOT arm
    thin = arc([[(20, 1), (20, 1), (20, 1)], [(20, 0)]])
    assert fo._coherent_block_nights(thin) == 0, "thin scattered rejection must NOT arm"


# --- stalls are recorded and rescued; nested trials and delivery are converged ----------

def test_lift_nongrav_state_units():
    """_lift_nongrav_state builds refine_nongrav's seed vector from a converged (sub-)model fit:
    the 6-state as it stands, each free coefficient at the source's PHYSICAL value divided by the
    TARGET model's scale (yarkovsky 1e-14, srp 1e-11 differ by three decades), 0 for an axis the
    source did not carry."""
    class _F:
        _state = np.array([1, 2, 3, 4, 5, 6, 0.0])
        nongrav_params = {"A2": -2.0e-13}
    p = fo._lift_nongrav_state(_F(), "srp2")           # (A1, A2) at scale 1e-11
    assert p.shape == (8,)
    assert np.allclose(p[:6], [1, 2, 3, 4, 5, 6])
    assert p[6] == 0.0                                  # A1: not carried by the source
    assert np.isclose(p[7], -2.0e-13 / ng.model("srp2")["scale"])
    q = fo._lift_nongrav_state(_F(), "yarkovsky")      # same model, same value, its own scale
    assert np.isclose(q[6], -2.0e-13 / ng.model("yarkovsky")["scale"])


def test_discovery_record_stall_rescue_and_nesting():
    """2008 NA (131 obs, 2 oppositions, 17 years) is the object that hid the stall defect for twelve days:
    every single-coefficient trial stalls at A = 0.000e+00 EXACTLY from a cold start. The delivered
    orbit must now carry the discovery record, and that record must show (1) the stall NAMED as a
    stall, not a null; (2) the transverse trial RESCUED off the corner; and (3) the nested srp2
    never worse than its best sub-model in ANY stage -- the invariant a cold trial used to break
    (836 vs 476 on the old round 2), which had the selection comparing a converged model with an
    unconverged one. (4) The settle loop ends at a fixed point: the delivered model is the model
    its mask was rejected under."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2008_NA")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    disc = fit.nongrav_discovery
    assert disc and "asteroid" in disc, "the discovery record must ride on the delivered orbit"
    info = disc["asteroid"]
    cmp_ = info["compare"]
    # (1) and (2) are claims about what a COLD start does, so they read the COLD record. 2008 NA's
    # coherent-block tell fires (block_nights = 3), so the seeded srp pass runs even
    # though round 1 already justified a model -- and a profile-seeded trial produces no stall /
    # rescue diagnostics at all, because seeding takes it out of refine_nongrav's diagnostic
    # branch. `compare` is therefore the SEEDED comparison for this object and cannot answer "did
    # the cold trial stall"; `compare_cold` is kept precisely so it still can.
    # `compare_cold` exists only when the coherent-block tell bought a seeded pass; whether it
    # fires is DATA-GEOMETRY dependent (the DE440 observer fix moved 2008 NA's gravity
    # fit enough that the block is no longer gutted, tell off, no seeded pass). Whichever record
    # is the cold one, (1) and (2) are claims about IT.
    cold = info.get("compare_cold", info["compare"])
    srp_diag = cold["srp"]["diag"]
    assert srp_diag["stalled"] is True and srp_diag["zero_axes"] == ["A1"], srp_diag
    # The yarkovsky claim is that the record EXPLAINS the trial, not which branch fired: before
    # the DE440 observer fix, 2008 NA's cold A2 LM sat in the gravity corner and only
    # the profile rescue reached the minimum (`rescued == "A2"`); with corrected geometry the
    # cold LM converges on its own (`stalled False`, nothing to rescue). Both are coherent
    # records of a determined coefficient -- what must never appear is an exact-zero A2 with a
    # diag that cannot say why (the stall rule).
    ydiag = cold["yarkovsky"]["diag"]
    y_a2 = float((cold["yarkovsky"].get("params") or {}).get("A2", 0.0))
    assert (ydiag.get("rescued") == "A2") or (ydiag["stalled"] is False and y_a2 != 0.0), \
        (ydiag, y_a2)
    for stage in ("compare", "retest"):
        m = info[stage]
        sub_best = min(m["srp"]["chi2"], m["yarkovsky"]["chi2"])
        assert m["srp2"]["chi2"] <= sub_best * (1 + 1e-6) + 1e-9, \
            f"{stage}: srp2 chi2 {m['srp2']['chi2']:.1f} > best sub-model {sub_best:.1f}"
    # (5) the converged srp2 is the better fit by chi2 but carries a NEGATIVE A1 (an inward radial
    # acceleration no mechanism in the model can produce), so it fails plausibility and the
    # transverse model is delivered -- the sign rule.
    assert cmp_["srp2"]["params"]["A1"] < 0 and cmp_["srp2"]["phys_ok"] is False, cmp_["srp2"]
    assert info.get("applied"), info.get("reason")
    assert info["model"] == "yarkovsky"
    last = info["settle"][-1]
    assert last["choice"] == last["under"] == info["model"], info["settle"]
    # and the delivered coefficient MOVED: a fitted A of exactly zero is a stall, never a result
    assert all(v != 0.0 for v in fit.nongrav_params.values()), fit.nongrav_params


def test_jackknife_nongrav_robustness():
    """The delete-one-tracklet JACKKNIFE: a robustness check the covariance / profile
    (and find_orb's independent-noise Monte Carlo) cannot give. The load-bearing guarantee is that it
    does NOT cry wolf on a genuine detection -- the bug that made 2007 AA2's clear A1 read as +-7e-9
    (non-converged refits inflating a std). On 2011 CL50's real Yarkovsky A2 (matches JPL) the
    corrected estimator (converged refits only, robust MAD scale, sign-based verdict) must:
      - ASSESS it (enough refits converge),
      - keep it robust (the sign never spans zero across the leave-one-out cloud), and
      - report a high robust-z, not a spuriously low one.
    Each refit is INDEPENDENT (re-seeded, not warm-started), with the profile seed forced when a
    single-coefficient plain LM would diverge, and refits that do not converge are rejected."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fx = _fixture("2011_CL50")
    fit = fo.fit_orbit(fx["obs"], _oc(), desig=fx["meta"]["designation"], meta=fx["meta"])
    assert fit.nongrav_model, "2011 CL50 should deliver a Yarkovsky model"
    jk = fit.nongrav_jackknife_detail()
    assert "A2" in jk and jk["A2"].get("assessed"), f"A2 robustness should be assessable: {jk.get('A2')}"
    assert not jk["A2"]["spans_zero"], "a real A2 detection must not be flagged not-robust"
    assert jk["A2"]["z"] > 4.0, f"a real A2 jackknife z should stay high, got {jk['A2']['z']:.1f}"
    assert "A2" not in fit.nongrav_not_robust
    # the robustness sigma is a real number and the coefficient's sign never flips
    assert jk["A2"]["se"] > 0 and jk["A2"]["flip"] == 0.0


def test_jackknife_lift_and_split():
    """Units of the jackknife building blocks, no fitting: the tracklet splitter groups by station +
    <=120 min, and _lift_nongrav_state carries a coefficient into a model's scale (shared with the
    stall rescue)."""
    import numpy as np
    # two stations, a >120 min gap splits within a station
    times = np.array([0.0, 0.01, 0.02, 0.2, 5.0, 5.005])   # days
    stn = np.array(["A", "A", "A", "A", "B", "B"])
    g = fo._split_tracklets_by_station(times, stn)
    assert [len(x) for x in g] == [3, 1, 2], g   # A: 0/.01/.02 together, .2 (>120min? .18d=259min) apart; B: pair


def _discovery_srp2_mask(grav):
    """srp2's own discovery-rejection mask (all obs minus worst-2, fit+seed, one reject round)."""
    import numpy as np; import pyroorb.arc_extend as ae
    ev = grav._ctx["ev"]; ev.restore_link_relaxation(); n = ev.n
    allm = np.ones(n, bool)
    st, result = ev.fit(np.asarray(grav._state, float), allm, thin=False)
    worst = np.argsort(ev.resid_sigmas(st))[::-1][:fo.NONGRAV_DROP_WORST]
    tm = allm.copy(); tm[worst] = False
    gall = fo._build_orbit(grav._ctx, st, tm, result, grav.history)
    f = fo.refine_nongrav(gall, "srp2", mask=tm, revert_if_worse=False, radial_seed_all=True, profile_sigmas=False)
    rm = tm.copy(); ae.auto_reject_within_arc(f._ctx["ev"], rm, np.asarray(f._state, float)[:6])
    return rm


def test_force_seed_reaches_the_rescue_on_an_a1_only_model():
    """`force_seed=True` must take precedence over the keep-the-better radial-seed branch.

    On an A1-only srp model (`_seed_radial` is srp-only), the default `profile_seed=True`
    used to route a forced refit through that branch, which is gated OUT of the diagnostic
    block -- so the rescue never ran, `force_seed` was silently a no-op, and the diag came
    back EMPTY. The delete-1 jackknife reads `rescue_chi2` off exactly this call to tell a
    coefficient that COLLAPSED to zero (evidence: it stays in the cloud) from one that merely
    stalled there (dropped) -- with an empty diag every genuine srp collapse was discarded as
    an unexplained stall, biasing the cloud away from zeros and inflating z, the
    phantom-hiding direction. The offline routing tests in test_nongrav_jackknife.py fake the
    engine, so THIS is the test that pins the engine half: the jackknife's own kwargs, a real
    srp refit, and a diag that can explain a zero. (The kept coefficient is unchanged by the
    precedence -- measured identical to 12 digits on 2008 NA -- so no golden moves.)"""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    fit = _fit("2011_CL50")
    f = fo.refine_nongrav(fit, "srp", revert_if_worse=False,
                          force_seed=True, profile_sigmas=False)
    assert "A1" in (f.nongrav_diag.get("rescue_chi2") or {}), f.nongrav_diag


def test_bic_selection_recovers_jpl_yarkovsky():
    """The BIC effect-size bar recovers Yarkovsky detections the old 0.15-gain / 5.5-z
    floors refused. 2003 HM -- the headline false negative, published by the MPC/JPL -- must now
    deliver yarkovsky (an 8% chi2 gain at F=23 that the 15% floor declined). And the A1-vs-A2
    preference must NOT regress: 2026 NA1, whose yarkovsky A2 is over the size-aware ceiling, must
    still deliver the radial srp2 model (the existing physical `a2_over_ceiling` clause handles it,
    even though BIC now makes its yarkovsky 'justified')."""
    if not _assist_available():
        pytest.skip("ASSIST backend unavailable")
    hm = _fixture("2003_HM")
    fit = fo.fit_orbit(hm["obs"], _oc(), desig=hm["meta"]["designation"], meta=hm["meta"])
    assert fit.nongrav_model == "yarkovsky", \
        f"BIC should recover 2003 HM's Yarkovsky, got {fit.nongrav_model}"
    assert fit.nongrav_params.get("A2", 0.0) < 0    # JPL sign

    na1 = _fixture("2026_NA1")
    f2 = fo.fit_orbit(na1["obs"], _oc(), desig=na1["meta"]["designation"], meta=na1["meta"])
    assert f2.nongrav_model in ("srp", "srp2"), \
        f"2026 NA1 must stay a radial model under BIC (A1/A2 preference), got {f2.nongrav_model}"


# --------------------------------------------------------------------------------------
# _ng_settle: the mask the delivered model was never allowed to judge
# --------------------------------------------------------------------------------------

class _SettleStub:
    """The three attributes `_ng_settle` touches on a refined orbit."""

    def __init__(self, ev):
        self._ctx = {"ev": ev}
        self._state = np.zeros(7)


def _settle_harness(monkeypatch, choices):
    """Drive `_ng_settle` with a scripted `evaluate`. `choices` is what each successive
    re-test prefers. Every rejection flips one more mask bit off, so the returned mask
    records exactly how many rejection rounds ran."""
    import pyroorb.arc_extend as ae

    state = {"n_reject": 0, "fitted_under": []}

    def _fake_refine(grav, model, mask=None, **kw):
        state["fitted_under"].append(model)
        return _SettleStub(object())

    def _fake_reject(ev, mask, state6, orig_n=None):
        mask[state["n_reject"]] = False           # one more obs drops each round
        state["n_reject"] += 1

    monkeypatch.setattr(fo, "refine_nongrav", _fake_refine)
    monkeypatch.setattr(fo, "_lift_nongrav_state", lambda fit, model: None)
    monkeypatch.setattr(ae, "auto_reject_within_arc", _fake_reject)

    grav = types.SimpleNamespace(_state=np.zeros(6))
    models = {k: {"fit": object()} for k in ("A", "B")}
    seq = list(choices)

    def _evaluate(seed6, mask, warm):
        return {"choice": seq.pop(0) if seq else None, "grav": grav, "models": models}

    sel = {"choice": "A", "grav": grav, "models": models}
    info = {}
    sel2, mask = fo._ng_settle(sel, np.ones(12, bool), _evaluate, 100, info, False)
    return sel2, mask, info, state


def test_ng_settle_records_a_reached_fixed_point(monkeypatch):
    """The ordinary case: the re-test agrees with the model the mask was rejected under."""
    sel2, mask, info, st = _settle_harness(monkeypatch, ["A"])
    assert info["settled"] is True
    assert sel2["choice"] == "A"
    assert st["fitted_under"] == ["A"]              # one rejection, under the delivered model
    assert int((~mask).sum()) == 1


def test_ng_settle_never_returns_a_mask_the_delivered_model_did_not_judge(monkeypatch):
    """The regression. With the models oscillating, the loop exhausted
    NONGRAV_SETTLE_ROUNDS and returned `sel2` -- choice B -- alongside a mask that had been
    rejected under A. The caller then refits B on it and delivers, which is precisely the
    "a model on observations it never got to judge" failure the function exists to prevent
    (its docstring costs 2008 NA 124/131 @ 0.971" for what settles at 117/131 @ 0.829").
    """
    sel2, mask, info, st = _settle_harness(monkeypatch, ["B", "A", "B", "A"])
    assert info["settled"] is False
    delivered = sel2["choice"]
    # the last rejection ran under the model that is about to be delivered
    assert st["fitted_under"][-1] == delivered
    last = info["settle"][-1]
    assert last["unsettled_final"] is True and last["under"] == delivered
    # ...and it really rejected: one pass per settle round, plus the final one
    assert st["n_reject"] == fo.NONGRAV_SETTLE_ROUNDS + 1
    assert int((~mask).sum()) == fo.NONGRAV_SETTLE_ROUNDS + 1


def test_ng_settle_declining_retest_is_settled_and_pays_no_extra_rejection(monkeypatch):
    """A re-test that justifies nothing (`choice` None) breaks the loop; the caller reverts
    to gravity, so there is no delivered model to judge a mask with."""
    sel2, mask, info, st = _settle_harness(monkeypatch, ["B", None])
    assert info["settled"] is True and sel2["choice"] is None
    assert st["n_reject"] == 2
    assert not any(r.get("unsettled_final") for r in info["settle"])
