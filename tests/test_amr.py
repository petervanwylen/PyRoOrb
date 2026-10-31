"""Offline tests for the AMR (area/mass ratio) report -- find_orb's

    Epoch 1997 Oct 13.0 TT; AMR 0.034 m^2/kg

which elem_out.cpp prints as ``orbit[6] * SOLAR_GM / SRP1AU`` for ONE force model,
FORCE_MODEL_SRP (its 1-parameter inverse-square radial term).  find_orb's internal
parameter ``orbit[6]`` enters the equations of motion as a dimensionless addition to
the solar term before the ``-SOLAR_GM/r^3`` factor (runge.cpp), so ``orbit[6]*SOLAR_GM``
IS the radial acceleration at 1 au in au/day^2 -- our A1.  Hence AMR = A1 / SRP1AU.

No network, no fit: constants and the model rule.
"""
import math

import pyroorb.nongrav as ng


# constant.h: SRP1AU = SOLAR_CONSTANT * seconds_per_day^2 / (AU_IN_METERS * SPEED_OF_LIGHT
# * METERS_PER_KM), with SOLAR_CONSTANT = 1361 W/m^2. find_orb's own note puts it at
# "about 2.3e-7 AU/day^2" for a 1 m^2/kg perfect absorber.
def test_srp1au_matches_find_orb_constant():
    assert math.isclose(ng.SRP1AU, 1361.0 * 86400.0 ** 2
                        / (1.495978707e11 * 299792.458 * 1000.0), rel_tol=1e-12)
    assert 2.2e-7 < ng.SRP1AU < 2.3e-7


def test_amr_is_a1_over_srp1au():
    # A unit area/mass ratio is, by construction, an A1 of exactly SRP1AU.
    assert math.isclose(ng.amr_m2_per_kg("srp", {"A1": ng.SRP1AU}), 1.0, rel_tol=1e-12)
    assert math.isclose(ng.amr_m2_per_kg("srp", {"A1": 3.4e-2 * ng.SRP1AU}), 0.034, rel_tol=1e-12)
    # find_orb computes it from its internal parameter; the two agree by definition.
    solar_gm = 0.01720209895 ** 2
    orbit6 = 1.0e-4
    assert math.isclose(ng.amr_m2_per_kg("srp", {"A1": orbit6 * solar_gm}),
                        orbit6 * solar_gm / ng.SRP1AU, rel_tol=1e-12)


def test_amr_only_for_the_one_parameter_srp_model():
    # find_orb prints AMR for FORCE_MODEL_SRP only -- not the two/three-parameter SRP
    # models, whose radial coefficient is no longer the whole photon push, and not for
    # out-gassing comets (a different g(r) entirely).
    assert ng.amr_m2_per_kg("srp", {"A1": 1e-9}) is not None
    for model in ("srp2", "srp3", "comet1", "comet2", "comet3", "none", None):
        assert ng.amr_m2_per_kg(model, {"A1": 1e-9, "A2": 1e-14}) is None
    # ...and never without an A1 to convert.
    assert ng.amr_m2_per_kg("srp", {}) is None
    assert ng.amr_m2_per_kg("srp", None) is None


def test_amr_sign_follows_a1():
    # A negative A1 is non-physical (SRP pushes outward; nongrav_physical_warnings says so),
    # but the conversion must not hide it behind an abs().
    assert ng.amr_m2_per_kg("srp", {"A1": -ng.SRP1AU}) == -1.0


def test_amr_report_rescales_a1s_interval():
    # AMR is a FIXED multiple of A1, so A1's profile interval rescales onto it -- no refits.
    r = ng.amr_report("srp", {"A1": 7.7e-9},
                      intervals={"A1": (4.4e-10, 4.6e-10)}, sigmas={"A1": 4.5e-10})
    assert math.isclose(r["amr"], 7.7e-9 / ng.SRP1AU, rel_tol=1e-12)
    assert math.isclose(r["minus"], 4.4e-10 / ng.SRP1AU, rel_tol=1e-12)
    assert math.isclose(r["plus"], 4.6e-10 / ng.SRP1AU, rel_tol=1e-12)
    assert r["unconstrained"] is False
    # The interval wins over the sigma; the sigma is the symmetric fallback.
    r2 = ng.amr_report("srp", {"A1": 7.7e-9}, sigmas={"A1": 4.5e-10})
    assert r2["minus"] == r2["plus"] == 4.5e-10 / ng.SRP1AU
    # Before the profile sigmas are computed there is a value and no interval at all.
    r3 = ng.amr_report("srp", {"A1": 7.7e-9})
    assert r3["minus"] is None and r3["plus"] is None
    # ...and no report at all off the 1-parameter model.
    assert ng.amr_report("srp2", {"A1": 7.7e-9}, sigmas={"A1": 4.5e-10}) is None


def test_amr_line_shapes():
    from pyroorb import presentation

    def line(**kw):
        return presentation.amr_line(ng.amr_report("srp", {"A1": 7.7e-9}, **kw))

    assert presentation.amr_line(None) is None
    assert line().startswith("AMR = 0.03399 m²/kg")                       # no ± yet
    assert " ± 0.002 m²/kg" in line(intervals={"A1": (4.4e-10, 4.5e-10)})  # symmetric
    assert "+0.004 / -0.00044 m²/kg" in line(intervals={"A1": (1e-10, 9e-10)})   # asymmetric
    # An axis the profile never bounded is a LOWER bound on the uncertainty, never a "±".
    loose = line(intervals={"A1": (1e-10, 9e-10)}, unconstrained=["A1"])
    assert "±" not in loose.split("unconstrained")[0]
    assert "± is at least 0.004" in loose
