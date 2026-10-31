"""Unit tests for the self-perturbing-asteroid guard (the "+16ast" analogue of the
Pluto ban).

The 16 massive main-belt asteroids in the sb441-n16 perturber kernel ARE the bodies
the +16ast force model adds. If the object being fit is itself one of those 16, turning
the perturbers on would propagate it in a field that already contains it -- it would be
perturbed by itself, exactly the Pluto paradox. Unlike Pluto (always a DE440 perturber,
so banned from being fit at all), the +16ast perturbers are OPT-IN, so these objects are
still fit normally with gravity -- we just refuse to enable +16ast for them. These tests
pin the identity layer (every form the object can arrive in), the look-alikes that must
NOT match, the FittedOrbit/obs-based detection refine_asteroids / refit_force_model use,
and -- when the real kernel is present -- that the hard-coded set matches it exactly.
"""
import os

import pytest

from pyroorb import guards

# refits with the asteroid perturber kernel
pytestmark = pytest.mark.slow
# Look-alikes that must NOT match: asteroid numbers/names NOT in the n16 set (5 Astraea,
# 6 Hebe, 19 Fortuna -> n373 not n16), near-miss numbers, Pluto, and unrelated objects.
NOT_PERTURBERS = ["5", "Astraea", "6", "Hebe", "19", "Fortuna", "40", "1073",
                  "134340", "Pluto", "433", "Eros", "2023 QJ3", "2004 AB",
                  "K16UL0A", "", None]


def test_every_perturber_matches_by_number_name_and_both():
    for n, name in guards.N16_PERTURBERS.items():
        for form in (str(n), f"({n})", name, name.lower(), f"{n} {name}",
                     f"({n}) {name}", str(n).zfill(5)):
            assert guards.perturber_number(form) == n, f"{form!r} should map to {n}"


def test_lookalikes_do_not_match():
    for d in NOT_PERTURBERS:
        assert guards.perturber_number(d) is None, f"{d!r} should NOT be a perturber"


def test_hygiea_alternate_spelling():
    assert guards.perturber_number("Hygeia") == 10        # common alternate of 10 Hygiea


def test_self_perturber_name_from_fit_designations():
    class Stub:
        designations = ["4 Vesta"]; designation = "4 Vesta"; _obs = []
    assert guards.self_perturber_name(fit=Stub()) == "4 Vesta"


def test_self_perturber_name_from_obs_self_identifying_by_number():
    """Astrometry that self-identifies as asteroid 4 (designation OR packed code) is
    caught even when the typed designation was something innocuous."""
    obs = [{"designation": "2023 QJ3", "packed": "K23Q03J"},
           {"designation": "4", "packed": "00004"}]
    assert guards.self_perturber_name(desig="2023 QJ3", obs=obs) == "4 Vesta"


def test_self_perturber_name_spares_clean_request():
    assert guards.self_perturber_name(
        desig="2023 QJ3",
        meta={"designation": "2023 QJ3", "designations": ["2023 QJ3"]},
        obs=[{"designation": "2023 QJ3", "packed": "K23Q03J"}]) is None


def test_constant_matches_the_real_kernel():
    """When the sb441-n16 kernel is present, N16_PERTURBERS must equal its SPK targets
    (2000000 + MP number) exactly -- so a future kernel swap can't silently drift the
    guard. Skipped (not failed) when the ~616 MB kernel hasn't been downloaded."""
    import pyroorb.force_assist as fa
    path = getattr(fa, "ASTEROIDS_PATH", None) or getattr(fa, "ASTEROIDS_DEFAULT_PATH", None)
    if not (path and os.path.exists(path)):
        pytest.skip("sb441-n16 kernel not present")
    from jplephem.spk import SPK
    k = SPK.open(path)
    try:
        targets = {s.target for s in k.segments if s.target > 2000000}
    finally:
        k.close()
    kernel_numbers = {t - 2000000 for t in targets}
    assert kernel_numbers == set(guards.N16_PERTURBERS), (
        f"kernel has {sorted(kernel_numbers)} but N16_PERTURBERS has "
        f"{sorted(guards.N16_PERTURBERS)} -- update the constant")


def test_web_banner_mentions_the_object():
    from pyroorb.web import engine_api
    msg = engine_api.self_perturber_banner("4 Vesta")
    assert "4 Vesta" in msg and "+16ast" in msg


# ---------------------------------------------------------------------------
# Enforcement: whole-fit +16ast mode (PYROORB_ASTEROIDS_PATH) turns the perturbers on
# GLOBALLY, underneath the per-call flag the refusals above police -- which is how (1) Ceres
# came to be fit inside a field containing itself (19 of 7993 obs kept, the modern arc reading
# ~1e5 sigma). These pin the suppression that closes that door.
# ---------------------------------------------------------------------------
def test_context_suppresses_the_perturbers_even_when_asked_for():
    import pyroorb.force_assist as fa
    assert fa.suppressed_perturber() is None
    with fa.perturbers_off_for("1 Ceres"):
        assert fa.suppressed_perturber() == "1 Ceres"
        # not merely the default: an EXPLICIT asteroids=True is overruled too, because
        # self-perturbation is meaningless regardless of who asked.
        assert fa._use_asteroids(True) is False
        assert fa._use_asteroids(None) is False
        assert fa._epsilon(True) == fa.IAS15_EPSILON      # gravity-only tolerance
    assert fa.suppressed_perturber() is None


def test_gravity_only_wrapper_pins_the_flag_off():
    import pyroorb.force_assist as fa
    seen = {}

    class Backend:
        marker = "passthrough"

        def predict_state(self, *a, asteroids=None, **k):
            seen["asteroids"] = asteroids
            return "state"

    b = fa.GravityOnlyForce(Backend())
    assert b.predict_state(1, 2, asteroids=True) == "state"
    assert seen["asteroids"] is False          # HARD pin, not a default
    assert b.marker == "passthrough"           # everything else falls through


def test_asteroid_wrapper_cannot_unwrap_the_gravity_pin():
    """_AsteroidForce unwraps another _AsteroidForce (idempotent) but must not unwrap the
    self-perturber pin -- the delivery-time force choice may not put an object back into the
    kernel it is a member of."""
    import pyroorb.fit_object as fo
    import pyroorb.force_assist as fa
    seen = {}

    class Backend:
        def predict_state(self, *a, asteroids=None, **k):
            seen["asteroids"] = asteroids

    pinned = fa.GravityOnlyForce(Backend())
    wrapped = fo._AsteroidForce(pinned)
    assert wrapped._backend is pinned
    wrapped.predict_state(1)
    assert seen["asteroids"] is False


def test_force_backend_hands_back_a_pinned_backend_in_context(monkeypatch, tmp_path):
    """In whole-fit +16ast mode the backend the geometry build lands in the fit's ctx must be
    gravity-pinned for a perturber object -- that is what makes the suppression outlive the
    fit (interactive steps, re-anchors, ephemerides)."""
    import pyroorb.force_assist as fa
    from pyroorb.obs_sources import _force_backend
    kern = tmp_path / "sb441-n16.bsp"; kern.write_text("x")
    monkeypatch.setattr(fa, "ASTEROIDS_PATH", str(kern))
    if not os.path.exists(fa.PLANETS_PATH):
        pytest.skip("DE440 planets kernel not present")
    backend, tag = _force_backend()
    assert not getattr(backend, "_gravity_pinned", False) and "+16ast)" in tag
    with fa.perturbers_off_for("1 Ceres"):
        backend, tag = _force_backend()
    assert backend._gravity_pinned and "self-perturber" in tag


def test_pinned_backend_is_preferred_for_products():
    """An ephemeris / impact scan resolves the backend itself; it must take the fit's pinned
    one when there is one, or it would recompute in the force model the fit had to refuse."""
    import pyroorb.force_assist as fa
    from pyroorb.obs_sources import _force_backend

    class Fit:
        force_model = "ASSIST(DE440+GR, +16ast off: self-perturber)"

        def __init__(self, backend): self._ctx = {"backend": backend}

    plain = object()
    pinned = fa.GravityOnlyForce(plain)
    assert _force_backend(Fit(pinned)) == (pinned, Fit.force_model)
    # an ordinary fit (and no fit at all) resolves the backend the usual way
    assert _force_backend(Fit(plain))[0] is not plain
    assert _force_backend(Fit(None))[0] is _force_backend()[0]
