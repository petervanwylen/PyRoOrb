"""Unit tests for the self-perturbing-asteroid guard (the "+16ast" analogue of the
Pluto ban).

The 16 massive main-belt asteroids in the sb441-n16 perturber kernel ARE the bodies
the +16ast force model adds. If the object being fit is itself one of those 16, turning
the perturbers on would propagate it in a field that already contains it -- it would be
perturbed by itself, exactly the Pluto paradox. Unlike Pluto (always a DE440 perturber,
so banned from being fit at all), the cure here is to EXCLUDE the object from the field and
keep the other 15 ("+15ast": (4) Vesta is fit with Ceres). These tests pin the identity
layer (every form the object can arrive in), the look-alikes that must NOT match, the
FittedOrbit/obs-based detection refine_asteroids / refit_force_model use, the one-body
exclusion and its fallback, and -- when the real kernel is present -- that the hard-coded
set matches it and that the excluded field neither self-perturbs nor drops the other 15.
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
# Enforcement: an object that IS one of the 16 is integrated with ITSELF excluded from the
# perturber field -- the other 15 stay ("+15ast"). Whole-fit +16ast mode (PYROORB_ASTEROIDS_PATH)
# turns the perturbers on globally, which is how (1) Ceres came to be fit inside a field
# containing itself (19 of 7993 obs kept, the modern arc reading ~1e5 sigma); until 2026-11-02
# the cure switched all 16 off. These pin the one-body exclusion that replaced it.
# ---------------------------------------------------------------------------
def _real_kernel():
    import pyroorb.force_assist as fa
    if not os.path.exists(fa.PLANETS_PATH) or fa.asteroid_kernel_path() is None:
        pytest.skip("DE440 + sb441-n16 kernels not present")
    return fa


def test_context_names_the_excluded_body():
    import pyroorb.force_assist as fa
    assert fa.suppressed_perturber() is None and fa._excluded_number() is None
    with fa.perturbers_off_for("4 Vesta"):
        assert fa.suppressed_perturber() == "4 Vesta"
        assert fa._excluded_number() == 4
    assert fa.suppressed_perturber() is None


def test_without_a_kernel_the_context_stays_gravity(monkeypatch):
    import pyroorb.force_assist as fa
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: None)
    monkeypatch.setattr(fa, "_field_kernel", lambda: None)   # nor the bundled kernel's 16
    with fa.perturbers_off_for("1 Ceres"):
        assert fa._use_asteroids(True) is False
        assert fa._epsilon(True) == fa.IAS15_EPSILON


def test_unverified_exclusion_falls_back_to_all_off(monkeypatch, capsys):
    """If the ctypes view of ASSIST's target table does not verify (a future ASSIST layout),
    the object must NOT be integrated in the full field: the perturbers stay off for it --
    the pre-exclusion guard -- with a one-time notice."""
    import pyroorb.force_assist as fa
    built = []

    class FakeEphem:
        def __init__(self, *a):
            built.append(a)

    monkeypatch.setattr(fa.assist, "Ephem", FakeEphem)
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: "ast.bsp")
    monkeypatch.setattr(fa, "ensure_planets_kernel", lambda *a, **k: "planets")
    monkeypatch.setattr(fa, "zero_asteroid_gm", lambda eph, n: False)
    monkeypatch.setattr(fa, "_EPHEM_EXCL", {})
    monkeypatch.setattr(fa, "_EXCL_WARNED", set())
    with fa.perturbers_off_for("1 Ceres"):
        assert fa._use_asteroids(True) is False
        assert fa._use_asteroids(True) is False          # cached: built once
        assert fa._epsilon(True) == fa.IAS15_EPSILON
    assert len(built) == 1
    assert "could not exclude asteroid 1" in capsys.readouterr().out
    # an unrelated object (no context) is unaffected by Ceres's failed exclusion
    assert fa._use_asteroids(True) is True


def test_layout_check_refuses_a_table_that_does_not_read_as_asteroids():
    """zero_asteroid_gm must refuse -- not write -- when the target codes are not the
    kernel's 2000000 + MP numbers (i.e. the struct view is misaligned)."""
    import pyroorb.force_assist as fa

    class NoTable:
        spk_asteroids = None
    assert fa.zero_asteroid_gm(NoTable(), 4) is False


def test_excluding_wrapper_runs_every_call_in_the_context():
    import pyroorb.force_assist as fa
    seen = {}

    class Backend:
        marker = "passthrough"

        def predict_state(self, *a, asteroids=None, **k):
            seen["asteroids"] = asteroids
            seen["excluded"] = fa.suppressed_perturber()
            return "state"

        def ensure_asteroid_kernel(self, verbose=False):
            seen["kernel_ctx"] = fa.suppressed_perturber()
            return "k"

    b = fa.SelfExcludedForce(Backend(), "4 Vesta")
    assert b.predict_state(1, 2, asteroids=True) == "state"
    assert seen == {"asteroids": True, "excluded": "4 Vesta"}   # flag passes through
    b.ensure_asteroid_kernel()
    assert seen["kernel_ctx"] == "4 Vesta"
    assert b.marker == "passthrough"               # non-callables fall through
    assert fa.suppressed_perturber() is None       # the context does not leak out
    again = fa.SelfExcludedForce(b, "4 Vesta")
    assert again._backend is b._backend            # idempotent


def test_asteroid_wrapper_keeps_the_exclusion(monkeypatch):
    """_AsteroidForce unwraps another _AsteroidForce (idempotent) but must not unwrap the
    self-perturber exclusion: wrapping it means perturbers ON with the object excluded.
    _self_excluding (the refine_asteroids / refit_force_model gate) also requires that the
    exclusion can be BUILT, or a fit labelled asteroids would really be gravity."""
    import pyroorb.fit_object as fo
    import pyroorb.force_assist as fa
    seen = {}

    class Backend:
        def predict_state(self, *a, asteroids=None, **k):
            seen["asteroids"] = asteroids
            seen["excluded"] = fa.suppressed_perturber()

    excl = fa.SelfExcludedForce(Backend(), "1 Ceres")
    wrapped = fo._AsteroidForce(excl)
    assert wrapped._backend is excl
    wrapped.predict_state(1)
    assert seen == {"asteroids": True, "excluded": "1 Ceres"}
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: "ast.bsp")
    monkeypatch.setattr(fa, "_masked_ephem", lambda n: object())
    assert fo._self_excluding(wrapped) and fo._self_excluding(excl)
    assert not fo._self_excluding(Backend()) and not fo._self_excluding(None)
    monkeypatch.setattr(fa, "_masked_ephem", lambda n: None)        # exclusion unavailable
    assert not fo._self_excluding(wrapped) and not fo._self_excluding(excl)


def test_force_backend_hands_back_an_excluding_backend_in_context(monkeypatch, tmp_path):
    """The backend the geometry build lands in the fit's ctx must exclude a perturber object
    -- in whole-fit mode AND gravity-default mode (a later polish / toggle may turn the
    perturbers on) -- which is what makes the exclusion outlive the fit."""
    import pyroorb.force_assist as fa
    from pyroorb.obs_sources import _force_backend, _self_excluded
    if not os.path.exists(fa.PLANETS_PATH):
        pytest.skip("DE440 planets kernel not present")
    backend, tag = _force_backend()
    assert not _self_excluded(backend)
    with fa.perturbers_off_for("1 Ceres"):
        backend, tag = _force_backend()            # gravity default
    assert _self_excluded(backend) and tag == "ASSIST(DE440+GR)"
    kern = tmp_path / "sb441-n16.bsp"; kern.write_text("x", encoding="utf-8")
    monkeypatch.setattr(fa, "ASTEROIDS_PATH", str(kern))
    monkeypatch.setattr(fa, "_masked_ephem", lambda n: object())
    backend, tag = _force_backend()
    assert not _self_excluded(backend) and "+16ast)" in tag
    with fa.perturbers_off_for("1 Ceres"):
        backend, tag = _force_backend()
    assert _self_excluded(backend) and tag == "ASSIST(DE440+GR+15ast, excl. 1 Ceres)"
    monkeypatch.setattr(fa, "_masked_ephem", lambda n: None)     # exclusion unavailable
    with fa.perturbers_off_for("1 Ceres"):
        backend, tag = _force_backend()
    assert _self_excluded(backend) and "self-perturber" in tag


def test_excluding_backend_is_preferred_for_products():
    """An ephemeris / impact scan resolves the backend itself; it must take the fit's
    excluding one when there is one, or it would recompute in a field containing the object."""
    import pyroorb.force_assist as fa
    from pyroorb.obs_sources import _force_backend

    class Fit:
        force_model = "ASSIST(DE440+GR+15ast, excl. 1 Ceres)"

        def __init__(self, backend): self._ctx = {"backend": backend}

    plain = object()
    excl = fa.SelfExcludedForce(plain, "1 Ceres")
    assert _force_backend(Fit(excl)) == (excl, Fit.force_model)
    # an ordinary fit gets its OWN backend too (it carries that fit's pinned force model);
    # only a fit without one (and no fit at all) resolves the backend the usual way
    assert _force_backend(Fit(plain))[0] is plain
    assert _force_backend(Fit(None))[0] is _force_backend()[0]


def test_masked_ephem_zeroes_exactly_one_gm():
    """On the real kernel: the excluded ephem has the object's GM at 0 and the other 15
    unchanged, and the SHARED full ephem is untouched."""
    fa = _real_kernel()
    full = fa.get_ephem(True)
    with fa.perturbers_off_for("4 Vesta"):
        masked = fa.get_ephem(True)
        assert fa._epsilon(True) == fa.IAS15_EPSILON_ASTEROIDS
    assert masked is not full
    a = {t.code - 2000000: t.mass for t in fa._spk_table(full)}
    b = {t.code - 2000000: t.mass for t in fa._spk_table(masked)}
    assert set(a) == set(guards.N16_PERTURBERS)
    assert a[4] > 0 and b[4] == 0.0
    assert {k: v for k, v in a.items() if k != 4} == {k: v for k, v in b.items() if k != 4}


def test_excluded_field_does_not_self_perturb_and_keeps_the_other_15():
    """A particle 1 km from Vesta's kernel position: in the full 16-body field it is flung
    away by Vesta's own point mass; with Vesta excluded it follows a Vesta orbit, and the
    other 15 still act on it (it differs from the gravity-only run)."""
    import numpy as np
    fa = _real_kernel()
    full = fa.get_ephem(True)
    codes = [t.code for t in fa._spk_table(full)]
    iv = fa._ASSIST_BODY_NPLANETS + codes.index(2000004)

    def pos(t):
        p = full.get_particle(iv, t - full.jd_ref)
        s = full.get_particle("Sun", t - full.jd_ref)
        return np.array([p.x - s.x, p.y - s.y, p.z - s.z])

    au_km = 149597870.7
    t0, t1, h = 2460000.5, 2460000.5 + 365.25, 0.01
    s0 = np.concatenate([pos(t0) + [1 / au_km, 0, 0], (pos(t0 + h) - pos(t0 - h)) / (2 * h)])
    truth = pos(t1)
    selfp = fa.predict_state(None, t0, s0, t1, asteroids=True)
    with fa.perturbers_off_for("4 Vesta"):
        excl = fa.predict_state(None, t0, s0, t1, asteroids=True)
        grav = fa.predict_state(None, t0, s0, t1, asteroids=False)
    km = lambda v: float(np.linalg.norm(v)) * au_km
    assert not np.all(np.isfinite(selfp)) or km(selfp[:3] - truth) > 1e6
    assert km(excl[:3] - truth) < 1000
    assert km(excl[:3] - grav[:3]) > 0.01           # the other 15 are acting
