"""The force model is STATE on the fit's backend, fixed when the fit is made.

Before: the evaluator's predict passed `asteroids=None` ("the process-wide default"), so a fit
made inside ``with pyroorb.asteroid_perturbers():`` re-corrected its LOV / range profile /
re-anchor planets-only after the block closed; ``refit_force_model(fit, False)`` relabelled a
perturbed fit "gravity" while reusing its perturbed evaluator; the products told the models
apart by a regex on the display label; and three different walks detected the self-exclusion
wrapper. Now forcemodel.PinnedForce carries the choice, with_force_model is the one swap, and
forcemodel.excluded_body the one predicate.
"""
import inspect
import types

import numpy as np
import pytest

import pyroorb.fit_object as fo
import pyroorb.force_assist as fa
import pyroorb.forcemodel as fm
from pyroorb import obs_sources, products


class _Assist:
    """A stand-in ASSIST backend: records the asteroids= each call was made with."""
    def __init__(self):
        self.calls = []

    def ensure_asteroid_kernel(self, verbose=False):
        return "kernel"

    def predict_radec_batch(self, *a, asteroids=None, **k):
        self.calls.append(("predict_radec_batch", asteroids))
        return asteroids

    def predict_radec_jac_batch(self, *a, asteroids=None, **k):
        self.calls.append(("predict_radec_jac_batch", asteroids))
        return asteroids

    def build_base_sim(self, t):
        return None


class _Rebound:
    """The rebound fallback: no perturbers, no asteroids= parameter."""
    def predict_radec_batch(self, *a, **k):
        assert "asteroids" not in k
        return "reb"


# --- the call list ---------------------------------------------------------------------------
def test_every_asteroid_aware_backend_call_is_pinned():
    """Every force_assist function taking `asteroids=` is a backend call the pin covers, except
    the three internal resolvers. predict_radec_jac_batch was missing: an asteroid-forced
    delivery's Jacobian ran in whatever the global switch said."""
    internal = {"_epsilon", "_use_asteroids", "get_ephem"}
    takes = {n for n, f in inspect.getmembers(fa, inspect.isfunction)
             if f.__module__ == fa.__name__ and "asteroids" in inspect.signature(f).parameters}
    assert takes - internal == set(fm.ASTEROID_AWARE_CALLS)
    assert fa.ASTEROID_AWARE_CALLS is fm.ASTEROID_AWARE_CALLS


# --- the pin -----------------------------------------------------------------------------
def test_the_pin_fills_the_default_and_an_explicit_choice_wins():
    b = _Assist()
    p = fm.PinnedForce(b, True)
    assert p.predict_radec_batch(1) is True                    # left out -> the pin
    assert p.predict_radec_batch(1, asteroids=None) is True    # "the global default" -> the pin
    assert p.predict_radec_batch(1, asteroids=False) is False  # an explicit choice wins
    assert p.predict_radec_jac_batch(1) is True
    assert fm.pinned_asteroids(p) is True and fm.pinned_asteroids(b) is None
    assert p.build_base_sim(0) is None and p.ensure_asteroid_kernel() == "kernel"


def test_rewrapping_replaces_the_pin_but_keeps_an_exclusion():
    excl = fa.SelfExcludedForce(_Assist(), "4 Vesta")
    p = fm.PinnedForce(fm.PinnedForce(excl, False), True)
    assert p._backend is excl                                  # one pin, the exclusion inside
    assert fm.pinned_asteroids(p) is True
    assert fm.excluded_body(p) == "4 Vesta" and fm.excluded_body(excl) == "4 Vesta"
    assert fm.excluded_body(fm.PinnedForce(_Assist(), True)) is None
    assert fm.excluded_body(None) is None
    # the _AsteroidForce wrapper is the True pin, and keeps the exclusion too
    a = fo._AsteroidForce(p)
    assert fm.pinned_asteroids(a) is True and fm.excluded_body(a) == "4 Vesta"


def test_the_rebound_fallback_is_never_pinned():
    r = _Rebound()
    assert fm.pin(r, True) is r and fm.pin(r, False) is r
    assert fm.pin(r, True).predict_radec_batch(0) == "reb"


# --- the geometry build pins the choice at fit time --------------------------------------
@pytest.mark.parametrize("on", [True, False])
def test_the_fit_context_pins_the_switch_at_fit_time(monkeypatch, on):
    b = _Assist()
    monkeypatch.setattr(fo, "_force_backend", lambda: (b, "ASSIST(DE440+GR)"))
    monkeypatch.setattr(fa, "_use_asteroids", lambda a: on)
    obs = [{"jd_utc": 2460000.5 + k, "ra_deg": 10.0 + k, "dec_deg": 5.0, "obscode": "500"}
           for k in range(3)]
    geom = fo._observer_geometry(obs, obs_sources.load_obscodes())
    assert fm.pinned_asteroids(geom["backend"]) is on
    # the switch flipping afterwards changes nothing: the pin answers, not the global
    monkeypatch.setattr(fa, "_use_asteroids", lambda a: not on)
    assert geom["backend"].predict_radec_batch(0) is on
    assert geom["jac_fn"](0) is on                             # the Jacobian too


# --- the one force-model swap ------------------------------------------------------------
def _ctx(backend, meta=None):
    ev = types.SimpleNamespace(_predict=None, _predict_jac=lambda *a, **k: None,
                               _resid_key=1, _sub_key=1, nongrav=None)
    return {"ev": ev, "backend": backend, "backend_name": "ASSIST(DE440+GR)",
            "meta": dict(meta or {})}


def test_toggling_off_really_turns_the_perturbers_off():
    """refit_force_model's "off" reused the fit's own evaluator and only relabelled it."""
    b = _Assist()
    ctx = _ctx(fm.PinnedForce(b, True), {"asteroids": True, "designations": ["x"]})
    ev2, ctx2 = fo.with_force_model(ctx, False)
    assert ev2 is not ctx["ev"]
    assert ev2._predict() is False and ev2._predict_jac() is False
    assert fm.pinned_asteroids(ctx2["backend"]) is False
    assert ctx2["asteroids"] is False and "asteroids" not in ctx2["meta"]
    assert ctx2["meta"]["designations"] == ["x"] and ctx2["backend_name"] == "ASSIST(DE440+GR)"
    assert ev2._resid_key is None and ev2._sub_key is None     # memo dropped


def test_toggling_on_binds_backend_flag_meta_and_label_together():
    ctx = _ctx(fm.PinnedForce(_Assist(), False))
    ev2, ctx2 = fo.with_force_model(ctx, True)
    assert ev2._predict() is True and ev2._predict_jac() is True
    assert ctx2["asteroids"] is True and ctx2["meta"]["asteroids"] is True
    assert ctx2["backend_name"] == "ASSIST(DE440+GR+16ast)"
    assert fm.pinned_asteroids(ctx["backend"]) is False        # the original is untouched


def test_an_exclusion_that_cannot_be_built_records_gravity(monkeypatch):
    excl = fa.SelfExcludedForce(_Assist(), "1 Ceres")
    monkeypatch.setattr(fo, "_self_excluding", lambda backend: False)
    ev2, ctx2 = fo.with_force_model(_ctx(fm.PinnedForce(excl, False)), True)
    assert ctx2["asteroids"] is False and "asteroids" not in ctx2["meta"]
    assert fm.pinned_asteroids(ctx2["backend"]) is False
    monkeypatch.setattr(fo, "_self_excluding", lambda backend: True)
    ev2, ctx2 = fo.with_force_model(_ctx(fm.PinnedForce(excl, False)), True)
    assert ctx2["asteroids"] is True
    assert ctx2["backend_name"] == "ASSIST(DE440+GR+15ast, excl. 1 Ceres)"
    assert fm.excluded_body(ctx2["backend"]) == "1 Ceres"


def test_the_rebound_fallback_cannot_be_toggled_on():
    ev2, ctx2 = fo.with_force_model(_ctx(_Rebound()), True)
    assert ctx2["asteroids"] is False and ev2._predict(0) == "reb"


def test_refine_asteroids_leaves_a_fit_already_made_with_the_perturbers():
    fit = types.SimpleNamespace(_ctx=_ctx(fm.PinnedForce(_Assist(), True)))
    assert fo.refine_asteroids(fit, force=True) is fit


# --- products and backend resolution read the fit, not the switch or the label ------------
def test_a_fitted_orbit_resolves_to_its_own_backend():
    b = fm.PinnedForce(_Assist(), True)
    fit = types.SimpleNamespace(_ctx={"backend": b}, force_model="whatever")
    assert obs_sources._force_backend(fit) == (b, "whatever")


def test_products_read_the_pin_not_the_label():
    on = types.SimpleNamespace(_ctx={"backend": fm.PinnedForce(_Assist(), True)},
                               asteroids=False, force_model="ASSIST(DE440+GR)")
    off = types.SimpleNamespace(_ctx={"backend": fm.PinnedForce(_Assist(), False)},
                                asteroids=True, force_model="ASSIST(DE440+GR+16ast)")
    assert products.fit_asteroids(on) is True and products.fit_asteroids(off) is False
    bare = types.SimpleNamespace(asteroids=True, force_model="")
    assert products.fit_asteroids(bare) is True                # no ctx: the fit's own flag


# --- end to end: whole-fit mode survives leaving the block --------------------------------
@pytest.mark.slow
def test_a_whole_fit_run_keeps_its_perturbers_after_the_block():
    """The bug as a user meets it: fit inside ``with asteroid_perturbers():``, leave the
    block, and every re-correction must still be perturbed."""
    if fa.asteroid_kernel_path() is None or not fa.Path(fa.PLANETS_PATH).exists():
        pytest.skip("DE440 + sb441-n16 kernels not present")
    import pyroorb
    from _fixtures import fixture_obs
    obs, meta = fixture_obs("2023_QJ3")
    oc = obs_sources.load_obscodes()
    with pyroorb.asteroid_perturbers():
        fit = fo.fit_orbit(obs, oc, desig="2023 QJ3", meta=meta)
    # the switch is off from here on
    assert fit.asteroids is True and products.fit_asteroids(fit) is True
    assert fm.pinned_asteroids(fit._ctx["backend"]) is True
    state = np.asarray(fit.state(), float)[:6]
    ev = fit._ctx["ev"]

    def resid(e):
        e._resid_key = e._sub_key = None                       # never a memo from before
        return np.concatenate(e.resid_arcsec(state))

    r_after = resid(ev)
    with pyroorb.asteroid_perturbers():
        r_inside = resid(ev)
    evg, ctxg = fo._build_evaluator(obs, oc, fit.epoch_tdb)    # built now: gravity
    assert fm.pinned_asteroids(ctxg["backend"]) is False
    r_grav = resid(evg)
    assert np.allclose(r_after, r_inside, atol=1e-9, rtol=0)   # pinned: same field either way
    assert np.max(np.abs(r_after - r_grav)) > 1e-6             # and it IS the perturbed field
    # toggling the fit off really is gravity now
    g = fo.refit_force_model(fit, False)
    assert g.asteroids is False and products.fit_asteroids(g) is False
    assert g.force_model == "ASSIST(DE440+GR)"


# --- the ctypes exclusion: the release gate comes before any pointer is followed ---------
def test_only_verified_assist_releases_are_read():
    assert fa.spk_layout_supported("1.2.3") and fa.spk_layout_supported("1.2")
    for v in ("1.3.0", "1.20.1", "1.1.9", "2.0", "", "1.2x"):
        assert not fa.spk_layout_supported(v), v
    assert fa.spk_layout_supported() == fa.spk_layout_supported(fa.assist.__version__)


def test_an_unverified_release_never_dereferences(monkeypatch):
    class _Untouchable:
        @property
        def spk_asteroids(self):
            raise AssertionError("the asteroid table pointer was read")
    monkeypatch.setattr(fa.assist, "__version__", "1.3.0")
    assert fa._spk_table(_Untouchable()) == []
    assert fa.zero_asteroid_gm(_Untouchable(), 4) is False


def test_an_unverified_release_builds_no_masked_ephem(monkeypatch, capsys):
    monkeypatch.setattr(fa.assist, "__version__", "1.3.0")
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: "kernel.bsp")
    monkeypatch.setattr(fa, "_EPHEM_EXCL", {})
    monkeypatch.setattr(fa, "_EXCL_WARNED", set())

    def _no_ephem(*a, **k):
        raise AssertionError("an Ephem was built for an unverified ASSIST")
    monkeypatch.setattr(fa.assist, "Ephem", _no_ephem)
    assert fa._masked_ephem(4) is None
    out = capsys.readouterr().out
    assert "ASSIST 1.3.0 is not a release whose struct layout is verified" in out
    # and so the perturbers stay off for that object: no field containing it, ever
    monkeypatch.setattr(fa, "_asteroids_on", lambda: True)
    with fa.perturbers_off_for("4 Vesta"):
        assert fa._use_asteroids(True) is False


def test_the_excluded_number_is_the_guards_number_not_a_string_split():
    with fa.perturbers_off_for("10 Hygiea"):
        assert fa._excluded_number() == 10
    with fa.perturbers_off_for("(4) Vesta"):
        assert fa._excluded_number() == 4
    assert fa._excluded_number() is None


def test_the_pyproject_assist_range_is_the_verified_one():
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    spec = re.search(r'"assist([^"]*)"', text).group(1)
    assert spec == ">=1.2,<1.3" and fa.SPK_LAYOUT_VERIFIED == ("1.2",)


def test_pinning_to_the_same_value_keeps_the_wrapper():
    """An identification fit's forcing _AsteroidForce survives the fit-time pin as itself."""
    forced = fo._AsteroidForce(_Assist())
    assert fm.pin(forced, True) is forced
    off = fm.pin(forced, False)
    assert off is not forced and fm.pinned_asteroids(off) is False
    assert not isinstance(off, fo._AsteroidForce)
