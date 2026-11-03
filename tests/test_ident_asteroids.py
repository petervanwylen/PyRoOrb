"""An identification hypothesis ("is A = B?", >= 2 designations requested) is fitted with the
16 asteroid perturbers ON from the first step -- no gate, no keep-if-it-helps. The gated
polish ran only after the gravity linker had already refused the
apparitions that need the perturbers to join, so it could not undo a false "no".

Checks: the switch reaches the ctx backend (so later web steps stay +16ast), the A/B toggle's
gravity side really turns the perturbers off, a single-object fit is untouched, and the
self-perturber exclusion still applies inside an ident request (+15ast)."""
import os

import pytest

from _fixtures import load_fixture, FIX
import pyroorb.fit_object as fo
import pyroorb.force_assist as fa
from pyroorb import obs_sources

pytestmark = pytest.mark.skipif(fa.asteroid_kernel_path() is None,
                                reason="needs the sb441-n16 asteroid kernel")

_OC = None


def _oc():
    global _OC
    if _OC is None:
        _OC = obs_sources.load_obscodes()
    return _OC


def _fit(label, **meta_over):
    fx = load_fixture(os.path.join(FIX, label + ".json"))
    meta = {**fx["meta"], **meta_over}
    return fo.fit_orbit([dict(o) for o in fx["obs"]], _oc(),
                        desig=meta["designation"], meta=meta)


def test_ident_fit_runs_with_perturbers_and_keeps_them_in_ctx():
    fit = _fit("PAIR_2003QC91")                          # n_requested == 2
    assert fit.asteroids is True
    assert isinstance(fit._ctx["backend"], fo._AsteroidForce)
    assert fa.forced_perturbers() is None                # the context did not leak out
    # refine_asteroids (the web's gated polish) leaves an ident fit alone
    assert fo.refine_asteroids(fit) is fit


def test_toggle_off_on_an_ident_fit_is_really_gravity():
    fit = _fit("PAIR_2003QC91")
    grav = fo.refit_force_model(fit, False)
    assert grav.asteroids is False
    assert not isinstance(grav._ctx["backend"], fo._AsteroidForce)
    ast = fo.refit_force_model(fit, True)
    assert ast.asteroids is True


def test_single_object_request_is_untouched():
    fit = _fit("PAIR_2003QC91", n_requested=1)
    assert fit.asteroids is False
    assert not isinstance(fit._ctx["backend"], fo._AsteroidForce)


def test_self_perturber_inside_the_ident_context_is_excluded_not_dropped():
    # One of the 16 in an identification request: +15ast with itself excluded -- the
    # forcing wrapper sits OUTSIDE the exclusion, which must survive it.
    with fa.perturbers_on_for("ident"), fa.perturbers_off_for("4 Vesta"):
        backend, tag = obs_sources._force_backend()
        on = fa._use_asteroids(None)
    assert isinstance(backend, fo._AsteroidForce)
    assert isinstance(backend._backend, fa.SelfExcludedForce)
    assert ("+15ast" in tag) == on
    with fa.perturbers_on_for("ident"):
        assert fa._use_asteroids(None) is True
        assert fa._use_asteroids(False) is False          # an explicit off is still honoured
