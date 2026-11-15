"""The fit page's asteroid-field banner promises a difference the force-model toggle shows:
the 16 alone are announced only past the same-obs RMS margin, extra bodies always."""
from types import SimpleNamespace

from pyroorb import fit_object as fo
from pyroorb import perturbers as pt
from pyroorb.web.engine_api import announce_field

GRAV = SimpleNamespace(asteroids=False)


def kept(asteroids, rms_drop):
    return SimpleNamespace(asteroids=asteroids,
                           perturber_screen={"kept": True, "gain": 2.5, "rms_drop": rms_drop})


def test_sixteen_below_margin_is_not_announced():
    # 2011 WK115: chi2 gain 2.5, RMS 0.0009" lower -- toggling shows no change
    assert not announce_field(kept(True, 0.0009), GRAV)


def test_sixteen_past_margin_is_announced():
    assert announce_field(kept(True, fo.ASTEROID_KEEP_RMS + 0.01), GRAV)


def test_extra_body_announced_whatever_the_drop_over_gravity():
    # 352870: Papagena's pull cancels Ceres's, 0.012" over gravity alone
    assert announce_field(kept(pt.Field({471}), 0.012), GRAV)


def test_gated_polish_without_screen_record_is_announced():
    # refine_asteroids (the fallback) enforces the margin itself and records no rms_drop
    assert announce_field(SimpleNamespace(asteroids=True, perturber_screen={"skipped": "x"}), GRAV)


def test_gravity_is_never_announced():
    assert not announce_field(SimpleNamespace(asteroids=False, perturber_screen=None), GRAV)
