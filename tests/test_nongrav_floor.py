"""The sub-model floor of the nongrav discovery (`fit_object._floor_sub_models`), on stand-in
trials: no fit runs, so this is in the fast suite."""
import pytest

import pyroorb.fit_object as fo


class _Trial:
    """A trial as _floor_sub_models reads it: the chi2 record, and a fit standing for its solution."""
    def __new__(cls, chi2, fit):
        return {"chi2": chi2, "fit": fit}


def test_sub_model_floor_ignores_the_gap_between_two_integrations():
    """srp2 settled ON the yarkovsky solution: the restricted evaluation is that same orbit, and
    evaluates 0.07 below the delivered chi2 only because it is a different integration. Nothing
    was demonstrated, so nothing moves and the nesting of the records holds."""
    m = {"yarkovsky": _Trial(227.8259, "y"), "srp2": _Trial(227.8259, "s")}
    moved = fo._floor_sub_models(m, lambda fit, key: 227.7513)
    assert moved is False
    assert m["yarkovsky"]["chi2"] == 227.8259 and "chi2_floored_by" not in m["yarkovsky"]


def test_sub_model_floor_still_takes_a_real_improvement():
    """A richer trial whose orbit, with the extra axis zeroed, beats the sub-model's own solution
    on the same evaluation floors the sub-model by that difference, in the trial's own chi2."""
    m = {"yarkovsky": _Trial(300.0, "y"), "srp2": _Trial(250.0, "s")}
    moved = fo._floor_sub_models(m, lambda fit, key: {"y": 299.0, "s": 279.0}[fit])
    assert moved is True
    assert m["yarkovsky"]["chi2"] == pytest.approx(280.0)
    assert m["yarkovsky"]["chi2_floored_by"] == "srp2"
    assert m["srp2"]["chi2"] == 250.0


def test_sub_model_floor_skips_what_it_cannot_evaluate():
    """A failed trial has no solution, and a failed evaluation demonstrates nothing."""
    def boom(fit, key):
        raise RuntimeError("integration failed")
    m = {"yarkovsky": _Trial(300.0, "y"), "srp2": _Trial(250.0, "s"), "srp": _Trial(float("inf"), None)}
    assert fo._floor_sub_models(m, boom) is False
    assert m["yarkovsky"]["chi2"] == 300.0
