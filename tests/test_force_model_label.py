"""The force_model tag follows the perturbers through the asteroid polish and the toggle.

refine_asteroids / refit_force_model rebuild the orbit from the original fit context, so
without an explicit relabel a polished orbit reports the gravity model it started from
while `fit.asteroids` says True. The label is derived from the context's own tag, so the
rebound backend (no perturbers) and the self-excluded perturber objects keep their spelling.
"""
from pyroorb import fit_object as fo


class _Backend:
    pass


def test_label_follows_the_perturbers(monkeypatch):
    ctx = {"backend_name": "ASSIST(DE440+GR)", "backend": _Backend()}
    monkeypatch.setattr(fo, "self_perturber_name", lambda fit=None: None)
    assert fo._force_model_label(ctx, True, fit=object()) == "ASSIST(DE440+GR+16ast)"
    assert fo._force_model_label(ctx, False, fit=object()) == "ASSIST(DE440+GR)"
    # a whole-fit +16ast context toggled off is relabelled, not left as it was
    ctx16 = {"backend_name": "ASSIST(DE440+GR+16ast)", "backend": _Backend()}
    assert fo._force_model_label(ctx16, False, fit=object()) == "ASSIST(DE440+GR)"
    assert fo._force_model_label(ctx16, True, fit=object()) == "ASSIST(DE440+GR+16ast)"


class _Excluding:
    """Marked the way force_assist.SelfExcludedForce marks itself."""
    def __init__(self, name):
        self._self_excluded = name


def test_rebound_and_self_excluded_keep_their_spelling(monkeypatch):
    reb = {"backend_name": "rebound(8-planet)", "backend": _Backend()}
    assert fo._force_model_label(reb, True, fit=object()) == "rebound(8-planet)"
    # the exclusion is read off the BACKEND (forcemodel.excluded_body), through any pin
    ctx = {"backend_name": "ASSIST(DE440+GR)",
           "backend": fo.PinnedForce(_Excluding("4 Vesta"), False)}
    monkeypatch.setattr(fo, "_self_excluding", lambda backend: True)
    assert fo._force_model_label(ctx, True, fit=object()) == "ASSIST(DE440+GR+15ast, excl. 4 Vesta)"
    # an exclusion that cannot be built here runs gravity, and says so
    monkeypatch.setattr(fo, "_self_excluding", lambda backend: False)
    assert fo._force_model_label(ctx, True, fit=object()) == \
        "ASSIST(DE440+GR, +16ast off: self-perturber)"
    # a backend that excludes nothing is the ordinary +16ast model
    plain = {"backend_name": "ASSIST(DE440+GR)", "backend": _Backend()}
    assert fo._force_model_label(plain, True, fit=object()) == "ASSIST(DE440+GR+16ast)"
