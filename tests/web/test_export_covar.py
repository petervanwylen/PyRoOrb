"""The /api/export/covar route: find_orb's covar.json for the displayed orbit.

The document itself is tested against a real fit in tests/test_egress_covar.py;
here we exercise the ROUTE -- that it is dispatched, that it is served as PLAIN TEXT
(the product's whole point is that it is byte-diffable against a real find_orb run, so
a JSON re-serialisation on the way out would defeat it), and that it obeys the
orbit-suppression refusal every other element export obeys. The handler is driven
directly (no socket), the same way test_export_state.py does it.
"""
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP = os.path.dirname(os.path.dirname(_HERE))

from pyroorb.web import server  # noqa: E402


class _StubFit:
    nongrav_model = None

    def __init__(self):
        self._state = np.array([1.1, -0.4, 0.2, 0.003, 0.012, -0.001])
        self._cov = np.diag([1e-8, 2e-8, 3e-8, 1e-12, 2e-12, 3e-12])

    def parameter_covariance(self):
        return self._cov.copy(), ["x", "y", "z", "dx", "dy", "dz"]

    def state(self):
        return self._state.copy()

    def epoch_jd(self):
        return 2460000.5


class _Recorder:
    def __init__(self, fit, refusal=None):
        self.fit = fit
        self.refusal = refusal
        self.sent = None
        self._export_covar = server.Handler._export_covar.__get__(self)

    def _resolve_fit(self, qs):
        if self.fit is None:
            raise ValueError("missing ?desig= or ?key=")
        return self.fit

    def _send(self, code, body, ctype="application/json"):
        self.sent = (code, body, ctype)


def _run(fit, refusal=None, monkeypatch=None):
    r = _Recorder(fit)
    if refusal is not None:
        monkeypatch.setattr(server, "_orbit_export_refusal", lambda f: refusal)
    else:
        monkeypatch.setattr(server, "_orbit_export_refusal", lambda f: None)
    r._export_covar({})
    return r.sent


def test_route_is_dispatched():
    import inspect
    body = inspect.getsource(server.Handler.do_GET)
    assert "/api/export/covar" in body and "_export_covar" in body, body[:400]


def test_served_as_plain_text_in_findorb_layout(monkeypatch):
    code, body, ctype = _run(_StubFit(), monkeypatch=monkeypatch)
    assert code == 200 and ctype.startswith("text/plain")
    assert body.startswith('{ "covar": [')          # find_orb's own opening bytes
    doc = json.loads(body)                          # still ordinary JSON
    assert list(doc) == ["covar", "state_vect", "epoch"]
    assert np.shape(doc["covar"]) == (6, 6)


def test_a_suppressed_orbit_is_refused(monkeypatch):
    """An export URL is a hand-openable GET, so the refusal lives on the server, not
    only in the hidden button -- the same rule the MPCORB/SOF/state exports obey."""
    code, body, ctype = _run(_StubFit(), refusal="no orbit: the arc does not determine one",
                             monkeypatch=monkeypatch)
    assert code == 200 and ctype.startswith("text/plain")
    assert "no orbit" in body and '"covar"' not in body


def test_unresolvable_fit_reports_the_failure(monkeypatch):
    code, body, ctype = _run(None, monkeypatch=monkeypatch)
    assert code == 200 and "export failed" in body


def test_the_menu_offers_it_and_suppression_hides_it():
    """The button and the suppression list must both know the kind -- a route with no
    menu entry is unreachable, and a menu entry outside ORBIT_EXPORTS would stay
    clickable on an orbit the server then refuses."""
    static = os.path.join(os.path.dirname(__import__("pyroorb.web", fromlist=["_"]).__file__), "static")
    assert 'data-export="covar"' in open(os.path.join(static, "index.html")).read()
    assert "covar:" in open(os.path.join(static, "app.js")).read()
    js = open(os.path.join(static, "orbitsuppress.js")).read()
    assert '"state", "covar"' in js
