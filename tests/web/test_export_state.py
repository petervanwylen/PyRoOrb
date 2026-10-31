"""The /api/export/state route: the cartesian state vector + 6x6 covariance the
Export menu hands back as JSON.

The document itself is tested against a real fit in tests/test_egress.py; here
we exercise the ROUTE -- that it is dispatched, that what comes back off the wire is
valid, pretty-printed JSON served as application/json, and that a degenerate fit
(NaN sigmas) still parses rather than emitting a bare NaN literal that JSON.parse
rejects. The handler is driven directly (no socket): BaseHTTPRequestHandler does its
work in __init__, so the test binds the bound methods onto a bare instance.
"""
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

from pyroorb.web import server  # noqa: E402


class _StubFit:
    """Enough of FittedOrbit for the state export."""
    designation = "2023 QJ3"
    designations = ["2023 QJ3"]
    packed_designation = "K23Q03J"
    nongrav_model = None

    def __init__(self, cov=None):
        self._state = np.array([1.1, -0.4, 0.2, 0.003, 0.012, -0.001])
        self._cov = np.eye(6) * 1e-8 if cov is None else cov

    def state(self):
        return self._state.copy()

    def covariance_matrix(self, frame="elements"):
        return self._cov.copy()

    def epoch_jd(self):
        return 2460000.5

    def n_observations(self):
        return 42

    def rms_residual(self):
        return 0.31


class _Recorder:
    """A handler stand-in that captures what _send was called with."""

    def __init__(self, fit):
        self.fit = fit
        self.sent = None
        self._export_state = server.Handler._export_state.__get__(self)

    def _resolve_fit(self, qs):
        if self.fit is None:
            raise ValueError("missing ?desig= or ?key=")
        return self.fit

    def _send(self, code, body, ctype="application/json"):
        self.sent = (code, body, ctype)


def _run(fit):
    r = _Recorder(fit)
    r._export_state({})
    return r.sent


def test_route_is_dispatched():
    """The GET dispatcher must actually reach _export_state -- an endpoint that only
    exists as a method is an endpoint nobody can call."""
    src = server.Handler.do_GET.__doc__ or ""
    import inspect
    body = inspect.getsource(server.Handler.do_GET)
    assert "/api/export/state" in body and "_export_state" in body, body[:400]


def test_state_export_is_pretty_printed_json():
    code, body, ctype = _run(_StubFit())
    assert code == 200 and ctype == "application/json"
    assert "\n  " in body, "expected an indented document for the copy/paste box"
    doc = json.loads(body)
    assert doc["object"] == "2023 QJ3"
    assert doc["covariance_order"] == ["x", "y", "z", "dx", "dy", "dz"]
    for frame in ("equatorial_J2000", "ecliptic_J2000"):
        blk = doc[frame]
        assert set(blk["state"]) == set(doc["covariance_order"])
        assert np.shape(blk["covariance"]) == (6, 6)
        assert np.shape(blk["correlation"]) == (6, 6)
    assert doc["units"] == {"position": "au", "velocity": "au/day"}
    assert doc["n_obs"] == 42


def test_degenerate_covariance_still_parses():
    """A NaN in the covariance (a hopelessly under-determined fit) must come back as
    null, not the JSON-invalid literal NaN."""
    cov = np.eye(6) * 1e-8
    cov[0, 0] = float("nan")
    code, body, _ = _run(_StubFit(cov=cov))
    assert "NaN" not in body, body[:200]
    doc = json.loads(body)                       # the real assertion: it parses
    assert doc["equatorial_J2000"]["sigma"]["x"] is None


def test_unresolvable_fit_returns_a_json_error():
    code, body, ctype = _run(None)
    assert code == 200 and ctype == "application/json"
    assert isinstance(body, dict) and "export failed" in body["error"]


def test_nongrav_fit_notes_the_marginalisation():
    """With an out-gassing model the 6x6 is a marginal covariance, and the document
    has to say so -- silently handing back six numbers from a 7-parameter fit is how
    a consumer under-reports its uncertainty."""
    fit = _StubFit()
    fit.nongrav_model = "comet"
    code, body, _ = _run(fit)
    doc = json.loads(body)
    assert doc["nongrav_model"] == "comet"
    assert "marginalised" in doc["nongrav_note"]


@pytest.mark.parametrize("frame", ["equatorial_J2000", "ecliptic_J2000"])
def test_covariance_is_symmetric_on_the_wire(frame):
    cov = np.diag([1e-8, 2e-8, 3e-8, 1e-12, 2e-12, 3e-12])
    cov[0, 1] = cov[1, 0] = 5e-9
    doc = json.loads(_run(_StubFit(cov=cov))[1])
    C = np.asarray(doc[frame]["covariance"])
    assert np.allclose(C, C.T)
