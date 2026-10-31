"""The a-pinned anchor fit must give the SAME orbit with the analytic Jacobian as with
scipy's finite-difference one.

`arc_extend._fit_anchor_fixed_a` is the multi-revolution bridge's hot loop: the a-secant
(`_bridge_far_by_a_secant`) calls it once per probe while root-finding the semi-major axis
that zeroes the far apparition's RA residual, and the recovery cascade re-runs the bridge
from every re-anchor. It used to hand scipy no `jac=`, so every LM step cost ~6 extra full
ASSIST sweeps; it now builds the analytic Jacobian from the backend's multi-particle sweep
and finite-differences ONLY the a-penalty row, through `quick_aeq`, which needs no
integration at all.

That is a pure conditioning change -- the minimum being sought is identical -- so the two
paths must agree to convergence tolerance. They are NOT bit-identical: the analytic backend
integrates the nominal orbit alongside 6 perturbed particles in ONE simulation and IAS15's
adaptive timestep is shared across them, so the nominal trajectory is stepped differently
than in a solo sweep. This test pins the agreement -- and thereby the Jacobian's row
ordering, scaling and penalty-row placement, which is what a plumbing mistake would break --
rather than byte-identity.

The inputs are the REAL ones: we run a fixture whose fit genuinely reaches the a-secant and
capture the (evaluator, seed, mask, a_target) of its bridge calls, instead of inventing an
anchor that might not resemble one.
"""
import os
import numpy as np
import pytest

import pyroorb.arc_extend as ae
from _fixtures import FIX, load_fixture

# runs a real (small) fit through the force model to capture live bridge inputs
pytestmark = pytest.mark.slow

# One of the nine regression objects whose fit reaches _fit_anchor_fixed_a at all, and the
# cheapest of them (~2 s): a 2-opposition NEO the multi-rev bridge has to a-secant across.
FIXTURE = "2008_KB12"


@pytest.fixture(scope="module")
def captured_calls():
    """(ev, seed, mask, a_target) for every a-secant probe of a real fixture fit."""
    import test_regression as tr

    calls = []
    original = ae._fit_anchor_fixed_a

    def recording(ev, seed, mask, a_target, **kw):
        calls.append((ev, np.array(seed, float), mask.copy(), float(a_target)))
        return original(ev, seed, mask, a_target, **kw)

    ae._fit_anchor_fixed_a = recording
    try:
        tr.refit(load_fixture(os.path.join(FIX, FIXTURE + ".json")))
    finally:
        ae._fit_anchor_fixed_a = original
    assert calls, f"{FIXTURE} no longer reaches the a-secant -- pick another fixture"
    return calls


def _numeric_jac_result(ev, seed, mask, a_target):
    """Run the SAME function down its finite-difference branch (the pre-change behaviour),
    by hiding the backend's analytic Jacobian for the duration of the call."""
    saved, ev._predict_jac = ev._predict_jac, None
    try:
        return ae._fit_anchor_fixed_a(ev, seed, mask, a_target)
    finally:
        ev._predict_jac = saved


def test_analytic_jac_matches_finite_difference_on_real_bridge_inputs(captured_calls):
    worst = 0.0
    for ev, seed, mask, a_target in captured_calls:
        assert ev._predict_jac is not None, "the backend must supply an analytic Jacobian"
        x_new = ae._fit_anchor_fixed_a(ev, seed, mask, a_target)
        x_old = _numeric_jac_result(ev, seed, mask, a_target)
        assert np.all(np.isfinite(x_new)) and np.all(np.isfinite(x_old))
        rel = float(np.max(np.abs(x_new - x_old) / (np.abs(x_old) + 1e-12)))
        worst = max(worst, rel)
        # the pinned quantity is what the secant root-finds on, so hold it tightest
        a_new = ae.quick_aeq(x_new)[0]
        a_old = ae.quick_aeq(x_old)[0]
        assert abs(a_new - a_old) < 1e-6, (
            f"a_target={a_target:.6f}: pinned a disagrees, {a_new!r} vs {a_old!r}")
    assert worst < 1e-4, f"states disagree by {worst:.2e} relative -- more than tolerance"


def test_the_pin_is_actually_enforced(captured_calls):
    """A wrong penalty-row gradient would still converge -- to an orbit that ignores the
    pin. Assert the analytic path really does land on the requested semi-major axis."""
    for ev, seed, mask, a_target in captured_calls:
        a_fit = ae.quick_aeq(ae._fit_anchor_fixed_a(ev, seed, mask, a_target))[0]
        assert np.isfinite(a_fit)
        assert abs(a_fit - a_target) < 1e-3 * abs(a_target), (
            f"a pinned to {a_target:.6f} but the fit came back at {a_fit:.6f}")
