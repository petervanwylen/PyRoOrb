"""The three interactive solve actions must each do what their button says.

"Full step" (single=True) -- ONE damped differential-correction step (find_orb's `f`):
from an off-minimum seed it strictly improves but does not iterate; clicked repeatedly it
converges to the same minimum the "Multi-step" (polish) solve reaches; at that minimum it
is an honest (near-)no-op. "Re-auto-solve" re-runs the full automatic pipeline on the open
fit's obs and returns a fresh session.

    python tests/web/test_step_modes.py
"""
import pytest
import os, sys, warnings; warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import _fixtures as FX
import pyroorb.web.engine_api as E

# each solve action drives a real engine fit on a cached fixture
pytestmark = pytest.mark.slow

FIXTURE = "2008_TE4"
KICK = 3e-7          # relative seed offset: in the convergence basin, well off the min


def _session():
    fx = FX.load_fixture(os.path.join(FX.FIX, FIXTURE + ".json"))
    payload = E.fit_obs_list(fx["obs"], fx["meta"], E._obscodes(), force=fx.get("force", False))
    sid = payload.get("session")
    assert sid, f"fit failed: {payload.get('error')}"
    s = E._SESSIONS[sid]
    s["model"] = None                 # exercise the plain-gravity branch
    s["constraints"] = {}
    return sid, s


def test_single_step_is_one_honest_step():
    """One step from an off-min seed improves; repeated steps converge to the
    multi-step minimum; a step AT the minimum barely moves (the no-op indicator)."""
    sid, s = _session()
    ev = s["ctx"]["ev"]
    mask = np.asarray(s["grav_fit"]._mask, dtype=bool)
    min_state, _ = ev.fit(np.asarray(s["state"], dtype=float), mask, thin=False, polish=True)
    min_rms = ev.rms_norm(min_state, mask)

    off = min_state * (1.0 + KICK)
    off_rms = ev.rms_norm(off, mask)
    st1, res1 = ev.step(off, mask)
    rms1 = ev.rms_norm(st1, mask)
    assert rms1 < off_rms, f"single step did not improve: {rms1:.5f} vs seed {off_rms:.5f}"

    st = st1
    for _ in range(24):                            # click repeatedly -> converge
        st, _ = ev.step(st, mask)
    walked = ev.rms_norm(st, mask)
    assert walked <= min_rms * 1.02, \
        f"repeated single steps missed the minimum: {walked:.5f} vs {min_rms:.5f}"

    at_min, _ = ev.step(min_state, mask)
    drift = ev.rms_norm(at_min, mask)
    assert drift <= min_rms * (1 + 1e-6), f"step at the minimum worsened it: {drift:.6f}"
    print(f"OK single: seed={off_rms:.5f} 1-step={rms1:.5f} walked={walked:.5f} "
          f"min={min_rms:.5f} at-min={drift:.5f}")


def test_fullstep_endpoint_single_and_multi():
    """End-to-end through full_step: both modes return ok payloads; the single mode
    honours the mask exactly like the multi mode does."""
    sid, s = _session()
    mask = np.asarray(s["grav_fit"]._mask, dtype=bool).tolist()
    p1 = E.full_step(sid, mask, single=True)
    assert p1.get("ok"), p1.get("error")
    p2 = E.full_step(sid, mask, single=False)
    assert p2.get("ok"), p2.get("error")
    r1 = p2["summary"]["rms_normalized"]
    print(f"OK endpoint: single ok, multi ok (rms_norm={r1:.5f})")


def test_reauto_solve_fresh_session():
    """Re-auto-solve returns a FRESH session over the same obs set (the pipeline's own
    auto-rejection decides inclusion afresh)."""
    sid, s = _session()
    n_total = len(s["fit"]._obs)
    p = E.reauto_solve(session=sid)
    assert p.get("ok"), p.get("error")
    assert p.get("session") and p["session"] != sid, "expected a fresh session"
    assert p["summary"]["n_obs_total"] == n_total, \
        f"obs set changed: {p['summary']['n_obs_total']} vs {n_total}"
    assert p.get("meta", {}).get("source") == "reauto"
    print(f"OK reauto: fresh session, {p['summary']['n_obs']}/{n_total} used, "
          f"rms={p['summary']['rms_residual']:.3f}")


if __name__ == "__main__":
    test_single_step_is_one_honest_step()
    test_fullstep_endpoint_single_and_multi()
    test_reauto_solve_fresh_session()
    print("all passed")


def test_stale_client_mask_is_refused_not_truncated():
    """A mask whose length is not the fit's obs count describes a DIFFERENT arc.

    The client builds it positionally from the residual table it last rendered, so a
    stale view (a second tab, an add-to-fit in between) sends the wrong shape. Nothing
    downstream noticed: np.flatnonzero on a short mask silently yields only the positions
    it covers, so the engine solved a TRUNCATED arc and then died with an opaque
    IndexError while building the products. Every interactive endpoint that takes a mask
    now says what is actually wrong, before any fitting.
    """
    sid, s = _session()
    n = len(s["grav_fit"]._obs)
    full = np.asarray(s["grav_fit"]._mask, dtype=bool).tolist()
    for name, call in (("full_step", lambda m: E.full_step(sid, m)),
                       ("set_asteroids", lambda m: E.set_asteroids(sid, False, mask_list=m)),
                       ("set_epoch", lambda m: E.set_epoch(sid, 2460000.5, mask_list=m))):
        for bad in (full[:-3], full + [True, True]):
            p = call(bad)
            assert not p.get("ok") and p.get("stale"), f"{name} accepted a {len(bad)}/{n} mask: {p}"
            assert "changed" in p["error"]
        assert call(full).get("ok"), f"{name} refused the aligned mask"
    print(f"OK stale-mask guard: 3 endpoints x 2 shapes refused, aligned mask still fits ({n} obs)")
