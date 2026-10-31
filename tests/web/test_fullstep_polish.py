"""The interactive Full step (engine_api.full_step -> _apply_interactive plain-gravity
branch) must DILIGENTLY re-solve, not freeze near its seed.

Regression guard for the "frozen Full step" bug. The endpoint seeds LM from the previous
converged orbit and then honours a changed obs mask, but with the loose final tolerance
MINPACK's badly-scaled first step overshot, the trust region collapsed below xtol*|x|, and
lmder returned the seed essentially UNCHANGED (nfev=2) while only the covariance/H updated
-- so toggling observations changed nothing about the orbit. Evaluator.fit(polish=True) now
runs a tight-tolerance re-seed loop that drives the seed to the minimum for the current mask.

The core guard (`test_polish_beats_loose_from_offmin_seed`) is object-independent: from an
identical off-minimum seed the loose solve (the old default) stays stranded above the
minimum, while the polish solve recovers it.

    python tests/web/test_fullstep_polish.py
    python -m pytest tests/web/test_fullstep_polish.py
"""
import pytest
import os, sys, warnings; warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import _fixtures as FX
import pyroorb.web.engine_api as E

# compares a full-step chain against the polish solve (real fits)
pytestmark = pytest.mark.slow

# A well-determined arc: its covariance is so tight that a tiny seed offset blows the
# residuals up (hundreds of sigma), so the loose solve visibly strands above the minimum
# while polish recovers it -- a clean, deterministic contrast.
FIXTURE = "2008_TE4"
KICK = 3e-7          # relative seed offset: in the convergence basin, but well off the min


def _session():
    fx = FX.load_fixture(os.path.join(FX.FIX, FIXTURE + ".json"))
    payload = E.fit_obs_list(fx["obs"], fx["meta"], E._obscodes(), force=fx.get("force", False))
    sid = payload.get("session")
    assert sid, f"fit failed: {payload.get('error')}"
    s = E._SESSIONS[sid]
    # Exercise the plain-gravity polish branch specifically (not the nongrav/constraint paths).
    s["model"] = None
    s["constraints"] = {}
    return sid, s


def test_polish_beats_loose_from_offmin_seed():
    """From an identical off-minimum seed, the loose solve (old default) stays stranded
    above the minimum; the polish solve reaches it. This is the fix, isolated."""
    sid, s = _session()
    ev = s["ctx"]["ev"]
    mask = np.asarray(s["grav_fit"]._mask, dtype=bool)
    min_state, _ = ev.fit(np.asarray(s["state"], dtype=float), mask, thin=False, polish=True)
    min_rms = ev.rms_norm(min_state, mask)
    off = min_state * (1.0 + KICK)
    loose_state, _ = ev.fit(off, mask, thin=False)                 # old behaviour
    polish_state, _ = ev.fit(off, mask, thin=False, polish=True)   # the fix
    loose_rms = ev.rms_norm(loose_state, mask)
    polish_rms = ev.rms_norm(polish_state, mask)
    # polish recovers the minimum...
    assert polish_rms <= min_rms * 1.02, f"polish missed the minimum: {polish_rms:.5f} vs {min_rms:.5f}"
    # ...and clearly beats the stranded loose solve
    assert polish_rms < loose_rms * 0.98, \
        f"polish did not beat the frozen loose solve: loose={loose_rms:.5f} polish={polish_rms:.5f}"
    print(f"OK contrast: min={min_rms:.5f} loose={loose_rms:.5f} polish={polish_rms:.5f}")


def test_fullstep_recovers_offmin_seed_end_to_end():
    """End-to-end through full_step: a stale (kicked) session seed is driven back to the
    minimum, not left stranded."""
    sid, s = _session()
    ev = s["ctx"]["ev"]
    mask = np.asarray(s["grav_fit"]._mask, dtype=bool)
    E.full_step(sid, mask.tolist())
    min_rms = ev.rms_norm(np.asarray(s["state"], dtype=float), mask)
    s["state"] = np.asarray(s["state"], dtype=float) * (1.0 + KICK)   # stale seed
    p = E.full_step(sid, mask.tolist())
    assert p.get("ok", True), p
    rec_rms = ev.rms_norm(np.asarray(s["state"], dtype=float), mask)
    assert rec_rms <= min_rms * 1.03, \
        f"Full step left the seed stranded: min={min_rms:.5f} recovered={rec_rms:.5f}"
    print(f"OK recover: min={min_rms:.5f} recovered={rec_rms:.5f}")


if __name__ == "__main__":
    test_polish_beats_loose_from_offmin_seed()
    test_fullstep_recovers_offmin_seed_end_to_end()
    print("all passed")
