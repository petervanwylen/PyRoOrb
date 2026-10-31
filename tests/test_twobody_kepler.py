"""Guards for the safeguarded universal-variable Kepler solver (twobody.py).

The solver is fed thousands of deliberately-wrong trial orbits by the IOD cascade, so
its FAILURE behaviour is as load-bearing as its success behaviour. Three properties:

  1. it never returns a finite state it did not converge to (a silently wrong seed);
  2. it fails silently and identically whatever type `dt` arrives as (np.float64 from
     iod.py/sr.py, plain float from arc_extend.py) -- a warning printed mid-batch reads
     as a crash to the user;
  3. it is correct where it succeeds, judged against an INDEPENDENT propagator.

The reference below is classical elements + anomaly: it shares no formulation with the
universal-variable solver, so agreement is real evidence rather than a tautology.
"""
import math
import warnings

import numpy as np
import pytest

from pyroorb.forward_model import MU_SUN
from pyroorb.twobody import kepler_propagate


def _reference(state, dt, mu=MU_SUN):
    """Classical-element propagation, or None where the element set is ill-conditioned."""
    r0 = np.asarray(state[:3], float); v0 = np.asarray(state[3:], float)
    rn = float(np.linalg.norm(r0)); vn2 = float(np.dot(v0, v0))
    h = np.cross(r0, v0); hn = float(np.linalg.norm(h))
    if hn == 0 or rn == 0:
        return None
    evec = np.cross(v0, h)/mu - r0/rn
    e = float(np.linalg.norm(evec))
    if abs(e - 1.0) < 1e-6 or e < 1e-8:
        return None
    a = 1.0/(2.0/rn - vn2/mu); p = hn*hn/mu
    ehat = evec/e; hhat = h/hn; phat = np.cross(hhat, ehat)
    nu0 = math.atan2(float(np.dot(r0, phat)), float(np.dot(r0, ehat)))
    if e < 1.0:
        E0 = 2*math.atan2(math.tan(nu0/2)*math.sqrt(1-e), math.sqrt(1+e))
        M = (E0 - e*math.sin(E0)) + math.sqrt(mu/a**3)*dt
        M = math.fmod(M, 2*math.pi)
        f = lambda E: E - e*math.sin(E) - M
        lo, hi = M - 1.0 - e, M + 1.0 + e
        for _ in range(200):
            mid = 0.5*(lo+hi)
            lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
        E = 0.5*(lo+hi)
        nu = 2*math.atan2(math.sqrt(1+e)*math.tan(E/2), math.sqrt(1-e))
    else:
        t = math.sqrt((e-1)/(e+1))*math.tan(nu0/2)
        if abs(t) >= 1.0:
            return None
        H0 = 2*math.atanh(t)
        M = (e*math.sinh(H0) - H0) + math.sqrt(mu/(-a)**3)*dt
        f = lambda H: e*math.sinh(H) - H - M
        try:
            hi = 1.0
            while f(hi) < 0: hi *= 2.0
            lo = -1.0
            while f(lo) > 0: lo *= 2.0
            for _ in range(400):
                mid = 0.5*(lo+hi)
                lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
        except OverflowError:
            return None
        H = 0.5*(lo+hi)
        nu = 2*math.atan2(math.sqrt(e+1)*math.tanh(H/2), math.sqrt(e-1))
    r = p/(1 + e*math.cos(nu))
    out = np.concatenate([r*(math.cos(nu)*ehat + math.sin(nu)*phat),
                          math.sqrt(mu/p)*(-math.sin(nu)*ehat + (e + math.cos(nu))*phat)])
    return out if np.all(np.isfinite(out)) else None


def _population(n, seed=7):
    """States spanning both regimes: sigma_v ~ 0.02 au/day straddles the escape speed at
    ~2 au, so roughly half of these are the hyperbolic trial orbits the IOD scan produces."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield (np.concatenate([rng.normal(0, 2, 3), rng.normal(0, 0.02, 3)]),
               float(rng.normal(0, 2000)))


def test_matches_independent_propagator_and_never_silently_wrong():
    """Every finite state returned must be the RIGHT one. A finite-but-wrong return is the
    defect this solver was rewritten to remove: the old Newton loop, started from the
    elliptic guess, crawled by a near-constant step through the region where the Stumpff
    functions are ~1e172, exhausted its 80 iterations far from the root, and handed back a
    state with relative error ~1e141 that passed every caller's isfinite() check."""
    checked = wrong = failed = 0
    for state, dt in _population(4000):
        ref = _reference(state, dt)
        if ref is None:
            continue
        checked += 1
        got = kepler_propagate(state, dt)
        if not np.all(np.isfinite(got)):
            failed += 1                      # an explicit failure is allowed; a wrong answer is not
            continue
        if np.max(np.abs(got - ref)/np.maximum(np.abs(ref), 1e-9)) > 1e-6:
            wrong += 1
    assert checked > 2000, f"population too degenerate to be a test ({checked} arbitrated)"
    assert wrong == 0, f"{wrong}/{checked} finite states disagree with the reference"
    # the safeguarded solver should also SOLVE these, not merely decline them
    assert failed <= checked // 100, f"{failed}/{checked} declined; regressed robustness"


def test_hyperbolic_trial_orbits_solve():
    """The regime the IOD distance scan generates and the old solver could not do: every
    failure measured over the regression fixtures had alpha < 0."""
    solved = total = 0
    for state, dt in _population(2000, seed=11):
        rn = float(np.linalg.norm(state[:3]))
        if 2/rn - float(np.dot(state[3:], state[3:]))/MU_SUN >= 0:
            continue                          # elliptic; covered above
        total += 1
        solved += bool(np.all(np.isfinite(kepler_propagate(state, dt))))
    assert total > 100
    assert solved / total > 0.98, f"only {solved}/{total} hyperbolic trials propagated"


@pytest.mark.parametrize("dt_type", [float, np.float64])
def test_failure_is_silent_and_type_independent(dt_type):
    """No warning may escape, whichever call path supplies dt. The identical divergence
    used to surface as a numpy RuntimeWarning when dt was an np.float64 (iod.py, sr.py)
    -- printed at the user's terminal in the middle of a batch run, where it reads as the
    app breaking -- while the plain-float path (arc_extend.py) signalled differently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # any warning becomes a test failure
        for state, dt in _population(3000, seed=3):
            out = kepler_propagate(state, dt_type(dt))
            assert out.shape == (6,)
            assert np.all(np.isfinite(out)) or np.all(np.isnan(out)), \
                "failure must be the documented all-NaN state, not a partial one"


def test_degenerate_inputs_return_the_failure_contract():
    """Callers rely on `np.all(np.isfinite(...))` to mean 'usable' -- so a degenerate input
    must produce NaNs rather than an exception or a plausible-looking state."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        good = np.array([1.0, 0, 0, 0, 0.01, 0])
        for state, dt in ((np.zeros(6), 100.0),                       # at the origin
                          (good, float("nan")),                       # no epoch to reach
                          (good, float("inf")),
                          (np.array([np.nan, 0, 0, 0, 0.01, 0]), 100.0),
                          (np.array([np.inf, 0, 0, 0, 0.01, 0]), 100.0)):
            assert np.all(np.isnan(kepler_propagate(state, dt)))


def test_zero_dt_is_the_identity():
    """A sanity anchor independent of the reference: propagating by no time at all must
    return the state unchanged, in both regimes."""
    for state, _ in _population(200, seed=5):
        out = kepler_propagate(state, 0.0)
        assert np.allclose(out, state, rtol=1e-10, atol=1e-12)


def test_radial_free_fall():
    """Zero angular momentum is NOT a failure case -- it is a rectilinear orbit, and the
    universal-variable formulation handles it analytically (which is one reason the IOD
    cascade can lean on it). Checked against numerical integration, short of the collision
    at t_ff = (pi/2)*sqrt(r0**3/(2*mu)) = 64.6 d, where the integrator itself breaks down."""
    from scipy.integrate import solve_ivp
    state = np.array([1.0, 0, 0, 0, 0, 0])

    def rhs(t, y):
        r = y[:3]; rn = float(np.linalg.norm(r))
        return np.concatenate([y[3:], -MU_SUN*r/rn**3])

    for dt in (10.0, 50.0):
        num = solve_ivp(rhs, [0, dt], state, rtol=1e-12, atol=1e-15, method="DOP853").y[:, -1]
        assert np.allclose(kepler_propagate(state, dt), num, rtol=1e-8, atol=1e-10)


def test_large_but_finite_state_is_not_a_failure():
    """A coordinate near the top of the double range is finite, and dt=0 propagates it to
    itself -- but sqrt(x*x + ...) overflows above ~1e154 and used to turn the identity into a
    NaN vector, i.e. the contract's "not propagable" for a state that plainly is. hypot is
    scale-safe. The same norm is taken again on the propagated position, so both are checked."""
    for scale in (1e120, 1e154, 1e200):
        state = np.array([scale, -0.5*scale, 0.25*scale, 0.0, 0.0, 0.0])
        out = kepler_propagate(state, 0.0)
        assert np.allclose(out, state, rtol=1e-12, atol=0.0), scale


def test_invalid_mu_fails_by_contract_rather_than_raising():
    """The documented failure signal is a NaN state, and it has to hold for every argument:
    mu=0 divided by zero on the way to alpha and a negative mu reached sqrt(mu) inside the
    solver, so an unusable mu came out as an exception through a function no caller wraps."""
    good = np.array([1.0, 0, 0, 0, 0.01, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for mu in (0.0, -1.0, float("nan"), float("inf")):
            assert np.all(np.isnan(kepler_propagate(good, 10.0, mu=mu))), mu
