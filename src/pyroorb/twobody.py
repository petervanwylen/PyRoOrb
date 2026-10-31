"""Standard two-body (Keplerian) mechanics — textbook, NOT from find_orb.

Universal-variable formulation (Stumpff functions), per Curtis, *Orbital Mechanics
for Engineering Students*. Used ONLY by the initial-orbit-determination cascade
(iod.py) for cheap seed orbits and by the orbit scorer; the actual fit integrates
with ASSIST/REBOUND. If you already know this material you can skip this file —
it's a Lambert solver, a Kepler propagator, and osculating-element extraction.
"""
import math
import numpy as np

from pyroorb.forward_model import MU_SUN


# --- Stumpff functions C(z), S(z) (Curtis Alg. 3.1) -------------------------
def stumpff_C(z):
    if z > 1e-12:  return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-12:
        s = math.sqrt(-z)
        if s > 700.0: return math.inf      # cosh would overflow: a divergent UV step
        return (math.cosh(s) - 1) / (-z)
    return 0.5


def stumpff_S(z):
    if z > 1e-12:
        s = math.sqrt(z);  return (s - math.sin(s)) / s**3
    if z < -1e-12:
        s = math.sqrt(-z)
        if s > 700.0: return math.inf      # sinh would overflow: a divergent UV step
        return (math.sinh(s) - s) / s**3
    return 1.0 / 6.0


def lambert(r1, r2, dt, mu=MU_SUN, prograde=True):
    """Universal-variable Lambert (Curtis Alg. 5.2). Returns v1 or None."""
    r1n, r2n = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    c12 = np.cross(r1, r2)
    dtheta = math.acos(max(-1, min(1, float(np.dot(r1, r2)) / (r1n * r2n))))
    if (prograde and c12[2] < 0) or (not prograde and c12[2] >= 0):
        dtheta = 2 * math.pi - dtheta
    # r1 parallel to r2 (dtheta 0 or 2pi) leaves no transfer plane and divides by zero
    # below -- bail to the IOD cascade's next method, as every other degeneracy here does.
    denom = 1 - math.cos(dtheta)
    if denom <= 0:
        return None
    A = math.sin(dtheta) * math.sqrt(r1n * r2n / denom)
    if abs(A) < 1e-15:
        return None
    def yz(z):
        cz = stumpff_C(z)
        if not math.isfinite(cz) or cz <= 0:
            return math.inf
        return r1n + r2n + A * (z * stumpff_S(z) - 1) / math.sqrt(cz)
    z = 0.0
    for _ in range(80):
        y = yz(z)
        if A > 0 and y < 0:
            z += 0.1; continue
        C, S = stumpff_C(z), stumpff_S(z)
        # a divergent universal-variable step pushes z so far negative that the Stumpff
        # functions overflow (math.inf here). Bail to the IOD cascade's next method
        # rather than crashing -- and guard the sqrt(y) below against y<0.
        # y <= 0 (not just < 0): the dF expression below divides by y, so a degenerate
        # zero-baseline transfer (dt ~ 0, r1 ~ r2) raised ZeroDivisionError instead of
        # falling through to the next IOD method.
        if not (math.isfinite(C) and math.isfinite(S) and math.isfinite(y)) or y <= 0 or C == 0:
            return None
        # F/dF themselves can overflow on a divergent step (e.g. (y/C)**1.5 or S**2/C
        # exceeding the float range): Python raises OverflowError rather than returning
        # inf, so catch it and bail to the IOD cascade's next method like the guards above.
        try:
            F = (y / C) ** 1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * dt
            if abs(z) < 1e-12:
                dF = math.sqrt(2)/40 * y**1.5 + A/8*(math.sqrt(y) + A*math.sqrt(1/(2*y)))
            else:
                dF = (y/C)**1.5*(1/(2*z)*(C - 1.5*S/C) + 0.75*S**2/C) \
                     + A/8*(3*S/C*math.sqrt(y) + A*math.sqrt(C/y))
        except OverflowError:
            return None
        if dF == 0 or not math.isfinite(dF):
            return None
        zn = z - F/dF
        if not math.isfinite(zn):
            return None
        if abs(zn - z) < 1e-10:
            z = zn; break
        z = zn
    y = yz(z)
    if not math.isfinite(y) or y < 0:
        return None
    g = A * math.sqrt(y / mu)
    f = 1 - y / r1n
    return (np.asarray(r2) - f * np.asarray(r1)) / g


_EPS = float(np.finfo(float).eps)
KEPLER_MAX_ITER = 80      # Newton steps before the safeguarded fallback takes over
KEPLER_TOL = 1e-9         # absolute convergence on chi, as Curtis Alg. 3.4
# NB: an appealing "relative for large chi" tolerance was tried and REVERTED. It buys
# nothing -- the non-convergences it was meant to accept are stalls sitting ~1e-3 from the
# root, far outside any sane relative band, and they are the bracketed solver's job -- while
# it silently costs precision on ORDINARY elliptic orbits: at chi ~ 9.3 it stops 9x early,
# which measured as 2e-6 relative error in the propagated state (vs 8e-10 for the absolute
# tolerance, checked against DOP853). chi is O(1-100) here, so absolute IS the right currency.


def _kepler_F(r0n, vr0, alpha, dt, mu, chi):
    """The universal Kepler equation F(chi) (Curtis eq. 3.62), or None if it is not
    representable at this chi (the Stumpff overflow guard, or a float overflow in the
    chi**3 term -- both mean the iteration has wandered outside the usable domain).

    dF/dchi is the orbital radius r(chi), which is strictly positive, so F is strictly
    INCREASING and its root is unique. That monotonicity is what makes the bracketed
    fallback below guaranteed to converge, and it is why a better starting guess can
    never change the answer -- only whether Newton finds it."""
    z = alpha * chi * chi
    C, S = stumpff_C(z), stumpff_S(z)
    if not (math.isfinite(C) and math.isfinite(S)):
        return None
    try:
        F = r0n*vr0/math.sqrt(mu)*chi*chi*C + (1-alpha*r0n)*chi**3*S + r0n*chi - math.sqrt(mu)*dt
    except OverflowError:
        return None
    return F if math.isfinite(F) else None


def _kepler_chi0(r0n, vr0, alpha, dt, mu):
    """Starting guess for the universal anomaly, by REGIME.

    Curtis' chi0 = sqrt(mu)*|alpha|*dt is derived for the ELLIPTIC case and is what this
    function returns for alpha > 0 (every real asteroid, and the overwhelming majority of
    calls -- keeping it byte-identical there keeps every converging fit unchanged).

    For alpha < 0 it is simply the wrong scale, and the IOD distance scan generates such
    trial orbits by the thousand: measured over the regression fixtures, EVERY failure of
    the old solver -- all 9 in 25786 calls -- was hyperbolic. Newton then either walked
    chi into the Stumpff overflow band (7 cases, NaN) or stalled short of the root and
    returned a silently WRONG finite state (2 cases). So the hyperbolic regime gets its
    own textbook logarithmic starter (Vallado, *Fundamentals of Astrodynamics*, Alg. 8;
    equivalently Bate-Mueller-White), which starts Newton inside the basin instead of
    launching it out of the representable domain."""
    if alpha > 0:
        return math.sqrt(mu) * abs(alpha) * dt          # Curtis Alg. 3.4, unchanged
    if alpha < 0:
        try:
            a = 1.0 / alpha                              # < 0
            sgn = 1.0 if dt >= 0 else -1.0
            num = -2.0 * mu * alpha * dt
            den = r0n*vr0 + sgn * math.sqrt(-mu*a) * (1.0 - r0n*alpha)
            if den != 0.0 and num / den > 0.0:
                return sgn * math.sqrt(-a) * math.log(num / den)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    # parabolic band, or a hyperbolic starter that is not representable: fall back to the
    # elliptic scale and let the bracketed solver below carry any case Newton cannot.
    return math.sqrt(mu) * abs(alpha) * dt


def _kepler_newton(r0n, vr0, alpha, dt, mu, chi):
    """Newton on F(chi). Returns (chi, converged). Never raises and never produces a
    NaN: a step that leaves the representable domain reports failure instead, which is
    what the old loop did NOT do -- it divided inf by inf, got NaN, and then spent its
    remaining 80 iterations propagating that NaN through the arithmetic (`abs(nan) <
    tol` is never true) before returning a NaN state."""
    smu = math.sqrt(mu)
    for _ in range(KEPLER_MAX_ITER):
        z = alpha * chi * chi
        C, S = stumpff_C(z), stumpff_S(z)
        if not (math.isfinite(C) and math.isfinite(S)):
            return chi, False
        try:
            F = r0n*vr0/smu*chi*chi*C + (1-alpha*r0n)*chi**3*S + r0n*chi - smu*dt
            dF = r0n*vr0/smu*chi*(1-alpha*chi*chi*S) + (1-alpha*r0n)*chi*chi*C + r0n
        except OverflowError:
            return chi, False
        if not (math.isfinite(F) and math.isfinite(dF)) or dF == 0.0:
            return chi, False
        d = F / dF
        if not math.isfinite(d):
            return chi, False
        chi -= d
        if abs(d) < KEPLER_TOL:
            return chi, True
    return chi, False


def _kepler_bracketed(r0n, vr0, alpha, dt, mu):
    """Safeguarded solve for chi, or None. Used ONLY when Newton fails.

    chi and dt share a sign (dchi/dt = sqrt(mu)/r > 0 and chi(0) = 0), and F is strictly
    increasing, so the root is bracketed by expanding outward from 0 in the direction of
    dt until F changes sign; bisection from there cannot diverge or stall. If the bracket
    cannot be built without leaving the representable domain, the orbit is genuinely not
    propagable in double precision and we say so rather than guessing."""
    sgn = 1.0 if dt >= 0 else -1.0
    lo, hi = 0.0, sgn * max(1.0, abs(_kepler_chi0(r0n, vr0, alpha, dt, mu)))
    for _ in range(200):                       # expand until F(hi) is past the root
        Fh = _kepler_F(r0n, vr0, alpha, dt, mu, hi)
        if Fh is None:
            return None
        if Fh * sgn > 0.0:
            break
        lo, hi = hi, hi * 2.0
    else:
        return None
    # Bisection halves the bracket every step, so 100 steps take ANY starting interval to
    # machine precision -- there is no non-convergence branch to report. It stops early only
    # when the interval can no longer be halved in double precision.
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        Fm = _kepler_F(r0n, vr0, alpha, dt, mu, mid)
        if Fm is None:
            return None
        if Fm * sgn > 0.0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= max(5e-324, 4 * _EPS * abs(hi)):
            break
    chi = 0.5 * (lo + hi)
    return chi if math.isfinite(chi) else None


_KEPLER_FAIL = np.full(6, np.nan)     # module-level template; copied on the (rare) failure path


def kepler_propagate(state, dt, mu=MU_SUN):
    """Universal-variable Kepler propagation (Curtis Alg. 3.4), safeguarded.

    Returns the propagated state, or a state of NaNs if this orbit cannot be propagated in
    double precision -- a real outcome here, because the IOD cascade feeds this function
    thousands of deliberately-wrong trial orbits (absurd hyperbolas from the assumed-distance
    scan) and asks which survive. EVERY caller already reads a non-finite state as "discard
    this seed and try the next method", so the NaN state is the failure CONTRACT, not an
    accident.

    Two things this must never do, both of which the plain Curtis loop did:

      - return a finite state it did not converge to. Started from the elliptic guess on a
        hyperbolic trial orbit, Newton lands where the Stumpff functions are ~1e172 and then
        crawls by a near-constant step, so it exhausts its iterations far from the root and
        returns a state with relative error ~1e141 -- which passes every caller's isfinite()
        check. Measured against an independent classical-element propagator, that was 38% of
        a hyperbolic-heavy synthetic population and 2 calls in 25786 over the fixture suite.

      - signal failure differently depending on the caller. dt arrives as np.float64 from
        iod.py/sr.py and as a plain float from arc_extend.py, and the identical divergence
        (inf/inf inside the Newton step) surfaced as a numpy RuntimeWarning on the first
        path -- printed at the user's terminal mid-batch, where it reads as the app breaking
        -- and would raise on the second.

    The scalar arithmetic below is therefore deliberately plain-float rather than numpy:
    floats raise (caught here) where numpy scalars warn and carry a NaN onward, so there is
    exactly one failure signal, it is silent, and it cannot depend on the caller's dtype.
    It is also faster -- for 3-vectors the numpy call overhead dominates the arithmetic, and
    this function is called ~25k times per fit by the multi-revolution period sweep."""
    # Indexed element-wise rather than sliced, so a 7-element nongrav state (x..vz, A1/A2)
    # propagates on its gravity-only part instead of tripping over the ragged slice the old
    # `state[3:]` produced. The return is always the 6-vector this is a propagator for.
    try:
        x, y, zc, vx, vy, vz = (float(state[0]), float(state[1]), float(state[2]),
                                float(state[3]), float(state[4]), float(state[5]))
        dt = float(dt)
    except (TypeError, ValueError, IndexError):
        return _KEPLER_FAIL.copy()
    # Validate up front rather than letting a non-finite input propagate through as a NaN
    # result: a caller handing us a broken state has already failed upstream, and our job is
    # to report that through the contract (which it checks) rather than to narrate it.
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(zc) and math.isfinite(vx)
            and math.isfinite(vy) and math.isfinite(vz) and math.isfinite(dt)):
        return _KEPLER_FAIL.copy()
    # mu is validated on the same contract rather than left to raise: a zero mu divides by zero
    # below and a negative one reaches sqrt(mu) inside the solver -- both an exception out of a
    # function whose documented failure signal is a NaN state, and neither is a way any caller is
    # prepared to be told "unpropagable". No caller passes mu at all today, so this guards an
    # argument that is structurally constant: contract, not bug.
    if not (math.isfinite(mu) and mu > 0.0):
        return _KEPLER_FAIL.copy()

    # hypot, not sqrt(x*x+...): the squares of a large-but-finite coordinate overflow to inf
    # (|x| >~ 1e154), which would file a finite input as a propagation failure. That magnitude is
    # not something this engine produces -- a diverged IOD/LM state arrives as inf/nan and the
    # entry guard above already has it -- and the scale-safety stops here anyway: v0n2 and the r.v
    # dot below overflow at the same size. What makes hypot worth using is the trade, not the
    # hazard: it is correctly rounded and, measured on this box, the faster of the two
    # (~0.083 vs ~0.136 us) in a path the multi-revolution sweep calls ~25k times per fit. It is
    # not bit-identical to the explicit form (1 ULP apart on ~36% of ordinary-scale inputs), so
    # every propagation moves fractionally; no golden moved when it landed.
    r0n = math.hypot(x, y, zc)
    if not (r0n > 0.0):
        return _KEPLER_FAIL.copy()
    v0n2 = vx*vx + vy*vy + vz*vz
    vr0 = (x*vx + y*vy + zc*vz) / r0n
    alpha = 2/r0n - v0n2/mu
    if not (math.isfinite(alpha) and math.isfinite(vr0)):
        return _KEPLER_FAIL.copy()

    chi, ok = _kepler_newton(r0n, vr0, alpha, dt, mu,
                             _kepler_chi0(r0n, vr0, alpha, dt, mu))
    if not ok:
        chi = _kepler_bracketed(r0n, vr0, alpha, dt, mu)
        if chi is None:
            return _KEPLER_FAIL.copy()

    z = alpha*chi*chi
    C, S = stumpff_C(z), stumpff_S(z)
    if not (math.isfinite(C) and math.isfinite(S)):
        return _KEPLER_FAIL.copy()
    try:
        f = 1 - chi*chi/r0n*C
        g = dt - chi**3/math.sqrt(mu)*S
        rx, ry, rz = f*x + g*vx, f*y + g*vy, f*zc + g*vz
        rn = math.hypot(rx, ry, rz)          # scale-safe, as for r0n above
        if not (rn > 0.0):
            return _KEPLER_FAIL.copy()
        fd = math.sqrt(mu)/(rn*r0n)*(alpha*chi**3*S - chi)
        gd = 1 - chi*chi/rn*C
        out = (rx, ry, rz, fd*x + gd*vx, fd*y + gd*vy, fd*zc + gd*vz)
    except (OverflowError, ZeroDivisionError):
        return _KEPLER_FAIL.copy()
    for c in out:
        if not math.isfinite(c):
            return _KEPLER_FAIL.copy()
    return np.array(out)


def quick_aeq(state):
    """Osculating (a, e, q) from a heliocentric state vector."""
    r, v = state[:3], state[3:]
    rn = float(np.linalg.norm(r)); v2 = float(np.dot(v, v))
    a = 1.0 / (2.0/rn - v2/MU_SUN)
    e = float(np.linalg.norm(np.cross(v, np.cross(r, v))/MU_SUN - r/rn))
    return a, e, a*(1 - e)
