"""Forward observation model: orbit state -> predicted RA/Dec.

The REBOUND backend (Sun + 8 planets as point masses; no GR, no non-gravs) plus the
geometry that turns an integrated heliocentric state into apparent equatorial
coordinates with light-time. This is the cheap, dependency-light force model used for
IOD scoring and as the fallback when ASSIST's DE440 ephemeris is absent; force_assist.py
is the higher-fidelity sibling (DE440 + GR) with the same predict_radec_batch signature.

All vectors are heliocentric, ICRS/equatorial frame.
"""
import math
import pyroorb.deadline as _deadline                  # cooperative wall-clock bound (leaf, stdlib only)
import pyroorb.sweepwatch as _sweepwatch              # what this thread is integrating right now (leaf, stdlib only)
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
import os
import rebound

# Floor on the IAS15 adaptive timestep (days) -- ONE floor for BOTH backends: this is the
# definition, and force_assist imports it, so there is one env knob and one default.
#
# IAS15 subdivides its step without limit near a point mass, and nothing here has a radius
# to collide with, so a trial state that passes -- or, worse, is BOUND -- within a hair of
# the Sun or a planet makes a single sim.integrate() call take millions of steps per
# simulated day; once the step is small enough that t + dt == t in double precision the
# call never returns at all.  Measured on this sim: a state bound 1e-5 au from the Earth
# point mass costs 2.1 s per 30 days unfloored and 0.64 s floored; at 1e-7 au the unfloored
# call did not return in 40 s while the floored one took milliseconds.  ASSIST got the floor
# first (the 2007 XB23 co-orbital hang); this module -- the propagator IOD, SR and the range
# profile score trial orbits with -- got it on 2014 HM543, where two ITF trial fits sat
# ~290 s each inside ONE integrate() call, out of reach of every between-call check
# including the 30 s trial-fit bound.  Two live ITF searches were later found parked inside
# rebound's integrate for 17 and 41 hours (pyroorb.net/memory, 2026-09-26) on a server
# that predated this floor; sweepwatch records what such a thread is integrating so the
# next one can be reproduced from the report.  Binds ONLY when the stepper wants a
# sub-floor step, so ordinary orbits are byte-identical (the whole suite is).  For the
# bodies here that is inside ~0.7 solar radii or ~1.5 Earth radii -- inside the body.
# 0 disables (rebound default).
IAS15_MIN_DT = float(os.environ.get("PYROORB_IAS15_MIN_DT", "1e-3"))

K = 0.01720209895                 # Gaussian gravitational constant
MU_SUN = K * K                    # au^3/day^2
C_AU_DAY = 173.1446326847         # speed of light


def _radec_from_helio(r, v, robs):
    """Heliocentric body state (position `r`, velocity `v`; au & au/day) plus the
    observer's heliocentric position `robs` -> apparent (ra_deg, dec_deg), applying the
    2-iteration linear light-time back-step along v (tau ~ minutes, so the curvature term
    is sub-microarcsecond). This is the single geometry kernel shared by BOTH force
    backends (rebound here and ASSIST in force_assist) — any fix (aberration, a third
    light-time pass) now lands in one place instead of the ~5 hand-copies it replaces.

    Dispatches on rank: a single body (r, v, robs shape (3,)) -> scalar (ra, dec); a stack
    (r, v shape (m,3); robs (3,)) -> arrays (ra (m,), dec (m,)). The 1-D path uses
    math.sqrt(rel.dot(rel)) and the (m,3) path np.linalg.norm(rel, axis=1). These are NOT
    interchangeable: np.linalg.norm(x, axis=-1) on a 1-D x differs from sqrt(dot) at ~1e-14
    and that perturbs the bridge, so the rank is dispatched explicitly to keep every
    existing caller bit-for-bit unchanged."""
    rel = r - robs
    if rel.ndim == 1:
        for _ in range(2):                          # light-time, linear back-step
            tau = math.sqrt(rel.dot(rel)) / C_AU_DAY
            rel = (r - v * tau) - robs
        # math.* on Python floats skips numpy's scalar-ufunc dispatch (this 1-D path runs
        # once per obs in every residual sweep). Bit-identical to the np.degrees/arctan2/
        # arcsin form -- both bottom out in libm; verified max diff 0 over 3e5 random draws.
        # The asin argument is clamped to [-1,1] to match np.arcsin's nan-free domain (the
        # ratio is <=1 by construction; the guard only covers a rounding tie at the poles).
        ra = math.degrees(math.atan2(rel[1], rel[0])) % 360
        s = rel[2] / math.sqrt(rel.dot(rel))
        dec = math.degrees(math.asin(-1.0 if s < -1.0 else 1.0 if s > 1.0 else s))
    else:
        for _ in range(2):                          # light-time, vectorized over the stack
            tau = np.sqrt((rel * rel).sum(axis=1)) / C_AU_DAY   # == norm(rel,axis=1), no dispatch
            rel = (r - v * tau[:, None]) - robs
        ra = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360
        dec = np.degrees(np.arcsin(rel[:, 2] / np.sqrt((rel * rel).sum(axis=1))))
    return ra, dec

# Planet masses in solar masses (Earth uses the Earth-Moon barycentre).
PLANETS = {
    "mercury": 1.66012e-7, "venus": 2.44784e-6, "earth": 3.04043e-6,  # EMB
    "mars": 3.22716e-7, "jupiter": 9.54791e-4, "saturn": 2.85886e-4,
    "uranus": 4.36624e-5, "neptune": 5.15139e-5,
}


def los_unit(ra_deg, dec_deg):
    """Line-of-sight unit vector (equatorial) for an RA/Dec."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def build_base_sim(t0_tdb):
    """A REBOUND sim of the Sun + 8 planets at epoch t0_tdb (COM frame)."""
    sim = rebound.Simulation()
    sim.G = MU_SUN
    sim.add(m=1.0)  # Sun at origin
    t = Time(t0_tdb, format="jd", scale="tdb")
    sun_p, sun_v = get_body_barycentric_posvel("sun", t)
    for name, mass in PLANETS.items():
        p, v = get_body_barycentric_posvel(name, t)
        pos = (p - sun_p).xyz.to(u.au).value
        vel = (v - sun_v).xyz.to(u.au/u.day).value
        sim.add(m=mass, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])
    sim.move_to_com()
    # The IAS15 step floor (see IAS15_MIN_DT above): every sweep integrates a copy()
    # of this sim, and copies inherit it.
    if IAS15_MIN_DT > 0:
        sim.ri_ias15.min_dt = IAS15_MIN_DT
    return sim


def helio_positions_batch(base_sim, t0_tdb, state0, times_tdb):
    """Heliocentric-equatorial positions (n,3 au) of the asteroid at each epoch in
    one integration sweep — used for photometry geometry (distances/phase)."""
    sim = base_sim.copy()
    sun0 = sim.particles[0]
    sim.add(m=0.0,
            x=sun0.x+state0[0], y=sun0.y+state0[1], z=sun0.z+state0[2],
            vx=sun0.vx+state0[3], vy=sun0.vy+state0[4], vz=sun0.vz+state0[5])
    ast, sun = sim.particles[-1], sim.particles[0]
    out = np.empty((len(times_tdb), 3))
    rec = _sweepwatch.begin("rebound", t0_tdb, state0, times_tdb, "helio")
    try:
        for i, t_tdb in enumerate(times_tdb):
            rec[_sweepwatch.AT] = i
            sim.integrate(t_tdb - t0_tdb)
            out[i] = [ast.x-sun.x, ast.y-sun.y, ast.z-sun.z]
    finally:
        _sweepwatch.end(rec, sim)
    return out


def predict_radec_batch(base_sim, t0_tdb, state0, times_tdb, Robs_helio):
    """Fast many-epoch forward model: rebound integration sweeps instead of
    re-integrating from the epoch per observation.

    The asteroid is advanced in place through the sorted epochs, so the whole batch
    costs ~one full-arc integration rather than n_obs of them. Light-time is applied
    as a linear back-step along the recorded heliocentric velocity (tau ~ minutes, so
    the curvature term is sub-microarcsecond). Returns an (n, 2) array of
    [ra_deg, dec_deg].

    Integration is split OUTWARD from t0 (forward through obs >= t0, backward through
    obs < t0), mirroring force_assist.predict_radec_batch. A single ascending sweep
    is cumulative, so a divergence at one far/excluded apparition (e.g. crossing an
    Earth close approach) would NaN-poison every obs after it — corrupting the residual
    of the GOOD obs too. Splitting confines any divergence to the far end of its own
    direction. A window entirely on one side of t0 integrates EXACTLY as the old single
    sweep did (the common IOD-scoring case is bit-for-bit unchanged)."""
    times_tdb = np.asarray(times_tdb, float)
    Robs_helio = list(Robs_helio)
    out = np.empty((len(times_tdb), 2))

    def _record(i, ast, sun, robs):
        r = np.array([ast.x-sun.x, ast.y-sun.y, ast.z-sun.z])
        v = np.array([ast.vx-sun.vx, ast.vy-sun.vy, ast.vz-sun.vz])
        out[i, 0], out[i, 1] = _radec_from_helio(r, v, robs)

    # The rebound fallback's half of the deadline seam -- the ASSIST path bounds
    # itself in force_assist._integrate_split. Read once per sweep; an unbounded
    # run pays nothing per observation. See deadline.py.
    _dl = _deadline.active()
    after = np.flatnonzero(times_tdb >= t0_tdb)           # ascending (forward from t0)
    before = np.flatnonzero(times_tdb < t0_tdb)[::-1]     # descending (backward, nearest t0 first)
    for grp in (after, before):
        if not len(grp):
            continue
        sim = base_sim.copy()
        sun0 = sim.particles[0]
        sim.add(m=0.0,
                x=sun0.x+state0[0], y=sun0.y+state0[1], z=sun0.z+state0[2],
                vx=sun0.vx+state0[3], vy=sun0.vy+state0[4], vz=sun0.vz+state0[5])
        ast, sun = sim.particles[-1], sim.particles[0]
        # Register the leg with sweepwatch: a thread parked inside integrate() then
        # reads on /memory as the state and the target it is stuck on, rather than as
        # a bare frame name (see sweepwatch.py).  One dict store per leg, one list
        # store per observation -- nothing is computed here.
        rec = _sweepwatch.begin("rebound", t0_tdb, state0, times_tdb)
        try:
            for i in grp:
                if _dl is not None and _deadline.expired(_dl):
                    _deadline.raise_for(_dl)
                rec[_sweepwatch.AT] = i
                sim.integrate(times_tdb[i] - t0_tdb)          # advance in place (monotonic within group)
                _record(int(i), ast, sun, Robs_helio[i])
        finally:
            _sweepwatch.end(rec, sim)
    return out


def predict_state(base_sim, t0_tdb, state0, t1_tdb, nongrav=None):
    """Heliocentric-equatorial 6-state (au, au/day) of the asteroid at t1_tdb,
    integrated from `state0` at t0_tdb in the 8-planet point-mass model. Used to
    RE-ANCHOR an orbit to a new osculating epoch (signature parity with the ASSIST
    backend; `nongrav` is unsupported here and must be None -- out-gassing lives in
    force_assist)."""
    if nongrav is not None:
        raise NotImplementedError("the rebound 8-planet backend has no nongrav force; "
                                  "re-anchor a nongrav orbit with the ASSIST backend")
    sim = base_sim.copy()
    sun0 = sim.particles[0]
    sim.add(m=0.0,
            x=sun0.x+state0[0], y=sun0.y+state0[1], z=sun0.z+state0[2],
            vx=sun0.vx+state0[3], vy=sun0.vy+state0[4], vz=sun0.vz+state0[5])
    ast, sun = sim.particles[-1], sim.particles[0]
    rec = _sweepwatch.begin("rebound", t0_tdb, state0, (t1_tdb,), "reanchor")
    rec[_sweepwatch.AT] = 0
    try:
        sim.integrate(t1_tdb - t0_tdb)
    finally:
        _sweepwatch.end(rec, sim)
    return np.array([ast.x-sun.x, ast.y-sun.y, ast.z-sun.z,
                     ast.vx-sun.vx, ast.vy-sun.vy, ast.vz-sun.vz])
