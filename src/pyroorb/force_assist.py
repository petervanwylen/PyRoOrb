"""ASSIST-backed force model — a drop-in alternative to the REBOUND forward_model backend.

ASSIST (Holman/Rein/Payne/Tamayo, the rebound small-body extension) integrates a
test particle in the EXACT JPL DE440 ephemeris background: Sun, planets and Moon as
SEPARATE bodies, **general relativity** (parameterized post-Newtonian), Sun/Earth
**oblateness**, and (when the asteroid ephemeris is present) 16 massive asteroid
perturbers. This is find_orb's architecture — the perturbers come from the ephemeris,
they are not themselves integrated — and it brings DE440, GR, oblateness and the asteroid
perturbers in one dependency.

**Oblateness (zonal harmonics), cf. find_orb.** ASSIST's EARTH_HARMONICS + SUN_HARMONICS
are ON by default here (`_make_extras` only swaps GR, never disables them), so we apply the
Earth's J2/J3/J4 and the Sun's J2. find_orb applies J2/J3/J4 for Earth/Mars/Jupiter/Saturn/
Uranus/Neptune when the object is within 0.05 au of that body (`runge.cpp:1127`; NOT the Sun —
its SOLAR_J2 is coded but left under `#ifdef NOT_USED_YET`, nor the Moon). So the OVERLAP with
find_orb is the **Earth's** harmonics; we additionally carry the Sun's J2, find_orb additionally
carries Mars + the four giant planets. We do NOT have the giant-planet terms: ASSIST's forces are
a fixed C bitmask (no giant-planet harmonics, no user-force hook), so they cannot be injected into
this integration from Python. They matter only inside 0.05 au — the regular-moon / artsat regime,
which also needs the planet's satellite-ephemeris kernels we don't carry — so they are deferred and
bundled with that satellite-kernel work.

We keep our internal state heliocentric-equatorial (ICRS); ASSIST works barycentric-
equatorial. The two differ only by the Sun's barycentric state (from the ephemeris),
so the topocentric vector (asteroid − observer) is identical in either frame.

Data files (downloaded once, see EPHEM_DIR):
  - planets:   linux_p1550p2650.440      (DE440, ~102 MB)
  - asteroids: sb441-n16.bsp  (optional, 16 asteroids, ~645 MB) -> set ASTEROIDS_PATH
"""
import contextlib
import contextvars
import ctypes
import os
import threading
from pathlib import Path
import numpy as np
import rebound
import assist
import pyroorb.deadline as _deadline                 # cooperative wall-clock bound (leaf, stdlib only)
import pyroorb.sweepwatch as _sweepwatch              # what this thread is integrating right now (leaf, stdlib only)
from pyroorb.forward_model import _radec_from_helio   # shared light-time + RA/Dec geometry kernel



# The tight tolerance, one order below IAS15_EPSILON. It is the default for the two sweeps that
# are NOT the hot gravity path -- the non-gravitational Jacobian (predict_radec_jac_ng) and the
# radar multi-state sweep (helio_states_multi) -- both of which differentiate through a small
# coefficient whose signal is well under the loose tolerance's error budget. Named _FALLBACK until
# 2026-08-27 for a retry path that no longer exists (see below); nothing falls back to it now.
IAS15_EPSILON_TIGHT = 1e-6
# REMOVED 2026-08-25: THRASH_SECONDS, a 4-second wall-clock timer that aborted a loose-epsilon
# Jacobian batch so the caller could redo it at the tight tolerance. It let MACHINE LOAD pick
# the integration tolerance, and therefore the numerical path the fit takes. That is not a
# tuning knob, it is a correctness hazard: with four concurrent ASSIST workers, 490581 fitted
# 173/1208 @ 625" over one opposition and 2011 JA 64/96 @ 8.5"; on a quiet box both are clean
# (1207/1208 @ 0.249" over six, 96/96 @ 0.365"). The wrong answers reproduce digit-for-digit,
# which is what a deterministic branch on a timing threshold looks like.
#
# It was deleted rather than repaired because it never fired: instrumented across ALL 184
# fixtures on an idle box, the bail triggered ZERO times. It defended a case this corpus has
# never contained, at the price of making every result depend on what else the machine was
# doing. If a genuine thrasher ever turns up -- an orbit where the loose tolerance is SLOWER
# than the tight one, which is the case this was built for -- the deterministic form is a STEP
# budget (rebound's sim.steps_done), because thrashing is a step-count phenomenon and elapsed
# time is only a proxy for it. Calibrate that on the actual object; do not guess it now.

C_AU_DAY = 173.1446326847

# IAS15 integration accuracy. The default (1e-9) is machine precision; we only need
# ~0.01" astrometry. 1e-5 keeps the fitted orbit/covariance essentially unchanged
# (verified across heavy/short-arc-NEO/multi-apparition objects: element shifts < 0.1σ,
# Δa sub-meter to ~100 m) while running ~12% faster than 1e-6 over the suite, with 0
# regressions. Do NOT loosen further: at 3e-5 the integrator THRASHES on some orbits
# (takes ever-smaller steps -> the suite ballooned 65s -> 144s). 1e-5 sits safely below
# that cliff; if a future object ever thrashes here, the catch is a per-object epsilon
# override / slow-path retry (the --force pattern).
IAS15_EPSILON = 1e-5

# With the 16 massive-asteroid perturbers active the force field is stiffer / higher-
# frequency, and the loose 1e-5 above is NO LONGER safe: the multi-particle finite-
# difference Jacobian accumulates differential integration error between the nominal and
# the 6 perturbed particles, which corrupts the LM step and DIVERGES the fit (2009 HV119:
# a clean 0.32" gravity fit blew up to 17" with asteroids on, purely from this). A tight
# tolerance restores it -- and the perturbation then genuinely helps (2009 HV119 →
# 73/85 @ 0.224", below the gravity-only 65/85 @ 0.320"). 1e-7 already suffices and is
# fast (~0.5 s/fit); 1e-8 is the default for a decade of margin. Env-overridable.
IAS15_EPSILON_ASTEROIDS = float(os.environ.get("PYROORB_IAS15_EPSILON_AST", "1e-8"))

# Floor on the IAS15 adaptive timestep (days) -- ONE floor for both backends, defined in
# forward_model (the fuller note and the measurements are there).  IAS15 subdivides its step
# without limit through a close planetary encounter; an Earth co-orbital fit (a~1, e~0,
# repeated close Earth approaches -- e.g. 2007 XB23) drives dt toward zero, so a SINGLE batch
# integration never returns and the LM grinds forever (the (cnt & 63) thrash check can't fire
# when cnt never advances). A min_dt floors the step so the integration always RETURNS
# (coarsely through the encounter -- fine for a fit that wants ~arcsec sky positions, and the
# resulting orbit was bit-identical at 1e-3 vs a 100x finer floor). Only BINDS when the stepper
# wants a sub-floor step (a real close encounter), so every normal NEO/MBA integration -- the
# whole suite -- is untouched. Same env knob (PYROORB_IAS15_MIN_DT); 0 disables.
from pyroorb.forward_model import IAS15_MIN_DT

from pyroorb import datadirs  # noqa: E402

EPHEM_DIR = Path(os.environ.get("PYROORB_EPHEM_DIR") or datadirs.table_dir("ephem"))
PLANETS_PATH = str(EPHEM_DIR / "linux_p1550p2650.440")
# JPL DE440 planetary ephemeris (1550-2650), the gravitational background ASSIST integrates
# in. ~102 MB; REQUIRED (unlike the optional asteroid kernel below) -- ensure_planets_kernel
# fetches it on first use if missing so a fresh box / container can self-provision.
PLANETS_URL = "https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440"
ASTEROIDS_DEFAULT_PATH = str(EPHEM_DIR / "sb441-n16.bsp")   # canonical local kernel location
ASTEROIDS_PATH = os.environ.get("PYROORB_ASTEROIDS_PATH")        # explicit override (else default)
# JPL SSD small-body perturber kernel (16 massive asteroids paired with DE441), ~616 MB.
ASTEROIDS_URL = ("https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/"
                 "asteroids_de441/sb441-n16.bsp")

ASTEROIDS_OVERRIDE = None    # explicit polish-kernel path (CLI --asteroids-path); does NOT
                             # enable whole-fit mode the way the PYROORB_ASTEROIDS_PATH env does

_EPHEM = None        # planets-only (gravity) ephemeris cache
_EPHEM_AST = None    # planets + 16 asteroid perturbers ephemeris cache
# Guards the two caches above. An assist.Ephem is a small Python object over a C
# mmap of the ephemeris kernels, so a DUPLICATE costs almost nothing to build and
# then quietly pays its own resident pages for the very same files as it is used
# (measured: one Ephem sweeping 40 yr adds 6 MB, a second sweeping the IDENTICAL
# span adds another 6 MB, and re-sweeping the first adds 0). Unlocked, the lazy
# "if None: build" below is a plain race -- six threads through it returned FIVE
# distinct Ephem objects -- so a cold web server whose fit workers all start at
# once could map the 747 MB of kernels several times over. That memory is
# invisible to the /memory page (the Python object is tiny; the pages are the C
# mapping's), which is exactly what made it hard to find.
_EPHEM_LOCK = threading.Lock()


def ensure_planets_kernel(verbose=False):
    """Return the DE440 planets-ephemeris path, downloading it to ephem/ on first use if
    absent. DE440 is REQUIRED (every fit integrates in it), so unlike the asteroid kernel a
    failed fetch re-raises -- there is no gravity-only fallback without planets. The ~102 MB
    download happens at most once; thereafter the cached file is reused. Lets a fresh
    container / clone self-provision instead of needing the file baked in or pre-staged."""
    dest = Path(PLANETS_PATH)
    if dest.exists():
        return str(dest)
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  downloading DE440 planetary ephemeris (~102 MB) from {PLANETS_URL} ...")
    tmp = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(PLANETS_URL, tmp)
        tmp.replace(dest)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return str(dest)


def asteroid_kernel_path():
    """Path to the asteroid perturber kernel to use -- a CLI override, then the
    PYROORB_ASTEROIDS_PATH env, then the canonical ephem/sb441-n16.bsp -- but ONLY if the file
    exists (None otherwise, so callers transparently fall back to gravity)."""
    for p in (ASTEROIDS_OVERRIDE, ASTEROIDS_PATH, ASTEROIDS_DEFAULT_PATH):
        if p and Path(p).exists():
            return p
    return None


def ensure_asteroid_kernel(verbose=False):
    """Return a usable asteroid-kernel path, downloading it to ephem/ on first use if
    absent. Returns None (caller falls back to gravity) when no kernel is present and the
    download fails -- offline, firewalled, or JPL unreachable. The ~616 MB fetch happens
    at most once; thereafter asteroid_kernel_path() finds the cached file."""
    p = asteroid_kernel_path()
    if p:
        return p
    dest = Path(ASTEROIDS_DEFAULT_PATH)
    try:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  downloading asteroid kernel (~616 MB) from {ASTEROIDS_URL} ...")
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(ASTEROIDS_URL, tmp)
        tmp.replace(dest)
        return str(dest)
    except Exception as ex:                    # offline / firewall / partial -> gravity
        if verbose:
            print(f"  asteroid kernel download failed ({ex}); continuing gravity-only")
        return None


def _asteroids_on():
    """The GLOBAL asteroid default (backward-compat for the env / --asteroids whole-fit
    mode): True when PYROORB_ASTEROIDS_PATH is explicitly set to an existing kernel."""
    return bool(ASTEROIDS_PATH and Path(ASTEROIDS_PATH).exists())


# ---------------------------------------------------------------------------
# Self-perturber exclusion -- the ENFORCEMENT arm of guards.self_perturber_name
# ---------------------------------------------------------------------------
# guards decides WHETHER a request names one of the 16 sb441-n16 perturbers; this is where
# that verdict bites. Such an object cannot be integrated in a field that contains IT: the
# test particle sits at zero distance from its own perturbing mass and IAS15 dutifully
# reports the singularity as a trajectory. (1) Ceres, seeded from its own published orbit,
# arrived 5 deg off its observations 43 days later and ~76 deg off across the arc, so the
# fit kept 19 of 7993 observations and called the whole modern arc a 1e5-sigma outlier.
#
# The cure is to take THAT ONE body out of the field, not all sixteen: (4) Vesta is fit with
# the other 15 (Ceres the largest of them), so "+16ast" for a perturber object means
# "+15ast, itself excluded". Until 2026-11-02 the guard switched all 16 off, which left the
# 16 perturber objects as the one main-belt class that could never get the perturbed model.
#
# Mechanism: a separate cached assist.Ephem per excluded body with that body's GM zeroed
# (_masked_ephem). ASSIST reads each asteroid's GM from its own Ephem's target table on every
# force evaluation (forces.c: prefac = GM/r^3), so GM = 0 removes exactly that one term --
# the test particle is never at r = 0 exactly, so there is no 0/0. The kernel is mmap'd
# MAP_SHARED, so the extra Ephem shares its pages with the full one.
#
# A ContextVar, not a module global: web fits run concurrently on pool threads, and one
# thread fitting Ceres must not take Ceres away from its neighbours. The context covers the
# fit; SelfExcludedForce (below) is the DURABLE half that re-enters it on every backend call.
_SELF_PERTURBER = contextvars.ContextVar("pyroorb_self_perturber", default=None)

# Every backend call that accepts an `asteroids=` flag -- the ONE list, kept in the
# dependency-free forcemodel module so the pin (forcemodel.PinnedForce) can read it on a box
# without ASSIST; a test checks it against every signature here.
from pyroorb.forcemodel import ASTEROID_AWARE_CALLS  # noqa: E402


def suppressed_perturber():
    """Display name ('1 Ceres') of the self-perturbing object being fitted in this context
    (the body excluded from the perturber field), or None. Also the re-entrancy test for
    perturbers_off_for."""
    return _SELF_PERTURBER.get()


def _excluded_number():
    """MP number of the body excluded in this context ('4 Vesta' -> 4), or None."""
    name = _SELF_PERTURBER.get()
    if name is None:
        return None
    from pyroorb.guards import perturber_number
    return perturber_number(name)


@contextlib.contextmanager
def perturbers_off_for(name):
    """Take `name` ('4 Vesta', one of the 16 -- guards.self_perturber_name) OUT of the
    perturber field for the duration of the block; the other 15 stay as configured. See
    _SELF_PERTURBER. (Named for the old behaviour, all 16 off, which is now only the fallback
    when the exclusion cannot be built -- see _masked_ephem.)"""
    token = _SELF_PERTURBER.set(name)
    try:
        yield
    finally:
        _SELF_PERTURBER.reset(token)


# The opposite switch: perturbers FORCED ON for one fit, whatever the global default. Used for
# an identification hypothesis ("is A = B?", orbit_common.ident_requested): a main-belt pair
# whose arcs only join under the 16 perturbers would otherwise be refused one apparition at a
# time by the gravity linker -- a false "no" that no post-fit polish can undo, because the
# polish starts from the gravity fit's links. A ContextVar for the same reason as above; the
# self-perturber context still wins (see _use_asteroids), and obs_sources._force_backend turns
# this into a durable fit_object._AsteroidForce on the fit's ctx.
_FORCE_PERTURBERS = contextvars.ContextVar("pyroorb_force_perturbers", default=None)


def forced_perturbers():
    """Why the perturbers are forced on in this context ('ident'), or None."""
    return _FORCE_PERTURBERS.get()


@contextlib.contextmanager
def perturbers_on_for(reason):
    """Force the 16 asteroid perturbers ON for the duration of the block (unless the object
    is itself one of them -- perturbers_off_for overrules). See _FORCE_PERTURBERS."""
    token = _FORCE_PERTURBERS.set(reason)
    try:
        yield
    finally:
        _FORCE_PERTURBERS.reset(token)


class SelfExcludedForce:
    """A force backend for one of the 16 perturber objects: EVERY call runs with that body
    excluded from the perturber field (perturbers_off_for), whoever makes it and whether or
    not the asteroids are on. It lands in the fit's ctx, so everything the ctx reaches later
    -- an interactive step, the +16ast toggle, a re-anchor, an ephemeris, an impact scan --
    keeps the object out of its own field after the fit's context has gone.

    It pins the EXCLUSION, not the on/off choice: asteroids= passes through, so gravity vs
    +15ast is decided outside it (the fit's forcemodel.PinnedForce, refine_asteroids'
    keep-gate, the web toggle). forcemodel.excluded_body reads `_self_excluded`.
    Non-callable attributes fall through unchanged, so hasattr-style capability checks are
    unaffected. Wrapping is idempotent (an inner SelfExcludedForce is unwrapped)."""

    def __init__(self, backend, name):
        if isinstance(backend, SelfExcludedForce):
            backend = backend._backend
        self._backend = backend
        self._self_excluded = name

    def __getattr__(self, name):
        f = getattr(self._backend, name)
        if not callable(f) or name.startswith("__"):
            return f
        excluded = self._self_excluded

        def _excluding(*a, **k):
            with perturbers_off_for(excluded):
                return f(*a, **k)
        return _excluding


def _use_asteroids(asteroids):
    """Resolve a per-call asteroids flag: None -> the global default; else honour it --
    but only when a kernel is actually available (so True can never crash a kernel-less box).
    In a self-perturber context the perturbers run with that body excluded; if the exclusion
    cannot be built (_masked_ephem None) they are refused outright -- an object that IS one of
    the 16 is never integrated in a field that contains it, whoever asked.
    Inside perturbers_on_for (an identification fit) None means ON."""
    if asteroids is None and _FORCE_PERTURBERS.get() is not None:
        asteroids = True
    use = _asteroids_on() if asteroids is None else bool(asteroids)
    if not (use and asteroid_kernel_path() is not None):
        return False
    n = _excluded_number()
    return n is None or _masked_ephem(n) is not None


def _epsilon(asteroids=None):
    """IAS15 accuracy for THIS call: the tight asteroid value when perturbers are active
    (the loose default is unstable with them), else the fast gravity-only default."""
    return IAS15_EPSILON_ASTEROIDS if _use_asteroids(asteroids) else IAS15_EPSILON


# ctypes views of ASSIST's `struct spk_s` / `struct spk_target` (src/spk.h, ASSIST 1.2.x):
# assist.Ephem exposes only the `spk_asteroids` pointer, not the target table behind it.
# The layout is VERIFIED before any mass is touched (the target codes must read back as the
# kernel's 2000000 + MP numbers, the mass field must equal get_particle's GM, and
# get_particle must then report exactly that one GM changed) -- a layout drift in a future
# ASSIST therefore disables the exclusion (all 16 off, the old guard) instead of writing
# into the wrong field.
#
# Those checks read THROUGH the views, so they cannot catch a layout that makes the read
# itself fault (a moved `targets` pointer or a different `num` field is followed before
# anything is compared). The first gate is therefore the release: only an ASSIST whose C
# library reports a version these structs were checked against is read at all; any other
# keeps the perturbers off for a perturber object, with the one-time notice. To admit a new
# release, compare src/spk.h with the two classes below, then add it here.
SPK_LAYOUT_VERIFIED = ("1.2",)       # ASSIST major.minor releases (1.2.3 checked 2026-11-02)


def spk_layout_supported(version=None):
    """True when the installed ASSIST C library (or `version`) is a release whose spk
    structs match _SpkS / _SpkTarget -- the gate checked before any pointer is followed."""
    v = str(getattr(assist, "__version__", "") if version is None else version).strip()
    return any(v == r or v.startswith(r + ".") for r in SPK_LAYOUT_VERIFIED)


class _SpkTarget(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("cen", ctypes.c_int), ("mass", ctypes.c_double),
                ("beg", ctypes.c_double), ("end", ctypes.c_double), ("res", ctypes.c_double),
                ("one", ctypes.POINTER(ctypes.c_int)), ("two", ctypes.POINTER(ctypes.c_int)),
                ("ind", ctypes.c_int), ("allocated_ind", ctypes.c_int)]


class _SpkS(ctypes.Structure):
    _fields_ = [("targets", ctypes.POINTER(_SpkTarget)), ("num", ctypes.c_int),
                ("allocated_num", ctypes.c_int), ("map", ctypes.c_void_p),
                ("len", ctypes.c_size_t)]


_ASSIST_BODY_NPLANETS = 11          # ASSIST body index of the first asteroid (assist.h)
_EPHEM_EXCL = {}                    # MP number -> Ephem with that body's GM zeroed, or None
_EXCL_WARNED = set()


def _spk_table(ephem):
    """The asteroid SPK target table of `ephem` as a list of live _SpkTarget views; empty
    (nothing dereferenced) for an ASSIST release whose layout is not verified."""
    if not spk_layout_supported():
        return []
    p = ephem.spk_asteroids
    if not p:
        return []
    spk = _SpkS.from_address(p)
    if not (0 < spk.num <= 64):                        # implausible -> layout is not ours
        return []
    return [spk.targets[i] for i in range(spk.num)]


def zero_asteroid_gm(ephem, number):
    """Zero the GM of asteroid `number` (MP number, one of the 16) in THIS ephem's perturber
    table, after verifying the struct layout. Returns True on a verified change; False (and
    the ephem untouched) when the layout or the result does not check out."""
    tbl = _spk_table(ephem)
    codes = [t.code for t in tbl]
    # layout check: every target code must read as a numbered asteroid (2000000 + n), and
    # the one wanted must be there exactly once
    if not tbl or not all(2000000 < c < 3000000 for c in codes) \
            or codes.count(2000000 + number) != 1:
        return False
    i = codes.index(2000000 + number)
    t_rel = 2451545.0 - ephem.jd_ref               # J2000: inside every sb441-n16 segment
    try:
        before = [ephem.get_particle(_ASSIST_BODY_NPLANETS + k, t_rel).m
                  for k in range(len(tbl))]
    except Exception:
        return False
    if not (before[i] > 0 and abs(before[i] - tbl[i].mass) <= 1e-12 * before[i]):
        return False                                   # mass field is not where we think
    tbl[i].mass = 0.0
    after = [ephem.get_particle(_ASSIST_BODY_NPLANETS + k, t_rel).m for k in range(len(tbl))]
    ok = after[i] == 0.0 and all(a == b for k, (a, b) in enumerate(zip(after, before))
                                 if k != i)
    if not ok:
        tbl[i].mass = before[i]
    return ok


def _masked_ephem(number):
    """The cached DE440 + 16-asteroid ephem with asteroid `number` excluded (GM 0), or None
    when no kernel is available or the exclusion could not be verified (a one-time notice;
    the caller then keeps the perturbers off for this object, the pre-2026-11-02 guard)."""
    if number in _EPHEM_EXCL:
        return _EPHEM_EXCL[number]
    kern = asteroid_kernel_path()
    if kern is None:
        return None
    with _EPHEM_LOCK:
        if number not in _EPHEM_EXCL:
            ok, eph = False, None
            if spk_layout_supported():
                eph = assist.Ephem(ensure_planets_kernel(), kern)
                try:
                    ok = zero_asteroid_gm(eph, number)
                except Exception:
                    ok = False
                why = "ASSIST struct layout did not verify"
            else:
                why = (f"ASSIST {getattr(assist, '__version__', '?')} is not a release whose "
                       f"struct layout is verified ({', '.join(SPK_LAYOUT_VERIFIED)}.x)")
            if not ok and number not in _EXCL_WARNED:
                _EXCL_WARNED.add(number)
                print(f"(force_assist: could not exclude asteroid {number} from the perturber "
                      f"field -- {why}; it stays gravity-only)")
            _EPHEM_EXCL[number] = eph if ok else None
    return _EPHEM_EXCL[number]


def get_ephem(asteroids=None):
    """Lazily construct (and cache) the ASSIST ephemeris. THREE kinds of cache: planets-only
    (gravity), planets + the 16 asteroid perturbers, and planets + the 16 with ONE excluded
    (a self-perturber context, see perturbers_off_for). `asteroids` selects gravity vs
    perturbed (None -> the global PYROORB_ASTEROIDS_PATH default); the asteroid ephem is built
    only when a kernel is available, else this transparently returns the gravity ephem."""
    global _EPHEM, _EPHEM_AST
    # Double-checked: the cached read stays lock-free (this is on every sweep's
    # hot path), and only the one-time construction serialises. See _EPHEM_LOCK.
    if _use_asteroids(asteroids):
        n = _excluded_number()
        if n is not None:
            return _masked_ephem(n)        # non-None: _use_asteroids checked it
        if _EPHEM_AST is None:
            with _EPHEM_LOCK:
                if _EPHEM_AST is None:
                    _EPHEM_AST = assist.Ephem(ensure_planets_kernel(),
                                              asteroid_kernel_path())
        return _EPHEM_AST
    if _EPHEM is None:
        with _EPHEM_LOCK:
            if _EPHEM is None:
                planets = ensure_planets_kernel()
                with _quiet_no_asteroid_notice():
                    _EPHEM = assist.Ephem(planets)
    return _EPHEM


# ASSIST's C library writes this to stderr whenever an Ephem is built without an asteroid
# kernel -- i.e. on every gravity-only run, the default. It reads like a failure but is the
# expected state (perturbers are opt-in: pyroorb-fetch-data --asteroids), so it is dropped.
_NO_ASTEROID_NOTICE = b"The JPL asteroid ephemeris file has not been found."


@contextlib.contextmanager
def _quiet_no_asteroid_notice():
    """Run the block with C-level stderr (fd 2) captured; re-emit every captured line except
    ASSIST's no-asteroid-kernel notice. Anything else -- a real ASSIST message, or another
    thread's output in this short window -- still reaches stderr."""
    import sys
    import tempfile
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError, AttributeError):    # no usable fd 2: nothing to quiet
        yield
        return
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            kept = [ln for ln in tmp.read().splitlines(keepends=True)
                    if _NO_ASTEROID_NOTICE not in ln]
            if kept:
                os.write(2, b"".join(kept))


def _make_extras(sim, ephem):
    """Attach ASSIST, swapping its default full Einstein-Infeld-Hoffmann GR (GR_EIH) for the
    cheaper solar-only term (GR_SIMPLE, Nobili & Roxburgh) -- which is what find_orb uses. EIH
    is a relativistic N-body sum over every perturber (the ASSIST authors flag it as "slow and
    computationally expensive"); for asteroid astrometry the difference is < 0.002" (far below
    our ~0.01" target) while the integration is ~10% faster."""
    ex = assist.Extras(sim, ephem)
    if IAS15_MIN_DT > 0:
        sim.ri_ias15.min_dt = IAS15_MIN_DT   # floor the adaptive step so a close-encounter integration
                                             # always returns (never collapses dt to zero -> hang)
    f = list(ex.forces)
    if "GR_EIH" in f:
        f[f.index("GR_EIH")] = "GR_SIMPLE"
        ex.forces = f
    return ex


def _set_gr(ex, gr):
    """Install a g(r) non-gravitational scaling law on an ASSIST Extras: the global
    alpha/r0/nm/nn/nk constants (see nongrav.py). The NON_GRAVITATIONAL force is on
    by ASSIST default; with no particle_params set it is a no-op, so a gravity-only
    integration never touches this. `gr` None leaves the defaults untouched."""
    if gr is None:
        return
    ex.alpha, ex.r0 = gr["alpha"], gr["r0"]
    ex.nm, ex.nn, ex.nk = gr["nm"], gr["nn"], gr["nk"]


def _apply_nongrav(ex, nongrav):
    """Set the single-particle A1/A2/A3 + g(r) for a nongrav propagation, or do
    nothing when `nongrav` is None (the pure-gravity path). `nongrav` is an
    (A, gr) pair: A a length-3 au/day^2 vector, gr a g(r) constant dict."""
    if nongrav is None:
        return
    A, gr = nongrav
    _set_gr(ex, gr)
    ex.particle_params = np.asarray(A, dtype=float)      # [A1,A2,A3] for particle 0


# The Sun's barycentric state at a given TDB instant depends ONLY on time -- never on the
# orbit being fit -- yet _sun_bary is called for every observation on every LM iteration /
# residual sweep, re-running the same DE440 Chebyshev evaluation thousands of times with
# identical arguments (the obs times and epochs are fixed for a fit). Memoize on the float
# time: the obs times recur identically every sweep, so this collapses ~n_obs*n_sweeps
# get_particle("Sun") calls down to ~n_obs per object (e.g. 71789 -> ~600 on 1864). The
# cached arrays are returned by reference and read-only everywhere (callers only subtract
# them), so we freeze them to catch any accidental mutation. Bounded so a long-lived
# process (web server) can't grow it without limit -- past the cap we simply stop caching.
_SUN_CACHE = {}
_SUN_CACHE_MAX = 300_000


def _sun_bary(ephem, jd_tdb):
    key = float(jd_tdb)
    hit = _SUN_CACHE.get(key)
    if hit is not None:
        return hit
    p = ephem.get_particle("Sun", key - ephem.jd_ref)
    pos = np.array([p.x, p.y, p.z]); vel = np.array([p.vx, p.vy, p.vz])
    pos.flags.writeable = False; vel.flags.writeable = False
    if len(_SUN_CACHE) < _SUN_CACHE_MAX:
        _SUN_CACHE[key] = (pos, vel)
    return pos, vel


def _integrate_split(ephem, t0_tdb, states, times_tdb, epsilon, setup=None):
    """Build the ASSIST sim and march MONOTONICALLY outward from t0 (forward through obs
    >= t0, backward through obs < t0), yielding (i, sim) after integrating to each obs
    time. The shared t0-split integration scaffold behind every ASSIST sweep
    (predict_radec_batch / _jac_batch / predict_radec_jac_ng / helio_positions_batch /
    predict_state).

    `states` is a list of heliocentric 6-states (au, au/day) added as massless particles
    (particle 0 nominal, 1.. perturbed for a Jacobian sweep); each is shifted to ASSIST's
    barycentric frame by the Sun's t0 state. `setup(ex)` runs once per direction after the
    Extras are built (out-gassing / g(r) / per-particle params). `epsilon` is the IAS15
    tolerance for this call, and it is FIXED for the call: nothing here may change the
    integration tolerance mid-flight (see the THRASH_SECONDS note at the top of the module).

    The split is required by integrate_or_interpolate (it interpolates within the last
    step, so each leg must advance one way) AND it confines a far-apparition divergence to
    its own direction instead of NaN-poisoning every later obs in a single sweep. ASSIST
    double-buffers the particles, so read sim.particles / serialize FRESH after each yield."""
    sunp0, sunv0 = _sun_bary(ephem, t0_tdb)
    # The one place all ASSIST propagation passes through, so it is where a bounded
    # unit of work agrees to stop (see deadline.py -- Python cannot kill a thread,
    # so a wedged fit can only be ended from the inside). Read ONCE here: an
    # unbounded run -- the CLI, the batch tool, the suite, an ITF search on its own
    # thread -- pays this single lookup per sweep and nothing per observation.
    _dl = _deadline.active()
    after = np.flatnonzero(times_tdb >= t0_tdb)            # ascending  (forward from t0)
    before = np.flatnonzero(times_tdb < t0_tdb)[::-1]      # descending (backward, nearest t0 first)
    for grp in (after, before):
        if not len(grp):
            continue
        sim = rebound.Simulation()
        sim.t = t0_tdb - ephem.jd_ref
        for s in states:
            sim.add(x=s[0]+sunp0[0], y=s[1]+sunp0[1], z=s[2]+sunp0[2],
                    vx=s[3]+sunv0[0], vy=s[4]+sunv0[1], vz=s[5]+sunv0[2])
        ex = _make_extras(sim, ephem)
        if setup is not None:
            setup(ex)
        sim.ri_ias15.epsilon = epsilon
        # Register the leg with sweepwatch, so a thread parked inside one
        # integrate_or_interpolate call reads on /memory as the state and target it is
        # stuck on rather than a bare frame name (see sweepwatch.py).  One dict store
        # per leg, one list store per observation; the finally runs when the consumer
        # finishes, breaks out, or the generator is dropped.
        rec = _sweepwatch.begin("assist", t0_tdb, states[0], times_tdb)
        try:
            for i in grp:
                if _dl is not None and _deadline.expired(_dl):
                    _deadline.raise_for(_dl)
                rec[_sweepwatch.AT] = i
                ex.integrate_or_interpolate(times_tdb[i] - ephem.jd_ref)
                yield int(i), sim
        finally:
            _sweepwatch.end(rec, sim)


def predict_radec_batch(base_sim, t0_tdb, state0, times_tdb, Robs_helio, nongrav=None,
                        asteroids=None):
    """ASSIST analogue of forward_model.predict_radec_batch (same signature; `base_sim`
    is ignored — ASSIST builds its own). `state0` is heliocentric-equatorial au,
    au/day at t0_tdb; `times_tdb` must be ascending. Returns (n,2) [ra_deg, dec_deg].

    `nongrav` (optional) is an (A, gr) pair adding the Marsden A1/A2/A3 out-gassing
    acceleration; None is the pure-gravity path (unchanged). `asteroids` selects the
    16-perturber ephemeris for THIS call (None -> the global default; True forces it on
    when a kernel is available)."""
    ephem = get_ephem(asteroids)
    times_tdb = np.asarray(times_tdb, float)
    Robs_helio = list(Robs_helio)
    out = np.empty((len(times_tdb), 2))
    for i, sim in _integrate_split(ephem, t0_tdb, [state0], times_tdb, _epsilon(asteroids),
                                   setup=lambda ex: _apply_nongrav(ex, nongrav)):
        p = sim.particles[0]                              # read FRESH (buffer may have swapped)
        sunp, sunv = _sun_bary(ephem, times_tdb[i])
        r = np.array([p.x, p.y, p.z]) - sunp             # barycentric -> heliocentric
        v = np.array([p.vx, p.vy, p.vz]) - sunv
        out[i, 0], out[i, 1] = _radec_from_helio(r, v, Robs_helio[i])
    return out


def predict_state(base_sim, t0_tdb, state0, t1_tdb, nongrav=None, asteroids=None):
    """Heliocentric-equatorial 6-state (au, au/day) of the orbit at t1_tdb, integrated
    from `state0` at t0_tdb with the DE440+GR force (and the optional A1/A2/A3
    out-gassing in `nongrav`). `base_sim` is ignored (ASSIST builds its own), kept for
    signature parity with predict_radec_batch. Used to RE-ANCHOR an orbit to a new
    osculating epoch without re-fitting: the physical trajectory is unchanged, so the
    residuals at the obs times are identical -- only the elements' reference epoch moves.

    That last sentence is only true when this propagation uses the SAME force model the
    orbit was fitted in, which is what `asteroids` is for (None -> the global default,
    exactly as predict_radec_batch reads it). An asteroid-POLISHED orbit re-anchored
    without it was moved through the 8-planet field and then had its residuals and
    covariance re-evaluated in the 16-perturber one -- disagreeing by precisely the
    perturbation the polish exists to capture (2009 HV119: 0.320" -> 0.224"). The caller
    that knows the model passes it (fit_object.reanchor_fit_to_epoch reads fit.asteroids)."""
    ephem = get_ephem(asteroids)
    for _i, sim in _integrate_split(ephem, t0_tdb, [state0], np.array([t1_tdb], float),
                                    _epsilon(asteroids),
                                    setup=lambda ex: _apply_nongrav(ex, nongrav)):
        p = sim.particles[0]                              # read FRESH (buffer may have swapped)
    sunp1, sunv1 = _sun_bary(ephem, t1_tdb)
    return np.array([p.x - sunp1[0], p.y - sunp1[1], p.z - sunp1[2],
                     p.vx - sunv1[0], p.vy - sunv1[1], p.vz - sunv1[2]])


def predict_radec_jac_batch(base_sim, t0_tdb, state0, times_tdb, Robs_helio, h=None,
                            asteroids=None):
    """Predictions AND the analytic-quality Jacobian d(ra,dec)/d(state0) in ONE
    integration. Instead of letting scipy finite-difference the Jacobian with 6
    SEPARATE integration sweeps per LM iteration, we integrate the nominal particle
    together with 6 perturbed copies (state0[k] += h_k) as massless test particles
    in a SINGLE simulation. ASSIST interpolates the DE440 ephemeris (the dominant
    cost) once per substep and shares it across all 7 particles, so the six extra
    columns come nearly for free (~2x faster than 6 separate sweeps, verified).

    Returns (pred, jac): pred is (n,2) [ra_deg, dec_deg] of the nominal orbit; jac
    is (n,2,6) with jac[i,0,k]=d(ra_i)/d(state0_k), jac[i,1,k]=d(dec_i)/d(state0_k)
    (forward differences; ra wrapped to (-180,180] before dividing by h_k).

    Integrates at IAS15_EPSILON, always. There is no timing-dependent retry: a tolerance chosen
    by how long the machine happened to take is a tolerance chosen by machine load (see the
    THRASH_SECONDS note at the top of the module).

    With the asteroid perturbers active we go STRAIGHT to the tight asteroid epsilon (the
    loose-first attempt is exactly the unstable path there -- see IAS15_EPSILON_ASTEROIDS)
    and skip the thrash heuristic, whose timer would false-fire on the legitimately slower
    16-body integration anyway."""
    if _use_asteroids(asteroids):
        return _jac_batch(t0_tdb, state0, times_tdb, Robs_helio, h,
                          IAS15_EPSILON_ASTEROIDS, asteroids=asteroids)
    return _jac_batch(t0_tdb, state0, times_tdb, Robs_helio, h, IAS15_EPSILON,
                      asteroids=asteroids)


def _jac_batch(t0_tdb, state0, times_tdb, Robs_helio, h, epsilon, asteroids=None):
    ephem = get_ephem(asteroids)
    state0 = np.asarray(state0, dtype=float)
    if h is None:                                   # scipy-style relative step
        h = 1.49e-8 * np.maximum(1.0, np.abs(state0))
    times_tdb = np.asarray(times_tdb, float)
    Robs_helio = list(Robs_helio)
    states = [state0] + [state0.copy() for _ in range(6)]   # particle 0 nominal, 1..6 perturbed
    for k in range(6):
        states[k+1][k] += h[k]
    n = len(times_tdb)
    pred = np.empty((n, 2)); jac = np.empty((n, 2, 6))
    pos = np.empty((7, 3)); vel = np.empty((7, 3))
    r_all = np.empty((n, 7, 3)); v_all = np.empty((n, 7, 3))   # helio state of all 7 particles, per obs
    # The 7 particles (nominal + 6 perturbed) ride one simulation; ASSIST interpolates the
    # closely-spaced obs within a step for all of them.
    # We only SERIALIZE inside the loop (the buffer swaps each step); the RA/Dec geometry is
    # deferred to one vectorized pass below -- same per-element math as the per-obs
    # _radec_from_helio (m,3) path (light-time back-step + arctan2/arcsin over the length-3
    # axis), just without the ~n python/ufunc-dispatch round-trips. Verified byte-identical.
    for i, sim in _integrate_split(ephem, t0_tdb, states, times_tdb, epsilon):
        sim.serialize_particle_data(xyz=pos, vxvyvz=vel)   # FRESH read (buffer may have swapped)
        sunp, sunv = _sun_bary(ephem, times_tdb[i])
        r_all[i] = pos - sunp; v_all[i] = vel - sunv       # (7,3) barycentric->helio
    robs = np.asarray(Robs_helio, dtype=float)[:, None, :]  # (n,1,3) observer helio position
    rel = r_all - robs
    for _ in range(2):                                      # light-time, linear back-step (vectorized)
        tau = np.sqrt((rel * rel).sum(axis=2)) / C_AU_DAY   # (n,7)
        rel = (r_all - v_all * tau[..., None]) - robs
    ra = np.degrees(np.arctan2(rel[..., 1], rel[..., 0])) % 360                    # (n,7)
    dec = np.degrees(np.arcsin(rel[..., 2] / np.sqrt((rel * rel).sum(axis=2))))    # (n,7)
    pred[:, 0] = ra[:, 0]; pred[:, 1] = dec[:, 0]
    jac[:, 0, :] = ((ra[:, 1:] - ra[:, :1] + 180) % 360 - 180) / h                 # (n,6) wrapped
    jac[:, 1, :] = (dec[:, 1:] - dec[:, :1]) / h
    return pred, jac


def predict_radec_jac_ng(t0_tdb, state0, A, free_idx, gr, times_tdb, Robs_helio,
                         hstate=None, hA=None, epsilon=IAS15_EPSILON_TIGHT,
                         asteroids=None):
    """Predictions + Jacobian d(ra,dec)/d[state(6), A_free(k)] for a NON-GRAVITATIONAL
    fit, in one ASSIST integration.

    Like predict_radec_jac_batch, but the parameter vector is augmented by the free
    out-gassing coefficients. We ride 1 nominal + 6 state-perturbed + k A-perturbed
    massless particles in a single simulation; each carries its own A1/A2/A3 through
    ASSIST's per-particle particle_params (forces.c reads params[3*j+0..2]), so the
    k extra Jacobian columns cost no extra ephemeris work -- the same trick the 6-D
    analytic Jacobian uses. `A` is the length-3 physical au/day^2 vector (fixed axes
    included as 0); `free_idx` are the axes (0,1,2 = A1,A2,A3) being solved for.

    Returns (pred (n,2) [ra,dec] of the nominal orbit, jac (n,2,6+k)); the trailing
    k columns are d(ra,dec)/dA_i in PHYSICAL units (per au/day^2) -- the caller scales.
    Uses the tight fallback epsilon (run once, and the nongrav accelerations are tiny,
    so we want the integration exact rather than fast). `asteroids` selects the
    16-perturber ephemeris for THIS call (None -> the global default); with them active
    the epsilon TIGHTENS to their required value, since a loose one diverges the
    multi-particle Jacobian (see IAS15_EPSILON_ASTEROIDS).

    No caller passes it today: refine_nongrav's population (auto-discovery gates on
    q < 1.4 au) and refine_asteroids' (main-belt, >= 2 oppositions) barely overlap,
    so a nongrav-on-polished orbit is rare. The knob is here so that
    combination is a CHOICE at the call site rather than a hard-coded gravity-only
    assumption -- which is the bug this parameter family was added to remove."""
    ephem = get_ephem(asteroids)
    epsilon = min(epsilon, _epsilon(asteroids))   # never looser than the perturbers demand
    state0 = np.asarray(state0, dtype=float)
    A = np.asarray(A, dtype=float)
    k = len(free_idx)
    if hstate is None:
        hstate = 1.49e-8 * np.maximum(1.0, np.abs(state0))
    if hA is None:                                   # absolute step in each free A axis
        hA = np.array([1e-3 * (abs(A[i]) if A[i] else 1.0) for i in free_idx])
    times_tdb = np.asarray(times_tdb, float)
    Robs_helio = list(Robs_helio)

    npart = 7 + k
    states = [state0.copy() for _ in range(npart)]   # particles 7.. keep the nominal state
    for j in range(6):
        states[1 + j][j] += hstate[j]                # 6 state-perturbed
    As = [A.copy() for _ in range(npart)]
    for j in range(k):
        As[7 + j][free_idx[j]] += hA[j]              # k A-perturbed
    pp = np.concatenate(As).astype(float)            # flat [A1,A2,A3] per particle

    n = len(times_tdb)
    pred = np.empty((n, 2)); jac = np.empty((n, 2, 6 + k))
    pos = np.empty((npart, 3)); vel = np.empty((npart, 3))
    r_all = np.empty((n, npart, 3)); v_all = np.empty((n, npart, 3))

    def _setup(ex):
        _set_gr(ex, gr)
        ex.particle_params = pp                       # per-particle A1,A2,A3

    # Serialize inside the loop (the buffer swaps each step), then do the RA/Dec geometry in
    # one vectorized pass, as _jac_batch does: the same per-element math as the per-obs
    # _radec_from_helio (m,3) path, without its per-observation dispatch. Byte-identical.
    for i, sim in _integrate_split(ephem, t0_tdb, states, times_tdb, epsilon, setup=_setup):
        sim.serialize_particle_data(xyz=pos, vxvyvz=vel)   # FRESH read (buffer may swap)
        sunp, sunv = _sun_bary(ephem, times_tdb[i])
        r_all[i] = pos - sunp; v_all[i] = vel - sunv
    robs = np.asarray(Robs_helio, dtype=float).reshape(n, 1, 3)
    rel = r_all - robs
    for _ in range(2):                                      # light-time, linear back-step
        tau = np.sqrt((rel * rel).sum(axis=2)) / C_AU_DAY   # (n,npart)
        rel = (r_all - v_all * tau[..., None]) - robs
    ra = np.degrees(np.arctan2(rel[..., 1], rel[..., 0])) % 360
    dec = np.degrees(np.arcsin(rel[..., 2] / np.sqrt((rel * rel).sum(axis=2))))
    pred[:, 0] = ra[:, 0]; pred[:, 1] = dec[:, 0]
    dra = (ra - ra[:, :1] + 180) % 360 - 180               # wrapped, (n,npart)
    ddec = dec - dec[:, :1]
    jac[:, 0, :6] = dra[:, 1:7] / hstate
    jac[:, 1, :6] = ddec[:, 1:7] / hstate
    if k:
        jac[:, 0, 6:] = dra[:, 7:] / hA
        jac[:, 1, 6:] = ddec[:, 7:] / hA
    return pred, jac


def helio_positions_batch(base_sim, t0_tdb, state0, times_tdb, nongrav=None,
                          asteroids=None):
    """Heliocentric-equatorial positions (n,3 au) of the asteroid at each epoch — used for
    photometry geometry (distances / phase angle). Same integrate_or_interpolate + t0-split
    sweep as predict_radec_batch (read sim.particles FRESH after each call -- buffer swaps).
    `nongrav` (optional, see predict_radec_batch) adds the A1/A2/A3 acceleration;
    `asteroids` selects the 16-perturber ephemeris (None -> the global default).

    Its callers (photometry H, ephemerides.py, impact.py, keyholes.py) leave `asteroids`
    alone deliberately: they consume DISTANCES and phase angles, where the perturbers move
    the answer by parts in 1e8 -- far below an H uncertainty or an impact corridor. The
    knob exists so the choice is the caller's rather than hard-coded here, which is what
    let predict_state silently re-anchor a polished orbit in the wrong model."""
    ephem = get_ephem(asteroids)
    times_tdb = np.asarray(times_tdb, float)
    out = np.empty((len(times_tdb), 3))
    for i, sim in _integrate_split(ephem, t0_tdb, [state0], times_tdb, _epsilon(asteroids),
                                   setup=lambda ex: _apply_nongrav(ex, nongrav)):
        p = sim.particles[0]                              # read FRESH (buffer may have swapped)
        sunp, _ = _sun_bary(ephem, times_tdb[i])
        out[i] = [p.x - sunp[0], p.y - sunp[1], p.z - sunp[2]]
    return out


def helio_states_multi(t0_tdb, states, times_tdb, nongrav=None,
                       epsilon=IAS15_EPSILON_TIGHT, asteroids=None):
    """Heliocentric-equatorial 6-states (au, au/day) of SEVERAL particles at each epoch,
    in one integration: returns (n, npart, 6) for `states` a list of npart 6-states at
    t0_tdb. The radar residual model's sweep (radar.py): the nominal orbit plus its 6
    state-perturbed copies ride one simulation, so the delay/Doppler Jacobian costs the
    same single integration the RA/Dec analytic Jacobian does (_jac_batch's trick, but
    delivering raw states -- radar needs the object's position AND velocity at the
    bounce epoch, not a sky direction). `nongrav` is an (A, gr) pair; A of shape (3,)
    applies the SAME out-gassing acceleration to every particle (the gravity-path radar
    refine holds A fixed and solves only the state), while shape (npart, 3) gives each
    particle its OWN A -- the joint state+A+radar solve (radar._refine_radar_nongrav)
    rides k A-perturbed particles for its d(delay,Doppler)/dA columns, the
    same per-particle-params trick predict_radec_jac_ng uses for the sky Jacobian.
    Tight fallback epsilon by default: it runs once per LM
    evaluation and radar sigmas are microseconds, so exact beats fast. `asteroids`
    selects the 16-perturber ephemeris (None -> the global default) and tightens the
    epsilon to what the perturbers require. Its callers (radar.py's refine, itf/lov_variants)
    leave it alone: radar targets are NEOs and the ITF variant scan is gravity-only by
    design, so neither meets the main-belt asteroid polish. As with predict_radec_jac_ng,
    the point is that the model is now a call-site choice, not a hard-coded one."""
    ephem = get_ephem(asteroids)
    epsilon = min(epsilon, _epsilon(asteroids))   # never looser than the perturbers demand
    times_tdb = np.asarray(times_tdb, float)
    npart = len(states)
    out = np.empty((len(times_tdb), npart, 6))
    pos = np.empty((npart, 3)); vel = np.empty((npart, 3))
    setup = None
    if nongrav is not None:
        A, gr = nongrav
        A = np.asarray(A, float)
        if A.ndim == 2:                               # (npart, 3): per-particle A1,A2,A3
            if A.shape != (npart, 3):
                raise ValueError(f"per-particle nongrav A must be ({npart}, 3), got {A.shape}")
            pp = A.ravel()
        else:                                         # (3,): same A for every particle
            pp = np.tile(A, npart)

        def setup(ex):
            _set_gr(ex, gr)
            ex.particle_params = pp
    for i, sim in _integrate_split(ephem, t0_tdb, list(states), times_tdb, epsilon,
                                   setup=setup):
        sim.serialize_particle_data(xyz=pos, vxvyvz=vel)   # FRESH read (buffer may swap)
        sunp, sunv = _sun_bary(ephem, times_tdb[i])
        out[i, :, :3] = pos - sunp
        out[i, :, 3:] = vel - sunv
    return out


def build_base_sim(t0_tdb):
    """Signature parity with forward_model.build_base_sim; ASSIST needs no prebuilt sim,
    so this is a no-op handle (returned object is unused by predict_radec_batch)."""
    return None
