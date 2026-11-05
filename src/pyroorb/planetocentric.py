"""Planetocentric orbits -- fit + report a natural satellite about a
KNOWN giant planet ("this is a Saturn satellite" -> ``--center saturn``).

Opt-in leaf module: the heliocentric N-body fit is UNCHANGED (the giant planets are
already perturbers, so a satellite's physics is fit correctly heliocentrically). This
adds only the two pieces the heliocentric pipeline lacks:

  1. A planet-hinted IOD seed (``planet_hinted_seed``). Ordinary angles-only IOD produces
     garbage for a satellite because a short arc cannot resolve the topocentric range. But
     an object inside a planet's Hill sphere sits at essentially the planet's known distance,
     so we pin rho ~ rho_planet and split the heliocentric velocity into the planet's bulk
     motion + a secant planetocentric velocity -- a seed that lands the LM in the right basin.

  2. Planetocentric element reporting (``planetocentric_elements`` / ``report``): reframe the
     converged heliocentric state onto the planet, compute classical elements with the
     planet's GM, and report a satellite class instead of a nonsense heliocentric one.

find_orb is the reference (logic openly ported, no source copied):
its ``forced_central_body`` path is exactly this "told-the-center" mode, central-body
reporting is done at OUTPUT over an unchanged heliocentric integration (``find_central_object``,
elem_out.cpp), and a hyperbolic planetocentric solution reverts to "fly-by, not a satellite"
(the ecc>1.01 reset, orb_func.cpp:3762). Auto-detecting the center (find_orb's
``find_best_fit_planet`` mass/dist^2.5 contest) is the deferred (b) path.

Scope note: DE440 carries the giant planets only as SYSTEM BARYCENTRES and has no satellite
kernels, so v1 targets IRREGULAR satellites (Himalia/Phoebe-class -- far out, Sun+planet
dominated) and treats the system barycentre as "the planet". Deep regular moons (needing the
major moons as perturbers + planetary J2) are out of scope.
"""
import math
from pathlib import Path

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u

from pyroorb.forward_model import MU_SUN, PLANETS, los_unit
from pyroorb.orbital_elements import element_vector, ELEM_NAMES

# The validated v1 targets. Inner planets are accepted (the math is identical) but flagged
# unvalidated -- no satellite-kernel perturbers / J2 either way, and a close artsat regime
# needs drag/oblateness we do not model.
GIANTS = ("jupiter", "saturn", "uranus", "neptune")

_ALIASES = {
    "jupiter": "jupiter", "jove": "jupiter", "jovian": "jupiter", "jovicentric": "jupiter",
    "saturn": "saturn", "saturnian": "saturn", "kronian": "saturn", "cronian": "saturn",
    "saturnicentric": "saturn",
    "uranus": "uranus", "uranian": "uranus",
    "neptune": "neptune", "neptunian": "neptune",
    "mars": "mars", "martian": "mars",
    "earth": "earth", "terran": "earth", "geocentric": "earth",
    "mercury": "mercury", "venus": "venus",
}

# find_orb's fly-by threshold: a planetocentric ecc > this is treated as an unbound pass, not
# a captured satellite (orb_func.cpp:3762). We report but flag it rather than fabricate elements.
ECC_UNBOUND = 1.01


def resolve_planet(name):
    """Normalise a user center name to a PLANETS key, or raise ValueError."""
    key = _ALIASES.get(str(name).strip().lower())
    if key is None or key not in PLANETS:
        raise ValueError(
            f"unknown/unsupported center {name!r}; supported: "
            f"{', '.join(GIANTS)} (giant planets, validated) "
            f"or an inner planet name (accepted, unvalidated)")
    return key


def is_validated(name):
    """True for the giant planets (the tested regime); False for accepted-but-unvalidated centers."""
    return resolve_planet(name) in GIANTS


def planet_gm(name):
    """Gravitational parameter GM of the planet (au^3/day^2), in MU_SUN units."""
    return MU_SUN * PLANETS[resolve_planet(name)]


def planet_helio_state(name, jd_tdb):
    """Heliocentric-EQUATORIAL (ICRS) state [x,y,z,vx,vy,vz] (au, au/day) of the planet's
    system barycentre at ``jd_tdb``, from the DE440 ephemeris the fit integrates in (the same
    source as the public body_states), so a planetocentric element or distance is measured
    against the planet the orbit was fitted with. astropy's built-in ephemeris, the previous
    source, puts Jupiter ~80,000 km from DE440: notebook 09 had to fetch its own Jupiter to get
    Ersa's elements within 700 km of JPL's. The built-in ephemeris remains the fallback when
    the planet ephemeris file is absent (the rebound backend, which integrates with it).

    The file is looked for before body_states is called: body_states builds the ASSIST
    ephemeris, which downloads the ~102 MB DE440 when it is missing, and auto-centring calls
    this on ordinary fits -- a rebound-only fit must not block on that download."""
    key = resolve_planet(name)
    from pyroorb import force_assist
    if Path(force_assist.PLANETS_PATH).exists():
        try:
            from pyroorb.products import body_states
            return np.asarray(body_states(key, [float(jd_tdb)])[0], float)
        except Exception:                               # an unreadable DE440: the fallback
            pass
    t = Time(jd_tdb, format="jd", scale="tdb")
    p, v = get_body_barycentric_posvel(key, t)
    sp, sv = get_body_barycentric_posvel("sun", t)
    pos = (p - sp).xyz.to(u.au).value
    vel = (v - sv).xyz.to(u.au / u.day).value
    return np.concatenate([np.asarray(pos, float), np.asarray(vel, float)])


def planet_hinted_seed(obs, geom, planet, vel_span=None, speed_scale=None):
    """Planet-hinted IOD seed for a satellite of ``planet``.

    Returns ``(state_eq_helio, seed_epoch_tdb)``: a heliocentric-equatorial 6-state anchored at
    the middle observation time (within-arc, per find_orb's epoch rule). Places the object at the
    planet's topocentric distance along each line of sight (it is inside the Hill sphere) and
    splits the heliocentric velocity into the planet's bulk motion + a planetocentric term.

    The planetocentric velocity DIRECTION is a central difference of the planetocentric offset over
    a short baseline bracketing the mid-epoch (``vel_span`` days, default ~1/8 of the arc, capped);
    a full-arc secant badly under-estimates the tangent once the arc spans a large fraction of the
    (~hundreds-of-days) period. The planetocentric SPEED is the single most basin-determining
    quantity (it sets the eccentricity) and angles-only cannot pin it, so ``_fit_from_planet_seed``
    MULTI-STARTS over ``speed_scale`` -- the speed as a multiple of the local circular speed
    sqrt(GM/r) -- and keeps the best-fitting orbit. ``speed_scale=None`` uses the raw secant speed."""
    Robs = geom["Robs"]
    times = np.asarray(geom["times_tdb"], float)
    ra = np.asarray(geom["obs_ra"], float)
    dec = np.asarray(geom["obs_dec"], float)
    n = len(times)
    if n < 2:
        raise ValueError("planet-hinted seed needs at least 2 observations")
    order = np.argsort(times)
    t_sorted = times[order]
    mid = n // 2
    i1 = order[mid]
    arc = float(t_sorted[-1] - t_sorted[0])
    if vel_span is None:
        vel_span = min(30.0, max(1.0, arc / 8.0))

    def offset(i):
        Rp = planet_helio_state(planet, times[i])
        Ro = np.asarray(Robs[i], float)
        rho_p = float(np.linalg.norm(Rp[:3] - Ro))          # planet's topocentric range
        r_obj = Ro + rho_p * los_unit(ra[i], dec[i])        # object pinned to that range along the LOS
        return r_obj, Rp, r_obj - Rp[:3]                    # (helio pos, planet state, planetocentric offset)

    rm, Pm, off_m = offset(i1)
    tm = t_sorted[mid]
    # central-difference planetocentric velocity over a short baseline bracketing tm
    lo = order[int(np.argmin(np.abs(t_sorted - (tm - vel_span / 2.0))))]
    hi = order[int(np.argmin(np.abs(t_sorted - (tm + vel_span / 2.0))))]
    if times[hi] == times[lo]:                              # degenerate -> widen to the whole arc
        lo, hi = order[0], order[-1]
    _, _, off_lo = offset(lo)
    _, _, off_hi = offset(hi)
    dt = times[hi] - times[lo]
    v_plan = (off_hi - off_lo) / dt if dt != 0 else np.zeros(3)
    if speed_scale is not None:
        # keep the observed direction, set the speed to speed_scale * local circular speed
        vn = float(np.linalg.norm(v_plan))
        direction = v_plan / vn if vn > 0 else np.array([0.0, 0.0, 1.0])
        r = float(np.linalg.norm(off_m))
        v_circ = math.sqrt(planet_gm(planet) / r) if r > 0 else 0.0
        v_plan = speed_scale * v_circ * direction
    v_seed = Pm[3:] + v_plan                                 # planet bulk motion + planetocentric
    state = np.concatenate([rm, v_seed])
    return state, float(tm)


# --------------------------------------------------------------------------- auto-detection (b)

def mass_contest(state_helio_eq, epoch_tdb, candidates=GIANTS):
    """find_orb's ``find_best_fit_planet`` gravitational-dominance test (runge.cpp:1346), ported:
    score each body by ``fitness = mass / dist**2.5`` (mass in solar masses, dist from the object
    in au) and return ``[(name, fitness, dist_au), ...]`` sorted best-first, with the Sun as the
    baseline (``i==0`` in find_orb). The extra ``dist**0.5`` over a pure ``mass/dist**2`` force
    biases toward the NEARER body when forces are comparable (find_orb's tidal-flavoured weighting).

    Crucially this depends on the object's POSITION only, so it names the dominating planet even
    when the heliocentric velocity -- hence the full converged orbit -- is garbage, which is exactly
    the state an ordinary heliocentric fit lands in for a satellite. Logic ported, no lines copied."""
    r = np.asarray(state_helio_eq, float).ravel()[:3]
    rsun = float(np.linalg.norm(r))
    scored = [("sun", 1.0 / rsun ** 2.5, rsun)]
    for name in candidates:
        d = float(np.linalg.norm(r - planet_helio_state(name, epoch_tdb)[:3]))
        scored.append((name, PLANETS[name] / d ** 2.5, d))
    scored.sort(key=lambda x: -x[1])
    return scored


def detect_center(state_helio_eq, epoch_tdb, candidates=GIANTS):
    """The winning giant planet if it dominates the Sun in the mass/dist^2.5 contest, else None
    (heliocentric). A pure POSITION test: the caller must still re-fit about the winner and confirm
    the planetocentric orbit is bound before declaring a satellite -- find_best_fit_planet is
    likewise followed by find_orb's ecc>1.01 fly-by reset (orb_func.cpp:3762).

    NOTE: this reads the converged heliocentric STATE, which for a satellite is garbage (an
    ordinary heliocentric fit can't place a planet-bound object) and on short arcs can drift
    outside the Hill sphere -> unreliable. Prefer ``satellite_candidate`` (below), which detects
    from the OBSERVATIONS directly and does not depend on the failed fit."""
    best = mass_contest(state_helio_eq, epoch_tdb, candidates)[0]
    return best[0] if best[0] != "sun" else None


def hill_radius_au(name, jd_tdb):
    """Hill-sphere radius (au) of the planet: r_H = a (m/3M_sun)^(1/3), with the planet's current
    heliocentric distance as a. A satellite orbits WITHIN this radius by definition."""
    a = float(np.linalg.norm(planet_helio_state(name, jd_tdb)[:3]))
    return a * (PLANETS[name] / 3.0) ** (1.0 / 3.0)


# Fraction of an arc's observations that must fall inside a planet's Hill cone to call it a
# satellite candidate. Below 1.0 so a stray outlier obs (common in real MPC astrometry) can't
# veto a genuine satellite; a Trojan/co-orbital scores ~0 inside, so the gap is wide.
SAT_INSIDE_FRAC = 0.7


def dominating_planet(state_helio_eq, epoch_tdb, candidates=GIANTS, factor=1.2):
    """The giant planet whose HILL SPHERE (sphere of influence) contains the object at this epoch,
    or None -- a 3-D position test on the (heliocentric) state. This is find_orb's ``find_best_fit_
    planet`` domain check recast as a boundary test: an object inside a planet's Hill sphere is
    gravitationally bound to it. ``factor`` scales the Hill radius for a little slack (a slightly
    off converged position / the system-barycentre-vs-centre approximation). Cheap -- one planet
    lookup per candidate -- and used as the auto-detect TRIGGER, so a satellite is caught even when
    its heliocentric fit is perfectly good (a short satellite arc fits heliocentrically fine). A
    normal asteroid at 2-3 au is nowhere near any giant planet, so this is free for the common case."""
    r = np.asarray(state_helio_eq, float).ravel()[:3]
    r_helio = float(np.linalg.norm(r))
    best = None
    for name in candidates:
        # free pre-filter: an object can only sit in a planet's Hill sphere if its heliocentric
        # distance is within ~a-few-au of the planet's -- skip the ephemeris lookup otherwise, so a
        # 2-3 au main-belt object (far from every giant planet) costs nothing on the common path.
        if abs(r_helio - _APPROX_HELIO_DIST[name]) > 1.5:
            continue
        Rp = planet_helio_state(name, epoch_tdb)[:3]       # one lookup per (surviving) candidate
        rH = float(np.linalg.norm(Rp)) * (PLANETS[name] / 3.0) ** (1.0 / 3.0)
        d = float(np.linalg.norm(r - Rp))
        if d < factor * rH and (best is None or d / rH < best[1]):
            best = (name, d / rH)
    return best[0] if best else None


# Nominal heliocentric distances (au) of the giant planets, for the cheap Hill pre-filter above.
_APPROX_HELIO_DIST = {"jupiter": 5.2, "saturn": 9.6, "uranus": 19.2, "neptune": 30.1}


def satellite_candidate(obs_ra, obs_dec, Robs, times_tdb, candidates=GIANTS, margin=1.3):
    """Detect a giant-planet satellite from the OBSERVATIONS ALONE -- robust to the garbage
    heliocentric fit (uses no fitted state). A satellite orbits inside its planet's Hill sphere,
    so its observed sky position stays within the Hill-sphere ANGULAR radius of the planet across
    the arc. For each candidate planet we compare, at every obs, the angle between the observed
    line of sight and the planet's topocentric direction against the Hill angular radius
    (``margin`` allows a little slack for the Hill edge + light-time), and require at least
    ``SAT_INSIDE_FRAC`` of the obs inside (tolerating outliers). Returns the best
    ``(planet, median_sep_deg, median_hill_deg)`` (smallest median separation relative to its Hill
    cone) or None.

    This is the position half of find_orb's ``find_best_fit_planet`` recast onto the data: a Trojan
    or co-orbital sits ~5 au (far outside the Hill cone) from the planet and is rejected; a real
    asteroid is nowhere near a giant planet. The caller still re-fits + confirms boundedness."""
    obs_ra = np.asarray(obs_ra, float); obs_dec = np.asarray(obs_dec, float)
    times = np.asarray(times_tdb, float)
    n = len(times)
    los = np.array([los_unit(obs_ra[k], obs_dec[k]) for k in range(n)])
    best = None
    for name in candidates:
        seps = np.empty(n); hills = np.empty(n)
        for k in range(n):
            geo = planet_helio_state(name, times[k])[:3] - np.asarray(Robs[k], float)
            gd = float(np.linalg.norm(geo))
            cang = float(np.dot(geo / gd, los[k]))
            seps[k] = math.degrees(math.acos(max(-1.0, min(1.0, cang))))
            hills[k] = math.degrees(math.asin(min(1.0, hill_radius_au(name, times[k]) / gd)))
        frac_inside = float(np.mean(seps <= margin * hills))
        if frac_inside >= SAT_INSIDE_FRAC:
            med_sep = float(np.median(seps)); med_hill = float(np.median(hills))
            score = med_sep / med_hill
            if best is None or score < best[3]:
                best = (name, med_sep, med_hill, score)
    return None if best is None else (best[0], best[1], best[2])


# --------------------------------------------------------------------------- reporting

def planetocentric_elements(state_eq_helio, epoch_tdb, planet):
    """Classical planetocentric elements (J2000 ecliptic) from a heliocentric-equatorial state.

    Reframes onto the planet (subtract its state) and uses the planet's GM. Returns a dict of the
    standard ELEM_NAMES plus ``planet``, ``planetocentric_range_au``, ``rel_state_eq`` and a
    ``bound`` flag (find_orb's ecc<1.01 captured-satellite test)."""
    key = resolve_planet(planet)
    Rp = planet_helio_state(key, epoch_tdb)
    rel = np.asarray(state_eq_helio, float).ravel()[:6] - Rp
    elem = element_vector(rel, epoch_tdb, mu=planet_gm(key))
    d = dict(zip(ELEM_NAMES, [float(x) for x in elem]))
    d["planet"] = key
    d["rel_state_eq"] = rel
    d["planetocentric_range_au"] = float(np.linalg.norm(rel[:3]))
    d["bound"] = bool(np.isfinite(d["e"]) and d["e"] < ECC_UNBOUND and d["a"] > 0)
    return d


def planetocentric_sigmas(fit, planet):
    """1-sigma element uncertainties for the planetocentric orbit, propagating the fit's STATE
    covariance through G = d(planet elements)/d(state). Because rel = state - planet_state(epoch)
    is a constant offset at the fixed epoch, d(rel)/d(state) = I, so the same state covariance the
    heliocentric fit already carries maps straight to the planetocentric elements. Returns a dict
    keyed by ELEM_NAMES (NaN where the covariance is unavailable)."""
    key = resolve_planet(planet)
    s0 = np.asarray(fit.state_vector(), float).ravel()[:6]
    ep = float(fit.epoch_tdb)
    mu = planet_gm(key)
    Rp = planet_helio_state(key, ep)

    def elems(s):
        return element_vector(np.asarray(s, float).ravel()[:6] - Rp, ep, mu=mu)

    e0 = elems(s0)
    try:
        cov_s = np.asarray(fit.covariance_matrix(frame="state"), float)[:6, :6]
    except Exception:
        return {nm: float("nan") for nm in ELEM_NAMES}
    G = np.zeros((len(e0), 6))
    for k in range(6):
        h = 1e-7 * max(1.0, abs(s0[k]))
        sp = s0.copy(); sp[k] += h
        de = elems(sp) - e0
        for a in (4, 5, 6, 7):                              # wrap angle differences (deg)
            de[a] = (de[a] + 180.0) % 360.0 - 180.0
        G[:, k] = de / h
    try:
        sig = np.sqrt(np.clip(np.diag(G @ cov_s @ G.T), 0.0, None))
    except Exception:
        sig = np.full(len(e0), np.nan)
    return dict(zip(ELEM_NAMES, [float(x) for x in sig]))


def classify(elem):
    """A short satellite class string from a planetocentric element dict."""
    planet = elem["planet"].capitalize()
    if not elem["bound"]:
        return f"unbound / hyperbolic pass of {planet} (not a captured satellite over this arc)"
    sense = "retrograde" if elem["i"] > 90.0 else "prograde"
    return f"{sense} satellite of {planet}"


def report(fit, planet):
    """Formatted multi-line CLI report of the planetocentric orbit for a converged fit."""
    key = resolve_planet(planet)
    elem = planetocentric_elements(fit.state_vector(), fit.epoch_tdb, key)
    sig = planetocentric_sigmas(fit, key)
    au_km = 1.495978707e8
    lines = []
    lines.append(f"Planetocentric orbit  ({classify(elem)})")
    lines.append(f"  center           : {key.capitalize()} (system barycentre; frame J2000 ecliptic)")
    lines.append(f"  epoch (TDB)      : JD {fit.epoch_tdb:.5f}")
    rng = elem["planetocentric_range_au"]
    lines.append(f"  range from {key[:3].capitalize():3}   : {rng:.6f} au ({rng * au_km:,.0f} km)")
    if not elem["bound"]:
        lines.append(f"  eccentricity     : {elem['e']:.4f}  -> UNBOUND (e > {ECC_UNBOUND})")
        lines.append("  (arc does not constrain a bound orbit about this planet; not reporting a/P.)")
    else:
        def pm(name, unit="", scale=1.0, fmt="{:.6f}"):
            v = elem[name] * scale
            s = sig.get(name, float("nan"))
            body = fmt.format(v)
            if np.isfinite(s):
                body += " +/- " + fmt.format(s * scale)
            return f"  {name:16s}: {body} {unit}".rstrip()
        a_km = elem["a"] * au_km
        lines.append(f"  a               : {elem['a']:.6f} au ({a_km:,.0f} km)"
                     + (f"  +/- {sig['a'] * au_km:,.0f} km" if np.isfinite(sig.get('a', float('nan'))) else ""))
        lines.append(pm("e", fmt="{:.5f}"))
        lines.append(pm("i", "deg", fmt="{:.4f}"))
        lines.append(pm("asc_node", "deg", fmt="{:.4f}"))
        lines.append(pm("arg_per", "deg", fmt="{:.4f}"))
        lines.append(pm("M", "deg", fmt="{:.4f}"))
        lines.append(f"  q / Q           : {elem['q']:.6f} / {elem['Q']:.6f} au")
        if np.isfinite(elem["P"]):
            lines.append(f"  period          : {elem['P'] * 365.25:.2f} d ({elem['P']:.4f} yr)")
    if not is_validated(key):
        lines.append("  NOTE: inner-planet center is UNVALIDATED in v1 (validated: "
                     + ", ".join(GIANTS) + ").")
    return "\n".join(lines)
