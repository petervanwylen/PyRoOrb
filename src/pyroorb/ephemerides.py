"""Output ephemeris: predicted observable positions over a time range, with the
sky-plane positional UNCERTAINTY (find_orb's headline ephemeris product).

`compute_ephemeris(fit, obscodes, jd_start, jd_stop, step_days, site=...)` returns a
list of per-epoch dicts (structured, for the GUI/JSON); `format_table(rows)` renders a
find_orb-style fixed-width text table from them.

Uncertainty: we propagate the fitted 6x6 state covariance to each epoch with the same
analytic Jacobian the fit uses (J = d(RA,Dec)/d(state0)). The 2x2 sky covariance
C = J Cov J^T (arcsec^2, with RA scaled by cos(Dec)) gives the full 1-sigma error
ellipse — semi-major / semi-minor axes and its position angle (N through E) — plus the
along-track / cross-track split relative to the object's sky motion. (find_orb's default
ephemeris only reports the major axis via a single variant orbit; the Jacobian gives the
whole ellipse directly.)

Row filtering (find_orb's two ephemeris knobs, `ephem0.cpp`): a FAINT LIMIT drops epochs
predicted fainter than a V magnitude (find_orb's `ephemeris_mag_limit`, default 22), and
SUPPRESS UNOBSERVABLES drops epochs the requested observatory could not actually shoot —
target below the horizon, Sun up, or the Moon in the way. Both are opt-in and cosmetic:
they hide rows, never change the orbit. See `filter_rows` for the ported predicate.
"""
import math
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz, get_body,
                                  GeocentricTrueEcliptic)

from pyroorb import photometry
from pyroorb import eop                          # the one Earth-orientation table
eop.pin()

DAY_MIN = 1440.0          # minutes per day
MOTION_DT = 0.01          # day; finite-difference step for sky motion (~14 min)
R_EARTH_M = 6378137.0     # Earth equatorial radius (m), for obscode -> EarthLocation
R_MOON_KM = 1737.4        # lunar radius (find_orb ephem0.cpp), for the "behind the Moon" test
AU_KM = 149597870.7

# find_orb's default ephemeris magnitude cut-off (`ephemeris_mag_limit`, ephem0.cpp):
# rows predicted fainter than this are dropped when a faint limit is in force.
FAINT_LIMIT_DEFAULT = 22.0


def _sky_cov(jac_i, dec_deg, state_cov):
    """2x2 sky covariance (arcsec^2) in the (RA*cosDec [East], Dec [North]) basis at one
    epoch, from the d(RA_deg,Dec_deg)/d(state) Jacobian and the 6x6 state covariance."""
    J = jac_i * 3600.0                                   # deg -> arcsec
    J = J.copy()
    J[0, :] *= math.cos(math.radians(dec_deg))           # RA -> RA*cos(Dec) (true sky angle)
    return J @ state_cov @ J.T


def _ellipse(C2):
    """Error ellipse from a 2x2 sky covariance: (semi_major", semi_minor", PA_deg).
    PA is the major-axis orientation measured North through East, reduced to 0..180."""
    w, V = np.linalg.eigh(C2)                            # ascending eigenvalues
    smin = math.sqrt(max(0.0, w[0]))
    smaj = math.sqrt(max(0.0, w[1]))
    e, n = V[0, 1], V[1, 1]                              # major eigenvector (East, North)
    pa = math.degrees(math.atan2(e, n)) % 180.0
    return smaj, smin, pa


def _nongrav_pair(fit):
    """``(A, gr)`` for a non-gravitational fit -- the pair every propagator here
    accepts -- or None for a pure-gravity one (and for any fit whose model is
    absent, "none", or carries no coefficients)."""
    model = getattr(fit, "nongrav_model", None)
    if not model or model == "none":
        return None
    params = getattr(fit, "nongrav_params", None) or {}
    try:
        import pyroorb.nongrav as ng
        A = [float(params.get(ng.A_AXES[ax], 0.0)) for ax in range(3)]
        if not any(A):
            return None
        return A, ng.model(model)["gr"]
    except Exception:
        return None


def linear_parameters(fit, nongrav=True):
    """What a LINEAR uncertainty of `fit` differentiates, with its covariance:
    ``(pair, free_idx, cov)``.

    A non-gravitational fit propagated with its force (`nongrav` True) is uncertain in its
    free A coefficients as well as its state, and the two are correlated: an uncertain
    acceleration spreads the orbit further the longer it acts. So the parameters are the six
    state components then the free A axes (`free_idx`, 0/1/2 = A1/A2/A3, fit order), and
    `cov` is the fit's full parameter covariance over them (`FittedOrbit.parameter_covariance`,
    au/day^2 for the A rows). Otherwise -- a gravity fit, or the same state propagated without
    its force -- the six state components and the state covariance (`pair` None, `free_idx`
    empty). `pair` is `_nongrav_pair(fit)`, the (A, gr) the propagators take."""
    pair = _nongrav_pair(fit) if nongrav else None
    if pair is None:
        return None, [], np.asarray(fit.covariance_matrix("state"), float)
    import pyroorb.nongrav as ng
    cov, names = fit.parameter_covariance()
    return pair, [ng.A_AXES.index(n) for n in names[6:]], np.asarray(cov, float)


# The Jacobian's displaced copies carry the covariance to the sky, so their step must stay in
# the LINEAR part of the orbit's own uncertainty. The fitter's step (1.49e-8 x |x|, scipy-style)
# is ~350 sigma in velocity for a well-observed NEA; through a deep planetary encounter a kick
# that size goes non-linear and the ellipse comes out too wide (Apophis at its 2029 pass,
# 38 000 km: 265" against a 300-clone cloud's 3.7", velocity columns up to 4.4x a 0.01-sigma
# central difference). Capping the step at 0.01 sigma per axis agrees with the clones there and
# leaves every orbit whose sigma already exceeds the fitter's step exactly as it was.
JAC_STEP_SIGMAS = 1e-2
JAC_STEP_FLOOR = 1e-12      # relative; a zero-sigma (dof 0) axis keeps a round-off-safe step


def _jacobian_steps(state, cov):
    """Per-axis step for the ephemeris Jacobian: the fitter's step, capped at JAC_STEP_SIGMAS
    of the axis's own 1-sigma (never below JAC_STEP_FLOOR relative)."""
    s = np.asarray(state, float)[:6]
    scale = np.maximum(1.0, np.abs(s))
    h = 1.49e-8 * scale
    sig = np.sqrt(np.clip(np.diag(np.asarray(cov, float))[:6], 0.0, None))
    cap = np.maximum(JAC_STEP_SIGMAS * sig, JAC_STEP_FLOOR * scale)
    return np.where(np.isfinite(cap), np.minimum(h, cap), h)


def _jacobian_A_steps(A, free_idx, cov):
    """Step for each free A axis of the ephemeris Jacobian: predict_radec_jac_ng's own
    default (1e-3 of the coefficient), capped at JAC_STEP_SIGMAS of the axis's 1-sigma like
    the state steps; an axis without a usable sigma keeps the default."""
    a = np.abs(np.asarray(A, float)[list(free_idx)])
    h = np.where(a > 0, 1e-3 * a, 1.0)
    sig = np.sqrt(np.clip(np.diag(np.asarray(cov, float))[6:6 + len(free_idx)], 0.0, None))
    cap = JAC_STEP_SIGMAS * sig
    return np.where(np.isfinite(cap) & (cap > 0), np.minimum(h, cap), h)


def compute_ephemeris(fit, obscodes, jd_start, jd_stop, step_days, site="500",
                      backend=None, full=True, observer=None):
    """Ephemeris for `fit` over [jd_start, jd_stop] (UTC JD) at `step_days`, as seen from
    MPC observatory `site` (default 500 = geocentre). Returns a list of per-epoch dicts.
    `full=True` adds the astropy-derived columns (alt/az, sun/moon, galactic/ecliptic, …).
    `observer` replaces the station offset by a geocentric equatorial position (au): one
    vector for every epoch, or one per epoch. That is how a space telescope is placed --
    the fit reads the same vector off each observation record (`sat_pos_au`); an ephemeris
    is given only a code, and C57 (TESS, up to ~100,000 km from the geocentre) came out
    ~800" off without it. The vector is already inertial, so it takes no Earth rotation,
    and the ground-site columns (altitude, airmass) are not computed for it."""
    from pyroorb.obs_sources import observer_positions_batch, _force_backend
    if backend is None:
        backend, _ = _force_backend(fit)      # keeps a self-perturber's pinned force model
    state = np.asarray(fit._state)
    t0 = fit.epoch_tdb
    # A NON-GRAVITATIONAL fit carries a 7+-element state (6 orbital + the free A
    # coefficients). The ACCELERATION goes to the propagators as the (A, gr) pair they
    # accept, and the sky-plane sigma differentiates the 6 state components AND the free A
    # axes (linear_parameters): the A coefficients' own uncertainty, and its correlation with
    # the state, widen the ellipse the longer the force has acted since the epoch.
    nongrav, free_idx, cov = linear_parameters(fit)
    state = state[:6]
    H, _ = fit.absolute_magnitude()
    G = fit.slope_parameter()

    jds = np.arange(jd_start, jd_stop + 1e-9, step_days)
    n = len(jds)
    tt = Time(jds, format="jd", scale="utc").tdb.jd
    base_sim = backend.build_base_sim(t0)

    # observer (heliocentric, equatorial) at each epoch, and a step later for sky motion.
    # A caller-supplied geocentric vector stands in for the station offset at both instants
    # (a spacecraft moves metres in MOTION_DT, far below the ephemeris's precision).
    sat_pos = _observer_vectors(observer, n)
    Robs = np.asarray(observer_positions_batch(jds, [site] * n, obscodes, sat_pos=sat_pos))
    Robs2 = np.asarray(observer_positions_batch(jds + MOTION_DT, [site] * n, obscodes,
                                                sat_pos=sat_pos))
    tt2 = Time(jds + MOTION_DT, format="jd", scale="utc").tdb.jd

    # The orbit is predicted in the force model it was DETERMINED in: an orbit fitted or
    # polished with the 16 asteroid perturbers (the fit backend's pin, products.fit_asteroids)
    # is propagated with them, and a planets-only orbit without them, whatever the
    # process-wide switch says at call time. Read from the switch instead, a polished (197)
    # Arete's ephemeris sat 0.08-0.38" (7-25 sigma of its own ellipse) off its own orbit across
    # its 2030 Vesta pass. Only the ASSIST backend takes the flag; the rebound fallback has no
    # perturbers to choose.
    from pyroorb.products import fit_asteroids
    ast = ({"asteroids": fit_asteroids(fit)} if hasattr(backend, "ensure_asteroid_kernel")
           else {})

    # nominal RA/Dec (+ analytic Jacobian if the backend supplies one) and the geometry
    jacfn = getattr(backend, "predict_radec_jac_batch", None)
    jacng = getattr(backend, "predict_radec_jac_ng", None)
    h = _jacobian_steps(state, cov)
    if nongrav is not None and jacng is not None:
        A, gr = nongrav
        # (n, 2, 6+k): the state columns, then one per free A axis (physical units)
        pred, jac = jacng(t0, state, list(A), free_idx, gr, tt, list(Robs),
                          hstate=h, hA=_jacobian_A_steps(A, free_idx, cov), **ast)
    elif jacfn is not None:
        pred, jac = jacfn(base_sim, t0, state, tt, list(Robs), h=h, **ast)
    else:
        pred = backend.predict_radec_batch(base_sim, t0, state, tt, list(Robs),
                                           nongrav=nongrav, **ast)
        jac = _finite_diff_jac(backend, base_sim, t0, state, tt, Robs)
    if np.shape(jac)[-1] < cov.shape[0]:
        cov = cov[:6, :6]            # state columns only: the state block, A marginalised
    pred2 = backend.predict_radec_batch(base_sim, t0, state, tt2, list(Robs2),
                                        nongrav=nongrav, **ast)                  # for sky motion

    helio = backend.helio_positions_batch(base_sim, t0, state, tt,
                                          nongrav=nongrav, **ast)                # asteroid helio (n,3)
    r = np.linalg.norm(helio, axis=1)                                        # heliocentric dist
    earth_sun = np.linalg.norm(Robs, axis=1)                                 # observer helio dist
    delta = np.linalg.norm(helio - Robs, axis=1)                             # observer->target dist
    elong = np.degrees(np.arccos(np.clip(
        (delta**2 + earth_sun**2 - r**2) / (2 * delta * earth_sun), -1, 1)))
    phase = np.degrees(np.arccos(np.clip(
        (r**2 + delta**2 - earth_sun**2) / (2 * r * delta), -1, 1)))

    rows = []
    for i in range(n):
        ra, dec = float(pred[i, 0]), float(pred[i, 1])
        cosd = math.cos(math.radians(dec))
        # sky motion (arcsec/min) + PA of motion (N through E)
        dE = ((pred2[i, 0] - ra + 180) % 360 - 180) * cosd * 3600.0          # East arcsec
        dN = (pred2[i, 1] - dec) * 3600.0                                    # North arcsec
        dt_min = MOTION_DT * DAY_MIN
        rate = math.hypot(dE, dN) / dt_min
        pa_motion = math.degrees(math.atan2(dE, dN)) % 360.0
        # magnitude
        red, _ = photometry.phase_reduction(r[i], delta[i], earth_sun[i], G)
        vmag = (H + red) if (H is not None and red < 9000.0) else None
        # positional uncertainty ellipse + along/cross-track
        C2 = _sky_cov(jac[i], dec, cov)
        smaj, smin, pa_unc = _ellipse(C2)
        if rate > 0:
            u = np.array([dE, dN]) / math.hypot(dE, dN)                      # along-motion unit
            up = np.array([-u[1], u[0]])                                     # cross-motion unit
            sig_along = math.sqrt(max(0.0, u @ C2 @ u))
            sig_cross = math.sqrt(max(0.0, up @ C2 @ up))
        else:
            sig_along = sig_cross = None
        rows.append({
            "jd_utc": float(jds[i]),
            "date_iso": Time(jds[i], format="jd", scale="utc").iso[:19],
            "ra_deg": ra, "dec_deg": dec,
            "ra_hms": _ra_hms(ra), "dec_dms": _dec_dms(dec),
            "delta_au": float(delta[i]), "r_au": float(r[i]),
            "elong_deg": float(elong[i]), "phase_deg": float(phase[i]),
            "v_mag": vmag,
            "motion_arcsec_min": rate, "motion_pa_deg": pa_motion,
            "unc_smaj_arcsec": smaj, "unc_smin_arcsec": smin, "unc_pa_deg": pa_unc,
            "unc_along_arcsec": sig_along, "unc_cross_arcsec": sig_cross,
        })
    if full:
        _add_observer_columns(rows, None if sat_pos is not None else site, obscodes)
    return rows


def _observer_vectors(observer, n):
    """`observer` as a per-epoch list of geocentric equatorial vectors (au), or None.
    One (3,) vector is repeated for every epoch; an (n, 3) array is used row by row."""
    if observer is None:
        return None
    v = np.asarray(observer, dtype=float)
    if v.shape == (3,):
        return [v] * n
    if v.shape == (n, 3):
        return list(v)
    raise ValueError(f"observer must be one (3,) vector or one per epoch (({n}, 3)); "
                     f"got shape {v.shape}")


def _earth_location(site, obscodes):
    """EarthLocation for an MPC obscode from its parallax constants, or None for a
    geocentric / space-based code (rho == 0)."""
    if site not in obscodes:
        return None
    lon, rcp, rsp = obscodes[site]
    if abs(rcp) < 1e-9 and abs(rsp) < 1e-9:        # geocentre (500) or roving/space
        return None
    x = R_EARTH_M * rcp * math.cos(math.radians(lon))
    y = R_EARTH_M * rcp * math.sin(math.radians(lon))
    z = R_EARTH_M * rsp
    return EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)


def _add_observer_columns(rows, site, obscodes):
    """Add the astropy-derived columns (alt/az, airmass, sun/moon, galactic & ecliptic
    coords, constellation, lunar elong/phase/illumination) to each row in place."""
    from astropy.coordinates import get_constellation
    jds = np.array([r["jd_utc"] for r in rows])
    t = Time(jds, format="jd", scale="utc")
    sc = SkyCoord([r["ra_deg"] for r in rows] * u.deg,
                  [r["dec_deg"] for r in rows] * u.deg, frame="icrs")
    gal = sc.galactic
    ecl = sc.transform_to(GeocentricTrueEcliptic())
    constel = get_constellation(sc)
    moon = get_body("moon", t)
    sun = get_body("sun", t)
    # angular separations must use DIRECTIONS only: get_body returns geocentric coords WITH
    # distance, and separation() against a barycentric ICRS target would inject a ~1 au
    # parallax shift for the nearby Moon. Strip the distance (the target is at infinity).
    moon_dir = SkyCoord(moon.ra, moon.dec)
    sun_dir = SkyCoord(sun.ra, sun.dec)
    lun_elong = sc.separation(moon_dir).deg
    illum = (1.0 - np.cos(sun_dir.separation(moon_dir).rad)) / 2.0   # illuminated fraction 0..1
    # apparent lunar semi-diameter: an object inside it is in front of / behind the Moon,
    # which find_orb marks unobservable ('l'/'L').
    moon_sd = np.degrees(np.arcsin(np.clip(
        R_MOON_KM / moon.distance.to(u.km).value, -1, 1)))
    loc = _earth_location(site, obscodes)
    if loc is not None:
        aa = AltAz(obstime=t, location=loc)
        alt = sc.transform_to(aa)
        sun_alt = sun.transform_to(aa).alt.deg
        moon_alt = moon.transform_to(aa).alt.deg
        tgt_alt = alt.alt.deg; tgt_az = alt.az.deg
        airmass = [float(a.secz) if a.alt.deg > 3 else None for a in alt]
    else:
        tgt_alt = tgt_az = sun_alt = moon_alt = [None] * len(rows)
        airmass = [None] * len(rows)
    for i, r in enumerate(rows):
        r.update({
            "gal_l_deg": float(gal.l.deg[i]), "gal_b_deg": float(gal.b.deg[i]),
            "ecl_lon_deg": float(ecl.lon.deg[i]), "ecl_lat_deg": float(ecl.lat.deg[i]),
            "constellation": str(np.atleast_1d(constel)[i]),
            "lunar_elong_deg": float(lun_elong[i]), "moon_illum": float(illum[i]),
            "moon_sd_deg": float(np.atleast_1d(moon_sd)[i]),
            "alt_deg": _f(tgt_alt, i), "az_deg": _f(tgt_az, i),
            "airmass": airmass[i],
            "sun_alt_deg": _f(sun_alt, i), "moon_alt_deg": _f(moon_alt, i),
        })
    return rows


def _f(arr, i):
    return None if arr is None or arr[i] is None else float(arr[i])


# ---------------------------------------------------------------------------
# row filtering: faint limit + suppress-unobservables
# ---------------------------------------------------------------------------

def unobservable_reason(row):
    """Why observatory `site` could not shoot this epoch, or None if it could —
    find_orb's `visibility_char` test (ephem0.cpp), reduced to what its DEFAULT site
    configuration ("500" in scope.json: alt 0..90, dec -90..90, HA/elong wide open)
    actually rejects, plus the nighttime-only rule that guards optical astrometry:

      * ``horizon`` — target altitude at or below 0 deg (find_orb's 'a'/'B');
      * ``daylight`` — the Sun is above the horizon (find_orb's `nighttime_only`);
      * ``moon``     — the target is inside the lunar disc ('l'/'L').

    A row with no altitude columns (a geocentric or space-based site, or `full=False`)
    has no horizon to be under: it is never suppressed — the same reason find_orb only
    offers the option for a topocentric code."""
    alt, sun = row.get("alt_deg"), row.get("sun_alt_deg")
    if alt is None or sun is None:
        return None
    if alt <= 0.0:
        return "horizon"
    if sun > 0.0:
        return "daylight"
    sd, sep = row.get("moon_sd_deg"), row.get("lunar_elong_deg")
    if sd is not None and sep is not None and sep < sd:
        return "moon"
    return None


def filter_rows(rows, faint_limit=None, suppress_unobservable=False):
    """Apply the two find_orb display cuts, returning (kept_rows, counts). `counts` is a
    reason -> number-dropped dict (``faint``/``horizon``/``daylight``/``moon``) so a caller
    can say what it hid rather than silently returning a short table. A row with no
    predicted magnitude (no H) survives the faint cut — unknown is not faint.

    Every RUN of dropped epochs also leaves a mark on the surviving row that follows it
    (``hidden_before``: the same reason -> count dict), or on the last kept row if the
    run runs off the end (``hidden_after``). That is what lets a renderer put a break
    where the time series jumps, the way find_orb prints a row of dots in place of a
    suppressed stretch — without it a table of 8pm and 4am rows reads as a continuous
    sequence. Kept rows are copied, never mutated in place."""
    kept, counts, run = [], {}, {}
    for r in rows:
        reason = None
        if (faint_limit is not None and r.get("v_mag") is not None
                and r["v_mag"] > float(faint_limit)):
            reason = "faint"
        elif suppress_unobservable:
            reason = unobservable_reason(r)
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
            run[reason] = run.get(reason, 0) + 1
        else:
            if run:
                r = dict(r, hidden_before=run)
                run = {}
            kept.append(r)
    if run and kept:                       # the table ends inside a suppressed stretch
        kept[-1] = dict(kept[-1], hidden_after=run)
    return kept, counts


def _finite_diff_jac(backend, base_sim, t0, state, tt, Robs):
    """Fallback d(RA_deg,Dec_deg)/d(state) by central differences, for a backend without
    an analytic Jacobian (e.g. the plain rebound model)."""
    n = len(tt); jac = np.empty((n, 2, 6))
    for k in range(6):
        h = 1.49e-8 * max(1.0, abs(state[k]))
        sp = state.copy(); sp[k] += h
        sm = state.copy(); sm[k] -= h
        pp = backend.predict_radec_batch(base_sim, t0, sp, tt, list(Robs))
        pm = backend.predict_radec_batch(base_sim, t0, sm, tt, list(Robs))
        jac[:, 0, k] = ((pp[:, 0] - pm[:, 0] + 180) % 360 - 180) / (2 * h)
        jac[:, 1, k] = (pp[:, 1] - pm[:, 1]) / (2 * h)
    return jac


# Canonical sexagesimal RA/Dec formatters live in obs80 (the OBS80 RA/Dec fields use the
# identical format); import them here so the ephemeris display and the astrometry I/O share
# one definition -- and pick up obs80's 60-second rounding carry, which the old local copies
# lacked (an RA of 23.99999 h used to print '24 00 00.000').
from pyroorb.obs80 import ra_hms as _ra_hms, dec_dms as _dec_dms  # noqa: E402


_HEADER = ("Date (UTC)           RA (J2000) Dec        delta    r      elong phase  V    "
           '"/min  mPA  sig-a" sig-b"  uPA  along" cross"')


_HIDDEN_WORDS = {"faint": "too faint", "horizon": "below the horizon",
                 "daylight": "in daylight", "moon": "behind the Moon"}


def hidden_note(counts):
    """One-line summary of what `filter_rows` dropped, or "" if it dropped nothing."""
    if not counts:
        return ""
    return f"{sum(counts.values())} epoch(s) hidden: " + hidden_reasons(counts)


def hidden_reasons(counts):
    """Just the "2 too faint, 3 below the horizon" clause of `hidden_note`, for a caller
    that supplies its own count (a per-break marker, say)."""
    return ", ".join(f"{counts[k]} {_HIDDEN_WORDS[k]}"
                     for k in ("faint", "horizon", "daylight", "moon") if counts.get(k))


def _break_line(counts):
    """The dotted stand-in for a suppressed run of epochs — find_orb prints a bare row of
    dots there; we keep the dots and say how many rows went and why."""
    return f"{'.' * 12} {sum(counts.values())} hidden ({hidden_reasons(counts)}) " \
           + "." * 12


def format_table(rows, site="500", hidden=None):
    """Render the structured ephemeris as a find_orb-style fixed-width text table.
    `hidden` is the counts dict from `filter_rows`, noted above the table (and, when it
    leaves nothing to show, standing in for it as find_orb's no-output message does).
    Rows carrying `hidden_before`/`hidden_after` get a dotted break line, so a stretch
    the display cuts removed is visible in the table and not just in the summary."""
    note = hidden_note(hidden)
    if not rows:
        return ("No ephemeris output. Every epoch was hidden by the display filters "
                f"({note}).") if note else "No ephemeris output."
    out = [f"Ephemeris  (observatory {site})"]
    if note:
        out.append(note)
    out += [_HEADER, "-" * len(_HEADER)]
    for x in rows:
        if x.get("hidden_before"):
            out.append(_break_line(x["hidden_before"]))
        v = f"{x['v_mag']:4.1f}" if x["v_mag"] is not None else "  - "
        al = f"{x['unc_along_arcsec']:6.2f}" if x["unc_along_arcsec"] is not None else "   -  "
        cr = f"{x['unc_cross_arcsec']:6.2f}" if x["unc_cross_arcsec"] is not None else "   -  "
        out.append(
            f"{x['date_iso']:19} {x['ra_hms']} {x['dec_dms']} "
            f"{x['delta_au']:7.4f} {x['r_au']:6.3f} {x['elong_deg']:5.1f} {x['phase_deg']:5.1f} "
            f"{v} {x['motion_arcsec_min']:6.2f} {x['motion_pa_deg']:5.1f} "
            f"{x['unc_smaj_arcsec']:6.2f} {x['unc_smin_arcsec']:6.2f} {x['unc_pa_deg']:4.0f} {al} {cr}")
    if rows[-1].get("hidden_after"):
        out.append(_break_line(rows[-1]["hidden_after"]))
    return "\n".join(out)
