"""A POPULATION PRIOR on a short arc's distance: "if it is compatible with a main-belt orbit, it
is probably in the main belt", written down as a posterior.

The range profile (rangescan.py) says what the astrometry says about the line-of-sight distance
rho: chi2(rho), each node re-fitted over the other five degrees of freedom. On a tracklet that is
usually a wide, flat valley -- every distance from 0.01 to 4 au fits -- and the nominal orbit is
then wherever the least-squares walk happened to stop. This module adds what the astrometry does
not know: how many real objects there are on each of those orbits. digest2's population model
(the Pan-STARRS Synthetic Solar System Model, Grav et al. 2011, binned in perihelion q,
eccentricity e, inclination i and absolute magnitude H; vendored as pydigest2/data/, the RAW
"All" flavour by choice) is turned into a density in phase space and multiplied into
the profile's likelihood:

    p(rho | data) ~ exp(-(chi2(rho) - chi2_min) / 2)                     (the profile)
                    x rho^4                                               (sky coords -> Cartesian)
                    x INTEGRAL over the bound radial velocities of n(q, e, i, H) / V6(bin) / dH

* rho^4 cos^2(dec) is the Jacobian from (ra, dec, ra-rate, dec-rate, rho, rho-dot) to (x, v); the
  four angular quantities are what the tracklet measures, so only rho^4 varies.
* V6(bin) is the bin's PHASE-SPACE volume. In Delaunay variables (L = sqrt(mu a), G = L sqrt(1-e^2),
  H_z = G cos i and three uniform angles) d3x d3v = dL dG dH_z dl dg dh, so the volume of a
  (q, e, i) box has a closed form:
      V6 = (2 pi)^3 mu^(3/2) / 2 * [2/3 q^(3/2)] * [2 (1-e)^(-1/2) + 2 (1-e)^(1/2)] * [-cos i]
  (each bracket differenced across the bin). Spreading a bin's count uniformly in PHASE SPACE
  rather than in (q, e, i) costs nothing at bin resolution and has no singularity -- the
  per-point Jacobian L G^2 e sin i / (2 q (1 - e^2)) vanishes at e = 0 and i = 0, so a uniform-in-
  (q,e,i) density would spike to infinity on every near-circular, near-ecliptic orbit.
* H comes from the observed magnitude at each trial distance (V - 5 log10(r Delta) + phase term,
  G = 0.15), so the prior also asks "is a body THAT bright AT THAT distance common?" -- which is
  what pulls a V = 21 slow mover to 2.5 au rather than 0.02 au. With no photometry the H axis is
  summed out and only the orbit counts.
* The model's last H bin is a catch-all ending at 25.5. Past it the density is EXTENDED with the
  model's own size-distribution slope in that (q, e, i) bin (between its 23-24 and 24-25.5 bins,
  clipped to 0-0.7 dex/mag, 0.35 where the 23-24 bin is empty), up to H = 32. Without that a close
  tiny-NEO solution would get no prior at all -- the imminent-impactor case starved by a table
  edge.

It may choose WHERE TO LOOK, never what is decided. The prior answers "where is the typical object
with this astrometry", which is the right question for choosing which orbit to draw when the user
asks to see one, and for the one extra variant an ITF search tries first (itf.lov_variants.
prior_variant, which adds to the scan and replaces nothing). It is the wrong question for anything
that asks whether this object is unusual: hazard, MOID, PHA, impact, NEOCP ranking, and any
acceptance of a linkage. Those keep the likelihood-only family. An NEO at low elongation near the
ecliptic is exactly the object this prior files under "main belt".
"""
import math
import threading

import numpy as np

from pyroorb.forward_model import MU_SUN
from pyroorb.orbital_elements import OBLIQ

_LOCK = threading.Lock()
_TABLE = None

H_EXTEND_MAX = 32.0        # the extended faint tail stops here (a 2008 TC3 is H 30.9)
H_SLOPE_DEFAULT = 0.35     # dex per mag where the model's own faint slope cannot be read
H_SLOPE_CLIP = (0.0, 0.7)
H_FIRST_LO = 3.0           # the model's first H bin is "< 6"; give it a finite width
E_CAP = 0.99               # the e bin straddling 1 is integrated to here (V6 diverges at e -> 1)
N_S = 41                   # radial-velocity samples per profile node
# Past one night the prior reads the line of variations, and only for a delivered orbit a real
# object rarely has: see most_probable_lov_fit for what each of these was measured to do.
LOV_W0_MAX = 0.01          # the delivered orbit's weight, as a fraction of the peak's, below
                           # which the prior may replace it
LOV_SHOWN_E_MAX = 0.9      # a replacement must itself be clear of near-parabolic
LOV_RMS_MAX = 1.5          # the replacement may fit this much worse than the delivered orbit
LOV_K_MAX = 3.0            # LOV sigmas scanned either side of the delivered orbit
LOV_K_STEP = 0.0005        # one peak was 0.015 sigma wide; a coarse step can step over it


def _edges(upper, first_lo):
    up = np.asarray(upper, float)
    return np.concatenate([[first_lo], up[:-1]]), up


def _table():
    """(density[q,e,i,h], marginal[q,e,i], h_slope[q,e,i], edges) built once from the model."""
    global _TABLE
    t = _TABLE
    if t is not None:
        return t
    with _LOCK:
        if _TABLE is not None:
            return _TABLE
        from pyroorb.digest2.model import load_model
        from pyroorb.digest2.paths import find_model_path
        from pyroorb.digest2.constants import QPART, EPART, IPART, HPART
        n = load_model(find_model_path()).all_ss                       # (29, 8, 11, 18) raw counts
        q_lo, q_hi = _edges(QPART, 0.0)
        e_lo, e_hi = _edges(EPART, 0.0)
        e_hi = np.minimum(e_hi, E_CAP)
        e_lo = np.minimum(e_lo, E_CAP)
        i_lo, i_hi = _edges(np.radians(IPART), 0.0)
        h_lo, h_hi = _edges(HPART, H_FIRST_LO)
        fq = (2.0 / 3.0) * (q_hi ** 1.5 - q_lo ** 1.5)
        F = lambda e: 2.0 / np.sqrt(1.0 - e) + 2.0 * np.sqrt(1.0 - e)
        fe = F(e_hi) - F(e_lo)
        fi = np.cos(i_lo) - np.cos(i_hi)
        vol = ((2 * math.pi) ** 3 * MU_SUN ** 1.5 / 2.0
               * fq[:, None, None] * fe[None, :, None] * fi[None, None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            per_vol = np.where(vol[..., None] > 0, n / vol[..., None], 0.0)
            dens = per_vol / (h_hi - h_lo)[None, None, None, :]
            marginal = np.where(vol > 0, n.sum(axis=3) / vol, 0.0)
            d16, d17 = dens[..., -2], dens[..., -1]
            slope = np.where((d16 > 0) & (d17 > 0),
                             np.log10(np.where(d16 > 0, d17 / np.where(d16 > 0, d16, 1), 1))
                             / (0.5 * (h_lo[-1] + h_hi[-1]) - 0.5 * (h_lo[-2] + h_hi[-2])),
                             H_SLOPE_DEFAULT)
        slope = np.clip(np.nan_to_num(slope, nan=H_SLOPE_DEFAULT), *H_SLOPE_CLIP)
        _TABLE = {"dens": dens, "marginal": marginal, "slope": slope,
                  "q_hi": np.asarray(QPART, float), "e_hi": np.asarray(EPART, float),
                  "i_hi": np.radians(np.asarray(IPART, float)), "h_hi": np.asarray(HPART, float),
                  "h_mid_last": 0.5 * (h_lo[-1] + h_hi[-1])}
        return _TABLE


def qei_ecliptic(states):
    """(q, e, i[rad]) of heliocentric EQUATORIAL states (N, 6), referred to the ecliptic as the
    model is. NaN where the state is unbound or degenerate."""
    s = np.atleast_2d(np.asarray(states, float))
    c, si = math.cos(OBLIQ), math.sin(OBLIQ)
    rot = np.array([[1, 0, 0], [0, c, si], [0, -si, c]])
    r = s[:, :3] @ rot.T
    v = s[:, 3:] @ rot.T
    rn = np.linalg.norm(r, axis=1)
    v2 = np.sum(v * v, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    evec = np.cross(v, h) / MU_SUN - r / rn[:, None]
    e = np.linalg.norm(evec, axis=1)
    energy = 0.5 * v2 - MU_SUN / rn
    with np.errstate(divide="ignore", invalid="ignore"):
        a = -MU_SUN / (2.0 * energy)
        q = a * (1.0 - e)
        inc = np.arccos(np.clip(h[:, 2] / hn, -1.0, 1.0))
    bad = ~(energy < 0) | ~np.isfinite(q) | (hn <= 0)
    q[bad] = np.nan
    e[bad] = np.nan
    inc[bad] = np.nan
    return q, e, inc


def _reduction(r, delta, R):
    """V - H for the H,G system at G = 0.15 (photometry.phase_reduction, vectorised)."""
    cos_a = np.clip((r * r + delta * delta - R * R) / (2.0 * delta * r), -1.0, 1.0)
    t = np.tan(np.arccos(cos_a) / 2.0)
    tval = 0.85 * np.exp(-3.33 * t ** 0.63) + 0.15 * np.exp(-1.87 * t ** 1.22)
    return 5.0 * np.log10(r * delta) - 2.5 * np.log10(np.maximum(tval, 1e-50))


def density(states, R_obs, vmag=None):
    """Population phase-space density at each heliocentric equatorial state (N, 6), per
    unit H when `vmag` (V band) is given, else summed over H. Zero outside the model."""
    T = _table()
    s = np.atleast_2d(np.asarray(states, float))
    q, e, inc = qei_ecliptic(s)
    out = np.zeros(len(s))
    ok = np.isfinite(q) & (e < 1.0) & (q >= 0)
    if not np.any(ok):
        return out
    iq = np.searchsorted(T["q_hi"], q[ok], side="right")
    ie = np.searchsorted(T["e_hi"], e[ok], side="right")
    ii = np.searchsorted(T["i_hi"], inc[ok], side="right")
    inb = (iq < len(T["q_hi"])) & (ie < len(T["e_hi"])) & (ii < len(T["i_hi"]))
    idx = np.flatnonzero(ok)[inb]
    iq, ie, ii = iq[inb], ie[inb], ii[inb]
    if vmag is None:
        out[idx] = T["marginal"][iq, ie, ii]
        return out
    R = float(np.linalg.norm(np.asarray(R_obs, float)))
    rr = np.linalg.norm(s[idx, :3], axis=1)
    dl = np.linalg.norm(s[idx, :3] - np.asarray(R_obs, float)[None, :], axis=1)
    H = float(vmag) - _reduction(rr, dl, R)
    hh = T["h_hi"]
    ih = np.searchsorted(hh, H, side="right")
    val = np.zeros(len(idx))
    inside = ih < len(hh)
    val[inside] = T["dens"][iq[inside], ie[inside], ii[inside], ih[inside]]
    tail = ~inside & (H <= H_EXTEND_MAX)
    if np.any(tail):
        last = T["dens"][iq[tail], ie[tail], ii[tail], -1]
        val[tail] = last * 10.0 ** (T["slope"][iq[tail], ie[tail], ii[tail]]
                                    * (H[tail] - T["h_mid_last"]))
    out[idx] = val
    return out


def _node_mass(row, R0, vmag, n_s=N_S):
    """The prior's weight at one profile node: rho^4 x the density integrated over every
    radial velocity that keeps the orbit bound, at the node's own fitted sky position and
    angular rates (rho-dot moves the heliocentric velocity along the line of sight only)."""
    st = np.asarray(row["state"], float)
    r, v = st[:3], st[3:]
    rho = float(row["rho"])
    u = (r - R0) / rho
    b = float(v @ u)
    c = float(v @ v) - 2.0 * MU_SUN / float(np.linalg.norm(r))
    disc = b * b - c
    if not disc > 0:
        return 0.0
    lo, hi = -b - math.sqrt(disc), -b + math.sqrt(disc)
    # midpoints: the integrand falls to the bound edge, never sample ON it (e = 1 exactly)
    s = lo + (np.arange(n_s) + 0.5) * (hi - lo) / n_s
    states = np.column_stack([np.repeat(r[None, :], n_s, 0), v[None, :] + s[:, None] * u[None, :]])
    return float(np.sum(density(states, R0, vmag)) * (hi - lo) / n_s) * rho ** 4


def posterior(prof, vmag=None, pct=(5.0, 50.0, 95.0)):
    """Population-weighted posterior over the profile's distance axis, or None when it cannot
    be formed (no profile rows, no observer position, or no node the model populates).

    ``{"rho": [...], "weight": [...], "likelihood": [...], "prior": [...],
       "rho_pct": {5: , 50: , 95: }, "rho_med": float, "state_med": [6], "t0_tdb": float,
       "vmag": float | None}``

    `state_med` is the profile's own best state at the posterior-median distance (interpolated
    in log rho between the bracketing nodes), so the orbit it describes fits the tracklet as
    well as the profile does there: the prior chooses the DISTANCE, the data still choose
    everything else."""
    rows = sorted((r for r in (prof or {}).get("rows") or [] if r.get("state") is not None),
                  key=lambda r: r["rho"])
    if len(rows) < 2 or prof.get("R0") is None:
        return None
    R0 = np.asarray(prof["R0"], float)
    rho = np.array([r["rho"] for r in rows])
    chi2 = np.array([r["chi2"] for r in rows], float)
    like = np.exp(-0.5 * (chi2 - np.nanmin(chi2)))
    prior = np.array([_node_mass(r, R0, vmag) for r in rows])
    w = like * prior
    if not np.any(w > 0):
        return None
    # trapezoid in rho: the nodes are log-spaced plus the zoomed fine passes
    seg = 0.5 * (w[1:] + w[:-1]) * np.diff(rho)
    cdf = np.concatenate([[0.0], np.cumsum(seg)])
    tot = cdf[-1]
    if not tot > 0:
        return None
    cdf /= tot

    def at(p):
        k = int(np.searchsorted(cdf, p, side="left"))
        k = min(max(k, 1), len(rho) - 1)
        c0, c1 = cdf[k - 1], cdf[k]
        t = 0.0 if c1 <= c0 else (p - c0) / (c1 - c0)
        return float(rho[k - 1] + t * (rho[k] - rho[k - 1])), k

    out = {f"{q:g}": at(q / 100.0)[0] for q in pct}
    rho_med, k = at(0.5)
    a, b = rows[k - 1], rows[k]
    t = (math.log(rho_med) - math.log(a["rho"])) / (math.log(b["rho"]) - math.log(a["rho"]))
    st = (1.0 - t) * np.asarray(a["state"], float) + t * np.asarray(b["state"], float)
    return {"rho": rho.tolist(), "weight": (w / tot).tolist(), "likelihood": like.tolist(),
            "prior": prior.tolist(), "rho_pct": out, "rho_med": rho_med,
            "state_med": st.tolist(), "t0_tdb": prof.get("t0_tdb"),
            "vmag": None if vmag is None else float(vmag)}


def tracklet_vmag(obs):
    """Mean V of the included observations (the engine's band table), or None without any."""
    from pyroorb import photometry
    vals = []
    for o in obs:
        m = o.get("mag")
        try:
            m = float(m)
        except (TypeError, ValueError):
            continue
        if math.isfinite(m) and m > 0:
            vals.append(m + photometry.band_shift(o.get("band")))
    return float(np.mean(vals)) if vals else None


def most_probable_fit(fit):
    """The orbit to show when the user asks for one on a tracklet that does not determine it:
    the range profile's own solution at the POSTERIOR-MEDIAN distance, as a FittedOrbit on the
    fit's own evaluator (so its residuals, covariance, H and every export are computed exactly
    as for any fit). Past one night, most_probable_lov_fit's orbit. None when no posterior can
    be formed -- the caller then shows the delivered orbit.

    The returned orbit carries `most_probable` = {rho_med, rho_pct, vmag}: the distance it was
    chosen at and the posterior's 5-95% span, which the page states beside it."""
    from pyroorb import constraint
    from pyroorb import rangescan
    rv = constraint._range_verdict(fit)
    if rv is None:
        return most_probable_lov_fit(fit)
    if not rv.get("profiled"):
        return None
    # "No orbit fits this arc at any distance": a mislinked tracklet, typically, where the
    # posterior is a tight answer to a meaningless question (all three PAIR fixtures that
    # read so land 13-70x off, one collapsing to zero width). Show the delivered fit instead.
    if (rv.get("chi2_per_dof") or 0.0) > rangescan.MAX_CHI2_PER_DOF:
        return None
    prof = fit.range_profile()
    post = posterior(prof, tracklet_vmag(fit.included_observations()))
    if not post:
        return None
    st = np.asarray(post["state_med"], float)
    t0 = post.get("t0_tdb")
    ep = float(fit.epoch_tdb)
    if t0 is None:
        return None
    if abs(float(t0) - ep) * 86400.0 > 1e-3:
        return None          # the profile works at the first included observation; a fit
                             # anchored elsewhere is not a single tracklet's (never seen)
    import pyroorb.fit_object as fo
    ctx = fit._ctx
    mask = np.asarray(fit._mask, bool)
    st = _choose_radial_velocity(st, prof, post["vmag"], ctx["ev"], mask, fit.epoch_tdb)
    st = _refit_attributable(st, np.asarray(prof["R0"], float), ctx["ev"], mask,
                             rho=post["rho_med"])
    res = ctx["ev"].evaluate(st, mask)
    mp = fo._build_orbit(ctx, st, mask, res, getattr(fit, "_history", []))
    mp.most_probable = {"rho_med": post["rho_med"], "rho_pct": post["rho_pct"],
                        "vmag": post["vmag"]}
    return mp


def lov_weights(fit, obscodes=None):
    """The population's weight along a fit's line of variations, or None without a LOV or any
    weight: Gaussian(k) x density x rho^4, the Jacobian from sky coordinates to Cartesian, read
    off the LINEAR LOV so it costs no fit. Returns {k, w, rho, R, vmag, k_peak}."""
    from pyroorb import obs_sources
    u, sigma, _refit = fit.lov_refit()
    if u is None:
        return None
    s0 = np.asarray(fit._state, float)[:6]
    k = np.arange(-LOV_K_MAX, LOV_K_MAX, LOV_K_STEP)
    states = s0[None, :] + (k * sigma)[:, None] * np.asarray(u, float)[None, :6]
    R = np.asarray(obs_sources.observer_positions_batch(
        np.array([fit.epoch_tdb]), ["500"], obscodes or obs_sources.load_obscodes()))[0]
    vmag = tracklet_vmag(fit.included_observations())
    rho = np.linalg.norm(states[:, :3] - R[None, :], axis=1)
    w = density(states, R, vmag) * np.exp(-0.5 * k * k) * rho ** 4
    if not w.sum() > 0:
        return None
    return {"k": k, "w": w, "rho": rho, "R": R, "vmag": vmag, "k_peak": float(k[int(np.argmax(w))])}


def lov_peak_state(fit, k_peak):
    """The orbit at LOV offset `k_peak`, re-corrected on the LOV hyperplane, or None."""
    u, sigma, refit = fit.lov_refit()
    if u is None:
        return None
    s0 = np.asarray(fit._state, float)[:6]
    st = np.asarray(refit(s0 + k_peak * sigma * np.asarray(u, float), u, sigma, k_peak), float)[:6]
    return st if np.all(np.isfinite(st)) else None


def most_probable_lov_fit(fit):
    """The orbit to show on a multi-night arc whose delivered orbit the population all but rules
    out: the peak of the population along the line of variations, re-corrected on the LOV
    hyperplane. None -- show the delivered orbit -- anywhere else.

    A short arc of a distant object often fits a parabola: 2008 SZ347 (a 5.20, e 0.14) seeded on
    3 nights delivers a 12134, e 0.9999. The orbit it really has lies 0.06 sigma away, and fits as
    well (rms 0.129" -> 0.132", a 5.27, e 0.12). The LOV is the valley the data leave open, so
    moving along it is the one move the observations do not refuse; the range profile with a
    chosen radial velocity leaves that valley, and fitted a median 2x worse on the same arcs.

    WHEN to replace is the delivered orbit's own weight on the LOV: at most 1% of the peak's
    (LOV_W0_MAX), i.e. the population holds almost nothing where the fit stopped. A near-parabolic
    or unbound orbit has none, so this covers them, and also ordinary-looking fits stranded off
    the population (2014 EJ64, a Trojan seeded on 2 nights: a 6.87 -> 5.79, truth 5.15). Scored on
    suppressed 2-5 night arcs against the full-arc truth, the threshold chosen on 106 and read on
    two holdouts -- 180 new arcs of the same objects, 41 arcs of 43 objects never used: a better
    on 19 / 51 / 8 of 21 / 57 / 9 fired, worse on 0 / 2 / 1, the distance newly 2x off on 0 / 2 / 0.
    Replacing every orbit is a coin flip (a delivered orbit is usually fair: median a error 3%),
    and an earlier rule on e >= 0.9 alone also replaced real long-period orbits the population
    does hold. A pinned re-correction can slip off the bound strip the linear LOV crossed, or stay
    near-parabolic, hence the test after it.

    The returned orbit carries the same `most_probable` as most_probable_fit, plus `k`, the LOV
    offset in sigmas; `rho_med` is the shown orbit's own distance, `rho_pct` the weighted span."""
    import pyroorb.fit_object as fo
    lov = lov_weights(fit)
    if lov is None:
        return None
    k_peak, w, rho, R, vmag = lov["k_peak"], lov["w"], lov["rho"], lov["R"], lov["vmag"]
    if w[int(np.argmin(np.abs(lov["k"])))] > LOV_W0_MAX * w.max():
        return None
    st = lov_peak_state(fit, k_peak)
    if st is None:
        return None
    ctx = fit._ctx
    mask = np.asarray(fit._mask, bool)
    mp = fo._build_orbit(ctx, st, mask, ctx["ev"].evaluate(st, mask), getattr(fit, "_history", []))
    shown = mp.elements()
    if not (shown and shown["a"] > 0 and shown["e"] < LOV_SHOWN_E_MAX
            and mp.rms_residual() <= LOV_RMS_MAX * fit.rms_residual()):
        return None
    order = np.argsort(rho)
    cdf = np.cumsum(w[order]) / w.sum()
    span = {str(p): float(rho[order][np.searchsorted(cdf, p / 100.0)]) for p in (5, 50, 95)}
    mp.most_probable = {"rho_med": float(np.linalg.norm(st[:3] - R)), "rho_pct": span,
                        "vmag": vmag, "k": k_peak}
    return mp


def _choose_radial_velocity(st, prof, vmag, ev, mask, epoch_tdb, n_s=61):
    """The radial velocity to show at the chosen distance -- picked the way the distance was.

    The profile node's own rho-dot is the chi2 optimum at that distance, and on one night chi2
    is nearly FLAT along rho-dot: the node sits wherever its bound barrier left the walk, often
    right at escape -- 2742 au for a main-belt tracklet whose population posterior had put it at
    a sensible distance. So scan rho-dot across the bound interval at the node's sky position
    and angular rates, weight each by the population density x exp(-dchi2/2) on the fit's OWN
    evaluator (so a multi-hour arc that does constrain rho-dot keeps that constraint), and take
    the weighted median. Falls back to the node's state when nothing is weighted."""
    R0 = np.asarray(prof["R0"], float)
    r, v = np.asarray(st[:3], float), np.asarray(st[3:], float)
    rho = float(np.linalg.norm(r - R0))
    if not rho > 0:
        return st
    u = (r - R0) / rho
    b = float(v @ u)
    c = float(v @ v) - 2.0 * MU_SUN / float(np.linalg.norm(r))
    disc = b * b - c
    if not disc > 0:
        return st
    lo, hi = -b - math.sqrt(disc), -b + math.sqrt(disc)
    s = lo + (np.arange(n_s) + 0.5) * (hi - lo) / n_s
    states = np.column_stack([np.repeat(r[None, :], n_s, 0), v[None, :] + s[:, None] * u[None, :]])
    dens = density(states, R0, vmag)
    n_inc = int(np.sum(mask))
    chi2 = np.full(n_s, np.inf)
    for k, x in enumerate(states):
        try:
            rn = float(ev.rms_norm(x, mask))
            chi2[k] = rn * rn * 2 * n_inc
        except Exception:
            pass
    ok = np.isfinite(chi2)
    if not np.any(ok):
        return st
    w = dens * np.where(ok, np.exp(-0.5 * (chi2 - np.min(chi2[ok]))), 0.0)
    if not np.any(w > 0):
        return st
    cdf = np.cumsum(w) / np.sum(w)
    k = int(np.searchsorted(cdf, 0.5))
    return states[min(k, n_s - 1)]


def _refit_attributable(st, R0, ev, mask, rho=None, max_nfev=200):
    """Re-fit what the tracklet MEASURES -- the sky position and its two rates -- holding what the
    prior CHOSE: the distance and the velocity along the line of sight. Without this the shown
    orbit was a state interpolated between two profile nodes with its radial velocity swapped,
    and it fitted its own tracklet a median 2.5x worse than the delivered orbit (31 of 169
    fixture tracklets past 2 sigma, worst 7.3).

    Parameters: offsets of the line of sight in RA / Dec (arcsec) and the two transverse velocity
    components in the local (RA, Dec) frame, all from the given state; the heliocentric velocity
    along the line of sight is held, which over one night is the radial velocity to the
    precision that matters. Least squares on the fit's own evaluator and inclusion mask."""
    from scipy.optimize import least_squares
    st = np.asarray(st, float)
    r, v = st[:3], st[3:]
    rho_st = float(np.linalg.norm(r - R0))
    if not rho_st > 0:
        return st
    u0 = (r - R0) / rho_st
    # the distance the prior chose, exactly: the state comes from interpolating two profile
    # nodes LINEARLY in position, which sits up to ~6% off the log-interpolated median
    rho = float(rho) if rho else rho_st
    ra0, dec0 = math.atan2(u0[1], u0[0]), math.asin(max(-1.0, min(1.0, u0[2])))
    vpar = float(v @ u0)

    def frame(ra, dec):
        u = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
        e_ra = np.array([-math.sin(ra), math.cos(ra), 0.0])
        e_dec = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
        return u, e_ra, e_dec

    _, e_ra0, e_dec0 = frame(ra0, dec0)
    w_ra0, w_dec0 = float(v @ e_ra0), float(v @ e_dec0)
    arcsec = math.pi / (180.0 * 3600.0)

    def build(p):
        ra = ra0 + p[0] * arcsec / max(math.cos(dec0), 1e-6)
        dec = dec0 + p[1] * arcsec
        u, e_ra, e_dec = frame(ra, dec)
        return np.concatenate([R0 + rho * u, vpar * u + (w_ra0 + p[2]) * e_ra + (w_dec0 + p[3]) * e_dec])

    idx = np.flatnonzero(mask)
    sra, sdec = np.asarray(ev.sig_ra, float)[idx], np.asarray(ev.sig_dec, float)[idx]

    def f(p):
        dra, ddec = ev.resid_arcsec(build(p))
        out = np.concatenate([np.asarray(dra)[idx] / sra, np.asarray(ddec)[idx] / sdec])
        return np.where(np.isfinite(out), out, 1e6)

    try:
        f0 = f(np.zeros(4))
        sol = least_squares(f, np.zeros(4), method="trf", x_scale="jac", max_nfev=max_nfev)
    except Exception:
        return st
    return build(sol.x) if float(np.sum(sol.fun ** 2)) <= float(np.sum(f0 ** 2)) else build(np.zeros(4))
