"""Ensembles of orbits the observations allow, and the population prior on a short arc's distance.

Public entry points (re-exported by ``pyroorb/__init__.py``, documented in docs/api.rst):
``sample_orbits`` -> :class:`OrbitEnsemble`, ``range_posterior``, ``most_probable_orbit``.

One ensemble type for the three ways the engine already characterises a non-linear uncertainty,
so a caller propagates, projects onto the sky, or scans close approaches the same way whichever
produced the members:

* ``"covariance"`` -- draws from the fit's FULL parameter covariance (state plus any free
  non-gravitational coefficients, `FittedOrbit.parameter_covariance`), equal weights. The
  find_orb Monte-Carlo (mc.py), extended to carry each clone's own A1/A2/A3 so a Yarkovsky or
  comet fit's force uncertainty propagates too (helio_states_multi takes per-particle A).
* ``"lov"`` -- the line-of-variations variants (`FittedOrbit.lov_uncertainty`, find_orb's
  improve_along_lov): re-corrected orbits at equal-probability offsets, equal weights. Gravity
  fits only (the LOV over the augmented nongrav space is not built).
* ``"sr"`` -- statistical ranging's accepted family for a short arc (sr.py, find_orb's SR):
  two-body trial orbits through the first and last lines of sight at sampled (range,
  range-rate), kept when they fit the astrometry, equal weights. Its spread is an upper bound
  on the n-body posterior's (sr_uncertainty's caveat).

Not the range-profile nodes: each node keeps its chi2-best radial velocity, which on a tracklet
sits on the bound/unbound edge (measured, 2024 MK's discovery night: 26 of the 27 heaviest nodes
at a < -2e8 au), so they are a profile of the distance, not a family of orbits; the distance
distribution is `range_posterior`.

The population prior is popprior.py's (digest2's Solar System model); every caveat there holds:
it answers "where is the typical object with this astrometry", never "is this object unusual",
so the engine uses it for display only and the library labels everything it touches.
"""
import math

import numpy as np

from pyroorb import products as _p


class OrbitEnsemble:
    """A set of orbits the observations allow, at one epoch, with weights.

    Made by :func:`pyroorb.sample_orbits`.

    Attributes
    ----------
    states : numpy.ndarray
        ``(m, 6)`` heliocentric equatorial states at the ensemble's epoch, au and au/day.
    weights : numpy.ndarray
        ``(m,)``, summing to 1.
    epoch_tdb : float
        The members' epoch, Julian date TDB. (For ``method="sr"`` this is the first included
        observation's time, where statistical ranging anchors its trials, not the fit's epoch.)
    method : str
        ``"covariance"``, ``"lov"`` or ``"sr"``.
    nongrav_A : numpy.ndarray or None
        ``(m, 3)`` the A1, A2, A3 each member is propagated with (au/day^2), or None when the
        members share the fit's non-gravitational parameters (or it has none).
    """

    def __init__(self, fit, states, weights, epoch_tdb, method, nongrav_A=None):
        self.fit = fit
        self.states = np.asarray(states, float).reshape(-1, 6)
        w = np.asarray(weights, float).ravel()
        self.weights = w / w.sum()
        self.epoch_tdb = float(epoch_tdb)
        self.method = method
        self.nongrav_A = None if nongrav_A is None else np.asarray(nongrav_A, float).reshape(-1, 3)

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return (f"OrbitEnsemble({len(self)} orbits, method={self.method!r}, "
                f"epoch_tdb={self.epoch_tdb:.5f})")

    # --- propagation ----------------------------------------------------------------------
    def _nongrav(self, nongrav):
        if not nongrav:
            return None
        pair = _p._nongrav(self.fit, True)
        if pair is None:
            return None
        A, gr = pair
        if self.nongrav_A is not None:
            return self.nongrav_A, gr
        return np.asarray(A, float), gr

    def _raw(self, times_tdb, nongrav=True, idx=None):
        t = np.asarray(times_tdb, float).ravel()
        order = np.argsort(t)
        sel = np.arange(len(self)) if idx is None else np.asarray(idx)
        ng = self._nongrav(nongrav)
        if ng is not None and np.ndim(ng[0]) == 2:
            ng = (ng[0][sel], ng[1])
        out = _p._backend(self.fit).helio_states_multi(
            self.epoch_tdb, list(self.states[sel]), t[order], nongrav=ng,
            asteroids=_p.fit_asteroids(self.fit))
        out = np.asarray(out, float)
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        return out[back]

    def propagate(self, times, *, nongrav=True):
        """Every member's state at other times.

        Parameters
        ----------
        times : str, float, astropy.time.Time, or a sequence of them
            As for :func:`pyroorb.propagate` (strings and numbers UTC).
        nongrav : bool, default True

        Returns
        -------
        numpy.ndarray
            ``(n_times, m, 6)`` heliocentric equatorial, au and au/day.
        """
        return self._raw(_p.tdb(times), nongrav=nongrav)

    # --- elements ---------------------------------------------------------------------------
    def elements(self):
        """Each member's osculating heliocentric elements at the ensemble's epoch.

        Returns
        -------
        dict
            ``{name: (m,) array}`` with the names and units of
            :meth:`FittedOrbit.elements`; NaN where an element does not exist (``Q`` and ``P``
            of an unbound member).
        """
        from pyroorb.orbital_elements import ELEM_NAMES, element_vector
        rows = np.array([element_vector(s, self.epoch_tdb) for s in self.states])
        return {name: rows[:, k] for k, name in enumerate(ELEM_NAMES)}

    def element_percentiles(self, pct=(5.0, 50.0, 95.0)):
        """Weighted percentiles of each element over the members.

        Angles are unwrapped about the weighted median first, so a family straddling 0/360 deg
        reports its real width. Unbound members are left out of ``Q`` and ``P``.

        Returns
        -------
        dict
            ``{name: [value at each pct]}``.
        """
        el = self.elements()
        out = {}
        for name, col in el.items():
            ok = np.isfinite(col)
            if not ok.any():
                out[name] = [float("nan")] * len(pct)
                continue
            x, w = col[ok], self.weights[ok]
            if name in ("asc_node", "arg_per", "M"):
                med = _wpct(x, w, [50.0])[0]
                x = med + (x - med + 180.0) % 360.0 - 180.0
            out[name] = [float(v) for v in _wpct(x, w, pct)]
        return out

    def bound_fraction(self):
        """The weighted fraction of members on bound (elliptic) heliocentric orbits."""
        from pyroorb.forward_model import MU_SUN
        r = np.linalg.norm(self.states[:, :3], axis=1)
        v2 = (self.states[:, 3:] ** 2).sum(axis=1)
        return float(self.weights[(0.5 * v2 - MU_SUN / r) < 0].sum())

    # --- on the sky -------------------------------------------------------------------------
    def sky_positions(self, times, site="500", *, obscodes=None, nongrav=True):
        """Where each member appears on the sky: the non-linear version of the ephemeris
        uncertainty ellipse.

        Astrometric (light-time corrected) RA/Dec, ICRF, as seen from an observatory.

        Parameters
        ----------
        times : str, float, astropy.time.Time, or a sequence of them
            Strings and numbers UTC.
        site : str, default "500"
            MPC observatory code.

        Returns
        -------
        dict
            ``jd_utc`` ``(n,)``; ``ra_deg``, ``dec_deg``, ``delta_au`` ``(n, m)``.
        """
        from pyroorb.obs_sources import load_obscodes, observer_positions_batch
        from pyroorb.forward_model import C_AU_DAY
        t_tdb = _p.tdb(times)
        jd_utc = _p._utc_jd(t_tdb)
        oc = load_obscodes() if obscodes is None else obscodes
        R = np.asarray(observer_positions_batch(jd_utc, [str(site)] * len(jd_utc), oc), float)
        S = self._raw(t_tdb, nongrav=nongrav)                       # (n, m, 6)
        r, v = S[..., :3], S[..., 3:]
        rho = r - R[:, None, :]
        for _ in range(3):                                          # light time, back-stepped
            tau = np.linalg.norm(rho, axis=-1, keepdims=True) / C_AU_DAY
            rho = r - v * tau - R[:, None, :]
        d = np.linalg.norm(rho, axis=-1)
        ra = np.degrees(np.arctan2(rho[..., 1], rho[..., 0])) % 360.0
        dec = np.degrees(np.arcsin(np.clip(rho[..., 2] / d, -1, 1)))
        return {"jd_utc": jd_utc, "ra_deg": ra, "dec_deg": dec, "delta_au": d}

    # --- encounters -------------------------------------------------------------------------
    def close_approaches(self, start, stop, *, body="earth", max_distance=0.05,
                         half_window=2.0, nongrav=True, chunk=200):
        """The spread of each close approach across the members.

        The NOMINAL orbit's approaches (:func:`pyroorb.close_approaches`) name the encounters;
        every member is then followed on one-minute samples within ``half_window`` days of each.

        Parameters
        ----------
        start, stop : str, float or astropy.time.Time
        body : str, default "earth"
        max_distance : float, default 0.05
            au, for the nominal orbit's approaches.
        half_window : float, default 2.0
            Days either side of the nominal approach in which each member's minimum is found.
            A member whose minimum falls on the window's edge is flagged.

        Returns
        -------
        list of dict
            Per encounter: ``nominal`` (the :func:`pyroorb.close_approaches` row), and
            per-member arrays ``distance_au``, ``jd_tdb``, ``at_window_edge``, plus
            ``weights``.
        """
        key = _p._resolve_body(body)
        nominal = _p.close_approaches(self.fit, start, stop, body=key, max_distance=max_distance,
                                      nongrav=nongrav, uncertainty=False)
        out = []
        for enc in nominal:
            fine = np.arange(enc["jd_tdb"] - half_window, enc["jd_tdb"] + half_window + 1e-9,
                             1.0 / 1440.0)
            B = _p.body_states(key, fine)
            dist = np.empty(len(self)); when = np.empty(len(self)); edge = np.zeros(len(self), bool)
            for lo in range(0, len(self), chunk):
                idx = np.arange(lo, min(lo + chunk, len(self)))
                S = self._raw(fine, nongrav=nongrav, idx=idx)       # (n_t, c, 6)
                r = S[..., :3] - B[:, None, :3]
                v = S[..., 3:] - B[:, None, 3:]
                d = np.linalg.norm(r, axis=-1)
                j = d.argmin(axis=0)
                for c, jj in enumerate(j):
                    rj, vj = r[jj, c], v[jj, c]
                    vv = float(vj @ vj)
                    dt = 0.0 if vv == 0 else max(-1 / 1440, min(1 / 1440, -float(rj @ vj) / vv))
                    dist[idx[c]] = float(np.linalg.norm(rj + vj * dt))
                    when[idx[c]] = float(fine[jj] + dt)
                    edge[idx[c]] = jj in (0, len(fine) - 1)
            out.append({"nominal": enc, "distance_au": dist, "jd_tdb": when,
                        "at_window_edge": edge, "weights": self.weights.copy()})
        return out


def _wpct(x, w, pct):
    order = np.argsort(x)
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    c = np.cumsum(w) - 0.5 * w
    c = c / w.sum()
    return np.interp(np.asarray(pct, float) / 100.0, c, x)


# --- sampling -------------------------------------------------------------------------------
def sample_orbits(fit, n=500, method="covariance", *, rng_seed=0, n_sigma=3.0):
    """See pyroorb.sample_orbits."""
    method = str(method).lower()
    if method == "covariance":
        return _covariance(fit, n, rng_seed)
    if method == "lov":
        lov = fit.lov_uncertainty(n_steps=int(n), n_sigma=float(n_sigma))
        if not lov or lov.get("variants") is None or not len(lov["variants"]):
            raise ValueError("no line-of-variations ensemble for this fit (a non-gravitational "
                             "fit, a degenerate covariance, or too few variants converged)")
        st = np.asarray(lov["variants"], float)
        return OrbitEnsemble(fit, st, np.ones(len(st)), fit.epoch_tdb, "lov")
    if method == "sr":
        return _sr(fit, n)
    raise ValueError(f"method must be 'covariance', 'lov' or 'sr', not {method!r}")


def _covariance(fit, n, rng_seed):
    cov, names = fit.parameter_covariance()
    s = np.asarray(fit.state(), float)[:6]
    A0 = np.zeros(3)
    for k, ax in enumerate(("A1", "A2", "A3")):
        A0[k] = float((fit.nongrav_params or {}).get(ax, 0.0))
    mean = np.concatenate([s, [A0[("A1", "A2", "A3").index(nm)] for nm in names[6:]]])
    rng = np.random.default_rng(rng_seed)
    draws = rng.multivariate_normal(mean, 0.5 * (cov + cov.T), size=int(n), check_valid="ignore")
    A = None
    if _p._nongrav(fit, True) is not None:
        A = np.tile(A0, (len(draws), 1))
        for j, nm in enumerate(names[6:]):
            A[:, ("A1", "A2", "A3").index(nm)] = draws[:, 6 + j]
    return OrbitEnsemble(fit, draws[:, :6], np.ones(len(draws)), fit.epoch_tdb, "covariance",
                         nongrav_A=A)


def _sr(fit, n):
    """Statistical ranging's accepted family, built the way FittedOrbit.sr_uncertainty builds
    its inputs (the INCLUDED observations, the fit's own weights), but keeping the states."""
    from astropy.time import Time
    from pyroorb import obs_sources as _os
    from pyroorb import sr as _srmod
    from pyroorb.forward_model import build_base_sim, los_unit
    ctx = getattr(fit, "_ctx", None) or {}
    mask = np.asarray(fit._mask, bool)
    sub = [o for o, keep in zip(fit._obs, mask) if keep]
    if len(sub) < 3:
        raise ValueError("statistical ranging needs at least three observations")
    times = np.array([o["jd_utc"] for o in sub])
    rhohats = np.array([los_unit(o["ra_deg"], o["dec_deg"]) for o in sub])
    Robs = np.array(list(_os.observer_positions_batch(
        list(times), [o["obscode"] for o in sub], _os.load_obscodes())))
    t0 = float(Time(times[0], format="jd", scale="utc").tdb.jd)
    sra, sdec = ctx.get("sigma_ra"), ctx.get("sigma_dec")
    sigmas = None
    if sra is not None and sdec is not None and len(sra) == len(mask) == len(sdec):
        sigmas = np.column_stack([np.asarray(sra, float)[mask], np.asarray(sdec, float)[mask]])
    med, kept, k = _srmod.statistical_ranging(sub, times, rhohats, Robs, build_base_sim(t0), t0,
                                              max_orbits=int(n), sigmas=sigmas)
    if med is None or kept is None or len(kept) < _srmod.SR_MIN_ACCEPT:
        span = 24.0 * (times.max() - times.min())
        raise ValueError(f"statistical ranging kept {0 if kept is None else len(kept)} of {int(n)} "
                         f"trial orbits on this {span:.1f} h arc, too few for a family (it needs "
                         f"{_srmod.SR_MIN_ACCEPT}). SR is a single-night tool: on two or more nights "
                         f"use method='lov' or 'covariance'; on one night, more trials (n) may "
                         f"help, and an arc no orbit fits (a mislinkage) has no family at all")
    st = np.asarray(kept, float)
    return OrbitEnsemble(fit, st, np.ones(len(st)), t0, "sr")


# --- the range posterior and the population prior -----------------------------------------
def range_posterior(fit, prior="population"):
    """See pyroorb.range_posterior."""
    if prior not in (None, "population"):
        raise ValueError("prior must be None or 'population'")
    from pyroorb import popprior
    prof = fit.range_profile()
    rows = sorted((r for r in (prof or {}).get("rows") or [] if r.get("state") is not None),
                  key=lambda r: r["rho"])
    if len(rows) < 2:
        why = (prof or {}).get("failure") or "the range profile has no rows"
        raise ValueError(f"no range profile for this fit: {why}")
    rho = np.array([r["rho"] for r in rows], float)
    chi2 = np.array([r["chi2"] for r in rows], float)
    like = np.exp(-0.5 * (chi2 - np.nanmin(chi2)))
    vmag = popprior.tracklet_vmag(fit.included_observations())
    pri = None
    if prior == "population":
        post = popprior.posterior(prof, vmag)
        if post is None:
            raise ValueError("the population prior cannot be formed for this arc")
        pri = np.array(post["prior"], float)
    w = like * (pri if pri is not None else 1.0)
    # node mass: trapezoid in rho (the nodes are log-spaced plus zoomed passes)
    edges = np.concatenate([[rho[0]], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]])
    mass = w * np.diff(edges)
    tot = mass.sum()
    if not tot > 0:
        raise ValueError("the range posterior has no mass")
    mass = mass / tot
    cdf = np.cumsum(mass)

    def at(p):
        return float(np.interp(p, cdf - 0.5 * mass, rho))

    return {"rho_au": rho, "chi2": chi2, "likelihood": like,
            "prior": pri, "weight": mass, "rho_percentiles": {5: at(0.05), 50: at(0.5),
                                                               95: at(0.95)},
            "rho_nominal_au": fit.geocentric_range(), "vmag": vmag,
            "epoch_tdb": float(prof.get("t0_tdb")), "prior_name": prior,
            "bound": np.array([bool(r["bound"]) for r in rows])}


def most_probable_orbit(fit):
    """See pyroorb.most_probable_orbit."""
    from pyroorb import popprior
    return popprior.most_probable_fit(fit)
