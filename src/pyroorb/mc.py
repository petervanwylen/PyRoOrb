"""Monte Carlo orbit uncertainty.

Logic ported from find_orb (GPL-2.0; logic openly ported, no source copied; see
PROVENANCE.md): `generate_mc_variant_from_covariance` (orb_fun2.cpp) draws each clone
by sampling the state covariance; `add_monte_orbit` / `compute_monte_sigmas` /
`dump_monte_data_to_file` (monte0.cpp) accumulate the element spread into per-element
sigmas and the MPC uncertainty parameter U.

find_orb steps along the covariance eigenvectors scaled by 1/sqrt(eigenvalue), one
unit-normal per axis -- exactly a correlated 6-D Gaussian draw from the state covariance
(np.random.multivariate_normal here). Each clone is converted to osculating elements and
the per-element standard deviation (angles unwrapped at +-180 deg) is the sigma.

Why bother when we already have a linear (Jacobian) covariance: the linear sigmas assume
element(state) is LINEAR across the uncertainty ellipsoid. Monte Carlo does not -- so it
stays honest when sigmas are large (eccentric or short-arc orbits), which is exactly the
regime where the linear approximation breaks (e.g. a single-night arc whose linear sigma
on a is a misleadingly tiny number; see sr.py).

Clones are summarised in the REGIME of the nominal orbit (bound or unbound). A hyperbola's
mean anomaly is not an angle and its aphelion/period do not exist, so a row of each kind cannot
share a column statistic; how much of the family lies on the other side of e = 1 is what
`bound_fraction` reports.
"""
import math
import numpy as np

GAUSS_K = 0.01720209895        # Gaussian gravitational constant (rad/day)
DAYS_PER_YEAR = 365.25
_ANGLES = ("i", "asc_node", "arg_per", "M")
_UNDEFINED_WHEN_UNBOUND = ("Q", "P")


def _names(elem_names):
    if elem_names is None:
        from pyroorb.orbital_elements import ELEM_NAMES
        return ELEM_NAMES
    return elem_names


def _is_unbound(row, elem_names):
    return float(row[elem_names.index("e")]) > 1.0


def _usable(row, elem_names, unbound):
    """A clone row in the requested regime whose DEFINED elements are all finite. A hyperbola's
    Q and P are undefined by construction, not a failed conversion, so they do not disqualify it."""
    if not np.isfinite(row[elem_names.index("e")]) or _is_unbound(row, elem_names) != unbound:
        return False
    skip = _UNDEFINED_WHEN_UNBOUND if unbound else ()
    return all(np.isfinite(v) for nm, v in zip(elem_names, row) if nm not in skip)


def _nominal_unbound(state, epoch_tdb, element_vector_fn, elem_names):
    try:
        return _is_unbound(element_vector_fn(np.asarray(state, float), epoch_tdb), elem_names)
    except Exception:
        return False


def _regime_rows(clones, epoch_tdb, element_vector_fn, elem_names, unbound):
    rows = []
    for cl in clones:
        try:
            v = element_vector_fn(cl, epoch_tdb)
        except Exception:
            continue
        if _usable(v, elem_names, unbound):
            rows.append(v)
    return rows


def monte_carlo_elements(state, state_cov, epoch_tdb, element_vector_fn, elem_names,
                         n=1000, rng_seed=0):
    """Sample the 6-D state covariance and convert each clone to osculating elements.

    Returns (sigmas, U, samples, bound_fraction):
      - sigmas : {name: (median_value, sigma)} for every element in elem_names (NaN for an
            element the regime does not define: Q and P of an unbound orbit)
      - U      : MPC uncertainty parameter from the in-orbit runoff, or None
      - samples: (m, len(elem_names)) array of clone element vectors (for downstream use)
      - bound_fraction : fraction of the drawn STATE clones that are gravitationally bound
            (energy sign; see _bound_fraction). Returned even when sigmas is None -- it is
            valid whenever the clones were drawn, which is exactly the unbound/degenerate
            regime where the element MC gives up but bound-vs-unbound is the question.
    (sigmas, U, samples) are None if too few clones yield finite elements.

    `element_vector_fn`/`elem_names` are passed in (orbital_elements.element_vector, ELEM_NAMES)
    so this module need not import fit_object."""
    clones = _draw_clones(state, state_cov, n, rng_seed)

    # Bound fraction from the SIGN of each clone's specific orbital energy, on the raw
    # STATE clones -- BEFORE the element-conversion filter below. A clone on the other side of
    # e = 1 from the nominal is dropped from `rows`, so a fraction computed over `rows` would
    # silently report ~100% agreement for an orbit whose bound-vs-unbound is wide open. This is
    # the honest number.
    bound_fraction = _bound_fraction(clones)

    unbound = _nominal_unbound(state, epoch_tdb, element_vector_fn, elem_names)
    rows = _regime_rows(clones, epoch_tdb, element_vector_fn, elem_names, unbound)
    if len(rows) < max(10, n // 10):
        return None, None, None, bound_fraction
    rows = np.asarray(rows)

    angle_idx = {i for i, nm in enumerate(elem_names)
                 if nm in _ANGLES and not (unbound and nm == "M")}
    sigmas = {}
    for j, nm in enumerate(elem_names):
        col = rows[:, j]
        if not np.all(np.isfinite(col)):    # undefined in this regime (Q, P of a hyperbola)
            sigmas[nm] = (float("nan"), float("nan"))
            continue
        med = float(np.median(col))
        if j in angle_idx:                  # unwrap about the median before the std
            col = (col - med + 180.0) % 360.0 - 180.0
            sigmas[nm] = (med, float(np.std(col)))
        else:
            sigmas[nm] = (med, float(np.std(col)))
    U = _uncertainty_parameter(rows, elem_names)
    return sigmas, U, rows, bound_fraction


def _draw_clones(state, state_cov, n, rng_seed):
    """`n` state clones from the covariance (find_orb's eigenvector steps == a correlated
    6-D Gaussian draw). A covariance that is not positive-semidefinite falls back to an
    uncorrelated diagonal draw rather than raising -- the degenerate fit is exactly the one
    a caller most needs an answer about."""
    rng = np.random.default_rng(rng_seed)
    state = np.asarray(state, float)
    C = 0.5 * (np.asarray(state_cov, float) + np.asarray(state_cov, float).T)  # symmetrize
    try:
        return rng.multivariate_normal(state, C, size=n)
    except Exception:                       # cov not positive-semidefinite -> diagonal draw
        return state + rng.standard_normal((n, 6)) * np.sqrt(np.clip(np.diag(C), 0, None))


def element_percentiles(rows, elem_names, pct=(5.0, 95.0), min_rows=20):
    """``{name: (lo, hi)}`` -- the central `pct` of the clone element family in `rows`, or None.

    Separate from `monte_carlo_elements` because the two have different standards of proof and
    must: a SIGMA claims the family is characterised, and is rightly refused when most clones
    convert to nothing; an INTERVAL only claims that the solutions it saw span this much, which
    stays true on a family that is 95% on the other side of e = 1 (those clones are not in the
    nominal's regime and simply are not here). That is the regime the interval exists for -- read
    it beside `_bound_fraction`, which says how much of the family it describes. Angles are
    unwrapped about the median first, so a family straddling 0/360 deg reports its width, not
    [1, 359]; an unbound family's mean anomaly is not an angle and is not unwrapped.
    """
    rows = np.asarray(rows, float)
    if rows.ndim != 2 or rows.shape[0] < min_rows:
        return None
    lo_p, hi_p = float(pct[0]), float(pct[1])
    e_col = rows[:, elem_names.index("e")] if "e" in elem_names else np.zeros(len(rows))
    unbound = bool(np.all(e_col > 1.0))
    out = {}
    for j, nm in enumerate(elem_names):
        col = rows[:, j]
        col = col[np.isfinite(col)]
        if col.size < min_rows:
            continue
        if nm in _ANGLES and not (unbound and nm == "M"):
            med = float(np.median(col))
            d = (col - med + 180.0) % 360.0 - 180.0
            lo, hi = med + np.percentile(d, lo_p), med + np.percentile(d, hi_p)
            if hi - lo >= 360.0:              # the angle is free; say so in its own units
                lo, hi = 0.0, 360.0
        else:
            lo, hi = np.percentile(col, lo_p), np.percentile(col, hi_p)
        if np.isfinite(lo) and np.isfinite(hi):
            out[nm] = (float(lo), float(hi))
    return out or None


def element_rows_from_cov(state, state_cov, epoch_tdb, element_vector_fn,
                          n=1000, rng_seed=0, elem_names=None):
    """The usable element vectors of `n` covariance clones (m <= n rows) in the nominal orbit's
    regime, for a caller that wants the family itself rather than `monte_carlo_elements`' verdict
    on it."""
    names = _names(elem_names)
    unbound = _nominal_unbound(state, epoch_tdb, element_vector_fn, names)
    rows = _regime_rows(_draw_clones(state, state_cov, n, rng_seed), epoch_tdb,
                        element_vector_fn, names, unbound)
    return np.asarray(rows, float) if rows else np.empty((0, 0))


def _bound_fraction(states):
    """Fraction of state clones that are gravitationally BOUND, from the sign of the
    specific orbital energy eps = v^2/2 - mu/r (mu = k^2). Computed on the raw heliocentric
    STATE clones so unbound/hyperbolic clones are counted rather than silently dropped at
    element conversion. Returns None if no clone has finite energy."""
    s = np.asarray(states, float)
    mu = GAUSS_K * GAUSS_K
    r = np.linalg.norm(s[:, :3], axis=1)
    v2 = np.sum(s[:, 3:] ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = 0.5 * v2 - mu / r
    finite = np.isfinite(eps)
    return float(np.mean(eps[finite] < 0.0)) if finite.any() else None


def bound_fraction_from_cov(state, state_cov, n=300, rng_seed=0):
    """Draw `n` clones from the state covariance and return the bound fraction (energy
    sign). For the degenerate/unbound regime where monte_carlo_elements declines on the
    element spread but the bound-vs-unbound question is exactly what the caller needs."""
    return _bound_fraction(_draw_clones(state, state_cov, n, rng_seed))


def _uncertainty_parameter(rows, elem_names):
    """MPC uncertainty parameter U from the clone element spread (find_orb
    dump_monte_data_to_file, monte0.cpp): U = ln(runoff)/1.49 + 1, where `runoff` is the
    in-orbit longitude runoff per decade (arcsec) built from sigma(1/a) and sigma(Tp)."""
    try:
        a = float(np.median(rows[:, elem_names.index("a")]))
        e = float(np.median(rows[:, elem_names.index("e")]))
        if a <= 0:
            return None
        inv_a = 1.0 / rows[:, elem_names.index("a")]
        sigma_inv_a = float(np.std(inv_a))
        sigma_tp = float(np.std(rows[:, elem_names.index("Tp")]))    # days
        per_yrs = a * math.sqrt(a)                                   # heliocentric P = a^1.5 yr
        runoff_coeff = 3.0 * 3600.0 * (180.0 / math.pi) * GAUSS_K
        runoff = runoff_coeff * (sigma_tp * e + 15.0 * DAYS_PER_YEAR * sigma_inv_a * a) / per_yrs
        if runoff <= 0 or not math.isfinite(runoff):
            return None
        return math.log(runoff) / 1.49 + 1.0
    except Exception:
        return None
