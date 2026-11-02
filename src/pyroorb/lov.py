"""Line-of-Variations (LOV) orbit uncertainty.

Logic ported from find_orb (GPL-2.0; logic openly ported, no source copied):
`improve_along_lov` (orb_fun2.cpp:694) steps the orbit along the
least-constrained direction of the covariance (the LOV) and re-corrects, generating
statistical clones that span the dominant uncertainty -- the input find_orb propagates
into close-approach tables and impact probability.

Complement to `mc.py` (covariance-sampling Monte Carlo): MC draws a full 6-D Gaussian
cloud; the LOV is the 1-D *spine* of that cloud -- the single most-uncertain direction.
For a short-arc / multi-revolution orbit nearly all the uncertainty lives along it, and
crucially the LOV is NONLINEAR: each variant is re-fit at its LOV offset (the other 5
d.o.f. re-minimized), so it follows the curved valley of best-fit orbits rather than the
straight covariance axis the linear sigmas assume. That curvature is exactly what a
linear (Jacobian) covariance -- and even a moderate MC -- can miss far from the nominal.

Design:
  - LOV direction = the largest-variance eigenvector of the STATE covariance (== the
    smallest-eigenvalue eigenvector of the normal matrix J^T J), 1-sigma extent = sqrt of
    that variance. Same direction `arc_extend._bridge_far_by_lov` slides for *search*;
    here it characterizes *uncertainty*.
  - Step in equal-probability increments along the LOV (find_orb's inverf spacing) out to
    +-`n_sigma`, re-correcting each via `refit_fn(seed, u, sigma, k)` -> variant state.
  - Convert each variant to osculating elements; report the per-element spread + the MPC
    uncertainty parameter U, the same shape `mc.py` returns so callers are symmetric.

This module is engine-agnostic: `refit_fn`, `element_vector_fn`, `elem_names` are injected
(by `fit_object.FittedOrbit.lov_uncertainty`) so it need not import fit_object/arc_extend.
"""
import math
import numpy as np

from pyroorb import mc   # reuse the MPC uncertainty-parameter U formula (find_orb dump_monte_data_to_file)


def lov_direction(state_cov):
    """Unit LOV direction `u` (6-vector) and its 1-sigma extent in state units, from the
    state covariance: the eigenvector of the LARGEST eigenvalue (the least-constrained
    direction), sigma = sqrt(eigenvalue). Returns (u, sigma) or (None, None) if degenerate."""
    C = 0.5 * (np.asarray(state_cov, float) + np.asarray(state_cov, float).T)
    try:
        evals, evecs = np.linalg.eigh(C)          # ascending
    except np.linalg.LinAlgError:
        return None, None
    lam, u = evals[-1], evecs[:, -1]              # largest variance == the LOV
    if not (np.isfinite(lam) and lam > 0):
        return None, None
    return u, math.sqrt(lam)


def _equal_prob_sigmas(n_steps, n_sigma):
    """`n_steps` LOV offsets in SIGMA units, equal-probability-spaced over +-n_sigma
    (find_orb improve_along_lov uses inverf for this -- each step carries equal probability
    mass, so the variants sample the Gaussian along the LOV rather than uniformly in sigma)."""
    from scipy.special import erfinv
    qs = (np.arange(n_steps) + 0.5) / n_steps            # (0,1) midpoints
    s = math.sqrt(2.0) * erfinv(2.0 * qs - 1.0)          # standard-normal quantiles
    s = s / s.max() * n_sigma                            # rescale the tails to exactly +-n_sigma
    return s


def lov_variants(state, state_cov, epoch_tdb, refit_fn, element_vector_fn, elem_names,
                 n_steps=21, n_sigma=3.0):
    """Generate nonlinear LOV variant orbits and their element spread.

    Returns (sigmas, U, info) where
      - sigmas : {name: (median_value, sigma)} per element in elem_names
      - U      : MPC uncertainty parameter from the variant element runoff (mc._uncertainty_parameter)
      - info   : {"u": LOV unit dir, "sigma": 1-sigma state extent, "k": sigma-offsets array,
                  "states": (m,6) re-corrected variant states, "elements": (m,len) element rows}
    or (None, None, None) if the LOV is degenerate or too few variants yield finite elements.

    `refit_fn(seed_state, u, sigma, k) -> variant_state` re-corrects the orbit pinned at the
    LOV coordinate k*sigma (see arc_extend.fit_lov_pinned)."""
    state = np.asarray(state, float)
    u, sigma = lov_direction(state_cov)
    if u is None:
        return None, None, None
    ks = _equal_prob_sigmas(n_steps, n_sigma)
    states, rows, kept_k = [], [], []
    for k in ks:
        seed = state + (k * sigma) * u                   # linear seed at the LOV offset...
        v = np.asarray(refit_fn(seed, u, sigma, k), float)   # ...then re-correct on the LOV hyperplane
        if not np.all(np.isfinite(v)):
            continue
        try:
            ev = element_vector_fn(v, epoch_tdb)
        except Exception:
            continue
        if np.all(np.isfinite(ev)):
            states.append(v); rows.append(ev); kept_k.append(k)
    if len(rows) < max(5, n_steps // 3):
        return None, None, None
    rows = np.asarray(rows); states = np.asarray(states)
    kept_k = np.asarray(kept_k)
    # The variants span +-n_sigma along the LOV, so their raw spread is NOT the 1-sigma
    # element uncertainty. `k` is in sigma units, so element-change PER SIGMA = std(elem)/std(k)
    # is the sampling-independent 1-sigma estimate (the LOV's contribution to that element's
    # uncertainty -- it matches the linear/MC sigma when, as for short-arc/multi-rev orbits,
    # the LOV dominates). sk also rescales the U-runoff inputs to true sigmas.
    sk = float(np.std(kept_k)) or 1.0

    angle_idx = {i for i, nm in enumerate(elem_names)
                 if nm in ("i", "asc_node", "arg_per", "M")}
    sigmas, pseudo = {}, rows.copy()
    for j, nm in enumerate(elem_names):
        col = rows[:, j]; med = float(np.median(col))
        if j in angle_idx:
            col = (col - med + 180.0) % 360.0 - 180.0
        sigmas[nm] = (med, float(np.std(col)) / sk)
        pseudo[:, j] = med + (col - med) / sk          # rescaled to true 1-sigma spread for U
    U = mc._uncertainty_parameter(pseudo, elem_names)
    info = {"u": u, "sigma": sigma, "k": kept_k, "states": states, "elements": rows}
    return sigmas, U, info
