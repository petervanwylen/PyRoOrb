"""The station-night error model: what residuals inside one night share, what a station's nights
share with each other, written down so a Delta chi2 between two orbits can be given its null.

THE MODEL. For observation i of station-night b (one station, consecutive gaps <=
`orbit_common.NIGHT_GAP_DAYS`), per axis, residual in arcsec:

    r_bi  =  delta_b  +  rho_b (t_bi - tbar_b)  +  eta_bi

    eta_bi ~ N(0, (k sigma_i)^2)    within-night scatter, k x the ASSIGNED per-obs sigma
    delta_b ~ N(0, tau^2)            the night's shared offset (catalogue, clock, centroiding)
    rho_b   ~ N(0, tau_rate^2)       the night's shared rate error
    Cov(delta_b, delta_b') = tau^2 rho0 exp(-|t_b - t_b'| / L)   for two nights of ONE station

so a station's observations of one object form one block whose covariance is
`S = k^2 D + tau^2 R + tau_rate^2 T`, D the assigned variances, R the night-offset correlation
(1 within a night, the decaying kernel across nights), T the per-night centred-time outer
products; a residual vector's chi2 under the model is r^T S^-1 r. (k, tau, tau_rate) per station
and (rho0, L) pooled are MEASURED on the full-arc residuals of the regression fixtures
(tabulated in `nightmodel_table.py`); a station not in the
table takes the pooled row.

WHY IT EXISTS. The assigned Veres/ADES sigma is a population rms that folds the night-to-night
systematics in, while a fit's residuals are within-night scatter (k ~ 0.45-0.8 of the sigma)
around offsets the six orbital parameters absorb when the arc is two or three nights. Under
independent-error weighting every orbit but the one that interpolates those offsets -- the
TRUE one included -- then pays the offsets times the number of observations each night
happened to hold, and a Delta chi2 between two orbits has no distribution anyone can name
(the independent-error rule read 2010 LP68's own orbit as refused at 9 sigma on two nights). Under S the
offsets cost what they are worth, tau, once per night; the within-night rates cost what the
scatter says; and the Delta chi2 of a one-parameter restriction between two S-optimal fits is
a chi2(1) variate, so "3 sigma" means 3 sigma on two nights as on two hundred. The
covariance floor (`FittedOrbit._param_inverse`) is the cheap stand-in for the same
model on the covariance side; a Delta chi2 cannot be floored, so here the model itself is used.

THE CROSS-NIGHT TERM. A station's night offsets for one object are not independent: the
night means correlate 0.50 (RA*) / 0.38 (Dec) at lags of 0-3 d, 0.27 at 10-30 d, 0.17 at
30-120 d and ~0 past a year, while DIFFERENT stations' night means correlate 0.00-0.03 at every
lag -- so this is the station's persistent systematic for that object (its catalogue's zonal
error over the field it crossed, its centroiding of that brightness and rate), not orbit or
force-model error. Fitted as rho0 exp(-dt / L) on the pooled bins (`NIGHT_CORR`); I41 (ZTF)
is the outlier, 0.6-0.8 within a month. A block is therefore a STATION-RUN, all of one
station's observations in the arc with no gap over RUN_BREAK_FACTOR x L (past which the kernel
is < 5% and a new block loses nothing), so a long arc still costs small eigendecompositions.

WHAT A NIGHT IS WORTH. The effective sample size of a correlated block is 1^T R^-1 1, R its
correlation matrix (for one equicorrelated night this is Kish's n_b / (1 + (n_b - 1) rho),
rho = tau^2 / (tau^2 + k^2 sigma^2)): the number of independent position measurements the
block's observations amount to per axis -- what the retired "4 measurements per station-night"
asserted. Measured, it is 1.7 per axis at F51 and 703/704
(tau ~ sigma_w) and 3.3 at G96/T08/T05 (tau << sigma_w): the same four observations are worth
3.4 or 6.6 depending on the station, and two nights of one station a week apart are worth
less than two independent ones. `n_eff` sums it over blocks and axes and is the N of the
refusal rule's BIC.

WHITENING. `NightWhitener` builds S^-1/2 per block and axis (eigendecomposition; blocks are
small) and applies it to any residual vector or Jacobian laid out like `Evaluator.evaluate`'s
-- interleaved [RA*, Dec] per observation, in units of the ASSIGNED sigma -- so the ordinary
LM machinery minimises r^T S^-1 r with no other change. The Veres over-observation factors
(`weights.overobserving_factors`, the fit's `fit_sig_*`) are a crude stand-in for the same
block correlation and are NOT applied on top of it. `draw` samples a residual vector from the
model, for the null studies and the tests.

PER-FIT SCALE. Objects differ from their station's k: the between-object variance of the
within-night variance ratio, its own sampling variance removed, is 0.4 (object-weighted over
the fixtures' stations), so a station's k is worth SCALE_PRIOR_DOF = 2 / 0.4 ~ 5 within-night
dof of an object's own -- measured, not chosen. The scatter about each night's OWN line is
orbit-independent (over a tracklet the motion is a line) and estimates the object's k from the
arc itself, on sum(n_b - 2) dof per axis; `within_scale` shrinks it toward the table with that
prior and hands back the posterior's dof, because a statistic scaled by an ESTIMATED variance
is an F(1, dof) variate and the rules must know it (hyperbolic.block_damaged). Symmetric: a
cleaner-than-population object earns its tighter scale.
"""
import math

import numpy as np

from pyroorb import orbit_common
from pyroorb.nightmodel_table import STATIONS, POOLED, SCALE_PRIOR_DOF, NIGHT_CORR

GAP_DAYS = orbit_common.NIGHT_GAP_DAYS
SCALE_MIN_DOF = 2            # below this the arc says nothing about its own scatter
MIN_SLOPE_OBS = 3            # a night needs this many obs for a line to leave scatter over
RUN_BREAK_FACTOR = 3.0       # a station's nights further apart than this x L are separate blocks


def station_night_index(jds, codes, gap_days=GAP_DAYS):
    """Block id per observation: one station, consecutive gaps <= `gap_days`."""
    jds = np.asarray(jds, float)
    order = np.argsort(jds, kind="stable")
    last, gid, ng = {}, np.empty(len(jds), int), 0
    for i in order:
        c = codes[i]
        if c in last and jds[i] - last[c][0] <= gap_days:
            gid[i] = last[c][1]
        else:
            gid[i] = ng
            ng += 1
        last[c] = (jds[i], gid[i])
    return gid


def station_run_index(jds, codes, break_days):
    """Block id per observation: one station, consecutive gaps <= `break_days` (a station-run)."""
    return station_night_index(jds, codes, gap_days=break_days)


def station_params(code):
    """``(k_ra, k_dec, tau_ra, tau_dec, tau_rate_ra, tau_rate_dec)`` for a station code."""
    row = STATIONS.get(str(code).strip(), POOLED)
    return tuple(float(v) for v in row[:6])


def night_kernel(dt_days, rho0=NIGHT_CORR[0], L=NIGHT_CORR[1]):
    """Correlation of one station's night offsets `dt_days` apart (1 at dt = 0)."""
    dt = np.abs(np.asarray(dt_days, float))
    return np.where(dt <= 0.0, 1.0, rho0 * np.exp(-dt / L))


def kish_neff(n_b, rho):
    """Independent measurements an equicorrelated block of `n_b` is worth (Kish 1965)."""
    return n_b / (1.0 + (n_b - 1) * rho)


def block_covariance(jds, sig, p, k_scale=1.0, gap_days=GAP_DAYS, corr=NIGHT_CORR, rate=True):
    """S for one station-run block and axis, in units of the ASSIGNED sigma: `jds` the block's
    epochs, `sig` its assigned sigmas, `p` = (k, tau, tau_rate) for the axis. `rate=False` leaves
    the rate term out (the POSITION covariance, which is what a night's worth is counted on)."""
    jds = np.asarray(jds, float); sig = np.asarray(sig, float)
    k, tau, tau_r = p[0] * k_scale, p[1], p[2]
    n = len(jds)
    nid = station_night_index(jds, ["x"] * n, gap_days)
    mids = np.array([jds[nid == g].mean() for g in range(nid.max() + 1)])
    R = night_kernel(mids[nid][:, None] - mids[nid][None, :], corr[0], corr[1])
    same = nid[:, None] == nid[None, :]
    R = np.where(same, 1.0, R)
    outer = 1.0 / np.outer(sig, sig)
    S = (k * k) * np.eye(n) + (tau * tau) * R * outer
    if rate:
        t = jds - mids[nid]                              # centred within each night
        S = S + (tau_r * tau_r) * np.where(same, np.outer(t, t), 0.0) * outer
    return S


class NightWhitener:
    """S^-1/2 for one set of observations, in the interleaved [RA*, Dec]-per-observation layout
    of the LM residual vector, all quantities in units of the ASSIGNED sigma.

    `jds`, `codes`, `sig_ra`, `sig_dec` are the INCLUDED observations in residual order.
    `k_scale` multiplies the table's k for every block (see :func:`within_scale`)."""

    def __init__(self, jds, codes, sig_ra, sig_dec, k_scale=1.0, gap_days=GAP_DAYS,
                 corr=NIGHT_CORR):
        jds = np.asarray(jds, float)
        sig = (np.asarray(sig_ra, float), np.asarray(sig_dec, float))
        self.n_obs = len(jds)
        self.k_scale = float(k_scale)
        rho0, L = corr
        rid = station_run_index(jds, codes, RUN_BREAK_FACTOR * L) if self.n_obs else np.zeros(0, int)
        self.n_blocks = int(rid.max()) + 1 if self.n_obs else 0
        self.n_nights = int(station_night_index(jds, codes, gap_days).max()) + 1 if self.n_obs else 0
        self.blocks = []                 # (rows into the 2m vector, W = S^-1/2, S^1/2) per block and axis
        self.n_eff = 0.0
        for g in range(self.n_blocks):
            sel = np.flatnonzero(rid == g)
            p = station_params(codes[sel[0]])
            for ax in range(2):
                s = sig[ax][sel]
                pax = (p[ax], p[2 + ax], p[4 + ax])
                S = block_covariance(jds[sel], s, pax, self.k_scale, gap_days, corr)
                w, V = np.linalg.eigh(S)
                w = np.maximum(w, 1e-30)
                W = (V * (1.0 / np.sqrt(w))) @ V.T
                self.blocks.append((2 * sel + ax, W, (V * np.sqrt(w)) @ V.T))
                # the block's worth in independent POSITIONS: 1^T R^-1 1 with R the correlation
                # matrix of the position covariance (scatter + offsets, no rate term -- a rate is
                # information about motion, not another position, and counting its diagonal would
                # read a long tracklet as more positions than it has observations)
                S0 = block_covariance(jds[sel], s, pax, self.k_scale, gap_days, corr, rate=False)
                d = np.sqrt(np.diag(S0))
                self.n_eff += float(d @ np.linalg.solve(S0, d))

    def apply(self, v):
        """Whiten a (2m,) residual vector or a (2m, p) Jacobian (rows interleaved RA*, Dec)."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        for rows, W, _ in self.blocks:
            out[rows] = W @ v[rows]
        return out

    def chi2(self, resid):
        """r^T S^-1 r for a residual vector in assigned-sigma units."""
        z = self.apply(resid)
        return float(np.dot(z, z))

    def draw(self, rng):
        """One residual vector (2m, assigned-sigma units) from the model: S^1/2 z, z ~ N(0, I)."""
        out = np.empty(2 * self.n_obs)
        for rows, _, S_half in self.blocks:
            out[rows] = S_half @ rng.standard_normal(len(rows))
        return out


def within_scale(jds, codes, resid_ra, resid_dec, sig_ra, sig_dec, gap_days=GAP_DAYS,
                 prior_dof=SCALE_PRIOR_DOF):
    """``(k_scale, dof, ratio, scale_dof)``: how the arc's own within-night scatter compares with
    the table's k, as a multiplier on k, shrunk toward 1 with a prior worth `prior_dof`.

    Per block (>= MIN_SLOPE_OBS obs) and axis the residuals (arcsec) are reduced by their own
    least-squares line -- offset and rate out, so the model's delta and rho cannot leak into the
    scatter -- and what remains, in units of (k sigma)^2, is chi2 on n_b - 2 degrees of freedom
    if the table's k is right for this object. `ratio` is the pooled chi2/dof (1.0 = the table),
    `k_scale` = sqrt((dof x ratio + prior_dof) / (dof + prior_dof)), and `scale_dof` = dof +
    prior_dof is what the resulting variance is known on -- the dof of the F a statistic scaled by
    it follows. (1.0, dof, None, None) when no block leaves any: the table's k stands, known
    exactly as far as this arc can say."""
    jds = np.asarray(jds, float)
    resid = (np.asarray(resid_ra, float), np.asarray(resid_dec, float))
    sig = (np.asarray(sig_ra, float), np.asarray(sig_dec, float))
    gid = station_night_index(jds, codes, gap_days)
    ss, dof = 0.0, 0
    for g in range(int(gid.max()) + 1 if len(jds) else 0):
        sel = np.flatnonzero(gid == g)
        if len(sel) < MIN_SLOPE_OBS:
            continue
        p = station_params(codes[sel[0]])
        t = jds[sel] - jds[sel].mean()
        stt = float(np.dot(t, t))
        for ax in range(2):
            r = resid[ax][sel] / (p[ax] * sig[ax][sel])       # units of k sigma
            r = r - r.mean()
            if stt > 0:
                r = r - t * (float(np.dot(t, r)) / stt)
            ss += float(np.dot(r, r))
            dof += len(sel) - 2
    if dof < SCALE_MIN_DOF or not ss > 0:
        return 1.0, dof, None, None
    ratio = ss / dof
    return math.sqrt((dof * ratio + prior_dof) / (dof + prior_dof)), dof, ratio, dof + prior_dof
