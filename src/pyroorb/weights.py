"""Per-observation astrometric uncertainties (sigmas) → normalized residuals.

find_orb (and JPL, digest2, OpenOrb) does NOT minimize raw angular residuals; it
minimizes residuals divided by each observation's 1-sigma uncertainty. Two sources:

  1. **ADES** astrometry carries `rmsRA`/`rmsDec` (arcsec) directly — the modern,
     preferred path. (ADES `rmsRA` is already multiplied by cos(Dec), matching the
     cross-track residual we form, so it pairs with dRA*cosδ.)
  2. **Older astrometry** has no per-obs uncertainty, so we fall back to a
     station-based lookup approximating **Veres et al. (2017)**, "Statistical
     analysis of astrometric errors for the most productive asteroid surveys",
     Icarus 296, 298 — the same table find_orb ships in `weight.txt`.

⚠️ This `VERES_2017` map is a DOCUMENTED SUBSET of that table (the high-volume
stations), with a 1.0" catch-all default. It is intentionally easy to extend; the
full per-program / per-epoch scheme is not yet implemented. ADES *parsing* is native
(ades.py); no maintained PyPI package provides these weights, so the table is
implemented here and cited.
"""

import math
import numpy as np

DEFAULT_SIGMA = 1.0      # arcsec — Veres 2017 catch-all for unlisted stations

# station (MPC obs code) -> 1-sigma astrometric uncertainty in arcsec.
# Subset of Veres et al. (2017) Table; values are the commonly-reproduced ones
# (also embedded in find_orb/digest2). RA and Dec share the station sigma unless
# ADES rms values override below.
VERES_2017 = {
    "703": 1.0,  "691": 0.6,  "644": 0.6,   # Catalina (old), Spacewatch, NEAT/Palomar
    "704": 1.0,                              # LINEAR
    "G96": 0.5,  "703s": 1.0, "I52": 0.5,    # Mt Lemmon / Catalina
    "F51": 0.2,  "F52": 0.2,                 # Pan-STARRS 1 & 2
    "G45": 0.6,  "699": 0.8,  "D29": 0.75,   # LONEOS, Lowell, Purple Mtn
    "E12": 0.75, "608": 0.6,  "J75": 1.0,
    "645": 0.7,  "673": 0.3,  "689": 0.5,    # Table Mtn, USNO Flagstaff
    "950": 0.5,  "H01": 0.3,  "J04": 0.4,
    "W84": 0.5,  "568": 0.25,                # CTIO/DECam, Mauna Kea (occultation-grade)
    "T05": 0.5,  "T08": 0.5,                 # ATLAS Haleakala / Mauna Loa
    "M22": 0.6,  "247": 1.0,  "C51": 1.0,    # roving, WISE (space)
    "C57": 1.0,  "094": 0.5,  "G83": 0.5,
    "V00": 0.5,  "W68": 0.5,                 # Bok/Kitt Peak, generic survey CCD
    "X05": 0.1,                              # Vera C. Rubin Obs (LSST) -- ~half F51/F52
}


# Floor on per-obs sigma (arcsec). Reported ADES rmsRA/rmsDec are often
# optimistically small (<0.05"), but star-catalog + centroiding systematics floor
# real single-observation accuracy around ~0.1", so weighting any obs tighter than
# this is unjustified -- and over-tight sigmas make good obs look like outliers.
# (JPL/find_orb apply a comparable minimum.)
SIGMA_FLOOR = 0.1

# Ceiling on per-obs sigma (arcsec). The flip side of the floor: a reported ADES
# rmsRA/rmsDec above this (some discovery tracklets self-report 72") is a
# data-quality FLAG, not a credible measurement precision -- left uncapped, such an
# obs can sit 70" off the orbit yet read as "<1sigma", so outlier rejection keeps it
# and it dominates the raw-arcsec RMS (find_orb instead weights it by its own station
# model, ~1", and rejects it). Capping the trusted sigma makes a gross error show up
# as the outlier it is. e.g. C/2025 R2: 3.9" -> 0.53" (matching find_orb), with no
# effect on the regression suite (its largest reported sigma is 5.0").
SIGMA_CEILING = 5.0


def _reported_sigma(x):
    """One reported ADES rms as a usable 1-sigma, or None.

    A value must be a real number AND strictly positive to be usable: a reported rms of
    0 (or a negative one) is not an infinitely precise measurement, it is a submitter
    that left the field at its default, and taking it literally would divide the
    residual by zero. Such an obs falls back to the station model like an obs with no
    rms at all."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def ades_rms_usable(rms_ra, rms_dec):
    """True when a pair of reported ADES rmsRA/rmsDec actually drives this obs's sigma.

    THE point of this predicate is that it is shared: `sigma_for` decides the sigma with
    it, and `ades.records_to_obs` labels the obs's `uncert_src` with it, so the label can
    never claim "ADES" while the value came from Veres-2017. They used to test the same
    thing two different ways -- and the label's way (a bare truthiness test) disagreed
    with the sigma's on a reported rms of exactly 0."""
    return _reported_sigma(rms_ra) is not None and _reported_sigma(rms_dec) is not None


# find_orb sigma.txt's catch-all rule, ported: an observation that PREDATES 1993-01-01 gets a
# default sigma of 3" no matter which observatory it came from ("older observations are usually
# photographic, and reduced with older catalogues such as the SAO; the assumption that they're
# just a little bit dodgy is usually right"). Found on 1862 Apollo: its 1932 plates
# scatter 10-35" against every modern solution, the Veres CCD-era lookup handed them 1.0-1.5",
# and the 18 the rejection loop kept out-voted a 96-year arc -- A2 delivered 6x below JPL's
# radar-anchored value WITH all 17 radar rows in the fit, because radar pins the orbit at its
# epochs while the mis-weighted 1932 block bends the secular drift between them. A FLOOR on the
# DEFAULT path only: a reported per-obs ADES rms still wins, since a remeasured old plate is
# exactly the observation that earned a tighter sigma.
PRE_CCD_JD = 2448988.5       # 1993-01-01, find_orb sigma.txt's date line
PRE_CCD_SIGMA = 3.0          # arcsec


def sigma_for(stn, astcat=None, rms_ra=None, rms_dec=None, jd_utc=None):
    """Return (sigma_RA, sigma_Dec) in arcsec for one observation, clamped to
    [SIGMA_FLOOR, SIGMA_CEILING]. ADES per-obs rms values win when present and
    positive (see _reported_sigma); otherwise the Veres-2017 station lookup (equal
    RA/Dec) with a 1.0" default, floored at PRE_CCD_SIGMA for an observation whose
    `jd_utc` predates 1993 (the find_orb sigma.txt rule -- see above)."""
    def _clamp(v):
        return min(max(v, SIGMA_FLOOR), SIGMA_CEILING)
    sra, sdec = _reported_sigma(rms_ra), _reported_sigma(rms_dec)
    if sra is not None and sdec is not None:
        return _clamp(sra), _clamp(sdec)
    s = VERES_2017.get(stn, DEFAULT_SIGMA)
    if jd_utc is not None and float(jd_utc) < PRE_CCD_JD:
        s = max(s, PRE_CCD_SIGMA)
    s = _clamp(s)
    return s, s


# --- timing-error smear (along-track), find_orb errors.cpp -----------------------
# find_orb folds each obs's TIME uncertainty into its POSITION uncertainty: a
# Gaussian timing error σ_t "smears" the measured position along the object's
# direction of motion by (sky rate)·σ_t, combined IN QUADRATURE with the
# astrometric σ (see the worked example in find_orb/errors.cpp:32-46). Formally the
# timing error adds a rank-1 covariance σ_t²·v·vᵀ (v = sky-velocity vector) to the
# position covariance; find_orb keeps the full rotated ellipse. Our least-squares
# weights each axis independently, so we apply the DIAGONAL projection of that term
# onto the RA/Dec axes (dropping the small off-diagonal cross term):
#     σ_RA_eff  = hypot(σ_RA,  v_RA ·σ_t)
#     σ_Dec_eff = hypot(σ_Dec, v_Dec·σ_t)
# This stretches the uncertainty of a fast mover along its track exactly where
# find_orb does, mis-weighting fewer NEO/close-approach obs. The sky rate is
# estimated orbit-free (no integration) from the observed positions themselves.
# Reference: find_orb errors.cpp; https://www.projectpluto.com/errors.htm
DEFAULT_TIME_SIGMA_SEC = 1.0   # assumed 1σ timestamp accuracy when none is reported
_MOTION_WINDOW_DAYS = 0.05     # pair an obs with same-station neighbours within ~72 min


def apparent_sky_motion(jds, ra_deg, dec_deg, codes, window=_MOTION_WINDOW_DAYS):
    """Per-obs apparent sky rate (v_RA·cosδ, v_Dec) in arcsec/day, estimated
    ORBIT-FREE by finite-differencing the OBSERVED positions against the nearest
    same-station neighbour within `window` days (a tracklet's own points). Returns
    zeros for a lone obs with no such neighbour — its rate is unknown, so it gets
    no timing inflation. Cheap (no integration); the rate need only be order-right
    since it is multiplied by a sub-second σ_t."""
    jds = np.asarray(jds, dtype=float)
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    codes = np.asarray(codes)
    n = len(jds)
    vra = np.zeros(n); vdec = np.zeros(n)
    cosd = np.cos(np.radians(dec))
    for code in np.unique(codes):
        sel = np.flatnonzero(codes == code)
        srt = np.argsort(jds[sel], kind="stable")
        si = sel[srt]; ts = jds[si]
        for a in range(len(si)):
            best, bdt = None, window
            for b in (a - 1, a + 1):                      # nearest neighbour in time
                if 0 <= b < len(si):
                    dt = abs(ts[b] - ts[a])
                    if 0 < dt <= bdt:
                        bdt, best = dt, b
            if best is None:
                continue
            i, j = si[a], si[best]
            dt = jds[j] - jds[i]
            dra = ((ra[j] - ra[i] + 180) % 360 - 180) * cosd[i] * 3600.0
            ddec = (dec[j] - dec[i]) * 3600.0
            vra[i] = dra / dt; vdec[i] = ddec / dt
    return vra, vdec


def timing_inflated_sigmas(sigma_ra, sigma_dec, v_ra, v_dec, sigma_time_days):
    """Inflate per-axis σ (arcsec) in quadrature with the along-track timing smear
    (sky rate × time σ) projected onto each axis. All inputs are arrays; sky rate in
    arcsec/day, σ_t in days. Returns (σ_RA_eff, σ_Dec_eff). σ only ever grows, so
    SIGMA_FLOOR is preserved."""
    st = np.asarray(sigma_time_days, dtype=float)
    add_ra = np.abs(np.asarray(v_ra, dtype=float)) * st
    add_dec = np.abs(np.asarray(v_dec, dtype=float)) * st
    return np.hypot(sigma_ra, add_ra), np.hypot(sigma_dec, add_dec)


# --- over-observation reweighting (find_orb reweight_for_overobserving) ----------
# When a station piles up many observations of one object in a short time, the random
# centroiding error has long since been beaten down and only the (correlated)
# systematic error remains — so N such obs carry far less than N obs' worth of
# independent information. find_orb down-weights them: an obs with `n_nearby`
# same-station obs within ~`time_span` is weighted by sqrt(Nmax/(n_nearby+Nmax-1)),
# so a dense night counts like ~Nmax independent obs, not its full count. Without
# this, an over-observed apparition (e.g. 700 obs in 12 days) dominates the χ² and
# its systematics bias the orbit, drowning out sparse but well-separated apparitions.
# Defaults match find_orb's FILTERING line: time span 1.0 day, ceiling (Nmax) 5.
# Reference: https://www.projectpluto.com/errors.htm
OVEROBS_TIME_SPAN = 1.0      # days (find_orb default)
OVEROBS_CEILING = 5          # Nmax (find_orb default)
_OVEROBS_WINDOW = 4.0        # count same-station obs within WINDOW*time_span (find_orb span_limit)


def overobserving_factors(jds, codes, included=None,
                          time_span=OVEROBS_TIME_SPAN, ceiling=OVEROBS_CEILING):
    """Per-obs weight-reduction factors (each ≤ 1) for over-observation, porting
    find_orb's reweight_for_overobserving / n_nearby_obs (orb_func.cpp). For obs i,
    `n_nearby` = 1 + Σ exp(−½(Δt/time_span)²) over OTHER included obs from the SAME
    station within WINDOW·time_span, and the factor is sqrt(Nmax/(n_nearby+Nmax−1)).
    Multiply the weight (1/σ) by this — equivalently inflate σ by 1/factor. Gaia
    (code 258) is returned at 1.0 (its astrometry arrives already de-correlated)."""
    jds = np.asarray(jds, dtype=float)
    codes = np.asarray(codes)
    n = len(jds)
    inc = np.ones(n, dtype=bool) if included is None else np.asarray(included, dtype=bool)
    Nmax = float(ceiling)
    window = _OVEROBS_WINDOW * time_span
    factors = np.ones(n)
    for code in np.unique(codes):
        sel = np.flatnonzero(codes == code)
        if str(code) == "258":                       # Gaia: already corrected
            continue
        t = jds[sel]
        srt = np.argsort(t, kind="stable")
        ts = t[srt]; incs = inc[sel][srt]
        for a in range(len(sel)):
            lo = np.searchsorted(ts, ts[a] - window)
            hi = np.searchsorted(ts, ts[a] + window)
            dt = (ts[lo:hi] - ts[a]) / time_span
            contrib = np.exp(-0.5 * dt * dt) * incs[lo:hi]
            contrib[a - lo] = 0.0                    # exclude self from the sum
            n_nearby = 1.0 + float(contrib.sum())
            factors[sel[srt[a]]] = math.sqrt(Nmax / (n_nearby + Nmax - 1.0))
    return factors
