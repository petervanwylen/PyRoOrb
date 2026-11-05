"""Progressive arc extension with automatic outlier rejection.

⚠️ PROVENANCE: This was originally a close port of Bill Gray's find_orb arc-extension +
blunder-rejection logic, read from the find_orb source and credited here:
  - extend_orbit_solution()       (find_orb/orb_func.cpp:4618)
  - attempt_extensions()          (find_orb/orb_func.cpp:4793)
  - auto_reject_obs()             (find_orb/orb_func.cpp:4716)
  - auto_reject_obs_within_arc()  (find_orb/orb_func.cpp:4779)
find_orb is GPL-2.0; this project is GPL-3.0 (see LICENSE.md). We reproduce the
*algorithm/structure*, not copied source lines.

The idea (which makes multi-opposition fits actually converge): never cold-start
a least-squares solve on a long, multi-loop arc — the residual landscape is full
of aliased minima. Instead:

  1. converge on the short initial sub-arc that the IOD seed came from (one basin);
  2. EXTEND: toggle on a few adjacent observations, refit, and KEEP THE RESULT
     ONLY IF IT IMPROVES (find_orb's best_orbit / score book-keeping). A bad
     extension is reverted, so it can never ruin the good sub-arc orbit;
  3. observations that won't fit are simply left OFF (find_orb leaves their
     is_included = 0; we expose this as the `mask`), letting the fit "step past"
     blunders. Adjacent obs are added almost unconditionally (within RESID_LIMIT
     sigmas); "fill" obs only if within MAX_SIGMA;
  4. at the very END, run a few rounds of outlier rejection that can both REJECT
     included obs (residual > outer) and REDEEM excluded ones (residual < inner),
     with hysteresis (inner < outer) so obs don't flip-flop, and with
     "evaporation" guards so we never reject away the arc or most of the data.

Residuals are in SIGMAS throughout, normalized per-observation (ADES rmsRA/rmsDec
or the Veres-2017 lookup, supplied by weights.py); the flat SIGMA_ARCSEC is only the
fallback when a caller hands the Evaluator no sigmas at all.
"""
import os as _os
import numpy as np
import time as _stime
from scipy.optimize import least_squares
from scipy.special import erfc

from pyroorb.forward_model import predict_radec_batch, C_AU_DAY
from pyroorb.twobody import quick_aeq, kepler_propagate
from pyroorb.orbit_common import (unphysical_penalty, IDENT_FAR_MIN_Q, IDENT_BRIDGE_MAX_NIGHTS,
                          count_nights, NIGHT_GAP_DAYS,
                          OPP_GAP_DAYS, SINGLE_NIGHT_SPAN_DAYS)

STAGE_TIMES = {}     # per-stage wall-clock accumulator (profiling; see tests/_stages.py)

# --- optional teaching-mode snapshots -------------------------------------
# When SNAPSHOTS is on, each history entry is tagged with the orbit state + the
# inclusion mask at that stage, so the web UI can replay that stage's residuals
# (WEB_UI_PLAN §5.2). Off by default -> zero cost / payload for normal fits; the
# web layer flips it on around a fit and back off afterwards.
SNAPSHOTS = False
def _hsnap(d, state, mask):
    """Tag a history dict with (_state,_mask) when SNAPSHOTS is on; else no-op."""
    if SNAPSHOTS:
        d["_state"] = [float(x) for x in np.asarray(state).ravel()]
        d["_mask"] = [bool(b) for b in np.asarray(mask).ravel()]
    return d


def _stadd(name, t0):
    STAGE_TIMES[name] = STAGE_TIMES.get(name, 0.0) + (_stime.perf_counter() - t0)

_BIG_RESID = 1e6     # arcsec sentinel for a non-finite (diverged-orbit) prediction


def _clamp(a):
    """Replace NaN/inf with a huge finite value so a diverged orbit (e.g. a far
    apparition propagated across NEO close approaches) can't crash the LM or poison
    RMS/score — the affected obs just read as gross outliers."""
    return np.where(np.isfinite(a), a, _BIG_RESID)


def _radec_resid(pred, ra, dec, cosd, sra, sdec):
    """Weighted (RA,Dec) residual vector in σ units. RA is wrapped to (-180,180] and
    scaled by cos(dec); the result interleaves per-obs [dRA/σ, dDec/σ, ...] (the
    column_stack(...).ravel() order the analytic Jacobian from `_radec_jac` is built
    against). NOT clamped — callers wrap in `_clamp` (and may append a penalty value
    first). This expression is the engine's single definition of an astrometric
    residual; it was previously copy-pasted across four LM fit bodies."""
    d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
    return np.column_stack([d * 3600.0 / sra, (pred[:, 1] - dec) * 3600.0 / sdec]).ravel()


def _radec_jac(J, cosd, sra, sdec):
    """Scale the analytic d(ra,dec)/d(params) Jacobian J:(m,2,k) into the weighted LM
    Jacobian (2m,k) matching `_radec_resid`'s row order. Diverged obs (non-finite) get
    a zero gradient. Callers needing a penalty row vstack it onto the result."""
    m, _, k = J.shape
    out = np.empty((m, 2, k))
    out[:, 0, :] = J[:, 0, :] * (cosd * 3600.0 / sra)[:, None]
    out[:, 1, :] = J[:, 1, :] * (3600.0 / sdec)[:, None]
    return np.where(np.isfinite(out), out, 0.0).reshape(2 * m, k)

# --- find_orb constants (sigmas, except where noted) ---------------------------
SIGMA_ARCSEC   = 1.0      # flat per-coordinate astrometric sigma (placeholder)
RESID_LIMIT    = 200.0    # extend_orbit_solution: residual_limit (adjacent obs)
MAX_SIGMA      = 2.0      # extend_orbit_solution: "safe" fill-obs criterion
REJECT_LIMIT   = 3.0      # automatic_outlier_rejection_limit (auto_reject start)
REJECT_CAP     = 60.0     # auto_reject_obs_within_arc: stop ramping past this
EXTEND_MAX_NFEV = int(_os.environ.get("PYROORB_EXTEND_MAX_NFEV", "100"))  # LM cap for the EXTENSION-
                          # stage fits only (subarc_extend seed-clean + extend loop + drop-suspect).
                          # Those refine from a near-converged orbit, so on heavy many-opposition
                          # arcs LM cannot reach xtol=1e-5 and grinds to the 400 cap WITHOUT moving
                          # the orbit (142561: a/e/i byte-identical 400->25 evals, 324s->39s). The
                          # LINKAGE/BRIDGE fits crawl from a swarm seed and genuinely need the budget
                          # (PAIR_2026EJ70 needs >=100 to link), so they keep the 400 default -- this
                          # is why the cap is applied per-call-site here, not globally to thin fits.
FIT_TOL_ITER   = 1e-5     # LM xtol/ftol for ITERATIVE fits (extension/linkage/rejection):
                          # only needs to be good enough for link/reject decisions. ~35%
                          # faster than the old machine-precision 1e-12 across the suite.
FIT_STALL_TOL_FACTOR = 1e-2   # stall guard (Evaluator._unstall): when a solve hands back the
                          # seed VERBATIM (nfev<=2, dx==0) it stopped on the tolerance rather
                          # than at a minimum, so re-solve once at this multiple of the
                          # caller's tolerance. 1e-2 clears the observed failures with room to
                          # spare (the 1e-5 stall converges at 1e-6) without approaching the
                          # machine-tight regime FIT_TOL_POLISH warns is a trap.
FIT_TOL_FINAL  = 1e-6     # FINAL covariance fit (thin=False): also loose -- verified that
                          # 1e-6 vs 1e-12 changes element sigmas by < 3e-6 relative (the
                          # covariance is J^T J at the minimum, unchanged by the last few
                          # iterations) and the orbit by sub-meter. Its own knob in case a
                          # future use needs it tighter.
# --- diligent "polish" solve for the INTERACTIVE full-step (fit(..., polish=True)) ---
# The web Full step seeds LM from the PREVIOUS converged orbit, then changes the obs mask.
# The Cartesian state is badly scaled (position ~1 au vs velocity ~5e-3 au/day), and scipy's
# least_squares(method="lm") default x_scale=1.0 hands MINPACK a UNIT diag (mode=2) -- i.e. NO
# variable scaling. Seeded at the old minimum, the first Gauss-Newton step then overshoots, the
# trust region collapses below xtol*|x|, and lmder returns the seed UNCHANGED (nfev=2) -- the
# orbit "freezes" while only the covariance/H update. The cure is x_scale="jac" (MINPACK
# mode=1, column-norm auto-scaling) plus a modestly tight tolerance in a RE-SEED loop: each
# least_squares call that stops early is re-seeded from its own result until the summed-square
# residual stops improving. Tighter is a trap: at 1e-10 a widely-spaced two-apparition NEO's
# curved chi2 valley (2007 AA2, 19-yr arc) never converges, so EVERY Full step burned the whole
# 600-eval budget (~10 s/click) to move `a` by ~1e-9 au (~1 formal sigma, <0.005" in displayed
# rms) past what the scaled 1e-8 solve reaches in <=10 evals (~0.15 s). See the web full_step
# handler (engine_api._apply_interactive).
FIT_TOL_POLISH   = 1e-8   # LM xtol/ftol/gtol for the interactive polish solve
POLISH_PER_NFEV  = 100    # LM budget per re-seed round (a scaled round rarely needs >10)
POLISH_MAX_ROUNDS = 6     # cap on re-seed rounds, so a pathological object stays interactive
POLISH_REL_IMPROVE = 1e-4 # stop re-seeding once a round cuts the SSQ residual by less than this
# --- obscode sigma relaxation (find_orb sigma.txt philosophy) ---
# An observer's stated (often very tight) uncertainties are, per find_orb, only DEFAULT
# values we may override. When one obscode's observations are CONSISTENTLY discordant for
# this object (median sigma-residual high) AND its claimed sigma is tighter than find_orb's
# sigma.txt default, we trust that sigma less and relax it to the default -- so the code's
# good sub-arcsecond obs stop reading as 5-sigma outliers while a true blunder stays a
# blunder. (find_orb sigma.txt: 0.5" modern / 3" pre-1993 catch-all.)
SIGMA_TXT_MODERN  = 0.5   # sigma.txt catch-all default (post-1993), arcsec
SIGMA_TXT_OLD     = 3.0   # sigma.txt pre-1993 default, arcsec
JD_1993           = 2448988.5    # 1993-01-01 (TDB≈UTC to <0.001 d, fine for this cutoff)
RELAX_TRIGGER_SIG = 2.5   # relax an obscode whose included obs sit at a median sigma-
                          # residual above this (systematic, not a lone blunder)
RELAX_MIN_OBS     = 3     # ...and only with at least this many obs to judge "consistently"
FIT_MIN_OBS       = 3     # an LM fit of the 6-element state needs >= 3 obs (6 residuals);
                          # scipy's method='lm' raises if residuals < variables. Trial masks
                          # that would drop below this are underdetermined -- skip them.
RELAX_MAX_RAW     = 1.0   # ...and only if the median RAW residual is small (arcsec): that
                          # is the over-confident-sigma signature. A code that's high-sigma
                          # because its RAW residuals are large is genuinely discordant
                          # (mislinkage/contaminant) -- relaxing it would mask a real
                          # problem and destabilise linkage, so leave it for rejection.
RELAX_REJECT_FRAC = 0.10  # ...the EXCLUDED-side relax trigger's fraction: this much of an
                          # obscode's obs REJECTED at small raw residual means an over-confident
                          # stated sigma, not bad positions (mechanism in
                          # _relax_overconfident_obscodes). The RELAX_MAX_RAW gate is what keeps
                          # a GENUINELY discordant night rejected (2025 KV14's 2016 F51
                          # tracklet, raw >> 1").
DROP_SUSPECT_RMS = 0.7    # ONE OF TWO triggers for the (expensive) single-night mislinkage
                          # troubleshooting: a POOR fit -- raw arcsec RMS above this -- with more
                          # than one excluded night. An UNLINKED APPARITION triggers it regardless
                          # of rms (see attempt_extensions), so this bound is not a floor on when
                          # the cascade can run.
DROP_SUSPECT_MIN_FRAC = 0.03  # ...and only TRIAL a single night that is at least this
                          # fraction of the included obs. _drop_suspect only pays off for a
                          # HIGH-LEVERAGE night (one that can DRAG a well-fit orbit); a tiny
                          # night in an overdetermined fit can't shift the orbit, so its
                          # trial always gains 0 (a real mislinkage there just shows high
                          # residual and is rejected normally). On a 5000-obs / 41-opp mega
                          # object this skips ~40 useless full re-fits (2738: 53s -> ~24s).
ARC_TIME_LIMIT = 36500.0  # extend cap (days). find_orb defaults to 3650 as an
                          # anti-lockup guard on load, then raises it; our batch
                          # propagator is fast, so we allow the full arc (~100 yr).
MAX_EXTEND_ITERS = 200    # backstop on the extension loop
EXTEND_SCORE_MARGIN = 0.1     # _score must improve by more than this to KEEP an extension step
EXTEND_GROW_OK_SCORE = 10.0   # ...but a step that ADDS obs is accepted if its score is below this,
                              # even without an improvement (a larger sane arc beats a marginally
                              # tighter shorter one). _score is normalized RMS (σ) + penalties.
EXTEND_BLOWUP_SCORE = 10000.0 # score above this = the orbit blew up; stop extending and revert
ACTIVATE_KEEP_NORM = 5.0      # normalized-rms ceiling for a cluster-activation / suspect-drop refit
                              # to be accepted as a converged, keepable joint fit
# OPP_GAP_DAYS (orbit_common's, imported above, 100 d): a >100-day jump from the
# current arc reaches a NEW opposition; extension's job is to grow the current
# opposition, cross-opposition linking is the bridge's/cluster-activation's.
# (Oppositions are >1 yr apart; obs within one span weeks.) The cluster splitter
# (_clusters), the over-obs cap and this extend-guard share it. This file used to
# keep its own copy as APPARITION_GAP_DAYS, same value and same consecutive-gap
# rule; the other opposition rules in the tree disagree with it.
# NIGHT_GAP_DAYS -- the seed-clean night split and the excluded-night count -- is
# orbit_common's (imported above), not a copy: one consecutive-gap rule in the tree.
CROSS_APP_MAX_RMS = 1.5   # ...so extension may KEEP a cross-apparition jump only if those obs
                          # fit this well jointly. A poorly-fit jump (1993 TZ's single 1993 night
                          # at 2-3" onto the 2016 arc) is a half-link that PREEMPTS the multi-rev
                          # bridge (the apparition no longer reads as "empty"); leave it instead.
CROSS_APP_MAX_Q = 1.5     # ...but only POLICE this for low-perihelion (q < 1.5 au) in-progress
                          # orbits -- the NEO/Mars-crosser regime where multi-revolution linkage
                          # actually arises and is worth a little extra bridge time. A well-behaved
                          # main-belt/Jupiter-crosser (944 Hidalgo q~1.9) keeps the fast old path.
SEED_CLEAN_SIGMA = 5.0    # a gross outlier (this many σ) the well-determined arc rejects
SEED_CLEAN_MIN = 5        # never let seed pre-cleaning shrink the seed below this many obs
SEED_POOR_RMS = 1.0       # a seed fitting worse than this (") may harbour a contaminant (a
                          # merely rough-but-clean seed sits below this and skips the screen)
SEED_SHARP = 0.5          # ...confirmed only if dropping its worst night cuts seed rms below
                          # this fraction of the original (a real contaminant; a merely rough
                          # seed won't sharply clean, so we leave it alone)
SEED_REJOIN_SIGMA = 2.5   # if the ENTIRE dropped night is gross, it may be a distant arc
                          # CONSTRAINT not contamination: keep it if a joint fit reconciles it
                          # below this normalised rms (else ban it as a real contaminant)
THIN_TRIGGER = 400        # only thin the iterative fit when more obs than this are in
THIN_PER_DAY = 6          # ...keeping at most this many per calendar day (dense NEOs)
MAX_FIT_OBS  = 300        # ...and cap the WHOLE iterative fit to ~this many, spread evenly
                          # across oppositions. A 6-parameter orbit is wildly over-
                          # determined by 300 well-distributed obs; the per-day cap alone
                          # barely dents a many-opposition arc (thousands of obs spread one
                          # or two per night over decades), so every LM iteration would
                          # otherwise propagate ~1700 obs. The FINAL covariance fit
                          # (thin=False) still uses every obs — this only thins the path,
                          # EXCEPT past HUGE_ARC_OBS, where it is capped too (below).

# A ceiling for the fits that deliberately do NOT thin -- the final covariance refit and
# the interactive polish -- and only for arcs far past anything the engine was tuned on.
#
# The iterative path has been bounded since MAX_FIT_OBS, so on a truly enormous arc the
# remaining cost is those un-thinned fits: every LM iteration propagates the WHOLE arc.
# Measured on the live deploy, 433 Eros (18151 observations, 59 oppositions) takes 628 s
# -- ten and a half minutes holding a fit worker -- and the objects most likely to be
# typed into a public site on its first day are exactly the famous, century-arc ones.
#
# 4000 obs spread evenly over the oppositions is still ~13x what MAX_FIT_OBS considers
# sufficient to DETERMINE the orbit; the extra rows here buy covariance precision, which
# scales as sqrt(N), so the sigma this delivers on a 18k-obs arc is within ~1.5x of the
# full-arc one and ERRS WIDE -- an overstated uncertainty, never an understated one.
# That direction is the whole reason for a cap rather than a rescale: a rescale would
# assume the residuals are independent, and on a century of survey astrometry they are
# emphatically not (correlated per night, per station, per catalogue), so correcting the
# sigma back down would claim a precision the subsample cannot support.
#
# HUGE_ARC_OBS sits ABOVE the largest regression fixture (4356 obs, 10052) on purpose:
# every golden in the suite therefore takes the identical un-capped path it always has,
# so this cannot silently move a delivered orbit that the suite is pinning.
HUGE_ARC_OBS = 6000       # only arcs past this are capped when thin=False
HUGE_ARC_FIT_OBS = 4000   # ...to this many, spread evenly across oppositions
LINK_MANY_APPS = 8        # >this many apparitions: skip wholesale activation, use the linker


class Evaluator:
    """Holds the full observation set and provides residuals / subset fits, all
    using the single-sweep batch propagator so each call costs ~one integration.

    Residuals are NORMALIZED by per-observation 1-sigma uncertainties (sigma_ra,
    sigma_dec, arcsec) — from ADES rmsRA/rmsDec or the Veres-2017 lookup — exactly
    as find_orb does. The LM minimizes (O-C)/sigma, the rejection thresholds are in
    real sigmas, and the resulting covariance is the proper weighted-LS one."""
    def __init__(self, base_sim, t0_tdb, times_tdb, Robs, obs_ra, obs_dec,
                 sigma_ra=None, sigma_dec=None, predict_fn=None, predict_jac_fn=None,
                 fit_sigma_ra=None, fit_sigma_dec=None, obscodes=None):
        self.base_sim, self.t0 = base_sim, t0_tdb
        self.obscodes = None if obscodes is None else np.asarray(obscodes)
        self._predict = predict_fn or predict_radec_batch   # force-model backend
        self._predict_jac = predict_jac_fn    # optional: predictions + analytic Jacobian
        self.times_tdb = np.asarray(times_tdb)
        self.Robs = list(Robs)
        self.ra = np.asarray(obs_ra)
        self.dec = np.asarray(obs_dec)
        self.cosd = np.cos(np.radians(self.dec))
        n = len(self.ra)
        self.sig_ra = (np.full(n, SIGMA_ARCSEC) if sigma_ra is None
                       else np.asarray(sigma_ra, dtype=float))
        self.sig_dec = (np.full(n, SIGMA_ARCSEC) if sigma_dec is None
                        else np.asarray(sigma_dec, dtype=float))
        # σ used to WEIGHT the least-squares fit. Normally == sig_ra/dec, but may be
        # inflated for over-observed obs (find_orb's reweight_for_overobserving) so a
        # dense apparition doesn't dominate the orbit. Rejection/residual reporting
        # always use the RAW sig_ra/dec — over-observation is a fit-weighting choice,
        # not a statement that the obs is an outlier.
        self.fit_sig_ra = self.sig_ra if fit_sigma_ra is None else np.asarray(fit_sigma_ra, float)
        self.fit_sig_dec = self.sig_dec if fit_sigma_dec is None else np.asarray(fit_sigma_dec, float)
        self.n = n
        self._resid_key = None          # single-slot memo for the full-arc residual sweep
        self._resid_val = None
        self._sub_key = None            # single-slot memo for the masked-subset sweep
        self._sub_val = None
        # Optional non-gravitational config: an (A, gr) pair (see nongrav.py / force_assist).
        # None is the pure-gravity path. When set (by the nongrav refine), every residual
        # sweep propagates WITH the out-gassing acceleration so the reported O-C is correct.
        self.nongrav = None
        # Phase-bridge link relaxation (see relax_link_opposition). A confirmed multi-revolution
        # link can leave a far apparition with a COHERENT (gravity-only) residual offset; we
        # temporarily inflate its sigmas so the reject loop keeps it, snapshotting the pre-relax
        # values here so restore_link_relaxation() can undo it EXACTLY before the nongrav refit.
        self._link_saved = None          # (sig_ra, sig_dec, fit_sig_ra, fit_sig_dec) pre-relax copies
        self._link_relaxed_mask = None   # bool[n]: which obs are currently link-relaxed

    def _predict_with_ng(self, state, tt, Rs):
        """Backend prediction, transparently adding the non-gravitational acceleration
        when `self.nongrav` is set (gravity-only otherwise -- the kwarg is only passed
        then, so a backend without nongrav support is untouched)."""
        if self.nongrav is None:
            return self._predict(self.base_sim, self.t0, state, tt, Rs)
        return self._predict(self.base_sim, self.t0, state, tt, Rs, nongrav=self.nongrav)

    @staticmethod
    def _ng_key(ng):
        """Stable CONTENT key for the nongrav config (None or an (A, gr) pair) used in
        the residual memo. Keying on id(ng) is a stale-sweep footgun: CPython recycles
        ids, so a reassigned nongrav tuple landing at a previously-seen address could
        return another config's cached sweep for the same state bytes. (A is a length-3
        accel vector; gr is a small g(r) dict -> repr is a deterministic hashable key.)"""
        if ng is None:
            return None
        A, gr = ng
        return (tuple(float(x) for x in A), repr(gr))

    # per-obs (dRA*cosδ, dDec) in arcsec, for ALL obs, at a given state. Memoized on the
    # state bytes: a converged state is residual-swept several times in a row (_is_clean,
    # resid_sigmas/auto_reject, scoring) and the full-arc integration is identical each
    # time -- it depends ONLY on `state` (base_sim/t0/times/Robs are fixed; fit_sigmas don't
    # enter here), so one sweep serves them all.
    def resid_arcsec(self, state):
        key = (state.tobytes(), self._ng_key(self.nongrav))
        if key == self._resid_key:
            return self._resid_val
        pred = self._predict_with_ng(state, self.times_tdb, self.Robs)
        dra = ((pred[:, 0] - self.ra + 180) % 360 - 180) * self.cosd * 3600.0
        ddec = (pred[:, 1] - self.dec) * 3600.0
        self._resid_key = key
        self._resid_val = (_clamp(dra), _clamp(ddec))
        return self._resid_val

    # per-obs (dRA*cosδ, dDec) arcsec over ONLY a masked subset — propagates just
    # those epochs, so a far diverging apparition can't NaN-poison the whole sweep.
    # Memoized on (state, mask): rms_arcsec, rms_norm and _score all hit the same
    # (state, mask) back-to-back, so one subset sweep serves the trio.
    def _check_mask(self, mask):
        """An inclusion mask is a POSITIONAL claim about THIS evaluator's obs list.

        Every masked path below reduces it with ``np.flatnonzero``, which accepts any
        length: a mask one observation short simply yields fewer indices, so the fit
        silently runs on a TRUNCATED arc and reports a clean rms for an arc nobody
        asked about. This is the one place every external mask (an interactive re-fit,
        a rescue's reduced arc, a client payload) has to pass through, so it is the
        one place worth checking. Cheap next to an integration."""
        if len(mask) != self.n:
            raise ValueError(f"inclusion mask has {len(mask)} entries but this "
                             f"evaluator holds {self.n} observations")

    def resid_subset(self, state, mask):
        self._check_mask(mask)
        key = (state.tobytes(), mask.tobytes(), self._ng_key(self.nongrav))
        if key == self._sub_key:
            return self._sub_val
        idx = np.flatnonzero(mask)
        tt = self.times_tdb[idx]; Rs = [self.Robs[i] for i in idx]
        pred = self._predict_with_ng(state, tt, Rs)
        dra = ((pred[:, 0] - self.ra[idx] + 180) % 360 - 180) * self.cosd[idx] * 3600.0
        ddec = (pred[:, 1] - self.dec[idx]) * 3600.0
        self._sub_key = key; self._sub_val = (dra, ddec)
        return dra, ddec

    # find_orb's total_residual_err(): hypot of per-obs sigma-normalised residuals
    def resid_sigmas(self, state):
        dra, ddec = self.resid_arcsec(state)
        return np.hypot(dra / self.sig_ra, ddec / self.sig_dec)

    def rms_arcsec(self, state, mask):
        dra, ddec = self.resid_subset(state, mask)        # propagate only included obs
        v = np.concatenate([_clamp(dra), _clamp(ddec)])
        return float(np.sqrt(np.mean(v**2)))

    def rms_norm(self, state, mask):
        """Normalized (dimensionless) RMS over included obs — ~1 for a fit
        consistent with the assigned uncertainties (find_orb's primary metric)."""
        idx = np.flatnonzero(mask)
        dra, ddec = self.resid_subset(state, mask)
        v = np.concatenate([_clamp(dra) / self.sig_ra[idx],
                            _clamp(ddec) / self.sig_dec[idx]])
        return float(np.sqrt(np.mean(v**2)))

    # --- phase-bridge link relaxation (scoped, reversible) ---------------------------------------
    def relax_link_opposition(self, obs_mask, far_norm, target_norm):
        """Reduce trust in the obs of a CONFIRMED multi-revolution-link apparition whose residuals
        share a COHERENT offset (the whole cloud is shifted, not scattered). On a gravity-only fit
        of a body with an unmodeled non-gravitational force the far apparition can sit a few-σ off
        even when the link is real; the wrong-revolution alternative was already rejected at the
        bridge's acceptance gate (it glues thousands of σ off), so this offset is real signal
        pending the nongrav refine -- NOT a blunder. Absorb that coherent offset into the obs'
        sigma: scale σ by far_norm/target_norm so the apparition reads ~target_norm σ, which is the
        standard treatment of a known systematic and keeps the downstream outlier rejection from
        gutting the whole (real) apparition. find_orb makes the same trust move for
        consistently-offset-but-real obs.

        This is SCOPED and REVERSIBLE (vs an anonymous in-place σ multiply): the pre-relax sigmas
        are snapshotted so restore_link_relaxation() can undo it EXACTLY before the nongrav refit,
        where the offset becomes MODELED and the apparition must be fit at full weight (see
        fit_object.refine_nongrav / discover_nongrav). It is non-compounding: a later relaxation of
        a DIFFERENT apparition snapshots from the same pre-relax base, never stacking scales."""
        if far_norm <= target_norm:
            return False                      # already within tolerance -- nothing to relax
        idx = np.flatnonzero(obs_mask)
        if idx.size == 0:
            return False
        if self._link_saved is None:          # first relaxation of this fit: snapshot the true base
            self._link_saved = (self.sig_ra.copy(), self.sig_dec.copy(),
                                self.fit_sig_ra.copy(), self.fit_sig_dec.copy())
            self._link_relaxed_mask = np.zeros(self.n, bool)
        scale = far_norm / target_norm
        for arr in (self.sig_ra, self.sig_dec, self.fit_sig_ra, self.fit_sig_dec):
            arr[idx] *= scale
        self._link_relaxed_mask[idx] = True
        self._resid_key = self._sub_key = None     # sigmas changed -> drop the stale residual memo
        return True

    def restore_link_relaxation(self):
        """Undo every relax_link_opposition() exactly, restoring the relaxed obs' pre-relax sigmas.
        Called before a non-gravitational refit (the coherent offset is now modeled, so the
        apparition must contribute at full weight) and idempotent (no-op if nothing was relaxed)."""
        if self._link_saved is None:
            return False
        m = self._link_relaxed_mask
        sr, sd, fr, fd = self._link_saved
        self.sig_ra[m] = sr[m]; self.sig_dec[m] = sd[m]
        self.fit_sig_ra[m] = fr[m]; self.fit_sig_dec[m] = fd[m]
        self._link_saved = self._link_relaxed_mask = None
        self._resid_key = self._sub_key = None
        return True

    def _thin(self, idx):
        """Down-sample an over-observed included set for the FITTING only: keep at
        most THIN_PER_DAY obs per calendar day (evenly spread). An over-observed night
        (find_orb's reweight_for_overobserving territory) carries little extra orbit
        information, so propagating all of it every LM iteration is the dominant cost
        on dense NEOs. Residuals, rejection and the final covariance fit still use ALL
        obs — this only thins the iterative least-squares."""
        if len(idx) <= THIN_TRIGGER:
            return idx
        days = np.floor(self.times_tdb[idx]).astype(np.int64)
        keep = []
        for d in np.unique(days):
            g = idx[days == d]
            if len(g) <= THIN_PER_DAY:
                keep.append(g)
            else:
                keep.append(g[np.linspace(0, len(g) - 1, THIN_PER_DAY).round().astype(int)])
        idx = np.sort(np.concatenate(keep))
        # global cap: spread MAX_FIT_OBS evenly OVER OPPOSITIONS (keeps a geometric
        # handful per apparition incl. each apparition's endpoints, which is all the
        # orbit needs) so a many-opposition arc doesn't drag every iteration.
        return self._spread_over_oppositions(idx, MAX_FIT_OBS)

    def _spread_over_oppositions(self, idx, cap):
        """At most `cap` of `idx`, spread evenly across its apparitions.

        Splitting on the apparition gap first is what makes the subsample keep its
        GEOMETRY: a flat every-nth stride over a century-long arc would happily spend
        its whole budget on the densest opposition and drop the far-separated ones that
        actually constrain the orbit. Each apparition's endpoints survive, since
        ``linspace`` includes both ends of every group."""
        if len(idx) <= cap:
            return idx
        t = self.times_tdb[idx]
        bnd = np.flatnonzero(np.diff(t) > OPP_GAP_DAYS) + 1
        groups = np.split(idx, bnd)
        per = max(4, cap // len(groups))
        out = [g if len(g) <= per
               else g[np.linspace(0, len(g) - 1, per).round().astype(int)]
               for g in groups]
        return np.concatenate(out)

    def _cap_huge_arc(self, idx):
        """Bound an UN-thinned fit on an arc far past anything the engine was tuned on.

        The fits that pass ``thin=False`` (the final covariance refit) or ``polish=True``
        (the interactive Full step) deliberately use every included observation, and on
        an ordinary arc they should: that is what makes the delivered covariance honest.
        On a 18000-observation century arc it is also what makes a single fit take ten
        minutes, and the cost is paid per LM ITERATION.

        So past HUGE_ARC_OBS -- above every regression fixture, so no golden changes --
        those fits run on HUGE_ARC_FIT_OBS observations spread across the apparitions.
        See the constants for why this errs toward a WIDER sigma and is not rescaled."""
        if len(idx) <= HUGE_ARC_OBS:
            return idx
        return self._spread_over_oppositions(idx, HUGE_ARC_FIT_OBS)

    def fit(self, seed, mask, max_nfev=None, thin=True, polish=False, stall_guard=False):
        """full_improvement analogue: weighted LM differential correction over the
        currently-included observations, minimizing NORMALIZED residuals. `max_nfev`
        is capped low for activation trials (a fit either converges fast or never).
        `thin=True` fits a per-day-capped subsample of an over-observed set (speed on
        dense NEOs); the FINAL covariance fit passes thin=False to use every obs.

        `polish=True` runs the DILIGENT interactive solve (the web Full step): a tight
        tolerance in a re-seed loop, so an orbit seeded from the previous minimum after
        an obs-mask change is driven to the NEW minimum instead of freezing at the seed
        (see FIT_TOL_POLISH and `_polish_solve`). It never thins (every included obs is
        used) and honours the same active force/nongrav model as any other `fit` call.

        When the backend supplies an analytic Jacobian (`predict_jac_fn`), we hand
        scipy a `jac` callable instead of letting it finite-difference: the nominal
        prediction and all 6 Jacobian columns come from ONE multi-particle
        integration (see force_assist.predict_radec_jac_batch), so each LM iteration
        costs ~1 sweep rather than ~7. A single-slot cache shares that integration
        between the paired fun/jac calls MINPACK makes at the same state."""
        self._check_mask(mask)
        if max_nfev is None:
            max_nfev = 400
        idx = np.flatnonzero(mask)
        # A loose LM tolerance is plenty: the last iterations to machine precision are
        # wasted (~35% faster across the suite at 1e-6, 0 used/rms/status regressions and
        # element sigmas unchanged to <3e-6 relative). thin=False keeps its own knob.
        # A polish solve overrides both with the tight FIT_TOL_POLISH (and never thins).
        tol = FIT_TOL_POLISH if polish else (FIT_TOL_ITER if thin else FIT_TOL_FINAL)
        if thin and not polish:
            idx = self._thin(idx)
        else:
            idx = self._cap_huge_arc(idx)   # the un-thinned fits still get a ceiling on
                                            # an enormous arc -- see _cap_huge_arc
        tt = self.times_tdb[idx]; Rs = [self.Robs[i] for i in idx]
        ra = self.ra[idx]; dec = self.dec[idx]; cosd = self.cosd[idx]
        sra = self.fit_sig_ra[idx]; sdec = self.fit_sig_dec[idx]; m = len(idx)  # fit weighting
        if self._predict_jac is not None:
            cache = {}
            def _compute(state):
                key = state.tobytes()
                hit = cache.get(key)
                if hit is None:
                    cache.clear()
                    hit = cache[key] = self._predict_jac(self.base_sim, self.t0, state, tt, Rs)
                return hit
            def resid(state):
                pred, _ = _compute(state)
                return _clamp(_radec_resid(pred, ra, dec, cosd, sra, sdec))  # never hand MINPACK a non-finite value
            def jac(state):
                _, J = _compute(state)            # J: (m,2,6) d(ra,dec)/d(state0)
                return _radec_jac(J, cosd, sra, sdec)         # row order matches resid; diverged obs → no gradient
            if polish:
                return self._polish_solve(resid, seed, tol, max_nfev, jac=jac)
            result = least_squares(resid, seed, jac=jac, method="lm",
                                xtol=tol, ftol=tol, max_nfev=max_nfev)
            if stall_guard:
                result = self._unstall(resid, seed, result, tol, max_nfev, jac=jac)
            return result.x, result
        def resid(state):
            pred = self._predict(self.base_sim, self.t0, state, tt, Rs)
            return _clamp(_radec_resid(pred, ra, dec, cosd, sra, sdec))
        if polish:
            return self._polish_solve(resid, seed, tol, max_nfev, jac=None)
        result = least_squares(resid, seed, method="lm",
                            xtol=tol, ftol=tol, max_nfev=max_nfev)
        if stall_guard:
            result = self._unstall(resid, seed, result, tol, max_nfev, jac=None)
        return result.x, result

    @staticmethod
    def _unstall(resid, seed, result, tol, max_nfev, jac=None):
        """STALL GUARD: a solve that returned the SEED VERBATIM stopped on the tolerance,
        not at a minimum -- re-solve it once, tighter.

        MINPACK's xtol test compares the trial step against `xtol * |x|`. The Cartesian
        state has |x| ~ 1 (au), and a real orbit correction after an obs-mask change is
        often only ~1e-5 RELATIVE -- the same order as FIT_TOL_ITER. So lmder can take its
        first step, judge it "converged", and hand back `nfev=2, status=3, dx=0`: the
        caller records a new stage, but the orbit never saw the obs that were just added.
        Downstream that is not a slow fit, it is a WRONG one -- the reject loop then scores
        every obs against an orbit blind to them and clips good astrometry (2002 QG17:
        extend1 added the whole 2026 apparition, returned extend0's state byte-identical
        at rms 0.826", and three sub-arcsecond obs were rejected at 3.9-7.4 sigma; the same
        mask re-solved at 1e-6 gives rms 0.395" and NOTHING above 3 sigma).

        Note this is a TOLERANCE effect, not a scaling one: `x_scale="jac"` (the cure for
        the interactive full-step freeze, see `_polish_solve`) makes no difference here --
        at 1e-5 the scaled and unscaled solves both stop at nfev=2, and at 1e-6 both
        converge. So the retry tightens the tolerance rather than rescaling.

        OPT-IN (`fit(..., stall_guard=True)`), and deliberately not the default. Every
        `ev.fit` feeds something -- IOD sub-arc scoring, linkage swarms, bridge candidate
        ranking -- that COMPARES fits against each other, and silently re-converging one
        of them reroutes those choices: switched on globally it moved 2026 NA1's IOD start
        arc from 35 obs to 23 and cost two fixtures their orbit. So it is armed only where
        a stalled fit is not merely imprecise but WRONG: the fits whose obs mask just
        changed and whose residuals are then judged against absolute sigma thresholds.

        Fires only on the exact no-movement signature, so a genuinely-converged refit
        (any step at all, or a cap-stop at status 0 -- that has its own retry contract in
        `_final_lm`) pays nothing. The retry keeps whichever result has the smaller
        residual sum, so this can never make a fit worse than it already was."""
        seed = np.asarray(seed, float)
        if result.nfev > 2 or np.any(result.x != seed) or getattr(result, "status", 1) == 0:
            return result
        kw = dict(method="lm", xtol=tol * FIT_STALL_TOL_FACTOR,
                  ftol=tol * FIT_STALL_TOL_FACTOR, max_nfev=max_nfev)
        if jac is not None:
            kw["jac"] = jac
        try:
            res2 = least_squares(resid, seed, **kw)
        except Exception:
            return result
        if float(np.dot(res2.fun, res2.fun)) < float(np.dot(result.fun, result.fun)):
            return res2
        return result

    @staticmethod
    def _polish_solve(resid, seed, tol, max_nfev, jac=None):
        """Diligent LM solve for the interactive Full step: a PROPERLY-SCALED
        (x_scale="jac" -> MINPACK mode-1 column-norm scaling) least_squares, re-seeded
        from its own result until the summed-square residual stops improving. Without
        the scaling, a single lmder call seeded at the previous minimum stalls at
        nfev=2 (the badly-scaled first step overshoots and the trust region collapses
        below xtol*|x|). The re-seed loop is kept as a cheap diligence backstop; see
        the FIT_TOL_POLISH block for why the tolerance must NOT be machine-tight.
        Returns (state, OptimizeResult) exactly like `fit`, so the covariance is built
        from the FINAL round's Jacobian at the reached minimum."""
        per_nfev = min(int(max_nfev), POLISH_PER_NFEV)
        kw = dict(method="lm", xtol=tol, ftol=tol, gtol=tol, x_scale="jac")
        if jac is not None:
            kw["jac"] = jac
        st = np.asarray(seed, dtype=float)
        result = least_squares(resid, st, max_nfev=per_nfev, **kw)
        prev = float(np.dot(result.fun, result.fun))
        for _ in range(POLISH_MAX_ROUNDS - 1):
            nxt = least_squares(resid, result.x, max_nfev=per_nfev, **kw)
            cur = float(np.dot(nxt.fun, nxt.fun))
            improved = (prev - cur) > POLISH_REL_IMPROVE * max(prev, 1e-30)
            if cur <= prev:                       # keep the better (or equal) round
                result, prev = nxt, cur
            if not improved:
                break
        return result.x, result

    def evaluate(self, state, mask):
        """Weighted residual vector + analytic Jacobian AT `state` over the included
        obs, packaged like a scipy least_squares result (.x/.fun/.jac), but WITHOUT any
        differential correction -- a zero-iteration "fit". This rebuilds the covariance
        when an orbit is re-anchored to a new osculating epoch (set-epoch): the orbit and
        its residuals are held fixed; only the epoch / state parameterization changes, so
        the covariance must be re-evaluated at the new epoch's Jacobian. Gravity-only
        (a re-anchored nongrav orbit is re-derived through refine_nongrav instead)."""
        self._check_mask(mask)
        from types import SimpleNamespace
        state = np.asarray(state, dtype=float)
        idx = np.flatnonzero(mask)
        tt = self.times_tdb[idx]; Rs = [self.Robs[i] for i in idx]
        ra = self.ra[idx]; dec = self.dec[idx]; cosd = self.cosd[idx]
        sra = self.fit_sig_ra[idx]; sdec = self.fit_sig_dec[idx]; m = len(idx)

        def _fun(p):
            pred = self._predict(self.base_sim, self.t0, p, tt, Rs)
            return _clamp(_radec_resid(pred, ra, dec, cosd, sra, sdec))

        if self._predict_jac is not None:
            pred, J = self._predict_jac(self.base_sim, self.t0, state, tt, Rs)
            fun = _clamp(_radec_resid(pred, ra, dec, cosd, sra, sdec))
            jac = _radec_jac(J, cosd, sra, sdec)
        else:                                            # forward-difference fallback
            fun = _fun(state)
            jac = np.empty((2*m, 6))
            for k in range(6):
                h = 1e-7 * max(1.0, abs(state[k]))
                sp = state.copy(); sp[k] += h
                jac[:, k] = (_fun(sp) - fun) / h
        return SimpleNamespace(x=state, fun=fun, jac=jac, success=True,
                               status=4, nfev=1, cost=0.5*float(np.dot(fun, fun)))

    # Levenberg damping sequence for the single interactive step: try the pure
    # Gauss-Newton correction first (what find_orb's full step is on a healthy arc),
    # then progressively damp toward gradient descent only if it made things worse.
    STEP_LAMBDAS = (0.0, 1e-3, 1e-1, 10.0)

    def step(self, seed, mask):
        """ONE damped differential-correction step -- find_orb's `f` "Full step"
        (one least-squares correction per keypress; Auto-Solve is the separate
        iterate-to-convergence action). Evaluates the analytic Jacobian + weighted
        residuals at `seed`, solves the column-scaled damped normal equations for a
        single correction, and takes it -- escalating the damping through
        STEP_LAMBDAS if the step worsens the residuals, and returning the seed
        UNCHANGED (an honest no-op: already converged) when nothing improves. No
        iteration: click again for the next step. Returns (state, result) shaped
        like fit(), with .fun/.jac re-evaluated at the DELIVERED state so the
        covariance/H rebuild is honest."""
        self._check_mask(mask)
        seed = np.asarray(seed, dtype=float)
        at = self.evaluate(seed, mask)
        ssq0 = float(np.dot(at.fun, at.fun))
        if not (np.all(np.isfinite(at.jac)) and np.all(np.isfinite(at.fun))):
            return seed, at                       # diverged geometry -> honest no-op
        # Column-scale the normal equations (the MINPACK mode-1 analogue): the raw
        # Cartesian state spans ~1 au positions vs ~5e-3 au/day velocities.
        d = np.linalg.norm(at.jac, axis=0)
        d[d == 0] = 1.0
        Jh = at.jac / d
        A = Jh.T @ Jh
        g = Jh.T @ at.fun
        # residual-only probe (no Jacobian) for the trial states
        idx = np.flatnonzero(mask)
        tt = self.times_tdb[idx]; Rs = [self.Robs[i] for i in idx]
        ra = self.ra[idx]; dec = self.dec[idx]; cosd = self.cosd[idx]
        sra = self.fit_sig_ra[idx]; sdec = self.fit_sig_dec[idx]

        def _ssq(p):
            pred = self._predict(self.base_sim, self.t0, p, tt, Rs)
            r = _clamp(_radec_resid(pred, ra, dec, cosd, sra, sdec))
            return float(np.dot(r, r))

        eye = np.eye(len(seed))
        for lam in self.STEP_LAMBDAS:
            try:
                delta = np.linalg.solve(A + lam * eye, -g) / d
            except np.linalg.LinAlgError:
                continue
            trial = seed + delta
            if np.all(np.isfinite(trial)) and _ssq(trial) < ssq0:
                return trial, self.evaluate(trial, mask)
        return seed, at                           # no improving step -> converged


# --- find_orb-style soft blunder reweighting (the "soft-weighting" paradigm) -------------
# find_orb (orb_func.cpp reweight_for_blunders, use_blunder_method) down-weights every obs each
# differential-correction step by a Bayesian factor instead of hard-rejecting it -- and crucially
# NEVER removes it (floored weight), so an obs the current orbit doesn't fit yet (a blunder, OR a
# far apparition mid-bridge whose period is still wrong) loses pull but RECOVERS as the orbit
# converges onto it. Here we use it NON-DESTRUCTIVELY as a final polish: a couple of IRLS passes
# that let the orbit slide off marginal blunders it would otherwise be dragged by, then a clean
# full-weight refit. (The old "Path A" was slow only because it ran fixed full-arc passes with no
# early exit, REPLACING the reject loop on mega-objects; as a gated, thinned, end-only polish it
# is ~free -- see path-a-blunder-reweight-fails.)
PROB_BLUNDER = 0.02          # find_orb default: assumed fraction of blunders
SOFT_W_FLOOR = 1e-6          # never drive a weight to zero (find_orb's 1e-90 guard, relaxed)
SOFT_PASSES = 2              # IRLS down-weight passes before the final full-weight refit


def _blunder_weight(resid_sig):
    """find_orb's Bayesian blunder weight: P(good)/(P_blunder + P(good)), with P(good) the
    two-sided normal tail erfc(r/sqrt2) at the obs's sigma-residual. ~0.96 at 0.67sigma,
    ~0.33 at 2.57sigma, ~0.09 at 3.1sigma. Floored, never zero."""
    tail = erfc(np.asarray(resid_sig) / np.sqrt(2.0))
    return np.maximum(tail / (PROB_BLUNDER + tail), SOFT_W_FLOOR)


SOFT_W_USED = 0.30           # an obs whose converged blunder weight falls below this is reported
                             # "unused" (find_orb's soft exclusion by threshold ~2.6 sigma). It is
                             # NOT removed -- it stays available for redemption -- just dropped
                             # from the reported rms/used set so the orbit reads at its true noise.


def _soft_polish(ev, state, mask):
    """END-OF-FIT soft polish (find_orb's Bayesian blunder reweighting), run ONCE on the final
    orbit -- never mid-link. A few IRLS passes down-weight (never drop) obs by their blunder
    weight, un-sticking the orbit from a marginal-blunder-dragged local minimum; then obs whose
    converged weight is below SOFT_W_USED are reported 'unused' (recoverable, not removed) and the
    orbit is refit on the kept set. Returns (state, result, kept_mask)."""
    base_ra, base_dec = ev.fit_sig_ra.copy(), ev.fit_sig_dec.copy()
    for _ in range(SOFT_PASSES):                          # IRLS: un-stick from the blunder drag
        w = _blunder_weight(ev.resid_sigmas(state))       # per-obs, raw-sigma residual
        ev.fit_sig_ra = base_ra / np.sqrt(w)              # inflate fit-sigma -> down-weight
        ev.fit_sig_dec = base_dec / np.sqrt(w)
        state, _ = ev.fit(state, mask, thin=True)         # thinned: just need the better basin
    ev.fit_sig_ra, ev.fit_sig_dec = base_ra, base_dec     # raw weighting -> clean escaped orbit
    state, result = ev.fit(state, mask, thin=False)
    keep = mask.copy()                                    # soft-exclude: report obs below the
    for _ in range(2):                                    # weight threshold as "unused", iterating
        new_keep = mask & (_blunder_weight(ev.resid_sigmas(state)) > SOFT_W_USED)  # (a refit can
        if int(new_keep.sum()) < 6 or np.array_equal(new_keep, keep):              # reveal more)
            break
        state, result = ev.fit(state, new_keep, thin=False)
        keep = new_keep
    return state, result, keep


def _score(ev, state, mask, allow_unbound=False):
    """find_orb evaluate_initial_orbit analogue: weighted RMS (sigmas) plus a
    big penalty for an unphysical orbit, so a blown-up extension scores terribly.
    `allow_unbound`: see attempt_extensions -- an arc shown to admit no bound orbit."""
    a, e, q = quick_aeq(state)
    pen = unphysical_penalty(a, e, q, allow_unbound=allow_unbound)   # unbound / sun-plunging / absurd-a
    idx = np.flatnonzero(mask)
    dra, ddec = ev.resid_subset(state, mask)          # only included obs, not full arc
    s = np.hypot(_clamp(dra) / ev.sig_ra[idx], _clamp(ddec) / ev.sig_dec[idx])
    return float(np.sqrt(np.mean(s**2))) + pen


# --- extend_orbit_solution() ---------------------------------------------------
def extend_solution(ev, mask, state, limit=RESID_LIMIT, max_sigma=MAX_SIGMA,
                    time_limit=ARC_TIME_LIMIT):
    """Toggle on observations adjacent to the current arc, per find_orb. Residuals
    are evaluated at `state` (the current best orbit). Mutates `mask` in place;
    returns the number of obs added (0 if none / arc would exceed time_limit)."""
    t = ev.times_tdb
    rs = ev.resid_sigmas(state)
    inc = np.flatnonzero(mask)
    if len(inc) == 0:
        return 0
    first_idx, last_idx = int(inc[0]), int(inc[-1])
    jd_low  = t[first_idx]*2.0 - t[last_idx]      # one-arc-length windows
    jd_high = t[last_idx]*2.0 - t[first_idx]
    n_added = 0

    # --- extend backward: greedy adjacent (within `limit`), then "safe" fill ---
    i = first_idx - 1
    while i >= 0 and rs[i] < limit and (t[last_idx] - t[i]) < time_limit:
        mask[i] = True; n_added += 1; i -= 1
    while i >= 0 and t[i] > jd_low:
        if rs[i] < max_sigma:
            mask[i] = True; n_added += 1
        i -= 1

    # --- extend forward: symmetric ---
    j = last_idx + 1
    while j < ev.n and rs[j] < limit and (t[j] - t[first_idx]) < time_limit:
        mask[j] = True; n_added += 1; j += 1
    while j < ev.n and t[j] < jd_high:
        if rs[j] < max_sigma:
            mask[j] = True; n_added += 1
        j += 1

    # --- if nothing qualified, extend by exactly ONE obs (smaller time-jump) ---
    if n_added == 0:
        direction = 0
        if first_idx == 0:
            if last_idx < ev.n - 1: direction = 1
        elif last_idx == ev.n - 1:
            direction = -1
        else:
            before = t[first_idx] - t[first_idx - 1]
            after  = t[last_idx + 1] - t[last_idx]
            direction = 1 if before > after else -1
        if direction == -1 and (t[last_idx] - t[first_idx - 1]) < time_limit:
            mask[first_idx - 1] = True; n_added = 1
        elif direction == 1 and (t[last_idx + 1] - t[first_idx]) < time_limit:
            mask[last_idx + 1] = True; n_added = 1
    return n_added


def _relax_overconfident_obscodes(ev, state, mask, verbose=False):
    """find_orb sigma.txt philosophy: relax (loosen) the assigned sigma for any obscode that
    is over-confident for this object -- its stated sigma is tighter than find_orb's sigma.txt
    default AND that over-tight sigma is costing it good obs. The whole obscode's sigma
    (rejection AND fit weighting) reverts to that default, so its good sub-arcsecond obs stop
    reading as gross outliers while a real blunder (large in arcsec) stays one. Two signatures
    trigger it, both requiring SMALL raw residuals (< RELAX_MAX_RAW) so a genuinely discordant
    code is never masked:
      * INCLUDED-side: the code's kept obs sit at a high median sigma-residual
        (> RELAX_TRIGGER_SIG) -- they survived but drag the fit; and
      * EXCLUDED-side: more than RELAX_REJECT_FRAC of the code's obs were REJECTED yet those
        rejected obs have small raw residuals -- they were clipped ONLY because the stated sigma
        is too tight (e.g. X05 reporting 0.1" against a 0.5" default). This is invisible to the
        included-side test (the obs are already out of the mask), so an over-confident survey
        can lose a whole arc to rejection; relaxing lets the reject/redeem round below reclaim
        them. 2025 KR274 = 2025 PY146 links 28/55 -> 49/55 through this path.
    Returns True if anything was relaxed (caller should refit)."""
    if ev.obscodes is None or not mask.any():
        return False
    rs = ev.resid_sigmas(state)
    dra, ddec = ev.resid_arcsec(state); raw = np.hypot(dra, ddec)
    default_all = np.where(ev.times_tdb < JD_1993, SIGMA_TXT_OLD, SIGMA_TXT_MODERN)
    relaxed = []
    for code in np.unique(ev.obscodes):
        allsel = ev.obscodes == code
        incl = allsel & mask
        excl = allsel & ~mask
        default = float(default_all[allsel].min())
        # INCLUDED-side: kept obs at high sigma-residual but small raw residual, over-tight sigma.
        over_incl = (int(incl.sum()) >= RELAX_MIN_OBS
                     and float(np.median(rs[incl])) > RELAX_TRIGGER_SIG
                     and float(np.median(raw[incl])) < RELAX_MAX_RAW
                     and float(np.median(ev.sig_ra[incl])) < default)
        # EXCLUDED-side: a meaningful fraction of the code's obs rejected, yet the rejected obs
        # have SMALL raw residuals (clipped only for the over-tight sigma), sigma tighter than default.
        # "Enough to judge" is measured on the code's TOTAL obs (over-confidence is a property of the
        # station's reported sigmas), NOT on the rejected count -- else a small over-confident night
        # split across the mask (e.g. D29's 3-obs night in 2013 CK176 = 2026 DO7: 1 kept, 2 clipped at
        # 4-5σ but only 0.5-0.9" raw) never reaches RELAX_MIN_OBS on the excluded side and loses good
        # obs. The small-raw-residual guard still keeps a genuine blunder (B74 at ~19") from relaxing.
        over_excl = (int(allsel.sum()) >= RELAX_MIN_OBS
                     and int(excl.sum()) > RELAX_REJECT_FRAC * int(allsel.sum())
                     and float(np.median(raw[excl])) < RELAX_MAX_RAW
                     and float(np.median(ev.sig_ra[excl])) < default)
        if over_incl or over_excl:
            d = default_all[allsel]
            ev.sig_ra[allsel] = np.maximum(ev.sig_ra[allsel], d)
            ev.sig_dec[allsel] = np.maximum(ev.sig_dec[allsel], d)
            ev.fit_sig_ra[allsel] = np.maximum(ev.fit_sig_ra[allsel], d)
            ev.fit_sig_dec[allsel] = np.maximum(ev.fit_sig_dec[allsel], d)
            med = float(np.median(rs[incl])) if incl.any() else float(np.median(rs[excl]))
            relaxed.append((str(code), int(allsel.sum()), med))
    if relaxed:
        ev._resid_key = ev._sub_key = None        # sigmas changed -> drop stale memo
        if verbose:
            for code, n, ms in relaxed:
                print(f"   relax obscode {code} ({n} obs, median {ms:.1f}σ) "
                      f"-> sigma.txt default")
    return bool(relaxed)


# --- auto_reject_obs() : bidirectional, hysteresis, evaporation guards ----------
def _included_span(ev, mask):
    """Time span (days) of the currently-included obs -- the evaporation-guard
    baseline. Multi-round rejection loops freeze this ONCE before round 1 and pass
    it down, so successive rounds can't each clip another 'legal' 50% of whatever
    survived the previous round (find_orb's auto_reject_obs measures every round
    against the same fixed obs[0]..obs[n-1] span of the arc under rejection)."""
    inc = np.flatnonzero(mask)
    return float(ev.times_tdb[inc[-1]] - ev.times_tdb[inc[0]]) if len(inc) else 0.0


def auto_reject(ev, mask, state, inner, outer, rs=None, orig_span=None, orig_n=None):
    """Within the current arc: reject included obs with residual > `outer`, and
    redeem excluded obs with residual < `inner` (inner < outer => hysteresis).
    Reverts (returns -1) if it would shrink the arc below 50% of `orig_span`
    (default: the span of the obs included at entry; multi-round callers pass the
    span frozen before their FIRST round) or drop below 2/3 of `orig_n` obs. Returns
    the number of toggles (>0), 0, or -1 on guard failure. `rs` (per-obs σ
    residuals) may be passed to skip the full-arc sweep when the state is
    unchanged across a rejection-limit ramp.

    `orig_n` defaults to the evaluator's WHOLE obs count (`ev.n`), the historical
    behaviour every golden is baked against: on a whole-arc fit that is the arc. A
    caller whose mask is a deliberate SUBSET of the evaluator -- one apparition of a
    comet kept on the full record, an apparition-wise extension -- passes the count
    it entered with, because measured against ev.n such a mask already sits under
    the 2/3 bar and the guard fires on every ramp step, so nothing is ever rejected
    (21P's 4197-of-7434 apparition kept every one of its obs at 1.84")."""
    if rs is None:
        rs = ev.resid_sigmas(state)
    orig = mask.copy()
    if orig_span is None:
        orig_span = _included_span(ev, orig)
    if orig_n is None:
        orig_n = ev.n
    n_changed = 0
    for i in range(ev.n):
        if not orig[i] and rs[i] < inner:        # redeem
            mask[i] = True;  n_changed += 1
        elif orig[i] and rs[i] > outer:          # reject
            mask[i] = False; n_changed += 1
    inc = np.flatnonzero(mask)
    n_found = len(inc)
    final_span = (ev.times_tdb[inc[-1]] - ev.times_tdb[inc[0]]) if n_found else 0.0
    if final_span < orig_span*0.5 or n_found < orig_n*2//3:   # evaporation guard
        mask[:] = orig
        return -1
    return n_changed


def auto_reject_within_arc(ev, mask, state, orig_span=None, orig_n=None):
    """Ramp the rejection limit (3σ ×1.5 …) until a round succeeds or we pass the
    cap — find_orb's auto_reject_obs_within_arc(). The state is fixed across the ramp,
    so the per-obs residuals are computed ONCE (the full-arc sweep is the dominant
    cost on long, dense arcs). `orig_span` (see _included_span) lets a multi-round
    caller pin the evaporation-guard baseline to the arc as it stood before its
    FIRST round."""
    rs = ev.resid_sigmas(state)
    limit = REJECT_LIMIT
    rval = -1
    while limit < REJECT_CAP:
        rval = auto_reject(ev, mask, state, limit, limit*1.1, rs=rs, orig_span=orig_span,
                           orig_n=orig_n)
        if rval >= 0:
            break
        limit *= 1.5
    return rval


ACTIVATE_BAIL_SIGMA = 120.0 # if a joint activation fit is worse than this, it didn't
                           # converge (period-extrapolation failure) -> skip rejection
                           # rounds and revert (speed). Must stay ABOVE the level of a
                           # salvageable activation (a few contaminant obs can put an
                           # otherwise-good apparition at ~40 sigma; rejection fixes it).
ACTIVATE_NFEV = 40         # LM iteration cap for activation trials (converges fast or never)


def _clusters(times_tdb):
    """Index ranges [(start,end), ...] of apparition clusters (gaps > OPP_GAP_DAYS)."""
    t = np.asarray(times_tdb)
    breaks = [0] + [i+1 for i in range(len(t)-1) if t[i+1]-t[i] > OPP_GAP_DAYS] + [len(t)]
    return [(breaks[b], breaks[b+1]-1) for b in range(len(breaks)-1)]


def _activate_clusters(ev, best_state, best_mask, history, verbose):
    """Adjacent extension can't bridge to an apparition years away (a single short
    arc doesn't pin the period). So try ACTIVATING each not-yet-included apparition
    WHOLESALE and refitting with outlier rejection: the long baseline of the joint
    fit pins the period and snaps to the linking orbit, and the rejection then
    exposes any genuinely-discordant obs (e.g. a mislinked night). Returns the
    possibly-extended (best_state, best_mask)."""
    clusters = _clusters(ev.times_tdb)
    improved = True
    while improved:
        improved = False
        for (cs, ce) in clusters:
            if best_mask[cs:ce+1].any():
                continue        # already included/processed (its outliers are
                                # handled by the joint rejection passes)
            if _ident_shortcircuit(ev, best_mask, (cs, ce)):
                # a period-known foreign designation in an explicit multi-designation linkage
                # request: it can't honestly join a period-locked anchor (see _ident_shortcircuit),
                # so don't spend a joint refit trying to activate it.
                continue
            trial = best_mask.copy()
            trial[cs:ce+1] = True                 # activate the whole apparition
            state, _ = ev.fit(best_state, trial, max_nfev=ACTIVATE_NFEV)  # joint refit
            # Early-bail: if the joint LM didn't snap (residual still huge), the
            # apparition can't be bridged from this orbit -- skip the (expensive)
            # rejection rounds and revert. Saves ~4 fits per doomed activation,
            # which is what made many-apparition --force time out.
            if ev.rms_norm(state, trial) < ACTIVATE_BAIL_SIGMA:
                span0 = _included_span(ev, trial)
                for _ in range(4):                # reject discordant obs, then refit
                    if auto_reject_within_arc(ev, trial, state, orig_span=span0) <= 0:
                        break
                    state, _ = ev.fit(state, trial, max_nfev=ACTIVATE_NFEV)
            survivors = int(trial[cs:ce+1].sum())
            ncl = ce - cs + 1
            rms_n = ev.rms_norm(state, trial)
            # accept only if the activation adds NET-NEW obs overall (else it's a
            # no-op re-activation of an already-processed apparition -> infinite loop)
            net_new = int(trial.sum()) - int(best_mask.sum())
            ok = survivors >= max(3, ncl//2) and rms_n < ACTIVATE_KEEP_NORM and net_new > 0
            if verbose:
                span = ev.times_tdb[ce] - ev.times_tdb[cs]
                print(f"   activate apparition obs[{cs}..{ce}] ({ncl} obs, "
                      f"{span:.0f}d): survivors {survivors}/{ncl}, +{net_new} net, "
                      f"total {int(trial.sum())} obs, rms={rms_n:.2f}σ -> "
                      f"{'KEEP' if ok else 'revert'}")
            history.append(_hsnap({"stage": f"activate[{cs}:{ce}]", "n_obs": int(trial.sum()),
                            "rms": ev.rms_arcsec(state, trial),
                            "score": _score(ev, state, trial), "kept": bool(ok)}, state, trial))
            if ok:
                best_state, best_mask = state.copy(), trial.copy()
                improved = True
    return best_state, best_mask


# --- incremental nearest-first apparition linker -------------------------------
# When wholesale activation can't bridge an apparition (a NEO whose orbit error
# accumulates over many periods / Earth close approaches into a wrong-revolution
# offset, so a joint fit diverges), grow the arc ONE apparition at a time, nearest
# gap first, gluing each in with a weight continuation (homotopy): the new apparition starts
# near-weightless and its weight rises slowly while the anchor stays fixed. If a step
# starts out-of-basin (residuals diverge), a small covariance clone-swarm finds an
# in-basin orbit first. This links e.g. Bennu (9/10 apparitions, 386 obs @ 0.36").
LINK_BASIN = 40.0       # arcsec: below this a weight continuation can pull an apparition in
MULTIREV_MIN_ANCHOR_DAYS = 4.0  # the Tier-2 period sweep only works from an anchor whose arc is
                        # long enough to constrain the period; a single short tracklet (e.g. a
                        # 2-day seed) can never glue, so skip the expensive sweep there entirely.
                        # (VM6's winning bridge anchors on an 11-day arc; UK199's doomed attempts
                        # were 2-day seeds -- this skips them, dropping UK199 ~56s -> ~13s.)
MULTIREV_MAX_ANCHOR_RMS = 1.0   # ...and the anchor must be WELL FIT: a garbage anchor (e.g. a
                        # bad IOD seed at tens of arcsec) can't be the precise base for a
                        # period sweep, so the dense swarm would just burn. (VM6's 2023 seed at
                        # 86" -> skip; only its clean 0.34" 2001 anchor runs Tier-2.)
CHEAP_SWARM_MAX_ANCHOR_RMS = 5.0  # a TRULY garbage anchor (well above any legit linkable arc,
                        # well below VM6's 86" bad seed) makes even the cheap Tier-1 swarm
                        # hopeless -- its covariance is too wide to land near a basin. Skip the
                        # swarm entirely above this, saving ~2.4s/attempt on dead-end seeds.
LINK_MULTIREV = 50000.0 # arcsec (~14°): only a gap this wide is a genuine multi-revolution
                        # ambiguity worth the EXPENSIVE period-sweep fallback. A smaller gap the
                        # cheap swarm couldn't bridge is just left unlinked (as it always was) --
                        # this keeps the dense swarm off the routine objects entirely.
LINK_CONTINUATION_SIGMA_MULT = [1e4, 1e3, 1e2, 1e1, 1.0]   # new-app σ multiplier high→low (homotopy)
CONTINUATION_NFEV = 8           # LM-iteration cap on the intermediate (non-final) continuation steps
IDENT_MULTI_OPP = 2     # opposition count at/above which a designation's own period (and so its sky
                        # position) is INDEPENDENTLY determined. Below it (a lone single-opposition
                        # tracklet) the period is unknown, so a large offset to another arc may be a
                        # genuine revolution ambiguity worth bridging -- such a candidate is NEVER
                        # short-circuited. See _ident_shortcircuit.
IDENT_LOCK_OPP = 4      # ...and at/above this an orbit is period-LOCKED: two arcs that are EACH this
                        # well-determined are virtually never the same body. In an explicit multi-
                        # designation linkage request (an ident-check: "is K14EK9X = K21Q75W?"), a
                        # period-locked arc sitting a hard multi-revolution jump off a period-KNOWN
                        # foreign designation is a real mismatch, not a revolution ambiguity (which
                        # only afflicts a short / single-opposition anchor). No honest bridge can glue
                        # them, so the expensive Tier-2 multi-rev sweep is wasted -- skip straight to
                        # "unlinked", the same verdict the grind reaches after seconds-to-minutes.
IDENT_LOCK_OPP_GATED = 3  # ...a LOWER lock threshold used on exactly ONE path: the OFFSET-GATED
                        # per-apparition guard inside _bridge_opposition, reached only once the offset
                        # already exceeds LINK_BASIN (the link is demonstrably NOT in-basin) and the
                        # cheap in-basin homotopy has had its chance. So lowering the bar to 3
                        # oppositions cannot skip a genuine identification -- a real match links
                        # in-basin and never reaches that branch -- it only spares the doomed Tier-2
                        # grind on a 3-vs-2-opp cross-designation non-match whose maturity tops out at
                        # 3 because its anchor is split across two designations (K14EE7R=2014 ER147
                        # [2 opp] + its 2025 arc, vs the foreign K21S69J=2021 SJ69 [3 opp] -- ~144 s
                        # a-secant grind -> ~0 s).
                        # EVERYTHING ELSE keeps the strict IDENT_LOCK_OPP (4): the paths that drop a
                        # cluster BEFORE measuring its offset (_link_oppositions, _activate_clusters),
                        # where a genuine in-basin match could otherwise be discarded unseen, AND the
                        # whole-fit gate fit_object._ident_nonmatch, which was deliberately reverted to
                        # 4 (firing it at 3 froze a poor initial anchor -- see its docstring). Do NOT
                        # assume this constant moves that gate.
LINK_ANCHOR_TOL = 0.5   # anchor rms must stay within this of its pre-step value (")
LINK_NEW_TOL = 3.0      # a newly linked apparition's post-glue rms (") must reach this to ACCEPT
                        # the bridge. This is only a real-vs-wrong-revolution discriminator, and
                        # that margin is enormous: a correctly-bridged apparition glues to a few
                        # arcsec (a couple of raw outliers in its 30+ obs inflate the rms; the
                        # downstream reject/redeem then cleans them to sub-arcsec), while a WRONG
                        # revolution glues to ~thousands-to-1e4". The old 1.5" gate sat inside that
                        # gap and rejected REAL links by a hair whenever the converged rms landed at
                        # ~2" — the multi-rev bridge's ~30%-coin-flip fragility (2023 VM6: the true
                        # link converged at 2.26" yet was thrown away). 3" keeps the wrong-revolution
                        # rejection intact (4 orders of magnitude away) while accepting real links.
SWARM_N = 50            # clones per swarm generation
SWARM_GENS = 5
SWARM_N_FAR = 120       # ...denser swarm when bridging a multi-revolution gap (off > LINK_MULTIREV):
SWARM_GENS_FAR = 6      #    we must populate the true-period basin well enough that one clone glues
LINK_BRIDGE_MAX_FAILS = 2  # consecutive per-apparition Tier-2 rejections before we stop bridging
#                          from this anchor -- see the note in _link_oppositions. They all test the
#                          same anchor orbit, nearest-first, so the third is not new information.
SWARM_CANDIDATES = 8    # closest candidate clones handed to the bridge LM (it keeps whichever
                        # yields a glued joint fit; covers several revolution-count basins)
BRIDGE_CAND_MAX_NFEV = 100  # LM budget per candidate clone in the period sweep. A RIGHT-period clone
                        # snaps to its glued minimum fast: across the whole suite every candidate that
                        # actually glues converges in <=64 iters (usually <20; 2023 VM6's cand#6 is the
                        # worst at 64). A WRONG-period clone can't glue (its anchor balloons) and just
                        # wanders to the cap -- at the old 200 those doomed grinders cost ~3s/candidate
                        # of pure waste (2026 NA1: two clones grinding to 184/200 = ~6s thrown away
                        # before the fallback slides link it, or force-mode re-solves). 100 sits in the
                        # clean valley between the two populations: it preserves every real glue with
                        # comfortable headroom while halving the doomed-grinder cost.
# a-constraint secant fallback (when the coarse sweep can't resolve the period from a short anchor)
A_PENALTY_W = 1e4       # weight pinning osculating a to the target in the constrained anchor fit
SECANT_DA = 0.005       # au: initial a-step to seed the secant (second bracket point)
SECANT_A_SPAN = 0.8     # au: clip the secant's a within +/- this of nominal (stay near-revolution)
SECANT_ITERS = 14       # max secant iterations to converge the far RA residual to zero. A real
                        # link converges in <=9 (1993 TZ); beyond ~14 it's a DOOMED link (e.g. the
                        # initial fit bridging from a poorly-constrained anchor that the recovery
                        # re-anchors anyway) just burning constrained fits -- bail and let recovery
                        # retry from the right anchor. (24 -> 14: TE4's doomed initial bridge cheaper.)
SECANT_RA_TOL = 0.5     # arcsec: far-apparition median RA residual small enough to declare the
                        # period found and hand off to the full 6-dof joint LM
SECANT_FAR_NORM_TOL = 3.0  # accept the glued far apparition at this NORMALIZED rms (per-coord
                        # sigmas): looser than the sweep's absolute LINK_NEW_TOL because the
                        # RA->0 + glued-anchor convergence is strong independent evidence the
                        # revolution is right, and an old single-night apparition's obs run loose.


def _off_sub(ev, state, mask):
    """Per-obs total offset (arcsec) over `mask`, with non-finite predictions CLAMPED
    to _BIG_RESID — the same divergence policy as the reject loop (rms_arcsec/_clamp).
    Previously _rms_sub/_med_off used nanmean/nanmedian, which SILENTLY DROPPED obs the
    orbit couldn't even predict: a bridge anchor could read as a 'good' RMS while half
    its predictions were NaN. Clamping makes the accept tests and the reject loop agree
    on the central 'does this orbit fit?' question (audit #6) — an obs you can't predict
    is a gross misfit, not a free pass. The median in _med_off stays robust (only blows
    up once a MAJORITY diverge, i.e. exactly when the orbit should be rejected)."""
    dra, ddec = ev.resid_subset(state, mask)
    return _clamp(np.hypot(dra, ddec))


def _rms_sub(ev, state, mask):
    return float(np.sqrt(np.mean(_off_sub(ev, state, mask)**2)))


def _med_off(ev, state, mask):
    return float(np.median(_off_sub(ev, state, mask)))


def _med_off_twobody(ev, state, idx):
    """Cheap two-body median offset (arcsec) of `state` at obs `idx` — used ONLY to RANK
    the multi-revolution period sweep, never to fit. The revolution a clone lands in is
    set by its osculating period (a two-body quantity); planetary perturbation only shifts
    it WITHIN a revolution, which the full-ASSIST candidate LM then absorbs. So a ~free
    Kepler propagation ranks the 720-clone sweep that a full integration made cost ~5s.
    state is heliocentric-equatorial (au, au/day); Robs is heliocentric-equatorial."""
    offs = np.empty(len(idx))
    for k, i in enumerate(idx):
        st = kepler_propagate(state, float(ev.times_tdb[i] - ev.t0))
        if not np.all(np.isfinite(st)):
            # this clone cannot be propagated to this epoch (see twobody.kepler_propagate:
            # a NaN state is its explicit failure contract). Score the OBS as missing rather
            # than letting NaNs run through the arithmetic below.
            offs[k] = np.nan
            continue
        r, v, robs = st[:3], st[3:], ev.Robs[i]
        rel = r - robs
        for _ in range(2):                          # light-time: linear back-step
            rel = (r - v * (np.linalg.norm(rel) / C_AU_DAY)) - robs
        ra = np.degrees(np.arctan2(rel[1], rel[0])) % 360
        dec = np.degrees(np.arcsin(rel[2] / np.linalg.norm(rel)))
        dra = ((ra - ev.ra[i] + 180) % 360 - 180) * ev.cosd[i] * 3600.0
        offs[k] = np.hypot(dra, (dec - ev.dec[i]) * 3600.0)
    # a clone that failed at EVERY epoch has no offset to report -- say so directly (the
    # caller drops non-finite offsets). np.nanmedian of an all-NaN slice returns NaN too,
    # but warns while doing it, which is noise at the user's terminal for a clone the
    # period sweep was always going to discard.
    if not np.any(np.isfinite(offs)):
        return float("nan")
    return float(np.nanmedian(offs))


SWARM_SCORE_OBS = 12    # rank clones against this many subsampled obs of the new
                        # apparition (it's only a ranking — full obs is wasteful, and
                        # the per-obs integrate calls dominate the swarm cost)
SWARM_BAIL_GENS = 2     # bail if no generation improves the offset this many times
                        # in a row (a hopeless far apparition won't suddenly converge)


def _swarm_to_basin(ev, state, linked, nbm, verbose):
    """Clone within the linked-arc covariance, keep whichever lands closest at the new
    apparition, respawn from the spread of the top few, widening if all clones diverge.
    Returns an in-basin state or None. Clones are ranked against a SUBSAMPLE of the new
    apparition and the search bails early when it stops improving.

    Uses a FIXED-seed RNG so a fit is fully reproducible (the orbit must not depend on
    random draws — and the regression archive needs deterministic goldens)."""
    rng = np.random.RandomState(0xB0BA)
    nb_idx = np.flatnonzero(nbm)
    sub = nb_idx[np.linspace(0, len(nb_idx) - 1, min(SWARM_SCORE_OBS, len(nb_idx))
                             ).round().astype(int)]
    subm = np.zeros(ev.n, bool); subm[sub] = True
    _, result = ev.fit(state, linked)
    J = np.asarray(result.jac); dof = max(1, J.shape[0] - 6)
    s2 = float(np.sum(result.fun**2)) / dof
    try:
        C = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        C = s2 * np.linalg.pinv(J.T @ J)
    best, best_off, scale, stall = result.x, _med_off(ev, result.x, subm), 1.0, 0
    for g in range(SWARM_GENS):
        try:
            clones = rng.multivariate_normal(best, C * scale, SWARM_N)
        except Exception:
            clones = best + rng.randn(SWARM_N, 6) * np.sqrt(np.abs(np.diag(C))) * scale
        scored = [(o, c) for c in clones for o in [_med_off(ev, c, subm)] if np.isfinite(o)]
        if not scored:
            scale *= 4; stall += 1
            if stall >= SWARM_BAIL_GENS:
                break
            continue
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_off:
            best, best_off, stall = scored[0][1], scored[0][0], 0
        else:
            stall += 1
        if best_off < LINK_BASIN:
            return best
        if stall >= SWARM_BAIL_GENS:
            break
        topK = np.array([c for _, c in scored[:max(3, SWARM_N // 10)]])
        if len(topK) >= 3:
            C = np.cov(topK.T) + np.eye(6) * 1e-18; scale = 1.0
        else:
            scale *= 4
    return best if best_off < LINK_BASIN else None


def _swarm_far_candidates(ev, state, linked, nbm):
    """EXPENSIVE multi-revolution fallback (only when the cheap swarm can't reach basin):
    sweep the period to bridge a gap where only the NOMINAL orbit is ~180° off. The poorly-
    constrained direction of the anchor covariance C IS the period; sampling C with a
    GEOMETRICALLY WIDENING scale walks the orbit through revolution after revolution at the
    far apparition, landing a clone near each revolution-count solution. Returns the closest
    candidate clones (within LM range) for the caller to joint-LM and glue-test, or None.
    Fixed-seed RNG for reproducibility."""
    rng = np.random.RandomState(0xB0BA)
    nb_idx = np.flatnonzero(nbm)
    sub = nb_idx[np.linspace(0, len(nb_idx) - 1, min(SWARM_SCORE_OBS, len(nb_idx))
                             ).round().astype(int)]
    subm = np.zeros(ev.n, bool); subm[sub] = True
    _, result = ev.fit(state, linked)
    J = np.asarray(result.jac); dof = max(1, J.shape[0] - 6)
    s2 = float(np.sum(result.fun**2)) / dof
    try:
        C = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        C = s2 * np.linalg.pinv(J.T @ J)
    nominal = result.x
    pool, scale = [], 1.0
    for g in range(SWARM_GENS_FAR):
        try:
            clones = rng.multivariate_normal(nominal, C * scale, SWARM_N_FAR)
        except Exception:
            clones = nominal + rng.randn(SWARM_N_FAR, 6) * np.sqrt(np.abs(np.diag(C))) * scale
        for c in clones:
            o = _med_off_twobody(ev, c, sub)         # ~free Kepler ranking, not a fit
            if np.isfinite(o):
                pool.append((o, c))
        scale *= 3.0                                 # widen the period sweep each generation
    # two-body offsets run larger than the full-ASSIST capture radius (22 yr of unmodelled
    # perturbation drift), so DON'T hard-filter on offset at all -- just hand the LOWEST-offset
    # clones (the ones nearest the right revolution) to the caller's full-ASSIST glue-test.
    pool.sort(key=lambda x: x[0])
    return [c for _, c in pool[:SWARM_CANDIDATES]] or None


def _homotopy_glue(ev, state, full, newm):
    """Glue a new apparition (`newm`) onto the linked set while keeping the anchor fixed, by a
    weight homotopy: start the new apparition (near-)weightless so it can't yank the orbit, ramp
    its weight to full over LINK_CONTINUATION_SIGMA_MULT, refit at each step. Returns the glued state. (Shared by
    _bridge_opposition and _bridge_bulk.)"""
    base_ra, base_dec = ev.fit_sig_ra.copy(), ev.fit_sig_dec.copy()
    cur = state
    for mult in LINK_CONTINUATION_SIGMA_MULT:
        ev.fit_sig_ra = base_ra.copy(); ev.fit_sig_dec = base_dec.copy()
        ev.fit_sig_ra[newm] *= mult; ev.fit_sig_dec[newm] *= mult
        nxt, _ = ev.fit(cur, full, max_nfev=(400 if mult == 1.0 else CONTINUATION_NFEV))
        if np.all(np.isfinite(nxt)):
            cur = nxt
    ev.fit_sig_ra, ev.fit_sig_dec = base_ra, base_dec
    return cur


LOV_PIN_W = 1e3        # weight pinning the LOV coordinate (in sigma units) in the constrained fit


def fit_lov_pinned(ev, seed, mask, state0, u, sigma, k, w=LOV_PIN_W, max_nfev=200):
    """LM fit over `mask` with the orbit's coordinate ALONG the Line-of-Variations direction
    `u` (a unit 6-vector in state space) PINNED to k sigmas from `state0`, i.e.
    (s - state0).u = k*sigma, by a heavy penalty -- the other 5 d.o.f. are optimized. The
    result is the best-fit orbit AT that LOV offset: one nonlinear LOV variant (find_orb
    improve_along_lov, orb_fun2.cpp). The penalty is normalized by sigma so it is
    dimensionless (sigma units), matching the per-obs sigma-normalized residuals; w=1e3
    pins to ~1e-3 sigma. Returns the variant state (or seed on a non-finite solve)."""
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
    ra = ev.ra[idx]; dec = ev.dec[idx]; cosd = ev.cosd[idx]
    sra = ev.fit_sig_ra[idx]; sdec = ev.fit_sig_dec[idx]
    state0 = np.asarray(state0, float); u = np.asarray(u, float)

    if ev._predict_jac is not None:
        # Analytic Jacobian, as _fit_anchor_fixed_a: ~1 integration per LM iteration
        # instead of the 7 a numeric jac forces. The pin is LINEAR in s, so its row is
        # exactly w*u/sigma. This is the LOV refinement's / LOV variants' hot loop (one
        # pinned fit per picket), which the ITF search runs on every weak seed.
        cache = {}
        def _compute(s):
            key = s.tobytes(); hit = cache.get(key)
            if hit is None:
                cache.clear(); hit = cache[key] = ev._predict_jac(ev.base_sim, ev.t0, s, tt, Rs)
            return hit
        prow = w * u / sigma
        def resid_a(s):
            pred, _ = _compute(s)
            r = _radec_resid(pred, ra, dec, cosd, sra, sdec)
            return _clamp(np.append(r, w * (np.dot(s - state0, u) / sigma - k)))
        def jac_a(s):
            _, J = _compute(s)
            full = np.vstack([_radec_jac(J, cosd, sra, sdec), prow])
            return np.where(np.isfinite(full), full, 0.0)
        out = least_squares(resid_a, np.asarray(seed, float), jac=jac_a, method="lm",
                            xtol=1e-8, ftol=1e-8, max_nfev=max_nfev).x
        return out if np.all(np.isfinite(out)) else np.asarray(seed, float)

    def resid(s):
        pred = ev._predict(ev.base_sim, ev.t0, s, tt, Rs)
        d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
        r = np.column_stack([d * 3600.0 / sra, (pred[:, 1] - dec) * 3600.0 / sdec]).ravel()
        pen = w * (np.dot(s - state0, u) / sigma - k)        # LOV coordinate pinned to k*sigma
        return _clamp(np.append(r, pen))

    out = least_squares(resid, np.asarray(seed, float), method="lm",
                        xtol=1e-8, ftol=1e-8, max_nfev=max_nfev).x
    return out if np.all(np.isfinite(out)) else np.asarray(seed, float)


def _fit_anchor_fixed_a(ev, seed, mask, a_target, w=A_PENALTY_W):
    """LM fit over `mask` with the osculating semi-major axis PINNED to a_target by a heavy
    penalty residual w*(a(state) - a_target). For a SHORT anchor arc the period is the
    under-constrained direction, so pinning a and fitting the other 5 d.o.f. slides the orbit
    cleanly along that direction while staying glued to the anchor.

    Uses the analytic Jacobian when the backend supplies one -- exactly as _fit_capped_ecc
    does for its eccentricity barrier: the a-penalty row's gradient is finite-differenced
    through quick_aeq, which is INTEGRATION-FREE, so the whole fit costs ~1 integration per
    LM iteration instead of the ~7 a fully numeric jac forces. This is the multi-revolution
    bridge's hot loop (_bridge_far_by_a_secant calls it once per secant probe, and the
    recovery cascade re-runs the bridge from every re-anchor), so those 6 extra sweeps per
    step were the dominant cost of every genuine multi-rev linkage."""
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
    ra = ev.ra[idx]; dec = ev.dec[idx]; cosd = ev.cosd[idx]
    sra = ev.fit_sig_ra[idx]; sdec = ev.fit_sig_dec[idx]

    def _pen(s):
        a, _, _ = quick_aeq(s); a = a if np.isfinite(a) else 1e3
        return w * (a - a_target)

    if ev._predict_jac is not None:
        cache = {}
        def _compute(state):
            key = state.tobytes(); hit = cache.get(key)
            if hit is None:
                cache.clear(); hit = cache[key] = ev._predict_jac(ev.base_sim, ev.t0, state, tt, Rs)
            return hit
        def resid(state):
            pred, _ = _compute(state)
            r = _radec_resid(pred, ra, dec, cosd, sra, sdec)
            return _clamp(np.append(r, _pen(state)))
        def jac(state):
            _, J = _compute(state)
            out = _radec_jac(J, cosd, sra, sdec)            # (2m,6) scaled + finite-filtered
            prow = np.zeros(6)
            p0 = _pen(state)
            for k in range(6):                              # two-sided pin: always has a gradient
                h = 1e-6 * (abs(state[k]) + 1e-6)
                sp = state.copy(); sp[k] += h
                prow[k] = (_pen(sp) - p0) / h
            full = np.vstack([out, prow])
            return np.where(np.isfinite(full), full, 0.0)
        result = least_squares(resid, seed, jac=jac, method="lm",
                            xtol=1e-10, ftol=1e-10, max_nfev=300)
        return result.x

    def resid(s):                                           # no analytic jac: numeric fallback
        pred = ev._predict(ev.base_sim, ev.t0, s, tt, Rs)
        r = _radec_resid(pred, ra, dec, cosd, sra, sdec)
        return _clamp(np.append(r, _pen(s)))
    return least_squares(resid, seed, method="lm",
                         xtol=1e-10, ftol=1e-10, max_nfev=300).x


def _bridge_far_by_a_secant(ev, nominal, mask, nbm, anchor0, verbose=False):
    """Multi-revolution bridge FALLBACK (when the period sweep finds no glued candidate): for
    a far apparition whose period is barely under-constrained by a SHORT anchor, the sweep is
    too coarse, but the far apparition's median RA residual is a smooth function of the pinned
    semi-major axis a -- so root-find the a where it crosses zero (the right period) while the
    anchor stays glued, then joint-LM. A targeted secant (no grid):
    constrain a, step in the sign-reducing direction, interpolate to the zero, repeat.

    Accept on a NORMALIZED far-apparition rms: the RA->0 convergence with a glued anchor is
    itself strong evidence the revolution is right, and an old single-night apparition's obs are
    looser than the sweep path's absolute LINK_NEW_TOL. Returns a glued joint state or None."""
    full = mask | nbm
    a0, _, _ = quick_aeq(nominal)
    if not np.isfinite(a0) or a0 <= 0:
        return None

    def far_ra(a_t):
        st = _fit_anchor_fixed_a(ev, nominal, mask, a_t)
        dra, _ = ev.resid_subset(st, nbm)
        return float(np.median(dra)), st

    a_lo, (r_lo, _) = a0, far_ra(a0)
    a_hi, (r_hi, _) = a0 + SECANT_DA, far_ra(a0 + SECANT_DA)
    for _ in range(SECANT_ITERS):
        if r_hi == r_lo:
            return None
        a_new = float(np.clip(a_hi - r_hi * (a_hi - a_lo) / (r_hi - r_lo),
                              a0 - SECANT_A_SPAN, a0 + SECANT_A_SPAN))
        r_new, st_new = far_ra(a_new)
        if abs(r_new) < SECANT_RA_TOL:                  # far RA residual ~ 0 -> period found
            jl, _ = ev.fit(st_new, full, max_nfev=400)  # glue with all 6 d.o.f. free
            if (np.all(np.isfinite(jl))
                    and _rms_sub(ev, jl, mask) < anchor0 + LINK_ANCHOR_TOL
                    and ev.rms_norm(jl, nbm) < SECANT_FAR_NORM_TOL):
                if verbose:
                    print(f"   a-secant: a={a_new:.5f} -> anchor {anchor0:.3f}->"
                          f"{_rms_sub(ev, jl, mask):.3f}\" far_normrms={ev.rms_norm(jl, nbm):.2f}")
                return jl
            return None
        if (r_new > 0) == (r_hi > 0):                   # keep the sign-bracketing endpoints
            a_hi, r_hi = a_new, r_new
        else:
            a_lo, r_lo = a_new, r_new
    return None


# --- PHASE-SCAN multi-revolution bridge (exp-phase) ----------------------------------------------
# The period sweep / a-secant slide the orbit along the SHAPE (a) direction. But the real
# under-constrained direction across a multi-year gap is the ORBITAL PHASE: where the object sits
# in its orbit (mean anomaly / time of perihelion) at the far epoch. The anchor pins a (hence the
# period P) and the shape well, but a short anchor barely pins the phase, so the far apparition can
# mispredict by tens-to-hundreds of degrees while a is essentially right. Fix: keep the anchor's
# state but advance it by a time offset dt (a Kepler propagation by dt is exactly a phase slide of
# 2*pi*dt/P), scan dt over one full period, rank each shifted seed cheaply against the far
# apparition, then full-LM glue the best few -- one phase sample lands the far apparition near its
# true place and the joint LM locks it in. "Reel them in / get close": accept a generous far rms.
PHASE_STEPS = 16           # phase samples across [0, P)
PHASE_TOPK = 3             # homotopy-glue only the best-ranked few (cheap median-off ranking first)
# (An a/period scan wrapping the phase scan was tried and dropped: it never reached the one genuine
#  wrong-basin strand -- 2003 LW2, where no (a, phase) pair gets the far apparition below ~39" median
#  without ballooning the anchor -- and it reordered the ranking enough to bump a real link off the
#  top-K. Scan phase at the anchor's a; the joint LM frees a once a sample is in basin.)
PHASE_FAR_NORM_TOL = 12.0  # "get-close" far-apparition normalized rms to accept a phase link. A
                           # WRONG revolution glues to thousands-to-1e5 normalized (see the rejected
                           # samples), so even 12 is four orders of magnitude clear of a false link;
                           # the bound just admits a real, gravity-only-elevated high-e NEO apparition.
PHASE_SCORE_OBS = 8        # rank each phase sample against this many subsampled far-apparition obs
PHASE_RELAX_NORM = 1.5     # after a confirmed link, relax far-app sigmas so its coherent (real but
                           # gravity-only-elevated) offset reads ~this many sigma -- keeps the whole
                           # apparition through the reject loop. The relaxation is SCOPED and
                           # REVERSIBLE (Evaluator.relax_link_opposition): the nongrav refit restores
                           # full weight and fits the offset (restore_link_relaxation).


def _bridge_far_by_phase(ev, state, mask, nbm, anchor0, verbose=False):
    """Phase scan at fixed a: refit the anchor, then slide the anchor state's orbital phase by a
    time offset dt across one full period (kepler_propagate by dt == advance mean anomaly). The
    period P=a^1.5*yr is ~right (a is the well-determined direction), so one phase sample slides
    the object onto the far apparition. Rank each shifted seed by its ACTUAL (ASSIST-integrated)
    far median offset -- a two-body offset is useless here, the propagation diverges over the
    multi-revolution horizon -- then homotopy-glue the best few (the new apparition starts
    near-weightless so it can't yank the anchor) and keep the first that glues the anchor and
    reels the far apparition within a generous "get-close" bound. Returns a glued state or None.

    GATED to TWO-apparition objects: the premise is "keep the anchor's a, slide the phase to reach
    THE one far apparition". With 3+ apparitions the period is over-constrained (the normal
    swarm/secant bridge already links them) and it's ambiguous which far apparition to phase onto,
    so the scan is pure wasted ASSIST integration there (2008 UK199: 2 useless calls, ~12 s)."""
    if len(_clusters(ev.times_tdb)) > 2:
        return None
    full = mask | nbm
    _, result = ev.fit(state, mask)
    seed = result.x
    if not np.all(np.isfinite(seed)):
        return None
    a, _, _ = quick_aeq(seed)
    if not np.isfinite(a) or a <= 0:
        return None
    # Subsample the far apparition for the ranking (the per-obs ASSIST integrate dominates cost).
    nb_idx = np.flatnonzero(nbm)
    sub = nb_idx[np.linspace(0, len(nb_idx) - 1, min(PHASE_SCORE_OBS, len(nb_idx))
                             ).round().astype(int)]
    subm = np.zeros(ev.n, bool); subm[sub] = True
    # Slide the orbital phase across one full period at the anchor's a (kepler_propagate by dt
    # advances the mean anomaly by 2*pi*dt/P); rank each shifted seed by its ACTUAL integrated far
    # offset (a two-body offset is useless here -- the propagation diverges over the multi-rev gap).
    P = (a ** 1.5) * 365.25                       # period in days (a in au)
    samples = []
    for k in range(PHASE_STEPS):
        dt = P * k / PHASE_STEPS
        sh = kepler_propagate(seed, dt)
        if not np.all(np.isfinite(sh)):
            continue
        off = _med_off(ev, sh, subm)
        if np.isfinite(off):
            samples.append((off, sh, dt, P))
    if not samples:
        return None
    samples.sort(key=lambda x: x[0])
    for off, sh, dt, P in samples[:PHASE_TOPK]:
        jl = _homotopy_glue(ev, sh, full, nbm)
        if not np.all(np.isfinite(jl)):
            continue
        jl2, _ = ev.fit(jl, full, max_nfev=400)  # a couple of extra full-weight joint iterations
        if np.all(np.isfinite(jl2)):             # tighten the far apparition once it's in basin
            jl = jl2
        a_rms = _rms_sub(ev, jl, mask)
        far_norm = ev.rms_norm(jl, nbm)
        if a_rms < anchor0 + LINK_ANCHOR_TOL and far_norm < PHASE_FAR_NORM_TOL:
            # CONFIRMED multi-revolution link: the anchor stays glued AND the far apparition's
            # whole obs cloud collapses to a coherent, modest offset (a wrong revolution glues to
            # thousands of arcsec, not single digits). The far residual is elevated only because
            # this is gravity-only on a high-e NEO whose exactness wants the nongrav routine --
            # "reel them in / get close". To stop the downstream reject loop from gutting the
            # whole (real) apparition as outliers, relax its fit/reject sigmas so its coherent
            # offset reads ~target sigma -- the obs stay LINKED; the nongrav refit then restores
            # full weight and fits the offset (Evaluator.relax_link_opposition / restore).
            ev.relax_link_opposition(nbm, far_norm, PHASE_RELAX_NORM)
            if verbose:
                print(f"   phase-scan: dt={dt:.0f}d (P={P:.0f}d) far_medoff={off:.0f}\" -> "
                      f"anchor {anchor0:.3f}->{a_rms:.3f}\" far_normrms={far_norm:.2f}"
                      f"{f' (relaxed x{far_norm/PHASE_RELAX_NORM:.1f})' if far_norm>PHASE_RELAX_NORM else ''}")
            return jl
    return None


# --- LINE-OF-VARIATIONS multi-revolution bridge (find_orb's improve_along_lov) -------------------
# The period sweep slides along the SHAPE (a), the phase scan along the orbital PHASE. Both are
# SINGLE axes, and both fail on the hardest 2-opposition pure-gravity strand (2003 LW2): there, a
# short anchor pins the plane well INDIVIDUALLY, but it is consistent with a small RANGE of a, and
# pinning a (or sliding phase) to seed the bridge distorts the anchor's plane by a hair -- which,
# multiplied over 7+ revolutions across an 18-yr gap, throws the far apparition off CROSS-TRACK,
# a direction no (a)- or (phase)-scan can correct. The fix is to slide along the anchor covariance's
# true least-constrained direction -- the Line of Variations -- which is the CORRELATED combination
# of (a, plane, phase) that keeps the anchor's chi^2 flat. Walking it explores exactly the family of
# orbits the anchor permits, and one member brings the far apparition in on BOTH sky axes at once.
# Ported (logic only) from find_orb orb_fun2.cpp::improve_along_lov: a dense 1-D scan along the LOV
# for the score minimum. find_orb scans 10000 divisions with a cheap LINEAR position model; we scan
# coarser with the REAL integrated offset, because over this multi-revolution horizon a two-body /
# linear-extrapolated position diverges (the same reason the phase scan ranks by the integrated
# offset, not two-body) -- so the linear model misses the (narrow) right-revolution dip entirely.
LOV_SCAN_SIG  = 4.0     # scan +/- this many LOV sigmas about the anchor orbit
LOV_SCAN_STEP = 0.2     # sigma step: the right-revolution dip can be <0.3 sigma wide (a is swept fast
                        # along the LOV), so the grid must be finer than that to land in its basin.
                        # We need only land NEAR the dip -- the homotopy glue's capture radius is wide
                        # (LW2: a 28000"~8deg offset reels in to 0.96") -- not pinpoint the minimum.
LOV_SCORE_OBS = 12      # subsample the far apparition to this many obs for the scan ranking (the
                        # per-point ASSIST integration to the far epoch dominates the cost)
LOV_TOPK      = 4       # homotopy-glue the best this-many scan minima (closest-revolution first)
LOV_FAR_TOL   = 3.0     # accept far-apparition rms (") -- same real-vs-wrong-revolution margin as the
                        # period sweep's LINK_NEW_TOL (a wrong revolution glues to thousands of ").
LOV_ANCHOR_TOL = 0.05   # accept the LOV link ONLY if it leaves the anchor essentially UNCHANGED (vs the
                        # standalone anchor fit). The LOV slide rides the anchor's flat-chi^2 valley by
                        # construction, so a CLEAN member keeps the anchor rms put (LW2: 0.881->0.872,
                        # an improvement). If pulling the far apparition in instead DEGRADES the anchor
                        # (2008 TE4: 0.991->1.113), the far data is fighting the anchor -- it is not a
                        # genuine LOV member, and accepting it would preempt the better re-anchor
                        # recovery the pipeline reaches otherwise (TE4 67/73 @ 0.38" vs a 63/73 @ 0.47"
                        # LOV link). Much tighter than the period sweep's LINK_ANCHOR_TOL (0.5"), which
                        # is a "stays glued" bound, not a "rode the valley" one.


def _bridge_far_by_lov(ev, state, mask, nbm, anchor0, verbose=False):
    """Multi-revolution bridge by a Line-of-Variations slide (find_orb improve_along_lov): refit the
    anchor, take the LEAST-constrained direction of its covariance (the smallest-eigenvalue
    eigenvector of the normal matrix J^T J -- the correlated a/plane/phase combination the anchor
    leaves free), and densely scan the orbit along it for the offset minimum at the far apparition.
    Unlike the a-secant (slides a only) and the phase scan (slides phase only), the LOV slide moves
    along the FULL correlated valley, so it threads the cross-track plane error that defeats both.
    Homotopy-glue the best few scan minima and keep the first that links. Returns a glued state or
    None. GATED to two-apparition objects (the premise is one anchor + one far apparition to reach).

    The scan ranks by the REAL integrated far offset, not two-body: over a multi-revolution gap a
    Kepler/linear-extrapolated position diverges by degrees, so it cannot locate the right-revolution
    dip (verified: two-body offset is monotonic in s where the integrated offset has a sharp minimum).
    """
    if len(_clusters(ev.times_tdb)) > 2:
        return None
    full = mask | nbm
    _, result = ev.fit(state, mask)
    st = result.x
    if not np.all(np.isfinite(st)):
        return None
    J = np.asarray(result.jac); dof = max(1, J.shape[0] - 6)
    s2 = float(np.sum(result.fun**2)) / dof
    try:
        evals, evecs = np.linalg.eigh(J.T @ J)        # ascending eigenvalues
    except np.linalg.LinAlgError:
        return None
    lam, u = evals[0], evecs[:, 0]                     # smallest eigenvalue -> largest variance (LOV)
    if not (np.isfinite(lam) and lam > 0):
        return None
    sig = np.sqrt(s2 / lam)                            # 1-sigma extent along the LOV (state units)
    nb_idx = np.flatnonzero(nbm)
    sub = nb_idx[np.linspace(0, len(nb_idx) - 1, min(LOV_SCORE_OBS, len(nb_idx))
                             ).round().astype(int)]
    subm = np.zeros(ev.n, bool); subm[sub] = True
    grid = np.arange(-LOV_SCAN_SIG, LOV_SCAN_SIG + 1e-9, LOV_SCAN_STEP)
    offs = np.array([_med_off(ev, st + (g * sig) * u, subm) for g in grid])
    # local minima of the integrated far offset, ranked closest-first (each is one revolution count)
    mins = sorted((offs[i], grid[i]) for i in range(1, len(grid) - 1)
                  if np.isfinite(offs[i]) and offs[i] < offs[i - 1] and offs[i] < offs[i + 1])
    for off, g in mins[:LOV_TOPK]:
        jl = _homotopy_glue(ev, st + (g * sig) * u, full, nbm)
        jl2, _ = ev.fit(jl, full, max_nfev=400)        # tighten once the far apparition is in basin
        if np.all(np.isfinite(jl2)):
            jl = jl2
        a_rms, n_rms = _rms_sub(ev, jl, mask), _rms_sub(ev, jl, nbm)
        if a_rms < anchor0 + LOV_ANCHOR_TOL and n_rms < LOV_FAR_TOL:
            if verbose:
                print(f"   lov-scan: s={g:+.2f}sig (sig={sig:.1e}) far_medoff={off:.0f}\" -> "
                      f"anchor {anchor0:.3f}->{a_rms:.3f}\" far={n_rms:.3f}\"")
            return jl
    return None


def _ident_shortcircuit(ev, mask, nb, lock_opp=IDENT_LOCK_OPP):
    """True when gluing apparition `nb` onto the masked anchor would be a cross-link between two
    INDEPENDENTLY well-determined designations in an explicit multi-designation linkage request --
    a hypothesis that is virtually always a non-match. The expensive Tier-2 bridge can only fail
    (or, worse, manufacture a chimera), so the caller skips it and leaves `nb` unlinked. Returns
    False (gate OFF) unless EVERY guard holds:

      * ev.ident_mode -- the caller REQUESTED >= 2 designations (n_requested > 1). A numbered
        object queried by its number carries many alias designations for ONE body (combined=True)
        but n_requested == 1, so this stays False and its apparitions link normally;
      * the anchor and the candidate apparition are DISJOINT in designation -- same designation on
        both sides (the ordinary multi-opposition self-link) is never blocked;
      * each side has a designation that is independently multi-opposition, and at least one is
        period-LOCKED (see the IDENT_MULTI_OPP / IDENT_LOCK_OPP thresholds below).

    Only designations that are INDEPENDENTLY multi-opposition (>= IDENT_MULTI_OPP) count on each
    side: such an arc has its own period determined, so its sky position is predictable and a hard
    multi-rev jump to it is a real disagreement -- whereas a lone single-opposition tracklet has an
    unknown period whose large offset may be a genuine revolution ambiguity, so it is NEVER blocked
    (and single-opp aliases mixed into the anchor are simply ignored). At least one side must be
    period-LOCKED (>= IDENT_LOCK_OPP). A genuine identification between two such mature arcs (rare --
    the MPC would usually have merged them already) predicts both within arcsec, so its bridge runs
    through the cheap in-basin homotopy (off < LINK_BASIN) and never reaches this guarded branch."""
    if not getattr(ev, "ident_mode", False):
        return False
    od = getattr(ev, "obs_desig", None)
    opp = getattr(ev, "desig_opp", None)
    if od is None or not opp:
        return False
    anchor = {od[i] for i in np.flatnonzero(mask) if opp.get(od[i], 0) >= IDENT_MULTI_OPP}
    cand = {od[i] for i in range(nb[0], nb[1] + 1) if opp.get(od[i], 0) >= IDENT_MULTI_OPP}
    if not anchor or not cand or (anchor & cand):
        return False    # a side has no period-determined designation, or they share one -> never block
    return max(opp[d] for d in (anchor | cand)) >= lock_opp


MULTIREV_GAP_REVS = 0.75  # a far apparition this many orbital PERIODS or more from the anchor is a
                          # genuine multi-revolution gap even if its SKY offset is modest. A near-1yr-
                          # period NEO (an Earth-resonant Aten/Apollo) returns to nearly the same sky
                          # place each apparition, so its wrong-revolution ambiguity shows only a
                          # few-thousand-arcsec offset -- far below LINK_MULTIREV (14deg), which would
                          # route it to the cheap swarm (that can't sweep whole revolutions) and give
                          # up. When the cheap swarm misses such a by-TIME multi-rev gap, escalate to
                          # the Tier-2 period sweep / a-secant instead of quitting (2024 LW: links its
                          # 2015/2021/2023/2026 single-night tracklets to the dense 2024 arc).


def _bridge_far_multirev(ev, state, mask, nbm, nb, off, anchor0, full, verbose):
    """TIER 2 (expensive, for a genuine multi-revolution gap): the cheap swarm cannot bridge whole
    revolutions, so densely sweep the period for distinct-revolution candidate clones, full-LM each,
    keeping whichever GLUES; fall back to the a-secant / phase / LOV single-axis slides. Returns
    (linked?, state).

    Extracted verbatim from _bridge_opposition's old inline `else` branch so it can now be reached
    from TWO callers: the original off>LINK_MULTIREV branch AND the new cheap-swarm-miss escalation
    for a modest-offset multi-revolution-by-TIME gap (a near-1yr-period Earth-resonant NEO) -- see
    MULTIREV_GAP_REVS. Same logic, one home."""
    cands = _swarm_far_candidates(ev, state, mask, nbm)
    if cands is None:
        if verbose:
            print(f"   link apparition obs[{nb[0]}..{nb[1]}]: off={off:.1f}\" -> reject (no candidate)")
        return False, state
    best_cur, best_a, best_n = None, None, None
    for c in cands:
        jl, _ = ev.fit(c, full, max_nfev=BRIDGE_CAND_MAX_NFEV)
        if not np.all(np.isfinite(jl)):
            continue
        a_rms, n_rms = _rms_sub(ev, jl, mask), _rms_sub(ev, jl, nbm)
        if a_rms < anchor0 + LINK_ANCHOR_TOL and n_rms < LINK_NEW_TOL:
            # candidates are ordered closest-period first, and a WRONG-period clone
            # can't glue (its anchor balloons), so the FIRST that glues is the true
            # solution -- stop rather than full-LM the remaining (up to 7) candidates.
            best_cur, best_a, best_n = jl, a_rms, n_rms
            break
    if best_cur is None:
        # the coarse sweep couldn't resolve the period (a SHORT anchor barely constrains
        # it): root-find the right period directly by sliding the pinned semi-major axis
        # to zero the far apparition's RA residual (the a-constraint secant).
        _, _res = ev.fit(state, mask)
        jl = _bridge_far_by_a_secant(ev, _res.x, mask, nbm, anchor0, verbose)
        if jl is not None:
            best_cur, best_a, best_n = jl, _rms_sub(ev, jl, mask), _rms_sub(ev, jl, nbm)
    if best_cur is None:
        # period sweep + a-secant both missed -- the loose direction is the orbital PHASE,
        # not the shape. Slide the phase at fixed a to land the far apparition, then glue.
        jl = _bridge_far_by_phase(ev, state, mask, nbm, anchor0, verbose)
        if jl is not None:
            best_cur, best_a, best_n = jl, _rms_sub(ev, jl, mask), _rms_sub(ev, jl, nbm)
    if best_cur is None:
        # a-, phase-, and period-scans all slide a SINGLE axis and all miss when the residual
        # is cross-track (a tiny anchor-plane error amplified over many revolutions). Last
        # resort: slide along the anchor covariance's full least-constrained valley (the Line
        # of Variations), which moves a/plane/phase in the correlated combination that keeps
        # the anchor glued while reaching the far apparition on both sky axes (find_orb's
        # improve_along_lov). This is what links the pure-gravity 2-opp strand (2003 LW2).
        jl = _bridge_far_by_lov(ev, state, mask, nbm, anchor0, verbose)
        if jl is not None:
            best_cur, best_a, best_n = jl, _rms_sub(ev, jl, mask), _rms_sub(ev, jl, nbm)
    if verbose:
        tail = (f"anchor {anchor0:.3f}->{best_a:.3f}\" new={best_n:.3f}\""
                if best_cur is not None else "no glued candidate")
        print(f"   link apparition obs[{nb[0]}..{nb[1]}] ({int(nbm.sum())}obs): off={off:.1f}\" "
              f"multirev {len(cands)} cand -> {'LINKED' if best_cur is not None else 'reject'} {tail}")
    return (True, best_cur) if best_cur is not None else (False, state)
    # (tried: a small a-nudge multistart can rescue a rare local-min stall here -- fixes
    #  2004 TE10 but fired on 1 in ~950 fingerprint NEOs, too narrow to keep. See git history
    #  for the dropped code.)


def _n_revs_to_opposition(ev, state, mask, nbm):
    """Number of orbital periods between the anchor arc and the far apparition `nbm` (the TIME gap
    over the anchor's period). Used to recognise a multi-revolution gap whose SKY offset is modest
    (a near-1yr-period Earth-resonant NEO), which the sky-offset LINK_MULTIREV threshold misses."""
    a_anchor, _, _ = quick_aeq(state)
    if not (np.isfinite(a_anchor) and a_anchor > 0):
        return 0.0
    period_days = (a_anchor ** 1.5) * 365.25
    lt = ev.times_tdb[mask]; ft = ev.times_tdb[nbm]
    if not (lt.size and ft.size):
        return 0.0
    gap = min(abs(ft.min() - lt.max()), abs(lt.min() - ft.max()))
    return gap / period_days


def _ident_far_bridge_skip(ev, state, mask=None):
    """True when the expensive Tier-2 far-apparition bridge should be SKIPPED for `state` as anchor:
    we are in an explicit ident-check hypothesis (ev.ident_mode) AND the anchor orbit is main-belt-
    or-beyond (perihelion q >= IDENT_FAR_MIN_Q) AND the anchor (`mask`) spans more than
    IDENT_BRIDGE_MAX_NIGHTS nights -- a 2-night anchor's far prediction is too loose for a Tier-1
    miss to mean anything. The cheap Tier-1 swarm still runs -- only the
    grind-prone multi-revolution machinery is cut. Never fires outside ident_mode. q here is the
    LIVE anchor state's (quick_aeq): mid-linkage the anchor sub-arc's orbit is the only one that
    exists, and a genuinely main-belt anchor reads main-belt from it. See orbit_common.IDENT_FAR_MIN_Q."""
    if not getattr(ev, "ident_mode", False):
        return False
    if mask is not None and count_nights(ev.times_tdb[mask]) <= IDENT_BRIDGE_MAX_NIGHTS:
        return False
    _, _, q = quick_aeq(state)
    return bool(np.isfinite(q) and q >= IDENT_FAR_MIN_Q)


def _bridge_opposition(ev, state, mask, nb, verbose):
    """Glue apparition `nb` onto the linked set via weight continuation (swarm first if it
    starts out of basin). Returns (linked?, state)."""
    nbm = np.zeros(ev.n, bool); nbm[nb[0]:nb[1] + 1] = True
    anchor0 = _rms_sub(ev, state, mask)
    off = _med_off(ev, state, nbm)
    if not np.isfinite(off):
        # the orbit's integration DIVERGES reaching this apparition's epoch (non-finite
        # prediction) -- it's unreachable from this state, and neither the swarm nor the
        # ramp can rescue a NaN. Skip straight to "unlinked" (saves ~1s/apparition of
        # wasted swarm+ramp on the far gaps a heavy NEO inevitably has).
        return False, state
    full = mask | nbm
    if off > LINK_BASIN:
        if _ident_shortcircuit(ev, mask, nb, lock_opp=IDENT_LOCK_OPP_GATED):
            # A period-locked (3+-opp, the offset-gated bar) designation and a period-known (2+-opp)
            # DIFFERENT designation the user asked us to link, sitting a hard multi-rev jump apart
            # (off > LINK_BASIN -- the cheap in-basin homotopy already ruled out a real match): no
            # honest bridge can glue them. Skip the whole expensive cascade (cheap swarm + Tier-2
            # period/phase/LOV/secant) and leave it unlinked -- the same verdict the grind reaches
            # after seconds-to-minutes, returned instantly.
            if verbose:
                print(f"   link apparition obs[{nb[0]}..{nb[1]}]: off={off:.1f}\" -> reject "
                      f"(multi-opposition arcs, distinct designations -- not the same object)")
            return False, state
        if anchor0 > CHEAP_SWARM_MAX_ANCHOR_RMS:
            # the anchor itself is garbage (tens of arcsec): its covariance is so wide that the
            # cheap swarm just wanders, and Tier-2 needs a PRECISE anchor for its period sweep.
            # Neither tier can glue from here -- skip both straight to "unlinked" rather than
            # burn ~2.4s of cheap swarm on a hopeless base. (VM6's 86" 2023 seed hits this.)
            return False, state
        anchor_span = float(ev.times_tdb[mask].max() - ev.times_tdb[mask].min()) if mask.any() else 0.0
        # IDENT-MODE far-bridge skip: in an explicit ident hypothesis, a main-belt/outer anchor
        # (q >= IDENT_FAR_MIN_Q) never needs the expensive Tier-2 multi-revolution bridge -- a
        # genuine such match links through the cheap swarm below; Tier-2 would only grind a non-match
        # to the pair_timeout (and re-run it from every recovery re-anchor). Skip Tier-2 and leave
        # `nb` unlinked. NEO/Mars-crosser anchors (q < 1.3) and <= 2-night anchors keep full
        # Tier-2 diligence.
        skip_far = _ident_far_bridge_skip(ev, state, mask)
        if off <= LINK_MULTIREV:
            # TIER 1 (cheap): a moderate gap the small covariance swarm can reach -- this is what
            # links the common multi-opposition objects (e.g. Bennu), so it stays first.
            st = _swarm_to_basin(ev, state, mask, nbm, verbose)
            if st is None:
                # The cheap swarm missed. A modest sky offset usually means a small gap it simply
                # can't close -> give up. BUT a near-1yr-period Earth-resonant NEO returns to nearly
                # the same sky place each apparition, so a genuine multi-REVOLUTION gap also shows a
                # modest offset (below LINK_MULTIREV). Recognise that by TIME: if the far apparition
                # is >= MULTIREV_GAP_REVS periods away and the anchor is long+precise enough to seed a
                # period sweep, escalate to Tier-2 (which the cheap swarm can't substitute for on a
                # whole-revolution ambiguity) rather than quitting. (2024 LW's 2015/2021/2023/2026
                # tracklets, 1-9 revs from the 2024 arc at 1-12 kilo-arcsec offsets, link only here.)
                # SCOPED to a SINGLE-apparition anchor: this is the near-1yr discovery-arc case where
                # the lone dense arc must reach a far tracklet across the revolution ambiguity. Once
                # 2+ apparitions are already linked the period is pinned, so a tracklet that STILL
                # won't cheap-swarm in is genuinely discordant, not a revolution ambiguity -- running
                # the expensive sweep on it only burns time (PAIR_2020TL64/2020XM18/2025KV14/2025QG135).
                if (not skip_far
                        and anchor_span >= MULTIREV_MIN_ANCHOR_DAYS and anchor0 <= MULTIREV_MAX_ANCHOR_RMS
                        and len(_clusters(ev.times_tdb[mask])) == 1
                        and _n_revs_to_opposition(ev, state, mask, nbm) >= MULTIREV_GAP_REVS):
                    return _bridge_far_multirev(ev, state, mask, nbm, nb, off, anchor0, full, verbose)
                return False, state                 # cheap swarm missed and the gap is too
            state = st                              # small to be a period ambiguity -> give up
        elif skip_far:
            # ident-mode main-belt/outer anchor at a multi-rev-scale gap: skip the Tier-2 phase/LOV
            # bridge (see IDENT_FAR_MIN_Q) and leave `nb` unlinked.
            if verbose:
                print(f"   link apparition obs[{nb[0]}..{nb[1]}]: off={off:.1f}\" -> reject "
                      f"(ident-mode q>={IDENT_FAR_MIN_Q} -- no Tier-2 far bridge)")
            return False, state
        elif anchor_span < MULTIREV_MIN_ANCHOR_DAYS or anchor0 > MULTIREV_MAX_ANCHOR_RMS:
            # a multi-revolution-scale gap, but the anchor can't seed a period sweep: too short
            # an arc to constrain the period (a lone tracklet), or too poorly fit to be a precise
            # bridge base. The PERIOD sweep needs a long well-fit anchor, but the PHASE scan does
            # not: it keeps the anchor's (well-determined) a and slides the orbital phase, which a
            # short arc constrains poorly precisely because phase is the loose direction. Try it.
            jl = _bridge_far_by_phase(ev, state, mask, nbm, anchor0, verbose)
            if jl is not None:
                return True, jl
            # ...and if the phase slide misses too, slide along the full covariance Line of
            # Variations (find_orb improve_along_lov). A SHORT anchor is exactly where the LOV wins:
            # its covariance is wide, so the least-constrained valley is well-defined, and walking
            # the correlated a/plane/phase combination reaches a far apparition that a single-axis
            # phase slide leaves cross-track-offset. This is the 2003 LW2 link (2.1-d 2021 anchor).
            jl = _bridge_far_by_lov(ev, state, mask, nbm, anchor0, verbose)
            if jl is not None:
                return True, jl
            return False, state
        else:
            # TIER 2 (expensive, ONLY for a genuine multi-revolution gap): the cheap swarm cannot
            # bridge whole revolutions, so skip it entirely and densely sweep the period.
            return _bridge_far_multirev(ev, state, mask, nbm, nb, off, anchor0, full, verbose)
    # In-basin (or cheap-swarm-positioned): a gentle weight homotopy keeps the anchor glued
    # while the new apparition's weight rises to full.
    cur = _homotopy_glue(ev, state, full, nbm)
    a_rms, n_rms = _rms_sub(ev, cur, mask), _rms_sub(ev, cur, nbm)
    ok = a_rms < anchor0 + LINK_ANCHOR_TOL and n_rms < LINK_NEW_TOL
    if verbose:
        print(f"   link apparition obs[{nb[0]}..{nb[1]}] ({int(nbm.sum())}obs): off={off:.1f}\" "
              f"anchor {anchor0:.3f}->{a_rms:.3f}\" new={n_rms:.3f}\" -> {'LINKED' if ok else 'reject'}")
    return (True, cur) if ok else (False, state)


def _bridge_bulk(ev, state, mask, apps, verbose):
    """Glue MANY already-in-basin apparitions on at once with a single joint weight
    ramp — far cheaper than one cycle each. The whole batch of new obs starts
    near-weightless and ramps to full while the anchor stays fixed. Returns
    (linked?, state)."""
    newm = np.zeros(ev.n, bool)
    for (cs, ce) in apps:
        newm[cs:ce + 1] = True
    anchor0 = _rms_sub(ev, state, mask)
    full = mask | newm
    cur = _homotopy_glue(ev, state, full, newm)
    a_rms = _rms_sub(ev, cur, mask)
    ok = a_rms < anchor0 + LINK_ANCHOR_TOL          # only require the anchor stays glued
    if verbose:
        print(f"   bulk-link {len(apps)} apparitions ({int(newm.sum())}obs): "
              f"anchor {anchor0:.3f}->{a_rms:.3f}\" -> {'LINKED' if ok else 'reject'}")
    return (True, cur) if ok else (False, state)


def _link_oppositions(ev, state, mask, history, verbose):
    """Grow the linked set to every reachable apparition. At each step, BULK-add every
    unlinked apparition the current orbit already predicts within the basin (one joint
    weight continuation for all of them) — once an opposition or two pins the orbit, most of
    the rest fall in at once, so there's no need to walk them one by one. Only the
    genuinely-hard far jumps drop to the per-apparition swarm path."""
    clusters = _clusters(ev.times_tdb)
    remaining = [c for c in clusters if not mask[c[0]:c[1] + 1].any()]
    if not remaining:
        return state, mask
    state, _ = ev.fit(state, mask)
    mask = mask.copy()
    bridge_fails = 0
    while remaining:
        # Drop clusters the ident gate refuses outright -- a period-known foreign designation that
        # cannot attach to a period-locked anchor (see _ident_shortcircuit). The per-apparition
        # bridge would only leave them unlinked anyway, so skip even MEASURING their offset: each
        # _med_off is a full integration to a far epoch, and a non-matching ident pair has several.
        # As the anchor only grows within its own designation, a rejected foreign cluster stays
        # rejected, so dropping it here is exactly byte-equivalent to bridging-and-rejecting it.
        # No-op unless ev.ident_mode is armed (an explicit multi-designation linkage request).
        remaining = [c for c in remaining if not _ident_shortcircuit(ev, mask, c)]
        if not remaining:
            break
        offs = [(c, _med_off(ev, state, _cluster_mask(ev.n, c))) for c in remaining]
        in_basin = [c for c, o in offs if np.isfinite(o) and o < LINK_BASIN]
        if in_basin:                                # bulk-add everything reachable now
            ok, state = _bridge_bulk(ev, state, mask, in_basin, verbose)
            if ok:
                for (cs, ce) in in_basin:
                    mask[cs:ce + 1] = True
                remaining = [c for c in remaining if c not in in_basin]
                history.append(_hsnap({"stage": f"bulklink+{len(in_basin)}", "n_obs": int(mask.sum()),
                                "rms": ev.rms_arcsec(state, mask), "score": 0.0, "kept": True}, state, mask))
                continue
        # nothing in basin (or bulk failed): take the nearest hard jump one at a time
        lt = ev.times_tdb[mask]
        nb = min(remaining, key=lambda c: min(abs(ev.times_tdb[c[0]] - lt.max()),
                                              abs(lt.min() - ev.times_tdb[c[1]])))
        # FAIL FAST. The per-apparition Tier-2 cascades are not independent experiments: every one
        # of them tests the SAME anchor orbit, and the loop takes the NEAREST unlinked apparition
        # first. So an anchor that cannot reach its nearest neighbours is not going to reach the
        # far ones, and running the remaining cascades is re-testing a hypothesis that has already
        # failed -- at ~700 integrations each. 162385 (15 apparitions) ran THIRTEEN of them from an
        # 11-obs/692-day IOD window that straddled apparition gaps: twelve rejected, one linked,
        # and the object was then solved by the recovery re-anchor whose plain arc extension walked
        # 8 -> 30 -> 49 -> 102 -> 336 obs. 585 CPU seconds, almost all of it spent proving one bad
        # anchor bad over and over.
        #
        # After LINK_BRIDGE_MAX_FAILS consecutive rejections we stop bridging for THIS anchor and
        # leave the rest unlinked, which is the honest verdict anyway -- the fit is then "troubled"
        # and the recovery cascade re-anchors, which is what fixes these objects.
        if bridge_fails >= LINK_BRIDGE_MAX_FAILS:
            if verbose:
                print(f"   bridge: {bridge_fails} consecutive rejections from this anchor -- "
                      f"{len(remaining)} apparitions left unlinked (re-anchor decides)", flush=True)
            break
        ok, state = _bridge_opposition(ev, state, mask, nb, verbose)
        bridge_fails = 0 if ok else bridge_fails + 1
        remaining.remove(nb)
        if ok:
            mask[nb[0]:nb[1] + 1] = True
            history.append(_hsnap({"stage": f"link[{nb[0]}:{nb[1]}]", "n_obs": int(mask.sum()),
                            "rms": ev.rms_arcsec(state, mask), "score": 0.0, "kept": True}, state, mask))
    return state, mask


def _cluster_mask(n, c):
    m = np.zeros(n, bool); m[c[0]:c[1] + 1] = True; return m


def _is_clean(ev, state, mask):
    """True if the converged fit needs no rejection/redemption/drop work, so the whole
    reject_drop stage can be skipped: raw RMS < 1" (no contaminant-apparition trigger),
    no INCLUDED obs above the first reject limit (3σ), and no EXCLUDED obs within the
    redemption thresholds. One full-arc sweep here replaces ~6 inside the stage."""
    dra, ddec = ev.resid_arcsec(state)              # single full sweep
    raw = np.hypot(dra, ddec)
    inc = mask
    if float(np.sqrt(np.mean(raw[inc]**2))) >= RMS_DROP_TRIGGER:
        return False
    rsig = np.hypot(dra / ev.sig_ra, ddec / ev.sig_dec)
    if float(rsig[inc].max(initial=0.0)) >= REJECT_LIMIT:
        return False
    exc = ~mask
    if exc.any() and bool(((rsig[exc] < REDEEM_SIGMA) | (raw[exc] < REDEEM_ARCSEC)).any()):
        return False
    return True


# --- attempt_extensions() : the driver -----------------------------------------
ECC_CAP_EXT = 0.9999      # bound-orbit ceiling for the under-constrained extension refit (matches
                          # fit_object.ECC_CAP; defined locally because fit_object imports
                          # arc_extend, so arc_extend can't import it back).
ECC_BARRIER_W = 1.0e4     # one-sided barrier weight on (e - cap), as in fit_object._refit_bounded_ecc
ECC_REINIT = 100.0        # if the free extension fit overshoots to e>this before the e>=1 capped-walk
                          # test fires, its state is a GROSS hyperbola whose integration recedes to
                          # enormous distance and costs ~10 s/eval over a long arc -- reseed the capped
                          # fit from the bound best_state instead (2010 TA192 e~8e3: 315s->23s). A
                          # merely-hyperbolic warm start (e up to tens, e.g. 2006 RH119 e=4.1)
                          # integrates fine and is a BETTER seed than the farther best_state, so keep
                          # it -- reseeding it would only perturb the greedy walk's path. The two
                          # regimes are ~3 orders of magnitude apart, so the threshold isn't delicate.
CAP_WALK_MAX_RMS = 50.0   # the keep-growing walk proceeds only while the e-capped orbit still FITS
                          # the included obs this well ("). A coherent under-constrained arc fits to
                          # a few arcsec (2010 LP68's walk peaks ~1.4"); a diverged wrong-revolution
                          # orbit (cross-apparition NEO, e.g. 2008 UK199) sits at thousands of arcsec
                          # -- that's the bridge's job, so above this we fall through to revert and
                          # let the linker own it (keeps UK199 fast instead of an obs-by-obs grind).


def _fit_capped_ecc(ev, seed, mask, cap=ECC_CAP_EXT, w=ECC_BARRIER_W, max_nfev=300):
    """LM fit over `mask` with eccentricity held <= cap by a one-sided barrier residual
    w*max(0, e(state) - cap), so an under-constrained short arc stays BOUND instead of drifting
    hyperbolic. Mirrors fit_object._refit_bounded_ecc but lives here (reads e from quick_aeq) for
    the extension's keep-growing path. Uses the analytic Jacobian when available -- the barrier
    row's gradient is finite-differenced through quick_aeq, which is integration-free, so the fit
    costs ~1 integration/iteration like Evaluator.fit (not ~7 from a fully numeric jac, which made
    the walk 9x slower on dense arcs). Returns the fitted state (caller checks finiteness)."""
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
    ra = ev.ra[idx]; dec = ev.dec[idx]; cosd = ev.cosd[idx]
    sra = ev.fit_sig_ra[idx]; sdec = ev.fit_sig_dec[idx]; m = len(idx)

    def _pen(s):
        _a, e, _q = quick_aeq(s)
        return w * max(0.0, (e if np.isfinite(e) else 2.0) - cap)

    if ev._predict_jac is not None:
        cache = {}
        def _compute(state):
            key = state.tobytes(); hit = cache.get(key)
            if hit is None:
                cache.clear(); hit = cache[key] = ev._predict_jac(ev.base_sim, ev.t0, state, tt, Rs)
            return hit
        def resid(state):
            pred, _ = _compute(state)
            r = _radec_resid(pred, ra, dec, cosd, sra, sdec)
            return _clamp(np.append(r, _pen(state)))
        def jac(state):
            _, J = _compute(state)
            out = _radec_jac(J, cosd, sra, sdec)            # (2m,6) scaled + finite-filtered
            brow = np.zeros(6)
            p0 = _pen(state)
            if p0 > 0.0:                                   # gradient only inside the barrier
                for k in range(6):
                    h = 1e-6 * (abs(state[k]) + 1e-6)
                    sp = state.copy(); sp[k] += h
                    brow[k] = (_pen(sp) - p0) / h
            full = np.vstack([out, brow])
            return np.where(np.isfinite(full), full, 0.0)
        result = least_squares(resid, seed, jac=jac, method="lm",
                            xtol=1e-8, ftol=1e-8, max_nfev=max_nfev)
        return result.x

    def resid(s):                                          # no analytic jac: numeric fallback
        pred = ev._predict(ev.base_sim, ev.t0, s, tt, Rs)
        r = _radec_resid(pred, ra, dec, cosd, sra, sdec)
        return _clamp(np.append(r, _pen(s)))
    return least_squares(resid, seed, method="lm",
                         xtol=1e-8, ftol=1e-8, max_nfev=max_nfev).x


def attempt_extensions(ev, seed, init_mask, verbose=False, activate=True):
    """Port of find_orb attempt_extensions(). Start from the IOD sub-arc, extend +
    refit while it improves (keeping the best), then reject/redeem outliers.

    Returns (best_state, best_mask, result, history): `result` is the scipy
    OptimizeResult of a final fit over the included obs (for covariance), and
    history is a list of dicts for diagnostics/logging."""
    _t = _stime.perf_counter()
    mask = init_mask.copy()
    best_state, _ = ev.fit(seed, mask)
    best_mask = mask.copy()
    best_score = _score(ev, best_state, best_mask)
    unbound_ok = False      # latched by the extension loop once no bound orbit fits (see there)
    history = [_hsnap({"stage": "subarc", "n_obs": int(best_mask.sum()),
                "rms": ev.rms_arcsec(best_state, best_mask), "score": best_score},
                best_state, best_mask)]
    if verbose:
        si = np.flatnonzero(best_mask)
        print(f"   initial sub-arc: obs[{si[0]}..{si[-1]}] = {len(si)} obs over "
              f"{ev.times_tdb[si[-1]]-ev.times_tdb[si[0]]:.0f}d, "
              f"rms={ev.rms_arcsec(best_state, best_mask):.3f}\"")

    # Ban list (find_orb-style early rejection): a clearly-bad obs inside the seed/extension
    # glues the orbit to a wrong solution that can't bridge to other apparitions. We BAN such
    # obs (extension may not re-add them) and give them one chance back at final redemption.
    banned = np.zeros(ev.n, bool)
    # seed pre-clean: a contaminant inside the seed (often clumped with GOOD obs on the same
    # night) glues the orbit to a wrong minimum where it no longer reads as an outlier, and the
    # fit then can't bridge to other apparitions. If the seed fits poorly, drop each of its
    # nights in turn; if one drop SHARPLY cleans the fit, that night carried a contaminant --
    # ban only the obs from it that stay gross at the cleaned orbit (its good obs return via the
    # final redemption). Extension may not re-add a banned obs.
    if ev.rms_arcsec(best_state, best_mask) > SEED_POOR_RMS and int(best_mask.sum()) >= 2 * SEED_CLEAN_MIN:
        si = np.flatnonzero(best_mask)
        nights, ns = [], 0
        for k in range(len(si)):
            if k + 1 == len(si) or ev.times_tdb[si[k + 1]] - ev.times_tdb[si[k]] > NIGHT_GAP_DAYS:
                nights.append(si[ns:k + 1]); ns = k + 1
        seed_rms = ev.rms_arcsec(best_state, best_mask)
        bd, br, bst = None, seed_rms, None
        for night in nights:
            trial = best_mask.copy(); trial[night] = False
            if int(trial.sum()) < SEED_CLEAN_MIN:
                continue
            st2, _ = ev.fit(best_state, trial, max_nfev=EXTEND_MAX_NFEV)
            r2 = ev.rms_arcsec(st2, trial)
            if r2 < br:
                bd, br, bst = night, r2, st2
        if bd is not None and br < seed_rms * SEED_SHARP:     # a night was dragging the orbit
            rs = ev.resid_sigmas(bst)
            bad = np.zeros(ev.n, bool); bad[bd] = True
            bad &= rs > SEED_CLEAN_SIGMA                       # ...only its genuine outliers
            if bad.any():
                clean_mask = best_mask & ~bad
                st_clean, _ = ev.fit(best_state, clean_mask, max_nfev=EXTEND_MAX_NFEV)
                rejoined = False
                # A real contaminant, once removed, leaves a BOUNDED, better-determined orbit.
                # If the "cleaned" orbit is instead UNBOUND (a<=0 or e>=1), dropping the night did
                # not expose a contaminant -- it merely under-determined an already-short arc (every
                # night-drop lowers the rms of a range-degenerate arc, so the SEED_SHARP cut fires
                # spuriously). Banning a whole good night then strands the arc on a single-night
                # hyperbola (2002 TD301: a 26-obs / 1.08-d / 2-night tracklet -> e=1e5 dropping
                # 14/26). Keep all the obs: leave best_state/best_mask as the bounded full-arc seed.
                a_cl, e_cl, _q_cl = quick_aeq(st_clean)
                if a_cl <= 0 or e_cl >= 1.0:
                    rejoined = True                       # under-determination, not contamination
                    if verbose:
                        print(f"   seed-clean: dropping the {int(bad.sum())}-obs night leaves an "
                              f"UNBOUND orbit (a={a_cl:.2f} e={e_cl:.3g}) -> under-determined short "
                              f"arc, not contamination; keeping all obs")
                # A whole gross night is not the "contaminant clumped with good obs" pattern --
                # it may be a temporally-distant night that the rough seed mis-predicts but a
                # JOINT fit reconciles (the night CONSTRAINS the orbit; banning it amputates the
                # arc and strands a near-parabolic short-arc solution -- e.g. 2019 EJ3). Try
                # keeping it: fit the seed jointly from the cleaned orbit, accept if it agrees.
                if not rejoined and int(bad.sum()) == len(bd):
                    st_join, _ = ev.fit(st_clean, best_mask, max_nfev=EXTEND_MAX_NFEV)
                    if ev.rms_norm(st_join, best_mask) < SEED_REJOIN_SIGMA:
                        best_state = st_join; rejoined = True
                        if verbose:
                            print(f"   seed-clean: kept a distant {len(bd)}-obs night via joint "
                                  f"fit ({ev.rms_norm(st_join, best_mask):.2f}σ); not contamination")
                if not rejoined:
                    banned |= bad; best_mask = clean_mask; best_state = st_clean
                    if verbose:
                        print(f"   seed-clean: banned {int(bad.sum())} contaminant(s) on worst seed "
                              f"night (rms {seed_rms:.2f}->{ev.rms_arcsec(best_state, best_mask):.2f}\")")
                best_score = _score(ev, best_state, best_mask)

    state = best_state.copy()
    for it in range(MAX_EXTEND_ITERS):
        trial = best_mask.copy()
        if extend_solution(ev, trial, state) == 0:
            break                                   # nothing left to add
        trial &= ~banned                            # never re-add a banned outlier
        if not (trial & ~best_mask).any():
            break                                   # extension only re-proposed banned obs
        state, _ = ev.fit(best_state, trial, max_nfev=EXTEND_MAX_NFEV,  # refit from current best
                          stall_guard=True)   # ...the next auto_reject scores obs against it
        # CROSS-APPARITION GUARD (low-q orbits only): obs the extension reached across a
        # >OPP_GAP_DAYS jump are a different apparition. Keep them only if they fit WELL
        # jointly -- a poor jump (a far single night dragged in at a wrong orbit) is a half-link
        # that leaves the apparition no longer "empty", preempting the multi-revolution bridge
        # that would link it properly. Policed only for low-perihelion (q < CROSS_APP_MAX_Q) in-
        # progress orbits, the regime where multi-rev linkage arises (and a little bridge time is
        # worth it); main-belt/Jupiter-crossers keep the fast old path. When it fires, stop
        # extending toward that apparition and let the bridge/cluster-activation own it.
        added_idx = np.flatnonzero(trial & ~best_mask)
        if best_mask.any() and len(added_idx) and quick_aeq(state)[2] < CROSS_APP_MAX_Q:
            cur_t = ev.times_tdb[best_mask]
            gap = np.abs(ev.times_tdb[added_idx][:, None] - cur_t[None, :]).min(axis=1)
            crossing = added_idx[gap > OPP_GAP_DAYS]
            if len(crossing):
                cm = np.zeros(ev.n, bool); cm[crossing] = True
                cross_rms = ev.rms_arcsec(state, cm)
                if cross_rms > CROSS_APP_MAX_RMS:
                    if verbose:
                        print(f"   extend: cross-apparition jump (+{len(crossing)} obs) "
                              f"rms={cross_rms:.2f}\" > {CROSS_APP_MAX_RMS}\" (q<{CROSS_APP_MAX_Q}) "
                              f"-> leave for bridge")
                    break
        # UNDER-CONSTRAINED ARC -> let the extension play out (keep the orbit BOUND, keep growing).
        # A short arc of a distant object is degenerate in (a,e): the free LM drifts hyperbolic
        # (a<=0 / e>=1), so _score's unphysical 1e6 penalty would trip the "blew up -> revert"
        # guard below on step one and strand the seed on its discovery night. Instead cap e with a
        # one-sided barrier so the orbit stays physical, and accept each net-new obs WITHOUT
        # reverting on the (meaningless-while-unpinned) rms -- extend across the intra-apparition
        # gap one obs at a time until a later apparition's baseline pins the orbit, at which point
        # the capped fit comes out bound (e<cap) and the normal greedy logic below resumes.
        # (find_orb grows the arc before the orbit is determined; this recovers e.g. 2010 LP68's
        # a=36 au TNO from a single-night seed Gauss/Herget could only fit as a spurious e>1 orbit.)
        a_uc, e_uc, _q_uc = quick_aeq(state)
        if ((a_uc <= 0 or e_uc >= 1.0) and int(trial.sum()) > int(best_mask.sum())
                and not unbound_ok):
            # A free fit that drifted to a GROSS hyperbola (e > ECC_REINIT) is a ruinous
            # seed for the capped refit -- every iteration integrates an orbit receding to
            # enormous distance at ~10 s/eval -- so start it from the previous BOUND
            # best_state, which converges to the same under-constrained minimum from inside
            # cheap-to-integrate territory. See ECC_REINIT for the measurement and for why a
            # MILD overshoot is deliberately left alone.
            seed_cap = (best_state if (not np.isfinite(e_uc) or e_uc > ECC_REINIT)
                        and np.all(np.isfinite(best_state)) else state)
            capped = _fit_capped_ecc(ev, seed_cap, trial)
            cap_rms = ev.rms_arcsec(capped, trial) if np.all(np.isfinite(capped)) else float("inf")
            # keep growing ONLY while the capped orbit still fits the included obs (a coherent,
            # merely under-constrained arc); a huge rms is a diverged wrong-revolution orbit that
            # belongs to the bridge -> fall through to the normal revert/break below.
            if cap_rms < CAP_WALK_MAX_RMS:
                state = capped
                best_state, best_mask = state.copy(), trial.copy()
                best_score = _score(ev, best_state, best_mask)
                if verbose:
                    ac, ec, _qc = quick_aeq(state)
                    print(f"   extend ->{int(best_mask.sum())} obs: under-constrained "
                          f"(a={ac:.1f} e={ec:.4f}) -> e-capped, keep growing  rms={cap_rms:.3f}\"")
                history.append(_hsnap({"stage": f"extend{it}", "n_obs": int(trial.sum()),
                                "rms": cap_rms, "score": best_score, "kept": True}, state, trial))
                continue
            # No bound orbit fits this arc even loosely. On an arc with the nights an unbound orbit
            # may be believed on (hyperbolic.unbound_arc_ok -- a mislinkage fits hyperbolic too) the
            # free unbound fit is not a blown-up extension, so from here on it is scored on its
            # residuals alone: 1I/2I/3I used to stop at their seed sub-arc on the 1e6 unbound term.
            # A diverged orbit still fails below, on those residuals.
            from pyroorb import hyperbolic
            if hyperbolic.unbound_arc_ok(ev.times_tdb[trial])[0]:
                unbound_ok = True
        rms = ev.rms_arcsec(state, trial)
        score = _score(ev, state, trial, allow_unbound=unbound_ok)
        improved = score < best_score - EXTEND_SCORE_MARGIN or (best_mask.sum() < trial.sum() and score < EXTEND_GROW_OK_SCORE)
        if verbose:
            print(f"   extend {int(best_mask.sum())}->{int(trial.sum())} obs: "
                  f"rms={rms:.3f}\" score={score:.3f} {'KEEP' if improved else 'revert'}")
        history.append(_hsnap({"stage": f"extend{it}", "n_obs": int(trial.sum()),
                        "rms": rms, "score": score, "kept": bool(improved)}, state, trial))
        if score > EXTEND_BLOWUP_SCORE:              # blew up -> stop, revert
            break
        if improved:
            best_state, best_mask, best_score = state.copy(), trial.copy(), score
        else:
            # converged: no improvement from this extension
            if int(trial.sum()) > int(best_mask.sum()):
                # we added obs but didn't improve the score; accept the larger arc
                # only if it's still a sane fit, else stop here.
                if score < EXTEND_GROW_OK_SCORE:
                    best_state, best_mask, best_score = state.copy(), trial.copy(), score
            break

    _stadd("subarc_extend", _t)

    # --- cluster activation: pull in whole apparitions adjacent extension
    #     couldn't bridge (multi-opposition linkage), refit + reject. (Skipped in
    #     the fast screen used by --force's drop-night search.) ---
    _t = _stime.perf_counter()
    if activate:
        # For few-apparition objects (the common multi-opposition case) the wholesale
        # activation is fast and proven, so keep it. For MANY apparitions (NEOs with
        # dozens) it just burns time trying — and reverting — far apparitions the
        # bulk-add linker handles directly, so skip straight to the linker.
        if len(_clusters(ev.times_tdb)) <= LINK_MANY_APPS:
            best_state, best_mask = _activate_clusters(ev, best_state, best_mask, history, verbose)
        if any(not best_mask[cs:ce+1].any() for (cs, ce) in _clusters(ev.times_tdb)):
            best_state, best_mask = _link_oppositions(ev, best_state, best_mask, history, verbose)
    _stadd("linkage", _t)

    # --- outlier rejection / redemption / contaminant-drops. SKIP the whole stage
    #     when the fit is already clean (no obs to reject, none to redeem, raw rms
    #     < 1") -- on the majority of objects this stage's ~6 full-arc sweeps would
    #     just confirm there's nothing to do. ---
    _t = _stime.perf_counter()
    state = best_state.copy(); mask = best_mask.copy()
    # find_orb sigma.txt: BEFORE final rejection, distrust+relax any obscode that is
    # systematically over-confident for this object, so its good sub-arcsec obs aren't
    # clipped as outliers (a real blunder, large in arcsec, still rejects below).
    if _relax_overconfident_obscodes(ev, state, mask, verbose):
        state, _ = ev.fit(state, mask, stall_guard=True)
    # CONVERGE before anything below judges residuals against absolute sigma (find_orb runs
    # full_improvement to convergence before auto-reject). The state handed in is whatever the
    # extension / activation / link stages left: loose-tolerance, nfev-capped solves that only
    # had to be good enough to ACCEPT a link. 2015 VZ1: the multirev bridge glued its 2026
    # close-approach recovery at new=1.9", the solve stopped short (it moved, so the stall
    # guard never armed), and round 0 clipped the whole sigma-0.1" F51 night at 11-25 sigma --
    # the refit without it then left F51 7" off for good. Converged on the same 35 obs it is
    # 0.49" with nothing over 3.5 sigma. The relax pass above still runs FIRST, on the
    # incoming state (2004 FH: T14's over-tight sigma shows there, and converging first would
    # pull the orbit toward it instead); a second pass then catches what only the converged
    # orbit shows (VZ1's D29 night, 3.2 sigma at sub-arcsec raw).
    state, _ = ev.fit(state, mask, polish=True)
    if _relax_overconfident_obscodes(ev, state, mask, verbose):
        state, _ = ev.fit(state, mask, stall_guard=True)
    if not _is_clean(ev, state, mask):
        span0 = _included_span(ev, mask)            # evaporation baseline: fixed across rounds
        for r in range(4):                          # up to 4 reject/redeem rounds
            changed = auto_reject_within_arc(ev, mask, state, orig_span=span0)
            if changed <= 0:
                break
            state, _ = ev.fit(state, mask, stall_guard=True)
            if verbose:
                print(f"   reject round {r}: {changed} toggled -> "
                      f"{int(mask.sum())} obs, rms={ev.rms_arcsec(state, mask):.3f}\"")
            history.append(_hsnap({"stage": f"reject{r}", "n_obs": int(mask.sum()),
                            "rms": ev.rms_arcsec(state, mask),
                            "score": _score(ev, state, mask), "kept": True}, state, mask))
        # single-night-apparition suspicion is mislinkage TROUBLESHOOTING (find_orb has
        # no equivalent): a ton of robustness machinery we should only spend on objects
        # that actually NEED it. We run it ONLY when the fit is genuinely TROUBLED -- either
        # (a) a whole apparition couldn't link ("can't extend to all the oppositions"), or
        # (b) the fit is POOR (raw rms > DROP_SUSPECT_RMS) AND its exclusions span more than
        # one night -- the signature that a single mislinkage is dragging the orbit so
        # several nights look discordant. A HEALTHY fit (good rms, all apparitions linked)
        # skips it entirely -- even a heavy 31-opposition object naturally has a few scattered
        # outliers across many nights, and trial-dropping each of them is pure waste
        # (944 Hidalgo: ~60 s of useless per-night refits at rms 0.49").
        exc = np.flatnonzero(~mask)
        n_excl_nights = (1 + int(np.sum(np.diff(ev.times_tdb[exc]) > NIGHT_GAP_DAYS))) if len(exc) else 0
        unlinked_app = any(not mask[cs:ce+1].any() for (cs, ce) in _clusters(ev.times_tdb))
        if unlinked_app or (n_excl_nights > 1 and ev.rms_arcsec(state, mask) > DROP_SUSPECT_RMS):
            state, mask = _drop_suspect_single_nights(ev, state, mask, history, verbose)
        # final redemption (within REDEEM_SIGMA OR ARCSEC).
        state, mask = _final_redeem(ev, state, mask, history, verbose)
    _stadd("reject_drop", _t)

    # final clean fit over the committed included set (covariance source) — ALL obs,
    # not the thinned subsample, so the reported orbit + covariance use everything.
    _t = _stime.perf_counter()
    final_state, final_res = _final_lm(ev, state, mask)
    final_state, mask, final_res = _settle_final_mask(
        ev, final_state, mask, final_res, history, verbose)
    _stadd("final_fit", _t)
    return final_state, mask, final_res, history


def _final_lm(ev, state, mask):
    """The delivery LM: one clean full-arc fit, with the status-0 retry contract.
    MINPACK status 0 = stopped at the max_nfev cap, not at a minimum: the elements
    would be a mid-iteration state and the covariance evaluated off-minimum. One
    warm restart from the reached state (a close seed converges in a few evals or
    proves the cap wasn't the obstacle); keep whichever has the smaller residual
    sum, ties to the restart. If the kept result STILL carries status 0, it flows
    into FittedOrbit, which flags the fit non-converged -- so a restart that
    converged but came back WORSE is discarded and the fit ships flagged."""
    final_state, final_res = ev.fit(state, mask, thin=False)
    if getattr(final_res, "status", 1) == 0:
        st2, res2 = ev.fit(final_state, mask, thin=False)
        if float(np.dot(res2.fun, res2.fun)) <= float(np.dot(final_res.fun, final_res.fun)):
            final_state, final_res = st2, res2
    return final_state, final_res


FINAL_SETTLE_ROUNDS = 3   # cap on _settle_final_mask rounds (it only ever ADDS obs, so it
                          # terminates on its own; this is a backstop, not the exit test)


def _settle_final_mask(ev, state, mask, result, history, verbose):
    """Settle the EXCLUSION mask against the orbit we are actually going to DELIVER.

    Every reject/redeem decision above is taken against an INTERMEDIATE orbit, and
    `_final_lm` -- which runs after all of them -- is the last thing that moves the orbit.
    So an obs clipped at 7 sigma from a half-converged fit stayed clipped even when the delivered
    orbit put it at 0.4 sigma: the fit reported observations as rejected while showing them
    sub-arcsecond O-C, which is not a fit decision the user can act on, it is a stale one
    (2002 QG17 shipped three obs marked rejected at 0.4/0.7/0.8 sigma).

    So re-run the `_final_redeem` amnesty against the DELIVERED orbit and, if it takes
    anything back, re-deliver. Redemption-only on purpose: the mask can only GROW, so the
    loop is monotone and cannot oscillate or evaporate an arc, and no obs can be clipped here
    that the rejection rounds above did not already judge. The gates are unchanged --
    REDEEM_SIGMA or REDEEM_ARCSEC, exactly as the pre-delivery pass applies them.

    Redeeming obs can RAISE the reported rms (more observations, same orbit) -- that is the
    honest direction: the rms now describes the whole arc instead of the subset that a
    stale mask happened to leave in.

    Returns the (possibly re-fit) state, the settled mask, and the matching LM result --
    the result must travel with them because it is the covariance source, and a covariance
    built on the pre-redemption mask would not describe the delivered orbit."""
    for _ in range(FINAL_SETTLE_ROUNDS):
        if mask.all():
            break                                   # nothing excluded -> nothing to settle
        state2, mask2 = _final_redeem(ev, state, mask, history, verbose)
        if int(mask2.sum()) == int(mask.sum()):
            break                                   # the delivered orbit agrees with the mask
        mask = mask2
        state, result = _final_lm(ev, state2, mask)
        history.append(_hsnap({"stage": "settle", "n_obs": int(mask.sum()),
                               "rms": ev.rms_arcsec(state, mask),
                               "score": _score(ev, state, mask), "kept": True}, state, mask))
    return state, mask, result


REDEEM_SIGMA = 3.0      # final amnesty: redeem an excluded obs within this many sigma
REDEEM_ARCSEC = 0.5     # ...OR this raw arcsec residual (a small raw resid is a good
                        # position regardless of an over-tight reported sigma)


def _final_redeem(ev, state, mask, history, verbose):
    """Add back any excluded obs whose residual to the (clean) orbit is within
    REDEEM_SIGMA sigmas OR REDEEM_ARCSEC arcsec, then refit once."""
    dra, ddec = ev.resid_arcsec(state)
    raw = np.hypot(dra, ddec)
    sig = ev.resid_sigmas(state)
    redeem = (~mask) & ((sig < REDEEM_SIGMA) | (raw < REDEEM_ARCSEC))
    n = int(redeem.sum())
    if n:
        mask = mask | redeem
        state, _ = ev.fit(state, mask, stall_guard=True)
        if verbose:
            print(f"   final redeem: +{n} obs (within {REDEEM_SIGMA:g}σ or "
                  f"{REDEEM_ARCSEC:g}\") -> {int(mask.sum())} obs, "
                  f"rms={ev.rms_arcsec(state, mask):.3f}\"")
        history.append(_hsnap({"stage": "redeem", "n_obs": int(mask.sum()),
                        "rms": ev.rms_arcsec(state, mask),
                        "score": _score(ev, state, mask), "kept": True}, state, mask))
    return state, mask


def _is_single_night(ev, cs, ce):
    """A cluster is a single night if all its obs fall within 12 h (no curvature,
    can't be IOD'd alone).  A SPAN test (first to last), which is why it reads
    SINGLE_NIGHT_SPAN_DAYS and not the consecutive-gap NIGHT_GAP_DAYS."""
    return ev.times_tdb[ce] - ev.times_tdb[cs] < SINGLE_NIGHT_SPAN_DAYS


def _drop_suspect_single_nights(ev, best_state, best_mask, history, verbose):
    """For each INCLUDED single-night apparition, test the hypothesis that it is a
    mislinkage: drop it, refit, and let the rejection/redemption pull back in the
    obs that now fit. Commit the drop only if it links strictly MORE observations
    (so a genuinely-good single night, whose removal gains nothing, is kept)."""
    # The keep-rule requires linking strictly MORE obs, which is impossible when
    # nothing is currently excluded — so the (expensive, per-night) trials would be
    # pure waste. Skip entirely if every obs is already included.
    if best_mask.all():
        return best_state, best_mask
    used = int(best_mask.sum())
    for (cs, ce) in _clusters(ev.times_tdb):
        if not _is_single_night(ev, cs, ce) or not best_mask[cs:ce+1].any():
            continue
        # leverage gate: a night too small a fraction of the data can't drag the orbit,
        # so dropping it can never redeem obs -- skip the (full-refit) trial.
        if int(best_mask[cs:ce+1].sum()) < DROP_SUSPECT_MIN_FRAC * used:
            continue
        trial = best_mask.copy()
        trial[cs:ce+1] = False                       # hypothesis: this night is bad
        if int(trial.sum()) < FIT_MIN_OBS:           # dropping it underdetermines the orbit
            continue                                 # (sparse arc, e.g. a 12-obs TNO) -- not a
                                                     # testable mislinkage hypothesis; skip
        state, _ = ev.fit(best_state, trial, max_nfev=EXTEND_MAX_NFEV)
        span0 = _included_span(ev, trial)
        for _ in range(2):                           # redeem now-fitting obs / reject bad
            if auto_reject_within_arc(ev, trial, state, orig_span=span0) <= 0:
                break
            trial[cs:ce+1] = False                   # keep the suspect night OUT
            if int(trial.sum()) < FIT_MIN_OBS:       # rejection thinned it below a fittable arc
                break
            state, _ = ev.fit(state, trial, max_nfev=EXTEND_MAX_NFEV)
        trial[cs:ce+1] = False
        gain = int(trial.sum()) - int(best_mask.sum())
        rms_n = ev.rms_norm(state, trial)
        keep = gain > 0 and rms_n < ACTIVATE_KEEP_NORM
        if verbose:
            print(f"   drop suspect single-night obs[{cs}..{ce}] ({ce-cs+1} obs): "
                  f"net {gain:+d} obs -> {int(trial.sum())} total, rms={rms_n:.2f}σ -> "
                  f"{'DROP IT' if keep else 'keep'}")
        history.append(_hsnap({"stage": f"drop1night[{cs}:{ce}]", "n_obs": int(trial.sum()),
                        "rms": ev.rms_arcsec(state, trial),
                        "score": _score(ev, state, trial), "kept": bool(keep)}, state, trial))
        if keep:
            best_state, best_mask = state.copy(), trial.copy()
    return best_state, best_mask


RMS_DROP_TRIGGER = 1.0       # arcsec: a converged fit below this is "clean" -- _is_clean uses
                             # it to skip the whole reject/redeem stage (a good fit is always
                             # well under the 0.6" "link OK" bar, so it's never disturbed).
