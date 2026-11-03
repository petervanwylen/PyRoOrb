"""Fetch real astrometry from the MPC observations API, fit an orbit, report elements.

Usage: pyroorb "2023 QJ3"   (or python -m pyroorb.fit_object "2023 QJ3")

Reuses the proven geometry/propagation from forward_model (forward model validated to
0.018" against Horizons). Generalizes two things for real data:
  - MPC observatory codes -> geocentric position via the MPC parallax table (ObsCodes)
  - many codes / long arcs (not just site 568 / 3 days)
Force model: ASSIST (JPL DE440 planets + GR, optionally the 16 asteroid perturbers), plus
non-gravitational terms when the fit earns them (see force_assist and nongrav).
"""
import os, sys, math
if __name__ == "__main__":
    # Run as a program (`python -m pyroorb.fit_object`): load the `.env` settings file before
    # the engine imports below read their settings.
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.console import utf8_stdio
    utf8_stdio()
import time as _stime
from pathlib import Path
import numpy as np
from astropy.time import Time

from pyroorb.forward_model import los_unit, build_base_sim
from pyroorb.obs80 import pack_designation, cache_stem
from pyroorb.iod import initial_orbit, select_subarc, best_seed_subarc
from pyroorb.arc_extend import (Evaluator, attempt_extensions, _clusters, _stadd, _soft_polish,
                        _hsnap, _clamp, FIT_MIN_OBS, _link_oppositions,
                        auto_reject_within_arc, _final_redeem, _relax_overconfident_obscodes,
                        _included_span, ACTIVATE_NFEV)
from pyroorb.orbit_common import IDENT_FAR_MIN_Q, ident_requested
from pyroorb.orbit_common import night_index
from pyroorb.weights import (sigma_for, overobserving_factors,
                     apparent_sky_motion, timing_inflated_sigmas)
from pyroorb import photometry
from pyroorb import datadirs
from pyroorb.mpc_orbit import fetch_mpc_orbit        # published-orbit seed for the well-observed fast path
# Leaf modules this engine builds on: observer geometry / backend pick (obs_sources), the
# fit-refusal guards (guards), and the classical-element math (orbital_elements).
from pyroorb.obs_sources import observer_positions_batch, _force_backend
from pyroorb.forcemodel import PinnedForce, pin, is_assist, excluded_body, pinned_asteroids
from pyroorb.guards import assert_not_pluto, self_perturber_name
from pyroorb.orbital_elements import (element_vector, state_eq_from_elements, ELEM_NAMES,
                              _ANGLE_IDX, earth_ecliptic_elements, n_oppositions)

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Orbit fit  (range-search IOD multi-start + LM differential correction)
# ---------------------------------------------------------------------------
def fit_orbit(obs, obscodes, desig="object", meta=None, verbose=False, force=False,
              auto_nongrav=None, _allow_sn_drop=True, mpc_seed=None, center=None,
              auto_center=None):
    """IOD (find_orb-style cascade) -> arc extension -> differential correction.

    Default ("regular") mode: select the best starting sub-arc with the quick
    nights*3+span heuristic, then extend/link. With `force=True`, if regular mode
    fails to link (an apparition is left with no included obs), brute-force it:
    solve EACH apparition alone, rank by semimajor-axis sigma (extensibility), and
    re-run the full link from the top three starts, keeping whichever links the
    most observations.

    `force=True` is an EXPERIMENTAL tool, deliberately NOT exposed in the web UI. It
    will drive arcs together (re-anchoring + dropping whichever nights resist) until
    SOMETHING links -- which means it can also manufacture a low-rms fit for two arcs
    that are NOT the same object (e.g. two members of the same dynamical family with
    look-alike osculating elements: it gluts a clean-looking orbit by amputating the
    observations that expose the mismatch). A clean `force` rms is therefore NOT
    evidence of a real identification -- use it to probe a hypothesis, then confirm
    the link honestly (residual structure, ident_check.systematic_misfit), never to
    declare a match. The default path is the one that decides whether obs belong
    together."""
    # Hard backstop: refuse minor planet (134340) Pluto in EVERY form -- typed name /
    # number, or self-identified by number in the fetched/pasted obs. It is a DE440
    # perturber, so it cannot be fit unperturbed by itself. Every fit path (CLI, web
    # designation, web paste, add-to-fit) funnels through here, so this one check
    # cannot be bypassed even if an upstream pre-check is missed. See is_pluto().
    assert_not_pluto(desig, meta, obs)
    # SELF-PERTURBER guard: one of the 16 sb441-n16 asteroids cannot be fit in a force field
    # that contains IT (guards.self_perturber_name; force_assist.perturbers_off_for) -- (1)
    # Ceres fit against its own mass kept 19 of 7993 observations while the modern arc read
    # ~1e5 sigma. Re-enter the fit with the object EXCLUDED from the perturber field (the
    # other 15 stay: (4) Vesta is fit with Ceres); the geometry build then lands an excluding
    # backend (force_assist.SelfExcludedForce) in the ctx, so every later step that reuses the
    # fit -- the asteroid polish, the web toggle, an ephemeris -- keeps the object out of its
    # own field. Guarded on suppressed_perturber() so the re-entry happens once.
    _sp = self_perturber_name(desig, meta, obs)
    _fa_sp = None
    if _sp is not None:
        try:
            import pyroorb.force_assist as _fa_sp
        except Exception:                 # no ASSIST -> no perturber kernel -> nothing to suppress
            _fa_sp = None
    if _fa_sp is not None and _fa_sp.suppressed_perturber() is None:
        with _fa_sp.perturbers_off_for(_sp):
            f = fit_orbit(obs, obscodes, desig=desig, meta=meta, verbose=verbose, force=force,
                          auto_nongrav=auto_nongrav, _allow_sn_drop=_allow_sn_drop,
                          mpc_seed=mpc_seed, center=center, auto_center=auto_center)
        f.self_perturber = _sp
        if verbose:
            print(f"  ({_sp}) is itself one of the 16 sb441-n16 perturbers; "
                  f"excluded from the perturber field (it cannot perturb itself)")
        return f
    # IDENTIFICATION hypothesis ("is A = B?", >= 2 designations requested -- CLI ident-check, a
    # web designation list, a multi-designation paste): fit the WHOLE thing with the 16
    # asteroid perturbers ON, no gate. The gated polish (refine_asteroids) runs after the
    # gravity linker has already decided which apparitions join, so a main-belt pair whose
    # arcs only meet under the perturbers is refused apparition by apparition -- a false "no".
    # So these always run +16ast, never gated. Same re-entry shape as the self-perturber guard
    # above; inside it (one of the 16 in the request) this means +15ast, the object excluded.
    if ident_requested(meta):
        try:
            import pyroorb.force_assist as _fa_id
        except Exception:                 # no ASSIST -> rebound, no perturbers to force
            _fa_id = None
        if (_fa_id is not None and _fa_id.forced_perturbers() is None
                and _fa_id.ensure_asteroid_kernel(verbose=verbose) is not None):
            with _fa_id.perturbers_on_for("ident"):
                f = fit_orbit(obs, obscodes, desig=desig, meta=meta, verbose=verbose,
                              force=force, auto_nongrav=auto_nongrav,
                              _allow_sn_drop=_allow_sn_drop, mpc_seed=mpc_seed, center=center,
                              auto_center=auto_center)
                on = _fa_id._use_asteroids(True)  # False only if a self-exclusion can't be built
            if on:
                if getattr(f, "_ctx", None) is not None:
                    f._ctx["meta"] = {**(f._ctx.get("meta") or {}), "asteroids": True}
                f.asteroids = True
            if verbose:
                print("  identification fit: asteroid perturbers "
                      + ("on" if on else "off (self-perturber exclusion unavailable)"))
            return f
    # Too few observations to determine anything: 6 orbital parameters need >= 3 obs
    # (2 residuals each), and below that the IOD cascade hands a degenerate geometry to
    # the two-body solvers -- a single pasted obs surfaced as "fit failed:
    # ZeroDivisionError: float division by zero" out of the Lambert solver. Say what is
    # actually wrong instead. (Guarded here, the ONE door every fit path -- CLI, web
    # designation, web paste, add-to-fit, ITF trial -- goes through.)
    if len(obs) < FIT_MIN_OBS:
        raise ValueError(f"need at least {FIT_MIN_OBS} observations to determine an orbit "
                         f"(got {len(obs)})")
    # A satellite does not out-gas and is not in our asteroid-Yarkovsky regime; its heliocentric
    # orbit (it moves with the planet) can trip the heliocentric comet/Yark gates falsely. Disable
    # auto nongrav discovery on the told-the-center path unless the caller explicitly asked for it.
    if center is not None and auto_nongrav is None:
        auto_nongrav = False
    def _finalize(f):
        if _allow_sn_drop:
            f = _rescue_single_night_poison(f, obs, obscodes, desig, meta, force, verbose)
        f = _prefer_bound_ecc(_final_soft_polish(f), verbose)
        # LAST DITCH, and it must be last: only a fit still catastrophic after every rescue above
        # (notably _rescue_single_night_poison, which routinely turns a 20" fit into a good one)
        # has anything to gain here, and running it earlier PREEMPTS those rescues with a worse
        # answer -- measured: 2024 LW 44/45 @ 0.392" -> 33/45 @ 0.345" when this ran first.
        f = _reject_catastrophic_link(f, obs, obscodes, desig, meta, verbose)
        f = _append_final_stage(f)
        # Automatically decide whether a NON-GRAVITATIONAL model is statistically warranted (chi2
        # effect-size + F-test + physical plausibility). Two parallel systems run here, both ON by
        # default and both controlled by the explicit `auto_nongrav` argument when given:
        #   1. COMET out-gassing (discover_comet_nongrav) -- triggered by a C//P/ designation or a
        #      cometary Tisserand parameter; the Marsden A1/A2/A3 sublimation model. PYROORB_AUTO_COMET_
        #      NONGRAV=0 to disable.
        #   2. asteroid Yarkovsky/SRP (discover_nongrav) -- gated q<1.4 au, multi-apparition;
        #      validated zero-false-positive against JPL. PYROORB_AUTO_NONGRAV=0 to disable.
        # The comet path runs FIRST and WINS the overlap: if a comet model is applied, the asteroid
        # discovery is skipped (a low-q comet would otherwise also trip the asteroid gate). If the
        # comet path only TRIGGERS but declines (no comet model justified), the asteroid path still
        # gets its chance -- a low-q object can be Yarkovsky-not-outgassing.
        do_comet = (auto_nongrav if auto_nongrav is not None
                    else os.environ.get("PYROORB_AUTO_COMET_NONGRAV", "1") != "0")
        applied_comet = False
        ng_disc = {}                       # the discovery record, attached to the delivered orbit below
        if do_comet:
            try:
                f_c, cinfo = discover_comet_nongrav(f, verbose=verbose)
                ng_disc["comet"] = cinfo
                if cinfo.get("applied"):
                    f, applied_comet = f_c, True
            except Exception as ex:
                if verbose:
                    print(f"  discover_comet_nongrav skipped: {type(ex).__name__}: {ex}")
        do_auto = (auto_nongrav if auto_nongrav is not None
                   else os.environ.get("PYROORB_AUTO_NONGRAV", "1") != "0")
        if do_auto and not applied_comet:
            try:
                f, ainfo = discover_nongrav(f, verbose=verbose)
                ng_disc["asteroid"] = ainfo
            except Exception as ex:
                if verbose:
                    print(f"  discover_nongrav skipped: {type(ex).__name__}: {ex}")
        if ng_disc:
            f.nongrav_discovery = ng_disc
        # AUTO-DETECT a natural-satellite (planetocentric) orbit -- ON BY DEFAULT.
        # Cheap: gated on a poor heliocentric rms, so a normal object costs one comparison and
        # returns unchanged (0 of the 142 regression objects trip the gate -> byte-identical).
        # Only on the heliocentric path (`center is None`); the recursive re-fit autodetect runs
        # sets center, so this never recurses. PYROORB_AUTO_CENTER=0 or auto_center=False disables it.
        if center is None:
            do_auto_center = (auto_center if auto_center is not None
                              else os.environ.get("PYROORB_AUTO_CENTER", "1") != "0")
            if do_auto_center:
                try:
                    f = autodetect_center(f, obs, obscodes, desig, meta, verbose)
                except Exception as ex:
                    if verbose:
                        print(f"  autodetect_center skipped: {type(ex).__name__}: {ex}")
        # RADAR delay/Doppler rows: when the fetch attached radar astrometry
        # (meta['radar'], from the JPL radar dump or a user file's R/r + ADES radar records),
        # fold it into a final differential correction (radar.refine_radar — the
        # refine_asteroids pattern; find_orb carries radar in every full_improvement, but its
        # IOD/Herget stages exclude it just as our sky-plane pipeline does). Opt-out:
        # PYROORB_RADAR=0. Offline fixtures carry no meta['radar'] -> suite byte-identical.
        radar_rows = (meta or {}).get("radar") or []
        if radar_rows and os.environ.get("PYROORB_RADAR", "1") != "0":
            try:
                import pyroorb.radar as _radar
                f = _radar.refine_radar(f, radar_rows, obscodes, verbose=verbose)
            except Exception as ex:
                if verbose:
                    print(f"  radar refine skipped: {type(ex).__name__}: {ex}")
        # LAST: re-judge the exclusions against the finished orbit -- the two
        # steps above are exactly the late movers whose improvements the rejection loop
        # never saw. No-op unless one of them actually moved the orbit.
        f = _final_reclaim(f, radar_rows, obscodes, verbose)
        # The delivered model must survive the unified verification at the
        # FINISHED orbit -- the reclaim's delivery-grade rebuild is what exposes it (2012 VP6).
        f = _delivered_verification(f, radar_rows, verbose)
        return f
    # PLANETOCENTRIC ("told-the-center") path: the user asserted this object
    # orbits a specific planet (--center saturn). Seed the fit with a planet-hinted IOD -- which
    # resolves the topocentric-range degeneracy ordinary angles-only IOD cannot -- then run the
    # same LM + drop/reclaim rejection as every other path. The heliocentric N-body fit itself is
    # UNCHANGED (the planet is already a perturber); only the seed differs, and the elements are
    # reported planetocentrically downstream (planetocentric.report). fit.center records the center.
    if center is not None:
        f = _fit_from_planet_seed(obs, obscodes, desig, meta, verbose, center)
        f = _finalize(f)
        import pyroorb.planetocentric as _pc
        f.center = _pc.resolve_planet(center)
        return f
    # FAST PATH: a well-observed, multi-opposition object already has a published MPC orbit
    # better than any IOD seed -- seed straight from it and run only drop/reclaim, skipping
    # the slow IOD/subarc/extend cascade. The seed is fetched OUTSIDE the fit (the production
    # layer calls fetch_mpc_seed and passes it as `mpc_seed`), so the fit stays deterministic
    # and the offline regression suite -- which never supplies a seed -- keeps the IOD path.
    # Falls back to IOD on any failure (no seed, non-finite state). See _fit_from_mpc_seed.
    if mpc_seed is not None and _qualifies_for_mpc_seed(obs, meta):
        try:
            fast = _fit_from_mpc_seed(obs, obscodes, desig, meta, verbose, mpc_seed)
            if fast is not None:
                return _finalize(fast)
        except Exception as ex:
            if verbose:
                print(f"  MPC-seed fast path failed ({type(ex).__name__}: {ex}); "
                      f"falling back to IOD")
    # IDENT ATTRIBUTION fast path (find_orb-style identification): in an explicit ident hypothesis
    # (n_requested>1) where one requested designation is independently WELL-DETERMINED (>= 5 opp),
    # attribute the other designation(s) to its precise orbit -- keep the obs that fit it, exclude the
    # rest -- instead of grinding a joint IOD/multi-rev bridge over the combined arc. The verdict is a
    # residual comparison against an authoritative orbit, so a doomed non-match of two dynamically
    # similar bodies (2016 UK176 vs 2008 VC46, ~8.8 deg apart) resolves in ~0.2s not ~52s, while a
    # genuine ident still keeps every obs (2014 EF94 = 1999 TC333 = 2025 UO106 -> 111/111). Skipped
    # under --force (grind the hypothesis as asked). See _ident_attribute_to_anchor.
    if not force:
        try:
            attr = _ident_attribute_to_anchor(obs, obscodes, desig, meta, verbose)
        except Exception as ex:
            attr = None
            if verbose:
                print(f"  ident-attribution skipped ({type(ex).__name__}: {ex}); falling back to IOD")
        if attr is not None:
            return _finalize(attr)
    fit = _fit_from_start(obs, obscodes, desig, meta, verbose)
    # Auto-recover a poisoned IOD seed (a bad night left a whole apparition unlinked)
    # with the cheap drop-suspect-night screen even without --force -- restoring
    # resilience to a contaminated starting arc. The expensive multi-start stays
    # --force-only (see part 2 below).
    if not (force or _still_troubled(fit, obs)):
        return _finalize(fit)


    # The fit is "troubled" only because an explicit linkage request joined a period-locked
    # object to a DIFFERENT period-known designation that cannot honestly attach (the multi-rev
    # linker already gate-rejected it: arc_extend._ident_shortcircuit). That is a settled non-match,
    # not a poisoned seed -- re-anchoring on any apparition reaches the same verdict, so the whole
    # (expensive) recovery cascade below is futile. Return the clean single-object fit now; the
    # leftover designation stays unlinked, so ident_check.identify() reports "not the same". (Skipped
    # when --force: the user explicitly asked to grind a hypothesis.) See _ident_nonmatch.
    if not force and _ident_nonmatch(fit, obs, meta):
        if verbose:
            print("[ident] settled non-match (period-locked vs period-known foreign "
                  "designation); skipping the recovery cascade")
        return _finalize(fit)

    # A whole apparition simply cannot be added to an ALREADY-SOLID orbit. A 4+-opposition orbit is
    # essentially never a mislinkage -- a 3-opposition one can
    # occasionally be constructed, but a clean 4+-opp fit is trustworthy -- so once the fit WITHOUT
    # the leftover apparition is a solid 4+-opp orbit (>= DOOM_MIN_OPP oppositions, sub-DOOM_SOLID_RMS
    # residual, no systematic per-apparition trend), an apparition its own linker could not glue on (a
    # multi-revolution jump away) is a near-certain non-match -- doomed. The recovery cascade would
    # only re-run that same doomed multi-rev bridge from every re-anchor: the K10OD3Q+K06W78O 3-minute
    # grind, where the intruder tracklet even becomes a short-seed anchor and fans the bridge across
    # all 8 real apparitions. Trust the solid orbit, leave the apparition excluded for good, and
    # return now. This is the general (non-ident) sibling of _ident_nonmatch above: same "the linker's
    # verdict is already settled, re-anchoring can't change it" logic, keyed on the orbit's own
    # 4+-opp maturity instead of an explicit multi-designation request. (Skipped under --force: the
    # user explicitly asked to grind the hypothesis.) See _doomed_orphan.
    if not force and _doomed_orphan(fit, obs):
        if verbose:
            print(f"[doom] solid {fit.n_oppositions()}-opp fit ({fit.rms_residual():.3f}\") cannot "
                  f"absorb a stranded apparition; banning it for good, skipping recovery")
        return _finalize(fit)

    # --- FORCE MODE -----------------------------------------------------------
    # Four recovery stages, each more expensive than the last and each gated on the fit
    # still being troubled (an apparition unlinked, or a linked apparition discordant) so
    # a stage that already resolved the link short-circuits the rest. See the individual
    # _recover_* functions for the per-stage rationale and the objects that exercise them.
    times = np.array([o["jd_utc"] for o in obs])
    clusters = _clusters(times)
    if len(clusters) <= 1:
        return _finalize(fit)   # nothing to brute-force
    best = fit
    if verbose:
        print(f"[force] regular linked {best.n_observations()}/{len(obs)}; "
              f"brute-forcing over {len(clusters)} apparitions")

    # 1) RE-ANCHOR from a RANKED candidate list -- cheap anchor selection, tried FIRST. Runs
    #    only when the fit is genuinely STUCK (under --force, or a SUBSTANTIAL apparition still
    #    stranded). See _recover_ranked_anchors for why the ranking key is what it is.
    tried_anchors = set()
    if ((force or _still_troubled(best, obs))
            and (best.n_excluded() >= FORCE_REJECT_THRESHOLD or best.n_observations() < len(obs)//2)):
        best = _recover_ranked_anchors(best, obs, obscodes, desig, meta, times, clusters,
                                       verbose, tried=tried_anchors)

    # DEGENERATE MAIN-BELT non-match -- checked NOW, AFTER the cheap ranked re-anchor, not before
    # it. An explicit IDENT hypothesis whose joint LM forced two arcs into a garbage minimum (included
    # obs fit >> IDENT_DEGEN_RMS) is USUALLY a settled non-match, so we skip the EXPENSIVE remaining
    # stages (night-drop / short-seed / joint refit) -- the slow-non-match tail. But a genuine
    # ident can land there purely from a POISONED ANCHOR (the base sub-arc pinned the wrong orbit): the
    # cheap re-anchor above rescues it (2015 KP420 = 2025 NH546: base 463" -> re-anchor 0.106"/25-28,
    # a y2 that regressed to "no" when this gate fired BEFORE the re-anchor). So gate on the RE-ANCHORED
    # `best`: only a fit STILL degenerate after re-anchoring is genuinely doomed. Two conditions, both
    # load-bearing: (1) an IDENT request (n_requested>1) -- _degenerate_anchor_q reads an INDEPENDENT q
    # only when a separate designation exists (a lone designation reproduces the garbage fit, and a
    # poisoned single-object fit is cascade-rescuable -- grind-necessity); (2) MAIN-BELT
    # (q>=IDENT_FAR_MIN_Q) -- LOW-q NEO idents need the later stages (2021 WO). See
    # _ident_doomed_degenerate; the _rescue_single_night_poison companion guard mirrors it.
    if not force and _ident_doomed_degenerate(best, obs, obscodes, meta):
        if verbose:
            print(f"[ident] still-degenerate main-belt joint fit (rms {best.rms_residual():.0f}\") after "
                  f"re-anchor -- settled non-match, skipping night-drop/short-seed/grid")
        return _finalize(best)

    # 2) DROP-SUSPECT-NIGHT -- fallback, only if re-anchoring left an apparition stranded.
    if _still_troubled(best, obs):
        best = _recover_drop_night(best, fit, obs, obscodes, desig, meta, times, clusters, verbose)

    # 3) SHORT-SEED ANCHOR -- only if an apparition is STILL stranded. Stage 1 already tries the
    #    short sub-arcs it ranked highest; `tried_anchors` stops this stage re-fitting those
    #    (identical) anchors a second time, which is pure duplicated work on a doomed object.
    if _still_troubled(best, obs):
        best = _recover_short_seed(best, obs, obscodes, desig, meta, times, clusters, verbose,
                                   tried=tried_anchors)

    # 4) JOINT REFIT -- the plain full step over every observation, one LM. Cheap, and the thing
    #    none of the gated link machinery above actually tries. See _recover_joint_refit.
    if _still_troubled(best, obs):
        best = _recover_joint_refit(best, obs, obscodes, desig, meta, times, clusters, verbose)

    return _finalize(best)


CATASTROPHIC_RMS_N = 5.0  # NORMALIZED (sigma) rms: a delivered orbit fitting its OWN included obs
#                           this badly is not an orbit, it is a hypothesis that failed. The currency
#                           is deliberately sigma, not arcsec: an arcsec bar cannot tell a broken fit
#                           from honest large-sigma astrometry. Measured on the 2026-08-25 NEO sweep,
#                           objects sitting at 2-3" raw come back at rms_n 0.87-1.02 -- they fit their
#                           own reported uncertainties fine -- while 2018 SY2's contaminated
#                           two-apparition fit was 339" at rms_n 365 and 2003 HM is 4.8" at rms_n 9.3.
#                           An arcsec threshold flags the former and would need to be set so high it
#                           risks missing the latter.
#                           CALIBRATION (2026-08-25, the 2000-object NEO-gate sweep): rms_normalized
#                           runs p50 0.65, p90 0.85, p99 1.06, p99.5 1.09 -- and then NOTHING until
#                           8.5. Four objects in 1975 sit above it (8.5, 9.3, 365, 663), and the
#                           same four are above every bar from 2 to 8. So this threshold sits in a
#                           two-order-of-magnitude empty band: anything from ~1.5 to ~8 selects the
#                           identical set, which is why 5.0 is not delicate. See
#                           _reject_catastrophic_link.


def _reject_catastrophic_link(fit, obs, obscodes, desig, meta, verbose=False):
    """LAST-DITCH: when the delivered orbit fits its own included observations catastrophically,
    check whether ONE apparition alone does better, and keep whichever `_better_link` prefers.

    Nothing else asks this. Every recovery stage is gated on `_still_troubled` -- an apparition
    left UNLINKED, or a linked one discordant -- and a fit that has contorted itself to span a
    contaminant is troubled by neither test: both apparitions carry observations, so the cascade
    is skipped and a garbage orbit is delivered. 2018 SY2 is the case. The MPC knows it as a
    29-observation, 27-day, ONE-opposition object (q=0.477, e=0.384, i=26.8, U=0); the obs API
    serves 6 further observations from 2020 under the same designation that are not this body.
    `iod.initial_orbit`'s vaisala-link accepted a gap-spanning seed 4104" (1.1 deg) off -- inside
    the generous LINK_ACCEPT of 5000" -- which committed the fit to a two-apparition hypothesis,
    and out came a=3.0, e=0.85, i=67 at **339.5"** keeping 25 of 35 obs. Fitting the 2018
    apparition alone gives 29/29 at 0.444" and reproduces the MPC's elements to four decimals.

    Note which way the comparison falls: the clean fit has MORE observations (29 > 25) AND a far
    better rms, so `_better_link` -- the same gate every recovery stage uses -- prefers it
    outright. The defect was never the comparison, it was that nobody ever offered the
    alternative. So this only has to construct it.

    Costs nothing on a healthy fit: gated on CATASTROPHIC_RMS_N, which a delivered orbit clears by
    two orders of magnitude in normal operation."""
    try:
        if fit.rms_normalized() <= CATASTROPHIC_RMS_N:
            return fit
        times = np.array([o["jd_utc"] for o in obs])
        clusters = _clusters(times)
        if len(clusters) < 2:
            return fit
        if getattr(fit, "iod_method", None) == "mpc" and getattr(fit, "_ctx", None):
            return _catastrophic_seeded_alternative(fit, obs, clusters, verbose)
        best = fit
        # Try each apparition alone, richest first -- the contaminant is usually the sparse one.
        for (s, e) in sorted(clusters, key=lambda c: c[1] - c[0], reverse=True):
            keep = [i for i in range(len(obs)) if s <= i <= e]
            if len(keep) < FIT_MIN_OBS:
                continue
            try:
                cand = _fit_from_start([obs[i] for i in keep], obscodes, desig, meta, False)
            except Exception:
                continue
            full = _represent_over_full_arc(cand, obs, keep, obscodes, desig, meta, verbose) or cand
            if _better_link(full, best):
                if verbose:
                    print(f"[catastrophic] delivered fit was {best.rms_residual():.1f}\" "
                          f"({best.rms_normalized():.1f} sigma); apparition "
                          f"obs[{s}..{e}] alone gives {full.n_observations()}/{len(obs)} @ "
                          f"{full.rms_residual():.3f}\" -- keeping that", flush=True)
                best = full
        return best
    except Exception:
        return fit


def _catastrophic_seeded_alternative(fit, obs, clusters, verbose=False):
    """_reject_catastrophic_link's alternative for a fit SEEDED FROM A PUBLISHED ORBIT
    (iod_method 'mpc'): the same question -- does one apparition alone fit cleanly where
    the whole arc does not -- answered on the fit's own evaluator from its own state, one
    short LM per apparition, instead of an IOD-path `_fit_from_start` + full-arc
    re-presentation per apparition.

    Why the seeded route needs its own: on a strongly out-gassing comet the seeded
    GRAVITY fit is catastrophic by construction (10P 10.9", 103P, 4P, 21P) -- the
    non-gravitational decision has not been made yet -- so this stage fired on every
    one, and its per-apparition IOD fits cost 33-52 s of a 63-90 s fit (2026-09-06
    survey) to find nothing the seed did not already know. A published orbit vouches that
    the apparitions are one body, so a from-scratch IOD of each is not information; what
    the stage is for -- a delivered orbit that is not an orbit -- is answered just as
    well by a seeded per-apparition LM at ~0.3 s each.

    The comparison is ALSO the one the IOD route cannot make: `_better_link` demands
    strictly MORE observations, and a catastrophic full-arc fit always has more of them
    than any one apparition -- 21P (7432 of 7434 obs at 2622", a published orbit that
    cannot hold a 124-year arc gravity-only) kept the garbage over a 1246-obs
    apparition at 1.76" for exactly that reason. A fit past CATASTROPHIC_RMS_N is not an
    orbit (that is the gate's own definition), so here a CLEAN apparition beats it
    regardless of count; among clean apparitions the richest wins. The chosen
    apparition is delivered on the full evaluator with the rest EXCLUDED (the
    published-seed analogue of _represent_over_full_arc), so the record stays whole
    and the discovery that follows can still re-include what its model fits.

    Deliberately NOT re-anchored at the apparition's own epoch and NOT the most recent
    apparition: the everything-mask comet discovery that follows fits the FULL arc from this
    orbit, and that fit wants the mid-arc evaluator it has (103P from a 2023 anchor: 1.154"
    in 216 s against 0.968" in 41 s from mid-arc, measured 2026-09-06). Re-anchoring and the
    prefer-the-recent rule live in `_comet_extend_apparitions`, the path for an arc the model
    cannot hold whole, where they are what makes the extension converge."""
    ctx = fit._ctx
    ev = ctx["ev"]
    state0 = np.asarray(fit._state, float)[:6]
    n = len(obs)
    cands = []
    for (s, e) in sorted(clusters, key=lambda c: c[1] - c[0], reverse=True):
        if e - s + 1 < FIT_MIN_OBS:
            continue
        mask = np.zeros(n, dtype=bool)
        mask[s:e + 1] = True
        try:
            st, _ = ev.fit(state0, mask, max_nfev=ACTIVATE_NFEV)
            rms_n = float(ev.rms_norm(st, mask))
        except Exception:
            continue
        if np.isfinite(rms_n) and rms_n <= CATASTROPHIC_RMS_N:
            cands.append((int(mask.sum()), -rms_n, s, e, st))
    if not cands:
        return fit
    n_keep, neg_rms_n, s, e, st = max(cands)
    mask = np.zeros(n, dtype=bool)
    mask[s:e + 1] = True
    st2, result = ev.fit(st, mask, thin=False)
    full = _build_orbit(ctx, st2, mask, result, fit.history)
    if full.rms_normalized() > CATASTROPHIC_RMS_N:
        return fit
    if verbose:
        print(f"[catastrophic] delivered fit was {fit.rms_residual():.1f}\" "
              f"({fit.rms_normalized():.1f} sigma) on {fit.n_observations()}/{n} obs; "
              f"apparition obs[{s}..{e}] alone (from the published seed) gives "
              f"{full.n_observations()}/{n} @ {full.rms_residual():.3f}\" -- keeping that",
              flush=True)
    full.history.append(_hsnap({"stage": "catastrophic", "n_obs": int(mask.sum()),
                                "rms": full.rms_residual(),
                                "score": full.rms_normalized(), "kept": True}, st2, mask))
    return full


def _reanchor_apparition(fit, s, e, verbose=False):
    """Re-anchor `fit` (one apparition obs[s..e] included, the rest excluded) at that
    apparition's own mid-epoch and run the standard drop/reclaim settle there. Returns the
    re-anchored FittedOrbit, or None when the ctx carries no epoch-free geometry (`geom`,
    stashed by the seeded path) or the result is no better (keep-the-better on the
    normalized rms). The full obs list stays on the orbit so every mask downstream keeps its
    indexing; only the evaluator's epoch moves. Used by the apparition-wise comet extension
    (21P: the 2018 apparition on the 1962 full-arc evaluator never converged, 4197/4197 at
    1.69" with a Dec trend; at its own epoch it settles to 3662 at 1.29")."""
    try:
        ctx = fit._ctx
        geom = ctx.get("geom")
        if geom is None:
            return None
        obs = ctx["obs"]
        backend = ctx["backend"]
        times_tdb = np.asarray(ctx["times_tdb"], float)
        epoch_new = 0.5 * (float(times_tdb[s]) + float(times_tdb[e]))
        st = np.asarray(fit._state, float)[:6]
        seed = np.asarray(backend.predict_state(ctx["fit_base"], ctx["epoch_tdb"], st,
                                                epoch_new), float)
        if not np.all(np.isfinite(seed)):
            return None
        ev2, ctx2 = _build_evaluator_at(geom, obs, epoch_new)
        for k, v in ctx.items():                    # desig/meta/method/mpc_seed/... ride along
            ctx2.setdefault(k, v)
        ctx2["geom"] = geom
        mask = np.asarray(fit._mask, bool).copy()
        history = list(fit.history)
        state, result = ev2.fit(seed, mask)
        # orig_n: the mask is one apparition of the full record, so the rejection guard
        # must be measured against IT, not the evaluator's 7434 (see auto_reject).
        state, result = _reject_settle(ev2, state, mask, result, history,
                                       orig_n=int(mask.sum()))
        out = _build_orbit(ctx2, state, mask, result, history)
        if not np.isfinite(out.rms_normalized()) or out.rms_normalized() >= fit.rms_normalized():
            return None
        if verbose:
            print(f"  extend: re-anchored at the apparition's own epoch (JD {epoch_new:.1f}): "
                  f"{out.n_observations()}/{len(obs)} @ {out.rms_residual():.3f}\" "
                  f"({out.rms_normalized():.2f} sigma)", flush=True)
        return out
    except Exception:
        return None


def _still_troubled(fit, obs):
    """True when the fit still has an apparition with no included obs (unlinked) or a
    linked apparition whose residuals are discordant. This is the condition each
    force-mode recovery stage checks to decide whether the next, more expensive stage
    should run -- when a stage resolves the link, the remaining stages short-circuit."""
    return _has_unlinked_opposition(fit, obs) or _has_discordant_opposition(fit, obs)


# --- ANCHOR RANKING (force-mode recovery stage 1) ---------------------------------------------
# The candidate anchors are every apparition AND every apparition's short find_orb sub-arc
# (best_seed_subarc). Each is probed with a cheap extend=False fit (~10 integrations), then they
# are tried in ranked order with an early exit. Two things about the ORDER are measured, not
# assumed:
#
#   SHORT SUB-ARCS FIRST. A multi-revolution bridge sizes its search on the anchor covariance,
#   and on a well-determined single-apparition arc that covariance is UNDER-DISPERSED -- on
#   2008 NA the true semi-major axis sits 6.2-6.9 sigma_a from what each of the three long
#   anchors reports, so every search sized on them misses, while the one-night anchor (sigma_a
#   4.8e-03, honest) contains the truth at 0.1 sigma and links immediately. Same ~6x factor the
#   ITF cross-track split study measured for held-out formal sigmas.
#
#   THEN BEST OWN-ARC RMS. Among anchors of the same kind, the one that fits its own data best
#   is the better base; the probe has already computed it, so the key costs nothing.
#
# The second clause is a PROXY for the first -- "short" stands in for "loose, honest sigma" -- so
# the obvious objection is that ranking by sigma_a DESCENDING would express the theory directly.
# Measured (2026-08-25), and it does not: every alternative reaches the SAME fits, more expensively.
#
#   integrations   2008 NA   2023 VM6   1998 VD32   2021 WO   (TE4/TE10/BV14 tied)
#   short+rms         1554        286         251       275   <- shipped
#   sigma desc        2049        653         308       275
#   short+sigma       2049          -           -         -
#   rms alone         2229        286         295       351
#
# Why the proxy beats the theory's direct form: among the SHORT sub-arcs sigma_a is uniformly
# huge, so ordering by its magnitude is ordering by noise, while own-arc rms picks the better
# CONDITIONED seed among them. Sigma honesty decides which KIND of anchor to prefer; it does not
# rank within that kind.
#
# Ranking by sigma_a (what this stage used to do) is actively WRONG here: it puts the tightest --
# hence most over-confident -- anchor first, and on 2008 NA that is an anchor that cannot work,
# so the object was answered only by stage 3, ~4x of work later. Measured over the whole
# 183-fixture suite: -12.0% integrations, no fixture's fit changed for the worse, and the two
# that changed improved (1998 VD32 0.5241 -> 0.5151", 2026 NA1 0.5568 -> 0.5505").
#
# The old stage kept the per-apparition probes but ranked them by sigma_a and never considered a
# short sub-arc (that was stage 3, reached only after the night-drop screen). Merging the two
# candidate sets into ONE ranked list is what collapses 2008 NA from 5054 to 1159 integrations;
# stage 3 remains as the fallback for anchors this stage did not get to.
ANCHOR_TRY_N = 3          # candidate anchors given a full extend=True re-link before giving up


def _anchor_rank_key(p):
    """Short sub-arcs first, then best own-arc rms. See the block comment above."""
    return (not p["short"], p["rms"])


def _anchor_candidates(obs, times, clusters, rhohats, Robs):
    """Every candidate anchor: each apparition, plus each apparition's short find_orb sub-arc
    (best_seed_subarc) where that differs. Returns [(s, e, is_short)]."""
    out = []
    for (s, e) in clusters:
        # Skip a NEAR-SIMULTANEOUS tracklet (no angular baseline): a cluster spanning only
        # minutes can't constrain a Gauss triangle, so its IOD seed is a degenerate
        # near-parabolic orbit whose integration crawls (an effective hang). PAIR_2016HH23
        # carries a 3-obs/10-minute cluster; without this guard probing it spins for minutes.
        if (times[e] - times[s]) >= PROBE_MIN_ANCHOR_DAYS:
            out.append((s, e, False))
        if times[e] - times[s] >= 0.5:            # multi-night: ask for an internal sub-arc
            t0 = Time(times[s], format="jd", scale="utc").tdb.jd
            try:
                ss, ee = best_seed_subarc(obs, times, rhohats, Robs,
                                          build_base_sim(t0), t0, s, e)
            except Exception:
                continue
            if (ss, ee) != (s, e) and (times[ee] - times[ss]) >= PROBE_MIN_ANCHOR_DAYS:
                out.append((ss, ee, True))
    return out


def _recover_ranked_anchors(best, obs, obscodes, desig, meta, times, clusters, verbose,
                            tried=None):
    """Force-mode recovery stage 1: probe every candidate anchor cheaply, then re-link the full
    fit from the best-ranked few. Records each anchor it fully tried in `tried` so stage 3 does
    not repeat it. Replaces the old sigma_a multi-start (see the block comment above)."""
    rhohats = [los_unit(o["ra_deg"], o["dec_deg"]) for o in obs]
    # Only observer positions are needed to pick sub-arcs, so use the epoch-free geometry
    # rather than building a full base_sim + Evaluator just to read Robs.
    Robs = _observer_geometry(obs, obscodes)["Robs"]
    probes = []
    for (s, e, short) in _anchor_candidates(obs, times, clusters, rhohats, Robs):
        try:
            p = _fit_from_start(obs, obscodes, desig, meta, False, start=(s, e), extend=False)
        except Exception:
            continue
        probes.append({"start": (s, e), "short": short, "rms": p.rms_residual()})
    probes.sort(key=_anchor_rank_key)
    for p in probes[:ANCHOR_TRY_N]:
        s, e = p["start"]
        try:
            if verbose:
                print(f"[force] try start obs[{s}..{e}] "
                      f"({'short' if p['short'] else 'full '} anchor, own rms={p['rms']:.3f}\")")
            cand = _fit_from_start(obs, obscodes, desig, meta, verbose, start=(s, e), extend=True)
            if tried is not None:
                tried.add((s, e))
            if _better_link(cand, best):
                best = cand
            # early exit: once a seed links EVERY apparition and the fit isn't a wrong-period
            # link (no stranded apparition, not discordant), the remaining seeds can only
            # re-derive the same orbit -- stop rather than re-run the expensive bridge from each.
            if not _still_troubled(best, obs):
                break
        except Exception:
            pass
    return best


def _recover_drop_night(best, fit, obs, obscodes, desig, meta, times, clusters, verbose):
    """Force-mode recovery stage 2: DROP-SUSPECT-NIGHT (fallback, only when re-anchoring left
    an apparition stranded). A single mislinked night -- esp. one inside the chosen sub-arc --
    poisons the seed so nothing links. When stage 1 already linked everything (TE4 / VD32 /
    VM6 / UK199 / HH23 all resolve there now), this is skipped -- the screen was the dominant
    wasted cost on the hard recovery objects. SCREEN each candidate fast (adjacent extension
    only, no cluster activation), then do the full link only on the night-drop that screens
    best. Gating on the actual trouble rather than the bare `force` flag skips ~8 wasted
    full-fit night-trials apiece (VM6 1286->605 sweeps, UK199 552->164, RH119 379->155;
    ~1300 suite sweeps, all at byte-identical rms)."""
    s0, e0 = select_subarc(times, [los_unit(o["ra_deg"], o["dec_deg"]) for o in obs])
    best_drop, best_screen = None, best.n_observations()
    for grp in _suspect_night_groups(times, clusters, (s0, e0)):
        reduced = [o for i, o in enumerate(obs) if i not in grp]
        if len(reduced) < 6:
            continue
        try:
            scr = _fit_from_start(reduced, obscodes, desig, meta, False, activate=False)
            if verbose:
                print(f"[force]   screen drop {len(grp)}-obs night -> {scr.n_observations()}")
            if scr.n_observations() > best_screen and scr.rms_residual() <= max(fit.rms_residual(), 0.6):
                best_screen, best_drop = scr.n_observations(), grp
                if best_screen >= 0.9 * len(obs):      # this drop already links ~everything
                    break                              # -- no need to screen the rest
        except Exception:
            pass
    if best_drop is not None:
        reduced_full = [i for i in range(len(obs)) if i not in best_drop]   # reduced[j] == obs[reduced_full[j]]
        reduced = [obs[i] for i in reduced_full]
        cand = _fit_from_start(reduced, obscodes, desig, meta, verbose)
        if _better_link(cand, best):
            if verbose:
                print(f"[force] dropping the {len(best_drop)}-obs suspect night links "
                      f"{cand.n_observations()}")
            best = _represent_over_full_arc(cand, obs, reduced_full, obscodes, desig, meta,
                                            verbose) or cand
    return best


def _represent_over_full_arc(cand, obs, reduced_full, obscodes, desig, meta, verbose=False):
    """Re-present a fit found on a REDUCED obs list (`cand`) as a fit over the FULL `obs`,
    with the dropped obs -- and cand's own rejections -- pinned OFF. Returns the full-arc
    FittedOrbit, or None to leave the caller with `cand`.

    Why this is not presentation polish: a fit whose `_obs` is the reduced list has no
    record of the dropped observations AT ALL. They are absent from n_total_observations(),
    from excluded_observations() (so no 'x rejected' line, no O-C, nothing in the web obs
    table), and from ident_check's `except:` echo -- the fit reports '45/45 obs' for an arc
    that was fetched with 51, and the provenance of the other 6 is gone. Pinning them off
    instead keeps the input set whole and states plainly which obs the orbit does not use.

    The re-fit is seeded with cand's own state over cand's own kept obs, so the orbit is the
    one already accepted; only obs the fit does not use are re-attached. If it fails or comes
    back materially worse (it should be identical to rounding), we return None rather than
    trade the accepted orbit for a bookkeeping improvement."""
    cand_mask = cand.is_included()
    excluded_full = ({i for i in range(len(obs)) if i not in reduced_full}
                     | {reduced_full[j] for j in range(len(reduced_full)) if not cand_mask[j]})
    try:
        full = _full_fit_excluding(
            cand._state, cand.epoch_tdb, obs, obscodes, desig, meta, excluded_full,
            method=cand.iod_method,
            history=_expand_history_masks(getattr(cand, "history", None), reduced_full, len(obs)))
    except Exception:
        return None
    if full is None or full.n_observations() != cand.n_observations():
        return None                      # the re-fit clipped/reclaimed obs -> not the same orbit
    if full.rms_residual() > cand.rms_residual() * 1.02 + 1e-9:
        return None                      # materially worse -> keep the accepted reduced fit
    if verbose:
        print(f"[force]   re-presented over the full arc: {full.n_observations()}/"
              f"{full.n_total_observations()} obs ({full.n_excluded()} excluded) "
              f"@ {full.rms_residual():.3f}\"")
    return full


def _recover_short_seed(best, obs, obscodes, desig, meta, times, clusters, verbose, tried=None):
    """Force-mode recovery stage 3: SHORT-SEED ANCHOR (last resort, only if an apparition is
    STILL stranded). Stage 1 anchors each apparition on its WHOLE arc; a long apparition (a
    ~70 d window near a slow NEO's stationary loop) seeds a revolution-wrapped orbit, so its
    probe diverges and the far apparition stays unlinked (2021 WO: 11/83, the 2018 apparition
    stranded 3 yr / ~N revs from the 2021-22 arc). Re-anchor each multi-night apparition on a
    SHORT internal sub-arc (find_orb's arc-shrink, best_seed_subarc) whose Gauss-friendly seed
    extends cleanly across the gap (-> 83/83 @ 0.43"). Gated on continued failure, so any
    object the earlier stages already linked is never reached."""
    rhohats = [los_unit(o["ra_deg"], o["dec_deg"]) for o in obs]
    # Only observer positions are needed here (each per-apparition sub-arc seed builds its own
    # base_sim at its own epoch below), so use the epoch-free geometry rather than building a
    # full base_sim + Evaluator just to read Robs.
    Robs = _observer_geometry(obs, obscodes)["Robs"]
    for (s, e) in clusters:
        if times[e] - times[s] < 0.5:        # single-night apparition: no internal sub-arc
            continue
        t0 = Time(times[s], format="jd", scale="utc").tdb.jd
        ss, ee = best_seed_subarc(obs, times, rhohats, Robs, build_base_sim(t0), t0, s, e)
        if (ss, ee) == (s, e):
            continue                          # apparition already short / no better window
        if tried and (ss, ee) in tried:
            continue                          # stage 1 already re-linked from this exact anchor
        try:
            if verbose:
                print(f"[force] short-seed anchor obs[{ss}..{ee}] within app[{s}..{e}]")
            cand = _fit_from_start(obs, obscodes, desig, meta, verbose, start=(ss, ee), extend=True)
            if _better_link(cand, best):
                best = cand
            if not _still_troubled(best, obs):
                break
        except Exception:
            pass
    return best


def _recover_joint_refit(best, obs, obscodes, desig, meta, times, clusters, verbose):
    """Force-mode recovery stage 4: THE PLAIN FULL STEP -- one unmasked differential correction
    over every observation, seeded from the stranded orbit and re-anchored at the mid-arc epoch.

    Nothing else in the pipeline does this. Every path that brings a far apparition in is GATED
    machinery: cluster activation refits and REVERTS on a normalized-rms test, `_homotopy_glue`
    ramps the new apparition's weight so it cannot yank the anchor, the multi-rev bridge seeds
    from swarms and slides. Each of those exists for a good reason -- they stop a wrong
    apparition dragging a good orbit -- but they all judge the link from INSIDE their own
    construction, and a construction can converge somewhere the plain fit does not. On 2016 BV14
    the homotopy settles with the far apparition at 3.079", is rejected by LINK_NEW_TOL=3.0 by
    eight hundredths of an arcsecond, and the object then needed a 340-cell (a, M) grid to
    rediscover what one unmasked LM from the same orbit reaches in FIVE integrations: 36/36 at
    0.209". This is find_orb's own habit -- `attempt_extensions` toggles observations on and
    calls `full_improvement` over the union; it has no weight continuation at all.

    Safety is the acceptance test, not the construction: the candidate goes through the same
    `_reject_settle` rounds and the same `_better_link` gate as every other recovery stage, so a
    joint fit that drags the anchor or fails to fit what it just linked is discarded. Cheap
    enough (one LM) to try before the grid on every troubled fit."""
    src = getattr(best, "_ctx", None)
    if src is None:
        return best
    geom = _observer_geometry(obs, obscodes)
    ttdb = np.asarray(geom["times_tdb"], float)
    # Mid-arc epoch: a Cartesian state expressed far from its data is ill-conditioned (the
    # velocity Jacobian columns scale with the epoch->obs leverage), which is the same reason
    # the grid re-anchors there. Cheap: one propagation.
    epoch = 0.5 * (float(ttdb.min()) + float(ttdb.max()))
    backend = geom["backend"]
    seed = np.asarray(backend.predict_state(
        backend.build_base_sim(src["epoch_tdb"]), src["epoch_tdb"],
        np.asarray(best._state, float), epoch), float)
    if not np.all(np.isfinite(seed)):
        return best
    ev, ctx = _build_evaluator_at(geom, obs, epoch)
    mask = np.ones(len(obs), dtype=bool)
    _ts = _stime.perf_counter()
    state, result = ev.fit(seed, mask, thin=False)
    if not np.all(np.isfinite(state)):
        _stadd("recover", _ts)
        return best
    history = [_hsnap({"stage": "joint-refit", "n_obs": int(mask.sum()),
                       "rms": ev.rms_arcsec(state, mask),
                       "score": ev.rms_norm(state, mask)}, state, mask)]
    state, result = _reject_settle(ev, state, mask, result, history)
    _stadd("recover", _ts)
    ctx.update(desig=desig, meta=meta, seed_d=0.0, method="jointrefit")
    cand = _build_orbit(ctx, state, mask, result, history)
    if verbose:
        print(f"[force] joint refit (plain full step): opp={cand.n_oppositions()} "
              f"{cand.n_observations()}/{len(obs)} rms={cand.rms_residual():.3f}\"", flush=True)
    return cand if _better_link(cand, best) else best


def _suspect_night_groups(times, clusters, subarc):
    """Candidate nights to hypothesize-drop in --force: every single-night
    apparition, plus each night within the default sub-arc (the seed is most
    poisoned by a contaminant there). Returns a list of frozensets of obs indices."""
    t = np.asarray(times)
    groups = []
    def nights(s, e):
        out, ns = [], s
        for k in range(s, e):
            if t[k+1] - t[k] > 0.5:
                out.append(frozenset(range(ns, k+1))); ns = k+1
        out.append(frozenset(range(ns, e+1)))
        return out
    for (cs, ce) in clusters:
        if t[ce] - t[cs] < 0.5:              # single-night apparition
            groups.append(frozenset(range(cs, ce+1)))
    groups += nights(*subarc)                # each night of the sub-arc
    seen, uniq = set(), []
    for g in groups:
        if g not in seen:
            seen.add(g); uniq.append(g)
    return uniq


FORCE_REJECT_THRESHOLD = 5     # >=5 obs turned off => regular mode "failed"

PROBE_MIN_ANCHOR_DAYS = 0.05   # ~1.2 h: a tracklet shorter than this is too near-simultaneous
                               # to give a non-degenerate IOD anchor -- skip it when probing
                               # apparitions for the sigma_a multi-start (see fit_orbit).


LINK_OPP_MAX_RMS = 2.0   # a candidate that links a whole EXTRA opposition is accepted up to this
                         # raw rms even if it exceeds the reference: a second opposition pins the
                         # period (a huge orbit-determination win), and it legitimately raises rms
                         # by pulling in an older, noisier apparition -- not a regression.


def _better_link(cand, ref):
    """Accept a recovery candidate that links strictly MORE obs without sacrificing fit
    quality -- its rms must stay within the reference (or below the 0.6" 'ok' bar). Guards
    against a night-drop that forces in a discordant cluster: more obs but a ballooned rms is
    not an improvement.

    EXCEPTION: a candidate that links a whole extra OPPOSITION is a major win (the long
    baseline pins the period), and naturally sits at a higher rms than a single-opposition fit
    of a pristine modern arc -- so accept it on a sane absolute rms (LINK_OPP_MAX_RMS) rather
    than requiring it to beat the reference. (2008 TE4: a 68/73 opp=2 link at 0.873" was being
    rejected for a 44/73 opp=1 fit at 0.700".)

    But the extra opposition must be EARNED, not FAKED: a genuine two-opposition link ADDS an
    apparition's obs, so it keeps at least as many obs as the reference (TE4 68 >= 44). A
    candidate that re-anchors on a SHORT modern tracklet and drags in only a sliver of the rich
    apparition -- rejecting the rest -- also reports n_oppositions()==2, but it keeps FEWER obs
    (the wrong-period "chimera": P12orpW+2016 BV14 kept 12 of 2026 + only 3 of 24 in 2016 = 15
    obs at a=4.87, vs the clean 24-obs single-opposition anchor). Rejecting it here leaves the
    honest 24/36 fit as `best` -- still troubled (2026 fully unlinked), so the joint refit
    (_recover_joint_refit) then glues the true 36/36 link at a=2.354. Without the obs-count guard
    the chimera wins, both apparitions carry SOME obs so _still_troubled goes quiet, and the
    joint refit never runs."""
    if (cand.n_oppositions() > ref.n_oppositions() and cand.rms_residual() <= LINK_OPP_MAX_RMS
            and cand.n_observations() >= ref.n_observations()):
        return True
    return (cand.n_observations() > ref.n_observations()
            and cand.rms_residual() <= max(ref.rms_residual(), 0.6))


def _has_unlinked_opposition(fit, obs):
    """A whole MULTI-NIGHT apparition is left with no included obs -- the clear sign that
    linkage genuinely failed on an apparition that SHOULD be linkable (multi-night => has
    curvature, is independently IOD-able), e.g. a poisoned IOD sub-arc left the real
    apparitions stranded. This auto-triggers the cheap drop-suspect-night recovery even
    without --force.

    A SINGLE-NIGHT unlinked tracklet does NOT trigger it: a lone night extension to a new
    apparition is the high-risk case -- it's often a genuine mislinkage, and forcing it in
    balloons the rms (the recovery's rms guard would reject it anyway, so attempting it
    just burns time). We extend to it opportunistically in the normal pass, but won't
    brute-force a single bad night back in."""
    # Cluster over the fit's OWN obs: is_included() aligns with fit._obs, which can be a
    # reduced subset of the caller's `obs` when a recovery stage dropped a suspect night
    # (force path) or upstream dropped companion lines. Indexing the mask by clusters built
    # from the longer `obs` silently mis-slices (numpy truncates, no error) -> wrong verdict.
    obs = fit._obs
    times = np.array([o["jd_utc"] for o in obs])
    clusters = _clusters(times)
    if len(clusters) < 2:
        return False
    inc = fit.is_included()
    # ...and the stranded apparition must be SUBSTANTIAL to be worth the (per-suspect-night,
    # full-refit) recovery: at least max(5, 3% of obs). A healthy heavy object naturally
    # leaves a sliver of a small apparition unlinked among its many; recovering a handful of
    # obs out of thousands is not worth ~90 screening fits. (944 Hidalgo fit 1928/2004 over 31
    # oppositions at 0.49" but one tiny stranded apparition triggered ~97 recovery fits = 195s.)
    floor = max(5, int(0.03 * len(obs)))
    return any(not inc[cs:ce + 1].any() and (times[ce] - times[cs]) > 0.5
               and (ce - cs + 1) >= floor
               for (cs, ce) in clusters)


DISCORDANT_FIT_RMS = 1.0     # only suspect a wrong-period link when the OVERALL fit is this poor
                             # (arcsec): a good multi-opposition fit (e.g. 944 Hidalgo, 0.49" over
                             # 31 oppositions) naturally has one apparition sitting a bit high
                             # among many -- not worth an expensive re-seed, so leave it alone.
DISCORDANT_SIGMA = 2.0       # ...with an apparition whose INCLUDED obs sit this many sigma off
                             # (median)...
DISCORDANT_RATIO = 2.0       # ...AND this many times worse than the best-fitting apparition (so
                             # ONE apparition stands out as off -- a wrong-period link -- vs a
                             # uniformly-noisy fit where every apparition sits similarly high)
DISCORDANT_MIN_OBS = 4       # judge an apparition's median only with at least this many obs


def _has_discordant_opposition(fit, obs):
    """A LINKED apparition whose included obs sit systematically many-sigma off (high MEDIAN
    normalized residual) while ANOTHER apparition fits cleanly -- the fingerprint of a
    wrong-PERIOD link, not scattered outliers. The orbit is pinned by one dense apparition and
    rides over the other; the discordant obs survive rejection only because old astrometry
    carries loose sigmas (and obscode relaxation softens them further), so the raw rms balloons
    while every individual obs stays just under the reject limit. Re-seeding on the discordant
    apparition usually threads BOTH (1998 VD32: the 1998 arc sits 3.8sigma off the 2020-pinned
    orbit; re-seed -> 0.86sigma, 0.525\" overall). Distinct from a uniformly-noisy fit (no clean
    apparition to contrast) and from a fully stranded one (_has_unlinked_opposition)."""
    if fit.rms_residual() <= DISCORDANT_FIT_RMS:     # the fit is already good -> leave it alone
        return False
    # Cluster over the fit's OWN obs (see _has_unlinked_opposition): residuals()/is_included()
    # align with fit._obs, which a recovery stage may have reduced below the caller's `obs`.
    obs = fit._obs
    times = np.array([o["jd_utc"] for o in obs])
    clusters = _clusters(times)
    if len(clusters) < 2:
        return False
    rs = np.array([r["resid_sigmas"] for r in fit.residuals()])   # raw-sigma normalized resid
    inc = fit.is_included()
    meds = [float(np.median(rs[cs:ce + 1][inc[cs:ce + 1]]))
            for (cs, ce) in clusters
            if int(inc[cs:ce + 1].sum()) >= DISCORDANT_MIN_OBS]
    return (len(meds) >= 2 and max(meds) > DISCORDANT_SIGMA
            and max(meds) > DISCORDANT_RATIO * min(meds))


SOFT_FIT_RMS = 0.6           # ...and only when the fit is ELEVATED (raw rms > this, the "suspect"
                             # bar) -- a fit already under 0.6" has nothing to gain, so a clean/
                             # heavy object (Atira 0.42", Hidalgo 0.49") skips the polish entirely
                             # rather than spend passes that don't move it.
SOFT_MARGINAL_LO = 1.5       # the soft-weight polish fires only when the linked fit carries
SOFT_MARGINAL_HI = 3.0       # marginal blunders -- included obs in [1.5, 3.0] sigma that survive
SOFT_MIN_MARGINAL = 3        # rejection but drag a high-leverage arc (>=3 of them). Clean fits
                             # (all obs < 1.5 sigma) skip it, so their goldens never move.


SOFT_REDEEM_ARCSEC = 2.0     # second-escape redemption radius. find_orb keeps an obs within a
                             # couple arcsec and DOWN-weights it rather than hard-excluding it
                             # (compute_rms is plain raw-arcsec over included obs -- NOT weighted --
                             # so a good position we clip for an over-tight sigma is pure loss). After
                             # the first escape un-sticks the orbit, reconsidering every obs within
                             # this raw arcsec lets it settle into find_orb's basin (2008 TE4
                             # 0.498"->0.383", matching find_orb's 0.38").


def _final_soft_polish(fit):
    """End-of-fit soft EXCLUSION on the FINAL best orbit (find_orb's threshold scheme), run
    exactly ONCE -- never mid-link -- so it can never strand a still-converging apparition (the
    failure mode of doing it per-attempt). Reports marginal blunders as 'unused' (recoverable,
    not removed) and refits on the kept set. Gated to an ELEVATED fit (rms > SOFT_FIT_RMS)
    carrying >= SOFT_MIN_MARGINAL marginal obs; a clean fit returns untouched."""
    ctx = getattr(fit, "_ctx", None)
    if ctx is None or os.environ.get("PYROORB_NO_SOFT"):
        return fit
    ev = ctx["ev"]
    state = np.asarray(fit._state, float).copy()
    mask = fit.is_included()
    if ev.rms_arcsec(state, mask) <= SOFT_FIT_RMS:
        return fit
    rs = ev.resid_sigmas(state)[mask]
    if int(np.sum((rs > SOFT_MARGINAL_LO) & (rs < SOFT_MARGINAL_HI))) < SOFT_MIN_MARGINAL:
        return fit
    # Escape 1: down-weight marginal blunders off the current included set, soft-exclude.
    state2, res2, keep = _soft_polish(ev, state, mask)
    if res2 is None or int(keep.sum()) < 6:
        return fit
    # Escape 2 (find_orb keep-within-2"): now the orbit has un-stuck, RECONSIDER every obs within
    # SOFT_REDEEM_ARCSEC raw -- a good position hard-excluded for an over-tight sigma re-enters and
    # lets the orbit settle into find_orb's basin. Take it only if it lowers rms without clipping
    # obs vs the first escape (else escape 1 stands).
    dra, ddec = ev.resid_arcsec(state2)
    cand = np.hypot(dra, ddec) < SOFT_REDEEM_ARCSEC
    if int(cand.sum()) >= 6:
        state3, res3, keep3 = _soft_polish(ev, state2, cand)
        if (res3 is not None and int(keep3.sum()) >= int(keep.sum()) - 1
                and ev.rms_arcsec(state3, keep3) < ev.rms_arcsec(state2, keep)):
            state2, res2, keep = state3, res3, keep3
    return _build_orbit(ctx, state2, keep, res2, getattr(fit, "_history", None))


def _final_reclaim(f, radar_rows, obscodes, verbose=False):
    """Rejection verdicts AGE as the orbit improves -- and the pipeline's last two
    movers (the delivered non-gravitational model, the radar joint) run AFTER both the
    rejection loop and the end-of-fit soft polish, so their improvements never reach the
    excluded observations. Measured on 1862 Apollo fit cold: 269 modern obs sat excluded
    that fit the FINAL orbit at median 0.59 arcsec (stale -- rejected against a worse
    intermediate orbit), the delivered rms was 1.26 arcsec, and one re-judgment pass against
    the finished orbit recovers 0.75 arcsec. This stage runs the ESTABLISHED machinery once
    more at the true end: the auto_reject sweep's own redeem/reject hysteresis to a fixed
    point (<= 3 rounds; it converged in 3 on Apollo), a refit per round that honours the
    active force model (`ev.fit` carries ev.nongrav and the asteroid-perturbed predict), then a
    delivery-grade rebuild -- refine_nongrav seeded from the delivered solution for a nongrav
    orbit, the plain rebuild otherwise -- the radar rejoin, and for an elevated gravity fit
    one more soft polish (whose Escape-2 keep-within-2" is find_orb's own late redemption).

    GATED to fire only when a late mover actually moved the orbit (a delivered nongrav model
    or applied radar): an offline gravity fixture has neither, so the archive is untouched by
    construction. Blind re-judgment of exclusions is the codebase's established semantics
    (auto_reject redeems anything under the inner threshold every round; soft Escape 2
    reconsiders every obs within 2 arcsec) -- structural drops stay dead because their obs
    genuinely misfit, and the evaporation/keep guards bound the rest. KEEP-THE-BETTER: a
    reclaim that loses observations or materially worsens the weighted fit is discarded."""
    import pyroorb.arc_extend as ae
    ctx = getattr(f, "_ctx", None)
    if ctx is None:
        return f
    if not (getattr(f, "nongrav_model", None)
            or (getattr(f, "radar", None) or {}).get("applied")):
        return f
    ev = ctx["ev"]
    mask = np.asarray(f._mask, bool).copy()
    state = np.asarray(f._state, float)[:6].copy()
    changed = 0
    result = None
    for _ in range(3):
        m2 = mask.copy()
        rv = ae.auto_reject_within_arc(ev, m2, state)
        if rv <= 0 or not (m2 ^ mask).any():
            break
        changed += int((m2 ^ mask).sum())
        mask = m2
        state, result = ev.fit(state, mask, thin=False, polish=True)
    if changed == 0:
        return f
    if verbose:
        print(f"  final reclaim: {changed} obs re-judged against the finished orbit "
              f"({int(mask.sum())}/{ev.n} now included)")
    try:
        if getattr(f, "nongrav_model", None):
            f2 = refine_nongrav(f, f.nongrav_model, mask=mask, revert_if_worse=False,
                                profile_sigmas=False, asteroids=bool(ctx.get("asteroids")),
                                seed_state=_lift_nongrav_state(f, f.nongrav_model))
        else:
            if result is None:
                return f
            f2 = _build_orbit(ctx, state, mask, result, getattr(f, "_history", None))
        if radar_rows and os.environ.get("PYROORB_RADAR", "1") != "0":
            import pyroorb.radar as _radar
            f2 = _radar.refine_radar(f2, radar_rows, obscodes, verbose=verbose)
        if not getattr(f2, "nongrav_model", None):
            f2 = _final_soft_polish(f2)
        # the reclaim changes the mask, not how the model was chosen -- carry the record
        # (refine_nongrav's fresh delivery does not, the same loss the radar joint had)
        f2.nongrav_discovery = getattr(f, "nongrav_discovery", None)
    except Exception as ex:
        if verbose:
            print(f"  final reclaim abandoned: {type(ex).__name__}: {ex}")
        return f
    if (f2.n_observations() < f.n_observations()
            or f2.rms_normalized() > f.rms_normalized() * 1.10):
        if verbose:
            print(f"  final reclaim: kept the pre-reclaim orbit "
                  f"({f.n_observations()}@{f.rms_normalized():.3f} vs "
                  f"{f2.n_observations()}@{f2.rms_normalized():.3f})")
        return f
    return f2


def _append_final_stage(fit):
    """Catch-all closing entry for the 'how the orbit was fit' trace.

    Several steps run AFTER arc-extension has finished recording its history but still
    change the delivered orbit: the final covariance refit (`thin=False`), the single-
    night-poison rescue, and the end-of-fit soft polish (`_final_soft_polish`). None of
    them log a stage of their own, so the last recorded stage's (n_obs, rms) can disagree
    with the orbit actually reported in the summary -- a silent jump the user sees as a
    "missing final stage". When that gap exists, append one synthetic `final` stage so the
    trace ends on the real orbit. When nothing moved since the last stage, add nothing
    (the trace is already accurate -- no duplicate-looking row)."""
    hist = getattr(fit, "history", None)
    if not hist:
        return fit
    last = hist[-1]
    n_obs = fit.n_observations()
    rms = fit.rms_residual()
    # Only when there is a VISIBLE change: a different obs count, or an rms that the trace
    # would render differently. The web shows rms to 3 decimals (fmt(h.rms, 3)), so compare
    # at that precision -- a sub-0.0005" covariance-refit nudge is invisible and gets no row,
    # while anything the user can actually see closes the gap with a `final` stage.
    if last.get("n_obs") == n_obs and round(float(last.get("rms") or 0.0), 3) == round(rms, 3):
        return fit
    ctx = getattr(fit, "_ctx", None)
    mask = fit.is_included()
    state = np.asarray(fit._state, float)
    score = ctx["ev"].rms_norm(state, mask) if ctx is not None else fit.rms_normalized()
    hist.append(_hsnap({"stage": "final", "n_obs": int(n_obs), "rms": float(rms),
                        "score": float(score), "kept": True}, state, mask))
    return fit


def _build_orbit(ctx, state, mask, result, history):
    """Assemble a FittedOrbit from a fitted (state, mask, result) plus the per-fit `ctx`
    (evaluator, backend, epoch, sigmas, ...). Shared by _fit_from_start and the end-of-fit
    soft-exclusion polish, so the latter can RE-FINALIZE the best orbit on a new (state, mask).
    Stashes ctx + history on the orbit so it can be re-finalized again."""
    ev, backend = ctx["ev"], ctx["backend"]
    _ts = _stime.perf_counter()
    dra, ddec = ev.resid_arcsec(state)            # O-C for ALL obs (incl. excluded)
    resid_all = np.column_stack([dra, ddec])
    ng = ctx.get("nongrav")                       # non-gravitational (A, gr) or None
    helio = (backend.helio_positions_batch(ctx["fit_base"], ctx["epoch_tdb"], state,
                                           ctx["times_tdb"], nongrav=ng) if ng is not None
             else backend.helio_positions_batch(ctx["fit_base"], ctx["epoch_tdb"], state,
                                                ctx["times_tdb"]))
    Robs_arr = np.array(ctx["Robs"])
    r_helio = np.linalg.norm(helio, axis=1)
    delta = np.linalg.norm(helio - Robs_arr, axis=1)
    earth_sun = np.linalg.norm(Robs_arr, axis=1)
    obs = ctx["obs"]
    phot_records = [{"obs_mag": o.get("mag"), "band": o.get("band", " "),
                     "mag_sigma": o.get("mag_sigma"),
                     "r": float(r_helio[k]), "delta": float(delta[k]),
                     "earth_sun": float(earth_sun[k]), "included": bool(mask[k])}
                    for k, o in enumerate(obs)]
    phot = photometry.fit_absolute_magnitude(phot_records)
    _stadd("products", _ts)
    # fit.asteroids is what the ctx backend integrates with, not a separately kept note: a
    # whole-fit +16ast run says True, a toggle back to gravity says False
    meta = ctx["meta"]
    pinned = pinned_asteroids(backend)
    if pinned is not None and bool((meta or {}).get("asteroids")) != pinned:
        meta = {k: v for k, v in (meta or {}).items() if k != "asteroids"}
        if pinned:
            meta["asteroids"] = True
    fit = FittedOrbit(desig=ctx["desig"], obs=obs, result=result, epoch_tdb=ctx["epoch_tdb"],
                      seed_d=ctx["seed_d"], iod_method=ctx["method"], mask=mask,
                      resid_all=resid_all, history=history, meta=meta,
                      sigma_ra=ctx["sigma_ra"], sigma_dec=ctx["sigma_dec"],
                      force_model=ctx["backend_name"], phot=phot)
    fit._ctx = ctx
    fit._history = history
    return fit


# Physical A2 grid (au/day^2) for the Yarkovsky profile seed, denser near zero. It spans the
# range a real transverse thermal acceleration occupies on the small NEOs this fires for --
# 2008 NA sits at -1.0e-12, 2011 CL50 at +4.4e-12 (JPL) -- and the size-aware plausibility
# ceiling vetoes anything unphysical downstream anyway. The grid only has to land ON the valley;
# the joint LM polishes the value.
A2_PROFILE_GRID = np.array([-1e-11, -5e-12, -2e-12, -1e-12, -5e-13, 0.0,
                            5e-13, 1e-12, 2e-12, 5e-12, 1e-11])
# The A1 (radial SRP) grid, in AREA/MASS ratio (m^2/kg; A1 = nongrav.SRP1AU * AMR): what a credible
# inner-solar-system SRP occupies, |AMR| up to ~0.05, denser near zero and skewed positive because
# SRP pushes OUTWARD (a negative A1 is unphysical and only there so the profile can see the sign).
A1_PROFILE_AMR_GRID = np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
# Which non-gravitational axes the in-trial STALL RESCUE below profiles: A1 and A2 are
# the two that trade against the semi-major axis and therefore corner a cold LM; A3 (normal) has
# no such valley. Env override exists for A/B measurement only (PYROORB_NG_RESCUE="1" is the
# pre-2026-08-26 behaviour, A2 only; "" disables the rescue).
NONGRAV_RESCUE_AXES = tuple(int(x) for x in os.environ.get("PYROORB_NG_RESCUE", "0,1").split(",") if x.strip())


def _profile_seed_axis(backend, t0, p0, scale, gr, tt, Rs, ra, dec, cosd, sra, sdec, m,
                       axis=0, grid_phys=None, slot=0, continue_from_zero=False, early_stop=False,
                       zero_cache=None):
    """Seed a joint (state, A) LM by PROFILING ONE non-gravitational axis: freeze A[axis] on a
    coarse physical grid, re-fit the 6-state at each, and return a p0 whose A (and its co-adapted
    state) sit at the lowest-chi2 grid point -- already on the curved (A <-> semimajor-axis)
    valley floor that the un-seeded LM cannot reach from the gravity corner.

    Why this is not radial-specific: a coefficient acting ALONG TRACK is near-degenerate with the
    semi-major axis, because both change the period. That is the documented A1 story, and it is if
    anything MORE true of the transverse A2, whose entire signature IS a secular along-track drift.
    `slot` is the axis's position in the parameter vector's A block (index 6 + slot).

    The joint LM then polishes the exact coefficient -- the profile only has to land ON the ridge
    near the minimum, so the grid is deliberately coarse (cheap: warm-started, loose-tol fits).

    `zero_cache` (a dict, optional) memoizes the grid's A = 0 point. With every coefficient at zero
    that state fit does not depend on which axis is being profiled, so the A1 and A2 profiles of one
    comparison (same data, same gravity seed, same g(r)) would otherwise run the identical fit twice."""
    import pyroorb.nongrav as ng
    from scipy.optimize import least_squares
    a1_grid = np.asarray(grid_phys, float)
    def _state_fit(a1, seed6):
        A = [0.0, 0.0, 0.0]
        A[axis] = float(a1)
        # The shared factory's one-slot cache lets the LM's jac(p) reuse the integration its
        # resid(p) at the same point already ran, instead of integrating the arc twice.
        resid, jac, _ = _make_ng_residual(backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr,
                                          np.empty(0), lambda p: A, [], np.empty(0), m)
        r = least_squares(resid, seed6, jac=jac, method="lm",
                          xtol=1e-6, ftol=1e-6, max_nfev=60)
        return np.asarray(r.x, float), float(np.sum(r.fun**2))
    # `continue_from_zero` picks the WARM-START PATH through the grid, and it matters because the
    # inner state fits are cheap and loose (60 evals, 1e-6):
    #   False (default, the shipped A1 behaviour): chain along the grid in index order. Tuned
    #     against 2026 NA1's comet2/srp descent -- left byte-identical.
    #   True: walk OUTWARD FROM ZERO in each direction independently, so every step is a small
    #     perturbation of an already-converged fit. Needed on the A2 axis, where starting from the
    #     far end of the grid throws the orbit far enough off that the loose fits never recover:
    #     every later warm start inherits the damage and the profile returns chi2 of 1e4-1e5
    #     against a ~9e2 gravity baseline -- a garbage curve whose minimum can sit on the WRONG
    #     SIGN (measured on 2008 NA: the chained profile put A2 at +2e-12 where the truth is
    #     -1.2e-12, and the seed was useless).
    # `early_stop` (only meaningful with continue_from_zero): walking outward from zero, stop a
    # side at the first grid point whose chi2 is HIGHER than the previous one. The profile is the
    # chi2 along a smooth curved valley -- convex about its minimum -- so once it rises the
    # minimum is behind us. On a null object (the common case: the gate population's rescue
    # "wins" by chi2 46.0 vs 46.1) each side stops after one step and the profile costs ~3 fits
    # instead of 9-11; on a real signal the walk continues to the minimum exactly as before.
    # The walk starts AT ZERO, and that is load-bearing rather than incidental. Starting it beside
    # an already-known coefficient -- the delivered value, for the delete-1 jackknife's refits --
    # makes it ~3x cheaper and was tried: the convexity stop then fires on the first point of each
    # side, the seed IS the known coefficient, and the tight LM polishes a couple of parts in 1e4
    # off it. Measured on 2008 NA (delete-1 of 14 tracklets): the cloud's MAD collapsed to 2e-15
    # against a range spanning 1.9e-12, and the robustness z went 2.15 -> 2047. It is the same
    # failure a warm start produces (z~700), reached by a different door: what makes a
    # delete-1 estimate independent is that its seed does not know the full-arc answer, and an
    # ORDERING that starts at that answer knows it.
    best_state, best_chi2, best_a1 = p0[:6].copy(), np.inf, 0.0
    if continue_from_zero:
        sides = (a1_grid[a1_grid >= 0], a1_grid[a1_grid < 0][::-1])
    else:
        sides = (a1_grid,)
    zero_state = p0[:6].copy()
    zero_chi2 = np.inf
    for side in sides:
        s = zero_state.copy()
        # Each side walks outward FROM zero, so its first step is judged against the A = 0 point,
        # the side's true predecessor. That holds the negative side to the rule the positive side
        # (which starts at zero) already follows: on a null profile both neighbours of zero rise
        # and, the profile being convex about its minimum, no farther grid point can win.
        prev = zero_chi2
        for a1 in side:
            zkey = (id(gr), s.tobytes()) if a1 == 0.0 and zero_cache is not None else None
            if zkey is not None and zkey in zero_cache:
                st, chi2 = zero_cache[zkey]
                st = st.copy()
            else:
                st, chi2 = _state_fit(a1, s)
                if zkey is not None:
                    zero_cache[zkey] = (st.copy(), chi2)
            s = st                              # warm-start the next grid point
            if continue_from_zero and a1 == 0.0:
                zero_state, zero_chi2 = st.copy(), chi2
            if chi2 < best_chi2:
                best_chi2, best_state, best_a1 = chi2, st, float(a1)
            if early_stop and continue_from_zero and a1 != 0.0 and chi2 > prev:
                break
            prev = chi2
    out = p0.copy()
    out[:6] = best_state
    out[6 + slot] = best_a1 / scale
    return out


def _profile_seed_radial(backend, t0, p0, scale, gr, tt, Rs, ra, dec, cosd, sra, sdec, m):
    """The A1 (radial SRP) profile seed: _profile_seed_axis on axis 0 over an area/mass-ratio grid
    (A1 = SRP1AU * AMR) spanning what a credible inner-solar-system SRP occupies -- |AMR| up to
    ~0.05 m^2/kg, denser near zero. Behaviour unchanged; see _profile_seed_axis for the rationale."""
    import pyroorb.nongrav as ng
    amr_grid = A1_PROFILE_AMR_GRID
    return _profile_seed_axis(backend, t0, p0, scale, gr, tt, Rs, ra, dec, cosd, sra, sdec, m,
                              axis=0, grid_phys=amr_grid * ng.SRP1AU, slot=0)


def _make_ng_residual(backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr, hA,
                      a_of_p, free_cols, col_scale, m):
    """Factory for the weighted (RA,Dec) residual/Jacobian closures every non-gravitational fit
    shares -- the ((pred-obs) wrapped)*cosd*3600/sigma residual and the cosd*3600/sigma-scaled,
    NaN-guarded Jacobian -- so the angle-wrap and weighting live in ONE place (they were hand-copied
    across refine_nongrav, _profile_seed_radial and the corrector, where a change to either had to be
    mirrored or silently diverge). `a_of_p(p)` maps the parameter vector to the physical [A1,A2,A3]
    (encapsulating fixed axes and the model/physical scale); `free_cols` are the A axes that get
    Jacobian columns; `col_scale` multiplies those physical columns into the variable the solver
    carries (model `scale` for the scipy LM, 1.0 for the physical-unit corrector). Returns
    (resid, jac, resid_only): resid/jac share a one-slot cache so the LM pays a single ASSIST
    integration per (resid,jac) pair; resid_only skips the Jacobian entirely (one integration) for
    the many derivative-free evals the corrector's half-steps and the seed's state fits make."""
    kk = len(free_cols)
    cs = np.asarray(col_scale, float) if kk else None
    cache = {}
    def _compute(p):
        key = p.tobytes()
        hit = cache.get(key)
        if hit is None:
            cache.clear()
            hit = cache[key] = backend.predict_radec_jac_ng(
                t0, p[:6], a_of_p(p), free_cols, gr, tt, Rs, hA=hA)
        return hit
    def _weighted_resid(pred):
        d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
        return _clamp(np.column_stack([d*3600.0/sra, (pred[:, 1]-dec)*3600.0/sdec]).ravel())
    def resid(p):
        pred, _ = _compute(p); return _weighted_resid(pred)
    def jac(p):
        _, J = _compute(p)
        out = np.empty((m, 2, 6 + kk))
        out[:, 0, :] = J[:, 0, :] * (cosd*3600.0/sra)[:, None]
        out[:, 1, :] = J[:, 1, :] * (3600.0/sdec)[:, None]
        if kk:
            out[:, :, 6:] *= cs                 # physical -> the variable the solver carries
        return np.where(np.isfinite(out), out, 0.0).reshape(2*m, 6 + kk)
    def resid_only(p):
        pred, _ = backend.predict_radec_jac_ng(t0, p[:6], a_of_p(p), [], gr, tt, Rs, hA=np.empty(0))
        return _weighted_resid(pred)
    return resid, jac, resid_only


def _full_step_descent(resid_only, jac, p0, arc_days, n_steps=12):
    """find_orb's `full_improvement` corrector (orb_func.cpp), ported: a bounded pure Gauss-Newton
    descent that reaches degenerate non-gravitational minima the damped scipy LM stalls short of.
    Each "full step" (find_orb's press-'f'): solve the FULL GN correction J.delta=-r (least-squares,
    so it steps far along the flat A1<->semimajor-axis valley where LM's damping suppresses motion);
    LIMIT it by find_orb's `scale_factor` so no STATE component moves more than 0.7*r (position) or
    0.7*r/(0.5*arc) (velocity); apply, recompute; HALF-STEP back toward the previous orbit while the
    rms jumped >1.5x (find_orb's HALF_STEPS). Best-seen is returned. The A columns must be ACCURATE
    for the GN step to point right (a fixed finite-diff step is roundoff-swamped for a ~1e-11 comet
    A1), so the caller passes `hA_nat`, each axis's ~1-sigma-effect step (find_orb's adaptive
    `delta_val`). p0/params are in PHYSICAL A units. Cold, ~n_steps ASSIST integrations, no seed:
    2026 NA1 comet2 chi2 47->39 (0.62"->0.49", near the 0.48" the ~5x-costlier radial seed reaches).
    NOT robust alone (stalls on a pure well-conditioned A2, or the A1-only degeneracy) -- the caller
    keeps min(this, the LM) so it can only ever improve on the LM, never regress. EARLY-EXITS once the
    iterate stops MOVING (|delta-chi2| < 0.1% for two steps): a genuinely stuck comet keeps changing
    chi2 every step -- it first OVERSHOOTS up, then walks steadily down the flat valley well past the
    start (2026 NA1: 48 -> 81 -> 70 -> ... -> 34) -- whereas a no-out-gassing comet candidate (a
    Damocloid that fits gravity, e.g. Hidalgo/Don Quixote) goes flat after one tiny step (941 -> 939
    -> 938.8 -> 938.8 ...), the GN already at the LM's own minimum. Bailing on CONVERGENCE, not on
    lack-of-improvement (which killed the descent during its overshoot recovery), trims the no-signal
    candidates from all `n_steps` to ~3 for a result keep-the-better discards anyway."""
    p = np.asarray(p0, float).copy()
    r = resid_only(p)
    best_chi2, best_p = float(np.sum(r**2)), p.copy()
    c_prev = best_chi2                                          # chi2 of the previous iterate
    rms = float(np.sqrt(np.mean(r**2)))
    maxd = np.empty(6)
    stale = 0                                                   # consecutive steps the iterate barely moved
    for _ in range(n_steps):
        J = jac(p)
        delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        rh = float(np.linalg.norm(p[:3]))                       # heliocentric distance (au)
        maxd[:3] = rh * 0.7
        maxd[3:] = rh * 0.7 / (arc_days * 0.5) if arc_days > 0 else rh * 0.7
        sf = max(1.0, float(np.max(np.abs(delta[:6] / maxd))))  # find_orb scale_factor (state-bounded)
        pnew = p + delta / sf
        rN = resid_only(pnew); rmsN = float(np.sqrt(np.mean(rN**2))); h = 0
        while rmsN > rms * 1.5 and h < 6:                       # HALF_STEPS backtracking
            pnew = (pnew + p) * 0.5
            rN = resid_only(pnew); rmsN = float(np.sqrt(np.mean(rN**2))); h += 1
        if not np.all(np.isfinite(pnew)) or np.linalg.norm(pnew[:3]) > 1e3:
            break                                               # is_unreasonable_orbit -> stop, keep best
        p, r, rms = pnew, rN, rmsN
        c = float(np.sum(rN**2))
        if c < best_chi2:
            best_chi2, best_p = c, p.copy()
        stale = stale + 1 if abs(c - c_prev) < 1e-3 * c_prev else 0   # iterate barely moved -> converged
        c_prev = c
        if stale >= 2:                                          # flat for two steps -> stop early
            break
    return best_p, best_chi2


def _adaptive_ng_hA(resid_phys, p0, k, scale, m):
    """find_orb's adaptive numerical-derivative step (`delta_val`): for each free A axis, pick the
    PHYSICAL finite-diff step whose worst single-observation residual effect is ~1 sigma, so the
    Jacobian A column is neither roundoff-swamped (a fixed step is, for a ~1e-11 comet A1) nor
    over-large. `resid_phys(p)` is the weighted residual with p carrying PHYSICAL A. Returns the
    per-axis step array (find_orb targets 1.0, accepts [0.03,3.0] -- we bracket [0.3,3.0] and cap the
    tweaks). A handful of derivative-free evals per axis, once, at the cold seed."""
    r0 = resid_phys(p0)
    hA = np.empty(k)
    for j in range(k):
        h = 1e-3 * scale
        for _ in range(8):
            pp = p0.copy(); pp[6 + j] += h
            eff = float(np.max(np.sqrt(((resid_phys(pp) - r0).reshape(m, 2)**2).sum(1)))) + 1e-15
            if 0.3 < eff < 3.0:
                break
            h *= 1.0 / eff
        hA[j] = h
    return hA


def _cold_cornered(result, obs_resid, fit, k, m, stuck_frac=0.9):
    """True when the cold LM barely improved on the gravity residual -- the signature of a degenerate
    radial-A1 model stuck in the gravity corner, where find_orb's full-step corrector pays off (and
    the only case worth its cost). A well-fitting model (rms well below gravity) is not cornered and
    is left to the LM. Cheap: one derivative-free eval of the A=0 (gravity) residual over this mask."""
    rms_lm = float(np.sqrt(np.mean(np.asarray(result.fun)[:2*m]**2)))
    r_grav = obs_resid(np.concatenate([np.asarray(fit._state, float)[:6], np.zeros(k)]))
    rms_grav = float(np.sqrt(np.mean(r_grav**2)))
    return rms_grav > 0 and rms_lm > stuck_frac * rms_grav


# --- non-gravitational JACKKNIFE uncertainty ----------------------------------------------------
# The covariance / profile sigma answer "how sharp is chi2 right here" and BOTH assume the per-obs
# errors are independent Gaussians of the stated sigma (find_orb's Monte Carlo makes the same
# assumption -- it jitters each obs independently). A non-gravitational coefficient is a WEAKLY
# IDENTIFIED parameter -- near-degenerate with the semimajor axis, its signal a slow secular drift
# whose leverage sits in a few tracklets at the arc's temporal extremes -- so a correlated
# systematic in those tracklets (a catalogue bias, a timing offset) is fitted AS a nongrav and
# every independence-assuming method reports it as a confident detection. The delete-one-tracklet
# jackknife is the one estimator that removes the correlated UNIT wholesale: drop each tracklet in
# turn, re-fit INDEPENDENTLY (each from A=0 via the profile seed -- NOT warm-started from the full
# solution, which pins the estimate in the flat valley and collapses the spread to nonsense), and
# take the spread. On 2008 NA's srp2 A1 it reads z=1.8 where the covariance says 6.4 and the
# profile 21; on a genuine A2 it agrees with them (2008 NA A2 z=6, 2011 CL50 z=20). Runs on the
# fit's OWN mask -- the delivered orbit's inclusion set -- because that is self-consistent (the +/-
# describes the orbit we ship) and it still catches a systematic there (dropping the high-leverage
# tracklets it rests on), while a gravity-"neutral" mask over-rejects the far apparition and
# reports a REAL Yarkovsky as unconstrained. Deterministic (no RNG); the delete-d fraction /
# bootstrap add a tuning knob and sampling noise for no gain once rescaled.
JACKKNIFE_MIN_TRACKLETS = 4       # below this a delete-1 spread is meaningless
JACKKNIFE_MAX_REFITS   = 16       # cap the refit count on dense arcs (systematic sample; see
                                  # _jackknife_sample). Was 30: the refits are the whole cost, and
                                  # 16 of them still votes the sign with every deletion it makes
                                  # while roughly halving what the estimator charges.
JACKKNIFE_PLAIN_STRIKES = 2       # after this many plain LMs in a row have diverged, stop
                                  # trying the plain LM on this object and seed straight away
JACKKNIFE_MIN_KEEP     = 5        # need at least this many CONVERGED refits, else robustness "not assessed"
JACKKNIFE_REJECT_FACTOR = 1.5     # a refit whose normalized rms exceeds this x the base fit's did NOT
                                  # converge on the subsample (removing data cannot legitimately make
                                  # the best fit this much worse) -- it is optimizer noise, not evidence
JACKKNIFE_REJECT_FLOOR = 0.3      # ... but never bar a refit within this much of the base in ABSOLUTE
                                  # normalized rms: on a well-fitting arc (base ~0.3) a 1.5x bar is
                                  # 0.45, close enough to the base that ordinary re-solve wobble
                                  # reads as divergence. Only bites below base 0.6; the real
                                  # divergence signature is enormous (2007 AA2: ~5 against 0.65).


def _split_tracklets_by_station(times_days, stations, order=None, gap_min=120.0):
    """Group observation indices into tracklets: same station AND <= `gap_min` minutes apart, in
    time order. `times_days` in days, `stations` a per-obs code array. Returns a list of index
    lists. The unit a systematic lives in (one telescope, one short session) -- the same
    definition itf.residual_trend.split_tracklets uses, kept local so fit_object carries no itf dependency."""
    idx = list(range(len(times_days))) if order is None else list(order)
    if not idx:
        return []
    groups, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        same = str(stations[b]) == str(stations[a]) if stations is not None else True
        if same and (times_days[b] - times_days[a]) * 1440.0 <= gap_min:
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)
    return groups


def _jackknife_sample(tracklets):
    """Which tracklets to delete. At or below JACKKNIFE_MAX_REFITS, all of them; above it, a
    systematic pass over the list in time order (which includes both ends of the arc, where a
    non-gravitational coefficient's leverage sits).

    A LINEARIZED delete-1 ranking was tried here and REJECTED on measurement (2026-08-27).
    It is free -- for the rows g of one tracklet the
    Gauss-Newton step is (J^T J - J_g^T J_g)^-1 J_g^T r_g, read straight off the delivered fit's own
    Jacobian -- and the idea was to spend the refit budget on the deletions most able to flip the
    sign. On 2011 CL50, whose A2 is sharp, it is superb: corr 0.998 with the exact refits, a cloud
    MAD within 1%. On 2008 NA, whose A2 is fragile (z~1.7) and which is the case this estimator
    exists for, it is corr -0.019, with a cloud MAD 14x too small and a per-tracklet
    |exact-base|/|pred-base| whose p90 is 202. That is not a surprise in hindsight and it is the
    whole point of the jackknife: the local chi2 surface is exactly what cannot say what removing a
    correlated unit does to a weakly identified parameter. A ranking that works only where nothing
    is at stake is not a ranking.

    So the sample is systematic and the only knob is how many. Note what that costs: the MAD of a
    ~15-value cloud has a p10/p90 span of 3-4x against the full-arc cloud's (measured on both
    objects' 30-refit clouds), so the reported SCALE is coarse at this budget while the VERDICT --
    the sign, which every refit votes on -- is not."""
    n = len(tracklets)
    if n <= JACKKNIFE_MAX_REFITS:
        return list(range(n))
    return sorted(set(int(i) for i in
                      np.linspace(0, n - 1, JACKKNIFE_MAX_REFITS).round().astype(int)))


def _jackknife_unexplained_zeros(fit, model, axes):
    """The axes of one delete-1 refit that came back EXACTLY 0.0 with nothing to explain it, split
    into (a forced profile seed could still explain these, it never will).

    The A seed is a bit-exact 0.0 and a damped LM that moves at all returns a tiny float, so an
    exact zero means the axis never left the seed -- the gravity-corner stall -- and this
    project's rule is that a fitted coefficient of exactly 0.0 is a stall, never a finding. The rms
    convergence bar cannot see it: a stalled refit's rms IS the gravity rms on that subsample,
    which for a real nongrav object sits well inside the bar (2008 NA: 0.95" gravity against 0.83"
    nongrav, 1.15x). Left alone, stalls enter the cloud as exact zeros and drag the median and the
    MAD toward zero -- inflating z, which is the phantom-hiding direction, and in the limit handing
    out MAD = 0 and z = infinity.

    But an exact zero is NOT always a stall, and the difference is free to read. refine_nongrav's
    own rescue profiles a cornered A1/A2 outward from zero and keeps the better, recording the axes
    it tried in nongrav_diag["rescue_chi2"]; a zero that survived THAT is a genuine collapse --
    "deleting this tracklet kills the coefficient" -- which is exactly the evidence a robustness
    test is asking for, and it stays in the cloud. Only an unexplained zero is worth the ~11s
    forced seed, and only where the seed can actually run: the rescue covers A1/A2 on the
    non-comet g(r) family, so an A3 or a comet-model zero is unexplainable and the refit is simply
    dropped (an unchecked stall is not evidence either way).

    `NONGRAV_RESCUE_AXES` (env `PYROORB_NG_RESCUE`, default both A1/A2) gates whether the FIRST,
    cold call here can send an axis to `can` -- i.e. whether the plain LM's own stall is worth a
    forced retry at all. It does NOT gate the forced retry's rescue itself: `refine_nongrav`'s
    `force_seed=True` path always profiles both axes regardless of this env var (its own gate is
    `not force_seed and axis not in NONGRAV_RESCUE_AXES`, which force_seed short-circuits), so a
    SECOND call here, on the forced retry's result, always sees the axis in `rescue_chi2` and
    routes it correctly as a survived collapse or a moved value -- never as unexplained. Only the
    cold-pass decision to BUY the forced seed is affected: with an axis outside the configured set,
    a stalled zero that would otherwise trigger a retry instead falls straight to `dead_z` and the
    refit is DROPPED -- the conservative direction (lost statistical power, never a wrong
    attribution)."""
    import pyroorb.nongrav as ng
    zeros = [a for a in axes if float(fit.nongrav_params.get(a, 0.0)) == 0.0]
    if not zeros:
        return [], []
    diag = dict(getattr(fit, "nongrav_diag", None) or {})
    profiled = set(diag.get("rescue_chi2") or {})
    zeros = [a for a in zeros if a not in profiled]
    if not zeros:
        return [], []
    seedable = ng.model(model)["gr"] is not ng.COMET_GR
    can = [a for a in zeros if seedable and ng.A_AXES.index(a) in NONGRAV_RESCUE_AXES]
    return can, [a for a in zeros if a not in can]


def _jackknife_stats(values, base, n_tracklets, attempted, rejected):
    """The cloud -> verdict arithmetic for ONE coefficient, as a PURE function (no fitting), so the
    statistics are testable without ASSIST -- see tests/test_nongrav_jackknife.py.

    `values` are the converged delete-1 estimates, `base` the delivered value, `n_tracklets` how
    many tracklets the arc HAS -- not how many of them were refit.

      * SCALE: robust MAD (one high-leverage tracklet genuinely moves a real coefficient's value
        without threatening the detection, and a std lets that one point wreck the sigma), scaled
        by sqrt(n_tracklets - 1). The delete-1 SE is sqrt(n-1) x the spread of the delete-1
        estimates, where n is the number of DELETABLE units; a sample of those estimates estimates
        the same spread and keeps the same constant. Scaling by the SAMPLE size instead -- which is
        what this did until 2026-08-27 -- shrinks sigma by sqrt((m-1)/(n-1)), 1.9x on a
        100-tracklet arc, and all of that error inflates z, i.e. it hides exactly the phantom the
        estimator exists to catch. Rejections shrink the sample, not the population, so they must
        not touch the constant either.
      * VERDICT: the sign spanning zero, INCLUSIVE -- a refit that lands exactly on 0.0 is the
        coefficient collapsing when that tracklet goes, which is a failure to survive, not a
        survival. (`{0, 0, 0, 1.9e-9, 2.0e-9}` used to report robust.)
      * a zero MAD is no measurable spread, NOT infinite significance: fall back to the half-range,
        and failing that report NOT ASSESSED. It used to divide by it and hand out z = inf, which
        reached the web UI as a null and rendered as a failed request.
      * more than a third of the attempts rejected is NOT ASSESSED as well: rejections are not
        random -- they cluster on the subsamples the LM finds hardest -- so what survives them is a
        spread biased low.

    One honest caveat no scaling constant can fix: the sample is systematic rather than random, and
    on a dense arc it is a fraction of the deletable units -- so the scale is an estimate of a
    spread, with its own sampling error (a p10/p90 span of 3-4x at this budget). See
    _jackknife_sample."""
    v = np.asarray([float(x) for x in values if np.isfinite(x)], float)
    # COVERAGE lives here, with the counts, rather than in the caller's loop. Every tracklet
    # that was actually tried incremented `attempted`, so `attempted < n_tracklets` catches
    # BOTH reasons one can go untried -- the refit budget never picked it, or deleting it
    # left too little arc to refit -- where the caller's old `len(picks) < n_trk` saw only
    # the first, and reported full coverage on an arc holding an undeletable tracklet.
    out = {"base": float(base), "n": int(v.size),
           "n_tracklets": int(n_tracklets), "attempted": int(attempted),
           "rejected": int(rejected), "sampled": bool(attempted < n_tracklets)}
    if v.size < JACKKNIFE_MIN_KEEP:
        return {**out, "assessed": False, "reason": "too few converged refits"}
    if rejected > max(1, attempted // 3):
        return {**out, "assessed": False, "reason": "too many non-converged refits"}
    mad = float(1.4826 * np.median(np.abs(v - np.median(v))))
    if mad <= 0:                                  # tied cloud -> half-range, ...
        mad = float((v.max() - v.min()) / 2.0)
    if not (mad > 0):                             # ... and failing that, no measurable spread
        return {**out, "assessed": False, "reason": "no spread across refits"}
    se = mad * float(np.sqrt(max(n_tracklets - 1, 1)))
    fv = float(base)
    return {**out, "assessed": True, "se": se, "z": abs(fv) / se,
            "spans_zero": bool(v.min() <= 0 <= v.max()),
            "flip": float(np.mean(np.sign(v) != np.sign(fv))) if fv else 1.0,
            "zeros": int(np.count_nonzero(v == 0.0)),
            "range": (float(v.min()), float(v.max()))}


def _profile_nongrav_sigmas(backend, t0, state6, A_hat, free_idx, gr, scale,
                            tt, Rs, ra, dec, cosd, sra, sde, chi2_min, lin_sig=None):
    """PROFILE-LIKELIHOOD 1-sigma for each free non-gravitational coefficient -- the RATIONAL
    uncertainty the linear covariance s^2 (J^T J)^-1 fails to give when a coefficient is degenerate
    with the 6-state (a near-singular Jacobian reports a tiny symmetric sigma for a parameter that
    is really unconstrained). For each free axis we hold it at a trial value, RE-FIT the 6-state AND
    the other free coefficients, and find the offset at which the weighted chi^2 rises by 1 (=1
    sigma) -- i.e. 'the range outside which the residuals actually get meaningfully worse'. Returns
    {axis: (w_lo, w_hi, unconstrained_bool)}: the SEPARATE minus/plus Delta-chi2=1 half-widths (the
    interval is genuinely asymmetric on a degenerate valley -- that asymmetry is the whole point of a
    profile, so we keep both sides; callers that want one number take max(w_lo, w_hi), the conservative
    symmetric sigma). Flagged unconstrained when chi^2 never rises by 1 on EITHER side within the
    search cap (a genuinely undetermined term). Physical and degeneracy-aware.

    Cost: a handful of 6-state re-fits per axis (ASSIST), so it runs only on the DELIVERED / CLI /
    UI fit, never on the auto-discovery comparison trials (which keep the fast linear sigma)."""
    from scipy.optimize import least_squares
    A_hat = list(A_hat)
    def _chi2_fixed(axis, value):
        """Re-fit the 6-state + the OTHER free axes with `axis` frozen at `value`; return chi^2."""
        others = [i for i in free_idx if i != axis]
        kk = len(others)
        hA = np.full(kk, 1e-3) * scale if kk else np.empty(0)
        def build(p):
            A = list(A_hat); A[axis] = value
            for j, ax in enumerate(others):
                A[ax] = p[6 + j] * scale
            return A
        def resid(p):
            pred, _ = backend.predict_radec_jac_ng(t0, p[:6], build(p), others, gr, tt, Rs, hA=hA)
            d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
            return _clamp(np.column_stack([d*3600.0/sra, (pred[:, 1]-dec)*3600.0/sde]).ravel())
        def jac(p):
            _, J = backend.predict_radec_jac_ng(t0, p[:6], build(p), others, gr, tt, Rs, hA=hA)
            out = np.empty((len(tt), 2, 6 + kk))
            out[:, 0, :] = J[:, 0, :] * (cosd*3600.0/sra)[:, None]
            out[:, 1, :] = J[:, 1, :] * (3600.0/sde)[:, None]
            if kk:
                out[:, :, 6:] *= scale
            return np.where(np.isfinite(out), out, 0.0).reshape(2*len(tt), 6 + kk)
        p0 = np.concatenate([np.asarray(state6, float),
                             [A_hat[i]/scale for i in others]])
        r = least_squares(resid, p0, jac=jac, method="lm", xtol=1e-8, ftol=1e-8, max_nfev=40)
        return float(np.sum(r.fun**2))

    sig = {}
    for j, axis in enumerate(free_idx):
        hat = A_hat[axis]
        cap = max(60.0 * abs(hat), 60.0 * (lin_sig[j] if lin_sig is not None else scale))
        step0 = max(lin_sig[j] if lin_sig is not None else scale, 1e-3*scale)  # start at the linear sigma
        def dchi(v):
            return _chi2_fixed(axis, v) - chi2_min
        # Seed the outward search AT the linear-sigma offset and interpolate PARABOLICALLY (chi^2 is
        # ~quadratic in the offset near the minimum, so Delta-chi2 is linear in offset^2). A
        # well-constrained axis lands near Delta-chi2=1 on the first eval and needs NO further fits
        # (offset = lin_sig / sqrt(Delta-chi2)); a degenerate one steps out geometrically until it
        # crosses 1 (or hits the cap = unconstrained). ~1 eval/side for a real detection, a handful
        # for a degenerate term -- cheap enough to run on every delivered fit.
        def half_width(sign):
            prev_off, prev_d = 0.0, 0.0; off = step0
            for _ in range(16):
                if off > cap:
                    return cap, True                            # never rose by 1 -> unconstrained
                d = dchi(hat + sign*off)
                if d >= 1.0:                                    # parabolic interp in offset^2 to Delta-chi2=1
                    denom = d - prev_d
                    o2 = (prev_off**2 + (off**2 - prev_off**2)*(1.0 - prev_d)/denom
                          if denom > 1e-12 else off**2)
                    return max(o2, 0.0)**0.5, False
                prev_off, prev_d = off, d
                off *= 1.8                                       # geometric outward search
            return cap, True
        w_hi, unc_hi = half_width(+1.0)
        w_lo, unc_lo = half_width(-1.0)
        sig[axis] = (w_lo, w_hi, unc_hi and unc_lo)             # keep both sides; unconstrained iff both
    return sig


def _deliver_nongrav_orbit(ctx, fit, model, result, mask, ng_diag=None,
                           profile_sigmas=True, extra_meta=None, asteroids=False):
    """Assemble the delivered FittedOrbit for a converged (6+k)-parameter non-gravitational LM.

    `result` carries x = [6-state, A's in `model`'s scale units] and fun/jac = ALL the sigma-
    normalized rows the covariance should reflect: refine_nongrav passes the weighted optical
    (+ constraint-penalty) rows; the joint state+A+radar solve (radar._refine_radar_nongrav,
    radar.py) passes optical + radar delay/Doppler rows. ONE definition of the params/sigmas
    arithmetic, the nongrav-aware evaluator clone and the meta plumbing, because the two
    deliverers must build identical orbits or a radar-refined fit would silently drift from a
    plain one in every downstream reader (residual rebuild, photometry, as_dict, the web UI).

    Factored out of refine_nongrav verbatim (2026-08-27); behavior-neutral for it."""
    import pyroorb.nongrav as ng
    import copy as _copy
    spec = ng.model(model)
    free_idx = ng.free_indices(model)
    k = len(free_idx)
    scale, gr = spec["scale"], spec["gr"]
    state = np.asarray(result.x[:6], float)
    A_phys = ng.full_A(model, result.x[6:])

    # Parameter 1-sigma from the weighted Jacobian at the solution, scaled by the
    # normalized-residual variance (same convention as the element sigmas).
    J = np.asarray(result.jac)
    dof = max(1, J.shape[0] - (6 + k))
    s2 = float(np.sum(result.fun**2)) / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(J.T @ J)
    sig_scaled = np.sqrt(np.maximum(np.diag(cov)[6:], 0.0))
    params, sigmas = {}, {}
    for j, axis in enumerate(free_idx):
        params[ng.A_AXES[axis]] = float(A_phys[axis])
        sigmas[ng.A_AXES[axis]] = float(sig_scaled[j] * scale)

    # `sigmas` above is the LINEAR covariance s^2 (J^T J)^-1 diagonal: fast, but it reports a tiny,
    # symmetric error for a coefficient that is degenerate with the state (near-singular Jacobian) --
    # calling an UNCONSTRAINED term a high-sigma 'detection'. The RATIONAL sigma is the
    # profile-likelihood interval (Delta-chi2=1, re-fitting everything else at each trial value; see
    # FittedOrbit.rational_nongrav_sigmas). A wrong sigma is a BUG, not an option -- so we mark this
    # fit to profile ON DEMAND whenever its sigma is DISPLAYED (CLI / as_dict / API). Doing it lazily
    # keeps the fit itself fast: the auto-discovery delivery + regression suite never touch the
    # sigma, so they pay nothing; only a user actually reading the +/- pays the ASSIST re-fits. The
    # opt-OUT (profile_sigmas=False) covers the internal comparison trials, whose fire/no-fire z
    # gating is JPL-calibrated on the linear sigma -- and the radar joint solve, whose profile
    # would re-fit WITHOUT the radar rows and so contradict the covariance it ships.
    # `_profile_sigmas` carries the flag to the orbit.
    _profile_sigmas_flag = bool(profile_sigmas and free_idx)

    # Rebuild WITH the out-gassing active so residuals + photometry reflect it. Use a
    # SHALLOW-CLONED evaluator carrying the nongrav config so the original gravity `fit`
    # (which shares this ctx's evaluator) is left untouched -- otherwise its residuals
    # would silently change under it. The clone shares the (read-only) obs arrays.
    nong = (list(A_phys), gr)
    if asteroids:
        # The delivered orbit LIVES under the 16-perturber model -- the ctx backend
        # (residual/photometry rebuild, the radar joint, later jackknife/web re-fits) and the
        # evaluator's own predict and Jacobian (O-C sweeps) both, or the state fitted under
        # perturbers is judged planets-only and every displayed residual carries the ~0.4"
        # divergence. with_force_model re-binds them through the ctx backend, so for one of
        # the 16 perturber objects the exclusion survives (a web re-fit, a jackknife).
        ev2, ctx2 = with_force_model(ctx, True)
    else:
        ev2 = _copy.copy(ctx["ev"])
        ctx2 = dict(ctx)
        ctx2["ev"] = ev2
    ev2.nongrav = nong; ev2._resid_key = None; ev2._sub_key = None
    ctx2["nongrav"] = nong
    # the flag rides the ctx, so a re-delivery (the radar joint, a jackknife refit, a web
    # re-fit) inherits it even when its own `asteroids` arg defaults False -- and the meta
    # records it where as_dict / reports can see which force model the value was made under
    _ast_active = bool(asteroids or ctx.get("asteroids"))
    # Advisory (NON-blocking) physical warning for the SRP/asteroid family: if the fitted A1 implies
    # a non-physical bulk density for this size (or A1<0, or A2 over the Yarkovsky ceiling), flag it
    # so the coefficients are read with care -- we still deliver the fit (rocket bodies / dark comets
    # are exactly such high-area/mass objects). Comet models are exempt: out-gassing has no density
    # limit and the comet model sequence has its own comet_a_is_plausible guard.
    ng_warn = None
    if gr is ng.INVSQ_GR:
        try:
            _H, _ = fit.absolute_magnitude()
        except Exception:
            _H = None
        ng_warn = ng.nongrav_physical_warnings(params, _H)
    ctx2["meta"] = {**(ctx.get("meta") or {}), "nongrav_model": model,
                    "nongrav_asteroids": _ast_active,
                    "nongrav_params": params, "nongrav_sigmas": sigmas,
                    "nongrav_warning": ng_warn,
                    "nongrav_profile": _profile_sigmas_flag,
                    "nongrav_diag": dict(ng_diag or {}),
                    **(extra_meta or {})}
    return _build_orbit(ctx2, state, mask, result, fit.history)


def refine_nongrav(fit, model="comet2", max_nfev=200, verbose=False,
                   mask=None, revert_if_worse=True, tol=1e-10, constraints=None,
                   profile_seed=True, profile_sigmas=True, radial_seed_all=False,
                   seed_state=None, rescue_cache=None, force_seed=False, asteroids=False):
    """Refit a converged orbit with a NON-GRAVITATIONAL model -- find_orb's "turn on
    the A1/A2/... parameters" step.

    `constraints` (optional, ``{element: target}`` over a/e/q/Q/i) holds those elements fixed
    WHILE the out-gassing is fitted -- the find_orb limited_orbit penalty rows (see
    refit_constrained) appended to the same LM, so a nongrav model and element constraints
    COMPOSE (e.g. a comet with a pinned perihelion). The penalty depends only on the 6-state,
    so its Jacobian columns over the A's are zero. None/empty = unconstrained (byte-identical to
    before this option).

    `tol` is the LM xtol/ftol. The default is tight (a delivered orbit); auto-discovery's
    model-COMPARISON trials pass a looser NONGRAV_DISCOVER_TOL, because they only need the
    chi2-gain to decide fire/no-fire (which is settled in a handful of evals) -- not a fully
    polished orbit. The chosen model is then re-fit here at the tight default for delivery.
    The seeded-srp discovery trial passes the intermediate NONGRAV_INTERACTIVE_TOL (tight
    enough that the profile-seed holds instead of sliding back down the degenerate valley to
    A1=0, but looser than delivery). The interactive web UI uses the tight default -- a manual
    model switch must reproduce the auto fit exactly (see engine_api._apply_interactive).

    The gravity fit (arc selection, outlier rejection, cluster activation) is left
    entirely alone; this holds the included-observation set FIXED and augments the
    solved-for vector with the model's free out-gassing coefficients (6 state + k of
    A1/A2/A3, per nongrav.MODELS).

    NOT opt-in: automatic discovery is ON by default (PYROORB_AUTO_NONGRAV and
    PYROORB_AUTO_COMET_NONGRAV, both defaulting to "1" in _finalize), so every default fit
    of an object inside the physical gate reaches here through discover_nongrav /
    discover_comet_nongrav. Explicit --nongrav and the UI's model picker call it directly.

    `model` is a nongrav.MODELS key
    (``comet2``/``comet3``/``yarkovsky``/``srp``/``srp2``/``srp3``/``none``).
    `mask` overrides which observations are included (default: the fit's own mask) --
    the interactive UI passes the user's currently-edited inclusion set so toggling
    obs and switching models compose. `revert_if_worse` (default True) keeps the
    gravity orbit if the model didn't improve the weighted rms; the UI sets it False
    so an explicitly chosen model is always applied. Returns a NEW FittedOrbit carrying
    the refined orbit plus the fitted A1/A2/A3 and their 1-sigma; ``none`` returns `fit`
    unchanged. Requires the ASSIST backend -- the nongrav force lives there."""
    import pyroorb.nongrav as ng
    from scipy.optimize import least_squares
    spec = ng.model(model)
    if not spec["params"]:
        return fit
    ctx = getattr(fit, "_ctx", None)
    if ctx is None:
        raise RuntimeError("fit carries no _ctx; cannot add a non-gravitational model")
    backend = ctx["backend"]
    if not hasattr(backend, "predict_radec_jac_ng"):
        raise RuntimeError("non-gravitational refit needs the ASSIST backend "
                           "(force_assist); the rebound 8-planet backend has no nongrav force")
    if asteroids:
        # The whole refit -- LM residuals, rescue profile, corrector -- under the
        # 16-perturber model; the delivered ctx carries the same wrapper (see
        # _deliver_nongrav_orbit) so nothing downstream drifts back to planets-only.
        backend = _AsteroidForce(backend)
    ev = ctx["ev"]
    # The phase-bridge link relaxation (Evaluator.relax_link_opposition) inflated a far
    # apparition's sigmas so the gravity-only reject loop would keep its coherent offset. That
    # offset is exactly the signal a nongrav model exists to fit, so restore full weight before
    # fitting it -- otherwise we down-weight the very obs that carry the out-gassing signature.
    ev.restore_link_relaxation()
    free_idx = ng.free_indices(model)
    k = len(free_idx)
    scale, gr = spec["scale"], spec["gr"]
    t0 = ctx["epoch_tdb"]
    mask = _caller_mask(fit, mask)
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
    ra, dec, cosd = ev.ra[idx], ev.dec[idx], ev.cosd[idx]
    sra, sdec = ev.fit_sig_ra[idx], ev.fit_sig_dec[idx]
    m = len(idx)
    arc_days = float(tt.max() - tt.min()) if m else 0.0
    hA_phys = np.full(k, 1e-3) * scale            # forward-diff step, physical au/day^2

    # Optional element constraints, composed with the out-gassing fit (mirror refit_constrained):
    # nc penalty residuals (element-target)*w appended to the LM, w = CONSTRAINT_TIGHTEN / free_sig.
    citems = []
    if constraints:
        src = constraints.items() if isinstance(constraints, dict) else constraints
        citems = [(str(kk), float(t)) for kk, t in src if str(kk) in _CONSTRAINABLE]
    if citems:
        ckinds = [kk for kk, _ in citems]
        ctargets = np.array([t for _, t in citems], dtype=float)
        nc = len(ckinds)
        _free_sig = fit.element_sigmas()
        _cbase = np.array([max(float(_free_sig.get(kk, 0.0)), 1e-6 * max(1.0, abs(t)))
                           for kk, t in citems])
        cw = float(CONSTRAINT_TIGHTEN) / _cbase   # length-nc penalty weights

    # The LM carries A in units of the model `scale`; expand to physical for the propagator.
    _obs_resid, _obs_jac, _ = _make_ng_residual(
        backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr, hA_phys,
        a_of_p=lambda p: ng.full_A(model, p[6:]), free_cols=free_idx,
        col_scale=np.full(k, scale), m=m)
    def resid(p):
        r = _obs_resid(p)
        if citems:
            pen = cw * (_constraint_values(p[:6], t0, ckinds) - ctargets)
            return np.concatenate([r, pen])
        return r
    def jac(p):
        obs_rows = _obs_jac(p)
        if not citems:
            return obs_rows
        # penalty Jacobian: w * d(element)/d(state) by forward diff; the A-columns (6:) are 0
        # because a constraint is a function of the 6-state only.
        c0 = _constraint_values(p[:6], t0, ckinds)
        G = np.zeros((nc, 6 + k))
        for kk in range(6):
            h = 1e-7 * max(1.0, abs(p[kk]))
            sp = p.copy(); sp[kk] += h
            G[:, kk] = (_constraint_values(sp[:6], t0, ckinds) - c0) / h
        return np.vstack([obs_rows, cw[:, None] * G])

    # _state is 6+k long on a fit already carrying a nongrav model (e.g. auto-discovery
    # fired, then the CLI --nongrav re-fits) -- seed from the 6-state only, A's from zero.
    p0 = np.concatenate([np.asarray(fit._state, float)[:6], np.zeros(k)])
    # A caller with a KNOWN-GOOD full seed (6-state + A's) -- e.g. the web UI re-seeding a model from
    # its own last-converged orbit -- passes it here to skip the profile-seed entirely and REPRODUCE
    # that orbit (the LM starts AT the minimum). Still fully polished on the current mask, so it adapts.
    if seed_state is not None:
        p0 = np.asarray(seed_state, float)
        if p0.shape[0] != 6 + k:                       # wrong arity -> ignore, fall back to the seed grid
            p0 = np.concatenate([np.asarray(fit._state, float)[:6], np.zeros(k)])
            seed_state = None
    # A RADIAL A1 is nearly degenerate with the semimajor axis -- both change only the orbital
    # period -- so the joint (state, A1) LM started from A1=0 sits in the gravity corner of a curved
    # narrow valley and never climbs it (and OVERSHOOTS to divergence when seeded off the ridge). We
    # PROFILE A1 first: freeze it on a coarse grid, re-fit the 6-state at each, and seed the joint
    # solve from the lowest-chi2 point -- already on the valley floor. This SEED is what reaches the
    # minimum find_orb's "several full steps" descent finds. (Rescaling the A axes by their natural
    # ~1-sigma size -- find_orb's adaptive derivative / MINPACK x_scale="jac" -- was measured to be a
    # strict no-op here once the seed lands on the valley floor: byte-identical chi2 and iteration
    # count, so it is not carried.)
    #   By DEFAULT we seed only the A1-only inverse-square `srp` model -- the one whose degeneracy is
    # acute enough that the un-seeded LM diverges. `radial_seed_all` widens the seed to EVERY model
    # carrying a radial A1 (srp2/srp3 and the whole comet model sequence), which is what lets comet2 descend
    # to its true out-gassing minimum (2026 NA1: gravity 0.68" -> comet2 0.48", srp 0.51"). It is opt-
    # in (the CLI --nongrav / an explicit UI model choice pass it) because auto-discovery ALREADY
    # delivers those models fine WITHOUT it -- most such fits are well-determined short arcs where the
    # profile is pure wasted cost -- so paying the seed on every discovery delivery only slows the
    # suite without changing an orbit. `profile_seed` is False on the discovery model-COMPARISON
    # trials (profiling every candidate across the gated NEO population would multiply discovery cost
    # several-fold; the winner is re-seeded at delivery). Default profile_seed True.
    _seed_radial = (seed_state is None) and (
        (0 in free_idx) if radial_seed_all else (free_idx == [0] and gr is ng.INVSQ_GR))
    # `force_seed` PRE-EMPTS the keep-the-better radial-seed branch below rather than composing
    # with it. The two grid the same axis, so running both would pay the profile twice for the
    # same seed -- but the real reason is the diagnostics: only the rescue block records
    # `rescue_chi2`, and force_seed's one caller (the delete-1 jackknife) reads that record to
    # tell a coefficient that COLLAPSED to zero (evidence -- it stays in the cloud) from one
    # that merely stalled there (not evidence -- dropped). Before this took precedence, an
    # A1-only srp model routed force_seed through the branch below, came back with an empty
    # diag, and every genuine collapse was discarded as an unexplained stall -- biasing the
    # jackknife cloud away from zeros, i.e. inflating z, the phantom-hiding direction. The
    # kept coefficient is the same either way when it is nonzero (measured on 2008 NA srp:
    # identical A1 to 12 digits); only the zero-explanation routing differs.
    _radial_branch = profile_seed and _seed_radial and not force_seed
    result = least_squares(resid, p0, jac=jac, method="lm",
                        xtol=tol, ftol=tol, max_nfev=max_nfev)
    if _radial_branch:
        # KEEP-THE-BETTER seed: also solve from the profile-seed and keep whichever chi2 is lower.
        # The bare-p0 LM above sticks in the gravity corner for a state-degenerate radial A1 (srp:
        # A1=0; srp2 on a longer arc: A1=A2=0 -- the web-UI "shows zero for both" bug), which the
        # seed escapes; but on a well-determined multi-A model (srp3 here) the bare LM can reach a
        # LOWER minimum than the seed lands on, so seeding unconditionally would regress it. Running
        # both and keeping the lower-chi2 solution is never-worse than either alone. (The un-seeded
        # branch -- discovery comparison trials, non-radial models -- is byte-identical to before.)
        p0_seed = _profile_seed_radial(backend, t0,
                                       np.concatenate([np.asarray(fit._state, float)[:6], np.zeros(k)]),
                                       scale, gr, tt, Rs, ra, dec, cosd, sra, sdec, m)
        res_seed = least_squares(resid, p0_seed, jac=jac, method="lm",
                                 xtol=tol, ftol=tol, max_nfev=max_nfev)
        if float(np.sum(res_seed.fun**2)) < float(np.sum(result.fun**2)):
            result = res_seed
    # STALL RESCUE. A model carrying a free A1 or A2, seeded at zero, sits in the gravity
    # corner of the A-vs-semimajor-axis valley: a radial A1 is near-degenerate with the period, and a
    # transverse A2 IS a secular along-track drift, so both trade against `a`. The damped LM's first
    # step is rejected and the trial returns A = 0.000e+00 EXACTLY, with chi2 bit-identical to
    # gravity -- which the selection reads as "+0% gain, not justified" and declines. That is not a
    # null: a fitted parameter that is exactly zero never moved. Found on 2008 NA (131 obs), which
    # declined a Yarkovsky worth F=769 that way for twelve days while its 2-parameter srp2 trial
    # escaped the same corner; then measured on the gate population (2000 objects, 2026-08-25):
    # 6.5% of fits had at least one trial stall, radial (srp / srp2) three times as often as A2.
    #
    # The cure is the profile seed the delivered srp fit already uses -- freeze the axis on a coarse
    # grid, re-fit the 6-state at each, start the joint LM from the lowest point, which is ON the
    # valley floor -- walked outward from zero (see _profile_seed_axis). Profiling EVERY trial was
    # measured several-fold slower on the gated population, so the rescue is gated on _cold_cornered,
    # the same cheap one-eval stall signature the comet corrector uses: a trial whose LM moves
    # normally pays nothing. Keep-the-better, so it can only lower a trial's chi2, never raise it.
    # Comet g(r) models are excluded: they have their own full-step corrector below. `ng_diag`
    # records what happened, because "+0% gain" from a coefficient that never moved and "+0% gain"
    # from a coefficient that was determined to be zero are different findings and must stay
    # distinguishable downstream (the discovery trace, a population sweep -- see _sweep.fit_one).
    ng_diag = {}
    if (seed_state is None and not _radial_branch and not citems
            and gr is not ng.COMET_GR and any(ax in (0, 1) for ax in free_idx)):
        chi2_cold = float(np.sum(result.fun**2))
        zero_axes = [ng.A_AXES[ax] for j, ax in enumerate(free_idx) if result.x[6 + j] == 0.0]
        ng_diag = {"stalled": len(zero_axes) == k, "zero_axes": zero_axes,
                   "cornered": bool(_cold_cornered(result, _obs_resid, fit, k, m)),
                   "status_cold": int(getattr(result, "status", -1)), "nfev_cold": int(getattr(result, "nfev", 0)),
                   "chi2_cold": chi2_cold, "rescued": None}
        if ng_diag["cornered"] or force_seed:
            p_cold = np.concatenate([np.asarray(fit._state, float)[:6], np.zeros(k)])
            solved = {}                                # seed bytes -> LM result, this rescue only
            for axis in free_idx:
                if axis not in (0, 1) or (not force_seed and axis not in NONGRAV_RESCUE_AXES):
                    continue
                # One profile per axis per comparison, whichever models ask for it: the profile
                # freezes THIS axis on the grid with every other A at zero and re-fits the 6-state,
                # so srp2's A1 profile is srp's A1 profile and its A2 profile is yarkovsky's --
                # the same fits, to the last bit. `rescue_cache` (one dict per _ng_evaluate) holds
                # the physical (state6, A) best point so a nested model's rescue costs it two
                # warm LMs instead of ~20 six-state fits (measured 3.4x on the whole fit).
                ckey = (axis, id(gr))
                if rescue_cache is not None and ckey in rescue_cache:
                    st6, a_phys = rescue_cache[ckey]
                    p0_ax = p_cold.copy(); p0_ax[:6] = st6; p0_ax[6 + free_idx.index(axis)] = a_phys / scale
                    ng_diag.setdefault("profile_cached", []).append(ng.A_AXES[axis])
                else:
                    grid = A1_PROFILE_AMR_GRID * ng.SRP1AU if axis == 0 else A2_PROFILE_GRID
                    p0_ax = _profile_seed_axis(backend, t0, p_cold, scale, gr, tt, Rs, ra, dec, cosd,
                                               sra, sdec, m, axis=axis, grid_phys=grid,
                                               slot=free_idx.index(axis), continue_from_zero=True,
                                               early_stop=True,
                                               zero_cache=(rescue_cache.setdefault("zero", {})
                                                           if rescue_cache is not None else None))
                    if rescue_cache is not None:
                        rescue_cache[ckey] = (p0_ax[:6].copy(),
                                              float(p0_ax[6 + free_idx.index(axis)]) * scale)
                # The TIGHT tol, for the same reason the seeded-srp trial takes it: at the loose
                # comparison tol the damped LM slides the profile seed straight back down to A = 0
                # and the seed is wasted (see NONGRAV_INTERACTIVE_TOL). Only a stalled trial pays it.
                # Two axes' profiles can hand back the SAME seed -- both minima at zero, which is
                # the common null case -- and the LM is deterministic, so the second solve would
                # repeat the first to the last bit. Reuse it.
                skey = p0_ax.tobytes()
                res_ax = solved.get(skey)
                if res_ax is None:
                    res_ax = solved[skey] = least_squares(
                        resid, p0_ax, jac=jac, method="lm",
                        xtol=min(tol, NONGRAV_INTERACTIVE_TOL),
                        ftol=min(tol, NONGRAV_INTERACTIVE_TOL), max_nfev=max_nfev)
                chi2_ax = float(np.sum(res_ax.fun**2))
                ng_diag.setdefault("rescue_chi2", {})[ng.A_AXES[axis]] = chi2_ax
                if chi2_ax < float(np.sum(result.fun**2)):
                    result = res_ax
                    ng_diag["rescued"] = ng.A_AXES[axis]
        ng_diag["chi2"] = float(np.sum(result.fun**2))

    # find_orb full-step corrector (COLD path only): the damped LM can stall in the gravity corner of
    # a degenerate COMET model (2026 NA1 comet2 sticks at 0.62"). When the LM barely improved on the
    # gravity residual (the stuck signature), run find_orb's bounded Gauss-Newton corrector from cold
    # and KEEP THE BETTER -- it reaches the out-gassing minimum the LM misses (comet2 -> 0.49", near
    # the 0.48" the ~5x-costlier radial seed reaches), yet being kept only when it wins it can never
    # regress. Restricted to COMET g(r) models: it is the ONLY family the corrector ever helps -- on
    # the radial-SRP srp/srp2 it just stalls in the same corner and keep-the-better discards it, so
    # firing it there is pure wasted cost on the whole gated-NEO population (most of which carry no
    # radial signal). Confining it to comet models means it runs only for genuine comet candidates
    # (Tisserand <= 3.01, i.e. discover_comet_nongrav's gated set) plus an explicit --nongrav comet
    # fit -- exactly where a stuck out-gassing signal might be hiding. Also skipped when radial-seeded
    # (the seed already lands deeper), constrained, or when the LM already fit well (rms << gravity).
    if (seed_state is None and not _radial_branch and not citems
            and gr is ng.COMET_GR and _cold_cornered(result, _obs_resid, fit, k, m)):
        def _a_phys(p):
            A = [0.0, 0.0, 0.0]
            for j, ax in enumerate(free_idx):
                A[ax] = p[6 + j]
            return A
        p_cold = np.concatenate([np.asarray(fit._state, float)[:6], np.zeros(k)])
        _, _, _ro0 = _make_ng_residual(backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr,
                                       np.empty(0), _a_phys, [], np.empty(0), m)
        hA_nat = _adaptive_ng_hA(_ro0, p_cold, k, scale, m)
        _, j_ph, ro_ph = _make_ng_residual(backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr,
                                           hA_nat, _a_phys, free_idx, np.ones(k), m)
        p_corr, chi2_corr = _full_step_descent(ro_ph, j_ph, p_cold, arc_days)
        if chi2_corr < float(np.sum(result.fun**2)):
            # polish the corrector's basin with the scaled LM so `result` (and its Jacobian, used for the
            # sigmas below) is a clean converged fit in the LM's units; keep it only if still better.
            res_c = least_squares(resid, np.concatenate([p_corr[:6], p_corr[6:] / scale]), jac=jac,
                                  method="lm", xtol=tol, ftol=tol, max_nfev=max_nfev)
            if float(np.sum(res_c.fun**2)) < float(np.sum(result.fun**2)):
                result = res_c
    # Delivery convergence: if the kept LM stopped at the max_nfev cap (status 0) rather
    # than at a minimum, warm-restart once from its own solution -- a close seed either
    # finishes in a few evals or proves the cap wasn't the obstacle. Keep the better; a
    # still-capped result carries status 0 into FittedOrbit's `converged` flag below.
    if getattr(result, "status", 1) == 0:
        res_r = least_squares(resid, result.x, jac=jac, method="lm",
                              xtol=tol, ftol=tol, max_nfev=max_nfev)
        if float(np.sum(res_r.fun**2)) <= float(np.sum(result.fun**2)):
            result = res_r
    extra_meta = {"constraints": {kk: float(t) for kk, t in citems}} if citems else None
    fit2 = _deliver_nongrav_orbit(ctx, fit, model, result, mask, ng_diag=ng_diag,
                                  profile_sigmas=profile_sigmas, extra_meta=extra_meta,
                                  asteroids=asteroids)
    # Keep the gravity orbit if the nongrav refit didn't actually improve the weighted
    # rms (a noisy/short arc carries no nongrav signal -- don't fabricate one). Skipped
    # when the caller forced the model (revert_if_worse=False) or fit a different mask.
    if revert_if_worse and fit2.rms_normalized() > fit.rms_normalized() + 1e-9:
        if verbose:
            print(f"  nongrav[{model}]: no improvement "
                  f"({fit.rms_normalized():.3f} -> {fit2.rms_normalized():.3f}); keeping gravity orbit")
        return fit
    if verbose:
        _p, _s = fit2.nongrav_params, fit2.nongrav_sigmas
        ps = ", ".join(f"{nm}={_p[nm]:.3e}±{_s[nm]:.1e}" for nm in _p)
        print(f"  nongrav[{model}]: rms_norm {fit.rms_normalized():.3f} -> "
              f"{fit2.rms_normalized():.3f}  ({ps})")
    return fit2


# --- automatic non-gravitational discovery -----------------------------------------------
# For a converged GRAVITY fit in the regime where a nongrav is detectable -- q < 1.4 au AND
# multi-apparition (Yarkovsky is a SECULAR along-track drift, so it needs many revolutions
# over multiple oppositions to accumulate) -- decide whether a non-gravitational model is
# warranted by turning the A1/A2 terms on, comparing, and keeping one ONLY if it clears a
# weighted-chi2 effect-size bar AND a nested F-test AND a size-aware physical-plausibility
# ceiling. The STATISTICS are the arbiter -- there is no heuristic systematic-trend pre-screen
# (a population check vs JPL showed it both blocks real large-A2 detections and passes
# force-model-deficient junk; the chi2 effect-size bar is what actually resists false
# positives). ON by default (PYROORB_AUTO_NONGRAV=0 to disable); validated zero-false-positive
# against JPL across the very-low-q strong-GR control class.
class _AsteroidForce(PinnedForce):
    """A force backend with the 16-perturber ephemeris FORCED ON for every call that accepts
    `asteroids=` (forcemodel.PinnedForce pinned True) -- e.g. the delivery-time force model for
    a long-arc, belt-sampling non-gravitational orbit. ONE wrapper object stored in the
    delivered ctx rather than per-call-site threading, so everything that ctx reaches -- the
    delivery LM's residuals and rescue profile, the radar joint's states sweep, the
    residual/photometry rebuild, a later jackknife or web re-fit -- runs under ONE force model
    and cannot drift planets-only halfway through (which is exactly how 1862 Apollo's radar
    "fit" a trajectory 20-50 km off: the divergence is ~0.4" to the optical and hundreds of
    sigma to a delay). An inner pin is replaced; a SelfExcludedForce inside is kept (+15ast)."""

    def __init__(self, backend):
        super().__init__(backend, True)


def with_force_model(ctx, asteroids, fit=None):
    """`ctx` re-bound to the asteroid perturbers ON or OFF: ``(ev2, ctx2)``.

    The ONE place a fit's force model is swapped (refine_asteroids, refit_force_model, the
    nongrav delivery, the web's restored sessions). The new ctx backend is pinned
    (forcemodel.PinnedForce, around any self-exclusion), the evaluator's predict and Jacobian
    are bound to THAT backend, and the ctx's flag, meta and label say the same thing -- so an
    orbit built from ctx2 cannot integrate in one field and report another. The evaluator is a
    shallow copy with its caches cleared; its `nongrav` is left for the caller."""
    import copy as _copy
    base = ctx["backend"]
    # what will actually run: no perturbers off ASSIST, and none for a perturber object whose
    # exclusion cannot be built here (force_assist then refuses them; say so, don't pretend)
    asteroids = (bool(asteroids) and is_assist(base)
                 and (excluded_body(base) is None or _self_excluding(base)))
    backend = pin(base, asteroids)
    ev = ctx["ev"]
    ev2 = _copy.copy(ev)
    ev2._predict = backend.predict_radec_batch
    ev2._predict_jac = (None if ev._predict_jac is None
                        else getattr(backend, "predict_radec_jac_batch", None))
    ev2._resid_key = ev2._sub_key = None
    ctx2 = dict(ctx)
    ctx2["ev"] = ev2
    ctx2["backend"] = backend
    ctx2["asteroids"] = asteroids
    meta = {k: v for k, v in (ctx.get("meta") or {}).items() if k != "asteroids"}
    if ctx2["asteroids"]:
        meta["asteroids"] = True
    ctx2["meta"] = meta
    ctx2["backend_name"] = _force_model_label(ctx, ctx2["asteroids"], fit)
    return ev2, ctx2


# The 16 asteroid perturbers at NONGRAV DELIVERY, gated. A long arc whose aphelion
# samples the belt accumulates tens of km of along-track divergence under the planets-only
# model over decades -- invisible to optical astrometry (~0.4" at NEO ranges), catastrophic at
# radar delay precision, and aliased directly into the secular A2 the delivery exists to
# measure (1862 Apollo: planets-only joint solve +1.3e-14, perturbed -5.1e-15 vs JPL's
# radar-anchored -3.66e-15). Discovery COMPARISON trials stay planets-only: selection is a
# relative chi2 question at ~100x the integration count, and the winner is re-fit at delivery
# under the gated model. A discovery gate on perihelion leaves aphelion unbounded, so a
# low-q, high-Q object with a >10-year arc does reach this delivery leg.
NONGRAV_ASTEROIDS_ARC_DAYS = 3650.0   # >10-year arcs: where the divergence outgrows the noise
NONGRAV_ASTEROIDS_Q_MIN = 1.7         # aphelion reaches the belt (the perturbers' home)


def _asteroids_detection_recheck(ctx, grav_fit, final, mask, model, N):
    """The perturbed delivery's sharpest caveat, closed: on a gated (perturbed-delivery) object the DETECTION
    was decided planets-only -- and the unmodeled perturber drift inflates the gravity
    baseline's chi2, which is exactly the direction that manufactures justification. So a
    belt-crossing decade arc could "detect a nongrav" that is really the missing perturbers.
    Re-ask the ONE decisive question under the perturbed model: does the chosen model still
    clear the BIC bar against a gravity baseline REFIT under the same physics? Costs one extra
    6-state LM (the model side is the delivery fit itself, already perturbed). Returns
    (record, still_justified); the caller records the number ALWAYS -- it is what a human
    adjudicating a gated detection should read -- and reverts to gravity when it collapses,
    mirroring the settle loop's own declined-retest path."""
    import math
    import pyroorb.nongrav as ng
    from scipy.optimize import least_squares
    ev = ctx["ev"]
    backend = _AsteroidForce(ctx["backend"])
    t0 = ctx["epoch_tdb"]
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
    ra, dec, cosd = ev.ra[idx], ev.dec[idx], ev.cosd[idx]
    sra, sdec = ev.fit_sig_ra[idx], ev.fit_sig_dec[idx]
    m = len(idx)
    spec = ng.model(model)
    resid, jac, _ = _make_ng_residual(backend, t0, tt, Rs, ra, dec, cosd, sra, sdec,
                                      spec["gr"], np.empty(0),
                                      a_of_p=lambda p: [0.0, 0.0, 0.0],
                                      free_cols=[], col_scale=np.empty(0), m=m)
    result = least_squares(resid, np.asarray(grav_fit._state, float)[:6], jac=jac,
                        method="lm", xtol=1e-10, ftol=1e-10, max_nfev=200)
    # Baseline floor: the gravity side must be the best gravity anyone demonstrated -- a
    # stalled re-fit LM would inflate chi2_grav, which is the direction that fakes
    # justification. The delivered orbit's own 6-state at A=0 is a free evaluation.
    chi2_grav = min(float(np.sum(result.fun ** 2)),
                    float(np.sum(resid(np.asarray(final._state, float)[:6]) ** 2)))
    chi2_model = float(np.sum(np.asarray(final._res.fun) ** 2))
    dp = len(spec["params"])
    dbic = _dbic(chi2_grav, chi2_model, dp, N)
    rec = {"chi2_grav": chi2_grav, "chi2_model": chi2_model, "dp": dp,
           "dbic": round(dbic, 3), "margin": NONGRAV_BIC_MARGIN}
    return rec, bool(dbic >= NONGRAV_BIC_MARGIN)


def _ng_mask_recheck(ctx, sel2, reject_mask, max_nfev, verbose=False, comet=False):
    """The settle's re-test scores gravity on the MODEL-chosen mask: the rejection round ran
    under the candidate's own residuals, so an observation kept because the model absorbs it
    (or dropped because gravity cannot) tilts the dBIC toward the model by construction. That
    tilt is not hypothetical -- 2021 LW3 delivered yarkovsky at retest dBIC 16.1 whose honest
    same-data margin is 12.4 (below the bar; one F51 obs costing gravity 12.9 chi2 but the
    model only 9.6), and 2022 OC3 claimed 34.6 riding two model-kept observations worth 23
    chi2 -- while the raw arcsec rms barely moved (0.407 vs 0.401), which is what a phantom
    looks like.

    So re-ask the ONE decisive question on data both sides accept: run the gravity-side
    rejection to ITS own fixed point (auto_reject's redeem/reject hysteresis, <=3 rounds --
    _final_reclaim's loop; verified to reproduce the pure-gravity pipeline's mask on the
    diagnosis objects), intersect with the model's settled mask, refit BOTH sides on the
    intersection, and require the same BIC margin there. Planets-only on both sides -- the
    perturbed-baseline question is _asteroids_detection_recheck's, asked separately.

    A DISPUTED observation (model mask keeps it, gravity's fixed point rejects it) is NOT
    simply excluded -- that was the first cut of this recheck, and the suite caught it
    amputating real signal: on 2003 BN4 gravity's own rejection clips 13+ observations that
    fit yarkovsky at 0.1-1.3 sigma while gravity fails them at 4-22 sigma in coherent night
    blocks (2010 VQ / 2004 JN1 / 2003 HM the same shape) -- those ARE the Yarkovsky evidence,
    and an intersection scored the model without it (BN4's honest gain collapsed to ~0.5 chi2,
    dBIC -4.8, a JPL-confirmed detection reverted). What separates that shape from the phantom
    is measurable per obs: a disputed obs COUNTS for the model iff the model actually fits it
    -- normalized residual under REDEEM_SIGMA (3.0), the redeem machinery's own "this obs
    belongs" bar. LW3's disputed obs sits at 3.05 under the MODEL TOO (absorbed junk, not
    explained signal); BN4's signal blocks sail under it. So the comparison runs on
    (model-mask AND gravity-mask) plus the disputed obs the model genuinely holds, and only
    both-sides-bad junk is left out of evidence.

    On a RADAR object the verdict is recorded but never vetoes (`veto_suppressed`): the
    optical-only trial's coefficient is provisional there -- 2003 HM's
    retest trial holds the radar-less A2 (-1.88e-14, 6x under JPL), under which the signal
    obs read ~3.1 sigma and every disputed obs looks unheld, while the delivered radar-joint
    orbit fits them at 0.1-0.4 sigma -- and the radar joint that runs next is the strongest
    same-data adjudicator there is. Vetoing pre-radar would drop the joint solve entirely,
    which would drop radar exactly when a nongrav is active.

    The mask comparison (Phase A) is skipped -- recorded, not vetoed -- when the fixed points
    are identical (the retest already measured exactly that) and, as a backstop, when the
    comparison mask still loses >20% of the model's observations. Phase B (the raw-visibility
    floor, NONGRAV_MIN_RMS_GAIN) ALWAYS runs: on the same mask, the model must improve the
    raw arcsec rms by the floor fraction -- the tell on 2021 LW3 and 2022 OC3, whose
    weighted-chi2 margins rode tracklet-correlated systematics while the raw rms moved ~1.5%
    (real detections on such arcs move it 6-42%). A leave-one-tracklet-out phase B was tried
    first and WITHDRAWN: with honest refits it taxes arc SPARSENESS, not phoniness -- it
    killed 2014 US7 (raw gain 14%, dBIC 45 on identical masks, i.e. real by every other
    light) and 19 of 37 JPL-positive controls, because deleting the strongest tracklet of a
    2-opposition arc removes a large share of one apparition's constraint no matter how real
    the force is. Costs 2-3 gravity LM fits + one model fit, on the rare detection path only.
    PYROORB_NG_MASK_RECHECK=0 disables (A/B + emergencies). Returns (record,
    still_justified); the caller records ALWAYS and reverts on collapse with `rec["why"]`
    naming which bar collapsed."""
    import math
    import pyroorb.arc_extend as ae
    import pyroorb.nongrav as ng
    if os.environ.get("PYROORB_NG_MASK_RECHECK", "1") == "0":
        return {"skipped": "env"}, True
    radar_pending = bool((ctx.get("meta") or {}).get("radar")
                         and os.environ.get("PYROORB_RADAR", "1") != "0")
    choice = sel2["choice"]
    ev = ctx["ev"]
    try:
        model_mask = np.asarray(reject_mask, bool)
        mask_g = model_mask.copy()
        state = np.asarray(sel2["grav"]._state, float)[:6].copy()
        st, _res = ev.fit(state, mask_g, thin=False, polish=True)
        state = np.asarray(st, float)[:6]
        for _ in range(3):
            m2 = mask_g.copy()
            rv = ae.auto_reject_within_arc(ev, m2, state)
            if rv <= 0 or not (m2 ^ mask_g).any():
                break
            mask_g = m2
            st, _res = ev.fit(state, mask_g, thin=False, polish=True)
            state = np.asarray(st, float)[:6]
        n_model = int(model_mask.sum())
        rec = {"n_model": n_model, "n_grav": int(mask_g.sum()),
               "margin": NONGRAV_BIC_MARGIN}
        dp = len(ng.model(choice)["params"])
        trial_fit = sel2["models"][choice]["fit"]
        trial_rows = trial_fit.residuals()

        def _msig(i, rows):
            r = rows[i]
            return float(np.hypot(r["d_ra_arcsec"] / ev.sig_ra[i],
                                  r["d_dec_arcsec"] / ev.sig_dec[i]))

        def _revert(why):
            rec["why"] = why
            if radar_pending:                  # the radar joint adjudicates; see docstring
                rec["veto_suppressed"] = "radar pending"
                return rec, True
            return rec, False

        # --- Phase A: the honest-mask comparison. Skipped (recorded) when it would only
        # repeat the retest's own numbers, or under the divergence backstop.
        both = model_mask & mask_g
        disputed = np.flatnonzero(model_mask & ~mask_g)
        kept = 0
        for i in disputed:
            # A disputed obs testifies for the model only if the model genuinely holds it
            # (normalized residual under the redeem bar); both-sides-bad junk stays out.
            # A non-finite residual (degenerate/zero sigma, a bad row) is UNMEASURABLE, not
            # a demonstrated hold -- it must not default to "kept for the model", or the
            # recheck this obs is meant to inform gets inflated evidence in exactly the
            # case it can least trust.
            ms = _msig(i, trial_rows)
            if np.isfinite(ms) and ms < ae.REDEEM_SIGMA:
                both[i] = True
                kept += 1
        if disputed.size:
            rec["disputed"] = int(disputed.size)
            rec["disputed_kept"] = kept
        n_both = int(both.sum())
        rec["n_both"] = n_both
        grav_cmp = model_cmp = None
        if not (model_mask ^ both).any():
            rec["skipped"] = ("masks agree" if not (model_mask ^ mask_g).any()
                              else "model holds every disputed obs")
            both = model_mask
        elif n_both < 0.8 * n_model:
            rec["skipped"] = "mask divergence"  # backstop; see docstring
            both = model_mask
        else:
            st_b, res_b = ev.fit(state.copy(), both, thin=False, polish=True)
            grav_cmp = _build_orbit(ctx, st_b, both, res_b, None)
            N2 = grav_cmp.n_resids()
            model_cmp = refine_nongrav(grav_cmp, choice, mask=both, max_nfev=max_nfev,
                                       revert_if_worse=False, profile_sigmas=False,
                                       seed_state=_lift_nongrav_state(trial_fit, choice))
            chi2_m = model_cmp.rms_normalized() ** 2 * N2
            # Baseline floor: a stalled gravity re-fit inflates chi2_g, faking margin -- the
            # model orbit's own 6-state at A=0 is a free demonstration of what gravity can do
            chi2_g = min(grav_cmp.rms_normalized() ** 2 * N2,
                         float(ev.rms_norm(np.asarray(model_cmp._state, float)[:6], both)
                               ** 2 * N2))
            dbic = _dbic(chi2_g, chi2_m, dp, N2)
            rec.update(chi2_grav=round(chi2_g, 3), chi2_model=round(chi2_m, 3), dp=dp,
                       dbic=round(dbic, 3))
            if dbic < NONGRAV_BIC_MARGIN:
                return _revert(f"dBIC {dbic:.1f} on shared data -- the justification rode "
                               "on the model-chosen mask")

        # --- Phase B: the raw-visibility floor. BIC counts weighted chi2
        # wherever it comes from, so a tracklet-correlated systematic on a few tight-sigma
        # obs can buy the whole margin while the arc's raw rms moves ~1.5% (2021 LW3,
        # 2022 OC3) -- a real detection on these arcs moves it 6-42%. Same-mask raw gain,
        # from the Phase A refits when they ran, else from the retest's own numbers.
        if grav_cmp is not None:
            # the same baseline floor in the raw currency: gravity's rms is the best demonstrated
            rms_g = min(grav_cmp.rms_residual(),
                        float(ev.rms_arcsec(np.asarray(model_cmp._state, float)[:6], both)))
            rms_m = model_cmp.rms_residual()
        else:                                  # comparison == the retest's own mask + fits
            rms_g, rms_m = sel2["rms0"], sel2["models"][choice]["rms"]  # (baseline-floored upstream)
        gain = (rms_g - rms_m) / rms_g if rms_g and rms_g > 0 else 0.0
        rec["rms_gain"] = round(gain, 4)
        # COMET path only: the 8% floor was calibrated on seven asteroids
        # and yields to an OVERWHELMING same-data margin -- the level the jackknife leg already
        # treats as beyond overturning. On a dense comet arc the thrust bends the whole arc,
        # so the raw rms moves little while gravity has to clip the arc's ends to cope: 2I
        # gains 5.2% raw at dBIC 584, where the phantom the floor exists for (2022 OC3, an
        # asteroid) gains 5.6% riding a claimed 34.6. Phase A's shared-data dBIC when it ran,
        # else the retest's own margin.
        if comet and gain < NONGRAV_MIN_RMS_GAIN:
            shared = rec.get("dbic")
            if shared is None and sel2.get("chi2_0") and sel2.get("N"):
                shared = _dbic(sel2["chi2_0"], sel2["models"][choice]["chi2"], dp,
                               sel2["N"])
            if shared is not None and shared >= NONGRAV_JK_SKIP_FACTOR * NONGRAV_BIC_MARGIN:
                rec["rms_floor_yielded"] = round(float(shared), 3)
                return rec, True
        if gain < NONGRAV_MIN_RMS_GAIN:
            return _revert(f"raw rms gain {100 * gain:.1f}% < {100 * NONGRAV_MIN_RMS_GAIN:.0f}% "
                           f"({rms_g:.3f}\" -> {rms_m:.3f}\" on the same mask) -- the model "
                           "explains nothing visible")
        return rec, True
    except Exception as ex:                    # a recheck that cannot run must not veto
        return {"skipped": f"{type(ex).__name__}: {ex}"[:120]}, True


def _nongrav_asteroids_gate(fit):
    """Should this orbit's non-gravitational DELIVERY carry the 16 asteroid perturbers?
    True iff the arc is long enough and the aphelion belt-sampling enough for the
    planets-only divergence to matter, AND the ASSIST backend + sb441-n16 kernel are
    actually available. PYROORB_NG_ASTEROIDS=0 disables (A/B + emergencies)."""
    if os.environ.get("PYROORB_NG_ASTEROIDS", "1") == "0":
        return False
    try:
        ctx = getattr(fit, "_ctx", None) or {}
        if not hasattr(ctx.get("backend"), "helio_states_multi"):
            return False                       # rebound fallback: no perturber support
        import pyroorb.force_assist as _fa
        if _fa.asteroid_kernel_path() is None:
            return False
        if fit.arc_length_days() < NONGRAV_ASTEROIDS_ARC_DAYS:
            return False
        Q, _sig = fit.aphelion_distance()
        return Q is not None and float(Q) > NONGRAV_ASTEROIDS_Q_MIN
    except Exception:
        return False


# Perihelion gate. Yarkovsky is a transverse 1/r^2 force, so its secular along-track drift per
# unit A2 scales as 1/(a^3 (1 - e^2)) and is earned near perihelion: an eccentric NEO with a
# above 2 au drifts as fast as a circular orbit at 1.6 au (2015 KJ19, a = 2.08, q = 0.53,
# sits exactly there). A semi-major-axis cut therefore shut out real detections: of JPL's 566
# published A2 at SNR >= 3, a < 1.6 admits 507 (90%), q < 1.3 admits 565 (the one left out is
# 362P, an active object the comet path owns). 1.4 sits a little past the NEO boundary as
# margin. The bars below were calibrated inside a < 1.6; the eccentric and Amor objects this
# gate adds are judged by the same bars.
NONGRAV_Q_MAX_AU = 1.4      # perihelion gate (the drift is earned near perihelion)
NONGRAV_MIN_OPP  = 2        # multi-apparition gate (drift must accumulate across oppositions)
# Model selection works on the WEIGHTED chi2 (sum of normalized residuals^2), not the raw-arcsec
# rms. Raw rms is noise-floor dominated, so it badly understates a real drift -- 2009 OS5's
# Yarkovsky cuts chi2 nearly in half (179->92, a 49% drop) yet moves the raw rms only +9%, which
# fell under the old flat 10% raw-rms floor and stranded a clear (z=14, JPL-confirmed) detection.
# A model must clear TWO bars: (1) a nested F-test on the chi2 drop vs gravity (the added A's are
# statistically warranted) AND (2) a real EFFECT SIZE -- a >=NONGRAV_CHI2_GAIN fractional chi2
# reduction.
#
# HISTORICAL, AND ITS CONTROL WAS WRONG (superseded 2026-08-27 -- selection is BIC
# now, see NONGRAV_BIC_MARGIN; this block survives only to explain the constants the comet model sequence
# still uses). The argument was that the effect-size bar is load-bearing because with long arcs
# even a 1-2% chi2 drop is "significant" by F-test/AIC/BIC, citing control 2002 NV16 (F=132,
# p=3e-29 on a +9% chi2 / +1.7% raw drop) as significance manufacturing a nongrav on a
# gravity-only object. 2002 NV16 IS NOT GRAVITY-ONLY. JPL publishes A2 = -2.33620767306597e-14
# +/- 4.318e-15 au/day^2 for it (363305), and the BIC rule that replaced this floor fits
# -2.266e-14 +/- 3.8e-15 on the same 8226-day arc -- 0.12 sigma agreement. The floor was not
# holding back a phantom, it was suppressing a real detection, which is the same thing it did to
# 2003 HM and Bennu. Note also the direction: under the null with independent errors a spurious
# parameter buys dchi2 ~ chi2(dp) REGARDLESS of N, so it is the FRACTIONAL bar that scales wrongly
# (chi2 grows with N, so a fixed fraction demands ever more from the long arcs where a Yarkovsky
# is most detectable) -- which is exactly the population it was refusing.
# The effect-size GAP sets the bar: genuine Yarkovskys run from ~13% (2015 XF261) up
# past 49% (2009 OS5), while gravity-only objects that reach selection mostly give <=2% -- but a
# low-q IEO with strong un-modeled dynamics (163693 Atira, JPL gravity-only) can absorb up to
# ~8% into a spurious A2 on the first pass. 10% sits above that worst-case true-negative (with
# margin) and below the weakest real, so a near-FP like Atira fails the bar outright instead of
# only being caught later by the post-rejection re-test (defense in depth + it avoids paying for
# the extra rejection round on every such object). Model choice applies a physical prior: in this
# low-q multi-opposition regime the secular trend is Yarkovsky's (A2) signature, so A2-only wins
# whenever justified and the radial-SRP A1 / A1+A2 models are a fallback (see _ng_choice) -- this
# blocks the SRP<->Yarkovsky mis-attribution (e.g. 2008 DB, where a degenerate A1 would otherwise
# edge out the clean Yarkovsky).
# 0.15, not the 0.10 that the GATED regime used. Removing the systematic-trend pre-screen lets
# every a<1.6 multi-opp object reach the selection, which shifts the gravity-only control chi2
# distribution UP, so the old 0.10 leaks marginal fires. On a 150-object broad no-A2 NEA sweep:
# 0.10 -> 4.7% fire, 0.15 -> 2.0% (3/150). 0.15 trades a touch of that ~2% rate (the
# accepted ceiling) for recovering 41/46 detectable JPL Yarkovskys -- +23 over the gated 18,
# i.e. the false-negative rate on detectable objects drops 61% -> 11%. The three 0.15 "fires"
# (2022 AV4, 2014 QS295, 2012 JR4) are NOT noise: chi2-gain 18-20%, z~5, physically plausible
# A2 on well-observed multi-opp NEOs JPL happens to fit gravity-only -- most likely real
# Yarkovskys JPL simply hasn't solved for, not engine artifacts (the EXTREME low-q strong-GR
# class, where a force-model deficiency would bite, fired 0/30).
NONGRAV_CHI2_GAIN  = 0.15   # required fractional weighted-chi2 reduction (effect size)
NONGRAV_FTEST_ALPHA = 0.05  # nested F-test significance for the added nongrav parameter(s)
# Detection significance required to REPORT the nongrav -- the cut that makes our fire/no-fire
# match JPL's published Yarkovsky catalog (i.e. emulate JPL: report a nongrav iff JPL would).
# Significance is sqrt(F) from the nested F-test (the chi2-drop t-statistic), NOT |A|/sigma from
# the linearized covariance: the latter is unreliable when the Jacobian is ill-conditioned (e.g.
# 2007 XB23 cuts chi2 70% with F=76 -- sqrt(F)=8.7 -- yet its covariance gives a spurious z=0.6,
# which a raw-z cut would wrongly reject). Calibrated on the JPL a<1.6 A2 population: all real
# detections clear sqrt(F)>=5.6, while the moderate-significance objects JPL fits gravity-only --
# the ones we'd otherwise "discover" (2022 AV4, 2014 QS295, 2012 JR4) -- sit at sqrt(F) 5.0-5.2.
# A 5.5 cut separates them: drops those three (control fire rate 2.0% -> 0%) at no cost to the
# real detections. (The effect-size bar still guards the large-N significant-but-trivial case.)
NONGRAV_MIN_Z      = 5.5    # min sqrt(F) significance to report (matches JPL's detection bar)
NONGRAV_SRP2_A1_Z  = 7.0    # to UPGRADE a justified Yarkovsky (A2) to the A1+A2 model, the radial
                            # A1 must be INDEPENDENTLY significant at this marginal z in the joint
                            # fit (not just lower the joint chi2 -- a degenerate A1 can do that).
                            # Set well above the degenerate cases (2008 DB A1z~4.3, 2011 CL50 ~2.4)
                            # and below the genuine SRP+Yarkovsky 2009 BD (A1z~11). See _ng_choice.
NONGRAV_DROP_WORST = 2      # worst-residual obs masked OFF before the model comparison
NONGRAV_DISCOVER_NFEV = 40  # LM cap for the model-comparison fits DURING auto-discovery (not the
                            # direct UI refine_nongrav, which keeps the full budget). A poorly-
                            # constrained candidate (e.g. one A-term on a NEO) can grind to the old
                            # 200 cap WITHOUT converging (status=0) while contributing nothing the
                            # selection uses; the well-determined models converge in <=21 evals, so
                            # 40 leaves ample margin. 2011 CL50's discovery: ~15s -> ~5s, result
                            # unchanged; XB23-class fits (all converge in <20) are untouched.
NONGRAV_DISCOVER_TOL = 1e-7  # LM xtol/ftol for the model-COMPARISON trials only. The selection
                            # statistic (weighted-chi2 gain / sqrt(F)) settles in ~5 evals, but at
                            # the tight 1e-10 the LM keeps taking sub-decision-relevant steps to the
                            # nfev cap. Loosening to 1e-7 leaves every decision unchanged (Bennu/VD32/
                            # 2018 KL still decline; 2003 BN4 / 2011 CL50 / 2009 OS5 still fire; 2008
                            # DB still picks a clean A2-only Yarkovsky over the degenerate A1+A2) while
                            # ~halving discovery on the common decline path (Bennu 3.8s->1.7s). The cut
                            # is calibrated to 2008 DB's fine SRP<->Yarkovsky chi2 margin: 1e-6 corrupts
                            # that ranking (it declines) and 1e-4 drops BN4 outright, so 1e-7 is the
                            # loosest safe value. The CHOSEN model is re-fit at the tight default before
                            # delivery, so an APPLIED nongrav orbit keeps full precision. Selection stays
                            # ungated by systematics -- this changes only the LM budget, not the criteria.
# BIC selection. A nongrav model is justified when its Bayesian Information Criterion
# improves on gravity by NONGRAV_BIC_MARGIN -- dBIC = (chi2_grav - chi2_model) - dp*ln(N), where dp
# is the added coefficient count. This subsumes the two hand-set floors below (NONGRAV_CHI2_GAIN,
# NONGRAV_MIN_Z) with ONE principled bar and removes their N-dependence (a fractional chi2 cut means
# something different on 30 residuals than on 1200; ln(N) is exactly that correction). Margin 15 is
# "very strong" on the Kass-Raftery scale and, scored against JPL's published Yarkovskys on the gate
# population, lifts recall 67% -> 86% (recovers 2003 HM and Bennu, which the 0.15/5.5 floors refuse)
# at ~+1 real false positive. The old floors are kept as constants for the comet model sequence / reference.
NONGRAV_BIC_MARGIN = 15.0

def _dbic(chi2_ref, chi2_trial, dp, N):
    """dBIC = (chi2_ref - chi2_trial) - dp*ln(N): the evidence for a model with ``dp`` more
    parameters than the reference, on the SAME N residuals -- the one bar every
    non-gravitational selection is decided against (>= NONGRAV_BIC_MARGIN). Written ONCE
    so every deciding site shares one N floor. The floor is N >= 2 (ln 1 = 0
    would charge nothing for a parameter; N is a residual count, never below 2 on a fit
    that exists) and ``None`` for N is that floor too."""
    return (float(chi2_ref) - float(chi2_trial)) - dp * math.log(max(float(N or 2), 2.0))


# Second bar on discovery: an AUTO-DISCOVERED nongrav must also
# improve the raw arcsec rms on the same mask by this fraction -- "does it actually explain
# anything you can see". BIC keeps significance; this keeps visibility; radar overrides both
# (an optical trial's coefficient is provisional on a radar object -- 2003 HM's real signal
# needs the joint solve, and the dense mega-arcs whose real signal is raw-invisible -- 65717
# 0.0%, Ryugu 0.5% -- are all radar objects, so the suppression IS the dense exemption).
# The 8% sits where the measured distributions put seven hand-adjudicated object-level verdicts,
# all reproduced exactly: phantoms 2021 LW3 2.0% / 2022 OC3 5.6% below it; real 2003 HM
# 11.7% / 2014 US7 14.3% / 2004 JN1 43% / 2003 BN4 68% / 2010 VQ 77% above it. Any floor in
# 6-11% does that; 8% is the middle. The measured price, accepted by design: ~26
# of 42 low-dBIC-tail JPL-published positives fall below 8% -- detections that cannot be
# seen in the data we fit; their values remain reachable through the forced leg, and every
# revert records its gain. PYROORB_NG_MASK_RECHECK=0 disables the whole recheck.
NONGRAV_MIN_RMS_GAIN = 0.08

# The unified delivered verification's jackknife leg is paid only in the marginal
# band -- necessity dBIC under this multiple of the margin. 16 capped delete-1 refits cannot
# plausibly overturn dBIC >= 60 (and the concentration-carried route to a FAKE overwhelming
# margin is already closed by the shared-mask and raw-visibility bars), while in [15, 60) the jackknife is the one
# instrument that sees a tracklet-correlated systematic no likelihood quantity can.
NONGRAV_JK_SKIP_FACTOR = 4.0

# The per-model fields kept on the discovery record (`compare` / `compare_cold` / `retest`). One
# tuple so the three views cannot drift apart -- they are read side by side.
_NG_COMPARE_KEYS = ("rms", "chi2", "chi2_gain", "F", "pval", "ftest_ok", "z", "phys_ok",
                    "params", "diag")

NONGRAV_INTERACTIVE_TOL = 1e-8  # LM xtol/ftol for the SEEDED-SRP discovery comparison trial
                            # (the one armed radial trial in _ng_evaluate). It is an intermediate
                            # tol chosen on CONVERGENCE grounds: tight enough that the profile-seed
                            # stays on the valley floor (at the looser NONGRAV_DISCOVER_TOL the
                            # damped LM slides a state-degenerate A1 back down to 0 -- the seed is
                            # wasted and the trial reads +0%), but looser than the 1e-10 delivery
                            # default (a throwaway fire/no-fire decision, not a delivered orbit;
                            # the CHOSEN model is re-fit at the tight default before delivery).
                            # (The web UI no longer uses this -- an interactive model switch runs
                            # at the tight default so it reproduces the auto fit; see 5e56b06.)

# A1 = radial SRP (1/r^2); A2 = transverse Yarkovsky (1/r^2); A1+A2 = the two together.
_NG_TRIALS = (("srp", ("A1",)), ("yarkovsky", ("A2",)), ("srp2", ("A1", "A2")),
              ("srp3", ("A1", "A2", "A3")))

# The srp3 step is FITTED only when the best 1-2 axis trial is still sigma-strained -- its
# normalized rms above this after explaining what it can. A normal delivery sits at 0.4-0.7;
# 2009 HC's A2-only yarkovsky sat at 0.958 because the object's real signal has an
# OUT-OF-PLANE component (A3 marginal z=35, chi2 432 -> 152, rms 0.762" -> 0.407") that no
# 2-axis model can express -- systematics visible by eye in the residuals. Gating the step on
# the strain keeps the discovery cost at three trials for the ~99% of objects a 1-2 axis
# model already explains (see _srp3_continue).
NONGRAV_SRP3_STRAIN = 0.85


def _srp3_continue(out, key):
    """continue_fn for the asteroid model sequence: everything runs eagerly except the srp3 step,
    which is fitted only when the best model so far is still strained (its normalized rms
    above NONGRAV_SRP3_STRAIN -- unexplained systematics remain). Judges the SUB-models'
    strain by construction: it is consulted before srp3 is fitted, so a settle round where
    srp3 cleaned the mask still re-fits it (the 1-2 axis trials stay strained on the signal
    the mask kept).

    STANDS DOWN on a radar object -- the same principle as _ng_mask_recheck's
    veto_suppressed: an optical-only trial's coefficients are provisional there, and
    the strain this step keys on is exactly what the radar joint resolves. 2003 HM is the
    case: its radar-less A2 sits 6x low, the signal obs misfit, and an optical srp3 absorbs
    the leftovers into a z=12 'A3' -- stealing a validated radar-anchored Yarkovsky. A radar
    object whose TRUE family needs A3 has no known example (JPL: 2 comets); revisit if the
    radar joint itself ever shows this strain."""
    if key != "srp2":
        return True
    N = out.get("N") or 0
    if not N:
        return False
    try:
        meta = (out.get("grav")._ctx or {}).get("meta") or {}
        if meta.get("radar") and os.environ.get("PYROORB_RADAR", "1") != "0":
            return False
    except Exception:
        pass
    best = min((d["chi2"] for d in out["models"].values() if d.get("fit") is not None),
               default=float("inf"))
    best = min(best, out.get("chi2_0", float("inf")))
    return bool(best / N > NONGRAV_SRP3_STRAIN ** 2)


def _asteroid_phys_ok(f, axes):
    """Physical-plausibility test for the asteroid Yarkovsky/SRP family: a loose flat area/mass
    ceiling on the radial A1. This is the SELECTION guard; it is deliberately generous so discovery
    still PICKS an SRP model for high-area/mass outliers (rocket bodies, dark comets).

    The transverse A2 is deliberately NOT gated here. A size-aware Yarkovsky ceiling
    is a statement about an ASSUMED diameter -- H via an assumed albedo, uncertain by a factor of a
    few -- and using it to veto a detection lets a soft size prior overrule the astrometry. 2008 NA
    is the case: A2 = -4.4e-12 against a ceiling of ~1.2e-12 at H=23.8, yet the detection is
    z=5.6 with a 22-40% chi2 gain and it takes the fit from 94/131 @ 0.951" to 117/131 @ 0.829".
    A body ~25 m rather than the assumed ~40 m clears the ceiling outright, and a genuinely
    non-thermal (outgassing / dark-comet) A2 is exactly the kind of object worth SEEING rather
    than silently declining. So it stays REPORTED and never blocks: ng.nongrav_physical_warnings
    raises "A2 exceeds the size-aware Yarkovsky ceiling for this body" on the delivered orbit --
    the same advisory treatment the sharper H-aware implied-density check on A1 already gets --
    and the user judges it. The statistical gates remain the sole arbiters of whether a term fires
    -- those are the BIC bar (NONGRAV_BIC_MARGIN) and the nested F-test, not the
    chi2-gain / NONGRAV_MIN_Z floors this sentence used to name.

    The SIGN of A1 is a different matter from its size. A negative A1 is
    a radial acceleration TOWARD the Sun, and neither mechanism the model stands for can produce
    one: solar radiation pressure pushes outward, and so does out-gassing (the Marsden A1 is the
    sunward-side sublimation recoil). A 6-sigma coefficient with an impossible sign is therefore
    not a weak detection to be reported with a caution -- it is the model absorbing something else
    (a station's catalogue bias, a timing offset, a pre-Gaia apparition), and selecting it ships a
    systematic as physics. 2008 NA is the case: once the srp2 trial CONVERGED (stall rescue) it beat
    yarkovsky by 120 in BIC with A1 = -8.4e-11 at z=6, and the A2-ceiling tie-break picked it. So a
    negative A1 fails selection; the size test stays as loose as ever, and the delivered orbit's
    advisory (nongrav_physical_warnings) still names a negative A1 when an explicit --nongrav /
    UI fit asks for one.

    The default for _ng_evaluate (the comet path supplies _comet_phys_ok instead)."""
    import pyroorb.nongrav as ng
    if "A1" not in axes:
        return True
    a1 = f.nongrav_params.get("A1", 0.0)
    return a1 > 0.0 and ng.a1_is_plausible(a1)


def _ng_choice(models, out=None):
    """Pure model-selection rule (no fitting). `out` (the _ng_evaluate result, unused here) is
    accepted only so this shares the generic choice_fn(models, out) signature with the comet
    model sequence. `models` maps each trial key to a dict with keys:
    chi2 (weighted chi2 of the fit), chi2_gain (fractional chi2 reduction vs gravity, 0..1),
    ftest_ok (the nested F-test vs gravity is significant at NONGRAV_FTEST_ALPHA), phys_ok (all
    params physically plausible). Returns the justified model key or None. A model is justified iff its BIC improves on
    gravity by NONGRAV_BIC_MARGIN AND ftest_ok AND phys_ok. dBIC = (chi2_grav -
    chi2_model) - dp*ln(N); this ONE principled bar subsumes the old hand-set floors
    (NONGRAV_MIN_Z, and NONGRAV_CHI2_GAIN as a justification floor -- both kept as the fallback for
    a caller that passes no gravity baseline; note NONGRAV_CHI2_GAIN is NOT retired, it still
    decides the srp2 upgrade via `_beats` on every call)
    and removes their N-dependence. Scored against JPL's published Yarkovskys on the gate population
    it lifts recall 67% -> 86% (recovers 2003 HM and Bennu, refused by the floors) at ~+1 false positive.

    PHYSICAL PRIOR (this regime is q<1.4, multi-opposition asteroids): the gate-triggering signal
    is a SECULAR residual trend that accumulates over revolutions -- which is the signature of the
    transverse Yarkovsky term A2. A RADIAL SRP (A1) produces no secular semi-major-axis drift, so
    it cannot create that signal; a fitted A1 is partly degenerate with A2 and mostly just absorbs
    systematics (the SRP<->Yarkovsky mis-attribution). So whenever the A2-only model is justified
    it WINS -- UNLESS the radial A1 is *also* independently, strongly detected (a genuine
    SRP+Yarkovsky object), in which case the A1+A2 model is reported so BOTH terms are recovered.
    A1 / A1+A2 are otherwise considered only as a fallback when Yarkovsky is not justified."""
    import pyroorb.nongrav as ng_mod
    def _sig(k):                                  # sqrt(F): the chi2-drop significance (robust)
        F = models[k].get("F") or 0.0
        return math.sqrt(F) if F > 0 else 0.0
    # BIC justification: dBIC over gravity >= NONGRAV_BIC_MARGIN, and physically plausible.
    # ftest_ok kept as a mild sanity floor (a model BIC-justified but F-insignificant would be odd).
    _chi2_0 = (out or {}).get("chi2_0"); _N = (out or {}).get("N")
    def _dbic_over_gravity(k):
        if _chi2_0 is None or not _N or _N <= 0:
            return None
        dp = len(ng_mod.model(k)["params"])          # coefficients this model adds over gravity
        return _dbic(_chi2_0, models[k]["chi2"], dp, _N)
    def _justified(k):
        d = _dbic_over_gravity(k)
        if d is None:                                # no baseline available -> fall back to the floors
            return (models[k]["chi2_gain"] >= NONGRAV_CHI2_GAIN and models[k]["ftest_ok"]
                    and models[k]["phys_ok"] and _sig(k) >= NONGRAV_MIN_Z)
        return d >= NONGRAV_BIC_MARGIN and models[k]["ftest_ok"] and models[k]["phys_ok"]
    just = {k: _justified(k) for k in models}
    def _beats(a, b): return (b - a) / b >= NONGRAV_CHI2_GAIN if b and b != float("inf") else False
    # UPGRADE a justified pick to the full 3-axis model, same shape as the srp2 upgrade below
    # PLUS the raw-visibility discipline: srp3 must be BIC-justified (dp=3), decisively beat
    # the pick on chi2, carry an independently-determined NEW axis (marginal z over
    # NONGRAV_SRP2_A1_Z -- not just a joint-chi2 drop a degenerate axis could buy), AND improve
    # the PICK's raw arcsec rms by NONGRAV_MIN_RMS_GAIN on this same mask. The
    # fourth clause is the same floor _ng_mask_recheck applies to every detection, aimed at the upgrade
    # comparison: weighted chi2 and a formal marginal z are exactly the currencies a
    # tracklet-correlated systematic can buy, raw visibility is not. The step is rarely even
    # fitted (_srp3_continue); when it is, this is 2009 HC's shape -- an out-of-plane A3 at
    # z=35 no 2-axis model can express, srp3 47% visibly better than the sigma-strained
    # A2-only pick -- while 2007 EG's strained arc offered an srp3 that cleared the first
    # three clauses yet barely moved the raw rms, and upgrading it stole a solid Yarkovsky.
    def _srp3_upgrade(cur):
        m3 = models.get("srp3")
        if not (m3 and just.get("srp3") and _beats(m3["chi2"], models[cur]["chi2"])):
            return None
        new_axes = set(ng_mod.model("srp3")["params"]) - set(ng_mod.model(cur)["params"])
        z3 = m3.get("z") or {}
        if max((z3.get(a, 0.0) for a in new_axes), default=0.0) < NONGRAV_SRP2_A1_Z:
            return None
        rms_cur, rms_3 = models[cur].get("rms"), m3.get("rms")
        if not rms_cur or rms_3 is None or (rms_cur - rms_3) / rms_cur < NONGRAV_MIN_RMS_GAIN:
            return None
        return "srp3"
    if just["yarkovsky"]:
        # UPGRADE to A1+A2 only when the radial term is independently, strongly significant: its
        # MARGINAL z in the joint fit clears NONGRAV_SRP2_A1_Z *and* adding it decisively beats
        # A2-only on chi2. The marginal-z gate (not chi2 alone) is what separates a real SRP+Yark
        # object (2009 BD: A1 z~11, both terms recovered near JPL) from a DEGENERATE A1 that lowers
        # the joint chi2 without being independently determined (2008 DB A1 z~4.3, 2011 CL50 ~2.4).
        a1z = (models.get("srp2", {}).get("z") or {}).get("A1", 0.0)
        # The size-aware A2 ceiling no longer VETOES a Yarkovsky detection (by design: see
        # _asteroid_phys_ok -- a soft size prior must not overrule the astrometry). It still gets to
        # break a CHOICE BETWEEN TWO JUSTIFIED MODELS, which is a different question: when A2-only
        # wants a coefficient beyond what a body of this size can plausibly produce thermally, and
        # a richer model explains the SAME data decisively better, the richer model is the better
        # description whether or not its A1 is independently pinned. 2026 NA1 is the case -- A2-only
        # asks for +8.3e-12 at rms 0.886" while A1+A2 reaches 0.530" (chi2 325 -> 81), and its A1's
        # marginal z of 5.2 sits just under NONGRAV_SRP2_A1_Z, so without this the A2-only model
        # wins on the physical prior alone and we ship the worse fit. Where the A2 IS within the
        # ceiling the rule is untouched, and where no richer model is decisively better (2008 NA:
        # srp2 stalls at +0%) Yarkovsky still wins with its ceiling breach merely REPORTED.
        _a2 = abs(float((models["yarkovsky"].get("params") or {}).get("A2", 0.0)))
        _H = None
        try:
            _H = (out or {}).get("grav").absolute_magnitude()[0]
        except Exception:
            pass
        _a2_over_ceiling = bool(_a2) and not ng_mod.a2_is_plausible(_a2, H=_H)
        pick = "yarkovsky"
        if (just.get("srp2") and _beats(models["srp2"]["chi2"], models["yarkovsky"]["chi2"])
                and (a1z >= NONGRAV_SRP2_A1_Z or _a2_over_ceiling)):
            pick = "srp2"
        return _srp3_upgrade(pick) or pick
    # Yarkovsky not justified -> fall back to the radial-SRP models (rare in this gate). Take the
    # two-parameter A1+A2 only if it decisively beats A1-only, else A1 if it stands on its own.
    # srp3 stays an UPGRADE of a justified pick only -- a 3-axis model justified while no
    # sub-model stands is not a shape we deliver unseen (no case yet; revisit if one appears).
    if just["srp2"] and _beats(models["srp2"]["chi2"], models["srp"]["chi2"]):
        return _srp3_upgrade("srp2") or "srp2"
    if just["srp"]:
        return _srp3_upgrade("srp") or "srp"
    return None


def _lift_nongrav_state(sub, model):
    """The (6-state + scaled A's) parameter vector `model`'s LM should START from, taken from a
    converged fit `sub` of the same or a NESTED model: the 6-state as it stands, each of `model`'s
    free coefficients at `sub`'s physical value (0 for an axis `sub` did not carry), divided by
    `model`'s own scale. refine_nongrav's `seed_state` takes exactly this shape (the web UI passes a
    same-model `_state` the same way). Physical units in the middle so models with different
    scales (yarkovsky 1e-14, srp 1e-11) compose."""
    import pyroorb.nongrav as ng
    spec = ng.model(model)
    free = ng.free_indices(model)
    p = np.zeros(6 + len(free))
    p[:6] = np.asarray(sub._state, float)[:6]
    for j, ax in enumerate(free):
        p[6 + j] = float((sub.nongrav_params or {}).get(ng.A_AXES[ax], 0.0)) / spec["scale"]
    return p


def _ng_eval_as(ctx, ev, mask, src, sub_key):
    """Weighted chi2 of `src`'s solution EVALUATED -- one residual sweep, no fitting --
    under the NESTED model `sub_key`: the 6-state as it stands, `sub_key`'s axes at src's
    fitted values, the extra axes zeroed. The sub-model side of the baseline floor:
    what the smaller model could achieve at the richer trial's own orbit. Uses ev.sig_ra/
    sig_dec so the number is in exactly the currency the trial chi2s (rms_norm^2 * N) are."""
    import pyroorb.nongrav as ng
    idx = np.flatnonzero(np.asarray(mask, bool))
    spec = ng.model(sub_key)
    A = [float((src.nongrav_params or {}).get(ax, 0.0)) if ax in spec["params"] else 0.0
         for ax in ("A1", "A2", "A3")]
    tt = ev.times_tdb[idx]
    Rs = [ev.Robs[i] for i in idx]
    resid, _jac, _ = _make_ng_residual(ctx["backend"], ctx["epoch_tdb"], tt, Rs,
                                       ev.ra[idx], ev.dec[idx], ev.cosd[idx],
                                       ev.sig_ra[idx], ev.sig_dec[idx],
                                       spec["gr"], np.empty(0),
                                       a_of_p=lambda p: A, free_cols=[],
                                       col_scale=np.empty(0), m=len(idx))
    r = resid(np.asarray(src._state, float)[:6])
    return float(np.sum(r ** 2))


# The delivered verification's revert is a keep-the-better net, not a second verdict: it only
# refuses a CATASTROPHIC rebuild (the wrong-basin rms_n-6.0 class), not the ~1% of noise between
# two orbits that are the same orbit. See _verification_outcome.
VERIFICATION_REVERT_MAX_FACTOR = 1.05


def _verification_outcome(why, rms_n_gravity, rms_n_model):
    """What a finished delivered-verification means, as a pure decision: `why` is None for a
    clean pass, else the failing chapter's sentence; the two numbers are the normalized rms of
    the REBUILT pinned-A=0 gravity orbit and of the delivered model orbit. Returns
    (outcome, note) with outcome one of:

      "pass"        -- every chapter held; the model stands, nothing to say.
      "revert"      -- a chapter failed AND the demonstrated gravity orbit is no worse than
                       VERIFICATION_REVERT_MAX_FACTOR x the model's; deliver gravity.
      "unverified"  -- a chapter failed but the gravity orbit is materially WORSE, i.e. the
                       pinned refit landed in the wrong basin. Reverting would hand back a
                       catastrophic orbit, so the model still ships -- and must ship SAYING
                       it is unverified. The first version had no name for this state: it
                       wrote the failing verdict, fell out of the gate with `applied` still
                       True, and left the record reading "necessity: dBIC 3.2 < 15" beside
                       "applied: true" with nothing anywhere naming the refusal.

    A non-finite rms on either side is "unverified" (a rebuild that produced no number is not
    a demonstration), which is what the bare `<=` comparison did with a NaN and is kept
    deliberately: the model is delivered under caution, never replaced by nothing."""
    if why is None:
        return "pass", None
    g, m = float(rms_n_gravity), float(rms_n_model)
    if np.isfinite(g) and np.isfinite(m) and g <= m * VERIFICATION_REVERT_MAX_FACTOR:
        return "revert", f"delivered verification failed -- {why}; reverted to gravity"
    return "unverified", (f"the verification failed ({why}) but the gravity orbit it would "
                          f"deliver instead is worse ({g:.2f} vs {m:.2f} sigma) -- the model "
                          f"was kept and is UNVERIFIED; treat the coefficients as provisional")


def _ng_warn_append(existing, note):
    """Add `note` to a FittedOrbit's advisory `nongrav_warning` without losing what is
    already there (the physical-plausibility text from refine_nongrav). Both are advisory
    sentences shown in the same banner, so they concatenate; never a veto either way."""
    note = str(note).strip()
    if not note:
        return existing
    sentence = note[0].upper() + note[1:].rstrip(".") + "."
    if not existing:
        return sentence
    if sentence in existing:
        return existing                            # idempotent: the tail can run twice
    return existing.rstrip() + " " + sentence


def _ng_warn_on(f, note):
    """Append `note` to `f`'s advisory `nongrav_warning`, defensively. Called from the
    delivered verification's two failure exits, one of which is an exception handler --
    which must not be able to raise on its own account, or a gate whose whole point is to
    fail loud dies silent instead. So the attribute is read with a default and the write
    is guarded: anything that is not a real FittedOrbit (a test double, a future orbit
    type) simply doesn't get the banner."""
    try:
        f.nongrav_warning = _ng_warn_append(getattr(f, "nongrav_warning", None), note)
    except Exception:
        pass


def _delivered_verification(f, radar_rows, verbose=False):
    """The UNIFIED post-delivery verification: there are several separate ways to see that an
    orbit does not need its nongrav, so one gate consults them all. It runs at the only place
    the state is actually converged (after _final_reclaim), asking the three independent
    questions every known phantom class failed in SOME instrument while a patchwork of
    separate checks consulted another:

    1. NECESSITY -- does the model beat the gravity it demonstrated? A pinned-A=0 refit
       SEEDED FROM THE DELIVERED 6-STATE (the model's own basin -- the baseline floor's lesson: fixed-state
       evaluation reads co-adaptation, and a cold gravity LM may sit in a worse basin) must
       lose by NONGRAV_BIC_MARGIN, and by the raw-visibility floor
       (NONGRAV_MIN_RMS_GAIN) in plain arcsec.
    2. SIGNIFICANCE -- is the delivered coefficient determined? This is the SAME likelihood
       ratio restated: z_joint = sqrt(chi2_pinned - chi2_model) is the honest (profile-form)
       joint significance of the delivered axes, and the necessity margin IS the z bar
       (dBIC >= 15 with dp=1, N~100 is z_joint >= ~4.4). The LINEAR sigma is recorded next
       to it and flagged when it disagrees -- on a curved A-vs-orbit valley it lies in
       either direction (2012 VP6 read z=0.17 linear on a dBIC-31 claim; both were telling
       the truth about different questions) -- but never decides.
    3. ROBUSTNESS -- does the sign survive resampling? The delete-1-tracklet jackknife
       (otherwise computed lazily, never consulted at delivery) must not have
       every assessed axis spanning zero. Correlated per-tracklet systematics are invisible
       to ANY likelihood quantity -- this is the one leg the first two cannot cover. Paid
       only in the marginal band (necessity dBIC < NONGRAV_JK_SKIP_FACTOR x margin):
       16 capped refits cannot plausibly overturn an overwhelming margin, and the
       concentration-carried route to a fake overwhelming margin is already closed by _ng_mask_recheck.
       An UNASSESSABLE jackknife (too few tracklets) is recorded, never a veto.

    Radar adjudicates, as everywhere in this family: rows pending or applied -> the whole
    verification records "radar adjudicates" and stands down (the joint solve re-derives A
    against radar; every optical-only comparison here is the wrong currency for that orbit).
    A perturber-gated delivery runs the pinned refit under the perturbed backend. On any
    failed chapter the demonstrated gravity orbit is delivered instead, built under a
    DE-CONTAMINATED context (ev.nongrav cleared + the re-anchor meta strip -- the delivered
    ctx/ev carry the active model and would dress the gravity orbit back up as it).
    PYROORB_NG_MASK_RECHECK=0 disables the family. The record lands on the discovery record
    as `verification`, one chapter per question, for the phase-2 scorer and the human.

    TWO ways out end with the model DELIVERED UNVERIFIED, and both say so out loud -- on
    the record AND in `nongrav_warning`, the one advisory channel the CLI and the web
    panel actually render (the discovery record is a diagnostic nothing displays):
      * an EXCEPTION anywhere in the gate (`verification.error`; fail loud, never open --
        a silent swallow would let any future bug disable the whole nongrav-verification family with
        no trace), and
      * a REFUSED REVERT: the chapters failed, but the rebuilt gravity orbit is more than
        1.05x worse in normalized rms, i.e. the pinned refit landed in the wrong basin
        that net exists to catch. Delivering a catastrophic gravity orbit helps nobody,
        so the model stands -- but `verification.revert_refused` records both rms values
        and the verdict becomes "unverified", never a bare failure sitting next to
        `applied: true`."""
    if not getattr(f, "nongrav_model", None):
        return f
    ctx = getattr(f, "_ctx", None)
    if ctx is None or os.environ.get("PYROORB_NG_MASK_RECHECK", "1") == "0":
        return f
    disc_all = getattr(f, "nongrav_discovery", None) or {}
    disc = disc_all.get("asteroid") if "asteroid" in disc_all else disc_all.get("comet")
    radar_pending = bool(radar_rows) and os.environ.get("PYROORB_RADAR", "1") != "0"
    if radar_pending or (getattr(f, "radar", None) or {}).get("applied"):
        if disc is not None:
            disc["verification"] = {"skipped": "radar adjudicates"}
        return f
    ev = ctx.get("ev")                       # hoisted so the fail-loud handler below can
    saved_ng = getattr(ev, "nongrav", None)  # always restore the model on ev (see except)
    try:
        import math
        import pyroorb.nongrav as ng
        from scipy.optimize import least_squares
        mask = np.asarray(f._mask, bool)
        idx = np.flatnonzero(mask)
        backend = _AsteroidForce(ctx["backend"]) if ctx.get("asteroids") else ctx["backend"]
        spec = ng.model(f.nongrav_model)
        resid, jac, _ = _make_ng_residual(backend, ctx["epoch_tdb"], ev.times_tdb[idx],
                                          [ev.Robs[i] for i in idx],
                                          ev.ra[idx], ev.dec[idx], ev.cosd[idx],
                                          ev.fit_sig_ra[idx], ev.fit_sig_dec[idx],
                                          spec["gr"], np.empty(0),
                                          a_of_p=lambda p: [0.0, 0.0, 0.0],
                                          free_cols=[], col_scale=np.empty(0), m=len(idx))
        result = least_squares(resid, np.asarray(f._state, float)[:6], jac=jac,
                            method="lm", xtol=1e-10, ftol=1e-10, max_nfev=200)
        chi2_grav = float(np.sum(result.fun ** 2))
        chi2_model = float(np.sum(np.asarray(f._res.fun) ** 2))
        dp = len(spec["params"])
        N = 2 * int(mask.sum())
        dbic = _dbic(chi2_grav, chi2_model, dp, N)
        # the pinned orbit's RAW rms, via a clean evaluation (layout-safe: never unpack fun)
        ev.nongrav = None
        try:
            rms_grav = float(ev.rms_arcsec(np.asarray(result.x, float), mask))
        finally:
            ev.nongrav = saved_ng
        rms_model = float(f.rms_residual())
        gain = (rms_grav - rms_model) / rms_grav if rms_grav > 0 else 0.0
        ver = {"margin": NONGRAV_BIC_MARGIN,
               "necessity": {"chi2_grav_refit": round(chi2_grav, 3),
                             "chi2_model": round(chi2_model, 3), "dp": dp,
                             "dbic": round(dbic, 3), "rms_grav": round(rms_grav, 4),
                             "rms_model": round(rms_model, 4), "rms_gain": round(gain, 4)}}
        why = None
        if dbic < NONGRAV_BIC_MARGIN:
            why = (f"necessity: dBIC {dbic:.1f} < {NONGRAV_BIC_MARGIN:.0f} against the "
                   "gravity orbit the model itself demonstrated")
        elif gain < NONGRAV_MIN_RMS_GAIN:
            # comet models: the floor yields to an overwhelming margin, as in _ng_mask_recheck
            if (str(f.nongrav_model).startswith("comet")
                    and dbic >= NONGRAV_JK_SKIP_FACTOR * NONGRAV_BIC_MARGIN):
                ver["necessity"]["rms_floor_yielded"] = True
            else:
                why = (f"visibility: raw rms gain {100 * gain:.1f}% < "
                       f"{100 * NONGRAV_MIN_RMS_GAIN:.0f}% over the demonstrated gravity")
        z_joint = math.sqrt(max(chi2_grav - chi2_model, 0.0))
        z_lin = {ax: (abs(v) / f.nongrav_sigmas[ax] if f.nongrav_sigmas.get(ax) else 0.0)
                 for ax, v in (f.nongrav_params or {}).items()}
        ver["significance"] = {"z_joint": round(z_joint, 2),
                               "z_linear": {a: round(z, 2) for a, z in z_lin.items()}}
        if z_lin and max(z_lin.values()) < 1.0 and dbic >= NONGRAV_BIC_MARGIN:
            ver["significance"]["linear_sigma_disagrees"] = True  # curved valley; recorded
        if why is None:
            if dbic >= NONGRAV_JK_SKIP_FACTOR * NONGRAV_BIC_MARGIN:
                ver["robustness"] = {"skipped": f"overwhelming (dBIC {dbic:.0f})"}
            else:
                jd = f.nongrav_jackknife_detail()
                assessed = {a: d for a, d in jd.items() if d.get("assessed")}
                ver["robustness"] = {a: {"z": round(float(d.get("z") or 0.0), 2),
                                         "spans_zero": bool(d.get("spans_zero"))}
                                     for a, d in assessed.items()}
                if not assessed:
                    ver["robustness"] = {"not_assessed": True}
                elif all(d.get("spans_zero") for d in assessed.values()):
                    why = ("robustness: every assessed coefficient's delete-1-tracklet "
                           "cloud spans zero -- the sign does not survive resampling")
        if disc is not None:
            ver["verdict"] = "pass" if why is None else why
            disc["verification"] = ver
        if why is None:
            return f
        # revert: deliver the demonstrated gravity orbit, decontaminated (see docstring)
        ctx_g = {k: v for k, v in ctx.items() if not k.startswith("nongrav")}
        meta_g = {**(ctx.get("meta") or {})}
        for key in ("nongrav_model", "nongrav_params", "nongrav_sigmas", "nongrav_warning",
                    "nongrav_asteroids"):
            meta_g.pop(key, None)                  # the re-anchor path's strip recipe
        ctx_g["meta"] = meta_g
        ev.nongrav = None
        try:
            orb = _build_orbit(ctx_g, np.asarray(result.x, float), mask, result, f.history)
        except Exception:
            ev.nongrav = saved_ng
            raise
        outcome, reason = _verification_outcome(why, orb.rms_normalized(),
                                                f.rms_normalized())
        if outcome == "revert":
            if disc is not None:
                disc["applied"] = False
                disc["model"] = None
                disc["reason"] = reason
            orb.nongrav_discovery = disc_all
            if verbose:
                print("  discover: " + reason)
            return orb
        # THE REVERT WAS REFUSED. The chapters failed, but the gravity orbit this gate
        # would deliver instead is materially WORSE than the model -- the wrong-basin
        # class the 1.05 net exists to catch -- so the model still ships. That is the
        # right call (delivering a catastrophic gravity orbit helps nobody), but it must
        # not ship looking verified: the first version wrote the failing verdict above
        # and then fell out of the gate leaving `applied` True, so the record said
        # "necessity: dBIC 3.2 < 15" and "applied: true" side by side with nothing
        # naming the refusal. Say both things, in the record and to the user -- this is
        # the same "delivered unverified" state the except-branch below is loud about,
        # reached through a different door.
        ev.nongrav = saved_ng                      # keeping the model; re-arm its ev
        if disc is not None:
            ver["revert_refused"] = {"rms_n_gravity": round(float(orb.rms_normalized()), 4),
                                     "rms_n_model": round(float(f.rms_normalized()), 4)}
            ver["verdict"] = "unverified: " + reason
        _ng_warn_on(f, reason)
        if verbose:
            print("  discover: " + reason)
    except Exception as ex:
        # FAIL LOUD, never open. A bare swallow here would deliver the model UNVERIFIED
        # with no trace -- the exact failure mode this gate exists to prevent, one door
        # down: any future bug in the gate silently disables the whole nongrav-verification family.
        # The model is still delivered (an erroring instrument is not evidence against
        # the detection), but the record says the verification did not run, so the
        # phase-2 scorer and a human reviewer can see the hole instead of a clean pass.
        if ev is not None:
            ev.nongrav = saved_ng                  # whatever probe cleared it, re-arm it
        msg = f"{type(ex).__name__}: {ex}"
        if disc is not None:
            rec = disc.get("verification")
            if not isinstance(rec, dict):
                rec = disc["verification"] = {}
            rec["error"] = msg                     # keeps any chapters already recorded
            rec.setdefault("verdict",
                           "error: verification did not complete; model delivered unverified")
        # ...and SAY so on the orbit. `nongrav_discovery` is a diagnostic record no
        # surface renders; the advisory warning is the one channel the CLI and the web
        # panel both show, so an unverified delivery reaches the human who has to judge
        # the coefficients instead of only the phase-2 scorer.
        _ng_warn_on(f, f"the delivered-model verification did not complete ({msg}) -- the "
                       f"model was kept and is UNVERIFIED; treat the coefficients as "
                       f"provisional")
        if verbose:
            print(f"  discover: delivered verification ERRORED ({msg}); "
                  "model delivered unverified")
    return f


def _ng_delivery_sane(final, fit, sel2, info, verbose=False):
    """The last word before a discovered model is delivered: it must be an ORBIT. A model
    orbit past CATASTROPHIC_RMS_N on its own included observations, replacing a gravity
    orbit that was not, is refused with a reason -- however large its dBIC. The selection
    compares the model against the gravity BASELINE on the re-included full arc, and when
    that baseline is itself garbage (21P: the published seed cannot hold a 124-year arc
    gravity-only, 2622" on 7432 obs) every step "gains +97%" against it and comet2 is
    delivered at 491" with A1 = 7e-8 -- a hundred times any comet's -- while a clean
    1246-obs apparition orbit sat one stage upstream. The re-include-everything premise
    (the gravity exclusions were the missing force) is what fails there, and this is the
    place it can be caught without touching the selection rule. A model orbit that is
    catastrophic when the incoming fit ALSO was is left to the existing rules (the
    verification chapters, the reclaim), since refusing it would deliver nothing better."""
    try:
        rn_final = float(final.rms_normalized())
        rn_in = float(fit.rms_normalized())
    except Exception:
        return True
    if rn_final > CATASTROPHIC_RMS_N and rn_in <= CATASTROPHIC_RMS_N:
        info["reason"] = (f"delivered model orbit is catastrophic ({rn_final:.1f} sigma on "
                          f"{final.n_observations()} obs) where the gravity orbit was not "
                          f"({rn_in:.1f} sigma on {fit.n_observations()}); the model cannot span "
                          "the re-included arc -- reverted to gravity")
        if verbose:
            print("  discover: " + info["reason"])
        return False
    return True


def _keep_better_gravity(ctx, fit, sel, info, verbose=False):
    """The baseline floor's keep-the-better corollary: a DECLINED discovery whose trials demonstrated
    a better-converged gravity orbit (the baseline floor's `grav_floor_state`) hands that
    orbit forward instead of discarding it -- the trial found gravity evidence, not force
    evidence, and the fit should not be poorer for having looked. One seeded polish LM
    (starting AT the demonstrated minimum), guarded keep-the-better; on any failure the
    original fit stands. 2012 VP6: the pipeline's own gravity sat at 0.323\" while a trial's
    6-state held 0.249\"."""
    st = (sel or {}).get("grav_floor_state")
    if st is None:
        return fit
    try:
        ev = ctx["ev"]
        mask = np.asarray(fit._mask, bool)
        st2, result = ev.fit(np.asarray(st, float)[:6].copy(), mask, thin=False, polish=True)
        orb = _build_orbit(ctx, st2, mask, result, fit.history)
        if orb.rms_normalized() < fit.rms_normalized():
            info["kept_better_gravity"] = {"rms_n": round(float(orb.rms_normalized()), 4),
                                           "was": round(float(fit.rms_normalized()), 4)}
            if verbose:
                print(f"  discover: declined, but kept the better gravity orbit a trial "
                      f"demonstrated (rms_n {fit.rms_normalized():.3f} -> "
                      f"{orb.rms_normalized():.3f})")
            return orb
    except Exception as ex:
        # Best-effort by design (the original fit legitimately stands), but a bare swallow
        # would also hide a real bug in the seeded polish forever -- record it on the
        # discovery info, where kept_better_gravity would have gone. No reader indexes
        # into this dict, so an error shape is safe.
        info["kept_better_gravity"] = {"error": f"{type(ex).__name__}: {ex}"}
        if verbose:
            print(f"  discover: keep-better-gravity errored ({type(ex).__name__}: {ex}); "
                  "original fit stands")
    return fit


def _ng_evaluate(ctx, seed_state, history, mask, max_nfev, verbose, warm_fits=None,
                 trials=_NG_TRIALS, phys_fn=None, choice_fn=None, continue_fn=None,
                 profile_srp=False, cold=True):
    """Fit gravity + each nongrav `trials` entry on a FIXED `mask`, score them, and return the
    justified choice. Pure comparison -- no rejection, no mask edits.

    `trials` is a sequence of (model_key, axes) pairs; `phys_fn(fit, axes)->bool` is the physical-
    plausibility test for the fitted coefficients (default: the asteroid Yarkovsky/SRP bounds);
    `choice_fn(models, out)->key|None` is the selection rule (default: _ng_choice). The comet path
    passes the comet model sequence, _comet_phys_ok and _ng_comet_choice; the asteroid path uses the
    defaults and is byte-identical to before this generalization.

    Scoring is on the WEIGHTED chi2 (sum of normalized residuals^2 over the included obs), the
    same data for gravity and every trial. Returns a dict: rms0 (gravity raw-arcsec rms, kept
    for display), chi2_0 (gravity chi2), per-model {rms, gain (raw, display), chi2, chi2_gain,
    F, pval, ftest_ok, z, phys_ok, params, sigmas, fit}, and `choice`. This function only
    MEASURES; which model (if any) is justified, and on what clauses, is _ng_choice's rule and
    is documented there -- deliberately not restated here, because the copy that used to live
    in this docstring drifted out of step with it.

    `warm_fits` ({model_key: fit}) is _ng_settle's previous-round solutions, each trial's
    second, warm LM start (see the NESTED WARM START block). `cold=False` SKIPS the from-A=0
    LM for any step that has a warm seed, and with a previous-round seed present skips the
    nested sub-model seed too (the model's own last solution is the better start of the
    two): the settle re-test after a SUBSAMPLED first round (_ng_first_round) already ran the
    cold trial on the subsample, and re-running it on the full arc is the one cost that
    round exists to avoid. A step with no seed still starts cold. Default True is the
    pre-2026-09-05 behaviour, bit for bit."""
    if continue_fn is None and trials is _NG_TRIALS:
        continue_fn = _srp3_continue          # the strain-gated srp3 step (asteroid model sequence)
    import pyroorb.nongrav as ng
    from scipy.stats import f as _f_dist
    phys_fn = phys_fn or _asteroid_phys_ok
    ev = ctx["ev"]
    st0, res0 = ev.fit(np.asarray(seed_state, float), mask, thin=False)
    grav = _build_orbit(ctx, st0, mask, res0, history)
    rms0 = grav.rms_residual()
    N = grav.n_resids()                                # scalar residuals = 2 * included obs
    chi2_0 = grav.rms_normalized() ** 2 * N            # weighted chi2 of the gravity fit
    out = {"rms0": rms0, "chi2_0": chi2_0, "N": N, "models": {}, "grav": grav, "choice": None}

    def _gain(rms):  return (rms0 - rms) / rms0 if rms0 > 0 else 0.0
    # SELECTION significance uses the LINEAR covariance z, while the DISPLAYED +/- is the profile-
    # likelihood interval -- two DIFFERENT questions, deliberately: the linear z (with the F-test /
    # chi2-gain, the primary gate) asks "does adding this parameter improve the fit?", the profile
    # asks "is this parameter's VALUE constrained?". For a degenerate term BOTH can hold at once --
    # the fit improves yet the value is unpinned -- so a coefficient can be SELECTED as a detection
    # and still DISPLAYED as ">±/unconstrained". That is not a contradiction, it is the honest state
    # of a degenerate model (the fit is better with the term, but the LOV direction is flat), and the
    # UI shows the ">" precisely to convey it. The weak link is that the z leans on the same fake-tight
    # linear sigma the profile was introduced to distrust; the F-test/gain is the load-bearing gate.
    def _z(f, nm):
        s = f.nongrav_sigmas.get(nm, 0.0)
        return abs(f.nongrav_params.get(nm, 0.0)) / s if s > 0 else 0.0

    rescue_cache = {}                                  # per-axis profile seeds shared by the trials

    def _failed(axes):
        return {"rms": float("inf"), "gain": 0.0, "chi2": float("inf"),
                "chi2_gain": 0.0, "F": 0.0, "pval": 1.0, "ftest_ok": False,
                "z": {nm: 0.0 for nm in axes}, "phys_ok": False,
                "params": {}, "sigmas": {}, "fit": None, "diag": {}}

    for key, axes in trials:
        # The warm seeds this step can start from (see NESTED WARM START below): its best
        # already-fitted sub-model, and the caller's `warm_fits` entry. Gathered first so a
        # cold=False caller can tell whether the cold LM is skippable for this step.
        own = set(ng.model(key)["params"])
        best_sub = None
        for k2, d2 in out["models"].items():
            if (d2.get("fit") is not None and set(ng.model(k2)["params"]) < own
                    and ng.model(k2)["gr"] is ng.model(key)["gr"]
                    and (best_sub is None or d2["chi2"] < best_sub[1]["chi2"])):
                best_sub = (k2, d2)
        seeds = []
        if best_sub is not None:
            seeds.append((best_sub[0], best_sub[1]["fit"]))
        if warm_fits and warm_fits.get(key) is not None:
            # BOTH seeds are kept even when `cold` is False. A previous-round seed used to
            # REPLACE the nested sub-model one here ("the model's own last solution
            # suffices"), which quietly dropped the guard the NESTED WARM START block below
            # exists to provide: a step's minimum chi2 can never exceed its sub-model's, and
            # the sub-model seed is what ENFORCES that on this mask. The previous-round seed
            # cannot stand in for it -- it was fitted on a DIFFERENT mask (on a subsampled
            # accept, ~600 obs of several thousand), so it says nothing about the nesting
            # relation here, and a step left to it alone can land above its own sub-model by
            # the optimizer artefact the block documents (2008 NA round 2: cold srp2 at
            # chi2 836 against yarkovsky's 476). Cost is one extra warm LM per seeded step.
            seeds.append(("previous round", warm_fits[key]))
        f, diag = None, {}
        if cold or not seeds:
            try:
                _seeded_srp = profile_srp and key == "srp"
                f = refine_nongrav(grav, key, mask=mask, max_nfev=max_nfev,
                                   revert_if_worse=False,
                                   # The seeded srp rescue needs the TIGHT tol: at the loose comparison
                                   # tol the profile-seed slides back to the gravity corner (A1->0) on a
                                   # deep-degeneracy arc, so it would read +0% and never be selected
                                   # (2026 NA1). Only this one rare armed trial pays it; every other
                                   # trial stays loose.
                                   tol=(NONGRAV_INTERACTIVE_TOL if _seeded_srp else NONGRAV_DISCOVER_TOL),
                                   profile_seed=_seeded_srp, rescue_cache=rescue_cache,
                                   profile_sigmas=False)   # comparison trials keep the fast linear-sigma z gating
            except Exception as ex:                   # a trial that won't converge can't justify itself
                if verbose:
                    print(f"  discover[{key}]: fit failed ({type(ex).__name__}: {ex})")
                out["models"][key] = _failed(axes)
                continue
            diag = dict(getattr(f, "nongrav_diag", None) or {})
            # model-unit chi2 (the diag's chi2_cold / chi2 are the LM's CLAMPED cost, not comparable)
            diag["chi2_trial"] = f.rms_normalized() ** 2 * N
        # NESTED WARM START. The trials are nested (srp2 = srp + yarkovsky, comet3 = comet2 + A3),
        # so a model's minimum chi2 can never exceed its sub-model's -- yet a COLD trial of the
        # larger model regularly does (2008 NA, round 2: srp2 836 against yarkovsky 476 and srp
        # 577), because the damped LM started at A=0 descends a curved 2-D valley partially and
        # runs out of evals or stalls somewhere its sub-model never had to visit. That is not the
        # data preferring the smaller model, it is an optimizer artefact, and the selection then
        # compares a converged model with an unconverged one. So the larger model ALSO starts from
        # its best already-fitted sub-model's solution (the extra axis at 0) -- one warm LM, at the
        # tol a valley-floor seed needs to hold -- and keeps whichever is lower. By construction
        # chi2(model) <= chi2(best sub-model), which is what nesting means. `warm_from` records it.
        # A second warm start, on a settle re-test: this SAME model's solution from the previous
        # round, whose mask differs by the handful of observations the rejection round moved. The
        # cold path starts every round in the gravity corner again, and only the seed on the valley
        # floor from the last round is guaranteed not to. `warm_fits` is {model: fit} from _ng_settle.
        for label, src in seeds:
            try:
                f_w = refine_nongrav(grav, key, mask=mask, max_nfev=max_nfev, revert_if_worse=False,
                                     tol=NONGRAV_INTERACTIVE_TOL, profile_seed=False,
                                     profile_sigmas=False, seed_state=_lift_nongrav_state(src, key))
                diag.setdefault("chi2_warm", {})[label] = f_w.rms_normalized() ** 2 * N
                if f is None or f_w.rms_normalized() < f.rms_normalized():
                    diag = {**diag, "warm_from": label}
                    f = f_w
            except Exception as ex:                    # the trial as it stands
                if verbose:
                    print(f"  discover[{key}]: warm start from {label} failed "
                          f"({type(ex).__name__}: {ex})")
        if f is None:                                  # warm-only step whose every seed failed
            out["models"][key] = _failed(axes)
            continue
        diag.setdefault("chi2_trial", f.rms_normalized() ** 2 * N)
        phys_ok = phys_fn(f, axes)
        chi2 = f.rms_normalized() ** 2 * N
        chi2_gain = (chi2_0 - chi2) / chi2_0 if chi2_0 > 0 else 0.0
        dp = len(axes); k = 6 + dp; dof = N - k       # nested F-test: gravity (6) vs gravity+dp A's
        F = ((chi2_0 - chi2) / dp) / (chi2 / dof) if (chi2 > 0 and dof > 0) else 0.0
        pval = float(_f_dist.sf(F, dp, dof)) if F > 0 else 1.0
        out["models"][key] = {"rms": f.rms_residual(), "gain": _gain(f.rms_residual()),
                              "chi2": chi2, "chi2_gain": chi2_gain, "F": F, "pval": pval,
                              "ftest_ok": pval < NONGRAV_FTEST_ALPHA,
                              "z": {nm: _z(f, nm) for nm in axes}, "phys_ok": phys_ok,
                              "params": dict(f.nongrav_params), "sigmas": dict(f.nongrav_sigmas),
                              "fit": f, "diag": diag}
        # Lazy model sequence: a nested model family (the comet path) stops fitting further steps once one
        # is not justified over the previous -- the selection would stop there anyway, so the extra
        # fits would be wasted (the common decline on dynamically-cometary asteroids). The asteroid
        # trials run eagerly (NOT a strict model sequence) except the final srp3 step, which
        # _srp3_continue gates on the best sub-model still being sigma-strained (2009 HC).
        if continue_fn is not None and not continue_fn(out, key):
            break

    m = out["models"]
    # BASELINE FLOOR, generalized (the old exact-zero clause was 2015 VO142's
    # special case of it). Evaluation is free, so gravity's chi2 is floored by every trial's
    # 6-state EVALUATED at A=0 -- the lowest gravity chi2 anyone demonstrated -- with rms0
    # re-derived in the same breath (the raw currency the visibility floor reads) and
    # the floor STATE kept so a declined discovery can hand a better gravity orbit forward
    # (_keep_better_gravity). The same floor runs one step down: each fitted sub-model's
    # chi2 is floored by every richer trial's solution restricted to it (shared axes at the
    # richer values, extras zeroed), so extra axes cannot take credit in the nested
    # comparisons (_beats, the srp2/srp3 upgrade clauses). Scope, measured on 2012 VP6:
    # these TRIAL-level floors fire only when the axes are already cosmetic AT the trial --
    # a basin escape usually hides inside a CO-ADAPTED (6-state, A) pair (the A-vs-orbit
    # degeneracy: VP6's trial states evaluate to chi2 8e3-3e7 at A=0 yet 35-49 with A), and
    # only the tight-tol delivery refit walks the pair far enough to expose it. THAT case is
    # caught by _delivered_model_recheck at delivery, where the state is actually converged.
    # All gains / F / p re-derived once at the end when any floor moved.
    grav_evals = []
    for d in m.values():
        if d["fit"] is None:
            continue
        try:
            st6 = np.asarray(d["fit"]._state, float)[:6]
            if d["params"] and all(v == 0.0 for v in d["params"].values()):
                grav_evals.append((d["chi2"], st6))     # literally a gravity solution
            else:
                grav_evals.append((float(ev.rms_norm(st6, mask) ** 2 * N), st6))
        except Exception:
            pass
    floored = False
    if grav_evals:
        best_chi2, best_st = min(grav_evals, key=lambda t: t[0])
        if best_chi2 < chi2_0:
            out["chi2_0_baseline"], chi2_0 = chi2_0, float(best_chi2)
            out["chi2_0"] = chi2_0
            out["grav_floor_state"] = np.asarray(best_st, float).copy()
            try:
                out["rms0"] = rms0 = float(ev.rms_arcsec(best_st, mask))
            except Exception:
                pass
            floored = True
    for kM, dM in m.items():
        if dM["fit"] is None:
            continue
        pM = set(ng.model(kM)["params"])
        for kS, dS in m.items():
            if dS["fit"] is None or kS == kM:
                continue
            if not (set(ng.model(kS)["params"]) < pM
                    and ng.model(kS)["gr"] is ng.model(kM)["gr"]):
                continue
            try:
                c = _ng_eval_as(ctx, ev, mask, dM["fit"], kS)
            except Exception:
                continue
            if c < dS["chi2"]:
                dS["chi2"] = c
                dS["chi2_floored_by"] = kM
                floored = True
    if floored:
        for key, d in m.items():
            if d["fit"] is None:
                continue
            dp = len(ng.model(key)["params"]); dof = N - 6 - dp
            d["gain"] = _gain(d["rms"])
            d["chi2_gain"] = (chi2_0 - d["chi2"]) / chi2_0 if chi2_0 > 0 else 0.0
            d["F"] = ((chi2_0 - d["chi2"]) / dp) / (d["chi2"] / dof) if (d["chi2"] > 0 and dof > 0) else 0.0
            d["pval"] = float(_f_dist.sf(d["F"], dp, dof)) if d["F"] > 0 else 1.0
            d["ftest_ok"] = d["pval"] < NONGRAV_FTEST_ALPHA
    out["choice"] = (choice_fn or _ng_choice)(m, out)
    if verbose:
        print(f"  discover: gravity rms0={rms0:.3f}\" chi2={chi2_0:.1f} (N={N})")
        for k, _ in trials:
            d = m.get(k)
            if d is None:                              # lazy model sequence skipped this (and later) steps
                continue
            zt = ", ".join(f"{nm} z={d['z'][nm]:.1f}" for nm in d["z"])
            dg = d.get("diag") or {}
            tell = (f"  warm={dg['warm_from']}" if dg.get("warm_from")
                    else f"  rescued={dg['rescued']}" if dg.get("rescued")
                    else "  STALLED (A never moved)" if dg.get("stalled") else "")
            print(f"  discover[{k:9}]: chi2 {d['chi2']:.1f} (gain {d['chi2_gain']*100:+.0f}%, "
                  f"rms {d['rms']:.3f}\") F={d['F']:.0f} p={d['pval']:.1e} {zt}  "
                  f"ftest={d['ftest_ok']} phys={d['phys_ok']}{tell}")
        print(f"  discover: choice={out['choice']}")
    return out


COHERENT_BLOCK_FRAC    = 0.30  # >= this fraction of ONE apparition's obs rejected = a gutted apparition
COHERENT_BLOCK_MINREJ  = 5     # ... and at least this many rejected obs (ignore tiny apparitions)
COHERENT_BLOCK_NIGHTS  = 2     # ... spanning >= this many DISTINCT nights (a multi-tracklet block)
COHERENT_BLOCK_MAX_OPP = 3     # ... on a FEW-apparition arc only (see below)


def _coherent_block_nights(fit):
    """Detect a GUTTED APPARITION: the GRAVITY fit rejected a large, multi-night block of a single
    apparition. Returns the number of distinct REJECTED nights in the worst such apparition (0 if
    none qualifies). This is the structural tell that a force model (SRP / Yarkovsky) may be wanted
    -- a real out-gassing/SRP signal makes a whole apparition inconsistent under gravity, so the
    reject loop guts it across many tracklets, whereas a lone bad tracklet or a mislink loses ONE
    night. Crucially it does NOT depend on the un-seeded nongrav trial escaping the (state,A1)
    degeneracy, so it arms the profile-seeded rescue exactly where discovery's cheap trial goes blind
    (deep-degeneracy arcs, e.g. 2026 NA1's 2007 apparition: 13/31 rejected over 3 nights). It also
    supplies the 'multiple tracklets' evidence for the delivery guard: a physically-implausible
    coefficient is reported when the block spans >=COHERENT_BLOCK_NIGHTS tracklets (candidate dark
    comet / lost rocket body), but reverted to gravity when it rests on a single tracklet (more likely
    bad astrometry). Gated on the REJECTION FRACTION of one apparition (>=COHERENT_BLOCK_FRAC), so a
    big arc's thin scattered rejections do not trip it.

    Restricted to FEW-apparition arcs (<=COHERENT_BLOCK_MAX_OPP): a continuous force cannot leave many
    apparitions pristine and gut exactly one, so on a long, well-observed arc a lone gutted apparition
    is an apparition-specific DATA problem (e.g. 163693 Atira: 14 clean apparitions + one 37%-rejected
    one -- SRP does not explain it), while a well-constrained real force would be caught by the cheap
    un-seeded trial anyway. The profile-seed rescue is only needed -- and only trustworthy -- where the
    (state,A1) degeneracy is deep: the 2-3 apparition short arc (2026 NA1). Returns 0 when unavailable."""
    jds = np.array([o["jd_utc"] for o in fit._obs], float)
    mask = np.asarray(fit._mask, bool)
    if jds.shape != mask.shape or len(jds) < 3:
        return 0
    order = np.argsort(jds)
    js, mk = jds[order], mask[order]
    app = np.zeros(len(js), int)                       # apparition id (find_orb's 238-day opposition gap)
    for i in range(1, len(js)):
        app[i] = app[i - 1] + (1 if js[i] - js[i - 1] > 238.0 else 0)
    if app[-1] + 1 > COHERENT_BLOCK_MAX_OPP:           # long, well-observed arc -> a lone gutted
        return 0                                       #   apparition is bad data, not a force model
    best = 0
    for a in range(app[-1] + 1):
        sel = app == a
        n = int(sel.sum())
        rej = ~mk[sel]
        n_rej = int(rej.sum())
        # NIGHTS: the 12-hour window anchored on each night's FIRST observation
        # (orbit_common.night_index).  round(jd - 0.5) put the boundary at noon
        # UT -- mid-night for F51/F52, T05/T08/T09, 703/G96 -- so one night of
        # rejections at those stations could count as two and clear
        # COHERENT_BLOCK_NIGHTS on its own.  Indexed over the WHOLE apparition,
        # not the rejected subset, so a window anchors on the night's true start.
        _ni = night_index(js[sel])
        rej_nights = len({ni for ni, r in zip(_ni, rej) if r})
        if (n_rej >= COHERENT_BLOCK_MINREJ and n_rej >= COHERENT_BLOCK_FRAC * n
                and rej_nights >= COHERENT_BLOCK_NIGHTS):
            best = max(best, rej_nights)
    return best


NONGRAV_SETTLE_ROUNDS = 3   # reject/re-test rounds allowed for the delivered model and its mask to agree
# SUBSAMPLED FIRST ROUND (2026-09-05). On a large arc the model-comparison trials -- three cold
# LMs plus their warm restarts, each an ASSIST sweep over every included observation -- are
# the whole cost of a seeded fit (10P: 56 of 76 s). The FIRST round only has to find each
# step's approximate solution and say whether anything is justified; so above
# NONGRAV_SUBSAMPLE_MIN_OBS included obs it runs on a stratified subsample (every night kept,
# evenly thinned within the night, then spread over apparitions to ~NONGRAV_SUBSAMPLE_TARGET
# -- the geometry the orbit needs, at a fraction of the sweeps), and the SETTLE round on the
# full arc warm-starts from it with the cold trials skipped (_ng_evaluate cold=False). The
# full-arc re-test stays the arbiter of the model, its mask and its plausibility. RECALL-SAFE
# by construction: a subsample round that declines is discarded and the full round runs as
# before, so no object justified today can be lost -- only the accepted ones get cheaper.
#
# That construction argument covers the DECLINE branch only; the ACCEPT branch was MEASURED
# (2026-09-08). Seventeen nongrav-delivering fixtures were forced down it (MIN=20 with
# NONGRAV_SUBSAMPLE_TARGET scaled per object, thinning 1.8x-8x) and compared against the
# same objects with the subsample off: NO model flip and NO sign flip anywhere, so the
# accept branch does not cost recall on this population. What it did cost, before the
# nested-seed fix in _ng_evaluate above, was a small fit-quality drift -- summed |rms -
# no-subsample| across the 17 fell 7.6x (0.0083" -> 0.0011") once the sub-model seed was
# restored, nearly all of it 2026 NA1's deep-degeneracy arc (0.0070" -> 0.00008",
# reproducible to 6 dp over two runs). Cost of the fix on that population: nil (662 vs 667
# CPU s, both ~9% under the no-subsample 730). CAVEAT: those are FORCED subsamples on
# 36-591-obs objects, not the >=2000-obs production regime -- 21P is the only regression
# fixture that subsamples for real, and it agreed to 8-9 significant figures in all arms.
NONGRAV_SUBSAMPLE_MIN_OBS = int(os.environ.get("PYROORB_NG_SUBSAMPLE_MIN", "2000"))
NONGRAV_SUBSAMPLE_TARGET = 600


def _ng_subsample_mask(ev, mask):
    """A stratified subsample of `mask` for a discovery first round, or None when the
    included set is at or below NONGRAV_SUBSAMPLE_MIN_OBS (the round then runs on `mask`).
    Every calendar night keeps >= 1 obs (evenly spread within the night, like the fitter's
    own _thin), then the evaluator's over-oppositions spread caps the total near
    NONGRAV_SUBSAMPLE_TARGET while keeping each apparition's endpoints. Deterministic --
    the fit must not depend on a random draw."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if len(idx) <= NONGRAV_SUBSAMPLE_MIN_OBS:
        return None
    days = np.floor(np.asarray(ev.times_tdb, float)[idx]).astype(np.int64)
    uniq = np.unique(days)
    per = max(1, NONGRAV_SUBSAMPLE_TARGET // len(uniq))
    keep = []
    for d in uniq:
        g = idx[days == d]
        if len(g) <= per:
            keep.append(g)
        else:
            keep.append(g[np.linspace(0, len(g) - 1, per).round().astype(int)])
    sub_idx = np.sort(np.concatenate(keep))
    sub_idx = ev._spread_over_oppositions(sub_idx, NONGRAV_SUBSAMPLE_TARGET)
    sub = np.zeros_like(np.asarray(mask, bool))
    sub[sub_idx] = True
    return sub


def _ng_first_round(ev, test_mask, evaluate, info, verbose):
    """Discovery round 1 -- `evaluate(mask) -> sel` -- on the subsample when the arc is large
    enough to warrant one (_ng_subsample_mask), falling back to the full `test_mask` when
    the subsample justifies nothing. Returns (sel, mask_used, subsampled). Records
    `info["subsample"]` = {n_sub, n_full, declined} whenever a subsample was tried."""
    sub = _ng_subsample_mask(ev, test_mask)
    if sub is None:
        return evaluate(test_mask), test_mask, False
    n_sub, n_full = int(sub.sum()), int(test_mask.sum())
    if verbose:
        print(f"  discover: round 1 on a {n_sub}-of-{n_full} stratified subsample")
    sel = evaluate(sub)
    if sel["choice"] is None:
        if verbose:
            print("  discover: the subsample justified nothing; re-running round 1 on the full arc")
        info["subsample"] = {"n_sub": n_sub, "n_full": n_full, "declined": True}
        return evaluate(test_mask), test_mask, False
    info["subsample"] = {"n_sub": n_sub, "n_full": n_full, "declined": False}
    return sel, sub, True


def _ng_settle(sel, test_mask, evaluate, max_nfev, info, verbose, reject_orig_n=None):
    """The rejection round and the model selection, iterated to a FIXED POINT: reject with the
    chosen model active, re-test on the new mask, and if the re-test prefers a DIFFERENT model,
    reject again under that one -- until the model the mask was rejected under is the model the
    re-test chooses, or NONGRAV_SETTLE_ROUNDS is spent.

    One round is not enough because the rejection depends on the model: a richer model (srp2)
    fits residuals a leaner one (yarkovsky) rejects, so the mask it leaves behind is the mask
    for ITS explanation. Delivering the leaner model on that mask -- which is what a single
    reject-then-retest does when the re-test switches -- ships a model on a set of observations
    it never got to judge (2008 NA: round 1 srp2, rejection under srp2 keeps 7 obs yarkovsky
    would drop, round 2 yarkovsky -- 124/131 @ 0.971" for what settles at 117/131 @ 0.829").

    `evaluate(seed_state, mask)` runs the path's own _ng_evaluate; the chosen model is re-fit at
    the tight tol SEEDED from its winning trial before rejecting (see discover_nongrav). Returns
    (sel2, reject_mask): the last re-test and its mask, sel2["choice"] None if a re-test declined
    (the caller reverts, as before). Records every round in info["settle"] and the totals in
    info["rejected_round"] / info["redeemed_round"].

    IF THE BUDGET RUNS OUT MID-FLIP, the fixed point was never reached: the last re-test
    prefers a model the returned mask was NOT rejected under, and returning that pair ships
    exactly the mismatch this function exists to prevent -- the caller delivers
    `sel2["choice"]` on `reject_mask` without ever noticing. So the exhausted case pays one
    final rejection under the model it is about to deliver, and records
    `info["settled"] = False`.

    What that trade buys and what it costs, stated plainly: after the final pass the mask
    HAS been judged by the delivered model (the property whose absence cost 2008 NA 0.14"
    above), but the model's justification was established on the PREVIOUS mask, since
    re-testing again is what the budget just ran out of. Both properties together need an
    unbounded loop, which is what NONGRAV_SETTLE_ROUNDS declines to run. Reverting to
    gravity instead was considered and rejected: an oscillation means every round DID
    justify some nongrav model (a `choice` of None breaks the loop immediately), so the
    disagreement is over WHICH model, and answering "no force at all" is the one answer the
    evidence rules out."""
    import pyroorb.arc_extend as ae

    def _reject_under(model_key, source_sel, mask):
        """Fit `model_key` at the tight tol on `mask` and run the rejection round under it.
        Returns the new mask. Factored out because the exhausted path must do exactly what
        every round does -- a second copy here is how the two would drift."""
        chosen = refine_nongrav(source_sel["grav"], model_key, mask=mask, max_nfev=max_nfev,
                                revert_if_worse=False, profile_sigmas=False,
                                seed_state=_lift_nongrav_state(
                                    source_sel["models"][model_key]["fit"], model_key))
        out = mask.copy()
        ae.auto_reject_within_arc(chosen._ctx["ev"],           # nongrav-aware evaluator clone
                                  out,
                                  # _state is the 6-state + k A's; the rejector wants the 6
                                  np.asarray(chosen._state, float)[:6],
                                  orig_n=reject_orig_n)      # None = the whole-record guard
        return out

    model, mask_cur, rounds = sel["choice"], test_mask, []
    # Set at the break, never inferred afterwards: the loop's last statement advances `model`
    # to the re-test's choice, so a post-loop `model == sel2["choice"]` is True whether the
    # fixed point was reached or the budget merely ran out.
    settled = False
    for _ in range(NONGRAV_SETTLE_ROUNDS):
        reject_mask = _reject_under(model, sel, mask_cur)
        # The re-test's GRAVITY baseline is seeded from the last converged GRAVITY orbit, never from
        # the chosen model's 6-state: a nongrav solution's `a` has been traded against its A, and a
        # gravity LM started there can settle in the wrong basin -- 2008 NA's third round put the
        # gravity baseline at 11" rms on a mask a clean gravity fit holds at 1.4", and every trial
        # then "gained +100%" against garbage. The trials, conversely, DO get the last round's
        # solutions as warm starts (see _ng_evaluate).
        grav6 = np.asarray(sel["grav"]._state, float)[:6]
        sel2 = evaluate(grav6, reject_mask,
                        {k: d["fit"] for k, d in sel["models"].items() if d.get("fit") is not None})
        rounds.append({"under": model, "rejected": int((mask_cur & ~reject_mask).sum()),
                       "redeemed": int((~mask_cur & reject_mask).sum()), "choice": sel2["choice"]})
        if sel2["choice"] is None or sel2["choice"] == model:
            settled = True
            break
        if verbose:
            print(f"  discover: re-test prefers {sel2['choice']} over the {model} the mask was "
                  f"rejected under -- settling again")
        model, mask_cur, sel = sel2["choice"], reject_mask, sel2
    if not settled:
        # Budget spent with the re-test still preferring something else. Deliver its choice
        # -- but on a mask that choice has judged (see the docstring).
        if verbose:
            print(f"  discover: {NONGRAV_SETTLE_ROUNDS} settle rounds spent and the re-test "
                  f"still prefers {sel2['choice']} over {model} -- rejecting once more under "
                  f"{sel2['choice']} and delivering unsettled")
        final = _reject_under(sel2["choice"], sel2, reject_mask)
        rounds.append({"under": sel2["choice"], "rejected": int((reject_mask & ~final).sum()),
                       "redeemed": int((~reject_mask & final).sum()), "choice": None,
                       "unsettled_final": True})
        reject_mask = final
    info["settled"] = bool(settled)
    info["settle"] = rounds
    info["rejected_round"] = sum(r["rejected"] for r in rounds)
    info["redeemed_round"] = sum(r["redeemed"] for r in rounds)
    return sel2, reject_mask


def discover_nongrav(fit, max_nfev=NONGRAV_DISCOVER_NFEV, verbose=False):
    """Automatically decide whether a NON-GRAVITATIONAL model is justified for a fit whose
    gravity solution still shows a systematic residual trend, and -- if so -- return the orbit
    refit with that model after a rejection round. Returns (fit_out, info).

    Gate (physical regime only): a bound orbit with q < 1.4 au AND >= 2 oppositions (see
    NONGRAV_Q_MAX_AU for why perihelion, not semi-major axis). Within that regime the
    STATISTICAL SELECTION is the sole arbiter -- a model fires only if it clears the weighted-
    chi2 effect-size bar AND the nested F-test AND the size-aware physical-plausibility ceiling
    (see _ng_evaluate / _ng_choice). We deliberately do NOT pre-screen on a heuristic
    systematic-trend flag: a population check vs JPL showed ident_check.systematic_misfit blocks
    ~13/13 large-A2 Yarkovskys it can't see as a coherent per-night trend (e.g. 2020 GE, A2
    -1.9e-12, recovered at chi2-gain 45%/z9.7), while passing force-model-deficient low-q
    objects the selection then has to reject anyway -- it is a poor filter in both directions.
    Dropping it recovers those detections at no measured false-positive cost (0 FP across the
    very-low-q strong-GR control class, where the chi2 effect-size bar rejects the ~0-2% gains
    a force-model trend produces). The caller is expected to have already done what it can
    gravitationally (including the asteroid-perturber model).

    Procedure (find_orb-style "turn the A's on and compare"): re-include every obs, mask off
    the worst NONGRAV_DROP_WORST by sigma-residual, and hold THAT mask fixed while fitting
    A1 / A2 / A1+A2 against the gravity baseline (see _ng_evaluate for the justification
    rules). If a model is justified, switch it on, run ONE outlier-rejection round with the
    nongrav active, then RE-TEST on the post-rejection mask -- keeping the nongrav only if it
    is still justified there, else reverting to the original gravity fit.

    NB: model selection is a raw-rms-gain + per-parameter significance heuristic; a nested
    F-test / information criterion would be more principled."""
    from pyroorb import ident_check
    import pyroorb.arc_extend as ae

    info = {"applied": None, "model": None, "reason": None}
    a, _ = fit.semimajor_axis()
    q, _ = fit.perihelion_distance()
    n_opp = fit.n_oppositions()
    sysdesc = ident_check.systematic_misfit(fit)     # reported for context, NOT gated on
    info["gate"] = {"a": a, "q": q, "n_opp": n_opp, "systematic": sysdesc}
    ctx = getattr(fit, "_ctx", None)
    backend = ctx["backend"] if ctx else None
    if ctx is None or not hasattr(backend, "predict_radec_jac_ng"):
        info["reason"] = "needs the ASSIST backend (no nongrav force otherwise)"
        return fit, info
    if not (a is not None and a > 0 and q is not None and 0 < q < NONGRAV_Q_MAX_AU
            and n_opp >= NONGRAV_MIN_OPP):
        info["reason"] = f"gate not met (a={a}, q={q}, n_opp={n_opp})"
        return fit, info
    if verbose:
        print(f"discover_nongrav: a={a:.3f} au, q={q:.3f} au, {n_opp} oppositions "
              f"(systematic flag: {'yes' if sysdesc else 'no'}, not gated)")

    ev = ctx["ev"]
    # Undo any phase-bridge link relaxation so the gravity baseline (rms0, worst-obs selection)
    # and every nongrav trial are compared at the obs' true weight -- the coherent far-apparition
    # offset is the signal under test here, not something to keep down-weighted. (refine_nongrav
    # also restores, defensively, for the direct UI path; both are idempotent.)
    ev.restore_link_relaxation()
    n = ev.n
    # Re-include everything, then drop the worst NONGRAV_DROP_WORST by sigma-residual.
    all_mask = np.ones(n, dtype=bool)
    st_all, _ = ev.fit(np.asarray(fit._state, float), all_mask, thin=False)
    worst = np.argsort(ev.resid_sigmas(st_all))[::-1][:NONGRAV_DROP_WORST]
    test_mask = all_mask.copy()
    test_mask[worst] = False
    info["dropped"] = [int(i) for i in worst]

    sel, _mask1, subsampled = _ng_first_round(
        ev, test_mask,
        lambda m: _ng_evaluate(ctx, st_all, fit.history, m, max_nfev, verbose), info, verbose)
    # SRP-shaped signal the un-seeded srp trial MISSED: a radial inverse-square SRP (A1) is nearly
    # degenerate with the semimajor axis, so the A1-only LM seeded from A1=0 sticks at the gravity
    # solution and the cheap srp trial reads +0% (see refine_nongrav's profile_seed). Seeding EVERY
    # discovery radial trial escapes this but is several-fold slower on the whole gated population
    # (the seed + the tight tol it needs cost real time on large arcs) -- so we pay the profile-seed
    # once, only when a NON-CIRCULAR structural tell says a force model is likely: the gravity fit
    # gutted a coherent multi-night block of one apparition (_coherent_block_nights). That signal does
    # NOT depend on the un-seeded trial escaping (unlike a z-threshold tell, which is circular -- it
    # would need the escape it is trying to enable), so it fires exactly where the cheap trial goes
    # blind (2026 NA1's grown 2-apparition arc: srp/srp2 stick at A1=0 while gravity guts the 2007
    # apparition). We fit even an unusual (e.g. ~0.006 g/cm^3) A1 -- rocket bodies / dark comets are
    # such high-area/mass objects -- and attach an advisory warning rather than declining it.
    # `block_nights` doubles as the 'multiple tracklets' evidence for the delivery guard below.
    block_nights = _coherent_block_nights(fit)
    info["block_nights"] = block_nights
    # The TELL alone buys the seed -- NOT "the tell fired and round 1 found nothing".
    #
    # The `sel["choice"] is None` conjunct that used to stand here was a COST guard, written when
    # the hand-set effect-size floors made "found nothing" the usual round-1 outcome. It was never
    # a claim that a justified model makes the radial seed pointless. Since selection became
    # a BIC bar, round 1 justifies something far more often, so that conjunct silently stopped
    # buying the seed on the few-apparition, deep-degeneracy arcs it exists for -- and having AN
    # answer is not having the right FAMILY. 2026 NA1 is the shape of that failure (A2-only 0.886"
    # against A1+A2's 0.530"); it survives today only because its round-1 choice happens to be None.
    # The tell itself is the cost guard and always was: `_coherent_block_nights` demands a gutted
    # coherent multi-night block, >=COHERENT_BLOCK_FRAC of ONE apparition, on a <=COHERENT_BLOCK_
    # MAX_OPP arc -- it explicitly refuses 163693 Atira's 14-clean-plus-one-gutted shape. Stacking a
    # second guard on a sufficient one bought almost no compute and cost reachability.
    profile_srp = block_nights >= COHERENT_BLOCK_NIGHTS
    if profile_srp:
        _t0, _c0 = _stime.perf_counter(), _stime.process_time()
        sel_seeded = _ng_evaluate(ctx, st_all, fit.history, _mask1, max_nfev, verbose,
                                  profile_srp=True)   # the same mask round 1 was judged on
        # What the seeded pass COST and DID, on the record: its price outside the suite is only
        # measurable on the gate population, and a sweep row is where that lives.
        info["profile_srp"] = {"cold": sel["choice"], "seeded": sel_seeded["choice"],
                               "secs": round(_stime.perf_counter() - _t0, 3),
                               "cpu": round(_stime.process_time() - _c0, 3)}
        # KEEP THE BETTER, the same rule the seeded trial itself follows. The seeded pass should
        # only ever ADD a radial option -- it changes the srp trial's seed and tol and nothing
        # else -- but it does re-fit every trial, and a profile-seeded srp skips the rescue block
        # that would otherwise populate `rescue_cache` for A1, so the two passes are not bit-equal.
        # This ran only on the choice-is-None path before, where overwriting could not lose
        # anything; now that it runs with a model already justified, refuse a swap that would
        # UNJUSTIFY it. Declining also withdraws the flag, so the settle rounds below stay
        # consistent with the selection they are re-testing.
        # Keep the COLD comparison either way. A profile-seeded trial takes the `seed_state is
        # None and not _radial_branch` branch out of refine_nongrav's diagnostic
        # block, so its `diag` carries no `stalled` / `zero_axes` / `rescued` -- and this project's
        # rule is that a coefficient of exactly 0.0 is a STALL, never a finding, with the two kept
        # distinguishable downstream. Overwriting `sel` therefore throws away the only record that
        # can tell them apart: 2008 NA's seeded srp still lands A1 = 0.000e+00, and without this
        # the discovery record reports that zero with nothing to say which it is. (The loss is
        # older than the decoupling above -- it applied to every choice-is-None object whose tell
        # fired -- but the decoupling widens it, so fix it here.)
        info["compare_cold"] = {k: {kk: v[kk] for kk in _NG_COMPARE_KEYS}
                                for k, v in sel["models"].items()}
        if sel_seeded["choice"] is not None or sel["choice"] is None:
            sel = sel_seeded
        else:
            profile_srp = False
        info["profile_srp"]["kept"] = profile_srp
    info["compare"] = {k: {kk: v[kk] for kk in _NG_COMPARE_KEYS}
                       for k, v in sel["models"].items()}
    info["rms0"] = sel["rms0"]
    info["compare_base"] = {"chi2_0": sel["chi2_0"], "N": sel["N"], "rms0": sel["rms0"]}
    if sel["choice"] is None:
        info["reason"] = "no nongrav model justified"
        return _keep_better_gravity(ctx, fit, sel, info, verbose), info

    # Justified on the frozen mask -> run a rejection round with the chosen model active,
    # then re-test: rejecting obs changes the included set, rms, and significance, which can
    # invalidate the selection (decision: re-test, keep only if still justified).
    # Re-fit the chosen model at the TIGHT tol first: the comparison trial that picked it ran at
    # the loose NONGRAV_DISCOVER_TOL, but the rejection round keys off this orbit's per-obs
    # residuals, so it must be fully converged or it would reject a slightly different obs set
    # (2007 XB23 is sensitive to exactly this). Only the rare detection path pays this fit.
    # SEEDED FROM THE TRIAL THAT WON, not cold from gravity: the trial already found the valley
    # floor (through its stall rescue or nested warm start, if it needed them), and a cold tight-
    # tol re-fit can land somewhere else -- 2008 NA's delivery once stalled at A2 = 1.7e-15 on a
    # mask where the trial had reached A2 z = 9.5, and shipped a 1.49" orbit for a 0.83" one.
    # The seeded LM starts AT the minimum and only polishes it at the tight tol.
    def _evaluate(seed6, mask, warm_fits):
        return _ng_evaluate(ctx, seed6, fit.history, mask, max_nfev, verbose, warm_fits=warm_fits,
                            profile_srp=profile_srp,
                            cold=not subsampled)      # the subsample already ran the cold trials
    sel2, reject_mask = _ng_settle(sel, test_mask, _evaluate, max_nfev, info, verbose)
    info["retest"] = {k: {kk: v[kk] for kk in _NG_COMPARE_KEYS}
                      for k, v in sel2["models"].items()}
    info["retest_base"] = {"chi2_0": sel2["chi2_0"], "N": sel2["N"], "rms0": sel2["rms0"]}
    if sel2["choice"] is None:
        info["reason"] = "nongrav not justified after rejection round; reverted to gravity"
        return _keep_better_gravity(ctx, fit, sel2, info, verbose), info

    _mrec, _mstill = _ng_mask_recheck(ctx, sel2, reject_mask, max_nfev, verbose)
    info["mask_recheck"] = _mrec
    if not _mstill:
        # `_mrec["why"]` already names its own bar and numbers (dBIC vs NONGRAV_BIC_MARGIN
        # for Phase A, raw rms gain vs NONGRAV_MIN_RMS_GAIN for Phase B) -- appending a fixed
        # "(margin N)" here used to cite the BIC margin even when Phase B's unrelated
        # raw-visibility floor was what actually collapsed, which reads as if the dBIC bar
        # failed when it was never evaluated.
        info["reason"] = (f"same-data re-check: {_mrec.get('why', 'collapsed')}; "
                          "reverted to gravity")
        if verbose:
            print("  discover: " + info["reason"])
        return _keep_better_gravity(ctx, fit, sel2, info, verbose), info

    # The comparison trials ran at the loose NONGRAV_DISCOVER_TOL (decision only). Re-fit the
    # chosen model at the tight default so the DELIVERED orbit is fully converged (same precision
    # as before this speedup -- the loose tol is confined to the throwaway comparison fits).
    # Seeded from the re-test's winning trial, for the reason given at `chosen`.
    _ast = _nongrav_asteroids_gate(fit)
    info["asteroids"] = bool(_ast)
    final = refine_nongrav(sel2["grav"], sel2["choice"], mask=reject_mask,
                           max_nfev=max_nfev, revert_if_worse=False, asteroids=_ast,
                           seed_state=_lift_nongrav_state(sel2["models"][sel2["choice"]]["fit"],
                                                          sel2["choice"]))
    if _ast:
        _rec, _still = _asteroids_detection_recheck(ctx, sel2["grav"], final, reject_mask,
                                                    sel2["choice"], sel2["N"])
        info["asteroids_recheck"] = _rec
        if not _still:
            info["reason"] = (f"perturbed re-check: dBIC {_rec['dbic']:.1f} < "
                              f"{NONGRAV_BIC_MARGIN:.0f} -- the planets-only justification was "
                              "absorbing the missing asteroid perturbers; reverted to gravity")
            if verbose:
                print("  discover: " + info["reason"])
            return _keep_better_gravity(ctx, fit, sel2, info, verbose), info
    # Delivery guard. This is a DATA-CONFIDENCE gate, NOT a physics test -- the
    # tracklet count does not change the implied density (that is fixed by A1 and H); it changes how
    # much we trust the DETECTION IS REAL vs. an artifact. An implausible density already tells us the
    # body is not a normal rock; the question the gate answers is whether that odd A1 is a genuine
    # signal or junk. If it is carried by a coherent systematic across MULTIPLE tracklets it is a real
    # object with unusual properties (candidate dark comet / lost rocket body) worth flagging even
    # though the numbers are odd; if it rests on a SINGLE tracklet it is far more likely bad astrometry
    # or a mislink, so default back to gravity (dropping that tracklet). So YES -- the same A1 can be a
    # detection on a well-covered arc and noise on a one-night one; that is correct (a coefficient
    # pinned by one bad night is genuinely less trustworthy), and it is a confidence call, not a claim
    # about the physics. A physically REASONABLE coefficient is always reported (no tracklet gate). See
    # _coherent_block_nights.
    # The guard reads the WARNING KINDS, not the warning: it is a rule about a RADIAL A1
    # whose implied density is off (or whose sign is), resting on too little data. The A2 size
    # ceiling became advisory on 2026-08-25 and its note joined the same warning string -- which
    # made this guard silently veto every over-ceiling Yarkovsky on a <2-block arc (2011 TB4 lost
    # a z=3.9 A2 that way while the trace said "choice=yarkovsky" twice). Advisory means advisory.
    _H = None
    try:
        _H = final.absolute_magnitude()[0]
    except Exception:
        pass
    import pyroorb.nongrav as _ng
    _, _kinds = _ng.nongrav_physical_warnings(final.nongrav_params, _H, with_kinds=True)
    if (_kinds & {"a1_negative", "a1_density"}) and block_nights < COHERENT_BLOCK_NIGHTS:
        info["model"] = None
        info["reason"] = (f"nongrav physically implausible and supported by <{COHERENT_BLOCK_NIGHTS} "
                          f"tracklets ({block_nights}); reverted to gravity")
        return _keep_better_gravity(ctx, fit, sel2, info, verbose), info
    if not _ng_delivery_sane(final, fit, sel2, info, verbose):
        return _keep_better_gravity(ctx, fit, sel2, info, verbose), info
    info["applied"] = True
    info["model"] = sel2["choice"]
    info["rms"] = final.rms_residual()
    info["params"] = dict(final.nongrav_params)
    info["sigmas"] = dict(final.nongrav_sigmas)
    if verbose:
        ps = "  ".join(f"{nm}={final.nongrav_params[nm]:+.3e}±{final.nongrav_sigmas[nm]:.1e}"
                       for nm in final.nongrav_params)
        print(f"discover_nongrav: applied {sel2['choice']}  rms={info['rms']:.3f}\"  {ps}")
    return final, info


# --- automatic COMET out-gassing discovery (the parallel system) -------------------------
# The comet analogue of discover_nongrav. Comet non-gravs are PHYSICALLY different from the
# asteroid Yarkovsky/SRP family: they are driven by water-ice sublimation, so they use the
# Marsden-Sekanina-Yeomans g(r) law (steep roll-off past ~2.8 au, nongrav.COMET_GR) rather than
# 1/r^2, the RADIAL term A1 is expected to dominate, and -- crucially -- the signal is detectable
# within a SINGLE perihelion passage rather than as a secular multi-revolution drift. So this path
# does NOT inherit the asteroid gate (q<1.4 au, >=2 oppositions); it triggers on a cometary
# designation (C//P//D//I/) OR a dynamically-cometary Tisserand parameter (T_J <= COMET_TJ_MAX),
# with no apparition requirement, and selects via a NESTED model sequence (radial A1 -> +transverse
# A2 -> +normal A3) keeping the most complex step whose added term is statistically justified over
# the previous one. The STATISTICS remain the sole arbiter inside the gate -- same chi2 effect-size
# + nested-F-test + plausibility philosophy as the asteroid path. ON by default
# (PYROORB_AUTO_COMET_NONGRAV=0 to disable); when it fires it WINS the overlap (the asteroid discovery
# is skipped for that object -- see the _finalize wiring).
COMET_TJ_MAX = 3.01         # Tisserand-vs-Jupiter trigger: at/below this the orbit is dynamically
                            # cometary (the Jupiter-family regime is 2<T_J<3); 3.01 leaves a thin
                            # margin for transition objects. A C//P//D//I/ designation triggers
                            # regardless.
COMET_MIN_OBS = 10          # need enough astrometry to meaningfully fit up to 3 out-gassing terms
COMET_Q_MAX = 4.0           # perihelion floor on the T_J trigger: the model is the Marsden WATER-ICE
                            # g(r) (scale r0=2.808 au, steep roll-off beyond), so an object whose
                            # perihelion never gets within ~3 au of the Sun cannot sublimate detectably
                            # -- at q=4 au g(r) is ~3e-5 of its 1-au value, so the comet fit can only
                            # ever DECLINE. Skipping it there is a pure speed win with no behavior
                            # change (e.g. Jupiter Trojans like 33822, q=4.34, T_J=2.88 -- dynamically
                            # cometary but never warm). A C//P//D//I/ designation still triggers
                            # regardless of q. The highest-q object we must still CHECK is the
                            # retrograde Jupiter co-orbital 514107 BZ509 (q=3.18), well under the floor.
# Each step is judged on its INCREMENTAL improvement over the previous one (see _ng_comet_choice):
# the added A-terms must clear the asteroid path's BIC bar (NONGRAV_BIC_MARGIN) AND a
# nested F-test, with every coefficient physically plausible. The hand-set effect-size floor
# (COMET_CHI2_GAIN) and z cut (COMET_MIN_Z) were retired from selection 2026-09-14 for the reason
# BIC selection retired the asteroid ones -- a FRACTIONAL chi2 bar demands ever more from exactly the dense
# long arcs where a force is most detectable: 2I's comet1 cut chi2 8322 -> 7697 over 6696
# residuals (8%, dBIC 616, z 28) and was refused, so the fit clipped the arc's ends to cope. Kept
# as constants for reference only.
COMET_CHI2_GAIN   = 0.15    # RETIRED from selection (see above)
COMET_FTEST_ALPHA = 0.05    # nested F-test significance for each added term
COMET_MIN_Z       = 5.5     # RETIRED from selection (see above)

# Comet out-gassing model sequence, in nested order of increasing complexity (radial first, the physical
# prior for sublimation thrust): A1, then +A2, then +A3. All ride the COMET_GR sublimation law.
_NG_COMET_TRIALS = (("comet1", ("A1",)),
                    ("comet2", ("A1", "A2")),
                    ("comet3", ("A1", "A2", "A3")))
_COMET_PREFIXES = ("C/", "P/", "D/", "I/")   # cometary designation prefixes (long-period, periodic,
                                             # defunct/lost, interstellar)


def _comet_phys_ok(f, axes):
    """Physical-plausibility test for the comet out-gassing family: a generous magnitude ceiling on
    every fitted A coefficient (nongrav.comet_a_is_plausible), since comet nongravs run orders of
    magnitude larger than asteroid Yarkovsky and the chi2/F-test bars do the real discrimination."""
    import pyroorb.nongrav as ng
    return all(ng.comet_a_is_plausible(f.nongrav_params.get(nm, 0.0)) for nm in axes)


def _comet_step_justified(prev_chi2, prev_k, chi2, k, N):
    """Is adding the terms that take a (prev_k-6)-A model to a (k-6)-A model statistically
    warranted? The INCREMENTAL BIC, (prev_chi2 - chi2) - dp*ln(N) over the dp=k-prev_k added
    parameters, must reach NONGRAV_BIC_MARGIN AND the nested F-test must be significant at
    COMET_FTEST_ALPHA. Shared by the lazy model sequence evaluator (_comet_continue) and
    the final selection (_ng_comet_choice) so the two stay exactly consistent."""
    from scipy.stats import f as _f_dist
    dp = k - prev_k
    dof = N - k
    if dof <= 0 or chi2 <= 0 or dp <= 0 or prev_chi2 <= 0:
        return False
    F_inc = ((prev_chi2 - chi2) / dp) / (chi2 / dof)
    ftest_ok = (float(_f_dist.sf(F_inc, dp, dof)) < COMET_FTEST_ALPHA) if F_inc > 0 else False
    return _dbic(prev_chi2, chi2, dp, N) >= NONGRAV_BIC_MARGIN and ftest_ok


def _comet_model_step(key):
    """(prev_chi2_key, prev_k, k) bookkeeping for model sequence step `key`: its own free-parameter count k
    and the previous step's index into the order ("comet1" -> gravity baseline). Returns (i, prev_k,
    k) where i is the step index (0-based)."""
    order = ["comet1", "comet2", "comet3"]
    axes_of = dict(_NG_COMET_TRIALS)
    i = order.index(key)
    prev_k = 6 if i == 0 else 6 + len(axes_of[order[i - 1]])
    k = 6 + len(axes_of[key])
    return order, i, prev_k, k


def _comet_continue(out, key):
    """Lazy model-sequence predicate for _ng_evaluate: AFTER step `key` has been fitted and scored, decide
    whether the NEXT step is worth fitting. Continue only if `key` itself is justified over the
    previous step -- if it is not, the nested selection would stop here anyway, so fitting comet2/
    comet3 would be wasted work (the common decline path on dynamically-cometary asteroids)."""
    m = out["models"].get(key)
    if m is None or not m.get("phys_ok") or not np.isfinite(m.get("chi2", float("inf"))):
        return False
    order, i, prev_k, k = _comet_model_step(key)
    prev_chi2 = out["chi2_0"] if i == 0 else out["models"][order[i - 1]]["chi2"]
    return _comet_step_justified(prev_chi2, prev_k, m["chi2"], k, out["N"])


def _ng_comet_choice(models, out):
    """Nested-model selection for the comet out-gassing family. Climb comet1 -> comet2 ->
    comet3, accepting each step only if it is justified OVER THE PREVIOUS one (see
    _comet_step_justified) AND every coefficient is physically plausible. The first step (comet1) is
    tested against the gravity baseline. Returns the highest justified step, or None. 'Most
    conservative justified' = stop adding terms as soon as the next term is not warranted (a comet3
    over a clean comet2, say, must EARN its A3). Steps the lazy evaluator never fitted are simply
    absent from `models`, which naturally stops the climb at the same place."""
    N = out["N"]
    prev_chi2 = out["chi2_0"]                          # gravity baseline for the first step
    chosen = None
    for key in ("comet1", "comet2", "comet3"):
        m = models.get(key)
        if m is None or not m.get("phys_ok") or not np.isfinite(m.get("chi2", float("inf"))):
            break
        _, _, prev_k, k = _comet_model_step(key)
        if _comet_step_justified(prev_chi2, prev_k, m["chi2"], k, N):
            chosen, prev_chi2 = key, m["chi2"]         # accept and keep climbing
        else:
            break                                      # next term not warranted -> stop (conservative)
    return chosen


def _ng_comet_choice_accum(models, out):
    """_ng_comet_choice with the comparison made against the best step ACCEPTED so far
    (gravity until one is), not the step immediately below: a step that is not justified
    is skipped, not a wall. Used by the apparition-wise extension's model sequence only. On a
    multi-apparition comet arc the radial A1 is nearly the semimajor axis and cannot
    express the transverse thrust that changes the period between perihelia, so comet1
    can gain 14% over gravity (one point under COMET_CHI2_GAIN) while comet2 gains 97% --
    and the strict model sequence, breaking at comet1, delivered nothing for an arc the extension
    had just shown the model holds. The added terms still have to EARN their place: the
    same incremental gain / F-test / z bars, on dp = the number of terms added over the
    accepted step (two, when comet2 is judged straight against gravity)."""
    N = out["N"]
    prev_chi2, prev_k = out["chi2_0"], 6
    chosen = None
    for key in ("comet1", "comet2", "comet3"):
        m = models.get(key)
        if m is None or not m.get("phys_ok") or not np.isfinite(m.get("chi2", float("inf"))):
            continue
        _, _, _, k = _comet_model_step(key)
        if _comet_step_justified(prev_chi2, prev_k, m["chi2"], k, N):
            chosen, prev_chi2, prev_k = key, m["chi2"], k
    return chosen


def _comet_continue_accum(out, key):
    """Lazy model-sequence predicate paired with _ng_comet_choice_accum: keep climbing after an
    unjustified step (the next one may be the answer), stop only after a step that failed
    to FIT (no finite chi2 -- the seeds are gone) or once comet3 has been tried."""
    m = out["models"].get(key)
    return m is not None and np.isfinite(m.get("chi2", float("inf")))


def _is_comet_designation(fit):
    """True if any of the fit's designations carries a cometary prefix (C//P//D//I/) -- i.e. the
    object was reported active. Absence is NOT evidence of an asteroid (dormant/mis-designated
    comets exist), which is why the Tisserand trigger backs this up."""
    import re
    desigs = [getattr(fit, "designation", "")] + list(getattr(fit, "designations", []) or [])
    for d in desigs:
        if not isinstance(d, str):
            continue
        s = d.strip()
        # "C/2025 N1", or a NUMBERED comet -- "10P", "3I", "73P-B": the MPC's numbered form carries
        # the same class letter the provisional prefix does, and "3I" missed the comet path entirely
        # (its Tisserand, the backup trigger, is undefined for an unbound orbit).
        if any(s.startswith(p) for p in _COMET_PREFIXES) or re.match(r"^\d+[PDI](?:$|[/-])", s):
            return True
    return False


def _comet_tisserand(fit):
    """Tisserand parameter vs Jupiter from the fitted a/e/i, computed CHEAPLY (no MOID, unlike
    FittedOrbit.tisserand_jupiter -> orbit_classification). Returns None if a/e/i are unavailable."""
    from pyroorb import hazard
    try:
        a, e, i = fit._val("a"), fit._val("e"), fit._val("i")
        if a is None or e is None or i is None:
            return None
        return hazard.tisserand_jupiter(a, e, i)
    except Exception:
        return None


# Apparition-wise extension under the comet model (see _comet_extend_apparitions). The
# acceptance bars are the ones the gravity linker already applies to an apparition it has
# just joined: its INCLUDED obs must sit at a sane median normalized residual after one
# rejection round (DISCORDANT_SIGMA), it must keep at least this fraction of its obs
# (an apparition the rejection guts is one the model does not fit, however low the
# survivors' median), and the whole fit's normalized rms must not blow up on joining.
#
# WHY NOT A BIC / LIKELIHOOD-RATIO BAR, like _comet_step_justified and NONGRAV_BIC_MARGIN?
# Because those answer a structurally DIFFERENT question. A model sequence step adds PARAMETERS to a
# FIXED dataset: chi2 can only fall, the models are nested, and dp*ln(N) is exactly the price
# of the extra freedom. A join adds OBSERVATIONS at a FIXED parameter count (comet3, 9
# parameters throughout): chi2 can only RISE, the two fits are densities over different sample
# spaces, and there is no dp to charge. "Does chi2 grow by less than ln(N) per added obs" is
# not a model-selection statement -- it is a goodness-of-fit statement about the new data, and
# it can be made to answer "join everything" or "join nothing" by choice of constant. The
# right instrument for "do these obs come from this orbit?" is a GOF test on the added
# residuals, and that is what these bars are -- a deliberately ROBUST one, because the
# alternative hypothesis is contamination (junk astrometry, a mislinked night), not a shifted
# mean, and a chi2 SUM is destroyed by the handful of blunders every real apparition carries.
#
# WHAT EACH BAR IS WORTH, measured on the one validation object (21P, full trace of the seven
# join trials, engine at 2788b05, seed rms_n 0.896 so the rms bar stood at 1.5):
#   obs[7416..7422]   7 obs 2020   kept 3/7    med 4.43  rn 0.91  refused -- median AND fraction
#   obs[7423..7433]  11 obs 2024   kept 11/11  med 2.42  rn 0.90  refused -- MEDIAN ALONE
#   obs[3200..3218]  19 obs 2012   kept 17/19  med 1.08  rn 0.87  JOINED
#   obs[2735..3199] 465 obs 2011   kept 415/465 med 1.55 rn 0.90  JOINED
#   obs[2364..2734] 371 obs 2005   rejection removed the whole apparition (no threshold reached)
#   obs[1573..2359] 787 obs 1997   kept 202/787 med 2.00 rn 1.01  refused -- FRACTION ALONE
#   obs[311..1564] 1254 obs 1985   rejection removed the whole apparition (no threshold reached)
# So only two of the seven verdicts were decided by a threshold at all, each by a DIFFERENT
# one, and 1997 is precisely the case the fraction bar exists for: rejection deleting 74% of an
# apparition to bring the survivors' median to the bar. A GOF test on post-rejection survivors
# would have taken it -- that test is circular, which is why the fraction clause is not
# redundant with the median clause. WHAT WOULD FALSIFY THIS: an object where an apparition the
# MPC's own solution includes is refused on the fraction bar while its survivors fit cleanly and
# the excluded obs are demonstrably a separate contaminant tracklet (then the bar wants to be
# per-tracklet, not per-apparition).
#
# Two of the three constants are NOT new to this path -- that is the point of reusing them:
# DISCORDANT_SIGMA (2.0) is the engine-wide discordant-apparition bar (_has_discordant_opposition)
# and COMET_EXT_KEEP_FRAC (0.5) is the gravity linker's own `survivors >= max(3, ncl//2)`
# (arc_extend._activate_clusters). COMET_EXT_RMS_N_FACTOR is the genuinely new one, and the
# trace above says plainly what it is worth on 21P: NOTHING. Every trial's rn landed in
# 0.87-1.01 against a 1.5 bar, so any factor >= ~1.13 gives 21P the same answer. Treat it as an
# UNCALIBRATED SAFETY NET against a join that bends the orbit to absorb a contaminant (the
# relative analogue of arc_extend.ACTIVATE_KEEP_NORM, relative because a comet's seed apparition
# sits near 1 sigma, not far below it) -- not as a tuned threshold. It cannot be calibrated from
# this object; doing so needs a second comet on this path.
COMET_EXT_KEEP_FRAC = 0.5
COMET_EXT_ANCHOR_FRAC = 0.25    # a clean apparition must carry this fraction of the richest clean
                                # one's obs to be preferred as the (most recent) anchor
COMET_EXT_RMS_N_FACTOR = 1.5
COMET_EXT_MODEL = "comet3"      # the step the extension is judged under: the FULL Marsden set,
                                # the most the model family can express, so an apparition is
                                # refused only when NO comet model holds it. The model sequence then
                                # re-decides comet1/2/3 on the final mask. Measured on 21P: 2011
                                # joins the 2018+2024 arc at median 1.41 sigma (399/465 kept)
                                # under comet3 and sits at 2.05 sigma (303/465) under comet2 --
                                # the same LM to the same tolerance, so not convergence.


# --- the extension's PURE decisions -----------------------------------------------------
# Everything below is arithmetic that used to sit inline in _comet_extend_apparitions,
# lifted out UNCHANGED (same expressions, same order, same short-circuits) so the
# orchestration rules -- which apparition seeds the growth, what order the rest are tried
# in, which side of the arc a refusal blocks, what counts as a join, which refusals get a
# second look -- can be unit-tested offline. Before the split the only thing that exercised
# any of them was the 21P benchmark fixture, which the routine suite does not run.


def _ext_seed_candidates(t, clusters, richest, hi0):
    """The apparitions eligible to RE-SEED the growth, most recent first: later than
    everything currently included (t[cs] > hi0) and rich enough to seed a comet fit --
    at least COMET_EXT_ANCHOR_FRAC of the richest included apparition's obs, so an 11-obs
    2024 fragment does not outrank 4197 obs from 2018. The caller tries them in this order
    and takes the first that actually fits (rms_n <= CATASTROPHIC_RMS_N)."""
    out = []
    for (cs, ce) in sorted(clusters, key=lambda c: -t[c[0]]):
        if t[cs] <= hi0 or ce - cs + 1 < max(FIT_MIN_OBS, COMET_EXT_ANCHOR_FRAC * richest):
            continue
        out.append((cs, ce))
    return out


class _ExtBlocks:
    """The refusal ledger: a refused apparition BLOCKS its side of the arc for every
    apparition beyond it that is no richer than itself. If the model cannot hold 1985's
    1254 obs it has no business claiming 1984's four beyond them -- a handful of loose old
    positions cannot refute the model, so they must not be allowed to extend it past an
    apparition that did. Two deliberate escapes: a RICHER apparition beyond a refusal is
    still tried (2011's 465 obs behind the refused 19-obs tail of its own apparition), and
    a refusal of fewer than COMET_MIN_OBS blocks nothing (too little astrometry to have
    said anything about the model).

    A block is judged against the arc AS IT STANDS (`blocked` takes the current lo/hi): a
    refusal bounds how far the model may be carried PAST it, so once a richer apparition has
    joined beyond it -- the arc went around it on the strength of that join -- it bounds
    nothing any more, and a poorer apparition further out gets its hearing. The hearing is
    the full trial (fit, rejection round, _ext_join_ok), so a sparse far apparition that IS
    a mislinkage is still refused on its own residuals; what the stale block denied was the
    trial, not the verdict. (Refusals are recorded when the apparition sits at the arc's
    edge -- the growth order never tries an interior one -- so this has to be decided at
    query time, not at `add`.)"""

    def __init__(self, t):
        self._t = np.asarray(t, float)
        self.blocks = []             # (side, time, n_obs) of each refusal

    def add(self, cs, ce, lo):
        """Record the refusal of obs[cs..ce] against an arc whose earliest included time is
        `lo`: the side is 'lo' when the apparition sits before the arc, 'hi' otherwise."""
        t = self._t
        self.blocks.append(("lo" if t[ce] < lo else "hi",
                            float(t[ce] if t[ce] < lo else t[cs]), ce - cs + 1))

    def blocked(self, cs, ce, lo, hi):
        """Is obs[cs..ce] foreclosed, given an arc currently spanning [lo, hi]? A block the
        arc has since grown around (a 'hi' refusal at or before `hi`, a 'lo' one at or after
        `lo`) is stale and counts for nothing."""
        t = self._t
        n_c = ce - cs + 1
        for side, tb, nb in self.blocks:
            if nb < COMET_MIN_OBS or nb < n_c:
                continue
            if side == "lo" and tb < lo and t[ce] < tb:
                return True
            if side == "hi" and tb > hi and t[cs] > tb:
                return True
        return False


def _ext_next_apparition(todo, t, hi):
    """Which of the remaining apparitions to try next: forward in time first (the nearest
    one later than the arc's current end `hi`), backward -- latest-first among what is left
    -- only once nothing later remains. By design: the apparitions being observed now are
    the ones the orbit is for."""
    later = [c for c in todo if t[c[0]] > hi]
    return (min(later, key=lambda c: t[c[0]]) if later
            else max(todo, key=lambda c: t[c[1]]))


def _ext_join_ok(rn, rn0, med, frac):
    """The join verdict for a trialled apparition: the whole fit must still be an orbit
    (CATASTROPHIC_RMS_N) and must not have blown up on joining (COMET_EXT_RMS_N_FACTOR x the
    seed's normalized rms, floored at 1.0 so a better-than-sigma seed does not set an
    unreachably tight bar), the joined apparition's surviving obs must sit at a sane median
    normalized residual (DISCORDANT_SIGMA), and it must have SURVIVED the rejection round
    (COMET_EXT_KEEP_FRAC) -- an apparition the rejection guts is one the model does not fit,
    however low the survivors' median. These are the bars the gravity linker already applies
    to an apparition it has just joined."""
    return bool(np.isfinite(rn) and rn <= CATASTROPHIC_RMS_N
                and rn <= COMET_EXT_RMS_N_FACTOR * max(rn0, 1.0)
                and med <= DISCORDANT_SIGMA and frac >= COMET_EXT_KEEP_FRAC)


def _ext_interior_refusals(refused, t, lo, hi):
    """Refusals that now sit strictly INSIDE the grown arc. An apparition refused against a
    shorter arc (the 19-obs tail of 2011 at 2.2 sigma, judged from 2018 alone) is a
    different question once the arc spans it, and joining an interior gap cannot extend the
    arc anywhere -- so those, and only those, get a second look."""
    return [r for r in refused if lo < t[r["obs"][0]] and t[r["obs"][1]] < hi]


def _comet_extend_apparitions(ctx, fit, max_nfev, verbose=False):
    """Grow the included set ONE APPARITION AT A TIME under the comet out-gassing model, from
    the apparition(s) `fit` holds, forward in time first (nearest later apparition), then
    backward. For each not-yet-included
    apparition: switch it on, re-fit COMET_EXT_MODEL seeded from the current solution, run
    one rejection round under the model, re-fit, and keep it only if it survived that
    round (COMET_EXT_KEEP_FRAC of its obs, median <= DISCORDANT_SIGMA) without inflating the
    whole fit's normalized rms past COMET_EXT_RMS_N_FACTOR x. Returns (ctx, fit, mask,
    warm_fits, record) -- the (possibly re-anchored) ctx and seed fit the extension ran on,
    the extended mask, the last converged comet fits keyed by model for the model sequence's warm
    start, and a record of what joined and what was refused -- or None if nothing joined.

    Why this exists: discover_comet_nongrav's premise is "re-include EVERYTHING; the gravity
    exclusions were the missing force". That is right when one A1/A2 spans the arc (10P,
    103P) and wrong when it does not: 21P's out-gassing has changed between apparitions
    (Sekanina 1993, Krolikowska 2001), so no constant Marsden set holds its 1900-2024 arc,
    the full-arc model sequence delivers a 491" orbit, _ng_delivery_sane refuses it, and the
    engine falls back to ONE apparition -- while the MPC's own published orbit fits three
    (2011-2024, A1+A2, 4322 of 4698 obs at rms_n 0.92). This is the middle answer: the
    LARGEST arc the model actually holds, decided apparition by apparition on the same
    residual bars the gravity linker uses, with the model sequence, its rejection settle and
    _ng_delivery_sane still the last word on the mask it hands back.

    Epoch: the seed apparition is re-anchored ONCE at its own mid-epoch
    (_reanchor_apparition) and the evaluator then stays there through the growth, like the
    MPC's own solution, whose epoch sits years outside its arc. Re-anchoring at each accepted
    join was considered and not done: it rebuilds the evaluator per step for an LM that
    converges as it is."""
    import pyroorb.arc_extend as ae
    ev = ctx["ev"]
    n = ev.n
    base_mask = np.asarray(fit._mask, bool).copy()
    clusters = _clusters(ev.times_tdb)
    if all(base_mask[cs:ce + 1].any() for (cs, ce) in clusters):
        return None
    t = np.asarray(ev.times_tdb, float)
    # THE SEED APPARITION: the most recent clean one that is well enough observed to seed a
    # comet fit -- at least COMET_EXT_ANCHOR_FRAC of the richest included apparition's obs, so
    # an 11-obs 2024 fragment does not outrank 4197 obs from 2018 -- rather than the richest
    # one the catastrophic stage handed over. By design: when the arc cannot
    # be held whole, the orbit that matters is the one for the apparitions being observed
    # NOW, not a good orbit for 1897; the growth order below is forward in time first for
    # the same reason.
    # REGRESSION NOTE: if comets on this path start delivering a poorer orbit that traces
    # back to a weak or badly placed seed apparition, A/B this choice first (richest vs
    # most-recent, and COMET_EXT_ANCHOR_FRAC) against the commit that introduced it before
    # looking elsewhere. The everything-mask path (10P, 103P) never comes through here.
    richest = max(int(base_mask[cs:ce + 1].sum()) for (cs, ce) in clusters)
    hi0 = float(t[base_mask].max())
    for (cs, ce) in _ext_seed_candidates(t, clusters, richest, hi0):
        try:
            m = np.zeros(n, dtype=bool); m[cs:ce + 1] = True
            st, result = ev.fit(np.asarray(fit._state, float)[:6], m, max_nfev=ACTIVATE_NFEV)
            if float(ev.rms_norm(st, m)) <= CATASTROPHIC_RMS_N:
                st, result = ev.fit(st, m, thin=False)
                cand = _build_orbit(ctx, st, m, result, fit.history)
                if cand.rms_normalized() <= CATASTROPHIC_RMS_N:
                    if verbose:
                        print(f"  extend: seeding from the most recent well-observed clean "
                              f"apparition obs[{cs}..{ce}] ({ce - cs + 1} obs) instead of the "
                              f"richest ({richest})")
                    fit, base_mask = cand, m
                    break
        except Exception:
            continue
    # ... re-anchored at ITS epoch (the full-arc evaluator can sit decades away), when the
    # included set is one apparition and the ctx can rebuild the evaluator.
    inc_apps = [(cs, ce) for (cs, ce) in clusters if base_mask[cs:ce + 1].any()]
    if len(inc_apps) == 1:
        re = _reanchor_apparition(fit, inc_apps[0][0], inc_apps[0][1], verbose)
        if re is not None:
            fit = re
            ctx = fit._ctx
            ev = ctx["ev"]
            base_mask = np.asarray(fit._mask, bool).copy()
    todo = [(cs, ce) for (cs, ce) in clusters if not base_mask[cs:ce + 1].any()]
    try:
        cur = refine_nongrav(fit, COMET_EXT_MODEL, mask=base_mask, max_nfev=max_nfev,
                             revert_if_worse=False, profile_sigmas=False,
                             tol=NONGRAV_DISCOVER_TOL)
    except Exception as ex:
        if verbose:
            print(f"  extend: seed {COMET_EXT_MODEL} fit failed ({type(ex).__name__}: {ex})")
        return None
    # Settle the seed apparition UNDER THE MODEL before judging anyone against it: the
    # gravity fit it arrives on could not reject (an out-gassing comet's single apparition
    # is misfit by gravity end to end, so nothing stands out) and every later join would be
    # measured against residuals inflated by that junk. One rejection round + re-fit, the
    # same treatment each joining apparition gets. `orig_n` on every rejection here: the
    # masks are subsets of the record (see auto_reject).
    cur_mask = base_mask.copy()
    try:
        ae.auto_reject_within_arc(cur._ctx["ev"], cur_mask, np.asarray(cur._state, float)[:6],
                                  orig_n=int(cur_mask.sum()))
        if cur_mask.sum() != base_mask.sum():
            cur = refine_nongrav(fit, COMET_EXT_MODEL, mask=cur_mask, max_nfev=max_nfev,
                                 revert_if_worse=False, profile_sigmas=False,
                                 tol=NONGRAV_DISCOVER_TOL,
                                 seed_state=_lift_nongrav_state(cur, COMET_EXT_MODEL))
    except Exception:
        cur_mask = base_mask.copy()
    rn0 = float(cur.rms_normalized())
    if not np.isfinite(rn0):
        return None
    if verbose:
        print(f"  extend: seed {COMET_EXT_MODEL} on {int(cur_mask.sum())} obs "
              f"({int(base_mask.sum())} in, one rejection round): rms_n {rn0:.2f}, growing "
              "apparition by apparition")
    joined, refused = [], []

    def _try(cs, ce, cur, cur_mask):
        """Join apparition obs[cs..ce] to `cur_mask` under the model: fit, one rejection
        round, re-fit, judge. Returns (ok, rec, cand, trial); on an exception ok is False
        and cand/trial are None."""
        trial = cur_mask.copy()
        trial[cs:ce + 1] = True
        tag = f"obs[{cs}..{ce}] ({ce - cs + 1} obs, JD {t[cs]:.0f})"
        try:
            cand = refine_nongrav(fit, COMET_EXT_MODEL, mask=trial, max_nfev=max_nfev,
                                  revert_if_worse=False, profile_sigmas=False,
                                  tol=NONGRAV_DISCOVER_TOL,
                                  seed_state=_lift_nongrav_state(cur, COMET_EXT_MODEL))
            ae.auto_reject_within_arc(cand._ctx["ev"], trial, np.asarray(cand._state, float)[:6],
                                      orig_n=int(trial.sum()))
            if not trial[cs:ce + 1].any():
                raise ValueError("rejection removed the whole apparition")
            cand = refine_nongrav(fit, COMET_EXT_MODEL, mask=trial, max_nfev=max_nfev,
                                  revert_if_worse=False, profile_sigmas=False,
                                  tol=NONGRAV_DISCOVER_TOL,
                                  seed_state=_lift_nongrav_state(cand, COMET_EXT_MODEL))
            rs = cand._ctx["ev"].resid_sigmas(np.asarray(cand._state, float)[:6])
            kept = trial[cs:ce + 1]
            med = float(np.median(rs[cs:ce + 1][kept]))
            frac = float(kept.sum()) / float(ce - cs + 1)
            rn = float(cand.rms_normalized())
        except Exception as ex:
            if verbose:
                print(f"  extend: {tag} refused ({type(ex).__name__}: {ex})")
            return False, {"obs": [cs, ce], "of": int(ce - cs + 1),
                           "why": f"{type(ex).__name__}: {ex}"}, None, None
        ok = _ext_join_ok(rn, rn0, med, frac)
        rec = {"obs": [cs, ce], "kept": int(kept.sum()), "of": int(ce - cs + 1),
               "median_sigma": round(med, 2), "rms_n": round(rn, 3)}
        if verbose:
            print(f"  extend: {tag}: kept {rec['kept']}/{rec['of']}, median {med:.2f} sigma, "
                  f"fit rms_n {rn:.2f} -> {'JOINED' if ok else 'refused'}")
        return ok, rec, cand, trial

    blocks = _ExtBlocks(t)           # the refusal ledger (see _ExtBlocks)
    while todo:
        lo = float(t[cur_mask].min()); hi = float(t[cur_mask].max())
        # Judged afresh each round, never culled: a block can go stale when a richer
        # apparition joins past it (see _ExtBlocks.blocked), and the apparition it held
        # back is only foreclosed for as long as the block stands.
        open_ = [c for c in todo if not blocks.blocked(*c, lo, hi)]
        if not open_:
            break
        cs, ce = _ext_next_apparition(open_, t, hi)   # forward in time first
        todo.remove((cs, ce))
        ok, rec, cand, trial = _try(cs, ce, cur, cur_mask)
        if ok:
            joined.append(rec)
            cur, cur_mask = cand, trial
        else:
            refused.append(rec)
            blocks.add(cs, ce, lo)
    # Second look at refusals now INSIDE the arc (see _ext_interior_refusals).
    if joined:
        lo = float(t[cur_mask].min()); hi = float(t[cur_mask].max())
        for rec in _ext_interior_refusals(refused, t, lo, hi):
            cs, ce = rec["obs"]
            ok, rec2, cand, trial = _try(cs, ce, cur, cur_mask)
            if ok:
                refused.remove(rec)
                rec2["retried"] = True
                joined.append(rec2)
                cur, cur_mask = cand, trial
    if not joined:
        return None
    warm = {COMET_EXT_MODEL: cur}
    # comet1's warm seed too: the nested model sequence must justify comet1 over gravity BEFORE it
    # looks at comet2, and a cold comet1 LM from the gravity baseline on a multi-apparition
    # comet arc (which gravity cannot hold) may never converge -- in which case the model sequence
    # would break at step one and decline the arc the extension just built.
    try:
        warm["comet1"] = refine_nongrav(fit, "comet1", mask=cur_mask, max_nfev=max_nfev,
                                        revert_if_worse=False, profile_sigmas=False,
                                        tol=NONGRAV_DISCOVER_TOL,
                                        seed_state=_lift_nongrav_state(cur, "comet1"))
    except Exception:
        pass
    return ctx, fit, cur_mask, warm, {"joined": joined, "refused": refused,
                                      "seed_obs": ([int(inc_apps[0][0]), int(inc_apps[0][1])]
                                                   if len(inc_apps) == 1 else None),
                                      "epoch_tdb": float(ctx["epoch_tdb"]),
                                      "seed_rms_n": round(rn0, 3),
                                      "n_included": int(cur_mask.sum())}


def discover_comet_nongrav(fit, max_nfev=NONGRAV_DISCOVER_NFEV, verbose=False):
    """Automatically decide whether a COMET OUT-GASSING model (Marsden A1/A2/A3 on the water-ice
    g(r)) is justified, and -- if so -- return the orbit refit with that model after a rejection
    round. The comet analogue of discover_nongrav. Returns (fit_out, info).

    Gate (trigger only -- see the module comment above): a C//P//D//I/ designation OR a Tisserand-
    vs-Jupiter <= COMET_TJ_MAX. There is NO multi-apparition requirement: comet out-gassing is
    detectable within a single perihelion passage, so a long-period one-apparition comet (e.g.
    C/2002 X5) qualifies. Within the gate the STATISTICAL SELECTION is the sole arbiter: a nested
    model sequence (comet1 -> comet2 -> comet3) keeps the most complex step whose added term clears the
    incremental chi2 effect-size bar AND a nested F-test AND the plausibility ceiling (see
    _ng_comet_choice).

    Procedure mirrors discover_nongrav: re-include EVERY obs, mask off the worst NONGRAV_DROP_WORST
    by sigma-residual, hold that mask fixed while fitting the model sequence, and -- if a model is justified
    -- switch it on, run ONE outlier-rejection round with the model active, then re-test on the
    post-rejection mask, keeping the model only if it is still justified there, else reverting to
    the gravity fit. The final delivered orbit is re-fit at the tight tol."""
    import pyroorb.arc_extend as ae

    info = {"applied": None, "model": None, "reason": None}
    a, _ = fit.semimajor_axis()
    tj = _comet_tisserand(fit)
    is_comet_desig = _is_comet_designation(fit)
    info["gate"] = {"a": a, "tisserand_jupiter": tj, "comet_designation": is_comet_desig}
    ctx = getattr(fit, "_ctx", None)
    backend = ctx["backend"] if ctx else None
    if ctx is None or not hasattr(backend, "predict_radec_jac_ng"):
        info["reason"] = "needs the ASSIST backend (no nongrav force otherwise)"
        return fit, info
    q, _ = fit.perihelion_distance()
    # A large-perihelion object on a cometary T_J (a Jupiter Trojan/co-orbital) never warms enough to
    # out-gas water ice, so the Marsden model can only decline -- skip the (~1 s) comet fit there. The
    # q-floor only EXCLUDES when q is known and clearly large; a comet designation always overrides.
    tj_trigger = (tj is not None and tj <= COMET_TJ_MAX
                  and not (q is not None and q > COMET_Q_MAX))
    triggered = is_comet_desig or tj_trigger
    if not triggered:
        info["reason"] = f"gate not met (comet_desig={is_comet_desig}, T_J={tj}, q={q})"
        return fit, info

    ev = ctx["ev"]
    # Restore any phase-bridge link relaxation: the coherent far-apparition offset is the signal
    # the comet model exists to fit, not something to keep down-weighted (see discover_nongrav).
    ev.restore_link_relaxation()
    n = ev.n
    if n < COMET_MIN_OBS:
        info["reason"] = f"too few obs for a comet fit (n={n} < {COMET_MIN_OBS})"
        return fit, info
    if verbose:
        print(f"discover_comet_nongrav: a={a if a is None else f'{a:.3f}'} au, "
              f"T_J={tj if tj is None else f'{tj:.3f}'}, comet_desig={is_comet_desig}")

    # Re-include everything, then drop the worst NONGRAV_DROP_WORST by sigma-residual (ranked at the
    # reactivated full-arc gravity solution, like discover_nongrav -- this gives a STABLE obs
    # selection; ranking at the clean-mask state instead destabilised which obs were dropped).
    all_mask = np.ones(n, dtype=bool)
    st_all, _ = ev.fit(np.asarray(fit._state, float), all_mask, thin=False)
    worst = np.argsort(ev.resid_sigmas(st_all))[::-1][:NONGRAV_DROP_WORST]
    test_mask = all_mask.copy()
    test_mask[worst] = False
    info["dropped"] = [int(i) for i in worst]

    final, sel_last = _comet_nested_models(ctx, fit, st_all, test_mask, info, max_nfev, verbose)
    if final is not None:
        return final, info

    # The full-arc premise failed -- nothing justified, or the model could not span the
    # re-included arc -- while the incoming fit holds whole apparitions EXCLUDED and gravity
    # cannot hold the everything-mask either. Then the exclusions were never a judgement
    # about those apparitions, only a gravity-only fit's inability to reach them, and the
    # honest answer lies between "everything" and "one apparition": grow the arc under the
    # model, apparition by apparition, and let the same model sequence judge THAT mask. See
    # _comet_extend_apparitions. Gated on both conditions so a comet whose gravity fit
    # legitimately rejected a contaminant apparition is left exactly as it was.
    own_mask = np.asarray(fit._mask, bool)
    stranded = any(not own_mask[cs:ce + 1].any() for (cs, ce) in _clusters(ev.times_tdb))
    try:
        rn_all = float(ev.rms_norm(st_all, all_mask))
    except Exception:
        rn_all = float("nan")
    if stranded and (not np.isfinite(rn_all) or rn_all > CATASTROPHIC_RMS_N):
        first_reason = info.get("reason")
        ext = _comet_extend_apparitions(ctx, fit, max_nfev, verbose)
        if ext is not None:
            ctx_e, fit_e, ext_mask, warm, rec = ext
            info["extension"] = rec
            info["extension"]["full_arc_reason"] = first_reason
            info["extension"]["full_arc_rms_n"] = round(rn_all, 2) if np.isfinite(rn_all) else None
            if verbose:
                print(f"  extend: full arc failed ({first_reason}); model sequence on the extended "
                      f"{int(ext_mask.sum())}-obs mask")
            ext_info = {}
            # The extended mask is a SUBSET of the record, so the settle's rejection guard is
            # measured against it (reject_orig_n); and its model sequence judges each step against
            # the best step ACCEPTED so far rather than the one just below it (see
            # _ng_comet_choice_accum): on a multi-apparition comet arc A1 alone can be
            # worthless (it is nearly the semimajor axis) while A1+A2 is the whole answer.
            final, sel_ext = _comet_nested_models(ctx_e, fit_e, np.asarray(fit_e._state, float)[:6],
                                           ext_mask, ext_info, max_nfev, verbose,
                                           warm_fits=warm,
                                           choice_fn=_ng_comet_choice_accum,
                                           continue_fn=_comet_continue_accum,
                                           reject_orig_n=int(ext_mask.sum()))
            info["extension"]["ladder"] = ext_info
            if final is not None:
                for k in ("applied", "model", "rms", "params", "sigmas", "reason"):
                    if k in ext_info:
                        info[k] = ext_info[k]
                info["reason"] = None
                return final, info
            info["reason"] = (f"{first_reason}; apparition-wise extension joined "
                              f"{len(rec['joined'])} apparition(s) but the model sequence then "
                              f"declined ({ext_info.get('reason')})")
            if verbose:
                print("  discover: " + info["reason"])
            return _keep_better_gravity(ctx, fit, sel_last, info, verbose), info
        elif verbose:
            print("  extend: no apparition joined under the comet model")
    return _keep_better_gravity(ctx, fit, sel_last, info, verbose), info


def _comet_nested_models(ctx, fit, st_seed, test_mask, info, max_nfev, verbose, warm_fits=None,
                  choice_fn=None, continue_fn=None, reject_orig_n=None):
    """The comet discovery proper on ONE frozen `test_mask`: model sequence round 1 (subsampled on
    a large arc), the reject/re-test settle, the same-data re-check, the tight delivery
    re-fit, the asteroid-perturber re-check and _ng_delivery_sane. Returns (final, sel) --
    `final` the delivered model orbit, or None with `info["reason"]` set and `sel` the last
    selection (for the caller's _keep_better_gravity). `st_seed` seeds the gravity baseline;
    `warm_fits` ({model_key: fit}) warm-starts round 1's steps (the apparition-wise
    extension hands its converged solutions in this way). Factored out of
    discover_comet_nongrav (2026-09-06) so the everything-mask attempt and the extended-mask
    attempt run the identical procedure; behavior-neutral for the former."""
    ev = ctx["ev"]
    choice_fn = choice_fn or _ng_comet_choice
    continue_fn = continue_fn or _comet_continue
    _keys = ("rms", "chi2", "chi2_gain", "F", "pval", "ftest_ok", "z", "phys_ok", "params", "diag")
    # NOT warm-started from a published orbit's A's: tried and measured a null (see
    # _fit_from_mpc_seed) -- the model sequence's cold trials plus the nested / previous-round warm
    # starts are what decide, on this arc as on every other.
    sel, _mask1, subsampled = _ng_first_round(
        ev, test_mask,
        lambda m: _ng_evaluate(ctx, st_seed, fit.history, m, max_nfev, verbose,
                               warm_fits=warm_fits,
                               trials=_NG_COMET_TRIALS, phys_fn=_comet_phys_ok,
                               choice_fn=choice_fn, continue_fn=continue_fn),
        info, verbose)
    info["compare"] = {k: {kk: v[kk] for kk in _keys} for k, v in sel["models"].items()}
    info["rms0"] = sel["rms0"]
    info["compare_base"] = {"chi2_0": sel["chi2_0"], "N": sel["N"], "rms0": sel["rms0"]}
    if sel["choice"] is None:
        info["reason"] = "no comet model justified"
        return None, sel

    # Justified on the frozen mask -> re-fit the chosen model at the TIGHT tol (the comparison
    # trials ran loose), run ONE rejection round with it active, then re-test on the new mask.
    # Seeded from the winning trial's solution (see discover_nongrav's `chosen`).
    def _evaluate(seed6, mask, warm_fits):
        return _ng_evaluate(ctx, seed6, fit.history, mask, max_nfev, verbose, warm_fits=warm_fits,
                            trials=_NG_COMET_TRIALS, phys_fn=_comet_phys_ok,
                            choice_fn=choice_fn, continue_fn=continue_fn,
                            cold=not subsampled)      # the subsample already ran the cold trials
    sel2, reject_mask = _ng_settle(sel, test_mask, _evaluate, max_nfev, info, verbose,
                                   reject_orig_n=reject_orig_n)
    info["retest"] = {k: {kk: v[kk] for kk in _keys} for k, v in sel2["models"].items()}
    info["retest_base"] = {"chi2_0": sel2["chi2_0"], "N": sel2["N"], "rms0": sel2["rms0"]}
    if sel2["choice"] is None:
        info["reason"] = "comet model not justified after rejection round; reverted to gravity"
        return None, sel2

    _mrec, _mstill = _ng_mask_recheck(ctx, sel2, reject_mask, max_nfev, verbose, comet=True)
    info["mask_recheck"] = _mrec
    if not _mstill:
        # `_mrec["why"]` already names its own bar and numbers (dBIC vs NONGRAV_BIC_MARGIN
        # for Phase A, raw rms gain vs NONGRAV_MIN_RMS_GAIN for Phase B) -- appending a fixed
        # "(margin N)" here used to cite the BIC margin even when Phase B's unrelated
        # raw-visibility floor was what actually collapsed, which reads as if the dBIC bar
        # failed when it was never evaluated.
        info["reason"] = (f"same-data re-check: {_mrec.get('why', 'collapsed')}; "
                          "reverted to gravity")
        if verbose:
            print("  discover: " + info["reason"])
        return None, sel2

    # Re-fit the chosen model at the tight default for delivery (full precision), seeded from the
    # re-test's winning trial (see discover_nongrav's `chosen`).
    _ast = _nongrav_asteroids_gate(fit)
    info["asteroids"] = bool(_ast)
    final = refine_nongrav(sel2["grav"], sel2["choice"], mask=reject_mask,
                           max_nfev=max_nfev, revert_if_worse=False, asteroids=_ast,
                           seed_state=_lift_nongrav_state(sel2["models"][sel2["choice"]]["fit"],
                                                          sel2["choice"]))
    if _ast:
        _rec, _still = _asteroids_detection_recheck(ctx, sel2["grav"], final, reject_mask,
                                                    sel2["choice"], sel2["N"])
        info["asteroids_recheck"] = _rec
        if not _still:
            info["reason"] = (f"perturbed re-check: dBIC {_rec['dbic']:.1f} < "
                              f"{NONGRAV_BIC_MARGIN:.0f} -- the planets-only justification was "
                              "absorbing the missing asteroid perturbers; reverted to gravity")
            if verbose:
                print("  discover: " + info["reason"])
            return None, sel2
    if not _ng_delivery_sane(final, fit, sel2, info, verbose):
        return None, sel2
    info["applied"] = True
    info["model"] = sel2["choice"]
    info["rms"] = final.rms_residual()
    info["params"] = dict(final.nongrav_params)
    info["sigmas"] = dict(final.nongrav_sigmas)
    if verbose:
        ps = "  ".join(f"{nm}={final.nongrav_params[nm]:+.3e}±{final.nongrav_sigmas[nm]:.1e}"
                       for nm in final.nongrav_params)
        print(f"discover_comet_nongrav: applied {sel2['choice']}  rms={info['rms']:.3f}\"  {ps}")
    return final, sel2


# --- constrained fits (find_orb's limited_orbit / evaluate_limited_orbit) ----------------
# Hold one or more orbital ELEMENTS fixed during the differential correction. find_orb
# (orb_func.cpp) reduces the free-parameter count; we use the equivalent PENALTY METHOD --
# append a residual row (element(state) - target)/sigma_pen to the weighted LM, so the solved
# 6-state is driven onto the constraint surface while the remaining freedom fits the
# astrometry. This is the same device already used internally to pin the semi-major axis for
# the multi-revolution bridge (arc_extend._fit_anchor_fixed_a), here generalized to a/e/q/Q/i
# and exposed as a user feature. The familiar presets are element constraints: a PARABOLIC
# orbit is e=1, a CIRCULAR orbit e=0. Opt-in: the default pipeline never calls this, so a
# gravity-only fit -- and the whole regression suite -- stays byte-identical.
#
# The penalty sigma is set RELATIVE to how tightly the *data* already constrain that element:
# sigma_pen = free_sigma / TIGHTEN. A fixed absolute weight fails -- it is far weaker than the
# astrometry's pull on a well-determined element (sigma_e ~ 1e-5) yet overkill on a loose one.
# Tying it to the free element sigma makes every constraint TIGHTEN x stiffer than the data
# (so the element is effectively held) with the SAME, bounded conditioning (~TIGHTEN^2)
# regardless of element scale or how well-determined the orbit is.
CONSTRAINT_TIGHTEN = 1e4    # hold each element this many x tighter than the data alone do
# Which classical elements can be constrained, by their index into element_vector():
# a (au), e (-), q (au), Q (au), i (deg). a/e/q/Q are frame-independent; i is the same
# ecliptic-J2000 inclination shown in the elements panel (element_vector does the rotation),
# so "fix i = 10" means the value the user sees. node/arg-perih/anomaly are intentionally
# omitted from the first cut (rarely constrained, phase-angle wrap complications).
_CONSTRAINABLE = {"a": 0, "e": 1, "q": 2, "Q": 3, "i": 4}

def _constraint_values(state, epoch_tdb, kinds):
    """The constrained elements' current values from a heliocentric-equatorial state,
    read straight from element_vector() so they match the displayed elements exactly."""
    ev = element_vector(state, epoch_tdb)
    return np.array([ev[_CONSTRAINABLE[k]] for k in kinds], dtype=float)


def _caller_mask(fit, mask):
    """A CALLER-SUPPLIED inclusion mask as a bool array aligned with ``fit._obs``.

    A mask is a positional claim about one specific observation list. Every public
    entry point that takes one (the interactive re-fits: nongrav, constraints, epoch
    re-anchor, asteroid polish, force-model swap) hands it straight to the evaluator,
    where ``np.flatnonzero`` accepts ANY length: a short mask silently fits a
    TRUNCATED arc and only dies later, deep in the products, as an unexplained
    IndexError. Fail here, where the caller can still be told what is wrong. ``None``
    means the fit's own mask."""
    if mask is None:
        return np.asarray(fit._mask, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1 or m.shape[0] != len(fit._obs):
        raise ValueError(f"inclusion mask has {m.shape[0] if m.ndim == 1 else m.shape} "
                         f"entries but the fit holds {len(fit._obs)} observations")
    return m


def refit_constrained(fit, constraints, max_nfev=200, verbose=False, mask=None,
                      tighten=CONSTRAINT_TIGHTEN):
    """Re-solve a converged orbit with one or more ELEMENT CONSTRAINTS held fixed --
    find_orb's limited_orbit / evaluate_limited_orbit.

    `constraints` is a dict ``{element: target}`` (or a list of ``(element, target)``
    pairs); ``element`` is one of ``a``/``q``/``Q`` (au), ``e`` (dimensionless) or ``i``
    (degrees). The common presets are just element constraints: a **parabolic** orbit is
    ``{"e": 1.0}``, a **circular** orbit ``{"e": 0.0}``, a fixed-perihelion comet
    ``{"q": ...}``, etc. Each constraint appends a penalty residual
    ``(element - target)/sigma_pen`` to the weighted LM (``sigma_pen`` = the element's
    free sigma / `tighten`), so the fitted 6-state stays on the constraint surface while
    the remaining freedom solves the astrometry. The covariance comes out CONSTRAINED for
    free: the penalty rows stiffen the Jacobian along each fixed direction, collapsing
    that element's sigma toward zero exactly as a held parameter should.

    The included-observation set is held FIXED (constraints don't re-reject); `mask`
    overrides it (the interactive UI passes the user's edited inclusion set). Works on
    EITHER backend -- a constraint is a pure function of the state, needing no special
    force. Opt-in: the default pipeline never calls it; the gravity fit and the whole
    regression suite are byte-identical. Returns a NEW FittedOrbit carrying the
    constrained orbit (with ``meta['constraints']`` set); an empty/invalid constraint set
    returns `fit` unchanged."""
    from scipy.optimize import least_squares
    items = list(constraints.items()) if isinstance(constraints, dict) else list(constraints)
    items = [(str(k), float(t)) for k, t in items if str(k) in _CONSTRAINABLE]
    if not items:
        return fit
    kinds = [k for k, _ in items]
    targets = np.array([t for _, t in items], dtype=float)
    nc = len(kinds)
    # Per-constraint penalty weight = tighten / sigma_pen_base, where sigma_pen_base is the
    # element's FREE 1-sigma (floored so a perfectly/under-determined element can't divide by
    # ~0). This holds each element `tighten`x tighter than the astrometry alone -- effectively
    # fixed -- at a scale-independent conditioning of ~tighten^2.
    free_sig = fit.element_sigmas()
    base = np.array([max(float(free_sig.get(k, 0.0)), 1e-6 * max(1.0, abs(t)))
                     for k, t in items])
    w = float(tighten) / base                          # length-nc penalty weights
    ctx = getattr(fit, "_ctx", None)
    if ctx is None:
        raise RuntimeError("fit carries no _ctx; cannot apply orbital constraints")
    ev = ctx["ev"]; t0 = ctx["epoch_tdb"]
    mask = _caller_mask(fit, mask)
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
    ra, dec, cosd = ev.ra[idx], ev.dec[idx], ev.cosd[idx]
    sra, sdec = ev.fit_sig_ra[idx], ev.fit_sig_dec[idx]
    m = len(idx)

    def _pen(state):
        return w * (_constraint_values(state, t0, kinds) - targets)   # nc penalty residuals

    if ev._predict_jac is not None:                       # analytic obs Jacobian path
        cache = {}
        def _compute(state):
            key = state.tobytes()
            hit = cache.get(key)
            if hit is None:
                cache.clear()
                hit = cache[key] = ev._predict_jac(ev.base_sim, t0, state, tt, Rs)
            return hit
        def resid(state):
            pred, _ = _compute(state)
            d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
            r = np.column_stack([d*3600.0/sra, (pred[:, 1]-dec)*3600.0/sdec]).ravel()
            return _clamp(np.concatenate([r, _pen(state)]))
        def jac(state):
            _, J = _compute(state)                        # (m,2,6)
            out = np.empty((m, 2, 6))
            out[:, 0, :] = J[:, 0, :] * (cosd*3600.0/sra)[:, None]
            out[:, 1, :] = J[:, 1, :] * (3600.0/sdec)[:, None]
            obs_rows = np.where(np.isfinite(out), out, 0.0).reshape(2*m, 6)
            # penalty Jacobian: w * d(element)/d(state) by forward finite difference --
            # element_vector is cheap (no integration), so the 6 extra evals are ~free.
            c0 = _constraint_values(state, t0, kinds)
            G = np.zeros((nc, 6))
            for kk in range(6):
                h = 1e-7 * max(1.0, abs(state[kk]))
                sp = state.copy(); sp[kk] += h
                G[:, kk] = (_constraint_values(sp, t0, kinds) - c0) / h
            return np.vstack([obs_rows, w[:, None] * G])
        result = least_squares(resid, np.asarray(fit._state, float), jac=jac, method="lm",
                            xtol=1e-10, ftol=1e-10, max_nfev=max_nfev)
    else:                                                 # finite-difference fallback
        def resid(state):
            pred = ev._predict(ev.base_sim, t0, state, tt, Rs)
            d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
            r = np.column_stack([d*3600.0/sra, (pred[:, 1]-dec)*3600.0/sdec]).ravel()
            return _clamp(np.concatenate([r, _pen(state)]))
        result = least_squares(resid, np.asarray(fit._state, float), method="lm",
                            xtol=1e-10, ftol=1e-10, max_nfev=max_nfev)
    state = np.asarray(result.x[:6], float)

    cons_meta = {k: float(t) for k, t in items}
    ctx2 = dict(ctx)
    ctx2["meta"] = {**(ctx.get("meta") or {}), "constraints": cons_meta}
    fit2 = _build_orbit(ctx2, state, mask, result, fit.history)
    if verbose:
        got = _constraint_values(state, t0, kinds)
        cs = ", ".join(f"{k}={t:g}(→{g:.4g})" for (k, t), g in zip(items, got))
        print(f"  constrained[{cs}]: rms_norm {fit.rms_normalized():.3f} -> "
              f"{fit2.rms_normalized():.3f}")
    return fit2


def reanchor_fit_to_epoch(fit, new_epoch_tdb, mask=None):
    """Re-express a converged GRAVITY orbit at a NEW osculating epoch WITHOUT a
    differential correction -- find_orb's "set epoch" (`e`). The physical orbit is held
    fixed: integrate the state from the fit's epoch to `new_epoch_tdb`, rebuild the
    evaluator anchored there (only t0 + the perturber base-sim depend on the epoch; the
    obs times, observer positions and sigmas do not), and re-evaluate the covariance at
    the new epoch. The osculating elements (and their epoch) move; the residuals are
    identical to propagation precision, because it is the same trajectory. A subsequent
    Full step then runs from this new epoch. Nongrav orbits are re-anchored by re-deriving
    the out-gassing on the new-epoch gravity baseline (refine_nongrav), not here.

    Returns a new FittedOrbit at `new_epoch_tdb`; the original `fit` is left untouched."""
    ctx = getattr(fit, "_ctx", None)
    if ctx is None:
        raise RuntimeError("fit carries no _ctx; cannot re-anchor its epoch")
    backend = ctx["backend"]
    old_epoch = ctx["epoch_tdb"]
    new_epoch_tdb = float(new_epoch_tdb)
    if abs(new_epoch_tdb - old_epoch) < 1e-9:
        return fit
    new_base = backend.build_base_sim(new_epoch_tdb)
    # Propagate in the force model this orbit was FITTED in. ev2 below reuses the fit's own
    # predictor (an asteroid-polished fit carries a 16-perturber partial), so propagating
    # gravity-only here would move the state through one field and then re-evaluate its
    # residuals and covariance in another -- disagreeing by exactly the perturbation the
    # polish exists to capture, while this function's contract claims the trajectory is
    # unchanged. `asteroids` is only honoured by the ASSIST backend; the rebound one takes
    # no such kwarg, so pass it only when the fit actually asked for perturbers.
    _ast = bool(getattr(fit, "asteroids", False))
    state1 = backend.predict_state(ctx["fit_base"], old_epoch,
                                   np.asarray(fit._state, float), new_epoch_tdb,
                                   **({"asteroids": True} if _ast else {}))
    ev = ctx["ev"]
    ev2 = Evaluator(new_base, new_epoch_tdb, ev.times_tdb, ev.Robs, ev.ra, ev.dec,
                    sigma_ra=ev.sig_ra, sigma_dec=ev.sig_dec,
                    fit_sigma_ra=ev.fit_sig_ra, fit_sigma_dec=ev.fit_sig_dec,
                    predict_fn=ev._predict, predict_jac_fn=ev._predict_jac,
                    obscodes=ev.obscodes)
    mask = _caller_mask(fit, mask)
    result = ev2.evaluate(state1, mask)
    ctx2 = dict(ctx)
    ctx2["ev"] = ev2
    ctx2["epoch_tdb"] = new_epoch_tdb
    ctx2["fit_base"] = new_base
    # a re-anchor is gravity-only; drop any nongrav config carried on the source ctx/meta
    ctx2.pop("nongrav", None)
    meta = {**(ctx.get("meta") or {})}
    for key in ("nongrav_model", "nongrav_params", "nongrav_sigmas", "nongrav_warning"):
        meta.pop(key, None)
    ctx2["meta"] = meta
    return _build_orbit(ctx2, state1, mask, result, fit.history)


# --- massive-asteroid perturber polish (find_orb's BC405 precision step) -----------------
# The 16 ASSIST perturbers (Ceres/Vesta/Pallas/...) matter only for LONG main-belt arcs --
# the perturbation has to accumulate -- and are negligible on short / single-opposition arcs
# while slowing every integration (they force the tight epsilon). So rather than carry them
# through the whole fit, run a POLISH at the very end: re-fit the converged gravity orbit
# with the perturbers on, and -- because the truer force model can pull previously-EXCLUDED
# obs back in-bounds (2009 HV119 redeemed 8 obs and dropped 0.32"->0.22") -- redeem/re-reject
# and re-fit, looping until the obs set and orbit settle. Gated to >=2-opposition main-belt
# objects; `force=True` bypasses the gate (the CLI --asteroids flag). Opt-in: the default
# library/CLI pipeline never calls this, so the regression suite stays byte-identical.
ASTEROID_MIN_OPP = 2
ASTEROID_A_RANGE = (1.7, 4.5)    # main-belt semi-major axis (au): where the 16 perturbers live
# Keep the polish ONLY when it meaningfully lowers the RMS on the SAME obs set (gravity vs
# asteroid orbit over the gravity fit's included obs). Reclaiming observations does not count
# as helping -- the test mirrors what the user sees when toggling the perturbers on a fixed obs
# set. Without a real margin, sub-mas noise (a 70-object sweep: median Δrms -0.1 mas) or a stray
# redeem would leave objects in 16-ast mode with no actual improvement (e.g. 220088). When the
# asteroid fit holds obs gravity rejected, the same test also runs on the ASTEROID fit's obs set
# (gravity's own set can exclude exactly the apparitions only the perturbers fit).
ASTEROID_KEEP_RMS = 0.02         # arcsec: minimum same-obs RMS improvement to keep the perturbers


# Auto-detect-center thresholds. The TRIGGER is find_orb's domain test
# (find_best_fit_planet): is the object inside a giant planet's Hill sphere / sphere of influence?
# -- run on the converged position, so a satellite is caught even when its heliocentric fit is
# perfectly good (a short satellite arc fits heliocentrically). A poor heliocentric rms is a
# SECONDARY trigger (there the converged position is unreliable, so we lean on the obs-based Hill
# cone). Acceptance: the planetocentric orbit must be BOUND (find_orb's ecc<1.01 fly-by reset) and
# fit at least as well as heliocentric (the planet frame is the correct one, not merely better).
AUTO_CENTER_HILL_FACTOR = 1.2    # trigger if within this * Hill radius of a giant planet
AUTO_CENTER_TRIGGER_RMS = 30.0   # arcsec -- above this the heliocentric state is garbage: re-fit, don't reframe
AUTO_CENTER_MAX_SAT_RMS = 5.0    # arcsec -- absolute quality bar for the garbage-case planet re-fit


def autodetect_center(fit, obs, obscodes, desig="object", meta=None, verbose=False):
    """Auto-detect a natural-satellite (planetocentric) orbit. OPT-IN post-fit step
    (mirrors refine_asteroids / refine_nongrav): given a converged HELIOCENTRIC ``fit``, decide
    whether the object is actually orbiting a giant planet and, if so, return a re-fitted
    planetocentric FittedOrbit (with ``.center`` set). Otherwise returns the input fit unchanged.

    Method (find_orb ``find_best_fit_planet``): the TRIGGER is the gravitational-domain test --
    is the object inside a giant planet's Hill sphere (`planetocentric.dominating_planet`)? -- run
    on EVERY fit's converged position, so a satellite is caught even when a short arc fits
    heliocentrically fine (find_orb likewise never gates on residual). A poor heliocentric rms is a
    secondary trigger for the garbage-fit case (long, curved arcs, where the converged position is
    unreliable). The planet is then confirmed + identified from the OBSERVATIONS
    (`satellite_candidate`, an obs-based Hill-cone test that doesn't depend on the fit), we re-fit
    about it (the planet-hinted seed via ``center=``), and ACCEPT only if the planetocentric orbit
    is BOUND (find_orb's ecc>1.01 fly-by reset) and fits at least as well as heliocentric. A Trojan
    / co-orbital sits ~5 au from the planet (outside the Hill sphere), so it stays heliocentric.
    Never raises: any failure returns the original heliocentric fit."""
    import pyroorb.planetocentric as _pc
    try:
        helio_rms = fit.rms_residual()
        near = _pc.dominating_planet(fit.state_vector(), fit.epoch_tdb,
                                     factor=AUTO_CENTER_HILL_FACTOR)
    except Exception:
        return fit
    poor = bool(helio_rms and helio_rms >= AUTO_CENTER_TRIGGER_RMS)
    if near is None and not poor:
        return fit                        # not inside any Hill sphere and fits fine -> not a satellite
    # Confirm + identify the planet from the OBSERVATIONS (robust to a garbage converged position):
    # does the object track inside a giant planet's Hill cone across the arc?
    try:
        geom = _observer_geometry(obs, obscodes)
        cand = _pc.satellite_candidate(geom["obs_ra"], geom["obs_dec"],
                                       geom["Robs"], geom["times_tdb"])
    except Exception as ex:
        if verbose:
            print(f"[auto-center] detection failed: {type(ex).__name__}: {ex}")
        return fit
    if cand is None:
        if verbose:
            why = f"inside {near}'s Hill sphere" if near else f"poor heliocentric fit (rms {helio_rms:.0f}\")"
            print(f"[auto-center] {why} but the arc is not inside any giant planet's Hill cone; "
                  f"staying heliocentric")
        return fit
    planet, med_sep, hill_deg = cand
    # CASE A -- the heliocentric fit already converged (not garbage): just REFRAME it about the
    # planet (find_orb's approach -- the central body is a reporting transform, NOT a re-fit). The
    # same physical orbit, reported planetocentrically, at the same (good) rms. A bound reframe is a
    # captured satellite; an unbound one (ecc>1.01) is a hyperbolic fly-by through the Hill sphere,
    # which find_orb resets to heliocentric.
    if not poor:
        pel = _pc.planetocentric_elements(fit.state_vector(), fit.epoch_tdb, planet)
        if pel["bound"]:
            fit.center = planet
            if verbose:
                print(f"[auto-center] DETECTED satellite of {planet} (reframed the converged "
                      f"heliocentric orbit, rms {helio_rms:.3f}\"): a={pel['a']:.5f} au e={pel['e']:.3f}")
            return fit
        if verbose:
            print(f"[auto-center] inside {planet}'s Hill sphere but the orbit is unbound "
                  f"(hyperbolic fly-by, ecc={pel['e']:.2f}); staying heliocentric")
        return fit
    # CASE B -- the heliocentric fit is garbage (a long, curved satellite arc the heliocentric IOD
    # can't fit): re-fit from the planet-hinted seed to RECOVER the orbit, and accept a bound orbit
    # that actually fits (absolute quality bar; the garbage heliocentric rms is no yardstick).
    if verbose:
        print(f"[auto-center] poor heliocentric fit (rms {helio_rms:.0f}\") within {planet}'s Hill "
              f"cone; re-fitting from the planet-hinted seed")
    try:
        sat = fit_orbit(obs, obscodes, desig=desig, meta=meta, verbose=verbose, center=planet)
        pel = _pc.planetocentric_elements(sat.state_vector(), sat.epoch_tdb, planet)
        sat_rms = sat.rms_residual()
    except Exception as ex:
        if verbose:
            print(f"[auto-center] satellite re-fit failed: {type(ex).__name__}: {ex}")
        return fit
    if not (pel["bound"] and sat_rms < AUTO_CENTER_MAX_SAT_RMS):
        if verbose:
            print(f"[auto-center] declined: satellite rms {sat_rms:.2f}\", bound={pel['bound']} "
                  f"(need bound + rms < {AUTO_CENTER_MAX_SAT_RMS}\")")
        return fit
    if verbose:
        print(f"[auto-center] DETECTED satellite of {planet}: rms {sat_rms:.3f}\" "
              f"(heliocentric was {helio_rms:.0f}\"), planetocentric a={pel['a']:.5f} au e={pel['e']:.3f}")
    return sat


def _self_excluding(backend):
    """True when `backend` excludes its object from the asteroid perturber field
    (force_assist.SelfExcludedForce, directly or under _AsteroidForce) AND that exclusion can
    actually be built on this box -- otherwise force_assist keeps the perturbers off for the
    object, and a fit labelled `asteroids` would really be gravity."""
    name = excluded_body(backend)
    if name is None:
        return False
    try:
        import pyroorb.force_assist as _fa
        with _fa.perturbers_off_for(name):
            return _fa._use_asteroids(True)
    except Exception:
        return False


def _force_model_label(ctx, asteroids, fit=None):
    """The `force_model` tag of an orbit rebuilt from `ctx` with the asteroid perturbers on
    or off. refine_asteroids and refit_force_model swap the evaluator's force model but
    rebuild the orbit from the ORIGINAL fit context, whose `backend_name` names the model
    that fit ran in; without this the polished orbit carried `asteroids=True` under an
    "ASSIST(DE440+GR)" label (2022 KX28, polished from gravity to the 16 perturbers)."""
    name = ctx.get("backend_name") or "?"
    if not name.startswith("ASSIST("):
        return name                             # rebound 8-planet: no perturbers either way
    if not asteroids:
        return "ASSIST(DE440+GR)"
    sp = excluded_body(ctx.get("backend"))
    if sp is not None:
        if _self_excluding(ctx.get("backend")):
            return f"ASSIST(DE440+GR+15ast, excl. {sp})"
        return "ASSIST(DE440+GR, +16ast off: self-perturber)"
    return "ASSIST(DE440+GR+16ast)"

def refine_asteroids(fit, force=False, max_iter=6, verbose=False, mask=None):
    """Polish a converged GRAVITY orbit with the 16 massive-asteroid perturbers.

    Gated to main-belt (`a` in ASTEROID_A_RANGE), >=`ASTEROID_MIN_OPP`-opposition objects --
    the only regime where the perturbation is worth the cost; `force=True` bypasses the gate.
    Re-fits with the asteroid force on an evaluator clone, then runs redemption + rejection
    and re-fits to convergence (the better force model commonly reclaims obs gravity had
    excluded). KEEPS the asteroid orbit iff both models refit on the same obs set -- the
    gravity fit's, or the asteroid fit's own when it holds obs gravity rejected -- show the
    perturbers lowering the RMS by more than ASTEROID_KEEP_RMS. Requires the ASSIST backend +
    the sb441-n16 kernel (auto-downloaded on first use; gravity fallback if unavailable).
    Returns a NEW FittedOrbit carrying the asteroid-aware evaluator (so a subsequent Full
    step keeps the perturbers), or `fit` unchanged when skipped/unavailable/not improving."""
    import functools, copy as _copy
    import pyroorb.arc_extend as _ae
    ctx = getattr(fit, "_ctx", None)
    if ctx is None:
        return fit
    if getattr(fit, "asteroids", False):     # already fitted WITH the perturbers (an ident fit)
        return fit
    backend = ctx["backend"]
    if not hasattr(backend, "ensure_asteroid_kernel"):        # rebound 8-planet backend: no-op
        return fit
    if pinned_asteroids(backend):           # fitted with the perturbers already (whole-fit run)
        return fit
    # Self-perturber guard: if the object IS one of the 16 sb441-n16 perturbers, the polish
    # runs with it EXCLUDED from the field (+15ast) -- never in a field containing itself.
    # The fit's ctx backend already excludes it (force_assist.SelfExcludedForce); anything
    # else (a backend from before the guard ran) is refused rather than trusted.
    sp = self_perturber_name(fit=fit)
    if sp is not None and not _self_excluding(backend):
        if verbose:
            print(f"  asteroid polish: ({sp}) is itself one of the 16 perturbers and this "
                  f"fit's backend does not exclude it; kept gravity orbit")
        return fit
    if not force:
        if fit.n_oppositions() < ASTEROID_MIN_OPP:
            return fit
        a = fit.semimajor_axis()[0]
        if not (ASTEROID_A_RANGE[0] <= a <= ASTEROID_A_RANGE[1]):
            return fit
    if backend.ensure_asteroid_kernel(verbose=verbose) is None:
        if verbose:
            print("  asteroid polish: no perturber kernel available; kept gravity orbit")
        return fit

    # Asteroid-aware evaluator: a shallow clone of the gravity evaluator re-bound to the
    # perturbers (with_force_model; obs, sigmas, base_sim are shared). The keep-gate below
    # compares it against the fit's own evaluator, which must therefore BE gravity.
    ev = ctx["ev"]
    ev2, ctx2 = with_force_model(ctx, True, fit)
    if not ctx2["asteroids"]:
        return fit                  # a perturber object whose exclusion cannot be built here
    ev2.nongrav = None

    mask_a = _caller_mask(fit, mask).copy()
    state = np.asarray(fit._state, float).copy()
    span0 = _ae._included_span(ev2, mask_a)       # evaporation baseline: fixed across rounds
    for _ in range(max_iter):
        state, _ = ev2.fit(state, mask_a, thin=False)
        before = mask_a.copy()
        _ae.auto_reject_within_arc(ev2, mask_a, state, orig_span=span0)  # redeem + reject, in place
        if np.array_equal(mask_a, before):
            break
    state, mask_a = _ae._final_redeem(ev2, state, mask_a, [], verbose)   # 3σ/0.5" amnesty
    state, result = ev2.fit(state, mask_a, thin=False)        # final covariance fit

    fit2 = _build_orbit(ctx2, state, mask_a, result, fit.history)

    # "Does it actually help?" -- judged FAIRLY on the SAME obs set (the gravity fit's
    # included obs), gravity orbit vs asteroid orbit. Reclaiming observations does NOT count
    # as helping (a redeem can leave an object in 16-ast mode with no real improvement, e.g.
    # 220088); only a MEANINGFUL drop in the same-obs RMS keeps the perturbers.
    # CRUCIAL: the asteroid orbit compared here must be a CLEAN refit on grav_mask, NOT the
    # redeemed `state` above -- that was fit on a redemption-grown obs set, so scoring it on
    # grav_mask conflates the obs-set change with the force-model change and reports a phantom
    # gain (e.g. 2010 TY11: redeemed-state 0.460" vs a true same-obs asteroid refit 0.491",
    # identical to gravity). The interactive A/B toggle (refit_force_model) refits both models
    # on the fixed mask; the keep-gate must measure the force model the same way or the auto-
    # polish keeps perturbers the toggle then shows make no difference.
    grav_mask = np.asarray(fit._mask, dtype=bool)
    rms_grav = fit.rms_residual()                       # gravity orbit on its obs
    state_ab, _ = ev2.fit(np.asarray(fit._state, float), grav_mask, thin=False)
    rms_ast = ev2.rms_arcsec(state_ab, grav_mask)       # asteroid orbit REFIT on the SAME obs
    rms_drop = rms_grav - rms_ast
    # The same test the other way round: the gravity fit's obs set is ITSELF a product of
    # the gravity force model -- when the missing perturbation is large, gravity's rejection
    # amputates exactly the apparitions it cannot fit, and on what is left the perturbers
    # change nothing. (613383) = 2022 KX28: gravity drops all of 2006/2010/2011 (35 obs,
    # 40-450" off) and fits 2013-2026 at 0.161"; +16ast holds all 153 at 0.263", while
    # gravity refit on those 153 manages 57.7". So when the asteroid fit holds obs gravity
    # rejected, refit gravity on the ASTEROID obs set too (seeded from both orbits, the
    # better kept -- a stalled gravity LM must not fake a gain) and keep the perturbers if
    # they win there by the same margin. A stray redeem (220088) that gravity also fits
    # stays below the margin.
    if not force and rms_drop <= ASTEROID_KEEP_RMS and (mask_a & ~grav_mask).any():
        rms_ast_a = fit2.rms_residual()
        rms_grav_a = min(ev.rms_arcsec(ev.fit(np.asarray(s, float)[:6].copy(), mask_a,
                                              thin=False)[0], mask_a)
                         for s in (fit._state, state))
        if verbose:
            print(f"  asteroid polish: on the asteroid fit's {int(mask_a.sum())} obs gravity "
                  f"{rms_grav_a:.3f}\" vs asteroids {rms_ast_a:.3f}\"")
        rms_drop = max(rms_drop, rms_grav_a - rms_ast_a)
    if not force and rms_drop <= ASTEROID_KEEP_RMS:
        if verbose:
            print(f"  asteroid polish: no real gain (same-obs rms {rms_grav:.3f}->"
                  f"{rms_ast:.3f}\", Δ{-rms_drop:+.4f}\" < {ASTEROID_KEEP_RMS}\"); kept gravity")
        return fit
    if verbose:
        print(f"  asteroid polish: same-obs rms {rms_grav:.3f}->{rms_ast:.3f}\" kept "
              f"(displayed {fit.n_observations()}->{fit2.n_observations()} obs @ "
              f"{fit2.rms_residual():.3f}\")")
    return fit2


def refit_force_model(fit, asteroids, mask=None, seed=None, max_nfev=200):
    """Re-fit on a FIXED observation set with or without the 16 asteroid perturbers -- a
    PURE force-model swap: NO redemption, NO rejection, the obs set is held exactly as `mask`.
    Unlike refine_asteroids (which redeems/re-rejects to find the asteroid orbit's natural obs
    set), this is the interactive toggle: flip the perturbers and re-solve the SAME obs. `fit`
    may carry either force model (its ctx is re-bound to the one asked for); `asteroids`
    True builds an asteroid-aware evaluator clone (kernel auto-downloaded; gravity fallback if
    unavailable). `seed` overrides the starting state -- IMPORTANT for an object whose obs set
    only makes sense with the perturbers (e.g. 2009 HV119): the asteroid fit on those obs has a
    spurious near-gravity local minimum, so turning asteroids ON must be seeded from a good
    asteroid state (the cached load-time polish), not the gravity orbit. Returns a NEW
    FittedOrbit on `mask`."""
    import functools, copy as _copy
    ctx = getattr(fit, "_ctx", None)
    if ctx is None:
        raise RuntimeError("fit carries no _ctx; cannot re-fit the force model")
    backend = ctx["backend"]
    ev = ctx["ev"]
    mask = _caller_mask(fit, mask)
    seed = np.asarray(fit._state if seed is None else seed, float)
    # Self-perturber guard (see refine_asteroids): an object that IS one of the 16 perturbers
    # gets the perturbers only with itself excluded (+15ast, the ctx backend's job); a
    # backend that does not exclude it stays on gravity whatever the caller asked for.
    use_ast = (bool(asteroids) and hasattr(backend, "ensure_asteroid_kernel")
               and backend.ensure_asteroid_kernel() is not None
               and (self_perturber_name(fit=fit) is None or _self_excluding(backend)))
    # Both directions re-bind the evaluator (with_force_model): "off" used to reuse the fit's own
    # evaluator and only relabel it, so a fit made with the perturbers on stayed perturbed under
    # a gravity label.
    ev2, ctx2 = with_force_model(ctx, use_ast, fit)
    if use_ast:
        ev2.nongrav = None
    # (An identification fit's ctx carries the forcing _AsteroidForce; re-pinning it False
    # unwraps it, so its gravity side is really gravity.)
    state, result = ev2.fit(seed, mask, thin=False, max_nfev=max_nfev)
    return _build_orbit(ctx2, state, mask, result, fit.history)


def _observer_geometry(obs, obscodes):
    """The EPOCH-FREE half of the fit context: observer positions, raw 1-sigma uncertainties
    (ADES rms when present, else Veres-2017 station lookup) and the over-observation-deweighted
    FIT sigmas (find_orb reweight_for_overobserving -- it inflates the fit weighting only;
    outlier rejection keeps the RAW sigma), per-obs TDB times, and the force-model backend +
    analytic-Jacobian fn. None of this depends on the osculating epoch, so it is built ONCE and
    the evaluator is then built at whatever epoch IOD chooses (see _build_evaluator_at). This is
    what lets initial_orbit own the epoch -- the caller no longer has to guess one up front.
    PYROORB_NO_OVEROBS=1 disables the de-weighting; PYROORB_FINITE_DIFF_JAC=1 forces finite differences."""
    Robs = list(observer_positions_batch([o["jd_utc"] for o in obs],
                                         [o["obscode"] for o in obs], obscodes,
                                         sat_pos=[o.get("sat_pos_au") for o in obs],
                                         site_itrs=[o.get("site_itrs_au") for o in obs]))
    obs_ra = np.array([o["ra_deg"] for o in obs])
    obs_dec = np.array([o["dec_deg"] for o in obs])
    # Use a pre-attached per-obs sigma (the ADES path carries rmsRA/rmsDec) only when it is
    # actually present AND non-None; otherwise fall back to the obscode-based model sigma. The
    # OBS80 path attaches no sigma at fetch (the key is absent) and a round-tripped fixture
    # materializes the key as None, so gate on the VALUE, not mere key presence.
    sig = [(o["sigma_ra"], o["sigma_dec"]) if o.get("sigma_ra") is not None
           else sigma_for(o["obscode"], jd_utc=o.get("jd_utc")) for o in obs]
    sigma_ra = np.array([s[0] for s in sig])
    sigma_dec = np.array([s[1] for s in sig])
    # Timing-error smear (find_orb errors.cpp): fold each obs's TIME uncertainty into
    # its position σ along the direction of motion. OPT-IN via PYROORB_TIME_SIGMA_SEC (the
    # assumed 1σ timestamp accuracy in seconds; DEFAULT_TIME_SIGMA_SEC=1.0 is the
    # find_orb-style value to use); per-obs ADES rmsTime overrides it. Default OFF (0):
    # find_orb applies it always, but here it shifts the σ-normalized residuals that the
    # multi-rev bridge / outlier-rejection thresholds are tuned against, destabilising the
    # hardest linkage objects (e.g. 2023 VM6) — so, like find_orb's continuous reweight,
    # it is opt-in rather than the suite default. When on, it inflates the RAW σ used
    # everywhere (fit weighting AND rejection): the smear is a genuine uncertainty
    # component, not a fit-only down-weight. The effect is sub-0.01" for ordinary
    # asteroids and only grows for fast movers / coarse timestamps, exactly as intended.
    t_sig_default = float(os.environ.get("PYROORB_TIME_SIGMA_SEC", 0))
    if t_sig_default > 0:
        sig_t_days = np.array([(o["sigma_time_sec"] if o.get("sigma_time_sec") else t_sig_default)
                               for o in obs]) / 86400.0
        v_ra, v_dec = apparent_sky_motion([o["jd_utc"] for o in obs], obs_ra, obs_dec,
                                          [o["obscode"] for o in obs])
        sigma_ra, sigma_dec = timing_inflated_sigmas(sigma_ra, sigma_dec, v_ra, v_dec, sig_t_days)
    fit_sigma_ra, fit_sigma_dec = sigma_ra, sigma_dec
    if not os.environ.get("PYROORB_NO_OVEROBS"):
        f = overobserving_factors([o["jd_utc"] for o in obs], [o["obscode"] for o in obs])
        fit_sigma_ra = sigma_ra / f
        fit_sigma_dec = sigma_dec / f
    # Opt-in per obs (``reject_sigma_floor``), applied AFTER the fit sigmas are derived so
    # it raises the REJECTION sigma only and leaves the WEIGHTING on the reported value.
    #
    # A station that self-reports tighter than it can measure turns a good observation
    # into an outlier: 2021 CP63's G96 tracklet reports 0.21"/0.23" against the ~0.5" the
    # station achieves, so an observation 0.72"/0.48" off the orbit reads 4.04 sigma and is
    # rejected -- 1.73 sigma, and kept, against the station model. It was one of THREE obs
    # in its tracklet, so losing it left 67% retention against the 75% floor, which cost the
    # tracklet and with it a correct 2-opposition linkage. The ITF acceptance gates already
    # ask their chi2 questions with station_floor=True (itf.fitter._obs_sigmas); this makes
    # the fit that feeds them judge outliers by the same convention, so the two can no
    # longer disagree about what a sigma means.
    #
    # NOT a duplicate of _relax_overconfident_obscodes, which cures this per OBSCODE and
    # only on a majority signature (> RELAX_REJECT_FRAC of the code\'s obs clipped, or its kept
    # obs dragging the fit). One observation of one tracklet of an otherwise healthy obscode
    # -- CP63\'s G96 is 1 clipped of 11 -- never trips it.
    if any(o.get("reject_sigma_floor") for o in obs):
        from pyroorb.weights import VERES_2017, DEFAULT_SIGMA
        floor = np.array([(VERES_2017.get(o["obscode"], DEFAULT_SIGMA)
                           if o.get("reject_sigma_floor") else 0.0) for o in obs])
        sigma_ra = np.maximum(sigma_ra, floor)
        sigma_dec = np.maximum(sigma_dec, floor)
    times_tdb = Time([o["jd_utc"] for o in obs], format="jd", scale="utc").tdb.jd
    backend, backend_name = _force_backend()
    if is_assist(backend):
        # The perturber choice is made HERE, once, and pinned on the backend every later step
        # of this fit reaches (forcemodel.PinnedForce). Left to the process-wide switch, a fit
        # made inside `with pyroorb.asteroid_perturbers():` re-corrected its LOV or range
        # profile planets-only once the block had closed.
        import pyroorb.force_assist as _fa
        backend = pin(backend, _fa._use_asteroids(None))
    jac_fn = (None if os.environ.get("PYROORB_FINITE_DIFF_JAC")
              else getattr(backend, "predict_radec_jac_batch", None))
    return {"Robs": Robs, "obs_ra": obs_ra, "obs_dec": obs_dec,
            "sigma_ra": sigma_ra, "sigma_dec": sigma_dec,
            "fit_sigma_ra": fit_sigma_ra, "fit_sigma_dec": fit_sigma_dec,
            "times_tdb": times_tdb, "backend": backend, "backend_name": backend_name,
            "jac_fn": jac_fn, "obscodes": [o["obscode"] for o in obs]}


def _build_evaluator_at(geom, obs, epoch_tdb):
    """Build the LM Evaluator + per-fit `ctx` at `epoch_tdb` from the epoch-free `geom`
    (_observer_geometry). Only the perturber base-sim and the Evaluator's osculating epoch
    depend on `epoch_tdb`; everything else is reused verbatim. Each caller then adds
    desig/meta/seed_d/method to the ctx."""
    backend = geom["backend"]
    fit_base = backend.build_base_sim(epoch_tdb)
    ev = Evaluator(fit_base, epoch_tdb, geom["times_tdb"], geom["Robs"],
                   geom["obs_ra"], geom["obs_dec"],
                   sigma_ra=geom["sigma_ra"], sigma_dec=geom["sigma_dec"],
                   fit_sigma_ra=geom["fit_sigma_ra"], fit_sigma_dec=geom["fit_sigma_dec"],
                   predict_fn=backend.predict_radec_batch,
                   predict_jac_fn=geom["jac_fn"], obscodes=geom["obscodes"])
    ctx = {"ev": ev, "obs": obs, "epoch_tdb": epoch_tdb, "times_tdb": geom["times_tdb"],
           "Robs": geom["Robs"], "backend": backend, "fit_base": fit_base,
           "backend_name": geom["backend_name"],
           "sigma_ra": geom["sigma_ra"], "sigma_dec": geom["sigma_dec"]}
    return ev, ctx


def _build_evaluator(obs, obscodes, epoch_tdb):
    """The LM Evaluator + per-fit `ctx` for `obs` at `epoch_tdb`. Thin wrapper over the
    epoch-free geometry build + the at-epoch build, for callers that already know the epoch
    (the single-night rescue, force-model swaps, epoch re-anchor)."""
    return _build_evaluator_at(_observer_geometry(obs, obscodes), obs, epoch_tdb)


SN_RESCUE_REJECT_FRAC = 0.10   # attempt the single-night-opp rescue when the fit clips at least
                               # this fraction of obs OR is rms-elevated (> SOFT_FIT_RMS) -- a clean
                               # fit (low rms AND <10% dropped) never trips it, so the vast majority
                               # pay only O(1) checks. A poisoned fit can hide the mess as high rms
                               # rather than rejections (it KEEPS the bad obs), so rms is checked too.
SN_RESCUE_SINGLE_NIGHT = 0.5   # days; an apparition shorter than this is a single-night opposition
RESCUE_OVERDET_OPP = 5         # >= this many oppositions at a clean rms => over-determined, so a
                               # single stray night cannot be poisoning the orbit: skip the rescue's
                               # expensive re-fit (it would only re-confirm the same orbit). See
                               # _rescue_single_night_poison; saved ~8 s on 2009 HV119.


def _full_fit_excluding(state, epoch_tdb, obs, obscodes, desig, meta, excluded_idx,
                        method="sn-rescue", history=None):
    """Evaluate `state` (an orbit at `epoch_tdb`, found on a REDUCED obs set) against the FULL obs
    set with `excluded_idx` pinned OFF, returning a FittedOrbit over ALL obs. Lets a stage that
    fit on a reduced arc PRESENT its clean orbit as a full-arc fit whose dropped obs are EXCLUDED
    (so they carry an O-C and an 'x', and --ident-check lists them under `except:`) instead of
    silently vanishing from the record. The kept obs are exactly the ones the reduced fit kept, so
    the orbit/rms come out the same -- this only re-attaches the excluded obs to the fit as
    rejections. (Same evaluator build as _fit_from_start via _build_evaluator, seeded with the
    known state rather than from IOD.) Two callers: the single-night poison rescue and the
    drop-suspect-night recovery stage, which pass their own `method` tag and `history`."""
    ev, ctx = _build_evaluator(obs, obscodes, epoch_tdb)
    ctx.update(desig=desig, meta=meta, seed_d=0.0, method=method)
    mask = np.ones(len(obs), dtype=bool)
    for i in excluded_idx:
        mask[i] = False
    state2, result = ev.fit(np.asarray(state, float), mask)
    return _build_orbit(ctx, state2, mask, result, history)


def _expand_history_masks(history, reduced_full, n_full):
    """Re-index a history recorded on a REDUCED obs list onto the FULL one.

    Teaching mode (arc_extend.SNAPSHOTS) tags each history entry with the `_mask` that
    stage held, so the web UI can replay the fit step by step. Those masks are indexed by
    the reduced list; attaching them unchanged to a full-arc fit would replay stages
    against the wrong observations. `reduced_full[j]` is the full index of reduced obs j,
    so each snapshot maps across with the dropped obs pinned off. Entries without a
    `_mask` (snapshots off) pass through untouched."""
    if not history:
        return history
    out = []
    for h in history:
        m = h.get("_mask")
        if m is None or len(m) != len(reduced_full):
            out.append(h)
            continue
        full = [False] * n_full
        for j, keep in enumerate(m):
            if keep:
                full[reduced_full[j]] = True
        h = dict(h)
        h["_mask"] = full
        out.append(h)
    return out


BRIDGE_RESCUE_MAX_RMS = 2.0    # accept a rescue-bridge result only if it stays this clean (") -- a
                               # genuinely-linked far apparition glues to well under an arcsec (the
                               # bridge's own LINK_NEW_TOL is 3"), so a bound here just guards against
                               # a degraded joint fit slipping through.


def _cand_positions(cand, reduced):
    """Position of each obs in `reduced` inside `cand._obs` -- or None where a recovery stage
    inside fit_orbit dropped it from the fit entirely.

    fit_orbit(reduced, ...) does NOT always come back fit on exactly `reduced`: the drop-suspect-
    night stage fits a shorter list and, when _represent_over_full_arc fails to re-present that
    orbit over the full arc, hands the caller the REDUCED fit (fit_object.py:381). cand.is_included()
    is then shorter than `reduced`, so indexing it by position in `reduced` either overruns the mask
    (IndexError out of the rescue) or -- when the drop was interior -- silently reads a DIFFERENT
    observation, which can pass a dominance/exclusion test off the wrong obs. Map by observation
    identity instead; the dicts are passed by reference throughout the fit, and the equal-length
    fast path costs nothing on the overwhelmingly common aligned case."""
    if len(cand._obs) == len(reduced):
        return list(range(len(reduced)))
    pos = {id(o): j for j, o in enumerate(cand._obs)}
    return [pos.get(id(o)) for o in reduced]


def _bridge_rescue_tracklets(cand, reduced_full, obs, obscodes, desig, meta, verbose=False):
    """The single-night rescue has just recovered a CLEAN single-apparition orbit (`cand`, fit on the
    `reduced` obs with the stray tracklets removed). That clean, well-determined anchor is exactly the
    base the multi-revolution bridge needs -- so, before we settle for merely EXCLUDING the stray
    tracklets, try to LINK them: rebuild the evaluator over the FULL obs at cand's epoch, seed it with
    cand's clean orbit over cand's kept obs, and run the apparition linker. For a near-1yr-period
    Earth-resonant NEO the stray 'tracklets' are in fact real precovery/later apparitions a few
    revolutions away whose modest sky offset hid them from the offset-gated router; the bridge's
    period sweep now snaps them in (2024 LW: 33/45 single-apparition -> 44/45 across five apparitions).
    Returns a FittedOrbit that links strictly MORE obs than `cand` at a clean rms, or None (leave the
    rescue to exclude the tracklets as before)."""
    if len(_clusters(np.array([o["jd_utc"] for o in obs]))) < 2:
        return None                                  # nothing to bridge across
    ev, ctx = _build_evaluator(obs, obscodes, cand.epoch_tdb)
    ctx.update(desig=desig, meta=meta, seed_d=0.0, method="sn-bridge")
    _arm_ident_shortcircuit(ev, obs, meta)
    # cand's KEPT obs (mapped reduced->full) are the clean anchor; everything else starts excluded.
    cand_inc = cand.is_included()
    cpos = _cand_positions(cand, [obs[i] for i in reduced_full])
    mask = np.zeros(len(obs), dtype=bool)
    for j, p in enumerate(cpos):
        if p is not None and cand_inc[p]:
            mask[reduced_full[j]] = True
    if int(mask.sum()) < FIT_MIN_OBS:
        return None
    try:
        state, _ = ev.fit(np.asarray(cand._state, float), mask)
        state, mask = _link_oppositions(ev, state, mask, [], verbose)
        if int(mask.sum()) <= int(cand.n_observations()):
            return None                              # linked nothing new -> no gain over cand
        span0 = _included_span(ev, mask)
        for _ in range(4):                           # settle: reject discordant / redeem now-fitting
            if auto_reject_within_arc(ev, mask, state, orig_span=span0) <= 0:
                break
            state, _ = ev.fit(state, mask)
        state, mask = _final_redeem(ev, state, mask, [], verbose)
        state2, result = ev.fit(state, mask, thin=False)
    except Exception:
        return None
    fit = _build_orbit(ctx, state2, mask, result, None)
    if fit.n_observations() > cand.n_observations() and fit.rms_residual() < BRIDGE_RESCUE_MAX_RMS:
        return fit
    return None


def _rescue_single_night_poison(best, obs, obscodes, desig, meta, force, verbose=False):
    """A lone single-night opposition (a stray / mislinked tracklet that is its own apparition)
    can drag the orbit so the reject loop clips GOOD multi-night obs -- the fit ends messy (many
    rejections, high rms) when a clean orbit exists for the rest (e.g. 2026 DV62 + 2024 UB12:
    55/729 @ 0.38" because a 4-obs night poisoned it; without it, 725/725 @ 0.04"). Re-fit FROM
    SCRATCH without the single-night oppositions, and keep that fit only if it RECOVERS obs (used
    does not drop -- so the dropped night wasn't load-bearing, e.g. a bridge stepping-stone like
    1993 TZ's 1993 night) AND cuts rms by >=20%. Gated to messy fits, so the common case is free.
    The kept orbit is then re-presented over the FULL arc with the single-night opps EXCLUDED (not
    dropped), so --ident-check reports them under `except:` -- the honest 'A = B except <stray tracklet>'
    rather than a bare 'A = B' that would wrongly claim the non-fitting obs belong to the object."""
    n_exc = best.n_excluded()
    if n_exc < 2:
        return best                                  # the user's floor: >=2 rejects to bother at all
    # A DEGENERATE MAIN-BELT IDENT fit (two designations forced together -> garbage rms on a q>=1.3
    # body) is a settled non-match, already short-circuited past the recovery cascade in fit_orbit.
    # Re-fitting from scratch without the single-night oppositions only re-confirms the non-fit at the
    # cost of a full recursive fit_orbit -- skip it. Same gate, same function (the anchor q it
    # computed at the fit_orbit short-circuit is memoized on `best`, so this re-check is free).
    # Real single-night poison rides at a FEW arcsec far below IDENT_DEGEN_RMS (2026 DV62's 2.4"),
    # and a LOW-q multi-rev NEO ident (2021 WO) keeps the rescue via the q gate.
    if _ident_doomed_degenerate(best, best._obs, obscodes, meta):
        return best
    # An over-determined orbit cannot be single-night-poisoned: a fit pinned by many
    # oppositions AT A CLEAN RMS is not being "dragged" by one stray night (the rescue's
    # failure mode needs an under-determined orbit one night can swing). The expensive
    # re-fit-from-scratch below could then only re-confirm the orbit it already has -- skip
    # it. The poison cases the rescue exists for are FEW-opposition (e.g. 2026 DV62's 2-
    # apparition link). HV119's 5-opposition 0.32" fit clips 20 Vesta-perturbation misfits
    # (a force-model issue the asteroid polish fixes, NOT poison); the doomed rescue cost ~8s.
    if best.n_oppositions() >= RESCUE_OVERDET_OPP and best.rms_residual() <= SOFT_FIT_RMS:
        return best
    # Work on the SAME obs list `best` was actually fit on, not the caller's pre-fit `obs`: a few
    # rows can be dropped upstream (unpaired satellite/roving companion lines), so the fetched arc
    # can be longer than best._obs. best.is_included() / the reduced->full index map below all align
    # with best._obs; indexing them by the longer arc overruns the mask (602242: fetch 184, fit 181).
    obs = best._obs
    times = np.array([o["jd_utc"] for o in obs])
    sn_idx = set()
    for (cs, ce) in _clusters(times):
        if times[ce] - times[cs] < SN_RESCUE_SINGLE_NIGHT:   # single-night opposition
            sn_idx.update(range(cs, ce + 1))
    if not sn_idx or len(sn_idx) >= len(obs):
        return best
    # Cheap common exit: a CLEAN fit needs no rescue. "Clean" = low rms AND it isn't clipping
    # GOOD (multi-night) obs -- so count only NON-single-night exclusions here. A couple of
    # within-night outliers dropped from a single-night opposition that is otherwise cleanly
    # LINKED is not the poison signature, and must not be confused with it: 2025 MM254 links
    # 2016+2021+2025 at 14/16 @ 0.076", clipping only 2 obs from inside the 2021 night -- the
    # rescue must NOT amputate those real precoveries just because a shorter 2025-only arc
    # trivially fits tighter (the old test counted those 2 within-night drops as 12.5% > 10%
    # and re-presented the 9-obs single-apparition orbit). The poison the rescue exists to undo
    # clips GOOD multi-night obs (2026 DV62: 55/729 @ 0.38") or rides badly (elevated rms, the
    # 727 @ 2.4" case) -- both still trip the test below.
    inc = best.is_included()
    non_sn_excluded = sum(1 for i in range(len(obs)) if not inc[i] and i not in sn_idx)
    if (best.rms_residual() <= SOFT_FIT_RMS
            and non_sn_excluded < SN_RESCUE_REJECT_FRAC * best.n_total_observations()):
        return best                                  # clean (low rms AND few good drops) -> cheap exit
    reduced_full = [i for i in range(len(obs)) if i not in sn_idx]   # reduced[j] == obs[reduced_full[j]]
    reduced = [obs[i] for i in reduced_full]
    try:
        # GRAVITY-ONLY: this is an internal structural re-fit (which nights to drop), not the
        # place to decide a non-grav model. Leaving auto_nongrav default-ON here would (a) leak a
        # 7-element nongrav state out of a fit_orbit(auto_nongrav=False) caller and (b) hand that
        # 7-vector as a seed to _full_fit_excluding's gravity evaluator below -> a (6,)/(7,) shape
        # crash (see 2010 VQ). The single authoritative non-grav decision is made once at the end
        # in _finalize, on the full settled obs set -- so force gravity here regardless of caller.
        # The published-orbit seed that started `best` (if any) is just as valid for the
        # reduced arc -- it is the same body -- and without it this re-fit goes through IOD
        # over the whole arc. On 10P (7839 obs, 52 oppositions, gravity rms 10" because the
        # comet out-gasses) that was 142 IOD-path fits and 844 of the fit's 919 s, for a
        # candidate the rescue then declined. With the seed the same re-fit is ~4 s and the
        # rescue's verdict is unchanged: it still compares the reduced fit on its own terms.
        _seed = (getattr(best, "_ctx", None) or {}).get("mpc_seed")
        cand = fit_orbit(reduced, obscodes, desig=desig, meta=meta, verbose=False,
                         force=force, _allow_sn_drop=False, auto_nongrav=False,
                         mpc_seed=_seed)
    except Exception:
        return best
    # Keep the reduced fit only if (a) dropping the single-night opps did NOT strand any of the
    # REMAINING obs -- cand links essentially all of `reduced`, so those nights weren't a load-
    # bearing bridge (cf. 1993 TZ, whose lone night bridges a 20-yr gap; without it a whole far
    # apparition strands and cand keeps far fewer than len(reduced)) -- AND (b) it cuts rms >=20%.
    # (best.n_observations() is NOT a valid baseline here: a poisoned best KEEPS the bad obs at high
    # rms, so 727 "included" @ 2.4" must lose to 725/725 @ 0.04", which an obs-count test gets wrong.)
    #
    # (a) is satisfied EITHER by that >=90% link rate OR by the DOMINANCE test below, which asks
    # the stranding question directly instead of through a fraction. The 90% form conflates two
    # different reasons cand can fail to link an obs: a genuinely stranded arc (the bridge case the
    # test exists to refuse) and ORDINARY outlier rejection of astrometry the poisoned fit had
    # ALREADY thrown out -- which costs nothing, because that obs is not in the orbit either way.
    # The fraction charges the rescue for both, so a second junk tracklet inside the retained arc
    # can veto a rescue that loses no real data at all: 2015 HP527 = 2025 PS30 clips its 4-obs 2021
    # single-night opposition and then fits 32/36 at 0.09" -- the 4 unlinked obs are a 1-degree-off
    # W84 tracklet `best` had also rejected -- and 32 lost to the 32.4 bar, leaving the object on a
    # 26/40 @ 0.43" orbit whose kept nights trend (an 'y-systematic' ident verdict) when a 32-obs
    # 0.09" orbit existed. Dominance is the honest statement of "nothing was stranded": every obs
    # `best` was keeping (outside the single-night opps we are deliberately setting aside) is still
    # linked by cand, so the rescue is strictly better on BOTH axes -- same-or-more real astrometry,
    # >=20% lower rms -- rather than merely better on one. A bridge case still fails it: 1993 TZ's
    # far apparition IS linked by best and strands in cand, so obs best kept go missing.
    inc_best = best.is_included()
    cand_mask = cand.is_included()
    cpos = _cand_positions(cand, reduced)          # cand may be fit on FEWER obs than `reduced`
    dominates = all(p is not None and cand_mask[p]
                    for p, i in zip(cpos, reduced_full) if inc_best[i])
    if not ((cand.n_observations() >= 0.9 * len(reduced) or dominates)
            and cand.rms_residual() <= 0.8 * best.rms_residual()):
        return best
    # cand's clean single-apparition orbit is precisely the anchor the multi-revolution bridge needs.
    # Before settling for merely EXCLUDING the stray single-night tracklets, try to LINK them onto it
    # -- for a near-1yr-period Earth-resonant NEO those "strays" are real precovery/later apparitions a
    # few revolutions away, whose modest sky offset hid them from the offset-gated router. If the
    # bridge snaps them in at a clean rms (linking strictly more obs), that is the real orbit.
    bridged = _bridge_rescue_tracklets(cand, reduced_full, obs, obscodes, desig, meta, verbose)
    if bridged is not None:
        if verbose:
            print(f"[sn-rescue] bridged single-night tracklets onto the clean anchor: "
                  f"{best.n_observations()}/{best.n_total_observations()}@{best.rms_residual():.2f}\" -> "
                  f"{bridged.n_observations()}/{bridged.n_total_observations()}@{bridged.rms_residual():.2f}\"")
        return bridged
    # Re-present cand's clean orbit over the FULL arc with the single-night opps (plus anything cand
    # itself rejected) pinned OFF, so they surface as `except:` rejections rather than vanishing.
    cand_mask = cand.is_included()
    cpos = _cand_positions(cand, reduced)
    excluded_full = set(sn_idx) | {reduced_full[j] for j, p in enumerate(cpos)
                                   if p is None or not cand_mask[p]}
    try:
        full = _full_fit_excluding(cand._state, cand.epoch_tdb, obs, obscodes, desig, meta, excluded_full)
    except Exception:
        full = None
    out = full if (full is not None and full.rms_residual() <= 0.8 * best.rms_residual()) else cand
    if verbose:
        print(f"[sn-rescue] exclude {len(excluded_full)} single-night-opp obs: "
              f"{best.n_observations()}/{best.n_total_observations()}@{best.rms_residual():.2f}\" -> "
              f"{out.n_observations()}/{out.n_total_observations()}@{out.rms_residual():.2f}\"")
    return out


# --- well-observed-object fast path: seed from the published MPC orbit ------------
# An object with hundreds of observations across many oppositions already has a published
# MPC orbit far better than any IOD triangle we could derive, and the IOD/subarc/extend
# cascade on such a long, dense arc is the slowest path through the engine. For these we
# seed the differential corrector straight from the MPC orbits API and run only the normal
# drop/reclaim rejection -- a much snappier fit at the same answer. Thresholds are
# deliberately conservative (a clear "this is a mature, well-determined object"); anything
# below them, or any multi-designation linkage request, keeps the full IOD path. The fast path
# degrades gracefully: any failure (no published orbit, network/SSL error, or a non-finite
# seed) falls back to IOD.
MPC_SEED_MIN_OBS = 500     # > this many observations
MPC_SEED_MIN_OPP = 5       # AND > this many oppositions => use the published-orbit seed
MPC_SEED_REJECT_ROUNDS = 6 # drop/reclaim rounds after the initial full-arc correction


def _seeded_midarc_evaluator(obs, obscodes, seed, seed_epoch, nongrav=None):
    """Shared setup for a SEEDED fast path (MPC-orbit seed, ident attribution): build the
    observer geometry and evaluator anchored at the MIDDLE of the observed arc (find_orb's
    "epoch within the arc" -- a published/anchor epoch sits at one end of a long arc, so fitting
    there integrates the full span on every sweep; the midpoint halves the worst-case integration
    distance), propagating `seed` from `seed_epoch` to the mid-arc epoch under the full force
    model (exact for a converged orbit -- only the osculating epoch moves). Returns
    (ev, ctx, seed_at_mid, mid_epoch), or None if the propagated seed goes non-finite (the
    caller falls back to its normal path). Time is accounted under the seed/IOD stage.

    `nongrav` is the (A, gr) pair the seed orbit was DETERMINED under, when it carries one:
    the propagation to mid-arc must run under the same force law, or the seed is a
    different orbit by the time it arrives. A published comet orbit's A1/A2 dragged
    gravity-only across half a 150-year arc landed 10P's seed 192" off its own
    observations; under its own out-gassing it lands within the arcsecond the MPC fit it
    to. None (the default) is the gravity-only propagation every earlier caller took."""
    _ts = _stime.perf_counter()
    geom = _observer_geometry(obs, obscodes)
    backend = geom["backend"]
    times_tdb = np.asarray(geom["times_tdb"], float)
    epoch_tdb = 0.5 * (float(times_tdb.min()) + float(times_tdb.max()))
    seed = np.asarray(seed, float)
    if abs(epoch_tdb - seed_epoch) > 1e-9:
        base = backend.build_base_sim(seed_epoch)
        if nongrav is not None:
            seed = np.asarray(backend.predict_state(base, seed_epoch, seed, epoch_tdb,
                                                    nongrav=nongrav), float)
        else:
            seed = np.asarray(backend.predict_state(base, seed_epoch, seed, epoch_tdb), float)
        if not np.all(np.isfinite(seed)):
            _stadd("iod", _ts)
            return None
    ev, ctx = _build_evaluator_at(geom, obs, epoch_tdb)
    ctx["geom"] = geom          # lets a later re-anchor (_reanchor_apparition) skip the rebuild
    _stadd("iod", _ts)                              # accounted under the seed/IOD stage
    return ev, ctx, seed, epoch_tdb


def _reject_settle(ev, state, mask, result, history, orig_n=None):
    """The standard post-seed drop/reclaim rounds shared by EVERY seeded fast path (MPC seed,
    planet seed, joint refit, ident attribution): reject discordant obs and redeem clean
    ones (auto_reject_within_arc does both, with hysteresis), refitting between rounds until the
    mask stops toggling -- find_orb's reject rounds, minus the extension. Mutates `mask` in place,
    appends one 'reject' history snapshot per round, and returns the settled (state, result).
    This is the ONE copy of that loop -- a change to the settle choreography belongs here, not in
    a per-path clone.

    Bounded in ROUNDS ONLY, never in wall-clock time. A time bound here was tried and
    reverted: this module is fit core, where elapsed time may be measured but never
    COMPARED (test_no_wallclock_branch), because a branch on the clock lets
    machine load choose the numerical path -- the exact defect that made 490581 land at
    0.249" on an idle box and 625" under four concurrent workers. A deadline that aborts
    and says so is a legitimate pattern, but it belongs OUTSIDE the fit core."""
    span0 = _included_span(ev, mask)
    for _ in range(MPC_SEED_REJECT_ROUNDS):
        if auto_reject_within_arc(ev, mask, state, orig_span=span0, orig_n=orig_n) <= 0:
            break
        state, result = ev.fit(state, mask)
        history.append(_hsnap({"stage": "reject", "n_obs": int(mask.sum()),
                               "rms": ev.rms_arcsec(state, mask),
                               "score": ev.rms_norm(state, mask)}, state, mask))
    return state, result


def _mpc_seed_enabled():
    return os.environ.get("PYROORB_MPC_SEED", "1") != "0"


def _mpc_comet_seed_enabled():
    """Whether a published comet orbit's A1/A2/A3 govern the seed's re-anchor to mid-arc.
    Off = the pre-2026-09-05 gravity-only propagation, for A/B."""
    return os.environ.get("PYROORB_MPC_COMET_SEED", "1") != "0"


def _qualifies_for_mpc_seed(obs, meta):
    """True iff this object is dense and long enough to seed from the published MPC orbit
    instead of IOD (see MPC_SEED_MIN_OBS / _MIN_OPP).

    Gate on how many designations the CALLER REQUESTED, not on `combined`: a numbered object
    queried by its number expands to all its historical aliases (combined=True) yet is one
    physical body with one published orbit -- exactly what we want to fast-path. Only a
    genuine multi-designation LINKAGE hypothesis (n_requested > 1) must keep the IOD path,
    since no single published orbit covers two separately-catalogued candidates. When the
    requested count is unknown (a meta from a path that predates the field), fall back to the
    conservative `combined` test."""
    if not _mpc_seed_enabled():
        return False
    n_req = (meta or {}).get("n_requested")
    if n_req is None:
        n_req = 2 if (meta and meta.get("combined")) else 1
    if n_req > 1:
        return False
    if len(obs) <= MPC_SEED_MIN_OBS:
        return False
    return n_oppositions([o["jd_utc"] for o in obs]) > MPC_SEED_MIN_OPP


def fetch_mpc_seed(obs, meta, verbose=False):
    """If `obs`/`meta` qualify for the published-orbit fast path, fetch the MPC orbit and
    return its seed dict (see mpc_orbit.fetch_mpc_orbit); else None.

    This is the ONLY network touch of the fast path, and it lives OUTSIDE fit_orbit on
    purpose: the fit itself stays deterministic and offline-testable (the regression suite
    re-fits frozen fixtures through the IOD path), while the production fetch/CLI/web layer
    calls this and passes the result to fit_orbit(mpc_seed=...). Never raises -- a network/
    SSL error or a missing orbit just returns None and the fit falls back to IOD."""
    if not _qualifies_for_mpc_seed(obs, meta):
        return None
    query = (meta or {}).get("designation")
    if not query:
        return None
    try:
        seed = fetch_mpc_orbit(query, cache=datadirs.cache_dir() / f"mpcorb_{cache_stem(query)}.json")
    except Exception as ex:
        if verbose:
            print(f"  MPC-seed fetch failed ({type(ex).__name__}: {ex}); using IOD")
        return None
    if seed is None and verbose:
        print(f"  MPC-seed: no published orbit for {query!r}; using IOD")
    return seed


def _fit_from_mpc_seed(obs, obscodes, desig, meta, verbose, seed_info):
    """Fast path for a well-observed multi-opposition object: seed the orbit from the
    published MPC orbit (`seed_info` from fetch_mpc_seed) rather than the IOD/subarc/extend
    cascade, then run the same drop/reclaim rejection the normal path uses. Returns a
    FittedOrbit, or None if the supplied seed is unusable (the caller falls back to IOD)."""
    if seed_info is None:
        return None
    seed = np.asarray(seed_info["state_eq"], float)
    mpc_epoch = float(seed_info["epoch_tdb"])
    if not np.all(np.isfinite(seed)):
        return None
    # A published COMET orbit carries its out-gassing A1/A2(/A3) beside the state, and the
    # state only means what it means under that force: dragged gravity-only from the published
    # epoch to mid-arc, 10P's seed arrived 192" off its own observations, 128" under its own
    # out-gassing (see _seeded_midarc_evaluator). So the re-anchor runs under the published
    # model. The A's go NO further: warm-starting discover_comet_nongrav's model sequence from them
    # was built and measured (2026-09-05, 10P) -- the warm LM never beat the cold one (the
    # MPC's A1 is 3.5x the engine's on this arc) and its extra LM per step cost 4-7 s -- so
    # the gravity fit below and the engine's own model selection stay exactly what they were.
    # PYROORB_MPC_COMET_SEED=0 re-anchors gravity-only (the pre-2026-09-05 seed).
    comet_A = seed_info.get("comet_nongrav") if _mpc_comet_seed_enabled() else None
    nong = None
    if comet_A:
        import pyroorb.nongrav as _ng
        A = [float(comet_A.get(ax, 0.0)) for ax in _ng.A_AXES]
        if all(np.isfinite(A)) and any(A):
            nong = (A, _ng.COMET_GR)
    sm = _seeded_midarc_evaluator(obs, obscodes, seed, mpc_epoch, nongrav=nong)
    if sm is None:
        return None
    ev, ctx, seed, epoch_tdb = sm
    # The seed rides on the ctx so a RE-FIT of this same arc can take the fast path again:
    # _rescue_single_night_poison's from-scratch re-fit used to re-derive 10P's 150-year,
    # 7839-obs arc through IOD (142 fits, 844 of the fit's 919 s) when the published orbit
    # that seeded the outer fit was equally valid for the reduced one.
    ctx["mpc_seed"] = seed_info
    mask = np.ones(len(obs), dtype=bool)            # start with the whole arc included
    if verbose:
        print(f"MPC-seed: published orbit re-anchored to mid-arc epoch JD {epoch_tdb:.1f}, "
              f"seed rms {ev.rms_arcsec(seed, mask):.2f}\" over {len(obs)} obs")
    history = [_hsnap({"stage": "mpc-seed", "n_obs": int(mask.sum()),
                       "rms": ev.rms_arcsec(seed, mask),
                       "score": ev.rms_norm(seed, mask)}, seed, mask)]
    # Initial differential correction on the full arc, then the shared drop/reclaim settle.
    state, result = ev.fit(seed, mask)
    history.append(_hsnap({"stage": "fit", "n_obs": int(mask.sum()),
                           "rms": ev.rms_arcsec(state, mask),
                           "score": ev.rms_norm(state, mask)}, state, mask))
    state, result = _reject_settle(ev, state, mask, result, history)
    ctx.update(desig=desig, meta=meta, seed_d=0.0, method="mpc")
    return _build_orbit(ctx, state, mask, result, history)


def fit_under_known_nongrav(obs, obscodes, ref_fit, desig="object", meta=None):
    """Fit ``obs`` under the non-gravitational force ``ref_fit`` already measured, held
    FIXED, warm-started from ``ref_fit``'s orbit: re-anchored to this arc's mid-epoch under
    that force, a 6-state LM and the standard reject/settle (:func:`_reject_settle`) -- no
    IOD, no gravity-only pass, no model discovery.

    For a caller that has established the model on a longer arc and now asks only whether a
    few more observations belong (the ITF search after a Yarkovsky rescue). A full fit
    re-derives both: its gravity pass cannot hold the drift and grinds through the rescue
    paths, then the discovery re-decides the model -- 2007 DB61's trial fits spent ~49 s
    and ~43 s of 93 s there.  The coefficients stay ``ref_fit``'s; the returned orbit
    carries them (``nongrav_model`` / ``nongrav_params``) so its ephemerides keep the force.
    Returns None when ``ref_fit`` has no nongrav model or the re-anchored seed is unusable."""
    from pyroorb.ephemerides import _nongrav_pair
    pair = _nongrav_pair(ref_fit)
    rctx = getattr(ref_fit, "_ctx", None)
    if pair is None or rctx is None:
        return None
    sm = _seeded_midarc_evaluator(obs, obscodes, np.asarray(ref_fit._state, float)[:6],
                                  float(rctx["epoch_tdb"]), nongrav=pair)
    if sm is None:
        return None
    ev, ctx, seed, _epoch = sm
    # Every residual AND Jacobian under the fixed force: Evaluator.fit predicts through
    # these two (the 6-column Jacobian is predict_radec_jac_ng with no free axis), the
    # reject/settle and the delivered residuals through _predict as well.
    import functools
    backend = ctx["backend"]
    A, gr = pair
    ev._predict = functools.partial(backend.predict_radec_batch, nongrav=pair)
    if ev._predict_jac is not None:
        ev._predict_jac = (lambda _base, t0, s, tt, Rs:
                           backend.predict_radec_jac_ng(t0, s, A, [], gr, tt, Rs))
    mask = np.ones(len(obs), dtype=bool)
    history = []
    state, result = ev.fit(seed, mask)
    state, result = _reject_settle(ev, state, mask, result, history)
    ctx.update(desig=desig, meta=meta, seed_d=0.0, method="known-nongrav")
    f = _build_orbit(ctx, state, mask, result, history)
    f.nongrav_model = ref_fit.nongrav_model
    f.nongrav_params = dict(ref_fit.nongrav_params)
    f.nongrav_sigmas = dict(getattr(ref_fit, "nongrav_sigmas", None) or {})
    return f


def _fit_from_planet_seed(obs, obscodes, desig, meta, verbose, center):
    """Planetocentric 'told-the-center' fit. Seed from the planet-hinted IOD
    (planetocentric.planet_hinted_seed -- topocentric range pinned to the planet's), anchor at the
    mid-arc epoch, then run the SAME LM + drop/reclaim rejection as _fit_from_mpc_seed. The
    heliocentric N-body fit is unchanged; the converged state is reported planetocentrically by
    planetocentric.report(). Returns a FittedOrbit."""
    import pyroorb.planetocentric as _pc
    planet = _pc.resolve_planet(center)
    _ts = _stime.perf_counter()
    geom = _observer_geometry(obs, obscodes)
    backend = geom["backend"]
    # Anchor the osculating epoch at the middle of the arc (find_orb's "epoch within the arc").
    times_tdb = np.asarray(geom["times_tdb"], float)
    epoch_tdb = 0.5 * (float(times_tdb.min()) + float(times_tdb.max()))
    ev, ctx = _build_evaluator_at(geom, obs, epoch_tdb)
    mask = np.ones(len(obs), dtype=bool)                # start with the whole arc included
    # MULTI-START over the planetocentric SPEED (as a multiple of the local circular speed). This
    # is the basin-determining knob -- angles-only cannot pin the speed, and the wrong speed drops
    # the LM into a spurious high-eccentricity minimum -- so sweep it and keep the best-fitting
    # orbit (find_orb-style: search the under-determined quantity, don't trust one seed). The raw
    # secant speed (scale=None) is included as one more start.
    speed_scales = [None, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4]
    best = None
    for ss in speed_scales:
        try:
            seed, seed_epoch = _pc.planet_hinted_seed(obs, geom, planet, speed_scale=ss)
            if abs(seed_epoch - epoch_tdb) > 1e-9:
                base = backend.build_base_sim(seed_epoch)
                seed = np.asarray(backend.predict_state(base, seed_epoch, seed, epoch_tdb), float)
            if not np.all(np.isfinite(seed)):
                continue
            st, cand_result = ev.fit(seed, mask)
            r = ev.rms_arcsec(st, mask)
            if np.isfinite(r) and (best is None or r < best[0]):
                best = (r, st, cand_result, ss)
        except Exception as ex:
            if verbose:
                print(f"  planet-seed speed x{ss} failed: {type(ex).__name__}: {ex}")
    if best is None:
        raise RuntimeError("planet-hinted seed produced no viable fit")
    _stadd("iod", _ts)                                  # accounted under the seed/IOD stage
    state, result = best[1], best[2]
    if verbose:
        print(f"planet-seed ({planet}): mid-arc epoch JD {epoch_tdb:.1f}, best of "
              f"{len(speed_scales)} speed starts -> fit rms {best[0]:.3f}\" (speed x{best[3]}) "
              f"over {len(obs)} obs")
    history = [_hsnap({"stage": "planet-seed", "n_obs": int(mask.sum()),
                       "rms": best[0], "score": ev.rms_norm(state, mask)}, state, mask)]
    state, result = _reject_settle(ev, state, mask, result, history)
    ctx.update(desig=desig, meta=meta, seed_d=0.0, method="planet:" + planet)
    return _build_orbit(ctx, state, mask, result, history)


def _arm_ident_shortcircuit(ev, obs, meta):
    """Attach the per-obs designation + per-designation opposition count the multi-rev linker uses
    to short-circuit a doomed cross-designation bridge (arc_extend._ident_shortcircuit). Only armed
    for an EXPLICIT multi-designation linkage request (n_requested > 1) -- an ident-check hypothesis
    like "is K14EK9X = K21Q75W?". A numbered object queried by its number expands to many alias
    designations for ONE body (combined=True) but n_requested == 1, so it is NOT armed and links
    exactly as before. Mirrors the n_requested gate in _qualifies_for_mpc_seed; when the field is
    absent (a meta from an older fetch path) the gate stays off (safe -- no speedup, no change)."""
    if not ident_requested(meta):
        return                                   # single requested object -> gate off, byte-identical
    desigs = [o.get("designation") for o in obs]
    by_desig = {}
    for d, o in zip(desigs, obs):
        by_desig.setdefault(d, []).append(o["jd_utc"])
    ev.obs_desig = desigs                         # aligned to ev/obs index order
    ev.desig_opp = {d: n_oppositions(ts) for d, ts in by_desig.items()}
    ev.ident_mode = True


def _ident_nonmatch(fit, obs, meta):
    """True when a multi-designation linkage request has SETTLED as a non-match: the fit linked one
    independently period-known designation (>= IDENT_MULTI_OPP oppositions) and left a DIFFERENT
    period-known designation fully unlinked, with at least one of the two period-LOCKED (>=
    IDENT_LOCK_OPP). This mirrors arc_extend._ident_shortcircuit at the whole-fit level (it is
    symmetric in which side ends up the anchor -- the linker may anchor on either of the two mature
    arcs): the linker already refused those cross-links one apparition at a time, so re-anchoring
    (the recovery cascade) can only reach the same verdict and is wasted. Conservative: only for an
    explicit n_requested > 1 request, needs period-known designations on BOTH sides with one locked,
    and leaves single-opposition orphans (a possible genuine attribution) to the normal recovery.

    NOTE this gate keeps the STRICT IDENT_LOCK_OPP (4), NOT the offset-gated 3 the per-apparition
    _ident_shortcircuit uses: it decides whether to skip the WHOLE recovery cascade, and that cascade
    is what re-anchors a fit that first settled on the WRONG arc onto the correct body. Firing it at 3
    on a 3-vs-2-opp non-match froze the poor initial anchor (K14EE7R+K21S69J: anchored on the foreign
    SJ69 arc, orphaning the real ER147+ND242 body). The per-apparition guard already makes the
    cascade's bridge passes cheap, so the cascade runs fast AND reaches the correct anchor."""
    from pyroorb.arc_extend import IDENT_MULTI_OPP, IDENT_LOCK_OPP
    if not ident_requested(meta):
        return False
    by_desig = {}
    for o in obs:
        by_desig.setdefault(o.get("designation"), []).append(o["jd_utc"])
    opp = {d: n_oppositions(ts) for d, ts in by_desig.items()}
    mask = np.asarray(fit._mask, dtype=bool)
    linked = {obs[i].get("designation") for i in range(len(obs)) if mask[i]}
    orphan = {obs[i].get("designation") for i in range(len(obs)) if not mask[i]} - linked
    linked_known = {d for d in linked if opp.get(d, 0) >= IDENT_MULTI_OPP}
    orphan_known = {d for d in orphan if opp.get(d, 0) >= IDENT_MULTI_OPP}
    if not linked_known or not orphan_known:
        return False                                  # a side has no period-determined designation
    return max(opp[d] for d in (linked_known | orphan_known)) >= IDENT_LOCK_OPP


DOOM_MIN_OPP = 4        # a 4+-opposition orbit is, in find_orb practice, essentially never a
                        # mislinkage (a 3-opposition one can occasionally be constructed), so we
                        # trust it enough to permanently BAN a leftover apparition it cannot absorb.
DOOM_SOLID_RMS = 0.5    # ...and only when that orbit is a CLEAN fit (arcsec). A noisier 4+-opp fit
                        # is left to the normal recovery path -- still linked-what-it-can, but not
                        # authorised to hard-ban (the leftover stays reconsiderable). Chosen tight
                        # (well under the 0.6-1.0" "suspect" bars) so only an unambiguously-good
                        # orbit earns the permanent ban.


def _doomed_orphan(fit, obs):
    """True when `fit` is a SOLID 4+-opposition orbit that simply cannot absorb a leftover
    apparition -- so that apparition is DOOMED (a near-certain non-match the linker already
    failed to glue), and the recovery cascade would only grind the same doomed multi-rev bridge
    from every re-anchor. This is the general sibling of _ident_nonmatch: it trusts the orbit's
    own 4+-opposition maturity (rather than an explicit multi-designation request) as the proof
    the linker's "can't attach" verdict is settled.

    ALL must hold:
      * >= DOOM_MIN_OPP oppositions among the INCLUDED obs -- a mature, virtually-never-mislinked
        orbit;
      * raw rms < DOOM_SOLID_RMS -- the orbit is a clean fit, not a marginal one;
      * an actual SUBSTANTIAL multi-night apparition is stranded (_has_unlinked_opposition) --
        else there is nothing to ban and no recovery to skip;
      * NO discordant INCLUDED apparition (_has_discordant_opposition) -- a wrong-PERIOD link
        hiding inside the linked set is exactly what re-anchoring fixes, so that trouble must
        NOT be short-circuited here;
      * no systematic within-apparition trend (ident_check.systematic_misfit) -- a coherent trend
        would mean the "solid" fit is itself a possible mislink, so we don't trust it to ban.

    On success the caller returns the fit as-is (the doomed apparition already excluded), never
    re-adding it. Deliberately independent of ident_mode: the reported case is a combined/ident
    request, but the principle -- don't grind a doomed apparition against a solid 4+-opp orbit --
    is general."""
    if fit.n_oppositions() < DOOM_MIN_OPP:
        return False
    if not (fit.rms_residual() < DOOM_SOLID_RMS):     # NaN-safe: an undefined rms is not "solid"
        return False
    if not _has_unlinked_opposition(fit, obs):
        return False
    if _has_discordant_opposition(fit, obs):
        return False
    from pyroorb import ident_check
    return ident_check.systematic_misfit(fit) is None


IDENT_DEGEN_RMS = 100.0  # arcsec. A DEGENERATE joint fit whose INCLUDED obs fit no better than this
                         # is two different objects forced into one orbit -- the LM landed in a garbage
                         # minimum reconciling incompatible arcs (the observed doomed non-matches sit at
                         # 680-38000"). It is NOT a rescuable state: a genuine pair that merely picked the
                         # wrong anchor keeps its anchor apparition fitting sub-arcsec (the other arc
                         # EXCLUDED, not inflating the rms), and real single-night poison rides at a few
                         # arcsec (2026 DV62's poisoned 2.4"). Chosen ~150x above the 0.6" ident bar and
                         # ~7x below the smallest observed doomed rms, so only unambiguous garbage trips it.
_DEGEN_Q_UNSET = object()  # memo sentinel for fit._degen_anchor_q (None is a valid computed value)


def _degenerate_anchor_q(obs, obscodes):
    """A RELIABLE perihelion q for a degenerate joint fit, whose own orbit q is meaningless. Re-fit
    the BEST-DETERMINED designation (most oppositions) on its OWN obs -- one physical body, so its
    base fit is well-posed -- and return that q, or None if it can't be determined. Cheap for the
    main-belt case (its apparitions link at Tier-1); only ever called on the rare rms>=IDENT_DEGEN_RMS
    path. Single-apparition IOD q is range-degenerate and unusable, which is why we fit the whole
    best designation, not one apparition.

    RELIES on there being a SEPARATE well-observed designation to re-fit: this is an independent q
    only in an explicit ident hypothesis (>= 2 requested designations). With a lone designation the
    "best" obs ARE the whole degenerate arc, so this would just reproduce the garbage fit's own q --
    which is why the callers gate on n_requested>1 (see _ident_doomed_degenerate)."""
    by = {}
    for o in obs:
        by.setdefault(o.get("designation"), []).append(o)
    if not by:
        return None
    best = max(by.values(), key=lambda os_: n_oppositions([o["jd_utc"] for o in os_]))
    try:
        f = _fit_from_start([dict(o) for o in best], obscodes, "anchor",
                            {"designation": "anchor", "n_requested": 1}, False)
        q, _ = f.perihelion_distance()
        return float(q) if np.isfinite(q) else None
    except Exception:
        return None


def _ident_doomed_degenerate(fit, obs, obscodes, meta):
    """True when an explicit IDENT hypothesis produced a DEGENERATE joint fit (INCLUDED obs fit
    >= IDENT_DEGEN_RMS -- the LM forced two designations' arcs together into a garbage minimum) AND
    the object is MAIN-BELT. Like _ident_nonmatch / _doomed_orphan this is a SETTLED non-match:
    re-anchoring / night-dropping / short-seeding (the recovery cascade) only re-derive the same
    non-fit at the cost of dozens of full re-fits.

    TWO conditions besides the rms, and BOTH are load-bearing:
      * ident hypothesis (n_requested > 1). The MAIN-BELT test below reads its q from
        _degenerate_anchor_q, which re-fits the best-observed designation ALONE -- a genuinely
        independent orbit ONLY when a SEPARATE designation exists. On a LONE designation the "best"
        obs are the whole degenerate arc, so that q just reproduces the garbage fit and is NOT a
        reliable discriminator; and a poisoned SINGLE-object garbage fit CAN be cascade-rescued (the
        grind-necessity investigation: a wrong-revolution long anchor short-seed-rescued). So we only
        fire under an explicit cross-designation request, where a clean single-object fallback exists.
      * MAIN-BELT (q >= IDENT_FAR_MIN_Q). The cascade rescues LOW-q (q < IDENT_FAR_MIN_Q) multi-
        revolution NEO idents whose long anchor wrapped a revolution (2021 WO: 48/83 @ 412" -> 83/83
        @ 0.43"), so we must NOT skip it for them. A grind study found 0/40 q>=1.3 objects rescued.

    Conservative on an undetermined q: no q => not confirmed main-belt => do NOT fire (keep the
    cascade). The anchor q is MEMOIZED on `fit`: the gate is consulted twice on the settled-non-
    match path (the fit_orbit short-circuit, then again in _rescue_single_night_poison via
    _finalize) and _degenerate_anchor_q is a full solo re-fit -- the second check must not pay it
    again. See IDENT_DEGEN_RMS / _degenerate_anchor_q."""
    if not ident_requested(meta):
        return False                                  # not an ident hypothesis -> no independent q
    if fit.rms_residual() < IDENT_DEGEN_RMS:
        return False
    q = getattr(fit, "_degen_anchor_q", _DEGEN_Q_UNSET)
    if q is _DEGEN_Q_UNSET:
        q = _degenerate_anchor_q(obs, obscodes)
        fit._degen_anchor_q = q                       # memo (q may legitimately be None)
    return bool(q is not None and q >= IDENT_FAR_MIN_Q)


IDENT_ATTR_MIN_OPP = 5          # a requested designation whose OWN obs span >= this many oppositions
                                # is independently well-determined -- authoritative enough to ATTRIBUTE
                                # the other designation(s) to, rather than grind a joint IOD/bridge over
                                # the combined arc. Matches RESCUE_OVERDET_OPP (a 5+-opp orbit is, in
                                # find_orb practice, precise: low U, sub-arcsec, near-never a mislink).
IDENT_ATTR_KEEP_ARCSEC = 300.0  # keep an ob if it fits the anchor orbit within this. A genuine same-
                                # body apparition fits to sub-arcsec even back-extrapolated decades
                                # (2014 EF94's 1999 obs fit the 10-opp 2025 UO106 orbit at 0.19"), while
                                # a non-match sits degrees off (2016 UK176 vs 2008 VC46: ~31700"). The
                                # gate only has to clip a GROSS non-match; the normal reject/redeem
                                # rounds below then fine-tune the survivors to sub-arcsec.


def _attr_fully_clipped(obs, mask):
    """First designation in `obs` with ZERO included obs under `mask`, or None if every
    designation kept at least one. The attribution fast path's 'non-match' verdict is exactly
    this state (a whole requested designation excluded), so this is what the low-q diligence
    fallback keys on. Pure -- unit-testable without a fit."""
    kept = {}
    for i, o in enumerate(obs):
        d = o.get("designation")
        kept[d] = kept.get(d, False) or bool(mask[i])
    for d, any_kept in kept.items():
        if not any_kept:
            return d
    return None


def _attr_low_q_inconclusive(q_anchor, obs, mask):
    """The attribution LOW-q DILIGENCE FALLBACK decision, pure so tests pin every arm: returns
    the clipped designation when the attribution outcome must be treated as INCONCLUSIVE (a whole
    requested designation excluded AND the anchor is NOT confirmed main-belt/outer -- q below
    IDENT_FAR_MIN_Q or undetermined, the conservative reading), else None (the fast verdict
    stands). See the caller for the science; the asymmetry is deliberate: clipping is only
    trustworthy where the anchor's back-prediction is (q >= 1.3), while ATTACHING is its own
    proof at any q (the foreign obs actually fit)."""
    clipped = _attr_fully_clipped(obs, mask)
    if clipped is None:
        return None                                  # every designation kept obs -> attached
    if np.isfinite(q_anchor) and q_anchor >= IDENT_FAR_MIN_Q:
        return None                                  # main-belt/outer anchor -> the clip IS the verdict
    return clipped


def _ident_attribute_to_anchor(obs, obscodes, desig, meta, verbose=False):
    """Ident-mode ATTRIBUTION fast path (find_orb-style identification). When an explicit ident
    hypothesis (n_requested > 1) contains a designation whose OWN obs are independently WELL-DETERMINED
    (>= IDENT_ATTR_MIN_OPP oppositions), that orbit is authoritative: fit it ALONE, then KEEP whichever
    of the other obs fit it (within IDENT_ATTR_KEEP_ARCSEC) and exclude the rest. The verdict is a
    residual comparison against a precise orbit -- not a joint IOD + multi-revolution bridge grind over
    the combined arc, which lands in a garbage minimum and burns dozens of full re-fits when the two
    bodies are different but dynamically similar.

    A genuine ident attaches the other arc (2014 EF94 = 1999 TC333 = 2025 UO106: all 111 obs fit the
    10-opposition 2025 UO106 orbit -> 111/111 @ 0.31"). A non-match's foreign arc sits degrees off and
    is excluded (2016 UK176 doesn't fit 2008 VC46's 8-opposition orbit -- ~8.8 deg away -> the UK176 obs
    drop; ~0.2 s instead of a ~52 s combined-arc grind). Returns a FittedOrbit, or None when no member
    qualifies or the anchor fit is unusable (the caller falls back to the normal IOD path). Never fires
    outside ident_mode or under --force (which asks to grind the hypothesis regardless).

    The clip-a-whole-designation verdict is trusted only for a MAIN-BELT/OUTER anchor
    (q >= IDENT_FAR_MIN_Q, the shared ident-policy cut): a low-q anchor's back-prediction can miss a
    GENUINE match by more than IDENT_ATTR_KEEP_ARCSEC (multi-revolution phase ambiguity, planetary-
    encounter divergence), so there a fully-clipped foreign side returns None -- inconclusive -- and the
    normal path grinds with full Tier-2 diligence (see the LOW-q DILIGENCE FALLBACK below)."""
    if not ident_requested(meta):
        return None
    by = {}
    for i, o in enumerate(obs):
        by.setdefault(o.get("designation"), []).append(i)
    if len(by) < 2:
        return None                                  # only one designation present -> nothing to attribute
    oppc = {d: n_oppositions([obs[i]["jd_utc"] for i in ix]) for d, ix in by.items()}
    well = max(oppc, key=oppc.get)
    if oppc[well] < IDENT_ATTR_MIN_OPP:
        return None                                  # no independently well-determined anchor member
    # 1) Fit the well-determined member ALONE -- gravity-only (the single non-grav decision is made
    #    later in _finalize on the settled arc). Its converged orbit is the attribution reference.
    solo = [obs[i] for i in sorted(by[well])]
    try:
        f_anchor = fit_orbit(solo, obscodes, desig=well,
                             meta={"designation": well, "n_requested": 1},
                             verbose=False, auto_nongrav=False)
    except Exception:
        return None
    if not (f_anchor.rms_residual() < SOFT_FIT_RMS
            and f_anchor.n_oppositions() >= IDENT_ATTR_MIN_OPP):
        return None                                  # anchor didn't settle cleanly -> normal path
    q_anchor, _ = f_anchor.perihelion_distance()     # for the low-q diligence fallback below
    # 2) Build a combined evaluator seeded at the anchor orbit, re-anchored to the combined mid-arc
    #    (_seeded_midarc_evaluator -- "epoch within the arc" over a possibly multi-decade span).
    sm = _seeded_midarc_evaluator(obs, obscodes, f_anchor._state, float(f_anchor.epoch_tdb))
    if sm is None:
        return None
    ev, ctx, seed, _mid = sm
    # 3) ATTRIBUTE: keep obs within IDENT_ATTR_KEEP_ARCSEC of the anchor orbit, exclude the rest.
    dra, ddec = ev.resid_arcsec(seed)
    r = np.hypot(np.asarray(dra, float), np.asarray(ddec, float))
    mask = r < IDENT_ATTR_KEEP_ARCSEC
    if int(mask.sum()) < FIT_MIN_OBS:
        return None                                  # nothing coherent to fit against the anchor
    # 4) Refit on the kept set, then the usual drop/reclaim rounds fine-tune the survivors.
    history = [_hsnap({"stage": "ident-attr", "n_obs": int(mask.sum()),
                       "rms": ev.rms_arcsec(seed, mask), "score": ev.rms_norm(seed, mask)}, seed, mask)]
    state, result = ev.fit(seed, mask)
    history.append(_hsnap({"stage": "fit", "n_obs": int(mask.sum()),
                           "rms": ev.rms_arcsec(state, mask), "score": ev.rms_norm(state, mask)}, state, mask))
    # Distrust over-confident obscodes BEFORE rejecting (find_orb sigma.txt) -- this evaluator is
    # fresh, so it must re-apply the relaxation attempt_extensions runs, else a station reporting
    # over-tight sigmas (e.g. D29 at ~0.1-0.2" against the 0.5" default) clips its own good
    # sub-arcsec obs here as 4-5σ outliers. A real blunder, large in arcsec, still rejects below.
    if _relax_overconfident_obscodes(ev, state, mask):
        state, result = ev.fit(state, mask)
    state, result = _reject_settle(ev, state, mask, result, history)
    # AFTER rejecting: a small over-confident night can slip past the pre-reject relax (its median
    # sigma-residual sits just under the trigger) yet lose an ob or two to the rounds above. Now the
    # clipped obs ARE on the excluded side, so re-run the relax (EXCLUDED-side, small-raw) and REDEEM the
    # obs it reclaims -- exactly attempt_extensions' end-of-reject relax+redeem, which the fresh
    # attribution evaluator skips (2013 CK176 = 2026 DO7: the D29 night is reclaimed here -> y2).
    if _relax_overconfident_obscodes(ev, state, mask):
        state, result = ev.fit(state, mask)
        state, mask = _final_redeem(ev, state, mask, [], False)
        state, result = ev.fit(state, mask)
        history.append(_hsnap({"stage": "relax-redeem", "n_obs": int(mask.sum()),
                               "rms": ev.rms_arcsec(state, mask), "score": ev.rms_norm(state, mask)}, state, mask))
    # LOW-q DILIGENCE FALLBACK: attribution's verdict is a residual comparison against the anchor's
    # back-prediction, and that back-prediction is authoritative ONLY where it is reliable -- the
    # main-belt/outer regime (q >= IDENT_FAR_MIN_Q, same policy as every ident short-circuit). Below
    # that q a GENUINE same-object arc can sit far outside IDENT_ATTR_KEEP_ARCSEC: a multi-revolution
    # phase ambiguity puts real matches kilo-arcsec off (2024 LW's own tracklets ride 1-12 kilo-arcsec
    # from its anchor), and an intervening planetary encounter diverges even a 5-opp orbit's
    # extrapolation. So when a low-q (or q-undetermined -- conservative) anchor CLIPS an entire
    # requested designation, that is INCONCLUSIVE, not a non-match: return None and let the normal
    # IOD/bridge path grind with full Tier-2 diligence. A low-q ident that genuinely ATTACHES (every
    # designation keeps obs) still enjoys the fast path, and main-belt anchors are unaffected.
    clipped = _attr_low_q_inconclusive(q_anchor, obs, mask)
    if clipped is not None:
        if verbose:
            print(f"[ident] attribution INCONCLUSIVE: low-q anchor {well} (q={q_anchor:.2f}) clipped "
                  f"all of {clipped} -- falling back to the full linkage path (NEO diligence)")
        return None
    if verbose:
        print(f"[ident] attributed to well-determined {well} ({oppc[well]} opp): "
              f"kept {int(mask.sum())}/{len(obs)} @ {ev.rms_arcsec(state, mask):.3f}\" "
              f"(other obs {'attached -- genuine' if int(mask.sum()) > f_anchor.n_observations() else 'clipped -- non-match'})")
    ctx.update(desig=desig, meta=meta, seed_d=0.0, method="ident-attr")
    return _build_orbit(ctx, state, mask, result, history)


def _fit_from_start(obs, obscodes, desig="object", meta=None, verbose=False,
                    start=None, extend=True, activate=True):
    """One orbit-fit pass. `start` forces the (start,end) sub-arc (else the quick
    heuristic picks it); `extend=False` fits ONLY that sub-arc (used to probe each
    apparition's extensibility). Returns a FittedOrbit."""
    times = np.array([o["jd_utc"] for o in obs])
    rhohats = [los_unit(o["ra_deg"], o["dec_deg"]) for o in obs]

    # IOD owns the epoch: initial_orbit picks the seed sub-arc (find_orb's geometric
    # select_subarc + the shrink-the-search-arc rescue) AND anchors the osculating epoch ON
    # that arc, returning both. This is the find_orb "epoch within the arc" rule -- a seed
    # expressed far from its own data makes the Cartesian-state LM hopelessly ill-conditioned.
    # The observer geometry (positions, sigmas, backend) is epoch-free, so build it once and
    # then build the evaluator at the epoch IOD returns; there is no preliminary epoch guess to
    # drift out of sync with the arc the shrink loop settles on.
    _ts = _stime.perf_counter()
    geom = _observer_geometry(obs, obscodes)
    seed, method, iod_score, (s_idx, e_idx), t0_tdb = initial_orbit(
        obs, times, rhohats, geom["Robs"], subarc=start)
    if seed is None:
        raise RuntimeError("IOD produced no viable seed orbit")
    ev, ctx = _build_evaluator_at(geom, obs, t0_tdb)
    _arm_ident_shortcircuit(ev, obs, meta)
    init_mask = np.zeros(len(obs), dtype=bool)
    init_mask[s_idx:e_idx + 1] = True
    # On sparse distant arcs (TNOs/Centaurs spread over many oppositions) the geometric
    # select_subarc can hand back a 2-obs window: within 45deg of sky and under the arc cap,
    # but its 4 residuals under-determine the 6-element state and least_squares('lm') raises
    # outright. attempt_extensions grows OUTWARD from this seed regardless, so widen the seed
    # mask by index to FIT_MIN_OBS here (the IOD seed/epoch is untouched -- only the first LM
    # fit gets enough residuals). No-op once the window already has >= FIT_MIN_OBS obs.
    lo, hi = s_idx, e_idx
    while int(init_mask.sum()) < FIT_MIN_OBS and (lo > 0 or hi < len(obs) - 1):
        if hi < len(obs) - 1:
            hi += 1; init_mask[hi] = True
        if int(init_mask.sum()) < FIT_MIN_OBS and lo > 0:
            lo -= 1; init_mask[lo] = True
    _stadd("iod", _ts)
    if verbose:
        print(f"IOD: method {method}, seed score {iod_score:.2f}; "
              f"sub-arc obs[{s_idx}..{e_idx}] of {len(obs)}")
    # LONG-WINDOW RESCUE (iod.rescue_window). Gated on four things, all needed: this pass chose its
    # own window (a forced `start` -- every recovery stage -- is never second-guessed, which also stops
    # this recursing); it will extend; the window is longer than find_orb's VAISALA_MAX_ARC and
    # straddles no opposition gap (initial_orbit's own rescue covers a straddle); and the seed window's
    # OWN differential correction is catastrophic. Over all 195 regression fixtures 12 have a
    # catastrophic sub-arc fit (multi-rev bridges and identification pairs that later stages repair,
    # every window <= 73 d) and 8 have a gap-free window > 230 d (sub-arc rms_n 0.42-1.56): none is
    # both, so the suite never enters. C/2013 A1 is both -- a 468-day window whose own fit is
    # 279863" (rms_n 286839) -- and extension then grows nothing for 12.9 s; placed AFTER extension
    # the rescue made the recovery cascade repeat that on every screen refit (6293 CPU s vs 506 s).
    # The length condition has ONE alternative: a whole arc that is a single apparition, where no
    # later stage exists to repair a catastrophic seed (every recovery stage and the catastrophic
    # last-ditch need a second apparition to re-anchor on). 2018 XQ2 -- a 28-day Aten seen from
    # 0.025 au, i = 0.08 deg, so its lines of sight and Earth's positions are all nearly in one
    # plane and Gauss over the 27-day geometric window is degenerate: seed 277" at i = 178 deg,
    # its own fit rms_n 132, and 33/44 delivered at 93" with the discovery night rejected. find_orb
    # shrinks here unconditionally; the first shrink (19 d) seeds a = 0.845, e = 0.197 (JPL 0.856,
    # 0.178) and fits 44/44 at 0.364". Over the fixtures this admits one more, 2026 LY (1.9-day
    # window, rms_n 5.47), for which rescue_window finds no better 3-night window.
    from pyroorb.iod import VAISALA_MAX_ARC, rescue_window
    from pyroorb.orbit_common import OPP_GAP_DAYS
    whole_single_app = not (times.size > 1 and float(np.max(np.diff(times))) > OPP_GAP_DAYS)
    if (start is None and extend
            and (float(times[e_idx] - times[s_idx]) > VAISALA_MAX_ARC or whole_single_app)):
        sub_t = times[s_idx:e_idx + 1]
        if not (sub_t.size > 1 and float(np.max(np.diff(sub_t))) > OPP_GAP_DAYS):
            try:
                st_sub, _res_sub = ev.fit(seed, init_mask)
                sub_rn = float(ev.rms_norm(st_sub, init_mask))
            except Exception:
                sub_rn = float("inf")
            if sub_rn > CATASTROPHIC_RMS_N:
                window = rescue_window(obs, times, rhohats, geom["Robs"], (s_idx, e_idx), iod_score)
                if window is not None:
                    if verbose:
                        print(f"[long-window rescue] seed window obs[{s_idx}..{e_idx}] fits its own obs at "
                              f"rms_n {sub_rn:.1f}; re-seeding on obs[{window[0]}..{window[1]}]", flush=True)
                    try:
                        cand = _fit_from_start(obs, obscodes, desig, meta, verbose,
                                               start=window, activate=activate)
                    except Exception:
                        cand = None
                    if cand is not None and cand.rms_normalized() <= CATASTROPHIC_RMS_N:
                        if verbose:
                            print(f"[long-window rescue] kept {cand.n_observations()}/{len(obs)} "
                                  f"at {cand.rms_residual():.3f}\"", flush=True)
                        return cand
    if extend:
        state, mask, result, history = attempt_extensions(ev, seed, init_mask,
                                                          verbose=verbose, activate=activate)
        # NB: the soft-weight polish is NOT done here (per-attempt) -- doing so on an intermediate
        # multi-start attempt strands a still-converging apparition. It runs ONCE on the final
        # best orbit in fit_orbit (_final_soft_polish), past the point of no return.
        # Record the RAW IOD-seed residual (pre-differential-correction) as the trace's first
        # step, so the history starts at the IOD orbit itself -- however terrible -- before the
        # sub-arc fit improves it. Cheap (one forward eval on the sub-arc); purely diagnostic.
        try:
            history = [_hsnap({"stage": f"iod:{method}", "n_obs": int(init_mask.sum()),
                        "rms": ev.rms_arcsec(seed, init_mask),
                        "score": ev.rms_norm(seed, init_mask)}, seed, init_mask)] + history
        except Exception:
            pass
    else:
        # probe mode: fit just the sub-arc (for the extensibility ranking)
        state, result = ev.fit(seed, init_mask)
        mask = init_mask
        history = [{"stage": "subarc", "n_obs": int(mask.sum()),
                    "rms": ev.rms_arcsec(state, mask), "score": ev.rms_norm(state, mask)}]
    ctx.update(desig=desig, meta=meta, seed_d=iod_score, method=method)
    return _build_orbit(ctx, state, mask, result, history)


# --- prefer a bound orbit over a marginally-hyperbolic one -----------------
# A short, under-constrained arc has a nearly flat chi-square in eccentricity, so the
# free LM optimum often drifts a hair past e=1. Two rules, both in
# hyperbolic.py: BOUND FIRST -- refit with e held at ECC_CAP and deliver the bound orbit
# unless the data refuse it at 3 sigma (hyperbolic.bound_damaged, a scaled Delta chi2 for the
# one restriction -- "hyperbolic" reads as "interstellar", so a close call goes bound); and an
# unbound orbit is also what a mislinkage looks like, so it is believed only on an arc with
# enough nights. Runs once at finalization, only when e>cap, so the bound-orbit majority is
# untouched.
ECC_CAP = 0.9999       # bound-orbit ceiling
ECC_PEN_W = 1.0e4      # one-sided barrier weight on (e - cap) in the constrained re-fit


def _eccentricity(state_eq, epoch_tdb):
    try:
        return float(element_vector(state_eq, epoch_tdb)[1])
    except Exception:
        return float("nan")


def _refit_bounded_ecc(ev, seed, mask, epoch_tdb, cap=ECC_CAP, w=ECC_PEN_W, max_nfev=400, whiten=None):
    """LM re-fit of the included obs with a one-sided barrier keeping e <= cap.
    Returns (bound_state, clean_result) where clean_result has the eccentricity
    penalty row stripped from .fun/.jac so the covariance stays honest, or
    (None, None) on failure. `cap=None` drops the barrier (a plain re-fit).

    `whiten` (a `nightmodel.NightWhitener` over the included obs, in residual order) makes
    the fit minimise r^T S^-1 r under the station-night error model instead of the diagonal
    fit_sig weighting: residuals are then taken in units of the ASSIGNED sigma (the Veres
    over-observation factors in fit_sig_* are a stand-in for the same block correlation and
    are not applied on top of it) and both the residual vector and the Jacobian go through
    S^-1/2 -- one linear map, so the LM is otherwise unchanged. This is how the unbound gate
    gets two optima whose Delta chi2 has a chi2(1) null (hyperbolic.py).

    Uses the backend's ANALYTIC Jacobian (one multi-particle integration yields the
    prediction plus all 6 columns, shared between the paired fun/jac calls MINPACK
    makes at the same state) exactly as Evaluator.fit does -- never scipy's finite
    difference, which re-integrates the whole arc per parameter. The eccentricity
    penalty row is a cheap two-body calc (no integration), so its gradient is finite-
    differenced essentially for free. This refit fires whenever the unconstrained orbit
    lands hyperbolic, which is common for short, poorly-constrained single-opposition
    arcs, and the finite-difference jac used to dominate them (2001 XB255: 1.6s -> ~0.1s)."""
    from pyroorb import deadline
    from scipy.optimize import least_squares
    from types import SimpleNamespace
    idx = np.flatnonzero(mask)
    tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
    ra = ev.ra[idx]; dec = ev.dec[idx]; cosd = ev.cosd[idx]
    if whiten is None:
        sra = ev.fit_sig_ra[idx]; sdec = ev.fit_sig_dec[idx]
    else:
        sra = ev.sig_ra[idx]; sdec = ev.sig_dec[idx]
    m = len(idx)

    cache = {}
    def _compute(state):
        key = state.tobytes()
        hit = cache.get(key)
        if hit is None:
            cache.clear()
            hit = cache[key] = ev._predict_jac(ev.base_sim, ev.t0, state, tt, Rs)
        return hit

    def resid(state):
        pred, _ = _compute(state)
        d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
        r = np.column_stack([d * 3600.0 / sra, (pred[:, 1] - dec) * 3600.0 / sdec]).ravel()
        r = np.where(np.isfinite(r), r, 0.0)
        if whiten is not None:
            r = whiten.apply(r)
        if cap is None:
            return r
        pen = w * max(0.0, _eccentricity(state, epoch_tdb) - cap)
        out = np.concatenate([r, [pen]])
        return np.where(np.isfinite(out), out, 0.0)

    def jac(state):
        _, J = _compute(state)                # J: (m,2,6) d(ra,dec)/d(state0)
        A = np.empty((m, 2, 6))
        A[:, 0, :] = J[:, 0, :] * (cosd * 3600.0 / sra)[:, None]
        A[:, 1, :] = J[:, 1, :] * (3600.0 / sdec)[:, None]
        A = A.reshape(2 * m, 6)               # row order matches resid's ravel
        A = np.where(np.isfinite(A), A, 0.0)
        if whiten is not None:
            A = whiten.apply(A)
        if cap is None:
            return A
        g = np.zeros(6)                       # penalty-row gradient (active only when e > cap)
        e0 = _eccentricity(state, epoch_tdb)
        if e0 > cap:
            for k in range(6):
                h = 1e-6 * max(1.0, abs(state[k]))
                sp = state.copy(); sp[k] += h
                g[k] = w * (_eccentricity(sp, epoch_tdb) - e0) / h
        out = np.vstack([A, g[None, :]])
        return np.where(np.isfinite(out), out, 0.0)

    try:
        result = least_squares(resid, np.asarray(seed, float), jac=jac, method="lm", x_scale="jac",
                            xtol=1e-8, ftol=1e-8, max_nfev=max_nfev)
    except deadline.DeadlineExceeded:
        raise                                 # out of time is not "this fit failed": several run here
    except Exception:
        return None, None
    if cap is None:
        return result.x, SimpleNamespace(x=result.x, fun=result.fun, jac=result.jac)
    jacm = result.jac[:-1] if (result.jac is not None and result.jac.shape[0] > 1) else result.jac
    return result.x, SimpleNamespace(x=result.x, fun=result.fun[:-1], jac=jacm)


def _note_unbound_check(fit, check):
    try:
        fit.unbound_check = check
    except AttributeError:
        pass
    return fit


def _fit_weighted_chi2(ev, state, mask):
    """chi2 of the included obs under the weighting the LM actually minimises (fit_sig_*), which
    is what a Delta chi2 between two of its solutions has to be measured in -- `rms_norm` divides
    by the display sigmas, which can differ (a pasted 2I arc: 1" display vs 1.1-1.66" fit)."""
    idx = np.flatnonzero(mask)
    dra, ddec = ev.resid_subset(state, mask)
    v = np.concatenate([dra / ev.fit_sig_ra[idx], ddec / ev.fit_sig_dec[idx]])
    return float(np.sum(v * v))


def _block_chi2(ev, state, mask, whiten):
    """chi2 of the included obs under the station-night error model: r^T S^-1 r with r in units
    of the ASSIGNED sigma, interleaved [RA*, Dec] per observation as `whiten` expects."""
    idx = np.flatnonzero(mask)
    dra, ddec = ev.resid_subset(state, mask)
    r = np.column_stack([dra / ev.sig_ra[idx], ddec / ev.sig_dec[idx]]).ravel()
    return whiten.chi2(np.where(np.isfinite(r), r, 0.0))


ECC_CAP_TOL = 1e-4          # a capped fit within this of the cap IS a bound orbit
BOUND_SEED_SCREEN_NFEV = 12 # quick correction of every admissible seed before the best are refined
BOUND_SEED_REFINE = 2       # seeds given a full capped fit
ECC_PEN_STIFFEN = (1.0e6, 1.0e8)   # retries when a capped fit must be brought UNDER the cap


def _bound_candidate(ev, seed, mask, epoch, w=ECC_PEN_W, max_nfev=400):
    """One capped fit, and what it is evidence of. ``exact`` when it ended under the cap: `chi2` is
    then a bound orbit's chi2. Otherwise the soft barrier balanced against the data a hair above
    the cap, and `bound_chi2` = chi2 + penalty^2 is a LOWER BOUND on what any bound orbit in this
    basin costs -- the capped optimum has no penalty to pay and cannot beat the penalised one."""
    bstate, bres = _refit_bounded_ecc(ev, seed, mask, epoch, w=w, max_nfev=max_nfev)
    if bstate is None or not np.all(np.isfinite(bstate)):
        return None
    e = _eccentricity(bstate, epoch)
    if not math.isfinite(e):
        return None
    chi2 = _fit_weighted_chi2(ev, bstate, mask)
    pen = w * max(0.0, e - ECC_CAP)
    return {"state": bstate, "res": bres, "e": e, "chi2": chi2, "exact": e <= ECC_CAP + ECC_CAP_TOL,
            "bound_chi2": chi2 + pen * pen}


def _admissible_bound_candidates(ev, mask, epoch):
    """Capped fits started ACROSS the admissible region (hyperbolic.bound_seeds) instead of at the
    unbound optimum -- the best `BOUND_SEED_REFINE` of them, fully refined.

    ``(candidates, n_seeds, status)``. The status is load-bearing: an empty list
    used to come back for five different reasons, and the record could not tell "the scan ran
    out to 300 au and the geometry admits no bound orbit" from "the scan was never run" -- so a
    comparator that fell back to the local walk did so silently. ``too-few-obs`` and
    ``arc-too-long`` are not attempted; ``no-admissible-range`` is the scan refusing;
    ``integration-failed`` and ``no-candidate-fit`` had seeds and lost them; ``ok`` has
    candidates."""
    from pyroorb import deadline; from pyroorb import force_assist; from pyroorb import hyperbolic
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return [], 0, "too-few-obs"
    i1, i2 = int(idx[0]), int(idx[-1])
    t1, t2 = float(ev.times_tdb[i1]), float(ev.times_tdb[i2])
    if not 0 < t2 - t1 <= hyperbolic.BOUND_SEED_MAX_ARC_DAYS:
        return [], 0, "arc-too-long"

    def _unit(i):
        ra, dec = math.radians(float(ev.ra[i])), math.radians(float(ev.dec[i]))
        return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])

    seeds = hyperbolic.bound_seeds(t1, ev.Robs[i1], _unit(i1), t2, ev.Robs[i2], _unit(i2))
    if not seeds:
        return [], 0, "no-admissible-range"
    try:                         # every seed shares t_mid: ONE integration carries them all to t0
        at_t0 = force_assist.helio_states_multi(seeds[0][0], [sd[1] for sd in seeds], [ev.t0])[0]
    except deadline.DeadlineExceeded:
        raise
    except Exception:
        return [], len(seeds), "integration-failed"
    quick = []
    for st, (_t, _s, rho) in zip(at_t0, seeds):
        c = _bound_candidate(ev, st, mask, epoch, max_nfev=BOUND_SEED_SCREEN_NFEV)
        if c is not None:
            c["rho_seed"] = rho
            quick.append(c)
    quick.sort(key=lambda c: c["bound_chi2"])
    out = []
    for c in quick[:BOUND_SEED_REFINE]:
        full = _bound_candidate(ev, c["state"], mask, epoch)
        if full is not None:
            full["rho_seed"] = c["rho_seed"]
            out.append(full)
    return out, len(seeds), ("ok" if out else "no-candidate-fit")


BRIDGE_REFINE = BOUND_SEED_REFINE   # bridged seeds given a full capped fit (the model sequence's count: each refit
                                    # integrates the whole multi-year arc, ~0.6 s on 107474's 39 years)


def _bridged_bound_candidates(ev, mask, epoch):
    """Capped fits started from bound orbits that JOIN the arc's apparitions over every revolution
    count (`boundbridge.bridge_seeds`): the comparator for a long arc made of short,
    widely separated apparitions, where the admissible-region chord is no velocity and the walk
    from a wrong-revolution hyperbola cannot cross to the bound orbit.

    ``(candidates, n_seeds, status)``, the status saying why there are none:
    ``single-apparition`` / ``long-apparition`` are not attempted (one apparition has no gap to
    bridge; an apparition longer than `BOUND_SEED_MAX_ARC_DAYS` pins its own range by curvature,
    the walk's regime -- and on a dense long arc this search would be the costliest thing in the
    fit for no gain); ``no-transfer`` is the enumeration finding no elliptic transfer at any
    range; ``integration-failed`` / ``no-candidate-fit`` had seeds and lost them; ``bridged`` has
    candidates. Every seed is screened with a short capped fit and the best `BRIDGE_REFINE`
    refined; every refined fit that converges is kept, and `_weigh` takes the cheapest."""
    from pyroorb import arc_extend; from pyroorb import boundbridge; from pyroorb import deadline; from pyroorb import force_assist; from pyroorb import hyperbolic
    groups = boundbridge.oppositions(ev.times_tdb, np.flatnonzero(mask), arc_extend.OPP_GAP_DAYS)
    if len(groups) < 2:
        return [], 0, "single-apparition"
    if any(float(np.ptp(ev.times_tdb[g])) > hyperbolic.BOUND_SEED_MAX_ARC_DAYS for g in groups):
        return [], 0, "long-apparition"
    t_ref, states, info = boundbridge.bridge_seeds(ev, groups)
    if states is None or not len(states):
        return [], 0, "no-transfer"
    try:                         # every seed is at the anchor's reference epoch: ONE integration
        at_t0 = force_assist.helio_states_multi(t_ref, list(states), [ev.t0])[0]
    except deadline.DeadlineExceeded:
        raise
    except Exception:
        return [], len(states), "integration-failed"
    quick = []
    for st, meta in zip(at_t0, info):
        c = _bound_candidate(ev, st, mask, epoch, max_nfev=BOUND_SEED_SCREEN_NFEV)
        if c is not None:
            c["bridge"] = meta
            quick.append(c)
    quick.sort(key=lambda c: c["bound_chi2"])
    out = []
    for c in quick[:BRIDGE_REFINE]:
        full = _bound_candidate(ev, c["state"], mask, epoch)
        if full is not None:
            full["bridge"] = c["bridge"]
            out.append(full)
    return out, len(states), ("bridged" if out else "no-candidate-fit")


def _prefer_bound_ecc(fit, verbose=False):
    """If the final orbit came out hyperbolic, find the best BOUND orbit and adopt it unless BOTH
    unbound rules (hyperbolic.py) let the unbound one stand: the data refuse the bound orbit at
    3 sigma (hyperbolic.bound_damaged), AND the arc has the nights an unbound orbit may be believed
    on (or refuses the bound one outright). What was weighed rides on the delivered orbit as
    `unbound_check`.

    The bound orbit the rules are shown is the best of: a capped walk from the unbound optimum, and
    -- on a short arc, whenever that walk leaves the bound orbit looking damaged -- capped fits
    started across the admissible region. EVERY exit goes through the rules: a walk that
    could not get under the cap is weighed by the lower bound it proves, not waved through."""
    from pyroorb import hyperbolic
    ctx = getattr(fit, "_ctx", None)
    if not ctx or ctx.get("ev") is None or ctx["ev"]._predict_jac is None:
        return fit
    ev = ctx["ev"]
    state, mask, epoch = fit._res.x, fit.is_included(), fit.epoch_tdb
    e_unc = _eccentricity(state, epoch)
    if not math.isfinite(e_unc) or e_unc <= ECC_CAP:
        return fit                                   # already bound (NaN -> leave the warning)
    arc_ok, nights, arc = hyperbolic.unbound_arc_ok(
        [o["jd_utc"] for o, m in zip(fit._obs, mask) if m])
    rms_u = ev.rms_arcsec(state, mask)
    check = {"e_free": e_unc, "rms_free": float(rms_u), "nights": nights,
             "arc_days": float(arc), "arc_ok": arc_ok}
    n_resid = 2 * int(np.sum(mask))
    inc = [o for o, m in zip(fit._obs, mask) if m]
    idx = np.flatnonzero(mask)
    jds = [o["jd_utc"] for o in inc]
    codes = [o.get("obscode") for o in inc]
    n_trk = hyperbolic.count_station_nights(jds, codes)
    check["station_nights"] = n_trk
    chi2_u = _fit_weighted_chi2(ev, state, mask)

    # THE CURRENCY THE RULES ARE WEIGHED IN: the station-night error model. The
    # free fit's own residuals say how this object's within-night scatter compares with its
    # stations' (orbit-independent: each night is reduced by its own line), the whitener then
    # carries that scale plus the table's per-night offset and rate systematics, and BOTH sides
    # of the comparison are re-optimised under it so their Delta chi2 is the chi2(1) variate the
    # rules' "3 sigma" and "dBIC >= 15" name. The delivered orbits stay the fit_sig-weighted
    # ones: this decides WHICH orbit, not how it is determined.
    from pyroorb import nightmodel
    dra_u, ddec_u = ev.resid_subset(state, mask)
    k_scale, within_dof, within_ratio, scale_dof = nightmodel.within_scale(
        jds, codes, dra_u, ddec_u, ev.sig_ra[idx], ev.sig_dec[idx])
    W = nightmodel.NightWhitener(jds, codes, ev.sig_ra[idx], ev.sig_dec[idx], k_scale=k_scale)
    free_S, _ = _refit_bounded_ecc(ev, state, mask, epoch, cap=None, whiten=W, max_nfev=200)
    if free_S is None or not np.all(np.isfinite(free_S)):
        free_S = state
    chi2_free_S = min(_block_chi2(ev, free_S, mask, W), _block_chi2(ev, state, mask, W))
    # LACK OF FIT. Under the model a free fit's chi2 is ~(2n - 6); when it is far above that the
    # residuals are not what the model describes -- a mislinkage, a catastrophic fit, a force the
    # gravity model lacks, a station whose row does not fit this object -- and a Delta chi2 in
    # the model's currency overstates the evidence (PAIR 1999 KK17's last three nights: five
    # observations over 17 years at 10", the model's sigma 0.3", "refused at 56 sigma"). So the
    # evidence is measured against the scatter the free fit ACTUALLY achieved when that is worse
    # than the model, and never better: the same one-sided floor the covariance uses
    # (`FittedOrbit._param_inverse`), from the same reasoning. For a fit the model describes the
    # scale is 1 and the statistic is the chi2(1) it is meant to be.
    dof_S = max(1, n_resid - 6)
    lack_of_fit = chi2_free_S / dof_S
    evidence_scale = max(1.0, lack_of_fit)
    check.update(n_eff=float(W.n_eff), k_scale=float(k_scale), within_dof=int(within_dof),
                 within_ratio=(None if within_ratio is None else float(within_ratio)),
                 scale_dof=(None if scale_dof is None else float(scale_dof)),
                 chi2_free_block=float(chi2_free_S), chi2_free_fit=float(chi2_u),
                 lack_of_fit=float(lack_of_fit), evidence_scale=float(evidence_scale))

    def _block(c):
        """The candidate's bound orbit re-optimised under the model, and what it proves: `chi2`
        at its optimum, `bound_chi2` = chi2 + penalty^2 -- the same LOWER-BOUND reading as the
        fit_sig-weighted candidate when the barrier stopped a hair above the cap."""
        if "block" not in c:
            bs, _ = _refit_bounded_ecc(ev, c["state"], mask, epoch, whiten=W, max_nfev=200)
            if bs is None or not np.all(np.isfinite(bs)):
                bs = c["state"]
            e_b = _eccentricity(bs, epoch)
            if not math.isfinite(e_b):
                e_b = c["e"]
            chi2 = _block_chi2(ev, bs, mask, W)
            pen = ECC_PEN_W * max(0.0, e_b - ECC_CAP)
            c["block"] = {"chi2": chi2, "bound_chi2": chi2 + pen * pen, "e": e_b}
        return c["block"]

    def _weigh(cands):
        """(best candidate, verdict) under rules 1 and 2, the bound side being the CHEAPEST bound
        orbit any candidate proves possible under the model.

        `dchi2_block` is the statistic the rules decide on -- chi2(1) under the station-night
        model (F(1, scale_dof) when the arc's own scatter set the scale; divided by the
        lack-of-fit scale when the free fit misses the model; 1 otherwise), `evidence` its
        one-dof equivalent, so `sigma` = sqrt(evidence) IS the sigma the rules' 3 is measured against
        and `damaged` is exactly `sigma >= sigma_bar`; `dbic` = dchi2_block - ln(n_eff) is what
        `bound_refused` compares with NONGRAV_BIC_MARGIN, n_eff the independent measurements
        the arc holds under the model (Kish). `dchi2_raw` is the retired PER-OBSERVATION
        statistic, kept beside it as the diagnostic old records and A/B studies are read in."""
        best = min(cands, key=lambda c: _block(c)["bound_chi2"])
        dchi2 = (_block(best)["bound_chi2"] - chi2_free_S) / evidence_scale
        damaged, evid = hyperbolic.block_damaged(dchi2, scale_dof)
        refused, dbic = hyperbolic.block_refused(dchi2, W.n_eff, NONGRAV_BIC_MARGIN, scale_dof)
        shown = hyperbolic.reported_evidence(evid) if evid is not None else None
        return best, {"damaged": bool(damaged), "dchi2_block": float(dchi2),
                      # ...as the one-dof chi2 its tail probability names when the scale was estimated
                      "evidence": (None if evid is None else float(evid)),
                      "dchi2_raw": hyperbolic.bound_dchi2(chi2_u, best["bound_chi2"], n_resid),
                      "sigma": (None if shown is None else math.sqrt(shown)),
                      "sigma_bar": math.sqrt(hyperbolic.BOUND_DAMAGE_DCHI2),
                      # ...and whether that sigma is the reportable ceiling rather than a
                      # measurement (1I's full arc is): `dchi2_block` says how far past it went
                      "sigma_is_floor": hyperbolic.saturated_evidence(shown),
                      "chi2_bound_block": float(_block(best)["bound_chi2"]),
                      "chi2_bound_fit": float(best["bound_chi2"]),
                      # the cheapest bound orbit under the fit's OWN weighting, whichever candidate
                      # proves it: what the retired per-observation rule would have weighed
                      "chi2_bound_fit_min": float(min(c["bound_chi2"] for c in cands)),
                      "dbic": dbic, "bound_refused": bool(refused),
                      "unbound": bool(damaged and (arc_ok or refused))}

    cands = [c for c in (_bound_candidate(ev, state, mask, epoch),) if c is not None]
    if cands:
        cands[0]["from"] = "walk"
        check.update(e_attempt=cands[0]["e"], rms_attempt=float(ev.rms_arcsec(cands[0]["state"], mask)))
    verdict = _weigh(cands)[1] if cands else None
    # The walk is a LOCAL answer. When it leaves the bound orbit looking damaged, or never got under
    # the cap (so there is no bound orbit to deliver even if the rules ask for one), ask the
    # question globally before believing it -- cheap, because it only runs on short arcs.
    if verdict is None or verdict["damaged"] or not cands[0]["exact"]:
        bound_cands, n_seeds, status = _admissible_bound_candidates(ev, mask, epoch)
        for c in bound_cands:
            c["from"] = "admissible"
        cands += bound_cands
        check["bound_seeds"] = n_seeds
        # ...and WHY that is the count: a 0 that was never attempted and a 0 the
        # scan came back with must not read alike on the record.
        check["admissible"] = status
        if status == "arc-too-long":
            # A long arc: no chord to seed from. When it is short apparitions years apart, the bound
            # orbit may sit a different number of revolutions away from the hyperbola the free fit
            # found -- search the revolution counts. `admissible` becomes "bridged" only
            # when that search produced candidates, so an arc it could not help keeps its label.
            bridged, n_bridge, b_status = _bridged_bound_candidates(ev, mask, epoch)
            for c in bridged:
                c["from"] = "bridge"
            cands += bridged
            check["bridge"] = b_status
            check["bridge_seeds"] = n_bridge
            if bridged:
                check["admissible"] = "bridged"
    else:
        check["admissible"] = "walk-sufficed"
    if not cands:
        check["kept"] = "unbound-no-bound-fit"       # no capped fit ran at all: nothing to weigh
        return _note_unbound_check(fit, check)
    best, verdict = _weigh(cands)
    # A verdict for BOUND has to deliver a bound orbit. If the cheapest candidate is only a lower
    # bound (it stopped a hair above the cap), stiffen the barrier until it is under, then weigh
    # again with the exact number. Never paid when the verdict is unbound (1I, 2I, 3I).
    for w in ECC_PEN_STIFFEN:
        if verdict["unbound"] or best["exact"]:
            break
        stiff = _bound_candidate(ev, best["state"], mask, epoch, w=w)
        if stiff is None:
            break
        stiff["from"], stiff["rho_seed"] = best.get("from"), best.get("rho_seed")
        if best.get("bridge") is not None:
            stiff["bridge"] = best["bridge"]
        cands = [c for c in cands if c is not best] + [stiff]
        best, verdict = _weigh(cands)
    check.update({k: v for k, v in verdict.items() if k != "unbound"})
    check.update(comparator=best.get("from"), e_bound=best["e"],
                 rms_bound=float(ev.rms_arcsec(best["state"], mask)))
    if best.get("rho_seed") is not None:
        check["rho_seed_au"] = best["rho_seed"]
    if best.get("bridge") is not None:
        check["bridge_revs"] = best["bridge"]["N"]    # the revolution count the comparator took
    if verdict["unbound"]:
        # EARNED or not. An unbound orbit on an arc under the night bar is believed only because the
        # data refuse the bound orbit -- and "the bound orbit" is whatever the comparator found. When
        # the comparator could only WALK from the hyperbolic optimum (`arc-too-long`: no admissible
        # seeds on a long arc; or seeds that never became a fit), a refusal is of a LOCAL bound orbit,
        # not a statement that no bound orbit fits: six observations over 39 years read e = 1.44 with
        # the walk stuck at 24" while the object's own orbit fits them at 0.3" (107474's first three
        # nights). The free fit is still the best orbit in hand and is delivered, but the
        # record and the web note say nothing earned it, as `unbound-no-bound-fit` already does. The
        # night bar itself, or a scan that ran (`ok`) or found the geometry admits no bound orbit at
        # any range (`no-admissible-range`), or a revolution-count search that produced bound fits
        # across the gaps and still lost (`bridged`), earns it.
        searched = check.get("admissible") in ("ok", "no-admissible-range", "bridged")
        check["kept"] = "unbound" if (arc_ok or searched) else "unbound-unsearched"
        return _note_unbound_check(fit, check)       # the data genuinely want e>1 -> keep it
    if not best["exact"]:
        # The rules say bound and no fit could be brought under the cap: deliver what we have, and
        # say that nothing earned it (it used to be the ONLY thing this exit said).
        check["kept"] = "unbound-no-bound-fit"
        return _note_unbound_check(fit, check)
    check["kept"] = "bound"
    if verbose:
        why = f"only {nights} nights" if verdict["damaged"] else "bounding it barely moves the residuals"
        print(f"   ecc-cap: e={e_unc:.4f}->{best['e']:.4f} (rms {rms_u:.3f}\"->{check['rms_bound']:.3f}\", "
              f"{best.get('from')}); kept the bound orbit ({why})")
    out = _build_orbit(ctx, best["state"], mask, best["res"], getattr(fit, "_history", None))
    return _note_unbound_check(out, check)


def _count_range_modes(values):
    """Count well-separated clusters ("range aliases") in a 1-D set of admissible distances.
    A short arc can fit cleanly at MORE THAN ONE distance (find_orb's range aliasing); the SR
    family then splits into separated clumps. Conservative gap-based clustering: sort, break at
    any gap wider than max(0.25 x total extent, 4 x median gap), and count clumps holding at
    least 15% of the family. Returns 1 for a single unimodal family (the common case)."""
    v = np.sort(np.asarray(values, float))
    v = v[np.isfinite(v)]
    if v.size < 6:
        return 1
    gaps = np.diff(v)
    extent = float(v[-1] - v[0])
    if extent <= 0:
        return 1
    med_gap = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 0.0
    thresh = max(0.25 * extent, 4.0 * med_gap)
    breaks = np.where(gaps > thresh)[0]
    sizes, start = [], 0
    for b in list(breaks) + [v.size - 1]:
        sizes.append(b - start + 1)
        start = b + 1
    floor = max(3, int(0.15 * v.size))
    return max(1, sum(1 for s in sizes if s >= floor))


# ---------------------------------------------------------------------------
# Fitted-orbit result object with getters mirroring find_orb's elements.json
# ---------------------------------------------------------------------------
class FittedOrbit:
    """A fitted orbit: the solution, its uncertainty, and how the observations fit it.

    Made by :func:`pyroorb.fit_orbit`; not constructed directly. The methods listed in the
    API reference (``docs/api.rst``) are supported; the rest are internal.

    Attributes
    ----------
    designation : str
        The object's name in the results.
    epoch_tdb : float
        The epoch of the elements and state, Julian date, TDB (same as :meth:`epoch_jd`).
    converged : bool
        False when the final least-squares correction stopped at its iteration cap.
    force_model : str
        The dynamical model the fit used, e.g. ``"ASSIST(DE440+GR)"``.
    nongrav_model : str or None
        The non-gravitational model the fit kept -- ``"comet1"`` / ``"comet2"`` /
        ``"comet3"`` (Marsden A1, A2, A3), ``"yarkovsky"`` (A2, 1/r^2) or ``"srp"`` -- or None
        for gravity only.
    nongrav_params : dict
        The fitted coefficients, ``{"A1": ..., "A2": ...}``, au/day^2.
    center : str
        Present only on a planetary satellite: the planet fitted around.
    radar : dict
        Present when radar astrometry was given: ``applied`` (whether it joined the fit),
        ``n``, ``n_used``, ``rows`` (per measurement, its ``z`` = O-C in sigmas and
        ``included``) and, when it was not applied, a ``note`` saying why.
        :func:`pyroorb.radar_residuals` gives the full residuals of any orbit.
    """

    def __init__(self, desig, obs, result, epoch_tdb, seed_d, iod_method="?",
                 mask=None, resid_all=None, history=None, meta=None,
                 sigma_ra=None, sigma_dec=None, force_model="?", phot=None):
        meta = meta or {}
        self.force_model = force_model
        self._phot = phot or {"H": None, "sigma_H": None, "n_mags": 0, "per": []}
        self.designation = meta.get("designation", desig)
        self.combined = bool(meta.get("combined", False))
        self.other_designations = list(meta.get("other_designations", []))
        self.designations = list(meta.get("designations", [self.designation]))
        # How many designations the CALLER asked for (orbit_common.ident_requested's field),
        # carried onto the fit because the LINKAGE layer needs it: `combined` alone cannot tell
        # "the user typed A and B" from "the user typed A and the MPC handed back A's already-
        # acknowledged aliases". Only the first is an identification anyone may submit.
        # None = an obs pile with no requested count (a paste, a fixture): unknown, not one.
        self.n_requested = meta.get("n_requested")
        # ...and WHICH ones, when the fetch recorded them (obs_sources.fetch_observations).
        # n_requested answers "is this an identification hypothesis at all"; the linkage
        # document needs the names, so that a search seeded from "A B" submits BOTH and not
        # just the alphabetically-earliest designation present.
        self.requested_designations = list(meta.get("requested_designations", []))
        self.packed_designation = pack_designation(self.designation)
        self._obs = obs
        self._res = result                 # scipy least_squares OptimizeResult (incl. obs only)
        # LM exit status: scipy sets status<=0 when MINPACK stopped for a reason other than
        # convergence (0 = the max_nfev cap). A capped delivery fit's elements are a
        # mid-iteration state and the covariance below is evaluated off-minimum -- carry the
        # flag so reports can say so instead of presenting it like a converged solution.
        # (A zero-iteration re-evaluation result -- Evaluator.evaluate's SimpleNamespace,
        # which has no .status -- is an assessment AT an already-converged state: default True.)
        self.converged = bool(getattr(result, "status", 1) > 0)
        self._state = np.asarray(result.x)  # heliocentric EQUATORIAL state at epoch
        self.epoch_tdb = epoch_tdb
        self.seed_d = seed_d                # IOD score of the winning seed
        self.iod_method = iod_method        # which find_orb-style method won
        # inclusion mask (find_orb's is_included) + per-obs O-C for ALL obs (arcsec)
        self._mask = (np.ones(len(obs), dtype=bool) if mask is None
                      else np.asarray(mask, dtype=bool))
        self._resid_all = resid_all         # (n,2) arcsec dRA*cosδ, dDec, or None
        n = len(obs)
        self._sig_ra = (np.ones(n) if sigma_ra is None else np.asarray(sigma_ra))
        self._sig_dec = (np.ones(n) if sigma_dec is None else np.asarray(sigma_dec))
        self.history = history or []
        # Non-gravitational solution (None unless a nongrav model was refined onto the
        # orbit): the model name, the fitted A1/A2/A3 (au/day^2) + their 1-sigma, and the
        # (A, gr) pair the propagator uses. Set by refine_nongrav via _attach_nongrav.
        self.nongrav_model = meta.get("nongrav_model")
        self.nongrav_params = dict(meta.get("nongrav_params") or {})
        # LINEAR covariance sigma (fast; used internally, e.g. discovery z gating). The RATIONAL
        # displayed uncertainty is the profile-likelihood interval computed lazily on first display
        # -- see rational_nongrav_sigmas(). `_nongrav_profile` flags a fit that should profile.
        self.nongrav_sigmas = dict(meta.get("nongrav_sigmas") or {})
        self._nongrav_profile = bool(meta.get("nongrav_profile"))
        self._rational_sigmas = None                 # lazy cache: (sigmas_dict, unconstrained_list, intervals_dict)
        self._jackknife_sigmas = None                # lazy cache: {axis: {se, z, spans_zero, flip, n}}
        # Advisory (non-blocking) notice that the fitted nongrav coefficients may be non-physical
        # (e.g. A1 implies a sub-rock density) -- surfaced as a UI banner / CLI note, never a veto.
        self.nongrav_warning = meta.get("nongrav_warning")
        # What the nongrav LM actually DID (refine_nongrav's stall rescue): whether the cold trial
        # stalled at A = 0 exactly, was cornered, and which axis's profile seed (if any) rescued it.
        # {} on a gravity orbit or a seeded fit. A coefficient reported as zero is only a finding
        # when this says the fit moved.
        self.nongrav_diag = dict(meta.get("nongrav_diag") or {})
        # The auto-discovery record ({"comet": info, "asteroid": info}) _finalize attaches to the
        # DELIVERED orbit: per-model chi2 / F / z / params / diag for every trial, whichever won.
        # None when discovery did not run. A sweep reads its trial outcomes from here.
        self.nongrav_discovery = None
        # Held-fixed orbital elements (find_orb limited_orbit), {element: target} or {} for
        # an unconstrained fit. Set by refit_constrained via meta; informational only.
        self.constraints = dict(meta.get("constraints") or {})
        # True when the 16 massive-asteroid perturbers are active (refine_asteroids polish).
        self.asteroids = bool(meta.get("asteroids"))
        # Set (to '1 Ceres', '4 Vesta', ...) when THIS object is one of the 16 perturbers, so
        # the fit ran with itself excluded from the perturber field (any asteroid perturbers
        # are the other 15) -- it cannot be perturbed by itself. See fit_orbit.
        self.self_perturber = None
        self._elem = element_vector(self._state, epoch_tdb)
        self._state_cov = self._compute_state_cov()
        self._elem_cov = self._compute_elem_cov()

    # --- covariance machinery -------------------------------------------
    def _param_inverse(self):
        """`s^2 (J^T J)^-1` over the WHOLE solved parameter vector, in the LM's own units.

        The one place the fit's covariance is formed; `_compute_state_cov` takes its 6x6
        corner and `parameter_covariance` publishes all of it."""
        J = np.asarray(self._res.jac)               # d(norm resid)/d(params), m x (6[+k])
        npar = J.shape[1]                           # 6, or 6+k with a nongrav model
        dof = max(1, J.shape[0] - npar)
        s2 = float(np.sum(self._res.fun**2)) / dof  # normalized-residual variance
        # ...FLOORED AT 1: the covariance may be scaled UP when the fit misses its
        # assigned sigmas, never DOWN when it beats them.
        #
        # WHY s2 < 1 IS THE NORMAL CASE, AND WHY IT MEANS NOTHING ABOUT THE ORBIT
        # (measured 2026-09-26; supersedes the first rationale,
        # which said "the model absorbs the noise"):  s2 compares two unlike
        # quantities.  The residual scatter of a fit is a WITHIN-NIGHT quantity --
        # measured over the station-nights of 68 full-arc fixtures it is 0.25x the
        # assigned sigma (median over stations; F51 0.31, G96 0.5, 703 0.33).  The
        # assigned sigma (Veres 2017 / ADES rms) is a population rms that folds in
        # the night-to-night systematics too (another 0.05-0.4" per station), so a
        # fit's chi2/dof against it is 0.06-0.25 BY CONSTRUCTION.  Measured: median
        # 0.15-0.21 in every arc class, below 1 on 98% of sub-arcs and on 100% of
        # full arcs -- where hundreds of residuals leave a 6-parameter model nothing
        # to absorb, so absorption was never the mechanism.  Scaling the covariance
        # by that ratio shrinks a systematic-inclusive uncertainty by a within-night
        # scatter, which is the systematic over-confidence the floor removes.
        #
        # The consequence is that this floor is not an occasional guard: it makes
        # the covariance the ASSIGNED-sigma covariance inv(J^T J) nearly always,
        # which is JPL's own practice (weights only, no chi2 rescaling).  Because
        # the transition at s2 = 1 is almost never visited, the FORM of the floor
        # does not matter -- find_orb's sigma_squared (orb_func.cpp:3188, an
        # ADDITIVE +1 per already-normalised residual, so scale = chi2/dof + ~1)
        # scores identically against the truth on every arc class.
        #
        # Measured against the truth (the full-arc orbit) on 268 sub-arc fits:
        # unfloored, a quarter to a half of 2-night, 3-night, one- and
        # two-apparition arcs sit beyond the 99.9% quantile of the truth's chi2(6);
        # floored, the tails roughly halve and the core is 2-6x too WIDE.  Full-arc
        # sigma_a is 0.95x JPL's (32 like-for-like objects) against 0.43x unfloored;
        # on 25 random objects the earlier study read 1.21x.  The original anchor
        # case stands: 2014 HM543's 3.9-day seed ran s2 = 0.0099, sigmas 10x too
        # small, its later apparitions at 14.5 sigma when they sit at 1.4, and the
        # ITF LOV fan (+-5 sigma) could not reach astrometry that was never more
        # than ~1.4 honest sigma away; floored, that search recovers the object with
        # no gate change.
        #
        # What the floor does NOT do: no covariance in this family is calibrated on
        # a short arc.  A per-station-night correlated error model (realistic
        # within-night sigma plus the measured night systematic) centres the core
        # and leaves the tails as heavy; 10-15% of 3-night and single-apparition
        # arcs sit past the 99.9% quantile under ANY Gaussian rescaling, and a third
        # of sparse two-apparition fits are in a wrong revolution basin outright.
        # Those tails are the non-Gaussian shape of a short arc's chi2 surface, and
        # the reason the ITF and identification paths use the LOV, clone clouds and
        # the range profile rather than this matrix. Not handled: no Gaussian
        # rescaling of this matrix reaches those tails.
        s2 = max(s2, 1.0)
        try:
            full = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            full = np.linalg.pinv(J.T @ J)
        return s2 * full

    def cov_floor_factor(self):
        """How much the s2 floor in :meth:`_param_inverse` widened this fit's
        sigmas: ``1/sqrt(s2_raw)`` when the fit beat its assigned sigmas, else 1.

        For consumers that must reproduce the PRE-floor scale exactly -- the ITF
        rankers were trained on covariance-derived features computed without the
        floor, and serving them the honest scale is a silent train/serve skew
        (itf.training.candidate_feature_row).  Not a correction to apply anywhere
        else: the floored covariance is the right one."""
        J = np.asarray(self._res.jac)
        dof = max(1, J.shape[0] - J.shape[1])
        s2 = float(np.sum(self._res.fun**2)) / dof
        return 1.0 / math.sqrt(s2) if 0.0 < s2 < 1.0 else 1.0

    def _compute_state_cov(self):
        # The state covariance is the top-left 6x6 block of the FULL inverse: with a
        # nongrav model that correctly MARGINALISES over the out-gassing parameters
        # (it is not the inverse of the 6x6 state block alone). k=0 -> the whole matrix.
        return self._param_inverse()[:6, :6]

    def parameter_covariance(self):
        """The FULL solved-parameter covariance, in PHYSICAL units, with its axis names.

        `_compute_state_cov` keeps only the top-left 6x6 (the state, marginalised over the
        out-gassing); this is the SAME inverse uncut -- the six cartesian state components
        followed by the model's free A coefficients in fit order. The LM carries those
        coefficients in `nongrav.MODELS[model]["scale"]` units, so their rows and columns
        are lifted back into au/day^2 here -- the units `nongrav_params` reports -- which
        makes the diagonal's square root equal `nongrav_sigmas` exactly.

        Returns ``(cov, names)``: an (6+k)x(6+k) array and the matching label list
        ``["x", "y", "z", "dx", "dy", "dz", "A1", ...]``. A gravity fit gives the plain
        6x6 and six names. Written for the find_orb `covar.json` export
        (`egress.covariance_document`), which reports the whole parameter vector."""
        import pyroorb.nongrav as ng
        full = self._param_inverse()
        npar = full.shape[0]
        # inv() of a symmetric matrix is symmetric only up to round-off, and this one is
        # published (a consumer may legitimately assert cov == cov.T); average the halves.
        cov = 0.5 * (full + full.T)
        names = ["x", "y", "z", "dx", "dy", "dz"]
        model = self.nongrav_model
        if not (model and model != "none" and npar > 6):
            return cov[:6, :6], names
        spec = ng.model(model)
        k = min(npar - 6, len(spec["params"]))
        names += [ng.A_AXES[a] for a in ng.free_indices(model)][:k]
        # scaled -> physical: one factor per row AND per column
        f = np.ones(npar)
        f[6:6 + k] = spec["scale"]
        return (cov * np.outer(f, f))[:6 + k, :6 + k], names

    def _elem_jacobian(self):
        G = np.zeros((len(ELEM_NAMES), 6))
        for k in range(6):
            h = 1e-7 * max(1.0, abs(self._state[k]))
            sp = self._state.copy(); sp[k] += h
            sm = self._state.copy(); sm[k] -= h
            dvec = element_vector(sp, self.epoch_tdb) - element_vector(sm, self.epoch_tdb)
            for a in _ANGLE_IDX:                      # unwrap angle differences
                dvec[a] = (dvec[a] + 180) % 360 - 180
            G[:, k] = dvec / (2*h)
        return G

    def _compute_elem_cov(self):
        G = self._elem_jacobian()
        return G @ self._state_cov @ G.T

    # --- elements + sigmas ----------------------------------------------
    def elements(self):
        """The osculating heliocentric elements at the epoch (:meth:`epoch_jd`, TDB).

        Two-body elements of the fitted state about the Sun, in the ecliptic and equinox of
        J2000 (mean obliquity 23.43928111 deg).

        Returns
        -------
        dict
            ``a`` semi-major axis (au; negative when unbound), ``e``, ``q`` perihelion and
            ``Q`` aphelion distance (au), ``i``, ``asc_node``, ``arg_per`` and ``M`` (deg; on
            an unbound orbit ``M`` is the hyperbolic mean anomaly, unwrapped), ``Tp`` time of
            perihelion (Julian date, TDB), ``n`` mean motion (deg/day), ``P`` period (years).
            ``Q`` and ``P`` are NaN on an unbound orbit.
        """
        return dict(zip(ELEM_NAMES, self._elem.tolist()))

    def element_sigmas(self):
        """The 1-sigma uncertainty of each of :meth:`elements`, from the linear covariance.

        The covariance is scaled by the fit's own residuals but never shrunk by them. Where
        the orbit is poorly constrained the linear sigma can be badly wrong: see
        :meth:`determinacy`, :meth:`element_ranges` and :meth:`monte_carlo_uncertainty`.

        Returns
        -------
        dict
            The same keys and units as :meth:`elements`.
        """
        sig = np.sqrt(np.clip(np.diag(self._elem_cov), 0, None))
        return dict(zip(ELEM_NAMES, sig.tolist()))

    def _val(self, name): return float(self._elem[ELEM_NAMES.index(name)])
    def _sig(self, name): return float(math.sqrt(max(0.0, self._elem_cov[ELEM_NAMES.index(name)][ELEM_NAMES.index(name)])))

    # individual getters (value, 1-sigma)
    def semimajor_axis(self):           return self._val("a"), self._sig("a")          # au
    def eccentricity(self):             return self._val("e"), self._sig("e")
    def perihelion_distance(self):      return self._val("q"), self._sig("q")          # au
    def aphelion_distance(self):        return self._val("Q"), self._sig("Q")          # au
    def inclination(self):              return self._val("i"), self._sig("i")          # deg
    def ascending_node(self):           return self._val("asc_node"), self._sig("asc_node")
    def argument_of_perihelion(self):   return self._val("arg_per"), self._sig("arg_per")
    def mean_anomaly(self):             return self._val("M"), self._sig("M")          # deg
    def perihelion_time(self):          return self._val("Tp"), self._sig("Tp")        # JD
    def mean_motion(self):              return self._val("n"), self._sig("n")          # deg/day
    def period(self):                   return self._val("P"), self._sig("P")          # yr

    # --- state + covariance getters -------------------------------------
    def state_vector(self):
        """Heliocentric equatorial (au, au/day): (position, velocity)."""
        return self._state[:3].copy(), self._state[3:].copy()
    def state(self):
        """The fitted state at the epoch (:meth:`epoch_jd`, TDB).

        Heliocentric, in the EQUATORIAL frame (ICRF axes) -- not the ecliptic frame of
        :meth:`elements`. A non-gravitational fit appends its free A coefficients
        (``fit.nongrav_params`` names them), so take ``state()[:6]`` for the orbit.

        Returns
        -------
        numpy.ndarray
            ``[x, y, z, vx, vy, vz, ...]`` in au and au/day. A copy.
        """
        return np.asarray(self._state, float).copy()
    def geocentric_range(self):
        """Geocentric distance (au) of the nominal orbit at the FIRST observation epoch, or
        None if the observer geometry isn't available. The external check on whether a short
        arc sits inside the range regime Statistical Ranging can trust. Public accessor so
        constraint.py need not reach into `_ctx["Robs"]`."""
        ctx = getattr(self, "_ctx", None)
        if not ctx or "Robs" not in ctx:
            return None
        try:
            R0 = np.asarray(ctx["Robs"])[0]
            return float(np.linalg.norm(self._state[:3] - R0))
        except Exception:
            return None
    def covariance_matrix(self, frame="elements"):
        """The linear covariance of the fit.

        Parameters
        ----------
        frame : {"elements", "state"}, default "elements"
            ``"elements"``: 11 x 11, in the order and units of :meth:`elements` (rank 6:
            ``q``, ``Q``, ``Tp``, ``n`` and ``P`` are functions of the other six).
            ``"state"``: 6 x 6 over the equatorial state of :meth:`state` (au, au/day),
            marginalised over any non-gravitational parameters;
            :meth:`parameter_covariance` keeps them.

        Returns
        -------
        numpy.ndarray
            A copy.
        """
        if frame not in ("elements", "state"):
            raise ValueError(f"frame must be 'elements' or 'state', not {frame!r}")
        return self._elem_cov.copy() if frame == "elements" else self._state_cov.copy()
    def correlation_matrix(self, frame="elements"):
        """:meth:`covariance_matrix` normalised to unit diagonal (same ``frame`` argument)."""
        C = self.covariance_matrix(frame); d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        return C / np.outer(d, d)

    def monte_carlo_uncertainty(self, n=1000, rng_seed=0):
        """Non-linear element uncertainty by Monte Carlo: sample the state covariance,
        convert each clone to osculating elements, take the per-element spread (find_orb's
        covariance-sampling MC, see mc.py). Returns {"sigmas": {name: (median, sigma)},
        "U": uncertainty_parameter, "n": n_clones} -- the sigmas agree with the linear
        element sigmas for a well-determined orbit and diverge (more honestly) when the
        orbit is poorly constrained. Cached per (n, rng_seed)."""
        from pyroorb import mc
        key = (n, rng_seed)
        cache = self.__dict__.setdefault("_mc_cache", {})
        if key not in cache:
            sigmas, U, samples, bf = mc.monte_carlo_elements(
                self._state, self._state_cov, self.epoch_tdb,
                element_vector, ELEM_NAMES, n=n, rng_seed=rng_seed)
            # `samples` (clone element vectors) is kept so a consumer can read the honest
            # non-linear spread directly -- the per-element percentiles the unconstrained-orbit
            # banner needs. `bound_fraction` is the energy-sign fraction over ALL drawn clones
            # (mc._bound_fraction) -- the honest bound-vs-unbound number, NOT recomputed over the
            # surviving element rows (which drop the unbound clones). See constraint.assess.
            cache[key] = None if sigmas is None else {"sigmas": sigmas, "U": U,
                                                      "n": len(samples), "samples": samples,
                                                      "bound_fraction": bf}
        return cache[key]

    def element_ranges(self, pct=(5.0, 95.0), n=1000, rng_seed=0):
        """Per-element ``{name: (lo, hi)}`` percentile interval over the Monte-Carlo clone
        family (`monte_carlo_uncertainty`'s ``samples``), or None when the MC declines.

        This is the honest product for an orbit whose SCALE is free (constraint.assess): a
        nominal value +/- a linear sigma is meaningless there -- the sigma is the tangent to a
        valley the solution is not confined to -- while the percentile span of the clones says
        what the astrometry actually allows. Angles are unwrapped about the median before the
        percentile so a family straddling 0/360 deg reports its real width, not [1, 359].

        Caveat the caller must carry: a clone that converts to no finite element vector (an
        unbound one, usually) is not in `samples`, so on a family that is part hyperbolic this
        interval describes the BOUND part only -- read it beside `bound_fraction`.
        """
        from pyroorb import mc
        mcu = self.monte_carlo_uncertainty(n=n, rng_seed=rng_seed)
        rows = None if not mcu else mcu.get("samples")
        if rows is None:
            # The MC DECLINED -- too few clones converted to finite elements. That is not a
            # reason to have no interval; it is the regime the interval is for. A free range on
            # a tracklet makes most of the cloud hyperbolic, and a hyperbolic clone has no
            # osculating a/e/Q to average, so `monte_carlo_elements` rightly refuses to publish
            # a SIGMA -- while the orbits that did convert still say what the bound part spans.
            # Draw the same cloud and keep whatever it yields (bound_fraction says how much of
            # the family that is).
            rows = mc.element_rows_from_cov(self._state, self._state_cov, self.epoch_tdb,
                                            element_vector, n=n, rng_seed=rng_seed)
        return mc.element_percentiles(rows, ELEM_NAMES, pct=pct)

    def bound_fraction(self, n=300, rng_seed=0):
        """Fraction of covariance-sampled clones that stay gravitationally bound (sign of the
        specific orbital energy). Honest even when monte_carlo_uncertainty declines on the
        element spread -- the degenerate/unbound regime where bound-vs-unbound is the question."""
        from pyroorb import mc
        return mc.bound_fraction_from_cov(self._state, self._state_cov, n=n, rng_seed=rng_seed)

    def range_profile(self, n_rho=None):
        """The RANGE PROFILE of the included arc (rangescan.profile) -- a profile that could
        not be run comes back in the same shape with no rows and a `failure` (rangescan.failed),
        never None: the verdict has to be able to say "could not measure".

        chi2 as a function of the line-of-sight distance, each node re-fitted over the other
        five degrees of freedom: the direct measurement of whether this arc pins the range at
        all. Asked only where it is the question -- a tracklet-length arc, where the local
        covariance has been shown to be confidently wrong (measured: 18 of the 24 fixture tracklets
        it called determined were >5% off in distance) -- but the
        computation itself has no arc-length assumption in it.

        Cached on the shared ctx keyed by the inclusion mask, exactly as `sr_uncertainty` is
        and for the same reason: every interactive re-solve builds a FRESH FittedOrbit, so an
        instance-only cache never hits and the profile would re-run on every web click.
        """
        mask = np.asarray(self._mask, bool)
        ctx = getattr(self, "_ctx", None)
        key = (mask.tobytes(), int(n_rho or 0))
        shared = ctx.setdefault("_rangeprof_cache", {}) if isinstance(ctx, dict) else None
        if shared is not None and key in shared:
            return shared[key]
        mine = self.__dict__.get("_rangeprof_cache")
        if isinstance(mine, dict) and key in mine:
            return mine[key]

        def _remember(info):
            self.__dict__.setdefault("_rangeprof_cache", {})[key] = info
            if shared is not None:
                if len(shared) > 32:
                    shared.clear()
                shared[key] = info
            return info

        from pyroorb import rangescan; import pyroorb.sr as _sr
        _remember(rangescan.failed("the range profile did not run"))   # unless we succeed below
        sub = [o for o, keep in zip(self._obs, mask) if keep]
        if len(sub) < 3:
            return _remember(rangescan.failed(f"only {len(sub)} observations are included",
                                              n_obs=len(sub)))
        # The fit's OWN weights where they are available (ctx sigma_ra/sigma_dec are
        # positionally aligned with _obs), else the per-record rule -- the same choice
        # sr_uncertainty makes, for the same reason: the profile's chi2 has to be the fit's.
        ctxd = ctx if isinstance(ctx, dict) else {}
        sra, sdec = ctxd.get("sigma_ra"), ctxd.get("sigma_dec")
        if sra is not None and sdec is not None and len(sra) == len(mask) == len(sdec):
            sig = np.column_stack([np.asarray(sra, float)[mask], np.asarray(sdec, float)[mask]])
        else:
            sig = _sr.obs_sigmas(sub)
        try:
            prof = rangescan.profile(sub, sig, seed_state=self._range_seed_state(sub),
                                     **({"n_rho": n_rho} if n_rho else {}))
        except Exception as ex:
            prof = rangescan.failed(f"the range profile raised {type(ex).__name__}",
                                    n_obs=len(sub))
        return _remember(prof)

    def _range_seed_state(self, sub):
        """This fit's heliocentric state at the epoch the range profile works in (the first
        included observation, TDB), or None. The profile takes it as a known solution so that
        its minimum can never be worse than the orbit it is asked to judge (rangescan.profile).
        Usually the fit epoch IS that observation and nothing is integrated."""
        try:
            from astropy.time import Time
            t0 = float(Time(sub[0]["jd_utc"], format="jd", scale="utc").tdb.jd)
            st = np.asarray(self._state, float)
            if abs(t0 - float(self.epoch_tdb)) * 86400.0 < 1e-3:
                return st
            from pyroorb.forward_model import predict_state
            from pyroorb.iod import build_base_sim
            return np.asarray(predict_state(build_base_sim(self.epoch_tdb), self.epoch_tdb,
                                            st, t0), float)[:6]
        except Exception:
            return None

    def range_verdict(self):
        """``rangescan.verdict`` for this fit's own profile. Says whether the astrometry
        MEASURES the distance, and if not, in what words -- including "it could not be
        measured" (`profiled` False), which is a verdict too and never None."""
        from pyroorb import rangescan
        prof = self.range_profile() or rangescan.failed("the range profile did not run")
        try:
            return rangescan.verdict(prof, rho_fit=self.geocentric_range())
        except Exception as ex:
            return rangescan.verdict(rangescan.failed(f"the verdict raised {type(ex).__name__}",
                                                      n_obs=prof.get("n_obs", 0)))

    def sr_uncertainty(self, max_orbits=300):
        """Statistical-Ranging uncertainty for a SHORT arc: sample the (range, range-rate)
        plane GLOBALLY (sr.py) and report the spread of the orbits that fit the angles.

        Why, when we already have the covariance MC: the MC samples the LOCAL covariance,
        which for a single night can be badly OVERCONFIDENT -- the linear fit lands in a tight
        valley and the MC then reports a small spread, hiding the true range ambiguity (see
        tests/test_sr_mc.py::test_sr_family_sigma_is_large). SR walks the whole admissible
        range, so it sees that under-determination. The trade-off: SR's two-body trial family
        is LOOSER than the true n-body posterior, so its spread is an upper bound -- the caller
        treats it conservatively (upgrade only on a clearly-large family).

        Returns a JSON-ish dict or None (SR couldn't characterise the arc -- too few accepting
        orbits, or outside SR's range regime, e.g. a distant TNO whose geometry the ported
        escape test rejects). Only meaningful for short arcs; the caller gates on arc length.
        Cached. Reconstructs SR's inputs the way tests/test_sr_mc drives it (observer positions
        + a base-sim at the first-obs TDB epoch), which is the validated path.

        Ranges the INCLUDED observations only. The caller (constraint._sr_assess) decides to
        ask on the strength of the fit's INCLUDED arc -- "6 days, one apparition, so the local
        covariance may be lying about the range" -- so SR has to be shown that same arc. Fed
        the whole obs list instead, it answers a question nobody asked, about observations the
        user switched off, and it does it slowly: on a 2-apparition object with one apparition
        deselected (included arc 3 d, full arc 15 yr) every one of the 300 two-body trials was
        integrated across the full 15 years -- ~11 s per web payload -- and the family that came
        back was then too small to pass constraint.SR_MIN_TRUST, so the answer was discarded.
        (constraint now consults SR only on a single-tracklet arc, for its mode count.)
        With the mask honoured the cost follows the arc SR was asked about."""
        mask = np.asarray(self._mask, dtype=bool)
        # Cache on the shared per-fit ctx, not just on self: every interactive re-solve (web
        # Full step, obs toggle, model switch) builds a FRESH FittedOrbit, so an instance-only
        # cache never hits and the banner re-ran SR on every click. Keyed by the inclusion mask
        # (the inputs) so a mask change recomputes rather than returning another mask's family.
        # The per-INSTANCE fallback (for a fit with no shared ctx) is keyed the same way:
        # `self._mask` is not frozen -- the rejection loop mutates it -- so an unkeyed
        # instance cache answers a second question with the first one's family, which is
        # exactly the staleness the shared key exists to prevent.
        ctx = getattr(self, "_ctx", None)
        key = (mask.tobytes(), int(max_orbits))
        shared = ctx.setdefault("_sr_cache", {}) if isinstance(ctx, dict) else None
        if shared is not None and key in shared:
            return shared[key]
        mine = self.__dict__.get("_sr_cache")
        if isinstance(mine, dict) and key in mine:
            return mine[key]

        def _remember(info):
            self.__dict__.setdefault("_sr_cache", {})[key] = info
            if shared is not None:
                if len(shared) > 32:               # bounded: a long toggling session can't grow it
                    shared.clear()
                shared[key] = info
            return info

        _remember(None)                            # default unless we succeed below
        if ctx is None:
            return None
        sub = [o for o, keep in zip(self._obs, mask) if keep]
        if len(sub) < 3:
            return None
        import pyroorb.sr as _sr; import pyroorb.mc as _mc; import pyroorb.obs_sources as _os
        try:
            times = np.array([o["jd_utc"] for o in sub])
            rhohats = np.array([los_unit(o["ra_deg"], o["dec_deg"]) for o in sub])
            Robs = np.array(list(_os.observer_positions_batch(
                [o["jd_utc"] for o in sub], [o["obscode"] for o in sub], _os.load_obscodes())))
            t0 = Time(times[0], format="jd", scale="utc").tdb.jd
            # SR accepts a trial on the SIGMA-weighted rms, so hand it the weights the fit
            # itself used (ctx sigma_ra/sigma_dec are positionally aligned with self._obs);
            # if that alignment is ever broken, fall back to the per-record rule.
            sra, sdec = ctx.get("sigma_ra"), ctx.get("sigma_dec")
            if sra is not None and sdec is not None and len(sra) == len(mask) == len(sdec):
                sigmas = np.column_stack([np.asarray(sra, float)[mask],
                                          np.asarray(sdec, float)[mask]])
            else:
                sigmas = None
            med, kept, n = _sr.statistical_ranging(
                sub, times, rhohats, Robs, build_base_sim(t0), t0, max_orbits=max_orbits,
                sigmas=sigmas)
        except Exception:
            return None
        if med is None or kept is None or len(kept) < _sr.SR_MIN_ACCEPT:
            return None
        states = np.asarray(kept)
        # Element rows of the accepted family (at the SR epoch t0) -> honest median/sigma and
        # 5-95 percentiles per element; angles unwrapped before the spread.
        rows = []
        for s in states:
            v = element_vector(s, t0)
            if np.all(np.isfinite(v)):
                rows.append(v)
        if len(rows) < _sr.SR_MIN_ACCEPT:
            return None
        rows = np.asarray(rows)
        ang = {i for i, nm in enumerate(ELEM_NAMES) if nm in ("i", "asc_node", "arg_per", "M")}
        sig, rng = {}, {}
        for j, nm in enumerate(ELEM_NAMES):
            col = rows[:, j]
            med_v = float(np.median(col))
            d = ((col - med_v + 180.0) % 360.0 - 180.0) if j in ang else col
            sig[nm] = (med_v, float(np.std(d)))
            rng[nm] = (float(np.percentile(col, 5)), float(np.percentile(col, 95)))
        geo = np.linalg.norm(states[:, :3] - Robs[0], axis=1)
        info = {"sigmas": sig, "ranges": rng, "n": int(len(rows)),
                "bound_fraction": _mc._bound_fraction(states),
                "geo_range": (float(geo.min()), float(geo.max())),
                "n_modes": _count_range_modes(geo)}
        return _remember(info)

    def predict_without(self, select, max_nfev=200):
        """Leave the observations ``select(obs)`` picks out of this fit and test whether the
        rest PREDICTS them, or whether they were only fitted because the orbit bent to take
        them.

        Re-converged WARM, from this fit's own state, over the remaining included
        observations; the left-out ones are then scored against that relaxed orbit:
        ``r`` their residuals in sigma units (two per observation), ``C`` the relaxed
        orbit's covariance ``inv(J^T J)``, and ``chi2 = r^T (J_t C J_t^T + I)^-1 r`` on
        ``2n`` degrees of freedom -- prediction uncertainty plus their own noise. Warm,
        never a fresh solution: re-running orbit determination on a sparse multi-opposition
        remainder can settle in another minimum, and then its prediction says nothing about
        the observations left out.

        Returns ``{"chi2", "dof", "p", "miss"}`` (``miss`` the mean offset, arcsec) or
        ``{"error": ...}`` when the test cannot be made: no live evaluator, a
        non-gravitational state, a selection this fit does not wholly include, or too
        little left. Never raises, and never changes this fit."""
        from scipy.stats import chi2 as _chi2
        ctx = getattr(self, "_ctx", None)
        if ctx is None or ctx.get("ev") is None:
            return {"error": "the fit carries no evaluator"}
        state0 = np.asarray(self._state, float)
        if state0.size != 6:
            return {"error": "a non-gravitational state"}
        ev = ctx["ev"]
        base = np.asarray(self._mask, bool)
        out = np.array([bool(select(o)) for o in self._obs], bool)
        if not out.any():
            return {"error": "no observation selected"}
        if not base[out].all():
            return {"error": "the fit does not include every selected observation"}
        rest = base & ~out
        if rest.sum() < 4:
            return {"error": "too little left to fit"}
        try:
            x, _res = ev.fit(state0.copy(), rest, max_nfev=max_nfev, thin=False)
            J = np.asarray(ev.evaluate(x, rest).jac, float)
            C = np.linalg.pinv(J.T @ J)
            e = ev.evaluate(x, out)
            r = np.asarray(e.fun, float)
            Jo = np.asarray(e.jac, float)
            q = float(r @ np.linalg.solve(Jo @ C @ Jo.T + np.eye(r.size), r))
            da, dd = ev.resid_arcsec(x)
            miss = float(np.hypot(np.mean(np.asarray(da)[out]), np.mean(np.asarray(dd)[out])))
        except Exception as ex:                          # noqa: BLE001 -- a test, never a crash
            return {"error": f"{type(ex).__name__}: {ex}"}
        return {"chi2": q, "dof": int(r.size), "p": float(_chi2.sf(q, r.size)), "miss": miss}

    def lov_uncertainty(self, n_steps=21, n_sigma=3.0):
        """Non-linear uncertainty along the Line of Variations: step the orbit through the
        least-constrained direction of its covariance and RE-CORRECT each variant on the LOV
        hyperplane (find_orb improve_along_lov; see lov.py). Returns {"sigmas": {name:
        (median, sigma)}, "U": uncertainty_parameter, "n": n_variants, "variants": (m,6)
        re-corrected states, "k": sigma-offsets} or None. Unlike the Monte-Carlo cloud (mc.py)
        this follows the CURVED valley of best-fit orbits, so it stays honest farther from the
        nominal than either the linear sigmas or a moderate MC. Needs the live evaluator
        (self._ctx); returns None if absent. Cached per (n_steps, n_sigma)."""
        from pyroorb import lov
        import pyroorb.arc_extend as _ae
        ctx = getattr(self, "_ctx", None)
        if ctx is None or ctx.get("ev") is None:
            return None
        if np.asarray(self._state).size != 6:
            # Nongrav-augmented fit (6 orbital + A-terms): the LOV-over-augmented-params
            # path is a later layer (and the existing covariance MC has the same gap), so
            # decline gracefully rather than mix 6/7-D states. TODO: LOV over the full
            # solved-parameter space for nongrav NEOs.
            return None
        ev, mask, state0 = ctx["ev"], self._mask, self._state
        key = (n_steps, round(n_sigma, 3))
        cache = self.__dict__.setdefault("_lov_cache", {})
        if key not in cache:
            refit_fn = lambda seed, u, sigma, k: _ae.fit_lov_pinned(ev, seed, mask, state0, u, sigma, k)
            # A nongrav fit carries an augmented state (6 orbital + A-terms): the LOV runs over
            # the full solved-parameter space (so the period direction stays correct), but only
            # the 6 orbital components map to osculating elements.
            elem_fn = lambda v, ep: element_vector(np.asarray(v, float)[:6], ep)
            sigmas, U, info = lov.lov_variants(
                self._state, self._state_cov, self.epoch_tdb,
                refit_fn, elem_fn, ELEM_NAMES, n_steps=n_steps, n_sigma=n_sigma)
            cache[key] = (None if sigmas is None else
                          {"sigmas": sigmas, "U": U, "n": len(info["elements"]),
                           "variants": info["states"], "k": info["k"]})
        return cache[key]

    def lov_refit(self):
        """Shared LOV machinery for downstream consumers (impact probability, close-approach
        analysis): returns ``(u, sigma, refit_fn)`` where ``u`` is the unit LOV direction,
        ``sigma`` its 1-sigma state extent, and ``refit_fn(seed, u, sigma, k) -> variant_state``
        re-corrects the orbit pinned at LOV coordinate k*sigma (arc_extend.fit_lov_pinned).
        Lets a caller sample the LOV at ARBITRARY k (denser/wider than ``lov_uncertainty``'s
        fixed grid). Returns ``(None, None, None)`` if the LOV is degenerate, the live evaluator
        is absent, or the state is nongrav-augmented (the 7-D LOV-IP path is a later layer)."""
        from pyroorb import lov
        import pyroorb.arc_extend as _ae
        ctx = getattr(self, "_ctx", None)
        if ctx is None or ctx.get("ev") is None or np.asarray(self._state).size != 6:
            return None, None, None
        u, sigma = lov.lov_direction(self._state_cov)
        if u is None:
            return None, None, None
        ev, mask, state0 = ctx["ev"], self._mask, self._state
        refit_fn = lambda seed, u_, sigma_, k: _ae.fit_lov_pinned(ev, seed, mask, state0, u_, sigma_, k)
        return u, sigma, refit_fn

    # --- fit quality / metadata (counts reflect INCLUDED obs, like find_orb) ---
    def epoch_jd(self):
        """The epoch of :meth:`elements` and :meth:`state`, as a Julian date in TDB (not UTC;
        also ``fit.epoch_tdb``)."""
        return self.epoch_tdb
    def epoch_iso(self):
        """:meth:`epoch_jd` as an ISO string, TDB."""
        return Time(self.epoch_tdb, format="jd", scale="tdb").iso
    def frame(self):                    return "ecliptic J2000 (heliocentric)"
    def _incl_obs(self):                return [o for o, m in zip(self._obs, self._mask) if m]
    def included_observations(self):
        """The observation records the fit kept (the rejected ones are left out)."""
        return self._incl_obs()               # public: leaf modules
    def n_observations(self):
        """How many optical observations the fit kept."""
        return int(self._mask.sum())          # included
    def n_total_observations(self):
        """How many optical observations the fit was given, kept or rejected."""
        return len(self._obs)                  # incl. rejected
    def n_excluded(self):
        """How many optical observations the fit rejected."""
        return int((~self._mask).sum())
    def n_debiased(self):               return sum(1 for o in self._obs if o.get("debiased"))
    def n_resids(self):                 return 2*int(self._mask.sum())
    def n_free_params(self):
        """Free parameters of the delivered solution: the 6-element state, plus one per fitted
        non-gravitational coefficient, minus one per HELD-FIXED element (refit_constrained
        pins those, so they cost no degree of freedom even though the solver carries them as
        penalty rows rather than as a reduced parameter vector)."""
        npar = int(np.asarray(self._state).size) - len(self.constraints)
        return max(1, npar)
    def degrees_of_freedom(self):
        """Statistical dof of the fit: residual ROWS minus free parameters, ``2n - npar``.

        Each optical observation supplies two rows (RA, Dec), so a plain 6-element fit is
        exactly determined at n=3 (dof 0) -- the same arithmetic find_orb uses
        (``n_included_observations * 2 - n_params``, orb_func.cpp). At dof <= 0 the residuals
        vanish identically and there is no scale to estimate: `_compute_state_cov` clamps its
        divisor to 1 so nothing raises, but the sigmas it then reports are arbitrary, not
        measured. Reported (never refused) so the UI can say so -- see the zero-dof banner."""
        return self.n_resids() - self.n_free_params()
    def n_oppositions(self):
        """How many oppositions (apparitions) the kept observations span (find_orb's rule)."""
        return n_oppositions([o["jd_utc"] for o in self._incl_obs()])
    def arc_length_days(self):
        """Days from the first kept observation to the last."""
        # sorted() so the span is correct even if _obs is not chronological (every
        # normal entry path sorts by jd_utc, but a paste "Add to current fit" merges
        # base+added without re-sorting) -- see loader.add_to_fit.
        ti = sorted(o["jd_utc"] for o in self._incl_obs())
        return ti[-1] - ti[0]
    def observatory_codes(self):
        """The MPC codes of the stations whose observations the fit kept, sorted."""
        return sorted({o["obscode"] for o in self._incl_obs()})
    def _compute_rational_sigmas(self):
        """Lazily compute the profile-likelihood (Delta-chi2=1) 1-sigma for each fitted nongrav
        coefficient, from this fit's own state / obs / backend (see _profile_nongrav_sigmas). Cached.
        Returns (sigmas_dict, unconstrained_list, intervals_dict) where sigmas_dict is the symmetric
        (conservative max-half-width) sigma per axis and intervals_dict is {axis: (minus, plus)} -- the
        honest asymmetric Delta-chi2=1 half-widths. Falls back to the linear sigma (and a symmetric
        interval) if anything is missing (e.g. a fit rebuilt without a live _ctx)."""
        import pyroorb.nongrav as ng
        if self._rational_sigmas is not None:
            return self._rational_sigmas

        def _fallback():
            lin = dict(self.nongrav_sigmas)
            return (lin, [], {k: (v, v) for k, v in lin.items()})   # symmetric interval from the linear sigma
        ctx = getattr(self, "_ctx", None)
        if (not self._nongrav_profile or ctx is None or self.nongrav_model in (None, "none")
                or not hasattr(ctx.get("backend"), "predict_radec_jac_ng")):
            self._rational_sigmas = _fallback()
            return self._rational_sigmas
        try:
            spec = ng.model(self.nongrav_model); gr = spec["gr"]; scale = spec["scale"]
            free_idx = ng.free_indices(self.nongrav_model)
            ev = ctx["ev"]; backend = ctx["backend"]; t0 = ctx["epoch_tdb"]
            idx = np.flatnonzero(self._mask)
            tt = ev.times_tdb[idx]; Rs = [ev.Robs[i] for i in idx]
            ra = ev.ra[idx]; dec = ev.dec[idx]; cosd = ev.cosd[idx]
            sra = ev.fit_sig_ra[idx]; sdec = ev.fit_sig_dec[idx]
            state6 = np.asarray(self._state, float)[:6]
            A_phys = [0.0, 0.0, 0.0]
            for ax in range(3):
                A_phys[ax] = float(self.nongrav_params.get(ng.A_AXES[ax], 0.0))
            # chi2 at the solution (obs-only, same weighting/clamp as the profile helper)
            pred, _ = backend.predict_radec_jac_ng(t0, state6, A_phys, [], gr, tt, Rs, hA=np.empty(0))
            d = ((pred[:, 0] - ra + 180) % 360 - 180) * cosd
            r0 = _clamp(np.column_stack([d*3600.0/sra, (pred[:, 1]-dec)*3600.0/sdec]).ravel())
            chi2_min = float(np.sum(r0**2))
            lin = [float(self.nongrav_sigmas.get(ng.A_AXES[a], scale)) for a in free_idx]
            prof = _profile_nongrav_sigmas(backend, t0, state6, A_phys, free_idx, gr, scale,
                                           tt, Rs, ra, dec, cosd, sra, sdec, chi2_min, lin_sig=lin)
            sig = dict(self.nongrav_sigmas); unc = []
            intervals = {k: (v, v) for k, v in self.nongrav_sigmas.items()}
            for axis in free_idx:
                w_lo, w_hi, is_unc = prof[axis]
                nm = ng.A_AXES[axis]
                sig[nm] = float(max(w_lo, w_hi))                 # symmetric = conservative half-width
                intervals[nm] = (float(w_lo), float(w_hi))       # honest asymmetric (minus, plus)
                if is_unc:
                    unc.append(nm)
            self._rational_sigmas = (sig, unc, intervals)
        except Exception:
            self._rational_sigmas = _fallback()
        return self._rational_sigmas

    def _compute_jackknife_sigmas(self):
        """Lazily compute the delete-one-tracklet JACKKNIFE robustness of each fitted nongrav
        coefficient, on this fit's OWN mask. Returns {axis: {se, z, spans_zero, assessed, n,
        n_tracklets, base, rejected}} ({} when it cannot run at all -- no model / too few tracklets
        / no live ctx). For each tracklet dropped in turn, the coefficient is re-fit INDEPENDENTLY
        (re-seeded from A=0, NOT warm-started from the full solution, which would pin the estimate on
        the flat valley); the spread of those refits is the robustness sigma.

        FIVE things make it honest rather than noisy -- each learned the hard way (2007 AA2 read a
        clear A1 detection as +-7e-9 before the first three):
          1. A single-coefficient model (srp A1-only, yarkovsky A2-only) DIVERGES from a bare A=0
             start -- it overshoots into a false basin of the A-vs-semimajor-axis valley. So a refit
             that lands far from the base fit (normalized rms > JACKKNIFE_REJECT_FACTOR x the base's)
             is REJECTED as non-converged: removing one tracklet cannot legitimately make the best
             fit that much worse, so it is optimizer noise, not evidence. Divergent refits get a
             second try with the forced profile seed (`force_seed`) -- and on a single-coefficient
             model that is nearly every refit, so after two divergences in a row the plain LM is
             skipped and the seed is bought straight away.
          2. A refit that STALLED is not a refit that found zero. An exact 0.0 with nothing to
             explain it is dropped rather than believed -- see _jackknife_unexplained_zeros for why
             the rms bar cannot see this and why the diag tells the two cases apart for free.
          3. The scale is the ROBUST MAD (1.4826 x median-abs-deviation), not the standard deviation
             -- one high-leverage tracklet genuinely moves a real coefficient's VALUE (2007 AA2's
             G96 tracklet carries much of its A1) without threatening the DETECTION, and the std lets
             that one point wreck the sigma while the MAD does not.
          4. The "not robust" verdict is the SIGN spanning zero across the converged refits -- the
             detection itself is fragile -- NOT a low z from the value moving. A coefficient can have
             real value-leverage and still be a rock-solid, always-same-sign detection.
          5. The scale's sqrt(n-1) counts TRACKLETS, not refits -- in _jackknife_stats, which is a
             pure function precisely so these can be tested without ASSIST.

        If fewer than JACKKNIFE_MIN_KEEP refits converge for an axis, or more than a third of the
        attempts are rejected, its robustness is reported as NOT ASSESSED (assessed=False) rather
        than as a garbage sigma.

        COST. Up to JACKKNIFE_MAX_REFITS refits: 0.2-0.6 s each where the plain LM converges, ~2.7 s
        where the model needs the profile seed (a single-coefficient model needs it on nearly every
        subsample, which is why the plain LM is dropped after two strikes). It stays lazy either
        way: only an explicit request (CLI flag / API / UI button) pays it, never the fit and never
        the regression suite."""
        import pyroorb.nongrav as ng
        if self._jackknife_sigmas is not None:
            return self._jackknife_sigmas
        self._jackknife_sigmas = {}
        ctx = getattr(self, "_ctx", None)
        model = self.nongrav_model
        if (ctx is None or model in (None, "none") or not self.nongrav_params
                or not hasattr(ctx.get("backend"), "predict_radec_jac_ng")):
            return self._jackknife_sigmas
        try:
            ev = ctx["ev"]
            mask = np.asarray(self._mask, bool)
            idx = np.flatnonzero(mask)
            if len(idx) < JACKKNIFE_MIN_TRACKLETS:
                return self._jackknife_sigmas
            times = np.asarray(ev.times_tdb)[idx]
            stn = ev.obscodes[idx] if getattr(ev, "obscodes", None) is not None else None
            order = np.argsort(times)
            groups_local = _split_tracklets_by_station(times[order], stn[order] if stn is not None else None)
            tracklets = [[int(idx[order[j]]) for j in g] for g in groups_local]
            n_trk = len(tracklets)
            if n_trk < JACKKNIFE_MIN_TRACKLETS:
                return self._jackknife_sigmas
            free = ng.free_indices(model)
            axes = [ng.A_AXES[a] for a in free]
            base = {a: float(self.nongrav_params.get(a, 0.0)) for a in axes}
            base_rn = self.rms_normalized()
            reject_rn = max(JACKKNIFE_REJECT_FACTOR * base_rn,
                            base_rn + JACKKNIFE_REJECT_FLOOR)
            picks = _jackknife_sample(tracklets)
            samples = {a: [] for a in axes}
            attempted = rejected = forced = zero_dropped = skipped = 0
            # A SINGLE-coefficient model (srp A1-only, yarkovsky A2-only) is degenerate enough that
            # the plain LM overshoots the A-vs-a valley on essentially every subsample -- 2008 NA
            # pays 0.58 s for a diverged plain fit before all 21 of its forced seeds. Two diverging
            # plains in a row is that object saying so, so stop buying them. It can only cost the
            # forced seed we were about to pay anyway, and keep-the-better still governs the result.
            skip_plain = False
            strikes = 0
            for ti in picks:
                trk = tracklets[ti]
                sub = mask.copy(); sub[trk] = False
                if int(sub.sum()) < max(JACKKNIFE_MIN_TRACKLETS, 6):
                    # Deleting this tracklet leaves too little arc to refit at all. COUNTED:
                    # it is neither attempted nor rejected, so without this it vanished from
                    # every coverage number, and an arc where all tracklets were picked but
                    # one was undeletable reported `sampled=False` -- which the UI renders as
                    # "the sign survives dropping ANY one tracklet" for a tracklet that was
                    # never tried. Same overclaim as a silent sample, by a different door.
                    skipped += 1
                    continue
                attempted += 1
                try:
                    # ADAPTIVE independent refit: a plain LM (fast -- 0.2-0.6 s on the suite's
                    # objects) seeded from THIS delivered 6-state converges for a well-behaved /
                    # multi-coefficient model. TWO signatures say it did not, and they are different
                    # failures: the rms (it landed far from the base fit -- single-axis overshoot),
                    # and an UNEXPLAINED exact zero (the coefficient never left its seed), which the
                    # rms cannot see at all -- see _jackknife_unexplained_zeros. Either buys the
                    # forced profile seed (`force_seed`, ~2.7 s), which grids the free axis from
                    # zero and cannot overshoot.
                    f, needs_seed = None, True
                    if not skip_plain:
                        f = refine_nongrav(self, model, mask=sub, revert_if_worse=False,
                                           profile_seed=False, profile_sigmas=False)
                        retry_z, dead_z = _jackknife_unexplained_zeros(f, model, axes)
                        needs_seed = bool(f.rms_normalized() > reject_rn or retry_z)
                        strikes = strikes + 1 if needs_seed else 0
                        skip_plain = strikes >= JACKKNIFE_PLAIN_STRIKES
                    if needs_seed:
                        forced += 1
                        f = refine_nongrav(self, model, mask=sub, revert_if_worse=False,
                                           force_seed=True, profile_sigmas=False)
                        retry_z, dead_z = _jackknife_unexplained_zeros(f, model, axes)
                    if f.rms_normalized() > reject_rn:   # still not converged -> not evidence
                        rejected += 1
                        continue
                    if retry_z or dead_z:                # a stall nothing explained -> not evidence
                        rejected += 1; zero_dropped += 1
                        continue
                    for a in axes:
                        samples[a].append(float(f.nongrav_params.get(a, 0.0)))
                except Exception:
                    rejected += 1
                    continue
            for a in axes:
                d = _jackknife_stats(samples[a], base[a], n_trk, attempted, rejected)
                # `sampled` (partial coverage) is set by _jackknife_stats from the counts;
                # `skipped` is reported beside it so the two reasons stay distinguishable.
                d.update(skipped=int(skipped), forced=int(forced),
                         zero_dropped=int(zero_dropped))
                self._jackknife_sigmas[a] = d
        except Exception:
            self._jackknife_sigmas = {}
        return self._jackknife_sigmas

    def jackknife_nongrav_sigmas(self):
        """{axis: jackknife 1-sigma} for the ASSESSED axes -- the delete-one-tracklet robustness
        sigma (see _compute_jackknife_sigmas). Omits axes that could not be assessed; empty when it
        cannot run at all. Lazy."""
        return {a: d["se"] for a, d in self._compute_jackknife_sigmas().items() if d.get("assessed")}

    def nongrav_jackknife_detail(self):
        """Full per-axis jackknife record {axis: {se, z, spans_zero, flip, n, base, ...}} for
        display -- the honest 'is this coefficient robust to dropping a tracklet' signal."""
        return {a: dict(d) for a, d in self._compute_jackknife_sigmas().items()}

    @property
    def nongrav_not_robust(self):
        """ASSESSED axes whose leave-one-tracklet cloud spans zero -- the DETECTION's sign does not
        survive dropping a single tracklet (a systematic or a lone high-leverage tracklet, not a
        force). The covariance / profile sigma cannot see this."""
        return [a for a, d in self._compute_jackknife_sigmas().items()
                if d.get("assessed") and d.get("spans_zero")]

    def rational_nongrav_sigmas(self):
        """The DISPLAYED nongrav 1-sigma: the profile-likelihood interval (Delta-chi2=1), computed
        lazily on first call. This is the honest uncertainty -- it correctly reports a degenerate
        coefficient as huge/unconstrained where the linear covariance sigma (self.nongrav_sigmas)
        gives a fake-tight symmetric error. Use this for any user-facing +/- (CLI / as_dict / UI)."""
        return dict(self._compute_rational_sigmas()[0])

    def rational_nongrav_intervals(self):
        """The honest ASYMMETRIC profile interval per coefficient: {axis: (minus, plus)}. On a
        degenerate valley the -/+ Delta-chi2=1 half-widths genuinely differ; rational_nongrav_sigmas
        collapses them to the conservative max, this keeps both for display (e.g. '+a / -b')."""
        return dict(self._compute_rational_sigmas()[2])

    @property
    def nongrav_unconstrained(self):
        """Coefficient axes whose profile interval never reached Delta-chi2=1 (unconstrained /
        degenerate); their sigma is only a lower bound. Presentation shows ">" instead of "+/-"."""
        return list(self._compute_rational_sigmas()[1])

    def rms_residual(self):
        """Unweighted RMS of O-C over the kept observations, arcsec, PER AXIS (the find_orb,
        MPC and JPL convention: the RA cos(dec) and Dec residuals pooled)."""
        if self._resid_all is None:
            return float(np.sqrt(np.mean(self._res.fun**2)))
        r = self._resid_all[self._mask]
        return float(np.sqrt(np.mean(r**2)))
    def rms_normalized(self):
        """RMS of (O-C)/sigma over included obs — dimensionless; ~1 means the fit is
        consistent with the assigned per-obs uncertainties (find_orb's metric)."""
        if self._resid_all is None:
            return float(np.sqrt(np.mean(self._res.fun**2)))
        m = self._mask
        v = np.concatenate([self._resid_all[m, 0] / self._sig_ra[m],
                            self._resid_all[m, 1] / self._sig_dec[m]])
        return float(np.sqrt(np.mean(v**2)))
    def is_included(self):              return self._mask.copy()
    # --- photometry ------------------------------------------------------
    def absolute_magnitude(self):
        """``(H, sigma_H)``: absolute magnitude in V on the H,G system with G fixed at
        :meth:`slope_parameter`, each band converted to V (find_orb's band table).
        ``(None, None)`` without usable photometry."""
        return self._phot.get("H"), self._phot.get("sigma_H")
    def slope_parameter(self):
        """The G slope parameter used for :meth:`absolute_magnitude` (fixed, 0.15)."""
        return photometry.ASTEROID_G
    def n_magnitudes(self):
        """How many reported magnitudes went into :meth:`absolute_magnitude`."""
        return self._phot.get("n_mags", 0)
    def magnitude_residuals(self):
        """Per-observation photometric residuals, for observations that reported a magnitude.

        Returns
        -------
        list of dict
            ``jd_utc``, ``obscode``, ``obs_mag``, ``band``, ``computed_mag`` (in that band),
            ``dmag`` = obs_mag - computed_mag, ``phase_deg``, and ``used`` (whether it went
            into H).
        """
        out = []
        for o, p in zip(self._obs, self._phot.get("per", [])):
            if o.get("mag") is None:
                continue
            out.append({"jd_utc": o["jd_utc"], "obscode": o["obscode"],
                        "obs_mag": o.get("mag"), "band": o.get("band", " "),
                        "computed_mag": p.get("computed_mag"), "dmag": p.get("dmag"),
                        "phase_deg": p.get("phase_deg"), "used": p.get("used")})
        return out
    # --- hazard / classification ----------------------------------------
    def _ecl_elements(self):
        return dict(zip(ELEM_NAMES, self._elem.tolist()))
    def moid_earth(self):
        """Minimum orbit intersection distance with Earth's osculating orbit at the epoch, au
        (None if it cannot be computed)."""
        if not hasattr(self, "_moid_cache"):
            from pyroorb import hazard
            try:
                self._moid_cache = hazard.moid(self._ecl_elements(),
                                               earth_ecliptic_elements(self.epoch_tdb))
            except Exception:
                self._moid_cache = None
        return self._moid_cache
    def orbit_classification(self):
        from pyroorb import hazard
        H, _ = self.absolute_magnitude()
        return hazard.classify(self._val("a"), self._val("e"), self._val("i"),
                               H=H, earth_moid=self.moid_earth())
    def orbit_class(self):
        """The dynamical class from a, e and q, as a label: e.g. ``"Apollo (NEO)"``,
        ``"Mars-crosser"``, ``"main-belt asteroid"``, ``"Jupiter Trojan"``, ``"Centaur"``,
        ``"hyperbolic/parabolic"``."""
        return self.orbit_classification()["class"]
    def is_pha(self):
        """Potentially hazardous: an NEO (q < 1.3 au) with Earth MOID <= 0.05 au and
        H <= 22."""
        return self.orbit_classification()["pha"]
    def tisserand_jupiter(self):
        """The Tisserand parameter with respect to Jupiter."""
        return self.orbit_classification()["tisserand_jupiter"]
    def uncertainty_parameter(self):
        """The MPC uncertainty parameter U, 0 (best) to 9, from the a and Tp sigmas
        (find_orb's formula)."""
        from pyroorb import hazard
        return hazard.uncertainty_parameter(self._val("a"), self._val("e"),
                                            self._sig("a"), self._sig("Tp"))
    def is_combined(self):
        """Whether the fit joins observations reported under more than one designation."""
        return self.combined
    def linked_designations(self):
        """Every designation the fitted observations were reported under."""
        return list(self.designations)
    def residuals(self, included_only=False):
        """Per-observation residuals, O-C = observed - computed.

        Parameters
        ----------
        included_only : bool, default False
            Leave out the observations the fit rejected. Otherwise the rows line up one to
            one with the observations given to :func:`pyroorb.fit_orbit`.

        Returns
        -------
        list of dict
            ``jd_utc``, ``obscode``, ``designation``, ``d_ra_arcsec`` (RA residual times
            cos(dec)), ``d_dec_arcsec``, ``sigma_ra`` / ``sigma_dec`` (the weights used,
            arcsec), ``resid_sigmas`` (the residual's size in sigmas), ``included``.

        Notes
        -----
        Implementation: per-obs O-C (arcsec) in the standard sense O-C = Observed - Computed
        (matches the photometric dmag = obs_mag - computed_mag and find_orb's sign
        convention). The engine stores residuals internally as Computed - Observed
        (the LM residual vector / _resid_all), so we negate here at the reporting
        boundary; the internal rejection/RMS math is sign-agnostic and untouched.
        Includes rejected obs (with 'included': False) so the astrometry view can
        render find_orb's 'x' marks.

        The rows are POSITIONALLY ALIGNED with `_obs` (unless included_only), and
        callers rely on that: they zip them against fit._obs to attach the sky
        position, the obs80 line, the photometry row. The no-_resid_all fallback
        below therefore emits a row PER OBS too -- an excluded one carries NaN
        residuals rather than being skipped, because a shorter list read
        positionally is a silently WRONG list, not a shorter one (it was: the
        fallback used to return the included rows alone, so every such caller read
        the residuals of one obs against the identity of another the moment a fit
        had a rejection)."""
        if self._resid_all is None:        # fallback: only the LM's included rows survive
            f = np.asarray(self._res.fun, float).reshape(-1, 2)
            it, out = iter(f), []
            for k, o in enumerate(self._obs):
                inc = bool(self._mask[k]) if k < len(self._mask) else False
                fk = next(it, None) if inc else None
                if included_only and not inc:
                    continue
                out.append({"jd_utc": o["jd_utc"], "obscode": o["obscode"],
                            "designation": o.get("designation", self.designation),
                            "d_ra_arcsec": -float(fk[0]) if fk is not None else float("nan"),
                            "d_dec_arcsec": -float(fk[1]) if fk is not None else float("nan"),
                            "included": inc})
            return out
        out = []
        for k, o in enumerate(self._obs):
            if included_only and not self._mask[k]:
                continue
            dra, ddec = -float(self._resid_all[k, 0]), -float(self._resid_all[k, 1])
            sra, sdec = float(self._sig_ra[k]), float(self._sig_dec[k])
            out.append({"jd_utc": o["jd_utc"], "obscode": o["obscode"],
                        "designation": o.get("designation", self.designation),
                        "d_ra_arcsec": dra, "d_dec_arcsec": ddec,
                        "sigma_ra": sra, "sigma_dec": sdec,
                        "resid_sigmas": float(math.hypot(dra/sra, ddec/sdec)),
                        "uncert_src": o.get("uncert_src", "Veres2017"),
                        "included": bool(self._mask[k])})
        return out
    def excluded_observations(self):
        """The :meth:`residuals` rows of the observations the fit rejected."""
        return [r for r in self.residuals() if not r["included"]]

    # --- find_orb elements.json-style dict ------------------------------
    def determinacy(self):
        """Is this orbit determined by its observations?

        Returns
        -------
        dict or None
            None when it is. Otherwise what the arc leaves open -- the verdict the web page's
            banner shows: ``headline`` and ``detail`` (text), ``ranges`` (``{element: [lo,
            hi]}``, the 5-95 % intervals the observations still allow, or None),
            ``bound_fraction``, ``U``, and further keys shaped for that banner (for
            digest2 scores call :func:`pyroorb.digest2_scores`).

        Notes
        -----
        A single tracklet is judged by its range profile (does the astrometry pin the
        distance at all?), never by U or the number of observations. Computed once per
        fitted state and included-observation set (``constraint.assess``).
        """
        key = (np.asarray(self._state, float).tobytes(), np.asarray(self._mask).tobytes())
        cached = getattr(self, "_determinacy_memo", None)
        if cached is None or cached[0] != key:
            from pyroorb import constraint
            cached = (key, constraint.assess(self))
            self._determinacy_memo = cached
        return cached[1]

    @property
    def is_determined(self):
        """False when the observations leave the orbit undetermined: the elements are then a
        nominal solution only (see :meth:`determinacy` for what is still known)."""
        return self.determinacy() is None

    def as_dict(self):
        """A JSON-ready summary of the fit (find_orb ``elements.json``-like): the designation,
        epoch, elements and sigmas, rms, observation counts, H, MOID, class, U and the
        non-gravitational solution. Provisional: keys may be added or renamed before 1.0."""
        el, sg = self.elements(), self.element_sigmas()
        d = {
            "object": self.designation, "packed": self.packed_designation,
            "combined": self.combined, "other_designations": self.other_designations,
            "frame": self.frame(), "epoch": self.epoch_jd(), "epoch_iso": self.epoch_iso(),
            "rms_residual": self.rms_residual(), "rms_normalized": self.rms_normalized(),
            "converged": self.converged,
            "force_model": self.force_model, "n_resids": self.n_resids(),
            "n_obs": self.n_observations(), "n_obs_total": self.n_total_observations(),
            "n_excluded": self.n_excluded(), "n_oppositions": self.n_oppositions(),
            "arc_days": self.arc_length_days(), "obs_codes": self.observatory_codes(),
            "H": self._phot.get("H"), "G": self.slope_parameter(),
            "H_sigma": self._phot.get("sigma_H"), "n_mags": self.n_magnitudes(),
            "moid_earth_au": self.moid_earth(), "class": self.orbit_class(),
            "pha": self.is_pha(), "tisserand_jupiter": self.tisserand_jupiter(),
            "U": self.uncertainty_parameter(),
            "nongrav_model": self.nongrav_model,
            # DISPLAYED sigma = the rational profile-likelihood interval (computed lazily here, only
            # when the fit is actually serialised for display -- the fit itself stays fast).
            "nongrav_params": self.nongrav_params, "nongrav_sigmas": self.rational_nongrav_sigmas(),
            # asymmetric profile interval {axis: [minus, plus]} -- the honest -/+ half-widths
            "nongrav_intervals": {k: list(v) for k, v in self.rational_nongrav_intervals().items()},
            "nongrav_warning": self.nongrav_warning,
            "nongrav_unconstrained": list(self.nongrav_unconstrained),
            # The delete-1-tracklet JACKKNIFE robustness sigma -- included ONLY when it has already
            # been computed (the CLI --nongrav-jackknife flag / the web +/- button populate the
            # cache), because it costs ~40 refits and as_dict must stay cheap. {} otherwise.
            "nongrav_jackknife": {k: {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in d.items()}
                                  for k, d in (self._jackknife_sigmas or {}).items()},
            "nongrav_not_robust": (self.nongrav_not_robust if self._jackknife_sigmas is not None else []),
            "constraints": dict(self.constraints),
            "asteroids": self.asteroids,
            "self_perturber": self.self_perturber,
        }
        for nm in ELEM_NAMES:
            d[nm] = el[nm]; d[f"{nm} sigma"] = sg[nm]
        return d


if __name__ == "__main__":
    # The command-line / batch front-end lives in cli.py (it imports this engine, so the
    # dependency runs one way). `python -m pyroorb.fit_object ...` -- the target of the
    # ident-check / batch subprocess re-invocations -- delegates to it here; the import is in
    # the guard so importing fit_object as a library never pulls in the CLI layer.
    from pyroorb.cli import main
    main()
