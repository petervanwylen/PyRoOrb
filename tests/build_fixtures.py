"""Build STATIC astrometry fixtures + golden fit results for the regression test.

Run this ONCE (needs network + the fetch cache) to snapshot every object we
rely on into tests/fixtures/*.json. Each fixture holds the fully-parsed obs
list (positions already debiased, per-obs sigmas attached -- exactly what
fit_orbit consumes), the meta, and the CURRENT fit result as the golden baseline.

Use this in a TARGETED way: to ADD a new object (extend OBJECTS, then build just
that one) or to REFRESH an object's astrometry snapshot when explicitly asked
(re-fetches from the network). To merely re-baseline the goldens after an
intentional engine change, prefer update_goldens.py (offline, deterministic, no
re-fetch). Shared golden/atomic-write logic lives in _fixtures.py.

    python tests/build_fixtures.py            # rebuild ALL objects (needs network)
    python tests/build_fixtures.py 2023\\ DW  # rebuild matching objects only
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys, os, time, warnings; warnings.filterwarnings("ignore")
from _fixtures import FIX, compute_golden, slim_obs, write_fixture
import pyroorb.fit_object as fo

from pyroorb import obs_sources
# Every linkage/object we want to stay working. (label, [designations]).
# The 34 identification pairs + sparse NEOs that now solve + single-opp sanity.
OBJECTS = [
    # --- single-opposition / sanity ---
    ("2023_QJ3", ["2023 QJ3"]), ("2015_FE570", ["2015 FE570"]),
    # --- sparse NEOs that used to crash at IOD ---
    ("2023_DW", ["2023 DW"]), ("2022_AP7", ["2022 AP7"]),
    # --- short-arc (3-4 d) single-opp 2026 NEOs ---
    ("2026_LW", ["2026 LW"]), ("2026_LL", ["2026 LL"]), ("2026_LX", ["2026 LX"]),
    ("2026_LY", ["2026 LY"]), ("2026_LA1", ["2026 LA1"]),
    # 8 obs over 6 days at 1.03 au, rms 0.098", a = 2.527 +/- 0.0018: the arc that was banned
    # "essentially unconstrained" by an SR family whose trial builder ignored light-time
    # (constraint.py, 2026-08-29). Guards: a short CURVED arc gets no banner.
    ("2021_NX4", ["2021 NX4"]),
    # --- multi-apparition NEOs linked across oppositions (incremental weight-ramp) ---
    ("Bennu_101955", ["101955"]),
    # 3-opp a~1.27 au NEO with a clear Yarkovsky signal -- the auto nongrav-discovery
    # acceptance object (gravity ejects an apparition; discover_nongrav picks A2). See
    # test_nongrav.test_discover_yarkovsky_bn4.
    ("2003_BN4", ["2003 BN4"]),
    # tiny (H=27.3, ~12 m) 2-opp Aten with a LARGE but physical Yarkovsky: JPL fits
    # A2=+4.4e-12, ~4x the old flat 1e-12 plausibility ceiling, so discover_nongrav saw the
    # 45% rms-cut trend (z=25) yet vetoed it as "implausible" and reverted to gravity. The
    # size-aware A2 ceiling (a2 ~ 1/(rho*D); scaled by H) admits it. See
    # test_nongrav.test_discover_yarkovsky_cl50.
    ("2011_CL50", ["2011 CL50"]),
    # 3-opp a=1.14 NEO whose Yarkovsky was the genuine MODEL-SELECTION case: the A2 fit is
    # significant (z=14, JPL A2=-3.3e-13) and plausible, but its raw-arcsec rms gain is only
    # +9% -- under the old flat 10% floor -- so it was rejected. It cuts the WEIGHTED chi2 by
    # ~49%, which the chi2-effect-size + nested F-test selection now correctly accepts. See
    # test_nongrav.test_discover_yarkovsky_os5.
    ("2009_OS5", ["2009 OS5"]),
    # 3-opp a=1.06 NEO that guards the SRP<->Yarkovsky physical prior: with both A1 and A2 free,
    # the degenerate A1 (z~4.5) makes A1+A2 edge out A2-only on raw chi2 -- but JPL fits A2 only
    # (A2=-1.04e-12) and the secular trend is Yarkovsky's. The A2 prior in _ng_choice keeps it on
    # a clean yarkovsky. See test_nongrav.test_discover_yarkovsky_db.
    ("2008_DB", ["2008 DB"]),
    # 3-opp a=0.86 Aten with a Yarkovsky signal that ALSO trips the single-night-poison rescue.
    # Guards a fixed crash: _rescue_single_night_poison re-fit with auto_nongrav default-ON, so its
    # internal structural re-fit returned a 7-element nongrav state, which (a) leaked out of a
    # fit_orbit(auto_nongrav=False) caller (crashing an external discover_nongrav with a (6,)/(7,)
    # shape error) and (b) was seeded into _full_fit_excluding's gravity evaluator. The rescue is now
    # gravity-only; the single nongrav decision is made once at the end. Fits 61/62 @ ~0.38" w/ A2.
    ("2010_VQ", ["2010 VQ"]),
    # More auto-Yarkovsky objects (JPL fits A2, our discover_nongrav must pick 'yarkovsky' with A2
    # matching JPL). Spread across the a<1.6 NEO range to broaden the population guard beyond
    # 2003 BN4 / 2011 CL50 / 2009 OS5 / 2008 DB / 2010 VQ. See test_nongrav.test_more_yarkovsky.
    ("2004_JN1", ["2004 JN1"]),     # a=1.085, JPL A2=-3.8e-13
    ("2015_SD",  ["2015 SD"]),      # a=0.862, JPL A2=-3.8e-13 (snr 48)
    ("2007_EG",  ["2007 EG"]),      # a=0.755, JPL A2=-4.3e-13
    ("2020_QJ1", ["2020 QJ1"]),     # a=1.210, 4-opp, JPL A2=-3.7e-13
    # 2009 BD: a tiny (H=28) 2-opp NEO where JPL fits BOTH radial SRP A1 and transverse A2 -- and
    # both are present in the optical astrometry (unlike most SRP objects, whose A1 needs radar).
    # discover_nongrav must report the A1+A2 'srp2' model (not A2-only): the radial term is
    # independently significant (A1 marginal z~11) and srp2 cuts chi2 ~26% below A2-only. Guards the
    # yarkovsky->srp2 upgrade gate (NONGRAV_SRP2_A1_Z). See test_nongrav.test_discover_srp2_2009bd.
    ("2009_BD", ["2009 BD"]),
    # 2026 NA1: tiny (H~25, ~35 m) 2-opp Apollo whose signal is an A1-only SRP (find_orb reports it;
    # our profile-seed solver reaches it, 1.08" -> 0.56", 66/66). Its A1 implies a ~0.006 g/cm^3
    # (sub-rock) density, so it is delivered WITH an advisory non-physical-values WARNING -- but it is
    # FITTED, not rejected: a high area/mass body (a dark comet, a rocket body, or an A1 absorbing
    # 2007 pre-Gaia systematics) is exactly what we want to surface. Guards that discovery PICKS the
    # SRP model and attaches the warning. See test_nongrav.test_discover_srp_2026na1_warns.
    ("2026_NA1", ["2026 NA1"]),
    ("2015_LM24", ["2015 LM24"]), ("2024_LH21", ["2024 LH21"]),
    ("2024_NP1", ["2024 NP1"]), ("2018_KL", ["2018 KL"]),
    # --- COMET out-gassing auto-discovery (the parallel system to the Yarkovsky path) ---
    # C/2002 X5 (Kudo-Fujikawa): a near-Sun (q~0.19 au) long-period comet with strong out-gassing.
    # A pure-gravity fit cannot reconcile the near-perihelion astrometry and ejects ~24% of the obs
    # (409/538 @ 0.81"); discover_comet_nongrav reactivates everything, finds a Marsden model
    # justified (comet2 under the old 15% model sequence floor, comet3 under per-step BIC since 2026-09-14),
    # and RESCUES ~108 of those obs (517/538) at comparable rms. Exercises
    # the T_J<<3 / C/ designation trigger and the comet model sequence selection. See
    # test_nongrav.test_discover_comet_x5.
    ("C_2002_X5", ["C/2002 X5"]),
    # 2I/Borisov: 3350 obs over 657 days, e 3.357. Under the comet model sequence's old 15% gain floor
    # comet1 (gain 8%, z 28, dBIC 616) was refused and the gravity fit clipped the arc to cope
    # (3067/3350, 2019-09..2020-05). Guards per-step BIC + the comet-only yield of the 8% raw-rms
    # floor at dBIC >= 60. On this ADES path master delivers gravity at
    # 2480/3347 with Aug-Oct 2020 wholly rejected; now comet1 2616/3347, included to the arc's end.
    # benchmark: a ~67 CPU-s fit; quality keeps the fit guards on, since the model is the point.
    ("2I", ["2I"], {"benchmark": True, "quality": True}),
    # 1I/'Oumuamua: 214 obs over 56 days, e 1.2002. The one real UNBOUND object whose early nights
    # are short enough to exercise the admissible-region bound comparator: nights 2-3
    # (18 obs, 6 tracklets) must stay unbound while two nights of a main-belt asteroid must not.
    # See test_unbound_door. Its full arc is weighed by the capped walk's LOWER BOUND (the walk
    # stops at e 1.14 and 39"), so this also pins that no exit skips the rules.
    ("1I", ["1I"], {"quality": True}),
    # C/2013 A1 (Siding Spring): 3248 obs 2012-2017, near-parabolic (JPL e 1.0000855, q 1.39976).
    # IOD chose a 468-day gap-free window no seed method can represent -- 603/3248 at 279863" with
    # q = 2e-5 au -- and every rescue was gated off. Guards iod.rescue_window + its early gate in
    # fit_object._fit_from_start (34-day pre-perihelion re-seed; comet3 3162/3248 at 0.654" since
    # the comet model sequence took per-step BIC, 2026-09-14 -- comet1 3094/3248 at 0.666" before).
    # benchmark: a ~170 CPU-s fit; quality opts its fit guards back in, since the orbit is the point.
    ("C_2013_A1", ["C/2013 A1"], {"benchmark": True, "quality": True}),
    # 2018 XQ2: a 28-day single-apparition Aten discovered at 0.025 au, i = 0.08 deg -- lines of
    # sight and Earth's positions all nearly coplanar, so Gauss over the 27-day geometric IOD window
    # seeds a retrograde (i = 178) orbit at 277" and nothing could repair it: no second apparition
    # for any recovery stage to re-anchor on. Delivered 33/44 at 93". Guards the single-apparition
    # arm of the seed-window rescue in fit_object._fit_from_start (19-day re-seed -> 44/44 at
    # 0.364", a/e/i matching JPL). find_orb fits it directly (its shrink loop is ungated).
    ("2018_XQ2", ["2018 XQ2"]),
    # --- many-opposition MBAs across the (formerly corrupt) pre-2001 ephemeris window ---
    ("1999_RO260", ["1999 RO260"]), ("107474", ["107474"]),
    # --- IOD epoch re-anchor (SPEED guard): clean 7-opp MBA where select_subarc anchored the
    #     2024 apparition but initial_orbit's shrink loop seeded from 2017 -- a 7-yr epoch/data
    #     gap left the Cartesian-state LM hopelessly ill-conditioned, so every subarc_extend
    #     fit ground to the nfev cap (12s @ a false 26" min). Fixed by re-anchoring the epoch
    #     onto the real seed arc; this fixture's ~2s baseline flags any re-slowdown. ---
    ("740715", ["740715"]),
    # --- heavy many-opposition (1000+ obs / ~20 opp) speed cases ---
    ("39976", ["39976"]), ("51661", ["51661"], {"benchmark": True}),
    # --- BENCHMARK-ONLY heavy objects: kept for speed measurement, skipped by the routine
    #     correctness run (not difficult to fit; fit-quality not guarded). Add speed-only
    #     objects here with {"benchmark": True}. ---
    ("13923", ["13923"], {"benchmark": True}),
    ("10052", ["10052"], {"benchmark": True}),
    # --- speed-only MBA benchmarks: the slowest-fitting objects (compute-time > 1.4 s) from a
    #     250-object random MBA draw (a 2.0-3.5 au, q>1.666, 2-7 opp). Clean multi-opposition
    #     fits, kept purely as a fit-time floor so a future slowdown on ordinary MBAs is caught.
    #     2019 QD60: 101 obs / 5 opp (~1.9 s); 2024 QH31: 31 obs / 5 opp (~1.6 s). ---
    ("2019_QD60", ["2019 QD60"], {"benchmark": True}),
    ("2024_QH31", ["2024 QH31"], {"benchmark": True}),
    # --- obscode sigma-relaxation case (over-confident D29/I41: reject only the blunder) ---
    ("559643", ["559643"]),
    # --- HARD-DYNAMICS coverage (low perihelion -> GR/inner-planet stress; Jupiter-crossing /
    #     co-orbital / quasi-Hilda -> strong Jovian perturbations). Guards the force model
    #     (GR_SIMPLE, planetary perturbers) and the integrator thrash path on the orbits that
    #     stress them, before we tune LM/integration aggressiveness. ---
    ("163693_Atira", ["163693"]),          # Atira/IEO prototype: q=0.50, entirely inside Earth
    ("594913_AyloChaxnim", ["594913"]),    # 2020 AV2: q=0.46, orbit inside Venus (extreme low-q)
    ("3552_DonQuixote", ["3552"]),         # classic ACO (asteroid on cometary orbit), T_J=2.31
    ("514107_BZ509", ["514107"]),          # 2015 BZ509: RETROGRADE Jupiter co-orbital, T_J=-0.75
    ("2009_DQ118", ["2009 DQ118"]),        # quasi-Hilda, frequent close Jupiter encounters, T_J=3.0
    ("944_Hidalgo", ["944"]),              # deep Jupiter-crosser, T_J=2.07, 2004 obs / 31 opp
                                           # (heavy multi-opposition; guards the recovery gate)
    ("33822_2000AA231", ["33822"]),        # Jupiter Trojan, a=5.18, 1708 obs / 23 opp; its long-arc
                                           # IOD overflowed the Lambert solver (Newton dF) -> crash;
                                           # guards the OverflowError bail-to-next-method in twobody.lambert
    ("2001_KQ76", ["2001 KQ76"]),          # sparse TNO, a=42.6, 12 obs / 7 nights; drop-suspect-night
                                           # trial thinned the mask below 3 obs -> scipy LM crash
                                           # (residuals < variables); guards the FIT_MIN_OBS skip
    ("2010_LP68", ["2010 LP68"]),          # distant TNO (a=36, e=0.03), 32 obs / 2 opp over 358 d, all
                                           # 568. Its single-night discovery seed is under-constrained
                                           # in (a,e), so Gauss/Herget could only fit it as a spurious
                                           # e>1 orbit; the OLD path stranded the seed (unphysical->revert)
                                           # and linked only the two endpoint tracklets -> wrong e=0.63
                                           # orbit dropping 10/32 obs. Guards the e-capped keep-growing
                                           # extension: walk across the intra-apparition gap until the
                                           # arc pins a=36, rejecting only the 1 genuinely-bad obs.
    # --- sparse distant TNOs whose geometric select_subarc returned a 2-OBS seed window
    #     (within 45deg + arc cap): 4 residuals under-determine the 6-element state, so the
    #     first least_squares('lm') raised "residuals < variables" and the whole fit CRASHED.
    #     Fix widens the seed mask by index to FIT_MIN_OBS before that first fit (the IOD
    #     seed/epoch are untouched; attempt_extensions grows outward from it anyway). Found by
    #     the a>15 TNO scan. ---
    ("2018_JU6", ["2018 JU6"]),            # 5 obs / 5 nights / 294 d -> now 5/5
    ("2019_QM111", ["2019 QM111"]),        # 8 obs / 4 nights / 768 d -> now 7/8, opp 2
    ("2014_FO69", ["2014 FO69"]),          # 10 obs / 6 nights / 1449 d -> now 10/10, opp 4
    # --- range-degenerate single-night ANCHOR of a distant multi-apparition object:
    #     a one-night anchor of a 40+ au TNO has no range info, so Gauss/Herget seed it at the
    #     wrong distance and the orbit predicts every OTHER apparition ~180deg off -> the arc
    #     stranded on its discovery night (3/15). The cross-apparition Vaisala range-sweep link
    #     (initial_orbit "vaisala-link") sweeps the anchor distance, links it across an opposition
    #     gap, and recovers the whole arc. Found by the a>15 TNO scan. ---
    ("2001_UN18", ["2001 UN18"]),          # classical KBO a=44 e=0.07 q=41, 15 obs / 6 opp -> 15/15
    ("2001_DV108", ["2001 DV108"]),        # TNO, 15 obs / 2 opp / 410 d -> 11/15 to 15/15 via the link
    # --- HANG guard: Earth co-orbital (a=1.03, e=0.04, q=0.99) whose multi-year extension
    #     integration repeatedly passes close to Earth, so IAS15 subdivided its step toward zero
    #     and a SINGLE batch integration never returned -- the fit ground forever (the (cnt&63)
    #     thrash check can't fire when cnt never advances). force_assist.IAS15_MIN_DT floors the
    #     adaptive step so the encounter integrates (coarsely) and RETURNS: now 18/22 @ ~3", 2 opp,
    #     ~2 s. Guards that the min_dt floor stays in place (a re-hang would time out the suite).
    #     Found by the 2-4 opp NEO scan. ---
    ("2007_XB23", ["2007 XB23"]),
    # --- BENCHMARK: a>15 TNO, 25 obs / 10 nights / 7 opp / 23 yr. Fits 25/25 @
    #     0.152" but USED to take ~315s: the free extension fit overshot to e~8e3 before the
    #     e>=1 capped-walk test fired, and the e-capped LM was seeded AT that gross hyperbola,
    #     integrating it over the full 23-yr arc at ~10 s/eval (94% of the fit). Fix reseeds the
    #     capped walk from the bound best_state on gross divergence (ECC_REINIT) -> 315s->23s.
    #     Benchmark-only baseline guards against re-slowdown. ---
    ("2010_TA192", ["2010 TA192"], {"benchmark": True}),
    # seed-clean over-banning guard: 26-obs/1.08-d/2-night tracklet whose
    # seed fits all 26 at 1.11" (>SEED_POOR_RMS); dropping a night let the other overfit to 0.18",
    # which seed-clean MISREAD as contamination and banned 14 good obs -> 12/26 hyperbolic. Fix:
    # don't ban a night if the cleaned orbit is UNBOUND. Guards OBS-RETENTION (25/26); the arc is
    # genuinely undetermined (SR family a in [1,223] au) so the orbit stays hyperbolic-with-huge-sigma.
    ("2002_TD301", ["2002 TD301"]),
    # --- wide-gap multi-apparition links that USED to need --force but now resolve on the
    #     DEFAULT path: a stranded early apparition auto-triggers the sigma_a multi-start
    #     re-anchor (_has_unlinked/_has_discordant), which pins the orbit and bridges the gap
    #     -- byte-identical orbit and sweep count with/without --force, so the flag is dropped
    #     and the test now exercises the real default path users get. ---
    ("2006_RH119", ["2006 RH119"]),
    ("2008_UK199", ["2008 UK199"]),
    # --- multi-REVOLUTION bridge: two short arcs 22 yr (~5.5 revs) apart, with 3 contaminant
    #     obs on one 2001 night that hide at the contaminated minimum. Seed-clean isolates the
    #     3 (keeping the good same-night obs), then the period-sweep bridge crosses the gap.
    #     Also now links on the default path (the contaminant strands an apparition -> auto-
    #     recovery), so --force is no longer required. ---
    ("2023_VM6", ["2023 VM6"]),
    # --- multi-REVOLUTION bridge on the DEFAULT (non-force) path: faint (H~18) two-opposition
    #     MBA, 2003 + 2024 arcs ~21 yr apart, where the nominal lands a full revolution off.
    #     The cheap swarm can't cross it; the two-body-ranked period sweep does. Guards that the
    #     bridge generalises beyond VM6 (found via mpcorb nopp==2 + long-arc + faint sampling). ---
    ("2003_YH136", ["2003 YH136"]),
    # --- multi-REVOLUTION bridge cracked by the a-constraint SECANT (sweep fallback): single-night
    #     1993 first apparition + 5-day 2016 arc, 23 yr (~8 revs) apart. The 5-day anchor is too
    #     short for the coarse period sweep, so the bridge root-finds the period directly -- pin a,
    #     slide it to zero the far apparition's RA residual while the anchor stays glued. Guards the
    #     secant fallback + the low-q cross-apparition extension guard that lets the bridge fire. ---
    ("1993_TZ", ["1993 TZ"]),
    # --- single-designation 3-apparition MBA whose two EARLIER apparitions (2016, 2021) are each
    #     a lone single-night precovery tracklet; the dense 2025 arc anchors and cluster-activation
    #     snaps both in (14/16 @ 0.076", all 3 opp). Guards that the single-night-poison rescue does
    #     NOT amputate cleanly-linked single-night precoveries: it counts only GOOD (multi-night)
    #     drops now, so a couple of within-night outliers no longer trip it into re-presenting the
    #     2025-only orbit (was 9/16 @ 0.057", opp 1). find_orb links all three at ~0.13". ---
    ("2025_MM254", ["2025 MM254"]),
    # --- 2-opposition Mars-crosser whose second apparition is a handful of very tight
    #     (sigma ~0.1") F51/F52 obs 24 years after the first. The extension fit that added
    #     that apparition used to stop on its TOLERANCE without moving at all (MINPACK
    #     nfev=2, dx=0), so the reject loop scored the new obs against an orbit blind to
    #     them and clip three of them at 3.9-7.4 sigma -- which the delivered orbit then
    #     showed at 0.4-0.8 sigma. Guards BOTH halves of that fix: the stall retry
    #     (Evaluator._unstall) and the settle of the mask against the DELIVERED orbit
    #     (_settle_final_mask). Must stay 39/39 -- any exclusion here is the bug back. ---
    ("2002_QG17", ["2002 QG17"]),
    #   ...and its sibling, where the solve MOVES (so the stall guard never arms) but still
    #   stops short: an 11-year multirev link glues the 2026 close-approach recovery (3 F51
    #   at sigma 0.1", 13"/min + 3 D29) at new=1.9", and the reject loop scored that
    #   unconverged orbit -- the whole F51 night went at 11-25 sigma, and the refit without it
    #   left F51 7" off. Converged on the same 35 obs it fits at 0.49" (the MPC's orbfit uses
    #   35/35). Guards the polish before the reject stage (attempt_extensions). Must stay
    #   35/35 -- any F51 exclusion here is the bug back. ---
    ("2015_VZ1", ["2015 VZ1"]),
    # --- bridge-required NEOs (q<1.3): two oppositions a short-period (many-rev) gap apart.
    #     A spread of which method links them, found via the mpcorb q/nopp/arc_years sampler:
    #     2002 CC26 by the period SWARM; 2008 KB12 + 2001 UQ16 by the a-constraint SECANT (the
    #     anchor too short for the sweep). Guards both bridge methods on real NEOs. ---
    ("2002_CC26", ["2002 CC26"]),
    ("2008_KB12", ["2008 KB12"]),
    ("2001_UQ16", ["2001 UQ16"]),
    # --- bridge-required NEOs whose best anchor is the LATER, denser apparition. The
    #     sigma_a-ranked multi-start (now enabled on the DEFAULT path when an apparition is
    #     stranded) re-anchors there and the bridge crosses back; both exercise _better_link's
    #     "an extra opposition is worth a higher rms" rule (2008 TE4 was being rejected for a
    #     smaller opp=1 fit) -- a noisy old apparition is signal, not a contaminant.
    #     Both now link cleanly on the DEFAULT path: 2008 TE4 via the stranded-apparition
    #     trigger, 1998 VD32 via the DISCORDANT-apparition trigger (its 1998 arc sat 3.8sigma off
    #     the 2020-pinned orbit -- the seed-retry re-anchors on 1998 and threads both at 0.525",
    #     matching find_orb). ---
    ("2008_TE4", ["2008 TE4"]),
    ("1998_VD32", ["1998 VD32"]),
    # --- the CHIMERA-REJECTION counterpart of the two above: guards _better_link's obs-count
    #     floor on the extra-opposition rule. P12orpW (a 12-obs, ~1-day 2026 NEOCP tracklet) is
    #     2016 BV14 re-observed 10.5 yr / ~3 revs later. Re-anchoring the sigma_a multi-start on the
    #     short 2026 arc pins a garbage a=4.87 orbit that drags in only 3 of the 24 rich-2016 obs and
    #     rejects the other 21 -- a 15/36 fit that still reports n_oppositions()==2, so WITHOUT the
    #     "cand keeps >= ref's obs" guard it wins _better_link over the honest 24/36 anchor, both
    #     apparitions then carry some obs so _still_troubled goes quiet, and the recovery stage that
    #     actually links it never runs (web UI + CLI showed a=4.87, 21 obs off). With the guard the
    #     chimera is rejected, the clean 24/36 stays best (2026 still fully unlinked -> troubled), and
    #     the JOINT REFIT -- one unmasked full step -- glues the true 36/36 link at a=2.354 / 0.21" in
    #     five integrations. (It was a 340-cell (a, M) grid until 2026-08-24.) NOTE: P12orpW is a live NEOCP tempdesig -- these obs are
    #     FROZEN here; a future rebuild can't re-fetch it (the union will re-fetch under 2016 BV14
    #     once MPC attributes it). ---
    ("P12orpW_2016BV14", ["P12orpW", "2016 BV14"]),
    # --- WIDE MULTI-REVOLUTION re-anchoring (BENCHMARK -- the suite's most expensive object).
    #     2008 NA = 2025 MF90, an Aten (a=0.96, q=0.67, P=0.94 yr) whose two arcs sit ~17 yr / ~18
    #     revolutions apart. Its true a sits 6.2-6.9 sigma_a from every well-determined anchor -- a
    #     short single-apparition arc's formal sigma is under-dispersed -- so every bridge search
    #     sized on those covariances misses, and only a re-anchor onto the LOOSE one-night sub-arc
    #     of the 2025 apparition links it (a-secant, a=0.96223). Guards the anchor RANKING: rank by
    #     tightest sigma_a and this object is answered ~4x of work later, or not at all. It needed a
    #     340-cell (a, M) grid until 2026-08-24. ---
    ("2008_NA", ["2008 NA"], {"benchmark": True, "quality": True}),
    # --- LINKAGE THE MPC HAS NOT MADE, once the seed stopped being wrong. 2018 SY2's obs API
    #     serves 6 observations from 2020 beyond the 29 the MPC fits (29 obs, 27 days, ONE
    #     opposition, q=0.47655 e=0.38391 i=26.7728, U=0). We used to fit those 35 obs as
    #     a=3.0/e=0.85/i=67 at 339" and I diagnosed the 2020 tracklet as a contaminant. It is not:
    #     with the IOD window confined to one apparition (see iod._scan_windows) the fit reaches
    #     35/35 @ 0.431" over 681 DAYS with q=0.476572 e=0.383968 i=26.765 -- the MPC's own
    #     elements to four decimals, on a 25x longer arc. The old 339" solution was a wrong
    #     two-apparition minimum reached from a gap-straddling seed, not evidence against the data.
    #     Guards that the seed fix keeps finding the real linkage. ---
    ("2018_SY2", ["2018 SY2"]),
    # --- RADAR THAT WRECKS THE OPTICAL FIT. 2003 HM (Aten, MOID 0.007 au) carries 9 JPL radar
    #     delay/Doppler measurements, folded into a final differential correction after the
    #     optical pipeline converges. The joint solve satisfies NEITHER dataset: radar median
    #     |z| 5.68 -- inside the one-sided RADAR_ABANDON_Z=25 guard -- while the optical goes
    #     0.735" (0.96 sigma, matching the MPC's published elements) to 4.836" (9.34 sigma), and
    #     that compromise was what got delivered. radar.RADAR_MAX_OPTICAL_DEGRADE now asks the
    #     symmetric question -- does the joint fit still represent the SKY? -- and keeps the
    #     optical orbit, reporting the disagreement. NB the fixture pins the OPTICAL path only:
    #     fixtures carry no meta['radar'], so the radar guard itself is covered by
    #     tests/test_radar.py (2005 YU55 still applies 19/19 at 0.52 sigma). Related: the MPC
    #     fits a Yarkovsky here (yarkovski -0.00118) that our NONGRAV_CHI2_GAIN=0.15 effect-size
    #     bar refuses at +8% despite F=23/p=3e-06/z=6.5 (since replaced by a BIC bar). ---
    ("2003_HM", ["2003 HM"]),
    # The 0.00033-au 2004 flyby whose 2-3 km observer-Earth error (astropy builtin vs
    # DE440) manufactured a jackknife-robust phantom Yarkovsky (A2 -3.4e-13 vs JPL's +1.7e-13).
    # The golden pins the post-fix truth: gravity-only at ~0.50" -- a nongrav key flipping back
    # on is this defect returning through any door (observer geometry, encounter threading,
    # near-encounter rejection).
    ("2004_FH", ["2004 FH"]),
    # The mask-chosen-justification class, pinned. 2021 LW3 (2-opp, 3-yr arc)
    # delivered a phantom yarkovsky at retest dBIC 16.1 whose same-data margin is 12.4 --
    # the gap was one F51 obs the model-side rejection kept that gravity's own fixed point
    # drops. The golden pins gravity-only: a nongrav key reappearing here is the same-data
    # re-check (_ng_mask_recheck) regressing through any door.
    ("2021_LW3", ["2021 LW3"]),
    # The raw-invisible class, pinned. 2022 OC3 (Arjuna, 12 nights over
    # 2 oppositions) clears the honest-mask dBIC at 34.6 -- weighted chi2 bought by a few
    # tight-sigma obs -- while the same-mask raw rms moves only 5.6% (0.274" -> 0.258"),
    # under the 8% visibility floor (NONGRAV_MIN_RMS_GAIN). The golden pins
    # gravity-only: a nongrav key here is the visibility floor regressing through any door.
    ("2022_OC3", ["2022 OC3"]),
    # The basin-escape class, pinned on its discovery object. 2012 VP6's
    # "yarkovsky" carried dBIC 31 with an A2 at z~0.2 by BOTH sigmas -- the model's
    # curved-valley traversal had found a gravity basin (chi2 48.7-equivalent) that every
    # plain gravity LM misses (they stall at 84 or diverge), and the accounting credited
    # the orbit improvement to the force. The final delivered-coefficient re-check (a
    # pinned-A=0 refit from the delivered seed) reverts it and DELIVERS that better gravity
    # orbit. The golden pins gravity-only at the reclaimed rms: a nongrav key here is the
    # end-of-pipeline guard regressing; an rms regression toward ~0.32 is the demonstrated
    # basin being lost.
    ("2012_VP6", ["2012 VP6"]),
    # The strain-gated srp3 step, pinned on its discovery object. 2009 HC's A2-only
    # yarkovsky sat sigma-strained at 0.762" (rms_n 0.96) because the real signal has an
    # out-of-plane component no 2-axis model can express -- srp3 lands 0.375" with A3 at
    # marginal z=35 through every upgrade clause (confirmed by eye first). The golden
    # pins srp3 with all three signs: a flip back to yarkovsky is the step, the upgrade
    # clauses, or the strain gate regressing through any door.
    ("2009_HC", ["2009 HC"]),
    # The mega-arc perturber class, pinned. 1862 Apollo (1930-2026 arc, Q=2.29,
    # 17 radar rows) is the object whose Yarkovsky was 6x small / wrong-sign under
    # planets-only physics and lands at JPL's order (-6.6e-15 vs their -3.66e-15) once the
    # delivery runs under the 16 perturbers with the detection re-checked under them. The
    # golden pins yarkovsky with A2 sign -1: a flip back to + (or to None) is the perturber
    # gate, the ctx coherence, or the radar joint regressing through any door. benchmark:
    # a ~3-min fit stays out of the routine loop; quality opts its fit-quality guards in.
    ("1862", ["1862"], {"benchmark": True, "quality": True, "mpc_seed": True}),
    # --- 21P/Giacobini-Zinner: the APPARITION-WISE COMET EXTENSION (fit_object
    #     _comet_extend_apparitions). 7434 obs over 1900-2024, twenty apparition clusters, and
    #     an out-gassing history no single Marsden A1/A2/A3 set spans: the seeded gravity fit is
    #     2622" on everything, the everything-mask comet model sequence delivers 465" and is refused,
    #     and the engine used to fall back to ONE apparition at 1.69" -- no rejection (the guard
    #     measured a 4197-obs mask against ev.n=7434) and a 1962 mid-arc epoch fifty-six years
    #     off. Now: the 2018 apparition re-anchored at its own epoch and settled (3662 @ 1.29"),
    #     then 2011 and its 2012.9 tail joined under comet3 (415/465 at median 1.55 sigma), 2005
    #     and everything earlier refused, 4180 obs over 2011-2019 at 0.868" with comet3 -- the
    #     MPC's own published orbit is 2011-2024 with A1+A2 at rms_n 0.92. Pins: comet3 with
    #     A1 +, A2 -, and >= 4000 used obs (a one-apparition regression drops it to ~3700).
    #     benchmark: an ~88 CPU-s fit; quality opts its fit guards in; mpc_seed: the production
    #     path. KEPT benchmark-gated on purpose (reviewed 2026-09-07): 88 CPU s is 13% of the
    #     whole serial offline suite and would land on one xdist worker, and the flag means
    #     exactly this -- a heavy object. What that costs is coverage of the extension's
    #     OUTCOME, not of its logic: the orchestration (seed choice, growth order, the refusal
    #     ledger, the interior second look, the join verdict) is now unit-tested offline and in
    #     the FAST loop by tests/test_comet_extension.py, which a fixture could never have done
    #     anyway -- a golden asserts an answer, not a control flow, so it cannot tell you the
    #     _blocked rule is wrong in a way that happens not to move 21P. RUN THIS ONE BY HAND
    #     (`pytest "tests/test_regression.py::test_fixture_still_links[21P]" --benchmark`,
    #     ~90 s) before landing any change to _comet_extend_apparitions / _reanchor_apparition /
    #     the COMET_EXT_* constants / the accumulative comet model sequence.
    ("21P", ["21P"], {"benchmark": True, "quality": True, "mpc_seed": True}),
    # --- MANY-APPARITION COLD FIT. 162385: 336 obs, 26 yr, FIFTEEN apparitions, an
    #     Aten at a=0.74. Cold (no MPC seed) it took 585 CPU SECONDS -- find_orb does it in ~8 --
    #     and tripped the sweep's 240 s watchdog, the only timeout in a 2000-object NEO sweep.
    #     Two causes, and neither was the linking itself. (1) select_subarc seeded it on 11 obs
    #     spanning 692 DAYS at rms 1078": a window straddling apparition gaps, which Gauss/Herget
    #     cannot represent. (2) From that anchor _link_oppositions ran THIRTEEN Tier-2 bridge
    #     cascades, twelve rejecting -- but they are not independent experiments, they all test the
    #     same anchor orbit nearest-first, so the third was already re-testing a failed hypothesis.
    #     Fail-fast alone: 585 -> 115 s. Plus the confined seed: 585 -> 14.5 s, same answer
    #     (333/336 @ 0.541", rms_n 0.72, all 15 apparitions -- better than the MPC's 305/336 at
    #     normRMS 0.946). Guards both. ---
    ("162385", ["162385"]),
    # --- PHASE-SCAN multi-revolution bridge: two-apparition NEO (2004 + 2016, ~12 yr apart) whose
    #     short anchor pins the orbit SHAPE/a but not the orbital PHASE, so the period sweep and the
    #     a-secant both miss (they nudge a and re-fit the under-determined short arc, corrupting the
    #     shape). _bridge_far_by_phase holds the anchor's a and slides the mean anomaly across one
    #     period to land the far apparition, then joint-glues -> 84/110 opp=1 stranded becomes
    #     108/110 @ 0.49", 2 opp (clean, no systematics). Guards the phase-scan fallback. ---
    ("2004_TE10", ["2004 TE10"]),
    # --- LINE-OF-VARIATIONS multi-revolution bridge: 2-opposition pure-gravity Amor (a=1.86,
    #     e=0.475, q=0.98), two short arcs 18 yr (~7.2 revs) apart -- 2003 (28 obs/7.4d) + 2021
    #     (42 obs/2.1d). Each apparition fits ~truth ALONE, but pinning a (a-secant) or sliding
    #     phase to seed the bridge distorts the anchor plane by a hair, which over 7 revs throws
    #     the far arc off CROSS-TRACK -- a direction no single-axis scan corrects, so all of the
    #     a-secant / period-sweep / phase-scan floor at ~39" far offset. _bridge_far_by_lov slides
    #     the anchor along its covariance's least-constrained eigendirection (the correlated
    #     a/plane/phase valley = find_orb's improve_along_lov), threading the cross-track error ->
    #     38/70 opp=1 stranded becomes 69/70 @ 0.54", 2 opp (elements match JPL). Guards the LOV
    #     bridge. ---
    ("2003_LW2", ["2003 LW2"]),
    # --- NEAR-1yr-PERIOD multi-apparition NEO whose multi-rev ambiguity shows only a MODEST sky
    #     offset. 2024 LW (Aten, a=0.985, e=0.41, P=0.977 yr -- near 1:1 with Earth) has a dense 2024
    #     discovery arc (34 obs / 22.7 d) plus four lone single-night tracklets 1-9 revolutions away
    #     (2015, 2021, 2023, 2026). Because an Earth-resonant object returns to nearly the same sky
    #     place each apparition, each wrong-revolution offset is only ~1-12 kilo-arcsec -- far below
    #     LINK_MULTIREV (14deg), so the router sent them to the cheap swarm (which can't sweep whole
    #     revolutions) and gave up, and the poisoned straddling IOD seed left the linker no clean
    #     anchor either -> 33/45 single-apparition, the four tracklets excluded. Now the single-night
    #     rescue's CLEAN 2024 orbit is handed to the bridge, and _bridge_opposition escalates a
    #     cheap-swarm miss to the Tier-2 period sweep when the gap is multi-rev BY TIME
    #     (MULTIREV_GAP_REVS) -> 44/45 @ 0.39", 5 opp (elements match MPCORB a=0.9845/e=0.4117). ---
    ("2024_LW", ["2024 LW"]),
    # --- short-arc, high-e retrograde NEO (Amor). 83 obs over 30 days: a dense March
    #     tracklet plus ONE lone night a month later. The rough Herget seed mis-predicts
    #     that far night by ~13" (50sigma), so seed-clean used to ban it as a contaminant,
    #     stranding an 8-day arc whose eccentricity is unconstrained -> a spurious near-
    #     parabolic orbit (e=1.008). seed-clean now keeps a WHOLE distant night when a joint
    #     fit reconciles it (0.85sigma here), threading all 80/83 obs at ~0.33" as a bound
    #     e=0.99 Amor. ---
    ("2019_EJ3", ["2019 EJ3"]),
    # --- DISTANT high-e object on a short single-opposition arc: a~280 au, e=0.993
    #     damocloid/TNO seen over 67 days (235 obs, no opposition gap). The line-of-sight
    #     range is as undetermined as a single night, so Gauss/Herget landed in a spurious
    #     a=0.74 au Aten basin (246", 68/235). The SR fallback now scans the range over the
    #     WHOLE arc -> 235/235 @ 0.51", a=282 au e=0.9935 (matching MPCORB). ---
    ("2017_SN33", ["2017 SN33"]),
    # --- two-apparition NEO whose IOD sub-arc STRADDLED the opposition gap. 2024 CL1
    #     returns to within 45deg of its discovery position a year later, so select_subarc
    #     grew a 365-day window across both apparitions and handed Herget a revolution-
    #     wrapped seed (rms ~7300"); the LM then ground forever. Guards find_orb's
    #     initial_orbit() shrink loop (arclen *= 0.7 until the seed score is acceptable),
    #     which collapses the seed back into the dense discovery tracklet. Apollo NEO,
    #     Earth MOID ~0.007 au (links all 26 obs / 2 opp @ ~0.41"). ---
    ("2024_CL1", ["2024 CL1"]),
    # --- two-apparition NEO stranded by a WITHIN-apparition over-long IOD window. 2021 WO's
    #     dense 2021-22 arc (76 obs / 99 d) sits near the object's stationary loop, so the
    #     geometric best sub-arc stayed <45deg yet spanned 73 days -- too long for Gauss, whose
    #     revolution-wrapped seed (rms ~3400") stranded the lone 2018 tracklet 3 yr / ~N revs
    #     away (11/83). Unlike 2024 CL1 the bad window has NO internal opposition gap, so the
    #     straddles-gap shrink does not fire; the recovery multi-start now re-anchors each
    #     apparition on a SHORT internal sub-arc (best_seed_subarc, find_orb's arc-shrink), whose
    #     Gauss-friendly seed extension carries across the gap -> 83/83 @ 0.43", 2 opp. ---
    ("2021_WO", ["2021 WO"]),
    # --- first COMET in the suite: P/2023 JN16, an asteroid-style provisional later
    #     flagged cometary (the 'P/' prefix that sbpy's packer chokes on). Guards that a
    #     comet designation now packs (obs80.pack_designation) and fits gravitationally
    #     like any MBA -- 3 opp / ~6.3 yr arc @ ~0.36". Non-gravitational forces are not
    #     modelled here; this is the purely-gravity baseline. ---
    ("P_2023_JN16", ["2023 JN16"]),
    # --- additional identification pairs (linked cleanly on first try) ---
    # identification with a single mislinked NIGHT to drop (tracklet-granularity ident-check):
    # the 2013 CK176 arc carries one stray 3-obs B74 night that the rest fits without.
    ("PAIR_2013CK176", ["2013 CK176", "2026 DO7"]),
    # identification where the stray tracklet shares a NIGHT with a kept tracklet from another
    # station: an F52 night-mate of a G96 tracklet is dropped (obscode-aware tracklet splitting).
    ("PAIR_2022YL13", ["2022 YL13", "2025 NF218"]),
    ("PAIR_2025OY448", ["2025 OY448", "2014 UO269"]),
    # identification where a stray tracklet shares a night AND station with a kept tracklet, but
    # under a DIFFERENT designation: a W84 night carries 5 kept 2015 KR274 obs plus 3 stray 2025
    # PY146 obs ~1.4 deg off the joint orbit (a 2015 detection mis-filed under the 2025 desig),
    # plus a lone stray 2023 V00 tracklet. Same obscode + same night made the old (night, obscode)
    # tracklet grouping read the 3 strays as a PARTIAL gutting of an 8-obs night -> rejected (None).
    # Splitting tracklets by SUBMISSION (night, obscode, designation) makes the strays their own
    # whole tracklet -> clean 'A = B except <2 tracklets>' (y2). Found via the ident-check batch.
    ("PAIR_2015KR274", ["2015 KR274", "2025 PY146"]),   # over-confident X05 (0.1" vs 0.5"): the
    #     sigma.txt reject-rate relaxation reclaims the obs the tight ADES sigma clipped, 28->49/55
    ("PAIR_2026BO14", ["2026 BO14", "2014 HM377"]),
    ("PAIR_2017EK62", ["2017 EK62", "2025 OQ641"]), ("PAIR_2016UD174", ["2016 UD174", "2025 OO559"]),
    ("PAIR_2016HH23", ["2016 HH23", "2025 NH144"], {"benchmark": True}), ("PAIR_2004GA47", ["2004 GA47", "2026 GP8"]),
    ("PAIR_2026EF28", ["2026 EF28", "2015 KY423"]), ("PAIR_2025QG135", ["2025 QG135", "2021 PP20"]),
    # --- the 34 identification pairs ---
    ("PAIR_2026EN44", ["2026 EN44", "2015 FE570"]), ("PAIR_2018FS73", ["2018 FS73", "2025 OH592"]),
    ("PAIR_2025MO385", ["2025 MO385", "2016 HG16"]), ("PAIR_2017ED53", ["2017 ED53", "2025 NL161"]),
    ("PAIR_2025OD444", ["2025 OD444", "2016 UJ129"], {"benchmark": True}),
    ("PAIR_2025KF32", ["2025 KF32", "2014 EK270"]), ("PAIR_2025NX399", ["2025 NX399", "2015 HU337"]),
    ("PAIR_2026DV62", ["2026 DV62", "2024 UB12"]), ("PAIR_2002QT68", ["2002 QT68", "2026 EO60"]),
    ("PAIR_2025OZ499", ["2025 OZ499", "2015 KL279"]), ("PAIR_2021GD209", ["2021 GD209", "2026 DB29"]),
    ("PAIR_2026EZ42", ["2026 EZ42", "2015 HF420"]), ("PAIR_2025OS355", ["2025 OS355", "2014 HZ280"]),
    ("PAIR_2025OA694", ["2025 OA694", "2020 BX154"]), ("PAIR_2026DQ58", ["2026 DQ58", "2016 FN137"]),
    ("PAIR_2014HM427", ["2014 HM427", "2026 DO62"]), ("PAIR_2026AD23", ["2026 AD23", "2014 ER287"]),
    ("PAIR_2025MU367", ["2025 MU367", "2021 RW229"]), ("PAIR_2025OW494", ["2025 OW494", "2015 HB441"]),
    ("PAIR_2025OA537", ["2025 OA537", "2025 PA77"]), ("PAIR_2015BR525", ["2015 BR525", "2026 DL31"]),
    ("PAIR_2015CY86", ["2015 CY86", "2026 DO29"]), ("PAIR_2022MT4", ["2022 MT4", "2005 JK163"]),
    ("PAIR_2025OA361", ["2025 OA361", "2021 PC149"]), ("PAIR_2014HZ280", ["2014 HZ280", "2018 BV65"]),
    # NEGATIVE: two-apparition chimera -- both oppositions multi-night, links only by gutting a
    # whole night of the older one (2015 HC310 = 2026 EM61). Must NOT identify (ident_check two-app ban).
    ("PAIR_2015HC310", ["2015 HC310", "2026 EM61"]),
    # NEGATIVE x5 -- the CLIP rule (ident_check.clip_doubt / the 2-opposition retention gate): a
    # 2-opposition link between two short, unconstrained arcs that closes ONLY by discarding a whole
    # night of one side's own coherent record is two different objects, and a flat 'no' -- not a
    # 'y-examine' candidate. Being years apart does NOT rescue it (one orbit can nearly always thread
    # two short arcs once it may clip a night). Hand-adjudicated from the ident-check batch:
    ("PAIR_2014EO135", ["2014 EO135", "2015 HT386"]),   # clips the 3rd of 3 consecutive W84 nights @ 2.5"
    ("PAIR_2014EN102", ["2014 EN102", "2022 TB28"]),    # both sides clip; the 2014 clip is 2.7" (soft)
    ("PAIR_2025QL180", ["2025 QL180", "2014 HA394"]),   # 18" clip collapses the only >3 d baseline
    ("PAIR_2016FH213", ["2025 YS17", "2016 FH213"]),    # 62% retention on the clipping side
    ("PAIR_2025DS43", ["2026 ER100", "2025 DS43"]),     # 3 nights clipped from a coherent 7-night O18 run
    ("PAIR_2025QR115", ["2025 QR115", "2021 VF77"]), ("PAIR_2020RL99", ["2020 RL99", "2025 VC35"]),
    ("PAIR_2026DR61", ["2026 DR61", "2024 RD211"]), ("PAIR_2025MV357", ["2025 MV357", "2014 HJ336"]),
    ("PAIR_2025PE28", ["2025 PE28", "2021 VB73"]), ("PAIR_2025NA115", ["2025 NA115", "2021 GX176"]),
    ("PAIR_2026JW6", ["2026 JW6", "2015 KP248"]), ("PAIR_2025OT173", ["2025 OT173", "2014 EN155"]),
    # --- idents_big.txt batch: 10 more identification pairs (packed desigs unpacked via sbpy) ---
    ("PAIR_2020TL64", ["2020 TL64", "2025 OQ395"]), ("PAIR_2001UW2", ["2001 UW2", "2015 BU642"]),
    ("PAIR_2025OB528", ["2025 OB528", "2025 PJ119"]),
    ("PAIR_2015FG554", ["2015 FG554", "2025 OA596"]), ("PAIR_2010CB37", ["2010 CB37", "2025 OZ395"]),
    ("PAIR_2014EW228", ["2014 EW228", "2026 FQ21"]), ("PAIR_2020XM18", ["2020 XM18", "2025 KL47"]),
    ("PAIR_2021SK57", ["2021 SK57", "2025 MZ315"]), ("PAIR_2015KJ307", ["2015 KJ307", "2025 NP409"]),
    # --- 22 clean cross-opposition identifications (all classify 'y': both arcs fully retained,
    #     sub-arcsec, photometrically consistent). Regression guards the joint fit reproducing. ---
    ("PAIR_2025KT34", ["2025 KT34", "2011 BF212"]), ("PAIR_2014HV328", ["2014 HV328", "2025 OV408"]),
    ("PAIR_2025OT488", ["2025 OT488", "2020 BM112"]), ("PAIR_2025OC671", ["2025 OC671", "2014 HM393"]),
    ("PAIR_2025HC95", ["2025 HC95", "2015 KC404"]), ("PAIR_2016UD169", ["2016 UD169", "2025 OU574"]),
    ("PAIR_2026EJ70", ["2026 EJ70", "2015 HW385"]), ("PAIR_2025OD445", ["2025 OD445", "2015 HN419"]),
    ("PAIR_2025OU403", ["2025 OU403", "2018 FQ73"]), ("PAIR_2002RF250", ["2002 RF250", "2025 NM484"]),
    ("PAIR_2011BB29", ["2011 BB29", "2025 OF519"]), ("PAIR_2026EW42", ["2026 EW42", "2015 FS479"]),
    ("PAIR_2025NR256", ["2025 NR256", "2014 HX73"]), ("PAIR_2015RJ179", ["2015 RJ179", "2014 HX430"]),
    ("PAIR_2001FA208", ["2001 FA208", "2016 OG18"]), ("PAIR_2025NF443", ["2025 NF443", "2015 KO294"]),
    ("PAIR_2026ES68", ["2026 ES68", "2015 HO415"]), ("PAIR_2015BW522", ["2015 BW522", "2026 FU33"]),
    ("PAIR_2025OP538", ["2025 OP538", "2014 HU557"]), ("PAIR_2012QR4", ["2012 QR4", "2017 QA37"]),
    ("PAIR_2003AR93", ["2003 AR93", "2025 UK53"]), ("PAIR_2003SR221", ["2003 SR221", "2025 QK106"]),
    # --- ident-check weighting-divergence pairs (see cli._ident_check_one / _fetch_ident_pair) ---
    #   2025 KV14 = 2025 NH520: a 16-yr 3-apparition arc + a fresh 4-night arc of the SAME 2025
    #   opposition. The canonical ADES per-obs sigmas reject the lone 2016 F51 precovery night
    #   (a faint sparse-arc period tension) and link 35/38 @ 0.10" (a y2); the looser OBS80
    #   Veres2017 defaults instead let that night pollute the orbit to ~2" and strand the whole
    #   NH520 arc -> 19/38 'no'. So this fixture must fit the ADES (default) path.
    ("PAIR_2025KV14", ["2025 KV14", "2025 NH520"]),
    #   2025 OB291 = 2016 UN180: a clean ID whose UN180 arc carries its own junk (gross outliers,
    #   one tracklet ~26000" off) -> 13 drops exceed the y2 tracklet budget, so the tool reports
    #   y-examine; the fit itself is a strong 53/66 @ 0.14" link.
    ("PAIR_2025OB291", ["2025 OB291", "2016 UN180"]),
    #   2021 TJ205 = 2015 KS376: a forced link. Each arc alone fits to ~0.06-0.09" (measurement
    #   noise), but the joint orbit can't thread both apparitions -- BOTH trend in RA (per-night mean
    #   0.45"/0.42" at ratio 4.8x/4.6x over ~0.09" within-night scatter). Neither apparition reaches
    #   the strict 0.5" absolute floor, so this is caught only by the corroborated-apparition
    #   relaxation (SYS_BETWEEN_CORROB); it guards that path (see systematic_misfit).
    ("PAIR_2021TJ205", ["2021 TJ205", "2015 KS376"]),
    #   2025 NF505 = 2015 HG272: a SUB-FLOOR trend. 32/32 kept at 0.29" (each arc alone 0.08"/0.06"),
    #   but the 2025 Dec means walk -0.59 -> +0.68" over four X05 nights each tight to ~0.05". The
    #   whole residual budget is under both arcsec floors, so only the trend arm (SYS_TREND_Z,
    #   z ~14) sees it. Submitted as a clean 'y' and rejected by the MPC ("orbit fit failed").
    ("PAIR_2025NF505", ["2025 NF505", "2015 HG272"]),
    #   2016 UJ137: a SINGLE object (no linkage at stake) that guards the OTHER dominance test,
    #   SYS_SE_RATIO. All 6 of its nights sit 3-8 sigma off the orbit in RA, and its 2017
    #   apparition clears the strict 0.5" floor at 0.53" -- yet the within-night-scatter test
    #   scored it only 2.19x (its nights carry 4-18 obs each, so averaging pulls the mean far
    #   below the single-obs scatter the old ratio compared against) and the flag was lost. Against
    #   the standard ERROR of those means it is 3.7x. Guards that a well-sampled coherent offset
    #   is still called structure (see systematic_misfit / SYS_SE_RATIO).
    ("SYS_2016UJ137", ["2016 UJ137"]),
    #   2025 QH122 = 2014 HV101: an emitted 'y2' that is a THIN-COLLAPSED-ARC false positive. 2014
    #   HV101's solid 4-night 2014 apparition keeps 23/23, but its 2011 and 2023 apparitions are
    #   dropped whole AND 2025 QH122's own 3-night / 11-day arc collapses to a ~0.8 d (< 24 h) kept
    #   sub-arc (its 08-17 night dropped) -- so nothing on the 2025 side independently confirms the
    #   identity. identify() still emits it (the drops are whole tracklets, the kept obs fit 0.14"),
    #   but ident_check.thin_arc_falsepos flags it and the classify code becomes 'y2-falsepos?'.
    ("PAIR_2025QH122", ["2025 QH122", "2014 HV101"]),
    # --- hand-picked y2 identifications, chosen for coverage across the hard regimes ---
    #   2024 WY87 = 2007 PQ8: a y2 across a ~17-YEAR gap (the longest bridge in the pair set) whose
    #   arcs' photometry disagrees enough to trip the advisory magnitude flag -> classify 'y2-mag'.
    #   Guards the long-bridge link AND the _mag_stats brightness-residual path together.
    ("PAIR_2024WY87", ["2024 WY87", "2007 PQ8"]),
    #   2026 GV5 = 2014 EE314: a clean y2 across a ~12-year gap -- exercises the multi-revolution
    #   bridge (the fragile linkage subsystem) on a genuine main-belt identification.
    ("PAIR_2026GV5", ["2026 GV5", "2014 EE314"]),
    #   2016 FS162 = 2025 YA14: a SIX-opposition y2 spanning ~16 years -- a many-apparition link where
    #   the clean oppositions anchor the orbit (contrast the 2-opposition chimera bans above).
    ("PAIR_2016FS162", ["2016 FS162", "2025 YA14"]),
    #   2015 FL474 = 2021 TY207: a straightforward 2-opposition y2 -- the clean-baseline control for
    #   the pair set (a real identification that must keep identifying as the gates evolve).
    ("PAIR_2015FL474", ["2015 FL474", "2021 TY207"]),
    #   2015 KP420 = 2025 NH546: a genuine y2 whose BASE fit lands in a poisoned-anchor garbage
    #   minimum (463"), so the degenerate short-circuit (_ident_doomed_degenerate) FIRED and returned
    #   'no' -- a regression vs 4/9 days ago. The cheap multi-start re-anchor recovers the real orbit
    #   (25/28 @ 0.106", q~1.86 main-belt), so the gate now runs AFTER the re-anchor: this fixture
    #   guards that the still-degenerate check no longer kills a re-anchorable main-belt ident.
    ("PAIR_2015KP420", ["2015 KP420", "2025 NH546"]),
    #   2015 OF201 = 2025 NA385: a genuine cross-apparition link (distant object, 8d/12d arcs ~10 yr
    #   apart at 0.10") that fits only by dropping a lone verified-bad T09 night (1 of 3 in the 2015
    #   opposition). The SHORT+SHORT veto used to auto-dismiss it as 'no', then the 2-opp chimera ban
    #   held it at 'y-examine'; the SHARPENED ban (reject only when a side collapses to a lone kept
    #   night) now lets identify() auto-confirm it -- the wounded opposition keeps 2 nights, still an
    #   independent multi-night arc -> a real y2.
    ("PAIR_2015OF201", ["2015 OF201", "2025 NA385"]),
    #   2014 HH101 = 2022 DQ27: same sharpened-ban regime from the other side -- a 2-opp
    #   both-multinight y2 whose 2014 opposition clips 1 of its 5 nights (a whole 8-obs F51 night)
    #   and keeps 4: solidly multi-night after the drop, so identify() must emit (27/35 @ 0.09").
    ("PAIR_2014HH101", ["2014 HH101", "2022 DQ27"]),
    #   2015 RC182 = 2026 ES49: chimera BOUNDARY guard. A genuine NON-match ~11 yr apart. With
    #   _better_link's obs-count floor on the extra-opposition rule (see fit_object._better_link),
    #   the sigma_a re-anchor no longer fabricates a wrong-period two-opposition chimera here: the
    #   fit cleanly keeps the dense 2026 arc (27/27) and rejects the entire 2015 side (0/11), a
    #   single-opposition 27/38 @ 0.05" -> identify() stays None and _examine_or_no classifies 'no'
    #   by 0% retention (was a 19/38 partial link before the floor). Guards that the floor rejects a
    #   fake link WITHOUT flipping a real cross-apparition ID to a miss.
    ("PAIR_2015RC182", ["2015 RC182", "2026 ES49"]),
    #   2025 NM2 = 2021 WL5 (MPC-confirmed y2, exceptions = one junk 126 tracklet at ~76"): guards
    #   three relaxations at once -- (a) the scatter cap SCALES with the arc (3 isolated outliers
    #   among 62 obs, over the flat cap of 2); (b) systematic_misfit is leave-worst-night-out robust
    #   in a 4+-night apparition (a lone L04 night +1.2" off in Dec faked a 0.58" "trend");
    #   (c) thin_arc_falsepos ignores blunder-level drops (the 126 pair never testified, so NM2's
    #   kept 2-night arc is everything real it has -- not a collapse).
    ("PAIR_2025NM2", ["2025 NM2", "2021 WL5"]),
    #   2015 XO232 = 2025 PD58: the COINCIDENCE gate's negative (ident_check two-opp coincidence
    #   downgrade). Night-for-night the SAME shape as the confirmed 2015 OF201 = 2025 NA385 y2 above
    #   -- a 2-night ~1 d arc (2015) + a 4-night ~13 d arc (2025), one 2015 night dropped -- so no
    #   night-count / arc-length rule can tell them apart. The discriminator is WHY the night was
    #   dropped: OF201 clips a 23" gross stray (an obvious blunder), while XO232 clips a night only
    #   ~4.5" off that plausibly belongs to its coherent 3-night F51 sequence. Neither side is secure
    #   (no >=3 kept nights, no >20 d arc), so the ID rests on that soft drop -> a coincidence.
    #   identify() must DECLINE (return None) and _examine_or_no routes it to 'y-examine' (human
    #   review, NOT a 'no'); the implied-H also disagrees ~0.9 mag, so the CLI code is 'y-examine-mag'.
    ("PAIR_2015XO232", ["2015 XO232", "2025 PD58"]),
    #   2003 QC91 = 2006 QX180: the COINCIDENCE gate's GROSS-STRAY-drop negative (the second
    #   discriminator beside XO232's soft-drop case). Two sparse ~1-day arcs three years apart:
    #   2006 QX180 is 3 obs / 1.0 d, and 2003 QC91's own 5-obs / 58-day record collapses to a 1.0 d
    #   kept arc once its far station-304 night (78" off the joint orbit) is clipped. The clipped night is
    #   a gross stray (>10"), so OF201's hard-drop rule would trust it -- but it was 2003 QC91's OWN
    #   longer baseline, and after clipping it NO opposition keeps even a multi-day arc (both ~1.0 d,
    #   vs OF201/HH101's retained 9-12 d). So one orbit threads two single-tracklet-pairs by
    #   coincidence -> identify() must DECLINE, and _examine_or_no returns 'no' (2003 QC91's 60%
    #   retention is below MPC_MIN_RETENTION). Guards COINCIDENCE_MIN_KEPT_ARC_DAYS.
    ("PAIR_2003QC91", ["2003 QC91", "2006 QX180"]),
    # --- doomed-orphan fast-exit (fit_object._doomed_orphan) ---
    #   2010 OQ133 + 2006 WO78: a combined/ident request where 2006 WO78 (= 626129) is a mature
    #   14-opposition MBA that fits 136/152 @ 0.23", while the 2010 OQ133 tracklet is a single
    #   opposition sitting ~170deg (611203") off -- it simply does not belong. Because a 4+-opposition
    #   orbit is essentially never a mislinkage, the fit BANS that stranded apparition and skips the
    #   recovery cascade instead of re-grinding the doomed multi-rev bridge from every re-anchor
    #   (which took ~3 min -- the intruder even became a short-seed anchor, fanning the grind across
    #   all 8 real apparitions). NON-PAIR label => STRICT speed guard: this fixture's ~5s baseline is
    #   the whole point (it flags any re-perseveration). Orbit is byte-identical to the old 3-min grind.
    ("DOOM_2010OQ133", ["2010 OQ133", "2006 WO78"]),
    # --- hand-picked single-object guards (pinned when the ident-mode recovery-cascade
    #     short-circuit was introduced; single fits are ident_mode-agnostic,
    #     so these lock that in). Two distinct regimes: ---
    #   2014 EF94: an 11-OPPOSITION mature MBA (111 obs @ ~0.31"), over-determined -- guards the
    #   many-apparition fit and the over-determination early-exit in _rescue_single_night_poison
    #   (n_oppositions >= RESCUE_OVERDET_OPP), which the new degenerate-ident guard sits beside.
    ("GUARD_2014EF94", ["2014 EF94"]),
    #   2013 CP266: an outer-belt MBA at q ~ 4.94 -- right on the comet-nongrav q>4 floor
    #   (COMET_Q_MAX), so it guards that gate boundary in a different dynamical regime.
    ("GUARD_2013CP266", ["2013 CP266"]),
    # --- ident-mode ATTRIBUTION fast path (_ident_attribute_to_anchor): when one requested
    #     designation is independently well-determined (>= 5 opp), attribute the others to its
    #     precise orbit instead of a joint IOD/bridge grind. Two fixtures pin both verdicts: ---
    #   NON-MATCH: 2016 UK176 (1 opp) vs 2008 VC46 (8 opp, U=0). UK176 sits ~8.8 deg off VC46's
    #   orbit, so attribution keeps VC46's 72 obs and excludes all 58 UK176 obs -> a clean 72/130
    #   @ ~0.29" (classify 'no'). Guards that a doomed well-determined-member non-match resolves by
    #   ATTRIBUTION in ~0.2s rather than the ~52s combined-arc grind it used to take.
    ("PAIR_2016UK176", ["2016 UK176", "2008 VC46"]),
    #   (The GENUINE attribution case -- attribution must KEEP a real match, not drop it -- is guarded
    #   by test_ident_check.test_ident_attribution_keeps_genuine_match: it re-fits GUARD_2014EF94's
    #   obs with n_requested=3 so 2025 UO106's 10-opp orbit anchors, and asserts all 111 obs survive.
    #   A fixture here would triplicate the arc, since each of the 3 merged designations fetches the
    #   whole 111-obs identification.)
    # --- degenerate-gate q-guard (_ident_doomed_degenerate is MAIN-BELT-only, gated on a reliable
    #     anchor q, NOT on n_requested). 2021 WO is a q~0.6 Earth-resonant NEO whose long 2021 anchor
    #     seeds the wrong revolution: the base fit is garbage (rms>100"), and ONLY the recovery
    #     cascade's short-seed re-anchor rescues it to 83/83 @ 0.43" (opp2). It must survive the
    #     degenerate gate -- a naive rms>=100 skip (or the earlier n_requested>1 gate under an ident
    #     request) would strand it at ~412". Guards that low-q cascade rescues are never gated off.
    ("MULTIREV_2021WO", ["2021 WO"]),
    # --- SHORT-LINK advisory (ident_check.thin_link / THIN_LINK_MIN_OBS) ---
    #   1999 KK17 = 2016 LL89: an identification that looks immaculate by every gate we had -- 9/9
    #   obs kept, nothing dropped, 0.56" -- because all of those gates judge what the fit DISCARDED.
    #   With 5 + 4 observations the joint orbit has 18 residuals against 6 parameters, so "it fits"
    #   is nearly free: nothing here TESTS the identification. identify() still emits it and the
    #   classify code stays a 'y', but it gains the '-short' suffix and the written record a
    #   '# FLAG short link' comment (advisory, never a rejection).
    ("PAIR_1999KK17", ["1999 KK17", "2016 LL89"]),
    # --- single-night-POISON dominance (_rescue_single_night_poison's acceptance test) ---
    #   2015 HP527 = 2025 PS30: 2025 PS30's record carries a 4-obs 2021 V00 night that is not this
    #   object, and it is its OWN apparition -- a lone tracklet with nothing beside it to contradict
    #   it, i.e. near-unit leverage. The joint fit therefore satisfies it EXACTLY and pays for it
    #   everywhere else: 26/40 @ 0.43", with the 2015 and 2025 nights each sitting coherently off the
    #   orbit -- which ident_check reads (rightly) as a residual TREND, so the verdict was
    #   'y-systematic'. Set that night aside and the remainder fits 32/36 @ 0.09" with the 2021 obs
    #   18' off. The sn-rescue used to REFUSE that better orbit on its >=90%-of-`reduced` link test:
    #   the 4 obs it does not link are a SECOND junk tracklet (a 1-degree-off 2015 W84 night filed
    #   under PS30) that the poisoned fit had rejected too, so 32 lost to the 32.4 bar over data that
    #   was in neither orbit. The DOMINANCE clause -- the rescue gives up nothing `best` was keeping
    #   -- accepts it. Guards the whole chain end to end: 32/40 @ 0.09" and a clean 'y2' excepting
    #   both junk tracklets (2015-04-22 and 2021-10-02).
    ("PAIR_2015HP527", ["2015 HP527", "2025 PS30"]),
    # --- HARD ASTROMETRY, not a mismatch (ident_check.kept_obs_fit's sigma clause) ---
    #   2025 MT356 = 243P, a COMET. The joint fit lands at 0.647" -- over the flat 0.6"
    #   MAX_KEPT_RMS that used to be an outright veto, so --ident-check filed a real
    #   identification as a flat 'no'. But the excess is 243P's OWN astrometry, not the link:
    #   243P fitted ALONE (the 243P fixture below) reaches only 0.649", with the same per-night
    #   trend and the same 85 rejects, so the identification makes the fit marginally BETTER --
    #   and 2025 MT356's own 6 obs are the TIGHTEST part of the whole fit at 0.036". A coma
    #   centroid plus two decades of mixed astrometry cannot deliver 0.6" for ANY set of
    #   designations, which is why the gate is now also asked in sigma-normalized currency
    #   (this fit is 0.75 sigma -- mid-population). THE ONLY FIXTURE THAT EXERCISES THAT
    #   CLAUSE: every one of the other ~98 pairs fits under 0.6" in arcsec (population max
    #   0.576"), so without this one the relaxation is untested. See
    #   test_ident_check.test_hard_astrometry_pair_is_examined.
    ("PAIR_2025MT356", ["2025 MT356", "243P"]),
    #   ...and 243P by ITSELF, which is what makes the claim above testable rather than
    #   asserted: the pair fixture alone cannot show that the link is not what costs the rms.
    #   The two are compared directly in test_hard_astrometry_pair_is_examined.
    ("243P", ["243P"]),
    # --- SHORT+SHORT systematic (ident_check.short_arc_systematic) ---
    #   2025 OC732 + 2014 HJ409: two 3 d arcs the joint orbit keeps whole (27/27 @ 0.57"), yet the
    #   Dec per-night means sit 0.78" apart at 27x their own error. With no curvature on either
    #   side there is nothing a reviewer could weigh that trend against, so it is a flat 'no' rather
    #   than a 'y-systematic'.
    ("PAIR_2025OC732", ["2025 OC732", "2014 HJ409"]),
    # --- SHORT+SHORT INTERLEAVED (ident_check._interleaved, MARGINAL_REJECT_SIGMA) ---
    #   2026 SW11 = 2026 SA12: two 0.9 d arcs from the SAME 22 hours (H36 on both, I41/I52,
    #   ...), 41/48 at 0.49" -- better than 2026 SA12 alone (0.586"). The seven drops are all
    #   marginal (2.7-3.6 sigma), two of them the 703 pair 2026 SW11's own solo fit rejects.
    #   The same-apparition short+short 'no' and the mostly-clipped-tracklet refusal both assume
    #   the drops BOUGHT the join; interleaved arcs test the orbit at the same epochs, so they
    #   did not -> a clean 'y'. test_ident_check also shifts 2026 SA12 onto a wrong track (1"
    #   offset, 3"/day drift) and asserts the same fit is still refused.
    ("PAIR_2026SA12", ["2026 SW11", "2026 SA12"]),
]


def main():
    os.makedirs(FIX, exist_ok=True)
    want = sys.argv[1:]
    oc = obs_sources.load_obscodes()
    for entry in OBJECTS:
        label, desigs = entry[0], entry[1]
        opts = entry[2] if len(entry) > 2 else {}
        force = opts.get("force", False)          # this object only links under --force
        benchmark = opts.get("benchmark", False)  # heavy: speed-benchmark only, skipped by
                                                  # the routine correctness run (no fit-qual guard)
        if want and not any(w in label or w in " ".join(desigs) for w in want):
            continue
        try:
            # Always the canonical ADES path. An over-confident survey sigma that used to force
            # a couple of pairs onto the OBS80 weighting (the old `obs80` opt) is now handled
            # inside the fit by the sigma.txt reject-rate relaxation, so every fixture mirrors
            # the one path the CLI / web / ident-check actually run (see
            # arc_extend._relax_overconfident_obscodes; 2015 KR274 = 2025 PY146 links here).
            obs, meta = obs_sources.fetch_observations(desigs, oc)
            # CPU seconds, not wall clock -- the fetch above is network-bound and the
            # timing baseline must be the same currency test_regression's speed guard
            # measures (see its module docstring).
            t0 = time.process_time()
            seed = None
            if opts.get("mpc_seed"):
                # PRODUCTION-PATH fixture (the _sweep docstring's "reproduce the production
                # path deliberately"): mega-arc objects are fit seeded in production (cli/web
                # auto-seed >500-obs multi-opposition objects), and the thing such a fixture
                # pins -- 1862 Apollo's perturbed-delivery Yarkovsky -- lives on that
                # path (the cold IOD fit of a 96-year arc is a still-open convergence
                # item, a different thing to measure). The seed STATE is frozen into the
                # fixture so the suite refit needs no network and cannot drift with the MPC's
                # own orbit updates.
                live = fo.fetch_mpc_seed(obs, meta)
                if live is not None:
                    seed = {"state_eq": [float(x) for x in live["state_eq"]],
                            "epoch_tdb": float(live["epoch_tdb"])}
                    # A published COMET orbit's A1/A2/A3 govern the seed's re-anchor
                    # (_fit_from_mpc_seed): frozen beside the state so the fixture takes the
                    # same path the CLI does (21P).
                    if live.get("comet_nongrav"):
                        seed["comet_nongrav"] = {k: float(v)
                                                 for k, v in live["comet_nongrav"].items()}
            fit = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta,
                               verbose=False, force=force, mpc_seed=seed)
            golden = compute_golden(fit, time.process_time() - t0)   # incl. timing baseline
            # EVERY declared flag, not a hand-picked two. `opts` is the declaration and the
            # fixture JSON is the artefact, and test_regression reads its flags off the ARTEFACT --
            # so a key named in OBJECTS but not copied here is silently lost on the next rebuild.
            # That is not hypothetical: `quality` was declared for 2008_NA in 83e4e13 and dropped
            # four commits later when the fixture was rebuilt with fresh astrometry, leaving the
            # one object with a fit-quality guard running speed-only for two days while the
            # declaration still said otherwise. test_fixture_flags.py now asserts the two agree.
            payload = {"label": label, "desigs": desigs, **opts,
                       "force": force, "benchmark": benchmark, "meta": meta,
                       "obs": slim_obs(obs), "golden": golden}
            if seed is not None:
                payload["mpc_seed_state"] = seed      # frozen; refit passes it back verbatim
            write_fixture(os.path.join(FIX, label + ".json"), payload)
            print(f"  {label:20} {golden['used']}/{golden['total']} @ {golden['rms']}\" {golden['status']}", flush=True)
        except Exception as ex:
            print(f"  {label:20} ERROR {str(ex)[:50]}", flush=True)


if __name__ == "__main__":
    main()
