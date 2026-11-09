# src/pyroorb/itf — native ITF search

Search the MPC **Isolated Tracklet File** (plus the pseudo-ITF of few-tracklet
/ no-orbit designations) for tracklets belonging to a known object, entirely
on the PyRoOrb engine — the find_orb-era "itfsearch" notebook pipeline rebuilt
as a first-class feature.  find_orb is not involved anywhere: seed fits, the
ephemeris corridor (geocentric, with the fit covariance projected to a sky-
plane sigma), every trial linkage fit, residual-based dropping and the U/q/rms
gates all run on `fit_object.fit_orbit` and friends.

## Two currencies: the line is the identity, ADES is the data

Everything the search *administers* is 80-column: the tracklet index stores raw
OBS80 records, an observation's identity is a column slice
(`obs80_fields.obs_unique_id` — date+time, sky position, station), nights and
oppositions are counted off dates in columns, and the MPC identification
document we submit is a pile of lines.

Everything the search *fits* is ADES.  Rendering an observation to 80 columns
drops the submitter's `rmsRA`/`rmsDec` (so the fit would fall back to the
Veres-2017 station model even where the real per-obs σ is known), `rmsMag`,
`logSNR`, `obsID` and `mode`, so those ride alongside in an `AdesRegistry`
(`obsreg.py`) keyed by line identity, and are put back on the observations
before every fit.  Two sources fill it: the seed fetch (ADES, like every other
fetch in the engine — the web UI hands over the observations it already has via
`seed_obs=`), and the candidate rows, because **an `obs_sbn` row IS an ADES
record** — the table's columns are the ADES field names (the MPC's `ADES_DF`
API serves the same table). The registry rides out on
`ItfSearchResult.registry`, so a caller re-fitting `combined_obs80` — the web
UI does — fits what the search fitted rather than its 80-column shadow.

Coverage is honest: ~26% of ITF observations were submitted natively in ADES
but only **~7.5% carry a reported rmsRA/rmsDec**, so most candidates still
weight off the station model exactly as before.  Positions and `astCat` are
deliberately NOT taken from the database row — see the module docstring in
`obsreg.py` for why.

## Flow

1. **Seed** — fetch the object's published astrometry (**ADES**) and fit it. The seed fit is
   screened once for asteroid perturbers beyond the 16 (`fitter.seed_perturbers`,
   `ItfConfig.seed_perturber_screen`): when its refit keeps any, the search latches that field
   (`SearchContext.perturbers`) and every trial fit runs in it — 2016 RM33's seed keeps (19)
   Fortuna, and the 2005 night past Fortuna's 2011 pass then fits in a trial (in gravity it is
   rejected). Trials themselves are never screened (thousands per search).
2. **Corridor** — sweep a geocentric ephemeris across the ITF era
   (1997→now by default) and cut it into quadratic corridor windows
   (`ephem.py`), each gated by predicted sky uncertainty and magnitude.
   The corridor half-width scales the fit's 1-σ sky sigma by `uncert_mult`
   (2.5, for the along-track tail of a moderately-weak single-opp seed).
   The magnitude gate (`mag_limit`, 23.5) drops the windows where the object
   is predicted too faint for the ITF to hold a detection at all — the ITF's
   own depth is a spike at 21.5–22.5 (p50 21.98, p90 23.48), and the window
   count falls off a cliff right there: 23.5 sweeps ~65% of what the legacy
   24.0 did, and 23.0 ~a third. A live 16-fixture A/B at 24.0 vs 23.0
   recovered **identical** tracklets 25% faster, and 23.5 sits above that
   measured-safe point. `--mag-limit MAG` overrides it on any surface
   (`itf search`, `itf batch`, `fit_object.py --itf-search`, the web options
   modal, which fetches the engine's value rather than carrying a literal);
   `--thorough` sweeps to `mag_limit_deep` (24.0) instead.
   That number is the DEFAULT, and it is deepened per object
   (`ephem.faint_cut`): an object never brighter than 22.5 since 1995 is swept
   to 1 mag past its brightest predicted V (`mag_floor_since` /
   `mag_floor_margin`), because a fixed cut deletes a faint object's whole
   corridor rather than trimming it (2026 DR69: zero windows at 23.5, linked
   at 24.81). A cut the user types (`ItfConfig.set_mag_limit`, every surface)
   is applied exactly, with no floor.
   An automatic *deeper retry* when nothing links exists but is **off**
   (`mag_limit_fallback`, `--mag-fallback` to arm it): the recall it was built
   to protect proved not to be at risk at 23.0, and it cost ~4% when it fired.
   Kept dormant rather than deleted — see the config comment.
   When the linear pass fails to LINK and the seed is under-determined
   (`U ≥ variant_corridor_U`, 4.0 — at which point the true position years
   away can be off by *tens* of degrees), the search then **scans variant orbits**
   sampled along the Line of Variations (`variants.py`): each a legitimate
   orbit, central-out, with a thin corridor, all K tracks from ONE shared
   ASSIST integration; early-exit on the first clean link, then the ordinary
   recursion — which now sees a collapsed U — picks up the rest.
3. **Candidates** — per window, pull candidate tracklets (`candidates.py`):
   * `index` path (default): the precomputed `itf.itf_tracklets` table
     (healpix prefilter + corridor + per-tracklet uncertainty, one row per
     tracklet) with the **certainty match-χ²** ranking.  A whole corridor is
     swept server-side, ROUTED by corridor geometry (`sweep_mode="auto"`,
     default; the router is `DbSource._route_sweep`): a narrow corridor uses
     the set-join (each window's (healpix,
     epoch, dec) box rides along in a joined `unnest(...)`; 5.4 s → 0.5 s on a
     907-window corridor), a wide-but-distinct corridor collapses each
     window's pixels to contiguous NESTED ranges + the motion bound in ~48
     window batches (32 s → 14.6 s on a 272-window corridor), and a
     many-overlapping-window corridor (the LOV variant union, ~19k thin
     windows) falls back to the union-box chunker, where a per-window join
     would multiply each row per matching window (109 s vs 285 s).  Identical
     candidates in every cell, measured; `sweep_mode="setjoin"` /
     `"rangejoin"` / `"chunk"` name a single path and FORCE it (no fallback)
     for an A/B;
   * `scan` path: the legacy box+quadratic-corridor SQL straight against
     `obs_sbn` (kept for A/B of candidate completeness).
The corridor test runs SERVER-SIDE.  Both index sweeps send each window's
quadratic and tolerances with the query -- the rangejoin as a join predicate,
the chunker as an `EXISTS` over the chunk's windows (so a row survives if ANY
of them can accept it, with no join multiplying rows per window) -- alongside
the motion bound that was already pushed down.  The union box a chunk fetches
is far wider than the thin corridors inside it, and on a LOV variant union
(thousands of corridors over the same era) nearly everything it returns can be
ruled out before it becomes a Python object: peak RSS on 2022 GL24 fell 2040
-> 913 MB at identical results and runtime, and objects that could not finish
at all on a 4 GB box (2023 HC31, 2015 VV222) now do.  The contract is the same
one the box bounds already had: every bound sent to SQL is rounded OUTWARD, and
`_survivors` re-applies the exact Python predicate to whatever comes back --
SQL narrows, Python decides.  Verified by sweeping a real 695-window corridor
with the pushdown on and off: identical candidate sets, both sweep modes.

4. **Score & trial-fit** — group obs into tracklets, score against the
   corridor (qual/dist/likely-maybe-unlikely), fit published+likely with the
   engine, back off to best-ranked when the fit chokes, recurse while the
   used-obs count grows (`search.py`).  Taken tracklets accumulate as a
   **hypothesis** that rides in every deeper trial fit (in by default, gone
   from the candidate pool — deeper levels only ask "which remaining
   candidate next?"), and each trial is fitted-then-**pruned**: a blunder
   tracklet's obs clipped individually (never a seed/hypothesis line) instead
   of one bad candidate burning the whole joint trial.  The per-recursion
   seed guard has an absolute noise floor (`recurse_seed_abs_arcsec`,
   0.5") — an overfit 3-night seed honestly widening to ~0.45" under the
   true multi-opposition orbit is not contamination — plus a seed-retention
   floor (`recurse_seed_retention`, 80% of seed obs must stay included).
   **A hypothesis is fitted ONCE per search** (`_hypothesis_key` / `ctx.tried_hypotheses`).
   The identity of a search is the ASTROMETRY SET it would fit — seed +
   accumulated hypothesis + this trial — plus the force model, and nothing
   else.  Everything the recursion does below a take follows from that set:
   the fit, the orbit, and a corridor that sweeps the WHOLE ITF era, so a
   second attempt at the same set cannot see anything the first did not.
   Which round asked, what that round's ranking called the trial, which LOV
   variant retrieved it and which untaken candidates that round carried down
   are all facts about the round, not about the question.  The trial queue's own
   `tried_keys` cannot express this: a `TrialQueue` is built per ROUND (per
   recursion level, per LOV variant), so eight variants meant eight empty
   sets, and its key is a tuple of positions in that round's ranking, which
   names a different tracklet in every round.  2006 UP17 is the case it was
   written for: `p00046@F51/2025-04-23` was taken, fitted (41/41, rms 0.505"),
   recursed on and vetted in EIGHT rounds — bit-identically, because the
   variant orbit picks which candidates are RETRIEVED and never enters the
   trial fit.
   **It is not a ban on a TRACKLET, and that distinction is the point.**
   `{seed+A}` asked once leaves `{seed+A+B}` askable, because that is a
   different question — and take A may have been barren *precisely* because A
   needs B to guide the convergence.  So barren-take backtracking no longer
   bans what rode in the take (`TrialQueue.mark_failed` used to, which barred A
   from every later joint trial of the round); it re-queues the members the fit
   clipped and resumes, and the ledger alone stops the repeat.  By the same rule a
   candidate refused on its own, or never taken at all, is still free to ride
   in a larger set later.
   **Two carve-outs, both the rule's own logic rather than exceptions to it.**
   The force model is IN the key: a Yarkovsky rescue flips `ctx.nongrav`
   mid-search and it stays on, so the same lines under gravity and under
   Yarkovsky are two different questions — the whole point of
   `_nongrav_rescue`.  And a take whose subtree DROPPED windows (a row budget,
   a query the server cancelled, the low-memory sweep floor — `_windows_dropped`
   reads the tally `_fold_sweep_truncation` keeps) searched only PART of the
   ITF, so its answer is not the ITF's answer and the hypothesis stays askable.
   A warm-screen kill is not recorded at all: it is a cheap per-round
   pre-filter judged against that round's parent orbit, not an answer about
   the astrometry.
   **`_CachedSource.lookup` is THE accessor** for a consumer that wants the
   variant scan's union-sweep rows rather than to sweep
   (`lov_profile.build_lov_profiles`).  That consumer used to reach into the private
   dict and spell the key itself, which made the cache's keying a fact two
   modules agreed on with nothing enforcing it: re-keying the cache made every
   LOV profile lookup miss, so the pass profiled NOTHING — no sharp candidates,
   hence no priority variants and no LOV-centered or cluster trial sources — and it
   degraded in silence with the whole offline suite green.  The only tell was
   a live trace reading `priority variants: none`.
   `test_lov_profile_reads_the_union_sweep_through_the_accessor` is the canary:
   every other LOV profile test drives `LovProfileSet.add` directly, so the
   sharpness maths was covered and the PLUMBING was not.
5. **Gate & link** — night/opposition/rms acceptance gates, then the MPC
   identification JSON (`linkage.py`).  Submission is **dry-run unless
   enabled twice** (`--itf-submit` *and* `ItfConfig.submit`).
   Two gates vet an accepted linkage, and they are complementary: the rms
   gate (`extension_degrades`) asks whether the seed obs that REMAIN still
   fit, and the **seed-clip ledger** (`_seed_clip_ledger` /
   `_clip_evidence_ok`) prices what the link threw AWAY — because an orbit
   that bends far enough for the engine to reject the object's own astrometry
   scores an *improvement* on the first test.  A clip is not forbidden (one
   bad published observation must not make an object permanently
   unrecoverable, and a recovery is often how a bad night is proven bad): a
   whole seed SPAN (`opp_profile.spans_of`, the gate's own 36-hour unit) may go,
   but only if bought with `clip_span_price_tracklets` (3) newly linked
   tracklets per span, only `clip_spans_max` (1) of them, never leaving under
   `clip_keep_spans_min` (2) spans of the object's own arc — and always
   REPORTED, in `result.warnings` and `vetting["seed_shed"]` (whose keys still
   say `nights`; the batch CSV and research JSONL carry them): *"we found this
   object, but span 4 of its published arc had to go; here is what says that
   span is bad."*  Losing scattered obs from nights that survive is ordinary outlier
   rejection: recorded in `vetting` and the log, deliberately NOT a
   warning — web submission makes the user acknowledge every warning, and
   routine rejection as a click-through is how real cautions stop being
   read.
   **A degraded seed fit is never `linked`** (owner rule 2026-08-02): when the
   seed's own rms rises past the warn level (`warn_rms_factor` 2.0, or
   `+ veto_rms_floor_arcsec`) — the same measurement as the `extension_degrades`
   veto, one threshold down — the linkage is demoted to `status=0.6`
   **possible** (`vetting["seed_degraded"]`) instead of accepted.  It keeps its
   fit, astrometry and warnings and is reported everywhere, but no surface
   submits a 0.6: not `itf batch --submit`, not the CLI, not the web button.
   The object's own observations fitting worse is what a contaminant looks
   like, and that is not a caution to click through unattended.
   **Two answers, one search** (`ItfConfig.prefer`): when the search holds a
   `linked` linkage AND a reviewable `possible` that links strictly more of the
   ITF (its extra oppositions too thin to confirm -- 2014 GV99: one 2014
   tracklet linked, against a 0.6 adding a lone night in 2006 and in 2022), it
   keeps both.  `prefer` picks which one leads; the other rides on
   `ItfSearchResult.alternative`.  `complete` (the default; the CLIs and the
   recovery harness) leads with the fuller 0.6, `submittable` with the linkage
   (automated submitters and the identification queue set it, and the
   CLIs default to it under `--submit`; `--prefer` / `--itf-prefer` choose).
   The CLIs print the other one as an `ALSO:` line; the web asks the user
   which to show.
   **A PARTIAL PICKUP of a designated object never claims that designation**
   (`itf/pickup.py`, owner rule 2026-08-19).  The search links TRACKLETS; the
   document names DESIGNATIONS.  When a linkage takes only part of a
   designated (pseudo-ITF) object, the archive's other tracklets of it are
   measured against the linked orbit at their own station and epoch, and if
   one does not fit (`pickup_off_arcsec` 10″ **and** `pickup_off_sigma` 5σ)
   then that designation is ITSELF a mislinkage — only part of it is this
   object.  An identification document cannot say so, so: the designation is
   dropped from the document (`vetting["linkage_dropped_designations"]`), the
   linkage is submitted without it when what remains still stands on its own
   (`_reduced_submittable`), and otherwise nothing is built at all and the find
   becomes a `status=0.6` **possible** whose warning names the MPC **Jira**
   route.  **That last test is the MPC's criteria (`itf/mpcrules.py`), not our
   own `_opp_profile_gate`** — including the owner's 1.5/1.75 tightening, which asks
   whether WE believe a thin profile and was already asked and answered on the
   full arc.  The ban removes a NAME, not evidence: the tracklet that convinced
   us is still in the fit and on screen.  2024 TF21 is why — its remainder (5
   published 2024 nights + a 2016 night + a 2017 night) reads 1.75 under
   `count_opps` and the belief gate refused it, while the MPC's rule for 3
   apparitions is one apparition of ≥3 nights, which 2024 satisfies five times
   over.  A 1.5 (one lone new night) still fails, and so does a 1.75 whose only
   multi-night apparition has two.  The
   fit, the astrometry and the web UI keep the tracklet either way — it is
   real astrometry of the object — and there is no refit: dropping a tracklet
   the orbit already accommodated does not move it.  The case that found this:
   2024 TF21 linked one of 2017 SS396's three tracklets (2017-10-15 G96, 0.4″
   on a 0.33″ orbit) while the other two sit 1.05° and 0.93° away.  A
   part-take whose leftovers DO fit is recorded in
   `vetting["partial_pickups"]` and logged, but raises no warning: that is
   just a corridor that never swept those nights.
   The one exception to "1.5 opps is never submittable" is a **NEO**
   (`_neo_orbit_waiver`): a link on a body that comes near the Earth is worth
   publishing to the MPC recovery page as a *proposed* identification, so
   acceptance waives the whole opposition profile when the trial fit has
   `q < neo_waiver_q` (1.3 au).  That waiver requires a **bound** orbit
   (`e < neo_waiver_max_e`), and a fit with no eccentricity at all fails it.
   The reason is 2002 PS152: a catalogued 39.4 au TNO, 4 observations over 3
   days, whose range is wholly unconstrained — it linked one ITF night and
   fitted at e 6.81, a −0.02 au, "q" 0.11 au.  That perihelion is an artefact
   of the negative semi-major axis, not a claim about the body, and it bought a
   degenerate arc a `linked` verdict at 1.5 oppositions.  A real sungrazer is
   unaffected (Phaethon's q is 0.14 au — on a bound orbit).
   Thin-profile tiers: **1.5 opps** = one single-night extension —
   shown as a POSSIBLE/LIKELY linkage (`status=0.6`, fit attached, "a lone
   best-fit tracklet, not submittable without a second night outside the
   initial opposition; search archival images for it") under ONE principled
   gate (`_show_lone_night`, three clauses, **path-agnostic** — linear OR
   LOV-variant): **(1) opposition profile** — a LONE tracklet (a single trial — one
   identification to vet, at any trial queue rank) lifting a 1-opp seed to 1.5 opps
   at depth 0, on a seed not itself badly
   under-determined (`pub.U < single_night_show_max_U`, 9.0 — distinct from
   the `variant_corridor_U` 4.0 fallback trigger); **(2) it fits, in σ units
   not overall arcsec rms** — the new tracklet's residuals consistent with
   *its own* uncertainties (χ² within the `show_new_tracklet_conf` ~3σ tail
   of its 2·n dof) AND the original astrometry not made statistically worse
   (a Wilks Δχ² test: re-fitting moves the seed obs' χ² by ≤ the seed's own
   6-dof `show_seed_preserve_conf` confidence region, σ held fixed at the
   seed's, a rejected seed ob counted at full residual) — so the overall rms
   may grow 0.1″→0.2″ freely when that is just the noisier new night;
   **(3) it is the only one that fits — or bounded multiplicity** — exclusivity
   is **measured, not inferred from the scores** (`_exhaustion_verdict`).  Walk
   the WHOLE ML-scored field (likely + maybe, NOT just the "likely" geometric
   bucket — a high-P "maybe" is still a comparable fit, and the trial queue only ever
   tries "likely"s; ranking only "likely" once falsely showed 2024 RD175's
   0.999/0.887/0.734 field as a sole standout) for the first candidate under
   `show_exhaust_frac` (⅓) × **the winner's** P — the *cliff* — looking no
   further than `show_exhaust_max_rank` (10).  Every rival above the cliff must
   be ruled out: the trial queue already tried and rejected it, or it is
   refitted here **against the bare seed** — an alternative to the winner, not
   an addition — and judged by the trial queue's own take criterion
   (`_trial_rejection`, shared code so the bar is identical).  None links →
   **sole standout**, shown with a single banner as "the object".  A rival
   links, or the field has no cliff inside rank 10 (exclusivity unclaimable),
   or the budget runs out mid-test (an unfinished exhaustion is not evidence) →
   not exclusive; while the top candidate is still strongly plausible
   (best ≥ `show_multi_min_ml_p`, 0.7, and best ≥ `show_min_ml_p_abs`
   throughout) it is **still shown**, with a SECOND same-severity banner ("not
   necessarily the only match … a lead to verify, not a confirmed
   identification") — both banners ride `result.warnings`, rendered as
   identical caution banners.  Below the 0.7 bar and not exclusive → silent.
   **How many fit, when clause 3 cannot say (web only, 2026-08-19).**  A field
   with no cliff inside rank 10 is shown on the winner's P alone, exclusivity
   having fitted *nothing* — and that is where the mistakes are: precision
   53/54 where exclusivity was measured, 26/49 where it was not
   (`research/onenight-gates/`, 103 shows).  No ML-score density proxy
   separates those (~2:1 at best; P is a ranker, not a fit test), but COUNTING
   does: `search.count_linking_rivals` fits the top `show_n_rivals`
   rivals against the bare seed and counts how many link — 0–2 for everything
   the search was right about, 3–4 for the crowded junk — so
   `show_rival_count_max_linkers` (3) withholds the show and says how many
   fitted.  It costs k trial fits (median 12 s, ~+10% of a search, on the
   ~quarter of searches that reach the path), so `show_n_rivals` is **0**
   in the engine and only `web/engine_api.py` sets it: the banner exists to
   send a person to the archives, and that is the surface where the price buys
   something.  The cut rests on one crowded object — a rare-event flag, not a
   calibrated threshold.
   **A foreclosed show says so UP FRONT** (`_note_show_floor`).  A 1-opposition
   seed's likeliest find is a lone new night, and only the show can report one
   — so when the whole depth-0 linear field scores under `show_min_ml_p_abs`,
   that verdict is settled before a single trial fit.  It is announced there,
   exactly as the seed-U foreclosure already is, and recorded as
   `telemetry["show_floor"]`.  2006 UP17: field max P 0.07 against the 0.15
   floor across all 23 LOV variants, reached after 35 s and eight rounds; the
   warning now lands at 7.8 s, before the variant fan that costs the rest.  A
   warning and nothing else — no gate moves, and a multi-night apparition does
   not need the show, so the search still runs in full.
   **A silence says why.**  Every `return None` in `_show_lone_night` goes
   through `_refuse`, which records the refusing clause as a sentence on
   `ctx.show_refusal`; the rejection ledger prints it (`nothing shown: …`) in
   place of the old `1.5 opps without progress,
   abandoning` — wording that described the trial queue rather than the
   finding, and that a reader reasonably took as "the search found nothing"
   when the truth was "something linked and we distrust it".  The crowded-field
   case says so outright: *the tracklet that linked is almost certainly one of
   many that would fit this orbit just as well … showing it would most likely
   be showing you a coincidence*.  The web payload flags the same distinction
   with `itf.suppressed`, and the UI raises a caution banner ("a candidate
   tracklet linked but is not being shown — …") instead of the "no additional
   observations found" notice.
   That sentence is printed **whichever branch of the ledger answers**: on 2006
   UP17 it was swallowed by the LEGACY `trial.rms >= rms_max` branch one clause
   earlier (0.505" over a 0.44" cap the acceptance path never asks — on an
   object whose own astrometry scatters 0.514"), so a night fitting at χ² 2.8
   against a 19.8 cap, seed intact, was reported as "too few
   nights/oppositions to submit".  And the **tracklet itself** is named at the
   end of the search (`_recurrent_notes` → `result.diagnostics`, never
   `warnings`): a silenced show leaves a 0.6 carrying no fit and no astrometry
   — deliberately, since that is what tells backtracking the branch was barren
   — so without this the candidate the search kept returning to disappeared
   entirely.  The ledger already knows it, because it is what stopped the ninth
   attempt.
   Clause 2 runs BEFORE clause 3: the σ tests are free, the exhaustion spends
   trial fits (2024 QT17: 4 refits, ~1 s).  NB a trial taken at rank 3 because
   ranks 1–2 were fitted and FAILED is exactly as exclusive as one taken at
   rank 1 — the retired `is_top_ranked` prefix rule silenced the former.
   Sole standouts are tiered **likely** vs **possible** by the new tracklet's
   reduced χ² and best P(match) (`show_likely_new_redchi2` /
   `show_likely_min_ml_p`); a multiplicity show is always **possible**.  The
   progress signal is "one ITF
   tracklet survived into a new opposition and still fits", not a raw
   obs-count gain — so it still surfaces when clipping a now-stale seed
   obs on the (far longer) linked arc drops the net used-count to at or
   below the seed's; **1.75** = two
   single-night extension opps (possibly combinatoric accumulation via
   recursion) — always shown as POSSIBLE with the combinatorics caution.
   A deeper recursion level's possible linkage supersedes a shallower
   frame's weaker verdict.  Possible linkages are never submittable (no
   submit, no linkage-file download).  **Confirmed-pair upgrade**: when the depth-0 sweep's
   top-2 tracklets by ML rank are what the search links, both `P(match) >=
   confirm_min_ml_p` (0.2), and every linked obs fits within
   `confirm_max_resid_arcsec` (0.5") in the final orbit, the 1.75 profile is
   declared **3 oppositions** and accepted normally (auto-submit allowed).
   Two ways to be that pair: taken together in ONE trial (a joint hypothesis
   from the seed orbit alone), or — owner rule 2026-08-02 — the accepted
   linkage turning out to be *exactly* those two however the trial queue got there
   (`ctx.depth0_top2`, recorded before any trial runs).  The trial queue authors no
   top-2 trial (top-10 → top-3 → singles), so the one-trial shape needs the ML
   floor to leave exactly two candidates; whether the pair was fitted at once
   or accumulated best-first-then-the-other is a fact about the trial queue, not
   about the evidence — the ranking that named them is the seed orbit's,
   formed before either was linked.  A linkage that is *more* than the pair is
   a different claim and still faces the seed-strength paths and the 0.6
   review.  **Any warning withdraws the upgrade** (owner rule 2026-08-02): the
   confirmation is a judgement about the PROFILE, made before the gates run, so
   a caution they then raise — a clipped seed night, a systematically offset
   linked night, a tracklet off the predicted brightness — is evidence against
   exactly what it waved through.  The upgraded linkage falls back to
   `status=0.6` (`vetting["confirm_withdrawn"]`), and the reduced-linkage fallback
   below is tried, as it is for any refused 1.75.  A warning on an
   linkage with an ordinary profile still accepts.

## Residual trends: the shape of a tracklet's residuals, not just their size

Every other gate measures how BIG a linked tracklet's residuals are — the
per-tracklet χ² against its own σ, the seed's Δχ², `extension_degrades`. None
looks at their ORDER, and a χ² cannot tell a trend from a scatter: ±1.25 σ either
way scores identically.

Over a tracklet's ~30–60 minutes the object's motion is a straight line to any
accuracy the astrometry resolves, so residuals that *walk* across it mean the
orbit has that tracklet moving at the wrong **rate** — right place, wrong speed,
which is what a plausible wrong linkage looks like. (The candidate ranker already
measures the pre-fit version as its `motion-mismatch` feature, but nothing gates
on it and the trial fit discards it.)

> **This gate was withdrawn on 2026-08-25 and is now a diagnostic.** The full
> calibration measured it firing on **2 of 58 correct** tracklets against **3 of
> 119 wrong** on the population where it is the last word, with no threshold
> separating them (false positives reach zero at 12×, where catches do too). It
> is computed and reported — `ItfSearchResult.diagnostics`, the CLI's `NOTE:`,
> a web info banner, and `n_diagnostics`/`diagnostics` in the batch
> CSVs — and it decides nothing in the search. Neither consumer below is armed
> any more. **Since 2026-09-28 (owner) the CSVs also count it as a review
> warning** (`itf.batch.review_warnings`): it lands in the `warnings` column and
> in a batch review's warned pile, like a slight magnitude disagreement.
> FINDINGS.md, Finding 6.

`itf/residual_trend.py` has **one clause**. A linked tracklet is flagged when all three
hold:

* **`mono = 1`** — the residuals *walk*, every adjacent step going the trend's
  way, and still do with any single observation dropped. `+2.0, −0.1, +0.1` is
  one bad first observation that a line fit reports as a steep slope; it scores
  0.5 and is not a trend.
* **`≥ 4.0×`** the worst trend anywhere else in the object's OWN arc — the **raw**
  swing on both sides of that ratio. A leave-one-out ratio shrinks its numerator
  only where LOO can run (n ≥ 4, median 0.67×) and not at all at n = 3, so it
  inflates a small candidate ~1.5× against a longer seed tracklet for arithmetic
  reasons; `ratio_raw` is also the sharper statistic (AUC 0.815 / 0.857 vs
  0.785 / 0.794). See FINDINGS.md Finding 5.
* **`≥ 2.0 σ`** end-to-end in absolute terms **after dropping the single most
  helpful observation** — this is where the support test lives, so neither an
  immaculate arc nor one bad endpoint can manufacture a trend.

…and a tracklet the fit **trimmed** does not fire at all when leave-one-out
cannot run on what remains: "no support test" plus "a known bad observation" is
the one combination that decides the gate on the very point LOO exists to
discount (408038's 2009-09-21 G96: 1.49× with four kept, 7.34× trimmed to three).

**Why only one clause, and why relative** — measured in `research/ramp-gate/`
(read FINDINGS.md), on matched pairs where each candidate is judged twice through
the same pipeline: once as a real tracklet of the object, once as a *designated*
tracklet of another object from the same night and field (wrong by the MPC's own
`provid`). Crucially the seed is ONE apparition — the weak U≈7 shape an ITF
search always has, and the reason wrong tracklets fit at all.

Two results shaped the design. First, **63% of wrong tracklets never reach this
test** — retention, χ² and the seed guard already reject them — so the gate has
to be judged on the remainder (39 good, 77 wrong). Second, on those an **absolute
σ-swing discriminates no better than chance (AUC 0.476)** while the **relative
ratio does (AUC 0.815)**: the χ² gate already tests σ-normalised residual *size*,
so a second size measure adds nothing, whereas comparing against the object's own
arc asks a new question. The absolute clause was therefore deleted — it cost 2/39
good tracklets to catch 2/77 wrong.

**Calibrate in the right regime.** A first version set thresholds from full fits
of 4+-opposition objects — strong orbits, where everything fits — and claimed 1
false positive in 1000. In the weak-seed regime the gate actually runs in,
`ratio ≥ 4.0` alone fires on **8.7%** of good tracklets. All three conditions
together, in-regime: **0 of 39 good, 2 of 77 wrong**. It is a rare-event flag,
not a workhorse, and it is not a basis for refusing a linkage.

Two consumers, **both disarmed 2026-08-25**. As **clause 2c of
`_show_lone_night`** it *was* a silencer — a lone new-night tracklet that trends
was not surfaced as a hunch; it now surfaces with the trend reported beside it,
because a 0.6 hunch is a thing a human reads and a measurement that is right
about as often as it is wrong belongs in front of them, not in place of them. On
the **accepted-linkage path** it *was* a warning beside `coherent_offset_misfit`
— never a veto, but like any warning it totally withdrew a three-thin
confirmation; it is now a `diagnostic`, which gates nothing and withdraws
nothing. What both consumers do today is *report*.

**One shape is a warning again (owner rule 2026-09-30): the two-axis walk.**
`two_axis_trend` flags a linked tracklet whose residuals walk monotonically
≥ 2.0 σ in **both** RA and Dec, with no seed yardstick and no leave-one-out —
measured on the kept observations and again with the fit's rejections put back.
It lands in `warnings` (so the web submit acknowledgement shows it, and the
batch review split counts it) but is appended **past** the 1.75
withdrawal and the seed demotion, so it never changes a verdict. Found on 2013
RD21's `C14PV12` (G96, 2019-10-08): RA −1.09, −0.38, −0.10, +0.48″, Dec −1.81,
−0.17, −0.03, +0.20″, the first observation rejected by the fit. The kept three
(1.7 σ / 0.8 σ) met the relative clause's trimmed-tracklet guard and said
nothing; all four walk +3.0 σ / +3.7 σ. Measured on the decisive calibration
population (kept observations only): 1 of 58 correct, 2 of 121 wrong — rare,
not discriminating, which is exactly a review prompt's profile. Either axis
alone fires on 10% of correct tracklets and was not adopted.

*2026 OE3 is the case this was found on.* `P21l7qL` (F52, 2021-09-13, 3 obs) was
shown as a 0.6 hunch: it fits its own σ at **0.55 per dof**, keeps 3/3
observations and moves the seed rms only 0.247″ → 0.268″ — it passes every size
test. Its Dec residuals walk **−2.5 σ end to end**, monotonically, against the
**0.32 σ** worst trend in the object's own 18-observation arc: **7.7×**.
`systematic_misfit` cannot see it (it needs ≥2 nights in an apparition; this is
one) and `coherent_offset_misfit` cannot (the trend is zero-mean).

## Use

```
# web UI: the "ITF search" toolbar button (options modal: max uncertainty,
# corridor multiplier, ranking, candidate source, seed noise). A found
# linkage replaces the view with the extended fit; the UI NEVER submits.
# Endpoint: POST /api/itf_search (engine_api.itf_search_web). The browser sends
# {async: true}, so the POST returns a job handle at ONCE and the search runs on
# a worker thread: a hosted deploy's reverse proxy will not hold one request
# open for a whole search, and a proxy that gives up answers with a non-JSON
# body (Railway's "upstream error") that the client cannot even read. The client
# collects the payload from GET /api/itf_result?id=<job> (engine_api.
# itf_search_status; result kept 30 min). While a search runs, the "see trace"
# link streams the live decision feed (itf/trace.py; GET /api/itf_trace
# polling), at the "Trace detail" level chosen in the SEARCH OPTIONS modal
# (default 2; see "Reading the trace" below) -- and the viewer's own Detail
# control re-levels the running search through the same poll (&level=).

# main CLI
pyroorb "2017 CL37" --itf-search [--itf-candidate-source scan] [--itf-rank-mode legacy]
                    [--itf-max-uncert ARCSEC] [--itf-jitter] [--itf-submit]
                    [--itf-trace]     # or PYROORB_VERBOSE=1
                    [--itf-trace-level 0-4]   # verbosity, default 2; 0 = no trace

# module CLI (from anywhere); the live trace prints by default
# (--quiet or --trace-level 0 to silence, --trace-level 1-4 for verbosity)
python -m pyroorb.itf search "2017 CL37"
python -m pyroorb.itf.batch --list objects.txt             # many objects: CSV + one JSON
python -m pyroorb.itf recovery-suite --n-objects 10 --seed 42 [--live]      # recovery test suite
python -m pyroorb.itf maintain refresh|reconcile|sweep|backfill|status
pyroorb-itf search "2017 CL37"                  # the installed command, same CLI
```

## Reading the trace

A search narrates itself to a `Tracer` (`itf/trace.py`) at one of **four
levels**.  The trace is read by eye, and a wide-corridor sweep emits tens of
thousands of lines -- so the levels exist to make the story legible, not merely
shorter:

| Level | Adds | Answers |
|---|---|---|
| 1 headline | seed, one line per round, every TAKEN/recursion, the verdict, warnings (`!`) | is this worth waiting for? |
| **2 narrative** (default) | the per-round rollup: what the sweep found, the P(match) distribution + top 3, which trials the trial queue tried and how each died, the acceptance gates, why a link was or was not submittable/displayable | what is it doing? |
| 3 detail | every trial, candidate, gate and rival verbatim (what the trace always used to print) | why didn't THAT tracklet link? |
| 4 firehose | per-window corridor rows and per-chunk DB queries, rolled up into one line each below | where is the time going? |

Levels 1-3 are cumulative narration: a level-2 read is a complete story, not
level 3 with holes in it.

All three surfaces take the level, deliberately: `python -m pyroorb.itf search
--trace-level N`, `fit_object.py --itf-search --itf-trace-level N`, and the
"Trace detail" selector in the web search-options modal.  On both CLIs 0 means
no trace and a value outside 0-4 is refused; the web selector offers 1-4, since
the page always records the trace for its viewer and there is nothing to turn
off.  The selector's default comes from `GET
/api/itf_defaults`, so no UI literal can shadow the engine's -- the lesson of
the corridor multiplier, which went stale in exactly that way.  It sits with
the other LAUNCH options because that is what it is: the search reads the level
once, when it starts.  It used to live only in the trace viewer, where picking
it looked like it did nothing (the running search kept its level) and where a
reload silently reset it -- so searches ran at the default whatever was chosen.
The viewer keeps a Detail control too, but that one now genuinely re-levels the
RUNNING search: it rides out on the feed poll (`GET /api/itf_trace&level=`) and
moves the live tracer's threshold, which changes what is written next and never
rewrites what is already on screen.

Three things shape the output beyond filtering:

* **Rounds.** One `_iterate` call -- a recursion level, or the Nth LOV variant
  -- is one round, opened by a blank line and a `=== ` header so boundaries are
  scannable without parsing.  Every round ends with an `outcome`.
* **Rollups.** Per-window and per-DB-query rows collapse to a corridor range, a
  `db:` line (queries / rows / seconds), and a tracklet-category count.  The
  candidate field prints a histogram (`P>=0.5: 2, 0.1-0.5: 1, <0.1: 9459`)
  before the top few, because "was anything promising?" is not answerable from
  a top-12 list followed by "and 9453 more".
* **Say-it-once.** A 31-variant scan would otherwise restate the same parent
  orbit and likely gate 31 times; `Tracer.field_changed` emits only on change.

A warning worth knowing: an `!` line right after the seed says the seed's `U`
is above `single_night_show_max_U`, which means a lone new-night tracklet
**cannot be reported** however well it fits (`_show_lone_night` clause 1).  On
such a seed only a multi-night apparition can produce a result, and it is worth
knowing that before spending the minutes.

Programmatic: `itf_search(DbSource(connect_pub(cfg), cfg), ["2017 CL37"],
cfg=ItfConfig())` → `ItfSearchResult` (status 1/0/0.6, linkage dict, linked
astrometry, final `FittedOrbit`).

### Batch: a list of objects → one CSV + one submission document

`batch.py` works a whole candidate list and produces the two artefacts a session
of recovery work actually needs — a **summary CSV, one row per object**, and
**one MPC identification document** holding every accepted linkage as
`link_0 … link_N` under a single header, so a night's finds go out as one
submission instead of N.

```
python -m pyroorb.itf.batch --list objects.txt \
       --out-csv runs/2026-07-29.csv \        # summary (one row per object)
       --out-json runs/2026-07-29.links.json  # combined submission document
       [--timeout 1800] [--resume] [--verbose]
        [--max-uncert ARCSEC] [--uncert-mult M] [--candidate-source index|scan] [--rank-mode ml|certainty|legacy] [--jitter] [--submit]
        [--prefer complete|submittable]
python -m pyroorb.itf.batch "2016 BV14,2016 BW14" "2014 SD57"   # objects as arguments
```

The object list is one object per line (`#` comments, blanks ignored); a line may
carry **several comma-separated designations**, which are seeded as one object
just like `itf search "A" "B"`.

CSV columns: `object, verdict, status, link_key, linked_tracklets, added_obs,
opps_pre, opps_post, nights_post, rms_pre, rms_post, seed_rms_pre, seed_rms_post, U_diff,
worst_mag_resid, seed_shed_nights, n_warnings, warnings, trksubs, seconds,
reason`.  `verdict` is `linked` (submittable), `possible` (the search's
unconfirmed thin-profile find — worth a human look, never submittable),
`none`, `timeout` or `error`.  `link_key` names the link a row became in the
combined JSON.  Read the `warnings` column before submitting: those are the
same cautions the web UI makes a submitter acknowledge.  `seed_rms_pre` /
`seed_rms_post` are the object's OWN observations before and after the link
(the pair the contaminant gates judge); `rms_pre` / `rms_post` are the OVERALL
fit.  **All four are per-axis** — see "RMS currency" below — so all four are
comparable to each other and to find_orb.

Operational properties, all deliberate:

* **one subprocess per object, with a timeout** — a search can wedge on a
  pathological orbit and a live run can outgrow the box's memory (an OOM kill is
  silent); neither takes the batch down, the row records `timeout`/`error`, and
  the memory comes back between objects;
* **both artefacts are rewritten after every object**, and each accepted linkage
  is also written under `<out-json>.parts/` — which is what lets `--resume`
  (skip what the CSV already records, retry its timeouts and errors) still emit a
  *whole* document rather than one holding only the last leg's finds;
* **it is dry-run by default** — pass `--submit` to transmit the final combined
  document for `linked` rows.  Submission still goes through the MPC
  deduplication, two-id backstop, and ledger path.  `possible` rows are never
  included.  Exit status reports the run, not the harvest: finding nothing is a
  good answer and exits 0.

## Linkages the user assembles (no ITF search involved)

An ITF find is not the only submittable identification.  Fit **two or more
designations together** — `fit_object.py "1999 TV43" OOn0024 --linkage`, or the
same pair typed into the web search box — and the joint orbit *is* a proposed
identification, including the NEOCP case (a temporary NEOCP id files under
`trksubs` and stamps `identification_type: "neocp"`, exactly as an ITF trksub
does).  Nothing about `build_linkage` was ever ITF-specific; what was missing
was anyone asking it for this document.

The gate is the engine's own ident rule, `ident_check.linkage_verdict` — the
verdict `--ident-check` prints, thresholds and all, so the UI cannot offer to
submit what the batch files as a non-match:

* `identify()` accepts → **clean identification**, offered with no caution;
* otherwise `examine_or_no()` = `y-examine` → an ambiguous-but-real candidate,
  offered **with its reason as a caution** the submitter must acknowledge;
* `no` (one side surviving only by clipping most of its observations, two
  baseline-less short arcs, …) → **not offered at all**.  The orbit may be fine;
  the identification is what the engine declines to back.

Advisory flags ride along as cautions in both cases (photometric ΔH between
apparitions, a systematic per-night residual trend) — they never veto, they make
a human look twice.

```
fit_object.py "1999 TV43" OOn0024 --linkage           # dry-run: verdict + the JSON
fit_object.py "1999 TV43" OOn0024 --linkage-submit    # ...and transmit it
```

In the web UI the same linkage raises the same **Submit linkage** button as an
ITF find (one block, `kind: "ident"`, one button), with a banner saying what the
engine concluded.  It follows the screen — switch a designation's observations
off and it leaves the document — is refused on an orbit that has not been
re-solved since the last toggle, requires every caution acknowledged
server-side, and goes out through the same vetted path as an ITF submission
(`submit_linkage`: DES=DES dedup against `current_identifications`, then the
ledger write).  Since 2026-10-11 the web button sends it to the identification
**queue** (`IDQUEUE_URL`) by default, which dedups, checks, paces and
forwards it; its ▾ menu offers the same document as a file (`ident_*.json`) and,
only where `DIRECT_SUBMIT_ALLOWED` is set, a direct send to the MPC through the path
above.  The header is always the person's own remembered name / e-mail.

### Which designations the document names

One id per identification, and only the ones that are news
(`linkage.requested_primaries` → `build_linkage(primary_desigs=…)`):

* every designation the **user requested** — one name typed in the search box or
  at the CLI is one identification, `K13M06C = K21C05V` is two, and both belong
  in `designations` (a search seeded from that pair used to submit the first
  name alone, silently dropping the identification the user had already made);
* plus any **designated pickup** the search newly linked (a pseudo-ITF tracklet
  filed under its own designation, from `itf_obs_out`);
* but **not** the aliases that merely rode along with a requested name — the MPC
  has already acknowledged those, and restating an *already-accepted* DES=DES is
  what it rejects (a DES=DES it has **not** accepted is precisely what an
  identification submission is for).
  A one-designation fetch is therefore one family primary however many
  designations came back, and `submit_linkage` re-checks the whole array against
  `current_identifications` on the way out.

The **shape rule** the document must satisfy is only this: a link names at least
**two** ids, in any mix — DES+DES, DES+ITF, DES+DES+ITF, ITF+ITF. One designation
alone, or one tracklet alone, claims no identification. `build_linkage` refuses to
construct such a document; `submit_linkage` re-checks it as a **backstop**
(`short_link`) because the dedup above runs *after* that check and can drop an id —
two designations that turn out to be one accepted family collapse to one. A
collapsed linkage is refused with a reason ("nothing new to submit"), never sent —
and the web **download** route (`itf_download_web`, the offline alternative to the
submit button) applies the same backstop after its own copy of the dedup, since a
file the user posts by hand has nothing downstream to catch it.

A **pickup is judged by the column it sits in**, not by the shape of its
characters (`linkage.line_designation`).  OBS80 columns 6-12 hold the packed
*provisional* or a trksub; a permanent number lives only in columns 1-5.  Some
SBN trksubs are all digits — `133651` is a real one, sibling to `10b51e` and
`1207c2` in the same linkage — and a string-only test round-trips them through
`unpack` as a numbered minor planet: the document then asserted an
identification with an object we never fitted, and lost that tracklet's nights
from `trksubs`.  So columns 6-12 count as a designation only for a packed
provisional (`K23R67C`, the `_`-extended form) or a survey desig (`PLS2040`);
everything else there is a trksub.  A line carrying *both* a number in 1-5 and a
trksub in 6-12 is a contradiction the MPC does not emit, and raises rather than
guessing.  (ADES/SBN have no such ambiguity: `permID`, `provID` and `trkSub` are
three separate columns.)

### The orbit block, and the one document assembler

Every link carries the MPC's optional `orbit` block (2026-10-05) -- the seven
fields the [submission format](https://minorplanetcenter.net/mpcops/documentation/identifications/submission-format/)
makes compulsory once an orbit is sent: `arg_pericenter`, `eccentricity`,
`epoch`, `inclination`, `lon_asc_node`, `pericenter_distance`,
`pericenter_time`.  Heliocentric J2000 **ecliptic** osculating elements of the
fit the linkage was judged on, angles in degrees in [0, 360), `epoch` and
`pericenter_time` as JD (TDB = TT to 2 ms) -- the layout find_orb's
`make_linkage_json` writes.  `linkage.orbit_section(fit)` builds it;
`build_linkage(..., fit=)` attaches it, and every caller passes its fit (the
search's `trial.fit`, the web's live session fit, the CLI `--linkage` joint
fit, the NEOCP monitor's joint / ITF fit).  It is **left out, never
approximated**, when we would not publish that orbit anywhere else: the arc
does not determine one (`constraint.assess`, the verdict the exports obey), the
user's "show an orbit anyway" opt-in is on (its orbit may be the
population-prior choice, display-only), or a field is not finite.
`short_link` refuses at send time a document whose orbit lost a field.

There is ONE link builder (`build_linkage`) and ONE document wrapper
(`linkage.submission_document(header, links)` + `submission_header(cfg)`):
`itf batch`'s merge (`batch.combine_links`, hence the megabatch and review
packets) and bundled submissions (per object, bundled, and the offered-pile record)
re-key the same link bodies through it, so a link's orbit travels with it.

## The tracklet index (we own it)

`itf.itf_tracklets` on the SBN server: one row per pure-ITF/pseudo-ITF
tracklet (weighted-LSQ motion + per-axis sigmas, healpix, photometry).
`index.py` carries the schema, feature math and the three idempotent
maintenance jobs — `refresh_incremental` (~60 s), `reconcile_membership`
(daily), `sweep_validity` (weekly) — all audited, watermarked and resumable;
`python -m pyroorb.itf.maintain <job>` invokes any one.  `scheduler.py` runs all three
on one clock, single-flight (sequential + a Postgres session advisory lock, so a
long reconcile/sweep just defers the next refresh), gated by the
`ITF_MAINTAIN_TURNED_ON` env var (**off** by default) with per-job last-run in
`itf.itf_maintain_runs`; on Railway it runs as a daemon thread in the web
service (`python -m pyroorb.itf.maintain scheduler` / `tick` run it standalone).  Check
health with `python -m pyroorb.itf.maintain events`.  **Setting up and maintaining
the index on your own SBN database: [`docs/ITF_SETUP.md`](../../../docs/ITF_SETUP.md).**  Server
etiquette: never seq-scan `obs_sbn` (~538 M rows), batch writes, throttle bulk
reads.

## Recovery test harness

`harness.py` seeds a multi-opposition object with ONE apparition and plants
the other apparitions as **fake ITF tracklets** (re-labeled obs80 under
synthetic trksubs, feature rows computed by the same `compute_tracklet_features`
as the real table). An `OverlaySource` serves them beside the real ITF (or
alone, offline), admitting each fake only through the same db-side predicates
a real row would face. Success = the search re-links the planted apparitions.
`python -m pyroorb.itf recovery-suite` runs it over random 5-opposition MPCORB objects;
offline unit tests live in `tests/test_itf_search.py`.

**Which apparition seeds** (`pick_seed_opposition`): the object's **discovery
apparition** — the one its designation names (`2024 NH2` → the apparition
holding 2024 observations) — when it carries at least `SEED_MIN_NIGHTS` (3)
observing nights; otherwise the most COMPACT apparition that does. A discovery
arc is the only seed shape that makes the test honest. Seeding the *fattest*
apparition (the original default) hands the search a well-observed arc no real
recovery ever gets, and seeding the chronologically first can land on a lone
precovery night that determines no orbit at all. Most-compact-alone was the
second wrong answer and is worth keeping named: it minimises SPAN, and span says
nothing about whether an arc carries information — on **2024 NH2** it preferred
a 7-obs/7-night/36 d precovery apparition (1.0 obs per night) over the real
28-obs/10-night/57 d 2024 discovery arc, whose first 3 nights are then three
*single-observation* nights, six measurements against six elements, exactly
determined at U 15.25. Such a seed cannot accept any extension at all
(`extension_degrades`' floor is below real astrometric noise), so the object was
unrecoverable before the search started: 0/12. On the discovery arc it recovers
7/9. A designation with no parseable year (a number, a trksub, a survey
designation) falls back to most-compact. `--seed-nights N` cuts the seed down to that
opposition's first N nights and plants the rest, so `--seed-nights 3` is the
discovery-arc stress shape: a ~2-day, 3-night seed, everything else to be found.

Nights here are **real observing nights** (a >0.5 d gap opens one,
`observing_nights`), deliberately NOT `opp_profile.spans_of`'s 36-hour SPAN (1.5 d
since gaps #63; 1.6 in the notebook). That unit is the ported convention the
acceptance gates are calibrated on and it reads observations 1.0 d apart as
one span on purpose — sound for gating,
wrong for building a seed, where it collapses a 3-consecutive-night discovery
arc into a single "night" and makes a 2-day arc look unfittable. Two rules,
each named for what it does.

A frozen fixture **pins its `seed_apparition` explicitly**
(`tests/test_itf_recovery.py`); inheriting the default would silently re-seed
every fixture the day the default changes, with no diff to show for it.

**The SQL pushdown is differentially tested** (`tests/itf_sql_fuzz.py`).
`_sql_corridor` / `_sql_motion_bound` re-implement `tracklet_in_window` /
`motion_bound_ok` server-side, and the arrangement is only sound while the SQL
is a **superset** — looser is free (`_survivors` re-applies the exact Python
test), stricter means candidates vanish with no error and the search reports a
clean negative. The fuzz generates rows sitting exactly ON each predicate's
accept boundary (where the 1e-6 pads earn their keep) and asserts the SQL never
drops what Python keeps: offline by transliterating the SQL into Python (runs in
`test_itf_search.py`, every commit), and live by handing the same rows and the
REAL SQL to the server via `unnest(...)` — read-only, no table touched:

```bash
python tests/itf_sql_fuzz.py 4000     # from the repo root; live check skipped w/o DB
```

### The harness is also the ML training-data factory

The candidate ranker (`training.py`, `ranking_mode="ml"`) learns from the same
rig: every candidate served into a corridor is one labeled row — **positive**
when it is the object's own planted (MPC-linked) truth, **negative** when it is
a real ITF tracklet that was served but does not belong. Features are frozen
at DECISION time (that recursion level's trial orbit), never the final one:
the model must see what the search saw.

```bash
python -m pyroorb.itf.collect --objects-per-split 250 --out train.csv     # collect (live DB)
python -m pyroorb.itf train-audit train.csv                  # what came out, + cap stats
python -m pyroorb.itf train-model train.csv --model v3.json  # xgboost
python -m pyroorb.itf train-compare train.csv \
    --model v2=training_data/itf_match_v2.json --model v3=v3.json
```

Four rules the collection holds to, each of which cost something to learn:

- **Train/serve parity.** Collection runs the search at its real defaults, and
  never with a different candidate population than production serves. The one
  deliberate exception is the motion pre-filter, which is bypassed so the
  negative class covers the whole corridor — its would-be verdict rides along
  in `motion_bound_ok` so a retrain can choose either population.
- **The train/eval wall is a hash of the designation** (`split_of`), permanent
  and per-object. Not the trailing digit: for NEOs that digit is the
  provisional cycle count, which encodes how late in a crowded half-month the
  object was designated, and splitting on it hands eval a systematically
  different NEO population (median cycle 18 vs 5).
- **Stratified sampling with matched mix on both sides** (`sampling.py`), plus
  an arc-length floor — an object whose whole arc becomes the seed has nothing
  left to plant and yields nothing.
- **A deadline costs the verdict, not the data.** Rows harvested before it
  fires are kept and labeled (`truncated` flags them); each object runs in its
  own process with a RAM precheck and a per-object ledger, because the v2 run
  was OOM-killed at object 47 and reported nothing.

## Setup

* Server and credentials in the gitignored `.env` settings file (`pyroorb/envfile.py`;
  or exported):
  `SBN_DB_HOST=...` (your SBN database server -- there is no built-in host; unset,
  every connection fails with a message naming it), `SBN_PUB_PASSWORD=...`
  (read-only search) and `ITF_DB_PASSWORD=...` (index owner).  See `dbio.py`.
  Optional (same `.env` / exported-var rules): `ITF_SUBMIT_NAME` /
  `ITF_SUBMIT_EMAIL` (the submitter identity the MPC receives from the CLI tools --
  there is NO built-in default since 2026-10-11, and `submit_linkage` refuses an
  anonymous document; an automated submitter carries its own).
  `DIRECT_SUBMIT_ALLOWED` is the server-side authorization for sending straight to
  the MPC from the web, skipping the queue (off by default; it gates the endpoint,
  not just the menu item),
  and web submission also requires the client to acknowledge every quality
  caution server-side.
* Extra deps beyond the engine: `psycopg2-binary==2.9.9` (2.9.11's wheel is
  broken on this box) and `healpy` — only needed for DB paths / index math.
* Needs direct TCP egress to the SBN PostgreSQL server; offline, the overlay
  harness and all unit tests still run.

Design/provenance: ported from the proven `ift outline/itflib` pipeline
(gitignored; the v23 notebook logic validated by real MPC submissions), with
find_orb's `make_linkage_json` (elem_out.cpp) consulted for the submission
document's edge rules.


## RMS currency (per-axis, everywhere a person reads)

An arcsec rms of astrometric residuals can be averaged two ways, √2 apart:

| | formula | estimates |
|---|---|---|
| **per-axis** | √( Σ(Δα\*² + Δδ²) / **2n** ) | σ — one coordinate's error |
| **per-observation** | √( Σ(Δα\*² + Δδ²) / **n** ) | σ√2 — a 2-D position offset |

**Everything reported is per-axis.**  That is find_orb's convention
(`compute_rms`, `orb_func.cpp`, which divides by `n_included * 2` — revised to
on 2010 Nov 4 "so that RMS residuals are computed as root-mean-square of RA and
dec treated separately … (Gareth Williams kindly steered me the right way on
this)"), and it is what the MPC, JPL and OrbFit quote.  It is also the only
currency comparable to a station's reported σ, to ADES `rmsRA`/`rmsDec`, and to
χ²/dof, since each observation carries two degrees of freedom.  Every engine
producer already is per-axis: `FittedOrbit.rms_residual()` / `rms_normalized()`,
`TrialFit.rms`, `arc_extend._Evaluator.rms_arcsec` / `rms_norm`.

Per-observation exists in **one** place — `itf.fitter.subset_rms` and its two
companions — because the seed-degradation gates compare a fit's rms on one
SUBSET of observations before and after a link, and both sides must be measured
the same way (mixing them cost a 1.414× handicap until 2026-07-29; see
`search._seed_pre_rms`).  Those gates are ratios, so the currency cancels.

The rule: **per-observation never reaches a report.**  Convert with
`itf.fitter.per_axis` at the display boundary, with a comment.  Three
exceptions, documented at their sites: `itf.gatestats` rows (an append-only
study feed), `itf.index.fit_rms_arcsec` (a written DB column, ML feature and
SQL prefilter term), and a single observation's |O−C| or a night's mean
displacement — 2-D offsets, not averages over components, so no conversion
applies.  Locked by `tests/test_itf_search.py::test_seed_rms_is_reported_per_axis`.
