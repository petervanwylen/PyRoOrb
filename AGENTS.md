# AGENTS.md — rules of the PyRoOrb engine

Guidance for anyone (person or coding agent) changing PyRoOrb. Each rule is one line naming
the symbol to read; the code beside it states the measured reason. Change a rule's behaviour
only with a measurement, and update its line here when you do.

**find_orb is the reference.** PyRoOrb ports Bill Gray's find_orb (GPL-2.0): logic openly
ported, no source lines copied. When a behaviour is unclear, read the find_orb code rather
than guess, and credit what was read or ported in the module's docstring (file + function).

## Layout (`src/pyroorb/`)

- `fit_object.py` — orchestration, combined objects, `FittedOrbit`; `cli.py` — the command line.
- `iod.py` — start-arc selection + Gauss/Väisälä/Herget IOD.
- `arc_extend.py` — extension, outlier rejection, cluster activation, single-night dropping.
- `forward_model.py` / `force_assist.py` — rebound 8-planet and ASSIST (DE440+GR) propagators + batch sweeps.
- `weights.py` / `debias.py` / `photometry.py` / `hazard.py` — σ, EFCC18 debias, H/band, MOID/class/U.
  `healpix.py` — numpy HEALPix (pixel lookup + inclusive NESTED disc), byte-identical to healpy.
- `hyperbolic.py`, `boundbridge.py`, `nightmodel.py` — the unbound-orbit gate.
- `constraint.py`, `rangescan.py`, `popprior.py`, `d2score.py` + `digest2/` — determinacy, range profile, display-only population prior, digest2.
- `radar.py`, `impact.py` / `bplane.py` / `keyholes.py`, `egress.py` — radar, impact products, export formats.
- `obs_sources.py`, `ades.py`, `obs80.py`, `sbn_obs.py` — observation fetch, parsing, the fetch cache.
- `datadirs.py` — where files live (per-user cache, reference tables); `localstore.py` —
  `LOCAL_PERSISTENT_FOLDER`, the one setting for restart-surviving storage.
- `mpcorb.py` — cached MPCORB + `sample_asteroids(n, …)` filtered random sampler.
- `itf/` — ITF search (seed fit → corridor → candidates from `itf.itf_tracklets` → ML-ranked
  trial linkages → recursion → gates → MPC identification JSON, dry-run by default). Needs your
  own SBN database. `pyroorb-itf` / `python -m pyroorb.itf`; see `itf/README.md`.
- `web/` — server, fit pool, ITF jobs, static UI, add-ons (`web/README.md`).
- `__init__.py` — the supported Python API (`read_observations`, `fetch_observations`,
  `fit_orbit`, `ephemeris`, `export`); everything else is internal.

## Testing

From the repo root (`./install.sh --extras itf,dev`, or `uv pip install -e ".[itf,dev]"`): `pytest -m "not slow"` (the fast loop),
`pytest -n 4` (everything offline: the root `conftest.py` blocks the network and uses an empty
cache; the goldens need the asteroid kernel, `pyroorb-fetch-data --asteroids`), `--live` (network / SBN database), `--benchmark`. `tests/test_regression.py` pins each
fixture's fit; re-baseline with `tests/update_goldens.py --write`, add objects with
`tests/build_fixtures.py`. Comments and docstrings state the rule and its measured reason
(`tests/test_public_code_hygiene.py`).

## ITF rules in one line each

- **Submission needs two REQUESTED ids** (`n_requested`, `linkage.requested_primaries`);
  MPC refuses a lone id and an already-accepted DES=DES. Enforced at build
  (`build_linkage`) AND send (`linkage.short_link`).
- **One link builder, one document wrapper** (`build_linkage(fit=)`,
  `linkage.submission_document`); the `orbit` block rides in every link unless
  `constraint.assess` says undetermined. **No unsigned document exists**
  (`linkage.header_problem`, `AnonymousDocument`).
- **Nothing submits a 0.6 (`possible`).** Rules that demote to 0.6: seed obs fit
  worse under the link (`warn_rms_factor`); a 1.75→3 confirmation that raised any
  warning (a seed degradation the seed's own extended opposition already carries
  excepted, `search._seed_opp_holds`); a partial pickup whose leftover tracklets misfit (`itf/pickup.py`, judged by
  `itf/mpcrules.py`, not our own gates); a comet/interstellar id (`linkage.comet_refusal`,
  column-aware); the MPC night rule (`mpcrules.submittable_nights`: 12 h apart = 2 nights).
- **A possible stays a possible through a web step** (`engine_api._pending_verdict`):
  only a GROWN obs set passing the search's own gate (`search.opp_profile_acceptable`)
  upgrades it; sole exception the opp-profile-only refusal after a re-solve
  (`search._thin_possible`, `promotable_on_step`).
- **After a Yarkovsky latch, trial fits hold the force fixed** (`fit_under_known_nongrav`,
  `SearchContext.nongrav_ref`): warm from the rescued orbit, no rediscovery; the rescue needs 2+
  apparitions (`_yarkovsky_retry`).
- **Two answers, consumer decides** (`ItfConfig.prefer`, `search._fuller_possible`): a LINKED
  linkage and a reviewable 0.6 linking strictly more both come back (`result.alternative`);
  inside the search the linkage always leads (`_pair`), `prefer` applies once at the end (`_choose`);
  `complete` default (CLIs, harness), `submittable` for every submitter (`--submit` and any automated sender); the web asks only when there are two (`itfchoice.js`).
- **One verdict for every linkage** (`search._judge`: `_assess` -> `_extension_check`);
  alternatives (`_repaired`, `_reduced_linkage`, the child) judged the same way. All seed
  tests live in `SeedCheck` (rule V; floor read as a measurement; one offending seed
  night dropped and priced, `_drop_seed_night`). Vouching (`_vouch`) accepts a tracklet
  the orbit does not move toward. What a repair earns rides on the `Assessment`.
- **Residual trends are diagnostics, not gates** (`itf/residual_trend.py`; withdrawn as a gate); they ARE review warnings in batch CSVs; the two-axis walk
  (`two_axis_trend`) is a search warning. Leave-out confirmation can demote a caution
  (`itf/leaveout.py`, never on 1.75→3).
- **Ranker = v4 only**; harness plants must not leak labels
  (`tests/test_itf_harness_leaks.py`). Continuous LOV refine (`itf/lov_refine.py`) runs
  ONLY after the whole fan failed — acting on it mid-scan causes wrong links; there its
  'unlikely' last-resort picks stay in the field and get trialled.
- **Fits ADES, administers OBS80** (`itf/obsreg.py` `AdesRegistry`; full band matters).
  The document names a tracklet by the DB `trksub` column, not cols 6-12 (`linkage.db_trksubs`).
- **Memory**: `sweep_min_free_mb` 1000 scaled by `memory.sweep_floor_mb`; concurrent
  searches derived in `web/capacity.py` (`PYROORB_ITF_MAX_RUNNING`); slots released by
  worker liveness (`_itf_reap`). Capacity messages never join `warnings`.
- **Corridor cap `max_uncert_deg` 40 deg** (measured; was an arbitrary 19.44); the carry pool ranks a re-swept tracklet by its fresh score (`_carried_pool`).
- **Flat field** (`search.flat_field`: best likely P < 0.15, linear corridor): 30 singles and barren-take backtracking at ANY depth, not only depth 0. A take reached by resuming after a barren one never restarts the no-progress clock (`TrialQueue.resumed`).
- **Terminal LOV fan** (`ItfConfig.terminal_fan`): a linear recursion ending on an orbit still at U >= `variant_corridor_U` scans the LOV fan from it, once per search, only from a take the vet calls reviewable (`_terminal_fan_due`), never inside a variant round (`SearchContext.terminal_fan_done`). A link the take's vet accepts is transmitted only if the fan does not supersede it (`_release_held`).
- Faint cut `ItfConfig.mag_limit` 23.5 (`--mag-limit`/`--thorough`); UI fetches defaults
  from `/api/itf_defaults`, never hard-codes them. The DEFAULT is deepened per object to
  1 mag past its brightest predicted V since 1995 (`ephem.faint_cut`); a typed cut
  (`set_mag_limit`, any surface, even 23.5) is used exactly; the web field stays empty.

## Engine invariants (one line each)
- **RMS is PER-AXIS everywhere** (find_orb/MPC/JPL convention). Per-obs only inside
  `itf.fitter.subset_rms`; convert with `itf.fitter.per_axis` at any display boundary.
- **Covariance `s2` floored at 1** (`FittedOrbit._param_inverse`): a fit's residuals may
  scale σ up, never down (chi2/dof ≈ 0.15–0.2 by construction).
  High-U gates were shifted for it; study-backed gates re-derived, not shifted.
- **NEOCP object = one object** (`ades.fetch_combined_neocp`): trkSub-only rows relabelled
  to the requested id; submitter trkSub kept on `o["trksub"]`.
- **Obs fetch falls back to the SBN replica** (`sbn_obs.py`) when the MPC API returns
  nothing: family = `permid` ∪ `provid` aliases, status `P`/`p`; `PYROORB_SBN_FALLBACK=0` off.
- **The fetch cache is burst protection, not an archive** (`obs80.astrometry_ttl`: 3 h for a
  published designation, 2 min for a NEOCP/temporary id; `datadirs.trim_cache` deletes expired
  entries and holds the cache under `PYROORB_CACHE_MAX_MB`, default 2000; reference data never trimmed).
- **One copy per detection** (`obs_sources.dedup_observations`, predicate
  `orbit_common.same_detection`): station+instant+sky; published designation wins.
- **A perturber is never fit inside itself** (`force_assist.perturbers_off_for`,
  ContextVar; `GravityOnlyForce` pinned on the fit's ctx).
- **Published-orbit seed** for >500 obs / >5 opps (`mpc_orbit.py`,
  `fit_object._fit_from_mpc_seed`, fetched in CLI/web only). Comet fallbacks:
  `_catastrophic_seeded_alternative`, `_comet_extend_apparitions` (prefer recent apparitions).
- **Unbound orbits are earned, never special-cased by designation** (`hyperbolic.py`):
  bound first (`_prefer_bound_ecc`), refused only at 3σ under the measured station-night
  error model (`nightmodel.py`) plus the night bar or BIC
  refusal; long-gap arcs searched by Lambert bridges (`boundbridge.py`); unsearched cases
  labelled, never silently earned.
- **IOD long-window rescue** (`iod.rescue_window`): re-seed a catastrophic gap-free window
  > 230 d before extension; any length on a single apparition.
- **No nominal orbit when undetermined** (`constraint.assess`; single tracklets decided by
  the range profile `rangescan.py`, never by U or tracklet count): elements blanked,
  intervals shown, exports refused except ephemeris. "Show anyway" uses the DISPLAY-ONLY
  population prior (`popprior.py`) — never feed it to hazard/ITF/NEOCP/classification.
- **Asteroid nongrav discovery gates on q < 1.4 au**, not a (`NONGRAV_Q_MAX_AU`; drift is earned at perihelion).
- **Nongrav discovery by BIC** (`_ng_choice`, `NONGRAV_BIC_MARGIN` 15; comet ladder
  `_comet_step_justified`); negative A1 refused; exact 0.0 = stall; marginal models
  jackknife-verified (`nongrav_jackknife_detail`, by tracklets, never warm-started).
- **Radar** joins the final correction (`radar.py`; joint state+A+radar for nongrav).
- **Egress**: `--out-format mpcorb,sof,elem,covar` / web Export (`egress.py`; `covar` is
  find_orb's own `covar.json` layout; state+covariance JSON web-only).
- Opt-in reports never touch the default fit: `--mag-audit`, `--impact`, `--impact-ip`,
  `--keyholes` (IP/keyholes CLI-only; they have not passed validation against Sentry).

## Web / ops invariants
- **Optional web features are add-ons** (`web/addons.py`): routes, pages, menu links and
  start hooks registered at start-up; the server names none of them.
- **Fits go through the bounded pool** (`web/fitqueue.py`): single-flight by content,
  `restore` tokens (never a shared session), 180 s cooperative deadline
  (`deadline.py`, web-only, `DeadlineExceeded` is a `BaseException`). Every heavy endpoint
  goes through `server._pooled`.
- **A user can stop their ITF search** (`POST /api/itf_cancel`, `static/itfcancel.js`):
  cooperative via `deadline.cancellable`; `deadline.Cancelled` must never subclass
  `DeadlineExceeded` (the trial-fit handler would eat it); slot freed only once unwound.
- **IAS15 step floor is ONE constant** (`forward_model.IAS15_MIN_DT`); in-flight sweeps
  visible via `sweepwatch.py` → `/api/memory`, `/healthz`.
- One Earth-orientation table (`eop.pin()`, no runtime download); one ephemeris
  (`force_assist.get_ephem`, locked).
- **One settings file** (`envfile.load_env_file`): every command and `python -m` entry loads
  `.env` BEFORE importing the engine (folders are read at import); a real variable wins; a
  library import reads none (`tests/test_envfile.py`).
- `/memory` (`web/webmem.py`): collects before counting, password-gated deep pass;
  `trim_allocator` returns glibc-held pages (verify with RSS).
