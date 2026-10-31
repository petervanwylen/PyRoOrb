# Testing framework

The regression suite re-fits a static, **offline** archive of real objects and asserts
each still links as well (and as fast) as its recorded baseline. It guards both fitting
quality and speed. This file is the single source of truth for how the test scripts fit
together and what counts as a regression.

## Running the suite

The runner is **pytest**, from the **repo root** (pyproject.toml's `pythonpath` puts `src/`
and `tests/` on the path; the repo-root `conftest.py` blocks the network and gives the run an
empty cache, so the default run is offline -- `--live` lifts both):

```bash
pytest -n 4                  # everything offline, in parallel (pytest-xdist)
pytest                       # the same serially — ~11 min (this is the timing-guarded run)
pytest -m "not slow"         # the fast unit tests only — ~3.6 s; the edit loop
pytest --durations=10        # where the time actually goes
pytest --live                # ALSO run the DB/network-backed tests (opt-in)
pytest --benchmark           # ALSO run the heavy speed-only regression fixtures
pytest tests/test_nongrav.py -k yarkovsky -x    # ordinary pytest selection
```

Install it with the declared dev extra: `uv pip install -e ".[dev]"` (pytest + xdist).

Two markers, declared in `pyproject.toml` under `--strict-markers` (so a typo fails
rather than silently selecting nothing):

- **`slow`** — does a full engine fit or a multi-object sweep. `-m "not slow"` is the
  edit/run loop; the full run is what you owe a commit. **Mark new fit-driven modules**
  — an unmarked one lands in the fast loop and quietly turns 3.6 s into minutes, which
  is exactly what the three heavy `tests/web/` modules did when they were first
  brought under pytest.
- **`live`** — needs the network or the SBN database. **Deselected unless `--live`.**
  These are minutes-long and, when the network is unreachable, they used to hang a
  plain run for a quarter of an hour before quietly testing something else.

`testpaths` covers **both** Python test trees — `tests/` (engine) and
`tests/web/` (the web server / engine_api layer). The **browser-side** tests in
`tests/web/*.js` are jsdom scripts outside pytest; run them with
`node tests/web/<file>.js` (they need a local `npm i jsdom`; `package.json` is
gitignored dev convenience). All thirteen pass in ~40 s.

`norecursedirs` in `pyproject.toml` is load-bearing, not tidiness: this repo keeps its
git worktrees *inside* the tree (`.worktrees/`, `.claude/worktrees/`), each a full copy
with its own `tests/` and ~200 fixtures, plus the gitignored `find_orb/` clone. Without
it, collection walks all of them and never finishes.

### Parallelism and the speed guard

`test_regression.py` makes **one test item per fixture** (not one loop inside one test),
which is what lets xdist spread ~180 independent fits across cores.

The timing baselines are **CPU seconds** (`time.process_time`), not wall clock. Wall
clock made the guard a function of whatever else the box was doing — it drifted ~9% run
to run on an idle machine — and it made the suite unparallelisable, since N fits sharing
M cores each "take longer" with nothing regressed. The engine is single-threaded
(measured CPU/wall = 1.00), so CPU time is the quantity it actually controls.

CPU time is *not* fully contention-free, though: measured on a 10-core box, **4 xdist
workers left CPU time flat, 6 inflated it ~40%** — past the 30% tolerance, failing
objects that had not regressed. So under xdist the **correctness** guards run on every
fixture and the **timing** guard steps aside. Measure speed with a serial run (or force
it with `--speed`), on a quiet machine, as this file has always advised. `-n 4` is the
sweet spot here; `-n 6` saves ~10 s and runs the box down to ~125 MB free.

## The three scripts (and who writes fixtures)

| script | reads | writes fixtures? | when to run |
|---|---|---|---|
| **`test_regression.py`** | `fixtures/*.json` | **never** (strictly read-only) | before every commit; freely |
| **`update_goldens.py`** | `fixtures/*.json` (stored obs) | only with `--write` | to re-baseline goldens after an *intentional, verified* engine change |
| **`build_fixtures.py`** | the MPC network + `OBJECTS` | yes (per object) | to ADD an object or refresh an object's raw astrometry snapshot |

Alongside them, **`pair_shapes.py`** is a read-only *diagnostic* (writes nothing): it dumps every
`PAIR_` fixture's night-level shape — retention, kept/full arc, per-night keep/DROP/residuals — plus
`identify()` / `examine_or_no()`. Run it before and after any change to the identification rule and
diff the verdict lines: that is how you prove a tuning aimed at two pairs left the other ninety alone.

Shared golden/atomic-write/slim-obs helpers live in **`_fixtures.py`** (one definition of
`link_status`, `compute_golden`, `nongrav_golden`, `write_fixture`, …), imported by all three.

**`test_fixture_flags.py`** guards the seam between the two: `build_fixtures.OBJECTS` is the
DECLARATION and `fixtures/*.json` is the ARTEFACT, and every flag is read at test time off the
artefact. They drifted once — `quality` was declared for 2008 NA and silently dropped by a later
rebuild, leaving the one protected benchmark running speed-only (gaps #43) — so the test asserts
each fixture carries the flags its entry declares, and `build_fixtures` now copies the whole opts
dict rather than a hand-picked pair.

Key consequence: **running the test never changes a fixture.** Goldens drift only when a
human deliberately runs `update_goldens.py --write`. Treat that as a commit-worthy act.

## Each fixture

`fixtures/<label>.json` holds the fully-parsed, debiased obs list (exactly what
`fit_orbit` consumes — no network needed to re-fit) plus the `golden` baseline:

```
golden = {used, total, rms, n_excluded, status, secs, nongrav, nongrav_signs}
status = "ok" if (rms < 0.6" and used >= max(6, 0.55*total)) else "suspect"   # link_status()
```

`nongrav` is the delivered force model (`None`, `"yarkovsky"`, `"srp2"`, …) and `nongrav_signs`
maps each fitted coefficient to `-1 / 0 / +1`. They are there because **nothing else in a golden
can see a selection flip**: when gaps #46 replaced the effect-size floors with a BIC bar, 2003 HM
went `None → yarkovsky` at `130 obs / 0.7355"` → `134 / 0.7466"` — inside every tolerance below, so
the suite could not have caught it. The model name is the guard; the *sign* rather than the value,
because a coefficient's magnitude moves with convergence, mask and epoch (pinning it would be
brittle for no gain) while its sign is the physical claim. An exact `0.0` records as sign `0`: this
project reads it as a **stall**, not a finding, so it must not compare equal to either sign.

Goldens written before this existed carry no `nongrav` key, and an absent key **skips** the check —
that is "not yet baselined", not "passing". Arm them with
`python tests/update_goldens.py --nongrav --write`, which records only the force-model keys and
leaves `used` / `rms` / `status` / `secs` untouched. That separation matters: a full `--write`
re-baseline also re-records every **timing** baseline, which has to be measured serially on a quiet
box, a far stronger precondition than "tell me which model each object delivers".

`secs` is a **CPU-second** timing baseline (`time.process_time`, machine-dependent,
refreshed on big speedups). It used to be wall clock. The engine is single-threaded, so
the two agree closely — measured CPU/wall = 1.00 per fit, and the whole suite stayed
green serially against baselines recorded the old way — which is why the existing
values were **not** bulk-rewritten at the switch. They become CPU-native as objects are
re-baselined normally. Don't read a pre-switch `secs` as a precise CPU figure; do trust
it as a tolerance-level one.

**Per-fixture flags** (in the JSON; set via the `OBJECTS` entry's opts dict in
`build_fixtures.py`, e.g. `("13923", ["13923"], {"benchmark": True})`):
- **`force`** — only links under `--force`; the test/updater fit it with `force=True`.
- **`quality`** — opts a `benchmark` fixture back into the fit-quality guards. Only meaningful
  alongside `benchmark`.
- **`mpc_seed`** — a PRODUCTION-PATH fixture: built (and re-fit) seeded from the published
  MPC orbit, whose **state is frozen into the fixture** (`mpc_seed_state`) so the suite refit
  stays offline and cannot drift with the MPC's own updates. For mega-arc objects that
  production always fits seeded (cli/web auto-seed >500-obs multi-opposition objects) and whose
  guarded behavior lives on that path — 1862 Apollo pins gaps #54's perturbed-delivery
  Yarkovsky (model + A2 sign −1 + all 17 radar rows, fully offline end-to-end), while its cold
  IOD fit is #54's separately-tracked open convergence item, a different thing to measure.
- **`benchmark`** — a heavy object kept for **speed measurement only**. It is **skipped by
  the routine run** (keeping the correctness suite fast), and under
  `pytest tests/test_regression.py --benchmark` only its **speed** is guarded — fit-quality
  (status/used/rms) is NOT. So a benchmark fixture's fit-quality can drift silently; only
  mark objects whose fit quality you're willing to stop guarding. `_stages.py` profiles
  them too.

**The `benchmark` + `quality` objects are the ones you must remember to run by hand.** They
carry real fit-quality and force-model guards, and the routine run never fires them — so a
change to *their* code path is unguarded until someone types `--benchmark`. As of 2026-09-07
that set is **1862** (gaps #54's perturbed-delivery Yarkovsky, 132 CPU s), **2008 NA** (gaps
#43, 30 s) and **21P** (the apparition-wise comet extension, 88 s). Run just the one you
touched:

```bash
pytest "tests/test_regression.py::test_fixture_still_links[21P]" --benchmark
```

Where a heavy fixture is the ONLY coverage of a large piece of logic, that is a gap the flag
cannot close — write offline unit tests for the logic and leave the fixture where it is. That
is what `test_comet_extension.py` does for `_comet_extend_apparitions`: a golden asserts an
*answer*, so it can never tell you a control-flow rule is wrong in a way that happens not to
move that one object.

## Normal workflow

1. Make an engine change.
2. `pytest tests/test_regression.py -n 4` — must stay green. A FAIL is either a real
   regression or an intended improvement the golden hasn't caught up to. (Parallel runs
   do not guard timing; re-run serially when the change was aimed at speed.)
3. If a change is intentional, eyeball the drift first: `python tests/update_goldens.py`
   (DRY RUN — re-fits stored obs offline, prints every `golden → current` delta and the
   suite-time delta, writes nothing). Confirm the changes are improvements / expected.
4. Only then `python tests/update_goldens.py --write` to bake the new baseline, and commit
   the fixtures with the code. (`update_goldens` re-fits through `test_regression.refit`,
   so a re-baseline always records exactly the currency the guard measures.)
5. To add a new object you wrestled into working: add it to `OBJECTS` in
   `build_fixtures.py`, run `python tests/build_fixtures.py "<desig>"`, commit its fixture.
   **Grow the suite every time you fix a new object / linkage / bug case.**

Avoid casual `--write`: re-baselining all goldens (esp. the noisy `secs`) on every tweak
is exactly the slow-drift we don't want. Re-baseline only the objects that genuinely
changed (the dry run shows which), and prefer reverting `secs`-only churn on the rest.

## What counts as an UNACCEPTABLE regression

`test_regression.py` flags an object (thresholds at the top of that file) when:

- **Linkage status** flips `ok → suspect`.
- **Used-obs drop**: `used` falls by more than **`USED_TOL = 2`** obs *and* rms did **not**
  also improve. (Fewer obs at equal-or-better rms is a cleaner fit, not a regression;
  more obs is always fine.)
- **RMS worsens** by more than `max(RMS_ABS_TOL = 0.05″, RMS_REL_TOL = 0.25 × golden_rms)`.
- **Force model changes** — `nongrav` differs from the golden, or a coefficient's sign flips.
  **No tolerance, and no "improvement" direction**: there is no such thing as a slightly different
  force model, and gaining a nongrav is as much a selection change as losing one. Both need looking
  at and re-baselining deliberately. Only checked once a fixture has been baselined with the key
  (see *Each fixture*).
- **Speed** (serial runs only): CPU time worsens by more than `max(30% × baseline, 0.5 s)` for a **bulk**
  object — but identification **PAIRs** (label `PAIR_*`, the deliberately hard cases) get a
  **lenient** guard of `max(~2× baseline, +1 s)`. We optimise for total-list time: speed up
  the bulk, and it's fine if a few hard pairs slow down a bit when a global win pushes them
  onto their careful (mislinkage-trick) path. A pair that *blows up* (≫2×) still fails.

Equivalently, an acceptable change keeps status, doesn't clip >2 obs without an rms win,
doesn't worsen rms past 25% / 0.05″, and doesn't slow a bulk object >30% (>0.5 s) — pairs
may slow up to ~2×.
"Rejecting more observations" is judged against the fixture's stored `used` count via the
used-drop rule above — losing >2 previously-used obs (without an rms improvement) fails.

> The speed guard is deliberately TIGHT (30% / 0.5 s). CPU time removes the sensitivity to
> unrelated load that wall clock had, but not the sensitivity to *co-running fits* — so it is
> disabled under xdist and you should measure timing baselines (and re-run the suite) serially
> on a quiet machine. Re-baseline `secs` with `update_goldens.py --write` after any real speed
> change so the guard tracks the current engine rather than flagging jitter.

## Profiling and scratch diagnostics

The `_`-prefixed modules in `tests/` are **tooling, not tests** — pytest ignores them,
and they are run by hand against the frozen fixtures:

| module | what it does |
|---|---|
| `_stages.py` | re-fits the suite and aggregates wall-clock by pipeline stage (`iod / subarc_extend / linkage / reject_drop / final_fit / products`) with per-stage CV, to spot the slow / high-variance stage. See `speed.md`. |
| `_sweeps.py` / `_phase_sweeps.py` | count `_fit_from_start` sweeps per object, and break them down by *phase* (initial bridge / drop-screen / sigma_a probe / re-link) so you can see which phase burns them. |
| `_sys_sweep.py` → `_sys_analyze.py` / `_sys_diff.py` | the measurement rig behind the systematic-misfit detector: sweep every fixture recording both the current gate's statistic and a proposed one, analyse where a threshold would land, and diff two runs for changed verdicts. This is how a gate threshold gets chosen from the observed distribution rather than guessed. |
| `pair_shapes.py` | the `PAIR_` night-shape dump described above. |
| `itf_sql_fuzz.py` | the SQL-vs-Python pushdown differential, described below. |

`_fixtures.py` and `_itf_fixtures.py` are the shared fixture helpers, not tools.

## ITF recovery fixtures (`itf_fixtures/`, `test_itf_recovery.py`)

A separate, ITF-specific fixture set — distinct from the fit-quality archive above.
Each `itf_fixtures/<label>.json` freezes one object's **published OBS80** (the ITF
harness is line-based, so it stores the raw 80-col lines, not parsed obs) plus the
OFFLINE recovery target. `build_itf_fixtures.py` writes them (fetches once); the seed
is then deterministic and network-free.

`test_itf_recovery.py` re-parses the frozen astrometry (via the same `read_obs_file`
path a live fetch uses), plants the object's other apparitions as fake ITF, and searches.
It is **two parametrized passes over the same 16 fixtures**:

- `test_recovers_offline` (fake-only): the **deterministic golden** — every planted
  apparition must recover under the fixture's rms cap. Reproducible anywhere, no DB.
  This is what a plain `pytest` run does: ~18 s under `-n 4`.
- `test_recovers_against_live_distractors` — marked **`live`**, so it runs only under
  `pytest --live`. Merges the **live ITF as real distractors** beside the planted fakes;
  the recovery must survive real neighbours. Requires SBN DB creds + egress, and SKIPS
  (rather than fails) when the database is unreachable.

Live used to be the *default*, with a silent fall-back to offline on any exception —
so a plain run sat on connection timeouts for a quarter of an hour and then tested
something other than what it reported. Splitting the two makes each attributable.
**Still run the live pass before trusting a recall-affecting change**: offline cannot
see anything on the DB side of the search, and a green offline run has twice hidden a
live recall loss.

Not run by the fast `test_itf_search.py` unit suite (it does full engine fits and
optionally hits the DB). The current set — 2019 PJ60, 2015 BE559, 2014 TM87, 2024 RX190
— pins one object per ITF recall fix (corridor mult, variant corridor, seed-guard noise
floor, recursion accumulation). The live mode paid off immediately: it caught the
`obs_unique_id` exposure-collision that dropped PJ60's true tracklet when real ITF
neighbours shared its G96 exposure.

### `itf_sql_fuzz.py` — the SQL pushdown vs its Python twin

The sweep pushes the corridor test and the motion pre-filter into SQL, so those
predicates exist twice. The contract is one-directional: the SQL may be LOOSER
than the Python predicate (`_survivors` re-applies the exact test to whatever
comes back), never stricter — a stricter SQL drops candidates server-side with
no error, and the search reports a clean negative. Silent recall loss is the one
failure this system cannot see in its own output.

`itf_sql_fuzz.py` generates rows on each predicate's exact accept boundary and
asserts that property. Offline (`check_shim`, run inside `test_itf_search.py`)
it transliterates the SQL into Python — proves the algebra. Live
(`python tests/itf_sql_fuzz.py [N]`) it hands the same rows and the real SQL
string to Postgres via `unnest(...)`, read-only, catching what float8 semantics
or NULL handling would do differently. Both are mutation-checked: flipping a
1e-6 pad's sign or dropping a half-extent term fails them.

### Frozen ADES fields, frozen goldens, allowed-missing apparitions (2026-10-30)

- **Frozen astrometry:** the verbatim OBS80 lines (`obs80`) and, since 2026-10-30, the ADES fields
  the search's registry carries for each line (`ades`: Veres-2017 sigma, uncert_src, rms, mode,
  band ...). `_itf_fixtures.load_obs` restores them, so a fixture weights observations as the live
  ADES fetch does. Fixtures built before that have no `ades` field and re-parse as before.
- **The golden is run on the FROZEN data**, not the fetch: OBS80 precision (~1 s, ~0.15") can tip
  a search near a decision boundary, and a golden the test cannot reproduce is worse than none.
- **`missing_apparitions`** (an optional dict as the 6th element of an `ITF_OBJECTS` entry): planted
  apparitions the search may leave out -- a per-object decision, explained in the note. Every other
  planted apparition is required, and the tracklet bar excludes theirs (2008 GU20: its lone 2000
  tracklet).
