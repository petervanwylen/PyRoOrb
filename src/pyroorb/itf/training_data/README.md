# ITF tracklet-match training data + models

Labeled datasets and xgboost classifiers for ranking candidate ITF tracklets
by probability of belonging to the seed object (see `../training.py`).

## Files

- `itf_match_v4.json` (+ `.meta.json`) -- the **deployed ranker, ONE model**
  (2026-09-27), retiring the v3e + v2 pair. Built in `research/ranker-v4/`
  (README there: data design, the harness label leaks it found, the DAgger
  rounds, the live A/B). Trained by `itf/trainv4.py`; its feature frame is
  honest-scale + decision context (`training._v4_frame`), served whenever the
  artifact's meta says `"v4": true`. The pair's artifacts below stay for A/Bs.

- `itf_match_v3e.json` (+ `.meta.json`) — the **deployed PRIMARY** ranker
  (2026-08-07), from the v3 line's "hard" recipe (2026-08-05). Deployed as a
  PAIR with the v2 artifact as the second opinion
  (`ItfConfig.ml_second_model_path` default; `training.default_model_path` /
  `default_second_model_path`) — the pair is the product. The live gap that
  blocked every v3 candidate (better on all offline measures, loses BE559/GL24
  live) was a control-flow defect — one greedy take foreclosing a round — and
  was resolved by barren-take backtracking + the woven second-opinion trial source
  (`../ladder-backtracking.md`), NOT by a model change. Decision benchmark
  (2026-08-07, all 24 live fixtures × {v2, v3e, v3e+v2, v2+v3e}, prod flags):
  v3e+v2 24/24, median wall 24 s (v2: 24/24, 29 s; loses WY29 153→63 s,
  NW264 89→34 s, SF15, GL24, KX6 to the pair); v3e ALONE 22/24 (misses BE559
  AND GL24 — never deploy a single ranker without the second-opinion trial source);
  v2+v3e no gain. Known cost: 2021 SA70 ~+55% under the pair (consistent on
  re-runs), accepted. Raw runs: `/home/agent/pyroorb-data/kcluster_bench.jsonl`.
- `itf_match_v2.json` (+ `.meta.json`) — the **deployed SECOND-OPINION**
  ranker (primary 2026-07-20 → 2026-08-07). 40 objects /
  39,988 rows / 976 positives, collected 2026-07-20 (`itf_train_v2.csv.gz.part-*`,
  reassemble with `cat`). Balanced (scale_pos_weight), object-level 80/20 split.
- `itf_match_v3.json` (+ `.meta.json`) — the FIRST v3 attempt (2026-07-30, 277
  objects), kept as history: it won every offline measure and lost three live
  fixtures (see "the live gap" below and git history around 15baeb2).
- **v3b (2026-08-05)** — the current candidate, NOT yet deployed. Trained under
  the owner's two-sided acceptance framework (below). Datasets and artifact live
  outside git (`the download cache/itf_train_v3b.csv`, `itf_holdout_v3b.csv`; safe copies
  + model on the LOCAL fs at `/home/agent/pyroorb-data/` — the repo mount tears
  large writes, see "filesystem" below).

## The two-sided acceptance framework (owner, 2026-08-05)

A replacement model must (1) show **no regression vs the v2 model's
out-of-fold predictions on the v2 training rows** — the trusted set; AUC-type
parity suffices — and (2) **beat the deployed v2 artifact on a fresh holdout**
of objects neither model trained on.

- The v2-side baseline is `python -m pyroorb.itf train-eval data/itf_train_v2.csv`:
  leave-one-object-out under v2's own recipe, so every row is predicted by a
  model that never saw its object. In-sample scoring would flatter v2; OOF is
  its honest self-baseline.
- The holdout is collected on the permanent eval side of the hash wall
  (`--splits eval`, seed 20260806, 62 usable objects / 60,455 rows / 1,832
  positives) **after the v3b recipe was frozen**, with every v2 object, every
  live-suite fixture and every v3b training object barred by `--exclude-csv`.

### Results (v3b-final)

v2 rows (39,988; baseline = v2-OOF):

| | v2-OOF | v3b-final |
|---|---|---|
| within-window AUC | 0.9905 | 0.9870 (paired bootstrap CI [-0.015,+0.006] — tie) |
| within-window top-1 | 0.985 | **1.000** |
| within-object top-1 | 0.935 | 0.839 (3 of 31 objects worse; two are saturated ties, one real miss: 2020 HR169) |
| global AUC | 0.9902 | **0.9980** |
| Brier | 0.0179 | **0.0043** |
| precision @ confirm 0.2 | 0.061 | **0.970** |

Fresh holdout (60,455 rows; deployed v2 artifact vs v3b-final):

| | v2 | **v3b-final** |
|---|---|---|
| within-window AUC | 0.9162 | **0.9993** |
| within-window top-1 | 0.878 | **0.985** |
| within-object top-1 | 0.946 | **0.982** |
| recall @ confirm 0.2 | 0.778 | **0.989** (equal precision) |

Both criteria: **met**.

## The v3b recipe, and why

Fresh collections kept "somehow" losing to v2-OOF on v2 rows. The diagnosis
(SHAP on the killed positives) was NOT dist-shortcuts and NOT feature-scale
era drift (both tested; a median-ratio rescale made things worse): the model
was leaning on **population-prior features** — trk_sig_pos, stn, magnitudes,
trial_U — "publishedness" cues that separate planted truths from ITF junk
within one collection and transfer badly to another. v2-OOF never faces that
shift on v2 rows: it is in-distribution with its own eval in era and
collection quirks. (The same home advantage exists for v3b on the holdout —
which is why the framework needs both sides.)

Recipe: v2 feature set (`--v2-features`, so the model scores v2-era rows),
unbalanced, eta 0.08, **fixed 250 rounds** (AUC early stopping fires on
saturation noise — one run stopped at 19 trees), plus:

- **shortcut-feature dropout**: the 12 prior-ish features masked to NaN with
  p=0.3 during training, pushing weight onto geometry/motion features that
  transfer;
- **natural-population object weights**:every object weighted so strata
  contribute like the real MPCORB nopp(1,6) mix (MBA .934 / NEO .059 / TNO
  .006 / Cen .001), not like the NEO-heavy stratified sample.

Training data: passes 1+2 (seeds 20260804 / 20260805, 420 objects, 401k
surviving rows, 31,813 positives; stratified store `--keep-maybe/unlikely 25`).
Seed sigma on the decider metrics ~0.001 ww-AUC; variant differences below
that are noise.

## The live gap (deployment blocker), isolated

The 12-fixture interleaved live A/B (`tests/itf_model_ab.py`, seed apparition
PINNED — the old script re-seeded fixtures and invalidated the comparison):
v3b-final recovers 2022 WY29 (which the first v3 lost) but still fails
2015 BE559 and 2022 GL24 with the historical signature "1.5 opps without
progress, abandoning" — v2 12/12, v3b 4/12 on these three.

Gate recalibration is NOT the fix: at the current four ml_p thresholds v3b
already passes MORE truth than v2 (holdout recall 0.989 vs 0.778;
`python -m pyroorb.itf train-recalibrate` moves the gates only slightly). The mechanism,
measured directly on BE559 (collector run, scoring every served candidate
with both models): in the decisive 518-candidate corridor window **v2 ranks
the true tracklet #1; v3b ranks it #74** — both at near-zero absolute p
(0.004), so recovery rides entirely on relative order, and the search burns
`no_progress_sec`/`min_fruitless_fits` on 73 wrong candidates first. The
regime is extreme-U weak-seed NEO corridors, which a random nopp 1-6 sample
under-produces.

A targeted weak-seed NEO pass (v3c, seed 20260807, `--strata NEO --nopp 2,4`,
60 objects / 28,329 rows / 8,330 positives, fixtures still excluded) settled
it: retrained on everything (`itf_match_v3c.json`, 480 objects / 40,143
positives), the model HOLDS both acceptance criteria (v2 rows: ww-AUC 0.9889,
ww-top1 0.985 = parity; holdout: ww-AUC 0.9997, ww-top1 1.000) and STILL
fails BE559/GL24 identically — the decisive-window truth moved rank 74 -> 87.
Three independent datasets and recipes (v3, v3b, v3c) now reproduce the same
live gap, so the BE559-class ordering is not reachable from fresh clean
training data.

The SEARCH-side follow-up (2026-08-06) split the gap in two:

- **2022 GL24 is fixed** by the second-opinion ranker
  (`cfg.ml_second_model_path` + `recurse_second_singles`, default off): each
  trial queue appends the top-2 singles as ranked by a second model artifact, so
  no single model's blind spot can bury a candidate.  Live with v3-primary +
  v2-second: GL24 0/4 -> 4/4 (183 s), WY29 stays green, cost ~2 warm-screen
  trials per round.
- **2015 BE559 is a BUDGET artifact, not a ranking failure.**  The v2-driven
  trace reaches its truth only at LOV variant 37/55 (k=+2.83 sigma, 381 s),
  surviving that long because dozens of junk trial-takes across variants 1-36
  (every one later rejected by vetting) kept resetting the no-progress
  clock: v2's inflated scores GAME the budget accounting.  A calibrated
  ranker takes fewer wrong trials and abandons sooner -- so any honest model
  loses this fixture regardless of its ordering.  Deciding whether take-churn
  should count as progress (e.g. only VETTED takes reset the clock, plus a
  deeper variant-scan allowance) is an owner-level budget-design question.
- **The hard-negative store closes the loop (v3e)**: the owner's good-vs-great
  diagnosis produced `--hard-negatives` collections (positives + the deployed
  ranker's top-10 negatives per window).  Trained on 711 objects / 60,868
  positives / 236k rows of hard-mined data, v3e holds parity on the v2 rows
  (ww-AUC 0.9886, ww-top1 0.985) and reaches ww-AUC 0.9995 / top-1 1.000 on
  the holdout.  The proposed deployment shape -- v3e primary + v2
  second-opinion -- goes 8/12 on the live A/B (GL24 4/4 at 186 s, ~40% FASTER
  than v2's 303 s; WY29 green; BE559 still the budget artifact above).
- The traces' repeated TAKEN-then-rejected trials (e.g. CA7E4P2@G96,
  P21QYUh@F52) are NOT related to 2015 BE559 (owner-confirmed: impossible to
  fit) -- they are exactly the junk take-churn described above: a weak
  variant orbit absorbs a wrong tracklet for one trial fit, vetting rejects
  it, and the no-progress clock resets anyway.  An earlier revision of this
  note wrongly floated them as possible real sightings from date proximity
  alone.

## Filesystem

The repo mount (virtiofs) TEARS large writes: pass 1 lost 9.6% of rows to
dropped spans; a disk-full stretch cost pass 2 ~half its negatives (93% of
positives survived); a "cleanup" rewrite corrupted the copy. Mitigations, all
in `training.py`: batched-fsync `write_rows`, and a stdlib-csv pre-scan in
`read_dataset` that hands pandas the exact torn-row indices (field count,
label sanity, NUL bytes — `on_bad_lines` alone misses NaN-padded short rows).
Write large artifacts to the LOCAL fs (`/home/agent/pyroorb-data/`).

## Repro

```bash
python -m pyroorb.itf train-eval data/itf_train_v2.csv --pred-out oof.csv   # baseline
python -m pyroorb.itf train-data --objects-per-split 250 --seed S --splits train \
    --exclude-csv EXCLUDE.csv --out DATA.csv                       # collection
python -m pyroorb.itf train-model DATA.csv --split train --v2-features ...  # variants
python -m pyroorb.itf train-compare HOLDOUT.csv --model v2=... --model v3=...
python -m pyroorb.itf train-recalibrate HOLDOUT.csv --model NEW --ref OLD
```
