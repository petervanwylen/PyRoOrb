# Vendored: PyDigest2

This directory is a verbatim copy of the `pydigest2` package from
**PyDigest2** — https://github.com/petervanwylen/PyDigest2 at commit `2023a93`
(copied 2026-10-02) — a pure-Python port of **digest2**, the Minor Planet
Center's short-arc orbit classifier (C original: github.com/Smithsonian/mpc-public,
`digest2/`; algorithm: Keys et al. 2019, PASP 131:064501, arXiv:1904.09188).
Both are public domain.

It is vendored, not depended on, so the engine needs no other package. Do not
edit it in place: fix upstream in PyDigest2 and re-copy, so the two cannot
drift. The tests travel with it (`tests/digest2/`, files renamed
`test_d2_*` so their module names cannot collide with the engine's tests; only
their data paths were changed).

What PyRoOrb uses it for: the digest2 class scores (raw and no-ID) shown for
every single tracklet and every undetermined orbit, and tracklet by tracklet in
the observations table for q < 1.8 (`src/pyroorb/d2score.py`), and the population model as a prior over
orbit space (`digest2.model.csv`, the **raw** "All" flavour). The prior is for
DISPLAY only — it is never allowed to decide a hazard, identification, ITF or
NEOCP question, since it would answer them by assuming the population.

`model.load_model` writes a parsed cache (`data/digest2.model.csv.npz`) beside
the CSV on first load; it is gitignored and regenerated as needed.
