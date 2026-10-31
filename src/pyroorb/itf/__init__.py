"""Native ITF (Isolated Tracklet File) search for PyRoOrb.

Turns a fitted orbit into an ephemeris corridor, searches the MPC ITF (plus the
pseudo-ITF of few-tracklet / no-orbit designations) for candidate tracklets,
trial-fits candidates with the PyRoOrb engine, and recurses on improvements --
the find_orb-era "itfsearch" workflow rebuilt on our own fitter/ephemerides.

Modules
-------
The search
  config          -- ``ItfConfig``: every switch and tunable of the search pipeline
  search          -- the core ``itf_search`` recursion driven by the PyRoOrb fitter
  verdict         -- ``SearchVerdict``: what a search concluded (linked / possible / none)
  ephem           -- ephemeris corridor windows from a PyRoOrb fit (geocentric, 500)
  lov_variants    -- LOV variant-orbit corridors, the fallback for under-determined seeds
  lov_profile     -- per-candidate LOV-offset profiles from the variant sweep
  lov_refine      -- the continuous LOV offset, measured (the last-resort retry)
  candidates      -- per-window candidate retrieval (index fast path, legacy scan,
                     and the overlay source used by the recovery test harness)
  fitter          -- the PyRoOrb fitting adapter: trial fits, subset statistics,
                     residual dropping, the warm-start rejection screen
  residual_trend  -- residuals that walk across a tracklet: a diagnostic, never a gate
  night_coherence -- per-night residual coherence of a linkage
  pickup          -- a partial pickup of a designated object, and what it implies
  mpcrules        -- the MPC's own acceptance rules for an identification
  linkage         -- MPC identification-submission JSON assembly (default: dry-run)
  trace           -- the live decision feed (CLI trace / web "see trace" stream)
  decisions       -- the search's decisions, logged for ranker training
Observation text
  obs80_fields    -- 80-column line helpers: obs identity, epochs, wrap handling
  opp_profile     -- night / span / opposition counting; ``OppProfile``
  obsreg          -- the ADES registry behind a line-based search
  jitter          -- the astrometry-jitter hack (off by default)
Data, ranking and maintenance
  dbio            -- SBN database connections (read-only ``pub`` / index-owner ``itf``)
  index           -- the precomputed ``itf.itf_tracklets`` table: features + maintenance
  maintain        -- CLI entry points for the index maintenance jobs
  scheduler       -- the always-on single-flight clock running those jobs
  training        -- ML candidate-ranking features, training and deployed scoring
  trainv4         -- the deployed v4 ranker's training
  traineval       -- dataset audit and head-to-head model evaluation
Running and measuring
  harness         -- the synthetic recovery harness (plants oppositions as fake ITF)
  gatelog         -- opt-in instrumentation of the thin-profile gates
  gatestats       -- clause statistics for those gates
  memory          -- memory actually left, inside a container
  memory_sampler  -- peak resident memory of a search, sampled
"""

from .config import ItfConfig                                    # noqa: F401
