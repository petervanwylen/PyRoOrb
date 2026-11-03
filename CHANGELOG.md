# Changelog

PyRoOrb follows semantic versioning from 1.0.0, the first release on PyPI; until then the
0.9.x releases on GitHub may still change the Python API.

## Unreleased

- Fixed: planetocentric elements and sigmas (`osculating_elements(center=...)`, the satellite
  detection and report) took the planet from astropy's built-in ephemeris, ~80,000 km from
  DE440 at Jupiter; they now use DE440, the ephemeris the fit integrates in (the built-in one
  remains the fallback without the DE440 file).
- `ephemeris(..., observer=[x, y, z])`: the observer's geocentric equatorial position (au)
  in place of the station's, one vector or one per epoch. This is how a space telescope is
  placed (the vector an observation carries as `sat_pos_au`); an ephemeris for C57 (TESS)
  by code alone was ~800" off.
- `body_states(body, times)`: heliocentric states of the Sun, planets, Moon, Pluto and the 16
  asteroid perturbers, from the ephemerides PyRoOrb integrates in; `close_approaches` (and an
  ensemble's) accept Pluto and those asteroids as bodies. `ref` (the publication reference) is a
  supported observation-record key. Notebooks 11, 12 and 14 now use only the public API.

- Removed `fit_orbit(force=...)` from the Python API. The experimental brute-force linker can
  drive two different objects into one clean-looking orbit, so it is not something to build
  on; the command line's `pyroorb --force` keeps it.

- Fixed: an orbit polished with the 16 asteroid perturbers (`--asteroids`, the web UI's toggle
  and its automatic main-belt polish) had its ephemeris computed with the planets only, so the
  prediction did not follow its own orbit (a polished (197) Arete: 0.08-0.38", 7-25 times its
  own 1-sigma). Every prediction now uses the force model the orbit was fitted in, whatever
  the process-wide switch says when it is called. Planets-only fits are unchanged.

- Fixed: `fit.force_model` after the automatic asteroid polish (and the perturber toggle).
  The polished orbit carried `asteroids=True` under the gravity label it started from,
  "ASSIST(DE440+GR)"; it now reads "ASSIST(DE440+GR+16ast)" (or "+15ast, excl. ..." for
  one of the 16 perturbers), and a toggle back to gravity relabels the other way.
- Fixed: the ephemeris uncertainty ellipse through a deep planetary encounter. Its Jacobian
  used the fitter's fixed step, which went non-linear at Apophis's 2029 pass (1-sigma 265"
  at closest approach, against 3.7" from 300 sampled orbits); the step is now capped at 0.01
  sigma per axis. Orbits whose uncertainty exceeds the old step are unchanged exactly.

- API reference (`docs/`, Sphinx + numpydoc): the supported functions and `FittedOrbit`
  members in stable and provisional tiers, and one page of conventions (units, frames, time
  scales: the orbit's epoch is TDB, input times UTC).
- `pyroorb.load_settings()`: load the `.env` settings file (was internal `envfile`).
- `fit_orbit(..., nongrav="auto"|"none", center=, detect_satellite=, mpc_seed=)`: the
  gravity-only comparison fit, planetary satellites, and the published-orbit start the
  command line uses. Defaults unchanged.
- `Observations.select(start, stop, stations=, exclude_stations=, radar=)`, and a documented
  set of supported observation-record keys.
- `pyroorb.digest2_scores(observations, per_tracklet=False)`: digest2 class scores with no fit.
- `FittedOrbit.covariance_matrix` refuses a `frame` other than "elements" or "state" (any
  other value used to return the state covariance).
- Forced non-gravitational models: `fit_orbit(..., nongrav="comet2" | "yarkovsky" | ...)`.
- `asteroid_perturbers()`: the 16 asteroid perturbers for every fit, or for a `with` block.
- Dynamics in the fit's own force model, with linear uncertainty: `propagate` (state and
  covariance at any time), `osculating_elements` (of any state, heliocentric or
  planetocentric), `close_approaches` (planets and the Moon: time, distance, relative speed,
  1-sigma; matches JPL's close-approach data for 2024 MK to 7 ms and 84 m).
- Radar: `radar_residuals` (any orbit against any radar list) and `radar_prediction`
  (delay and Doppler with 1-sigma, for planning).
- Ensembles: `sample_orbits(fit, n, method="covariance" | "lov" | "sr")` returns an
  `OrbitEnsemble` to propagate, turn into elements and percentiles, project onto the sky
  (`sky_positions`) and scan for close approaches; covariance clones carry their own
  non-gravitational coefficients.
- Short arcs: `range_posterior` (the range profile as a distance distribution, with or
  without the population prior) and `most_probable_orbit`; new notebook 11 measures the
  prior on 24 tracklets against JPL.
- The notebooks use the public API wherever it exists; cells that still need an internal
  module are marked.

## 0.9.0 — first public release (GitHub)

- Orbit determination from ADES or MPC 80-column astrometry: initial orbits,
  arc extension and linking across apparitions, outlier rejection, differential correction
  over REBOUND/ASSIST (DE440 + GR), EFCC18 debiasing, non-gravitational parameters,
  radar astrometry, covariance and Monte-Carlo uncertainties.
- Products: elements and covariance, ephemerides with uncertainty, MOID / class / U,
  H and G, impact and keyhole reports (experimental), exports to MPCORB, SOF, eight-line
  elements and find_orb's covar.json.
- Python API: `read_observations`, `fetch_observations`, `fit_orbit` (flags an undetermined
  orbit with `UndeterminedOrbitWarning`), `ephemeris`, `export`.
- Command line: `pyroorb`, `pyroorb-web` (the web UI), `pyroorb-itf` (ITF search; needs your
  own SBN database), `pyroorb-fetch-data`.
- The MPC station table ships with the package; downloads are cached briefly and the cache
  is capped.
