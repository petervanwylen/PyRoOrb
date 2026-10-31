# Changelog

PyRoOrb follows semantic versioning from 1.0.0, the first release on PyPI; until then the
0.9.x releases on GitHub may still change the Python API.

## 0.9.0 — first public release (GitHub)

- Orbit determination from ADES or MPC 80-column astrometry: find_orb-style initial orbits,
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
