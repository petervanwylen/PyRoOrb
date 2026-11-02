# Example notebooks

These notebooks are **examples**: short, worked science questions that PyRoOrb answers in a
handful of lines, starting from the raw astrometry. Each one asks a question, fits the orbit(s)
from the observations, and **checks the answer against JPL's published values**. Where PyRoOrb
and JPL differ, the notebook says by how much and why.

They are demonstrations, not a test suite or a tutorial on every option. The supported Python
API is the one in the main [README](../README.md#quickstart) (`fetch_observations`,
`fit_orbit`, `ephemeris`, `export` and the `FittedOrbit` methods). A few notebooks reach into
internal modules to go further; those cells are marked *internal API, may change*.

| Notebook | The question |
|---|---|
| [01 Asteroid close approach](01_asteroid_close_approach.ipynb) | (4) Vesta and (197) Arete meet every ~18 years: when and how close is the 2030 pass, and can you see Vesta's 2012 tug in Arete's residuals? |
| [02 Apophis 2029 flyby](02_apophis_2029_flyby.ipynb) | How close will Apophis come on 13 April 2029, how sure are we, and what does Earth do to its orbit? |
| [03 Short-arc NEO](03_short_arc_neo.ipynb) | Replaying the discovery of 2024 MK: what do one, two, three nights tell you, and when is its 29 June close approach known? |
| [04 Yarkovsky on a near-Earth asteroid](04_yarkovsky_nea.ipynb) | Bennu: the fit finds the thermal (Yarkovsky) push unaided; how does its orbital drift compare with OSIRIS-REx? |
| [05 When is Yarkovsky detectable?](05_yarkovsky_detectability.ipynb) | Bennu again, with the data cut at successive years: when does the drift stand out of the noise, with and without radar? |
| [06 Comet outgassing](06_comet_nongrav_encke.ipynb) | 2P/Encke: how outgassing forces show up in a comet's orbit, and its A1/A2 against JPL |
| [07 Interstellar objects](07_interstellar_unbound.ipynb) | Are 'Oumuamua, Borisov and 3I/ATLAS really unbound, and how does a near-parabolic comet (C/1980 E1) compare? |
| [08 Radar's leverage](08_radar_leverage.ipynb) | 2017 BQ6: how much do five Goldstone radar measurements shrink a one-apparition orbit's uncertainty? |
| [09 Photometry](09_photometry_h_phase.ipynb) | (1036) Ganymed: absolute magnitude, the phase curve and station offsets from 11,000 archived magnitudes |
| [10 Planetary satellite](10_planetary_satellite.ipynb) | Ersa (a Jovian moon) vs (11351) Leucus (a Trojan): which one orbits the planet? |

## Running them

```bash
uv pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb" jupyterlab
pyroorb-fetch-data               # DE440 + the debias tables (once)
pyroorb-fetch-data --asteroids   # the 16-asteroid perturber kernel (~600 MB), used by 01, 02 and 04
jupyter lab notebooks/
```

The notebooks need network access: observations come from the Minor Planet Center, and the
comparisons query JPL's public APIs (SBDB, close-approach data, Horizons). Each one runs in a
few minutes on a laptop. The committed copies include their outputs, so you can read them
without running anything. Re-running them later uses newer observations, so the numbers can
shift a little.
