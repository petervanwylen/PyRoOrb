<p align="center"><img src="docs/logo.png" width="120" alt="PyRoOrb"></p>

<h1 align="center">PyRoOrb</h1>

<p align="center">
  <strong>Raw astrometry in. A robust orbit out. All in Python.</strong>
</p>

<p align="center">
  <em><strong>Py</strong>thon <strong>Ro</strong>bust <strong>Orb</strong>it determination</em>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#web-ui-quickstart">Web UI Quickstart</a> ·
  <a href="#python-quickstart">Python Quickstart</a> ·
  <a href="notebooks/README.md">Examples</a> ·
  <a href="#the-web-ui">Web UI</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="docs/INSTALL.md">Install guide</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.14-blue" alt="Python 3.11–3.14">
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License: GPL-3.0">
  <img src="https://github.com/petervanwylen/PyRoOrb/actions/workflows/ci.yml/badge.svg" alt="CI">
</p>

---

**PyRoOrb** (**Py**thon **Ro**bust **Orb**it Determination) turns raw asteroid/comet astrometry (ADES or MPC 80-column) into a
high-quality orbit — with realistic uncertainties, identifications, and ITF search — **entirely in Python, on every OS** (Linux, macOS and Windows). It ships with a **browser workbench**, but it is equally intended for science in Python notebooks. ITF search is primarily limited to web-based service a pyroorb.net but is possible for the motivated user to run locally with an SBN mirror of their own (please coordinate with the author before doing mass pipeline submissions using it in order to deconflict large number of duplicates arriving at MPC and to discuss submission credit).

## Why it exists

🪐 **See it.** The **web UI** renders the orbit in 3-D, plots O−C residuals,
lets you toggle observations and re-fit live, and *replays how the orbit was fit* —
a working tool that's also the clearest way to learn orbit determination. Use PyRoOrb for teaching and explaining and developing your own spatial intuition.

🛡️ **Trust it.** It links apparitions across many revolutions, rejects contaminants
instead of good data, recovers from a problematic data even during ITF search, and **tells you** when the
residuals say something might be wrong (or interesting!) **rather than hiding it**.

🐍 **Use it anywhere.** Linux, macOS and Windows; a small
Python API drops into your
pipeline, with covariance + Monte-Carlo uncertainties and ephemerides that carry a
sky-plane error ellipse.

There are excellent orbit codes already — find_orb (C/C++) and OrbFit (Fortran)
are some of the most widely used and trusted. What PyRoOrb
adds is a pure-Python, cross-platform, library-first, solution that is robust by default.

## Install

**Only want the web UI?** Nothing to install: use **[pyroorb.net](https://www.pyroorb.net)**.
To run the web UI yourself without installing Python — on your own computer, or as a shared
server for a lab — use the Docker image:

```bash
docker run -d --name pyroorb -p 8000:8000 -v pyroorb-data:/data ghcr.io/petervanwylen/pyroorb
```

then open http://localhost:8000 (the first start downloads ~110 MB of reference tables into
the `pyroorb-data` volume; if the page does not load yet, `docker logs pyroorb` shows the download). A ready `compose.yaml` for a shared server, updating and the
options: **[`docs/INSTALL.md`](docs/INSTALL.md#docker-the-web-ui-without-python)**.

**For the Python API, the notebooks and the command line** (macOS and Linux) — one script does everything: checks for a C compiler
(and starts Apple's installer if it is missing), installs [uv](https://docs.astral.sh/uv/)
if needed, creates an environment with its own Python, installs PyRoOrb and downloads the
reference tables:

```bash
curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash
```

It clones into `~/PyRoOrb` (or run `./install.sh` inside a clone you already have). Then
`cd ~/PyRoOrb && source .venv/bin/activate` and `pyroorb-web` opens the web UI.

It needs `curl`, `git` and a C compiler. On **macOS** Apple's Command Line Tools provide
the compiler and git (the script starts their installer if needed; `curl` is built in). On
**Linux** the script prints the install command for anything missing, e.g.
`sudo apt update && sudo apt install build-essential git curl` on Ubuntu/Debian; to let it run that for you,
pass `--yes` like this: `curl -LsSf … | bash -s -- --yes`. Expect about 400 MB on disk: the
environment (~250 MB) plus ~150 MB of reference tables in the cache. The ITF search, which
needs your own SBN database, is left out; `--extras itf` adds it (~500 MB more).

**By hand, with uv** (recommended) or pip:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you don't have uv yet
uv venv && uv pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"
source .venv/bin/activate   # in every new terminal, before the pyroorb commands
pyroorb-fetch-data          # once: DE440 (~102 MB) + the debias tables (~10 MB)
```

Python 3.11–3.14 on Linux or macOS; Python 3.11–3.12 on 64-bit Windows, where the installer is
`install.ps1` ([Windows](docs/INSTALL.md#windows)). One dependency, ASSIST, is compiled during
the install, so a C compiler is needed: Xcode command-line tools on macOS, `gcc` on Linux, the
Visual Studio Build Tools on Windows. Details, pip and conda:
**[`docs/INSTALL.md`](docs/INSTALL.md)**.

## Web UI Quickstart

Want to look at an orbit, not write code? The web UI is a point-and-click workbench; you
never need to touch Python.

1. **Open it.**
   - **Nothing to install:** go to **[pyroorb.net](https://www.pyroorb.net)**.
   - **On your own computer, no Python needed:** run the Docker image, then open
     <http://localhost:8000> (the first start downloads ~110 MB of reference tables):

     ```bash
     docker run -d --name pyroorb -p 8000:8000 -v pyroorb-data:/data ghcr.io/petervanwylen/pyroorb
     ```

   - **Already installed PyRoOrb** (see [Install](#install))**:** run `pyroorb-web` in a terminal
     and open <http://127.0.0.1:8000>.
2. **Load an object.** Type a name or designation (say `2023 QJ3` or `433`) into the box at
   the top, or paste your own MPC 80-column or ADES observations into the paste area.
3. **Fit it.** The orbit appears in 3-D with the residuals beside it. Click observations to
   include or exclude them and press **Full step** to re-fit.
4. **Go further.** Export an ephemeris or a pseudo-MPEC, or replay how the orbit was fit.
   Everything the UI offers is described under [The web UI](#the-web-ui).

Wanting to script it or work with the results in Python? Continue with the
[Python Quickstart](#python-quickstart).

## Python Quickstart

For scripts, notebooks and pipelines: the Python API and the command line (install the
Python package first, see [Install](#install)).

```bash
pyroorb "2023 QJ3"                              # fetch from the MPC and fit
pyroorb "2014 HM410" "2016 UA210"               # fit two designations as one object
```

```python
import pyroorb

pyroorb.load_settings()                         # your .env settings file, if you keep one
obs = pyroorb.fetch_observations("2023 QJ3")     # or pyroorb.read_observations("my.psv")
fit = pyroorb.fit_orbit(obs, designation="2023 QJ3")
print(fit.is_determined, fit.rms_residual(), fit.n_observations())
print(fit.elements(), fit.element_sigmas())     # a, e, i, Ω, ω, M … (+1σ), epoch
print(fit.moid_earth(), fit.orbit_class(), fit.uncertainty_parameter())

eph = pyroorb.ephemeris(fit, "2026-11-01", "2026-12-01", step=1.0, site="568")
print(pyroorb.export(fit, "mpcorb"))            # also "sof", "eightline", "covar"

grav = pyroorb.fit_orbit(obs, nongrav="none")   # gravity only, to compare
d2 = pyroorb.digest2_scores(obs.select(stop=obs[0]["jd_utc"] + 0.5))   # no fit needed

state, cov = pyroorb.propagate(fit, "2030-01-01", covariance=True)
print(pyroorb.close_approaches(fit, "2026-01-01", "2036-01-01", body="mars", max_distance=0.2))
clones = pyroorb.sample_orbits(fit, 500)        # or method="lov" / "sr"
cloud = clones.sky_positions("2027-01-01", site="568")   # non-linear sky uncertainty
```

When the observations do not determine an orbit (a single short tracklet, say),
`fit_orbit` still returns the nominal solution, warns, and sets `fit.is_determined` to
`False`; `fit.determinacy()` says what is known.

**API reference:** [`docs/api.rst`](docs/api.rst) lists the supported functions and
`FittedOrbit` methods, with stable and provisional tiers; [`docs/conventions.rst`](docs/conventions.rst)
states the units, frames and time scales (the orbit's epoch is TDB, input times are UTC).
Build it as HTML with `pip install -e ".[docs]"` and `sphinx-build -b html docs docs/_build/html`.
Everything not listed there is internal and may change before 1.0.

## Example notebooks

**[`notebooks/`](notebooks/README.md)** holds eleven short worked examples. Each one fits the raw
observations and checks its answer against JPL. The topics are Apophis's 2029 flyby, Vesta
passing (197) Arete, a new NEO's first nights, Yarkovsky on Bennu, comet outgassing, the
interstellar objects, radar, photometry, a Jovian moon, and a population prior for a single
night's orbit.

## How it works

```
astrometry (ADES / OBS80)
      │
      ▼  IOD seed ─────────── Gauss · Herget · Väisälä cascade, with sub-arc selection
      ▼  extend & link ────── grow the arc; bridge apparitions across revolutions
      ▼  differential corr. ─ Levenberg–Marquardt with an analytic, variational
      │                       (multi-particle) Jacobian over REBOUND/ASSIST
      ▼  reject / redeem ──── drop blunders, recover over-tight-σ obs, heal a bad seed
      ▼  products ─────────── elements + covariance, Monte-Carlo / U uncertainties,
                              ephemerides w/ error ellipse, MOID · PHA · class · H/G
```

## The web UI

`pyroorb-web` opens a single-page workbench:

- **3-D heliocentric orbit** — drag to rotate, scroll to zoom, scrub the date to watch the body move against the planets.
- **Orbarium** — the same scene flipped around the observer: each observation drawn as an Earth→object sightline, so you can *see* the geometry the fit was solved from.
- **Observations & residuals** — click any observation to include/exclude (excluded rows go deep red), then **Full step** to re-fit and watch the RMS respond. The **O−C vs time** plot toggles **arcsec ↔ σ**.
- **"How the orbit was fit"** — the fit trace as a timeline; click a stage to replay the orbit *as it was then* (with a clear **NOT FINAL ORBIT** banner and a way back).
- **Paste & export** — paste raw OBS80/ADES to fit on the spot; export a find_orb-style **pseudo-MPEC** or an **ephemeris** table.

<!-- ![The PyRoOrb web UI](docs/img/webui.png) -->

## Robustness — the part we're proud of

A fit that quietly converges on the *wrong* orbit is worse than one that fails
loudly. PyRoOrb is built to:

- link a clean apparition to a contaminated one **and reject the contaminant**, not gut a good night to force a fit;
- **bridge apparitions separated by many revolutions** (period-swept multi-start);
- recover when the initial-orbit seed is poisoned by a bad night;
- **flag per-apparition residual trends** — the tell of a bad identification — instead of burying them in dropped observations.

## ITF search

PyRoOrb includes a search of the MPC's Isolated Tracklet File for unlinked observations of an
object (`pyroorb-itf search "2017 CL37"`, or the web UI's **ITF search**).

- **The place to run ITF searches is [pyroorb.net](https://www.pyroorb.net)**, which keeps
  the index the search needs current.
- **Searching yourself** needs access to an SBN database: either your own clone of it, or
  the author's permission to use his (ask in a GitHub issue). A clone takes some setup —
  the tracklet index and its maintenance jobs — described in
  **[`docs/ITF_SETUP.md`](docs/ITF_SETUP.md)**. The search's extra packages are not
  installed by default: `./install.sh --extras itf` (or `pip install "pyroorb[itf] @ …"`).
- **Please give the author a polite heads-up before any mass ITF searching or submitting**
  (a GitHub issue). He already runs these searches across the ITF and submits what they
  find; more people running the same code over the same tracklets produces duplicate
  identifications, and every duplicate is work the Minor Planet Center has to process and
  discard.

ITF search is outside the supported Python API.

## Standing on the shoulders of giants

PyRoOrb is small because the giants are large. The basketball lineage is
deliberate: **REBOUND** grabs the rebound, **ASSIST** makes the pass, and
**PyRoOrb** finishes the play.

- **[REBOUND](https://github.com/hannorein/rebound)** — N-body integrator.
- **[ASSIST](https://github.com/matthewholman/assist)** — ephemeris-quality forces (DE440, GR).
- **[find_orb](https://github.com/Bill-Gray/find_orb)** (Bill Gray) — our reference oracle. Algorithms studied and some are **ported in pure Python; no source lines copied**.
- **EFCC18** — star-catalog astrometric debiasing.
- **[Minor Planet Center](https://minorplanetcenter.net/)** — observations & observatory codes.
- **astropy · numpy · scipy** — the scientific bedrock.

## License & citing

**GPL-3.0** (required by REBOUND/ASSIST).

The API may shift before 1.0, but only slightly. If it helps your work, please cite PyRoOrb. Questions and bug reports: open a GitHub issue.