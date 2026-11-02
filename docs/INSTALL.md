# Installing PyRoOrb

PyRoOrb runs on Linux and macOS with Python 3.11–3.14. **Windows is not yet supported;
support is coming in late 2026.** Pick one route.

## The install script (recommended)

```bash
curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash
# or, inside a clone:  ./install.sh
```

It checks for a C compiler (on macOS it starts the Command Line Tools installer and waits; on
Linux it prints the package-manager command, or runs it with `--yes`), installs
[uv](https://docs.astral.sh/uv/) when it is missing, creates `.venv` with a uv-managed
Python 3.12, installs PyRoOrb (with the ITF extra; on macOS that needs Homebrew's `libomp`,
which the script installs, or leaves the extra out when there is no Homebrew), and downloads
the reference tables. Options: `--dir DIR` (where to clone; default `~/PyRoOrb`),
`--extras itf,dev`, `--asteroids` (adds the 616 MB asteroid ephemeris), `--no-data`, `--yes`.
Through `curl`, options go after `bash -s --`:
`curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash -s -- --yes`.

It needs `curl`, `git` and a C compiler: on macOS the Command Line Tools bring git and the
compiler; on Ubuntu/Debian `sudo apt install build-essential git curl` (the script prints
the right command for apt, dnf, pacman and zypper, or runs it with `--yes`). The script uses
its own Python 3.12, whatever Python your system has. The environment takes about 750 MB
(most of it xgboost, for the ITF ranker; `--extras ""` leaves it out), the reference tables
about 150 MB more.

**To update** an install made by the script: `cd ~/PyRoOrb && git pull && ./install.sh`
(the install is editable; re-running it picks up new dependencies).

## By hand: uv (the default) or pip

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you don't have uv yet
uv venv && uv pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"
source .venv/bin/activate          # in every new terminal, before the pyroorb commands
pyroorb-fetch-data                 # once: the reference tables (see below)
```

uv installs Python itself if you have none. With pip, make a virtual environment first:
current Ubuntu, Debian and Homebrew Pythons refuse a bare `pip install`
(`externally-managed-environment`).

```bash
python3 -m venv .venv              # Ubuntu/Debian: needs  sudo apt install python3-venv
source .venv/bin/activate
pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"
pyroorb-fetch-data
```

To update a by-hand install, run its install line again so it refetches from GitHub:
`uv pip install --reinstall-package pyroorb "pyroorb @ git+…"`, or with pip
`pip install --force-reinstall "pyroorb @ git+…"` (which also recompiles ASSIST).

**A C compiler is needed** for now: ASSIST, the force model PyRoOrb integrates with, has no
pre-built wheels on PyPI, so the install compiles it (about half a minute), against REBOUND 4
(ASSIST 1.2 requires `rebound<5`, and PyRoOrb pins the same).

- **macOS** — the Xcode command-line tools: `xcode-select --install`.
- **Linux** — your distribution's `gcc` (e.g. `sudo apt install build-essential`).

## conda-forge (Linux and macOS, no compiler)

conda-forge ships ASSIST and REBOUND pre-built for Linux (x86-64, arm64) and macOS (Intel,
Apple silicon). Install those two from conda-forge, then PyRoOrb with pip:

```bash
conda create -n pyroorb -c conda-forge python=3.12 assist rebound
conda activate pyroorb
pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"
```


## The ITF search extra

The ITF search reads an SBN database and ranks candidates with a trained model:

```bash
uv pip install "pyroorb[itf] @ git+https://github.com/petervanwylen/PyRoOrb"   # or pip install ...
```

On macOS the ranker's xgboost also needs the OpenMP runtime: `brew install libomp`.
It needs an SBN database; most people should run ITF searches on
[pyroorb.net](https://www.pyroorb.net) instead. Setting up your own: [`ITF_SETUP.md`](ITF_SETUP.md).

## Reference data

A fit integrates against the JPL DE440 ephemeris (~102 MB) and debiases with the EFCC18
tables (~10 MB). Both download on first use; to fetch them ahead of time:

```bash
pyroorb-fetch-data                 # DE440 + debias tables
pyroorb-fetch-data --asteroids     # + the 16-asteroid perturber kernel (~616 MB, optional)
```

They go to your per-user cache directory (`~/.cache/pyroorb` on Linux,
`~/Library/Caches/pyroorb` on macOS).
`PYROORB_CACHE_DIR` moves the whole cache; `PYROORB_EPHEM_DIR` and `PYROORB_DEBIAS_DIR` move
the two tables. Downloaded observations are cached for a few hours only, and the cache is held
under 2 GB (`PYROORB_CACHE_MAX_MB`).

## Settings: one `.env` file

Every setting can live in one `.env` file of `KEY=value` lines, which every PyRoOrb command
(`pyroorb`, `pyroorb-web`, `pyroorb-itf`, `pyroorb-fetch-data`, and their `python -m` forms)
reads at start-up, before anything else:

```bash
PYROORB_EPHEM_DIR=/path/to/ephem
PYROORB_DEBIAS_DIR=/path/to/debias
```

The first file found is used: the path in `PYROORB_ENV_FILE`; `.env` at the root of a source
checkout; `.env` in the folder you run the command from, or in any folder above it. A variable
already set in your shell always wins over the file. Keep `.env` out of version control (the
repository's `.gitignore` already does).

## Run it

```bash
pyroorb "2023 QJ3"                      # fetch and fit
pyroorb "2014 HM410" "2016 UA210"       # two designations as one object
pyroorb --help
pyroorb-web                             # the web UI -> http://127.0.0.1:8000 ($PORT / $HOST honoured)
```

From Python:

```python
import pyroorb

obs = pyroorb.fetch_observations("2023 QJ3")      # or pyroorb.read_observations("file.psv")
fit = pyroorb.fit_orbit(obs, designation="2023 QJ3")
print(fit.is_determined, fit.rms_residual(), fit.elements())
eph = pyroorb.ephemeris(fit, "2026-11-01", "2026-12-01", step=1.0, site="568")
print(pyroorb.export(fit, "mpcorb"))
```

## From a source checkout (development)

```bash
git clone https://github.com/petervanwylen/PyRoOrb && cd PyRoOrb
./install.sh --extras itf,dev --asteroids         # or: uv venv && uv pip install -e ".[itf,dev]"
source .venv/bin/activate
pytest -m "not slow"                              # the fast tests
pytest -n 4                                       # everything, offline (needs the asteroid
                                                  # ephemeris: pyroorb-fetch-data --asteroids)
python -m pyroorb.web.devserver                   # web UI that restarts on code edits
```

The web UI's page tests use jsdom: `npm i jsdom`, then `node tests/web/<test>.js`.
