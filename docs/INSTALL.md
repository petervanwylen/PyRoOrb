# Installing PyRoOrb

PyRoOrb runs on Linux, macOS and Windows with Python 3.11–3.14. Pick one route.

## pip or uv (the main route)

```bash
pip install "rebound<5" numpy setuptools
pip install --no-build-isolation assist
pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"
```

or, with [uv](https://docs.astral.sh/uv/), which also installs Python itself if you have
none:

```bash
uv venv && uv pip install "rebound<5" numpy setuptools
uv pip install --no-build-isolation-package assist \
    "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"
```

**A C compiler is needed** for now: ASSIST, the force model PyRoOrb integrates with, has no
pre-built wheels on PyPI, so the install compiles it -- and its build reads REBOUND's
headers from the installed `rebound` package -- REBOUND 4, which ASSIST 1.2 requires -- which is
why REBOUND goes in first and ASSIST
is built without build isolation (in an isolated build it fails with *"rebound.h" not
found*).

- **macOS** — the Xcode command-line tools: `xcode-select --install`.
- **Windows** — "Build Tools for Visual Studio", workload *Desktop development with C++*.
- **Linux** — your distribution's `gcc` (e.g. `sudo apt install build-essential`).

## conda-forge (Linux and macOS, no compiler)

conda-forge ships ASSIST and REBOUND pre-built for Linux (x86-64, arm64) and macOS (Intel,
Apple silicon). Install those two from conda-forge, then PyRoOrb with pip:

```bash
conda create -n pyroorb -c conda-forge python=3.12 assist rebound
conda activate pyroorb
pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"
```

conda-forge has no Windows build of ASSIST; on Windows use the compiler route above.

## The ITF search extra

The ITF search reads an SBN database and ranks candidates with a trained model:

```bash
pip install "pyroorb[itf] @ git+https://github.com/petervanwylen/PyRoOrb"
```

It needs your own clone of the SBN database (`SBN_DB_HOST`, `SBN_PUB_PASSWORD`, and for the
index `ITF_DB_PASSWORD`, in the environment or a `.env` file).

## Reference data

A fit integrates against the JPL DE440 ephemeris (~102 MB) and debiases with the EFCC18
tables (~10 MB). Both download on first use; to fetch them ahead of time:

```bash
pyroorb-fetch-data                 # DE440 + debias tables
pyroorb-fetch-data --asteroids     # + the 16-asteroid perturber kernel (~616 MB, optional)
```

They go to your per-user cache directory (`~/.cache/pyroorb` on Linux,
`~/Library/Caches/pyroorb` on macOS, `%LOCALAPPDATA%\pyroorb\Cache` on Windows).
`PYROORB_CACHE_DIR` moves the whole cache; `PYROORB_EPHEM_DIR` and `PYROORB_DEBIAS_DIR` move
the two tables. Downloaded observations are cached for a few hours only, and the cache is held
under 2 GB (`PYROORB_CACHE_MAX_MB`).

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
python -m venv .venv && . .venv/bin/activate      # or: uv venv
pip install -e ".[dev]"
pytest -m "not slow"                              # the fast tests
pytest -n 4                                       # everything, offline
python -m pyroorb.web.devserver                   # web UI that restarts on code edits
```

The web UI's page tests use jsdom: `npm i jsdom`, then `node tests/web/<test>.js`.
