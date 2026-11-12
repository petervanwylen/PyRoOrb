# Installing PyRoOrb

PyRoOrb runs on Linux and macOS with Python 3.11–3.14, and natively on 64-bit Windows with
Python 3.11–3.12 ([Windows](#windows) has its own installer). Pick one route. Only want the web UI? Use
[pyroorb.net](https://www.pyroorb.net), or run it yourself without Python through
[Docker](#docker-the-web-ui-without-python).

## The install script (recommended)

```bash
curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash
# or, inside a clone:  ./install.sh
```

It checks for a C compiler (on macOS it starts the Command Line Tools installer and waits; on
Linux it prints the package-manager command, or runs it with `--yes`), installs
[uv](https://docs.astral.sh/uv/) when it is missing, creates `.venv` with a uv-managed
Python 3.12, installs PyRoOrb, downloads the reference tables, and links the PyRoOrb
commands into `~/.local/bin` (the folder uv installs itself into) so they run from any folder.
Options: `--dir DIR`
(where to clone; default `~/PyRoOrb`), `--extras itf` (the ITF search, see below; on macOS
the script also installs Homebrew's `libomp` for it), `--extras notebooks` (what the
[example notebooks](../notebooks/README.md) need, see below), several at once such as
`--extras itf,dev`, `--asteroids`
(adds the 616 MB asteroid ephemeris), `--no-data`, `--yes`.
Through `curl`, options go after `bash -s --`:
`curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash -s -- --yes`.

It needs `curl`, `git` and a C compiler: on macOS the Command Line Tools bring git and the
compiler; on Ubuntu/Debian `sudo apt update && sudo apt install build-essential git curl` (the script prints
the right command for apt, dnf, pacman and zypper, or runs it with `--yes`). The script uses
its own Python 3.12, whatever Python your system has. The environment takes about 250 MB
and the reference tables about 150 MB more; the ITF extra adds about 500 MB (mostly
xgboost).

**Then open a new terminal** and run `pyroorb-web` (the web UI) or `pyroorb "2023 QJ3"` from
any folder. You only need `source ~/PyRoOrb/.venv/bin/activate` to use PyRoOrb from your own
Python scripts or notebooks: it switches that terminal to PyRoOrb's Python, so `python` and
`import pyroorb` find it. If `~/.local/bin` is not on your PATH (uv's installer normally puts it
there), the script says so and prints the line to add.

**To update** an install made by the script: `cd ~/PyRoOrb && git pull && ./install.sh`
(the install is editable; re-running it picks up new dependencies). **To remove it:**
`~/PyRoOrb/install.sh --uninstall` ([Uninstall](#uninstall)).

## By hand: uv (the default) or pip

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you don't have uv yet
uv venv && uv pip install pyroorb
source .venv/bin/activate          # in every new terminal, before the pyroorb commands
pyroorb-fetch-data                 # once: the reference tables (see below)
```

uv installs Python itself if you have none. With pip, make a virtual environment first:
current Ubuntu, Debian and Homebrew Pythons refuse a bare `pip install`
(`externally-managed-environment`).

```bash
python3 -m venv .venv              # Ubuntu/Debian: needs  sudo apt install python3-venv
source .venv/bin/activate
pip install pyroorb
pyroorb-fetch-data
```

To update a by-hand install: `uv pip install --upgrade pyroorb` (or `pip install -U pyroorb`).
To run the development version instead, install it from the GitHub repository:
`uv pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"`.

### If PyRoOrb stops with "ASSIST cannot run against the installed REBOUND"

ASSIST is compiled against REBOUND's headers, so the two must be the same release. They drift
apart when an older REBOUND was already in the environment while `pip` built ASSIST, after
`pip install -U rebound`, or when `uv` reuses an ASSIST wheel it built for another REBOUND.
PyRoOrb checks this before its first fit; without the check the symptom is a process or Jupyter
kernel that dies with no traceback (`free(): invalid size`, `corrupted size vs. prev_size`).
Rebuild ASSIST against the REBOUND you have (`install.sh` does the same):

```bash
pip install setuptools wheel      # --no-build-isolation needs them in the environment
pip install --no-build-isolation --force-reinstall --no-deps --no-cache-dir assist
```

With uv: `uv cache clean assist`, then `uv pip install setuptools wheel`, then
`uv pip install --reinstall --no-build-isolation --no-deps assist`.

**A C compiler is needed** for now: ASSIST, the force model PyRoOrb integrates with, has no
pre-built wheels on PyPI, so the install compiles it (about half a minute), against REBOUND 4
(ASSIST 1.2 requires `rebound<5`, and PyRoOrb pins the same).

- **macOS** — the Xcode command-line tools: `xcode-select --install`.
- **Linux** — your distribution's `gcc` (e.g. `sudo apt update && sudo apt install build-essential`).

## Windows

PyRoOrb runs natively on 64-bit Windows (no WSL, no Docker). Two of its dependencies set the
terms. REBOUND 4 publishes Windows wheels for CPython 3.10–3.12 only, so the Python is 3.11 or
3.12. ASSIST is compiled during the install with the MSVC compiler, and the ASSIST release on
PyPI does not build on Windows yet: until one does, ASSIST comes from
[a fork with native Windows support](https://github.com/petervanwylen/assist-win-support/tree/windows-native-support),
pinned to a commit in `install.ps1`.

**The install script.** In PowerShell:

```powershell
irm https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.ps1 | iex
# or, inside a clone:  powershell -ExecutionPolicy Bypass -File install.ps1
```

It does what `install.sh` does: checks for the Visual Studio Build Tools (the C compiler) and git
and prints the `winget` command that installs them (or runs it with `-Yes`), installs uv,
creates `.venv` with a uv-managed Python 3.12, installs REBOUND, ASSIST and PyRoOrb,
downloads the reference tables, and copies the PyRoOrb commands into `~\.local\bin`, adding
that folder to your user PATH if it is not there yet. The options are the same,
PowerShell-style: `-Dir`, `-Extras itf,dev`, `-Asteroids`, `-NoData`, `-Yes`, `-Uninstall`,
`-KeepData`. Through `irm`, options need the script block form:
`& ([scriptblock]::Create((irm https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.ps1))) -Yes`.

Then open a new terminal and run `pyroorb-web` or `pyroorb "2023 QJ3"` from any folder. Only
to use PyRoOrb from your own Python scripts or notebooks, switch the terminal to PyRoOrb's
Python with `& ~\PyRoOrb\.venv\Scripts\Activate.ps1` (if PowerShell refuses to run it, allow
local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`).

**By hand with pip.** Install the Build Tools once (no need for a Developer Command Prompt
afterwards; the build finds the compiler itself):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

then, with a Python 3.12 (or 3.11):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install "rebound>=4.4.11,<5" numpy setuptools wheel
pip install --no-build-isolation "assist @ git+https://github.com/petervanwylen/assist-win-support@20d12ef89b3c97e09dfb70f11d59c0d840b52eee"
pip install pyroorb
pyroorb-fetch-data
```

REBOUND goes first and ASSIST is built without isolation so it compiles against exactly the
REBOUND it will run with. The order matters: a bare `pip install pyroorb` with no ASSIST yet
builds the ASSIST release from PyPI, which stops with `cl : Command line error D8021`. Once
the fork is installed, pip keeps it, since it already satisfies PyRoOrb's ASSIST requirement.
For the latest unreleased code, the last line becomes
`pip install "pyroorb @ git+https://github.com/petervanwylen/PyRoOrb"`. The reference
tables live under `%LOCALAPPDATA%\pyroorb\Cache`.

What differs from Linux and macOS: the web UI's `/memory` page reports no process figures
(they come from `/proc`), and the ITF collection harness bounds each object with a timer
thread instead of `SIGALRM`. Fits, ephemerides and every product are the same.

## conda-forge (Linux and macOS, no compiler)

conda-forge ships ASSIST and REBOUND pre-built for Linux (x86-64, arm64) and macOS (Intel,
Apple silicon). Install those two from conda-forge, then PyRoOrb with pip:

```bash
conda create -n pyroorb -c conda-forge python=3.12 assist rebound
conda activate pyroorb
pip install pyroorb
```


## Docker: the web UI without Python

For the web UI alone — on your own computer, or as a shared server for a lab or a group — the
Docker image needs no Python, no compiler and no install script; only
[Docker](https://docs.docker.com/get-docker/). It is built for Intel/AMD and ARM (Apple
silicon, ARM servers).

```bash
docker run -d --name pyroorb -p 8000:8000 -v pyroorb-data:/data ghcr.io/petervanwylen/pyroorb
```

Open http://localhost:8000 (on a server: `http://<server>:8000`). The first start downloads the
reference tables (~110 MB) into the `pyroorb-data` volume; every later start, and every newer
image, reuses them. Until the download finishes the page does not load: `docker logs pyroorb`
shows its progress.

| Option | What it does |
|---|---|
| `-e PYROORB_ASTEROIDS=1` | also downloads the 616 MB asteroid ephemeris (the 16 massive perturbers); the engine switches them on for the objects that need them (e.g. a close pass of Ceres), not for every fit |
| `-p 9000:8000` | serve on another port of the host |
| `-e PYROORB_MEMORY_PASSWORD=…` | unlocks the `/memory` diagnostics page |
| `-v pyroorb-data:/data` | keeps the tables and caches across restarts and updates; leave it out and every new container downloads them again |

**Update** to the newest version:

```bash
docker pull ghcr.io/petervanwylen/pyroorb
docker rm -f pyroorb && docker run -d --name pyroorb -p 8000:8000 -v pyroorb-data:/data ghcr.io/petervanwylen/pyroorb
```

**A shared server or lab** — the repository's [`compose.yaml`](../compose.yaml) runs it with the
asteroid ephemeris, the data volume and a restart policy:

```bash
docker compose up -d                              # start (and restart after reboots)
docker compose pull && docker compose up -d       # update
```

The server is plain HTTP; on a network beyond your own, put it behind a reverse proxy that
adds HTTPS (Caddy, nginx, Traefik). It limits each client's rate on the costly routes by
itself. Images are tagged `latest` (the newest from `main`), `X.Y.Z` (each release) and
`sha-…` (each commit). To build it yourself from a clone: `docker build -f docker/Dockerfile -t pyroorb .`

The image is the web UI only: the Python API, the notebooks and the command-line tools need
a normal install (above).

## The notebooks extra

The [example notebooks](../notebooks/README.md) also need matplotlib, pandas and JupyterLab.
They are not installed by default:

```bash
./install.sh --extras notebooks          # with the script
uv pip install -e ".[notebooks]"         # or by hand, inside a clone (or pip install -e ...)
```

The notebooks themselves come with the clone, not with the installed package; open them with
`jupyter lab notebooks/` from the clone. Notebooks 01, 02, 04, 11 and 12 also use the asteroid
ephemeris: add `--asteroids` to the script, or run `pyroorb-fetch-data --asteroids`.

## The ITF search extra

The ITF search reads an SBN database and ranks candidates with a trained model. It is not
installed by default:

```bash
./install.sh --extras itf                                                      # with the script
uv pip install "pyroorb[itf]"                                                  # or by hand (or pip install ...)
```

On macOS the ranker's xgboost also needs the OpenMP runtime: `brew install libomp` (the
script runs it when Homebrew is present). xgboost's macOS wheels link against Homebrew's
`libomp` rather than bundling an OpenMP runtime; on Linux nothing extra is needed.
It needs an SBN database; most people should run ITF searches on
[pyroorb.net](https://www.pyroorb.net) instead. Setting up your own: [`ITF_SETUP.md`](ITF_SETUP.md).

## Reference data

A fit integrates against the JPL DE440 ephemeris (~102 MB) and debiases with the EFCC18
tables (~10 MB). Both download on first use; to fetch them ahead of time:

```bash
pyroorb-fetch-data                   # DE440 + debias tables
pyroorb-fetch-data --asteroids       # + JPL's sb441-n16 (the 16 over DE441's span, ~616 MB, optional)
```

The **asteroid-perturber kernel** (`pyroorb-perturbers-1900-2100-20km.bsp`, ~33 MB) holds the 343
massive asteroids for 1900-2100 (the 16 most massive for 1800-2200) and lets every fit be screened for the ones its observations
need (`fit_orbit(perturbers="auto")`, the default) -- or fitted in all 343 whether or not they
matter (`perturbers="all"`, `pyroorb --perturbers all`, the web page's "+343 ast" toggle; 5-18x
slower than the 16); it also supplies the 16 used by
identification fits when sb441-n16 is not installed. It **ships inside the package**, so every
install has it with nothing to download. On first use a ~75 MB position table is derived from
it into the cache folder. (An install without the bundled copy builds it with
`pyroorb-fetch-data` from JPL's small-body ephemeris sb441-n373: about a minute, reading
~165 MB of JPL's 15 GB file over range requests.)

It is a modified JPL kernel produced by the PyRoOrb project -- not a JPL product -- and is
redistributed under JPL's data-use policy and NAIF's rules for modified kernels; its comment
area (readable with any SPICE tool, e.g. `jplephem`'s `SPK.comments()`) states the source
(sb441-n373, Park et al. 2021, Courtesy NASA/JPL-Caltech), the modifications and the validation.

They go to your per-user cache directory (`~/.cache/pyroorb` on Linux,
`~/Library/Caches/pyroorb` on macOS, `%LOCALAPPDATA%\pyroorb\Cache` on Windows).
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

## Uninstall

**An install made by the install script** removes itself:

```bash
~/PyRoOrb/install.sh --uninstall
```

```powershell
powershell -ExecutionPolicy Bypass -File $HOME\PyRoOrb\install.ps1 -Uninstall
```

It first lists everything it will remove and keep, and why, then asks (`--yes` / `-Yes` skips
that question). It removes only what the script made:

- the PyRoOrb commands in `~/.local/bin`, and on Windows that folder's PATH entry when the
  script added it and nothing else is left in the folder;
- the folder's `.venv`, the Python environment;
- the reference tables and caches; `--keep-data` / `-KeepData` keeps them, for a reinstall
  without downloading them again. They are shared by every PyRoOrb install of yours, so they
  are also kept while another one is on PATH.

The PyRoOrb folder itself (`~/PyRoOrb`, or your `--dir`) is offered for deletion only when the
script downloaded it and it holds nothing of yours: no files, no stash, no commit that is on no
remote, no other git worktree. Even then it is a separate question, asked last, that `--yes` /
`-Yes` does not answer; with no one to answer it, the folder stays and the command to delete it
is printed. A folder updated rather than downloaded by the script, or a clone of your own where
you ran `./install.sh`, keeps everything but `.venv`. With `--dir` / `-Dir`, that folder is the
install removed, wherever you run the command from.

It leaves uv, Python, the C compiler and git alone, since other software can use them, and
does not uninstall packages from any environment it did not create.

**Any other install** is undone by deleting what you created, plus the data folders below:

- **PyPI or pip into an environment of your own:** `pip uninstall pyroorb`. That leaves its
  dependencies (ASSIST, REBOUND, numpy, ...) in place, in case something else uses them; if you
  made the environment just for PyRoOrb, delete it instead.
- **conda:** `conda env remove -n pyroorb` (the environment the conda route made).
- **Docker:** `docker rm -f pyroorb && docker volume rm pyroorb-data`, then
  `docker rmi ghcr.io/petervanwylen/pyroorb`.

| Data folder | Linux | macOS | Windows |
|---|---|---|---|
| Reference tables and download cache (150 MB to 2 GB) | `~/.cache/pyroorb` | `~/Library/Caches/pyroorb` | `%LOCALAPPDATA%\pyroorb` |
| Web UI fit cache (up to 256 MB) | `/tmp/pyroorb-fitcache` | `$TMPDIR/pyroorb-fitcache` | `%TEMP%\pyroorb-fitcache` |
| Saved state | `~/.local/share/pyroorb` | `~/Library/Application Support/pyroorb` | `%APPDATA%\pyroorb` |

The first two are caches: deleting them loses nothing that cannot be downloaded or computed
again (`PYROORB_CACHE_DIR` and `PYROORB_FIT_CACHE_DIR` move them). The saved-state folder is
the one place PyRoOrb keeps anything meant to last (`LOCAL_PERSISTENT_FOLDER` moves it); the
uninstall never deletes it, so look inside and delete it yourself if you no longer need it.

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
