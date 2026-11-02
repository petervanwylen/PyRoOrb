#!/usr/bin/env bash
# PyRoOrb installer for macOS and Linux (Windows is not yet supported; support comes late 2026).
#
#   ./install.sh                  from a checkout: installs into ./.venv
#   curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash
#                                 clones into ~/PyRoOrb first (--dir to choose), then installs
#
# What it does, in order (each step is skipped when already satisfied):
#   1. checks for a C compiler -- ASSIST, the ephemeris integrator, is compiled during the
#      install. macOS: the Xcode Command Line Tools (it starts their installer and waits);
#      Linux: gcc/cc (it prints, or with --yes runs, the package-manager command);
#   2. installs uv (https://docs.astral.sh/uv/) when it is missing;
#   3. creates .venv with a uv-managed Python 3.12 (which carries its own C headers);
#   4. installs PyRoOrb and its dependencies in ONE uv command. ASSIST builds against REBOUND 4;
#      a build constraint (rebound<5) pins that for the build;
#   5. on macOS, the ITF ranker's xgboost needs the OpenMP runtime: installed with Homebrew when
#      Homebrew is present, otherwise the [itf] extra is left out (fitting and the web UI work
#      without it);
#   6. downloads the reference tables (DE440, the debias tables; --asteroids adds the 616 MB
#      asteroid ephemeris). --no-data skips this.
#
# Options: --dir DIR  --extras LIST (default: itf; "" for none; e.g. itf,dev)  --asteroids
#          --no-data  --yes (run system package installs without asking)  -h|--help
set -euo pipefail

REPO_URL="${PYROORB_REPO_URL:-https://github.com/petervanwylen/PyRoOrb.git}"
PYTHON_VERSION="3.12"
DIR="${PYROORB_DIR:-$HOME/PyRoOrb}"
EXTRAS="itf"
EXTRAS_SET=0
ASTEROIDS=0
DATA=1
YES=0

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --extras) EXTRAS="$2"; EXTRAS_SET=1; shift 2 ;;
    --asteroids) ASTEROIDS=1; shift ;;
    --no-data) DATA=0; shift ;;
    --yes|-y) YES=1; shift ;;
    -h|--help)
      if [ -f "${BASH_SOURCE[0]:-}" ]; then sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      else echo "options: --dir DIR --extras LIST --asteroids --no-data --yes  (see install.sh in the repository)"; fi
      exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

OS="$(uname -s)"
case "$OS" in
  Darwin|Linux) ;;
  *) die "PyRoOrb supports macOS and Linux for now; Windows support is coming in late 2026." ;;
esac

# --- 0. the checkout -----------------------------------------------------------------------
is_checkout() { [ -f "$1/pyproject.toml" ] && grep -q '^name = "pyroorb"' "$1/pyproject.toml"; }
HERE=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -n "$HERE" ] && is_checkout "$HERE"; then
  ROOT="$HERE"
elif is_checkout "$PWD"; then
  ROOT="$PWD"
else
  command -v git >/dev/null || die "git is needed to download PyRoOrb (or download it yourself and run ./install.sh inside it)"
  if is_checkout "$DIR"; then
    say "Updating the PyRoOrb checkout in $DIR"
    git -C "$DIR" pull --ff-only
  else
    say "Downloading PyRoOrb into $DIR"
    git clone --depth 1 "$REPO_URL" "$DIR"
  fi
  ROOT="$DIR"
fi
cd "$ROOT"
say "Installing PyRoOrb from $ROOT"

# --- 1. a C compiler -----------------------------------------------------------------------
if [ "$OS" = Darwin ]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    say "Installing Apple's Command Line Tools (a dialog opens; accept it)"
    xcode-select --install >/dev/null 2>&1 || true
    printf 'Waiting for the Command Line Tools to finish installing'
    for _ in $(seq 1 180); do                       # up to 30 minutes
      xcode-select -p >/dev/null 2>&1 && break
      printf '.'; sleep 10
    done
    echo
    xcode-select -p >/dev/null 2>&1 || die "the Command Line Tools are not installed yet; run ./install.sh again when they are"
  fi
else
  if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
    if   command -v apt-get >/dev/null; then cmd="sudo apt-get install -y build-essential"
    elif command -v dnf     >/dev/null; then cmd="sudo dnf install -y gcc"
    elif command -v pacman  >/dev/null; then cmd="sudo pacman -S --noconfirm gcc"
    elif command -v zypper  >/dev/null; then cmd="sudo zypper install -y gcc"
    else die "no C compiler found; install gcc with your package manager and run this again"; fi
    if [ "$YES" = 1 ]; then
      say "Installing a C compiler: $cmd"; $cmd
    else
      die "no C compiler found. Install one with:  $cmd   (or run this script with --yes)"
    fi
  fi
fi

# --- 2. uv ---------------------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  say "Installing uv"
  command -v curl >/dev/null || die "curl is needed to install uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || die "uv was installed but is not on PATH; open a new terminal and run this again"
fi

# --- 5 (before installing). the OpenMP runtime for xgboost on macOS ------------------------
if [ "$OS" = Darwin ] && [[ ",$EXTRAS," == *",itf,"* ]]; then
  if command -v brew >/dev/null; then
    if ! brew list libomp >/dev/null 2>&1; then
      say "Installing the OpenMP runtime (brew install libomp) for the ITF ranker"
      brew install libomp
    fi
  elif [ "$EXTRAS_SET" = 0 ]; then
    warn "Homebrew is not installed, so the ITF extra is left out (its ranker needs Homebrew's libomp). Fitting and the web UI work without it."
    EXTRAS=""
  fi
fi

# --- 3 + 4. the environment and the one install command ----------------------------------
if [ ! -x .venv/bin/python ]; then
  say "Creating .venv with Python $PYTHON_VERSION"
  uv venv --managed-python --python "$PYTHON_VERSION" .venv
fi
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
echo 'rebound>=4.4.11,<5' > "$CONSTRAINTS"
TARGET="."
[ -n "$EXTRAS" ] && TARGET=".[$EXTRAS]"
say "Installing PyRoOrb ${EXTRAS:+with [$EXTRAS] }(compiles ASSIST; about a minute)"
uv pip install --python .venv/bin/python --build-constraint "$CONSTRAINTS" -e "$TARGET"
.venv/bin/python -c "import pyroorb, rebound, assist" \
  || die "the install finished but PyRoOrb or ASSIST does not import; please open an issue with this output"

# --- 6. reference tables -------------------------------------------------------------------
if [ "$DATA" = 1 ]; then
  say "Downloading the reference tables"
  if [ "$ASTEROIDS" = 1 ]; then .venv/bin/pyroorb-fetch-data --asteroids; else .venv/bin/pyroorb-fetch-data; fi
fi

cat <<EOF

PyRoOrb is installed in $ROOT/.venv

  cd "$ROOT"
  source .venv/bin/activate
  pyroorb "2023 QJ3"          # fit an object from the command line
  pyroorb-web                 # the web UI at http://localhost:8000

Settings (data folders, credentials) can live in one .env file; see docs/INSTALL.md.
EOF
