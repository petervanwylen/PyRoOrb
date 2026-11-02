#!/usr/bin/env bash
# PyRoOrb installer for macOS and Linux (Windows is not yet supported; support comes late 2026).
#
#   ./install.sh                  from a checkout: installs into ./.venv
#   curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash
#                                 clones into ~/PyRoOrb first (--dir to choose), then installs
#
# What it does, in order (each step is skipped when already satisfied):
#   1. checks for a C compiler -- ASSIST, the ephemeris integrator, is compiled during the
#      install -- and for git and curl. macOS: the Xcode Command Line Tools, which also bring
#      git (it starts their installer and waits); Linux: gcc/cc, git, curl (it prints, or
#      with --yes runs, the package-manager command);
#   2. installs uv (https://docs.astral.sh/uv/) when it is missing;
#   3. creates .venv with a uv-managed Python 3.12 (which carries its own C headers);
#   4. installs PyRoOrb and its dependencies in ONE uv command. ASSIST builds against REBOUND 4;
#      a build constraint (rebound<5) pins that for the build;
#   5. only with --extras itf (the ITF search, which needs your own SBN database): on macOS its
#      ranker's xgboost needs the OpenMP runtime, installed with Homebrew (brew install libomp);
#   6. downloads the reference tables (DE440, the debias tables; --asteroids adds the 616 MB
#      asteroid ephemeris). --no-data skips this.
#
# Options: --dir DIR  --extras LIST (default: none; e.g. itf, or itf,dev)  --asteroids
#          --no-data  --yes (run system package installs without asking)  -h|--help
#          Through curl, options go after "bash -s --":  curl -LsSf …/install.sh | bash -s -- --yes
set -euo pipefail

# Everything runs inside main(), called on the last line: piped through `curl | bash`, bash
# then has the whole file before running any of it, so a cut-off download runs nothing, and a
# command that reads standard input (sudo, apt-get, brew) cannot swallow the rest of the script.
main() {

REPO_URL="${PYROORB_REPO_URL:-https://github.com/petervanwylen/PyRoOrb.git}"
PYTHON_VERSION="3.12"
DIR="${PYROORB_DIR:-$HOME/PyRoOrb}"
EXTRAS=""
ASTEROIDS=0
DATA=1
YES=0

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir|--extras)
      [ $# -ge 2 ] || die "$1 needs a value, e.g.  --dir ~/PyRoOrb  or  --extras itf,dev  (see --help)"
      if [ "$1" = --dir ]; then DIR="$2"; else EXTRAS="$2"; fi
      shift 2 ;;
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

is_checkout() { [ -f "$1/pyproject.toml" ] && grep -q '^name = "pyroorb"' "$1/pyproject.toml"; }
HERE=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# How to re-run with --yes: piped through curl, flags go after "bash -s --".
if [ -n "$HERE" ]; then RERUN_YES="./install.sh --yes"
else RERUN_YES="curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash -s -- --yes"; fi

# The Linux package-manager command that installs the named packages (apt names; dnf,
# pacman and zypper get their own names for a compiler). Empty when none is found.
linux_install_cmd() {   # $1 = apt package, $2 = the package name elsewhere
  if   command -v apt-get >/dev/null; then echo "sudo apt-get install -y $1"
  elif command -v dnf     >/dev/null; then echo "sudo dnf install -y $2"
  elif command -v pacman  >/dev/null; then echo "sudo pacman -S --noconfirm $2"
  elif command -v zypper  >/dev/null; then echo "sudo zypper install -y $2"
  fi
}
# Install a missing Linux prerequisite: run the command with --yes, else print it and stop.
need_linux_pkg() {      # $1 = what, $2 = apt package, $3 = the package name elsewhere
  local cmd; cmd="$(linux_install_cmd "$2" "$3")"
  [ -n "$cmd" ] || die "$1 is not installed; install it with your package manager and run this again"
  if [ "$YES" = 1 ]; then
    say "Installing $1: $cmd"; $cmd
  else
    die "$1 is not installed. Install it with:
    $cmd
then run this again (or let the script do it, asking for your password:  $RERUN_YES)"
  fi
}

# --- 1. a C compiler (first: on macOS the Command Line Tools also provide git) -------------
if [ "$OS" = Darwin ]; then
  # /usr/bin/git exists on every Mac, but until the Command Line Tools are installed it is
  # only a stub that opens their installer, so this must run before the clone below.
  if ! xcode-select -p >/dev/null 2>&1; then
    say "Installing Apple's Command Line Tools (a dialog opens; accept it)"
    xcode-select --install >/dev/null 2>&1 || true
    printf 'Waiting for the Command Line Tools to finish installing'
    for _ in $(seq 1 180); do                       # up to 30 minutes
      xcode-select -p >/dev/null 2>&1 && break
      printf '.'; sleep 10
    done
    echo
    xcode-select -p >/dev/null 2>&1 || die "the Command Line Tools are not installed yet; run this again when they are"
  fi
else
  if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
    need_linux_pkg "a C compiler" build-essential gcc
  fi
fi

# --- 0. the checkout -----------------------------------------------------------------------
if [ -n "$HERE" ] && is_checkout "$HERE"; then
  ROOT="$HERE"
elif is_checkout "$PWD"; then
  ROOT="$PWD"
else
  if ! command -v git >/dev/null; then
    [ "$OS" = Darwin ] && die "git is needed to download PyRoOrb; it comes with Apple's Command Line Tools:  xcode-select --install"
    need_linux_pkg "git" git git
  fi
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

# --- 2. uv ---------------------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  say "Installing uv"
  if ! command -v curl >/dev/null; then
    [ "$OS" = Darwin ] && die "curl is needed to install uv (https://docs.astral.sh/uv/)"
    need_linux_pkg "curl" curl curl
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || die "uv was installed but is not on PATH; open a new terminal and run this again"
fi

# --- 5 (before installing; --extras itf only). the OpenMP runtime for xgboost on macOS -----
# xgboost's macOS wheels link against Homebrew's libomp.dylib instead of bundling an OpenMP
# runtime (Apple's compiler ships none); on Linux the runtime comes with the wheel.
if [ "$OS" = Darwin ] && [[ ",$EXTRAS," == *",itf,"* ]]; then
  if command -v brew >/dev/null; then
    if ! brew list libomp >/dev/null 2>&1; then
      say "Installing the OpenMP runtime (brew install libomp) for the ITF ranker"
      brew install libomp
    fi
  else
    warn "You asked for the ITF extra, but Homebrew is not installed: its ranker (xgboost) needs Homebrew's libomp and will not load until you install Homebrew (https://brew.sh) and run  brew install libomp. Fitting and the web UI are unaffected."
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
if [[ ",$EXTRAS," == *",itf,"* ]] && ! .venv/bin/python -c "import xgboost" >/dev/null 2>&1; then
  if [ "$OS" = Darwin ]; then
    warn "the ITF ranker's xgboost does not load (on macOS it needs: brew install libomp). Fitting and the web UI work without it."
  else
    warn "the ITF ranker's xgboost does not load (try: .venv/bin/python -c 'import xgboost'). Fitting and the web UI work without it."
  fi
fi

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

To update later:  cd "$ROOT" && git pull && ./install.sh   (re-running picks up new dependencies)
Settings (data folders, credentials) can live in one .env file; see docs/INSTALL.md.
EOF
}

main "$@"
