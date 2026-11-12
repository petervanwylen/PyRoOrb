#!/usr/bin/env bash
# PyRoOrb installer for macOS and Linux (on Windows, use install.ps1 in PowerShell).
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
#   6. downloads the reference tables (DE440 and the debias tables; the ~26 MB asteroid-perturber
#      kernel ships inside the package; --asteroids adds the 616 MB
#      asteroid ephemeris). --no-data skips this;
#   7. puts the PyRoOrb commands on PATH: links them into ~/.local/bin, where uv installs itself.
#
# --uninstall first lists what it would remove and keep, then undoes what this script made: the
# command links in ~/.local/bin, .venv, and the reference tables and caches (--keep-data keeps
# those; they are also kept while another PyRoOrb install is on PATH). The PyRoOrb folder itself
# is offered only when this script cloned it and it holds nothing of yours -- no files, stashes,
# commits on no remote or other worktrees -- and only by a question of its own that --yes does
# not answer. It never removes uv, Python, the compiler or git.
#
# Options: --dir DIR  --extras LIST (default: none; e.g. itf, notebooks, or itf,dev)  --asteroids
#          --no-data  --yes (run system package installs, or the uninstall, without asking;
#          never deletes the PyRoOrb folder)
#          --uninstall  --keep-data  -h|--help
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
UNINSTALL=0
KEEP_DATA=0
CLONED=0
DIR_GIVEN=0; [ -n "${PYROORB_DIR:-}" ] && DIR_GIVEN=1
# The commands put on PATH (pyproject.toml [project.scripts]) and where their links go.
COMMANDS="pyroorb pyroorb-web pyroorb-itf pyroorb-fetch-data"
BIN="$HOME/.local/bin"

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir|--extras)
      [ $# -ge 2 ] || die "$1 needs a value, e.g.  --dir ~/PyRoOrb  or  --extras itf,dev  (see --help)"
      if [ "$1" = --dir ]; then DIR="$2"; DIR_GIVEN=1; else EXTRAS="$2"; fi
      shift 2 ;;
    --asteroids) ASTEROIDS=1; shift ;;
    --no-data) DATA=0; shift ;;
    --yes|-y) YES=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --keep-data) KEEP_DATA=1; shift ;;
    -h|--help)
      if [ -f "${BASH_SOURCE[0]:-}" ]; then awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
      else echo "options: --dir DIR --extras LIST --asteroids --no-data --yes --uninstall --keep-data  (see install.sh in the repository)"; fi
      exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

OS="$(uname -s)"
case "$OS" in
  Darwin|Linux) ;;
  MINGW*|MSYS*|CYGWIN*) die "this is the macOS and Linux installer; on Windows, run install.ps1 in PowerShell (see docs/INSTALL.md#windows)." ;;
  *) die "PyRoOrb's installers support macOS, Linux and Windows; $OS is not one of them." ;;
esac

is_checkout() { [ -f "$1/pyproject.toml" ] && grep -q '^name = "pyroorb"' "$1/pyproject.toml"; }
HERE=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# How to re-run with --yes: piped through curl, flags go after "bash -s --".
# What this script did that --uninstall must undo, kept in .venv: "cloned" when it downloaded
# the PyRoOrb folder. A later run from inside the folder (the update route) keeps the mark.
has_mark() { [ -f "$1/.venv/pyroorb-install" ] && grep -qx "$2" "$1/.venv/pyroorb-install"; }

# --- --uninstall -----------------------------------------------------------------------------
# Ask on the terminal: piped through curl, standard input is the script itself. With no terminal
# to ask on, the answer is no.
ask() {
  local ans=""
  printf '%s [y/N] ' "$1"
  read -r ans 2>/dev/null < /dev/tty || { echo; return 1; }
  case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

uninstall() {
  local root venv c link t links="" folder=0 why="" own="" cache state fitcache caches="" others=""
  # An explicit --dir (or PYROORB_DIR) is the install to remove, wherever this runs from: the
  # folder the script sits in or the current one could be a different PyRoOrb checkout.
  if [ "$DIR_GIVEN" = 1 ]; then
    is_checkout "$DIR" || die "no PyRoOrb install found in $DIR"
    root="$(cd "$DIR" && pwd)"
  elif [ -n "$HERE" ] && is_checkout "$HERE"; then root="$HERE"
  elif is_checkout "$PWD"; then root="$PWD"
  elif is_checkout "$DIR"; then root="$(cd "$DIR" && pwd)"
  else die "no PyRoOrb install found in $DIR (use --dir to name its folder)"; fi
  venv="$root/.venv"
  for c in $COMMANDS; do      # only links into THIS .venv; another install's stay
    link="$BIN/$c"
    [ -L "$link" ] || continue
    t="$(readlink "$link")"
    if [ "$t" = "$venv/bin/$c" ]; then links="$links${links:+
}$link"
    else case "$t" in */.venv/bin/"$c") others="$others${others:+
}$link -> $t" ;; esac; fi
  done
  # The folder itself is offered for deletion only when this script downloaded it and nothing
  # in it is the user's: no changed, new or ignored files (but the environment and build
  # leftovers), no commit or stash that is on no remote, no other worktree, and it is its own
  # repository (inside another one, git would answer for that one). Even then it is asked about
  # on its own, after the rest, and --yes does not answer that question.
  if ! has_mark "$root" cloned; then why="this script did not download it"
  elif ! t="$(git -C "$root" rev-parse --show-toplevel 2>/dev/null)"; then
    why="git could not check it for work of your own"
  elif [ "$(cd "$t" && pwd -P)" != "$(cd "$root" && pwd -P)" ]; then
    why="it is inside another git repository ($t)"
  elif ! own="$(git -C "$root" status --porcelain --ignored 2>/dev/null)"; then
    own=""; why="git could not check it for work of your own"
  else
    own="$(printf '%s\n' "$own" | sed 's/^...//' \
           | grep -Ev '(^|/)(\.venv|__pycache__|\.pytest_cache|[^/]*\.egg-info)/$' | grep . || true)"
    if [ -n "$own" ]; then why="it holds files of your own"
    elif [ -n "$(git -C "$root" stash list 2>/dev/null)" ]; then why="it holds stashed changes"
    elif ! t="$(git -C "$root" rev-list --all --not --remotes 2>/dev/null)"; then
      why="git could not check it for commits of your own"
    elif [ -n "$t" ]; then
      why="it holds $(printf '%s\n' "$t" | wc -l | tr -d ' ') commit(s) that are on no remote"
    elif [ "$(git -C "$root" worktree list --porcelain 2>/dev/null | grep -c '^worktree ')" != 1 ]; then
      why="other git worktrees use it"
    else folder=1; fi
  fi
  if [ "$OS" = Darwin ]; then
    cache="$HOME/Library/Caches/pyroorb"; state="$HOME/Library/Application Support/pyroorb"
  else
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/pyroorb"; state="${XDG_DATA_HOME:-$HOME/.local/share}/pyroorb"
  fi
  fitcache="${TMPDIR:-/tmp}"; fitcache="${fitcache%/}/pyroorb-fitcache"
  # The tables are shared by every PyRoOrb install of this user: kept while another one is on PATH.
  if [ "$KEEP_DATA" = 0 ] && [ -z "$others" ]; then
    [ -d "$cache" ] && caches="$cache"
    [ -d "$fitcache" ] && caches="$caches${caches:+
}$fitcache"
  fi

  say "Uninstalling PyRoOrb from $root"
  echo "This removes:"
  if [ -n "$links" ]; then printf '%s\n' "$links" | sed 's/^/  /'; fi
  if [ -d "$venv" ]; then echo "  $venv (the Python environment)"; fi
  if [ -n "$caches" ]; then
    printf '%s\n' "$caches" | while IFS= read -r c; do
      echo "  $c ($(du -sm "$c" 2>/dev/null | cut -f1) MB of downloaded tables and caches)"
    done
  fi
  if [ "$folder" = 1 ]; then
    echo "Then it asks, separately, whether to delete the whole folder $root ($(du -sm "$root" 2>/dev/null | cut -f1) MB),"
    echo "which this script downloaded and which holds nothing of yours."
  else
    echo "It keeps the folder $root ($why)${own:+:}"
    if [ -n "$own" ]; then printf '%s\n' "$own" | head -10 | sed 's/^/    /'; fi
  fi
  if [ -n "$others" ] && [ "$KEEP_DATA" = 0 ]; then
    echo "It keeps the downloaded tables ($cache): another PyRoOrb install uses them:"
    printf '%s\n' "$others" | sed 's/^/    /'
  fi
  if [ -n "${PYROORB_CACHE_DIR:-}" ]; then
    echo "It keeps PYROORB_CACHE_DIR ($PYROORB_CACHE_DIR); delete it yourself if you no longer need it."
  fi
  if [ -d "$state" ]; then
    echo "It keeps $state (PyRoOrb's saved state); look inside, and delete it yourself if you no longer need it."
  fi
  echo "It keeps uv, Python, the compiler and git, which other software can use."
  if [ "$YES" != 1 ] && ! ask "Remove these?"; then echo "Nothing removed."; return 0; fi

  if [ -n "$links" ]; then printf '%s\n' "$links" | while IFS= read -r link; do rm -f "$link"; done; fi
  rmdir "$BIN" 2>/dev/null || true              # only when empty
  rm -rf "$venv"
  if [ -n "$caches" ]; then printf '%s\n' "$caches" | while IFS= read -r c; do rm -rf "$c"; done; fi
  if [ "$folder" = 1 ]; then
    if ask "Also delete the whole folder $root? Everything in it goes, its git history too."; then
      case "$PWD/" in "$root"/*) cd "$HOME" ;; esac
      rm -rf "$root"
    else
      echo "Kept $root. To delete it yourself:  rm -rf '$root'"
    fi
  fi
  say "PyRoOrb is uninstalled."
}
if [ "$UNINSTALL" = 1 ]; then uninstall; exit 0; fi

if [ -n "$HERE" ]; then RERUN_YES="./install.sh --yes"
else RERUN_YES="curl -LsSf https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.sh | bash -s -- --yes"; fi

# The Linux package-manager command that installs the named packages, or empty when no
# known package manager is found. apt first refreshes its package lists: a fresh machine
# or image can have none, and the install then fails with "Unable to locate package".
# Root (a server, a container) needs no sudo.
linux_install_cmd() {   # $1 = apt packages, $2 = the package names elsewhere
  local s="sudo "; [ "$(id -u)" = 0 ] && s=""
  if   command -v apt-get >/dev/null; then echo "${s}apt-get update && ${s}apt-get install -y $1"
  elif command -v dnf     >/dev/null; then echo "${s}dnf install -y $2"
  elif command -v pacman  >/dev/null; then echo "${s}pacman -Sy --noconfirm $2"
  elif command -v zypper  >/dev/null; then echo "${s}zypper install -y $2"
  fi
}
# Install the missing Linux prerequisites in ONE command (one password prompt): run it with
# --yes, else print it and stop.
need_linux_pkgs() {     # $1 = what (for messages), $2 = apt packages, $3 = names elsewhere
  local cmd; cmd="$(linux_install_cmd "$2" "$3")"
  [ -n "$cmd" ] || die "$1 not installed; install with your package manager:  $3   then run this again"
  if [ "$YES" = 1 ]; then
    if [ "$(id -u)" != 0 ] && ! command -v sudo >/dev/null; then
      die "$1 not installed, and sudo is not available. As root, run:  ${cmd//sudo /}"
    fi
    say "Installing what is missing: $cmd"
    bash -c "$cmd" || die "the package install failed (see above)"
  else
    die "$1 not installed. Install with:
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
  # Everything missing, gathered into one install: a compiler always; git when PyRoOrb
  # must be downloaded; curl when uv must be installed.
  what=(); apt=(); other=()
  if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
    what+=("a C compiler"); apt+=(build-essential); other+=(gcc)
  fi
  if ! { [ -n "$HERE" ] && is_checkout "$HERE"; } && ! is_checkout "$PWD" && ! command -v git >/dev/null; then
    what+=(git); apt+=(git); other+=(git)
  fi
  if ! command -v uv >/dev/null && ! command -v curl >/dev/null; then
    what+=(curl); apt+=(curl ca-certificates); other+=(curl)
  fi
  if [ ${#what[@]} -gt 0 ]; then
    w="$(IFS=,; echo "${what[*]}")"; w="${w//,/, }"
    if [ ${#what[@]} -eq 1 ]; then w="$w is"; else w="$w are"; fi
    need_linux_pkgs "$w" "${apt[*]}" "${other[*]}"
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
    die "git is needed to download PyRoOrb; install it with your package manager"
  fi
  if is_checkout "$DIR"; then
    say "Updating the PyRoOrb checkout in $DIR"
    git -C "$DIR" pull --ff-only
  else
    say "Downloading PyRoOrb into $DIR"
    git clone --depth 1 "$REPO_URL" "$DIR"
    CLONED=1          # only a folder this run downloaded is marked; an update is the user's
  fi
  ROOT="$DIR"
fi
cd "$ROOT"
ROOT="$(pwd)"     # a relative --dir, made absolute: the command links name it
say "Installing PyRoOrb from $ROOT"

# --- 2. uv ---------------------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  say "Installing uv"
  command -v curl >/dev/null || die "curl is needed to install uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh \
    || die "could not download or run uv's installer from astral.sh; check your internet connection (or a firewall/proxy), or install uv yourself (https://docs.astral.sh/uv/) and run this again"
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
# ASSIST is compiled against REBOUND's C headers and runs against REBOUND's library, so the two
# must be the SAME release: REBOUND is installed first (an existing one is kept), and ASSIST is
# then built, and the rest resolved, against exactly that version. uv keeps the ASSIST wheel it
# builds and would reuse it in a later environment whatever REBOUND that has, so ASSIST's cache
# entry is dropped first (it compiles in a few seconds; an ASSIST already installed here is
# kept, and already matches).
uv pip install --python .venv/bin/python 'rebound>=4.4.11,<5'
REBOUND_VERSION="$(.venv/bin/python -c 'import rebound; print(rebound.__version__)')" \
  || die "REBOUND did not install; please open an issue with this output"
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
echo "rebound==$REBOUND_VERSION" > "$CONSTRAINTS"
TARGET="."
[ -n "$EXTRAS" ] && TARGET=".[$EXTRAS]"
say "Installing PyRoOrb ${EXTRAS:+with [$EXTRAS] }(compiles ASSIST against REBOUND $REBOUND_VERSION; about a minute)"
uv cache clean assist
uv pip install --python .venv/bin/python --constraint "$CONSTRAINTS" \
  --build-constraint "$CONSTRAINTS" -e "$TARGET"
.venv/bin/python -c "import pyroorb, rebound, assist" \
  || die "the install finished but PyRoOrb or ASSIST does not import; please open an issue with this output"
if [[ ",$EXTRAS," == *",itf,"* ]] && ! .venv/bin/python -c "import xgboost" >/dev/null 2>&1; then
  if [ "$OS" = Darwin ]; then
    warn "the ITF ranker's xgboost does not load (on macOS it needs: brew install libomp). Fitting and the web UI work without it."
  else
    warn "the ITF ranker's xgboost does not load (try: .venv/bin/python -c 'import xgboost'). Fitting and the web UI work without it."
  fi
fi

if [ "$CLONED" = 1 ] && ! has_mark "$ROOT" cloned; then echo cloned >> .venv/pyroorb-install; fi

# --- 6. reference tables -------------------------------------------------------------------
if [ "$DATA" = 1 ]; then
  say "Downloading the reference tables"
  if [ "$ASTEROIDS" = 1 ]; then .venv/bin/pyroorb-fetch-data --asteroids; else .venv/bin/pyroorb-fetch-data; fi
fi

# --- 7. the commands on PATH ----------------------------------------------------------------
mkdir -p "$BIN"
LINKED=0
for c in $COMMANDS; do
  [ -x ".venv/bin/$c" ] || continue
  # Replace an earlier PyRoOrb install's link; leave anything else of that name alone.
  if [ -e "$BIN/$c" ] || [ -L "$BIN/$c" ]; then
    case "$(readlink "$BIN/$c" 2>/dev/null || true)" in
      */.venv/bin/"$c") ;;
      *) warn "$BIN/$c already exists and is not a PyRoOrb link from this script; left in place"; continue ;;
    esac
  fi
  ln -sfn "$ROOT/.venv/bin/$c" "$BIN/$c"
  LINKED=1
done
case ":$PATH:" in *":$BIN:"*) ON_PATH=1 ;; *) ON_PATH=0 ;; esac
case "${SHELL##*/}" in zsh) RC="~/.zshrc" ;; *) RC="~/.bashrc" ;; esac

if [ "$LINKED" = 1 ] && [ "$ON_PATH" = 1 ]; then
cat <<EOF

PyRoOrb is installed in $ROOT

Open a new terminal; then, from any folder:
  pyroorb "2023 QJ3"          # fit an object from the command line
  pyroorb-web                 # the web UI at http://localhost:8000

Only for your own Python scripts and notebooks, switch to PyRoOrb's Python first:
  source "$ROOT/.venv/bin/activate"

To update later:  cd "$ROOT" && git pull && ./install.sh   (re-running picks up new dependencies)
To uninstall:     "$ROOT/install.sh" --uninstall
Settings (data folders, credentials) can live in one .env file; see docs/INSTALL.md.
EOF
else
if [ "$LINKED" = 1 ]; then
  warn "$BIN is not on your PATH, so the pyroorb commands there are not found by name. To fix that for good:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> $RC  and open a new terminal."
fi
cat <<EOF

PyRoOrb is installed in $ROOT/.venv

  cd "$ROOT"
  source .venv/bin/activate
  pyroorb "2023 QJ3"          # fit an object from the command line
  pyroorb-web                 # the web UI at http://localhost:8000

To update later:  cd "$ROOT" && git pull && ./install.sh   (re-running picks up new dependencies)
To uninstall:     "$ROOT/install.sh" --uninstall
Settings (data folders, credentials) can live in one .env file; see docs/INSTALL.md.
EOF
fi
}

main "$@"
