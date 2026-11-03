# PyRoOrb installer for Windows (64-bit). install.sh is the macOS and Linux one.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1      from a checkout: installs into .\.venv
#   irm https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.ps1 | iex
#                                 clones into ~\PyRoOrb first (-Dir to choose), then installs
#
# What it does, in order (each step is skipped when already satisfied):
#   1. checks for the MSVC C compiler (the Visual Studio Build Tools) -- ASSIST, the ephemeris
#      integrator, is compiled during the install -- and for git; it prints, or with -Yes runs,
#      the winget command that installs what is missing;
#   2. installs uv (https://docs.astral.sh/uv/) when it is missing;
#   3. creates .venv with a uv-managed Python 3.12: REBOUND 4 publishes Windows wheels for
#      CPython 3.10-3.12 only, and PyRoOrb needs 3.11 or later;
#   4. installs REBOUND 4, then ASSIST with Windows support built against exactly that REBOUND,
#      then PyRoOrb. The ASSIST release on PyPI does not build on Windows yet; until it does,
#      ASSIST comes from the fork named in $AssistSource below;
#   5. downloads the reference tables (DE440, the debias tables; -Asteroids adds the 616 MB
#      asteroid ephemeris). -NoData skips this.
#
# Options: -Dir DIR  -Extras LIST (default: none; e.g. itf, notebooks, or itf,dev)  -Asteroids
#          -NoData  -Yes (run the winget installs without asking)  -Help
# Through irm, options need the script block form:
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.ps1))) -Yes
[CmdletBinding()]
param(
    [string]$Dir = $(if ($env:PYROORB_DIR) { $env:PYROORB_DIR } else { Join-Path $HOME "PyRoOrb" }),
    [string]$Extras = "",
    [switch]$Asteroids,
    [switch]$NoData,
    [switch]$Yes,
    [switch]$Help
)
$ErrorActionPreference = "Stop"

$RepoUrl = if ($env:PYROORB_REPO_URL) { $env:PYROORB_REPO_URL } else { "https://github.com/petervanwylen/PyRoOrb.git" }
$PythonVersion = "3.12"
# ASSIST with native Windows support, pinned to a commit. Replace with the plain "assist>=1.2,<1.3"
# once an ASSIST release on PyPI builds on Windows. Keep in step with .github/workflows/ci.yml.
$AssistSource = "assist @ git+https://github.com/petervanwylen/assist-win-support@20d12ef89b3c97e09dfb70f11d59c0d840b52eee"

function Say($msg)  { Write-Host "==> $msg" -ForegroundColor White }
function Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }
# Stop: exit from a script file; through irm | iex an exit would close the user's window, so throw.
function Die($msg)  {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    if ($PSCommandPath) { exit 1 } else { throw "PyRoOrb install stopped" }
}
function Run {      # a native command; stop on a non-zero exit, which PowerShell 5.1 ignores
    $exe, $rest = $args
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { Die "$exe $($rest -join ' ') failed (exit code $LASTEXITCODE; see above)" }
}

if ($Help) {
    if ($PSCommandPath) { Get-Content $PSCommandPath -TotalCount 22 | ForEach-Object { $_ -replace '^# ?', '' } }
    else { "options: -Dir DIR -Extras LIST -Asteroids -NoData -Yes  (see install.ps1 in the repository)" }
    return
}
if (-not [Environment]::Is64BitOperatingSystem) { Die "PyRoOrb on Windows needs 64-bit Windows." }

function Test-Checkout($path) {
    $toml = Join-Path $path "pyproject.toml"
    (Test-Path $toml) -and (Select-String -Path $toml -Pattern '^name = "pyroorb"' -Quiet)
}
# How to re-run with -Yes: from a file, or through irm.
$RerunYes = if ($PSCommandPath) { "powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Yes" }
            else { "& ([scriptblock]::Create((irm https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.ps1))) -Yes" }

# --- 1. the C compiler and git ------------------------------------------------------------
# setuptools finds MSVC through vswhere on its own, so no Developer Command Prompt is needed;
# what must exist is the VC++ toolset itself.
function Test-Msvc {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return [bool]$found
}
$Here = if ($PSScriptRoot -and (Test-Checkout $PSScriptRoot)) { $PSScriptRoot } else { "" }
$needGit = -not $Here -and -not (Test-Checkout $PWD.Path) -and -not (Get-Command git -ErrorAction SilentlyContinue)
$missing = @()
if (-not (Test-Msvc)) {
    $missing += @{ what = "the Visual Studio Build Tools (the C compiler)";
                   cmd = 'winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"' }
}
if ($needGit) { $missing += @{ what = "git"; cmd = "winget install --id Git.Git -e" } }
if ($missing.Count -gt 0) {
    $what = ($missing | ForEach-Object { $_.what }) -join " and "
    if ($Yes) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Die "$what not installed, and winget is not available. Install by hand, then run this again:`n    $(($missing | ForEach-Object { $_.cmd }) -join "`n    ")"
        }
        foreach ($m in $missing) {
            Say "Installing $($m.what): $($m.cmd)"
            Invoke-Expression $m.cmd
            if ($LASTEXITCODE -ne 0) { Die "the install of $($m.what) failed (see above)" }
        }
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not (Test-Msvc)) { Die "the Build Tools installed, but their C++ compiler was not found; run this again in a new terminal" }
    } else {
        Die "$what not installed. Install with:`n    $(($missing | ForEach-Object { $_.cmd }) -join "`n    ")`nthen run this again (or let the script do it:  $RerunYes)"
    }
}

# --- 0. the checkout -----------------------------------------------------------------------
if ($Here) {
    $Root = $Here
} elseif (Test-Checkout $PWD.Path) {
    $Root = $PWD.Path
} else {
    if (Test-Checkout $Dir) {
        Say "Updating the PyRoOrb checkout in $Dir"
        Run git -C $Dir pull --ff-only
    } else {
        Say "Downloading PyRoOrb into $Dir"
        Run git clone --depth 1 $RepoUrl $Dir
    }
    $Root = $Dir
}
Set-Location $Root
Say "Installing PyRoOrb from $Root"

# --- 2. uv ---------------------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv"
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Die "could not download or run uv's installer from astral.sh; check your internet connection (or a firewall/proxy), or install uv yourself (https://docs.astral.sh/uv/) and run this again"
    }
    $env:Path = (Join-Path $HOME ".local\bin") + ";" + $env:Path
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Die "uv was installed but is not on PATH; open a new terminal and run this again" }
}

# --- 3 + 4. the environment and the install ------------------------------------------------
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Say "Creating .venv with Python $PythonVersion"
    & uv venv --managed-python --python $PythonVersion .venv
    if ($LASTEXITCODE -ne 0) {
        # Seen with uv 0.13 on Windows: the Python installs, but uv then cannot follow its own
        # cpython-3.12-... junction ("Missing expected target directory for Python minor version
        # link"). Name the installed interpreter by its full path instead.
        $pyDir = (& uv python dir).Trim()
        $exact = Get-ChildItem $pyDir -Directory -Filter "cpython-$PythonVersion.*-windows-x86_64-none" -ErrorAction SilentlyContinue |
                 Where-Object { Test-Path (Join-Path $_.FullName "python.exe") } |
                 Sort-Object { [version]($_.Name -replace '^cpython-([\d.]+)-.*$', '$1') } | Select-Object -Last 1
        if (-not $exact) { Die "uv could not create .venv with Python $PythonVersion (see above)" }
        Warn "uv could not use its Python $PythonVersion link; using $($exact.FullName)\python.exe directly"
        Run uv venv --python (Join-Path $exact.FullName "python.exe") .venv
    }
}
# ASSIST is compiled against REBOUND's C headers and runs against REBOUND's library, so the two
# must be the SAME release: REBOUND first (an existing one is kept), then ASSIST built, and the
# rest resolved, against exactly that version. ASSIST's uv cache entry is dropped first so a
# wheel built against another REBOUND is never reused (install.sh does the same).
Run uv pip install --python $Py "rebound>=4.4.11,<5"
$ReboundVersion = & $Py -c "import rebound; print(rebound.__version__)"
if ($LASTEXITCODE -ne 0) { Die "REBOUND did not install; please open an issue with this output" }
$Constraints = New-TemporaryFile
try {
    Set-Content -Path $Constraints -Value "rebound==$ReboundVersion" -Encoding ascii
    $Target = if ($Extras) { ".[$Extras]" } else { "." }
    Say "Installing PyRoOrb $(if ($Extras) { "with [$Extras] " })(compiles ASSIST against REBOUND $ReboundVersion; about a minute)"
    Run uv cache clean assist
    Run uv pip install --python $Py --constraint $Constraints --build-constraint $Constraints $AssistSource -e $Target
} finally {
    Remove-Item $Constraints -ErrorAction SilentlyContinue
}
& $Py -c "import pyroorb, rebound, assist"
if ($LASTEXITCODE -ne 0) { Die "the install finished but PyRoOrb or ASSIST does not import; please open an issue with this output" }
if ((",$Extras," -like "*,itf,*")) {
    & $Py -c "import xgboost" 2>$null
    if ($LASTEXITCODE -ne 0) { Warn "the ITF ranker's xgboost does not load (try: .venv\Scripts\python -c 'import xgboost'). Fitting and the web UI work without it." }
}

# --- 5. reference tables -------------------------------------------------------------------
if (-not $NoData) {
    Say "Downloading the reference tables"
    $fetch = Join-Path $Root ".venv\Scripts\pyroorb-fetch-data.exe"
    if ($Asteroids) { Run $fetch --asteroids } else { Run $fetch }
}

@"

PyRoOrb is installed in $Root\.venv

  cd "$Root"
  .venv\Scripts\activate
  pyroorb "2023 QJ3"          # fit an object from the command line
  pyroorb-web                 # the web UI at http://localhost:8000

To update later:  cd "$Root"; git pull; powershell -ExecutionPolicy Bypass -File install.ps1
Settings (data folders, credentials) can live in one .env file; see docs/INSTALL.md.
"@
