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
#      asteroid ephemeris). -NoData skips this;
#   6. puts the PyRoOrb commands on PATH: copies them into ~\.local\bin (where uv installs
#      itself) and adds that folder to your user PATH when it is not there yet.
#
# -Uninstall undoes what this script made: the commands in ~\.local\bin (and the PATH entry,
# when this script added it and nothing else is left there), .venv, the downloaded PyRoOrb
# folder when this script cloned it and it holds no files of yours, and the reference tables
# and caches (-KeepData keeps those). It never removes uv, Python, the Build Tools or git.
#
# Options: -Dir DIR  -Extras LIST (default: none; e.g. itf, notebooks, or itf,dev)  -Asteroids
#          -NoData  -Yes (run the winget installs, or the uninstall, without asking)
#          -Uninstall  -KeepData  -Help
# Through irm, options need the script block form:
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/petervanwylen/PyRoOrb/main/install.ps1))) -Yes
[CmdletBinding()]
param(
    [string]$Dir = $(if ($env:PYROORB_DIR) { $env:PYROORB_DIR } else { Join-Path $HOME "PyRoOrb" }),
    [string]$Extras = "",
    [switch]$Asteroids,
    [switch]$NoData,
    [switch]$Yes,
    [switch]$Uninstall,
    [switch]$KeepData,
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
    if ($PSCommandPath) { foreach ($l in Get-Content $PSCommandPath) { if ($l -notmatch '^#') { break }; $l -replace '^# ?', '' } }
    else { "options: -Dir DIR -Extras LIST -Asteroids -NoData -Yes -Uninstall -KeepData  (see install.ps1 in the repository)" }
    return
}
if (-not [Environment]::Is64BitOperatingSystem) { Die "PyRoOrb on Windows needs 64-bit Windows." }

function Test-Checkout($path) {
    $toml = Join-Path $path "pyproject.toml"
    (Test-Path $toml) -and (Select-String -Path $toml -Pattern '^name = "pyroorb"' -Quiet)
}
$Here = if ($PSScriptRoot -and (Test-Checkout $PSScriptRoot)) { $PSScriptRoot } else { "" }

# The commands put on PATH (pyproject.toml [project.scripts]) and where they go. A copy of a
# .venv launcher runs from anywhere: it names its .venv's python.exe by full path, and that path
# is also how the uninstall tells its own copies from another install's.
$Commands = @("pyroorb", "pyroorb-web", "pyroorb-itf", "pyroorb-fetch-data")
$Bin = Join-Path $HOME ".local\bin"
function Test-Launcher($file, $venvPy) {     # a .venv launcher; with $venvPy, that .venv's
    if (-not (Test-Path $file)) { return $false }
    $text = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($file))
    if ($venvPy) { return $text.IndexOf($venvPy, [StringComparison]::OrdinalIgnoreCase) -ge 0 }
    return $text -match '\\\.venv\\Scripts\\python\.exe'
}
# The user PATH is read and written raw: [Environment]::SetEnvironmentVariable would store it as
# a plain string and expand every %VARIABLE% in it for good.
function Get-UserPath { (Get-Item HKCU:\Environment).GetValue("Path", "", "DoNotExpandEnvironmentNames") }
function Set-UserPath($value) {
    Set-ItemProperty HKCU:\Environment -Name Path -Value $value -Type ExpandString
    # Writing any user variable announces the change, so terminals opened from now on see it.
    [Environment]::SetEnvironmentVariable("PYROORB_PATH_CHANGED", $null, "User")
}
function Test-OnPath($folder) {
    foreach ($scope in "User", "Machine") {
        foreach ($p in ([Environment]::GetEnvironmentVariable("Path", $scope) -split ";")) {
            if ($p -and [Environment]::ExpandEnvironmentVariables($p).TrimEnd("\") -ieq $folder.TrimEnd("\")) { return $true }
        }
    }
    return $false
}
# What this script did that the uninstall must undo, one word per line, kept in .venv:
# "cloned" (it downloaded the PyRoOrb folder) and "path-added" (it put ~\.local\bin on PATH).
function Test-Mark($root, $word) {
    $m = Join-Path $root ".venv\pyroorb-install"
    (Test-Path $m) -and [bool](Select-String -Path $m -Pattern "^$word$" -Quiet)
}
function Add-Mark($root, $word) {
    if (-not (Test-Mark $root $word)) { Add-Content -Path (Join-Path $root ".venv\pyroorb-install") -Value $word -Encoding ascii }
}
function Get-Busy($root) {     # PyRoOrb processes running from this .venv or its PATH copies
    $venvPy = Join-Path $root ".venv\Scripts\python.exe"
    $scripts = Join-Path $root ".venv\Scripts"
    $copies = @($Commands | ForEach-Object { Join-Path $Bin "$_.exe" } | Where-Object { Test-Launcher $_ $venvPy })
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and ($_.Path.StartsWith($scripts, [StringComparison]::OrdinalIgnoreCase) -or $copies -contains $_.Path) }
}
function Get-SizeMB($path) {
    $sum = (Get-ChildItem $path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    [math]::Round($sum / 1MB)
}

# --- -Uninstall ----------------------------------------------------------------------------
if ($Uninstall) {
    $Root = if ($Here) { $Here } elseif (Test-Checkout $PWD.Path) { $PWD.Path }
            elseif (Test-Checkout $Dir) { (Resolve-Path $Dir).Path } else { "" }
    if (-not $Root) { Die "no PyRoOrb install found in $Dir (use -Dir to name its folder)" }
    $venv = Join-Path $Root ".venv"
    $venvPy = Join-Path $venv "Scripts\python.exe"
    $busy = Get-Busy $Root
    if ($busy) { Die "PyRoOrb is running ($(($busy | ForEach-Object { $_.Name } | Sort-Object -Unique) -join ', ')); close it, then run this again" }

    $copies = @($Commands | ForEach-Object { Join-Path $Bin "$_.exe" } | Where-Object { Test-Launcher $_ $venvPy })
    $pathAdded = (Test-Mark $Root "path-added")
    $othersInBin = @(Get-ChildItem $Bin -Force -ErrorAction SilentlyContinue | Where-Object { $copies -notcontains $_.FullName })
    $dropPath = $pathAdded -and $othersInBin.Count -eq 0
    # The whole folder goes only when this script downloaded it and nothing in it is yours:
    # git lists changed, new and ignored files, all but the environment and build leftovers.
    $whole = $false; $why = ""; $own = @()
    if (-not (Test-Mark $Root "cloned")) { $why = "this script did not download it" }
    else {
        $status = & git -C $Root status --porcelain --ignored 2>$null
        if ($LASTEXITCODE -ne 0) { $why = "git could not check it for files of your own" }
        else {
            $own = @($status | ForEach-Object { $_.Substring(3) } |
                     Where-Object { $_ -notmatch '(^|/)(\.venv|__pycache__|\.pytest_cache|[^/]*\.egg-info)/$' })
            if ($own.Count -eq 0) { $whole = $true } else { $why = "it holds files of your own" }
        }
    }
    $caches = @()
    if (-not $KeepData) {
        $caches = @((Join-Path $env:LOCALAPPDATA "pyroorb"), (Join-Path $env:TEMP "pyroorb-fitcache")) | Where-Object { Test-Path $_ }
    }
    $state = Join-Path $env:APPDATA "pyroorb"

    Say "Uninstalling PyRoOrb from $Root"
    "This removes:"
    foreach ($c in $copies) { "  $c" }
    if ($dropPath) { "  $Bin from your PATH (this script added it)" }
    if ($whole) { "  $Root (the PyRoOrb folder this script downloaded)" }
    elseif (Test-Path $venv) { "  $venv" }
    foreach ($c in $caches) { "  $c ($(Get-SizeMB $c) MB of downloaded tables and caches)" }
    if (-not $whole) {
        "It keeps $Root ($why)$(if ($own) { ':' })"
        foreach ($f in ($own | Select-Object -First 10)) { "    $f" }
    }
    if ($env:PYROORB_CACHE_DIR) { "It keeps PYROORB_CACHE_DIR ($env:PYROORB_CACHE_DIR); delete it yourself if you no longer need it." }
    if (Test-Path $state) { "It keeps $state (PyRoOrb's saved state); look inside, and delete it yourself if you no longer need it." }
    "It keeps uv, Python, the Build Tools and git, which other software can use."
    if (-not $Yes) {
        try { $answer = Read-Host "Remove these? [y/N]" } catch { $answer = "" }   # no console: no
        if ($answer -notmatch '^(y|yes)$') { "Nothing removed."; return }
    }

    $failed = @()
    foreach ($c in $copies) { try { Remove-Item $c -Force } catch { $failed += $c } }
    if ($dropPath) {
        $kept = @((Get-UserPath) -split ";" | Where-Object { $_ -and [Environment]::ExpandEnvironmentVariables($_).TrimEnd("\") -ine $Bin.TrimEnd("\") })
        Set-UserPath ($kept -join ";")
        Remove-Item $Bin -ErrorAction SilentlyContinue     # only when empty
    }
    # Windows will not delete the folder a process is in: step out of it first, PowerShell's
    # location and the process's own working folder both (Set-Location moves only the first).
    if ($PWD.Path.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) { Set-Location $HOME }
    [Environment]::CurrentDirectory = $HOME
    $gone = if ($whole) { $Root } else { $venv }
    $emptyLeft = $false
    if (Test-Path $gone) {
        try { Remove-Item $gone -Recurse -Force }
        catch {
            # Everything inside went, but a terminal whose current folder is this one (the update
            # route leaves people there) keeps the empty folder itself.
            if (-not (Get-ChildItem $gone -Force -ErrorAction SilentlyContinue)) { $emptyLeft = $true }
            else { $failed += $gone }
        }
    }
    foreach ($c in $caches) { try { Remove-Item $c -Recurse -Force } catch { $failed += $c } }
    if ($emptyLeft) {
        Warn "$gone is empty now, but a terminal is still in it; delete the folder once you have left it (cd ~)"
    }
    if ($failed) {
        Warn "could not remove: $($failed -join ', '). Close any program or terminal using them (a terminal whose current folder is inside), then delete them."
        if ($PSCommandPath) { exit 1 } else { return }
    }
    Say "PyRoOrb is uninstalled."
    return
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
    $Cloned = $true
}
Set-Location $Root
$Root = $PWD.Path      # a relative -Dir, made absolute: the commands on PATH name it
Say "Installing PyRoOrb from $Root"

# --- 2. uv ---------------------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv"
    # uv's own documented command, in a process of its own: run inside this script it inherits
    # $ErrorActionPreference = "Stop", which turns the installer's non-fatal errors into fatal ones.
    & powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) {
        Die "could not download or run uv's installer from astral.sh (see above); check your internet connection (or a firewall/proxy), or install uv yourself (https://docs.astral.sh/uv/) and run this again"
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
# Windows will not replace a running .exe: a re-install while pyroorb-web (or any other PyRoOrb
# command from this .venv) runs fails half-way, so say so before touching anything.
$scripts = Join-Path $Root ".venv\Scripts"
$busy = Get-Busy $Root
if ($busy) {
    Die "PyRoOrb is running from this .venv ($(($busy | ForEach-Object { $_.Name } | Sort-Object -Unique) -join ', ')); close it, then run this again"
}
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

# Recorded for -Uninstall. A later run from inside the folder (the update route) keeps the mark.
if ($Cloned) { Add-Mark $Root "cloned" }

# --- 5. reference tables -------------------------------------------------------------------
if (-not $NoData) {
    Say "Downloading the reference tables"
    $fetch = Join-Path $Root ".venv\Scripts\pyroorb-fetch-data.exe"
    if ($Asteroids) { Run $fetch --asteroids } else { Run $fetch }
}

# --- 6. the commands on PATH ----------------------------------------------------------------
New-Item -ItemType Directory -Force $Bin | Out-Null
$onPath = $false
foreach ($c in $Commands) {
    $from = Join-Path $scripts "$c.exe"
    $to = Join-Path $Bin "$c.exe"
    if (-not (Test-Path $from)) { continue }
    # Replace an earlier PyRoOrb install's copy; leave anything else of that name alone.
    if ((Test-Path $to) -and -not (Test-Launcher $to $null)) {
        Warn "$to already exists and is not a PyRoOrb command from this script; left in place"
        continue
    }
    Copy-Item $from $to -Force
    $onPath = $true
}
if ($onPath -and -not (Test-OnPath $Bin)) {
    Say "Adding $Bin to your PATH"
    $current = (Get-UserPath).TrimEnd(";")
    Set-UserPath ($(if ($current) { "$current;" } else { "" }) + $Bin)
    Add-Mark $Root "path-added"
}
# This window too: through irm | iex this is the user's own session.
if ($onPath -and -not (($env:Path -split ";") -contains $Bin)) { $env:Path = "$Bin;$env:Path" }

$UninstallCmd = "powershell -ExecutionPolicy Bypass -File `"$Root\install.ps1`" -Uninstall"
if ($onPath) {
@"

PyRoOrb is installed in $Root

Open a new terminal; then, from any folder:
  pyroorb "2023 QJ3"          # fit an object from the command line
  pyroorb-web                 # the web UI at http://localhost:8000

Only for your own Python scripts and notebooks, switch to PyRoOrb's Python first:
  & "$Root\.venv\Scripts\Activate.ps1"
(PowerShell may first need:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

To update later:  cd "$Root"; git pull; powershell -ExecutionPolicy Bypass -File install.ps1
To uninstall:     $UninstallCmd
Settings (data folders, credentials) can live in one .env file; see docs/INSTALL.md.
"@
} else {
@"

PyRoOrb is installed in $Root\.venv

  cd "$Root"
  .venv\Scriptsctivate
  pyroorb "2023 QJ3"          # fit an object from the command line
  pyroorb-web                 # the web UI at http://localhost:8000

To update later:  cd "$Root"; git pull; powershell -ExecutionPolicy Bypass -File install.ps1
To uninstall:     $UninstallCmd
Settings (data folders, credentials) can live in one .env file; see docs/INSTALL.md.
"@
}
