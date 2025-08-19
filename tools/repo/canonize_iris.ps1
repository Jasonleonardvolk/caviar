<#  canonize_iris.ps1
    Canonicalize PFS app name to "iris" with OS-level junctions (no code changes).
    - Renames:  <repo>\standalone-holo  ->  <repo>\iris   (if iris not already present)
    - Junction: <repo>\standalone-holo  ->  <repo>\iris   (back-compat)
    - Routes:   <repo>\frontend\app\routes\iris      -> well-played (junction)
                <repo>\frontend\app\routes\renderer  -> well-played (junction)
    - Optional: append "standalone-holo/" to <repo>\.gitignore (skip with -NoGitIgnore)

    Usage:
      pwsh -File .\tools\repo\canonize_iris.ps1 -RepoRoot "C:\Users\jason\Desktop\tori\kha" -Force
      pwsh -File .\tools\repo\canonize_iris.ps1 -DryRun     # preview only

    Notes:
      - Safe to run multiple times; it's idempotent.
      - Requires Windows (junctions). Run PowerShell (pwsh) with normal user; admin not required.
#>

[CmdletBinding()]
param(
  [string]$RepoRoot,
  [switch]$Force,
  [switch]$DryRun,
  [switch]$NoGitIgnore
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok  ([string]$msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Info([string]$msg) { Write-Host "* $msg" -ForegroundColor Gray }
function Write-Warn([string]$msg) { Write-Host "! $msg" -ForegroundColor Yellow }
function Write-Err ([string]$msg) { Write-Host "[X] $msg" -ForegroundColor Red }

function Resolve-RepoRoot {
  param([string]$Path)
  if ([string]::IsNullOrWhiteSpace($Path)) {
    # Default: two levels up from this script (tools\repo\ -> repo root)
    $Path = Join-Path -Path $PSScriptRoot -ChildPath '..\..'
  }
  return (Resolve-Path -LiteralPath $Path).Path
}

function Test-IsJunction {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return $false }
  try {
    $attr = (Get-Item -LiteralPath $Path -Force).Attributes
    return ($attr -band [IO.FileAttributes]::ReparsePoint) -ne 0
  } catch { return $false }
}

function New-DirJunction {
  param(
    [Parameter(Mandatory)] [string]$LinkPath,
    [Parameter(Mandatory)] [string]$TargetPath
  )
  if (Test-Path -LiteralPath $LinkPath) {
    if (Test-IsJunction $LinkPath) {
      Write-Info "Junction already exists: $LinkPath"
      return
    } else {
      throw "Cannot create junction. A real directory/file exists at: $LinkPath"
    }
  }
  if (-not (Test-Path -LiteralPath $TargetPath)) {
    throw "Target for junction does not exist: $TargetPath"
  }

  if ($DryRun) {
    Write-Info "[DRYRUN] mklink /J '$LinkPath' '$TargetPath'"
    return
  }

  try {
    # Prefer PowerShell junction API if available
    New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
  } catch {
    # Fallback to cmd
    $cmdStr = "mklink /J `"$LinkPath`" `"$TargetPath`""
    cmd /c $cmdStr | Out-Null
  }
  Write-Ok "Created junction: $LinkPath  ->  $TargetPath"
}

# --- Resolve paths
$RepoRoot = Resolve-RepoRoot -Path $RepoRoot
Write-Step "Repo root: $RepoRoot"

$stdPath        = Join-Path $RepoRoot 'standalone-holo'
$irisPath       = Join-Path $RepoRoot 'iris'
$routesRoot     = Join-Path $RepoRoot 'frontend\app\routes'
$wellPlayedPath = Join-Path $routesRoot 'well-played'
$irisRoute      = Join-Path $routesRoot 'iris'
$rendererRoute  = Join-Path $routesRoot 'renderer'
$gitignorePath  = Join-Path $RepoRoot '.gitignore'

Write-Info "PFS old path : $stdPath"
Write-Info "PFS new path : $irisPath"
Write-Info "Routes root  : $routesRoot"

# --- Sanity pre-scan (optional ripgrep)
try {
  $rg = Get-Command rg -ErrorAction SilentlyContinue
  if ($rg) {
    Write-Step "Pre-scan for references (ripgrep)"
    & $rg -n --hidden -S --glob '!**/dist/**' --glob '!**/node_modules/**' 'standalone-holo|well-played' $RepoRoot
  } else {
    Write-Info "ripgrep (rg) not found; skipping pre-scan."
  }
} catch { Write-Warn "Pre-scan skipped: $($_.Exception.Message)" }

# --- Confirm (unless -Force)
if (-not $Force -and -not $DryRun) {
  $resp = Read-Host "Proceed with rename + junction creation? (y/N)"
  if ($resp -notin @('y','Y')) { Write-Warn "Aborted by user."; exit 1 }
}

# --- Step 1: Rename standalone-holo -> iris (only if needed)
if ((Test-Path -LiteralPath $stdPath) -and -not (Test-Path -LiteralPath $irisPath)) {
  Write-Step "Renaming PFS directory: standalone-holo -> iris"
  if ($DryRun) {
    Write-Info "[DRYRUN] Rename-Item -LiteralPath '$stdPath' -NewName 'iris'"
  } else {
    Rename-Item -LiteralPath $stdPath -NewName 'iris'
    Write-Ok "Renamed to: $irisPath"
  }
} elseif (Test-Path -LiteralPath $irisPath) {
  Write-Info "PFS directory already canonicalized at: $irisPath"
} elseif (-not (Test-Path -LiteralPath $stdPath)) {
  Write-Warn "No standalone-holo directory found; will only create route junctions if possible."
}

# --- Step 2: Ensure back-compat junction standalone-holo -> iris
if (-not (Test-IsJunction $stdPath)) {
  if (Test-Path -LiteralPath $stdPath) {
    # Real dir still exists (wasn't renamed) -> move aside safely
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $backup = "${stdPath}.backup_$stamp"
    if ($DryRun) {
      Write-Info "[DRYRUN] Move-Item '$stdPath' '$backup'"
    } else {
      Move-Item -LiteralPath $stdPath -Destination $backup
      Write-Ok "Backed up original folder to: $backup"
    }
  }
  if (Test-Path -LiteralPath $irisPath) {
    Write-Step "Creating back-compat junction: standalone-holo -> iris"
    New-DirJunction -LinkPath $stdPath -TargetPath $irisPath
  }
} else {
  Write-Info "Back-compat junction already present: $stdPath"
}

# --- Step 3: Route aliases via junctions (iris, renderer -> well-played)
if (Test-Path -LiteralPath $wellPlayedPath) {
  Write-Step "Route alias junctions (no code changes)"
  if (-not (Test-IsJunction $irisRoute)) {
    New-DirJunction -LinkPath $irisRoute -TargetPath $wellPlayedPath
  } else { Write-Info "Route /iris already aliased." }

  if (-not (Test-IsJunction $rendererRoute)) {
    New-DirJunction -LinkPath $rendererRoute -TargetPath $wellPlayedPath
  } else { Write-Info "Route /renderer already aliased." }
} else {
  Write-Warn "Route source not found: $wellPlayedPath. Skipping route aliases."
}

# --- Step 4: .gitignore back-compat (optional)
if (-not $NoGitIgnore) {
  Write-Step "Ensuring .gitignore ignores standalone-holo/ (junction)"
  $entry = 'standalone-holo/'
  try {
    if ($DryRun) {
      Write-Info "[DRYRUN] Append '$entry' to $gitignorePath if missing"
    } else {
      if (Test-Path -LiteralPath $gitignorePath) {
        $content = Get-Content -LiteralPath $gitignorePath -ErrorAction SilentlyContinue
        if ($content -notcontains $entry) {
          Add-Content -LiteralPath $gitignorePath -Value $entry
          Write-Ok "Appended to .gitignore: $entry"
        } else {
          Write-Info ".gitignore already contains: $entry"
        }
      } else {
        Set-Content -LiteralPath $gitignorePath -Value $entry
        Write-Ok "Created .gitignore with: $entry"
      }
    }
  } catch { Write-Warn "Could not update .gitignore: $($_.Exception.Message)" }
} else {
  Write-Info "Skipping .gitignore update per -NoGitIgnore"
}

# --- Step 5: Verify invariants (apps untouched)
Write-Step "Verifying apps folder integrity"
$pcDonor     = Join-Path $RepoRoot 'frontend\apps\pc_donor'
$phoneClient = Join-Path $RepoRoot 'frontend\apps\phone_client'
if ((Test-Path -LiteralPath $pcDonor -PathType Container) -and
    (Test-Path -LiteralPath $phoneClient -PathType Container)) {
  Write-Ok "apps\pc_donor and apps\phone_client present and untouched."
} else {
  Write-Warn "apps\pc_donor or apps\phone_client not found. (This script does not modify them.)"
}

Write-Step "Done."
Write-Info "PFS canonical path       : $irisPath"
Write-Info "Back-compat junction     : $stdPath  -> iris"
Write-Info "Route alias (browser)    : /iris, /renderer  -> /well-played"
Write-Info "Next: run PFS dev: 'iris\start.bat'  and Svelte dev: 'frontend\... dev', open /iris and /renderer"