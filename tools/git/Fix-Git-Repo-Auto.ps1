Param(
  [string]$RepoRoot = "D:\Dev\kha",
  [string]$FeatureBranch = "feat/wowpack-prores-hdr10-pipeline",
  [switch]$RelocateGitDir # (advanced) not used by default
)

# ===== Utilities =====
function Ensure-Dir { param($p) if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null } }
function Line-In-File {
  param([string]$Path,[string]$Line)
  if (!(Test-Path $Path)) { New-Item -ItemType File -Path $Path -Force | Out-Null }
  $exists = Select-String -Path $Path -Pattern ([regex]::Escape($Line)) -SimpleMatch -Quiet
  if (-not $exists) { Add-Content -Path $Path -Value $Line }
}
function Robo-CopyTree {
  param([string]$src, [string]$dst, [string[]]$xd, [string[]]$xf, [string]$logPath)
  Ensure-Dir $dst
  $args = @($src, $dst, "/E", "/R:1", "/W:1", "/MT:16", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
  if ($xd -and $xd.Count) { $args += "/XD"; $args += $xd }
  if ($xf -and $xf.Count) { $args += "/XF"; $args += $xf }
  if ($logPath) { $args += "/LOG+:$logPath" }
  & robocopy @args | Out-Null
  $code = $LASTEXITCODE
  # Robocopy: 0-7 = success/warnings, >=8 = failure
  if ($code -ge 8) { throw "Robocopy failed with code $code" }
}

# ===== Setup =====
$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$VerifyDir = Join-Path $RepoRoot "verification_reports"
Ensure-Dir $VerifyDir
$Log = Join-Path $VerifyDir "git_autofix_$ts.log"
$Json = Join-Path $VerifyDir "git_autofix_last.json"

Start-Transcript -Path $Log -Append | Out-Null
Write-Host "=== GIT AUTO-FIX ($ts) ==="
Write-Host "RepoRoot: $RepoRoot"
if (-not (Test-Path (Join-Path $RepoRoot ".git"))) { throw "No .git at $RepoRoot" }

Push-Location $RepoRoot
$summary = [ordered]@{
  timestamp = $ts
  repoRoot = $RepoRoot
  mode = "unknown"
  origin = ""
  inPlaceRepair = $false
  reclone = $false
  branch = $FeatureBranch
  result = "unknown"
  notes = @()
}

# Harden against Drive/GC races during repair
try { git config --global gc.auto 0 | Out-Null } catch {}
try { git config --global gc.autoDetach false | Out-Null } catch {}
try { git config --global maintenance.auto false | Out-Null } catch {}
try { git maintenance stop 2>$null | Out-Null } catch {}
$originUrl = (& git config --get remote.origin.url) 2>$null
if (-not $originUrl) { throw "Missing remote.origin.url - cannot re-fetch/re-clone." }
$summary.origin = $originUrl

# Helper: run fsck and decide health
function Test-GitHealth {
  $out = (& git fsck --full 2>&1)
  $bad = ($out -match 'missing ' -or $out -match 'error:' -or $out -match 'fatal:')
  return @{ Bad = $bad; Output = $out }
}

# 1) IN-PLACE REPAIR
$summary.mode = "in-place"
Write-Host "`n[1/5] In-place verification: git fsck --full"
$h1 = Test-GitHealth
$summary.notes += $h1.Output -split "`n" | Where-Object { $_ }

if ($h1.Bad) {
  Write-Host "[2/5] Force re-fetch packs: git fetch origin --prune --tags --force"
  & git fetch origin --prune --tags --force
  $h2 = Test-GitHealth
  $summary.notes += $h2.Output -split "`n" | Where-Object { $_ }

  if ($h2.Bad) {
    Write-Host "[3/5] Backup all .git\objects\pack and re-fetch"
    $pack = Join-Path $RepoRoot ".git\objects\pack"
    $save = Join-Path $pack "_save_$ts"
    Ensure-Dir $save
    Get-ChildItem "$pack\pack-*.pack","$pack\pack-*.idx" -ErrorAction SilentlyContinue | ForEach-Object {
      Move-Item $_.FullName $save -Force
    }
    & git fetch origin --prune --tags --force
    $h3 = Test-GitHealth
    $summary.notes += $h3.Output -split "`n" | Where-Object { $_ }

    if (-not $h3.Bad) {
      $summary.inPlaceRepair = $true
    }
  } else {
    $summary.inPlaceRepair = $true
  }
} else {
  $summary.inPlaceRepair = $true
}

if ($summary.inPlaceRepair) {
  Write-Host "[4/5] In-place looks good. Normalizing EOL + setting branch."
  & git switch -C $FeatureBranch
  & git config core.autocrlf false
  & git config core.eol lf
  & git add --renormalize .
  # Enforce .gitignore media rules (idempotent)
  $gi = Join-Path $RepoRoot ".gitignore"
  "content/wowpack/input/"                  | ForEach-Object { Line-In-File -Path $gi -Line $_ }
  "content/wowpack/video/"                  | ForEach-Object { Line-In-File -Path $gi -Line $_ }
  "tori_ui_svelte/static/media/wow/*.mp4"   | ForEach-Object { Line-In-File -Path $gi -Line $_ }
  "!tori_ui_svelte/static/media/wow/wow.manifest.json" | ForEach-Object { Line-In-File -Path $gi -Line $_ }
  "tori_ui_svelte/static/media/hls/"        | ForEach-Object { Line-In-File -Path $gi -Line $_ }
  & git add .gitignore

  # Commit if there's anything to commit
  $status = (& git status --porcelain) 2>$null
  if ($status) {
    & git commit -m "chore(git): auto-repair packs, normalize EOL, enforce .gitignore (auto-fix $ts)"
  }
  # Try to set upstream and pull fast-forward (won't create merge commits)
  try { & git branch --set-upstream-to=origin/$FeatureBranch 2>$null | Out-Null } catch {}
  & git pull --ff-only 2>$null | Out-Null
  # Push branch (create if missing)
  & git push -u origin $FeatureBranch
  $summary.result = "OK (in-place)"
  $summary.mode   = "in-place:success"
  Pop-Location
  $summary | ConvertTo-Json -Depth 5 | Set-Content $Json
  Stop-Transcript | Out-Null
  exit 0
}

# 2) RECLONE PATH (fallback)
$summary.mode = "reclone"
$summary.reclone = $true
$SnapRoot  = "D:\Dev\kha.WORKTREE_SNAPSHOT_$ts"
$CleanRoot = "D:\Dev\kha.CLEAN_$ts"
Write-Host "`n[5/5] In-place failed. Proceeding with snapshot + clean clone."
Write-Host "Snapshot: $SnapRoot"
Write-Host "Clean clone: $CleanRoot"

# Snapshot working tree (exclude heavy/build/media)
$excludeDirs = @(
  (Join-Path $RepoRoot ".git"),
  (Join-Path $RepoRoot "node_modules"),
  (Join-Path $RepoRoot "tori_ui_svelte\.svelte-kit"),
  (Join-Path $RepoRoot "tori_ui_svelte\build"),
  (Join-Path $RepoRoot "tori_ui_svelte\dist"),
  (Join-Path $RepoRoot "tori_ui_svelte\static\media\wow"),
  (Join-Path $RepoRoot "tori_ui_svelte\static\media\hls"),
  (Join-Path $RepoRoot "content\wowpack\video"),
  (Join-Path $RepoRoot "content\wowpack\input")
)
$excludeFiles = @("*.mp4","*.mov","*.m4s","*.mkv","*.ts")
Robo-CopyTree -src $RepoRoot -dst $SnapRoot -xd $excludeDirs -xf $excludeFiles -logPath $Log

# Clean clone
& git clone $originUrl $CleanRoot
if ($LASTEXITCODE -ne 0) { throw "git clone failed." }
Push-Location $CleanRoot
& git switch -c $FeatureBranch

# Copy snapshot into clean clone (same excludes)
$xd2 = @(
  (Join-Path $SnapRoot ".git"),
  (Join-Path $SnapRoot "node_modules"),
  (Join-Path $SnapRoot "tori_ui_svelte\.svelte-kit"),
  (Join-Path $SnapRoot "tori_ui_svelte\build"),
  (Join-Path $SnapRoot "tori_ui_svelte\dist"),
  (Join-Path $SnapRoot "tori_ui_svelte\static\media\wow"),
  (Join-Path $SnapRoot "tori_ui_svelte\static\media\hls"),
  (Join-Path $SnapRoot "content\wowpack\video"),
  (Join-Path $SnapRoot "content\wowpack\input")
)
Robo-CopyTree -src $SnapRoot -dst $CleanRoot -xd $xd2 -xf $excludeFiles -logPath $Log

# Normalize EOL + .gitignore enforcement
& git config core.autocrlf false
& git config core.eol lf
& git add --renormalize .
$gi2 = Join-Path $CleanRoot ".gitignore"
"content/wowpack/input/"                  | ForEach-Object { Line-In-File -Path $gi2 -Line $_ }
"content/wowpack/video/"                  | ForEach-Object { Line-In-File -Path $gi2 -Line $_ }
"tori_ui_svelte/static/media/wow/*.mp4"   | ForEach-Object { Line-In-File -Path $gi2 -Line $_ }
"!tori_ui_svelte/static/media/wow/wow.manifest.json" | ForEach-Object { Line-In-File -Path $gi2 -Line $_ }
"tori_ui_svelte/static/media/hls/"        | ForEach-Object { Line-In-File -Path $gi2 -Line $_ }
& git add .gitignore

# Commit/push if changes exist
$status2 = (& git status --porcelain) 2>$null
if ($status2) {
  & git commit -m "feat(wowpack): ProRes->HDR10/AV1/SDR pipeline + preflight/verify + runtime source selection (rescue $ts)"
}
& git push -u origin $FeatureBranch

Pop-Location
$summary.result = "OK (reclone)"
$summary | ConvertTo-Json -Depth 5 | Set-Content $Json
Stop-Transcript | Out-Null
Write-Host "`nDONE. Clean clone at: $CleanRoot"
Write-Host "Snapshot kept at: $SnapRoot"