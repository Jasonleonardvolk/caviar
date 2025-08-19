# Test-WowPack-E2E.ps1
# Location: D:\Dev\kha\tools\encode\Test-WowPack-E2E.ps1
#
# One-button smoke test for the WOW Pack pipeline:
#  1) (Optional) Generate ProRes 10-bit test masters from existing MP4s
#  2) Check-ProRes-Masters.ps1 (strict 10-bit ProRes enforcement)
#  3) Batch-Encode-Simple.ps1 -> HDR10 / AV1 / SDR outputs
#  4) Verify-WowPack.ps1 -> asserts HDR atoms & codecs
#  5) Start-Services-Now.ps1 -> backend + frontend
#  6) Opens /hologram?clip=<detected or specified>

[CmdletBinding()]
param(
  [switch]$SkipGenerate,
  [string]$Clip
)

$ErrorActionPreference = "Stop"

# ----- Paths
$here = Split-Path -Parent $PSCommandPath                   # ...\tools\encode
$tools = Split-Path -Parent $here                           # ...\tools
$repo  = Split-Path -Parent $tools                          # D:\Dev\kha

$encodeDir   = Join-Path $repo "tools\encode"
$releaseDir  = Join-Path $repo "tools\release"
$inputDir    = Join-Path $repo "content\wowpack\input"
$staticWow   = Join-Path $repo "tori_ui_svelte\static\media\wow"
$startSvc    = Join-Path $repo "Start-Services-Now.ps1"
$startShow   = Join-Path $repo "Start-Show.ps1"

function Run-Step([string]$name, [scriptblock]$body) {
  Write-Host "==> $name" -ForegroundColor Cyan
  & $body
  Write-Host "OK: $name" -ForegroundColor Green
  Write-Host ""
}

# ----- Preflight: required script presence
$required = @(
  (Join-Path $encodeDir "Check-ProRes-Masters.ps1"),
  (Join-Path $encodeDir "Batch-Encode-Simple.ps1"),
  (Join-Path $releaseDir "Verify-WowPack.ps1")
)
if (-not $SkipGenerate) { 
  $required += (Join-Path $encodeDir "Generate-Test-ProRes.ps1") 
}
foreach ($f in $required) { 
  if (-not (Test-Path $f)) { 
    throw "Missing required script: $f" 
  } 
}
if (-not (Test-Path $inputDir)) { 
  throw "Missing input folder: $inputDir" 
}
if (-not (Test-Path $staticWow)) { 
  New-Item -ItemType Directory -Path $staticWow -Force | Out-Null 
}

# ----- 1) Generate 10-bit ProRes masters (optional)
if (-not $SkipGenerate) {
  Run-Step "Generate ProRes test masters in $inputDir" {
    Push-Location $encodeDir
    & .\Generate-Test-ProRes.ps1
    Pop-Location
  }
}

# ----- 2) Strict preflight (must be ProRes 10-bit)
Run-Step "Preflight masters in $inputDir" {
  Push-Location $encodeDir
  & .\Check-ProRes-Masters.ps1
  Pop-Location
}

# ----- 3) Encode all variants
Run-Step "Encode WowPack to $staticWow" {
  Push-Location $encodeDir
  & .\Batch-Encode-Simple.ps1
  Pop-Location
}

# ----- 4) Verify outputs
Run-Step "Verify outputs via Verify-WowPack.ps1" {
  Push-Location $releaseDir
  & .\Verify-WowPack.ps1
  Pop-Location
}

# ----- 5) Start services (API + frontend)
Run-Step "Start services with Start-Services-Now.ps1" {
  if (Test-Path $startSvc) {
    & $startSvc
  } else {
    Write-Host "Starting services manually..." -ForegroundColor Yellow
    
    # Check if already running
    $api8002 = Get-NetTCPConnection -LocalPort 8002 -ErrorAction SilentlyContinue
    $frontend3000 = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
    
    if (-not $api8002) {
      Write-Host "  Starting backend API on port 8002..." -ForegroundColor Gray
      Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$repo'; python enhanced_launcher.py --api-mode"
      Start-Sleep -Seconds 3
    } else {
      Write-Host "  Backend API already running on 8002" -ForegroundColor Green
    }
    
    if (-not $frontend3000) {
      Write-Host "  Starting frontend on port 3000..." -ForegroundColor Gray
      Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$repo\tori_ui_svelte'; npm run dev"
      Start-Sleep -Seconds 5
    } else {
      Write-Host "  Frontend already running on 3000" -ForegroundColor Green
    }
  }
}

# ----- 6) Open show (clip detection or override)
if (-not $Clip) {
  $candidates = @("holo_flux_loop","mach_lightfield","kinetic_logo_parade")
  foreach ($b in $candidates) {
    $hasHdr = Test-Path (Join-Path $staticWow "${b}_hdr10.mp4")
    $hasAv1 = Test-Path (Join-Path $staticWow "${b}_av1.mp4")
    $hasSdr = Test-Path (Join-Path $staticWow "${b}_sdr.mp4")
    
    if ($hasHdr -or $hasAv1 -or $hasSdr) { 
      $Clip = $b
      break 
    }
  }
  
  if (-not $Clip) {
    $first = Get-ChildItem $staticWow -Filter "*_hdr10.mp4" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($first) { 
      $Clip = ($first.BaseName -replace "_hdr10$","")
    }
  }
  
  if (-not $Clip) { 
    $Clip = "holo_flux_loop"  # final fallback
  }
}

Run-Step "Open viewer /hologram?clip=$Clip" {
  if (Test-Path $startShow) { 
    & $startShow -Clip $Clip | Out-Null 
  }
  Start-Sleep -Seconds 2
  $url = "http://localhost:3000/hologram?clip=$Clip"
  Write-Host "  Opening: $url" -ForegroundColor Yellow
  Start-Process $url
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "     E2E Pipeline Complete!                            " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Inputs   : $inputDir" -ForegroundColor Yellow
Write-Host "Outputs  : $staticWow" -ForegroundColor Yellow
Write-Host "Viewer   : http://localhost:3000/hologram?clip=$Clip" -ForegroundColor Yellow
Write-Host ""
Write-Host "Available clips:" -ForegroundColor Cyan
Write-Host "  ?clip=holo_flux_loop" -ForegroundColor Gray
Write-Host "  ?clip=mach_lightfield" -ForegroundColor Gray
Write-Host "  ?clip=kinetic_logo_parade" -ForegroundColor Gray
