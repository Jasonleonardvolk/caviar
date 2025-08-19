param(
  [string]$RepoRoot = "D:\Dev\kha",
  [string[]]$Targets = @(),  # Empty = auto-detect all available
  [switch]$Strict = $true
)
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

# 0) Ensure paths
$limitsDir  = Join-Path $RepoRoot "tools\shaders\device_limits"
$reportsDir = Join-Path $RepoRoot "tools\shaders\reports"
$logsDir    = Join-Path $RepoRoot "tools\release\error_logs"
New-Item -ItemType Directory -Force -Path $reportsDir,$logsDir | Out-Null

# Auto-detect targets if not specified
if ($Targets.Count -eq 0) {
  Write-Host "[shader-gate] Auto-detecting device profiles..."
  $deviceFiles = Get-ChildItem -Path $limitsDir -Filter "*.json" -ErrorAction SilentlyContinue
  
  if ($deviceFiles) {
    foreach ($file in $deviceFiles) {
      $targetName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
      $Targets += $targetName
      Write-Host "  Found: $targetName"
    }
  } else {
    # Fallback to common targets
    Write-Host "  No profiles found, using defaults: iphone11, iphone15"
    $Targets = @("iphone11", "iphone15")
  }
}

Write-Host "[shader-gate] Testing against: $($Targets -join ', ')"

# 1) Optional Tint (for MSL/HLSL/Spir-V cross checks)
$tintBin = Join-Path $RepoRoot "tools\shaders\bin\tint.exe"
$useTint = Test-Path $tintBin
if ($useTint) { $env:PATH = "$([System.IO.Path]::GetDirectoryName($tintBin));$env:PATH" }
Write-Host ("[shader-gate] Tint: " + ($(if ($useTint) {"FOUND"} else {"MISSING (Naga-only)"})))

# 2) Pick validator script (new vs legacy)
$valNew = Join-Path $RepoRoot "tools\shaders\validate-wgsl.js"
$valOld = Join-Path $RepoRoot "tools\shaders\validate_and_report.mjs"
$validator = $(if (Test-Path $valNew) { $valNew } elseif (Test-Path $valOld) { $valOld } else { "" })
if (-not $validator) { Write-Host "ERROR: validator script not found under tools\shaders\"; exit 2 }

# 3) Run per device-limit target
$ts = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$overallFail = $false
foreach ($t in $Targets) {
  $limits = Join-Path $limitsDir "$t.json"
  if (-not (Test-Path $limits)) { 
    Write-Host "WARN: missing $limits (skipping)"
    continue 
  }
  
  $label = $t
  $out = Join-Path $reportsDir "shader_validation_${t}_$ts"
  $args = @("--dir=frontend", "--limits=$limits", "--label=$label", "--out=$reportsDir")
  if ($Strict) { $args += "--strict" }
  if ($useTint) { $args += "--tint" }
  
  Write-Host "[shader-gate] node $validator $($args -join ' ')"
  & node $validator @args
  $exit = $LASTEXITCODE
  if ($exit -ne 0) { $overallFail = $true }
  
  # Normalize report filenames (JSON, JUnit, txt summary expected)
  Get-ChildItem $reportsDir -Filter "*$label*.{json,xml,txt}" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  report -> $($_.FullName)"
  }
}

if ($overallFail) {
  Write-Host "[shader-gate] FAIL: one or more targets failed"
  exit 2
} else {
  Write-Host "[shader-gate] PASS: all targets passed (or only warnings)"
  exit 0
}
