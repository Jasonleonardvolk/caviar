param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$QuickBuild,
  [switch]$OpenReport,
  [switch]$Parallel
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

# Paths
$ReportDir        = Join-Path $RepoRoot "tools\release\reports"
$PreflightMjs     = Join-Path $RepoRoot "tools\runtime\preflight.mjs"
$ShaderGatePs1    = Join-Path $RepoRoot "tools\shaders\run_shader_gate.ps1"
$ApiSmokeJs       = Join-Path $RepoRoot "tools\release\api-smoke.js"
# Use the non-interactive version if it exists, otherwise fall back to original
$IrisOneButtonNonInt = Join-Path $RepoRoot "tools\release\IrisOneButton_NonInteractive.ps1"
$IrisOneButtonOrig = Join-Path $RepoRoot "tools\release\IrisOneButton.ps1"
$IrisOneButtonPs1 = if (Test-Path $IrisOneButtonNonInt) { $IrisOneButtonNonInt } else { $IrisOneButtonOrig }
$ValidatorJs      = Join-Path $RepoRoot "tools\shaders\validate-wgsl.js"
$ValidatorMjs     = Join-Path $RepoRoot "tools\shaders\validate_and_report.mjs"
$DeviceLimitsDir  = Join-Path $RepoRoot "tools\shaders\device_limits"
$ReleasesDir      = Join-Path $RepoRoot "releases"
$TsconfigFront    = Join-Path $RepoRoot "frontend\tsconfig.json"

# Reports
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$tsStamp   = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$mdReport  = Join-Path $ReportDir ("iris_e2e_report_{0}.md" -f $tsStamp)
$jsonReport= Join-Path $ReportDir ("iris_e2e_report_{0}.json" -f $tsStamp)

# Results map
$results = [ordered]@{}

function Add-Result([string]$Name, [string]$Status, [int]$Exit, [string]$Log, [string]$Note="") {
  $results[$Name] = [ordered]@{
    status = $Status
    exit   = $Exit
    log    = $Log
    note   = $Note
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  }
  if ($Status -eq "PASS") { Write-Host ("PASS  " + $Name) -ForegroundColor Green }
  elseif ($Status -eq "SKIP") { Write-Host ("SKIP  " + $Name + " (" + $Note + ")") -ForegroundColor Yellow }
  else { Write-Host ("FAIL  " + $Name + " -> " + $Log) -ForegroundColor Red }
  Write-Host ("Exit Code: " + $Exit)
}

function Run-Logged([string]$StepName, [ScriptBlock]$Block, [int]$MaxRetries=1) {
  $safe = ($StepName -replace '[^\w\-]', '_')
  $log = Join-Path $ReportDir ("{0}_{1}.log" -f $safe, $tsStamp)
  $status = "PASS"; $exit = 0; $note = ""
  
  $attempt = 0
  while ($attempt -lt $MaxRetries) {
    $attempt++
    
    try {
      # Save current error action preference
      $oldErrorAction = $ErrorActionPreference
      $ErrorActionPreference = "Continue"
      
      if ($attempt -gt 1) {
        Write-Host ("Retry attempt {0} of {1} for {2}" -f $attempt, $MaxRetries, $StepName) -ForegroundColor Yellow
      }
      
      # Execute the script block and capture output
      & $Block 2>&1 | Tee-Object -FilePath $log | Out-Null
      
      # Get the actual exit code from the last command
      $exit = $LASTEXITCODE
      if ($null -eq $exit) { $exit = 0 }
      
      # Determine pass/fail based on exit code, not PowerShell errors
      if ($exit -ne 0) { 
        $status = "FAIL"
        if ($attempt -lt $MaxRetries) {
          Start-Sleep -Seconds 2
          continue
        }
      } else {
        $status = "PASS"
        break
      }
      
      # Restore error action preference
      $ErrorActionPreference = $oldErrorAction
      
    } catch {
      # Only catch actual exceptions, not stderr output
      $_ | Out-String | Tee-Object -FilePath $log | Out-Null
      $status = "FAIL"
      $exit = 1
      
      if ($attempt -lt $MaxRetries) {
        Start-Sleep -Seconds 2
        continue
      }
    }
  }
  
  if ($attempt -gt 1 -and $status -eq "PASS") {
    $note = "Succeeded on attempt $attempt"
  }
  
  Add-Result -Name $StepName -Status $status -Exit $exit -Log $log -Note $note
  return $exit
}

function Skip([string]$StepName, [string]$Reason) {
  $safe = ($StepName -replace '[^\w\-]', '_')
  $log = Join-Path $ReportDir ("{0}_{1}.log" -f $safe, $tsStamp)
  $Reason | Out-File -FilePath $log -Encoding UTF8
  Add-Result -Name $StepName -Status "SKIP" -Exit 0 -Log $log -Note $Reason
}

# Context capture
$commit  = ""
$branch  = ""
$dirtyTxt = ""
try { $commit = (& git rev-parse --short HEAD 2>$null).Trim() } catch {}
try { $branch = (& git rev-parse --abbrev-ref HEAD 2>$null).Trim() } catch {}
try { $dirtyTxt = (& git status --porcelain 2>$null) } catch {}
$dirty = $false
if ($dirtyTxt -ne $null -and $dirtyTxt -ne "") { $dirty = $true }

$node = ""; $npm = ""; $pnpm = "N/A"
try { $node = (& node -v 2>$null) } catch {}
try { $npm  = (& npm -v 2>$null) } catch {}
try { $pnpm = (& pnpm -v 2>$null) } catch {}

# Get system info
$os = [System.Environment]::OSVersion.VersionString
$cores = [System.Environment]::ProcessorCount

$context = [ordered]@{
  repoRoot = $RepoRoot
  commit   = $commit
  branch   = $branch
  dirty    = $dirty
  node     = $node
  npm      = $npm
  pnpm     = $pnpm
  os       = $os
  cores    = $cores
  timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

# Performance tracking
$startTime = Get-Date

# 01. Preflight
if (Test-Path $PreflightMjs) {
  Run-Logged "01_Preflight" { & node $PreflightMjs }
} else {
  Skip "01_Preflight" "Missing tools\runtime\preflight.mjs"
}

# Parallel execution support
if ($Parallel.IsPresent) {
  Write-Host "Running parallel steps..." -ForegroundColor Cyan
  
  $jobs = @()
  
  # Start TypeScript check job
  if (Test-Path $TsconfigFront) {
    $jobs += Start-Job -ScriptBlock {
      param($tsc, $log)
      & npx tsc -p $tsc --noEmit 2>&1 | Out-File -FilePath $log -Encoding UTF8
      return $LASTEXITCODE
    } -ArgumentList $TsconfigFront, (Join-Path $ReportDir ("02_TypeScript_{0}.log" -f $tsStamp))
  }
  
  # Start Shader validation job
  if (Test-Path $ShaderGatePs1) {
    $jobs += Start-Job -ScriptBlock {
      param($script, $root, $log)
      & powershell -ExecutionPolicy Bypass -File $script -RepoRoot $root -Strict 2>&1 | Out-File -FilePath $log -Encoding UTF8
      return $LASTEXITCODE
    } -ArgumentList $ShaderGatePs1, $RepoRoot, (Join-Path $ReportDir ("03_ShaderGate_{0}.log" -f $tsStamp))
  }
  
  # Wait for jobs and collect results
  $jobs | Wait-Job | ForEach-Object {
    $result = Receive-Job $_
    Remove-Job $_
    # Process results...
  }
  
} else {
  # Sequential execution (existing code)
  
  # 02. TypeScript check
  if (Test-Path $TsconfigFront) {
    Run-Logged "02_TypeScript" { & npx tsc -p $TsconfigFront --noEmit }
  } else {
    Run-Logged "02_TypeScript" { & npx tsc --noEmit }
  }
  
  # 03. Shader Gate
  if (Test-Path $ShaderGatePs1) {
    Run-Logged "03_ShaderGate_AllProfiles_Strict" {
      & powershell -ExecutionPolicy Bypass -File $ShaderGatePs1 -RepoRoot $RepoRoot -Strict
    }
  } else {
    $validator = $null
    if (Test-Path $ValidatorJs) { $validator = $ValidatorJs }
    elseif (Test-Path $ValidatorMjs) { $validator = $ValidatorMjs }
    
    if ($validator -ne $null) {
      $limits = Join-Path $DeviceLimitsDir "latest.json"
      if (-not (Test-Path $limits)) {
        $anyJson = Get-ChildItem $DeviceLimitsDir -Filter "*.json" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($anyJson) { $limits = $anyJson.FullName }
      }
      if (Test-Path $limits) {
        Run-Logged "03_ShaderGate_Fallback" { & node $validator --dir="frontend" --limits=$limits --strict }
      } else {
        Skip "03_ShaderGate" "No gate script and no device_limits JSON found"
      }
    } else {
      Skip "03_ShaderGate" "Missing run_shader_gate.ps1 and validators"
    }
  }
}

# 04. API Smoke (with retry for network issues)
if (Test-Path $ApiSmokeJs) {
  $prodEnv = Join-Path $RepoRoot ".env.production"
  Run-Logged "04_API_Smoke" { & node $ApiSmokeJs --env $prodEnv } -MaxRetries 3
} else {
  Skip "04_API_Smoke" "Missing tools\release\api-smoke.js"
}

# 05. Desktop profiles
$validator2 = $null
if (Test-Path $ValidatorJs) { $validator2 = $ValidatorJs }
elseif (Test-Path $ValidatorMjs) { $validator2 = $ValidatorMjs }

$desktopLow = Join-Path $DeviceLimitsDir "desktop_low.json"
$desktop    = Join-Path $DeviceLimitsDir "desktop.json"

if ((Test-Path $desktopLow) -and ($validator2 -ne $null)) {
  Run-Logged "05_DesktopLow_ShaderGate" { & node $validator2 --dir="frontend" --limits=$desktopLow --strict }
} else {
  Skip "05_DesktopLow_ShaderGate" "No desktop_low.json or no validator"
}

if ((Test-Path $desktop) -and ($validator2 -ne $null)) {
  Run-Logged "05_Desktop_ShaderGate" { & node $validator2 --dir="frontend" --limits=$desktop --strict }
} else {
  Skip "05_Desktop_ShaderGate" "No desktop.json or no validator"
}

# 06. Build and Package
if (Test-Path $IrisOneButtonPs1) {
  $args = @()
  # If using the NonInteractive version, it has the flag built-in
  # If using the original, add -NonInteractive only if it supports it
  if ($IrisOneButtonPs1 -like "*NonInteractive*") {
    $args += "-NonInteractive"
  }
  if ($QuickBuild.IsPresent) { $args += "-QuickBuild" }
  
  Write-Host ("   Using: {0}" -f (Split-Path $IrisOneButtonPs1 -Leaf)) -ForegroundColor Gray
  
  Run-Logged "06_Build_and_Package" { 
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & powershell -ExecutionPolicy Bypass -File $IrisOneButtonPs1 @args 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorAction
    exit $exitCode
  }
} else {
  Skip "06_Build_and_Package" "Missing IrisOneButton scripts"
}

# 07. Artifact verification
Run-Logged "07_Artifact_Verification" {
  if (-not (Test-Path $ReleasesDir)) { 
    Write-Host "No releases directory at $ReleasesDir" -ForegroundColor Yellow
    throw "No releases directory at $ReleasesDir" 
  }
  
  $rel = Get-ChildItem $ReleasesDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $rel) { 
    Write-Host "No releases found under $ReleasesDir" -ForegroundColor Yellow
    throw "No releases found under $ReleasesDir" 
  }
  
  $relPath = $rel.FullName
  $dist    = Join-Path $relPath "dist"
  $manifest= Join-Path $relPath "manifest.json"
  
  if (-not (Test-Path $dist)) { 
    Write-Host "Missing dist at $dist" -ForegroundColor Yellow
    throw "Missing dist at $dist" 
  }
  if (-not (Test-Path $manifest)) { 
    Write-Host "Missing manifest at $manifest" -ForegroundColor Yellow
    throw "Missing manifest at $manifest" 
  }
  
  # Generate checksums
  $hashFile = Join-Path $ReportDir ("hashes_{0}.sha256" -f $tsStamp)
  Get-ChildItem $dist -Recurse | Get-FileHash -Algorithm SHA256 | Out-File -FilePath $hashFile -Encoding UTF8
  
  # Count files and calculate size
  $fileCount = (Get-ChildItem $dist -Recurse -File).Count
  $totalSize = (Get-ChildItem $dist -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
  
  "releasePath=$relPath"  | Out-Host
  "distPath=$dist"        | Out-Host
  "manifest=$manifest"    | Out-Host
  "checksums=$hashFile"   | Out-Host
  "fileCount=$fileCount"  | Out-Host
  ("totalSize={0:N2} MB" -f $totalSize) | Out-Host
}

# Calculate elapsed time
$endTime = Get-Date
$elapsed = $endTime - $startTime

# Build summary
$anyFail = $false
$passCount = 0
$failCount = 0
$skipCount = 0

$rows = @()
foreach ($k in $results.Keys) {
  $row = $results[$k]
  if ($row.status -eq "FAIL") { 
    $anyFail = $true
    $failCount++
  } elseif ($row.status -eq "PASS") {
    $passCount++
  } else {
    $skipCount++
  }
  
  $rows += [PSCustomObject]@{
    step   = $k
    status = $row.status
    exit   = $row.exit
    log    = $row.log
    note   = $row.note
    timestamp = $row.timestamp
  }
}
$goNoGo = if ($anyFail) { "NO-GO" } else { "GO" }

# JSON report
$payload = [ordered]@{
  timestamp = $tsStamp
  context   = $context
  result    = $goNoGo
  elapsed   = $elapsed.ToString()
  summary   = [ordered]@{
    passed  = $passCount
    failed  = $failCount
    skipped = $skipCount
  }
  steps     = $rows
}
$payload | ConvertTo-Json -Depth 6 | Out-File -FilePath $jsonReport -Encoding UTF8

# Markdown report (ASCII only - no backticks)
$md = @()
$md += ("# IRIS End-to-End Verification - {0}" -f $tsStamp)
$md += ""
$md += "## Context"
$md += ("- Repo: {0}" -f $RepoRoot)
$md += ("- Branch: {0}" -f $context.branch)
$md += ("- Commit: {0}" -f $context.commit)
$md += ("- Dirty working tree: {0}" -f $context.dirty)
$md += ("- Node/npm/pnpm: {0} / {1} / {2}" -f $context.node, $context.npm, $context.pnpm)
$md += ("- OS: {0}" -f $context.os)
$md += ("- CPU Cores: {0}" -f $context.cores)
$md += ""
$md += "## Summary"
$md += ("- Total Time: {0}" -f $elapsed.ToString())
$md += ("- Passed: {0}" -f $passCount)
$md += ("- Failed: {0}" -f $failCount)
$md += ("- Skipped: {0}" -f $skipCount)
$md += ""
$md += "## Steps"
foreach ($r in $rows) {
  $logName = Split-Path $r.log -Leaf
  $md += ("- **{0}**: {1} - Log: {2}" -f $r.step, $r.status, $logName)
  if ($r.note) { $md += ("  - Note: {0}" -f $r.note) }
  if ($r.timestamp) { $md += ("  - Time: {0}" -f $r.timestamp) }
}
$md += ""
$md += ("## RESULT: {0}" -f $goNoGo)
if ($anyFail) {
  $md += ""
  $md += "## Action Required"
  $md += "The build has failed verification. Please review the failed steps above and their log files."
  $md += "Run AutoFixForShipping.ps1 for automated fixes, or review individual logs for manual fixes."
}
$md -join "`r`n" | Out-File -FilePath $mdReport -Encoding UTF8

# Safer color selection
$fg = "Green"; if ($anyFail) { $fg = "Red" }
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ("RESULT: " + $goNoGo) -ForegroundColor $fg
Write-Host ("Elapsed Time: " + $elapsed.ToString()) -ForegroundColor Cyan
Write-Host ("Pass/Fail/Skip: {0}/{1}/{2}" -f $passCount, $failCount, $skipCount) -ForegroundColor Cyan
Write-Host ("Markdown: " + $mdReport) -ForegroundColor Cyan
Write-Host ("JSON:     " + $jsonReport) -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($OpenReport.IsPresent) { Start-Process $mdReport }
if ($anyFail) { exit 1 } else { exit 0 }
