param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$QuickBuild,
  [switch]$OpenReport,
  [switch]$Parallel,
  [switch]$SkipPrompts  # Force skip any interactive prompts
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

# Paths
$ReportDir        = Join-Path $RepoRoot "tools\release\reports"
$PreflightMjs     = Join-Path $RepoRoot "tools\runtime\preflight.mjs"
$ShaderGatePs1    = Join-Path $RepoRoot "tools\shaders\run_shader_gate.ps1"
$ApiSmokeJs       = Join-Path $RepoRoot "tools\release\api-smoke.js"
$IrisOneButtonPs1 = Join-Path $RepoRoot "tools\release\IrisOneButton.ps1"
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
$diagReport = Join-Path $ReportDir ("iris_e2e_diagnostic_{0}.txt" -f $tsStamp)

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
      
      # Determine pass/fail based on exit code
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
      # Only catch actual exceptions
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

# Write diagnostic header
$diag = @()
$diag += "IRIS E2E Verification Diagnostic Report"
$diag += "========================================"
$diag += "Started: " + $startTime.ToString("yyyy-MM-dd HH:mm:ss")
$diag += ""
$diag += "Context:"
$diag += "  Repo: $RepoRoot"
$diag += "  Branch: $branch"
$diag += "  Commit: $commit"
$diag += "  Dirty: $dirty"
$diag += ""

# 01. Preflight
if (Test-Path $PreflightMjs) {
  Run-Logged "01_Preflight" { & node $PreflightMjs }
} else {
  Skip "01_Preflight" "Missing tools\runtime\preflight.mjs"
}

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

# 06. Build and Package - FIXED to handle prompts
Write-Host "`nStep 06: Build and Package" -ForegroundColor Yellow
if (Test-Path $IrisOneButtonPs1) {
  $buildLog = Join-Path $ReportDir ("06_Build_and_Package_{0}.log" -f $tsStamp)
  $buildExit = 0
  
  try {
    # Create a script block that automatically answers 'N' to any prompts
    $buildScript = @"
`$ErrorActionPreference = 'Continue'
Set-Location '$RepoRoot'

# Prepare arguments
`$args = @()
$(if ($QuickBuild) { "`$args += '-QuickBuild'" })

# Run IrisOneButton with automatic 'N' response to prompts
`$process = Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File '$IrisOneButtonPs1' `$args" -NoNewWindow -PassThru -RedirectStandardOutput '$buildLog.out' -RedirectStandardError '$buildLog.err'

# Wait for process but send 'N' if it's waiting for input
`$timeout = 300  # 5 minutes timeout
`$elapsed = 0
while (-not `$process.HasExited -and `$elapsed -lt `$timeout) {
    Start-Sleep -Seconds 1
    `$elapsed++
    
    # Check if process is waiting (CPU usage near 0)
    if (`$elapsed -gt 10 -and `$process.CPU -lt 1) {
        # Likely waiting for input, send N
        Add-Type -TypeDefinition @'
        using System;
        using System.Runtime.InteropServices;
        public class Win32 {
            [DllImport("user32.dll")]
            public static extern bool SetForegroundWindow(IntPtr hWnd);
        }
'@
        [System.Windows.Forms.SendKeys]::SendWait('n{ENTER}')
        break
    }
}

# Wait for completion
`$process.WaitForExit()
exit `$process.ExitCode
"@

    # Execute the build script
    $buildResult = & powershell -Command $buildScript 2>&1
    $buildExit = $LASTEXITCODE
    
    # Combine output logs
    if (Test-Path "$buildLog.out") {
      Get-Content "$buildLog.out" | Out-File $buildLog -Encoding UTF8
      Remove-Item "$buildLog.out" -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path "$buildLog.err") {
      Get-Content "$buildLog.err" | Out-File $buildLog -Append -Encoding UTF8
      Remove-Item "$buildLog.err" -Force -ErrorAction SilentlyContinue
    }
    
    # If still no log, write the result
    if (-not (Test-Path $buildLog)) {
      $buildResult | Out-String | Out-File $buildLog -Encoding UTF8
    }
    
    if ($buildExit -eq 0) {
      Add-Result -Name "06_Build_and_Package" -Status "PASS" -Exit 0 -Log $buildLog
    } else {
      Add-Result -Name "06_Build_and_Package" -Status "FAIL" -Exit $buildExit -Log $buildLog
    }
    
  } catch {
    $_.Exception.Message | Out-File $buildLog -Append -Encoding UTF8
    Add-Result -Name "06_Build_and_Package" -Status "FAIL" -Exit 1 -Log $buildLog -Note "Exception during build"
  }
} else {
  Skip "06_Build_and_Package" "Missing tools\release\IrisOneButton.ps1"
}

# 07. Artifact verification
Write-Host "`nStep 07: Artifact Verification" -ForegroundColor Yellow
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
  
  Write-Host "Checking release: $relPath" -ForegroundColor Gray
  
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
  Get-ChildItem $dist -Recurse -File | Get-FileHash -Algorithm SHA256 | Out-File -FilePath $hashFile -Encoding UTF8
  
  # Count files and calculate size
  $files = Get-ChildItem $dist -Recurse -File
  $fileCount = $files.Count
  $totalSize = ($files | Measure-Object -Property Length -Sum).Sum / 1MB
  
  "releasePath=$relPath"  | Out-Host
  "distPath=$dist"        | Out-Host
  "manifest=$manifest"    | Out-Host
  "checksums=$hashFile"   | Out-Host
  "fileCount=$fileCount"  | Out-Host
  ("totalSize={0:N2} MB" -f $totalSize) | Out-Host
  
  # Add to diagnostic
  $script:diag += ""
  $script:diag += "Artifact Details:"
  $script:diag += "  Release: $relPath"
  $script:diag += "  Files: $fileCount"
  $script:diag += ("  Size: {0:N2} MB" -f $totalSize)
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
    
    # Add failure details to diagnostic
    $diag += ""
    $diag += "FAILED: $k"
    $diag += "  Exit Code: " + $row.exit
    $diag += "  Log: " + $row.log
    if (Test-Path $row.log) {
      $diag += "  Last 10 lines:"
      Get-Content $row.log -Tail 10 | ForEach-Object { $diag += "    $_" }
    }
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

# Complete diagnostic report
$diag += ""
$diag += "Summary:"
$diag += "  Result: $goNoGo"
$diag += "  Passed: $passCount"
$diag += "  Failed: $failCount"
$diag += "  Skipped: $skipCount"
$diag += "  Duration: " + $elapsed.ToString()
$diag += ""
$diag += "Completed: " + $endTime.ToString("yyyy-MM-dd HH:mm:ss")
$diag -join "`r`n" | Out-File -FilePath $diagReport -Encoding UTF8

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

# Markdown report (ASCII only)
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
  $md += ("### {0}: {1}" -f $r.step, $r.status)
  $md += ("- Exit Code: {0}" -f $r.exit)
  $md += ("- Log: {0}" -f $logName)
  if ($r.note) { $md += ("- Note: {0}" -f $r.note) }
  if ($r.timestamp) { $md += ("- Time: {0}" -f $r.timestamp) }
  
  # Add failure analysis for failed steps
  if ($r.status -eq "FAIL" -and (Test-Path $r.log)) {
    $md += ""
    $md += "#### Error Summary:"
    $errorLines = Get-Content $r.log -Tail 20 | Where-Object { $_ -match "error|fail|exception" -or $_ -match "^\s*\+" }
    if ($errorLines) {
      foreach ($line in $errorLines) {
        $md += ("    {0}" -f $line)
      }
    } else {
      $md += "    Check full log for details"
    }
  }
  $md += ""
}
$md += ("## RESULT: {0}" -f $goNoGo)
if ($anyFail) {
  $md += ""
  $md += "## Action Required"
  $md += "The build has failed verification. Please:"
  $md += "1. Review the failed steps above and their error summaries"
  $md += "2. Check the diagnostic report: " + (Split-Path $diagReport -Leaf)
  $md += "3. Run AutoFixForShipping.ps1 for automated fixes"
  $md += "4. Or review individual logs for manual fixes"
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
Write-Host ("Diagnostic: " + $diagReport) -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Show failure details if any
if ($anyFail) {
  Write-Host ""
  Write-Host "Failed Steps:" -ForegroundColor Red
  foreach ($k in $results.Keys) {
    if ($results[$k].status -eq "FAIL") {
      Write-Host ("  - {0} (Exit: {1})" -f $k, $results[$k].exit) -ForegroundColor Red
    }
  }
  Write-Host ""
  Write-Host "See diagnostic report for details: $diagReport" -ForegroundColor Yellow
}

if ($OpenReport.IsPresent) { Start-Process $mdReport }
if ($anyFail) { exit 1 } else { exit 0 }
