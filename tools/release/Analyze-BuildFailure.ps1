param(
  [string]$ReportDir = "D:\Dev\kha\tools\release\reports",
  [string]$LogFile = ""  # Optional specific log file to analyze
)

Write-Host "IRIS Build Failure Diagnostic Analyzer" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# Find latest report if not specified
if ($LogFile -eq "") {
  $latestReport = Get-ChildItem $ReportDir -Filter "iris_e2e_report_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latestReport) {
    $LogFile = $latestReport.FullName
    Write-Host "Analyzing latest report: $($latestReport.Name)" -ForegroundColor Yellow
  } else {
    Write-Host "No reports found in $ReportDir" -ForegroundColor Red
    exit 1
  }
}

# Load the JSON report
$report = Get-Content $LogFile | ConvertFrom-Json

Write-Host ""
Write-Host "Build Context:" -ForegroundColor Yellow
Write-Host "  Branch: $($report.context.branch)" -ForegroundColor White
Write-Host "  Commit: $($report.context.commit)" -ForegroundColor White
Write-Host "  Result: $($report.result)" -ForegroundColor $(if ($report.result -eq "GO") { "Green" } else { "Red" })
Write-Host ""

# Analyze each failed step
$failedSteps = $report.steps | Where-Object { $_.status -eq "FAIL" }

if ($failedSteps.Count -eq 0) {
  Write-Host "No failures found! Build passed successfully." -ForegroundColor Green
  exit 0
}

Write-Host "Failed Steps Analysis:" -ForegroundColor Red
Write-Host "======================" -ForegroundColor Red

foreach ($step in $failedSteps) {
  Write-Host ""
  Write-Host "Step: $($step.step)" -ForegroundColor Yellow
  Write-Host "Exit Code: $($step.exit)" -ForegroundColor White
  
  # Try to read the log file
  $logPath = $step.log
  if (-not [System.IO.Path]::IsPathRooted($logPath)) {
    $logPath = Join-Path $ReportDir $step.log
  }
  
  if (Test-Path $logPath) {
    Write-Host "Analyzing log: $(Split-Path $logPath -Leaf)" -ForegroundColor Gray
    
    $logContent = Get-Content $logPath -Raw
    
    # Common error patterns
    $patterns = @{
      "TypeScript Errors" = "error TS\d+:"
      "Module Not Found" = "Cannot find module|Module not found"
      "Shader Errors" = "shader.*error|WGSL.*error"
      "Build Errors" = "Build failed|npm ERR!"
      "Permission Denied" = "permission denied|access denied"
      "Network Errors" = "ECONNREFUSED|ETIMEDOUT|network"
      "Memory Issues" = "out of memory|heap"
      "File Not Found" = "no such file|cannot find|not found"
      "Syntax Errors" = "SyntaxError|unexpected token"
      "API Failures" = "API.*failed|401|403|404|500"
    }
    
    Write-Host ""
    Write-Host "  Error Categories Found:" -ForegroundColor White
    $foundErrors = $false
    
    foreach ($category in $patterns.Keys) {
      $matches = [regex]::Matches($logContent, $patterns[$category], [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
      if ($matches.Count -gt 0) {
        $foundErrors = $true
        Write-Host ("    - {0}: {1} occurrence(s)" -f $category, $matches.Count) -ForegroundColor Red
        
        # Show first few examples
        $examples = $matches | Select-Object -First 3
        foreach ($match in $examples) {
          $startIndex = [Math]::Max(0, $match.Index - 50)
          $length = [Math]::Min(150, $logContent.Length - $startIndex)
          $context = $logContent.Substring($startIndex, $length).Trim()
          $context = $context -replace "`r`n", " " -replace "`n", " "
          Write-Host ("      Example: ...{0}..." -f $context) -ForegroundColor Gray
        }
      }
    }
    
    if (-not $foundErrors) {
      Write-Host "    No common error patterns found. Showing last 10 lines:" -ForegroundColor Yellow
      Get-Content $logPath -Tail 10 | ForEach-Object { Write-Host ("      {0}" -f $_) -ForegroundColor Gray }
    }
    
    # Step-specific analysis
    Write-Host ""
    Write-Host "  Recommended Fix:" -ForegroundColor Green
    
    switch -Wildcard ($step.step) {
      "*TypeScript*" {
        Write-Host "    1. Run: npx tsc --noEmit to see all errors" -ForegroundColor White
        Write-Host "    2. Fix type errors in the reported files" -ForegroundColor White
        Write-Host "    3. Or use -SkipTypeCheck flag for quick build" -ForegroundColor White
      }
      "*Shader*" {
        Write-Host "    1. Check shader files for WGSL syntax errors" -ForegroundColor White
        Write-Host "    2. Verify device limits compatibility" -ForegroundColor White
        Write-Host "    3. Run: node tools\shaders\validate-wgsl.js for details" -ForegroundColor White
      }
      "*Build*" {
        Write-Host "    1. Clear node_modules and reinstall: rm -rf node_modules && npm install" -ForegroundColor White
        Write-Host "    2. Check for missing dependencies in package.json" -ForegroundColor White
        Write-Host "    3. Try quick build: -QuickBuild flag" -ForegroundColor White
      }
      "*API*" {
        Write-Host "    1. Check .env.production for correct API endpoints" -ForegroundColor White
        Write-Host "    2. Verify API server is running and accessible" -ForegroundColor White
        Write-Host "    3. Check network connectivity and firewall settings" -ForegroundColor White
      }
      "*Artifact*" {
        Write-Host "    1. Ensure build completed successfully" -ForegroundColor White
        Write-Host "    2. Check releases folder exists and has write permissions" -ForegroundColor White
        Write-Host "    3. Verify dist folder was created during build" -ForegroundColor White
      }
      default {
        Write-Host "    Check the full log file for specific error details" -ForegroundColor White
      }
    }
    
  } else {
    Write-Host "  Log file not found: $logPath" -ForegroundColor Red
  }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Quick Fix Commands:" -ForegroundColor Green
Write-Host ""

if ($failedSteps | Where-Object { $_.step -match "TypeScript" }) {
  Write-Host "Fix TypeScript errors:" -ForegroundColor Yellow
  Write-Host "  npx tsc --noEmit --pretty" -ForegroundColor White
  Write-Host ""
}

if ($failedSteps | Where-Object { $_.step -match "Shader" }) {
  Write-Host "Validate shaders:" -ForegroundColor Yellow
  Write-Host "  node tools\shaders\validate-wgsl.js --dir=frontend --strict" -ForegroundColor White
  Write-Host ""
}

if ($failedSteps | Where-Object { $_.step -match "Build" }) {
  Write-Host "Clean rebuild:" -ForegroundColor Yellow
  Write-Host "  npm ci && npm run build" -ForegroundColor White
  Write-Host ""
}

Write-Host "Run with fixes applied:" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File .\tools\release\AutoFixForShipping.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or skip validations for quick test:" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File .\tools\release\IrisOneButton.ps1 -SkipTypeCheck -SkipShaderCheck -QuickBuild" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
