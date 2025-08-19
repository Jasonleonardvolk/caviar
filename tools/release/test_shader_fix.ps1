# Test shader bundle fix with comprehensive logging
param(
    [string]$RepoRoot = "D:\Dev\kha",
    [switch]$Verbose = $false
)

Push-Location $RepoRoot

# Create timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$errorDir = Join-Path $RepoRoot "tools\release\error_logs\$timestamp"

# Create error log directory
New-Item -ItemType Directory -Path $errorDir -Force | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " TESTING SHADER BUNDLE FIX WITH LOGGING" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Logging to: $errorDir" -ForegroundColor Yellow
Write-Host ""

# Initialize test results
$testResults = @{
    Timestamp = $timestamp
    ShaderBundle = "NOT_RUN"
    TypeScript = "NOT_RUN"
    TotalErrors = 0
    ErrorDetails = @()
}

# Step 1: Regenerate shader bundle
Write-Host "Step 1: Regenerating shader bundle..." -ForegroundColor Cyan
$shaderLog = Join-Path $errorDir "shader_bundle.log"

$shaderProcess = Start-Process -FilePath "node" -ArgumentList "scripts\bundleShaders.mjs" `
    -WorkingDirectory $RepoRoot -NoNewWindow -PassThru -Wait `
    -RedirectStandardOutput $shaderLog `
    -RedirectStandardError "$shaderLog.err"

if ($shaderProcess.ExitCode -eq 0) {
    Write-Host "✅ SUCCESS: Shader bundle generated" -ForegroundColor Green
    $testResults.ShaderBundle = "PASSED"
    
    if ($Verbose) {
        Get-Content $shaderLog | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    }
} else {
    Write-Host "❌ ERROR: Shader bundler failed!" -ForegroundColor Red
    $testResults.ShaderBundle = "FAILED"
    
    Write-Host "Error output:" -ForegroundColor Yellow
    if (Test-Path "$shaderLog.err") {
        Get-Content "$shaderLog.err" | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }
    Get-Content $shaderLog | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    
    # Save test results and exit
    $testResults | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $errorDir "test_results.json")
    Pop-Location
    exit 1
}

Write-Host ""

# Step 2: Test TypeScript compilation
Write-Host "Step 2: Testing TypeScript compilation..." -ForegroundColor Cyan
$tsLog = Join-Path $errorDir "typescript_errors.txt"

$tsProcess = Start-Process -FilePath "npx" -ArgumentList "tsc", "-p", ".\frontend\tsconfig.json", "--noEmit" `
    -WorkingDirectory $RepoRoot -NoNewWindow -PassThru -Wait `
    -RedirectStandardOutput $tsLog `
    -RedirectStandardError "$tsLog.err"

# Combine output and error streams
if (Test-Path "$tsLog.err") {
    Get-Content "$tsLog.err" | Add-Content $tsLog
    Remove-Item "$tsLog.err"
}

if ($tsProcess.ExitCode -eq 0) {
    Write-Host "✅ SUCCESS: TypeScript compilation passed!" -ForegroundColor Green
    $testResults.TypeScript = "PASSED"
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " ALL TESTS PASSED!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ready to run full release gate:" -ForegroundColor Cyan
    Write-Host "  .\tools\release\IrisOneButton.ps1" -ForegroundColor White
    
    # Create success summary
    @"
Test Run Summary
================
Timestamp: $timestamp
Shader Bundle: PASSED
TypeScript: PASSED

Ready to run full release gate:
  .\tools\release\IrisOneButton.ps1
"@ | Set-Content (Join-Path $errorDir "SUCCESS.txt")
    
} else {
    Write-Host "❌ TypeScript compilation failed!" -ForegroundColor Red
    $testResults.TypeScript = "FAILED"
    
    # Parse TypeScript errors
    $tsErrors = Get-Content $tsLog | Select-String "error TS"
    $errorCount = $tsErrors.Count
    $testResults.TotalErrors = $errorCount
    
    Write-Host ""
    Write-Host "Total TypeScript errors: $errorCount" -ForegroundColor Red
    Write-Host ""
    
    # Group errors by file
    $errorsByFile = @{}
    foreach ($error in $tsErrors) {
        if ($error -match "(.+?)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)") {
            $file = $Matches[1]
            $line = $Matches[2]
            $col = $Matches[3]
            $code = $Matches[4]
            $message = $Matches[5]
            
            if (-not $errorsByFile.ContainsKey($file)) {
                $errorsByFile[$file] = @()
            }
            
            $errorsByFile[$file] += @{
                Line = $line
                Column = $col
                Code = $code
                Message = $message
            }
        }
    }
    
    # Show errors by file
    Write-Host "Errors by file:" -ForegroundColor Yellow
    foreach ($file in ($errorsByFile.Keys | Sort-Object)) {
        $fileErrors = $errorsByFile[$file]
        $relPath = $file.Replace($RepoRoot, ".")
        Write-Host "  $relPath ($($fileErrors.Count) errors)" -ForegroundColor Cyan
        
        if ($Verbose) {
            foreach ($err in $fileErrors[0..4]) {  # Show first 5 errors per file
                Write-Host "    Line $($err.Line): $($err.Code) - $($err.Message)" -ForegroundColor Gray
            }
            if ($fileErrors.Count -gt 5) {
                Write-Host "    ... and $($fileErrors.Count - 5) more" -ForegroundColor DarkGray
            }
        }
    }
    
    $testResults.ErrorDetails = $errorsByFile
    
    # Create failure summary
    @"
Test Run Summary
================
Timestamp: $timestamp
Shader Bundle: PASSED
TypeScript: FAILED
Total Errors: $errorCount

Files with errors: $($errorsByFile.Keys.Count)

Top files with most errors:
$(
    $errorsByFile.GetEnumerator() | 
    Sort-Object { $_.Value.Count } -Descending | 
    Select-Object -First 5 | 
    ForEach-Object { "  - $($_.Key.Replace($RepoRoot, '.')): $($_.Value.Count) errors" }
)

Full log: $tsLog
"@ | Set-Content (Join-Path $errorDir "FAILURE.txt")
    
    if (-not $Verbose) {
        Write-Host ""
        Write-Host "Run with -Verbose to see error details" -ForegroundColor Yellow
    }
}

# Save test results as JSON
$testResults | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $errorDir "test_results.json")

Write-Host ""
Write-Host "All logs saved to:" -ForegroundColor Cyan
Write-Host "  $errorDir" -ForegroundColor White
Write-Host ""

# Create a symlink to latest results
$latestLink = Join-Path (Split-Path $errorDir -Parent) "latest"
if (Test-Path $latestLink) {
    Remove-Item $latestLink -Force
}
New-Item -ItemType SymbolicLink -Path $latestLink -Target $errorDir -Force | Out-Null
Write-Host "Latest results: $latestLink" -ForegroundColor Gray

Pop-Location

# Return exit code
if ($testResults.TypeScript -eq "PASSED") {
    exit 0
} else {
    exit 1
}
