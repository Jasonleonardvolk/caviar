#!/usr/bin/env pwsh
# Test-GitWorkflow.ps1
# Verify the Git workflow system is properly configured

param(
    [switch]$Verbose
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "   Git Workflow System Test Suite" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$tests = @()
$repoRoot = git rev-parse --show-toplevel 2>$null

if (!$repoRoot) {
    Write-Host "ERROR: Not in a Git repository" -ForegroundColor Red
    exit 1
}

$repoRoot = $repoRoot -replace '/', '\'
Set-Location $repoRoot

# Test 1: Git hooks configured
$hooksPath = git config core.hooksPath
if ($hooksPath -eq ".githooks") {
    $tests += @{Name="Git hooks path"; Status="PASS"; Details="Configured to .githooks"}
} elseif ($hooksPath) {
    $tests += @{Name="Git hooks path"; Status="WARN"; Details="Set to: $hooksPath (expected: .githooks)"}
} else {
    $tests += @{Name="Git hooks path"; Status="FAIL"; Details="Not configured (run Setup-GitHooks.ps1)"}
}

# Test 2: Pre-push hook exists
if (Test-Path ".githooks\pre-push") {
    $tests += @{Name="Pre-push hook (Unix)"; Status="PASS"; Details="Present"}
} else {
    $tests += @{Name="Pre-push hook (Unix)"; Status="FAIL"; Details="Missing"}
}

if (Test-Path ".githooks\pre-push.cmd") {
    $tests += @{Name="Pre-push hook (Windows)"; Status="PASS"; Details="Present"}
} else {
    $tests += @{Name="Pre-push hook (Windows)"; Status="FAIL"; Details="Missing"}
}

# Test 3: Workflow scripts exist
$scripts = @(
    "tools\git\Git-Workflow.ps1",
    "tools\git\Setup-GitHooks.ps1",
    "tools\release\verify-setup.ps1",
    "tools\release\Reset-And-Ship.ps1"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        $tests += @{Name=$script; Status="PASS"; Details="Present"}
    } else {
        $tests += @{Name=$script; Status="FAIL"; Details="Missing"}
    }
}

# Test 4: Test hook execution (dry run)
if ($Verbose) {
    Write-Host "`nTesting hook execution (dry run)..." -ForegroundColor Yellow
    try {
        $result = & ".\.githooks\pre-push.cmd" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $tests += @{Name="Hook execution"; Status="PASS"; Details="Runs successfully"}
        } else {
            $tests += @{Name="Hook execution"; Status="WARN"; Details="Runs but returns non-zero"}
        }
    } catch {
        $tests += @{Name="Hook execution"; Status="FAIL"; Details=$_.Exception.Message}
    }
}

# Test 5: Git workflow script syntax
try {
    $null = Test-Path "tools\git\Git-Workflow.ps1" -ErrorAction Stop
    $parseErrors = $null
    $null = [System.Management.Automation.PSParser]::Tokenize((Get-Content "tools\git\Git-Workflow.ps1" -Raw), [ref]$parseErrors)
    if ($parseErrors.Count -eq 0) {
        $tests += @{Name="Git-Workflow.ps1 syntax"; Status="PASS"; Details="Valid PowerShell"}
    } else {
        $tests += @{Name="Git-Workflow.ps1 syntax"; Status="FAIL"; Details="$($parseErrors.Count) syntax errors"}
    }
} catch {
    $tests += @{Name="Git-Workflow.ps1 syntax"; Status="SKIP"; Details="Could not parse"}
}

# Display results
Write-Host "Test Results:" -ForegroundColor Yellow
Write-Host "-------------" -ForegroundColor Gray

$passCount = 0
$failCount = 0
$warnCount = 0

foreach ($test in $tests) {
    $color = switch ($test.Status) {
        "PASS" { $passCount++; "Green" }
        "FAIL" { $failCount++; "Red" }
        "WARN" { $warnCount++; "Yellow" }
        default { "Gray" }
    }
    
    $symbol = switch ($test.Status) {
        "PASS" { "[OK]" }
        "FAIL" { "[X]" }
        "WARN" { "[!]" }
        default { "[o]" }
    }
    
    Write-Host "$symbol " -NoNewline -ForegroundColor $color
    Write-Host "$($test.Name): " -NoNewline
    Write-Host $test.Details -ForegroundColor $color
}

# Summary
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Passed: $passCount" -ForegroundColor Green
if ($warnCount -gt 0) {
    Write-Host "  Warnings: $warnCount" -ForegroundColor Yellow
}
if ($failCount -gt 0) {
    Write-Host "  Failed: $failCount" -ForegroundColor Red
}

# Overall status
Write-Host "`nOverall Status: " -NoNewline
if ($failCount -eq 0) {
    Write-Host "GIT WORKFLOW READY! [OK]" -ForegroundColor Green
    Write-Host "`nYou can now use:" -ForegroundColor Cyan
    Write-Host "  .\tools\git\Git-Workflow.ps1 status" -ForegroundColor White
    Write-Host "  .\tools\git\Git-Workflow.ps1 branch feat api 'new-feature'" -ForegroundColor White
    Write-Host "  .\tools\git\Git-Workflow.ps1 release '1.0.0' -Push" -ForegroundColor White
} else {
    Write-Host "SETUP INCOMPLETE [X]" -ForegroundColor Red
    Write-Host "`nRun the following to complete setup:" -ForegroundColor Yellow
    Write-Host "  .\tools\git\Setup-GitHooks.ps1" -ForegroundColor White
}

Write-Host ""

# Quick command reference
if ($failCount -eq 0) {
    Write-Host "Quick Command Reference:" -ForegroundColor Cyan
    Write-Host "------------------------" -ForegroundColor Gray
    Write-Host "Setup hooks:    .\setup-git-hooks.bat" -ForegroundColor White
    Write-Host "Check status:   .\git-status.bat" -ForegroundColor White
    Write-Host "New feature:    .\tools\git\Git-Workflow.ps1 branch feat scope name" -ForegroundColor White
    Write-Host "Internal tag:   .\tools\git\Git-Workflow.ps1 internal '0.1.1'" -ForegroundColor White
    Write-Host "RC tag:         .\tools\git\Git-Workflow.ps1 rc '1.0.0-rc.1'" -ForegroundColor White
    Write-Host "Release:        .\tools\git\Git-Workflow.ps1 release '1.0.0'" -ForegroundColor White
    Write-Host ""
}