#!/usr/bin/env powershell
# Quick verification script for iRis setup
param([switch]$SkipAssets)

# Error trap for unexpected failures
trap {
    Write-Host "[X] verify-setup.ps1 fatal: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== iRis v0.1.0 Setup Verification ===" -ForegroundColor Cyan

# Set project directory
$projectDir = "D:\Dev\kha\tori_ui_svelte"

# Check for skip assets flag
if ($SkipAssets) {
    Write-Host "[i] SkipAssets enabled -> skipping 3D asset validation" -ForegroundColor Yellow
    Write-Host ""
}

# Check for dirty working tree (optional - warn but don't fail)
$dirty = git status --porcelain 2>$null
if ($dirty) {
    Write-Host "[!] Warning: Working tree has uncommitted changes" -ForegroundColor Yellow
    Write-Host "   Consider committing before push" -ForegroundColor Gray
    Write-Host ""
}

$checks = @()

# --- ASSET MANIFEST VALIDATION (blocks on fail) ---
if (-not $SkipAssets) {
    $manifest = Join-Path $projectDir "assets\3d\luxury\ASSET_MANIFEST.json"
    if (Test-Path $manifest) {
        Write-Host "Validating asset manifest..." -ForegroundColor Yellow
        $nodeExists = Get-Command node -ErrorAction SilentlyContinue
        if ($nodeExists) {
            $validatorPath = Join-Path $projectDir "tools\assets\validate-manifest.mjs"
            if (Test-Path $validatorPath) {
                & node $validatorPath $manifest `
                    --maxTris=100000 `
                    --maxTexDim=4096 `
                    --maxTexBytes=16777216 `
                    --maxAlphaDim=2048 `
                    --maxAlphaBytes=8388608 `
                    --maxSetBytes=52428800 2>&1
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[X] Asset manifest validation FAILED" -ForegroundColor Red
                    Write-Host "Fix the issues above before pushing." -ForegroundColor Red
                    exit 1
                }
                Write-Host "[OK] Asset manifest validation passed" -ForegroundColor Green
                Write-Host ""
            } else {
                Write-Host "[!] Asset validator not found at $validatorPath" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[!] Node.js not found - skipping asset validation" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[i] No asset manifest found (skipping 3D asset validation)" -ForegroundColor Gray
    }
}

# Check 1: Project directory exists
if (Test-Path $projectDir) {
    $checks += @{Name="Project directory"; Status="PASS"; Details=$projectDir}
} else {
    $checks += @{Name="Project directory"; Status="FAIL"; Details="Not found: $projectDir"}
}

# Check 2: Environment files
if (Test-Path "$projectDir\.env.local") {
    $checks += @{Name=".env.local"; Status="PASS"; Details="Present"}
} else {
    $checks += @{Name=".env.local"; Status="FAIL"; Details="Missing"}
}

if (Test-Path "$projectDir\.env.production") {
    $checks += @{Name=".env.production"; Status="PASS"; Details="Present"}
} else {
    $checks += @{Name=".env.production"; Status="WARN"; Details="Missing (optional)"}
}

# Check 3: Upload directory
if (Test-Path "$projectDir\var\uploads") {
    $checks += @{Name="Upload directory"; Status="PASS"; Details="var\uploads"}
} else {
    $checks += @{Name="Upload directory"; Status="FAIL"; Details="Missing"}
}

# Check 4: Key files
$keyFiles = @(
    "src\routes\+page.server.ts",
    "src\routes\renderer\+page.svelte", 
    "src\routes\api\health\+server.ts",
    "src\routes\api\list\+server.ts",
    "src\routes\api\pdf\stats\+server.ts",
    "src\routes\api\memory\state\+server.ts",
    "src\lib\services\storage.ts"
)

foreach ($file in $keyFiles) {
    $fullPath = Join-Path $projectDir $file
    if (Test-Path $fullPath) {
        $checks += @{Name=$file; Status="PASS"; Details="Present"}
    } else {
        $checks += @{Name=$file; Status="FAIL"; Details="Missing"}
    }
}

# Check 5: Node modules
if (Test-Path "$projectDir\node_modules") {
    $moduleCount = (Get-ChildItem "$projectDir\node_modules" -Directory).Count
    $checks += @{Name="Node modules"; Status="PASS"; Details="$moduleCount packages"}
} else {
    $checks += @{Name="Node modules"; Status="WARN"; Details="Not installed (run: pnpm install)"}
}

# Check 6: Build directory
if (Test-Path "$projectDir\build") {
    $checks += @{Name="Build directory"; Status="PASS"; Details="Ready"}
} else {
    $checks += @{Name="Build directory"; Status="INFO"; Details="Not built yet (run: pnpm build)"}
}

# Check 7: AR Assets (skip if SkipAssets is enabled)
if (-not $SkipAssets) {
    if (Test-Path "$projectDir\assets\3d\luxury\ASSET_MANIFEST.json") {
        $checks += @{Name="AR Asset Manifest"; Status="PASS"; Details="Present"}
    } else {
        $checks += @{Name="AR Asset Manifest"; Status="WARN"; Details="Missing (optional)"}
    }

    if (Test-Path "$projectDir\src\routes\assets\+page.svelte") {
        $checks += @{Name="Assets viewer route"; Status="PASS"; Details="Present"}
    } else {
        $checks += @{Name="Assets viewer route"; Status="WARN"; Details="Missing (optional)"}
    }

    if (Test-Path "$projectDir\src\routes\api\assets\+server.ts") {
        $checks += @{Name="Assets API endpoint"; Status="PASS"; Details="Present"}
    } else {
        $checks += @{Name="Assets API endpoint"; Status="WARN"; Details="Missing (optional)"}
    }
}

# Display results
Write-Host "`nVerification Results:" -ForegroundColor Yellow
Write-Host "---------------------" -ForegroundColor Gray

$passCount = 0
$failCount = 0
$warnCount = 0

foreach ($check in $checks) {
    $color = switch ($check.Status) {
        "PASS" { $passCount++; "Green" }
        "FAIL" { $failCount++; "Red" }
        "WARN" { $warnCount++; "Yellow" }
        default { "Gray" }
    }
    
    $statusSymbol = switch ($check.Status) {
        "PASS" { "[OK]" }
        "FAIL" { "[X]" }
        "WARN" { "[!]" }
        default { "[i]" }
    }
    
    Write-Host "$statusSymbol " -NoNewline -ForegroundColor $color
    Write-Host "$($check.Name): " -NoNewline
    Write-Host $check.Details -ForegroundColor $color
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
    Write-Host "READY TO BUILD AND SHIP! [OK]" -ForegroundColor Green
    Write-Host "`nNext step: Run .\tools\release\Reset-And-Ship.ps1" -ForegroundColor Cyan
} elseif ($failCount -le 2) {
    Write-Host "ALMOST READY (minor issues) [!]" -ForegroundColor Yellow
    Write-Host "`nFix the failed checks above, then run .\tools\release\Reset-And-Ship.ps1" -ForegroundColor Yellow
} else {
    Write-Host "NOT READY (critical issues) [X]" -ForegroundColor Red
    Write-Host "`nPlease fix the failed checks above before proceeding." -ForegroundColor Red
}

Write-Host ""

# Explicit exit codes
# 0 = OK or warnings only, >=1 = failures
if ($failCount -eq 0) {
    exit 0
} else {
    exit 1
}
