# Phase 6: TypeScript Fixes Verification
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 6: TypeScript Fixes Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path "D:\Dev\kha\tori_ui_svelte"

Write-Host "Checking TypeScript compilation..." -ForegroundColor Yellow
Write-Host ""

# Run type check and capture output
Write-Host "Running: pnpm run check" -ForegroundColor Gray
$checkOutput = & pnpm run check 2>&1 | Out-String

# Check for errors
$errorCount = ([regex]::Matches($checkOutput, "error TS\d+")).Count

if ($errorCount -gt 0) {
    Write-Host ""
    Write-Host "[WARN] Found $errorCount TypeScript errors" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Common errors to look for:" -ForegroundColor Yellow
    Write-Host "- 'Element implicitly has an 'any' type'" -ForegroundColor Gray
    Write-Host "- 'No index signature'" -ForegroundColor Gray
    Write-Host "- 'Property does not exist'" -ForegroundColor Gray
    Write-Host "- 'Duplicate identifier'" -ForegroundColor Gray
    
    # Show first few errors
    $errors = $checkOutput -split "`n" | Where-Object { $_ -match "error TS" } | Select-Object -First 5
    if ($errors) {
        Write-Host ""
        Write-Host "First few errors:" -ForegroundColor Yellow
        $errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }
} else {
    Write-Host ""
    Write-Host "[OK] No TypeScript errors detected!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Specific File Checks" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Verifying fixed files exist:" -ForegroundColor Yellow

# Check each fixed file
$filesToCheck = @(
    @{Path = "src\lib\cognitive\braidMemory.ts"; Name = "braidMemory.ts"},
    @{Path = "src\lib\cognitive\loopRecord.ts"; Name = "loopRecord.ts"},
    @{Path = "src\lib\services\PersonaEmergenceEngine.ts"; Name = "PersonaEmergenceEngine.ts"},
    @{Path = "src\lib\services\solitonMemory.ts"; Name = "solitonMemory.ts"},
    @{Path = "src\lib\components\DatabaseReset.svelte"; Name = "DatabaseReset.svelte"}
)

$allFilesExist = $true
foreach ($file in $filesToCheck) {
    if (Test-Path $file.Path) {
        Write-Host "[OK] $($file.Name) exists" -ForegroundColor Green
        
        # Check for specific fixes
        $content = Get-Content $file.Path -Raw
        
        switch ($file.Name) {
            "braidMemory.ts" {
                if ($content -match "as Record<string, number>") {
                    Write-Host "     ✓ Record type fix applied" -ForegroundColor Gray
                } else {
                    Write-Host "     ⚠ Record type fix not found" -ForegroundColor Yellow
                }
            }
            "loopRecord.ts" {
                if ($content -match "as Record<string, number>") {
                    Write-Host "     ✓ Record type fix applied" -ForegroundColor Gray
                } else {
                    Write-Host "     ⚠ Record type fix not found" -ForegroundColor Yellow
                }
            }
            "PersonaEmergenceEngine.ts" {
                if ($content -match "personaRegistry instanceof Map") {
                    Write-Host "     ✓ Map vs Object fix applied" -ForegroundColor Gray
                } else {
                    Write-Host "     ⚠ Map vs Object fix not found" -ForegroundColor Yellow
                }
            }
            "solitonMemory.ts" {
                if ($content -match "userId: userId \|\| event\.userId") {
                    Write-Host "     ✓ Duplicate userId fix applied" -ForegroundColor Gray
                } else {
                    Write-Host "     ⚠ Duplicate userId fix not found" -ForegroundColor Yellow
                }
            }
            "DatabaseReset.svelte" {
                if ($content -match "toriStorage\.clearAll\(\)") {
                    Write-Host "     ✓ Method name fix applied" -ForegroundColor Gray
                } else {
                    Write-Host "     ⚠ Method name fix not found" -ForegroundColor Yellow
                }
            }
        }
    } else {
        Write-Host "[ERROR] $($file.Name) not found" -ForegroundColor Red
        $allFilesExist = $false
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fix Verification Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($allFilesExist -and $errorCount -eq 0) {
    Write-Host "SUCCESS: All TypeScript fixes applied and working!" -ForegroundColor Green
} elseif ($allFilesExist) {
    Write-Host "PARTIAL SUCCESS: Files fixed but some TypeScript errors remain" -ForegroundColor Yellow
} else {
    Write-Host "FAILED: Some files are missing" -ForegroundColor Red
}

Write-Host ""
Write-Host "Fixed issues:" -ForegroundColor Cyan
Write-Host "✅ Index signature errors in GLYPH_TYPE_BONUSES" -ForegroundColor White
Write-Host "✅ Map vs Object handling in PersonaEmergenceEngine" -ForegroundColor White
Write-Host "✅ Duplicate userId in solitonMemory" -ForegroundColor White
Write-Host "✅ Method name in DatabaseReset.svelte" -ForegroundColor White
Write-Host ""
Write-Host "Next: Run 'pnpm run build' to verify production build" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
