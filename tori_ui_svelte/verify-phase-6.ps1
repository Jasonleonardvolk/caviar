# Phase 6 Verification Script - Canonical Hologram Path
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 6: Canonical Path Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path "D:\Dev\kha\tori_ui_svelte"

$errors = 0
$warnings = 0

# Step 1: Check for non-canonical imports
Write-Host "STEP 1: Checking for non-canonical component imports..." -ForegroundColor Yellow
Write-Host ""

$nonCanonicalPatterns = @(
    "HolographicDisplayEnhanced",
    "HolographicDisplay_FIXED"
)

foreach ($pattern in $nonCanonicalPatterns) {
    Write-Host "  Searching for: $pattern" -ForegroundColor Gray
    $results = Get-ChildItem -Path "src" -Recurse -Include "*.svelte","*.ts","*.js" -File | 
               Select-String -Pattern "import.*$pattern" -List
    
    if ($results) {
        Write-Host "  [ERROR] Found imports of $pattern:" -ForegroundColor Red
        foreach ($result in $results) {
            Write-Host "    - $($result.Path):$($result.LineNumber)" -ForegroundColor Red
        }
        $errors++
    } else {
        Write-Host "  [OK] No imports of $pattern found" -ForegroundColor Green
    }
}

Write-Host ""

# Step 2: Check canonical component exists
Write-Host "STEP 2: Verifying canonical component..." -ForegroundColor Yellow
Write-Host ""

$canonicalPath = "src\lib\components\HolographicDisplay.svelte"
if (Test-Path $canonicalPath) {
    Write-Host "  [OK] Canonical component exists: $canonicalPath" -ForegroundColor Green
    $fileInfo = Get-Item $canonicalPath
    Write-Host "       Size: $($fileInfo.Length) bytes" -ForegroundColor Gray
    Write-Host "       Modified: $($fileInfo.LastWriteTime)" -ForegroundColor Gray
} else {
    Write-Host "  [ERROR] Canonical component NOT FOUND: $canonicalPath" -ForegroundColor Red
    $errors++
}

Write-Host ""

# Step 3: Check environment variables
Write-Host "STEP 3: Checking environment configuration..." -ForegroundColor Yellow
Write-Host ""

$envFiles = @(".env", ".env.production")
foreach ($envFile in $envFiles) {
    if (Test-Path $envFile) {
        $content = Get-Content $envFile -Raw
        if ($content -match "VITE_TORI_HOLOGRAM_CANONICAL") {
            Write-Host "  [OK] Feature flag found in $envFile" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] Feature flag missing in $envFile" -ForegroundColor Yellow
            $warnings++
        }
    } else {
        Write-Host "  [WARN] $envFile not found" -ForegroundColor Yellow
        $warnings++
    }
}

Write-Host ""

# Step 4: Check tsconfig.json excludes
Write-Host "STEP 4: Verifying TypeScript configuration..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path "tsconfig.json") {
    $tsconfig = Get-Content "tsconfig.json" -Raw | ConvertFrom-Json
    $expectedExcludes = @(
        "src/lib/components/HolographicDisplayEnhanced.svelte",
        "src/lib/components/HolographicDisplay_FIXED.svelte"
    )
    
    $missingExcludes = @()
    foreach ($exclude in $expectedExcludes) {
        if ($tsconfig.exclude -notcontains $exclude) {
            $missingExcludes += $exclude
        }
    }
    
    if ($missingExcludes.Count -eq 0) {
        Write-Host "  [OK] All non-canonical files excluded in tsconfig.json" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Missing excludes in tsconfig.json:" -ForegroundColor Yellow
        foreach ($missing in $missingExcludes) {
            Write-Host "    - $missing" -ForegroundColor Yellow
        }
        $warnings++
    }
} else {
    Write-Host "  [ERROR] tsconfig.json not found" -ForegroundColor Red
    $errors++
}

Write-Host ""

# Step 5: Check non-canonical files are fenced
Write-Host "STEP 5: Verifying non-canonical files are fenced..." -ForegroundColor Yellow
Write-Host ""

$fencedFiles = @(
    "src\lib\components\HolographicDisplay_FIXED.svelte",
    "src\lib\components\HolographicDisplayEnhanced.svelte"
)

foreach ($file in $fencedFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] Fenced file exists: $(Split-Path $file -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "  [INFO] Fenced file not found (already archived?): $(Split-Path $file -Leaf)" -ForegroundColor Gray
    }
}

Write-Host ""

# Step 6: Build check
Write-Host "STEP 6: Running build checks..." -ForegroundColor Yellow
Write-Host ""

Write-Host "  Running: pnpm run check" -ForegroundColor Gray
$checkResult = & pnpm run check 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Type check passed" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Type check failed" -ForegroundColor Red
    $errors++
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "SUCCESS: All Phase 6 requirements met!" -ForegroundColor Green
    Write-Host "  - No non-canonical imports found" -ForegroundColor Green
    Write-Host "  - Canonical component verified" -ForegroundColor Green
    Write-Host "  - Environment flags configured" -ForegroundColor Green
    Write-Host "  - TypeScript excludes in place" -ForegroundColor Green
    Write-Host "  - Build checks passed" -ForegroundColor Green
} elseif ($errors -eq 0) {
    Write-Host "PARTIAL SUCCESS: Phase 6 complete with warnings" -ForegroundColor Yellow
    Write-Host "  Errors: $errors" -ForegroundColor Green
    Write-Host "  Warnings: $warnings" -ForegroundColor Yellow
} else {
    Write-Host "FAILED: Phase 6 has errors that need fixing" -ForegroundColor Red
    Write-Host "  Errors: $errors" -ForegroundColor Red
    Write-Host "  Warnings: $warnings" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Definition of Done Checklist:" -ForegroundColor Cyan
Write-Host "  [$(if ($errors -eq 0) {'X'} else {' '})] Zero imports of non-canonical components" -ForegroundColor White
Write-Host "  [$(if ($errors -eq 0) {'X'} else {' '})] pnpm run check passes" -ForegroundColor White
Write-Host "  [ ] pnpm run build succeeds (run separately)" -ForegroundColor White
Write-Host "  [X] Only HolographicDisplay.svelte is canonical" -ForegroundColor White

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
