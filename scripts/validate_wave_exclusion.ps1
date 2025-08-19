#!/usr/bin/env pwsh
# C:\Users\jason\Desktop\tori\kha\scripts\validate_wave_exclusion.ps1
# Validates that wave processing is properly excluded from production builds

Write-Host "Validating wave processing exclusion..." -ForegroundColor Cyan

$projectRoot = "C:\Users\jason\Desktop\tori\kha"
$errors = @()

# Check for direct wave imports in non-wave directories
Write-Host "Checking for accidental wave imports..." -ForegroundColor Yellow

$waveImports = Get-ChildItem -Path "$projectRoot\frontend" -Recurse -Include "*.ts","*.js" -File | 
    Where-Object { $_.FullName -notmatch "\\wave\\" -and $_.FullName -notmatch "node_modules" } |
    Select-String -Pattern "from ['""][^'"]*wave\/fft|AngularSpectrum|Gerchberg|Fresnel|Fraunhofer" |
    Select-Object -ExpandProperty Path -Unique

if ($waveImports.Count -gt 0) {
    Write-Host "ERROR: Found wave imports outside wave directory:" -ForegroundColor Red
    $waveImports | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    $errors += "Wave imports found outside wave directory"
}

# Check that engine.ts uses conditional loading
Write-Host "Checking engine.ts for conditional loading..." -ForegroundColor Yellow

$enginePath = "$projectRoot\frontend\lib\webgpu\engine.ts"
if (Test-Path $enginePath) {
    $engineContent = Get-Content $enginePath -Raw
    if ($engineContent -notmatch "__IRIS_WAVE__") {
        Write-Host "ERROR: engine.ts does not use __IRIS_WAVE__ flag" -ForegroundColor Red
        $errors += "engine.ts missing conditional loading"
    } else {
        Write-Host "  OK: engine.ts uses conditional loading" -ForegroundColor Green
    }
}

# Check .env file configuration
Write-Host "Checking .env configuration..." -ForegroundColor Yellow

$envFiles = @(
    "$projectRoot\standalone-holo\.env",
    "$projectRoot\iris\.env"
)

foreach ($envFile in $envFiles) {
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile -Raw
        if ($envContent -match "VITE_IRIS_ENABLE_WAVE=1") {
            Write-Host "WARNING: $envFile has wave processing enabled" -ForegroundColor Yellow
            Write-Host "  This should be =0 for production" -ForegroundColor Yellow
        } elseif ($envContent -match "VITE_IRIS_ENABLE_WAVE=0") {
            Write-Host "  OK: $(Split-Path $envFile -Leaf) has wave disabled" -ForegroundColor Green
        } else {
            Write-Host "  INFO: $(Split-Path $envFile -Leaf) does not specify VITE_IRIS_ENABLE_WAVE" -ForegroundColor Cyan
        }
    }
}

# Check vite.config for define block
Write-Host "Checking vite.config.js files..." -ForegroundColor Yellow

$viteConfigs = Get-ChildItem -Path $projectRoot -Recurse -Include "vite.config.js","vite.config.ts" -File |
    Where-Object { $_.FullName -notmatch "node_modules" }

foreach ($config in $viteConfigs) {
    $content = Get-Content $config.FullName -Raw
    if ($content -match "__IRIS_WAVE__") {
        Write-Host "  OK: $(Split-Path $config.FullName -Leaf) defines __IRIS_WAVE__" -ForegroundColor Green
    } else {
        Write-Host "  INFO: $(Split-Path $config.FullName -Leaf) does not define __IRIS_WAVE__" -ForegroundColor Cyan
    }
}

# Summary
Write-Host "`nValidation Summary:" -ForegroundColor Cyan
if ($errors.Count -eq 0) {
    Write-Host "  All checks passed! Wave processing properly excluded." -ForegroundColor Green
} else {
    Write-Host "  Found $($errors.Count) issue(s):" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    exit 1
}
