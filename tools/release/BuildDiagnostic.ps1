param(
  [string]$RepoRoot = "D:\Dev\kha"
)

$ErrorActionPreference = "Continue"  # Don't stop on errors, we want to see everything
Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DETAILED BUILD DIAGNOSTIC" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Check Node/NPM
Write-Host "`n1. Checking Node/NPM installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    $npmVersion = npm --version
    Write-Host "   Node: $nodeVersion" -ForegroundColor Green
    Write-Host "   NPM: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "   ERROR: Node/NPM not found!" -ForegroundColor Red
}

# Step 2: Check the import fix
Write-Host "`n2. Checking realGhostEngine.js imports..." -ForegroundColor Yellow
$realGhostPath = Join-Path $RepoRoot "tori_ui_svelte\src\lib\realGhostEngine.js"

if (Test-Path $realGhostPath) {
    $imports = Get-Content $realGhostPath | Where-Object { $_ -match "^import" }
    Write-Host "   Found imports:" -ForegroundColor Gray
    $imports | ForEach-Object { Write-Host "     $_" -ForegroundColor White }
}

# Step 3: Check if imported files exist
Write-Host "`n3. Verifying imported files exist..." -ForegroundColor Yellow

$filesToCheck = @(
    "frontend\lib\holographicEngine.ts",
    "frontend\lib\holographicRenderer.ts", 
    "frontend\lib\webgpu\fftCompute.ts",
    "frontend\lib\webgpu\hologramPropagation.ts",
    "tools\quilt\WebGPU\QuiltGenerator.ts",
    "tori_ui_svelte\src\lib\conceptHologramRenderer.js"
)

foreach ($file in $filesToCheck) {
    $fullPath = Join-Path $RepoRoot $file
    if (Test-Path $fullPath) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file - NOT FOUND!" -ForegroundColor Red
    }
}

# Step 4: Try simple Vite build directly
Write-Host "`n4. Trying direct Vite build in tori_ui_svelte..." -ForegroundColor Yellow
Set-Location (Join-Path $RepoRoot "tori_ui_svelte")

Write-Host "   Running: npx vite build" -ForegroundColor Gray
$viteOutput = npx vite build 2>&1 | Out-String

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ Vite build succeeded!" -ForegroundColor Green
} else {
    Write-Host "   ✗ Vite build failed!" -ForegroundColor Red
    Write-Host "`n   Error output:" -ForegroundColor Yellow
    
    # Extract just the error message
    $errorLines = $viteOutput -split "`n" | Where-Object { 
        $_ -match "error" -or 
        $_ -match "Could not resolve" -or 
        $_ -match "Error:" -or
        $_ -match "RollupError"
    }
    
    if ($errorLines) {
        $errorLines | ForEach-Object { Write-Host "     $_" -ForegroundColor Red }
    } else {
        # Show last 10 lines if no specific error found
        $lines = $viteOutput -split "`n"
        $start = [Math]::Max(0, $lines.Count - 10)
        Write-Host "   Last 10 lines of output:" -ForegroundColor Yellow
        $lines[$start..($lines.Count-1)] | ForEach-Object { Write-Host "     $_" -ForegroundColor Gray }
    }
}

# Step 5: Check what directories exist after build attempt
Write-Host "`n5. Checking build output directories..." -ForegroundColor Yellow
Set-Location $RepoRoot

$outputDirs = @(
    "tori_ui_svelte\build",
    "tori_ui_svelte\dist",
    "tori_ui_svelte\.svelte-kit\output",
    "tori_ui_svelte\.svelte-kit\output\client",
    "tori_ui_svelte\.svelte-kit\output\server",
    "dist",
    "build"
)

foreach ($dir in $outputDirs) {
    $fullPath = Join-Path $RepoRoot $dir
    if (Test-Path $fullPath) {
        $count = (Get-ChildItem $fullPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($count -gt 0) {
            Write-Host "   ✓ $dir - $count files" -ForegroundColor Green
        } else {
            Write-Host "   ⚠ $dir - exists but empty" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ✗ $dir - does not exist" -ForegroundColor Gray
    }
}

# Step 6: Check TypeScript compilation
Write-Host "`n6. Checking TypeScript issues..." -ForegroundColor Yellow
Write-Host "   Running: npx tsc --noEmit" -ForegroundColor Gray

$tscOutput = npx tsc --noEmit 2>&1 | Out-String
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ No TypeScript errors" -ForegroundColor Green
} else {
    $errorCount = 0
    if ($tscOutput -match "Found (\d+) error") {
        $errorCount = $Matches[1]
    }
    Write-Host "   ⚠ TypeScript errors: $errorCount" -ForegroundColor Yellow
}

# Step 7: Package.json check
Write-Host "`n7. Checking package.json build script..." -ForegroundColor Yellow
$packagePath = Join-Path $RepoRoot "package.json"
if (Test-Path $packagePath) {
    $package = Get-Content $packagePath | ConvertFrom-Json
    $buildScript = $package.scripts.build
    Write-Host "   Build script: $buildScript" -ForegroundColor White
}

$sveltePackagePath = Join-Path $RepoRoot "tori_ui_svelte\package.json"
if (Test-Path $sveltePackagePath) {
    $sveltePackage = Get-Content $sveltePackagePath | ConvertFrom-Json
    $svelteBuildScript = $sveltePackage.scripts.build
    Write-Host "   Svelte build script: $svelteBuildScript" -ForegroundColor White
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DIAGNOSTIC COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nSummary:" -ForegroundColor Yellow
Write-Host "- Check the error messages above" -ForegroundColor White
Write-Host "- Look for missing files (marked with ✗)" -ForegroundColor White
Write-Host "- Review the Vite build errors" -ForegroundColor White

exit 0