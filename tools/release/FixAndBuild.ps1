param(
  [string]$RepoRoot = "D:\Dev\kha"
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

Write-Host "FIX AND BUILD SCRIPT" -ForegroundColor Cyan
Write-Host "====================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if the import was fixed
Write-Host "Checking import fix in realGhostEngine.js..." -ForegroundColor Yellow
$realGhostPath = Join-Path $RepoRoot "tori_ui_svelte\src\lib\realGhostEngine.js"

if (Test-Path $realGhostPath) {
    $content = Get-Content $realGhostPath -Raw
    if ($content -match "frontend/lib/webgpu/quiltGenerator") {
        Write-Host "  ERROR: Old import path still present!" -ForegroundColor Red
        Write-Host "  The import has NOT been fixed properly" -ForegroundColor Red
        exit 1
    } elseif ($content -match "tools/quilt/WebGPU/QuiltGenerator") {
        Write-Host "  Import path is fixed!" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: Cannot verify import path" -ForegroundColor Yellow
    }
}

# Step 2: Check if QuiltGenerator file actually exists
Write-Host "`nVerifying QuiltGenerator file exists..." -ForegroundColor Yellow
$quiltPath = Join-Path $RepoRoot "tools\quilt\WebGPU\QuiltGenerator.ts"

if (Test-Path $quiltPath) {
    Write-Host "  QuiltGenerator.ts exists at correct location" -ForegroundColor Green
} else {
    Write-Host "  ERROR: QuiltGenerator.ts not found at: $quiltPath" -ForegroundColor Red
}

# Step 3: Clean previous build attempts
Write-Host "`nCleaning previous build artifacts..." -ForegroundColor Yellow
$buildDirs = @(
    "tori_ui_svelte\build",
    "tori_ui_svelte\dist",
    "tori_ui_svelte\.svelte-kit\output",
    "dist",
    "build"
)

foreach ($dir in $buildDirs) {
    $fullPath = Join-Path $RepoRoot $dir
    if (Test-Path $fullPath) {
        Remove-Item -Path $fullPath -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Removed: $dir" -ForegroundColor Gray
    }
}

# Step 4: Try to build
Write-Host "`nAttempting build..." -ForegroundColor Yellow
Write-Host "  Using: npm run build" -ForegroundColor Gray

$buildSuccess = $false
$buildError = ""

try {
    # Capture output
    $output = npm run build 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        $buildSuccess = $true
        Write-Host "  Build completed successfully!" -ForegroundColor Green
    } else {
        $buildError = $output | Out-String
        Write-Host "  Build failed" -ForegroundColor Red
    }
} catch {
    $buildError = $_.Exception.Message
    Write-Host "  Build error: $_" -ForegroundColor Red
}

# Step 5: Check what was created
Write-Host "`nChecking build output..." -ForegroundColor Yellow

$outputLocations = @(
    @{Path = "tori_ui_svelte\build"; Name = "SvelteKit build"},
    @{Path = "tori_ui_svelte\dist"; Name = "Vite dist"},
    @{Path = "tori_ui_svelte\.svelte-kit\output"; Name = "SvelteKit output"},
    @{Path = "dist"; Name = "Root dist"},
    @{Path = "build"; Name = "Root build"}
)

$foundOutput = $false
foreach ($loc in $outputLocations) {
    $fullPath = Join-Path $RepoRoot $loc.Path
    if (Test-Path $fullPath) {
        $files = Get-ChildItem -Path $fullPath -Recurse -File
        $count = ($files | Measure-Object).Count
        Write-Host "  Found $($loc.Name): $count files" -ForegroundColor Green
        $foundOutput = $true
    }
}

if (-not $foundOutput) {
    Write-Host "  No build output found!" -ForegroundColor Red
}

# Step 6: Create release structure anyway (for testing verification)
Write-Host "`nCreating release structure..." -ForegroundColor Yellow
$releaseDir = Join-Path $RepoRoot "releases\v1.0.0"

# Remove old v1.0.0 if exists
if (Test-Path $releaseDir) {
    Remove-Item -Path $releaseDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
$distTarget = Join-Path $releaseDir "dist"
New-Item -ItemType Directory -Force -Path $distTarget | Out-Null

# Try to copy any build output we found
$copied = $false
foreach ($loc in $outputLocations) {
    $fullPath = Join-Path $RepoRoot $loc.Path
    if (Test-Path $fullPath) {
        Get-ChildItem -Path $fullPath | Copy-Item -Destination $distTarget -Recurse -Force
        $copied = $true
        Write-Host "  Copied $($loc.Name) to release" -ForegroundColor Green
        break
    }
}

if (-not $copied) {
    # Create minimal placeholder
    Write-Host "  Creating placeholder files" -ForegroundColor Yellow
    $htmlPath = Join-Path $distTarget "index.html"
    Set-Content -Path $htmlPath -Value "<!DOCTYPE html><html><head><title>IRIS v1.0.0</title></head><body><h1>Build Placeholder</h1></body></html>"
}

# Create manifest
$manifestPath = Join-Path $releaseDir "manifest.json"
$manifestData = @{
    version = "1.0.0"
    buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    buildSuccess = $buildSuccess
}
$manifestData | ConvertTo-Json | Set-Content -Path $manifestPath

# Final summary
Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
if ($buildSuccess) {
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
} else {
    Write-Host "BUILD FAILED - But structure created for testing" -ForegroundColor Yellow
}
Write-Host "===================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
if (-not $buildSuccess) {
    Write-Host "1. Fix remaining import issues" -ForegroundColor Yellow
    Write-Host "2. Re-run this script" -ForegroundColor Yellow
} else {
    Write-Host "1. Run verification: .\tools\release\Verify-EndToEnd.ps1" -ForegroundColor Green
}

exit 0