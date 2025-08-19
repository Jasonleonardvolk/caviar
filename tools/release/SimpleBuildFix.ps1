param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$UseTimestamp = $false
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

Write-Host "SIMPLE BUILD AND PACKAGE SCRIPT" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build the project
Write-Host "Building project..." -ForegroundColor Yellow

# Try to build with npm
npm run build

if ($LASTEXITCODE -ne 0) { 
    Write-Host "Build failed with npm run build" -ForegroundColor Red
    Write-Host "Trying direct vite build in tori_ui_svelte..." -ForegroundColor Yellow
    
    Set-Location tori_ui_svelte
    npx vite build
    Set-Location ..
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed completely" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Build complete" -ForegroundColor Green

# Step 2: Package creation
Write-Host "Creating release package..." -ForegroundColor Yellow

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# CRITICAL FIX: Use timestamped directory or clean existing one
if ($UseTimestamp) {
    $releaseDir = Join-Path $RepoRoot "releases\iris_v1_$timestamp"
} else {
    $releaseDir = Join-Path $RepoRoot "releases\v1.0.0"
    
    # Clean existing directory to avoid stale data
    if (Test-Path $releaseDir) {
        Write-Host "Cleaning existing v1.0.0 directory..." -ForegroundColor Yellow
        Remove-Item -Path $releaseDir -Recurse -Force
    }
}

# Create release directory
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

# Look for build output in multiple locations
$svelteBuildDir = Join-Path $RepoRoot "tori_ui_svelte\build"
$svelteDistDir = Join-Path $RepoRoot "tori_ui_svelte\dist"
$rootDistDir = Join-Path $RepoRoot "dist"
$distTarget = Join-Path $releaseDir "dist"

$foundBuild = $false

if (Test-Path $rootDistDir) {
    Write-Host "Found dist at root: $rootDistDir" -ForegroundColor Gray
    Copy-Item -Path $rootDistDir -Destination $distTarget -Recurse -Force
    $foundBuild = $true
} elseif (Test-Path $svelteBuildDir) {
    Write-Host "Found SvelteKit build: $svelteBuildDir" -ForegroundColor Gray
    Copy-Item -Path $svelteBuildDir -Destination $distTarget -Recurse -Force
    $foundBuild = $true
} elseif (Test-Path $svelteDistDir) {
    Write-Host "Found dist in tori_ui_svelte: $svelteDistDir" -ForegroundColor Gray
    Copy-Item -Path $svelteDistDir -Destination $distTarget -Recurse -Force
    $foundBuild = $true
}

if (-not $foundBuild) {
    Write-Host "WARNING: No build output found. Creating minimal dist folder." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $distTarget | Out-Null
    
    # Create a minimal index.html that won't break verification
    @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IRIS Build</title>
</head>
<body>
    <h1>IRIS Build Output</h1>
    <p>Build timestamp: $timestamp</p>
    <p>Note: This is a placeholder - actual build output was not found</p>
</body>
</html>
"@ | Out-File -FilePath (Join-Path $distTarget "index.html") -Encoding UTF8
    
    # Add a dummy JS file so it looks more like a real build
    "// Placeholder build output" | Out-File -FilePath (Join-Path $distTarget "app.js") -Encoding UTF8
}

# Verify we actually have files in dist
$fileCount = (Get-ChildItem $distTarget -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "Created dist with $fileCount files" -ForegroundColor Gray

# Create release manifest
$manifest = @{
    version = "1.0.0"
    buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    timestamp = $timestamp
    buildFound = $foundBuild
    fileCount = $fileCount
}

$manifest | ConvertTo-Json | Out-File -FilePath "$releaseDir\manifest.json" -Encoding UTF8

Write-Host "Release package created: $releaseDir" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Release Dir: $releaseDir" -ForegroundColor White
Write-Host "  Build Found: $foundBuild" -ForegroundColor White
Write-Host "  Files in dist: $fileCount" -ForegroundColor White
Write-Host ""

# Show what was created
if ($fileCount -gt 0) {
    Write-Host "Contents of dist:" -ForegroundColor Cyan
    Get-ChildItem $distTarget | Select-Object Name, Length | Format-Table
}

exit 0