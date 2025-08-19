param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "CREATE RELEASE AND VERIFY" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nBuild was successful! Creating release structure..." -ForegroundColor Green

# Check where the build output is
$svelteKitOutput = Join-Path $RepoRoot "tori_ui_svelte\.svelte-kit\output"
$svelteBuildPath = Join-Path $RepoRoot "tori_ui_svelte\build"
$svelteDistPath = Join-Path $RepoRoot "tori_ui_svelte\dist"

$buildPath = $null
$fileCount = 0

# SvelteKit creates output in .svelte-kit/output
if (Test-Path $svelteKitOutput) {
    Write-Host "Found SvelteKit output directory" -ForegroundColor Green
    # The client files are what we need for deployment
    $clientPath = Join-Path $svelteKitOutput "client"
    if (Test-Path $clientPath) {
        $buildPath = $clientPath
        $fileCount = (Get-ChildItem $buildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "Using client build: $fileCount files" -ForegroundColor Green
    }
} elseif (Test-Path $svelteBuildPath) {
    $buildPath = $svelteBuildPath
    $fileCount = (Get-ChildItem $buildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "Using SvelteKit build: $fileCount files" -ForegroundColor Green
} elseif (Test-Path $svelteDistPath) {
    $buildPath = $svelteDistPath
    $fileCount = (Get-ChildItem $buildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "Using Vite dist: $fileCount files" -ForegroundColor Green
}

if ($buildPath) {
    # Create release structure
    Write-Host "`nCreating release at releases\v1.0.0..." -ForegroundColor Yellow
    $releaseDir = Join-Path $RepoRoot "releases\v1.0.0"
    
    # Clean existing release
    if (Test-Path $releaseDir) {
        Remove-Item $releaseDir -Recurse -Force
        Write-Host "Cleaned existing release directory" -ForegroundColor Gray
    }
    
    # Create new release structure
    New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
    $distTarget = Join-Path $releaseDir "dist"
    
    # Copy build output to dist
    Write-Host "Copying build output to dist..." -ForegroundColor Yellow
    Copy-Item -Path $buildPath -Destination $distTarget -Recurse -Force
    
    # Create manifest
    $manifest = @{
        version = "1.0.0"
        buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        fileCount = $fileCount
        buildPath = $buildPath
        buildSuccess = $true
    }
    $manifest | ConvertTo-Json | Set-Content -Path (Join-Path $releaseDir "manifest.json")
    
    Write-Host "`nRelease structure created successfully:" -ForegroundColor Green
    Write-Host "  Location: $releaseDir" -ForegroundColor White
    Write-Host "  Files in dist: $fileCount" -ForegroundColor White
    Write-Host "  Manifest: created" -ForegroundColor White
    
    # Show some of the files
    Write-Host "`nSample files in dist:" -ForegroundColor Cyan
    Get-ChildItem $distTarget -File | Select-Object -First 5 | ForEach-Object {
        Write-Host "  - $($_.Name)" -ForegroundColor Gray
    }
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "RUNNING VERIFICATION" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    # Run verification
    & ".\tools\release\Verify-EndToEnd.ps1"
    
} else {
    Write-Host "`nERROR: Could not find build output!" -ForegroundColor Red
    Write-Host "Checked:" -ForegroundColor Yellow
    Write-Host "  - $svelteKitOutput" -ForegroundColor Gray
    Write-Host "  - $svelteBuildPath" -ForegroundColor Gray
    Write-Host "  - $svelteDistPath" -ForegroundColor Gray
}

exit 0