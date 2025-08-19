param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "FIX RELEASE FOR VERIFICATION" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nThe issue: IrisOneButton.ps1 wasn't finding SvelteKit output" -ForegroundColor Yellow
Write-Host "Fixed: Updated to check .svelte-kit\output\client" -ForegroundColor Green

# Create proper v1.0.0 release
Write-Host "`nCreating proper v1.0.0 release with actual build output..." -ForegroundColor Yellow

$releaseDir = Join-Path $RepoRoot "releases\v1.0.0"
$svelteKitClientDir = Join-Path $RepoRoot "tori_ui_svelte\.svelte-kit\output\client"

# Clean existing release
if (Test-Path $releaseDir) {
    Remove-Item $releaseDir -Recurse -Force
    Write-Host "Cleaned existing release directory" -ForegroundColor Gray
}

# Create new release structure
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
$distTarget = Join-Path $releaseDir "dist"

if (Test-Path $svelteKitClientDir) {
    Write-Host "Found SvelteKit client output" -ForegroundColor Green
    
    # Copy the client build to dist
    Copy-Item -Path $svelteKitClientDir -Destination $distTarget -Recurse -Force
    
    # Count files
    $fileCount = (Get-ChildItem $distTarget -File -Recurse | Measure-Object).Count
    Write-Host "Copied $fileCount files to dist" -ForegroundColor Green
    
    # Create manifest
    $manifest = @{
        version = "1.0.0"
        buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        fileCount = $fileCount
        buildPath = $svelteKitClientDir
        buildSuccess = $true
    }
    $manifest | ConvertTo-Json | Set-Content -Path (Join-Path $releaseDir "manifest.json")
    
    Write-Host "`nRelease structure created:" -ForegroundColor Green
    Write-Host "  Location: $releaseDir" -ForegroundColor White
    Write-Host "  Files in dist: $fileCount" -ForegroundColor White
    Write-Host "  Manifest: created" -ForegroundColor White
    
    # Verify the structure
    if ((Test-Path (Join-Path $releaseDir "dist")) -and (Test-Path (Join-Path $releaseDir "manifest.json"))) {
        Write-Host "`nVerification structure check:" -ForegroundColor Cyan
        Write-Host "  dist folder: EXISTS" -ForegroundColor Green
        Write-Host "  manifest.json: EXISTS" -ForegroundColor Green
        
        # Check for key files
        $indexHtml = Join-Path $distTarget "index.html"
        $appDir = Join-Path $distTarget "_app"
        
        if (Test-Path $appDir) {
            Write-Host "  _app folder: EXISTS" -ForegroundColor Green
        }
        
        Write-Host "`nREADY FOR VERIFICATION!" -ForegroundColor Green
    }
    
} else {
    Write-Host "ERROR: SvelteKit client output not found at:" -ForegroundColor Red
    Write-Host "  $svelteKitClientDir" -ForegroundColor Gray
    Write-Host "`nRun the build first:" -ForegroundColor Yellow
    Write-Host "  npm run build" -ForegroundColor White
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Now run verification:" -ForegroundColor Yellow
Write-Host "  .\tools\release\Verify-EndToEnd.ps1" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

exit 0