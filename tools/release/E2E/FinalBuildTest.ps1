param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "FINAL BUILD TEST - ALL EXPORTS ADDED" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nFixes Applied:" -ForegroundColor Green
Write-Host "✅ Added missing export: runElfinScript" -ForegroundColor White
Write-Host "✅ Added missing export: globalElfinInterpreter" -ForegroundColor White
Write-Host "✅ Fixed QuiltGenerator import path" -ForegroundColor White

Write-Host "`nRunning build..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

npm run build

$exitCode = $LASTEXITCODE

Write-Host "-------------------------------------------------------" -ForegroundColor Gray

if ($exitCode -eq 0) {
    Write-Host "`n🎉 BUILD SUCCESSFUL!" -ForegroundColor Green
    
    # Check what was created
    Write-Host "`nChecking build output..." -ForegroundColor Yellow
    
    # Check SvelteKit build output
    $svelteBuildPath = Join-Path $RepoRoot "tori_ui_svelte\build"
    if (Test-Path $svelteBuildPath) {
        $fileCount = (Get-ChildItem $svelteBuildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "  SvelteKit build: $fileCount files" -ForegroundColor Green
        
        # Create the release structure
        Write-Host "`nCreating release structure..." -ForegroundColor Yellow
        $releaseDir = Join-Path $RepoRoot "releases\v1.0.0"
        
        if (Test-Path $releaseDir) {
            Remove-Item $releaseDir -Recurse -Force
        }
        
        New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
        $distTarget = Join-Path $releaseDir "dist"
        
        # Copy build to dist
        Copy-Item -Path $svelteBuildPath -Destination $distTarget -Recurse -Force
        
        # Create manifest
        @{
            version = "1.0.0"
            buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            fileCount = $fileCount
        } | ConvertTo-Json | Set-Content -Path (Join-Path $releaseDir "manifest.json")
        
        Write-Host "  Created release at: $releaseDir" -ForegroundColor Green
        Write-Host "  Files in dist: $fileCount" -ForegroundColor Green
    }
    
    Write-Host "`n✅ READY FOR VERIFICATION!" -ForegroundColor Green
    Write-Host "`nRun: .\tools\release\Verify-EndToEnd.ps1" -ForegroundColor Yellow
    
} else {
    Write-Host "`n❌ Build still failing" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
}

exit $exitCode