param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "BUILD TEST - UNUSED IMPORT REMOVED" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nFix Applied:" -ForegroundColor Green
Write-Host "✅ Removed unused import: systemCoherence" -ForegroundColor White
Write-Host "   (It was imported but never used or exported)" -ForegroundColor Gray

Write-Host "`nAll fixes so far:" -ForegroundColor Yellow
Write-Host "1. QuiltGenerator import path" -ForegroundColor White
Write-Host "2. Added runElfinScript export" -ForegroundColor White
Write-Host "3. Added globalElfinInterpreter export" -ForegroundColor White
Write-Host "4. Removed unused systemCoherence import" -ForegroundColor White

Write-Host "`nRunning build..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

npm run build

$exitCode = $LASTEXITCODE

Write-Host "-------------------------------------------------------" -ForegroundColor Gray

if ($exitCode -eq 0) {
    Write-Host "`n🎉 BUILD SUCCESSFUL!" -ForegroundColor Green
    
    # Check for build output
    $svelteBuildPath = Join-Path $RepoRoot "tori_ui_svelte\build"
    if (Test-Path $svelteBuildPath) {
        $fileCount = (Get-ChildItem $svelteBuildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "`nBuild created $fileCount files" -ForegroundColor Green
        
        # Create release structure
        Write-Host "`nCreating release structure..." -ForegroundColor Yellow
        $releaseDir = Join-Path $RepoRoot "releases\v1.0.0"
        
        if (Test-Path $releaseDir) {
            Remove-Item $releaseDir -Recurse -Force
        }
        
        New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
        $distTarget = Join-Path $releaseDir "dist"
        
        Copy-Item -Path $svelteBuildPath -Destination $distTarget -Recurse -Force
        
        @{
            version = "1.0.0"
            buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            fileCount = $fileCount
        } | ConvertTo-Json | Set-Content -Path (Join-Path $releaseDir "manifest.json")
        
        Write-Host "✅ Release structure created at: $releaseDir" -ForegroundColor Green
    }
    
    Write-Host "`n✅ READY FOR VERIFICATION!" -ForegroundColor Green
    Write-Host "`nNext: .\tools\release\Verify-EndToEnd.ps1" -ForegroundColor Yellow
    
} else {
    Write-Host "`n❌ Build still has errors" -ForegroundColor Red
    Write-Host "Check the output above for remaining issues" -ForegroundColor Yellow
}

exit $exitCode