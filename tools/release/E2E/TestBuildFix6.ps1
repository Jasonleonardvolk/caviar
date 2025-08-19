param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "BUILD TEST - UNUSED STORES REMOVED" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nLatest Fix:" -ForegroundColor Green
Write-Host "Removed unused store imports from elfin/+page.svelte:" -ForegroundColor White
Write-Host "  - ghostState (doesn't exist)" -ForegroundColor Gray
Write-Host "  - conceptGraph (doesn't exist)" -ForegroundColor Gray
Write-Host "  - conversationLog (doesn't exist)" -ForegroundColor Gray

Write-Host "`nAll 6 fixes applied:" -ForegroundColor Yellow
Write-Host "1. QuiltGenerator import path fixed" -ForegroundColor White
Write-Host "2. Added runElfinScript export" -ForegroundColor White
Write-Host "3. Added globalElfinInterpreter export" -ForegroundColor White
Write-Host "4. Removed unused systemCoherence import" -ForegroundColor White
Write-Host "5. Commented out Node.js CLI code in QuiltGenerator" -ForegroundColor White
Write-Host "6. Removed non-existent store imports" -ForegroundColor White

Write-Host "`nRunning build..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

npm run build

$exitCode = $LASTEXITCODE

Write-Host "-------------------------------------------------------" -ForegroundColor Gray

if ($exitCode -eq 0) {
    Write-Host "`nBUILD SUCCESSFUL!" -ForegroundColor Green
    
    # Check for build output
    $svelteBuildPath = Join-Path $RepoRoot "tori_ui_svelte\build"
    if (Test-Path $svelteBuildPath) {
        $fileCount = (Get-ChildItem $svelteBuildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "Found build output: $fileCount files" -ForegroundColor Green
        
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
        
        Write-Host "Release created at: $releaseDir" -ForegroundColor Green
    }
    
    Write-Host "`nREADY FOR VERIFICATION!" -ForegroundColor Green
    Write-Host "Run: .\tools\release\Verify-EndToEnd.ps1" -ForegroundColor Yellow
    
} else {
    Write-Host "`nBuild failed - check for more missing exports" -ForegroundColor Red
}

exit $exitCode