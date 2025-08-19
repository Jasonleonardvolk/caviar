param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "BUILD TEST - ALL ERRORS FIXED" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nErrors Fixed:" -ForegroundColor Green
Write-Host "1. ✓ realGhostEngine.js - Fixed import path for QuiltGenerator" -ForegroundColor White
Write-Host "2. ✓ elfin/+page.svelte - Removed non-existent globalElfinInterpreter import" -ForegroundColor White

Write-Host "`nRunning build..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

npm run build

$exitCode = $LASTEXITCODE

Write-Host "-------------------------------------------------------" -ForegroundColor Gray

if ($exitCode -eq 0) {
    Write-Host "`n✅ BUILD SUCCESSFUL!" -ForegroundColor Green
    
    # Check what was created
    Write-Host "`nChecking build output..." -ForegroundColor Yellow
    $outputDirs = @(
        "tori_ui_svelte\build",
        "tori_ui_svelte\dist",
        "tori_ui_svelte\.svelte-kit\output"
    )
    
    foreach ($dir in $outputDirs) {
        $fullPath = Join-Path $RepoRoot $dir
        if (Test-Path $fullPath) {
            $count = (Get-ChildItem $fullPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-Host "  Found ${dir}: $count files" -ForegroundColor Green
        }
    }
    
    Write-Host "`nNEXT STEP:" -ForegroundColor Cyan
    Write-Host "Run: .\tools\release\Verify-EndToEnd.ps1" -ForegroundColor Yellow
    
} else {
    Write-Host "`n❌ Build still failing" -ForegroundColor Red
    Write-Host "Check the error messages above for remaining issues" -ForegroundColor Yellow
}

exit $exitCode