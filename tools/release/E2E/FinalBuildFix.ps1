param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "FINAL BUILD - NODE.JS IMPORT FIXED" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nLatest Fix:" -ForegroundColor Green
Write-Host "Commented out Node.js import in QuiltGenerator.ts" -ForegroundColor White
Write-Host "  Line 123: import from url module" -ForegroundColor Gray
Write-Host "  This was causing browser build to fail" -ForegroundColor Gray

Write-Host "`nAll fixes applied:" -ForegroundColor Yellow
Write-Host "1. QuiltGenerator import path fixed" -ForegroundColor White
Write-Host "2. Added runElfinScript export" -ForegroundColor White
Write-Host "3. Added globalElfinInterpreter export" -ForegroundColor White
Write-Host "4. Removed unused systemCoherence import" -ForegroundColor White
Write-Host "5. Commented out Node.js CLI code in QuiltGenerator" -ForegroundColor White

Write-Host "`nRunning build..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

npm run build

$exitCode = $LASTEXITCODE

Write-Host "-------------------------------------------------------" -ForegroundColor Gray

if ($exitCode -eq 0) {
    Write-Host "`nBUILD SUCCESSFUL!" -ForegroundColor Green
    
    # Check for build output
    $svelteBuildPath = Join-Path $RepoRoot "tori_ui_svelte\build"
    $svelteDistPath = Join-Path $RepoRoot "tori_ui_svelte\dist"
    $svelteKitOutput = Join-Path $RepoRoot "tori_ui_svelte\.svelte-kit\output"
    
    $buildPath = $null
    $fileCount = 0
    
    if (Test-Path $svelteBuildPath) {
        $buildPath = $svelteBuildPath
        $fileCount = (Get-ChildItem $buildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "`nFound SvelteKit build: $fileCount files" -ForegroundColor Green
    } elseif (Test-Path $svelteDistPath) {
        $buildPath = $svelteDistPath
        $fileCount = (Get-ChildItem $buildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "`nFound Vite dist: $fileCount files" -ForegroundColor Green
    } elseif (Test-Path $svelteKitOutput) {
        $buildPath = $svelteKitOutput
        $fileCount = (Get-ChildItem $buildPath -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "`nFound SvelteKit output: $fileCount files" -ForegroundColor Green
    }
    
    if ($buildPath) {
        # Create release structure
        Write-Host "`nCreating release structure..." -ForegroundColor Yellow
        $releaseDir = Join-Path $RepoRoot "releases\v1.0.0"
        
        if (Test-Path $releaseDir) {
            Remove-Item $releaseDir -Recurse -Force
        }
        
        New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
        $distTarget = Join-Path $releaseDir "dist"
        
        Copy-Item -Path $buildPath -Destination $distTarget -Recurse -Force
        
        @{
            version = "1.0.0"
            buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            fileCount = $fileCount
            buildSuccess = $true
        } | ConvertTo-Json | Set-Content -Path (Join-Path $releaseDir "manifest.json")
        
        Write-Host "Release structure created at: $releaseDir" -ForegroundColor Green
        Write-Host "Files in dist: $fileCount" -ForegroundColor Green
    }
    
    Write-Host "`nBUILD COMPLETE - READY FOR VERIFICATION!" -ForegroundColor Green
    Write-Host "`nNext step: .\tools\release\Verify-EndToEnd.ps1" -ForegroundColor Yellow
    
} else {
    Write-Host "`nBuild failed - check for more import issues" -ForegroundColor Red
    Write-Host "Look for more Node.js specific imports in browser code" -ForegroundColor Yellow
}

exit $exitCode